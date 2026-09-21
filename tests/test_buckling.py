"""线性屈曲的验证。

Euler 的四种经典边界条件给了四个不同的系数，正好把几何刚度阵的
各项都验到：两端铰接 1、悬臂 1/4、两端固接 4、一固一铰 2.0457。
只验其中一种的话，把某个耦合项写错了照样能蒙对。

和模态一样，一致的几何刚度阵**从上方**单调逼近精确解。这条比单点比对更灵敏：
哪次改动让它从下方逼近，就说明矩阵写错了。
"""

import numpy as np
import pytest

from buckling import assemble_geometric, buckling, geometric_stiffness
from frame3d import Frame, Material, Member, Node, Section, solve
from units import MM, SI, convert_model

E, NU, RHO = 2.1e11, 0.3, 7850.0
A, IY, IZ, J = 0.01, 4e-5, 3e-4, 8e-7
L = 6.0
P = 1e3                                   # 参考轴压，1 kN
EULER = np.pi ** 2 * E * IZ / L ** 2      # π²EI/L²

# Euler 系数：λ_cr = k · π²EI/L² / P
ENDS = {
    "pinned": (1.0, "两端铰接"),
    "cantilever": (0.25, "悬臂"),
    "clamped": (4.0, "两端固接"),
    "propped": (2.0457504, "一端固接一端铰接"),
}


def column(n: int, ends: str, load: float = P) -> Frame:
    """沿全局 x 的压杆。只留 ux、uz、ry —— 屈曲发生在 x–z 平面内，用 Iz。"""
    f = Frame()
    for k in range(n + 1):
        f.nodes[k + 1] = Node(k + 1, L * k / n, 0.0, 0.0)
        f.supports[k + 1] = (0, 1, 0, 1, 0, 1)
    for k in range(n):
        f.members[k + 1] = Member(k + 1, k + 1, k + 2, "S", "M")
    f.sections["S"] = Section("S", A, IY, IZ, J)
    f.materials["M"] = Material("M", E, NU, RHO)

    def fix(nid: int, *dofs: int) -> None:
        mask = list(f.supports[nid])
        for d in dofs:
            mask[d] = 1
        f.supports[nid] = tuple(mask)

    if ends == "pinned":
        fix(1, 0, 2)
        fix(n + 1, 2)
    elif ends == "cantilever":
        fix(1, 0, 2, 4)
    elif ends == "clamped":
        fix(1, 0, 2, 4)
        fix(n + 1, 2, 4)
    else:                                  # propped
        fix(1, 0, 2, 4)
        fix(n + 1, 2)
    f.nodal_loads[n + 1] = (-load, 0.0, 0.0, 0.0, 0.0, 0.0)
    return f


def exact_factor(ends: str, load: float = P) -> float:
    return ENDS[ends][0] * EULER / load


# ------------------------------------------------- Euler 闭合解

@pytest.mark.gold
@pytest.mark.parametrize("ends", list(ENDS))
def test_critical_factor_matches_euler(ends):
    got = buckling(column(16, ends), num_modes=2).critical
    assert got == pytest.approx(exact_factor(ends), rel=1e-4)


@pytest.mark.parametrize("ends", list(ENDS))
def test_eight_elements_is_within_a_tenth_of_a_percent(ends):
    """实测 8 单元：铰接 +0.003%、悬臂 +0.0002%、固接 +0.05%、一固一铰 +0.014%。"""
    got = buckling(column(8, ends), num_modes=2).critical
    assert abs(got / exact_factor(ends) - 1.0) < 1e-3


@pytest.mark.gold
@pytest.mark.parametrize("ends", list(ENDS))
def test_convergence_is_monotone_from_above(ends):
    """一致几何刚度阵必然高估临界荷载，加密时单调下降。

    从下方逼近就说明矩阵写错了——这条比任何单点比对都灵敏。
    """
    exact = exact_factor(ends)
    got = [buckling(column(n, ends), num_modes=1).critical for n in (2, 4, 8, 16)]
    assert all(v > exact for v in got), got
    assert all(a > b for a, b in zip(got, got[1:], strict=False)), got


@pytest.mark.gold
def test_higher_modes_follow_the_square_of_the_half_wave_number():
    """两端铰接柱的各阶临界荷载之比是 1 : 4 : 9（n²π²EI/L²）。

    只验一阶的话，几何刚度阵里的耦合项写错了也发现不了。
    """
    r = buckling(column(24, "pinned"), num_modes=3)
    for k, n in enumerate((1, 2, 3)):
        assert r.factors[k] == pytest.approx(n ** 2 * exact_factor("pinned"),
                                             rel=3e-3), k


# ------------------------------------------------- 缩放关系

def test_the_factor_is_inversely_proportional_to_the_reference_load():
    """参考荷载翻倍，临界因子减半——λ 是"再放大多少倍"，不是绝对荷载。"""
    a = buckling(column(8, "pinned", load=P), num_modes=1).critical
    b = buckling(column(8, "pinned", load=2 * P), num_modes=1).critical
    assert b == pytest.approx(a / 2, rel=1e-9)


def test_the_factor_is_proportional_to_bending_stiffness():
    """EI 翻倍，临界荷载翻倍。"""
    base = buckling(column(8, "pinned"), num_modes=1).critical
    stiff = column(8, "pinned")
    stiff.sections["S"] = Section("S", A, IY, 2 * IZ, J)
    assert buckling(stiff, num_modes=1).critical == pytest.approx(2 * base, rel=1e-6)


def test_the_weak_axis_governs_when_both_planes_are_free():
    """两个方向都不约束时，先绕弱轴失稳。λ 之比等于 Iy/Iz。

    强弱轴搞混是刚架计算里最贵的一类错误——柱子按强轴校核通过、实际按弱轴失稳。
    """
    free = column(12, "pinned")
    for nid in list(free.supports):
        mask = list(free.supports[nid])
        mask[1] = mask[5] = 0              # 放开 uy 与 rz
        free.supports[nid] = tuple(mask)
    free.supports[1] = tuple(
        1 if k in (0, 1, 2) else v for k, v in enumerate(free.supports[1]))
    free.supports[max(free.nodes)] = tuple(
        1 if k in (1, 2) else v for k, v in enumerate(free.supports[max(free.nodes)]))
    got = buckling(free, num_modes=1).critical
    assert got == pytest.approx(exact_factor("pinned") * IY / IZ, rel=5e-3)


def test_the_factor_is_the_same_in_both_unit_systems():
    """λ 无量纲，换单位制必须一个字不变。"""
    from model_io import from_dict

    si = {"units": SI,
          "materials": [{"name": "M", "E": E, "nu": NU, "density": RHO}],
          "sections": [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}],
          "nodes": [{"id": k + 1, "x": L * k / 4, "y": 0.0, "z": 0.0}
                    for k in range(5)],
          "members": [{"id": k + 1, "i": k + 1, "j": k + 2,
                       "section": "S", "material": "M"} for k in range(4)],
          "supports": ([{"node": 1, "fix": [1, 1, 1, 1, 0, 1]}]
                       + [{"node": k, "fix": [0, 1, 0, 1, 0, 1]}
                          for k in range(2, 5)]
                       + [{"node": 5, "fix": [0, 1, 1, 1, 0, 1]}]),
          "nodal_loads": [{"node": 5, "load": [-P, 0, 0, 0, 0, 0]}]}
    a = buckling(from_dict(si), num_modes=2).factors
    b = buckling(from_dict(convert_model(si, MM)), num_modes=2).factors
    assert b == pytest.approx(a, rel=1e-8)


# ------------------------------------------------- 矩阵本身

def test_the_geometric_matrix_is_symmetric():
    kg = geometric_stiffness(L, -P, (IY + IZ) / A)
    assert np.allclose(kg, kg.T, atol=0)


def test_tension_and_compression_give_opposite_matrices():
    """K_g 与轴力成正比。受拉使结构变硬，受压削弱——符号必须整体翻转。"""
    pull = geometric_stiffness(L, P, (IY + IZ) / A)
    push = geometric_stiffness(L, -P, (IY + IZ) / A)
    assert np.allclose(pull, -push, atol=0)


def test_a_zero_axial_force_contributes_nothing():
    assert np.count_nonzero(geometric_stiffness(L, 0.0, (IY + IZ) / A)) == 0
    f = column(4, "pinned")
    assert assemble_geometric(f, {mid: 0.0 for mid in f.members}).nnz == 0


def test_the_axial_forces_come_back_with_the_result():
    """看是哪根压杆先失稳就靠这个。受拉为正，所以压杆是负值。"""
    r = buckling(column(4, "pinned"), num_modes=1)
    assert set(r.axial) == set(column(4, "pinned").members)
    assert all(N < 0 for N in r.axial.values()), r.axial
    assert all(N == pytest.approx(-P, rel=1e-9) for N in r.axial.values())


def test_the_mode_shape_is_zero_at_supports_and_nonzero_elsewhere():
    f = column(8, "pinned")
    r = buckling(f, num_modes=1)
    shape = r.shapes[:, 0]
    assert np.abs(shape).max() > 0
    for nid, mask in f.supports.items():
        for k, flag in enumerate(mask):
            if flag:
                assert shape[f.node_dofs(nid)[k]] == 0.0


# ------------------------------------------------- 失败路径

def test_a_structure_in_tension_is_refused():
    """全受拉不存在屈曲问题。给个巨大的 λ 或者算出负值都不如直接说清楚。"""
    f = column(4, "pinned", load=-P)        # 反向 = 受拉
    with pytest.raises(ValueError, match="没有受压杆件"):
        buckling(f, num_modes=1)


def test_an_empty_model_is_refused():
    with pytest.raises(ValueError, match="没有杆件"):
        buckling(Frame(), num_modes=1)


def test_an_unknown_case_is_refused():
    f = column(4, "pinned")
    with pytest.raises(ValueError, match="NOPE"):
        buckling(f, case="NOPE", num_modes=1)


def test_a_prepared_solution_can_be_reused():
    """已经算过静力就别再算一遍——大模型一句话里既要内力又要屈曲时会走这条路。"""
    f = column(8, "pinned")
    sol = solve(f)
    a = buckling(f, num_modes=1).critical
    b = buckling(f, num_modes=1, solution=sol).critical
    assert b == pytest.approx(a, rel=1e-12)


# --------------------------------------------------------------- 粗网格闸

def _beam_session(n: int):
    """8 m 简支梁切成 n 段，端部受压。n 越小，网格越粗。"""
    from agent import Session

    E, A, IY, IZ, JT = 2.06e11, 8.6e-3, 3.0e-5, 1.0e-4, 1.0e-6
    s = Session()
    s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": k + 1, "x": 8.0 * k / n, "y": 0, "z": 0}
                  for k in range(n + 1)],
        "members": [{"id": k + 1, "i": k + 1, "j": k + 2,
                     "material": "S", "section": "B"} for k in range(n)],
        "materials": [{"name": "S", "E": E, "nu": 0.3, "density": 7850.0}],
        "sections": [{"name": "B", "A": A, "Iy": IY, "Iz": IZ, "J": JT}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": n + 1, "fix": [0, 1, 1, 1, 0, 0]}],
    })
    s.set_nodal_load(n + 1, [-1.0e6, 0, 0, 0, 0, 0])
    return s


def test_coarse_mesh_is_called_out_in_buckling():
    """一跨一个单元时屈曲报得太高，必须当场说出来。

    实测单跨门式刚架每构件一个单元：λ=1.66，而收敛值 0.82——**偏高 102%**，
    方向是偏不安全那一侧。λ=1.66 在说"还有 66% 余量"，真相是它已经失稳。
    静力内力是解析恢复的、与网格无关，几何刚度阵不是，这个差别看不见，
    所以只能由工具自己讲出来。
    """
    r = _beam_session(1).buckling_analysis()
    assert r.ok
    warn = r.payload.get("mesh_warning")
    assert warn is not None, "一跨一个单元却没有任何网格提示"
    assert warn["elements_in_span"] == 1
    assert "偏不安全" in warn["message"]


def test_a_properly_meshed_span_is_not_nagged():
    """**这条是防喊狼来了的。**

    判据数的是"每跨几个单元"而不是"每根构件几个单元"——两者不一样。
    用户把一根 8 m 梁手工拆成 4 根构件时，每根仍只有 1 个单元，但那一跨
    实际有 4 个，屈曲误差 0.05%。按构件数判会在这种正确模型上误报。
    """
    r = _beam_session(4).buckling_analysis()
    assert r.ok
    assert "mesh_warning" not in r.payload, "细分够了还提示，属于误报"


def test_coarse_mesh_beats_load_direction_as_the_suspect():
    """网格太粗导致解不出临界因子时，别把人指向荷载方向。

    两端固接的梁只剖一个单元时，跨内没有可屈曲的自由度，屈曲必然失败。
    原来只报"没有找到正的临界荷载因子，检查荷载方向与约束"——照着这句话
    去查荷载方向永远查不出问题，因为荷载方向是对的。
    """
    s = _beam_session(1)
    s.model["supports"] = [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                           {"node": 2, "fix": [0, 1, 1, 1, 1, 1]}]
    r = s.buckling_analysis()
    assert not r.ok
    assert "likely_cause" in r.payload, "失败了却没说最可能的原因"
    assert "单元" in r.payload["likely_cause"]


def test_coarse_mesh_is_called_out_in_modal_too():
    """一致质量阵和几何刚度阵同理，都从上方逼近——模态也要提示。

    实测 8 m 简支梁一个单元 f1 偏高 11.0%，四个单元降到 0.026%。
    模态走的是**物理模型**（不编译），所以这里单独验一遍它也接上了。
    """
    coarse = _beam_session(1).modal_analysis(num_modes=4)
    fine = _beam_session(4).modal_analysis(num_modes=4)
    assert coarse.ok and fine.ok
    assert "mesh_warning" in coarse.payload
    assert "mesh_warning" not in fine.payload
    # 偏高的方向也一并钉住：粗网格的频率必须更高
    assert (coarse.payload["modes"][0]["frequency_Hz"]
            > fine.payload["modes"][0]["frequency_Hz"])
