"""模态分析的验证。

参照项目 FEM-Python 没有模态分析，这一块是我们多出来的能力，
所以更得验到位。判据分三类：

1. **闭合解** —— 悬臂、简支、两端固接梁的频率系数 β，扭转与轴向的一维解；
2. **收敛方向** —— 一致质量矩阵必然**从上方**单调逼近精确解（形函数比真实振型硬）。
   若哪次改动让它从下方逼近，那说明质量阵被写成集中质量或者干脆写错了，
   这条比单点比对更能抓住问题；
3. **恒等式** —— 有效质量之和等于总质量，频率随 √E 与 1/√ρ 缩放。

采到的最低阶未必是弯曲：扭转和轴向也在里面。所以夹具把无关自由度全约束掉，
只留一族模态——否则"第一阶"到底是什么全靠猜。
"""

import numpy as np
import pytest

from frame3d import Frame, Material, Member, Node, Section
from modal import assemble_mass, consistent_mass, modal
from units import MM, SI, convert_model

E, NU, RHO = 2.1e11, 0.3, 7850.0
A, IY, IZ, J = 0.01, 4e-5, 3e-4, 8e-7
L = 6.0
G = E / (2.0 * (1.0 + NU))


def _bar(n: int, mask: tuple[int, ...]) -> Frame:
    """沿全局 x 的杆，每个节点都按 mask 约束——用来隔离出单一族模态。"""
    f = Frame()
    for k in range(n + 1):
        f.nodes[k + 1] = Node(k + 1, L * k / n, 0.0, 0.0)
        f.supports[k + 1] = mask
    for k in range(n):
        f.members[k + 1] = Member(k + 1, k + 1, k + 2, "S", "M")
    f.sections["S"] = Section("S", A, IY, IZ, J)
    f.materials["M"] = Material("M", E, NU, RHO)
    return f


def _fix(f: Frame, nid: int, *dofs: int) -> None:
    mask = list(f.supports[nid])
    for d in dofs:
        mask[d] = 1
    f.supports[nid] = tuple(mask)


def bending(n: int, ends: str) -> Frame:
    """只留 x–z 平面内的弯曲：自由度只剩 uz 与 ry。

    全局 z 向位移在这套局部坐标下绕**局部 z** 弯曲，所以用的是 Iz。
    （拿 Iy 去对闭合解会差 √(Iz/Iy) = 2.74 倍——第一次就是这么错的。）
    """
    f = _bar(n, (1, 1, 0, 1, 0, 1))
    if ends == "cantilever":
        _fix(f, 1, 2, 4)
    elif ends == "simple":
        _fix(f, 1, 2)
        _fix(f, n + 1, 2)
    else:
        _fix(f, 1, 2, 4)
        _fix(f, n + 1, 2, 4)
    return f


def beam_frequency(beta_L: float, I: float = IZ) -> float:
    """Euler-Bernoulli 梁的自振频率闭合解。"""
    return beta_L ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * I / (RHO * A))


# ------------------------------------------------- 闭合解

CASES = {
    # β₁L：悬臂 1.8751、简支 π、两端固接 4.7300
    "cantilever": 1.87510407,
    "simple": np.pi,
    "clamped": 4.73004074,
}


@pytest.mark.parametrize("ends, beta", CASES.items())
def test_fundamental_frequency_matches_the_closed_form(ends, beta):
    got = modal(bending(16, ends), 3).frequencies[0]
    assert got == pytest.approx(beam_frequency(beta), rel=1e-4)


@pytest.mark.parametrize("ends, beta", CASES.items())
def test_four_elements_per_span_is_within_two_tenths_of_a_percent(ends, beta):
    """一致质量阵收敛得快：每跨四个单元，三种边界都在 0.2% 以内。

    实测：悬臂 +0.007%、简支 +0.026%、两端固接 +0.19%。
    这是这套做法真实的精度，不是"一个单元就够"——那句只对悬臂成立。
    """
    got = modal(bending(4, ends), 2).frequencies[0]
    assert abs(got / beam_frequency(beta) - 1.0) < 2e-3


def test_a_single_element_cantilever_is_within_half_a_percent():
    """最省的一档：悬臂一个单元 +0.47%。

    换成简支同样一个单元就差 11% —— 边界条件越"硬"，形函数越不够用。
    所以文档里不写"一个单元就够"这种一概而论的话。
    """
    one = modal(bending(1, "cantilever"), 1).frequencies[0]
    assert abs(one / beam_frequency(CASES["cantilever"]) - 1.0) < 5e-3
    simple = modal(bending(1, "simple"), 1).frequencies[0]
    assert abs(simple / beam_frequency(CASES["simple"]) - 1.0) > 0.05


@pytest.mark.parametrize("ends, beta", CASES.items())
def test_convergence_is_monotone_from_above(ends, beta):
    """一致质量矩阵**必然高估**频率，加密时单调下降逼近精确解。

    若哪次改动让它从下方逼近，说明质量阵写成了集中质量或者写错了——
    这条比任何单点比对都更能抓住问题。
    """
    exact = beam_frequency(beta)
    got = [modal(bending(n, ends), 2).frequencies[0] for n in (2, 4, 8, 16)]
    assert all(a > exact for a in got), got
    assert all(a > b for a, b in zip(got, got[1:])), got


def test_higher_cantilever_modes():
    """前三阶 β = 1.8751 / 4.6941 / 7.8548。只验一阶的话，
    质量阵里那些耦合项写错了也发现不了。"""
    got = modal(bending(24, "cantilever"), 3).frequencies
    for k, beta in enumerate((1.87510407, 4.69409113, 7.85475744)):
        assert got[k] == pytest.approx(beam_frequency(beta), rel=2e-3), k


def test_torsional_frequency():
    """扭转一维解 f₁ = (1/4L)·√(GJ / ρ·Ip)，一端固定一端自由。

    Ip 按 (Iy + Iz) 取——质量阵里的转动惯量项就是这么定的，
    这条测试同时验了那个取法。
    """
    f = _bar(12, (1, 1, 1, 0, 1, 1))       # 只留 rx
    _fix(f, 1, 3)
    ip = IY + IZ
    exact = np.sqrt(G * J / (RHO * ip)) / (4.0 * L)
    assert modal(f, 2).frequencies[0] == pytest.approx(exact, rel=2e-3)


def test_axial_frequency():
    """轴向一维解 f₁ = (1/4L)·√(E/ρ)。"""
    f = _bar(12, (0, 1, 1, 1, 1, 1))       # 只留 ux
    _fix(f, 1, 0)
    exact = np.sqrt(E / RHO) / (4.0 * L)
    assert modal(f, 2).frequencies[0] == pytest.approx(exact, rel=2e-3)


def test_the_weak_axis_gives_the_lower_frequency():
    """同一根杆两个方向的频率之比等于 √(Iz/Iy) —— 强弱轴没搞混。"""
    strong = modal(bending(16, "cantilever"), 2).frequencies[0]
    weak = _bar(16, (1, 0, 1, 1, 1, 0))    # 只留 uy 与 rz
    _fix(weak, 1, 1, 5)
    assert (strong / modal(weak, 2).frequencies[0]
            == pytest.approx(np.sqrt(IZ / IY), rel=2e-3))


# ------------------------------------------------- 恒等式与缩放

def test_frequency_scales_with_sqrt_stiffness_over_mass():
    """f ∝ √(E/ρ)。E 翻四倍频率翻倍，ρ 翻四倍频率减半。"""
    base = modal(bending(8, "cantilever"), 1).frequencies[0]

    stiff = bending(8, "cantilever")
    stiff.materials["M"] = Material("M", 4 * E, NU, RHO)
    assert modal(stiff, 1).frequencies[0] == pytest.approx(2 * base, rel=1e-9)

    heavy = bending(8, "cantilever")
    heavy.materials["M"] = Material("M", E, NU, 4 * RHO)
    assert modal(heavy, 1).frequencies[0] == pytest.approx(base / 2, rel=1e-9)


def test_effective_masses_sum_to_the_participating_mass():
    """取满全部模态时，有效质量之和是一个恒等式，必须精确成立。

    工程上就靠这个判断"取的阶数够不够"，所以它本身不能有偏差。
    """
    f = bending(6, "cantilever")
    n_free = f.num_dofs - int(np.sum([sum(m) for m in f.supports.values()]))
    r = modal(f, n_free)
    M = assemble_mass(f).toarray()
    direction = np.zeros(f.num_dofs)
    direction[2::6] = 1.0                   # 全局 z 向刚体平动
    free = [d for d in range(f.num_dofs)
            if not f.supports[d // 6 + 1][d % 6]]
    exact = float(direction[free] @ M[np.ix_(free, free)] @ direction[free])
    assert r.effective_mass[:, 2].sum() == pytest.approx(exact, rel=1e-8)


def test_total_mass_is_rho_A_L():
    f = bending(4, "cantilever")
    assert modal(f, 1).total_mass == pytest.approx(RHO * A * L, rel=1e-12)


def test_the_element_mass_matrix_totals_rho_A_L():
    """12×12 单元质量阵的平动行之和必须等于杆件质量。"""
    M = consistent_mass(L, RHO, A, (IY + IZ) / A)
    axial = M[np.ix_((0, 6), (0, 6))].sum()
    assert axial == pytest.approx(RHO * A * L, rel=1e-12)
    lateral = M[np.ix_((1, 7), (1, 7))].sum()
    assert lateral == pytest.approx(RHO * A * L, rel=1e-12)


def test_the_mass_matrix_is_symmetric_and_positive_definite():
    M = consistent_mass(L, RHO, A, (IY + IZ) / A)
    assert np.allclose(M, M.T, atol=0)
    assert np.linalg.eigvalsh(M).min() > 0


def test_frequencies_are_the_same_in_both_unit_systems():
    """Hz 与单位制无关。E、ρ、长度各换一次，抵消之后频率必须一模一样。"""
    from model_io import from_dict

    si = {"units": SI,
          "materials": [{"name": "M", "E": E, "nu": NU, "density": RHO}],
          "sections": [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}],
          "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                    {"id": 2, "x": L / 2, "y": 0, "z": 0},
                    {"id": 3, "x": L, "y": 0, "z": 0}],
          "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"},
                      {"id": 2, "i": 2, "j": 3, "section": "S", "material": "M"}],
          "supports": [{"node": 1, "fix": [1] * 6}]}
    a = modal(from_dict(si), 4).frequencies
    b = modal(from_dict(convert_model(si, MM)), 4).frequencies
    assert b == pytest.approx(a, rel=1e-9)


# ------------------------------------------------- 失败路径

def test_zero_density_is_refused_with_a_useful_message():
    """没有密度就没有质量。给个零频率或者除零 nan 都比不上直接说清楚。"""
    f = bending(4, "cantilever")
    f.materials["M"] = Material("M", E, NU, 0.0)
    with pytest.raises(ValueError, match="density"):
        modal(f, 2)


def test_a_fully_constrained_model_is_refused():
    f = _bar(1, (1,) * 6)
    with pytest.raises(ValueError, match="没有可振动"):
        modal(f, 2)


def test_an_empty_model_is_refused():
    with pytest.raises(ValueError, match="没有杆件"):
        modal(Frame(), 2)


def test_asking_for_too_many_modes_just_returns_what_exists():
    """要 100 阶但只有几个自由度，返回全部而不是报错。"""
    r = modal(bending(2, "cantilever"), 100)
    assert 0 < len(r.frequencies) < 100


def test_periods_are_the_reciprocal_of_frequencies():
    r = modal(bending(4, "cantilever"), 3)
    assert r.periods == pytest.approx(1.0 / r.frequencies)
