"""支座沉降与自重的验证。

支座沉降是**给定位移**，不是荷载。最常见的错法是把它当成等效荷载去凑，
那样算出来的量级看着也对，只是数不对。所以这里全部对闭合解验：

    两端固接梁，一端竖向沉降 Δ：
        端弯矩 M = 6EIΔ/L²，两端等大反向；剪力 V = 12EIΔ/L³
    悬臂梁，固端沉降 Δ：整根杆平动 Δ，内力恒为零

后一条尤其要紧——静定结构在支座沉降下不产生内力。若代码把沉降当荷载加，
这里立刻会冒出一堆本不该存在的弯矩。
"""

import numpy as np
import pytest

from frame3d import (Frame, Material, Member, Node, Section, check_equilibrium,
                     check_model, self_weight_loads, solve)
from internal_forces import member_diagram
from span_loads import UNIFORM
from units import system as unit_system

GRAVITY = unit_system("N-m-Pa").gravity

E, L, DELTA = 2.1e11, 6.0, -0.01      # 沉降 10 mm 向下
IZ, AREA = 3e-4, 0.01
SEC = Section("S", AREA, 4e-5, IZ, 8e-7)
MAT = Material("STEEL", E, 0.3)
STEEL_RHO = 7850.0


def bar(density: float = 0.0) -> Frame:
    f = Frame()
    f.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    f.nodes[2] = Node(2, L, 0.0, 0.0)
    f.members[1] = Member(1, 1, 2, "S", "STEEL")
    f.sections["S"] = SEC
    f.materials["STEEL"] = Material("STEEL", E, 0.3, density)
    return f


# ------------------------------------------------- 支座沉降

def test_fixed_fixed_settlement_end_moments():
    """两端固接、一端沉降 Δ：端弯矩 6EIΔ/L²，两端等大反向。"""
    f = bar()
    f.supports[1] = (1,) * 6
    f.supports[2] = (1,) * 6
    f.settlements[2] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    d = member_diagram(f, solve(f), 1, stations=3)
    expected = 6 * E * IZ * DELTA / L ** 2
    assert abs(d.Mz[0]) == pytest.approx(abs(expected), rel=1e-9)
    assert d.Mz[0] == pytest.approx(-d.Mz[-1], rel=1e-9)


def test_fixed_fixed_settlement_shear():
    """剪力沿杆恒定，大小 12EIΔ/L³。"""
    f = bar()
    f.supports[1] = (1,) * 6
    f.supports[2] = (1,) * 6
    f.settlements[2] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    d = member_diagram(f, solve(f), 1, stations=11)
    assert abs(d.Vy[0]) == pytest.approx(abs(12 * E * IZ * DELTA / L ** 3), rel=1e-9)
    assert np.allclose(d.Vy, d.Vy[0])


def test_the_settled_support_actually_moves_by_the_given_amount():
    """给多少就必须动多少 —— 否则说明沉降被当成荷载加了进去。"""
    f = bar()
    f.supports[1] = (1,) * 6
    f.supports[2] = (1,) * 6
    f.settlements[2] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    sol = solve(f)
    assert sol["default"].U[f.node_dofs(2)[2]] == pytest.approx(DELTA, rel=1e-12)


def test_a_statically_determinate_structure_has_no_settlement_forces():
    """悬臂固端沉降：整根杆平动，内力恒为零。

    静定结构在支座沉降下不产生内力。把沉降当等效荷载算的实现，
    在这里会凭空冒出一堆弯矩 —— 这是最灵敏的一条判据。
    """
    f = bar()
    f.supports[1] = (1,) * 6
    f.settlements[1] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    sol = solve(f)
    d = member_diagram(f, sol, 1, stations=21)
    scale = abs(E * IZ * DELTA / L ** 2)
    assert np.abs(d.Mz).max() < 1e-9 * scale
    assert np.abs(d.Vy).max() < 1e-9 * scale / L
    # 自由端与固定端一起平动同样的量
    assert sol["default"].U[f.node_dofs(2)[2]] == pytest.approx(DELTA, rel=1e-9)


def test_settlement_reactions_still_balance():
    """沉降不引入外荷载，反力必须自平衡。"""
    f = bar()
    f.supports[1] = (1,) * 6
    f.supports[2] = (1,) * 6
    f.settlements[2] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    assert check_equilibrium(f, solve(f))["ok"]


def test_settlement_scales_linearly():
    f = bar()
    f.supports[1] = (1,) * 6
    f.supports[2] = (1,) * 6
    f.case("A").settlements[2] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    f.case("B").settlements[2] = (0.0, 0.0, 3 * DELTA, 0.0, 0.0, 0.0)
    f.load_cases.pop("default", None)
    sol = solve(f)
    a = member_diagram(f, sol, 1, case="A", stations=5).Mz
    # 反弯点处 Mz 过零，纯相对误差在那里没有意义，给一个按量级定的绝对容差
    assert (member_diagram(f, sol, 1, case="B", stations=5).Mz
            == pytest.approx(3 * a, rel=1e-9, abs=1e-9 * np.abs(a).max()))


def test_a_combo_carries_settlement_through():
    f = bar()
    f.supports[1] = (1,) * 6
    f.supports[2] = (1,) * 6
    f.case("SET").settlements[2] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    f.load_cases.pop("default", None)
    f.combos["0.5SET"] = {"SET": 0.5}
    sol = solve(f)
    assert (member_diagram(f, sol, 1, case="0.5SET", stations=5).Mz
            == pytest.approx(0.5 * member_diagram(f, sol, 1, case="SET", stations=5).Mz))


def test_settlement_on_a_free_direction_is_reported():
    """未约束方向上"指定位移"不会生效，用户却以为施加了 —— 必须说出来。"""
    f = bar()
    f.supports[1] = (1,) * 6
    f.supports[2] = (0, 1, 1, 1, 0, 0)     # uz 约束住了，ux 没有
    f.settlements[2] = (DELTA, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert any("不会生效" in m for m in check_model(f)), check_model(f)


def test_settlement_on_a_non_support_node_is_reported():
    f = bar()
    f.supports[1] = (1,) * 6
    f.settlements[2] = (0.0, 0.0, DELTA, 0.0, 0.0, 0.0)
    assert any("不是支座节点" in m for m in check_model(f))


def test_a_model_without_settlement_is_bit_for_bit_unchanged():
    """没有沉降时结果必须和改动前完全一致 —— 新分支不能碰老路径。"""
    f = bar()
    f.supports[1] = (1,) * 6
    f.member_loads[1] = (0.0, 0.0, -20e3)
    d = member_diagram(f, solve(f), 1, stations=11)
    assert d.Mz[0] == pytest.approx(-20e3 * L ** 2 / 2, rel=1e-12)


# ------------------------------------------------- 自重

def test_self_weight_equals_rho_A_g():
    f = bar(STEEL_RHO)
    loads = self_weight_loads(f)
    assert set(loads) == {1}
    assert loads[1].kind == UNIFORM
    assert loads[1].w1[2] == pytest.approx(-STEEL_RHO * AREA * GRAVITY)
    assert loads[1].w1[0] == 0.0 and loads[1].w1[1] == 0.0


def test_self_weight_total_matches_the_reaction():
    """自重总量 ρALg 必须等于支座竖向反力 —— 从密度一路到反力全链路验一遍。"""
    f = bar(STEEL_RHO)
    f.supports[1] = (1,) * 6
    for mid, load in self_weight_loads(f).items():
        f.member_spans.setdefault(mid, []).append(load)
    sol = solve(f)
    assert (sol["default"].R[f.node_dofs(1)[2]]
            == pytest.approx(STEEL_RHO * AREA * L * GRAVITY, rel=1e-9))


def test_a_zero_density_material_weighs_nothing():
    """密度默认为零，老模型的结果一个字都不会变。"""
    assert self_weight_loads(bar(0.0)) == {}


def test_the_self_weight_factor_scales_it():
    a = self_weight_loads(bar(STEEL_RHO))[1].w1[2]
    b = self_weight_loads(bar(STEEL_RHO), factor=1.3)[1].w1[2]
    assert b == pytest.approx(1.3 * a)


def test_self_weight_can_point_somewhere_else():
    """方向可指定，并且会被归一化 —— 传 (0,0,-2) 不该把自重放大一倍。"""
    load = self_weight_loads(bar(STEEL_RHO), direction=(0.0, 0.0, -2.0))[1]
    assert load.w1[2] == pytest.approx(-STEEL_RHO * AREA * GRAVITY)


def test_a_zero_direction_is_refused():
    with pytest.raises(ValueError, match="零向量"):
        self_weight_loads(bar(STEEL_RHO), direction=(0.0, 0.0, 0.0))


def test_gravity_follows_the_unit_system():
    """g 是这套代码里唯一真正依赖单位制的常数。

    写死成 9.80665 m/s² 的话，选 N-mm-MPa 时自重会小 1000 倍，
    而且不报错、不告警 —— 结果看着完全正常。
    """
    si = bar(STEEL_RHO)
    mm = bar(STEEL_RHO)
    mm.units = "N-mm-MPa"
    assert self_weight_loads(mm)[1].w1[2] == pytest.approx(
        1000.0 * self_weight_loads(si)[1].w1[2])


def test_an_unknown_unit_system_is_refused():
    f = bar(STEEL_RHO)
    f.units = "kg-cm-s"
    with pytest.raises(ValueError, match="未知单位制"):
        self_weight_loads(f)
