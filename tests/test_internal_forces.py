"""沿杆长内力分布的验证。

核心主张是**解析恢复、与网格无关**：单跨划一个单元，弯矩图就是精确解。
所以除了对经典闭合解，还专门验"1 个单元与 8 个单元结果相同"——
那条才是这套做法区别于逐单元取积分点的地方。
"""
import numpy as np
import pytest

from frame3d import Frame, Material, Member, Node, Section, solve
from internal_forces import COMPONENTS, all_diagrams, envelope, member_diagram

E, L = 2.1e11, 6.0
W, P = -20e3, -30e3
SEC = Section("S", 0.01, 4e-5, 3e-4, 8e-7)
MAT = Material("STEEL", E, 0.3)


def bar(n: int = 1) -> Frame:
    f = Frame()
    for k in range(n + 1):
        f.nodes[k + 1] = Node(k + 1, L * k / n, 0.0, 0.0)
    for k in range(n):
        f.members[k + 1] = Member(k + 1, k + 1, k + 2, "S", "STEEL")
    f.sections["S"] = SEC
    f.materials["STEEL"] = MAT
    return f


def simply_supported(n: int = 1) -> Frame:
    f = bar(n)
    f.supports[1] = (1, 1, 1, 1, 0, 0)
    f.supports[n + 1] = (0, 1, 1, 1, 0, 0)
    for k in range(n):
        f.member_loads[k + 1] = (0.0, 0.0, W)
    return f


def cantilever() -> Frame:
    f = bar(1)
    f.supports[1] = (1, 1, 1, 1, 1, 1)
    f.nodal_loads[2] = (0.0, 0.0, P, 0.0, 0.0, 0.0)
    return f


def fixed_fixed() -> Frame:
    f = bar(1)
    f.supports[1] = (1, 1, 1, 1, 1, 1)
    f.supports[2] = (1, 1, 1, 1, 1, 1)
    f.member_loads[1] = (0.0, 0.0, W)
    return f


# ------------------------------------------------- 闭合解

def test_simply_supported_midspan_moment_from_one_element():
    """单跨一个单元，跨中弯矩就是 wL²/8——不需要任何加密。"""
    f = simply_supported(1)
    d = member_diagram(f, solve(f), 1, stations=101)
    x, moment = d.extreme("Mz")
    assert abs(moment) == pytest.approx(abs(W) * L ** 2 / 8, rel=1e-12)
    assert x == pytest.approx(L / 2, rel=1e-12)


def test_simply_supported_end_moments_vanish():
    f = simply_supported(1)
    d = member_diagram(f, solve(f), 1)
    scale = abs(W) * L ** 2
    assert abs(d.Mz[0]) < 1e-9 * scale
    assert abs(d.Mz[-1]) < 1e-9 * scale


def test_simply_supported_end_shear():
    f = simply_supported(1)
    d = member_diagram(f, solve(f), 1)
    assert abs(d.Vy[0]) == pytest.approx(abs(W) * L / 2, rel=1e-12)
    assert abs(d.Vy[-1]) == pytest.approx(abs(W) * L / 2, rel=1e-12)


def test_cantilever_moment_is_linear_and_vanishes_at_the_tip():
    f = cantilever()
    d = member_diagram(f, solve(f), 1, stations=51)
    assert abs(d.Mz[0]) == pytest.approx(abs(P) * L, rel=1e-12)
    assert abs(d.Mz[-1]) < 1e-9 * abs(P) * L
    assert abs(d.Mz[25]) == pytest.approx(abs(P) * L / 2, rel=1e-12)
    assert np.allclose(d.Vy, d.Vy[0]), "无跨间荷载时剪力沿杆恒定"


def test_fixed_fixed_end_and_midspan_moments():
    f = fixed_fixed()
    d = member_diagram(f, solve(f), 1, stations=101)
    assert abs(d.Mz[0]) == pytest.approx(abs(W) * L ** 2 / 12, rel=1e-12)
    assert abs(d.Mz[50]) == pytest.approx(abs(W) * L ** 2 / 24, rel=1e-12)


def test_fixed_fixed_inflection_point_location():
    """反弯点在 x = L(3−√3)/6 ≈ 0.2113L，按零点插值找，不靠取样点碰巧对齐。"""
    f = fixed_fixed()
    d = member_diagram(f, solve(f), 1, stations=2001)
    sign = np.sign(d.Mz)
    crossings = np.where(np.diff(sign) != 0)[0]
    assert len(crossings) == 2, "两端固接梁应当有两个反弯点"
    k = crossings[0]
    x0 = d.x[k] - d.Mz[k] * (d.x[k + 1] - d.x[k]) / (d.Mz[k + 1] - d.Mz[k])
    assert x0 == pytest.approx(L * (3 - np.sqrt(3)) / 6, rel=1e-4)


# ------------------------------------------------- 与网格无关

@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_midspan_moment_is_independent_of_the_mesh(n):
    """1 个单元和 8 个单元给出同一个跨中弯矩。

    这是解析恢复相对"逐单元取一个积分点"的关键差别：后者要靠加密逼近。
    """
    f = simply_supported(n)
    sol = solve(f)
    peak = max(abs(member_diagram(f, sol, mid).extreme("Mz")[1])
               for mid in f.members)
    assert peak == pytest.approx(abs(W) * L ** 2 / 8, rel=1e-9)


def test_diagram_matches_the_end_forces_it_came_from():
    """x=0 处等于 −f[0..5]，x=L 处等于 f[6..11]——与杆端力必须自洽。"""
    f = fixed_fixed()
    sol = solve(f)
    d = member_diagram(f, sol, 1, stations=3)
    forces = sol.member_forces[1]
    for k, name in enumerate(COMPONENTS):
        assert d.component(name)[0] == pytest.approx(-forces[k], abs=1e-6)
        assert d.component(name)[-1] == pytest.approx(forces[6 + k], abs=1e-6)


# ------------------------------------------------- 工况、组合与查询

def test_combo_diagram_is_the_superposition_of_its_cases():
    f = bar(1)
    f.supports[1] = (1, 1, 1, 1, 1, 1)
    f.case("D").member_loads[1] = (0.0, 0.0, -10e3)
    f.case("L").nodal_loads[2] = (0.0, 0.0, -5e3, 0.0, 0.0, 0.0)
    f.combos["C"] = {"D": 1.3, "L": 1.5}
    sol = solve(f)
    d = member_diagram(f, sol, 1, "D", stations=21)
    l = member_diagram(f, sol, 1, "L", stations=21)
    c = member_diagram(f, sol, 1, "C", stations=21)
    assert np.allclose(c.Mz, 1.3 * d.Mz + 1.5 * l.Mz, rtol=1e-10)
    assert np.allclose(c.Vy, 1.3 * d.Vy + 1.5 * l.Vy, rtol=1e-10)


def test_envelope_finds_the_worst_member_and_location():
    f = simply_supported(4)
    sol = solve(f)
    worst = envelope(f, sol, "Mz", stations=201)
    assert abs(worst["value"]) == pytest.approx(abs(W) * L ** 2 / 8, rel=1e-6)
    assert worst["member"] in f.members
    assert 0.0 <= worst["x"] <= L / 4 + 1e-9


def test_all_diagrams_covers_every_member():
    f = simply_supported(4)
    diagrams = all_diagrams(f, solve(f))
    assert set(diagrams) == set(f.members)


def test_unknown_member_and_component_are_rejected():
    f = cantilever()
    sol = solve(f)
    with pytest.raises(KeyError, match="99"):
        member_diagram(f, sol, 99)
    with pytest.raises(KeyError, match="内力分量"):
        member_diagram(f, sol, 1).component("Q")
    with pytest.raises(ValueError, match="stations"):
        member_diagram(f, sol, 1, stations=1)


def test_tied_peaks_resolve_to_the_first_one_regardless_of_roundoff():
    """对称结构里两处极值解析上相等，算出来只差末几位。

    求解器换了求和顺序（SuperLU 对称模式）或模型换了单位制，末位谁大谁小
    就翻过来，报出的最不利位置在两处之间跳——单位不变性测试就是这么红的。
    差在 PEAK_TIE_RTOL 以内算并列，取先遇到的那个。
    """
    from internal_forces import PEAK_TIE_RTOL, exceeds, peak_index

    a = 123.456
    for wobble in (1.0 + 1e-13, 1.0 - 1e-13):
        assert peak_index([a, 0.5, -a * wobble]) == 0
        assert not exceeds(-a * wobble, a)
    assert peak_index([a, 0.5, -a * (1.0 + 10 * PEAK_TIE_RTOL)]) == 2
    assert exceeds(a * (1.0 + 10 * PEAK_TIE_RTOL), a)
    assert peak_index([]) == 0
