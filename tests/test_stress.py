"""极端纤维正应力（stress.py）。

测五件事：
1. 悬臂端力的固定端应力与 σ = M·c/I 逐位相符
2. 圆截面双向弯曲按平方和开方合成，不是两项相加
3. 轴力与弯矩的叠加，拉压两侧分别正确
4. 截面缺极端纤维距离时明确拒绝，不估算
5. 换算单位制后应力是同一个物理量（守 units.py 里的 cy/cz 换算）
"""

from __future__ import annotations

import math

import pytest

import sections as sec
from agent import Session
from frame3d import Section
from stress import (StressUnavailable, extreme_normal_stress,
                    max_normal_stress, worst_normal_stress_all_cases)
from units import MM, convert_model


L = 3.0
P = 10e3


def _cantilever(section: dict, axial: float = 0.0) -> dict:
    """悬臂梁：i 端固接，j 端加 -z 集中力，可选叠加轴向拉力。"""
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "S", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
        "sections": [section],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": section["name"], "material": "S"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": [{"name": "P", "nodal_loads": [
            {"node": 2, "load": [axial, 0, -P, 0, 0, 0]}]}],
    }


def _solved(model: dict) -> Session:
    session = Session()
    session.set_model(model=model)
    assert session.solve_model().ok
    return session


def test_cantilever_stress_matches_moment_over_section_modulus():
    """σ = M·c/I，M = P·L。这是独立于本实现的经典公式。"""
    rect = sec.rectangle("R", width=0.2, height=0.4)
    session = _solved(_cantilever(rect))
    got = max_normal_stress(session.frame, session.solution, "P", stations=51)
    expected = P * L * rect["cy"] / rect["Iz"]
    assert got["value"] == pytest.approx(expected, rel=1e-9)
    assert got["x"] == pytest.approx(0.0, abs=1e-12)     # 固定端
    assert got["sigma_max"] == pytest.approx(expected, rel=1e-9)
    assert got["sigma_min"] == pytest.approx(-expected, rel=1e-9)


def test_circular_section_combines_biaxial_bending_as_resultant():
    """圆截面 (cy, cz) 处没有材料，两个弯曲项不能相加。"""
    circ = sec.solid_circle("C", diameter=0.3)
    section = Section("C", circ["A"], circ["Iy"], circ["Iz"], circ["J"],
                      None, None, circ["cy"], circ["cz"], circ["circular"])
    my, mz = 40e3, 30e3
    _, high = extreme_normal_stress(section, 0.0, my, mz)
    resultant = math.hypot(my, mz) * circ["cy"] / circ["Iy"]
    naive_sum = (my + mz) * circ["cy"] / circ["Iy"]
    assert high == pytest.approx(resultant, rel=1e-12)
    assert high < naive_sum                       # 相加会高估
    assert circ["circular"] is True


def test_rectangle_adds_both_bending_terms():
    """矩形的角点真实存在，双向弯曲就是两项相加。"""
    rect = sec.rectangle("R", width=0.2, height=0.4)
    section = Section("R", rect["A"], rect["Iy"], rect["Iz"], rect["J"],
                      None, None, rect["cy"], rect["cz"], rect["circular"])
    my, mz = 40e3, 30e3
    _, high = extreme_normal_stress(section, 0.0, my, mz)
    expected = my * rect["cz"] / rect["Iy"] + mz * rect["cy"] / rect["Iz"]
    assert high == pytest.approx(expected, rel=1e-12)


def test_axial_shifts_both_extremes():
    """轴力平移整条应力分布：拉侧增大、压侧减小，差值仍是 2·M·c/I。"""
    rect = sec.rectangle("R", width=0.2, height=0.4)
    section = Section("R", rect["A"], rect["Iy"], rect["Iz"], rect["J"],
                      None, None, rect["cy"], rect["cz"], rect["circular"])
    n = 100e3
    low, high = extreme_normal_stress(section, n, 0.0, 30e3)
    bending = 30e3 * rect["cy"] / rect["Iz"]
    assert high == pytest.approx(n / rect["A"] + bending, rel=1e-12)
    assert low == pytest.approx(n / rect["A"] - bending, rel=1e-12)


def test_section_without_geometry_is_refused():
    """只有 A/Iy/Iz/J 的截面必须拒绝，不能拿任何近似顶上。"""
    bare = Section("BARE", 0.01, 1e-5, 1e-5, 1e-6)
    with pytest.raises(StressUnavailable, match="cy/cz"):
        extreme_normal_stress(bare, 1.0, 1.0, 1.0)


def test_one_bare_section_fails_the_whole_query():
    """范围内混进一根缺几何的杆件时整体报错，而不是跳过它给个偏小的值。"""
    rect = sec.rectangle("R", width=0.2, height=0.4)
    model = _cantilever(rect)
    model["sections"].append({"name": "BARE", "A": 0.01, "Iy": 1e-5,
                              "Iz": 1e-5, "J": 1e-6})
    model["nodes"].append({"id": 3, "x": 2 * L, "y": 0, "z": 0})
    model["members"].append({"id": 2, "i": 2, "j": 3,
                             "section": "BARE", "material": "S"})
    session = _solved(model)
    with pytest.raises(StressUnavailable):
        max_normal_stress(session.frame, session.solution, "P")


def test_stress_is_the_same_physical_quantity_in_both_unit_systems():
    """守 units.py 对 cy/cz 的换算：漏掉会让应力差 1000 倍。"""
    rect = sec.rectangle("R", width=0.2, height=0.4)
    si = _cantilever(rect)
    mm = convert_model(si, MM)
    assert mm["sections"][0]["cy"] == pytest.approx(rect["cy"] * 1e3)

    si_stress = worst_normal_stress_all_cases(
        *(lambda s: (s.frame, s.solution))(_solved(si)))          # Pa
    mm_stress = worst_normal_stress_all_cases(
        *(lambda s: (s.frame, s.solution))(_solved(mm)))          # MPa
    assert si_stress == pytest.approx(mm_stress * 1e6, rel=1e-9)


# ------------------------------------------------- 剪应力与四个强度理论

def test_shear_stress_matches_the_closed_forms():
    """τ = V·S/(I·b) 对经典截面必须**精确**命中闭合解。

    这两条是教科书上背下来的数：矩形 τmax = 1.5V/A、实心圆 4V/(3A)。
    对不上就说明静矩或中性轴宽度写错了，而那种错会一路传到折算应力，
    在那里已经看不出来源。
    """
    import sections as sec
    from frame3d import Section
    from stress import shear_stress

    V = 100e3
    rect = Section(**{k: v for k, v in sec.rectangle("R", 0.20, 0.40).items()
                      if k != "name"}, name="R")
    tau_y, _ = shear_stress(rect, V, 0.0)
    assert float(tau_y) == pytest.approx(1.5 * V / rect.A, rel=1e-12)

    circle = Section(**{k: v for k, v in sec.solid_circle("C", 0.20).items()
                        if k != "name"}, name="C")
    tau_y, _ = shear_stress(circle, V, 0.0)
    assert float(tau_y) == pytest.approx(4.0 * V / (3.0 * circle.A), rel=1e-12)


def test_the_i_section_web_carries_more_than_the_crude_average():
    """工字形腹板 τ 比"腹板均匀受剪"高一截，这个差别是真的，不是误差。

    V/(hw·tw) 把剪力当成在腹板上均匀分布；真实分布是抛物线，峰值更高。
    实测高 4.7%。若两者相等，多半是静矩被写成了只含腹板的那一半。
    """
    import sections as sec
    from frame3d import Section
    from stress import shear_stress

    raw = sec.i_section("I", 0.400, 0.200, 0.008, 0.013)
    isec = Section(**{k: v for k, v in raw.items() if k != "name"}, name="I")
    V = 100e3
    tau_y, _ = shear_stress(isec, V, 0.0)
    crude = V / ((0.400 - 2 * 0.013) * 0.008)
    assert float(tau_y) > crude
    assert float(tau_y) / crude == pytest.approx(1.047, abs=5e-3)


def test_shear_is_refused_when_the_geometry_is_missing():
    """只给 A/Iy/Iz/J 的截面不许"估"一个剪应力出来。

    来路不明的应力比没有应力更危险——正应力那一路已经这么定过，
    剪应力照同一条规矩。
    """
    from frame3d import Section
    from stress import StressUnavailable, shear_stress

    bare = Section("bare", A=0.01, Iy=1e-5, Iz=2e-5, J=1e-6)
    with pytest.raises(StressUnavailable):
        shear_stress(bare, 1.0, 0.0)


@pytest.mark.parametrize("sigma,tau", [
    (100e6, 0.0), (0.0, 50e6), (100e6, 50e6), (-100e6, 50e6),
    (200e6, 80e6), (-150e6, 30e6), (50e6, -50e6)])
def test_the_third_and_fourth_theories_agree_with_their_closed_forms(sigma, tau):
    """σr3、σr4 有两条算法，必须给出同一个数。

    一条从主应力合成，一条是梁里 (σ, τ) 的闭式 √(σ²+4τ²) / √(σ²+3τ²)。
    实现走的是主应力那条，闭式在这里当独立对照——闭式写错了主应力那条
    不会跟着错，所以这是真的交叉校验，不是自己验自己。
    """
    from stress import equivalent_stress, von_mises

    assert float(equivalent_stress(sigma, tau, "3")) == pytest.approx(
        math.sqrt(sigma ** 2 + 4 * tau ** 2), rel=1e-12)
    assert float(equivalent_stress(sigma, tau, "4")) == pytest.approx(
        math.sqrt(sigma ** 2 + 3 * tau ** 2), rel=1e-12)
    # 折算应力就是第四强度理论，GB 50017 §6.1.5 用的是同一个式子
    assert float(von_mises(sigma, tau)) == pytest.approx(
        float(equivalent_stress(sigma, tau, "4")), rel=1e-12)


def test_pure_shear_reproduces_the_textbook_principal_stresses():
    """纯剪 τ 下 σ1=+τ、σ2=0、σ3=−τ，于是 σr3=2τ、σr4=√3·τ。

    这是四个理论里差别最大的一个状态：第三理论比第四理论保守 15.5%。
    两者若算出同一个数，说明某一条的公式套错了。
    """
    from stress import equivalent_stress, principal_stresses

    tau = 50e6
    s1, s2, s3 = principal_stresses(0.0, tau)
    assert float(s1) == pytest.approx(tau)
    assert float(s2) == pytest.approx(0.0, abs=1e-6)
    assert float(s3) == pytest.approx(-tau)
    assert float(equivalent_stress(0.0, tau, "3")) == pytest.approx(2 * tau)
    assert float(equivalent_stress(0.0, tau, "4")) == pytest.approx(
        math.sqrt(3.0) * tau)


def test_the_second_theory_is_the_only_one_that_needs_poisson():
    """σr2 = σ1 − ν(σ2+σ3)，纯剪下等于 (1+ν)·τ。

    它是四条里唯一吃材料参数的。ν 传错会静默改变结论，所以钉死。
    """
    from stress import equivalent_stress

    tau = 50e6
    for nu in (0.0, 0.3, 0.5):
        got = equivalent_stress(0.0, tau, "2", nu=nu)
        assert float(got) == pytest.approx((1.0 + nu) * tau)


def test_an_unknown_theory_is_rejected_rather_than_guessed():
    from stress import equivalent_stress

    with pytest.raises(ValueError, match="1/2/3/4"):
        equivalent_stress(1.0, 1.0, "5")
