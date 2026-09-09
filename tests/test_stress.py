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
