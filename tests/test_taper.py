"""变截面杆件（两端截面按几何线性插值、编译为阶梯段）的回归。

真值：线性变化 Iz(x) 的悬臂梁端部挠度 δ = P∫(L−x)²/(E·Iz(x))dx 有含对数的闭合解；
楔形工字钢的 Iz(x) 非线性，参照值取同一积分的高精度数值积分（被积函数由截面轮廓
精确给出，与阶梯离散无关）。
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad

import section_geometry as sg
import sections as S
import taper
from agent import Session
from frame3d import solve
from model_compiler import compile_model
from model_io import validate_payload
from units import convert_model

E, LOAD = 2.06e11, 1.0e4


def _cantilever(sec_i: dict, sec_j: dict, length: float, divisions: int | None = None) -> dict:
    model = {"units": "N-m-Pa",
             "materials": [{"name": "S", "E": E, "nu": 0.3, "density": 7850.0,
                            "allow_tension": 215e6}],
             "sections": [sec_i, sec_j],
             "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": length, "y": 0, "z": 0}],
             "members": [{"id": 1, "i": 1, "j": 2, "section": sec_i["name"],
                          "section_j": sec_j["name"], "material": "S"}],
             "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
             "load_cases": [{"name": "LC", "nodal_loads": [
                 {"node": 2, "load": [0, 0, -LOAD, 0, 0, 0]}]}]}
    if divisions:
        model["mesh"] = {"divisions": divisions}
    return model


def _tip(model: dict) -> tuple[float, int]:
    frame = compile_model(model).analysis_model
    result = solve(frame)
    return -result["LC"].U[frame.node_dofs(2)[2]], len(frame.members)


def test_linearly_varying_stiffness_matches_the_closed_form_and_converges_quadratically():
    """防止变截面只能整根换一个截面（原 retaper）：宽度线性变化的矩形悬臂梁，

    Iz 沿杆线性，端部挠度与闭合解一致；阶梯近似误差按段数平方下降。
    """
    length, si, sj = 4.0, S.rectangle("R0", 0.2, 0.3), S.rectangle("R1", 0.1, 0.3)
    a, c = si["Iz"], (sj["Iz"] - si["Iz"]) / length
    k = a + c * length
    exact = LOAD / E / c**3 * (k**2 * math.log(k / a) - 2 * k * (k - a) + (k * k - a * a) / 2)
    tip, count = _tip(_cantilever(si, sj, length))
    assert count == taper.TAPER_MIN_DIVISIONS
    assert tip == pytest.approx(exact, rel=1e-3)
    err16 = abs(tip / exact - 1)
    err32 = abs(_tip(_cantilever(si, sj, length, divisions=32))[0] / exact - 1)
    assert err32 == pytest.approx(err16 / 4, rel=0.05)


def test_web_tapered_i_beam_matches_the_exact_flexibility_integral():
    """楔形工字钢（腹板高 600→300）默认段数下端部挠度误差 < 0.2%（标定见 TAPER_MIN_DIVISIONS）。"""
    length = 6.0
    si = S.i_section("H600", 0.6, 0.2, 0.008, 0.012)
    sj = S.i_section("H300", 0.3, 0.2, 0.008, 0.012)

    def iz(x: float) -> float:
        geometry = taper.interpolated_geometry(si["geometry"], sj["geometry"], x / length)
        return sg.section_properties(geometry)["Iz"]

    exact = quad(lambda x: LOAD * (length - x)**2 / (E * iz(x)), 0, length,
                 epsabs=0, epsrel=1e-12, limit=200)[0]
    assert _tip(_cantilever(si, sj, length))[0] == pytest.approx(exact, rel=2e-3)


def test_segments_carry_interpolated_profiles_with_traceable_names():
    """每段截面取段中点的插值轮廓；名字写明两端截面、杆件号与段号，结果可追溯。"""
    si = S.i_section("H600", 0.6, 0.2, 0.008, 0.012)
    sj = S.i_section("H300", 0.3, 0.2, 0.008, 0.012)
    frame = compile_model(_cantilever(si, sj, 6.0)).analysis_model
    first = frame.sections[frame.members[1].section]
    assert first.name == "H600→H300@1.1"
    n = taper.TAPER_MIN_DIVISIONS
    expected_h = 0.6 - 0.3 * 0.5 / n
    assert first.geometry["dimensions"]["h"] == pytest.approx(expected_h, rel=1e-12)
    assert first.A == pytest.approx(sg.profile(first.geometry)["area"], rel=1e-12)


def test_tapered_model_is_unit_invariant():
    """米制与毫米制的变截面悬臂梁端部挠度只差长度换算。"""
    model = _cantilever(S.i_section("A", 0.6, 0.2, 0.008, 0.012),
                        S.i_section("B", 0.3, 0.2, 0.008, 0.012), 6.0)
    in_m = _tip(model)[0]
    in_mm = _tip(convert_model(model, "N-mm-MPa"))[0]
    assert in_mm == pytest.approx(in_m * 1000, rel=1e-9)


@pytest.mark.parametrize("change, message", [
    (lambda m: m["sections"][1].pop("geometry"), "真实轮廓"),
    (lambda m: m["sections"].__setitem__(1, S.rectangle("B", 0.2, 0.3)), "类型须相同"),
    (lambda m: m["sections"][1]["geometry"]["dimensions"].update(r=0.01), "尺寸项不一致"),
    (lambda m: m["members"][0].update(section_j="不存在"), "不存在的截面"),
])
def test_invalid_taper_definitions_are_rejected(change, message):
    """两端截面不能一一插值时，模型校验直接点名杆件与原因，不静默按等截面算。"""
    model = _cantilever(S.i_section("A", 0.6, 0.2, 0.008, 0.012),
                        S.i_section("B", 0.3, 0.2, 0.008, 0.012), 6.0)
    change(model)
    errors = validate_payload(model)
    assert any("变截面杆件 1" in e and message in e for e in errors), errors


def _tapered_session() -> Session:
    model = _cantilever(S.i_section("A", 0.6, 0.2, 0.008, 0.012),
                        S.i_section("B", 0.3, 0.2, 0.008, 0.012), 6.0)
    model["members"][0].pop("section_j")
    return Session(model=model)


def test_retaper_sets_and_clears_a_true_taper():
    """retaper 给 end_section 为真正的变截面；不给即恢复等截面并清掉旧的 section_j。"""
    s = _tapered_session()
    assert s.retaper([1], "A", end_section="B").ok
    assert s.model["members"][0]["section_j"] == "B"
    assert s.retaper([1], "A").ok
    assert "section_j" not in s.model["members"][0]


def test_retaper_refuses_an_impossible_taper_without_touching_the_model():
    s = _tapered_session()
    s.define_materials_and_sections(s.model["materials"],
                                    s.model["sections"] + [S.rectangle("R", 0.2, 0.3)])
    before = [dict(m) for m in s.model["members"]]
    result = s.retaper([1], "A", end_section="R")
    assert not result.ok and "类型须相同" in result.payload["errors"][0]
    assert s.model["members"] == before


def test_strength_check_skips_tapered_members_and_says_so():
    """逐杆强度验算按一个代表截面算，用在变截面上会错：剔除并写明，不冒充通过。"""
    s = _tapered_session()
    s.retaper([1], "A", end_section="B")
    result = s.check_strength()
    assert result.ok
    assert result.payload["count"] == 0
    assert any("变截面杆件 [1]" in note for note in result.payload["notes"])


def test_abaqus_export_refuses_tapered_members():
    """inp 按整根杆件写一个截面，变截面会被静默当成等截面，必须拒绝。"""
    s = _tapered_session()
    s.retaper([1], "A", end_section="B")
    result = s.solve_with_abaqus()
    assert not result.ok and "变截面杆件 [1]" in result.payload["error"]


def test_lateral_torsional_buckling_treats_a_tapered_member_as_continuous():
    """变截面杆件各分段截面名不同，翘曲仍应沿杆连续；临界值介于两端等截面之间。"""
    pytest.importorskip("gmsh")
    from buckling import continuous_warping_groups

    deep, shallow = S.i_section("A", 0.6, 0.2, 0.008, 0.012), S.i_section("B", 0.3, 0.2, 0.008, 0.012)

    def ltb(section_j: str | None, section: str = "A") -> float:
        model = {"units": "N-m-Pa",
                 "materials": [{"name": "S", "E": E, "nu": 0.3, "density": 7850.0}],
                 "sections": [deep, shallow],
                 "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 8.0, "y": 0, "z": 0}],
                 "members": [{"id": 1, "i": 1, "j": 2, "section": section, "material": "S",
                              **({"section_j": section_j} if section_j else {})}],
                 "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                              {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}],
                 "load_cases": [{"name": "LC", "nodal_loads": [
                     {"node": 1, "load": [0, 0, 0, 0, 1e5, 0]},
                     {"node": 2, "load": [0, 0, 0, 0, -1e5, 0]}]}]}
        result = Session(model=model).buckling_analysis(case="LC", lateral_torsional=True)
        assert result.ok, result.payload
        return result.payload["critical_factor"]

    tapered = ltb("B")
    assert ltb(None, "B") < tapered < ltb(None, "A")
    compiled = compile_model(_cantilever(deep, shallow, 8.0), min_divisions=4)
    groups = continuous_warping_groups(
        compiled.analysis_model,
        same_member={e: compiled.mapping.physical_member_id(e)
                     for e in compiled.analysis_model.members})
    assert len(set(groups.values())) == 1
