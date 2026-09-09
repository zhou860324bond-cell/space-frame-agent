"""绘图验证。不比对像素，只确认图画得出来、数值摘要对、符号约定没搞反。"""
from pathlib import Path

import pytest

from agent import Session
from plot3d import plot_axial, plot_deformed, plot_diagram

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7},
]


@pytest.fixture
def solved():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6, 6], storeys=[3.6], bays=[6],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL").payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]} for m in beams]}])
    s.solve_model()
    return s


def test_deformed_plot_writes_a_file_and_reports_the_peak(solved, tmp_path):
    out = tmp_path / "d.png"
    info = plot_deformed(solved.frame, solved.solution, "DL", out)
    assert out.exists() and out.stat().st_size > 5000
    expected = solved.query_results(what="max_displacement", case="DL").payload
    assert info["max_displacement_mm"] == pytest.approx(expected["magnitude_mm"], rel=1e-9)
    assert info["at_node"] == expected["node"]
    assert info["max_centerline_displacement_mm"] >= info["max_displacement_mm"]
    assert info["at_member"] in solved.frame.members


def test_axial_plot_writes_a_file(solved, tmp_path):
    out = tmp_path / "a.png"
    info = plot_axial(solved.frame, solved.solution, "DL", out)
    assert out.exists() and out.stat().st_size > 5000
    assert info["max_abs_axial_kN"] > 0


def test_axial_plot_uses_tension_positive_forces(solved):
    """重力作用下柱子受压，绘图取的分量必须让它们是负值。"""
    res = solved.solution["DL"]
    columns = [m.id for m in solved.frame.members.values() if m.section == "COLUMN"]
    assert columns
    assert all(res.member_forces[mid][6] < 0 for mid in columns)


def test_plot_tool_refuses_before_solving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not Session().plot_results(kind="deformed").ok


def test_plot_tool_writes_into_results_dir(solved, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = solved.plot_results(kind="axial", case="DL")
    assert r.ok
    assert Path(r.payload["path"]).exists()
    assert "results" in r.payload["path"]


def test_plot_tool_rejects_unknown_case(solved, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = solved.plot_results(kind="deformed", case="NOPE")
    assert not r.ok and "available" in r.payload


def test_static_diagram_keeps_physical_values_after_switching_to_mm(solved, tmp_path):
    si = plot_diagram(solved.frame, solved.solution, "Mz", "DL", tmp_path / "si.png")
    assert solved.set_units("N-mm-MPa").ok
    assert solved.solve_model().ok
    mm = plot_diagram(solved.frame, solved.solution, "Mz", "DL", tmp_path / "mm.png")
    assert mm["peak"] == pytest.approx(si["peak"], rel=1e-8)
    assert mm["at_x"] == pytest.approx(si["at_x"], abs=1e-4)
