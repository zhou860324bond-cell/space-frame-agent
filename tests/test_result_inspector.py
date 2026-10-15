"""结果探针、极值定位、截面正应力与显示控制。"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from PySide6.QtWidgets import QApplication                         # noqa: E402

from agent import Session                                          # noqa: E402
from desktop import scene                                          # noqa: E402
from desktop.main_window import MainWindow                         # noqa: E402
from desktop.result_inspector import (global_extreme, probe_member,
                                      projected_station,
                                      section_stress_grid)          # noqa: E402
from sections import rectangle                                     # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def solved() -> Session:
    section = rectangle("B", 0.2, 0.4)
    session = Session()
    assert session.set_model(model={
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "B", "material": "Steel"}],
        "materials": [{"name": "Steel", "E": 2.06e11, "nu": 0.3}],
        "sections": [section],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": [{"name": "P", "nodal_loads": [
            {"name": "Tip", "node": 2,
             "load": [0, 0, -50e3, 0, 0, 0]}]}],
    }).ok
    assert session.solve_model().ok
    return session


def test_probe_projects_to_member_and_returns_six_signed_components_and_stress():
    session = solved()
    data = probe_member(session.frame, session.solution, 1,
                        np.array([2.25, 0.4, -0.2]), "P")
    assert data["x"] == pytest.approx(2.25)
    assert set(data["forces"]) == {"N", "Vy", "Vz", "T", "My", "Mz"}
    assert data["force_unit"] == "kN"
    assert data["moment_unit"] == "kN·m"
    assert data["displacement_unit"] == "mm"
    assert data["stress"] is not None
    assert data["stress"]["min"] < 0 < data["stress"]["max"]


def test_projected_station_clamps_outside_points_to_member_ends():
    session = solved()
    assert projected_station(session.frame, 1, [-3, 0, 0]) == 0.0
    assert projected_station(session.frame, 1, [20, 0, 0]) == 6.0


def test_global_extreme_returns_a_traceable_member_station_and_point():
    session = solved()
    extreme = global_extreme(session.frame, session.solution, "Mz", "P")
    assert extreme["member"] == 1
    assert extreme["x"] == pytest.approx(0.0)
    assert abs(extreme["value"]) == pytest.approx(300.0)
    assert extreme["point"] == pytest.approx([0, 0, 0])


def test_circular_stress_grid_masks_points_outside_the_real_section():
    class Circular:
        name = "Tube"
        A = 0.01
        Iy = Iz = 1e-4
        cy = cz = 0.2
        circular = True

    grid = section_stress_grid(Circular(), 0.0, 1e3, 2e3, samples=21)
    assert np.isnan(grid["sigma"][0, 0])
    assert np.isfinite(grid["sigma"][10, 10])


def test_sign_filter_changes_only_a_copy():
    source = np.array([-2.0, 0.0, 3.0])
    assert scene.sign_filtered(source, "positive") == pytest.approx([0, 0, 3])
    assert scene.sign_filtered(source, "negative") == pytest.approx([-2, 0, 0])
    assert source == pytest.approx([-2, 0, 3])


def test_result_panel_controls_are_wired_to_contour_options(qt_app):
    window = MainWindow(solved())
    window.results.range_mode.setCurrentIndex(1)
    window.results.sign_mode.setCurrentIndex(1)
    window.results.overlay_deformed.setChecked(True)
    window.results.levels.setCurrentText("8 级")
    options = window.result_display_options
    assert options == {"percentile": None, "sign": "positive", "levels": 8,
                       "palette": "turbo", "shading": True,
                       "overlay_deformed": True, "show_extrema": True}


def test_member_probe_populates_result_table_and_extreme_can_be_located(qt_app):
    session = solved()
    window = MainWindow(session)
    from agent import ToolResult
    window.result = ToolResult(True, {"cases": ["P"]})
    window.case = "P"
    window.probe_member_result(1, np.array([3.0, 0.0, 0.0]))
    assert window.results.table.rowCount() >= 12
    assert "结果探针" in window.results.caption.text()
    window.locate_current_extreme()
    assert window.viewport.selection == ("member", 1)
    assert "绝对极值" in window.results.caption.text()


class _AcceptLoad:
    """替身荷载编辑框：直接确定，返回 -40 kN 竖向集中力。"""

    def __init__(self, *args, **kwargs):
        pass

    def exec(self):
        from PySide6.QtWidgets import QDialog
        return QDialog.DialogCode.Accepted

    def vector(self):
        return [0.0, 0.0, -40e3]


def test_bc_edit_invalidates_backend_and_desktop_result_together(qt_app, monkeypatch):
    """改荷载要同时作废后端结果与界面上的结果（原来经旧边界表单，现在经「创建载荷」）。"""
    session = solved()
    window = MainWindow(session)
    from agent import ToolResult
    window.result = ToolResult(True, {"cases": ["P"]})
    window.case = "P"
    monkeypatch.setattr("desktop.window_objects.EditLoadDialog", _AcceptLoad)
    window._finish_load("Tip-2", "P", "concentrated_force", {"nodes": [2]})
    assert session.solution is None and session.result_db is None
    assert window.result is None and window.case is None
    assert window.mode == "模型"
    assert "失效" in window.results.notes.toPlainText()
    assert not window.results.notes.isHidden()


def test_load_can_be_copied_and_support_has_an_explicit_delete(qt_app, monkeypatch):
    """旧边界表单的「复制荷载」「删除支座」由对象管理器的复制、删除承担。"""
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    session = solved()
    window = MainWindow(session)
    assert session.create_load("Tip-2", "P", "concentrated_force", {"nodes": [2]},
                               force=[0.0, 0.0, -1e3]).ok
    assert session.create_boundary_condition("BC-Base", {"nodes": [2]}, fix=[0, 0, 0, 1, 0, 0]).ok
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Tip-Copy", True))
    window.manage_object("loads", "copy", "Tip-2")
    names = {f["name"] for f in session.model["features"]["loads"]}
    assert {"Tip-2", "Tip-Copy"} <= names
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    window.manage_object("bcs", "delete", "BC-Base")
    assert "BC-Base" not in {f["name"] for f in session.model["features"].get("bcs", [])}


def test_the_stress_field_is_the_worse_extreme_fibre_and_keeps_its_sign(qt_app):
    """σ 取上下缘里绝对值大的那一侧，**并保留符号**——受拉为正、受压为负，
    正好落在发散色标的两端。取 |σ| 会把受压画成受拉那一头。"""
    from internal_forces import member_diagram

    session = solved()
    frame, sol = session.frame, session.solution
    mid = sorted(frame.members)[0]
    diagram = member_diagram(frame, sol, mid, sol.primary, stations=21)
    values = scene.member_scalar(frame, frame.members[mid], diagram, scene.STRESS)

    from stress import extreme_normal_stress
    low, high = extreme_normal_stress(frame.sections[frame.members[mid].section],
                                      diagram.N, diagram.My, diagram.Mz)
    for k in range(len(values)):
        assert values[k] in (low[k], high[k])
        assert abs(values[k]) == pytest.approx(max(abs(low[k]), abs(high[k])))


def test_a_section_without_fibre_distances_refuses_the_stress_contour(qt_app):
    """缺 cy/cz 时**明确拒绝**，不估算一个来路不明的应力。"""
    from frame3d import Section
    from internal_forces import member_diagram
    from stress import StressUnavailable

    session = solved()
    frame, sol = session.frame, session.solution
    mid = sorted(frame.members)[0]
    name = frame.members[mid].section
    old = frame.sections[name]
    frame.sections[name] = Section(name, A=old.A, Iy=old.Iy, Iz=old.Iz, J=old.J)
    diagram = member_diagram(frame, sol, mid, sol.primary, stations=5)
    with pytest.raises(StressUnavailable):
        scene.member_scalar(frame, frame.members[mid], diagram, scene.STRESS)


def test_the_palette_and_shading_controls_reach_the_viewport(qt_app):
    """色系和"立体"是显示选项，必须一路走到视口，不能停在面板上。"""
    window = MainWindow(solved())
    window.results.palette.setCurrentIndex(
        window.results.palette.findData("diverging"))
    window.results.shading.setChecked(False)
    window.set_mode("云图")
    assert window.viewport.contour_palette == "diverging"
    assert window.viewport.contour_shading is False
