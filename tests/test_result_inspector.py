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
    options = window.result_display_options
    assert options == {"percentile": None, "sign": "positive",
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


def test_bc_edit_invalidates_backend_and_desktop_result_together(qt_app):
    session = solved()
    window = MainWindow(session)
    from agent import ToolResult
    window.result = ToolResult(True, {"cases": ["P"]})
    window.case = "P"
    window.bc.set_selection("node", 2)
    window.bc.spn_fz.setValue(-40e3)
    window.bc._apply_nodal_load()
    assert session.solution is None and session.result_db is None
    assert window.result is None and window.case is None
    assert window.mode == "模型"
    assert "失效" in window.results.caption.text()


def test_load_can_be_copied_and_support_has_an_explicit_delete(qt_app, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    session = solved()
    window = MainWindow(session)
    window.bc.set_selection("node", 2)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Tip-Copy", True))
    window.bc._copy_nodal_load()
    case = session.model["load_cases"][0]
    assert {entry["name"] for entry in case["nodal_loads"]} == {
        "Tip", "Tip-Copy"}

    window.bc.set_selection("node", 1)
    window.bc._clear_support()
    assert all(int(item["node"]) != 1 for item in session.model["supports"])
