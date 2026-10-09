"""节点实体主视口：数据关联、只读显示与求解版本失效回归。"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402
from agent import ToolResult                                         # noqa: E402
from desktop.main_window import MainWindow                           # noqa: E402
from desktop.solid_result import read_result                         # noqa: E402
from solid3d import EDGE_PAIRS                                       # noqa: E402
from tools.render_smoke import build_session                         # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def files(tmp_path):
    corners = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tetra = np.vstack([corners, [(corners[i] + corners[j]) / 2 for i, j in EDGE_PAIRS]])
    nodes = np.vstack([tetra, tetra + [3, 0, 0]])
    elements = np.arange(20).reshape(2, 10)
    displacement = np.zeros_like(nodes)
    displacement[-1] = [3., 4., 0.]
    for index in range(2):
        np.savez(tmp_path / f"native_joint_{index}.npz", nodes=nodes, elements=elements,
                 displacement=displacement, element_mises=[100., 1000. * (index + 1)],
                 element_abs_principal=[200., 2000. * (index + 1)])
    summary = dict(schema="solid-joint-analysis/native-v1", node_id=65, case="DL",
                   meshes=[dict(vtu=f"Z:/old/native_joint_{i}.vtu", nodes=20, elements=2,
                                mesh_size_mm=2.0 / (i + 1)) for i in range(2)],
                   files={"directory": str(tmp_path)})
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path, summary


@pytest.mark.parametrize("suffix", ["npz", "vtu", "json"])
def test_saved_results_open_from_each_format_and_keep_cell_stresses(files, suffix):
    """已完成分析可直接打开；目录搬移或截图失败不应迫使用户重新求解。"""
    directory, _summary = files
    grid, *_ = read_result(directory / "native_joint_1.npz")
    grid.save(directory / "native_joint_1.vtu")
    path = directory / ("summary.json" if suffix == "json" else f"native_joint_1.{suffix}")
    loaded, _summary, paths, index, caption = read_result(path)
    assert loaded.n_cells == 2 and loaded.n_points == 20
    np.testing.assert_array_equal(loaded.cell_data["Mises_MPa"], [100, 2000])
    assert "Mises_MPa" not in loaded.point_data
    assert loaded.point_data["DisplacementMagnitude_mm"][-1] == 5
    assert index == 1 and all(path.parent == directory for path in paths)
    assert caption == "节点 65 · 工况 DL"


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1])
def test_invalid_solid_values_are_rejected_before_rendering(files, bad_value):
    """非法单元应力不得进入色标或制造看似有效的实体云图。"""
    directory, _summary = files
    path = directory / "native_joint_0.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["element_mises"] = np.array([100, bad_value])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="应力结果无效"):
        read_result(path)


def test_clipped_range_keeps_true_peak_and_deformation_is_read_only(app, files):
    """P99 着色不得隐藏真实峰值；放大变形不得写回原始坐标和计算结果。"""
    directory, _summary = files
    window = MainWindow()
    assert window._load_solid_result(directory / "summary.json")
    original = window._solid_grid.points.copy()
    window.solid_panel.scale.setValue(10)
    got = window.viewport._solid_last
    assert got["peak"] == 2000 and got["clim"][1] < 2000
    assert "2000" in window.viewport.result_overlay.peak.text()
    np.testing.assert_array_equal(window._solid_grid.points, original)
    np.testing.assert_allclose(got["mesh"].points[-1], original[-1] + [30, 40, 0])
    window.solid_panel.range.setCurrentIndex(1)
    assert window.viewport._solid_last["clim"] == (0, 2000)


def test_displacement_uses_nodes_and_mesh_levels_keep_options(app, files):
    """位移按节点显示；换网格档位不能沿用上一档应力或丢失显示选项。"""
    directory, _summary = files
    window = MainWindow()
    window._load_solid_result(directory / "summary.json")
    window.solid_panel.field.setCurrentIndex(2)
    assert window.viewport._solid_last["association"] == "point"
    assert window.viewport._solid_last["peak"] == 5
    window.solid_panel.field.setCurrentIndex(0)
    window.solid_panel.edges.setChecked(True)
    window.solid_panel.levels.setCurrentIndex(0)
    assert window.viewport._solid_last["peak"] == 1000
    assert window.viewport._solid_last["show_edges"] is True
    assert window.solid_panel.levels.currentIndex() == 0


def test_imported_result_opens_without_model_and_returns_to_modelling(app, files):
    """既有实体结果独立可看；返回建模时不能误用实体的 mm 坐标。"""
    directory, _summary = files
    window = MainWindow()
    window.show()
    assert window._load_solid_result(directory / "summary.json")
    window.refresh()
    assert window.mode == "实体云图" and window.mode_actions["实体云图"].isChecked()
    assert window.empty_state.isHidden()
    assert window.viewport._frame is None and window.viewport.pick_mode is None
    assert window.quickbar.context.currentIndex() == window.quickbar._context_pages["实体结果"]
    assert "独立于当前整体模型" in window.solid_panel.caption.text()
    window.actions_by_name["model_node"].trigger()
    assert window.mode == "模型" and window.viewport.model_mode == "node"
    window.show_solid_contour()
    assert window.viewport.model_mode is None
    window.clear_results()
    assert window.mode == "模型" and window._solid_grid is None


def test_analysis_enters_solid_view_and_model_edit_invalidates_it(app, files, monkeypatch):
    """实体分析完成后自动进入主视口；整体模型改变时旧实体结果必须退出。"""
    _directory, summary = files
    session = build_session()
    assert session.solve_model().ok
    window = MainWindow(session)
    monkeypatch.setattr(window.runner, "submit", lambda fn, **kw: kw["on_done"](fn()))
    window._analyse("solid_joint", "节点实体", lambda: ToolResult(True, summary))
    assert window.mode == "实体云图" and window._solid_grid is not None
    assert window._solid_binding[0] is session.solution
    assert "独立于当前整体模型" not in window.solid_panel.caption.text()
    assert session.add_nodes([[50, 50, 50]]).ok
    window.refresh()
    assert window._solid_grid is None and window.mode == "模型"
    assert "实体结果已失效" in window.solid_panel.caption.text()
    assert window.results.table.rowCount() == 0


def test_failed_import_and_cancel_preserve_view(app, files, monkeypatch):
    """损坏文件或取消打开不能覆盖有效结果，也不能错误点亮实体模式。"""
    directory, _summary = files
    window = MainWindow()
    window._load_solid_result(directory / "summary.json")
    previous = window._solid_grid
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    assert not window._load_solid_result(directory / "missing.npz")
    assert window._solid_grid is previous and window.mode == "实体云图"
    window.clear_results()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    window.actions_by_name["solid_contour"].trigger()
    assert window.mode == "模型" and window.mode_actions["模型"].isChecked()


def test_window_cleanup_releases_native_objects_on_the_main_thread(app):
    """防止窗口与无父测试控件滞留到后台垃圾回收，造成全量回归原生退出。"""
    from threading import get_ident

    import shiboken6
    from conftest import _close_windows
    from PySide6.QtWidgets import QWidget

    window = MainWindow()
    standalone = QWidget()
    destroyed_on = []
    window.destroyed.connect(lambda: destroyed_on.append(get_ident()))
    standalone.destroyed.connect(lambda: destroyed_on.append(get_ident()))
    cleanup = _close_windows.__wrapped__()
    next(cleanup)
    with pytest.raises(StopIteration):
        next(cleanup)
    assert not shiboken6.isValid(window)
    assert not shiboken6.isValid(standalone)
    assert destroyed_on == [get_ident()] * 2
