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


def test_solid_display_controls_keep_raw_results_and_expose_average_and_palette(app, files):
    """防止默认色谱偏离Abaqus十二级Rainbow，或换配色与平均显示改写原应力。"""
    from desktop.solid_result import SolidResultPanel
    directory, _ = files
    grid, *_ = read_result(directory/"summary.json")
    raw = grid.cell_data["Mises_MPa"].copy()
    panel = SolidResultPanel()
    assert panel.options()["palette"] == "abaqus"
    from desktop.theme import solid_cmap
    from matplotlib.colors import to_rgba
    expected = ["#0000FF", "#005CFF", "#00B9FF", "#00FFE7", "#00FF8B", "#00FF2E",
                "#2EFF00", "#8BFF00", "#E7FF00", "#FFB900", "#FF5C00", "#FF0000"]
    cmap = solid_cmap()
    assert cmap.N == 12
    np.testing.assert_allclose(cmap(np.arange(12)), [to_rgba(color) for color in expected])
    assert panel.palette.findData("engineering") >= 0
    assert panel.options()["averaging"] is True
    panel.averaging.setChecked(False)
    panel.palette.setCurrentIndex(panel.palette.findData("viridis"))
    assert panel.options()["palette"] == "viridis"
    assert panel.options()["averaging"] is False
    assert grid.cell_data["Mises_MPa"] == pytest.approx(raw)
    assert grid.cell_data["Mises_MPa"][1] == 2000
    panel.bind([directory/'summary.json'],0,'示例',geometry={
        'algorithm':'g2-bezier-sweep','minimum_centerline_radius_mm':140.5})
    assert 'G2曲率连续' in panel.source.text() and '140.5 mm' in panel.source.text()
    panel.bind([directory/'summary.json'],0,'示例',geometry={
        'algorithm':'halfspace-miter+rolling-ball','applied_fillet_radius_mm':0.,'fillet_status':'kernel_unavailable'})
    assert '未形成圆角' in panel.source.text()
    panel.bind([directory/'summary.json'],0,'示例',geometry={'algorithm':'g2-bezier-sweep'})
    assert '几何记录无效' in panel.source.text()
    panel.bind([directory/'summary.json'],0,'示例',geometry={
        'algorithm':'column-host-butt-joint','host_member_id':1,'head_extension_mm':100.})
    assert '柱截面保持' in panel.source.text() and '梁端贴合' in panel.source.text()
    panel.bind([directory/'summary.json'],0,'示例',geometry={
        'algorithm':'column-host+constrained-rolling-ball','applied_fillet_radius_mm':4.5})
    assert '一体连续实体' in panel.source.text() and '4.5 mm' in panel.source.text()
    panel.close()


def test_solid_analysis_dialog_exposes_geometry_mesh_and_backend(app, monkeypatch):
    """防止用户只能接受自动粗网格，或所选连接几何和后端未进入分析参数。"""
    from PySide6.QtWidgets import QComboBox, QDialog, QDoubleSpinBox, QWidget
    host = QWidget()
    def choose(box):
        for combo in box.findChildren(QComboBox):
            if combo.findData("abaqus") >= 0:
                combo.setCurrentIndex(combo.findData("abaqus"))
            if combo.findData("intersection") >= 0:
                combo.setCurrentIndex(combo.findData("intersection"))
        box.findChild(QDoubleSpinBox).setValue(5)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog,"exec",choose)
    assert MainWindow._ask_joint_levels(host,2) == 3
    assert host._joint_analysis_options == {
        "backend":"abaqus", "geometry_mode":"intersection", "mesh_sizes_mm":[11.25,7.5,5.]}
    host.close()


def test_feature_outline_cannot_replace_the_stress_legend(app, files, monkeypatch):
    """防止截面轮廓成为最后一个图层后，MPa色标错误绑定轮廓数组或默认色谱。

    色标改由 Qt 绘制后，颜色显式取自 _solid_result 那一层的映射器；
    这里守住量程与分级仍然跟实体结果走，而不是跟最后加进来的轮廓线走。
    """
    import pyvista as pv
    from desktop import viewport
    view=viewport.Viewport()
    class LegendPlotter(viewport._NullPlotter):
        def __init__(self):
            super().__init__()
            self.actors={}
            self.last_mapper=None
            self.legend_range=None
            self.solid_options=None
            self.legend_options=None
        def add_mesh(self,dataset,**options):
            if options.get('name') == '_solid_result':
                self.solid_options=options
            mapper=pv.DataSetMapper(dataset=dataset)
            mapper.scalar_range=options.get('clim',(-1.,1.))
            actor=pv.Actor(mapper=mapper)
            if options.get('name') == '_solid_result':
                self.solid_actor=actor
            self.actors[options.get('name')]=actor
            self.last_mapper=mapper
            return actor
    plotter=LegendPlotter()
    view.plotter=plotter
    monkeypatch.setattr(viewport,'CAN_RENDER',True)
    monkeypatch.setattr(view,'_apply_view',lambda *args:None)
    monkeypatch.setattr(view,'_inset_camera',lambda:None)
    grid,*_=read_result(files[0]/'native_joint_1.npz')
    result=view.show_solid_result(grid,'节点65',show_edges=False)
    assert plotter.solid_options['cmap'].N == 12
    assert plotter.solid_options['n_colors'] == 12
    assert '_solid_feature_edges' in plotter.actors
    assert plotter.last_mapper.scalar_range == (-1.,1.)
    scale = view.color_scale
    assert scale.active and scale.banded, "Abaqus 十二级色谱的色标也要分级"
    assert scale.clim == pytest.approx(result['clim'])
    assert scale.clim[1] > 1000
    assert "MPa" in scale.title
    view.close()
