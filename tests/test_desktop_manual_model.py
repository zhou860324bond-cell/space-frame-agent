"""人工建模精确化的验证。

覆盖这次深挖人工建模时修的几件事：

1. 早期只有节点、材料/截面/杆件还没齐时，正式 preview_frame 会报“不合法”，
   以前视口会被清空——刚点的节点看不见、也点不中，根本没法连杆件。
   scene.draft_frame 宽松装配，只装几何，保证这一阶段可见、可拾取。
2. 没有杆件时 member_polylines/tubes 不能越界或抛“空 mesh”。
3. 工作平面投影、网格捕捉、已有节点吸附——鼠标点选精度的核心。
4. 主窗口早期人工建模链路：点节点→草稿帧可见→点两节点建杆件→Esc 退出。
"""

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="未安装 pyvista，跳过")
pv.OFF_SCREEN = True

from desktop import scene                                   # noqa: E402


# ---------------------------------------------------------- draft_frame

def test_draft_frame_only_nodes():
    model = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                       {"id": 2, "x": 5, "y": 0, "z": 0}],
             "members": [], "materials": [], "sections": [], "supports": []}
    f = scene.draft_frame(model)
    assert set(f.nodes) == {1, 2}
    assert f.members == {}


def test_draft_frame_skips_member_with_missing_endpoint():
    # 杆件引用了还没建出来的节点 9，应跳过而不是抛错
    model = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}],
             "members": [{"id": 1, "i": 1, "j": 9}], "supports": []}
    f = scene.draft_frame(model)
    assert f.members == {}
    # 把缺的端点补上后，杆件保留
    model["nodes"].append({"id": 9, "x": 3, "y": 0, "z": 0})
    f = scene.draft_frame(model)
    assert set(f.members) == {1}


def test_draft_frame_tolerates_garbage_and_missing_keys():
    # 缺键、空模型、坏数据都不崩
    assert scene.draft_frame({}).nodes == {}
    model = {"nodes": [{"id": "x"}, {"id": 2, "x": 1, "y": 2, "z": 3}],
             "members": [{"bad": 1}]}
    f = scene.draft_frame(model)
    assert set(f.nodes) == {2}


def test_empty_members_do_not_crash():
    f = scene.draft_frame({"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}]})
    lines = scene.member_polylines(f)
    assert lines.n_points == 0            # 以前这里 blocks[0] 越界
    tubes = scene.member_tubes(f)
    assert tubes.n_cells == 0


# ---------------------------------------------------------- 视口精确建模

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_work_plane_projection(qt_app):
    from desktop.viewport import Viewport
    vp = Viewport()
    vp.set_work_plane("XY", 2.0)
    assert np.allclose(vp._project_to_work_plane(np.array([1., 3., 9.])),
                       [1, 3, 2])
    vp.set_work_plane("XZ", -1.0)
    assert np.allclose(vp._project_to_work_plane(np.array([1., 3., 9.])),
                       [1, -1, 9])
    vp.set_work_plane("YZ", 0.5)
    assert np.allclose(vp._project_to_work_plane(np.array([1., 3., 9.])),
                       [0.5, 3, 9])


def test_grid_snap(qt_app):
    from desktop.viewport import Viewport
    vp = Viewport()
    vp.set_work_plane("XY", 0.0)
    vp.set_snap_size(0.5)
    p = vp._snap_point(np.array([0.26, 0.74, 9.0]))
    assert np.allclose(p, [0.5, 0.5, 0.0])
    vp.set_snap_size(0.0)                 # 关闭：只压平面，不取整
    p = vp._snap_point(np.array([2.41, -1.63, 9.0]))
    assert np.allclose(p, [2.41, -1.63, 0.0])


def test_snap_to_existing_node(qt_app):
    from agent import Session
    from desktop.viewport import Viewport
    s = Session()
    s.add_nodes([[0, 0, 0], [5, 0, 0], [5, 4, 0]])
    vp = Viewport()
    vp._frame = scene.draft_frame(s.model)
    assert vp._snap_to_existing_node(np.array([4.97, 4.02, 0.0])) == 3
    assert vp._snap_to_existing_node(np.array([100., 100, 100])) is None


def test_mode_badge_text_and_hide(qt_app):
    from desktop.viewport import Viewport
    vp = Viewport()
    vp.set_work_plane("XY", 0.0)
    vp.set_snap_size(0.5)
    vp.set_model_mode("node")
    txt = vp.mode_badge.text()
    assert "建节点" in txt and "XY" in txt and "0.5" in txt and "Esc" in txt
    assert not vp.mode_badge.isHidden()      # 父窗口未 show，用 isHidden 判断
    vp.set_model_mode(None)
    assert vp.mode_badge.isHidden()


def test_quickbar_sets_the_work_plane_axis_and_offset(qt_app):
    from desktop.main_window import MainWindow

    w = MainWindow()
    w.quickbar.plane.setCurrentText("XZ")
    w.quickbar.offset.setValue(2.75)

    assert w.viewport.work_plane == "XZ"
    assert w.viewport.work_offset == pytest.approx(2.75)
    assert w.quickbar.offset_label.text() == "y="
    assert np.allclose(w.viewport._snap_point([4.0, 99.0, 6.0]),
                       [4.0, 2.75, 6.0])
    w.close()


def test_exact_coordinate_reuses_an_existing_node_without_crashing(qt_app):
    from desktop.main_window import MainWindow

    w = MainWindow()
    w._create_node_at([1.0, 2.0, 3.0])
    before_history = len(w.session.history.steps)

    w._create_node_at([1.0, 2.0, 3.0])

    assert len(w.session.model["nodes"]) == 1
    assert len(w.session.history.steps) == before_history
    assert w._selected_kind == "node" and w._selected_id == 1
    assert "已有节点 1" in w.statusBar().currentMessage()
    w.close()


# ---------------------------------------------------------- 主窗口早期链路

def test_early_manual_modeling_flow(qt_app, monkeypatch):
    from conftest import opengl_available
    if not opengl_available():
        pytest.skip("无可用 OpenGL，跳过主窗口链路")
    import pyvista as pv
    pv.OFF_SCREEN = True
    from desktop.main_window import MainWindow
    w = MainWindow()
    vp = w.viewport
    # 建三个节点，z 给乱，验证被工作平面压到 0
    w.actions_by_name["model_node"].setChecked(True)
    w.set_model_node()
    vp._on_pick([0.0, 0.0, 7.0])
    vp._on_pick([5.0, 0.0, -3.0])
    vp._on_pick([5.0, 4.0, 9.0])
    assert len(w.session.model["nodes"]) == 3
    assert all(abs(n["z"]) < 1e-9 for n in w.session.model["nodes"])
    # 早期草稿帧必须持有节点，供拾取
    assert vp._frame is not None and len(vp._frame.nodes) == 3
    # 点两节点建杆件
    w.actions_by_name["model_node"].setChecked(False)
    w.set_model_node()
    w.actions_by_name["model_member"].setChecked(True)
    w.set_model_member()
    vp._on_pick([0.05, 0.02, 0.0])
    assert vp._pending_node == 1
    vp._on_pick([4.95, 0.03, 0.0])
    members = w.session.model.get("members", [])
    assert len(members) == 1 and members[0]["i"] == 1 and members[0]["j"] == 2

    # 再连一根，避免留下会让全局刚度矩阵出现零行的孤立节点。
    vp._on_pick([5.02, 0.01, 0.0])
    vp._on_pick([5.01, 3.98, 0.0])
    assert [(m["i"], m["j"]) for m in w.session.model["members"]] == [(1, 2), (2, 3)]

    # 走主窗口实际对话框入口，覆盖材料 → 截面 → 指派 → 支座 → 荷载。
    from PySide6.QtWidgets import QDialog
    from desktop import (assignment_dialog, bc_dialog, load_dialog,
                         material_dialog, section_dialog)

    class Accepted:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

    class Material(Accepted):
        def get_material(self):
            return {"name": "Q355", "E": 2.06e11, "nu": 0.3,
                    "density": 7850.0}

    class Section(Accepted):
        def get_section(self):
            return {"name": "B", "type": "矩形", "A": 0.02,
                    "Iy": 6e-4, "Iz": 8e-4, "J": 1e-5,
                    "cy": 0.1, "cz": 0.15, "circular": False}

    class Assignment(Accepted):
        def get_assignment(self):
            return "B", "Q355"

    class Boundary(Accepted):
        def get_fix(self):
            return [1, 1, 1, 1, 1, 1]

        def get_name(self):
            return "Fixed-1"

    class Load(Accepted):
        def get_load_type(self):
            return "concentrated"

        def get_load(self):
            return [0, 0, -1000, 0, 0, 0]

        def get_name(self):
            return "Tip-load"

        def get_case(self):
            return "Load-1"

    monkeypatch.setattr(material_dialog, "MaterialDialog", Material)
    monkeypatch.setattr(section_dialog, "SectionDialog", Section)
    monkeypatch.setattr(assignment_dialog, "AssignmentDialog", Assignment)
    monkeypatch.setattr(bc_dialog, "BCDialog", Boundary)
    monkeypatch.setattr(load_dialog, "LoadDialog", Load)

    w.create_material()
    w.create_section()
    for member_id in (1, 2):
        w.locate("member", member_id)
        w.assign_section()
    w.locate("node", 1)
    w.create_bc()
    w.locate("node", 3)
    w.create_load()

    assert w.session.validate_model().ok
    w.solve()
    assert w.runner.wait(30000), "人工建模后的求解没有在时限内完成"
    assert w.result is not None and w.result.ok
    assert w.mode == "变形" and w.case == "Load-1"
    assert "1 个工况" in w.lbl_solve.text()

    # 求解成功之后，结果必须能从视口落到可审查的数值表。
    w.probe_member_result(2, np.array([5.0, 2.0, 0.0]))
    assert w.results.table.rowCount() >= 12
    assert "结果探针" in w.results.caption.text()
    # Esc 退出
    w.cancel_interaction()
    assert not w.actions_by_name["model_member"].isChecked()
    assert vp.model_mode is None
    w.close()
