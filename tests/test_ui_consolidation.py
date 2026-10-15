"""界面收敛：同一件事只留一个入口，单击就能选、右键就有对象操作。

用户 2026-10-15 实机看完 CAE 重搭分支后的反馈是"很多都是重复冗余又不贴合人的操作"。
当时实测到的问题，每条对应下面一个测试：
- 流程有四套导航（模块页签、分析流程条、左侧窄栏、九阶段清单），提示有三处；
- 边界条件有三个入口，左侧旧表单与管理器存两份数据；
- 进视口要先按「选择杆件」才能点选，右键不先选中光标下的对象；
- 一百多个命令散在十一页里，知道名字却找不到按钮；
- 参数化建模对话框收起快捷指派后不缩回。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
pytest.importorskip("pyvistaqt", reason="未安装 pyvistaqt，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyvista as pv                                        # noqa: E402

pv.OFF_SCREEN = True

from conftest import opengl_available                       # noqa: E402
pytestmark = pytest.mark.skipif(                             # noqa: E402
    not opengl_available(),
    reason="无可用 OpenGL，桌面端 VTK 视口无法初始化，跳过")

from PySide6.QtWidgets import QApplication, QToolButton     # noqa: E402

from desktop.main_window import MainWindow                  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _surface_actions(w):
    """功能区各页、常驻工作条、页签行角落上挂着的 QAction（按出现次数）。"""
    seen = []
    for page in w.ribbon.pages.values():
        seen += [b.defaultAction() for b in page.findChildren(QToolButton) if b.defaultAction()]
    seen += [b.defaultAction() for b in w.quickbar.findChildren(QToolButton) if b.defaultAction()]
    seen += [b.defaultAction() for b in w.ribbon.corner.findChildren(QToolButton)
             if b.defaultAction()]
    return seen


def test_every_command_has_one_visible_entry(qt_app):
    """同一个命令在功能区、工作条和页签行上只能出现一次（右键菜单与命令面板不算）。"""
    w = MainWindow()
    actions = _surface_actions(w)
    names = {a: n for n, a in w.actions_by_name.items()}
    dupes = sorted({names.get(a, a.text()) for a in actions if actions.count(a) > 1})
    assert not dupes, f"这些命令有多个常驻入口：{dupes}"


def test_single_navigation_spine(qt_app):
    """不再有左右窄栏与整行流程条；流程收进页签行右端，模型栏常驻在视口左侧。"""
    w = MainWindow()
    assert not hasattr(w, "left_rail") and not hasattr(w, "right_rail")
    assert w.workflow_bar.parent() is not None
    assert w.ribbon.corner.isAncestorOf(w.workflow_bar)
    assert w.agent_splitter.widget(0) is w.model_panel
    assert w.agent_splitter.widget(w.agent_splitter.count() - 1) is w.chat
    assert w.ribbon.corner.isAncestorOf(w.agent_button)
    assert "modeling_workflow" not in {
        n for n, a in w.actions_by_name.items() if a in _surface_actions(w)}


def test_next_step_marked_on_one_tab(qt_app):
    """页签自己说明下一步在哪一页：空模型标建模，只标一个。"""
    w = MainWindow()
    marked = [t for t, p in w.ribbon.pages.items()
              if not w.ribbon.tabIcon(w.ribbon.indexOf(p)).isNull()]
    assert marked == ["建模"]


def test_old_bc_form_removed_and_settlement_kept(qt_app, monkeypatch):
    """旧边界表单删掉后，它独有的给定位移（支座沉降）必须仍能从「创建载荷」施加。"""
    from desktop.object_dialogs import LOAD_REGION, LOAD_TYPES
    from tools.render_smoke import build_session
    assert "settlement" in [v for _, v in LOAD_TYPES] and LOAD_REGION["settlement"] == "node"
    w = MainWindow(build_session())
    assert not hasattr(w, "bc") and not hasattr(w, "bc_dock")
    assert w.session.add_load_case("Settle").ok
    case = "Settle"
    supports = [int(s["node"]) for s in w.session.model.get("supports") or []][:2]
    w._finish_settlement("Settle", case, {"nodes": supports}, [0.0, 0.0, -0.01])
    entries = next(c for c in w.session.model["load_cases"] if c["name"] == case)["settlements"]
    assert sorted(e["node"] for e in entries) == sorted(supports)
    assert all(e["d"][2] == -0.01 for e in entries)


def test_click_selects_without_entering_pick_mode(qt_app, monkeypatch):
    """默认就能单击选中；点空白取消选择；节点优先于杆件。"""
    from tools.render_smoke import build_session
    w = MainWindow(build_session())
    vp = w.viewport
    assert vp.pick_mode is None and vp.model_mode is None
    nodes = np.array([[100.0, 100.0], [300.0, 100.0]])
    segments = np.array([[[100.0, 100.0], [300.0, 100.0]]])
    member = sorted(vp._frame.members)[0]
    monkeypatch.setattr(vp, "_screen_geometry", lambda: ([1, 2], nodes, [member], segments))
    assert vp.click_select(101, 101) == ("node", 1)
    assert w._selected_kind == "node" and w._selected_id == 1
    assert vp.click_select(200, 102)[0] == "member"
    assert vp.click_select(600, 600) is None
    assert w._selected_kind is None and vp.selection is None


def test_right_click_selects_object_under_cursor(qt_app, monkeypatch):
    """右键先选中光标下的对象，菜单才会给出指派截面、施加载荷这类针对它的操作。"""
    from PySide6.QtCore import QPoint
    from tools.render_smoke import build_session
    w = MainWindow(build_session())
    vp = w.viewport
    member = sorted(vp._frame.members)[0]
    monkeypatch.setattr(vp, "_screen_geometry", lambda: (
        [], np.zeros((0, 2)), [member], np.array([[[0.0, 50.0], [400.0, 50.0]]])))
    shown = []
    monkeypatch.setattr(w, "_show_viewport_context_menu", lambda pos: shown.append(pos))
    vp.context_requested.disconnect()
    vp.context_requested.connect(w._show_viewport_context_menu)
    vp._context_at(QPoint(200, 52))
    assert shown and w._selected_kind == "member"
    actions = w._viewport_context_menu().actions()
    assert w.actions_by_name["assign_section"] in actions
    assert w.actions_by_name["create_load"] in actions


def test_command_palette_finds_and_runs_commands(qt_app, monkeypatch):
    """命令面板：按中文或英文名找到命令，回车执行；找不到时可以交给 AI 助手。"""
    from PySide6.QtCore import Qt
    from desktop.command_palette import CommandPalette
    w = MainWindow()
    palette = CommandPalette(w)
    palette.search.setText("模型树")
    names = [palette.items.item(i).data(Qt.ItemDataRole.UserRole)
             for i in range(palette.items.count())]
    assert "tree_panel" in names and names[-1] == "__ask_ai__"
    palette.search.setText("model tree")
    assert palette.items.currentItem().data(Qt.ItemDataRole.UserRole) == "tree_panel"
    before = w.tree_dock.isVisible()
    palette._run_current()
    assert w.tree_dock.isVisible() != before
    sent = []
    monkeypatch.setattr(w.chat, "send", lambda text=None: sent.append(text))
    palette = CommandPalette(w)
    palette.search.setText("帮我把柱子换成 HN300")
    palette._run(palette.items.item(palette.items.count() - 1))
    assert sent == ["帮我把柱子换成 HN300"]


def test_parametric_dialog_shrinks_after_collapsing_quick_setup(qt_app):
    from agent import Session
    from desktop.dialogs import ParametricDialog
    dialog = ParametricDialog(Session())
    dialog.show()
    QApplication.processEvents()
    collapsed = dialog.height()
    dialog.quick_setup.setChecked(True)
    QApplication.processEvents()
    assert dialog.height() > collapsed
    dialog.quick_setup.setChecked(False)
    QApplication.processEvents()
    assert dialog.height() <= collapsed + 4
    dialog.close()


def test_direct_supports_can_be_converted_to_manageable_objects(qt_app, monkeypatch):
    """框架生成器写的是「直接定义」支座，边界条件管理器看不见；旧表单删掉后
    它们曾无处可改。现在点模型树里的直接支座，可以一次转成边界条件对象。"""
    from PySide6.QtWidgets import QMessageBox
    from tools.render_smoke import build_session
    from agent import Session
    if not hasattr(Session, "convert_direct_supports"):
        pytest.skip("等待模型层的 Session.convert_direct_supports 合入")
    w = MainWindow(build_session())
    before = {(int(s["node"]), tuple(s["fix"])) for s in w.direct_supports()}
    assert before, "示例模型应带直接定义的支座"
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert w.convert_direct_supports()
    assert not w.direct_supports()
    owned = {(int(s["node"]), tuple(s["fix"])) for s in w.session.model["supports"]}
    assert owned == before, "约束方向与节点必须原样保留"
    assert w.session.model["features"]["bcs"]


def test_clicking_a_member_before_materials_still_identifies_it(qt_app):
    """实机发现：只有几何时点中杆件，属性区仍显示「未选中对象」，像是点选没生效。"""
    w = MainWindow()
    assert w.session.generate_frame(spans=[6.0], storeys=[3.0], bays=[6.0]).ok
    w.refresh()
    member = int(w.session.model["members"][0]["id"])
    w._on_picked("member", member)
    assert w.properties.title.text().startswith(f"杆件 {member}")
    assert "材料" in w.properties.hint.text()
