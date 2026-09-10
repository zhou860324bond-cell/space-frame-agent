"""功能区、图标与动作表的验证。

界面层最容易出的不是崩溃，是**按了没反应**——处理函数名打错一个字，
按钮照样显示、照样能点，点下去弹一句"还没接上"。这种错误
自己测不出来，只能靠用户一个个点，所以这里把它变成测试：
**动作表里的每个处理函数都必须在主窗口上真的存在。**

图标同理：少一个图标不会崩（`icon()` 返回空的），但工具栏上会出现
一个没图案的按钮。也一并查掉。
"""

from __future__ import annotations

import os

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

from PySide6.QtWidgets import QApplication, QPushButton     # noqa: E402

from desktop import commands, icons, ribbon                 # noqa: E402
from desktop.main_window import MODES, MainWindow           # noqa: E402
from agent import Session                                   # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------- 动作表

def test_every_command_has_a_handler_that_exists(qt_app):
    """名字打错一个字就是一个死按钮，而且**点下去才知道**。"""
    w = MainWindow()
    missing = [(c.name, c.handler) for c in commands.COMMANDS
               if not callable(getattr(w, c.handler, None))]
    assert not missing, f"这些动作没有处理函数：{missing}"


def test_every_command_has_an_icon(qt_app):
    missing = [c.name for c in commands.COMMANDS
               if c.icon not in icons.DRAWERS]
    assert not missing, f"这些动作没有图标：{missing}"


def test_every_icon_actually_draws_something(qt_app):
    """空的 QIcon 不报错，但工具栏上是个没图案的按钮。"""
    for name in icons.names():
        pixmap = icons.icon(name, 32).pixmap(32, 32)
        assert not pixmap.isNull(), name
        image = pixmap.toImage()
        painted = sum(1 for y in range(0, 32, 2) for x in range(0, 32, 2)
                      if image.pixelColor(x, y).alpha() > 20)
        assert painted > 8, f"{name} 几乎是空白的（只有 {painted} 个像素）"


def test_an_unknown_icon_name_does_not_crash():
    """少一个图标是小事，开不了窗是大事。"""
    assert icons.icon("这个名字不存在").isNull()


def test_every_command_explains_itself():
    """每个按钮都要有说明——图标再好也有人认不出来。"""
    for c in commands.COMMANDS:
        assert c.label.strip() and c.tip.strip(), c.name
        assert len(c.tip) > 4, f"{c.name} 的说明太敷衍：{c.tip!r}"


# ------------------------------------------------- 功能区

def test_the_ribbon_pages_follow_the_workflow(qt_app):
    """页签顺序就是分析流程：建模 → 加荷载 → 算 → 看结果。
    从左往右点一遍正好走完一次完整分析，这是它的主要价值。"""
    w = MainWindow()
    titles = [w.ribbon.tabText(i) for i in range(w.ribbon.count())]
    assert titles == ["项目", "建模", "属性", "载荷", "分析", "结果", "视图"]


def test_each_module_explains_its_next_operation(qt_app):
    """模块页不只分类，还应告诉用户当前该做什么。"""
    w = MainWindow()
    for index in range(w.ribbon.count()):
        w.ribbon.setCurrentIndex(index)
        assert w.lbl_prompt.text().strip() and "就绪" not in w.lbl_prompt.text()


def test_the_ribbon_only_offers_what_we_actually_have(qt_app):
    """**不摆我们没有的功能。** 摆了就是骗人，答辩时一点就穿。"""
    w = MainWindow()
    titles = [w.ribbon.tabText(i) for i in range(w.ribbon.count())]
    for absent in ("网格", "接触", "非线性", "热分析", "疲劳"):
        assert absent not in titles, f"我们并没有{absent}"


def test_ribbon_buttons_share_the_window_actions(qt_app):
    """按钮不许自己存一份状态，否则会出现"菜单是灰的、功能区是亮的"。"""
    from PySide6.QtWidgets import QToolButton

    w = MainWindow()
    owned = set(w.actions_by_name.values())
    # 只看我们自己放进去的（`ribbon` 属性是 ribbon._button 打的标记）。
    # 不过滤的话会捞到 QTabBar 自带的翻页箭头——那也是 QToolButton
    buttons = [b for b in w.ribbon.findChildren(QToolButton)
               if b.property("ribbon")]
    assert len(buttons) >= 20, f"功能区里只找到 {len(buttons)} 个按钮"
    for b in buttons:
        assert b.defaultAction() in owned, b.text()


def test_workspace_bar_uses_one_compact_row_and_marks_primary_actions(qt_app):
    """Frequent controls use one compact row and key actions still stand out."""
    from PySide6.QtWidgets import QToolButton

    w = MainWindow()
    assert w.quickbar.objectName() == "workspaceBar"
    assert w.quickbar.layout().__class__.__name__ == "QHBoxLayout"
    primary = [button.defaultAction().objectName() for button in
               w.ribbon.findChildren(QToolButton)
               if button.property("role") == "primary"]
    assert set(primary) == {"sketch_ai", "solve"}


def test_empty_viewport_offers_real_starting_paths(qt_app):
    w = MainWindow()
    labels = {button.text() for button in w.empty_state.findChildren(QPushButton)}
    assert labels == {"AI 识别图纸", "参数化框架", "二维草图"}
    assert not w.empty_state.isHidden()
    assert w.tree_dock.isHidden()


def test_model_tree_returns_when_a_model_exists(qt_app):
    session = Session()
    assert session.add_nodes([[0, 0, 0], [1, 0, 0]]).ok
    w = MainWindow(session)
    assert not w.tree_dock.isHidden()


def test_workflow_action_opens_the_editor_for_the_actual_blocker(qt_app):
    """The persistent next action must route from backend facts, not a fixed tab."""
    session = Session()
    assert session.add_nodes([[0, 0, 0], [1, 0, 0]]).ok
    assert session.add_members([[1, 2]]).ok
    w = MainWindow(session)

    w.workflow_bar.next_button.click()

    assert w.ribbon.tabText(w.ribbon.currentIndex()) == "属性"
    assert "定义材料" in w.lbl_prompt.text()


def test_ribbon_does_not_repeat_actions_that_are_always_in_the_quickbar(qt_app):
    """常驻快捷栏已经提供的操作，不应再占一遍 Ribbon 空间。"""
    from PySide6.QtWidgets import QToolButton

    w = MainWindow()
    ribbon_actions = {button.defaultAction() for button in
                      w.ribbon.findChildren(QToolButton) if button.property("ribbon")}
    quick_actions = {button.defaultAction() for button in
                     w.quickbar.findChildren(QToolButton) if button.property("ribbon")}
    assert ribbon_actions.isdisjoint(quick_actions)


def test_the_menus_use_the_same_actions(qt_app):
    w = MainWindow()
    owned = set(w.actions_by_name.values())
    seen = 0
    assert w.menuBar().isHidden()
    assert w.menu_button.menu() is w.main_menu
    for menu in [w.main_menu, *w.main_menu.findChildren(type(w.main_menu))]:
        for act in menu.actions():
            if act.menu() is not None or act.parent() is getattr(w, "recent_menu", None):
                continue
            if not act.isSeparator() and act.text():
                assert act in owned, act.text()
                seen += 1
    assert seen > 15, "菜单里的动作太少，八成没建起来"


# ------------------------------------------------- 行为

def test_no_command_uses_agent_prefill():
    """功能区只提供已实现的直接操作，不能把示例话术伪装成功能。"""
    assert not [c.name for c in commands.COMMANDS if c.handler == "prefill"]


def test_the_display_modes_are_mutually_exclusive(qt_app):
    """看得见当前在哪个模式。变形图和模型图在小位移下长得很像，
    靠图猜是猜不出来的。"""
    w = MainWindow()
    for mode in MODES:
        assert mode in w.mode_actions, mode
    w.set_mode("模型")
    checked = [m for m, a in w.mode_actions.items() if a.isChecked()]
    assert checked == ["模型"]


def test_a_result_mode_reverts_when_there_is_no_result(qt_app, monkeypatch):
    """没有结果时点"云图"，按钮不能勾在云图上而视口画着模型——
    那是界面在说谎。"""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    w._run_command(commands.BY_NAME["contour"])
    assert w.mode == "模型"
    assert w.mode_actions["模型"].isChecked()
    assert not w.mode_actions["云图"].isChecked()


def test_solving_is_disabled_without_a_model(qt_app):
    w = MainWindow()
    assert not w.act_solve.isEnabled()


def test_a_missing_handler_says_so_instead_of_doing_nothing(qt_app, monkeypatch):
    """**按了没动静是最糟的交互**——用户分不清是没点中还是坏了。"""
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown.append(a))
    w = MainWindow()
    broken = commands.Command("x", "测试", "new", "说明说明", "根本不存在的方法")
    w._run_command(broken)
    assert shown


# ------------------------------------------------- 对话框边界

DIALOGS = {"frame", "portal", "tables", "json", "open", "save", "camera"}


def test_no_test_accidentally_opens_a_modal_dialog():
    """**测试永远不该触发模态对话框**：offscreen 环境下 `exec()` 不会返回，
    表现为整个套件卡死。所以凡是会开对话框的动作，测试里只能查它的接线，
    不能真去 `_run_command`。这条把边界写下来，免得下次又踩。
    """
    import inspect

    from desktop.main_window import MainWindow as MW

    for name in DIALOGS:
        cmd = commands.BY_NAME.get(name)
        if cmd is None:
            continue
        source = inspect.getsource(getattr(MW, cmd.handler))
        assert (".exec()" in source or "getOpenFileName" in source
                or "getSaveFileName" in source), \
            f"{name} 被列为对话框，但 {cmd.handler} 里看不到模态调用"
