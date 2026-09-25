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

from PySide6.QtCore import Qt                               # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton     # noqa: E402

from desktop import commands, icons                 # noqa: E402
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

def test_panels_live_in_three_drawers_not_loose_windows(qt_app):
    """面板收进左、右、底三个抽屉，每侧一个，多个面板用页签切。

    上一版是九个各自漂着的浮窗：会压在功能区与快捷栏上把按钮挡住（实拍里
    模型树正好盖住「侧视/顶视」），拖走了找不回来。抽屉的位置由视口决定。
    """
    w = MainWindow()
    drawers = {w.left_drawer, w.right_drawer, w.bottom_drawer}
    assert set(w.drawers.drawers.values()) == drawers
    for name in w.PANELS:
        handle = getattr(w, name)
        assert handle._drawer in drawers, f"{name} 不在任何一个抽屉里"
        assert handle.windowTitle().strip(), f"{name} 没有标题，页签会是空的"


def test_opening_panels_never_shrinks_the_viewport(qt_app):
    """抽屉叠在视口上，**视口尺寸从头到尾不变**——包括三个抽屉同时开着。"""
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    qt_app.processEvents()
    before = (w.viewport.width(), w.viewport.height())
    for name in w.PANELS:
        getattr(w, name).setVisible(True)
        qt_app.processEvents()
        assert (w.viewport.width(), w.viewport.height()) == before, (
            f"开了 {name} 后视口从 {before} 变成了 "
            f"{(w.viewport.width(), w.viewport.height())}")
    w.close()


def test_drawers_are_tool_windows_owned_by_the_main_window(qt_app):
    """Tool 窗：浮在主窗之上、跟着主窗最小化、不在任务栏各占一格。"""
    w = MainWindow()
    for drawer in w.drawers.drawers.values():
        assert drawer.parent() is w, "脱离了主窗，就不会跟着主窗关闭"
        assert drawer.windowFlags() & Qt.WindowType.Tool, "不是 Tool 窗"


def test_drawers_stay_inside_the_viewport_and_never_overlap(qt_app):
    """抽屉永远在视口矩形之内，底部抽屉让开两侧的抽屉。

    这条是实拍抓到的：截面优化页的最小尺寸是 1200×390，QStackedWidget 取
    各页最小尺寸的最大值，于是 Qt 无视给定位置，把底部抽屉撑到盖住右抽屉、
    越出视口下沿。每页包一层滚动区之后才老实待在分给它的矩形里。
    """
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    w.chat_dock.show()
    w.tree_dock.show()
    w.section_opt_dock.show()           # 最小尺寸最大的那一页
    qt_app.processEvents()
    area = w.drawers.viewport_rect()
    left, right, bottom = (w.left_drawer.geometry(), w.right_drawer.geometry(),
                           w.bottom_drawer.geometry())
    for side, geometry in (("left", left), ("right", right), ("bottom", bottom)):
        assert area.contains(geometry), f"{side} 抽屉 {geometry} 越出了视口 {area}"
        assert geometry == w.drawers.geometry_for(side), (
            f"{side} 抽屉没待在分给它的位置：{geometry}")
    assert not bottom.intersects(left) and not bottom.intersects(right)
    w.close()


def test_drawers_never_cover_the_toolbars(qt_app):
    """抽屉的上沿就是视口的上沿——功能区、快捷栏、流程条上的按钮永远点得着。"""
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    w.tree_dock.show()
    w.chat_dock.show()
    qt_app.processEvents()
    top = w.drawers.viewport_rect().top()
    for drawer in (w.left_drawer, w.right_drawer):
        assert drawer.geometry().top() >= top
    bar = w.workflow_bar
    bar_bottom = bar.mapToGlobal(bar.rect().bottomLeft()).y()
    assert w.left_drawer.geometry().top() > bar_bottom
    w.close()


def test_one_button_on_the_right_summons_and_dismisses_the_assistant(qt_app):
    """右侧窄栏最上面那颗「AI」：一点召唤，再点收起；菜单里的开关同步。"""
    w = MainWindow()
    assert w.chat_dock.isHidden()
    w.agent_button.click()
    assert w.chat_dock.isVisible()
    assert w.agent_button.isChecked() and w.act_chat.isChecked()
    w.agent_button.click()
    assert w.chat_dock.isHidden()
    assert not w.agent_button.isChecked() and not w.act_chat.isChecked()


def test_rail_buttons_follow_the_real_drawer_state(qt_app):
    """抽屉被别的途径关掉（×、Esc、切到别的页）时，窄栏按钮不能还亮着。"""
    w = MainWindow()
    results = w.right_rail.buttons["results"]
    diagram = w.right_rail.buttons["diagram"]
    results.click()
    assert results.isChecked() and w.results_dock.isVisible()
    diagram.click()                      # 同一个抽屉换页
    assert diagram.isChecked() and not results.isChecked()
    w.bottom_drawer.close_button.click()
    assert not diagram.isChecked() and w.diagram_dock.isHidden()


def test_escape_folds_a_drawer_when_there_is_nothing_else_to_cancel(qt_app):
    w = MainWindow()
    w.results_dock.show()
    w.cancel_interaction()
    assert w.results_dock.isHidden()


def test_opening_a_drawer_moves_viewport_overlays_out_from_under_it(qt_app):
    """左上角模式提示、左下角坐标轴要让开被抽屉盖住的边。"""
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    qt_app.processEvents()
    w.tree_dock.show()
    qt_app.processEvents()
    left, _right, _bottom = w.drawers.insets()
    assert left == w.left_drawer.geometry().width()
    assert w.viewport._insets[0] == left
    w.tree_dock.hide()
    qt_app.processEvents()
    assert w.viewport._insets == (0, 0, 0)
    w.close()


def test_every_workspace_button_says_what_it_does(qt_app):
    """常驻工具栏的按钮必须带文字，纯图标只留给含义早已固化的那几个。

    原来这条栏是 18 个 20px 的无字线框图标（视角×5、选择×2、编辑×3、
    标注/属性/撤销/重做×4、显示×4）。不逐个悬停认不出任何一个——
    这是整个界面最劝退的一处。

    白名单只放撤销/重做和两个显示开关：⟲⟳ 是跨软件的通用约定，
    配字反而占地方。其余一律配字。
    """
    from PySide6.QtWidgets import QToolButton

    from PySide6.QtCore import Qt

    # 查的是**显示样式**，不是 b.text()。挂了 defaultAction 的 QToolButton，
    # b.text() 永远返回动作文本，跟屏幕上有没有字无关——按它断言等于没测。
    icon_only_ok = {"撤销", "重做", "编号标注", "属性"}
    w = MainWindow()
    naked = []
    for b in w.quickbar.findChildren(QToolButton):
        if not b.property("ribbon"):
            continue                      # 非命令按钮（如坐标建点）另算
        act = b.defaultAction()
        if act is None or act.text() in icon_only_ok:
            continue
        if b.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly:
            naked.append(act.text())
    assert len(naked) == 0, f"这些按钮只画了图标、屏幕上没有字：{naked}"
    # 白名单之外确实有一批按钮被查到，否则这条断言是空过的
    examined = [b for b in w.quickbar.findChildren(QToolButton)
                if b.property("ribbon") and b.defaultAction() is not None
                and b.defaultAction().text() not in icon_only_ok]
    assert len(examined) >= 12, f"只查到 {len(examined)} 个按钮，白名单开太大了"


def test_the_display_mode_is_visible_as_a_labelled_row(qt_app):
    """显示模式是**状态不是动作**，必须一眼看出当前在哪个。

    原来它和旁边的动作按钮长得一模一样——四个无字小图标，既看不出
    自己在看模型还是看云图，也容易误点。现在是一排带字的互斥按钮。
    """
    w = MainWindow()
    assert set(w.quickbar.mode_buttons) == set(MODES), (
        f"模式按钮与 MODES 不一致：{sorted(w.quickbar.mode_buttons)}")
    for label, button in w.quickbar.mode_buttons.items():
        assert button.text() == label, (
            f"模式按钮该显示短名 {label!r}，实际是 {button.text()!r}")
    w.set_mode("模型")
    assert w.quickbar.mode_buttons["模型"].defaultAction().isChecked()


def test_the_context_strip_follows_the_mode_before_the_page(qt_app):
    """中段控件按阶段切换，且**显示模式优先于功能区页**。

    刚建完模型就去看变形图、页签还停在"建模"，是很常见的路径。
    这时该给的是工况与放大倍数，不是"建节点/建杆件"。
    """
    w = MainWindow()
    build, result = 0, 1

    w.ribbon.setCurrentIndex(w.ribbon.indexOf(w.ribbon.pages["建模"]))
    assert w.quickbar.context.currentIndex() == build

    w.ribbon.setCurrentIndex(w.ribbon.indexOf(w.ribbon.pages["结果"]))
    assert w.quickbar.context.currentIndex() == result

    # 页签退回建模，但显示模式仍是变形 —— 模式说了算
    w.set_mode("变形")
    w.ribbon.setCurrentIndex(w.ribbon.indexOf(w.ribbon.pages["建模"]))
    assert w.quickbar.context.currentIndex() == result, (
        "正在看变形图时不该把中段换回建模控件")


def test_precise_modelling_controls_only_appear_while_placing(qt_app):
    """工作平面/捕捉/坐标建点只在真的要放点时出现。

    这一组实测占 425px，是整条工具栏原先放不下模式切换的直接原因；
    而不点节点的时候它一个都用不上——它们只影响"下一个点落在哪儿"。
    """
    w = MainWindow()
    assert w.quickbar.precise.isHidden(), "默认不该占着位置"
    w.actions_by_name["model_node"].setChecked(True)
    assert not w.quickbar.precise.isHidden(), "开始建节点了就该出现"
    w.actions_by_name["model_node"].setChecked(False)
    assert w.quickbar.precise.isHidden(), "退出建点应当收起"


def test_the_ribbon_can_be_collapsed_to_give_the_viewport_its_height_back(qt_app):
    """功能区可收起，但页签必须留着。

    顶部原来四层吃掉约 260px，在 1000px 高的窗口上是 26%，而功能区是其中
    最高、又最"查完就不用"的一层。收起时只收内容：页签同时是"我在哪个
    模块"的指示，一起藏掉是另一种反人类。
    """
    w = MainWindow()
    tall = w.ribbon.maximumHeight()
    assert not w.ribbon.is_collapsed()

    w.ribbon.toggle_collapsed()
    assert w.ribbon.is_collapsed()
    assert w.ribbon.maximumHeight() < tall
    assert w.ribbon.tabBar().isVisibleTo(w.ribbon), "页签不能跟着收掉"
    assert w.ribbon.count() == 7, "收起不该动页签本身"

    w.ribbon.toggle_collapsed()
    assert not w.ribbon.is_collapsed()
    assert w.ribbon.maximumHeight() == tall


def test_the_assistant_starts_out_of_the_way_but_reachable(qt_app):
    """AI 面板默认收起，右侧窄栏上的「AI」按钮是它的入口。

    这个面板固定占 400px、四分之一屏，只在想让 AI 帮忙建模时才用；
    没配密钥时开局还是一大段配置说明——第一眼看到的不该是一条错误。
    """
    w = MainWindow()
    assert w.chat_dock.isHidden(), "AI 面板不该默认占着四分之一屏"
    assert not w.agent_button.isHidden(), "收起后必须留下可见的入口"


def test_toolbar_inputs_fit_their_worst_case_text(qt_app):
    """常驻工具栏上的输入控件必须放得下它们**真会显示**的最长文本。

    这条闸拦的是"写死像素宽度"。工作平面的 z 输入框原先是
    `setFixedWidth(78)`，而它量程 ±1e6、3 位小数——实拍里 "0.000" 的最后
    一位被切掉、上下箭头压在数字上，显示成 "0.00("。

    界面裁切不崩、不报错、不影响任何计算，只会让人**读错数字**，
    所以自己测不出来，只能变成测试。

    比的是控件的最小宽度与 `_fits` 算出来的需求。两边用同一套字体度量，
    所以换字号、换 DPI、切英文都不会假红；而一旦有人改回硬编码像素，
    这里立刻就红。
    """
    from desktop.ribbon import _fits

    w = MainWindow()
    bar = w.quickbar
    worst = [
        # (控件, 它真会显示的最长文本)
        (bar.plane, "XZ"),
        (bar.offset, "-99999.999"),       # 工作平面位置，负值 + 3 位小数
        (bar.snap, "关闭"),
        (bar.scale, "自动"),
    ]
    for widget, sample in worst:
        need = _fits(widget, sample)
        assert widget.minimumWidth() >= need, (
            f"{widget.objectName() or type(widget).__name__} 最小宽度 "
            f"{widget.minimumWidth()} 放不下 {sample!r}（需要 {need}）——"
            "别写死像素，用 ribbon._fits 推导")


def test_fits_is_derived_from_the_widget_not_a_constant(qt_app):
    """`_fits` 必须随文本变长而变宽，否则它只是换了个地方的魔数。"""
    from desktop.ribbon import _fits

    w = MainWindow()
    short = _fits(w.quickbar.offset, "0")
    long = _fits(w.quickbar.offset, "-99999.999")
    assert long > short, "_fits 没有真的度量文本"


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


def test_switching_to_english_translates_drawer_tabs_and_rail_buttons(qt_app):
    """抽屉页签、单页抽屉标题与两侧窄栏都要跟着切语言。

    这些原来是停靠窗，语言切换按停靠窗翻标题；改成抽屉之后传进去的是
    PanelHandle（不是控件），什么都翻不到——切到英文后一屏中英混排，
    而且不报错，没人发现。
    """
    from desktop import i18n

    w = MainWindow()
    try:
        w.actions_by_name["lang"].setChecked(True)
        w.toggle_language()
        tabs = [w.left_drawer.tabs.tabText(i) for i in range(w.left_drawer.tabs.count())]
        assert tabs == ["Model Tree", "Property", "History"]
        assert w.right_rail.buttons["chat"].text() == "AI"
        assert w.left_rail.buttons["timeline"].text() == "Steps"
        w.actions_by_name["lang"].setChecked(False)
        w.toggle_language()
        assert w.left_drawer.tabs.tabText(0) == "模型树"
        assert w.right_rail.buttons["chat"].text() == "AI\n助手"
    finally:
        i18n.set_language("zh")


def test_closing_a_drawer_window_by_the_system_keeps_its_state_honest(qt_app):
    """Alt+F4 走 Qt 底层的关闭；抽屉的"开着"状态与窄栏按钮必须跟着变。"""
    from PySide6.QtGui import QCloseEvent

    w = MainWindow()
    w.results_dock.show()
    assert w.right_rail.buttons["results"].isChecked()
    event = QCloseEvent()
    w.bottom_drawer.closeEvent(event)
    assert not event.isAccepted()
    assert w.results_dock.isHidden()
    assert not w.right_rail.buttons["results"].isChecked()
