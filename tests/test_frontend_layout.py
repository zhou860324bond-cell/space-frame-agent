"""结果摘要与窄屏判读回归，防止精简界面隐藏校核信息或误报通过。"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                      # noqa: E402
from PySide6.QtWidgets import QApplication, QToolButton             # noqa: E402

from desktop import result_rows                                   # noqa: E402
from desktop.panels import ResultPanel                             # noqa: E402
from agent import Session                                          # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def payload(load=200e3, yield_stress=235e6):
    session = Session()
    material = {"name": "M", "E": 2e11, "nu": 0.3, "allow_tension": 100e6}
    if yield_stress is not None:
        material["yield_stress"] = yield_stress
    assert session.set_model(model={
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 0.5 if load < 0 else 5, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
        "materials": [material],
        "sections": [{"name": "S", "A": 4e-3, "Iy": 8e-6, "Iz": 8e-6,
                      "J": 1.6e-5, "cy": 0.05, "cz": 0.05}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 2, "fix": [0, 1, 1, 1, 1, 1]}],
        "load_cases": [{"name": "P", "nodal_loads": [{"node": 2, "load": [load, 0, 0, 0, 0, 0]}]}],
    }).ok
    assert session.solve_model().ok
    result = session.check_strength()
    assert result.ok
    return result.payload


def present(panel, data):
    title, columns, rows, locators = result_rows.to_rows("strength", data)
    panel.show_rows(title, columns, rows, locators,
                    result_rows.row_marks("strength", data),
                    overview=result_rows.strength_overview(data),
                    primary_columns={"杆件", "截面", "应力比 σ/[σ]", "折算比", "N/(φA)/f", "结论"})


@pytest.mark.parametrize("ratio", [0.99999, 1.00001])
def test_summary_verdict_uses_unrounded_tool_ratio(ratio):
    """真实轴力校核在阈值两侧显示值都约为 1，摘要不能用显示值重新判断。"""
    data = payload(ratio * 100e6 * 4e-3)
    cards = result_rows.strength_overview(data)
    assert cards[0]["value"] == "1"
    assert (cards[0]["state"] == "fail") == (ratio > 1)
    assert cards[0]["target"] == ("member", data["worst_strength"]["member"])


def test_inconclusive_compression_does_not_claim_every_member_passed():
    """材料缺少屈服应力时，粗短压杆未判定，标题不能仍宣称全部通过。"""
    data = payload(-200e3, yield_stress=None)
    assert data["inconclusive_members"] == [1]
    title, *_ = result_rows.to_rows("strength", data)
    assert "全部通过" not in title
    assert "可判定杆件均通过" in title
    assert result_rows.strength_overview(data)[3]["value"] == "1"


def test_key_columns_keep_shear_and_code_stability_visible(app):
    """精简结果列时必须同时保留剪切折算比和规范稳定比，避免只读正应力。"""
    panel = ResultPanel()
    present(panel, payload())
    visible = {panel.table.horizontalHeaderItem(i).text()
               for i in range(panel.table.columnCount()) if not panel.table.isColumnHidden(i)}
    assert visible == {"杆件", "截面", "应力比 σ/[σ]", "折算比", "N/(φA)/f", "结论"}
    panel.all_columns.setChecked(True)
    assert not any(panel.table.isColumnHidden(i) for i in range(14))
    panel.show_rows("位移", ["节点", "位移"], [[1, 0.1]])
    assert not panel.table.isColumnHidden(1)
    assert panel.all_columns.isHidden()
    assert panel.summary.isHidden()


def test_calculation_notes_preserve_tool_limits_and_errors_are_expanded(app):
    """折叠冗长说明不能丢失工具的适用限制；计算失败的下一步应直接可见。"""
    panel = ResultPanel()
    data = payload()
    present(panel, data)
    assert panel.notes.isHidden()
    assert data["limitation"].replace("**", "") in panel.notes.toPlainText()
    panel.details_button.click()
    assert not panel.notes.isHidden()
    panel.show_message("验算未完成", "材料参数不全，请补充屈服应力后重新验算。")
    assert not panel.notes.isHidden()
    assert "请补充屈服应力" in panel.notes.toPlainText()
    assert panel.summary.isHidden()


@pytest.mark.parametrize("load,yield_stress,card,mark", [
    (600e3, 235e6, 2, "fail"), (-200e3, None, 3, "unclear")])
def test_summary_links_use_tool_member_and_filter_after_sort(app, load, yield_stress, card, mark):
    """摘要定位和超限/未判定清单必须关联真实物理杆件，不能依赖表格行号。"""
    panel = ResultPanel()
    data = payload(load, yield_stress)
    present(panel, data)
    located = []
    panel.locate.connect(lambda *target: located.append(target))
    panel.table.sortItems(0, Qt.SortOrder.DescendingOrder)
    panel.summary_cards[0].findChild(QToolButton).click()
    assert located == [("member", 1)]
    panel.summary_cards[card].findChild(QToolButton).click()
    assert panel.row_filter.currentData() == mark
    assert not panel.table.isRowHidden(0)


def test_narrow_summary_keeps_metrics_above_readable_table(app):
    """窄屏摘要应换为两列，不能将指标挤成空白条。"""
    panel = ResultPanel()
    present(panel, payload())
    panel.resize(450, 420)
    panel.show()
    app.processEvents()
    assert panel.summary_grid.getItemPosition(2)[:2] == (1, 0), (panel.width(), panel.minimumSizeHint().width())
    assert all(60 <= card.height() < 92 for card in panel.summary_cards)
    assert panel.table.height() >= 60
    panel.close()


def test_result_overlay_follows_mode_badge_and_clears_with_scene(app):
    """拾取提示出现时不能压住云图摘要，切回模型也不能留下旧的峰值卡片。"""
    from desktop.viewport import Viewport
    v = Viewport()
    v.resize(800, 500)
    v.show()
    v._show_result_summary(dict(title="弯矩合量 M · kN·m", peak="真实峰值：69.2",
                               range_text="色标范围：0 ～ 40.4", status="95% 裁剪",
                               details="峰值来自完整结果。"))
    v.set_overlay_insets(240, 80, 180)
    v.pick_mode = "member"
    v._update_mode_badge()
    app.processEvents()
    assert v.result_overlay.x() >= 240
    assert v.result_overlay.geometry().right() < v.width() - 80
    assert v.result_overlay.y() > v.mode_badge.geometry().bottom()
    assert v.result_overlay.geometry().bottom() < v.height() - 180
    v.clear()
    assert v.result_overlay.isHidden()
    assert not v.result_overlay.active
    v.close()


@pytest.mark.parametrize("scale", [1.25, 1.5, 2.0])
def test_vector_icons_have_dpi_padding_and_a_distinct_disabled_state(app, scale):
    """分数缩放时不能复用模糊位图，支座剖面线与屈曲箭头也不能贴边被截断。"""
    from math import ceil
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon
    from desktop import icons

    for name in icons.names():
        original = icons.icon(name)
        copy = QIcon(original)
        copy.setIsMask(True)       # 强制 Qt 分离并复制引擎，不能留下 Python 临时对象指针。
        rendered = copy.pixmap(QSize(24, 24), scale)
        assert not original.isMask(), name
        assert rendered.toImage() == original.pixmap(QSize(24, 24), scale).toImage(), name
        assert rendered.width() == ceil(24 * scale), name
        assert rendered.devicePixelRatioF() == scale, name
        picture = rendered.toImage()
        edge = picture.width() - 1
        assert all(picture.pixelColor(x, y).alpha() == 0
                   for i in range(picture.width())
                   for x, y in ((0, i), (edge, i), (i, 0), (i, edge))), name
        disabled = icons.icon(name).pixmap(QSize(24, 24), scale, QIcon.Mode.Disabled).toImage()
        assert disabled != picture, name


def test_creation_and_selection_commands_have_distinct_semantic_icons(app):
    """草图、识别、空间框架、材料及创建/选择不能继续复用同一图案。"""
    from desktop import commands, icons
    by_name = {command.name: command for command in commands.COMMANDS}
    for group in (("sketch", "sketch_ai", "frame"),
                  ("material", "beam_section", "assign_section"),
                  ("model_node", "pick_node"), ("model_member", "pick_member"),
                  ("hinge", "create_bc"), ("analysis_mesh", "model_member")):
        images = [icons.icon(by_name[name].icon).pixmap(32, 32).toImage() for name in group]
        assert all(images[i] != images[j] for i in range(len(images))
                   for j in range(i)), group


def test_tooltips_wait_before_appearing_and_include_shortcut(app):
    """经过工具栏时提示不应立即闪现，工具名称与快捷键应同时可见。"""
    from PySide6.QtWidgets import QStyle, QStyleFactory
    from desktop.qt_style import FrameStyle
    from desktop.window_commands import WindowCommandsMixin
    from desktop.commands import COMMANDS
    from PySide6.QtWidgets import QWidget
    from types import SimpleNamespace
    style = FrameStyle(QStyleFactory.create("Fusion"))
    assert style.styleHint(QStyle.StyleHint.SH_ToolTip_WakeUpDelay) == 500
    class CommandsWindow(WindowCommandsMixin, QWidget):
        def _sync_selection_actions(self):
            pass

        def cancel_interaction(self):
            pass

    window = CommandsWindow()
    window.viewport = SimpleNamespace(load_labels=True)
    window.mode = "模型"
    window._build_actions()
    command = next(command for command in COMMANDS if command.name == "solve")
    assert window.actions_by_name["solve"].toolTip() == f"求解（F5）\n{command.tip}"
    assert window.actions_by_name["solve"].statusTip() == command.tip
    window.close()


def test_detaching_and_redocking_preserve_widget_input_and_page_order(app):
    """独立窗口必须搬动同一面板；往返不能丢失输入、乱页序或留下空抽屉。"""
    from PySide6.QtWidgets import QWidget, QLineEdit
    from desktop.drawers import DrawerHost, LEFT
    window = QWidget()
    window.resize(1000, 700)
    viewport = QWidget(window)
    viewport.setGeometry(40, 100, 900, 550)
    host = DrawerHost(window, viewport)
    drawer = host.drawer(LEFT, 300)
    field = QLineEdit("未提交的属性")
    handle = drawer.add_page("props", "属性", field)
    other = drawer.add_page("tree", "模型树", QWidget())
    window.show()
    handle.show()
    app.processEvents()
    container = drawer.stack.widget(0)
    handle.float_panel()
    app.processEvents()
    assert handle.isVisible() and not drawer.is_open()
    assert field.isVisibleTo(handle._window), "移出的原页面不能仍处于隐藏状态"
    assert container.height() > 100
    window.activateWindow()
    app.processEvents()
    handle.show()
    app.processEvents()
    assert app.activeWindow() is window, "已可见面板的属性刷新不能抢走视口焦点"
    assert field.text() == "未提交的属性"
    field.setText("保留修改")
    other.show()
    assert other.isVisible() and handle.isVisible()
    handle._window.close()
    assert not handle.isVisible()
    drawer.toggle_page("props")
    assert handle.isVisible() and other.isVisible()
    handle._window.dock_button.click()
    app.processEvents()
    assert handle.isVisible() and not other.isVisible()
    assert drawer.stack.widget(0) is container
    assert drawer.keys() == ["props", "tree"]
    assert field.text() == "保留修改"
    drawer.close()
    assert not handle.isVisible()
    window.close()


def test_resize_grip_can_restore_default_and_ignores_unchanged_extent(app):
    """误拖动面板边缘后应能双击恢复，重复鼠标位置不能反复触发布局。"""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QWidget
    from desktop.drawers import DrawerHost, LEFT, _EdgeGrip
    window = QWidget()
    window.resize(1000, 700)
    viewport = QWidget(window)
    viewport.resize(900, 550)
    host = DrawerHost(window, viewport)
    drawer = host.drawer(LEFT, 300)
    drawer.add_page("tree", "模型树", QWidget()).show()
    window.show()
    app.processEvents()
    drawer.set_extent(360)
    calls = []
    host.on_insets = lambda *args: calls.append(args)
    drawer.set_extent(360)
    assert not calls
    QTest.mouseDClick(drawer.findChild(_EdgeGrip), Qt.MouseButton.LeftButton, pos=QPoint(3, 100))
    assert drawer.extent == 300
    window.close()
