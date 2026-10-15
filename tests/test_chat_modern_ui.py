"""对话面板的现代化排版与交互。

用户反馈对话面板"太古老"：顶栏是大标题加一排文字按钮、再一行整宽下拉框，
像一张表单；记录区套着边框；每条消息上常驻「复制」「编辑」文字；工具调用前面
是 ›√ 字符，一轮五六次调用把回答挤出屏幕；发送键是「发送 / 思考中…」文字，
运行时按钮变宽。这里守住改后的排版与交互，不让它退回去。
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QFrame  # noqa: E402

from agent import Session  # noqa: E402


@pytest.fixture()
def panel(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("FRAMELAB_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setattr("credentials.load_api_key", lambda *a, **k: None)
    from desktop.chat_panel import ChatPanel
    from desktop.worker import Runner
    widget = ChatPanel(Session(), Runner())
    widget.resize(420, 800)
    widget.show()
    app.processEvents()
    yield widget
    assert widget.runner.wait(30000)
    widget.shutdown()
    widget.close()
    app.processEvents()


def _turn(panel, text):
    panel.entry.setText(text)
    panel.send()
    assert panel.runner.wait(60000)
    QApplication.processEvents()


def _assistant_widget(panel, last=True):
    widgets = [w for w, e in zip(panel.view.widgets, panel.view.entries, strict=True) if e["type"] == "assistant"]
    return widgets[-1] if last else widgets[0]


def test_consecutive_tool_calls_fold_into_one_card_with_the_call_chain(panel):
    """一轮建模五六次调用，每次一张卡片会把回答正文挤到屏幕外。

    连续调用收进一张卡片；收起时标题仍写明调用链与成败，数值依据一眼可见。
    """
    from desktop.chat_widgets import ToolGroup
    _turn(panel, "建个 24 米门式刚架并求解")
    widget = _assistant_widget(panel)
    groups = widget.findChildren(ToolGroup)
    assert len(groups) == 1
    group = groups[0]
    tools = [s for s in widget.entry["segments"] if s["kind"] == "tool"]
    assert len(group.cards) == len(tools) >= 2
    assert "生成门式刚架" in group.header.detail.full_text()
    assert "求解" in group.header.detail.full_text()
    assert group.header.status.state() == "ok"
    assert f"{len(tools)} 次工具调用" in group.header.name.text()
    assert not group.body.isVisible(), "本轮结束后调用组应收起"
    group.header.click()
    assert group.body.isVisible()


def test_a_single_tool_call_stays_a_plain_card(panel):
    """只调用一次工具时不必再套一层分组。"""
    from desktop.chat_widgets import ToolCard, ToolGroup
    _turn(panel, "建个 24 米门式刚架并求解")
    _turn(panel, "/位移")
    widget = _assistant_widget(panel)
    assert not widget.findChildren(ToolGroup)
    assert len(widget.findChildren(ToolCard)) == 1


def test_a_failed_call_marks_the_group_as_partial():
    """组里有失败的调用时，收起的标题不能显示成全部成功。"""
    QApplication.instance() or QApplication([])
    from desktop.chat_widgets import ToolGroup
    group = ToolGroup()
    group.update_run([(0, {"kind": "tool", "name": "solve_model", "args": {}, "state": "ok", "ms": 5}),
                      (1, {"kind": "tool", "name": "query_results", "args": {}, "state": "failed", "ms": 1})],
                     live=False)
    assert group.header.status.state() == "partial"
    assert "1 次失败" in group.header.name.text()
    assert group.cards[1].header.status.state() == "failed"


def test_user_message_is_a_right_aligned_bubble_sized_to_its_text(panel):
    """用户消息原来和助手回答一样通栏铺满，分不清谁说的。

    短句气泡贴着文字收窄并靠右，长段落最多占一栏的 85%。
    """
    panel._append_user("最大位移")
    panel._append_user("请把" * 200)
    QApplication.processEvents()
    short, long = panel.view.widgets[-2], panel.view.widgets[-1]
    assert short.bubble.width() < short.width() * 0.5
    assert short.bubble.geometry().right() >= short.width() - 2, "气泡应靠右"
    assert long.bubble.width() <= long.width() * long.BUBBLE_RATIO + 1


def test_message_actions_are_icons_shown_on_hover_except_the_last_reply(panel):
    """「复制」「编辑」「重新生成」原来是每条消息上常驻的文字按钮。

    改成小图标：最后一条回答常显，其余消息鼠标移上去才显出。
    """
    _turn(panel, "建个 24 米门式刚架并求解")
    _turn(panel, "/位移")
    first, last = _assistant_widget(panel, last=False), _assistant_widget(panel)
    for widget in (first, last):
        assert not widget.buttons["copy"].icon().isNull()
        assert widget.buttons["copy"].text() == ""
    assert last._effect.opacity() == 1.0
    assert first._effect.opacity() == 0.0
    QApplication.sendEvent(first, QEvent(QEvent.Type.Enter))
    assert first._effect.opacity() == 1.0
    QApplication.sendEvent(first, QEvent(QEvent.Type.Leave))
    assert first._effect.opacity() == 0.0


def test_header_is_one_row_with_model_and_icon_buttons(panel):
    """顶栏原来是标题加文字按钮一行、整宽模型下拉再一行。

    现在模型名与四个图标按钮同一行，按钮名称和用途写在悬停提示里。
    """
    QApplication.processEvents()
    buttons = (panel.btn_history, panel.btn_new_chat, panel.btn_tasks, panel.btn_settings)
    tops = {panel.model_selector.mapTo(panel, panel.model_selector.rect().center()).y()}
    for button, name in zip(buttons, ("对话记录", "新建", "协作任务", "连接设置"), strict=True):
        assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
        assert not button.icon().isNull()
        assert button.toolTip().startswith(name)
        tops.add(button.mapTo(panel, button.rect().center()).y())
    assert max(tops) - min(tops) <= 4, "模型名与图标按钮应在同一行"
    assert panel.view.frameShape() == QFrame.Shape.NoFrame
    assert panel.view.geometry().top() < 60


def test_send_and_stop_share_one_round_button_slot(panel):
    """发送键原来运行时改成「思考中…」，按钮变宽把引用状态挤走。

    现在发送与停止是同一位置的圆形按钮，运行中换成停止键并获得焦点，按 Esc 即停。
    """
    from desktop.chat_panel import RoundButton
    assert isinstance(panel.btn_send, RoundButton) and isinstance(panel.btn_stop, RoundButton)
    size = panel.btn_send.size()
    panel.entry.setFocus()
    QApplication.processEvents()
    stopped = []
    panel.stop = lambda: stopped.append(True)
    panel.btn_stop.clicked.disconnect()
    panel.btn_stop.clicked.connect(panel.stop)
    panel._set_busy(True)
    QApplication.processEvents()
    assert panel.btn_stop.isVisible() and not panel.btn_send.isVisible()
    assert panel.btn_stop.size() == size
    assert panel.btn_stop.hasFocus()
    QApplication.sendEvent(panel.btn_stop, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                                      Qt.KeyboardModifier.NoModifier))
    assert stopped == [True]
    panel._set_busy(False)
    QApplication.processEvents()
    assert panel.btn_send.isVisible() and not panel.btn_stop.isVisible()
    assert panel.entry.hasFocus(), "结束后焦点回到输入框"


def test_up_arrow_in_an_empty_box_recalls_the_last_message(panel):
    """取回上一条消息只放回文字，不撤销那一轮，也不改对话记录。"""
    _turn(panel, "建个 24 米门式刚架并求解")
    count = len(panel.record["entries"])
    panel.entry.clear()
    QApplication.sendEvent(panel.entry, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up,
                                                  Qt.KeyboardModifier.NoModifier))
    assert panel.entry.toPlainText() == "建个 24 米门式刚架并求解"
    assert len(panel.record["entries"]) == count
    assert panel.session.model.get("members")


def test_streaming_reply_shows_typing_dots_until_first_content(panel):
    """请求发出到第一个字之间只有「思考中」三个字，看起来像卡住了。"""
    from desktop.chat_widgets import TypingDots
    panel._add_entry({"type": "assistant", "status": "streaming", "activity": "思考中…",
                      "segments": [], "plan": []})
    QApplication.processEvents()
    widget = panel.view.widgets[-1]
    dots = widget.findChildren(TypingDots)[0]
    assert dots.isVisible() and widget.busy_mark.isVisible()
    widget.entry["segments"].append({"kind": "text", "text": "好的"})
    widget.refresh()
    assert not dots.isVisible()
    widget.entry["status"] = "done"
    widget.refresh()
    assert not widget.busy_mark.isVisible()


def test_scroll_to_bottom_button_appears_after_scrolling_up(panel):
    """往上翻看旧消息后，要能一键回到最新消息。"""
    for i in range(30):
        panel._append_note(f"第 {i} 条说明：模型已更新，请核对后继续。", level="info")
    for _ in range(10):         # 布局请求是延迟处理的，多转几圈事件循环
        QApplication.processEvents()
    bar = panel.view.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(0)
    QApplication.processEvents()
    assert panel.view.btn_bottom.isVisible()
    panel.view.btn_bottom.click()
    QApplication.processEvents()
    assert bar.value() == bar.maximum() and not panel.view.btn_bottom.isVisible()


def test_history_is_grouped_by_day():
    """对话一多，平铺的列表找不到昨天那段；按今天、昨天、近 7 天、更早分组。"""
    from desktop.chat_panel import day_group
    now = time.time()
    today = time.localtime(now)
    assert day_group(time.localtime(now), today) == "今天"
    assert day_group(time.localtime(now - 86400), today) == "昨天"
    assert day_group(time.localtime(now - 3 * 86400), today) == "近 7 天"
    assert day_group(time.localtime(now - 30 * 86400), today) == "更早"


def test_history_list_shows_unselectable_day_headers(panel):
    _turn(panel, "建个 24 米门式刚架并求解")
    panel.show_history()
    first = panel.history_list.item(0)
    assert first.text() == "今天"
    assert first.flags() == Qt.ItemFlag.NoItemFlags
    assert panel.history_list.item(1).data(Qt.ItemDataRole.UserRole) == panel.record["id"]
