"""对话面板的 Agent 桌面端能力：流式、工具卡片、计划、改动审阅、持久化与输入增强。

用户验收时指出原面板"只是排版"：回答整段等到最后才出现，工具调用只有一行灰字，
改了模型没有地方撤销，关掉软件对话就没了。这里的每条测试对应其中一项真实缺口。
测试只用离线脚本和假密钥，配置与对话目录都在临时目录里。
"""
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from agent import ScriptedProvider, Session  # noqa: E402
from chat_store import ChatStore, new_record  # noqa: E402
from conversation import Conversation  # noqa: E402
from llm_stream import collect_anthropic_events, collect_openai_chunks  # noqa: E402


class StreamingScript(ScriptedProvider):
    """按脚本回放，并把文字拆成几段流式交出。"""

    def complete_stream(self, messages, tools, on_text):
        item = self.complete(messages, tools)
        text = item.get("content") or ""
        for start in range(0, len(text), 3):
            on_text(text[start:start + 3])
        return item


def _chunk(content=None, calls=None):
    delta = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def _call(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=arguments))


# --- 流式解析 ---

def test_openai_stream_rebuilds_split_tool_arguments():
    """工具参数按片到达；拼错一个字符就会把错误参数交给求解器。"""
    pieces = []
    out = collect_openai_chunks([
        _chunk("先建"), _chunk("模型"),
        _chunk(calls=[_call(0, "c1", "generate_frame", '{"spans": [6')]),
        _chunk(calls=[_call(0, None, None, ', 6], "storeys": 2}')]),
    ], pieces.append)
    assert "".join(pieces) == "先建模型"
    assert out["tool_calls"] == [{"id": "c1", "name": "generate_frame",
                                  "arguments": {"spans": [6, 6], "storeys": 2}}]


def test_anthropic_sse_parses_text_and_tool_use():
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"好的"}}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"solve_model"}}',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
    ]
    pieces = []
    out = collect_anthropic_events([line.encode() for line in lines], pieces.append)
    assert pieces == ["好的"]
    assert out["tool_calls"] == [{"id": "t1", "name": "solve_model", "arguments": {}}]


def test_anthropic_sse_reports_truncated_output():
    """截断的回答不能当作完整结论显示。"""
    with pytest.raises(RuntimeError, match="输出上限"):
        collect_anthropic_events(['data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}'],
                                 None)


# --- 对话事件与计划 ---

def test_conversation_emits_stream_tool_and_plan_events():
    """界面要的：文字边到边显示、每个工具有开始/结束与耗时、计划单独一张卡。"""
    script = StreamingScript([
        {"tool_calls": [{"id": "p", "name": "update_plan", "arguments": {"steps": [
            {"title": "校验模型", "status": "in_progress"}]}},
            {"id": "v", "name": "validate_model", "arguments": {}}]},
        {"content": "模型还没有几何。"}])
    events = []
    session = Session()
    conv = Conversation(script, session, plan_tool=True)
    result = conv.ask("检查一下当前状态好吗", on_event=lambda k, d: events.append((k, d)))
    kinds = [k for k, _ in events]
    assert kinds.count("round") == 2 and "plan" in kinds
    end = next(d for k, d in events if k == "tool_end")
    assert end["name"] == "validate_model" and end["ms"] >= 0 and end["summary"]
    assert "".join(d["delta"] for k, d in events if k == "text") == result.reply
    # 计划工具不进 Session，也不算一次工程工具调用
    assert [name for name, _ in result.tool_calls] == ["validate_model"]
    assert conv.plan == [{"title": "校验模型", "status": "in_progress"}]


def test_plan_tool_only_offered_when_enabled():
    """评测集与单轮调用的工具集不能被桌面功能悄悄改变。"""
    seen = []

    class Capture:
        def complete(self, messages, tools):
            seen.append({t["function"]["name"] for t in tools})
            return {"content": "好"}

    Conversation(Capture(), Session()).ask("你好呀朋友")
    Conversation(Capture(), Session(), plan_tool=True).ask("你好呀朋友")
    assert "update_plan" not in seen[0] and "update_plan" in seen[1]


def test_history_export_import_round_trip_and_drops_broken_pairs():
    conv = Conversation(ScriptedProvider([{"content": "第一轮答复"}]), Session())
    conv.ask("第一轮问题是什么")
    records = json.loads(json.dumps(conv.export_history(), ensure_ascii=False))
    broken = dict(records[0], messages=[{"role": "tool", "tool_call_id": "x", "content": "{}"}])
    restored = Conversation(ScriptedProvider([]), Session())
    restored.import_history(records + [broken])
    assert len(restored.history) == 1 and restored.history[0].reply == "第一轮答复"


# --- 对话落盘 ---

def test_chat_store_round_trip_search_rename_delete(tmp_path):
    store = ChatStore(tmp_path)
    record = new_record()
    record["entries"] = [{"type": "user", "text": "门式刚架 24 米"}]
    store.save(record, secrets=["sk-secret-value"])
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert [item["id"] for item in store.list()] == [record["id"]]
    assert store.search("24 米")[0]["id"] == record["id"] and not store.search("网壳")
    store.rename(record["id"], "  我的   刚架 ")
    assert store.load(record["id"])["title"] == "我的 刚架"
    store.delete(record["id"])
    assert store.list() == []


def test_chat_store_never_writes_keys(tmp_path):
    store = ChatStore(tmp_path)
    record = new_record()
    record["entries"] = [{"type": "user", "text": "我的密钥是 sk-secret-value"}]
    store.save(record, secrets=["sk-secret-value"])
    assert "sk-secret-value" not in (tmp_path / f"{record['id']}.json").read_text(encoding="utf-8")


def test_chat_store_rejects_path_like_ids(tmp_path):
    with pytest.raises(ValueError):
        ChatStore(tmp_path).load("../agent_models")


# --- 面板 ---

@pytest.fixture
def panel(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("FRAMELAB_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setattr("credentials.load_api_key", lambda *a, **k: None)
    from desktop.chat_panel import ChatPanel
    from desktop.worker import Runner
    widget = ChatPanel(Session(), Runner())
    widget.resize(420, 800)
    widget.show()
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


def test_tool_calls_become_cards_with_timing_and_change_card(panel):
    """工具调用曾只有一行灰字；改了模型也没有任何地方可以撤销这一轮。"""
    _turn(panel, "建个 24 米门式刚架并求解")
    entries = panel.record["entries"]
    reply = next(e for e in entries if e["type"] == "assistant")
    tools = [s for s in reply["segments"] if s["kind"] == "tool"]
    assert {"generate_portal_frame", "solve_model"} <= {s["name"] for s in tools}
    assert all(s["state"] == "ok" and s["ms"] is not None and s["summary"] for s in tools)
    change = next(e for e in entries if e["type"] == "change")
    assert any(line.startswith("杆件") for line in change["lines"])
    assert panel.session.model.get("members")


def test_revert_turn_restores_model_before_the_turn(panel):
    _turn(panel, "建个 24 米门式刚架并求解")
    index = next(i for i, e in enumerate(panel.record["entries"]) if e["type"] == "change")
    changed = []
    panel.model_changed.connect(changed.append)
    panel._entry_action("revert", index)
    assert panel.record["entries"][index]["status"] == "reverted"
    assert not panel.session.model.get("members") and changed == [True]


def test_revert_refused_after_later_manual_edit(panel):
    """本轮之后又手工改过模型时，整轮撤销会连带抹掉那次修改，必须拒绝。"""
    _turn(panel, "建个 24 米门式刚架并求解")
    index = next(i for i, e in enumerate(panel.record["entries"]) if e["type"] == "change")
    panel.session.model["nodes"][0]["x"] += 1.0
    panel._entry_action("revert", index)
    assert panel.record["entries"][index]["status"] == "stale"
    assert panel.session.model.get("members")


def test_regenerate_replaces_last_turn_without_duplicating_model(panel):
    _turn(panel, "建个 24 米门式刚架并求解")
    members = len(panel.session.model["members"])
    reply = next(i for i, e in enumerate(panel.record["entries"]) if e["type"] == "assistant")
    panel._entry_action("regenerate", reply)
    assert panel.runner.wait(60000)
    QApplication.processEvents()
    assert len(panel.session.model["members"]) == members
    assert [e["type"] for e in panel.record["entries"]].count("user") == 1
    assert len(panel.conversation.history) == 1


def test_streamed_text_matches_final_reply(panel, monkeypatch):
    """流式拼出的文字必须与最终回答一致，不能重复或缺字。"""
    script = StreamingScript([{"content": "这是一段流式回答，分几次到达。"}])
    monkeypatch.setattr(panel, "_make_provider", lambda: script)
    _turn(panel, "随便聊两句吧")
    reply = next(e for e in panel.record["entries"] if e["type"] == "assistant")
    texts = [s["text"] for s in reply["segments"] if s["kind"] == "text"]
    assert texts == ["这是一段流式回答，分几次到达。"]


def test_plan_card_rendered_from_plan_tool(panel, monkeypatch):
    script = StreamingScript([
        {"tool_calls": [{"id": "p", "name": "update_plan", "arguments": {"steps": [
            {"title": "建立几何", "status": "done"}, {"title": "求解", "status": "pending"}]}}]},
        {"content": "计划已列出。"}])
    monkeypatch.setattr(panel, "_make_provider", lambda: script)
    _turn(panel, "分步骤帮我建模")
    reply = next(e for e in panel.record["entries"] if e["type"] == "assistant")
    assert [s["title"] for s in reply["plan"]] == ["建立几何", "求解"]
    assert "✓ 建立几何" in panel.view.toPlainText()


def test_pending_batch_change_gets_confirm_card(panel, monkeypatch):
    """批量修改闸门原来只能靠用户手打"确认"；现在卡片上点一下发出同样的确认。"""
    sent = []
    _turn(panel, "建个 24 米门式刚架并求解")
    panel.session.pending_change = {"preview_id": "P1", "tool": "remove_members", "arguments": {},
                                    "diff": {"collections": {"members": {
                                        "before": 4, "after": 3, "added": [], "removed": [1],
                                        "changed": []}}}}
    panel._turn = {"before": panel.session.model, "signature": None, "pending": None, "cursor": 0}
    panel._review_changes()
    panel._turn = None
    index = next(i for i, e in enumerate(panel.record["entries"]) if e["type"] == "approval")
    assert "杆件：4 → 3；删除 1" in panel.record["entries"][index]["lines"]
    monkeypatch.setattr(panel, "send", lambda text=None: sent.append(text))
    panel._entry_action("approve", index)
    assert sent == ["确认 P1"]


def test_slash_commands(panel):
    panel.entry.setText("/帮助")
    panel.send()
    assert "/求解" in panel.view.toPlainText()
    panel.entry.setText("/不存在")
    panel.send()
    assert "没有命令 /不存在" in panel.view.toPlainText()
    assert panel._slash("/位移") == "最大位移"


def test_mention_adds_reference(panel, monkeypatch):
    """输入 @杆件3 与在选择窗口里勾选杆件 3 等价。"""
    panel.session.model.update({"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                                          {"id": 2, "x": 1, "y": 0, "z": 0}],
                                "members": [{"id": 3, "i": 1, "j": 2}]})
    assert "杆件3" in panel._mention_names()
    script = StreamingScript([{"content": "收到"}])
    monkeypatch.setattr(panel, "_make_provider", lambda: script)
    _turn(panel, "这根 @杆件3 是什么截面")
    assert ("member", 3) in panel.references
    assert "杆件 3" in panel.record["entries"][0]["refs"]


def test_chat_persists_and_restores_with_context(panel, monkeypatch, tmp_path):
    """关掉软件对话就没了：现在落盘，重新打开后模型仍能看到前文。"""
    _turn(panel, "建个 24 米门式刚架并求解")
    chat_id = panel.record["id"]
    panel.shutdown()
    from desktop.chat_panel import ChatPanel
    from desktop.worker import Runner
    other = ChatPanel(Session(), Runner())
    assert other.store.list()[0]["id"] == chat_id
    other.open_chat(chat_id)
    assert other.record["id"] == chat_id
    assert "generate_portal_frame" in other.view.toPlainText()
    assert len(other._ensure_conversation().history) == 1
    other.shutdown()
    other.close()


def test_dropped_drawing_goes_to_sketch_recognition(panel):
    got = []
    panel.image_dropped.connect(got.append)
    from PySide6.QtCore import QMimeData, QUrl
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("C:/图纸/刚架.png")])
    accepted = []
    event = SimpleNamespace(mimeData=lambda: mime, acceptProposedAction=lambda: accepted.append(1))
    panel.dragEnterEvent(event)
    panel.dropEvent(event)
    assert got == ["C:/图纸/刚架.png"] and len(accepted) == 2
    assert "草图识别" in panel.view.toPlainText()


def test_transcript_never_scrolls_sideways(panel):
    """长参数曾把卡片撑宽，右侧被截断。"""
    _turn(panel, "建个 24 米门式刚架并求解")
    assert panel.view.widget().width() <= panel.view.viewport().width()


def test_mention_completion_follows_ime_input(panel):
    """中文经输入法提交时不产生按键事件；补全只挂在 keyPressEvent 上时，
    键入「@杆」列表不会过滤，选中后也不会加入引用。"""
    from PySide6.QtGui import QInputMethodEvent
    from PySide6.QtTest import QTest
    panel.session.model.update({"nodes": [{"id": i, "x": i, "y": 0, "z": 0} for i in range(1, 4)],
                                "members": [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 3}]})
    panel.entry.setFocus()
    QApplication.processEvents()
    QTest.keyClicks(panel.entry, "@")
    event = QInputMethodEvent()
    event.setCommitString("杆")
    QApplication.sendEvent(panel.entry, event)
    assert panel.entry.completer.completionCount() == 2
    panel.entry.completer.activated.emit("@杆件2")
    assert panel.entry.toPlainText() == "@杆件2 "
    assert panel.references == [("member", 2)]
