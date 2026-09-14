"""多轮对话的验证。

两件事要验，第二件才是难点：

1. **状态连续** —— 模型跨轮保留，"再算一遍"是在同一个刚架上改；
2. **裁剪不拆散工具调用配对** —— 接口的硬约束是每条 tool 结果必须紧跟在
   请求它的 assistant.tool_calls 之后，且一条 assistant 请求的全部结果都要到齐。
   朴素的"保留最后 N 条消息"必然拆散它，接口回 400 而且信息含糊。
   所以裁剪以**回合**为单位，这里逐条验那个不变量。
"""

import pytest

from agent import ScriptedProvider, Session
from conversation import (Conversation, check_pairing, tool_message_content,
                          tools_for_turn)

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
            {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}]


def call(cid, name, **args):
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def say(text):
    return {"content": text}


def build_script():
    """第一轮建模求解，第二轮只改荷载再算——第二轮**不重建模型**。"""
    return [
        call("a1", "define_materials_and_sections",
             materials=MATERIALS, sections=SECTIONS),
        call("a2", "generate_frame", spans=[6.0], storeys=[3.6],
             column_section="COLUMN", beam_section="BEAM", material="STEEL"),
        call("a3", "set_load_cases",
             cases=[{"name": "DL", "member_loads":
                     [{"member": 3, "w": [0, 0, -20e3]}]}]),
        call("a4", "solve_model"),
        call("a5", "query_results", what="max_displacement", case="DL"),
        say("第一轮：最大位移见上。"),
        # 第二轮：只重设荷载
        call("b1", "set_load_cases",
             cases=[{"name": "DL", "member_loads":
                     [{"member": 3, "w": [0, 0, -40e3]}]}]),
        call("b2", "solve_model"),
        call("b3", "query_results", what="max_displacement", case="DL"),
        say("第二轮：荷载翻倍后的位移见上。"),
    ]


# ------------------------------------------------- 状态连续

def test_the_model_survives_between_turns():
    """第二轮没有重新建模，却照样算得出来——这就是"多轮"的全部意义。"""
    chat = Conversation(ScriptedProvider(build_script()))
    chat.ask("三跨框架，梁上 20 kN/m")
    nodes_after_first = len(chat.session.model["nodes"])

    chat.ask("荷载改成 40 kN/m 再算一遍")
    assert len(chat.session.model["nodes"]) == nodes_after_first
    tools = [name for name, _ in chat.history[1].tool_calls]
    assert "generate_frame" not in tools, "第二轮不该重建模型"
    assert "solve_model" in tools


def test_doubling_the_load_doubles_the_displacement():
    """状态真的连上了才会有这个结果——重建一个模型的话数字对不上。"""
    chat = Conversation(ScriptedProvider(build_script()))
    chat.ask("建模")
    first = chat.session.query_results(what="max_displacement",
                                       case="DL").payload["magnitude_mm"]
    chat.ask("荷载翻倍")
    second = chat.session.query_results(what="max_displacement",
                                        case="DL").payload["magnitude_mm"]
    # query_results 报的是 6 位小数的显示值，2× 之后末位对不上是取整的事，
    # 不是物理的事——所以放到 1e-5
    assert second == pytest.approx(2 * first, rel=1e-5)


def test_the_second_turn_sees_the_first_turn():
    """第二轮发出去的 messages 里必须带着第一轮的内容。"""
    provider = ScriptedProvider(build_script())
    chat = Conversation(provider)
    chat.ask("第一个问题")
    before = len(provider.seen[-1])
    chat.ask("第二个问题")
    assert len(provider.seen[-1]) > before
    texts = [m.get("content") or "" for m in provider.seen[-1]]
    assert any("第一个问题" in t for t in texts)


def test_a_continuation_hint_appears_from_the_second_turn():
    """第一轮不该看到"你在多轮对话里"——单轮调用（评测集就是）
    看到它会以为存在并不存在的历史状态。"""
    provider = ScriptedProvider(build_script())
    chat = Conversation(provider)
    assert not any("多轮" in (m.get("content") or "")
                   for m in chat.build_messages("第一问") if m["role"] == "system")
    chat.ask("第一问")
    systems = [m["content"] for m in chat.build_messages("第二问")
               if m["role"] == "system"]
    assert any("多轮" in t for t in systems)


# ------------------------------------------------- 配对不变量

def test_pairing_checker_accepts_a_well_formed_history():
    chat = Conversation(ScriptedProvider(build_script()))
    chat.ask("建模")
    chat.ask("改荷载")
    assert check_pairing(chat.build_messages("第三问")) == []


def test_pairing_checker_catches_an_orphan_tool_result():
    """没有 assistant 请求就冒出来的 tool 结果。"""
    bad = [{"role": "user", "content": "x"},
           {"role": "tool", "tool_call_id": "z", "content": "{}"}]
    assert any("没有对应" in p for p in check_pairing(bad))


def test_pairing_checker_catches_a_missing_tool_result():
    """assistant 请求了两个工具，只回来一个。"""
    bad = [{"role": "assistant", "content": None,
            "tool_calls": [{"id": "a", "type": "function",
                            "function": {"name": "n", "arguments": "{}"}},
                           {"id": "b", "type": "function",
                            "function": {"name": "n", "arguments": "{}"}}]},
           {"role": "tool", "tool_call_id": "a", "content": "{}"}]
    assert any("未应答" in p for p in check_pairing(bad))


def test_pairing_checker_catches_an_interrupted_call():
    """工具结果还没回齐就插了一条 user 消息——朴素裁剪最常造出这种。"""
    bad = [{"role": "assistant", "content": None,
            "tool_calls": [{"id": "a", "type": "function",
                            "function": {"name": "n", "arguments": "{}"}}]},
           {"role": "user", "content": "打断"},
           {"role": "tool", "tool_call_id": "a", "content": "{}"}]
    assert any("打断" in p for p in check_pairing(bad))


def test_trimming_keeps_pairing_intact():
    """**这是这个模块最要紧的一条测试。**

    只保留一个回合，逼裁剪真正丢东西，然后逐条验配对不变量。
    如果裁剪从回合中间切开，这里立刻会红——而不是等接口回一个 400。
    """
    chat = Conversation(ScriptedProvider(build_script()), keep_turns=1)
    # 用系统提示里绝不会出现的词，否则"丢掉了第一轮"这条会被系统提示骗过去
    chat.ask("第一问 ZZZALPHA")
    chat.ask("第二问 ZZZBETA")
    messages = chat.build_messages("第三问")
    assert check_pairing(messages) == []
    texts = [m.get("content") or "" for m in messages]
    assert not any("ZZZALPHA" in t for t in texts), "第一轮应当被裁掉"
    assert any("ZZZBETA" in t for t in texts), "第二轮应当留着"


def test_trimming_never_cuts_inside_a_turn():
    """裁剪以回合为单位：留下的消息要么是某回合的全部，要么一条都不留。"""
    chat = Conversation(ScriptedProvider(build_script()), keep_turns=1)
    chat.ask("建模")
    chat.ask("改荷载")
    kept = chat.build_messages("第三问")
    last_turn = chat.history[-1].messages
    for m in last_turn:
        assert m in kept, "保留的那个回合必须整段都在"


def test_keep_turns_must_be_at_least_one():
    with pytest.raises(ValueError, match="至少"):
        Conversation(ScriptedProvider([]), keep_turns=0)


def test_context_budget_drops_old_complete_turns_without_breaking_pairs():
    chat = Conversation(ScriptedProvider(build_script()), context_chars=900)
    chat.ask("第一问 ZZZALPHA")
    chat.ask("第二问 ZZZBETA")
    messages = chat.build_messages("第三问")
    assert check_pairing(messages) == []
    texts = [m.get("content") or "" for m in messages]
    assert not any("ZZZALPHA" in text for text in texts)
    assert any("ZZZBETA" in text for text in texts)


def test_large_tool_result_is_compacted_to_valid_json():
    from agent import ToolResult
    import json

    result = ToolResult(True, {"stations": list(range(5000)), "peak": 4999})
    content = tool_message_content(result, max_chars=1000)
    decoded = json.loads(content)
    assert decoded["ok"] is True
    assert decoded["peak"] == 4999
    assert decoded["_context_note"]
    assert len(content) < len(result.to_json())


def test_next_step_uses_local_fast_path_without_calling_model():
    provider = ScriptedProvider([])
    chat = Conversation(provider)
    out = chat.ask("下一步")
    assert out.metrics["fast_path"] is True
    assert out.metrics["provider_calls"] == 0
    assert provider.seen == []
    assert "建立几何" in out.reply


def test_max_displacement_query_uses_local_fast_path():
    session = Session()
    session.define_materials_and_sections(MATERIALS, SECTIONS)
    session.generate_frame(spans=[6], storeys=[3.6], beam_load=20e3)
    assert session.solve_model().ok
    provider = ScriptedProvider([])
    out = Conversation(provider, session=session).ask("查看最大位移")
    assert out.metrics["fast_path"] is True
    assert provider.seen == []
    assert "mm" in out.reply
    assert [name for name, _ in out.tool_calls] == ["query_results"]


@pytest.mark.parametrize("prompt, expected_tool", [
    ("检查模型", "validate_model"),
    ("重新求解", "solve_model"),
    ("查看反力", "query_results"),
])
def test_common_engineering_commands_do_not_need_a_model_round_trip(
        prompt, expected_tool):
    session = Session()
    session.define_materials_and_sections(MATERIALS, SECTIONS)
    session.generate_frame(spans=[6], storeys=[3.6], beam_load=20e3)
    assert session.solve_model().ok
    provider = ScriptedProvider([])
    out = Conversation(provider, session=session).ask(prompt)
    assert out.metrics["provider_calls"] == 0
    assert provider.seen == []
    assert out.tool_calls[0][0] == expected_tool


def test_low_frequency_tool_schemas_are_loaded_only_for_matching_intent():
    ordinary = {tool["function"]["name"]
                for tool in tools_for_turn("检查当前模型并求解")}
    joint = {tool["function"]["name"]
             for tool in tools_for_turn("继续做节点实体分析")}
    assert "analyze_joint_solid" not in ordinary
    assert "analyze_joint_solid" in joint
    assert len(ordinary) < len(joint) <= 49


def test_conversation_uses_streaming_provider_when_available():
    class StreamingProvider:
        def complete(self, *_args):
            raise AssertionError("有流式接口时不应退回完整等待")

        def complete_stream(self, messages, tools, on_text):
            on_text("正在")
            on_text("完成")
            return {"content": "正在完成"}

    chunks = []
    out = Conversation(StreamingProvider()).ask(
        "给我一个简短说明", on_text=chunks.append)
    assert chunks == ["正在", "完成"]
    assert out.reply == "正在完成"
    assert out.metrics["tool_schema_count"] < out.metrics["all_tool_count"]


# ------------------------------------------------- 其他

def test_the_transcript_lists_the_tools_each_turn_used():
    chat = Conversation(ScriptedProvider(build_script()))
    chat.ask("建模")
    chat.ask("改荷载")
    log = chat.transcript()
    assert len(log) == 2
    assert "generate_frame" in log[0]["tools"]
    assert "generate_frame" not in log[1]["tools"]


def test_resetting_history_keeps_the_model():
    """"重新问一遍，但别推倒重建"——清对话不清模型。"""
    chat = Conversation(ScriptedProvider(build_script()))
    chat.ask("建模")
    nodes = len(chat.session.model["nodes"])
    chat.reset_history()
    assert chat.history == []
    assert len(chat.session.model["nodes"]) == nodes


def test_running_out_of_rounds_does_not_poison_the_next_turn():
    """轮数用尽那一回合里全是未收尾的工具调用，不能原样留进历史。"""
    endless = [call(f"c{k}", "validate_model") for k in range(20)]
    chat = Conversation(ScriptedProvider(endless), max_rounds=3)
    out = chat.ask("绕不出来的问题")
    assert out.stopped_by_limit
    assert check_pairing(chat.build_messages("下一问")) == []


def test_an_existing_session_can_be_handed_in():
    """界面上已经建好的模型，直接交给对话接着聊。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    chat = Conversation(ScriptedProvider([say("好的")]), session=s)
    chat.ask("在这个模型上继续")
    assert chat.session is s
    assert s.model["materials"] == MATERIALS


def test_workflow_state_is_refreshed_after_a_tool_call():
    provider = ScriptedProvider([
        call("n1", "add_nodes", coordinates=[[0.0, 0.0, 0.0]]),
        say("节点已建立，仍需补全模型。"),
    ])
    chat = Conversation(provider)

    chat.ask("先建立一个节点")

    first = [m for m in provider.seen[0] if m.get("name") == "workflow_state"]
    second = [m for m in provider.seen[1] if m.get("name") == "workflow_state"]
    assert len(first) == len(second) == 1
    assert '"phase": "empty"' in first[0]["content"]
    assert '"phase": "draft"' in second[0]["content"]
    assert '"nodes": 1' in second[0]["content"]


def test_the_agent_cannot_preview_and_self_confirm_in_one_user_turn():
    session = Session()
    assert session.add_nodes([[0, 0, 0], [6, 0, 0]]).ok
    assert session.add_members([[1, 2]]).ok
    preview = session.preview_change("remove_members", {"ids": [1]})
    preview_id = preview.payload["preview_id"]
    # 清除刚才仅为取得确定性 id 而创建的待确认项，模拟新的一轮。
    session.pending_change = None
    provider = ScriptedProvider([
        call("p1", "preview_change", tool="remove_members",
             arguments={"ids": [1]}),
        call("p2", "apply_preview", preview_id=preview_id),
        say("需要你确认后才能删除。"),
    ])

    Conversation(provider, session=session).ask("删掉这根杆")

    assert len(session.model["members"]) == 1
    assert any(not entry[2].ok for entry in session.tool_log
               if entry[0] == "apply_preview")


def test_a_new_exact_confirmation_message_authorizes_the_pending_preview():
    session = Session()
    assert session.add_nodes([[0, 0, 0], [6, 0, 0]]).ok
    assert session.add_members([[1, 2]]).ok
    preview = session.preview_change("remove_members", {"ids": [1]})
    preview_id = preview.payload["preview_id"]
    provider = ScriptedProvider([
        call("a1", "apply_preview", preview_id=preview_id),
        say("已按确认执行。"),
    ])

    Conversation(provider, session=session).ask("确认")

    assert session.model["members"] == []
