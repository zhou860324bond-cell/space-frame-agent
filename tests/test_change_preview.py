"""Agent 变更预演：先看差异，当前 Session 必须纹丝不动。"""

from copy import deepcopy

from agent import Session
from change_preview import diff_models


def test_model_diff_reports_added_removed_and_changed_entities():
    before = {
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
        "materials": [{"name": "steel", "E": 2.0e11, "nu": 0.3}],
    }
    after = deepcopy(before)
    after["nodes"] = [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 1.0},
        {"id": 3, "x": 2.0, "y": 0.0, "z": 0.0},
    ]

    delta = diff_models(before, after)

    assert delta["changed"] is True
    assert delta["changed_top_level"] == ["nodes"]
    assert delta["collections"]["nodes"]["added"] == [3]
    assert delta["collections"]["nodes"]["removed"] == [2]
    assert delta["collections"]["nodes"]["changed"] == [1]


def test_preview_change_runs_on_a_copy_and_returns_a_versioned_diff():
    marker_frame = object()
    marker_solution = object()
    marker_db = object()
    session = Session(frame=marker_frame, solution=marker_solution,
                      result_db=marker_db)
    before_model = deepcopy(session.model)
    before_history = session.history.timeline()
    before_log = list(session.tool_log)

    result = session.preview_change("add_nodes", {
        "coordinates": [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
    })

    assert result.ok
    assert result.payload["schema"] == "agent-change-preview/v1"
    assert result.payload["tool"] == "add_nodes"
    assert result.payload["diff"]["collections"]["nodes"]["added"] == [1, 2]
    assert result.payload["would_invalidate_results"] is True
    assert session.model == before_model
    assert session.frame is marker_frame
    assert session.solution is marker_solution
    assert session.result_db is marker_db
    assert session.history.timeline() == before_history
    assert session.tool_log == before_log


def test_preview_rejects_read_only_unknown_and_nested_preview_tools():
    session = Session()

    for tool in ("validate_model", "missing_tool", "preview_change"):
        result = session.preview_change(tool, {})
        assert not result.ok
        assert "可预演" in result.payload["error"]


def test_failed_preview_preserves_the_previewed_error_and_current_state():
    session = Session()

    result = session.preview_change("add_nodes", {"coordinates": [[1.0, 2.0]]})

    assert not result.ok
    assert result.payload["schema"] == "agent-change-preview/v1"
    assert result.payload["tool"] == "add_nodes"
    assert "坐标" in result.payload["preview_error"]
    assert session.model == {}


def _small_frame() -> Session:
    session = Session()
    assert session.add_nodes([[0, 0, 0], [6, 0, 0]]).ok
    assert session.add_members([[1, 2]]).ok
    return session


def test_agent_dispatch_cannot_delete_without_a_confirmed_preview():
    session = _small_frame()

    denied = session.dispatch("remove_members", {"ids": [1]})

    assert not denied.ok
    assert denied.payload["confirmation_required"] is True
    assert len(session.model["members"]) == 1


def test_set_model_cannot_bypass_delete_confirmation():
    session = _small_frame()
    replacement = deepcopy(session.model)
    replacement["members"] = []

    denied = session.dispatch("set_model", {"model": replacement})

    assert not denied.ok
    assert denied.payload["confirmation_required"] is True
    assert len(session.model["members"]) == 1


def test_preview_must_be_confirmed_by_a_later_user_message_before_apply():
    session = _small_frame()
    preview = session.dispatch("preview_change", {
        "tool": "remove_members", "arguments": {"ids": [1]},
    })
    preview_id = preview.payload["preview_id"]

    denied = session.dispatch("apply_preview", {"preview_id": preview_id})
    assert not denied.ok
    assert denied.payload["confirmation_required"] is True
    assert len(session.model["members"]) == 1

    assert session.receive_user_confirmation("确认") is True
    applied = session.dispatch("apply_preview", {"preview_id": preview_id})
    assert applied.ok
    assert applied.payload["applied_tool"] == "remove_members"
    assert session.model["members"] == []


def test_a_stale_preview_cannot_be_applied_to_a_changed_model():
    session = _small_frame()
    preview = session.preview_change("remove_members", {"ids": [1]})
    preview_id = preview.payload["preview_id"]
    assert session.receive_user_confirmation(f"确认 {preview_id}")
    assert session.add_nodes([[3, 0, 0]]).ok

    result = session.apply_preview(preview_id)

    assert not result.ok
    assert "失效" in result.payload["error"] or "待确认" in result.payload["error"]


def test_cancelling_a_preview_prevents_a_later_unrelated_confirmation():
    session = _small_frame()
    preview = session.preview_change("remove_members", {"ids": [1]})

    assert session.receive_user_confirmation("取消") is False
    assert session.pending_change is None
    assert session.receive_user_confirmation("确认") is False
    result = session.apply_preview(preview.payload["preview_id"])
    assert not result.ok
    assert len(session.model["members"]) == 1


# ---------------- 批量修改闸门 ----------------
# 评测 P02：批量换柱截面，三轮里有一轮没先预演就直接改了。原先代码只对删除
# 强制预演，现在对「本轮之前已有实体」的累计修改达到 2 个也强制。

_MS = {"materials": [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}],
       "sections": [{"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
                    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}]}


def _existing_frame():
    """两跨单层平面刚架：杆 1-3 为柱、4-5 为梁。建完再开始一轮用户消息。"""
    s = Session()
    s.dispatch("define_materials_and_sections", _MS)
    s.dispatch("generate_frame", {"spans": [6, 6], "storeys": [4], "column_section": "COLUMN",
                                  "beam_section": "BEAM", "material": "STEEL", "base": "fixed"})
    s.begin_user_turn("把三根柱子的截面全部换成 BEAM")
    return s


def _sections(s):
    return {m["id"]: m["section"] for m in s.model["members"]}


def test_batch_property_change_on_existing_members_requires_preview():
    s = _existing_frame()
    before = deepcopy(s.model)
    r = s.dispatch("assign_properties", {"member_ids": [1, 2, 3], "section": "BEAM",
                                         "material": "STEEL"})
    assert not r.ok and r.payload["confirmation_required"]
    assert r.payload["required_tool"] == "preview_change"
    assert sorted(r.payload["would_modify"]) == ["member 1", "member 2", "member 3"]
    assert s.model == before


def test_splitting_a_batch_into_single_edits_is_still_caught():
    """三次 edit_member 各改一根——按单次计数就绕过去了，所以按本轮累计。"""
    s = _existing_frame()
    first = s.dispatch("edit_member", {"member_id": 1, "section": "BEAM"})
    assert first.ok
    second = s.dispatch("edit_member", {"member_id": 2, "section": "BEAM"})
    assert not second.ok and second.payload["confirmation_required"]
    # 第一根已经改了，回包必须点名，好让模型如实告诉用户
    assert second.payload["already_modified_this_turn"] == ["member 1"]
    assert _sections(s)[2] == "COLUMN"


def test_single_edit_of_an_existing_member_is_not_gated():
    s = _existing_frame()
    assert s.dispatch("edit_member", {"member_id": 4, "section": "COLUMN"}).ok
    assert _sections(s)[4] == "COLUMN"


def test_entities_created_in_the_same_turn_can_be_edited_freely():
    """「先生成几何、再指派截面」是正常建模路径，不能被闸门拦住。"""
    s = Session()
    s.begin_user_turn("建一个两跨框架")
    s.dispatch("define_materials_and_sections", _MS)
    s.dispatch("generate_frame", {"spans": [6, 6], "storeys": [4], "base": "fixed"})
    r = s.dispatch("assign_properties", {"member_ids": [1, 2, 3, 4, 5], "section": "BEAM",
                                         "material": "STEEL"})
    assert r.ok, r.payload


def test_gate_is_off_outside_a_conversation_turn():
    """界面与离线脚本直接 dispatch、不经过用户消息——那里没有「模型自作主张」的问题。"""
    s = Session()
    s.dispatch("define_materials_and_sections", _MS)
    s.dispatch("generate_frame", {"spans": [6, 6], "storeys": [4], "column_section": "COLUMN",
                                  "beam_section": "BEAM", "material": "STEEL", "base": "fixed"})
    assert s.dispatch("assign_properties", {"member_ids": [1, 2, 3], "section": "BEAM",
                                            "material": "STEEL"}).ok


def test_batch_change_goes_through_after_preview_and_confirmation():
    s = _existing_frame()
    args = {"member_ids": [1, 2, 3], "section": "BEAM", "material": "STEEL"}
    preview = s.dispatch("preview_change", {"tool": "assign_properties", "arguments": args})
    assert preview.ok
    pid = preview.payload["preview_id"]
    assert not s.dispatch("apply_preview", {"preview_id": pid}).ok   # 同一轮不许自批
    assert s.begin_user_turn("确认") is True
    applied = s.dispatch("apply_preview", {"preview_id": pid})
    assert applied.ok, applied.payload
    assert _sections(s) == {1: "BEAM", 2: "BEAM", 3: "BEAM", 4: "BEAM", 5: "BEAM"}


def test_editing_loads_is_not_treated_as_a_batch_modification():
    """荷载本来就是整体重写，把它也拦住，每次调荷载都要确认一遍，闸门就成了噪声。"""
    s = _existing_frame()
    r = s.dispatch("set_load_cases", {"cases": [{"name": "DL", "member_loads": [
        {"member": m, "w": [0, 0, -20e3]} for m in (4, 5)]}]})
    assert r.ok, r.payload
