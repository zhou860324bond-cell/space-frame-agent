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
