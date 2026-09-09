"""MM2-07 structured preview, atomic commit, and sidecar tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent import Session
from change_preview import model_digest
from draft_commit import (DraftCommitError, commit_prepared, load_sidecar,
                          materialize_candidate, prepare_commit, save_sidecar,
                          sidecar_path)
from multimodal_workflow import MultimodalControllerState
from recognition_draft import DRAFT_FORMAT, migrate_to_v2


def ready_draft() -> dict:
    v1 = {
        "format": DRAFT_FORMAT, "scale": {"status": "unknown"},
        "model": {
            "units": "N-m-Pa", "materials": [], "sections": [],
            "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                      {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
            "members": [{"id": 1, "i": 1, "j": 2,
                         "material": "", "section": ""}],
            "supports": [], "load_cases": [],
        }, "entities": [], "questions": [], "warnings": [],
    }
    draft = migrate_to_v2(v1, image_hash="a" * 64, source_path="drawing.png")
    draft["source"]["preprocessing"]["perspective_status"] = "accepted"
    draft["work_plane"]["status"] = "confirmed"
    draft["scale"].update(status="confirmed", length_per_pixel=0.01)
    draft["model"] = deepcopy(v1["model"])
    draft["issues"] = []
    return draft


def armed_state(draft: dict, preview: dict) -> MultimodalControllerState:
    state = MultimodalControllerState()
    state.load_image("a" * 64)
    job = state.start_recognition("job")
    assert state.complete_recognition(job, draft)
    state.set_commit_preview(preview)
    return state


def test_prepare_is_pure_and_preview_is_structured():
    session = Session()
    before = deepcopy(session.model)
    prepared, preview = prepare_commit(ready_draft(), session.model)
    assert session.model == before
    assert prepared["merge_plan"]["mode"] == "replace_empty"
    assert [item["target_id"] for item in prepared["merge_plan"]["node_actions"]] == [1, 2]
    assert {item["kind"] for item in preview["operations"]} == {"node", "member"}
    assert "candidate_model" not in preview


def test_commit_is_one_undo_step_and_provenance_undoes_with_model():
    session = Session()
    prepared, preview = prepare_commit(ready_draft(), session.model)
    state = armed_state(prepared, preview)
    result = commit_prepared(session, state, confirmed_at="2026-09-08T00:00:00Z")
    assert result.ok
    assert len(session.history) == 1
    assert session.multimodal_provenance["image_hash"] == "a" * 64
    committed = deepcopy(session.model)
    assert session.undo() and session.model == {}
    assert session.multimodal_provenance is None
    assert session.redo() and session.model == committed
    assert session.multimodal_provenance["image_hash"] == "a" * 64


def test_stale_baseline_rejected_without_session_or_history_change():
    session = Session()
    prepared, preview = prepare_commit(ready_draft(), session.model)
    state = armed_state(prepared, preview)
    session.model = {"units": "N-m-Pa", "nodes": [{"id": 9, "x": 9, "y": 0, "z": 0}],
                     "members": []}
    before = deepcopy(session.model)
    with pytest.raises(DraftCommitError, match="陈旧|状态"):
        commit_prepared(session, state, confirmed_at="2026-09-08T00:00:00Z")
    assert session.model == before
    assert len(session.history) == 0
    assert session.multimodal_provenance is None


def test_session_failure_rolls_back_model_provenance_and_history():
    session = Session(model={"units": "N-m-Pa"})
    baseline_hash = model_digest(session.model)
    bad = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}],
           "members": [{"id": 1, "j": 1}], "supports": []}
    result = session.apply_multimodal_draft(
        bad, {"image_hash": "a" * 64}, expected_baseline_hash=baseline_hash)
    assert not result.ok
    assert session.model == {"units": "N-m-Pa"}
    assert session.multimodal_provenance is None
    assert len(session.history) == 0


def test_sidecar_round_trip_mismatch_and_legacy_absence(tmp_path):
    model_path = tmp_path / "frame.json"
    model = {"units": "N-m-Pa", "nodes": [], "members": []}
    provenance = {"image_hash": "a" * 64}
    written = save_sidecar(model_path, model, provenance)
    assert written == sidecar_path(model_path)
    loaded, warning = load_sidecar(model_path, model)
    assert loaded == provenance and warning is None
    loaded, warning = load_sidecar(model_path, {**model, "units": "N-mm-MPa"})
    assert loaded is None and "哈希不匹配" in warning
    assert load_sidecar(tmp_path / "legacy.json", model) == (None, None)


def test_add_only_can_reuse_one_node_and_remap_support_and_named_load():
    baseline = {
        "schema_version": 1, "units": "N-m-Pa", "materials": [], "sections": [],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "", "section": ""}],
        "supports": [], "load_cases": [],
    }
    draft = ready_draft()
    draft["model"]["nodes"] = [
        {"id": 1, "x": 1.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 2.0, "y": 0.0, "z": 0.0},
    ]
    draft["model"]["supports"] = [
        {"name": "new-end", "node": 2, "fix": [1, 1, 1, 0, 0, 0]}]
    draft["model"]["load_cases"] = [{
        "name": "LC1", "nodal_loads": [
            {"name": "P", "node": 2, "load": [0, -1, 0, 0, 0, 0]}]}]

    prepared, preview = prepare_commit(draft, baseline, node_reuse={1: 2})
    candidate = materialize_candidate(prepared, baseline)
    assert prepared["merge_plan"]["node_actions"] == [
        {"draft_id": 1, "action": "reuse", "target_id": 2},
        {"draft_id": 2, "action": "create", "target_id": 3},
    ]
    assert candidate["members"][-1].get("i") == 2
    assert candidate["members"][-1].get("j") == 3
    assert candidate["supports"][-1]["node"] == 3
    assert candidate["load_cases"][-1]["nodal_loads"][0]["node"] == 3
    assert any(item["op"] == "reuse" and item["kind"] == "node"
               for item in preview["operations"])
    assert any(item["kind"] == "load" and item["key"] == "LC1:nodal_loads:P"
               for item in preview["operations"])


def test_add_only_refuses_duplicate_member_created_by_reuse():
    baseline = {
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2}], "supports": []}
    draft = ready_draft()
    with pytest.raises(DraftCommitError, match="重复"):
        prepared, _ = prepare_commit(draft, baseline, node_reuse={1: 1, 2: 2})
        materialize_candidate(prepared, baseline)
