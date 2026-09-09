"""MM2-01 v2 draft migration and authoritative state tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from change_preview import model_digest
from multimodal_contract import canonical_digest
from multimodal_workflow import (MultimodalControllerState, MultimodalPhase,
                                 draft_digest, validate_v2_draft)
from recognition_draft import DRAFT_FORMAT, migrate_to_v2


def v1_payload() -> dict:
    return {
        "format": DRAFT_FORMAT,
        "scale": {"status": "unknown"},
        "model": {
            "units": "N-m-Pa", "materials": [], "sections": [],
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                      {"id": 2, "x": 6, "y": 0, "z": 0}],
            "members": [{"id": 1, "i": 1, "j": 2,
                         "material": "", "section": ""}],
            "supports": [], "load_cases": [],
        },
        "entities": [], "questions": [], "warnings": [],
    }


def review_draft() -> dict:
    return migrate_to_v2(v1_payload(), image_hash="a" * 64,
                         source_path="drawing.png")


def ready_draft(session_model: dict) -> tuple[dict, dict]:
    draft = review_draft()
    draft["source"]["preprocessing"]["perspective_status"] = "accepted"
    draft["work_plane"]["status"] = "confirmed"
    draft["scale"]["status"] = "confirmed"
    draft["scale"]["length_per_pixel"] = 0.01
    draft["model"] = {
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "material": "", "section": ""}],
        "supports": [], "materials": [], "sections": [],
    }
    draft["issues"] = []
    draft["merge_plan"] = {
        "mode": "replace_empty", "baseline_model_hash": model_digest(session_model),
        "translation_ab_m": [0.0, 0.0], "node_actions": [],
        "topology_actions": [], "member_id_map": {"1": 1},
        "support_name_map": {}, "load_name_map": {},
    }
    preview = {
        "preview_id": "preview-1", "draft_hash": draft_digest(draft),
        "merge_plan_hash": canonical_digest(draft["merge_plan"]),
        "baseline_model_hash": model_digest(session_model),
        "operations": [], "diff_hash": canonical_digest([]),
    }
    return draft, preview


def test_v1_migration_uses_image_graph_and_requires_review():
    draft = review_draft()
    assert validate_v2_draft(draft) == []
    assert draft["model"] is None
    assert draft["image_model"]["nodes"] == [
        {"id": 1, "u": 0.0, "v": 0.5},
        {"id": 2, "u": 1.0, "v": 0.5},
    ]
    assert draft["issues"][0]["severity"] == "blocking"


def test_v1_migration_synthesizes_traceable_support_and_load_entities():
    payload = v1_payload()
    payload["model"]["supports"] = [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}]
    payload["model"]["nodal_loads"] = [
        {"node": 2, "load": [0, -1, 0, 0, 0, 0]},
    ]
    draft = migrate_to_v2(payload, image_hash="a" * 64)
    assert draft["image_model"]["supports"][0]["name"] == "S1-1"
    assert draft["image_model"]["nodal_loads"][0]["name"] == "nodal_loads-1"
    kinds = [entity["kind"] for entity in draft["entities"]]
    assert kinds.count("support") == 1
    assert kinds.count("load") == 1


def test_initial_cancel_returns_to_loaded_and_stale_callback_is_ignored():
    state = MultimodalControllerState()
    assert state.phase({}) == MultimodalPhase.IDLE
    state.load_image("a" * 64)
    job = state.start_recognition("job-1")
    assert state.phase({}) == MultimodalPhase.RECOGNIZING
    assert state.cancel_recognition(job)
    assert state.phase({}) == MultimodalPhase.IMAGE_LOADED
    assert not state.complete_recognition(job, review_draft())


def test_rerecognition_cancel_and_failure_restore_old_draft():
    state = MultimodalControllerState()
    state.load_image("a" * 64)
    first = state.start_recognition("first")
    assert state.complete_recognition(first, review_draft())
    old = deepcopy(state.draft)
    retry = state.start_recognition("retry")
    assert state.cancel_recognition(retry)
    assert state.draft == old
    retry = state.start_recognition("retry-2")
    assert state.fail_recognition(retry, "offline")
    assert state.draft == old
    assert state.phase({}) == MultimodalPhase.REVIEW_REQUIRED


def test_first_failure_has_explicit_failed_phase():
    state = MultimodalControllerState()
    state.load_image("a" * 64)
    job = state.start_recognition("job")
    assert state.fail_recognition(job, "bad response")
    assert state.phase({}) == MultimodalPhase.RECOGNITION_FAILED


def test_ready_preview_commit_is_single_flight_and_receipted():
    session_model = {}
    draft, preview = ready_draft(session_model)
    state = MultimodalControllerState()
    state.load_image("a" * 64)
    job = state.start_recognition("job")
    assert state.complete_recognition(job, draft)
    state.set_commit_preview(preview)
    assert state.phase(session_model) == MultimodalPhase.READY_TO_COMMIT
    assert state.start_commit(session_model, confirmed_at="2026-09-08T00:00:00Z")
    assert not state.start_commit(session_model, confirmed_at="later")
    assert state.phase(session_model) == MultimodalPhase.COMMITTING
    committed_model = deepcopy(draft["model"])
    state.finish_commit(success=True, model_hash=model_digest(committed_model),
                        committed_at="2026-09-08T00:00:01Z", history_step_id="7")
    assert state.phase(committed_model) == MultimodalPhase.COMMITTED


def test_edit_invalidates_confirmation_preview_and_receipt():
    session_model = {}
    draft, preview = ready_draft(session_model)
    state = MultimodalControllerState()
    state.load_image("a" * 64)
    job = state.start_recognition("job")
    assert state.complete_recognition(job, draft)
    state.set_commit_preview(preview)
    edited = deepcopy(state.draft)
    edited["image_model"]["nodes"][1]["u"] = 0.9
    state.edit_draft(edited)
    assert state.draft["revision"] == 1
    assert state.draft["merge_plan"] is None
    assert state.commit_preview is None
    assert state.commit_receipt is None
    assert state.phase(session_model) == MultimodalPhase.REVIEW_REQUIRED


def test_preview_rejects_stale_draft_hash():
    state = MultimodalControllerState(draft=review_draft(), image_hash="a" * 64)
    with pytest.raises(ValueError, match="draft_hash"):
        state.set_commit_preview({"draft_hash": "stale"})


def test_recognition_rejects_a_result_for_another_image():
    state = MultimodalControllerState()
    state.load_image("b" * 64)
    job = state.start_recognition("job")
    assert not state.complete_recognition(job, review_draft())
    assert state.phase({}) == MultimodalPhase.RECOGNITION_FAILED
    assert "image_hash" in state.last_error
