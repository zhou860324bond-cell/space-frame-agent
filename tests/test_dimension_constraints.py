"""MM2-04 dimension evidence and scale resolution tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from dimension_constraints import (add_member_length_dimension, apply_scale_to_draft,
                                   default_anchor_node, dimension_pixel_length,
                                   solve_scale)
from multimodal_workflow import MultimodalControllerState, MultimodalPhase


IMAGE_MODEL = {
    "nodes": [{"id": 3, "u": 0.1, "v": 0.2},
              {"id": 2, "u": 0.1, "v": 0.8},
              {"id": 1, "u": 0.9, "v": 0.8}],
    "members": [{"id": 7, "i": 2, "j": 1}],
}


def dimension(identifier="span", *, value=8.0, unit="m", status="confirmed",
              kind="member_length", target=None):
    return {"id": identifier, "kind": kind, "text": str(value), "value": value,
            "unit": unit, "image_geometry": {"line": [[0.1, 0.8], [0.9, 0.8]]},
            "target": target or {"member": 7}, "confidence": 0.9,
            "recognition_confidence": 0.9, "verified": False,
            "source": "vision", "status": status}


def test_single_dimension_calibrates_scale_and_default_anchor():
    scale, _ = solve_scale(IMAGE_MODEL, [dimension()], 101, 101)
    assert scale["status"] == "confirmed"
    assert scale["length_per_pixel"] == pytest.approx(0.1)
    assert scale["evidence_ids"] == ["dimension:span"]
    assert default_anchor_node(IMAGE_MODEL) == 2


def test_redundant_consistent_dimensions_use_sorted_median():
    dimensions = [dimension("b", value=8080, unit="mm"),
                  dimension("a", value=8.0, unit="m"),
                  dimension("c", value=800, unit="cm")]
    scale, _ = solve_scale(IMAGE_MODEL, dimensions, 101, 101)
    assert scale["status"] == "confirmed"
    assert scale["length_per_pixel"] == pytest.approx(0.1)
    assert scale["evidence_ids"] == ["dimension:a", "dimension:b", "dimension:c"]


def test_conflicting_dimensions_block_controller_commit():
    scale, _ = solve_scale(IMAGE_MODEL,
                           [dimension("a", value=8), dimension("b", value=16)],
                           101, 101)
    assert scale["status"] == "conflict"
    assert scale["length_per_pixel"] is None
    state = MultimodalControllerState(image_hash="a" * 64, draft={"scale": scale})
    assert state.phase({}) == MultimodalPhase.REVIEW_REQUIRED
    assert not state.can_commit({})


def test_exact_two_percent_from_median_is_not_a_conflict():
    scale, _ = solve_scale(
        IMAGE_MODEL, [dimension("lo", value=7.84), dimension("hi", value=8.16)],
        101, 101)
    assert scale["status"] == "confirmed"
    assert scale["length_per_pixel"] == pytest.approx(0.1)


def test_trusted_dimension_wins_exactly_and_rejects_other_evidence():
    dimensions = [dimension("a", value=8), dimension("b", value=16)]
    scale, updated = solve_scale(
        IMAGE_MODEL, dimensions, 101, 101, trusted_dimension_id="b")
    assert scale["length_per_pixel"] == pytest.approx(0.2)
    assert scale["evidence_ids"] == ["dimension:b"]
    assert updated[0]["status"] == "rejected_conflict"
    assert updated[1]["status"] == "confirmed"
    assert dimensions[0]["status"] == "confirmed"


def test_missing_unit_and_angle_do_not_enter_scale_solution():
    scale, _ = solve_scale(
        IMAGE_MODEL, [dimension(unit=None), dimension("angle", kind="angle")],
        101, 101)
    assert scale["status"] == "unknown"
    assert scale["evidence_ids"] == []


def test_pixel_length_kinds_and_zero_length_validation():
    vertical = dimension("height", kind="vertical_distance",
                         target={"nodes": [3, 2]})
    assert dimension_pixel_length(vertical, IMAGE_MODEL, 101, 101) == pytest.approx(60)
    horizontal = dimension("span", kind="horizontal_distance",
                           target={"nodes": [2, 1]})
    assert dimension_pixel_length(horizontal, IMAGE_MODEL, 101, 101) == pytest.approx(80)
    with pytest.raises(ValueError, match="像素长度为零"):
        dimension_pixel_length(
            dimension("zero", kind="horizontal_distance", target={"nodes": [3, 2]}),
            IMAGE_MODEL, 101, 101)


def test_apply_scale_is_pure_and_preserves_user_anchor():
    draft = {"image_model": deepcopy(IMAGE_MODEL), "dimensions": [dimension()],
             "source": {"width_px": 101, "height_px": 101},
             "scale": {"status": "unknown", "length_per_pixel": None,
                       "anchor_node": 3,
                       "anchor_coordinates_xyz": [1.0, 2.0, 3.0]},
             "issues": [{"id": "old-scale", "category": "scale_unknown",
                         "severity": "blocking", "entity_refs": [],
                         "message": "old", "status": "open",
                         "resolution": None, "resolved_by": None}]}
    original = deepcopy(draft)
    updated = apply_scale_to_draft(draft)
    assert draft == original
    assert updated["scale"]["status"] == "confirmed"
    assert updated["scale"]["anchor_node"] == 3
    assert updated["scale"]["anchor_coordinates_xyz"] == [1.0, 2.0, 3.0]
    assert updated["issues"][0]["status"] == "resolved"
    assert updated["issues"][0]["resolved_by"] == "system"


def test_scale_issue_is_created_and_reopened_deterministically():
    draft = {"image_model": deepcopy(IMAGE_MODEL),
             "dimensions": [dimension(unit=None)],
             "source": {"width_px": 101, "height_px": 101},
             "scale": {}, "issues": []}
    unknown = apply_scale_to_draft(draft)
    assert [(item["category"], item["status"]) for item in unknown["issues"]] == [
        ("scale_unknown", "open")]
    unknown["dimensions"] = [dimension()]
    confirmed = apply_scale_to_draft(unknown)
    assert confirmed["issues"][0]["status"] == "resolved"
    confirmed["dimensions"] = [dimension(unit=None)]
    reopened = apply_scale_to_draft(confirmed)
    assert len(reopened["issues"]) == 1
    assert reopened["issues"][0]["status"] == "open"
    assert reopened["issues"][0]["resolution"] is None


def test_manual_member_length_creates_traceable_confirmed_evidence():
    draft = {"image_model": deepcopy(IMAGE_MODEL), "dimensions": [],
             "source": {"width_px": 101, "height_px": 101}, "scale": {},
             "issues": []}
    with_dimension = add_member_length_dimension(draft, 7, 8000, "mm")
    evidence = with_dimension["dimensions"][0]
    assert draft["dimensions"] == []
    assert evidence["source"] == "user" and evidence["verified"] is True
    assert evidence["recognition_confidence"] is None
    calibrated = apply_scale_to_draft(with_dimension)
    assert calibrated["scale"]["length_per_pixel"] == pytest.approx(0.1)
