"""多模态识别草稿的人机确认边界。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from recognition_draft import DRAFT_FORMAT, RecognitionDraft


def payload(scale: str = "confirmed") -> dict:
    return {
        "format": DRAFT_FORMAT,
        "scale": {"status": scale, "evidence": "图中标注 6 m"},
        "model": {
            "units": "N-m-Pa",
            "materials": [],
            "sections": [],
            "nodes": [
                {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"id": 2, "x": 2.0, "y": 0.0, "z": 0.0},
            ],
            "members": [{"id": 7, "i": 1, "j": 2}],
            "supports": [],
            "load_cases": [],
        },
        "entities": [{"kind": "member", "id": 7, "confidence": 0.8,
                      "image_geometry": {"line": [[0.1, 0.5], [0.9, 0.5]]}}],
        "questions": ["支座符号是否存在？"],
        "warnings": [],
    }


def test_geometry_only_recognition_is_a_valid_editing_draft():
    draft = RecognitionDraft.from_payload(payload())

    assert draft.ready_to_load
    assert draft.model["materials"] == []
    assert draft.model["sections"] == []
    assert draft.model["members"][0]["section"] == ""
    assert draft.model["members"][0]["material"] == ""
    assert any("尚未指派截面" in note.message
               for note in draft.to_frame_draft().editing_diagnostics())


def test_unknown_scale_blocks_loading_until_user_calibrates_a_member():
    draft = RecognitionDraft.from_payload(payload("unknown"))

    assert not draft.ready_to_load
    assert any("尺度" in question for question in draft.questions)
    with pytest.raises(ValueError, match="尺度"):
        draft.to_frame_draft()

    draft.calibrate(7, 6.0)

    assert draft.ready_to_load
    assert draft.scale_status == "confirmed"
    assert draft.model["nodes"][1]["x"] == pytest.approx(6.0)
    assert "用户标定" in draft.scale_evidence


def test_calibration_does_not_mutate_the_parser_payload():
    original = payload("unknown")
    before = deepcopy(original)
    draft = RecognitionDraft.from_payload(original)

    draft.calibrate(7, 3.0)

    assert original == before


def test_legacy_recognition_cannot_silently_assume_coordinates_are_metres():
    legacy = deepcopy(payload()["model"])

    draft = RecognitionDraft.from_payload(legacy)

    assert draft.scale_status == "unknown"
    assert not draft.ready_to_load
    assert any("尺度" in question for question in draft.questions)


def test_bad_topology_is_rejected_before_confirmation():
    invalid = payload()
    invalid["model"]["members"][0]["j"] = 99

    with pytest.raises(ValueError, match="不存在的节点"):
        RecognitionDraft.from_payload(invalid)


def test_entity_confidence_must_be_a_probability():
    invalid = payload()
    invalid["entities"][0]["confidence"] = 1.5

    with pytest.raises(ValueError, match="confidence"):
        RecognitionDraft.from_payload(invalid)


def test_entity_overlay_coordinates_and_references_are_validated():
    invalid = payload()
    invalid["entities"].extend([
        {"kind": "member", "id": 7, "confidence": 0.8,
         "image_geometry": {"line": [[0, 0], [1, 2]]}},
        {"kind": "node", "id": 99, "confidence": 0.8,
         "image_geometry": {"point": [0.5, 0.5]}},
    ])

    with pytest.raises(ValueError) as caught:
        RecognitionDraft.from_payload(invalid)

    message = str(caught.value)
    assert "重复" in message
    assert "介于 0 和 1" in message
    assert "不存在的节点" in message


def test_bad_support_or_load_reference_is_rejected():
    invalid = payload()
    invalid["model"]["supports"] = [{"node": 99, "fix": [1, 1, 1]}]
    invalid["model"]["load_cases"] = [{
        "name": "D", "member_loads": [{"member": 88, "w": [0, 0, -1]}],
    }]

    with pytest.raises(ValueError) as caught:
        RecognitionDraft.from_payload(invalid)

    message = str(caught.value)
    assert "支座" in message and "不存在的节点" in message
    assert "fix" in message
    assert "不存在的杆件" in message


def test_malformed_load_entries_are_reported_not_crashed():
    invalid = payload()
    invalid["model"]["load_cases"] = [None, {
        "name": "D", "nodal_loads": [None], "member_loads": ["bad"],
    }]

    with pytest.raises(ValueError, match="必须是对象"):
        RecognitionDraft.from_payload(invalid)


def test_dragging_a_node_updates_model_coordinates_and_member_overlay():
    data = payload()
    data["entities"].append({
        "kind": "node", "id": 1, "confidence": 0.9,
        "image_geometry": {"point": [0.1, 0.5]},
    })
    data["entities"].append({
        "kind": "node", "id": 2, "confidence": 0.9,
        "image_geometry": {"point": [0.9, 0.5]},
    })
    draft = RecognitionDraft.from_payload(data)

    draft.move_node_image(2, 0.7, 0.3)

    assert draft.model["nodes"][1]["x"] == pytest.approx(1.5)
    assert draft.model["nodes"][1]["z"] == pytest.approx(0.5)
    member = next(item for item in draft.entities if item["kind"] == "member")
    assert member["image_geometry"]["line"][1] == [0.7, 0.3]


def test_user_can_add_and_remove_a_member_without_bypassing_topology():
    data = payload()
    data["model"]["nodes"].append({"id": 3, "x": 2, "y": 0, "z": 2})
    data["entities"].extend([
        {"kind": "node", "id": 2, "confidence": 0.9,
         "image_geometry": {"point": [0.8, 0.8]}},
        {"kind": "node", "id": 3, "confidence": 0.9,
         "image_geometry": {"point": [0.8, 0.2]}},
    ])
    draft = RecognitionDraft.from_payload(data)

    member_id = draft.add_member(2, 3)

    assert draft.model["members"][-1]["section"] == ""
    assert any(item.get("source") == "user" for item in draft.entities)
    with pytest.raises(ValueError, match="已有杆件"):
        draft.add_member(3, 2)

    draft.remove_member(member_id)
    assert all(int(item["id"]) != member_id for item in draft.model["members"])


def test_member_with_recognized_load_cannot_be_silently_deleted():
    data = payload()
    data["model"]["nodes"].append({"id": 3, "x": 2, "y": 0, "z": 2})
    data["model"]["members"].append({"id": 8, "i": 2, "j": 3})
    data["model"]["load_cases"] = [{
        "name": "L", "member_loads": [{"member": 8, "w": [0, 0, -1]}],
    }]
    draft = RecognitionDraft.from_payload(data)

    with pytest.raises(ValueError, match="仍被荷载引用"):
        draft.remove_member(8)


def test_support_can_be_added_changed_and_removed_on_a_recognized_node():
    data = payload()
    data["entities"].append({
        "kind": "node", "id": 1, "confidence": 0.9,
        "image_geometry": {"point": [0.1, 0.5]},
    })
    draft = RecognitionDraft.from_payload(data)

    draft.set_support(1, [1, 1, 1, 0, 0, 0], "BC-1")
    assert draft.model["supports"] == [{
        "node": 1, "fix": [1, 1, 1, 0, 0, 0], "name": "BC-1"}]
    support_entity = next(item for item in draft.entities
                          if item["kind"] == "support")
    assert support_entity["target"]["node"] == 1
    assert "bbox" in support_entity["image_geometry"]

    draft.set_support(1, [1, 1, 1, 1, 1, 1], "BC-Fixed")
    assert len(draft.model["supports"]) == 1
    assert draft.model["supports"][0]["fix"] == [1] * 6

    draft.remove_support(1)
    assert draft.model["supports"] == []
    assert all(item["kind"] != "support" for item in draft.entities)


def test_user_added_support_overlay_follows_a_dragged_node():
    data = payload()
    data["entities"].extend([
        {"kind": "node", "id": 1, "confidence": 0.9,
         "image_geometry": {"point": [0.1, 0.5]}},
        {"kind": "node", "id": 2, "confidence": 0.9,
         "image_geometry": {"point": [0.9, 0.5]}},
    ])
    draft = RecognitionDraft.from_payload(data)
    draft.set_support(2, [1, 1, 1, 0, 0, 0])

    draft.move_node_image(2, 0.7, 0.3)

    support = next(item for item in draft.entities if item["kind"] == "support")
    assert support["image_geometry"]["bbox"] == pytest.approx(
        [0.665, 0.265, 0.735, 0.335])


def test_named_nodal_load_can_be_added_updated_and_removed():
    data = payload()
    data["entities"].append({
        "kind": "node", "id": 2, "confidence": 0.9,
        "image_geometry": {"point": [0.9, 0.5]},
    })
    draft = RecognitionDraft.from_payload(data)
    draft.add_load_case("Wind")

    draft.set_nodal_load("Wind", 2, [100, 0, 0, 0, 0, 0], "P1")
    draft.set_nodal_load("Wind", 2, [200, 0, 0, 0, 0, 0], "P1")

    entries = draft.model["load_cases"][0]["nodal_loads"]
    assert entries == [{
        "name": "P1", "node": 2,
        "load": [200.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]
    load_entity = next(item for item in draft.entities if item["kind"] == "load")
    assert load_entity["target"] == {"case": "Wind", "node": 2, "name": "P1"}

    draft.remove_nodal_load("Wind", "P1")
    assert draft.model["load_cases"][0]["nodal_loads"] == []
    assert all(item["kind"] != "load" for item in draft.entities)


def test_support_and_load_editing_rejects_implicit_or_invalid_actions():
    draft = RecognitionDraft.from_payload(payload())

    with pytest.raises(ValueError, match="全零约束"):
        draft.set_support(1, [0] * 6)
    with pytest.raises(ValueError, match="不存在"):
        draft.set_nodal_load("Missing", 2, [1, 0, 0, 0, 0, 0], "P")
    draft.add_load_case("L")
    with pytest.raises(ValueError, match="不能全为零"):
        draft.set_nodal_load("L", 2, [0] * 6, "P")
