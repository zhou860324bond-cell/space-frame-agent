"""局部分布荷载人工范围、单位与审核证据回归。"""

from copy import deepcopy

import pytest

from model_io import from_dict, validate_payload
from sketch_load_edit import (member_length, refresh_partial_geometry,
                              remove_partial_load, set_partial_load)
from sketch_parser import _fill_bookkeeping
from sketch_topology import materialize_geometry


def beam_draft():
    draft = _fill_bookkeeping({"image_model": {
        "nodes": [{"id": 1, "u": 0.1, "v": 0.5}, {"id": 2, "u": 0.9, "v": 0.5}],
        "members": [{"id": 7, "i": 1, "j": 2, "material": "Steel", "section": "S"}],
        "supports": [], "load_cases": [{"name": "D", "member_spans": []},
                                      {"name": "W", "member_spans": []}]}}, "a" * 64, "image.png")
    draft["source"].update(width_px=101, height_px=201)
    draft["work_plane"].update(status="confirmed", plane="XY")
    draft["scale"].update(status="confirmed", length_per_pixel=0.1, anchor_node=1)
    return draft


def test_partial_load_reaches_solver_with_exact_si_range_and_intensity():
    """人工局部荷载不能物化成满跨或带额外单位属性，求解器必须收到相同范围。"""
    draft = beam_draft()
    original = deepcopy(draft)
    updated = set_partial_load(draft, "D", "q", 7, 1, 3, [0, -1000, 0],
                               kind="partial_trapezoid", w2=[0, -2000, 0])
    assert draft == original and updated["image_model"]["load_cases"][1] == original["image_model"]["load_cases"][1]
    model = materialize_geometry(updated)
    model["materials"] = [{"name": "Steel", "E": 210e9, "nu": 0.3}]
    model["supports"] = [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}]
    model["sections"] = [{"name": "S", "A": 0.01, "Iy": 1e-4, "Iz": 1e-4, "J": 1e-4}]
    assert validate_payload(model) == []
    frame = from_dict(model)
    load = frame.case("D").member_spans[7][0]
    assert load.a == 1 and load.b == 3 and load.w1 == (0, -1000, 0) and load.w2 == (0, -2000, 0)
    entity = updated["entities"][-1]
    assert entity["image_geometry"]["line"] == [[pytest.approx(0.2), 0.5], [pytest.approx(0.4), 0.5]]
    assert entity["verified"] and entity["source"] == "user" and entity["confidence"] is None
    assert updated["confirmation"] is None


@pytest.mark.parametrize(("a", "b"), [(-1, 2), (1, 1), (3, 1), (0, 9),
                                     (float("nan"), 3), (1, float("inf")), (False, 3)])
def test_bad_range_is_rejected_without_mutating_draft(a, b):
    """缺失、逆序、越界及非有限范围不能被自动截短或扩大为整跨。"""
    draft = beam_draft()
    original = deepcopy(draft)
    with pytest.raises(ValueError, match="起止距离"):
        set_partial_load(draft, "D", "q", 7, a, b, [0, -1, 0])
    assert draft == original


@pytest.mark.parametrize("vector", [None, [0, -1], [0, float("nan"), 0], [0, True, 0], [0, "1", 0]])
def test_unknown_intensity_is_not_replaced_with_zero(vector):
    """图上没有可用强度时，人工编辑也必须提供三个有限数，不能默认零荷载。"""
    with pytest.raises(ValueError, match="强度"):
        set_partial_load(beam_draft(), "D", "q", 7, 1, 3, vector)


def test_unconfirmed_scale_cannot_be_used_as_metres():
    """图像百分比不是实际米数；尺度未知时不能应用实际作用区间。"""
    draft = beam_draft()
    draft["scale"].update(status="unknown", length_per_pixel=None)
    with pytest.raises(ValueError, match="尺度"):
        set_partial_load(draft, "D", "q", 7, 1, 3, [0, -1, 0])


def test_oblique_reversed_member_uses_pixel_aspect_and_i_end():
    """非正方形图片和反向杆件不能按归一化距离或左端位置误算局部范围。"""
    draft = beam_draft()
    draft["source"].update(width_px=201, height_px=101)
    draft["image_model"]["nodes"] = [{"id": 1, "u": 0.1, "v": 0.1}, {"id": 2, "u": 0.4, "v": 0.9}]
    draft["image_model"]["members"][0].update(i=2, j=1)
    assert member_length(draft, 7) == pytest.approx(10)
    updated = set_partial_load(draft, "D", "q", 7, 2, 6, [0, -1, 0])
    line = updated["entities"][-1]["image_geometry"]["line"]
    assert line[0] == pytest.approx([0.34, 0.74]) and line[1] == pytest.approx([0.22, 0.42])


def test_repair_resolves_only_the_exact_named_load_issue_and_preserves_observation():
    """同一杆件可能有多个缺值荷载，修一个不能清除全局、另一荷载或杆件级问题。"""
    draft = beam_draft()
    draft["image_model"]["load_cases"][0]["member_spans"] = [{"name": "q", "member": 7, "kind": "partial"}]
    target = {"case": "D", "collection": "member_spans", "name": "q"}
    draft["issues"] = [
        {"id": "specific", "category": "load_incomplete", "entity_refs": ["member:7"], "load_target": target, "status": "open"},
        {"id": "global", "category": "load_incomplete", "entity_refs": [], "status": "open"},
        {"id": "member", "category": "load_incomplete", "entity_refs": ["member:7"], "status": "open"},
        {"id": "other", "category": "load_incomplete", "entity_refs": ["load:D:member_spans:q2"], "status": "open"}]
    draft["entities"].append({"id": "old-q", "kind": "load", "source": "vision", "target": {"load": target}, "confidence": 0.2})
    updated = set_partial_load(draft, "D", "q", 7, 1, 3, [0, -100, 0])
    assert [i["status"] for i in updated["issues"]] == ["resolved", "open", "open", "open"]
    assert updated["entities"][-1]["original_observation"]["entities"][0]["confidence"] == 0.2
    assert updated["entities"][-1]["original_observation"]["loads"][0]["kind"] == "partial"


def test_shrinking_member_blocks_stale_range_and_repair_reopens_review():
    """已填局部范围在拖短杆件后必须重新阻断，不能继续显示原来的有效覆盖。"""
    draft = set_partial_load(beam_draft(), "D", "q", 7, 1, 6, [0, -1, 0])
    draft["image_model"]["nodes"][1]["u"] = 0.4
    updated = refresh_partial_geometry(draft)
    issue = next(i for i in updated["issues"] if i["id"].startswith("user-partial-range:"))
    assert issue["status"] == "open" and issue["severity"] == "blocking"
    assert updated["entities"][-1]["image_geometry"] == {}
    repaired = set_partial_load(updated, "D", "q", 7, 1, 2, [0, -1, 0])
    assert all(i["status"] == "resolved" for i in repaired["issues"])
    assert repaired["image_model"]["load_cases"][0]["member_spans"][0]["b"] == 2


def test_delete_keeps_same_named_load_in_other_case_and_audit_evidence():
    """删除局部荷载不能删掉另一工况的同名荷载，且需保留被删除的观察。"""
    draft = set_partial_load(beam_draft(), "D", "q", 7, 1, 3, [0, -1, 0])
    draft = set_partial_load(draft, "W", "q", 7, 2, 4, [0, -2, 0])
    updated = remove_partial_load(draft, "D", "q")
    assert updated["image_model"]["load_cases"][0]["member_spans"] == []
    assert updated["image_model"]["load_cases"][1]["member_spans"][0]["w1"] == [0, -2, 0]
    assert updated["edit_history"][-1]["loads"][0]["w1"] == [0, -1, 0]
    assert updated["edit_history"][-1]["entities"]


def test_existing_other_kind_cannot_be_overwritten_by_typing_same_name():
    """名称可编辑时，同名跨间集中力不能被局部均布编辑器悄悄替换。"""
    draft = beam_draft()
    draft["image_model"]["load_cases"][0]["member_spans"] = [{"name": "q", "kind": "point", "member": 7, "a": 2, "w1": [0, -5, 0]}]
    with pytest.raises(ValueError, match="其他类型"):
        set_partial_load(draft, "D", "q", 7, 1, 3, [0, -1, 0])


def test_generated_missing_range_issue_has_exact_load_target():
    """同一杆件多条缺范围的问题必须保留各自工况与名称，供人工精确解除。"""
    draft = beam_draft()
    draft["image_model"]["load_cases"][0]["member_spans"] = [
        {"member": 7, "kind": "partial", "w1": [0, -1, 0]},
        {"member": 7, "kind": "partial", "w1": [0, -2, 0]}]
    updated = _fill_bookkeeping(draft, "a" * 64, "image.png")
    issues = [i for i in updated["issues"] if i["category"] == "load_incomplete"]
    assert [i["load_target"]["name"] for i in issues] == ["member_spans-1", "member_spans-2"]
    repaired = set_partial_load(updated, "D", "member_spans-1", 7, 1, 3, [0, -1, 0])
    assert [i["status"] for i in repaired["issues"]] == ["resolved", "open"]
