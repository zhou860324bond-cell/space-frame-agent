"""MM2-05 deterministic intersections, deletion guards, and compiler handoff."""

from __future__ import annotations


from copy import deepcopy

import pytest

from agent import Session
from model_compiler import compile_model
from sketch_topology import (delete_member, delete_node, detect_topology,
                             insert_member_node, materialize_geometry, resolve_intersection)


def draft(nodes, members):
    return {
        "format": "space-frame-recognition-draft/v2",
        "image_model": {"nodes": nodes, "members": members, "supports": [],
                        "load_cases": []},
        "model": None, "merge_plan": None,
        "source": {"width_px": 101, "height_px": 101},
        "work_plane": {"status": "confirmed", "plane": "XY", "offset": 0.0},
        "scale": {"status": "confirmed", "length_per_pixel": 0.1,
                  "anchor_node": nodes[0]["id"],
                  "anchor_coordinates_xyz": [0.0, 0.0, 0.0]},
        "entities": [], "dimensions": [], "intersections": [], "issues": [],
        "revision": 0, "confirmation": None,
    }


def crossing(kind="x"):
    nodes = [{"id": 1, "u": 0.0, "v": 0.5},
             {"id": 2, "u": 1.0, "v": 0.5},
             {"id": 3, "u": 0.5, "v": 0.0},
             {"id": 4, "u": 0.5, "v": 1.0 if kind == "x" else 0.5}]
    return draft(nodes, [{"id": 1, "i": 1, "j": 2, "material": None, "section": None},
                         {"id": 2, "i": 3, "j": 4, "material": None, "section": None}])


def _single_beam():
    return draft([{"id": 1, "u": 0.1, "v": 0.5}, {"id": 2, "u": 0.9, "v": 0.5}],
                 [{"id": 1, "i": 1, "j": 2, "material": "Steel", "section": "S"}])


def test_user_can_insert_missing_action_node_and_keep_geometry_and_boundary_data():
    """模型漏掉集中作用点时，用户可补节点；原图、尺度、支座和端节点荷载必须保留。"""
    from copy import deepcopy
    value = _single_beam()
    value["image_model"]["supports"] = [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}]
    value["image_model"]["load_cases"] = [{"name": "D", "nodal_loads": [
        {"node": 2, "load": [0, 0, -10, 0, 0, 0]}]}]
    value["confirmation"] = {"old": True}
    original = deepcopy(value)
    updated = insert_member_node(value, 1, 0.25)
    assert value == original
    assert updated["image_model"]["nodes"][-1] == {"id": 3, "u": pytest.approx(0.3), "v": 0.5}
    assert [(m["id"], m["i"], m["j"]) for m in updated["image_model"]["members"]] == [(1, 1, 3), (2, 3, 2)]
    assert all(m["section"] == "S" and m["material"] == "Steel" for m in updated["image_model"]["members"])
    assert updated["image_model"]["supports"] == original["image_model"]["supports"]
    assert updated["image_model"]["load_cases"] == original["image_model"]["load_cases"]
    assert updated["scale"] == original["scale"] and updated["source"] == original["source"]
    assert updated["confirmation"] is None
    assert all(e["source"] == "user" and e["verified"] and e["confidence"] is None for e in updated["entities"])


@pytest.mark.parametrize("fraction", [0, 1, True, float("nan"), 0.01])
def test_action_node_cannot_duplicate_an_endpoint_or_use_invalid_position(fraction):
    """防止手工补作用点时生成零长度杆件或让非有限坐标进入草稿。"""
    with pytest.raises(ValueError):
        insert_member_node(_single_beam(), 1, fraction)


@pytest.mark.parametrize("reference", ["member_loads", "member_spans", "dimension"])
def test_action_node_split_refuses_existing_member_loads_and_dimensions(reference):
    """分段不能静默把旧整跨荷载或尺寸转移到第一小段，需先处理引用。"""
    value = _single_beam()
    if reference == "dimension":
        value["dimensions"] = [{"id": "D1", "target": {"member": 1}}]
    else:
        value["image_model"]["load_cases"] = [{"name": "D", reference: [{"member": 1}]}]
    with pytest.raises(ValueError, match="仍被引用"):
        insert_member_node(value, 1, 0.5)
    assert len(value["image_model"]["nodes"]) == 2


def test_split_reopens_original_member_issues_for_both_segments():
    """已有符号缺值问题不能因补节点消失或只错误附着在第一段。"""
    value = _single_beam()
    value["issues"] = [{"id": "q", "category": "load_incomplete", "entity_refs": ["member:1"],
        "severity": "blocking", "status": "resolved", "resolution": "ignored", "resolved_by": "user"}]
    updated = insert_member_node(value, 1, 0.5)
    issue = next(i for i in updated["issues"] if i["id"] == "q")
    assert issue["entity_refs"] == ["member:1", "member:2"] and issue["status"] == "open"
    assert issue["resolution"] is None and issue["resolved_by"] is None


def test_split_does_not_duplicate_end_releases_on_new_internal_node():
    """杆件含端部释放时直接复制会使新内节点错误铰接，因此拒绝未经核对的分段。"""
    value = _single_beam()
    value["image_model"]["members"][0]["releases"] = ["rz_i"]
    with pytest.raises(ValueError, match="端部释放"):
        insert_member_node(value, 1, 0.5)


def test_t_junction_reuses_branch_endpoint():
    detected = detect_topology(crossing("t"))
    item = detected["intersections"][0]
    assert item["member_s"] == {"1": pytest.approx(0.5), "2": 1.0}
    resolved = resolve_intersection(detected, item["id"], "connect")
    assert resolved["intersections"][0]["node_id"] == 4
    assert len(resolved["image_model"]["nodes"]) == 4


def test_x_junction_connect_creates_one_shared_internal_node():
    detected = detect_topology(crossing())
    assert detected["intersections"][0]["decision"] == "unknown"
    resolved = resolve_intersection(detected, "I-1-2", "connect")
    assert resolved["intersections"][0]["node_id"] == 5
    assert resolved["image_model"]["nodes"][-1] == {"id": 5, "u": 0.5, "v": 0.5}
    assert resolved["issues"][0]["status"] == "resolved"


def test_cross_decision_does_not_create_a_node():
    detected = detect_topology(crossing())
    resolved = resolve_intersection(detected, "I-1-2", "cross")
    assert resolved["intersections"][0]["decision"] == "cross"
    assert resolved["intersections"][0]["node_id"] is None
    assert len(resolved["image_model"]["nodes"]) == 4


def test_near_endpoint_uses_closest_endpoint_and_snaps_it_exactly():
    value = crossing("t")
    value["image_model"]["nodes"][2] = {"id": 3, "u": 0.02, "v": 0.0}
    value["image_model"]["nodes"][3] = {"id": 4, "u": 0.02, "v": 0.5}
    detected = detect_topology(value)
    resolved = resolve_intersection(detected, "I-1-2", "connect")
    assert resolved["intersections"][0]["node_id"] == 4
    reused = next(item for item in resolved["image_model"]["nodes"] if item["id"] == 4)
    assert reused == {"id": 4, "u": pytest.approx(0.02), "v": pytest.approx(0.5)}


def test_duplicate_member_is_a_blocking_topology_issue():
    value = crossing()
    value["image_model"]["members"][1].update(i=2, j=1)
    detected = detect_topology(value)
    assert detected["intersections"] == []
    assert detected["issues"][0]["category"] == "topology"
    assert "重复杆件" in detected["issues"][0]["message"]


def test_existing_shared_endpoint_is_already_connected_without_a_question():
    value = draft(
        [{"id": 1, "u": 0, "v": 0.5}, {"id": 2, "u": 0.5, "v": 0.5},
         {"id": 3, "u": 0.5, "v": 0}],
        [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 3}])
    detected = detect_topology(value)
    assert detected["intersections"][0]["decision"] == "connect"
    assert detected["intersections"][0]["node_id"] == 2
    assert not any(item["category"] == "intersection_unknown"
                   for item in detected["issues"])


def test_separate_collinear_members_are_not_reported_as_overlapping():
    value = draft(
        [{"id": 1, "u": 0, "v": 0.5}, {"id": 2, "u": 0.2, "v": 0.5},
         {"id": 3, "u": 0.8, "v": 0.5}, {"id": 4, "u": 1, "v": 0.5}],
        [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 3, "j": 4}])
    detected = detect_topology(value)
    assert detected["intersections"] == []
    assert detected["issues"] == []


@pytest.mark.parametrize("points", [
    [(0, 0.5), (0.5, 0.5), (1, 0.5)],
    [(0.5, 0), (0.5, 0.5), (0.5, 1)],
    [(0, 0), (0.5, 0.5), (1, 1)],
])
@pytest.mark.parametrize("ends", [((1, 2), (2, 3)), ((2, 1), (2, 3)),
                                  ((1, 2), (3, 2)), ((2, 1), (3, 2))])
def test_collinear_shared_endpoint_is_recorded_in_either_member_direction(points, ends):
    """防止水平、竖直或斜向共线接头被平行分支跳过，反向杆件也须引用同一节点。"""
    value = draft([{"id": n, "u": p[0], "v": p[1]} for n, p in enumerate(points, 1)],
                  [{"id": n, "i": e[0], "j": e[1]} for n, e in enumerate(ends, 1)])
    original = deepcopy(value)
    detected = detect_topology(value)
    assert detected["intersections"] == [{
        "id": "I-1-2", "members": [1, 2], "point": list(points[1]),
        "decision": "connect", "node_id": 2,
        "member_s": {str(n): 0.0 if e[0] == 2 else 1.0 for n, e in enumerate(ends, 1)},
    }]
    assert detected["issues"] == []
    assert value == original
    assert detect_topology(detected) == detected


@pytest.mark.parametrize("last", [(0.52, 0.5), (0.51, 0.51)])
def test_short_shared_members_keep_exact_endpoint_parameters(last):
    """防止两端均在像素吸附范围内时，把已共享的 j 端错误改记为 i 端。"""
    value = draft([{"id": 1, "u": 0.5, "v": 0.5}, {"id": 2, "u": 0.51, "v": 0.5},
                   {"id": 3, "u": last[0], "v": last[1]}],
                  [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 3}])
    detected = detect_topology(value)
    assert detected["intersections"][0]["member_s"] == {"1": 1.0, "2": 0.0}
    assert detected["intersections"][0]["point"] == [0.51, 0.5]
    resolved = resolve_intersection(detected, "I-1-2", "connect")
    assert resolved["image_model"]["nodes"] == value["image_model"]["nodes"]


def test_collinear_shared_endpoint_does_not_hide_overlap():
    """共线杆件同向重叠仍须阻断，不能因具有同一端节点被当成正常接头。"""
    value = draft([{"id": 1, "u": 0, "v": 0.5}, {"id": 2, "u": 0.5, "v": 0.5},
                   {"id": 3, "u": 1, "v": 0.5}],
                  [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 1, "j": 3}])
    detected = detect_topology(value)
    assert detected["intersections"] == []
    assert [(i["id"], i["severity"]) for i in detected["issues"]] == [("overlap-1-2", "blocking")]


def test_coincident_distinct_collinear_endpoints_are_not_merged():
    """防止仅坐标重合的两个不同节点被新增共享端点逻辑擅自合并。"""
    value = draft([{"id": 1, "u": 0, "v": 0.5}, {"id": 2, "u": 0.5, "v": 0.5},
                   {"id": 3, "u": 0.5, "v": 0.5}, {"id": 4, "u": 1, "v": 0.5}],
                  [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 3, "j": 4}])
    assert detect_topology(value)["intersections"] == []
    assert detect_topology(value)["image_model"] == value["image_model"]


def test_degenerate_shared_member_does_not_gain_a_valid_joint():
    """防止零长杆件从新增共线端点分支获得看似有效的连接记录。"""
    value = draft([{"id": 1, "u": 0.5, "v": 0.5}, {"id": 2, "u": 0.5, "v": 0.5},
                   {"id": 3, "u": 1, "v": 0.5}],
                  [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 3}])
    assert detect_topology(value)["intersections"] == []


def test_inserted_collinear_action_node_compiles_without_extra_segments():
    """用户补作用点后须留下连接记录，导入求解仍只有原两段，不产生零长单元。"""
    detected = insert_member_node(_single_beam(), 1, 0.25)
    assert detected["intersections"][0]["node_id"] == 3
    assert detected["intersections"][0]["member_s"] == {"1": 1.0, "2": 0.0}
    model = materialize_geometry(detected)
    session = Session()
    assert session.add_nodes([[n["x"], n["y"], n["z"]] for n in model["nodes"]]).ok
    assert session.add_members([[m["i"], m["j"]] for m in model["members"]]).ok
    assert session.define_materials_and_sections(
        [{"name": "Steel", "E": 2e11, "nu": 0.3}],
        [{"name": "S", "A": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 5e-6}],
    ).ok
    assert session.assign_properties([1, 2], "S", "Steel").ok
    assert session.set_supports([1], [1, 1, 1, 1, 1, 1]).ok
    compiled = compile_model(session.model)
    assert [len(compiled.mapping.element_ids(n)) for n in (1, 2)] == [1, 1]
    assert {(m.i, m.j) for m in compiled.analysis_model.members.values()} == {(1, 3), (3, 2)}


def test_collinear_generated_joints_do_not_block_further_split_or_delete():
    """补作用点产生的共线共享端点记录可自动重建，不能误阻止再次分段或删除杆件。"""
    first = insert_member_node(_single_beam(), 1, 0.5)
    second = insert_member_node(first, 2, 0.5)
    assert [(i["members"], i["node_id"]) for i in second["intersections"]] == [([1, 2], 3), ([2, 3], 4)]
    deleted = delete_member(second, 2)
    assert deleted["intersections"] == []
    assert deleted["issues"] == []
    assert len(deleted["image_model"]["members"]) == 2


def test_collinear_joint_agrees_with_independently_written_reference():
    """新参考要求共线接头完整；运行时结果应匹配手写参考，不能从预测补参考。"""
    from multimodal_eval.reference_topology import FORMAT, check_reference_topology

    value = draft([{"id": 1, "u": 0, "v": 0.5}, {"id": 2, "u": 0.5, "v": 0.5},
                   {"id": 3, "u": 1, "v": 0.5}],
                  [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 3}])
    truth = {**deepcopy(value["image_model"]), "width_px": 101, "height_px": 101,
             "nodes": [{"id": n["id"], "point": [n["u"], n["v"]]}
                       for n in value["image_model"]["nodes"]],
             "topology_reference": {"format": FORMAT, "scope": "all_member_pairs", "status": "verified"},
             "intersections": [{"id": "joint", "members": [1, 2], "point": [0.5, 0.5],
                                "decision": "connect", "node_id": 2}]}
    assert check_reference_topology(truth)["valid"]
    observed = detect_topology(value)["intersections"][0]
    assert {k: observed[k] for k in ("members", "point", "decision", "node_id")} == {
        k: truth["intersections"][0][k] for k in ("members", "point", "decision", "node_id")}


def test_deletion_refuses_member_and_node_references():
    value = crossing()
    value["dimensions"] = [{"id": "D", "target": {"member": 1}}]
    with pytest.raises(ValueError, match="dimension"):
        delete_member(value, 1)
    value = crossing()
    value["image_model"]["supports"] = [{"name": "S", "node": 1,
                                          "fix": [1, 1, 1, 1, 1, 1]}]
    with pytest.raises(ValueError, match="member.*support|support.*member"):
        delete_node(value, 1)


@pytest.mark.parametrize(("kind", "shared", "expected_splits"),
                         [("x", 5, (2, 2)), ("t", 4, (2, 1))])
def test_confirmed_intersection_materializes_and_compiles_to_shared_analysis_node(
        kind, shared, expected_splits):
    detected = detect_topology(crossing(kind))
    resolved = resolve_intersection(detected, "I-1-2", "connect")
    model = materialize_geometry(resolved)
    session = Session()
    assert session.add_nodes([[item["x"], item["y"], item["z"]]
                              for item in model["nodes"]]).ok
    assert session.add_members([[item["i"], item["j"]]
                                for item in model["members"]]).ok
    assert session.define_materials_and_sections(
        [{"name": "Steel", "E": 2e11, "nu": 0.3}],
        [{"name": "S", "A": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 5e-6}],
    ).ok
    assert session.assign_properties([1, 2], "S", "Steel").ok
    assert session.set_supports([1], [1, 1, 1, 1, 1, 1]).ok
    compiled = compile_model(session.model)
    assert tuple(len(compiled.mapping.element_ids(mid)) for mid in (1, 2)) == expected_splits
    for member_id in (1, 2):
        element_ids = compiled.mapping.element_ids(member_id)
        endpoints = {(compiled.analysis_model.members[eid].i,
                      compiled.analysis_model.members[eid].j) for eid in element_ids}
        assert any(shared in pair for pair in endpoints)
