"""MM2-05 deterministic intersections, deletion guards, and compiler handoff."""

from __future__ import annotations


import pytest

from agent import Session
from model_compiler import compile_model
from sketch_topology import (delete_member, delete_node, detect_topology,
                             materialize_geometry, resolve_intersection)


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
