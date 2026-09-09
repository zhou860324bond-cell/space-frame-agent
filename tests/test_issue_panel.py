"""MM2-06 problem-driven v2 review UI tests."""

from __future__ import annotations

import os
from copy import deepcopy

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from agent import Session  # noqa: E402
from desktop.issue_panel import IssuePanel  # noqa: E402
from desktop.sketch_panel import SketchPanel  # noqa: E402
from sketch_topology import detect_topology  # noqa: E402
from multimodal_workflow import MultimodalControllerState  # noqa: E402
from dimension_constraints import add_member_length_dimension  # noqa: E402


class IdleRunner:
    busy = False


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def issue(identifier, category, severity="blocking", refs=None):
    return {"id": identifier, "category": category, "severity": severity,
            "entity_refs": refs or [], "message": identifier, "status": "open",
            "resolution": None, "resolved_by": None}


def base_draft():
    return {
        "image_model": {
            "nodes": [{"id": 1, "u": 0.0, "v": 0.5},
                      {"id": 2, "u": 1.0, "v": 0.5},
                      {"id": 3, "u": 0.5, "v": 0.0},
                      {"id": 4, "u": 0.5, "v": 1.0}],
            "members": [{"id": 1, "i": 1, "j": 2},
                        {"id": 2, "i": 3, "j": 4}],
            "supports": [], "load_cases": []},
        "source": {"width_px": 101, "height_px": 101,
                   "preprocessing": {"perspective_status": "unconfirmed"}},
        "work_plane": {"status": "proposed", "plane": "XZ"},
        "scale": {"status": "unknown"}, "entities": [
            {"kind": "node", "id": "node-1", "source": "vision",
             "confidence": 0.5, "recognition_confidence": 0.5, "verified": False,
             "image_geometry": {"point": [0.0, 0.5]}, "target": {"node": 1},
             "payload": {}}],
        "dimensions": [], "intersections": [], "issues": [], "revision": 4,
        "model": None, "merge_plan": {"stale": True},
        "confirmation": {"stale": True},
    }


def test_queue_orders_blockers_by_workflow_priority_and_counts(qt_app):
    panel = IssuePanel()
    value = {"issues": [issue("warn", "topology", "warning"),
                        issue("low", "low_confidence"),
                        issue("plane", "work_plane_unconfirmed")]}
    panel.set_draft(value)
    assert panel.summary.text() == "阻断 2 · 警告 1"
    assert panel.list.item(0).data(256) == "plane"
    assert panel.list.item(1).data(256) == "low"
    assert panel.list.item(2).data(256) == "warn"


def test_intersection_issue_exposes_only_explicit_topology_actions(qt_app):
    panel = IssuePanel()
    emitted = []
    panel.resolution_requested.connect(lambda identifier, action: emitted.append((identifier, action)))
    panel.set_draft({"issues": [issue("crossing", "intersection_unknown")]})
    assert panel.btn_confirm.isHidden()
    assert not panel.btn_connect.isHidden()
    panel.btn_cross.click()
    assert emitted == [("crossing", "cross")]


def test_low_confidence_selection_links_issue_to_overlay_and_confirmation(qt_app):
    value = base_draft()
    value["issues"] = [issue("low", "low_confidence", refs=["node:1"])]
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(value)
    assert panel._highlight_refs == {"node:1"}
    panel.issue_panel.btn_confirm.click()
    entity = panel._v2_draft["entities"][0]
    assert entity["source"] == "user" and entity["verified"] is True
    assert entity["confidence"] is None
    assert panel._v2_draft["revision"] == 5
    assert panel._v2_draft["confirmation"] is None
    assert panel._v2_draft["merge_plan"] is None
    assert panel.issue_panel.summary.text() == "阻断 0 · 警告 0"


def test_intersection_action_updates_draft_and_cannot_use_global_checkbox(qt_app):
    value = detect_topology(base_draft())
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(value)
    assert not panel.chk_confirm.isEnabled()
    assert not panel.btn_load.isEnabled()
    panel.issue_panel.btn_connect.click()
    crossing = panel._v2_draft["intersections"][0]
    assert crossing["decision"] == "connect"
    assert crossing["node_id"] == 5
    assert panel._v2_draft["revision"] == 5


def test_perspective_confirmation_is_a_specific_recorded_action(qt_app):
    value = base_draft()
    value["issues"] = [issue("perspective", "perspective")]
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(value)
    panel.issue_panel.btn_confirm.click()
    assert panel._v2_draft["source"]["preprocessing"]["perspective_status"] == "accepted"
    resolved = panel._v2_draft["issues"][0]
    assert resolved["status"] == "resolved" and resolved["resolved_by"] == "user"


def test_issue_edit_stays_synchronized_with_controller_revision(qt_app):
    value = base_draft()
    value.update(format="space-frame-recognition-draft/v2",
                 merge_plan=None, confirmation=None)
    value["issues"] = [issue("perspective", "perspective")]
    value.setdefault("dimensions", [])
    value.setdefault("intersections", [])
    state = MultimodalControllerState(image_hash="a" * 64, draft=value)
    panel = SketchPanel(Session(), IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)
    panel.issue_panel.btn_confirm.click()
    assert panel._v2_draft == state.draft
    assert state.draft["revision"] == 5
    assert state.draft["confirmation"] is None


def ready_v2_draft():
    value = base_draft()
    value.update(format="space-frame-recognition-draft/v2",
                 model=None, merge_plan=None, confirmation=None)
    value["source"].update(image_hash="a" * 64, original_path="drawing.png")
    value["source"]["preprocessing"]["perspective_status"] = "accepted"
    value["work_plane"].update(status="confirmed", offset=0.0,
                               axis_mapping={"first_axis": "X", "second_axis": "Z",
                                             "offset_axis": "Y",
                                             "image_right_sign": 1,
                                             "image_up_sign": 1})
    value["scale"] = {"status": "confirmed", "length_per_pixel": 0.01,
                      "unit": "m/px", "anchor_node": 1,
                      "anchor_coordinates_xyz": [0.0, 0.0, 0.0],
                      "evidence_ids": ["dimension:test"]}
    value["issues"] = []
    return value


def test_v2_review_reaches_atomic_commit_and_single_undo_step(qt_app):
    session = Session()
    value = ready_v2_draft()
    state = MultimodalControllerState(
        workflow_instance_id="workflow", image_hash="a" * 64,
        draft=deepcopy(value))
    panel = SketchPanel(session, IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)

    assert panel.chk_confirm.isEnabled()
    assert state.commit_preview is not None
    assert not panel.btn_load.isEnabled()
    panel.chk_confirm.setChecked(True)
    assert panel.btn_load.isEnabled()
    panel._load_model()

    assert len(session.model["nodes"]) == 4
    assert session.multimodal_provenance["image_hash"] == "a" * 64
    assert len(session.history) == 1
    assert "原子加载" in panel.lbl_status.text()
    assert session.undo() and session.model == {}
    assert session.multimodal_provenance is None


def test_v2_unknown_scale_uses_reference_member_control(qt_app):
    value = ready_v2_draft()
    value["scale"].update(status="unknown", length_per_pixel=None,
                          evidence_ids=[])
    value["issues"] = [issue("scale_unknown", "scale_unknown")]
    state = MultimodalControllerState(
        workflow_instance_id="workflow", image_hash="a" * 64,
        draft=deepcopy(value))
    panel = SketchPanel(Session(), IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)

    assert not panel.scale_widget.isHidden()
    panel.spn_reference_length.setValue(2.0)
    panel._apply_scale()
    assert panel._v2_draft["scale"]["status"] == "confirmed"
    assert panel.scale_widget.isHidden()
    assert panel.chk_confirm.isEnabled()


def test_v2_add_only_requires_explicit_node_reuse_before_commit(qt_app):
    session = Session(model={
        "schema_version": 1, "units": "N-m-Pa", "materials": [], "sections": [],
        "nodes": [{"id": 10, "x": 0.0, "y": 0.0, "z": 0.0}],
        "members": [], "supports": [], "load_cases": [],
    })
    value = ready_v2_draft()
    state = MultimodalControllerState(
        workflow_instance_id="workflow", image_hash="a" * 64,
        draft=deepcopy(value))
    panel = SketchPanel(session, IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)

    assert panel.issue_panel.current_issue()["category"] == "merge_collision"
    assert not panel.issue_panel.btn_reuse.isHidden()
    assert not panel.chk_confirm.isEnabled()
    panel.issue_panel.btn_reuse.click()
    assert panel._v2_node_reuse == {1: 10}
    assert panel.chk_confirm.isEnabled()
    assert any(item["op"] == "reuse"
               for item in state.commit_preview["operations"])
    panel.chk_confirm.setChecked(True)
    panel._load_model()
    assert len(session.model["nodes"]) == 4
    assert len(session.history) == 1


def test_v2_node_move_updates_evidence_and_invalidates_old_preview(qt_app):
    value = ready_v2_draft()
    value["image_model"]["nodes"] = value["image_model"]["nodes"][:2]
    value["image_model"]["members"] = value["image_model"]["members"][:1]
    value["entities"] = [
        {"kind": "node", "id": "node-1", "source": "vision", "confidence": 0.9,
         "recognition_confidence": 0.9, "verified": False,
         "image_geometry": {"point": [0.0, 0.5]}, "target": {"node": 1},
         "payload": {}},
        {"kind": "node", "id": "node-2", "source": "vision", "confidence": 0.9,
         "recognition_confidence": 0.9, "verified": False,
         "image_geometry": {"point": [1.0, 0.5]}, "target": {"node": 2},
         "payload": {}},
        {"kind": "member", "id": "member-1", "source": "vision", "confidence": 0.9,
         "recognition_confidence": 0.9, "verified": False,
         "image_geometry": {"line": [[0.0, 0.5], [1.0, 0.5]]},
         "target": {"member": 1}, "payload": {}},
    ]
    value = add_member_length_dimension(value, 1, 1.0)
    state = MultimodalControllerState(
        workflow_instance_id="workflow", image_hash="a" * 64,
        draft=deepcopy(value))
    panel = SketchPanel(Session(), IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)
    old_preview = state.commit_preview["preview_id"]

    panel._move_v2_node_preview(2, 0.9, 0.4)
    assert panel._v2_draft["merge_plan"] is None
    panel._finish_v2_node_move(2)

    node = panel._v2_draft["image_model"]["nodes"][1]
    entity = panel._v2_draft["entities"][1]
    member = panel._v2_draft["entities"][2]
    assert (node["u"], node["v"]) == (0.9, 0.4)
    assert entity["source"] == "user" and entity["verified"] is True
    assert member["image_geometry"]["line"][1] == [0.9, 0.4]
    assert state.commit_preview["preview_id"] != old_preview
    assert panel.chk_confirm.isEnabled()


def test_v2_member_add_and_delete_update_authoritative_draft(qt_app):
    value = ready_v2_draft()
    value["image_model"]["nodes"] = [
        {"id": 1, "u": 0.0, "v": 0.5}, {"id": 2, "u": 0.5, "v": 0.5},
        {"id": 3, "u": 1.0, "v": 0.5}]
    value["image_model"]["members"] = [{"id": 1, "i": 1, "j": 2}]
    value["entities"] = []
    state = MultimodalControllerState(
        workflow_instance_id="workflow", image_hash="a" * 64,
        draft=deepcopy(value))
    panel = SketchPanel(Session(), IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)
    revision = state.draft["revision"]

    panel.cmb_member_i.setCurrentIndex(panel.cmb_member_i.findData(2))
    panel.cmb_member_j.setCurrentIndex(panel.cmb_member_j.findData(3))
    panel._add_member()
    assert len(state.draft["image_model"]["members"]) == 2
    assert state.draft["revision"] == revision + 1
    assert any(item.get("target", {}).get("member") == 2
               for item in state.draft["entities"])

    panel.cmb_remove_member.setCurrentIndex(panel.cmb_remove_member.findData(2))
    panel._remove_member()
    assert state.draft["image_model"]["members"] == [{"id": 1, "i": 1, "j": 2}]
    assert state.draft["revision"] == revision + 2


def test_v2_support_editor_updates_image_model_and_trace_entity(qt_app):
    value = ready_v2_draft()
    state = MultimodalControllerState(
        workflow_instance_id="workflow", image_hash="a" * 64,
        draft=deepcopy(value))
    panel = SketchPanel(Session(), IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)
    panel.cmb_support_node.setCurrentIndex(panel.cmb_support_node.findData(1))
    panel.txt_support_name.setText("Base-1")
    panel.cmb_support_type.setCurrentIndex(1)
    panel._apply_support_edit()

    support = state.draft["image_model"]["supports"][0]
    assert support == {"name": "Base-1", "node": 1,
                       "fix": [1, 1, 1, 0, 0, 0]}
    entity = next(item for item in state.draft["entities"]
                  if item["kind"] == "support")
    assert entity["target"]["support"] == {"node": 1, "name": "Base-1"}
    assert entity["source"] == "user"

    panel._remove_support_edit()
    assert state.draft["image_model"]["supports"] == []
    assert not any(item["kind"] == "support" for item in state.draft["entities"])


def test_v2_named_nodal_load_editor_keeps_traceable_units(qt_app):
    value = ready_v2_draft()
    state = MultimodalControllerState(
        workflow_instance_id="workflow", image_hash="a" * 64,
        draft=deepcopy(value))
    panel = SketchPanel(Session(), IdleRunner())
    panel._v2_state = state
    panel.set_v2_draft(value)
    panel.txt_new_case.setText("Wind")
    panel._add_load_case_edit()
    panel.cmb_load_node.setCurrentIndex(panel.cmb_load_node.findData(2))
    panel.cmb_load_name.setEditText("P-top")
    panel.load_spins[1].setValue(-1200.0)
    panel._apply_nodal_load_edit()

    load = state.draft["image_model"]["load_cases"][0]["nodal_loads"][0]
    assert load["name"] == "P-top" and load["node"] == 2
    assert load["load"][1] == -1200.0
    assert load["units"] == ["N", "N", "N", "N*m", "N*m", "N*m"]
    entity = next(item for item in state.draft["entities"] if item["kind"] == "load")
    assert entity["target"]["load"]["name"] == "P-top"

    panel._remove_nodal_load_edit()
    assert state.draft["image_model"]["load_cases"][0]["nodal_loads"] == []
    assert not any(item["kind"] == "load" for item in state.draft["entities"])
