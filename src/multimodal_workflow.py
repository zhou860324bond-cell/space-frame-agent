"""Deterministic v2 multimodal draft validation and controller state."""

from __future__ import annotations

import math
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from change_preview import model_digest
from multimodal_contract import canonical_digest

V2_DRAFT_FORMAT = "space-frame-recognition-draft/v2"


class MultimodalPhase(str, Enum):
    IDLE = "idle"
    IMAGE_LOADED = "image_loaded"
    RECOGNIZING = "recognizing"
    REVIEW_REQUIRED = "review_required"
    READY_TO_COMMIT = "ready_to_commit"
    COMMITTING = "committing"
    COMMITTED = "committed"
    RECOGNITION_FAILED = "recognition_failed"


_DRAFT_HASH_FIELDS = (
    "format", "image_model", "model", "merge_plan", "source", "work_plane",
    "scale", "entities", "dimensions", "intersections", "issues", "revision",
)


def draft_digest(draft: Mapping[str, Any]) -> str:
    return canonical_digest({key: draft.get(key) for key in _DRAFT_HASH_FIELDS})


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def validate_v2_draft(draft: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(draft, dict):
        return ["v2 草稿必须是对象"]
    required = {*_DRAFT_HASH_FIELDS, "confirmation"}
    missing = sorted(required - set(draft))
    if missing:
        errors.append(f"v2 草稿缺少字段: {missing}")
        return errors
    if draft.get("format") != V2_DRAFT_FORMAT:
        errors.append(f"format 必须是 {V2_DRAFT_FORMAT}")
    if not isinstance(draft.get("revision"), int) or draft["revision"] < 0:
        errors.append("revision 必须是非负整数")

    image_model = draft.get("image_model")
    if not isinstance(image_model, dict):
        errors.append("image_model 必须是对象")
        return errors
    nodes = image_model.get("nodes")
    members = image_model.get("members")
    if not isinstance(nodes, list) or not isinstance(members, list):
        errors.append("image_model.nodes/members 必须是数组")
        return errors
    node_ids: set[int] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), int):
            errors.append("image_model 节点必须包含整数 id")
            continue
        node_id = node["id"]
        if node_id <= 0 or node_id in node_ids:
            errors.append(f"image_model 节点 id 无效或重复: {node_id}")
        node_ids.add(node_id)
        if not (_finite(node.get("u")) and _finite(node.get("v"))
                and 0.0 <= float(node["u"]) <= 1.0
                and 0.0 <= float(node["v"]) <= 1.0):
            errors.append(f"image_model 节点 {node_id} 的 u/v 无效")
    member_ids: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("id"), int):
            errors.append("image_model 杆件必须包含整数 id")
            continue
        member_id = member["id"]
        i, j = member.get("i"), member.get("j")
        if member_id <= 0 or member_id in member_ids:
            errors.append(f"image_model 杆件 id 无效或重复: {member_id}")
        member_ids.add(member_id)
        if i not in node_ids or j not in node_ids:
            errors.append(f"image_model 杆件 {member_id} 引用悬空")
            continue
        edge = tuple(sorted((int(i), int(j))))
        if i == j or edge in edges:
            errors.append(f"image_model 杆件 {member_id} 自连接或重复")
        edges.add(edge)

    work_plane = draft.get("work_plane")
    if (not isinstance(work_plane, dict)
            or work_plane.get("status") not in ("proposed", "confirmed")
            or work_plane.get("plane") not in ("XY", "XZ", "YZ")):
        errors.append("work_plane 无效")
    scale = draft.get("scale")
    if not isinstance(scale, dict) or scale.get("status") not in (
            "unknown", "confirmed", "conflict"):
        errors.append("scale 无效")
    elif scale["status"] == "confirmed":
        if not _finite(scale.get("length_per_pixel")) \
                or float(scale["length_per_pixel"]) <= 0:
            errors.append("confirmed scale 必须包含有限正比例")
    elif scale.get("length_per_pixel") is not None:
        errors.append("未确认 scale 的 length_per_pixel 必须为 null")

    for key in ("entities", "dimensions", "intersections", "issues"):
        if not isinstance(draft.get(key), list):
            errors.append(f"{key} 必须是数组")
    model = draft.get("model")
    if model is not None and not isinstance(model, dict):
        errors.append("model 必须是对象或 null")
    merge_plan = draft.get("merge_plan")
    if merge_plan is not None and not isinstance(merge_plan, dict):
        errors.append("merge_plan 必须是对象或 null")
    return errors


def _proxy_points(model: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    nodes = model.get("nodes") or []
    ranges = []
    for order, axis in enumerate(("x", "y", "z")):
        values = [float(node.get(axis, 0.0)) for node in nodes]
        spread = max(values, default=0.0) - min(values, default=0.0)
        ranges.append((-spread, order, axis, values))
    ranges.sort()
    first, second = ranges[0], ranges[1]

    def normalized(values: list[float], value: float) -> float:
        low, high = min(values, default=0.0), max(values, default=0.0)
        return 0.5 if high == low else (value - low) / (high - low)

    result = {}
    for index, node in enumerate(nodes):
        result[int(node["id"])] = (
            normalized(first[3], first[3][index]),
            1.0 - normalized(second[3], second[3][index]),
        )
    return result


def migrate_v1_payload(payload: Mapping[str, Any], *, image_hash: str = "",
                       source_path: str = "") -> dict[str, Any]:
    """Deterministically preserve a v1 graph without trusting its scale."""
    model = deepcopy(payload.get("model", payload))
    if not isinstance(model, dict) or not model.get("nodes") or not model.get("members"):
        raise ValueError("v1 草稿缺少 nodes/members")
    points = _proxy_points(model)
    old_entities = payload.get("entities") or []
    point_entities = {
        int(item["id"]): item["image_geometry"]["point"]
        for item in old_entities
        if item.get("kind") == "node" and isinstance(item.get("image_geometry"), dict)
        and isinstance(item["image_geometry"].get("point"), list)
        and len(item["image_geometry"]["point"]) == 2
    }
    use_evidence = len(point_entities) == len(model["nodes"])
    if use_evidence:
        points = {node_id: (float(point[0]), float(point[1]))
                  for node_id, point in point_entities.items()}
    image_nodes = [{"id": int(node["id"]), "u": points[int(node["id"])][0],
                    "v": points[int(node["id"])][1]}
                   for node in model["nodes"]]
    image_members = [{"id": int(item["id"]), "i": int(item["i"]),
                      "j": int(item["j"]),
                      "material": item.get("material") or None,
                      "section": item.get("section") or None}
                     for item in model["members"]]
    old_by_kind_id = {(str(item.get("kind")), str(item.get("id"))): item
                      for item in old_entities if isinstance(item, dict)}

    def migrated_entity(kind: str, legacy_id: Any, geometry: dict[str, Any],
                        target: dict[str, Any], entity_payload: dict[str, Any]) -> dict[str, Any]:
        previous = old_by_kind_id.get((kind, str(legacy_id)), {})
        previous_confidence = previous.get("confidence")
        recognition_confidence = (float(previous_confidence)
                                  if _finite(previous_confidence)
                                  and 0.0 <= float(previous_confidence) <= 1.0
                                  else None)
        return {"kind": kind, "id": f"{kind}-{legacy_id}",
                "confidence": None, "recognition_confidence": recognition_confidence,
                "verified": False, "source": "migration",
                "image_geometry": deepcopy(previous.get("image_geometry") or geometry),
                "target": target, "payload": deepcopy(entity_payload)}

    entities = []
    for node in image_nodes:
        entities.append(migrated_entity(
            "node", node["id"], {"point": [node["u"], node["v"]]},
            {"node": node["id"]}, {}))
    for member in image_members:
        a = points[member["i"]]
        b = points[member["j"]]
        entities.append(migrated_entity(
            "member", member["id"], {"line": [list(a), list(b)]},
            {"member": member["id"]}, {}))
    supports = deepcopy(model.get("supports") or [])
    for index, support in enumerate(supports):
        support.setdefault("name", f"S{support['node']}-{index + 1}")
        node_id = int(support["node"])
        point = points[node_id]
        entities.append(migrated_entity(
            "support", support["name"], {"point": list(point)},
            {"support": {"node": node_id, "name": support["name"]}}, support))

    top_level_loads = {key: deepcopy(model.get(key) or []) for key in (
        "nodal_loads", "member_loads", "member_spans", "settlements")}
    load_cases = deepcopy(model.get("load_cases") or [])

    def add_load_entities(case_name: str, container: dict[str, Any]) -> None:
        for collection in ("nodal_loads", "member_loads", "member_spans", "settlements"):
            for index, load in enumerate(container.get(collection) or []):
                load.setdefault("name", f"{collection}-{index + 1}")
                if "node" in load:
                    geometry = {"point": list(points[int(load["node"])])}
                else:
                    member = next(item for item in image_members
                                  if item["id"] == int(load["member"]))
                    geometry = {"line": [list(points[member["i"]]),
                                         list(points[member["j"]])]}
                legacy_id = f"{case_name}:{load['name']}"
                entities.append(migrated_entity(
                    "load", legacy_id, geometry,
                    {"load": {"case": case_name, "collection": collection,
                              "name": load["name"]}}, load))

    add_load_entities("__top__", top_level_loads)
    for case in load_cases:
        add_load_entities(str(case.get("name", "default")), case)
    refs = [f"node:{item['id']}" for item in image_nodes]
    return {
        "format": V2_DRAFT_FORMAT,
        "image_model": {
            "nodes": image_nodes, "members": image_members,
            "supports": supports,
            **top_level_loads,
            "load_cases": load_cases,
        },
        "model": None,
        "merge_plan": None,
        "source": {"original_path": source_path, "image_hash": image_hash,
                   "preprocessing": {"perspective_status": "unconfirmed"}},
        "work_plane": {"status": "proposed", "plane": "XZ", "offset": 0.0,
                       "axis_mapping": {"first_axis": "X", "second_axis": "Z",
                                        "offset_axis": "Y", "image_right_sign": 1,
                                        "image_up_sign": 1}},
        "scale": {"status": "unknown", "length_per_pixel": None,
                  "unit": "m/px", "anchor_node": image_nodes[0]["id"],
                  "anchor_coordinates_xyz": [0.0, 0.0, 0.0],
                  "evidence_ids": []},
        "entities": entities,
        "dimensions": [], "intersections": [],
        "issues": [{"id": "migration-review", "category": "low_confidence",
                    "severity": "blocking", "entity_refs": refs,
                    "message": "旧版图片坐标需要人工确认",
                    "status": "open", "resolution": None, "resolved_by": None}],
        "revision": 0, "confirmation": None,
    }


@dataclass
class MultimodalControllerState:
    workflow_instance_id: str = ""
    image_hash: str = ""
    draft: dict[str, Any] | None = None
    job_status: str = "idle"
    job_id: str | None = None
    commit_status: str = "idle"
    commit_preview: dict[str, Any] | None = None
    commit_receipt: dict[str, Any] | None = None
    last_error: str | None = None
    commit_error: str | None = None
    _previous_draft: dict[str, Any] | None = field(default=None, repr=False)

    def load_image(self, image_hash: str) -> None:
        if not image_hash:
            raise ValueError("image_hash 不能为空")
        self.workflow_instance_id = str(uuid.uuid4())
        self.image_hash = image_hash
        self.draft = None
        self.job_status, self.job_id = "idle", None
        self.commit_status = "idle"
        self.commit_preview = self.commit_receipt = None
        self.last_error = self.commit_error = None
        self._previous_draft = None

    def start_recognition(self, job_id: str | None = None) -> str:
        if not self.image_hash or self.job_status == "running":
            raise RuntimeError("当前不能开始识别")
        self._previous_draft = deepcopy(self.draft)
        if self._previous_draft is not None:
            self._previous_draft["confirmation"] = None
            self._previous_draft["merge_plan"] = None
            self.draft = deepcopy(self._previous_draft)
        self.job_status = "running"
        self.job_id = job_id or str(uuid.uuid4())
        self.last_error = None
        self.commit_preview = self.commit_receipt = None
        return self.job_id

    def cancel_recognition(self, job_id: str) -> bool:
        if self.job_status != "running" or job_id != self.job_id:
            return False
        self.job_status, self.job_id = "idle", None
        self.draft = self._previous_draft
        self._previous_draft = None
        return True

    def complete_recognition(self, job_id: str, draft: dict[str, Any]) -> bool:
        if self.job_status != "running" or job_id != self.job_id:
            return False
        errors = validate_v2_draft(draft)
        if (draft.get("source") or {}).get("image_hash") != self.image_hash:
            errors.append("识别草稿 image_hash 与当前图片不一致")
        if errors:
            self.fail_recognition(job_id, "；".join(errors))
            return False
        self.draft = deepcopy(draft)
        self.job_status, self.job_id = "idle", None
        self.last_error = None
        self._previous_draft = None
        return True

    def fail_recognition(self, job_id: str, error: str) -> bool:
        if self.job_status != "running" or job_id != self.job_id:
            return False
        self.job_status, self.job_id = "idle", None
        self.draft = self._previous_draft
        self._previous_draft = None
        self.last_error = str(error)
        return True

    def edit_draft(self, draft: dict[str, Any]) -> None:
        if self.draft is None:
            raise RuntimeError("没有可编辑草稿")
        edited = deepcopy(draft)
        edited["revision"] = int(self.draft.get("revision", 0)) + 1
        edited["confirmation"] = None
        edited["merge_plan"] = None
        errors = validate_v2_draft(edited)
        if errors:
            raise ValueError("；".join(errors))
        self.draft = edited
        self.commit_preview = self.commit_receipt = None
        self.commit_error = None

    def set_commit_preview(self, preview: Mapping[str, Any]) -> None:
        if self.draft is None:
            raise RuntimeError("没有草稿")
        if preview.get("draft_hash") != draft_digest(self.draft):
            raise ValueError("preview draft_hash 已陈旧")
        if preview.get("merge_plan_hash") != canonical_digest(self.draft.get("merge_plan")):
            raise ValueError("preview merge_plan_hash 已陈旧")
        self.commit_preview = deepcopy(dict(preview))

    def install_commit_preview(self, prepared_draft: Mapping[str, Any],
                               preview: Mapping[str, Any]) -> None:
        """Install a plan produced from the current draft without faking an edit."""
        if self.draft is None:
            raise RuntimeError("没有草稿")
        current = deepcopy(self.draft)
        prepared = deepcopy(dict(prepared_draft))
        # model 是已确认 image_model/work-plane/scale 的确定性物化结果，
        # 与 merge_plan 一起安装，不属于一次新的人工编辑。
        current["model"] = deepcopy(prepared.get("model"))
        current["merge_plan"] = prepared["merge_plan"] = None
        current["confirmation"] = prepared["confirmation"] = None
        if draft_digest(current) != draft_digest(prepared):
            raise ValueError("合并计划不是由当前草稿生成")
        installed = deepcopy(dict(prepared_draft))
        errors = validate_v2_draft(installed)
        if errors:
            raise ValueError("；".join(errors))
        self.draft = installed
        self.set_commit_preview(preview)

    def phase(self, session_model: Mapping[str, Any]) -> MultimodalPhase:
        if self.commit_status == "running":
            return MultimodalPhase.COMMITTING
        if self.job_status == "running":
            return MultimodalPhase.RECOGNIZING
        if self.last_error and self.draft is None:
            return MultimodalPhase.RECOGNITION_FAILED
        if self._receipt_matches(session_model):
            return MultimodalPhase.COMMITTED
        if self.image_hash and self.draft is None:
            return MultimodalPhase.IMAGE_LOADED
        if self.draft is not None:
            return (MultimodalPhase.READY_TO_COMMIT
                    if self._ready(session_model) else MultimodalPhase.REVIEW_REQUIRED)
        return MultimodalPhase.IDLE

    def can_commit(self, session_model: Mapping[str, Any]) -> bool:
        return (self.job_status == "idle" and self.commit_status == "idle"
                and self.phase(session_model) == MultimodalPhase.READY_TO_COMMIT)

    def start_commit(self, session_model: Mapping[str, Any], *, confirmed_at: str) -> bool:
        if not self.can_commit(session_model) or self.draft is None \
                or self.commit_preview is None:
            return False
        self.draft["confirmation"] = {
            "revision": self.draft["revision"],
            "draft_hash": draft_digest(self.draft),
            "baseline_model_hash": model_digest(dict(session_model)),
            "preview_id": self.commit_preview["preview_id"],
            "diff_hash": self.commit_preview["diff_hash"],
            "confirmed_at": confirmed_at,
        }
        self.commit_status = "running"
        self.commit_error = None
        return True

    def finish_commit(self, *, success: bool, model_hash: str = "",
                      committed_at: str = "", history_step_id: str = "") -> None:
        if self.commit_status != "running" or self.draft is None:
            raise RuntimeError("没有进行中的提交")
        self.commit_status = "idle"
        if not success:
            self.commit_error = "提交失败"
            return
        self.commit_receipt = {
            "workflow_instance_id": self.workflow_instance_id,
            "image_hash": self.image_hash,
            "draft_hash": draft_digest(self.draft),
            "revision": self.draft["revision"],
            "model_hash": model_hash,
            "committed_at": committed_at,
            "history_step_id": history_step_id,
        }

    def _ready(self, session_model: Mapping[str, Any]) -> bool:
        if self.draft is None or validate_v2_draft(self.draft):
            return False
        draft = self.draft
        blocking = any(issue.get("severity") == "blocking"
                       and issue.get("status") == "open"
                       for issue in draft["issues"])
        perspective = ((draft.get("source") or {}).get("preprocessing") or {}).get(
            "perspective_status")
        preview = self.commit_preview
        return bool(
            draft["work_plane"].get("status") == "confirmed"
            and draft["scale"].get("status") == "confirmed"
            and isinstance(draft.get("model"), dict)
            and isinstance(draft.get("merge_plan"), dict)
            and perspective == "accepted" and not blocking and preview
            and preview.get("draft_hash") == draft_digest(draft)
            and preview.get("merge_plan_hash") == canonical_digest(draft["merge_plan"])
            and preview.get("baseline_model_hash") == model_digest(dict(session_model))
        )

    def _receipt_matches(self, session_model: Mapping[str, Any]) -> bool:
        if self.draft is None or not self.commit_receipt:
            return False
        receipt = self.commit_receipt
        return bool(
            receipt.get("workflow_instance_id") == self.workflow_instance_id
            and receipt.get("image_hash") == self.image_hash
            and receipt.get("draft_hash") == draft_digest(self.draft)
            and receipt.get("revision") == self.draft.get("revision")
            and receipt.get("model_hash") == model_digest(dict(session_model))
        )
