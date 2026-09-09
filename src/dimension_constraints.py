"""Deterministic dimension evidence and image-scale resolution for v2 drafts."""

from __future__ import annotations

import math
from copy import deepcopy
from statistics import median
from typing import Any, Mapping

from multimodal_contract import SCALE_CONFLICT_RATIO
from units import MM, SI, system


LINEAR_KINDS = {
    "member_length", "node_distance", "horizontal_distance", "vertical_distance",
}
_UNIT_TO_M = {
    "m": system(SI).length_to_m,
    "mm": system(MM).length_to_m,
    "cm": 0.01,
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def length_to_m(value: Any, unit: Any) -> float | None:
    """Convert a positive dimension to metres; missing/unknown units are unusable."""
    if not _finite(value) or float(value) <= 0 or not isinstance(unit, str):
        return None
    factor = _UNIT_TO_M.get(unit.strip().lower())
    return None if factor is None else float(value) * factor


def default_anchor_node(image_model: Mapping[str, Any]) -> int:
    """Choose left-most, then lowest, then smallest-id image node."""
    nodes = image_model.get("nodes") or []
    if not nodes:
        raise ValueError("image_model 没有节点")
    try:
        return int(min(nodes, key=lambda item: (
            float(item["u"]), -float(item["v"]), int(item["id"])))["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("image_model 节点坐标无效") from exc


def _node_points(image_model: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for node in image_model.get("nodes") or []:
        node_id = int(node["id"])
        point = float(node["u"]), float(node["v"])
        if node_id in result or not all(_finite(value) and 0 <= value <= 1
                                        for value in point):
            raise ValueError("image_model 节点 ID 或坐标无效")
        result[node_id] = point
    return result


def _dimension_nodes(dimension: Mapping[str, Any],
                     image_model: Mapping[str, Any]) -> tuple[int, int]:
    target = dimension.get("target") or {}
    if dimension.get("kind") == "member_length":
        member_id = target.get("member")
        members = [item for item in image_model.get("members") or []
                   if item.get("id") == member_id]
        if len(members) != 1:
            raise ValueError(f"尺寸 {dimension.get('id')} 引用未知杆件 {member_id}")
        return int(members[0]["i"]), int(members[0]["j"])
    nodes = target.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 2:
        raise ValueError(f"尺寸 {dimension.get('id')} 必须引用两个节点")
    return int(nodes[0]), int(nodes[1])


def dimension_pixel_length(dimension: Mapping[str, Any],
                           image_model: Mapping[str, Any],
                           width_px: int, height_px: int) -> float:
    if width_px <= 1 or height_px <= 1:
        raise ValueError("派生图宽高必须大于 1")
    points = _node_points(image_model)
    i, j = _dimension_nodes(dimension, image_model)
    try:
        a, b = points[i], points[j]
    except KeyError as exc:
        raise ValueError(f"尺寸 {dimension.get('id')} 引用悬空节点") from exc
    dx = abs((b[0] - a[0]) * (width_px - 1))
    dy = abs((b[1] - a[1]) * (height_px - 1))
    kind = dimension.get("kind")
    if kind == "horizontal_distance":
        pixels = dx
    elif kind == "vertical_distance":
        pixels = dy
    else:
        pixels = math.hypot(dx, dy)
    if pixels <= 0 or not math.isfinite(pixels):
        raise ValueError(f"尺寸 {dimension.get('id')} 的目标像素长度为零")
    return pixels


def solve_scale(image_model: Mapping[str, Any], dimensions: list[dict[str, Any]],
                width_px: int, height_px: int, *,
                trusted_dimension_id: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a v2 scale block and copied dimensions with conflict decisions."""
    updated = deepcopy(dimensions)
    candidates: list[tuple[str, float, dict[str, Any]]] = []
    for item in updated:
        if item.get("kind") not in LINEAR_KINDS or item.get("status") != "confirmed":
            continue
        metres = length_to_m(item.get("value"), item.get("unit"))
        if metres is None:
            continue
        evidence_id = f"dimension:{item.get('id')}"
        pixels = dimension_pixel_length(item, image_model, width_px, height_px)
        candidates.append((evidence_id, metres / pixels, item))
    candidates.sort(key=lambda entry: entry[0])

    if trusted_dimension_id is not None:
        trusted_ref = (trusted_dimension_id if str(trusted_dimension_id).startswith("dimension:")
                       else f"dimension:{trusted_dimension_id}")
        chosen = [entry for entry in candidates if entry[0] == trusted_ref]
        if len(chosen) != 1:
            raise ValueError("指定的可信尺寸不是可用的 confirmed 线性尺寸")
        for evidence_id, _, item in candidates:
            if evidence_id != trusted_ref:
                item["status"] = "rejected_conflict"
        return ({"status": "confirmed", "length_per_pixel": chosen[0][1],
                 "unit": "m/px", "anchor_node": default_anchor_node(image_model),
                 "anchor_coordinates_xyz": [0.0, 0.0, 0.0],
                 "evidence_ids": [trusted_ref]}, updated)

    if not candidates:
        return ({"status": "unknown", "length_per_pixel": None,
                 "unit": "m/px", "anchor_node": default_anchor_node(image_model),
                 "anchor_coordinates_xyz": [0.0, 0.0, 0.0],
                 "evidence_ids": []}, updated)
    middle = float(median(entry[1] for entry in candidates))
    ratios = [abs(value - middle) / middle for _, value, _ in candidates]
    conflict = any(ratio > SCALE_CONFLICT_RATIO
                   and not math.isclose(ratio, SCALE_CONFLICT_RATIO,
                                        rel_tol=0.0, abs_tol=1e-12)
                   for ratio in ratios)
    return ({"status": "conflict" if conflict else "confirmed",
             "length_per_pixel": None if conflict else middle,
             "unit": "m/px", "anchor_node": default_anchor_node(image_model),
             "anchor_coordinates_xyz": [0.0, 0.0, 0.0],
             "evidence_ids": [] if conflict else [entry[0] for entry in candidates]}, updated)


def apply_scale_to_draft(draft: Mapping[str, Any], *,
                         trusted_dimension_id: str | None = None) -> dict[str, Any]:
    """Purely resolve scale from a v2 draft; the input draft is never modified."""
    updated = deepcopy(dict(draft))
    source = updated.get("source") or {}
    scale, dimensions = solve_scale(
        updated.get("image_model") or {}, updated.get("dimensions") or [],
        int(source.get("width_px", 0)), int(source.get("height_px", 0)),
        trusted_dimension_id=trusted_dimension_id)
    old = updated.get("scale") or {}
    for key in ("anchor_node", "anchor_coordinates_xyz"):
        if old.get(key) is not None:
            scale[key] = deepcopy(old[key])
    updated["scale"] = scale
    updated["dimensions"] = dimensions
    _sync_scale_issues(updated)
    return updated


def add_member_length_dimension(draft: Mapping[str, Any], member_id: int,
                                value: float, unit: str = "m") -> dict[str, Any]:
    """Add one traceable user-confirmed member-length dimension to a draft copy."""
    updated = deepcopy(dict(draft))
    image_model = updated.get("image_model") or {}
    members = [item for item in image_model.get("members") or []
               if item.get("id") == member_id]
    if len(members) != 1 or length_to_m(value, unit) is None:
        raise ValueError("杆件或实际长度无效")
    points = _node_points(image_model)
    member = members[0]
    try:
        start, end = points[int(member["i"])], points[int(member["j"])]
    except KeyError as exc:
        raise ValueError("杆件引用悬空节点") from exc
    dimensions = updated.setdefault("dimensions", [])
    used = {str(item.get("id")) for item in dimensions}
    sequence = 1
    while f"user-member-{member_id}-{sequence}" in used:
        sequence += 1
    identifier = f"user-member-{member_id}-{sequence}"
    dimensions.append({
        "id": identifier, "kind": "member_length",
        "text": f"{float(value):g} {unit}", "value": float(value), "unit": unit,
        "image_geometry": {"line": [list(start), list(end)]},
        "target": {"member": member_id}, "confidence": None,
        "recognition_confidence": None, "verified": True, "source": "user",
        "status": "confirmed",
    })
    return updated


def _sync_scale_issues(draft: dict[str, Any]) -> None:
    status = draft["scale"]["status"]
    active = {"unknown": "scale_unknown", "conflict": "scale_conflict"}.get(status)
    issues = draft.setdefault("issues", [])
    by_category = {item.get("category"): item for item in issues
                   if item.get("category") in {"scale_unknown", "scale_conflict"}}
    for category in ("scale_unknown", "scale_conflict"):
        issue = by_category.get(category)
        if category == active:
            if issue is None:
                issue = {"id": category, "category": category,
                         "severity": "blocking", "entity_refs": [],
                         "message": ("缺少可用的线性尺寸证据" if category == "scale_unknown"
                                     else "尺寸证据相互冲突"),
                         "status": "open", "resolution": None,
                         "resolved_by": None}
                issues.append(issue)
            else:
                issue.update(status="open", resolution=None, resolved_by=None,
                             severity="blocking")
        elif issue is not None and issue.get("status") == "open":
            issue.update(status="resolved", resolution="尺度状态已重新计算",
                         resolved_by="system")
