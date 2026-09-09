"""Deterministic 2-D intersection detection and topology decisions for v2 drafts."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from image_preprocess import image_to_model
from multimodal_contract import (ENDPOINT_MATCH_ABS_PX,
                                 ENDPOINT_MATCH_DIAGONAL_RATIO,
                                 point_match_threshold)


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _point(node: Mapping[str, Any]) -> tuple[float, float]:
    point = float(node["u"]), float(node["v"])
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in point):
        raise ValueError("节点图片坐标无效")
    return point


def _geometry(image_model: Mapping[str, Any]):
    nodes = {int(item["id"]): _point(item) for item in image_model.get("nodes") or []}
    members = {int(item["id"]): item for item in image_model.get("members") or []}
    if len(nodes) != len(image_model.get("nodes") or []) \
            or len(members) != len(image_model.get("members") or []):
        raise ValueError("节点或杆件 ID 重复")
    for member in members.values():
        if int(member["i"]) not in nodes or int(member["j"]) not in nodes:
            raise ValueError(f"杆件 {member['id']} 引用悬空节点")
    return nodes, members


def _intersection(a, b, c, d):
    r, s = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
    denominator = _cross(r, s)
    q = c[0] - a[0], c[1] - a[1]
    if abs(denominator) <= 1e-12:
        return None
    ta, tb = _cross(q, s) / denominator, _cross(q, r) / denominator
    if -1e-12 <= ta <= 1 + 1e-12 and -1e-12 <= tb <= 1 + 1e-12:
        return ta, tb, (a[0] + ta * r[0], a[1] + ta * r[1])
    return None


def _collinear_overlap(a, b, c, d) -> bool:
    axis = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1
    overlap = min(max(a[axis], b[axis]), max(c[axis], d[axis])) \
        - max(min(a[axis], b[axis]), min(c[axis], d[axis]))
    return overlap > 1e-12


def _endpoint_tolerance(width: int, height: int) -> float:
    return max(ENDPOINT_MATCH_ABS_PX,
               math.hypot(width - 1, height - 1) * ENDPOINT_MATCH_DIAGONAL_RATIO)


def _snap_parameter(value: float, length_px: float, tolerance_px: float) -> float:
    if value * length_px <= tolerance_px:
        return 0.0
    if (1.0 - value) * length_px <= tolerance_px:
        return 1.0
    return float(value)


def _issue(identifier: str, category: str, refs: list[str], message: str) -> dict[str, Any]:
    return {"id": identifier, "category": category, "severity": "blocking",
            "entity_refs": sorted(set(refs)), "message": message, "status": "open",
            "resolution": None, "resolved_by": None}


def detect_topology(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute pairwise 2-D intersections and stable blocking issues."""
    updated = deepcopy(dict(draft))
    image_model = updated.get("image_model") or {}
    nodes, members = _geometry(image_model)
    source = updated.get("source") or {}
    width, height = int(source.get("width_px", 0)), int(source.get("height_px", 0))
    if width <= 1 or height <= 1:
        raise ValueError("派生图宽高必须大于 1")
    tolerance = _endpoint_tolerance(width, height)
    old = {tuple(item.get("members") or []): item
           for item in updated.get("intersections") or []}
    intersections, generated_issues = [], []
    member_items = sorted(members.items())
    for index, (aid, first) in enumerate(member_items):
        ai, aj = int(first["i"]), int(first["j"])
        a, b = nodes[ai], nodes[aj]
        for bid, second in member_items[index + 1:]:
            bi, bj = int(second["i"]), int(second["j"])
            c, d = nodes[bi], nodes[bj]
            pair = [aid, bid]
            if {ai, aj} == {bi, bj}:
                generated_issues.append(_issue(
                    f"duplicate-member-{aid}-{bid}", "topology",
                    [f"member:{aid}", f"member:{bid}"], "重复杆件"))
                continue
            hit = _intersection(a, b, c, d)
            if hit is None:
                # Parallel collinear overlap is blocking rather than guessed.
                r = (b[0] - a[0], b[1] - a[1])
                if (abs(_cross(r, (c[0] - a[0], c[1] - a[1]))) <= 1e-12
                        and _collinear_overlap(a, b, c, d)):
                    generated_issues.append(_issue(
                        f"overlap-{aid}-{bid}", "topology",
                        [f"member:{aid}", f"member:{bid}"], "共线杆件重叠"))
                continue
            sa, sb, point = hit
            la = math.hypot((b[0] - a[0]) * (width - 1),
                            (b[1] - a[1]) * (height - 1))
            lb = math.hypot((d[0] - c[0]) * (width - 1),
                            (d[1] - c[1]) * (height - 1))
            sa, sb = (_snap_parameter(sa, la, tolerance),
                      _snap_parameter(sb, lb, tolerance))
            previous = old.get(tuple(pair), {})
            shared_endpoints = {ai, aj} & {bi, bj}
            if shared_endpoints:
                decision, node_id = "connect", min(shared_endpoints)
            else:
                decision = previous.get("decision", "unknown")
                node_id = previous.get("node_id") if decision == "connect" else None
            identifier = f"I-{aid}-{bid}"
            intersections.append({
                "id": identifier, "members": pair, "point": [float(point[0]), float(point[1])],
                "decision": decision, "node_id": node_id,
                "member_s": {str(aid): sa, str(bid): sb},
            })
            if decision == "unknown":
                generated_issues.append(_issue(
                    f"intersection-unknown-{aid}-{bid}", "intersection_unknown",
                    [f"intersection:{identifier}"], "交点连接关系尚未确认"))
    retained = [item for item in updated.get("issues") or []
                if not (str(item.get("id", "")).startswith(("duplicate-member-", "overlap-",
                                                             "intersection-unknown-")))]
    updated["intersections"] = intersections
    updated["issues"] = retained + generated_issues
    return updated


def resolve_intersection(draft: Mapping[str, Any], intersection_id: str,
                         decision: str) -> dict[str, Any]:
    """Resolve one intersection as connect or cross on a copied draft."""
    if decision not in {"connect", "cross"}:
        raise ValueError("交点决策必须是 connect 或 cross")
    updated = deepcopy(dict(draft))
    matches = [item for item in updated.get("intersections") or []
               if str(item.get("id")) == str(intersection_id)]
    if len(matches) != 1:
        raise ValueError("未知交点")
    item = matches[0]
    image_model = updated["image_model"]
    nodes, members = _geometry(image_model)
    endpoint_ids = []
    for member_id in item["members"]:
        member = members[int(member_id)]
        s = item["member_s"][str(member_id)]
        if s == 0.0:
            endpoint_ids.append(int(member["i"]))
        elif s == 1.0:
            endpoint_ids.append(int(member["j"]))
    if decision == "cross":
        if endpoint_ids:
            raise ValueError("构件端点形成的 T/端点连接不能标记为跨越")
        point = tuple(float(value) for value in item["point"])
        source = updated.get("source") or {}
        width, height = int(source.get("width_px", 0)), int(source.get("height_px", 0))
        tolerance = point_match_threshold(width, height)
        if any(math.hypot((point[0] - location[0]) * (width - 1),
                          (point[1] - location[1]) * (height - 1)) <= tolerance
               for location in nodes.values()):
            raise ValueError("交点处已有节点，必须连接或先删除该节点")
        item.update(decision="cross", node_id=None)
    else:
        if endpoint_ids:
            point = tuple(float(value) for value in item["point"])
            ranked = sorted((math.dist(point, nodes[node_id]), node_id)
                            for node_id in set(endpoint_ids))
            if len(ranked) > 1 and math.isclose(ranked[0][0], ranked[1][0], abs_tol=1e-12):
                raise ValueError("交点附近存在多个不同端点，需先合并或移动节点")
            node_id = ranked[0][1]
            # The tolerance decision becomes exact formal topology: move the reused
            # endpoint onto the mathematical intersection so the compiler sees it.
            for node in image_model["nodes"]:
                if int(node["id"]) == node_id:
                    node["u"], node["v"] = point
                    break
        else:
            node_id = max(nodes, default=0) + 1
            u, v = (float(value) for value in item["point"])
            image_model.setdefault("nodes", []).append({"id": node_id, "u": u, "v": v})
            updated.setdefault("entities", []).append({
                "kind": "node", "id": f"node-{node_id}", "confidence": None,
                "recognition_confidence": None, "verified": True,
                "image_geometry": {"point": [u, v]}, "source": "user",
                "target": {"node": node_id}, "payload": {},
            })
        item.update(decision="connect", node_id=node_id)
    for issue in updated.get("issues") or []:
        if issue.get("category") == "intersection_unknown" \
                and f"intersection:{intersection_id}" in issue.get("entity_refs", []):
            issue.update(status="resolved", resolution=decision, resolved_by="user")
    updated["revision"] = int(updated.get("revision", 0)) + 1
    updated["model"] = updated["merge_plan"] = updated["confirmation"] = None
    return updated


def delete_member(draft: Mapping[str, Any], member_id: int) -> dict[str, Any]:
    """Delete an unreferenced member, refusing dimensions, loads and intersections."""
    image_model = draft.get("image_model") or {}
    refs = []
    if any(item.get("target", {}).get("member") == member_id
           for item in draft.get("dimensions") or []):
        refs.append("dimension")
    members = {int(item["id"]): item for item in image_model.get("members") or []}
    for item in draft.get("intersections") or []:
        pair = [int(value) for value in item.get("members") or []]
        if member_id not in pair or len(pair) != 2:
            continue
        first, second = members.get(pair[0]), members.get(pair[1])
        shared = (set((int(first["i"]), int(first["j"])))
                  & set((int(second["i"]), int(second["j"])))) \
            if first and second else set()
        # 共享端点关系由几何自动重建，不是需要保护的人工交点决定。
        if not shared:
            refs.append("intersection")
    for block in [image_model, *(image_model.get("load_cases") or [])]:
        for collection in ("member_loads", "member_spans"):
            if any(item.get("member") == member_id for item in block.get(collection) or []):
                refs.append("load")
    if refs:
        raise ValueError(f"杆件 {member_id} 仍被引用: {', '.join(sorted(set(refs)))}")
    updated = deepcopy(dict(draft))
    before = len(updated["image_model"].get("members") or [])
    updated["image_model"]["members"] = [item for item in updated["image_model"]["members"]
                                                   if item.get("id") != member_id]
    if len(updated["image_model"]["members"]) == before:
        raise ValueError("未知杆件")
    updated["entities"] = [item for item in updated.get("entities") or []
                           if item.get("target", {}).get("member") != member_id]
    return detect_topology(updated)


def delete_node(draft: Mapping[str, Any], node_id: int) -> dict[str, Any]:
    image_model = draft.get("image_model") or {}
    refs = []
    if any(node_id in (item.get("i"), item.get("j")) for item in image_model.get("members") or []):
        refs.append("member")
    if any(item.get("node") == node_id for item in image_model.get("supports") or []):
        refs.append("support")
    for block in [image_model, *(image_model.get("load_cases") or [])]:
        for collection in ("nodal_loads", "settlements"):
            if any(item.get("node") == node_id for item in block.get(collection) or []):
                refs.append("load")
    if any(node_id in (item.get("target", {}).get("nodes") or [])
           for item in draft.get("dimensions") or []):
        refs.append("dimension")
    if refs:
        raise ValueError(f"节点 {node_id} 仍被引用: {', '.join(sorted(set(refs)))}")
    updated = deepcopy(dict(draft))
    before = len(updated["image_model"].get("nodes") or [])
    updated["image_model"]["nodes"] = [item for item in updated["image_model"]["nodes"]
                                                if item.get("id") != node_id]
    if len(updated["image_model"]["nodes"]) == before:
        raise ValueError("未知节点")
    updated["entities"] = [item for item in updated.get("entities") or []
                           if item.get("target", {}).get("node") != node_id]
    return detect_topology(updated)


def materialize_geometry(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize confirmed image geometry to SI Domain IR without engineering defaults."""
    work_plane, scale = draft.get("work_plane") or {}, draft.get("scale") or {}
    if work_plane.get("status") != "confirmed" or scale.get("status") != "confirmed":
        raise ValueError("工作平面和尺度必须先确认")
    image_model = draft.get("image_model") or {}
    source = draft.get("source") or {}
    nodes, _ = _geometry(image_model)
    anchor_id = int(scale["anchor_node"])
    if anchor_id not in nodes:
        raise ValueError("尺度锚点不存在")
    result_nodes = []
    for node_id, (u, v) in sorted(nodes.items()):
        xyz = image_to_model(
            u, v, width=int(source["width_px"]), height=int(source["height_px"]),
            anchor_uv=nodes[anchor_id], length_per_pixel=float(scale["length_per_pixel"]),
            plane=work_plane["plane"], offset=float(work_plane["offset"]),
            anchor_xyz=scale["anchor_coordinates_xyz"])
        result_nodes.append({"id": node_id, "x": xyz[0], "y": xyz[1], "z": xyz[2]})
    return {"schema_version": 1, "units": "N-m-Pa", "materials": [], "sections": [],
            "nodes": result_nodes, "members": deepcopy(image_model.get("members") or []),
            "supports": deepcopy(image_model.get("supports") or []),
            "nodal_loads": deepcopy(image_model.get("nodal_loads") or []),
            "member_loads": deepcopy(image_model.get("member_loads") or []),
            "member_spans": deepcopy(image_model.get("member_spans") or []),
            "settlements": deepcopy(image_model.get("settlements") or []),
            "load_cases": deepcopy(image_model.get("load_cases") or [])}
