"""独立参考的几何与交汇完整性检查；不读取预测，也不生成连接答案。"""

from __future__ import annotations

import math
from itertools import combinations

FORMAT = "space-frame-reference-topology/v1"
EPS = 1e-10
REFERENCE_POINT_TOLERANCE_PX = 1.0


def _id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("invalid_id")
    return str(value)


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in value)):
        raise ValueError("invalid_point")
    return tuple(value)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _meet(a, b, c, d):
    """直接检查两线段的唯一交汇，共线重叠不能充当可评分交点。"""
    r, s = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
    q = c[0] - a[0], c[1] - a[1]
    divisor = _cross(r, s)
    if abs(divisor) > EPS:
        t, u = _cross(q, s) / divisor, _cross(q, r) / divisor
        if -EPS <= t <= 1 + EPS and -EPS <= u <= 1 + EPS:
            return (a[0] + t * r[0], a[1] + t * r[1])
        return None
    if abs(_cross(q, r)) > EPS:
        return None
    axis = 0 if abs(r[0]) >= abs(r[1]) else 1
    low = max(min(a[axis], b[axis]), min(c[axis], d[axis]))
    high = min(max(a[axis], b[axis]), max(c[axis], d[axis]))
    if high < low - EPS:
        return None
    if high - low > EPS:
        raise ValueError("overlapping_members")
    t = (low - a[axis]) / r[axis]
    return a[0] + t * r[0], a[1] + t * r[1]


def check_reference_topology(truth: dict, *, required: bool = True) -> dict:
    """逐对检查调用前人工参考；检查不修改标注，也不从模型响应推断答案。"""
    contract = truth.get("topology_reference")
    if contract is None and not required:
        return {"valid": True, "legacy": True, "errors": []}
    errors = []
    if (not isinstance(contract, dict) or contract.get("format") != FORMAT
            or contract.get("scope") != "all_member_pairs" or contract.get("status") != "verified"):
        errors.append({"code": "contract_not_verified"})
    try:
        width, height = truth["width_px"], truth["height_px"]
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 1 for v in (width, height)):
            raise ValueError("invalid_image_dimensions")

        def distance(a, b):
            return math.hypot((a[0] - b[0]) * (width - 1), (a[1] - b[1]) * (height - 1))

        nodes, members = {}, {}
        for node in truth["nodes"]:
            ident = _id(node["id"])
            if ident in nodes:
                raise ValueError("duplicate_node")
            nodes[ident] = _point(node["point"])
        for member in truth["members"]:
            ident, i, j = _id(member["id"]), _id(member["i"]), _id(member["j"])
            if ident in members:
                raise ValueError("duplicate_member")
            if i not in nodes or j not in nodes:
                raise ValueError("dangling_member")
            if math.dist(nodes[i], nodes[j]) <= EPS:
                raise ValueError("zero_length_member")
            members[ident] = (i, j)
        if not nodes or not members:
            raise ValueError("empty_geometry")
        expected = {}
        for aid, bid in combinations(sorted(members), 2):
            first, second = members[aid], members[bid]
            if set(first) == set(second):
                errors.append({"code": "duplicate_member_geometry", "members": [aid, bid]})
                continue
            try:
                point = _meet(*(nodes[n] for n in (*first, *second)))
            except ValueError as exc:
                errors.append({"code": str(exc), "members": [aid, bid]})
                continue
            if point is not None:
                shared = set(first) & set(second)
                expected[(aid, bid)] = (point, shared)
        seen, identifiers = set(), set()
        for record in truth["intersections"]:
            ident = _id(record["id"])
            if ident in identifiers:
                errors.append({"code": "duplicate_intersection_id", "id": ident})
            identifiers.add(ident)
            pair = record.get("members")
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid_intersection_members")
            pair = tuple(sorted(_id(n) for n in pair))
            if pair in seen:
                errors.append({"code": "duplicate_intersection_pair", "members": list(pair)})
            seen.add(pair)
            if pair not in expected:
                errors.append({"code": "unexpected_intersection", "members": list(pair)})
                continue
            point, shared = expected[pair]
            if distance(_point(record["point"]), point) > REFERENCE_POINT_TOLERANCE_PX:
                errors.append({"code": "intersection_point_mismatch", "members": list(pair)})
            decision = record.get("decision")
            if decision not in ("connect", "cross"):
                errors.append({"code": "intersection_decision_unverified", "members": list(pair)})
            elif decision == "connect":
                node = _id(record.get("node_id"))
                if (node not in nodes or distance(nodes[node], point) > REFERENCE_POINT_TOLERANCE_PX
                        or (shared and node not in shared)):
                    errors.append({"code": "connection_node_mismatch", "members": list(pair)})
            elif shared or record.get("node_id") is not None:
                errors.append({"code": "cross_has_connected_node", "members": list(pair)})
        for pair in sorted(set(expected) - seen):
            errors.append({"code": "missing_intersection", "members": list(pair)})
    except (KeyError, TypeError, ValueError) as exc:
        errors.append({"code": "invalid_reference_geometry", "detail": str(exc)})
    return {"valid": not errors, "legacy": False, "errors": errors}
