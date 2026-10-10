"""识别草稿中局部分布荷载的人工编辑与几何校核。"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from sketch_topology import materialize_geometry


PARTIAL_KINDS = {"partial", "partial_trapezoid"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def member_length(draft: Mapping[str, Any], member_id: int) -> float:
    """按已确认尺度计算实际杆长，不能以图像比例代替米。"""
    model = materialize_geometry(draft)
    member = next((m for m in model["members"] if m["id"] == member_id), None)
    if member is None:
        raise ValueError("所选杆件不存在，请重新选择杆件。")
    nodes = {n["id"]: (n["x"], n["y"], n["z"]) for n in model["nodes"]}
    if member["i"] not in nodes or member["j"] not in nodes:
        raise ValueError("杆件端点不存在，请先修正节点连接。")
    length = math.dist(nodes[member["i"]], nodes[member["j"]])
    if not math.isfinite(length) or length <= 0:
        raise ValueError("杆件长度无效，请先修正端点和尺度。")
    return length


def _range_geometry(draft: Mapping[str, Any], load: Mapping[str, Any]) -> dict[str, Any]:
    length = member_length(draft, load["member"])
    a, b = load.get("a"), load.get("b")
    if not (_finite(a) and _finite(b) and 0 <= a < b <= length):
        raise ValueError(f"起止距离须满足 0 ≤ a < b ≤ 杆长 {length:g} m，请修正作用范围。")
    model = draft["image_model"]
    member = next(m for m in model["members"] if m["id"] == load["member"])
    nodes = {n["id"]: n for n in model["nodes"]}
    start, end = nodes[member["i"]], nodes[member["j"]]
    return {"line": [[start[axis] + (end[axis] - start[axis]) * distance / length
                      for axis in ("u", "v")] for distance in (a, b)]}


def _case(draft: dict, case_name: str) -> dict:
    matches = [c for c in draft["image_model"].get("load_cases") or [] if c.get("name") == case_name]
    if len(matches) != 1:
        raise ValueError("工况不存在或名称重复，请先在支座与节点荷载中创建唯一工况。")
    return matches[0]


def _resolve_exact_issues(draft: dict, target: dict, resolution: str) -> None:
    # 杆件级、全局及涉及多个荷载的问题仍需分别审核。
    ref = f"load:{target['case']}:{target['collection']}:{target['name']}"
    for issue in draft.get("issues") or []:
        if (issue.get("category") == "load_incomplete" and issue.get("status") == "open"
                and (set(issue.get("entity_refs") or []) == {ref}
                     or (issue.get("load_target") == target
                         and len(issue.get("entity_refs") or []) <= 1))):
            issue.update(status="resolved", resolved_by="user", resolution=resolution)


def set_partial_load(draft: Mapping[str, Any], case_name: str, name: str,
                     member_id: int, a: float, b: float, w1: list[float], *,
                     kind: str = "partial", w2: list[float] | None = None) -> dict:
    """只写入用户明确填写的 SI 范围和全局强度，保留其他工况及观察证据。"""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("荷载名称不能为空，请填写名称。")
    name = name.strip()
    if not isinstance(member_id, int) or isinstance(member_id, bool) or member_id <= 0:
        raise ValueError("杆件编号无效，请重新选择杆件。")
    if kind not in PARTIAL_KINDS:
        raise ValueError("请选择局部均布或局部梯形荷载。")
    for vector in ([w1, w2] if kind == "partial_trapezoid" else [w1]):
        if not isinstance(vector, (list, tuple)) or len(vector) != 3 or not all(map(_finite, vector)):
            raise ValueError("荷载强度必须包含三个有限分量，请按全局 X/Y/Z 方向填写 N/m。")
    updated = deepcopy(dict(draft))
    case = _case(updated, case_name)
    old = [item for item in case.get("member_spans") or [] if item.get("name") == name]
    if len(old) > 1 or (old and old[0].get("kind") not in PARTIAL_KINDS):
        raise ValueError("同名荷载重复或属于其他类型，请使用不同名称。")
    payload = {"name": name, "member": member_id, "kind": kind,
               "a": a, "b": b, "w1": list(w1)}
    if kind == "partial_trapezoid":
        payload["w2"] = list(w2)
    geometry = _range_geometry(updated, payload)
    target = {"load": {"case": case_name, "collection": "member_spans", "name": name}}
    entities = updated.setdefault("entities", [])
    observations = [deepcopy(e) for e in entities if e.get("kind") == "load" and e.get("target") == target]
    entities[:] = [e for e in entities if not (e.get("kind") == "load" and e.get("target") == target)]
    ident = f"user-span-{case_name}-{name}"
    while any(e.get("id") == ident for e in entities):
        ident += "-new"
    entity = {"id": ident, "kind": "load", "target": target, "payload": deepcopy(payload),
              "image_geometry": geometry, "source": "user", "verified": True,
              "confidence": None, "recognition_confidence": None}
    if observations or old:
        entity["original_observation"] = {"entities": observations, "loads": deepcopy(old)}
    entities.append(entity)
    case["member_spans"] = [item for item in case.get("member_spans") or [] if item.get("name") != name]
    case["member_spans"].append(payload)
    _resolve_exact_issues(updated, target["load"], "用户已重新填写范围与强度")
    updated["model"] = updated["merge_plan"] = updated["confirmation"] = None
    return refresh_partial_geometry(updated)


def remove_partial_load(draft: Mapping[str, Any], case_name: str, name: str) -> dict:
    """明确删除选定局部荷载，观察保存在审核记录，不改变其他引用的问题。"""
    updated = deepcopy(dict(draft))
    case = _case(updated, case_name)
    old = [item for item in case.get("member_spans") or [] if item.get("name") == name]
    if len(old) != 1 or old[0].get("kind") not in PARTIAL_KINDS:
        raise ValueError("所选局部荷载不存在或名称重复，请重新选择。")
    case["member_spans"] = [item for item in case["member_spans"] if item.get("name") != name]
    target = {"load": {"case": case_name, "collection": "member_spans", "name": name}}
    entities = updated.setdefault("entities", [])
    observations = [deepcopy(e) for e in entities if e.get("kind") == "load" and e.get("target") == target]
    entities[:] = [e for e in entities if not (e.get("kind") == "load" and e.get("target") == target)]
    updated.setdefault("edit_history", []).append({"action": "delete_partial_load",
        "target": target, "loads": old, "entities": observations})
    _resolve_exact_issues(updated, target["load"], "用户明确删除此局部荷载")
    updated["model"] = updated["merge_plan"] = updated["confirmation"] = None
    return refresh_partial_geometry(updated)


def refresh_partial_geometry(draft: Mapping[str, Any]) -> dict:
    """拖点或重设尺度后重新校核人工范围；越界时清除旧覆盖并保持阻断。"""
    updated = deepcopy(dict(draft))
    active = set()
    issues = updated.setdefault("issues", [])
    for entity in updated.get("entities") or []:
        target = (entity.get("target") or {}).get("load") or {}
        if (entity.get("kind") != "load" or entity.get("source") != "user"
                or not entity.get("verified") or target.get("collection") != "member_spans"):
            continue
        case = next((c for c in updated["image_model"].get("load_cases") or []
                     if c.get("name") == target.get("case")), {})
        load = next((s for s in case.get("member_spans") or [] if s.get("name") == target.get("name")), {})
        if load.get("kind") not in PARTIAL_KINDS:
            continue
        ident = f"user-partial-range:{entity['id']}"
        try:
            entity["image_geometry"] = _range_geometry(updated, load)
        except (ValueError, KeyError, TypeError) as exc:
            entity["image_geometry"] = {}
            active.add(ident)
            issue = next((i for i in issues if i.get("id") == ident), None)
            if issue is None:
                issue = {"id": ident, "category": "load_incomplete", "severity": "blocking",
                         "entity_refs": [f"load:{target['case']}:member_spans:{target['name']}"]}
                issues.append(issue)
            issue.update(status="open", message=f"局部荷载范围需要重新校核：{exc}",
                         resolution=None, resolved_by=None)
    for issue in issues:
        if str(issue.get("id", "")).startswith("user-partial-range:") and issue["id"] not in active:
            issue.update(status="resolved", resolution="局部范围已重新校核或明确删除", resolved_by="user")
    return updated
