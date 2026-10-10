"""聚焦集中作用点的视觉复核；只提出拓扑候选，不生成工程数值。"""

from copy import deepcopy
import math

from multimodal_contract import point_match_threshold
from sketch_topology import insert_member_node


ACTION_REVIEW_PROMPT = """你只复核图片中外部集中力/力矩的作用位置和已有支座的可见符号，不输出完整结构模型。
上下文给出的节点/杆件是上一阶段的待审核几何，可能遗漏中间作用节点。
请逐一查看所有直箭头和圆弧箭头：直箭头取与杆件接触的箭头尖或作用位置，
圆弧力矩取圆弧中心在杆件上的作用点，不能取圆弧箭头尖。归一化坐标以图片左上角为原点，
u 向右，v 向下，范围 0..1。不能取文字中心、图像中心或按杆长均匀分配位置。
排除明确的支座反力、尺寸箭头、坐标轴和分布荷载箭头列；不能仅凭颜色判定外力或反力。
无法区分时将疑问写入 warnings，不编造点。即使力/矩没有数值，也要保留可见的作用点。
member 使用上下文已存在的杆件编号。不要生成荷载大小、单位换算、真实长度、应力或位移。
支座只查看 support_nodes 中已有的节点：报告可见图形，不能按静定性或习惯假设左右一铰一滚。
triangle 表示普通三角形且未见滚轮；triangle_on_rollers 表示三角形下明确有滚轮；
circle 表示梁下独立圆形支座；wall 表示杆端嵌入带剖线墙；看不清用 unknown。
没有明确的圆形/滚轮证据，不能因位于右端就写滚动。支座 point 取与梁附着位置，
不能取圆形支座中心或反力箭头。不要生成约束掩码、刚度或求解数值。
仅输出 JSON：{"actions":[{"id":"A1","kind":"force","member":1,
"point":[0.4,0.5],"text":"图上集中力符号","confidence":null}],
"supports":[{"node":1,"symbol":"triangle","point":[0.1,0.5],"text":"梁下三角形，未见滚轮"}],"warnings":[]}。
kind 仅 force/moment。id 唯一。text 只描述观察，不填数值荷载。warnings 为字符串列表。
没有集中作用点时 actions 为空；存在无法定位的符号时必须在 warnings 中说明。
"""


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_action_review(value):
    if (not isinstance(value, dict) or not {"actions", "warnings"} <= set(value)
            or set(value) - {"actions", "warnings", "supports"}):
        raise ValueError("作用点复核必须包含 actions 和 warnings，不能输出工程数值字段。")
    if not isinstance(value["actions"], list) or len(value["actions"]) > 64:
        raise ValueError("作用点列表无效或超过 64 项，请人工分区审核。")
    if not isinstance(value["warnings"], list) or any(not isinstance(w, str) for w in value["warnings"]):
        raise ValueError("作用点复核疑问必须为文字列表。")
    ids = set()
    for action in value["actions"]:
        if not isinstance(action, dict) or set(action) - {"id", "kind", "member", "point", "text", "confidence"}:
            raise ValueError("作用点观察包含无效或工程数值字段。")
        ident = action.get("id")
        point = action.get("point")
        confidence = action.get("confidence")
        if (not isinstance(ident, str) or not ident.strip() or ident in ids
                or action.get("kind") not in {"force", "moment"}
                or not isinstance(action.get("member"), int) or isinstance(action["member"], bool)
                or not isinstance(point, list) or len(point) != 2
                or not all(_finite(p) and 0 <= p <= 1 for p in point)
                or not isinstance(action.get("text", ""), str)
                or (confidence is not None and not (_finite(confidence) and 0 <= confidence <= 1))):
            raise ValueError("作用点编号、类型、引用或图片坐标无效，请人工核对。")
        ids.add(ident)
    supports = value.get("supports", [])
    if not isinstance(supports, list) or len(supports) > 64:
        raise ValueError("支座观察必须为不超过 64 项的列表。")
    nodes = set()
    for support in supports:
        if (not isinstance(support, dict) or not {"node", "symbol", "point", "text"} <= set(support)
                or set(support) - {"node", "symbol", "point", "text", "confidence"}
                or not isinstance(support.get("node"), int) or isinstance(support["node"], bool)
                or support["node"] in nodes
                or support.get("symbol") not in {"triangle", "triangle_on_rollers", "circle", "wall", "unknown"}
                or not isinstance(support.get("point"), list) or len(support["point"]) != 2
                or not all(_finite(p) and 0 <= p <= 1 for p in support["point"])
                or not isinstance(support.get("text"), str) or not support["text"].strip()
                or (support.get("confidence") is not None and not (
                    _finite(support["confidence"]) and 0 <= support["confidence"] <= 1))):
            raise ValueError("支座观察的节点、可见符号或附着坐标无效，不能输出数值约束。")
        nodes.add(support["node"])


def _project(model, member, point, width, height):
    nodes = {n["id"]: n for n in model["nodes"]}
    a, b = nodes[member["i"]], nodes[member["j"]]
    dx, dy = (b["u"] - a["u"]) * (width - 1), (b["v"] - a["v"]) * (height - 1)
    px, py = (point[0] - a["u"]) * (width - 1), (point[1] - a["v"]) * (height - 1)
    if dx * dx + dy * dy <= 0:
        raise ValueError("所引用杆件的图片长度为零。")
    t = (px * dx + py * dy) / (dx * dx + dy * dy)
    distance = math.hypot(px - t * dx, py - t * dy)
    return t, distance


def apply_action_review(draft, review, *, image_path=None):
    """按原杆件引用逐一分段；已有工程引用或歧义时保留观察，不迁移荷载。"""
    validate_action_review(review)
    updated = deepcopy(draft)
    source = updated.get("source") or {}
    width, height = source.get("width_px", 0), source.get("height_px", 0)
    if width <= 1 or height <= 1:
        raise ValueError("图片尺寸未知，请重新加载图片后复核作用点。")
    originals = {m["id"]: deepcopy(m) for m in updated["image_model"]["members"]}
    origins = {ident: ident for ident in originals}
    records = []
    protected = updated.get("scale", {}).get("status") == "confirmed" or any(
        e.get("verified") for e in updated.get("entities") or [])
    from sketch_moment_refinement import moment_centres
    centres = {} if protected else moment_centres(review["actions"], image_path)
    for action in review["actions"]:
        record = {"observation": deepcopy(action), "node": None, "reason": None}
        refs = []
        try:
            if protected:
                raise ValueError("已有人工审核或已确认尺度，请通过人工补节点处理。")
            if action["member"] not in originals:
                raise ValueError("观察引用的杆件不存在，请核对原图。")
            pixel_evidence = centres.get(action["id"])
            point = pixel_evidence["point"] if pixel_evidence else action["point"]
            t, distance = _project(draft["image_model"], originals[action["member"]], point, width, height)
            tolerance = point_match_threshold(width, height)
            if not 0 <= t <= 1 or distance > tolerance * 2:
                raise ValueError("观察位置偏离所引用杆件，请人工定位作用点。")
            if pixel_evidence:
                record["pixel_refinement"] = deepcopy(pixel_evidence)
            original = originals[action["member"]]
            original_nodes = {n["id"]: n for n in draft["image_model"]["nodes"]}
            a, b = original_nodes[original["i"]], original_nodes[original["j"]]
            point = [a[k] + t * (b[k] - a[k]) for k in ("u", "v")]
            nearby = [n for n in updated["image_model"]["nodes"] if math.hypot(
                (n["u"] - point[0]) * (width - 1), (n["v"] - point[1]) * (height - 1)) <= tolerance]
            hosts = [m for m in updated["image_model"]["members"] if origins.get(m["id"]) == action["member"]]
            if nearby:
                if len(nearby) != 1 or not any(nearby[0]["id"] in (m["i"], m["j"]) for m in hosts):
                    raise ValueError("作用位置附近存在歧义节点，请人工核对连接。")
                node_id = nearby[0]["id"]
            else:
                candidates = [(m, _project(updated["image_model"], m, point, width, height)[0]) for m in hosts]
                candidates = [(m, fraction) for m, fraction in candidates if 0 < fraction < 1]
                if len(candidates) != 1:
                    raise ValueError("作用位置不能唯一绑定到杆件分段，请人工处理。")
                member, fraction = candidates[0]
                old_member = next((deepcopy(e) for e in updated.get("entities") or []
                                   if (e.get("target") or {}).get("member") == member["id"]), {})
                updated = insert_member_node(updated, member["id"], fraction)
                node_id = max(n["id"] for n in updated["image_model"]["nodes"])
                new_member = max(m["id"] for m in updated["image_model"]["members"])
                origins[new_member] = action["member"]
                refs.extend([f"member:{member['id']}", f"member:{new_member}"])
                for entity in updated["entities"]:
                    target = entity.get("target") or {}
                    if (target.get("node") == node_id or target.get("member") in {member["id"], new_member}):
                        entity.update(source="derived", verified=False, confidence=None,
                                      recognition_confidence=None,
                                      position_refinement={"method": "focused-action-projection/v1",
                                          "observation": deepcopy(action),
                                          "pixel_evidence": deepcopy(pixel_evidence)})
                        if entity["kind"] == "member":
                            entity["original_observation"] = deepcopy(old_member.get("original_observation") or {
                                "image_geometry": old_member.get("image_geometry"), "confidence": old_member.get("confidence")})
            record["node"] = node_id
            refs.append(f"node:{node_id}")
        except ValueError as exc:
            record["reason"] = str(exc)
            refs = [f"member:{action['member']}"] if action["member"] in originals else []
        records.append(record)
        ident = f"action-review-{action['id']}"
        while any(i.get("id") == ident for i in updated.get("issues") or []):
            ident += "-new"
        message = "集中力" if action["kind"] == "force" else "集中力矩"
        if record["node"] is not None:
            updated.setdefault("issues", []).append({"id": ident + "-position", "category": "low_confidence",
                "severity": "blocking", "status": "open", "entity_refs": refs,
                "message": f"{message}作用点及分段为候选位置，请对照原图核对。此确认不补充荷载数值。",
                "resolution": None, "resolved_by": None})
            refs = [f"node:{record['node']}"]
        updated.setdefault("issues", []).append({"id": ident, "category": "load_incomplete",
            "severity": "blocking", "status": "open", "entity_refs": refs,
            "message": f"{message}作用位置候选待核对；请确认外荷载/反力含义、位置、方向、数值和单位。"
                       + (record["reason"] or "候选分段尚未经人工确认。"),
            "observations": [deepcopy(record)], "resolution": None, "resolved_by": None})
    if review["warnings"]:
        updated.setdefault("issues", []).append({"id": "action-review-warnings", "category": "load_incomplete",
            "severity": "blocking", "status": "open", "entity_refs": [],
            "message": "作用点复核仍有疑问，请人工检查：" + "；".join(review["warnings"]),
            "resolution": None, "resolved_by": None})
    support_records = _review_supports(updated, review.get("supports", []), protected, width, height)
    updated["action_review"] = {"status": "completed", "records": records, "warnings": deepcopy(review["warnings"]),
                                "support_records": support_records}
    updated["model"] = updated["merge_plan"] = updated["confirmation"] = None
    return updated


def _review_supports(draft, observations, protected, width, height):
    """按可见符号生成已有支座的未审核候选，未知、远点或人工数据不改。"""
    from sketch_parser import _as_support
    kinds = {"triangle": "pinned", "triangle_on_rollers": "roller", "circle": "roller", "wall": "fixed"}
    records = []
    for observation in observations:
        node_id = observation["node"]
        record = {"observation": deepcopy(observation), "applied": False, "reason": None}
        matches = [s for s in draft["image_model"].get("supports", []) if s["node"] == node_id]
        node = next((n for n in draft["image_model"]["nodes"] if n["id"] == node_id), None)
        refs = []
        if len(matches) == 1:
            support = matches[0]
            refs = [f"support:{node_id}:{support.get('name')}"]
            record["original_support"] = deepcopy(support)
        if protected:
            record["reason"] = "已有人工审核或已确认尺度，支座保持原设置；请人工核对观察。"
        elif len(matches) != 1 or node is None:
            record["reason"] = "观察无法唯一对应已有支座，请人工选择节点并设置支座。"
        elif math.hypot((node["u"] - observation["point"][0]) * (width - 1),
                        (node["v"] - observation["point"][1]) * (height - 1)) > point_match_threshold(width, height) * 2:
            record["reason"] = "观察附着位置偏离节点，请人工核对支座位置。"
        elif observation["symbol"] not in kinds:
            record["reason"] = "可见符号不能确定支座类型，原约束暂保留；请人工设置并核对。"
        else:
            candidate, _ = _as_support({"node": node_id, "kind": kinds[observation["symbol"]]})
            support["fix"] = candidate["fix"]
            record["applied"] = True
            for entity in draft.get("entities", []):
                if entity.get("kind") == "support" and (entity.get("target", {}).get("support") or {}).get("node") == node_id:
                    entity.update(source="derived", verified=False, confidence=None, recognition_confidence=None,
                                  symbol_review=deepcopy(record), payload=deepcopy(support))
        records.append(record)
        ident = f"support-symbol-review-{node_id}"
        while any(i.get("id") == ident for i in draft.get("issues", [])):
            ident += "-new"
        draft.setdefault("issues", []).append({"id": ident,
            "category": "low_confidence" if record["applied"] else "support_unknown",
            "severity": "blocking", "status": "open", "entity_refs": refs,
            "support_review_node": node_id, "observations": [deepcopy(record)],
            "message": record["reason"] or "支座类型按可见符号提出候选；请核对实际约束，不能按静定性推断。",
            "resolution": None, "resolved_by": None})
    return records
