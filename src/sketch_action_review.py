"""聚焦集中作用点的视觉复核；只提出拓扑候选，不生成工程数值。"""

from copy import deepcopy
import math

from multimodal_contract import point_match_threshold
from sketch_topology import insert_member_node


ACTION_REVIEW_PROMPT = """你只复核图片中外部集中力和集中力矩的作用位置，不输出完整结构模型。
上下文给出的节点/杆件是上一阶段的待审核几何，可能遗漏中间作用节点。
请逐一查看所有直箭头和圆弧箭头：直箭头取与杆件接触的箭头尖或作用位置，
圆弧力矩取圆弧中心在杆件上的作用点，不能取圆弧箭头尖。归一化坐标以图片左上角为原点，
u 向右，v 向下，范围 0..1。不能取文字中心、图像中心或按杆长均匀分配位置。
排除明确的支座反力、尺寸箭头、坐标轴和分布荷载箭头列；不能仅凭颜色判定外力或反力。
无法区分时将疑问写入 warnings，不编造点。即使力/矩没有数值，也要保留可见的作用点。
member 使用上下文已存在的杆件编号。不要生成荷载大小、单位换算、真实长度、应力或位移。
仅输出 JSON：{"actions":[{"id":"A1","kind":"force","member":1,
"point":[0.4,0.5],"text":"图上集中力符号","confidence":null}],"warnings":[]}。
kind 仅 force/moment。id 唯一。text 只描述观察，不填数值荷载。warnings 为字符串列表。
没有集中作用点时 actions 为空；存在无法定位的符号时必须在 warnings 中说明。
"""


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_action_review(value):
    if not isinstance(value, dict) or set(value) != {"actions", "warnings"}:
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


def apply_action_review(draft, review):
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
    for action in review["actions"]:
        record = {"observation": deepcopy(action), "node": None, "reason": None}
        refs = []
        try:
            if protected:
                raise ValueError("已有人工审核或已确认尺度，请通过人工补节点处理。")
            if action["member"] not in originals:
                raise ValueError("观察引用的杆件不存在，请核对原图。")
            t, distance = _project(draft["image_model"], originals[action["member"]], action["point"], width, height)
            tolerance = point_match_threshold(width, height)
            if not 0 <= t <= 1 or distance > tolerance * 2:
                raise ValueError("观察位置偏离所引用杆件，请人工定位作用点。")
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
                                          "observation": deepcopy(action)})
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
    updated["action_review"] = {"status": "completed", "records": records, "warnings": deepcopy(review["warnings"])}
    updated["model"] = updated["merge_plan"] = updated["confirmation"] = None
    return updated
