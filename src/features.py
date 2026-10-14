"""建模对象层（Domain IR v2 的对象部分）：Abaqus 式「对象施加在区域上」。

Abaqus/CAE 里截面指派、边界条件、荷载都是**有名字的对象**，各自指向一个
区域（集合或现场拾取的一组实体）。改集合，引用它的对象跟着变；对象可以
编辑、复制、改名、抑制、删除。原来的 Domain IR 只有逐节点、逐杆件的条目：
同名条目是对象唯一的痕迹，集合在写入时就被展开成编号，之后再改集合，
荷载和约束不会跟着变；对一串杆件施加荷载时每根还会被改名成「名字-杆号」。

这一层的做法与 Abaqus 的「特征 + 再生成」相同：

* ``model["features"]`` 是这些对象的**唯一定义**；
* 逐节点、逐杆件条目由 :func:`regenerate` 从对象**派生**，带上来源标记
  （支座 ``owners``、荷载 ``owner``、杆件 ``assignment``）；
* 没有来源标记的条目是「直接定义」（旧模型、旧工具写入的），原样保留。

求解内核、结果库和所有读逐条目的代码都不需要改：它们看到的仍是 v1 条目。
冲突（同一自由度被两个对象约束、同一杆件被两次指派截面）不在再生成时
静默取舍，而是由 :func:`check` 报出，校验时拒绝求解。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

KINDS = ("section_assignments", "bcs", "loads")
LOAD_TYPES = ("concentrated_force", "moment", "line_load", "gravity")
INITIAL = "Initial"

_REGION = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "set": {"type": "string", "minLength": 1},
        "nodes": {"type": "array", "items": {"type": "integer"}},
        "members": {"type": "array", "items": {"type": "integer"}},
    },
}
_VEC3 = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}

#: 并入 model_io.MODEL_SCHEMA 的 ``features`` 字段。
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "section_assignments": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "region", "section", "material"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "region": _REGION,
                "section": {"type": "string", "minLength": 1},
                "material": {"type": "string", "minLength": 1},
                "suppressed": {"type": "boolean"},
            }}},
        "bcs": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "region", "fix"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "region": _REGION,
                "step": {"type": "string", "minLength": 1},
                "fix": {"type": "array", "minItems": 6, "maxItems": 6,
                        "items": {"type": "integer", "enum": [0, 1]}},
                "spring": {"type": "array", "minItems": 6, "maxItems": 6,
                           "items": {"type": "number", "minimum": 0}},
                "suppressed": {"type": "boolean"},
            }}},
        "loads": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "region", "case", "type"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "region": _REGION,
                "case": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": list(LOAD_TYPES)},
                "force": _VEC3,          # concentrated_force：力
                "moment": _VEC3,         # moment：力×长度
                "w": _VEC3,              # line_load：力/长度，全局分量
                "g": _VEC3,              # gravity：加速度，全局分量
                "suppressed": {"type": "boolean"},
            }}},
    },
}

_LABELS = {"section_assignments": "截面指派", "bcs": "边界条件", "loads": "荷载"}
_LOAD_FIELD = {"concentrated_force": "force", "moment": "moment", "line_load": "w",
               "gravity": "g"}


def _features(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return payload.get("features") or {}


def region_kind(kind: str, feature: dict[str, Any]) -> str:
    """对象需要哪一类实体：截面指派要杆件，边界条件要节点，荷载看类型。"""
    if kind == "section_assignments":
        return "members"
    if kind == "bcs":
        return "nodes"
    return "nodes" if feature.get("type") in ("concentrated_force", "moment") else "members"


def region_ids(payload: dict[str, Any], region: dict[str, Any], what: str
               ) -> tuple[list[int], str | None]:
    """区域 → 编号列表（``what`` 为 nodes 或 members）。返回 (编号, 问题或 None)。

    集合是**活引用**：每次再生成都按集合的当前内容展开。集合里只有另一类
    实体（给边界条件选了杆件集合）算错，而不是静默得到空区域。
    """
    noun = "节点" if what == "nodes" else "杆件"
    if "set" in region:
        entry = (payload.get("sets") or {}).get(region["set"])
        if entry is None:
            return [], f"集合「{region['set']}」不存在"
        ids = [int(k) for k in entry.get(what) or []]
        if not ids:
            return [], f"集合「{region['set']}」里没有{noun}"
    else:
        ids = [int(k) for k in region.get(what) or []]
        if not ids:
            return [], f"区域里没有{noun}"
    known = {int(e["id"]) for e in payload.get(what) or []}
    missing = sorted(set(ids) - known)
    if missing:
        return [], f"区域引用了不存在的{noun} {missing}"
    return ids, None


def _active(kind: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in _features(payload).get(kind) or [] if not f.get("suppressed")]


def regenerate(payload: dict[str, Any]) -> list[str]:
    """按对象重写派生条目（就地修改 payload），返回无法派生的问题清单。

    幂等：连续调用两次结果相同。直接定义的条目不动。
    """
    problems: list[str] = []
    _regenerate_sections(payload, problems)
    _regenerate_bcs(payload, problems)
    _regenerate_loads(payload, problems)
    return problems


def _regenerate_sections(payload, problems):
    members = {int(m["id"]): m for m in payload.get("members") or []}
    for member in members.values():            # 先撤掉旧的派生指派
        if member.pop("assignment", None) is not None:
            member["section"] = ""
            member["material"] = ""
    claimed: dict[int, str] = {}
    for feature in _active("section_assignments", payload):
        ids, problem = region_ids(payload, feature["region"], "members")
        if problem:
            problems.append(f"截面指派「{feature['name']}」：{problem}")
            continue
        for mid in ids:
            if mid in claimed:
                problems.append(f"杆件 {mid} 被截面指派「{claimed[mid]}」与"
                                f"「{feature['name']}」重复指派；请从其中一个的区域里去掉它。")
                continue
            claimed[mid] = feature["name"]
            member = members[mid]
            member["section"], member["material"] = feature["section"], feature["material"]
            member["assignment"] = feature["name"]


def _merge_support(entry: dict[str, Any], feature: dict[str, Any]) -> str | None:
    """把一个边界条件并进节点上已有的派生条目；自由度相交则返回冲突说明。

    Abaqus 允许同一节点的不同自由度分属不同边界条件（如 U1 归一个、U2 归
    另一个），同一自由度被两个约束则是过约束。
    """
    fix = list(feature["fix"])
    spring = list(feature.get("spring") or [0.0] * 6)
    old_fix = list(entry["fix"])
    old_spring = list(entry.get("spring") or [0.0] * 6)
    taken = [k for k in range(6)
             if (fix[k] or spring[k]) and (old_fix[k] or old_spring[k])]
    if taken:
        return (f"边界条件「{feature['name']}」与「{entry['owners'][0]}」在同一节点的"
                f"方向 {taken} 上重复约束")
    entry["fix"] = [a | b for a, b in zip(old_fix, fix, strict=True)]
    merged = [a + b for a, b in zip(old_spring, spring, strict=True)]
    if any(merged):
        entry["spring"] = merged
    entry["owners"].append(feature["name"])
    return None


def _regenerate_bcs(payload, problems):
    supports = [s for s in payload.get("supports") or [] if not s.get("owners")]
    direct = {int(s["node"]) for s in supports}
    steps = {s["name"]: s for s in payload.get("steps") or []}
    for step in steps.values():
        table = step.get("supports")
        if table:
            kept = {k: v for k, v in table.items() if not v.get("owners")}
            if kept:
                step["supports"] = kept
            else:
                step.pop("supports", None)
    derived: dict[str, dict[int, dict[str, Any]]] = {}
    for feature in _active("bcs", payload):
        step = feature.get("step") or INITIAL
        if step != INITIAL and step not in steps:
            problems.append(f"边界条件「{feature['name']}」所在的分析步「{step}」不存在")
            continue
        ids, problem = region_ids(payload, feature["region"], "nodes")
        if problem:
            problems.append(f"边界条件「{feature['name']}」：{problem}")
            continue
        table = derived.setdefault(step, {})
        for nid in ids:
            if step == INITIAL and nid in direct:
                problems.append(f"节点 {nid} 既有直接定义的支座，又在边界条件「{feature['name']}」"
                                "的区域里；请删除直接定义，或把它转成边界条件对象。")
                continue
            if nid in table:
                conflict = _merge_support(table[nid], feature)
                if conflict:
                    problems.append(conflict + "；请调整其中一个的区域或自由度。")
                continue
            entry = {"name": feature["name"], "fix": list(feature["fix"]),
                     "owners": [feature["name"]]}
            if any(feature.get("spring") or []):
                entry["spring"] = list(feature["spring"])
            table[nid] = entry
    for nid, entry in sorted((derived.get(INITIAL) or {}).items()):
        supports.append({"node": nid, **entry})
    payload["supports"] = supports
    for step_name, table in derived.items():
        if step_name == INITIAL:
            continue
        target = steps[step_name].setdefault("supports", {})
        for nid, entry in sorted(table.items()):
            if str(nid) in target:
                problems.append(f"节点 {nid} 在分析步「{step_name}」里既有直接改写的支座，"
                                f"又在边界条件「{entry['owners'][0]}」的区域里")
                continue
            target[str(nid)] = entry


def _case_blocks(payload) -> dict[str, dict[str, Any]]:
    blocks = {"default": payload}
    for case in payload.get("load_cases") or []:
        blocks[str(case["name"])] = case
    return blocks


def _regenerate_loads(payload, problems):
    blocks = _case_blocks(payload)
    for block in blocks.values():
        for key in ("nodal_loads", "member_loads", "member_spans"):
            if block.get(key):
                block[key] = [e for e in block[key] if not e.get("owner")]
    members = {int(m["id"]): m for m in payload.get("members") or []}
    sections = {s["name"]: s for s in payload.get("sections") or []}
    density = {m["name"]: float(m.get("density") or 0.0)
               for m in payload.get("materials") or []}
    for feature in _active("loads", payload):
        block = blocks.get(feature["case"])
        if block is None:
            problems.append(f"荷载「{feature['name']}」所在的工况「{feature['case']}」不存在")
            continue
        ids, problem = region_ids(payload, feature["region"], region_kind("loads", feature))
        if problem:
            problems.append(f"荷载「{feature['name']}」：{problem}")
            continue
        load_type = feature["type"]
        vec = [float(v) for v in feature.get(_LOAD_FIELD[load_type]) or (0.0, 0.0, 0.0)]
        if load_type in ("concentrated_force", "moment"):
            six = vec + [0.0] * 3 if load_type == "concentrated_force" else [0.0] * 3 + vec
            block.setdefault("nodal_loads", []).extend(
                {"node": nid, "load": list(six), "owner": feature["name"]} for nid in ids)
        elif load_type == "line_load":
            block.setdefault("member_loads", []).extend(
                {"member": mid, "w": list(vec), "owner": feature["name"]} for mid in ids)
        else:                                          # gravity：w = ρ·A·g
            for mid in ids:
                member = members[mid]
                section = sections.get(member.get("section") or "")
                rho = density.get(member.get("material") or "", 0.0)
                if section is None or rho <= 0.0:
                    problems.append(f"重力荷载「{feature['name']}」：杆件 {mid} 缺截面或材料密度，"
                                    "无法计算自重")
                    continue
                block.setdefault("member_loads", []).append(
                    {"member": mid, "w": [rho * float(section["A"]) * gk for gk in vec],
                     "owner": feature["name"]})


def check(payload: dict[str, Any]) -> list[str]:
    """对象层的语义检查：名字唯一、引用存在、派生无冲突。不修改 payload。"""
    errors: list[str] = []
    features = _features(payload)
    for kind, label in _LABELS.items():
        names = [f["name"] for f in features.get(kind) or []]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            errors.append(f"{label}名称重复：{dup}")
    sections = {s["name"] for s in payload.get("sections") or []}
    materials = {m["name"] for m in payload.get("materials") or []}
    for feature in features.get("section_assignments") or []:
        if feature["section"] not in sections:
            errors.append(f"截面指派「{feature['name']}」引用了不存在的截面「{feature['section']}」")
        if feature["material"] not in materials:
            errors.append(f"截面指派「{feature['name']}」引用了不存在的材料「{feature['material']}」")
    for feature in features.get("bcs") or []:
        clash = [k for k, (f, s) in enumerate(zip(feature["fix"], feature.get("spring") or [0] * 6,
                                                  strict=True)) if f and s]
        if clash:
            errors.append(f"边界条件「{feature['name']}」的方向 {clash} 既固定又给了弹簧，只能取一个")
    direct_names = {case: {e.get("name") for key in ("nodal_loads", "member_loads", "member_spans")
                           for e in block.get(key) or [] if e.get("name")}
                    for case, block in _case_blocks(payload).items()}
    for feature in features.get("loads") or []:
        if feature["name"] in direct_names.get(feature["case"], set()):
            errors.append(f"荷载「{feature['name']}」与工况「{feature['case']}」里直接定义的"
                          "同名荷载冲突；请改名")
        field = _LOAD_FIELD[feature["type"]]
        if field not in feature:
            errors.append(f"荷载「{feature['name']}」是 {feature['type']}，缺少 {field}")
    errors += regenerate(deepcopy(payload))
    return errors


def owner_of(entry: dict[str, Any]) -> str | None:
    """条目来自哪个对象；直接定义返回 None。界面据此把条目归到对象下。"""
    if entry.get("owners"):
        return entry["owners"][0]
    return entry.get("owner") or entry.get("assignment")
