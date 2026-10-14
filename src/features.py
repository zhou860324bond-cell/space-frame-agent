"""建模对象层（Domain IR v2 的对象部分）：Abaqus 式「对象施加在区域上」。

Abaqus/CAE 里截面指派、边界条件、荷载都是**有名字的对象**，各自指向一个
区域（集合或现场拾取的一组实体）。改集合，引用它的对象跟着变；对象可以
编辑、复制、改名、抑制、删除。原来的 Domain IR 只有逐节点、逐杆件的条目：
同名条目是对象唯一的痕迹，集合在写入时就被展开成编号，之后再改集合，
荷载和约束不会跟着变；对一串杆件施加荷载时每根还会被改名成「名字-杆号」。

这一层的做法与 Abaqus 的「特征 + 再生成」相同：

* ``model["features"]`` 是这些对象的**唯一定义**；
* 逐节点、逐杆件条目由 :func:`regenerate` 从对象**派生**，带上来源标记
  （支座 ``owners``、荷载 ``owner``、杆件 ``assignment`` 与 ``orientation``）；
* 没有来源标记的条目是「直接定义」（旧模型、旧工具写入的），原样保留。

求解内核、结果库和所有读逐条目的代码都不需要改：它们看到的仍是 v1 条目。
冲突（同一自由度被两个对象约束、同一杆件被两次指派截面）不在再生成时
静默取舍，而是由 :func:`check` 报出，校验时拒绝求解。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

KINDS = ("section_assignments", "beam_orientations", "bcs", "loads", "constraints")
LOAD_TYPES = ("concentrated_force", "moment", "line_load", "gravity")
#: Interaction 模块的约束类型（Abaqus 名）。beam / tie / pin 是 MPC，coupling 是运动耦合。
#: 值是 (被约束的从节点自由度, 是否带刚臂)；coupling 的自由度由对象的 dofs 给出。
CONSTRAINT_TYPES = {
    "beam": ((0, 1, 2, 3, 4, 5), True),       # MPC BEAM：刚性连杆
    "tie": ((0, 1, 2, 3, 4, 5), False),       # MPC TIE：重合节点全部自由度相等
    "pin": ((0, 1, 2), False),                # MPC PIN：重合节点只连平动（铰）
    "coupling": (None, True),                 # Kinematic Coupling：所选自由度随参考点刚体运动
}
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
        "beam_orientations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "region", "n1"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "region": _REGION,
                # 截面局部 y 轴（Abaqus 的 n1 方向）的近似方向，全局分量；只看方向不看长度
                "n1": _VEC3,
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
                # Abaqus 管理器里在后面某一步 Modify：步名 → 那一步起的约束与弹簧
                "modifications": {"type": "object", "additionalProperties": {
                    "type": "object", "additionalProperties": False, "required": ["fix"],
                    "properties": {
                        "fix": {"type": "array", "minItems": 6, "maxItems": 6,
                                "items": {"type": "integer", "enum": [0, 1]}},
                        "spring": {"type": "array", "minItems": 6, "maxItems": 6,
                                   "items": {"type": "number", "minimum": 0}}}}},
                # Abaqus 管理器里的 Deactivate：从这一步起不再生效
                "deactivated": {"type": "string", "minLength": 1},
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
        "constraints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "type", "control", "region"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": list(CONSTRAINT_TYPES)},
                # 主节点：MPC 的 control point、耦合的 reference point
                "control": {"type": "integer"},
                # 从节点区域
                "region": _REGION,
                # 只有 coupling 用：六个 0/1，顺序 ux uy uz rx ry rz
                "dofs": {"type": "array", "minItems": 6, "maxItems": 6,
                         "items": {"type": "integer", "enum": [0, 1]}},
                "suppressed": {"type": "boolean"},
            }}},
    },
}

_LABELS = {"section_assignments": "截面指派", "beam_orientations": "梁方向",
           "bcs": "边界条件", "loads": "荷载", "constraints": "约束"}
_LOAD_FIELD = {"concentrated_force": "force", "moment": "moment", "line_load": "w",
               "gravity": "g"}


def _features(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return payload.get("features") or {}


def region_kind(kind: str, feature: dict[str, Any]) -> str:
    """对象需要哪一类实体：截面指派要杆件，边界条件要节点，荷载看类型。"""
    if kind in ("section_assignments", "beam_orientations"):
        return "members"
    if kind in ("bcs", "constraints"):
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
    _regenerate_orientations(payload, problems)
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


def _regenerate_orientations(payload, problems):
    """梁方向（Abaqus 的 Assign Beam Orientation）：把 n1 写成杆件的参考向量。

    参考向量与杆轴平行时局部轴无法确定，求解会失败；这里逐杆检查并报出杆号，
    不把错误留到求解时才以「参考向量与杆轴平行」的形式出现。
    """
    import numpy as np
    members = {int(m["id"]): m for m in payload.get("members") or []}
    nodes = {int(n["id"]): np.array([n["x"], n["y"], n["z"]], dtype=float)
             for n in payload.get("nodes") or []}
    for member in members.values():
        if member.pop("orientation", None) is not None:
            member.pop("ref_vector", None)
    claimed: dict[int, str] = {}
    for feature in _active("beam_orientations", payload):
        ids, problem = region_ids(payload, feature["region"], "members")
        if problem:
            problems.append(f"梁方向「{feature['name']}」：{problem}")
            continue
        n1 = np.asarray(feature["n1"], dtype=float)
        for mid in ids:
            if mid in claimed:
                problems.append(f"杆件 {mid} 被梁方向「{claimed[mid]}」与「{feature['name']}」"
                                "重复指定；请从其中一个的区域里去掉它。")
                continue
            member = members[mid]
            if int(member["i"]) in nodes and int(member["j"]) in nodes:
                axis = nodes[int(member["j"])] - nodes[int(member["i"])]
                if np.linalg.norm(np.cross(axis, n1)) <= 1e-9 * max(
                        np.linalg.norm(axis) * np.linalg.norm(n1), 1e-300):
                    problems.append(f"梁方向「{feature['name']}」的 n1 与杆件 {mid} 的轴线平行，"
                                    "无法确定截面朝向；请换一个垂直于杆轴的方向。")
                    continue
            claimed[mid] = feature["name"]
            member["ref_vector"] = [float(v) for v in n1]
            member["orientation"] = feature["name"]


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


def _step_order(payload: dict[str, Any]) -> list[tuple[str, bool]]:
    """[(步名, 是否线性摄动步)]，Initial 在最前。"""
    from steps import PERTURBATION
    return [(INITIAL, False)] + [(str(s["name"]), s.get("analysis", "linear") in PERTURBATION)
                                 for s in payload.get("steps") or []]


def _bc_values(source: dict[str, Any]) -> dict[str, Any] | None:
    """{fix, spring}；一个方向都不约束、也没有弹簧时返回 None（等于不起作用）。"""
    fix = [int(v) for v in source["fix"]]
    spring = [float(v) for v in source.get("spring") or []]
    if not any(fix) and not any(spring):
        return None
    return {"fix": fix, **({"spring": spring} if any(spring) else {})}


def bc_problems(payload: dict[str, Any], feature: dict[str, Any]) -> list[str]:
    """边界条件的分析步引用：创建、修改、失活的步要存在，先后顺序要对。"""
    order = [name for name, _ in _step_order(payload)]
    label = f"边界条件「{feature['name']}」"
    created = feature.get("step") or INITIAL
    if created not in order:
        return [f"{label}所在的分析步「{created}」不存在"]
    out = []
    first = order.index(created)
    off = feature.get("deactivated")
    if off is not None:
        if off not in order:
            out.append(f"{label}在不存在的分析步「{off}」失活")
        elif order.index(off) <= first:
            out.append(f"{label}在「{off}」失活，但它在「{created}」才创建；失活要在创建之后的步")
    for step in feature.get("modifications") or {}:
        if step not in order:
            out.append(f"{label}在不存在的分析步「{step}」被修改")
        elif order.index(step) <= first:
            out.append(f"{label}在「{step}」的修改早于或等于创建步「{created}」；创建步里直接编辑它")
        elif off in order and order.index(step) >= order.index(off):
            out.append(f"{label}在「{step}」的修改不早于失活步「{off}」，不会生效")
    return out


def bc_states(payload: dict[str, Any], feature: dict[str, Any]
              ) -> list[tuple[str, dict[str, Any] | None, str]]:
    """边界条件在每一步的状态：(步名, 生效的 {fix, spring} 或 None, 状态)。

    对应 Abaqus 边界条件管理器的分析步矩阵：创建、传播、修改、失活。通用步里的
    修改与失活一直延续到后面；线性摄动步里的只在本步（摄动步不改变基态）。
    在摄动步里创建的边界条件也只在那一步。
    """
    created = feature.get("step") or INITIAL
    mods = feature.get("modifications") or {}
    off = feature.get("deactivated")
    out: list[tuple[str, dict[str, Any] | None, str]] = []
    started, ended, current, tail = False, False, None, ""
    for name, perturb in _step_order(payload):
        if ended or (not started and name != created):
            out.append((name, None, tail if ended else ""))
            continue
        if name == created:
            started = True
            values = _bc_values(feature)
            if perturb:
                ended = True                      # 摄动步里创建的只在本步
                out.append((name, values, "创建（本步）"))
                continue
            current = values
            out.append((name, values, "创建"))
            continue
        if name == off:
            out.append((name, None, "失活（本步）" if perturb else "失活"))
            if not perturb:
                current, ended, tail = None, True, "失活"
            continue
        if name in mods:
            values = _bc_values(mods[name])
            out.append((name, values, "修改（本步）" if perturb else "修改"))
            if not perturb:
                current = values
            continue
        out.append((name, current, "传播" if current is not None else "失活"))
    return out


def _regenerate_bcs(payload, problems):
    supports = [s for s in payload.get("supports") or [] if not s.get("owners")]
    direct = {int(s["node"]) for s in supports}
    steps = {s["name"]: s for s in payload.get("steps") or []}
    for step in steps.values():                 # 先撤掉旧的派生条目
        table = step.get("supports")
        if table:
            kept = {k: v for k, v in table.items() if not v.get("owners")}
            if kept:
                step["supports"] = kept
            else:
                step.pop("supports", None)
        owned = step.pop("deactivation_owners", None) or {}
        if owned:
            kept_off = [n for n in step.get("deactivate_supports") or [] if str(n) not in owned]
            if kept_off:
                step["deactivate_supports"] = kept_off
            else:
                step.pop("deactivate_supports", None)

    order = _step_order(payload)
    tables: list[dict[int, dict[str, Any]]] = [{} for _ in order]
    seen: set[str] = set()

    def report(text: str) -> None:
        if text not in seen:
            seen.add(text)
            problems.append(text)

    for feature in _active("bcs", payload):
        bad = bc_problems(payload, feature)
        if bad:
            for text in bad:
                report(text)
            continue
        ids, problem = region_ids(payload, feature["region"], "nodes")
        if problem:
            report(f"边界条件「{feature['name']}」：{problem}")
            continue
        clash = sorted(set(ids) & direct)
        if clash:
            report(f"节点 {clash} 既有直接定义的支座，又在边界条件「{feature['name']}」"
                   "的区域里；请删除直接定义，或把它转成边界条件对象。")
            ids = [n for n in ids if n not in direct]
        for k, (step_name, values, _status) in enumerate(bc_states(payload, feature)):
            if values is None:
                continue
            table = tables[k]
            for nid in ids:
                if nid in table:
                    conflict = _merge_support(table[nid], {"name": feature["name"], **values})
                    if conflict:
                        where = "" if step_name == INITIAL else f"分析步「{step_name}」里，"
                        report(where + conflict + "；请调整其中一个的区域或自由度。")
                    continue
                table[nid] = {"name": feature["name"], "fix": list(values["fix"]),
                              "owners": [feature["name"]],
                              **({"spring": list(values["spring"])} if values.get("spring") else {})}

    for nid, entry in sorted(tables[0].items()):
        supports.append({"node": nid, **entry})
    payload["supports"] = supports

    # 分析步只写相对基态的变化：改写的节点进 supports，不再约束的节点进
    # deactivate_supports 并记下来源，下次再生成时据此撤掉
    base = tables[0]
    for k, (step_name, perturb) in enumerate(order[1:], start=1):
        step, now = steps[step_name], tables[k]
        target = step.get("supports") or {}
        for nid, entry in sorted(now.items()):
            if base.get(nid) == entry:
                continue
            if str(nid) in target:
                report(f"节点 {nid} 在分析步「{step_name}」里既有直接改写的支座，"
                       f"又在边界条件「{entry['owners'][0]}」的区域里")
                continue
            target[str(nid)] = entry
        if target:
            step["supports"] = target
        off = list(step.get("deactivate_supports") or [])
        owners: dict[str, list[str]] = {}
        for nid in sorted(set(base) - set(now)):
            if str(nid) in target or nid in off:
                continue
            off.append(nid)
            owners[str(nid)] = list(base[nid]["owners"])
        if owners:
            step["deactivate_supports"] = off
            step["deactivation_owners"] = owners
        if not perturb:
            base = now


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
    known = {int(n["id"]) for n in payload.get("nodes") or []}
    for feature in features.get("constraints") or []:
        if feature.get("suppressed"):
            continue
        label = f"约束「{feature['name']}」"
        if int(feature["control"]) not in known:
            errors.append(f"{label}的主节点 {feature['control']} 不存在")
        slaves, problem = region_ids(payload, feature["region"], "nodes")
        if problem:
            errors.append(f"{label}：{problem}")
        elif int(feature["control"]) in slaves:
            errors.append(f"{label}的主节点 {feature['control']} 也在从节点区域里；从区域里去掉它")
        if feature["type"] == "coupling" and not any(feature.get("dofs") or [1]):
            errors.append(f"{label}是运动耦合，至少要选一个自由度")
    errors += regenerate(deepcopy(payload))
    return errors


def resolved_constraints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """生效的约束对象 → 内核能用的形式：主节点、从节点、自由度、是否带刚臂。

    约束不派生逐条目，求解装配（model_io.from_dict）和 Abaqus 导出直接读这里，
    两边同源。区域解析不了的跳过，由 :func:`check` 报出。
    """
    out = []
    for feature in _active("constraints", payload):
        slaves, problem = region_ids(payload, feature["region"], "nodes")
        if problem:
            continue
        dofs, rigid = CONSTRAINT_TYPES[feature["type"]]
        if dofs is None:
            dofs = tuple(k for k, on in enumerate(feature.get("dofs") or [1] * 6) if on)
        out.append({"name": feature["name"], "type": feature["type"],
                    "master": int(feature["control"]), "slaves": tuple(slaves),
                    "dofs": tuple(dofs), "rigid": rigid})
    return out


def constraint_nodes(payload: dict[str, Any]) -> set[int]:
    """约束用到的节点（主节点与从节点）。它们可以不连杆件，清理孤立节点时要留下。"""
    nodes: set[int] = set()
    for feature in _active("constraints", payload):
        nodes.add(int(feature["control"]))
        nodes.update(region_ids(payload, feature["region"], "nodes")[0])
    return nodes


def owner_of(entry: dict[str, Any]) -> str | None:
    """条目来自哪个对象；直接定义返回 None。界面据此把条目归到对象下。"""
    if entry.get("owners"):
        return entry["owners"][0]
    return entry.get("owner") or entry.get("assignment") or entry.get("orientation")
