"""Agent 与求解器之间的唯一契约：一份 JSON 结构模型。

大模型只需产出符合 MODEL_SCHEMA 的 JSON，其余全部由确定性代码接管。
校验失败时把 errors 原样回喂给模型重试——这就是自修复闭环的一轮。
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np

from frame3d import (DEFAULT_CASE, LOCAL_DOF_NAMES, Frame, LoadCase, Material,
                     Member, Node, Section, check_model, member_endpoints)
from span_loads import KINDS as SPAN_KINDS
from span_loads import SpanLoad

CURRENT_SCHEMA_VERSION = 1

_LOAD6 = {"type": "array", "minItems": 6, "maxItems": 6, "items": {"type": "number"}}
_VEC3 = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
_RELEASE_LIST = {
    "type": "array",
    "maxItems": 6,
    "uniqueItems": True,
    "items": {"type": "string", "enum": list(LOCAL_DOF_NAMES)},
}
_NODAL_LOADS = {
    "type": "array",
    "items": {
        "type": "object", "required": ["node", "load"], "additionalProperties": False,
        "properties": {"name": {"type": "string", "minLength": 1},
                       "node": {"type": "integer"}, "load": _LOAD6},
    },
}
_MEMBER_SPANS = {
    "type": "array",
    "items": {
        "type": "object", "required": ["member", "kind", "w1"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "member": {"type": "integer"},
            "kind": {"type": "string", "enum": list(SPAN_KINDS)},
            "w1": _VEC3,
            "w2": _VEC3,
            "a": {"type": "number", "minimum": 0},
            # 自由备注，求解器不读。自重生成的荷载靠它标记，
            # 才能在重复调用时替换掉上一次生成的而不是叠加上去
            "note": {"type": "string"},
        },
    },
}
_SETTLEMENTS = {
    "type": "array",
    "items": {
        "type": "object", "required": ["node", "d"], "additionalProperties": False,
        "properties": {"name": {"type": "string", "minLength": 1},
                       "node": {"type": "integer"}, "d": _LOAD6},
    },
}
# 初应变：装配误差与温度。讲义 §3-9 五、六把温度应力"转化为装配内力问题"，
# 两者本来就是同一个机制，所以放同一张表里，可以同时给。
# lack_of_fit 是制造误差 Δl = 实际长度 − 设计长度，**正值表示做长了**——
# 与升温同向（都想变长），这样两种输入的符号含义一致，不用记两套规则。
_MEMBER_STRAINS = {
    "type": "array",
    "items": {
        "type": "object", "required": ["member"], "additionalProperties": False,
        "properties": {"name": {"type": "string", "minLength": 1},
                       "member": {"type": "integer"},
                       "lack_of_fit": {"type": "number"},
                       "delta_t": {"type": "number"}},
    },
}
_MEMBER_LOADS = {
    "type": "array",
    "items": {
        "type": "object", "required": ["member", "w"], "additionalProperties": False,
        "properties": {"name": {"type": "string", "minLength": 1},
                       "member": {"type": "integer"}, "w": _VEC3,
                       "frame": {"type": "string", "enum": ["global"]}},
    },
}

MODEL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SpaceFrameModel",
    "type": "object",
    "required": ["units", "materials", "sections", "nodes", "members", "supports"],
    "additionalProperties": False,
    "properties": {
        # 旧项目没有版本字段，仍按 v0 接受并在进入 Session 时迁移到 v1。
        # 明确写出版本后则必须是当前实现支持的版本，禁止把未来格式静默误读。
        "schema_version": {"type": "integer", "enum": [CURRENT_SCHEMA_VERSION]},
        "units": {"type": "string", "enum": ["N-m-Pa", "N-mm-MPa"]},

        # 命名集合。**纯粹是给人用的名字，求解器根本看不到它**——
        # 工具层在调用时就把名字展开成编号了。
        #
        # 存在的理由：编号规则由生成器决定，用户并不知道 27 号在哪。
        # 没有名字，"给顶层所有梁加 5 kN/m" 这句话就没有落点，
        # 无论是人手点还是 Agent 说。有了名字，这组杆件在整个会话里
        # 都能被稳定引用，即使中途增删了别的杆件。
        "sets": {
            "type": "object",
            "additionalProperties": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "members": {"type": "array", "items": {"type": "integer"}},
                    "nodes": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string"},
                },
            },
        },
        "materials": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "required": ["name", "E", "nu"], "additionalProperties": False,
                "properties": {"name": {"type": "string", "minLength": 1},
                               "E": {"type": "number", "exclusiveMinimum": 0},
                               "nu": {"type": "number", "minimum": 0, "maximum": 0.5},
                               # 只有算自重时才用得上，缺省为 0 即不计自重
                               "density": {"type": "number", "minimum": 0},
                               "yield_stress": {"type": "number", "exclusiveMinimum": 0},
                               "hardening_ratio": {"type": "number", "minimum": 0,
                                                   "exclusiveMaximum": 1},
                               # 线膨胀系数 1/℃，只有算温度应力时才用得上
                               "alpha": {"type": "number", "minimum": 0}},
            },
        },
        "sections": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "required": ["name", "A", "Iy", "Iz", "J"],
                "additionalProperties": False,
                "properties": {"name": {"type": "string", "minLength": 1},
                               "A": {"type": "number", "exclusiveMinimum": 0},
                               "Iy": {"type": "number", "exclusiveMinimum": 0},
                               "Iz": {"type": "number", "exclusiveMinimum": 0},
                               "J": {"type": "number", "exclusiveMinimum": 0},
                               "Ay": {"type": "number", "exclusiveMinimum": 0},
                               "Az": {"type": "number", "exclusiveMinimum": 0},
                               # 极端纤维距离；缺省时正应力不可算，明确拒绝。
                               "cy": {"type": "number", "exclusiveMinimum": 0},
                               "cz": {"type": "number", "exclusiveMinimum": 0},
                               "circular": {"type": "boolean"}},
            },
        },
        "nodes": {
            "type": "array", "minItems": 2,
            "items": {
                "type": "object", "required": ["id", "x", "y", "z"], "additionalProperties": False,
                "properties": {"id": {"type": "integer", "minimum": 1},
                               "x": {"type": "number"}, "y": {"type": "number"},
                               "z": {"type": "number"}},
            },
        },
        "members": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "required": ["id", "i", "j", "section", "material"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "i": {"type": "integer"}, "j": {"type": "integer"},
                    "section": {"type": "string"}, "material": {"type": "string"},
                    "ref_vector": _VEC3,
                    "offset_i": _VEC3,
                    "offset_j": _VEC3,
                    "releases": {
                        "type": "object", "additionalProperties": False,
                        "properties": {"i": _RELEASE_LIST, "j": _RELEASE_LIST},
                    },
                },
            },
        },
        "supports": {
            # 当前求解器只做静力分析；零约束的自由体必然形成机构。
            # 在 Schema 层拦住，错误才能准确指向 supports，而不是等线性
            # 求解器抛出难懂的“刚度矩阵奇异”。
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "required": ["node", "fix"], "additionalProperties": False,
                "properties": {"name": {"type": "string", "minLength": 1},
                               "node": {"type": "integer"},
                               "fix": {"type": "array", "minItems": 6, "maxItems": 6,
                                       "items": {"type": "integer", "enum": [0, 1]}}},
            },
        },
        # 单工况模型直接写这几项，等价于名为 "default" 的工况
        "nodal_loads": _NODAL_LOADS,
        "member_loads": _MEMBER_LOADS,
        "member_spans": _MEMBER_SPANS,
        "settlements": _SETTLEMENTS,
        "member_strains": _MEMBER_STRAINS,
        # 多工况模型写这一项
        "load_cases": {
            "type": "array", "minItems": 0,
            "items": {
                "type": "object", "required": ["name"], "additionalProperties": False,
                "properties": {"name": {"type": "string", "minLength": 1},
                               "nodal_loads": _NODAL_LOADS,
                               "member_loads": _MEMBER_LOADS,
                               "member_spans": _MEMBER_SPANS,
                               "settlements": _SETTLEMENTS,
                               "member_strains": _MEMBER_STRAINS},
            },
        },
        "combos": {
            "type": "array",
            "items": {
                "type": "object", "required": ["name", "factors"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "factors": {"type": "object", "minProperties": 1,
                                "additionalProperties": {"type": "number"}},
                },
            },
        },
    },
}


def migrate_payload(data: dict[str, Any]) -> dict[str, Any]:
    """把旧模型复制并升级到当前 Domain IR，不修改调用方的数据。

    当前唯一的旧格式是没有 ``schema_version`` 的 Alpha JSON（记作 v0）。
    v0 到 v1 只补版本标记，字段语义保持不变；未来版本必须新增显式迁移步骤，
    不能在这里猜测其含义。
    """
    if not isinstance(data, dict):
        raise TypeError("模型必须是 JSON 对象")
    version = data.get("schema_version")
    if version not in (None, CURRENT_SCHEMA_VERSION):
        raise ValueError(
            f"不支持的模型 schema_version={version!r}；"
            f"当前仅支持旧版无版本模型和 v{CURRENT_SCHEMA_VERSION}")
    migrated = deepcopy(data)
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    return migrated


def _fill_case(frame: Frame, case: LoadCase, data: dict[str, Any]) -> LoadCase:
    for e in data.get("nodal_loads", []):
        node = int(e["node"])
        previous = case.nodal_loads.get(node, (0.0,) * 6)
        case.nodal_loads[node] = tuple(a + float(b)
                                       for a, b in zip(previous, e["load"]))
    for e in data.get("member_loads", []):
        member = int(e["member"])
        previous = case.member_loads.get(member, (0.0,) * 3)
        case.member_loads[member] = tuple(a + float(b)
                                          for a, b in zip(previous, e["w"]))
    for e in data.get("member_spans", []):
        case.member_spans.setdefault(int(e["member"]), []).append(
            SpanLoad.from_dict(e))
    for e in data.get("settlements", []):
        case.settlements[int(e["node"])] = tuple(float(v) for v in e["d"])
    for e in data.get("member_strains", []):
        member = frame.members.get(int(e["member"]))
        if member is None:
            continue           # 语义校验会单独报"引用了不存在的杆件"
        # 换算成初应变。**存应变不存 Δl/ΔT**：应变沿杆是常量，杆件被自动
        # 剖分时各段直接继承，不需要按段长重新分配。
        strain = 0.0
        if e.get("lack_of_fit") is not None:
            pi, pj = member_endpoints(frame, member)
            length = float(np.linalg.norm(np.asarray(pj) - np.asarray(pi)))
            if length > 0.0:
                strain += float(e["lack_of_fit"]) / length
        if e.get("delta_t") is not None:
            strain += frame.materials[member.material].alpha * float(e["delta_t"])
        case.member_strains[member.id] = (
            case.member_strains.get(member.id, 0.0) + strain)
    return case


def from_dict(data: dict[str, Any]) -> Frame:
    """把 JSON 模型装成 Frame。不做 schema 校验——先用 jsonschema 校验再调用。"""
    f = Frame(units=str(data.get("units", "N-m-Pa")))
    for m in data["materials"]:
        f.materials[m["name"]] = Material(
            m["name"], float(m["E"]), float(m["nu"]), float(m.get("density", 0.0)),
            float(m["yield_stress"]) if m.get("yield_stress") is not None else None,
            float(m.get("hardening_ratio", 0.01)),
            float(m.get("alpha", 0.0)))
    for s in data["sections"]:
        f.sections[s["name"]] = Section(
            s["name"], float(s["A"]), float(s["Iy"]), float(s["Iz"]), float(s["J"]),
            float(s["Ay"]) if s.get("Ay") is not None else None,
            float(s["Az"]) if s.get("Az") is not None else None,
            float(s["cy"]) if s.get("cy") is not None else None,
            float(s["cz"]) if s.get("cz") is not None else None,
            bool(s.get("circular", False)))
    for n in data["nodes"]:
        f.nodes[int(n["id"])] = Node(int(n["id"]), float(n["x"]), float(n["y"]), float(n["z"]))
    for m in data["members"]:
        rel = m.get("releases") or {}
        f.members[int(m["id"])] = Member(
            int(m["id"]), int(m["i"]), int(m["j"]), m["section"], m["material"],
            tuple(m["ref_vector"]) if m.get("ref_vector") else None,
            tuple(rel.get("i", ())), tuple(rel.get("j", ())),
            tuple(float(v) for v in m.get("offset_i", (0.0, 0.0, 0.0))),
            tuple(float(v) for v in m.get("offset_j", (0.0, 0.0, 0.0))),
        )
    for s in data["supports"]:
        f.supports[int(s["node"])] = tuple(int(v) for v in s["fix"])

    _fill_case(f, f.case(DEFAULT_CASE), data)
    for c in data.get("load_cases", []):
        _fill_case(f, f.case(str(c["name"])), c)
    # 只写了多工况时，别留一个空的 default 干扰结果表
    if data.get("load_cases") and not any(
            data.get(k) for k in ("nodal_loads", "member_loads",
                                  "member_spans", "settlements",
                                  "member_strains")):
        f.load_cases.pop(DEFAULT_CASE, None)

    for c in data.get("combos", []):
        f.combos[str(c["name"])] = {str(k): float(v) for k, v in c["factors"].items()}
    return f


def validate_payload(data: dict[str, Any]) -> list[str]:
    """两级校验：schema 结构 + 模型语义。返回给大模型的错误清单。"""
    errors: list[str] = []
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(MODEL_SCHEMA)
        for e in sorted(v.iter_errors(data), key=lambda e: list(e.path)):
            loc = "/".join(str(p) for p in e.path) or "(根)"
            errors.append(f"[结构] {loc}: {e.message}")
    except ImportError:
        errors.append("[提示] 未安装 jsonschema，跳过结构校验")
    except AttributeError:
        # 装着 4.20 以前的 jsonschema：没有 Draft202012Validator。直接抛异常
        # 会把"依赖过旧"伪装成一整片校验失败，这里降级并说清真正的原因。
        errors.append("[提示] jsonschema 版本过低（需 >=4.20），跳过结构校验")
    if any(x.startswith("[结构]") for x in errors):
        return errors

    # jsonschema 的 number 在 Python 中可能接受 NaN/Infinity；这些数一旦进入
    # 刚度矩阵会一路传播，最后才以“求解器失败”的面貌出现。入口处直接指出路径。
    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"[结构] {path or '(根)'}: 数值必须有限，收到 {value}")
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}/{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}/{index}" if path else str(index))

    def unique(items: list[dict[str, Any]], key: str, label: str) -> None:
        values = [item.get(key) for item in items]
        duplicate = sorted({value for value in values
                            if value is not None and values.count(value) > 1},
                           key=str)
        if duplicate:
            errors.append(f"[语义] {label}重复：{duplicate}")

    walk(data)
    unique(data.get("materials") or [], "name", "材料名")
    unique(data.get("sections") or [], "name", "截面名")
    unique(data.get("nodes") or [], "id", "节点编号")
    unique(data.get("members") or [], "id", "杆件编号")
    unique(data.get("supports") or [], "node", "支座节点")
    unique(data.get("load_cases") or [], "name", "工况名")
    unique(data.get("combos") or [], "name", "组合名")

    blocks = [("顶层默认工况", data)] + [
        (f"工况 {case.get('name')!r}", case) for case in data.get("load_cases") or []]
    for label, block in blocks:
        # 多条力在物理上相加；给定位移却不能对同一节点写两套值。
        unique(block.get("settlements") or [], "node", f"{label}的给定位移节点")
        # 与 Abaqus 的 Load 对象一致：同一分析步内，一个载荷名只能代表一个
        # 载荷对象，不能在节点力、均布力和跨间力三张表里各藏一份同名载荷。
        named_loads = [item for key in ("nodal_loads", "member_loads", "member_spans")
                       for item in block.get(key) or [] if item.get("name")]
        unique(named_loads, "name", f"{label}的载荷名称")
        named_bc = [item for item in block.get("settlements") or [] if item.get("name")]
        unique(named_bc, "name", f"{label}的给定位移名称")
    if errors:
        return errors
    try:
        errors += [f"[语义] {msg}" for msg in check_model(from_dict(data))]
    except Exception as exc:
        errors.append(f"[语义] 模型装配失败: {exc}")
    return errors


def validate_definitions(materials: list[dict[str, Any]],
                         sections: list[dict[str, Any]], *,
                         allow_incomplete: bool = False) -> list[str]:
    """只校验 Property 阶段的材料/截面，不要求几何已经存在。

    Abaqus 允许先建材料、稍后再建截面；因此空模型编辑属性时可以暂时只
    有其中一类。已有杆件时则由调用方使用 ``validate_payload`` 做完整校验。
    """
    errors: list[str] = []
    if not allow_incomplete and (not materials or not sections):
        errors.append("[结构] 材料和截面都至少需要一个定义")
    if allow_incomplete and not materials and not sections:
        errors.append("[结构] 至少保留一个材料或截面定义")
    try:
        import jsonschema
        for key, values in (("materials", materials), ("sections", sections)):
            if not values and allow_incomplete:
                continue
            validator = jsonschema.Draft202012Validator(
                MODEL_SCHEMA["properties"][key])
            for error in sorted(validator.iter_errors(values),
                                key=lambda item: list(item.path)):
                loc = "/".join(str(part) for part in error.path) or key
                errors.append(f"[结构] {key}/{loc}: {error.message}")
    except ImportError:
        errors.append("[提示] 未安装 jsonschema，跳过结构校验")
    except AttributeError:
        # 装着 4.20 以前的 jsonschema：没有 Draft202012Validator。直接抛异常
        # 会把"依赖过旧"伪装成一整片校验失败，这里降级并说清真正的原因。
        errors.append("[提示] jsonschema 版本过低（需 >=4.20），跳过结构校验")

    for label, values in (("材料", materials), ("截面", sections)):
        names = [str(item.get("name", "")).strip() for item in values
                 if isinstance(item, dict)]
        duplicates = sorted({name for name in names if name and names.count(name) > 1})
        if duplicates:
            errors.append(f"[语义] {label}名称重复：{'、'.join(duplicates)}")
    return errors
