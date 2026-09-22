"""Session 的建模域：把"三跨两层"这类描述展开成节点与杆件，以及此后对
图元的一切增删改。

从 agent.py 搬出来的，代码一个字节没改。

这一组是整个 Session 里最大的一块（二十来个工具、九百多行），也是最该单独
成文件的一块：它只跟 Domain IR 打交道，不碰求解器、不碰结果——读它的时候
不需要知道任何关于求解的事。

拓扑由代码展开、不让大模型逐个列节点坐标，是 agent.py 开头那条设计原则的
第 2 条；这个文件就是那条原则的实现。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frame3d import LOCAL_DOF_NAMES
from generator import GeneratorError, beam_member_ids, describe, generate_frame, generate_portal_frame, rafter_member_ids
from model_io import validate_definitions, validate_payload
from session_base import ToolResult, _UNSET, _records
from units import convert_model
import bent as _bent
import changes as _changes


#: Abaqus 的命名边界条件类型 → 六个自由度的约束掩码（ux uy uz rx ry rz）。
#:
#: **为什么要有名字。** 让人去填六个 0/1 是这一步最容易出错的地方：
#: "对称面上该约束哪几个"没有几个人记得住，而填错了不会报错——结构照样
#: 算得出来，只是算的不是你想要的那个结构。Abaqus 用名字解决这件事，
#: 这里照搬同一套名字，迁移过来的人不用重新记。
#:
#: 对称面的法向决定约束哪些：**法向的平动**被约束（不能穿过对称面），
#: **面内两个轴的转动**被约束（转了就不对称了）。反对称正好相反：
#: 面内两个平动被约束，法向转动被约束。
BC_TYPES: dict[str, tuple[int, int, int, int, int, int]] = {
    "ENCASTRE": (1, 1, 1, 1, 1, 1),     # 完全固定
    "PINNED": (1, 1, 1, 0, 0, 0),       # 三向铰接
    "XSYMM": (1, 0, 0, 0, 1, 1),        # 对称面法向 X：U1=UR2=UR3=0
    "YSYMM": (0, 1, 0, 1, 0, 1),        # 法向 Y：U2=UR1=UR3=0
    "ZSYMM": (0, 0, 1, 1, 1, 0),        # 法向 Z：U3=UR1=UR2=0
    "XASYMM": (0, 1, 1, 1, 0, 0),       # 反对称，法向 X：U2=U3=UR1=0
    "YASYMM": (1, 0, 1, 0, 1, 0),
    "ZASYMM": (1, 1, 0, 0, 0, 1),
    "FREE": (0, 0, 0, 0, 0, 0),         # 解除约束
}

#: 给人看的一句话说明，列在 BC 管理器里
BC_TYPE_NOTES = {
    "ENCASTRE": "完全固定：六个自由度全约束",
    "PINNED": "铰接：三个平动约束，转动自由",
    "XSYMM": "对称面法向 X：不能穿过对称面，也不能绕面内两轴转",
    "YSYMM": "对称面法向 Y",
    "ZSYMM": "对称面法向 Z",
    "XASYMM": "反对称面法向 X：面内两个平动与法向转动被约束",
    "YASYMM": "反对称面法向 Y",
    "ZASYMM": "反对称面法向 Z",
    "FREE": "自由：解除该节点的全部约束",
}


def _next_bc_name(existing: list[dict]) -> str:
    """下一个没被占用的 BC-n。

    **默认名以前是常量 "BC-1"。** 于是不指定名字连建四个支座，四个都叫
    BC-1——而 delete_boundary_condition 是按名字匹配的，删「一条」会把四条
    一起删掉。删完模型没有支座了，那一步倒是会被校验挡住；但只要还剩别的
    支座，它就悄悄删多了，而"少了几个约束"从结果里看不出来，只会表现为
    位移偏大。Abaqus 的 BC 名是自动递增的，这里照做。
    """
    used = {str(item.get("name") or "") for item in existing}
    index = 1
    while f"BC-{index}" in used:
        index += 1
    return f"BC-{index}"


class ModelingMixin:
    """建模：几何生成与图元增删改。见模块 docstring。"""
    # --- 工具实现 ---
    @_records
    def define_materials_and_sections(self, materials, sections) -> ToolResult:
        # 单位制是项目设置，不是“定义属性”的副作用。用户若先选了毫米制，
        # 创建材料时不能又被悄悄改回 SI。
        from copy import deepcopy

        if not isinstance(materials, list) or not isinstance(sections, list):
            return ToolResult(False, {"error": "材料和截面必须分别用列表提供"})
        new_materials = deepcopy(materials)
        new_sections = deepcopy(sections)
        candidate = deepcopy(self.model)
        candidate.setdefault("units", "N-m-Pa")
        candidate["materials"] = new_materials
        candidate["sections"] = new_sections

        definition_errors = validate_definitions(
            new_materials, new_sections, allow_incomplete=True)
        if definition_errors:
            return ToolResult(False, {
                "errors": definition_errors,
                "hint": "属性定义未写入，原模型保持不变",
            })
        if (self.model.get("materials", []) == new_materials
                and self.model.get("sections", []) == new_sections):
            return ToolResult(True, {
                "materials": [m["name"] for m in new_materials],
                "sections": [s["name"] for s in new_sections],
                "no_change": True,
            })
        self.model = candidate
        self._invalidate()
        errors = validate_payload(candidate)
        return ToolResult(True, {"materials": [m["name"] for m in new_materials],
                                 "sections": [s["name"] for s in new_sections],
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def generate_frame(self, **kwargs) -> ToolResult:
        if not self.model.get("materials"):
            kwargs.setdefault("material", "")
        if not self.model.get("sections"):
            kwargs.setdefault("column_section", "")
            kwargs.setdefault("beam_section", "")
        try:
            generated = generate_frame(**kwargs)
        except (GeneratorError, TypeError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # 生成器的公开参数固定用 m / N / N·m，产物也是 SI。当前模型若已
        # 切到毫米制，必须先把新几何和荷载整体换到毫米制再沿用已有属性；
        # 只改 units 标签会把 6 m 的跨误成 6 mm，而且不会立即报错。
        target_units = self.model.get("units", "N-m-Pa")
        if target_units != generated.get("units", "N-m-Pa"):
            try:
                generated = convert_model(generated, target_units)
            except ValueError as exc:
                return ToolResult(False, {"error": str(exc)})
        keep = {k: self.model[k] for k in ("materials", "sections") if k in self.model}
        candidate = {**generated, **keep}
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        beams = beam_member_ids(self.model, kwargs.get("beam_section", "BEAM"))
        return ToolResult(True, {
            "summary": describe(self.model),
            "beam_member_ids": beams,
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "节点与杆件已按参数展开，不要再自行罗列坐标；"
                    + ("当前是几何草稿，请在 Property 模块指派材料和截面。"
                       if errors else
                       "要分工况加载就用 set_load_cases，并引用上面的梁编号"),
        })

    @_records
    def generate_portal_frame(self, **kwargs) -> ToolResult:
        if not self.model.get("materials"):
            kwargs.setdefault("material", "")
        if not self.model.get("sections"):
            kwargs.setdefault("column_section", "")
            kwargs.setdefault("rafter_section", "")
            kwargs.setdefault("tie_section", "")
        try:
            generated = generate_portal_frame(**kwargs)
        except (GeneratorError, TypeError) as exc:
            return ToolResult(False, {"error": str(exc)})
        target_units = self.model.get("units", "N-m-Pa")
        if target_units != generated.get("units", "N-m-Pa"):
            try:
                generated = convert_model(generated, target_units)
            except ValueError as exc:
                return ToolResult(False, {"error": str(exc)})
        keep = {k: self.model[k] for k in ("materials", "sections") if k in self.model}
        candidate = {**generated, **keep}
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "summary": describe(self.model),
            "rafter_member_ids": rafter_member_ids(self.model),
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "斜梁编号按几何判别，已排除纵向系杆；分工况加载请引用它",
        })



    # --- 命名集合 ---
    #
    # **编号规则是生成器定的，用户并不知道 27 号在哪。**
    # 没有名字，"给顶层所有梁加 5 kN/m" 这句话就没有落点。
    # 有了名字，这组杆件在整个会话里都能被稳定引用。
    #
    # 集合是纯粹的命名层：求解器看不到它，工具在调用时就展开成编号了。

    @_records
    def define_set(self, name: str, member_ids=None, node_ids=None,
                   note: str | None = None, **selector) -> ToolResult:
        """给一组杆件或节点起个名字。

        两种给法，二选一：
        · 直接给 `member_ids` / `node_ids`
        · 给几何条件（同 select_members 的参数），由代码挑出来
        """
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        name = str(name).strip()
        if not name:
            return ToolResult(False, {"error": "集合名不能为空"})

        entry: dict[str, Any] = {}
        if member_ids is None and node_ids is None:
            if not selector:
                return ToolResult(False, {
                    "error": "要么直接给 member_ids / node_ids，"
                             "要么给几何条件（section / orientation / x_range 等）"})
            try:
                picked = _bent.select_members(self.model, **selector)
            except (GeneratorError, TypeError, ValueError) as exc:
                return ToolResult(False, {"error": str(exc)})
            if not picked:
                return ToolResult(False, {
                    "error": "按这些条件一根杆件都没挑到。"
                             "**空集合不要存**——存下来之后引用它的地方会静默什么都不做，"
                             "比当场报错难查得多。检查 x/y/z_range 是否框对了位置"})
            entry["members"] = picked
        else:
            known_m = {int(m["id"]) for m in self.model.get("members") or []}
            known_n = {int(n["id"]) for n in self.model.get("nodes") or []}
            if member_ids is not None:
                bad = sorted({int(k) for k in member_ids} - known_m)
                if bad:
                    return ToolResult(False, {"error": f"杆件 {bad} 不存在"})
                entry["members"] = sorted({int(k) for k in member_ids})
            if node_ids is not None:
                bad = sorted({int(k) for k in node_ids} - known_n)
                if bad:
                    return ToolResult(False, {"error": f"节点 {bad} 不存在"})
                entry["nodes"] = sorted({int(k) for k in node_ids})
        if note:
            entry["note"] = str(note)

        sets = dict(self.model.get("sets") or {})
        replaced = name in sets
        sets[name] = entry
        self.model["sets"] = sets
        return ToolResult(True, {
            "name": name, **entry, "replaced": replaced,
            "sets": sorted(sets),
            "note": "之后凡是要杆件编号的地方，都可以直接写这个名字。",
        })

    def list_sets(self) -> ToolResult:
        sets = self.model.get("sets") or {}
        return ToolResult(True, {
            "sets": {k: {"members": len(v.get("members") or []),
                         "nodes": len(v.get("nodes") or []),
                         "note": v.get("note", "")}
                     for k, v in sets.items()},
            "count": len(sets),
        })

    def _expand_members(self, value) -> list[int]:
        """把"编号列表或集合名"统一展开成编号列表。

        **允许混着写**：`["顶层梁", 12, 13]` 是合法的——用户脑子里
        本来就是"那一组，再加这两根"。
        """
        sets = self.model.get("sets") or {}
        if isinstance(value, (str, int)):
            value = [value]
        out: list[int] = []
        for item in value or []:
            if isinstance(item, str):
                entry = sets.get(item)
                if entry is None:
                    raise ValueError(
                        f"没有名为「{item}」的集合。已有的集合："
                        + ("、".join(sorted(sets)) if sets else "（一个都没有）"))
                out.extend(entry.get("members") or [])
            else:
                out.append(int(item))
        seen, unique = set(), []
        for k in out:                       # 去重但保持顺序，便于人工核对
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique

    def _expand_nodes(self, value) -> list[int]:
        """把节点编号、编号列表或节点集合名展开为去重后的编号列表。"""
        sets = self.model.get("sets") or {}
        if isinstance(value, (str, int)):
            value = [value]
        out: list[int] = []
        for item in value or []:
            if isinstance(item, str):
                entry = sets.get(item)
                if entry is None:
                    raise ValueError(
                        f"没有名为「{item}」的集合。已有集合："
                        + ("、".join(sorted(sets)) if sets else "（无）"))
                out.extend(int(node) for node in entry.get("nodes") or [])
            else:
                out.append(int(item))
        return list(dict.fromkeys(out))

    # --- 一榀 / 拉伸 / 算子 ---
    #
    # **模板是加法，算子是乘法。** 两个模板给两种结构；一榀 + 拉伸 + 五个算子
    # 覆盖的是绝大多数刚架。而且每一个都是一句话能说清的，
    # 正好落在自然语言这一侧。

    def _length_factor_from_metres(self) -> float:
        """公开建模工具统一收米；这里换成当前模型内部长度单位。"""
        return 1000.0 if self.model.get("units") == "N-mm-MPa" else 1.0

    def _scaled_ranges(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """把工具参数里的 x/y/z 区间由米换到当前模型长度单位。"""
        out = dict(kwargs)
        factor = self._length_factor_from_metres()
        for name in ("x_range", "y_range", "z_range"):
            if out.get(name) is not None:
                out[name] = [float(value) * factor for value in out[name]]
        return out

    @_records
    def generate_bent(self, **kwargs) -> ToolResult:
        if not self.model.get("materials"):
            kwargs.setdefault("material", "")
        if not self.model.get("sections"):
            kwargs.setdefault("column_section", "")
            kwargs.setdefault("beam_section", "")
        try:
            model, ids = _bent.generate_bent(**kwargs)
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # generate_bent 的公开参数固定为米，底层产物也是 SI；不能只把
        # units 标签改成毫米制，否则 12 m 会静默变成 12 mm。
        model["units"] = "N-m-Pa"
        target_units = self.model.get("units", "N-m-Pa")
        if target_units != "N-m-Pa":
            try:
                model = convert_model(model, target_units)
            except ValueError as exc:
                return ToolResult(False, {"error": str(exc)})
        keep = {k: self.model[k] for k in ("materials", "sections")
                if k in self.model}
        candidate = {**model, **keep}
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "summary": describe(self.model), **ids,
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "一榀平面刚架。屋面形状全由 profile 折线决定："
                    "平屋面两个点，双坡三个点（中间那个是屋脊），"
                    "悬挑就把折线两端伸到柱子外面。要做成空间结构，"
                    "接着调 extrude_bents 沿纵向复制。",
        })

    @_records
    def extrude_bents(self, bays, tie_section=None) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型，请先生成一榀刚架"})
        try:
            factor = self._length_factor_from_metres()
            widths = [float(value) * factor for value in bays]
            model, info = _bent.extrude_bents(self.model, widths,
                                              tie_section=tie_section)
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        info.pop("node_map", None)              # 太大，不往回传
        return ToolResult(True, {
            "summary": describe(self.model), **info,
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "已沿 Y 方向复制成多榀并生成连系梁。"
                    "单榀时补的面外约束已经去掉——拉伸后不再是平面刚架，"
                    "留着会把结构在 Y 方向钉死，横向刚度算出来偏刚且不报错。",
        })

    def select_members(self, **kwargs) -> ToolResult:
        """按几何条件挑杆件。**不是让用户报编号**——编号规则是生成器定的。"""
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            ids = _bent.select_members(self.model, **self._scaled_ranges(kwargs))
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        return ToolResult(True, {
            "member_ids": ids, "count": len(ids),
            "note": "区间判定用的是杆件**中点**，这样「这一跨的梁」才有确定含义。",
        })

    @_records
    def add_bracing(self, **kwargs) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            model, info = _bent.add_bracing(
                self.model, **self._scaled_ranges(kwargs))
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        return ToolResult(True, {
            "summary": describe(self.model), **info,
            "analysis_ready": not errors, "warnings": errors,
            "note": "支撑两端默认铰接：它靠轴力工作，按刚接建会高估它对弯矩的贡献。"
                    "评价加撑效果时要看**被撑那一榀的侧移**，不要看全结构最大位移——"
                    "只撑局部时最大值会跑到没撑的地方去，看起来像没生效。",
        })

    @_records
    def raise_nodes(self, dz, **kwargs) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            factor = self._length_factor_from_metres()
            model, info = _bent.raise_nodes(
                self.model, dz=float(dz) * factor,
                **self._scaled_ranges(kwargs))
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # 区域移动可能把两个节点压到同一点，或把一根杆压成零长度。
        # 这种不是“尚未建完”，必须当场拒绝，不能留到求解时才报奇异。
        from draft import FrameDraft, SNAP_TOL
        draft = FrameDraft.from_model(model)
        coincident = draft.coincident_nodes(tol=SNAP_TOL / self.units.length_to_m)
        zero = [int(member["id"]) for member in model.get("members") or []
                if (draft.length(int(member["id"])) is not None
                    and draft.length(int(member["id"])) <
                    SNAP_TOL / self.units.length_to_m)]
        if coincident or zero:
            return ToolResult(False, {
                "error": "节点移动会造成重合节点或零长度杆件，改动未写入",
                "coincident_nodes": coincident, "zero_length_members": zero})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        info["dz"] = float(dz)       # 工具回报仍按公开接口的米
        return ToolResult(True, {"summary": describe(self.model), **info,
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def retaper(self, member_ids, section) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        if section not in {str(item["name"]) for item in self.model.get("sections") or []}:
            return ToolResult(False, {"error": f"截面 {section!r} 未定义"})
        try:
            member_ids = self._expand_members(member_ids)
            model, info = _bent.retaper(self.model, member_ids, section)
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        return ToolResult(True, {"summary": describe(self.model), **info,
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def add_nodes(self, coordinates: list[list[float]]) -> ToolResult:
        # 节点属于几何，不应偷偷创建 Q355/COL/BEAM。允许模型处于尚未指派
        # 属性的编辑态，等创建杆件时再明确要求材料与截面。
        from copy import deepcopy

        parsed = []
        for c in coordinates:
            try:
                valid_length = len(c) == 3
            except TypeError:
                valid_length = False
            if not valid_length:
                return ToolResult(False, {"error": f"坐标必须是三个数，收到 {c}"})
            try:
                point = [float(c[0]), float(c[1]), float(c[2])]
            except (TypeError, ValueError):
                return ToolResult(False, {"error": f"坐标必须是三个数，收到 {c}"})
            if not all(np.isfinite(value) for value in point):
                return ToolResult(False, {"error": f"坐标必须是有限数，收到 {c}"})
            parsed.append(point)
        candidate = deepcopy(self.model)
        candidate.setdefault("nodes", [])
        candidate.setdefault("units", "N-m-Pa")
        candidate.setdefault("materials", [])
        candidate.setdefault("sections", [])
        candidate.setdefault("supports", [])
        candidate.setdefault("load_cases", [])
        existing = {int(n["id"]) for n in candidate["nodes"]}
        next_id = (max(existing) + 1) if existing else 1
        snap_tol = 1e-6 / self.units.length_to_m
        added = []
        reused = []
        input_ids = []
        for x, y, z in parsed:
            hit = next((int(node["id"]) for node in candidate["nodes"]
                        if float(np.linalg.norm(
                            np.array([node["x"] - x, node["y"] - y, node["z"] - z],
                                     dtype=float))) <= snap_tol), None)
            if hit is not None:
                reused.append(hit)
                input_ids.append(hit)
                continue
            candidate["nodes"].append({"id": next_id, "x": x, "y": y, "z": z})
            added.append(next_id)
            input_ids.append(next_id)
            next_id += 1
        if not added:
            return ToolResult(True, {"added_node_ids": [],
                                     "reused_node_ids": reused,
                                     "input_node_ids": input_ids,
                                     "no_change": True,
                                     "summary": describe(self.model)})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"added_node_ids": added,
                                 "reused_node_ids": reused,
                                 "input_node_ids": input_ids,
                                 "summary": describe(self.model)})

    # --- 就地修改单个对象 ---
    #
    # 属性面板靠这两个。**面板不能自己去改 model 字典**——那样就绕开了
    # 建模过程记录，改完按 Ctrl+Z 什么也不会发生，而用户完全预期能撤销。
    # 所以走 @_records，和 Agent 改的是同一条链。

    @_records
    def edit_member(self, member_id: int, section: str | None = None,
                    material: str | None = None,
                    releases_i: list[str] | None = None,
                    releases_j: list[str] | None = None,
                    ref_vector: list[float] | None | object = _UNSET,
                    offset_i: list[float] | None | object = _UNSET,
                    offset_j: list[float] | None | object = _UNSET,
                    mu_y: float | None | object = _UNSET,
                    mu_z: float | None | object = _UNSET) -> ToolResult:
        """改一根杆件的截面、材料、局部轴向或端部释放。只传要改的项。

        释放传空列表表示**取消释放**（刚接）；不传表示不动它——
        这两件事必须分得开，否则用户没法把一根铰接梁改回刚接。

        ``ref_vector`` 是局部 y 轴的全局参考向量，用于非对称截面；传
        ``None`` 或空列表可恢复自动定向。参考向量与杆轴平行会在这里拒绝，
        不把一个求解时才爆出的错误留给用户。
        """
        from copy import deepcopy

        candidate = deepcopy(self.model)
        target = None
        for m in candidate.get("members") or []:
            if int(m["id"]) == int(member_id):
                target = m
                break
        if target is None:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})

        changed: dict[str, Any] = {}
        if section is not None:
            known = {s["name"] for s in candidate.get("sections") or []}
            if section not in known:
                return ToolResult(False, {
                    "error": f"截面 {section!r} 未定义",
                    "hint": f"已定义的截面：{'、'.join(sorted(known)) or '（无）'}"})
            if section != target.get("section"):
                changed["section"] = (target.get("section"), section)
                target["section"] = section
        if material is not None:
            known = {m["name"] for m in candidate.get("materials") or []}
            if material not in known:
                return ToolResult(False, {
                    "error": f"材料 {material!r} 未定义",
                    "hint": f"已定义的材料：{'、'.join(sorted(known)) or '（无）'}"})
            if material != target.get("material"):
                changed["material"] = (target.get("material"), material)
                target["material"] = material

        rel = dict(target.get("releases") or {})
        for end, value in (("i", releases_i), ("j", releases_j)):
            if value is None:
                continue
            bad = [k for k in value if k not in LOCAL_DOF_NAMES]
            if bad:
                return ToolResult(False, {
                    "error": f"{bad} 不是有效的自由度名",
                    "hint": f"可用：{'、'.join(LOCAL_DOF_NAMES)}"})
            before = list(rel.get(end) or [])
            if value:
                rel[end] = list(value)
            else:
                rel.pop(end, None)          # 空列表 = 取消释放
            if before != list(value):
                changed[f"releases_{end}"] = (before, list(value))
        if rel:
            target["releases"] = rel
        else:
            target.pop("releases", None)

        if ref_vector is not _UNSET:
            before = target.get("ref_vector")
            if ref_vector is None or ref_vector == []:
                vector = None
            else:
                try:
                    vector = [float(v) for v in ref_vector]
                except (TypeError, ValueError):
                    return ToolResult(False, {
                        "error": "梁方向参考向量必须是三个数字，例如 [0, 0, 1]"})
                if len(vector) != 3 or not all(np.isfinite(vector)):
                    return ToolResult(False, {
                        "error": "梁方向参考向量必须是三个有限数字，例如 [0, 0, 1]"})
                if float(np.linalg.norm(vector)) < 1e-12:
                    return ToolResult(False, {"error": "梁方向参考向量不能是零向量"})
                nodes = {int(n["id"]): n for n in candidate.get("nodes") or []}
                ni, nj = nodes.get(int(target["i"])), nodes.get(int(target["j"]))
                if ni is None or nj is None:
                    return ToolResult(False, {"error": "杆件端点不存在，无法设置梁方向"})
                axis = np.array([float(nj[k]) - float(ni[k]) for k in ("x", "y", "z")])
                if float(np.linalg.norm(np.cross(axis, vector))) < 1e-12:
                    return ToolResult(False, {
                        "error": "梁方向参考向量不能与杆件轴线平行",
                        "hint": "请换一个垂直于杆件轴线的方向，例如 [0, 0, 1]。"})
            if before != vector:
                changed["ref_vector"] = (before, vector)
                if vector is None:
                    target.pop("ref_vector", None)
                else:
                    target["ref_vector"] = vector

        for key, raw in (("offset_i", offset_i), ("offset_j", offset_j)):
            if raw is _UNSET:
                continue
            before = target.get(key)
            if raw is None or raw == []:
                vector = None
            else:
                try:
                    vector = [float(v) for v in raw]
                except (TypeError, ValueError):
                    return ToolResult(False, {"error": f"{key} 必须是三个数字"})
                if len(vector) != 3 or not all(np.isfinite(vector)):
                    return ToolResult(False, {"error": f"{key} 必须是三个有限数字"})
                if np.linalg.norm(vector) <= 1e-14:
                    vector = None
            if before != vector:
                changed[key] = (before, vector)
                if vector is None:
                    target.pop(key, None)
                else:
                    target[key] = vector

        for key, raw in (("mu_y", mu_y), ("mu_z", mu_z)):
            if raw is _UNSET:
                continue
            before = target.get(key)
            if raw is None:
                value = None
            else:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    return ToolResult(False, {"error": f"{key} 必须是数字"})
                if not np.isfinite(value) or value <= 0.0:
                    return ToolResult(False, {
                        "error": f"{key} 必须是正数",
                        "hint": "计算长度系数常用值：两端铰支 1.0、一端固定一端自由 2.0、"
                                "一端固定一端铰支 0.7、两端固定 0.5"})
            if before != value:
                changed[key] = (before, value)
                if value is None:
                    target.pop(key, None)
                else:
                    target[key] = value

        if not changed:
            # **值没变就当没发生。** 属性面板每次重填表单都会把当前值
            # 再写一遍控件；照记不误的话，建模过程里会堆满"修改杆件 3："
            # 这种空步骤，撤销一次什么都不动，用户会以为撤销坏了
            return ToolResult(True, {"member": int(member_id), "changed": {},
                                     "summary": describe(self.model),
                                     "no_change": True,
                                     "note": "值与原来相同，未作改动"})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "member": int(member_id),
            "changed": {k: {"was": a, "now": b} for k, (a, b) in changed.items()},
            "summary": describe(self.model),
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "改动已生效，原有分析结果已失效，请重新求解。",
        })

    @_records
    def edit_node(self, node_id: int, x: float | None = None,
                  y: float | None = None, z: float | None = None,
                  fix: list[int] | None = None) -> ToolResult:
        """改一个节点的坐标或约束。只传要改的项。

        `fix` 是六个 0/1，顺序 ux uy uz rx ry rz；传空列表表示**取消全部约束**。
        """
        from copy import deepcopy

        candidate = deepcopy(self.model)
        target = None
        for n in candidate.get("nodes") or []:
            if int(n["id"]) == int(node_id):
                target = n
                break
        if target is None:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})

        changed: dict[str, Any] = {}
        for key, value in (("x", x), ("y", y), ("z", z)):
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                return ToolResult(False, {"error": f"节点坐标 {key} 必须是数字"})
            if not np.isfinite(number):
                return ToolResult(False, {"error": f"节点坐标 {key} 必须是有限数"})
            if number != float(target[key]):
                changed[key] = (float(target[key]), number)
                target[key] = number

        if any(value is not None for value in (x, y, z)):
            point = np.array([target[k] for k in ("x", "y", "z")], dtype=float)
            for other in candidate.get("nodes") or []:
                if int(other["id"]) == int(node_id):
                    continue
                other_point = np.array([other[k] for k in ("x", "y", "z")], dtype=float)
                if float(np.linalg.norm(point - other_point)) < 1e-9:
                    return ToolResult(False, {
                        "error": f"移动后会与节点 {other['id']} 重合，改动未写入"})

        if fix is not None:
            if fix and (len(fix) != 6 or any(v not in (0, 1) for v in fix)):
                return ToolResult(False, {
                    "error": "fix 必须是六个 0 或 1，顺序 ux uy uz rx ry rz"})
            supports = [s for s in candidate.get("supports") or []
                        if int(s["node"]) != int(node_id)]
            before_entry = next((s for s in candidate.get("supports") or []
                                 if int(s["node"]) == int(node_id)), None)
            before = before_entry["fix"] if before_entry else None
            if fix:
                entry = {"node": int(node_id), "fix": [int(v) for v in fix]}
                if before_entry and before_entry.get("name"):
                    entry["name"] = before_entry["name"]
                supports.append(entry)
            if before != (list(fix) if fix else None):
                changed["fix"] = (before, list(fix) if fix else None)
            candidate["supports"] = sorted(supports, key=lambda s: s["node"])

        if not changed:
            return ToolResult(True, {"node": int(node_id), "changed": {},
                                     "summary": describe(self.model),
                                     "no_change": True,
                                     "note": "值与原来相同，未作改动"})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "node": int(node_id),
            "changed": {k: {"was": a, "now": b} for k, (a, b) in changed.items()},
            "summary": describe(self.model),
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "改动已生效，原有分析结果已失效，请重新求解。",
        })

    @_records
    def add_members(self, pairs: list[list[int]], section: str | None = None,
                    material: str | None = None,
                    releases_i: list[str] | None = None,
                    releases_j: list[str] | None = None) -> ToolResult:
        from copy import deepcopy

        if not pairs:
            return ToolResult(False, {"error": "至少需要一对节点来创建杆件"})
        bad_releases = [value for value in (releases_i or []) + (releases_j or [])
                        if value not in LOCAL_DOF_NAMES]
        if bad_releases:
            return ToolResult(False, {
                "error": f"{bad_releases} 不是有效的局部自由度",
                "hint": f"可用：{'、'.join(LOCAL_DOF_NAMES)}"})
        known = {int(n["id"]) for n in self.model.get("nodes", [])}
        nodes = {int(n["id"]): n for n in self.model.get("nodes", [])}
        existing_members = self.model.get("members") or []
        known_sections = {str(s["name"]) for s in self.model.get("sections") or []}
        known_materials = {str(m["name"]) for m in self.model.get("materials") or []}
        if section and known_sections and section not in known_sections:
            return ToolResult(False, {
                "error": f"截面 {section!r} 未定义",
                "hint": "请选择已有截面，或留空后稍后统一指派"})
        if material and known_materials and material not in known_materials:
            return ToolResult(False, {
                "error": f"材料 {material!r} 未定义",
                "hint": "请选择已有材料，或留空后稍后统一指派"})
        if section is None:
            if existing_members and existing_members[0].get("section") in known_sections:
                section = existing_members[0]["section"]
            elif len(known_sections) == 1:
                section = next(iter(known_sections))
        if material is None:
            if existing_members and existing_members[0].get("material") in known_materials:
                material = existing_members[0]["material"]
            elif len(known_materials) == 1:
                material = next(iter(known_materials))
        section_name = str(section or "")
        mat = str(material or "")
        next_id = (max(int(m["id"]) for m in existing_members) + 1) if existing_members else 1
        seen = {(min(int(m["i"]), int(m["j"])), max(int(m["i"]), int(m["j"])))
                for m in existing_members}
        parsed_pairs = []
        for pair in pairs:
            try:
                valid_length = len(pair) == 2
            except TypeError:
                valid_length = False
            if not valid_length:
                return ToolResult(False, {"error": f"每项必须是两个节点编号，收到 {pair}"})
            try:
                i, j = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                return ToolResult(False, {"error": f"节点编号必须是整数，收到 {pair}"})
            missing = [n for n in (i, j) if n not in known]
            if missing:
                return ToolResult(False, {"error": f"节点 {missing} 不存在",
                                          "hint": "先用 add_nodes 建节点"})
            if i == j:
                return ToolResult(False, {"error": f"杆件起点和终点不能都是节点 {i}"})
            a, b = nodes[i], nodes[j]
            length = float(np.linalg.norm(np.array(
                [b[k] - a[k] for k in ("x", "y", "z")], dtype=float)))
            if length < 1e-6 / self.units.length_to_m:
                return ToolResult(False, {
                    "error": f"节点 {i} 与 {j} 重合或距离过小，不能创建零长度杆件"})
            key = (min(i, j), max(i, j))
            if key in seen:
                return ToolResult(False, {"error": f"节点 {i} 与 {j} 之间已经有杆件"})
            seen.add(key)
            parsed_pairs.append((i, j))

        candidate = deepcopy(self.model)
        candidate.setdefault("members", [])
        added = []
        for i, j in parsed_pairs:
            entry: dict[str, Any] = {"id": next_id, "i": i, "j": j,
                                     "section": section_name, "material": mat}
            rel = {}
            if releases_i:
                rel["i"] = list(releases_i)
            if releases_j:
                rel["j"] = list(releases_j)
            if rel:
                entry["releases"] = rel
            candidate["members"].append(entry)
            added.append(next_id)
            next_id += 1
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"added_member_ids": added,
                                 "summary": describe(self.model),
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def add_step(self, name: str, analysis: str = "linear",
                 loads: dict | None = None,
                 deactivate_loads: list | None = None,
                 supports: dict | None = None,
                 deactivate_supports: list | None = None,
                 increments: int = 10,
                 after: str | None = None) -> ToolResult:
        """新增一个分析步（Abaqus 的 Step），追加到末尾或插在某一步之后。

        **荷载与边界条件在步之间传播。** 这一步只写新建或改写的东西，上一步
        有而这里没提的会自动沿用——要让它消失必须显式写进 deactivate。这是
        Abaqus 的语义，也是最容易被误解的一点：漏写不等于撤销。

        ``loads`` 是 工况名 -> 幅值曲线名，例如 {"DL": "STEP", "WX": "RAMP"}。
        ``supports`` 是 节点号 -> {"fix": [...], "spring": [...]}，用来在这一步
        改写某个支座（比如把固接换成铰接）。

        analysis 目前支持 linear 与 pdelta。材料非线性没放进来是因为一个分析步
        里常有几个工况同时生效，而材料非线性不能事后叠加，缺一个合成工况的
        增量入口；硬放进来只能偷偷按单工况各算各的，那是错的答案。
        """
        import steps as step_module

        try:
            step = step_module.Step(
                name=str(name), analysis=str(analysis),
                loads={str(k): str(v) for k, v in (loads or {}).items()},
                deactivate_loads=tuple(str(c) for c in (deactivate_loads or ())),
                supports={int(k): dict(v) for k, v in (supports or {}).items()},
                deactivate_supports=tuple(int(n) for n in
                                          (deactivate_supports or ())),
                increments=int(increments))
        except (ValueError, TypeError) as exc:
            return ToolResult(False, {"error": str(exc)})

        existing = step_module.from_payload(self.model)
        names = [s.name for s in existing]
        if step.name in names:
            return ToolResult(False, {
                "error": f"已经有名为 {step.name!r} 的分析步", "steps": names})
        if after is None:
            existing.append(step)
        elif after in names:
            existing.insert(names.index(after) + 1, step)
        else:
            return ToolResult(False, {
                "error": f"没有名为 {after!r} 的分析步，插不进去", "steps": names})

        return self._write_steps(existing, {"added": step.name})

    def list_steps(self) -> ToolResult:
        """列出全部分析步，并给出每一步**实际生效**的荷载与边界条件。

        传播是隐式的：第三步写着一行「失活活载」，实际生效的却是第一步那几个
        荷载减掉活载。光看声明看不出这件事，所以这里把结算结果一并给出——
        声明与生效分两栏，哪一步到底在算什么一眼可见。
        """
        import steps as step_module

        declared = step_module.from_payload(self.model)
        if not declared:
            return ToolResult(True, {"steps": [], "count": 0,
                                     "note": "模型里还没有分析步"})
        payload = {"declared": step_module.to_payload(declared),
                   "count": len(declared)}
        try:
            payload["effective"] = [
                {"step": e.name, "analysis": e.analysis,
                 "increments": e.increments, "loads": dict(e.loads),
                 "supports": sorted(e.supports), "changes": list(e.changes)}
                for e in step_module.resolve(declared, self.model)]
        except ValueError as exc:
            payload["resolve_error"] = str(exc)
            payload["hint"] = "这串分析步结算不出来，求解会直接报同样的错"
        return ToolResult(True, payload)

    @_records
    def delete_step(self, name: str) -> ToolResult:
        """删除一个分析步。

        **后面的步会跟着变**：传播是按顺序结算的，删掉中间一步，它建立的荷载
        或支座改写就不再存在，后面某一步的「失活」可能因此失去对象而报错。
        返回里会把删除后的结算结果一并给出，好核对这件事。
        """
        import steps as step_module

        existing = step_module.from_payload(self.model)
        names = [s.name for s in existing]
        if str(name) not in names:
            return ToolResult(False, {
                "error": f"没有名为 {name!r} 的分析步", "steps": names})
        keep = [s for s in existing if s.name != str(name)]
        return self._write_steps(keep, {"deleted": str(name)})

    def _write_steps(self, steps: list, extra: dict) -> ToolResult:
        """把分析步写回模型，写之前先结算一遍——结算不过就不写。"""
        from copy import deepcopy

        import steps as step_module

        candidate = deepcopy(self.model)
        candidate["steps"] = step_module.to_payload(steps)
        before = set(validate_payload(self.model))
        errors = [e for e in validate_payload(candidate) if e not in before]
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "分析步未写入，原模型保持不变"})
        try:
            effective = step_module.resolve(steps, candidate)
        except ValueError as exc:
            return ToolResult(False, {
                "error": str(exc),
                "hint": "这串分析步结算不出来，没有写入；先把它改成立得住的顺序"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            **extra, "count": len(steps),
            "steps": [{"step": e.name, "analysis": e.analysis,
                       "loads": dict(e.loads), "supports": sorted(e.supports),
                       "changes": list(e.changes)} for e in effective],
            "note": "loads 与 supports 是**结算后实际生效**的，含从前面步传播"
                    "过来的部分，不只是这一步声明的"})

    #: 常见节点形式的连接刚度，按**梁线刚度 EI/L 的倍数**给。
    #:
    #: 绝对刚度没法预设——同一个端板连接装在 H400 上和装在 H200 上，
    #: 相对刚度差好几倍。规范和文献里的分类也都是按 EI/L 归一的
    #: （EC3 把 k ≥ 25EI/L 算刚接、k ≤ 0.5EI/L 算铰接，有侧移框架门槛更高）。
    CONNECTION_TYPES = {
        "刚接": (None, "完全刚性。用 bc/releases 表达，不必设弹簧"),
        "端板": (20.0, "外伸端板、加劲：能传约 80% 的梁端弯矩，接近刚接"),
        "平端板": (8.0, "平齐端板：半刚性的典型值"),
        "顶底角钢": (3.0, "顶底角钢带双腹板角钢：明显半刚性"),
        "腹板角钢": (1.0, "双腹板角钢：偏柔，接近铰接但不是铰"),
        "铰接": (None, "理想铰。用 releases 表达，那是 k=0 的极限"),
    }

    @_records
    def set_member_connection(self, member_ids, end: str = "j",
                              dof: str = "rz",
                              stiffness: float | None = None,
                              connection_type: str | None = None,
                              clear: bool = False) -> ToolResult:
        """给杆端装半刚性连接（转动弹簧）。

        **真实的梁柱节点既不是铰也不是刚接。** 两头都按极限算，弯矩分布差
        得很远：两跨连续梁上，刚接支座弯矩 90 kN·m，连接刚度取 3EI/L 时只剩
        53 kN·m，差出来的那部分全跑到跨中去了——而两种算法都不会报错。

        刚度可以直接给绝对值（力·长度/弧度），也可以用 ``connection_type``
        按**梁线刚度 EI/L 的倍数**给。后者更可用：绝对刚度离开截面和跨度
        就没有意义，同一个端板装在不同梁上相对刚度差几倍。

        ``end`` 取 "i"/"j"/"both"，``dof`` 是局部自由度名（梁端弯矩通常是
        "rz"）。``clear=True`` 去掉该端的连接弹簧，恢复刚接。
        """
        from copy import deepcopy

        try:
            ids = self._expand_members(member_ids)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "没有选中任何杆件"})
        if end not in ("i", "j", "both"):
            return ToolResult(False, {"error": 'end 只能是 "i"、"j" 或 "both"'})
        if dof not in LOCAL_DOF_NAMES:
            return ToolResult(False, {
                "error": f"dof 只能取自 {list(LOCAL_DOF_NAMES)}"})

        ends = ("i", "j") if end == "both" else (end,)
        candidate = deepcopy(self.model)
        index = {int(m["id"]): m for m in candidate["members"]}
        missing = [i for i in ids if i not in index]
        if missing:
            return ToolResult(False, {"error": f"没有杆件 {missing}"})

        applied = {}
        for member_id in ids:
            entry = index[member_id]
            table = entry.setdefault("connections", {})
            if clear:
                for side in ends:
                    (table.get(side) or {}).pop(dof, None)
                    if not table.get(side):
                        table.pop(side, None)
                if not table:
                    entry.pop("connections", None)
                applied[member_id] = None
                continue
            value = stiffness
            if value is None:
                if connection_type is None:
                    return ToolResult(False, {
                        "error": "要么给 stiffness，要么给 connection_type",
                        "connection_types": {
                            k: v[1] for k, v in self.CONNECTION_TYPES.items()}})
                spec = self.CONNECTION_TYPES.get(str(connection_type))
                if spec is None:
                    return ToolResult(False, {
                        "error": f"不认识的节点形式 {connection_type!r}",
                        "connection_types": {
                            k: v[1] for k, v in self.CONNECTION_TYPES.items()}})
                ratio, note = spec
                if ratio is None:
                    return ToolResult(False, {
                        "error": f"{connection_type} 不是半刚性连接：{note}"})
                value = ratio * self._beam_line_stiffness(entry, candidate)
            value = float(value)
            if value <= 0:
                return ToolResult(False, {
                    "error": "连接刚度必须是正数；理想铰请用杆端释放表达"})
            for side in ends:
                table.setdefault(side, {})[dof] = value
            applied[member_id] = value

        errors = validate_payload(candidate)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "连接未写入，原模型保持不变"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "members": ids, "end": end, "dof": dof,
            "stiffness": applied, "cleared": bool(clear),
            "note": "连接刚度的单位是 力·长度/弧度。理想铰是 k=0 的极限、"
                    "刚接是 k=∞ 的极限，半刚性落在中间——"
                    "两头都按极限算，弯矩分布差得很远。"})

    def _beam_line_stiffness(self, entry: dict, payload: dict) -> float:
        """梁线刚度 EI/L，用来把"几倍 EI/L"换算成绝对刚度。

        I 取绕局部 z 的惯性矩（梁端弯矩通常绕它），L 取杆件两端的直线距离。
        """
        import math

        sections = {s["name"]: s for s in payload["sections"]}
        materials = {m["name"]: m for m in payload["materials"]}
        nodes = {int(n["id"]): n for n in payload["nodes"]}
        section = sections[entry["section"]]
        modulus = float(materials[entry["material"]]["E"])
        start, finish = nodes[int(entry["i"])], nodes[int(entry["j"])]
        length = math.dist((start["x"], start["y"], start["z"]),
                           (finish["x"], finish["y"], finish["z"]))
        return modulus * float(section["Iz"]) / length

    # 只读的清单工具不记建模历史：它不改模型，记进去只会让时间轴上多出
    # 一堆"什么也没做"的条目，把真正的建模步骤淹掉。
    def list_boundary_conditions(self) -> ToolResult:
        """列出全部边界条件——相当于 Abaqus 的 BC Manager。

        以前只能去读模型 JSON。边界条件是**最容易改错又最难看出来**的一类
        对象：多约束一个自由度，结构照样算得出来，只是算的不是你想要的那个。
        所以要能一眼看全：谁、在哪些节点、约束了什么、是不是某个标准类型。
        """
        rows: list[dict[str, Any]] = []
        by_name: dict[str, dict[str, Any]] = {}
        for entry in self.model.get("supports") or []:
            name = str(entry.get("name") or f"(未命名-节点{entry['node']})")
            mask = tuple(int(v) for v in entry.get("fix") or (0,) * 6)
            spring = entry.get("spring")
            key = (name, mask, tuple(spring) if spring else None)
            bucket = by_name.get(str(key))
            if bucket is None:
                matched = next((k for k, v in BC_TYPES.items() if v == mask), None)
                bucket = {
                    "name": name, "nodes": [], "fix": list(mask),
                    "type": matched or "自定义",
                    "means": BC_TYPE_NOTES.get(matched or "",
                                               "自定义掩码，不对应标准类型"),
                    **({"spring": list(spring)} if spring else {}),
                }
                by_name[str(key)] = bucket
                rows.append(bucket)
            bucket["nodes"].append(int(entry["node"]))
        for row in rows:
            row["nodes"] = sorted(row["nodes"])
            row["count"] = len(row["nodes"])
        free = [row["name"] for row in rows if not any(row["fix"])
                and not row.get("spring")]
        return ToolResult(True, {
            "count": len(rows), "boundary_conditions": rows,
            "restrained_dofs": sum(sum(r["fix"]) * r["count"] for r in rows),
            **({"warning": f"边界条件 {free} 什么都没约束——写了名字但六个自由度"
                           "全是 0，多半是想删没删干净"} if free else {}),
            "note": "type 是按掩码反查出来的标准类型名；对不上任何一种就标"
                    "「自定义」。Initial 阶段的约束在这里；分析步里的给定位移"
                    "属于荷载工况，用 query_results 或看模型的 settlements。",
        })

    @_records
    def delete_boundary_condition(self, name: str) -> ToolResult:
        """按名字删掉一条边界条件。

        这个操作**会让结构少掉约束**，所以删完立刻跑一次奇异诊断：约束不够
        的话求解会在很后面才失败，而那时错误信息指向的是刚度矩阵，
        不是"你刚才删掉的那条"。
        """
        from copy import deepcopy

        target = str(name).strip()
        supports = list(self.model.get("supports") or [])
        keep = [s for s in supports if str(s.get("name")) != target]
        if len(keep) == len(supports):
            return ToolResult(False, {
                "error": f"没有名为 {target!r} 的边界条件",
                "available": sorted({str(s.get("name")) for s in supports})})
        candidate = deepcopy(self.model)
        candidate["supports"] = keep
        errors = validate_payload(candidate)
        if errors:
            return ToolResult(False, {
                "errors": errors,
                "hint": f"删掉 {target!r} 之后模型不合法，原模型保持不变"})
        self.model = candidate
        self._invalidate()
        removed = len(supports) - len(keep)
        payload: dict[str, Any] = {"deleted": target, "entries": removed,
                                   "remaining": len(keep)}
        diagnosis = self.diagnose_supports()
        if diagnosis.ok and diagnosis.payload.get("modes"):
            payload["warning"] = ("删掉之后结构出现了刚体位移模态——约束已经不够，"
                                  "现在求解会失败。")
            payload["modes"] = diagnosis.payload["modes"]
        return ToolResult(True, payload)

    def set_supports(self, node_ids, fix: list[int] | None = None,
                     name: str | None = None,
                     spring: list[float] | None = None,
                     bc_type: str | None = None) -> ToolResult:
        """给节点或节点集合统一施加位移边界；边界属于 Initial 阶段。

        ``spring`` 给出六个方向的**支承刚度**（平动 力/长度，转动
        力·长度/弧度），0 表示该方向没有弹簧。桩基、弹性地基、橡胶支座、
        相邻结构的约束刚度都是这个形状——以前只能在"完全固定"和"完全自由"
        之间二选一。

        同一方向不能既 fix=1 又给弹簧：刚性约束会把自由度整个划掉，
        弹簧永远用不上，而用户以为它在起作用。这里当场拒绝。
        """
        from copy import deepcopy

        try:
            ids = self._expand_nodes(node_ids)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "至少选择一个节点"})
        if bc_type is not None:
            key = str(bc_type).upper()
            if key not in BC_TYPES:
                return ToolResult(False, {
                    "error": f"bc_type 只能取 {sorted(BC_TYPES)}",
                    "hint": "对称面用 XSYMM/YSYMM/ZSYMM，法向取对称面的法线方向"})
            if fix is not None and list(fix) != list(BC_TYPES[key]):
                # 两个都给且不一致时不能默默挑一个——那正是"算的不是你想要
                # 的那个结构"的来源
                return ToolResult(False, {
                    "error": f"同时给了 bc_type={key} 和 fix={list(fix)}，"
                             "而两者不一致，无法判断你要哪个",
                    "bc_type_means": list(BC_TYPES[key])})
            fix = list(BC_TYPES[key])
        if fix is None:
            return ToolResult(False, {
                "error": "要么给 fix（六个 0/1），要么给 bc_type",
                "bc_types": sorted(BC_TYPES)})
        if len(fix) != 6 or any(value not in (0, 1) for value in fix):
            return ToolResult(False, {
                "error": "fix 必须是六个 0 或 1，顺序 ux uy uz rx ry rz"})
        stiffness: list[float] | None = None
        if spring is not None:
            try:
                stiffness = [float(v) for v in spring]
            except (TypeError, ValueError):
                return ToolResult(False, {"error": "spring 必须是六个数字"})
            if len(stiffness) != 6 or not all(np.isfinite(v) and v >= 0
                                              for v in stiffness):
                return ToolResult(False, {
                    "error": "spring 必须是六个非负有限数，顺序 ux uy uz rx ry rz"})
            clash = [k for k, (f, v) in enumerate(zip(fix, stiffness,
                                                      strict=True)) if f and v]
            if clash:
                return ToolResult(False, {
                    "error": f"方向 {clash} 既写了 fix=1 又给了弹簧刚度。"
                             "刚性约束会让弹簧完全失效，两者只能取一个。",
                    "hint": "要刚接就把 spring 那一项设为 0，"
                            "要弹性支承就把 fix 那一项设为 0。"})
            if not any(stiffness):
                stiffness = None
        known = {int(node["id"]) for node in self.model.get("nodes") or []}
        missing = sorted(set(ids) - known)
        if missing:
            return ToolResult(False, {"error": f"节点 {missing} 不存在"})

        candidate = deepcopy(self.model)
        selected = set(ids)
        supports = [support for support in candidate.get("supports") or []
                    if int(support["node"]) not in selected]
        bc_name = str(name).strip() if name else _next_bc_name(supports)
        if any(fix) or stiffness:
            supports.extend({"name": bc_name, "node": node,
                             "fix": [int(v) for v in fix],
                             **({"spring": list(stiffness)} if stiffness else {})}
                            for node in ids)
        candidate["supports"] = sorted(supports, key=lambda item: int(item["node"]))
        if candidate.get("supports") == self.model.get("supports", []):
            return ToolResult(True, {"nodes": ids, "name": name,
                                     "step": "Initial", "no_change": True})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "nodes": ids, "fix": list(fix), "name": bc_name,
            "step": "Initial", "analysis_ready": not errors,
            "warnings": errors, "summary": describe(self.model),
        })

    @_records
    def assign_properties(self, member_ids, section: str,
                          material: str) -> ToolResult:
        """给一批杆件统一指派属性；对应 Abaqus 的 Section Assignment。"""
        from copy import deepcopy

        try:
            ids = self._expand_members(member_ids)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "至少选择一根要指派的杆件"})
        sections = {str(item["name"]) for item in self.model.get("sections") or []}
        materials = {str(item["name"]) for item in self.model.get("materials") or []}
        if section not in sections:
            return ToolResult(False, {"error": f"截面 {section!r} 未定义"})
        if material not in materials:
            return ToolResult(False, {"error": f"材料 {material!r} 未定义"})
        known = {int(item["id"]) for item in self.model.get("members") or []}
        missing = sorted(set(ids) - known)
        if missing:
            return ToolResult(False, {"error": f"杆件 {missing} 不存在"})

        candidate = deepcopy(self.model)
        changed = []
        selected = set(ids)
        for member in candidate.get("members") or []:
            if int(member["id"]) not in selected:
                continue
            if (member.get("section") != section
                    or member.get("material") != material):
                member["section"] = section
                member["material"] = material
                changed.append(int(member["id"]))
        if not changed:
            return ToolResult(True, {"members": sorted(selected), "no_change": True,
                                     "analysis_ready": not validate_payload(candidate)})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "members": changed, "section": section, "material": material,
            "analysis_ready": not errors, "warnings": errors,
            "summary": describe(self.model),
        })

    @_records
    def remove_members(self, ids) -> ToolResult:
        from copy import deepcopy

        if not self.model.get("members"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            ids = self._expand_members(ids)     # 允许直接写集合名
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        try:
            drop = {int(k) for k in ids}
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "杆件编号必须是整数列表或集合名"})
        if not drop:
            return ToolResult(False, {"error": "至少选择一个要删除的杆件"})
        known = {int(m["id"]) for m in self.model["members"]}
        missing = sorted(drop - known)
        if missing:
            return ToolResult(False, {"error": f"杆件 {missing} 不存在"})
        candidate = deepcopy(self.model)
        candidate["members"] = [m for m in candidate["members"]
                                if int(m["id"]) not in drop]

        # 清理走 changes.apply，**不在这里手写**。
        #
        # 原来这里按记忆逐条清，只过滤了 member_loads——而荷载有四类
        # （nodal_loads / member_loads / member_spans / settlements），
        # 而且顶层和 load_cases 两处都能写。自重恰好写的是 member_spans，
        # 于是"加自重 → 删一根柱"会留下指向已删杆件的荷载，
        # 校验在下一步才报错，指向的却不是删除这一步。
        report = _changes.apply(candidate, _changes.Change.GEOMETRY,
                                dropped_members=drop)

        errors = validate_payload(candidate)
        payload = {"removed": sorted(drop),
                   "removed_orphan_nodes": report.get("removed_orphan_nodes", []),
                   "summary": describe(candidate), **report}
        self.model = candidate
        self._invalidate()
        payload["analysis_ready"] = not errors
        payload["warnings"] = errors
        bits = []
        if report.get("removed_orphan_nodes"):
            bits.append(f"清除了 {len(report['removed_orphan_nodes'])} 个孤立节点"
                        "（留着会让刚度矩阵出现整行零，求解报奇异）")
        if report.get("pruned_loads"):
            total = sum(report["pruned_loads"].values())
            bits.append(f"清除了 {total} 条指向已删对象的荷载")
        if report.get("emptied_sets"):
            bits.append(f"集合 {'、'.join(report['emptied_sets'])} 已被删空，一并移除")
        if bits:
            payload["note"] = "顺带" + "；".join(bits) + "。"
        return ToolResult(True, payload)

    @_records
    def remove_nodes(self, ids) -> ToolResult:
        """删除节点。连接到这些节点的杆件会一并删除（否则会留下指向不存在节点的杆）。"""
        from copy import deepcopy

        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        if isinstance(ids, (str, int)):
            ids = [ids]
        try:
            drop = {int(k) for k in ids}
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "节点编号必须是整数列表"})
        if not drop:
            return ToolResult(False, {"error": "至少选择一个要删除的节点"})
        known = {int(n["id"]) for n in self.model["nodes"]}
        missing = sorted(drop - known)
        if missing:
            return ToolResult(False, {"error": f"节点 {missing} 不存在"})
        candidate = deepcopy(self.model)
        connected = [int(m["id"]) for m in candidate.get("members", [])
                     if int(m["i"]) in drop or int(m["j"]) in drop]
        connected_set = set(connected)
        candidate["members"] = [m for m in candidate.get("members", [])
                                if int(m["id"]) not in connected_set]
        candidate["nodes"] = [n for n in candidate["nodes"]
                              if int(n["id"]) not in drop]
        candidate["supports"] = [s for s in candidate.get("supports", [])
                                  if int(s["node"]) not in drop]
        report = _changes.apply(
            candidate, _changes.Change.GEOMETRY,
            dropped_members=connected_set, dropped_nodes=drop)
        errors = validate_payload(candidate)
        payload = {"removed": sorted(drop), "removed_connected_members": connected,
                   "removed_orphan_nodes": report.get("removed_orphan_nodes", []),
                   "summary": describe(candidate), **report}
        self.model = candidate
        self._invalidate()
        payload["analysis_ready"] = not errors
        payload["warnings"] = errors
        return ToolResult(True, payload)
