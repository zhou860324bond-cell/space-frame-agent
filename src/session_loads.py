"""Session 的荷载域：工况、节点荷载、杆件荷载、强制位移、初应变、自重。

从 agent.py 搬出来的，代码一个字节没改。

单独成文件的理由：荷载是唯一一处"工况"这个概念贯穿始终的地方，而工况的
重复与覆盖语义（同名工况怎么合、自重反复调用怎么替换）正是这个项目踩过坑
的地方——SELF_WEIGHT_NOTE 那个标记就是为此存在的。放在一起才看得出这套
规则是一致的。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frame3d import self_weight_loads
from model_io import from_dict, validate_payload
from session_base import SELF_WEIGHT_NOTE, ToolResult, _records


class LoadsMixin:
    """荷载：工况、节点/杆件荷载、位移与初应变。见模块 docstring。"""
    @_records
    def set_load_cases(self, cases: list[dict], combos: list[dict] | None = None) -> ToolResult:
        """一次性定义全部荷载工况与组合，覆盖此前的荷载定义。

        多工况务必用这个，不要把几个工况的荷载先加起来当单一工况——
        那样组合结果虽对，却拿不到各工况的分项内力，也无法做包络。
        """
        if not cases:
            return ToolResult(False, {"error": "至少要给一个工况"})
        from copy import deepcopy

        candidate = deepcopy(self.model)
        for key in ("nodal_loads", "member_loads", "member_spans", "settlements"):
            candidate.pop(key, None)
        candidate["load_cases"] = deepcopy(cases)
        candidate["combos"] = deepcopy(combos or [])
        errors = validate_payload(candidate)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "荷载定义未写入，原模型保持不变"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"cases": [c.get("name") for c in cases],
                                 "combos": [c.get("name") for c in (combos or [])]})

    @_records
    def add_load_case(self, name: str) -> ToolResult:
        """新建一个空荷载工况。"""
        from copy import deepcopy

        name = str(name).strip()
        if not name:
            return ToolResult(False, {"error": "工况名称不能为空"})
        candidate = deepcopy(self.model)
        candidate.setdefault("load_cases", [])
        if any(c.get("name") == name for c in candidate["load_cases"]):
            return ToolResult(False, {"error": f"工况 {name!r} 已存在"})
        candidate["load_cases"].append({"name": name, "nodal_loads": [],
                                          "member_loads": [], "member_spans": [],
                                          "settlements": []})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"case": name, "cases": [c["name"] for c in self.model["load_cases"]]})

    @_records
    def set_nodal_load(self, node_id: int, load: list[float],
                       case_name: str | None = None,
                       name: str | None = None) -> ToolResult:
        """设置节点集中力。load 是 [Fx,Fy,Fz,Mx,My,Mz]，单位为 N 与 N·当前长度单位。
        传全零列表表示删除该节点的荷载。"""
        from copy import deepcopy

        if len(load) != 6:
            return ToolResult(False, {"error": "load 必须是六个数 [Fx,Fy,Fz,Mx,My,Mz]"})
        try:
            values = [float(value) for value in load]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "集中力六个分量必须都是数字"})
        if not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "集中力分量必须是有限数"})
        node_id = int(node_id)
        if node_id not in {int(n["id"]) for n in self.model.get("nodes", [])}:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})
        candidate = deepcopy(self.model)
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                         "member_loads": [], "member_spans": [],
                                         "settlements": []}]
        case_name = case_name or candidate["load_cases"][0]["name"]
        case = next((c for c in candidate["load_cases"] if c["name"] == case_name), None)
        if case is None:
            return ToolResult(False, {"error": f"工况 {case_name!r} 不存在"})
        same_target = [entry for entry in case.get("nodal_loads", [])
                       if int(entry["node"]) == node_id]
        if name is None and len(same_target) > 1:
            return ToolResult(False, {
                "error": f"节点 {node_id} 上有多条载荷，请给出要编辑的载荷名称",
                "names": [entry.get("name") for entry in same_target]})
        old = same_target[0] if same_target else None
        load_name = str(name or (old or {}).get("name") or
                        f"CF-Node-{node_id}").strip()
        if not load_name:
            return ToolResult(False, {"error": "载荷名称不能为空"})
        entries = [entry for entry in case.get("nodal_loads", [])
                   if not (entry.get("name") == load_name
                           or (old is entry and not entry.get("name")))]
        for key in ("member_loads", "member_spans"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") != load_name]
        if any(value != 0 for value in values):
            entries.append({"name": load_name, "node": node_id, "load": values})
        case["nodal_loads"] = entries
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": load_name,
                                 "node": node_id, "case": case_name, "load": values})

    @_records
    def set_member_load(self, member_id: int, load: list[float],
                        case_name: str | None = None,
                        name: str | None = None) -> ToolResult:
        """设置杆件均布荷载。load 是三个数 [wx,wy,wz]，单位为 N/当前长度单位。
        传全零列表表示删除该杆件的荷载。"""
        from copy import deepcopy

        if len(load) != 3:
            return ToolResult(False, {"error": "load 必须是三个数 [wx,wy,wz]"})
        try:
            values = [float(value) for value in load]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "均布荷载三个分量必须都是数字"})
        if not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "均布荷载分量必须是有限数"})
        member_id = int(member_id)
        if member_id not in {int(m["id"]) for m in self.model.get("members", [])}:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})
        candidate = deepcopy(self.model)
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                         "member_loads": [], "member_spans": [],
                                         "settlements": []}]
        case_name = case_name or candidate["load_cases"][0]["name"]
        case = next((c for c in candidate["load_cases"] if c["name"] == case_name), None)
        if case is None:
            return ToolResult(False, {"error": f"工况 {case_name!r} 不存在"})
        same_target = [entry for entry in case.get("member_loads", [])
                       if int(entry["member"]) == member_id]
        if name is None and len(same_target) > 1:
            return ToolResult(False, {
                "error": f"杆件 {member_id} 上有多条均布荷载，请给出要编辑的载荷名称",
                "names": [entry.get("name") for entry in same_target]})
        old = same_target[0] if same_target else None
        load_name = str(name or (old or {}).get("name") or
                        f"Line-Member-{member_id}").strip()
        if not load_name:
            return ToolResult(False, {"error": "载荷名称不能为空"})
        entries = [entry for entry in case.get("member_loads", [])
                   if not (entry.get("name") == load_name
                           or (old is entry and not entry.get("name")))]
        for key in ("nodal_loads", "member_spans"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") != load_name]
        if any(value != 0 for value in values):
            entries.append({"name": load_name, "member": member_id, "w": values})
        case["member_loads"] = entries
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": load_name,
                                 "member": member_id, "case": case_name, "load": values})

    def _load_case_in(self, candidate: dict[str, Any],
                      case_name: str | None) -> tuple[dict[str, Any] | None, str]:
        """在候选模型中取分析步；增量载荷工具共用，避免默认工况逻辑漂移。"""
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                          "member_loads": [], "member_spans": [],
                                          "settlements": []}]
        chosen = case_name or candidate["load_cases"][0]["name"]
        case = next((item for item in candidate["load_cases"]
                     if item.get("name") == chosen), None)
        return case, str(chosen)

    @_records
    def set_member_span_load(self, member_id: int, kind: str, w1: list[float],
                             w2: list[float] | None = None,
                             a: float | None = None,
                             case_name: str | None = None,
                             name: str | None = None) -> ToolResult:
        """创建梯形、三角形或杆中集中力，按名称编辑而不是重复叠加。"""
        from copy import deepcopy

        kind = str(kind)
        if kind not in {"uniform", "trapezoid", "point"}:
            return ToolResult(False, {"error": "kind 只能是 uniform / trapezoid / point"})
        try:
            start = [float(value) for value in w1]
            end = [float(value) for value in (w2 if w2 is not None else w1)]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "w1/w2 必须是三个数字"})
        if len(start) != 3 or len(end) != 3 or not all(
                np.isfinite(value) for value in start + end):
            return ToolResult(False, {"error": "w1/w2 必须是三个有限数字"})
        member_id = int(member_id)
        member = next((item for item in self.model.get("members") or []
                       if int(item["id"]) == member_id), None)
        if member is None:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})
        position = None
        if kind == "point":
            try:
                position = float(a)
            except (TypeError, ValueError):
                return ToolResult(False, {"error": "杆中集中力必须给出距 i 端位置 a"})
            nodes = {int(node["id"]): node for node in self.model.get("nodes") or []}
            ni, nj = nodes[int(member["i"])], nodes[int(member["j"])]
            length = float(np.linalg.norm(np.array(
                [nj[key] - ni[key] for key in ("x", "y", "z")], dtype=float)))
            if not np.isfinite(position) or not 0 <= position <= length:
                return ToolResult(False, {
                    "error": f"位置 a 必须在 0 到杆长 {length:g} 之间（当前模型长度单位）"})
        load_name = str(name or f"{kind.title()}-Member-{member_id}").strip()
        if not load_name:
            return ToolResult(False, {"error": "载荷名称不能为空"})

        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})
        entries = [entry for entry in case.get("member_spans", [])
                   if entry.get("name") != load_name]
        for key in ("nodal_loads", "member_loads"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") != load_name]
        values_for_delete = start + (end if kind == "trapezoid" else [])
        if any(value != 0 for value in values_for_delete):
            entry: dict[str, Any] = {"name": load_name, "member": member_id,
                                     "kind": kind, "w1": start}
            if kind == "trapezoid":
                entry["w2"] = end
            if kind == "point":
                entry["a"] = position
            entries.append(entry)
        case["member_spans"] = entries
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": load_name, "member": member_id,
                                 "case": chosen, "kind": kind,
                                 "analysis_ready": not errors, "warnings": errors})

    @_records
    def set_prescribed_displacement(self, node_id: int, d: list[float],
                                    case_name: str | None = None,
                                    name: str | None = None) -> ToolResult:
        """创建分析步内给定位移；只有 Initial 中已约束的方向会参与求解。"""
        from copy import deepcopy

        try:
            values = [float(value) for value in d]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "d 必须是六个数字"})
        if len(values) != 6 or not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "d 必须是六个有限数字"})
        node_id = int(node_id)
        if node_id not in {int(node["id"]) for node in self.model.get("nodes") or []}:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})
        bc_name = str(name or f"Displacement-Node-{node_id}").strip()
        if not bc_name:
            return ToolResult(False, {"error": "边界条件名称不能为空"})
        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})
        entries = [entry for entry in case.get("settlements", [])
                   if entry.get("name") != bc_name and int(entry["node"]) != node_id]
        if any(value != 0 for value in values):
            entries.append({"name": bc_name, "node": node_id, "d": values})
        case["settlements"] = entries
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": bc_name, "node": node_id,
                                 "case": chosen, "d": values,
                                 "analysis_ready": not errors, "warnings": errors,
                                 "note": "给定位移属于分析步边界条件，不计入外力。"})

    @_records
    def set_member_strain(self, member_id: int,
                          lack_of_fit: float | None = None,
                          delta_t: float | None = None,
                          case_name: str | None = None,
                          name: str | None = None) -> ToolResult:
        """装配误差与温度变化（讲义 §3-9 五、六）。

        讲义把温度应力"转化为装配内力问题"，这里就照这个思路做成一个入口：
        两者都只是给杆件一个初始长度变化，差别仅在于 Δl 是直接给的还是由
        α·ΔT 算出来的。
        """
        from copy import deepcopy

        member_id = int(member_id)
        members = {int(m["id"]): m for m in self.model.get("members") or []}
        if member_id not in members:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})
        for label, value in (("lack_of_fit", lack_of_fit), ("delta_t", delta_t)):
            if value is not None and not np.isfinite(float(value)):
                return ToolResult(False, {"error": f"{label} 必须是有限数"})
        if delta_t:
            material = next(
                (item for item in self.model.get("materials") or []
                 if item["name"] == members[member_id]["material"]), None)
            if not (material or {}).get("alpha"):
                return ToolResult(False, {
                    "error": f"材料 {members[member_id]['material']!r} 没有线膨胀系数 alpha，"
                             "算不了温度应力",
                    "hint": "在 define_materials_and_sections 里给该材料补 alpha（1/℃），"
                            "钢约 1.2e-5"})

        entry_name = str(name or f"Strain-Member-{member_id}").strip()
        if not entry_name:
            return ToolResult(False, {"error": "名称不能为空"})
        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})
        entries = [entry for entry in case.get("member_strains", [])
                   if entry.get("name") != entry_name
                   and int(entry["member"]) != member_id]
        payload = {"name": entry_name, "member": member_id}
        if lack_of_fit:
            payload["lack_of_fit"] = float(lack_of_fit)
        if delta_t:
            payload["delta_t"] = float(delta_t)
        if len(payload) > 2:
            entries.append(payload)
        case["member_strains"] = entries
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "name": entry_name, "member": member_id, "case": chosen,
            "lack_of_fit": lack_of_fit, "delta_t": delta_t,
            "analysis_ready": not errors, "warnings": errors,
            "note": "初应变不是外力：杆件能自由伸缩时轴力为零，"
                    "被约束住才产生内力。"})

    @_records
    def add_self_weight(self, case: str | None = None,
                        factor: float = 1.0) -> ToolResult:
        """按 ρ·A·g 生成各杆自重，作为均布荷载追加到某个工况。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "模型不合法，先修好再加自重"})
        frame = from_dict(self.model)
        loads = self_weight_loads(frame, factor=factor)
        if not loads:
            missing = sorted({m["material"] for m in self.model["members"]})
            return ToolResult(False, {
                "error": "没有一种材料定义了 density，自重为零",
                "hint": f"给材料 {missing} 加上 density（kg/m³，钢约 7850）"
                        "再调一次；define_materials_and_sections 可以带这个字段"})
        cases = self.model.setdefault("load_cases", [])
        if not cases:
            cases.append({"name": case or "SW"})
        name = case or cases[0]["name"]
        target = next((c for c in cases if c.get("name") == name), None)
        if target is None:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况",
                                      "available": [c.get("name") for c in cases]})
        spans = target.setdefault("member_spans", [])
        # 反复调用会把自重叠加好几遍，先把上一次生成的清掉
        kept = [e for e in spans if e.get("note") != SELF_WEIGHT_NOTE]
        replaced = len(spans) - len(kept)
        spans[:] = kept
        for mid, load in sorted(loads.items()):
            spans.append({"member": mid, **load.to_dict(),
                          "note": SELF_WEIGHT_NOTE})
        self._invalidate()
        total = sum(abs(load.w1[2]) for load in loads.values())
        return ToolResult(True, {
            "case": name, "members": len(loads),
            "replaced_previous": replaced,
            "total_vertical_kN_per_m": round(total * self.units.line_load_scale, 6),
            "note": "自重已写成显式均布荷载，可在模型里看到。改截面后需重新调用。"})
