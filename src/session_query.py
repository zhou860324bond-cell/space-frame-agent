"""Session 的查询域：从算完的结果里取数、出图、做参数扫掠与包络。

从 agent.py 搬出来的，代码一个字节没改。

这一组只读不写：不改 Domain IR，也不重新求解（sweep 例外，它按定义就要
反复求解）。读它的时候可以假定模型和结果都已经就位。

"大模型只产结构，不产数值"这条原则的落点就在这里——报告里每个数字都从
这些方法返回，能溯源到某次工具调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from session_base import ToolResult


class QueryMixin:
    """查询与出图：结果读取、内力图、参数扫掠、包络。见模块 docstring。"""
    def query_results(self, what: str, case: str | None = None,
                      member_id: int | None = None,
                      span: float | None = None) -> ToolResult:
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        name = case or self._controlling_case()
        U = self.units
        if name not in self.result_db.steps:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况或组合",
                                      "available": list(self.result_db.steps)})
        if what == "max_displacement":
            displacements = self.result_db.field(name, "U")
            node = max(displacements.values,
                       key=lambda item: float(np.linalg.norm(displacements.value(item))))
            vector = displacements.value(node)
            mag = float(np.linalg.norm(vector))
            n = self.frame.nodes[node]
            comp = [round(float(value) * U.disp_scale, 6) for value in vector]
            payload: dict[str, Any] = {
                "case": name, "node": node,
                "coordinates_xyz": [round(v * U.length_to_m, 6)
                                    for v in (n.x, n.y, n.z)],
                "coordinate_unit": "m",
                "magnitude_mm": round(mag * U.disp_scale, 6),
                "components_mm": {"ux": comp[0], "uy": comp[1], "uz": comp[2]},
                "note": "位置用上面的坐标描述，方向用 components_mm 判断，"
                        "两者都不要靠节点编号推测",
            }
            origin = self.compilation.mapping.generated_nodes.get(node)
            if origin is not None:
                payload.update({"node_kind": "generated_analysis_node",
                                "physical_member": origin[0],
                                "member_s": round(origin[1], 9)})
            else:
                payload["node_kind"] = "physical_node"
            return ToolResult(True, payload)
        if what == "max_deflection":
            try:
                from internal_forces import max_deflection
            except ImportError as exc:
                return ToolResult(False, {"error": f"内力模块不可用：{exc}"})
            # 梯形积分的误差随测点数二阶下降。101 点时单跨一个单元差 0.016%，
            # 201 点降到 0.004%——够了，再加只是烧时间
            best = max_deflection(
                self.frame, self.result_db.solution_view(self.frame), name,
                stations=201,
                mapping=self.compilation.mapping)
            if best["member"] is None:
                return ToolResult(False, {"error": "模型里没有杆件"})
            length = float(best["length"])
            value = abs(best["value"])
            payload: dict[str, Any] = {
                "case": name,
                "magnitude_mm": round(value * U.disp_scale, 6),
                "at_member": best["member"],
                "at_x_m": round(best["x"] * U.length_to_m, 4),
                "member_length_m": round(length * U.length_to_m, 4),
                "member_length_over_deflection": (round(length / value, 1)
                                                  if value > 0 else None),
            }
            if span is not None:
                payload["span_m"] = float(span)
                payload["span_over_deflection"] = (round(float(span) /
                                                          (value * U.length_to_m), 1)
                                                   if value > 0 else None)
            payload["note"] = (
                "这是**单元内**的最大横向挠度，由精确弯矩两次积分得到，与网格疏密无关。"
                "校核挠跨比用它，不要用 max_displacement——后者只看节点。"
                "注意 member_length_over_deflection 的分母是**这一根杆件的长度**，"
                "不是设计跨度：门式刚架的斜梁是两根杆，梁划成几个单元时更是差几倍。"
                "要和 L/400 这类限值比，请把设计跨度作为 span 传进来，"
                "工具会给出 span_over_deflection。")
            return ToolResult(True, payload)

        if what == "reactions":
            reactions = self.result_db.field(name, "RF")
            out = {}
            for nid in sorted(self.frame.supports):
                n = self.frame.nodes[nid]
                out[str(nid)] = {"xyz": [round(v * U.length_to_m, 6)
                                         for v in (n.x, n.y, n.z)],
                                 "R": [round(float(value) * U.force_scale, 6)
                                       for value in reactions.value(nid)]}
            total = U.force_scale * sum(float(reactions.value(nid)[2])
                                        for nid in self.frame.supports)
            return ToolResult(True, {"case": name, "unit": "kN",
                                     "coordinate_unit": "m",
                                     "reactions": out,
                                     "vertical_total_kN": round(total, 6)})
        if what == "member_forces":
            physical_ids = self.result_db.physical_to_elements
            if member_id is None:
                # 不给编号就一次返回全部，省得一根根查
                rows = {}
                for mid in sorted(physical_ids):
                    axial = self.result_db.physical_member_value(name, "N", mid)
                    mz = self.result_db.physical_member_value(name, "Mz", mid)
                    # 保留位数与 query_diagram 一致：同一个量在两处对不上，
                    # 模型会以为是两个不同的结果
                    rows[str(mid)] = {
                        "N": round(float(axial[1]) * U.force_scale, 6),
                        "Mz_i": round(float(mz[0]) * U.moment_scale, 6),
                        "Mz_j": round(float(mz[1]) * U.moment_scale, 6)}
                return ToolResult(True, {"case": name, "unit": "kN, kN·m",
                                         "sign_convention": "N 以受拉为正，受压为负",
                                         "all_members": rows})
            if member_id not in physical_ids:
                return ToolResult(False, {"error": f"没有编号为 {member_id} 的杆件"})
            axial = self.result_db.physical_member_value(name, "N", member_id)
            my = self.result_db.physical_member_value(name, "My", member_id)
            mz = self.result_db.physical_member_value(name, "Mz", member_id)
            return ToolResult(True, {"case": name, "member": member_id, "unit": "kN, kN·m",
                                     "sign_convention": "N 以受拉为正，受压为负",
                                     "N": round(float(axial[1]) * U.force_scale, 6),
                                     "Mz_i": round(float(mz[0]) * U.moment_scale, 6),
                                     "Mz_j": round(float(mz[1]) * U.moment_scale, 6),
                                     "My_i": round(float(my[0]) * U.moment_scale, 6),
                                     "My_j": round(float(my[1]) * U.moment_scale, 6)})
        return ToolResult(False, {"error": f"不支持的查询 {what!r}"})

    def plot_results(self, kind: str, case: str | None = None) -> ToolResult:
        """出图。只回路径与摘要，不回图像——结论仍须引用 query_results 的数字。"""
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        try:
            from plot3d import plot_axial, plot_deformed, plot_diagram
        except ImportError as exc:
            return ToolResult(False, {"error": f"绘图依赖缺失：{exc}"})
        name = case or self._controlling_case()
        if name not in self.result_db.steps:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况或组合",
                                      "available": list(self.result_db.steps)})
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        result_view = self.result_db.solution_view(self.frame)
        try:
            if kind == "deformed":
                info = plot_deformed(self.frame, result_view, name,
                                     out_dir / f"deformed_{safe}.png",
                                     mapping=self.compilation.mapping)
            elif kind == "axial":
                info = plot_axial(self.frame, result_view, name,
                                  out_dir / f"axial_{safe}.png",
                                  mapping=self.compilation.mapping)
            elif kind in {"moment", "shear"}:
                comp = "Mz" if kind == "moment" else "Vy"
                info = plot_diagram(self.frame, result_view, comp, name,
                                    out_dir / f"{kind}_{safe}.png",
                                    mapping=self.compilation.mapping)
            else:
                return ToolResult(False, {"error": f"不支持的图类型 {kind!r}"})
        except Exception as exc:
            return ToolResult(False, {"error": f"绘图失败：{type(exc).__name__}: {exc}"})
        info["note"] = "图已保存供用户查看；引用数字请用 query_results，不要从图上读数"
        return ToolResult(True, info)

    def query_diagram(self, component: str, member: int | None = None,
                      case: str | None = None,
                      stations: int | None = None) -> ToolResult:
        """沿杆长的内力。解析恢复，与网格无关。"""
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        try:
            from internal_forces import COMPONENTS, physical_member_diagram
        except ImportError as exc:
            return ToolResult(False, {"error": f"内力模块不可用：{exc}"})
        if component not in COMPONENTS:
            return ToolResult(False, {"error": f"内力分量只能取 {list(COMPONENTS)}"})
        name = case or self._controlling_case()
        if name not in self.result_db.steps:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况或组合",
                                      "available": list(self.result_db.steps)})
        U = self.units
        moment = component in {"T", "My", "Mz"}
        unit = U.moment_unit if moment else U.force_unit
        scale = U.moment_scale if moment else U.force_scale
        result_view = self.result_db.solution_view(self.frame)

        if member is None:
            worst = None
            for physical_id in sorted(self.compilation.mapping.physical_to_elements):
                diagram = physical_member_diagram(
                    self.frame, result_view, self.compilation.mapping,
                    physical_id, name, stations=201)
                x_at, value = diagram.extreme(component)
                if worst is None or abs(value) > abs(worst["value"]):
                    worst = {"value": value, "member": physical_id, "x": x_at}
            if worst is None:
                return ToolResult(False, {"error": "模型里没有杆件"})
            return ToolResult(True, {
                "case": name, "component": component, "unit": unit,
                "peak": round(worst["value"] * scale, 6),
                "at_member": worst["member"],
                "at_x_m": round(worst["x"] * U.length_to_m, 4),
                "note": "全结构极值。要看某根杆件的完整分布，传 member 与 stations。"})

        if member not in self.compilation.mapping.physical_to_elements:
            return ToolResult(False, {"error": f"没有编号为 {member} 的杆件"})
        d = physical_member_diagram(
            self.frame, result_view, self.compilation.mapping, member, name,
            stations=stations or 21)
        x_at, peak = d.extreme(component)
        payload: dict[str, Any] = {
            "case": name, "member": member, "component": component, "unit": unit,
            "length_m": round(d.length * U.length_to_m, 4),
            "peak": round(peak * scale, 6),
            "at_x_m": round(x_at * U.length_to_m, 4),
            "at_i_end": round(float(d.component(component)[0]) * scale, 6),
            "at_j_end": round(float(d.component(component)[-1]) * scale, 6)}
        if stations:
            payload["x_m"] = [round(float(v) * U.length_to_m, 4) for v in d.x]
            payload["values"] = [round(float(v) * scale, 6)
                                 for v in d.component(component)]
        return ToolResult(True, payload)

    def _sweep_metric(self, metric: str, case: str | None):
        """在当前 self.model 上算一个指标。失败时返回 (None, 原因)。"""
        solved = self.solve_model()
        if not solved.ok:
            return None, "求解失败"
        if metric in ("max_displacement", "max_deflection"):
            r = self.query_results(what=metric, case=case)
            return (r.payload["magnitude_mm"], None) if r.ok else (None, "查询失败")
        if metric == "max_abs_Mz":
            r = self.query_diagram(component="Mz", case=case)
            return (abs(r.payload["peak"]), None) if r.ok else (None, "查询失败")
        if metric == "buckling_factor":
            r = self.buckling_analysis(case=case, num_modes=1)
            return (r.payload["critical_factor"], None) if r.ok else \
                (None, r.payload.get("error", "屈曲失败"))
        if metric == "first_frequency":
            r = self.modal_analysis(num_modes=1)
            return (r.payload["modes"][0]["frequency_Hz"], None) if r.ok else \
                (None, r.payload.get("error", "模态失败"))
        return None, f"不认识的指标 {metric!r}"

    def sweep(self, what: str, target: str, values: list[float], metric: str,
              prop: str | None = None, case: str | None = None,
              limit: float | None = None) -> ToolResult:
        """参数扫描。一次调用跑完 N 个取值，返回对照表与结论。"""
        import copy

        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "还没有模型"})
        if not values or len(values) < 2:
            return ToolResult(False, {"error": "至少要给两个取值才谈得上扫描"})
        if what == "section_property" and prop not in ("A", "Iy", "Iz", "J"):
            return ToolResult(False, {"error": "扫截面特性时 prop 必须是 A/Iy/Iz/J"})

        names = {"section_property": [s["name"] for s in self.model["sections"]],
                 "material_E": [m["name"] for m in self.model["materials"]],
                 "load_scale": [c.get("name")
                                for c in self.model.get("load_cases", [])]}.get(what)
        if names is None:
            return ToolResult(False, {"error": f"不认识的扫描类型 {what!r}"})
        if target not in names:
            return ToolResult(False, {"error": f"没有名为 {target!r} 的对象",
                                      "available": names})

        # 扫描**绝不能动**当前模型：用户扫完还要接着用原模型算别的
        original = copy.deepcopy(self.model)
        original_solution = self.solution
        rows: list[dict] = []
        try:
            for v in values:
                probe = copy.deepcopy(original)
                if what == "section_property":
                    for sec in probe["sections"]:
                        if sec["name"] == target:
                            sec[prop] = float(v)
                elif what == "material_E":
                    for mat in probe["materials"]:
                        if mat["name"] == target:
                            mat["E"] = float(v)
                else:
                    probe.setdefault("combos", [])
                    probe["combos"] = [c for c in probe["combos"]
                                       if c["name"] != "__sweep__"]
                    probe["combos"].append({"name": "__sweep__",
                                            "factors": {target: float(v)}})
                self.model = probe
                self._invalidate()
                use_case = "__sweep__" if what == "load_scale" else case
                got, why = self._sweep_metric(metric, use_case)
                rows.append({"value": float(v),
                             "metric": None if got is None else round(float(got), 6),
                             "failed": why})
        finally:
            # 一定要还原成扫描前的样子：模型、装配好的 frame、以及结果。
            # 只还原 model 不还原 frame/solution 的话，接下来的查询会拿
            # 最后一个探针的结果去回答原模型的问题——不报错，但数全是错的
            self.model = original
            self._invalidate()
            if original_solution is not None:
                self.solve_model()

        good = [r for r in rows if r["metric"] is not None]
        if not good:
            return ToolResult(False, {"error": "每个取值都没算出来",
                                      "rows": rows})

        payload: dict[str, Any] = {
            "what": what, "target": target, "prop": prop, "metric": metric,
            "unit": {"max_displacement": "mm", "max_deflection": "mm",
                     "max_abs_Mz": "kN·m", "buckling_factor": "倍",
                     "first_frequency": "Hz"}[metric],
            "rows": rows,
            "note": "扫描没有改动当前模型；要真正采用某个取值，"
                    "再单独调一次相应的工具把它写进模型。",
        }
        if limit is not None:
            bigger_better = metric in self._SWEEP_BIGGER_IS_BETTER
            ok_rows = [r for r in good
                       if (r["metric"] >= limit if bigger_better
                           else r["metric"] <= limit)]
            payload["limit"] = float(limit)
            payload["satisfying_values"] = [r["value"] for r in ok_rows]
            # 满足限值所需的最省取值。位移/弯矩类是"越大的截面越好"，
            # 所以取满足者里最小的那个值；两类指标方向相反，别写反
            payload["smallest_sufficient"] = (min(r["value"] for r in ok_rows)
                                              if ok_rows else None)
            if not ok_rows:
                payload["hint"] = ("给的这些取值都不满足限值，"
                                   "把范围往有利方向再扩一档再扫一次。")
        return ToolResult(True, payload)

    def query_envelope(self, component: str, member: int | None = None,
                       cases: list[str] | None = None) -> ToolResult:
        """内力包络与控制组合。"""
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        try:
            from envelope import governing_summary, member_envelope
            from internal_forces import COMPONENTS
        except ImportError as exc:
            return ToolResult(False, {"error": f"包络模块不可用：{exc}"})
        if component not in COMPONENTS:
            return ToolResult(False, {"error": f"内力分量只能取 {list(COMPONENTS)}"})
        picked = tuple(cases) if cases else None
        U = self.units
        moment = component in {"T", "My", "Mz"}
        unit = U.moment_unit if moment else U.force_unit
        scale = U.moment_scale if moment else U.force_scale
        result_view = self.result_db.solution_view(self.frame)

        try:
            if member is None:
                s = governing_summary(
                    self.frame, result_view, picked, stations=101,
                    mapping=self.compilation.mapping)
                worst = s["worst"][component]
                return ToolResult(True, {
                    "component": component, "unit": unit,
                    "cases": s["cases"],
                    "peak": round(worst["value"] * scale, 6),
                    "at_member": worst["member"],
                    "at_x_m": round(worst["x"] * U.length_to_m, 4),
                    "governing_case": worst["case"],
                    "controls_global_extreme": s["controls_global_extreme"],
                    "never_governs_anywhere": s["never_governs_anywhere"],
                    "note": "全结构最不利。要看某根杆件的逐点包络，传 member。"
                            "controls_global_extreme 表示该组合拿下了几个分量的"
                            "全结构极值；它为 0 并不等于这个组合没用——"
                            "它完全可能在某根梁的端部说了算。"
                            "真正一处都不控制的在 never_governs_anywhere 里，"
                            "那种才可以考虑删掉，或者检查它的系数是不是写错了。"})
            if member not in self.compilation.mapping.physical_to_elements:
                return ToolResult(False, {"error": f"没有编号为 {member} 的杆件"})
            env = member_envelope(
                self.frame, result_view, member, picked, stations=101,
                mapping=self.compilation.mapping)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})

        worst = env.extreme(component)
        return ToolResult(True, {
            "member": member, "component": component, "unit": unit,
            "cases": list(env.cases),
            "length_m": round(env.length * U.length_to_m, 4),
            "peak": round(worst["value"] * scale, 6),
            "at_x_m": round(worst["x"] * U.length_to_m, 4),
            "governing_case": worst["case"],
            "at_i_end": {"upper": round(float(env.upper[component][0]) * scale, 6),
                         "lower": round(float(env.lower[component][0]) * scale, 6)},
            "at_j_end": {"upper": round(float(env.upper[component][-1]) * scale, 6),
                         "lower": round(float(env.lower[component][-1]) * scale, 6)},
            "cases_that_govern_somewhere": env.governing(component),
            "note": "上下包线是逐点取的。同一根杆上跨中与支座常由不同组合控制，"
                    "所以引用结论时要连位置和组合名一起说。"})
