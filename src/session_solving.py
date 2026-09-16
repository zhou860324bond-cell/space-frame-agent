"""Session 的求解域：自研内核求解、Abaqus 三段链路、节点局部实体子模型，
以及三者的横向对比。

从 agent.py 搬出来的，代码一个字节没改。

这一组的共同点是"会真的去算"：每个方法都要先编译 Domain IR、再交给某个
求解器、最后把结果收进 result_db。算前守门（先 validate 再算）是 agent.py
开头设计原则的第 3 条，solve_model 里那一段就是它。

求解后自动跑的两项质检——静默失败检测与实验胶囊存档——失败不影响求解
结果本身，所以都用 try/except 兜住、只记警告。别把它们改成硬失败。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from frame3d import check_equilibrium, solve
from model_compiler import compile_model
from model_io import validate_payload
from result_db import from_solution as result_db_from_solution
from session_base import ToolResult
import capsule as _capsule
import silent_failures as _silent


class SolvingMixin:
    """求解：自研内核、Abaqus 对标、实体子模型。见模块 docstring。"""
    def solve_model(self, analysis: str = "linear", increments: int = 10,
                    max_iter: int = 40, tolerance: float = 1e-7) -> ToolResult:
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "先按上面的清单修正模型，再重新求解"})
        self.compilation = compile_model(self.model)
        self.frame = self.compilation.analysis_model
        self.result_db = None
        try:
            if analysis == "linear":
                self.solution = solve(self.frame)
            elif analysis == "pdelta":
                from nonlinear import solve_pdelta
                self.solution = solve_pdelta(
                    self.frame, increments=increments, max_iter=max_iter,
                    tolerance=tolerance)
            elif analysis == "material":
                from nonlinear import solve_material_nonlinear
                self.solution = solve_material_nonlinear(
                    self.frame, increments=increments, max_iter=max_iter,
                    tolerance=tolerance)
            else:
                self.solution = None
                return ToolResult(False, {"error": f"未知分析类型 {analysis!r}"})
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            self.solution = None
            self.result_db = None
            payload: dict[str, Any] = {"error": str(exc)}
            diag = self.diagnose_supports()       # 直接把诊断带上，不必再调一次
            if diag.ok and diag.payload.get("modes"):
                payload["diagnosis"] = diag.payload["modes"]
                payload["hint"] = ("诊断已在上面的 diagnosis 里给出，不必再调 diagnose_supports。"
                                   "补上约束再求解，或向用户说明缺什么。注意三维梁两端都放开 rx "
                                   "会产生绕杆轴的扭转机构，至少要约束一端的 rx。")
            else:
                payload["hint"] = "检查约束是否足以消除全部刚体位移"
            return ToolResult(False, payload)
        self.result_db = result_db_from_solution(
            self.compilation.source_ir, self.frame, self.solution,
            self.compilation.mapping)
        span = self._reference_length()
        U = self.units
        results = {}
        for name, res in self.solution.all_results().items():
            node, mag = self._max_displacement(res)
            eq = check_equilibrium(self.frame, self.solution, name)
            entry = {"max_displacement_mm": round(mag * U.disp_scale, 6),
                     "at_node": node,
                     "equilibrium_ok": eq["ok"],
                     "equilibrium_residual": float(f"{eq['relative']:.3e}")}
            if mag <= 1e-12 and self._applied_load_magnitude(name) > 0.0:
                entry["warning"] = (
                    "有荷载但位移为零。常见原因：荷载全部作用在被约束死的方向上——"
                    "例如给平面刚架（bays 留空）加了面外荷载，"
                    "平面刚架的面外自由度是被自动约束的。检查荷载方向。")
            elif span > 0 and mag > span / 200.0:
                entry["warning"] = (
                    f"最大位移达到最短杆件长度的 1/{max(1, int(span / mag))}，量级异常。"
                    "常见原因：荷载方向写错（重力应为全局 -Z，即 w=[0,0,-w]）、"
                    "单位没换算成 N-m-Pa、或截面惯性矩填小了。")
            results[name] = entry

        # 求解成功后自动跑两项质检：静默失败检测 + 实验胶囊存档。
        # 这两项是"附加价值"，失败不影响求解结果本身——用 try/except 兜住，
        # 只在 payload 里记警告，不让主流程红。
        payload: dict[str, Any] = {
            "cases": results,
            "analysis": dict(self.solution.analysis),
            "compilation": {
                "physical_members": len(
                    self.compilation.mapping.physical_to_elements),
                "analysis_elements": len(
                    self.compilation.mapping.element_to_physical),
                "generated_nodes": len(self.compilation.mapping.generated_nodes),
                "diagnostics": list(self.compilation.diagnostics),
            },
        }

        # 1. 静默失败检测：8 项检查，把"算出来了但结果可疑"的情况挑出来
        try:
            findings = _silent.detect_silent_failures(self.frame, self.solution)
            summary = _silent.summarize_findings(findings)
            payload["silent_failures"] = {"summary": summary, "findings": findings}
        except Exception as exc:                      # noqa: BLE001
            payload["silent_failures"] = {
                "summary": {"status": "error",
                            "error": f"{type(exc).__name__}: {exc}"},
                "findings": [],
            }

        # 2. 实验胶囊存档：完整输入+结果摘要，便于复现和回归对比
        try:
            cap_path = _capsule.save_capsule(
                self.model, self.frame, self.solution, source="solve_model")
            payload["capsule"] = str(cap_path)
        except Exception as exc:                      # noqa: BLE001
            payload["capsule_error"] = f"{type(exc).__name__}: {exc}"

        # 判定为「结果不可用」的静默失败要让 ok=False。
        #
        # 理由不是"critical 就该失败"，而是**自修复闭环靠 ok 触发**：系统提示
        # 第 4 条写着"solve_model 返回错误清单时，读懂它、改模型、重试"。
        # 以前这些发现只躺在 payload 里，Agent 照样把一套全零结果如实汇报给
        # 用户——可溯源，但是个废数。
        #
        # 不是所有 critical 都拦：excessive_displacement 的数字是真的，只是
        # 大得可疑，用户可能就是要看它。拦的是「这组数不代表任何受力状态」
        # 的那几条，名单写在 silent_failures.RESULT_INVALIDATING 里。
        blocking = [f for f in payload.get("silent_failures", {}).get("findings", [])
                    if f.get("status") == "fail"
                    and f.get("id") in _silent.RESULT_INVALIDATING]
        if blocking:
            payload["unusable"] = {
                "reason": "静默失败检测判定这组结果不代表任何受力状态",
                "checks": [f["id"] for f in blocking],
                "next": [f.get("suggestion") for f in blocking if f.get("suggestion")],
            }
        return ToolResult(not blocking, payload)

    def solve_with_abaqus(self, case: str | None = None,
                          element: str = "B33") -> ToolResult:
        """用 Abaqus 求解。结果格式与 solve_model 对齐，便于直接比对。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "模型都不合法，换求解器也没用"})
        try:
            import abaqus_backend
        except ImportError as exc:
            return ToolResult(False, {"error": f"Abaqus 后端不可用：{exc}"})
        out_dir = Path("results") / "abaqus"
        try:
            summary = abaqus_backend.solve(self.model, out_dir, case=case,
                                           element=element)
        except KeyError as exc:
            return ToolResult(False, {"error": str(exc)})
        except abaqus_backend.AbaqusError as exc:
            return ToolResult(False, {"error": str(exc),
                                      "hint": "自研求解器 solve_model 不需要 Abaqus，"
                                              "可以改用它"})
        summary["note"] = ("这是 Abaqus 的结果。与 solve_model 的结果比对时请注意单元格式："
                           "B33 与本程序同为 Euler-Bernoulli，误差应到数值精度量级；"
                           "B31 含剪切变形，偏差是格式差异不是错误。")
        return ToolResult(True, summary)

    def analyze_joint_solid(self, node_id: int, case: str | None = None,
                            anchor_member: int | None = None,
                            mesh_sizes_mm: list[float] | None = None,
                            backend: str = "native",
                            dry_run: bool = False) -> ToolResult:
        """节点局部实体子模型：看杆件相交处的应力集中。

        分两段，是有意的。**建规格**（把梁模型的杆端力翻译成实体子模型的
        边界条件、反演圆管尺寸、算名义应力）是纯计算；**跑实体**默认调用
        自研 C3D10 内核，也可切换 Abaqus 对标。dry_run 不是调试开关，而是
        在划网格前先审查几何和载荷映射的入口。
        """
        import solid_joint

        try:
            spec = solid_joint.prepare_joint_spec(
                self, int(node_id), case, anchor_member)
        except solid_joint.SolidJointError as exc:
            return ToolResult(False, {"error": str(exc)})
        except (KeyError, ValueError) as exc:
            return ToolResult(False, {"error": f"节点局部实体建模失败：{exc}"})

        if dry_run:
            payload = spec.to_dict()
            payload["note"] = ("只建了规格，没有调用 Abaqus。名义正应力即 Kt 的分母，"
                               "可以先手算核对再决定是否跑实体分析。")
            return ToolResult(True, payload)
        try:
            if backend == "native":
                import native_joint
                summary = native_joint.run_native_joint_analysis(
                    self, int(node_id), case, anchor_member, mesh_sizes_mm)
            elif backend == "abaqus":
                summary = solid_joint.run_joint_analysis(
                    self, int(node_id), case, anchor_member, mesh_sizes_mm)
            else:
                return ToolResult(False, {"error": f"未知实体求解后端 {backend!r}"})
        except solid_joint.SolidJointError as exc:
            return ToolResult(False, {
                "error": str(exc),
                "hint": "可以先用 dry_run=true 看规格；或切换 native/abaqus 后端定位问题",
            })
        return ToolResult(True, summary)

    def _native_rows(self, case: str) -> dict[int, dict[str, float]]:
        """把自研结果摊成与 Abaqus CSV 同一套字段，才能逐分量比。

        单位取当前模型的原生单位（m 或 mm、rad、N），两边同源，不做缩放。
        """
        res = self.solution[case]
        rows: dict[int, dict[str, float]] = {}
        keys = ("u1", "u2", "u3", "ur1", "ur2", "ur3")
        for nid in sorted(self.frame.nodes):
            d = self.frame.node_dofs(nid)
            row = {k: float(res.U[d[i]]) for i, k in enumerate(keys)}
            # 非支座节点反力恒为零，写进去让两侧字段对齐
            for i, k in enumerate(("rf1", "rf2", "rf3")):
                row[k] = float(res.R[d[i]]) if nid in self.frame.supports else 0.0
            rows[nid] = row
        return rows

    def compare_solvers(self, case: str | None = None,
                        element: str = "B33") -> ToolResult:
        """自研 + Abaqus 各算一遍，逐分量给归一化偏差。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "模型都不合法，比对没有意义"})
        native = self.solve_model()
        if not native.ok:
            return ToolResult(False, {"error": "自研求解器先失败了",
                                      "detail": native.payload})
        name = case or self.model["load_cases"][0]["name"]
        if self.solution is None or name not in set(self.solution.all_results()):
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况"})
        ab = self.solve_with_abaqus(case=name, element=element)
        if not ab.ok:
            return ToolResult(False, {"error": "Abaqus 侧失败，无法比对",
                                      "detail": ab.payload})
        try:
            import abaqus_backend
            rows_ab = abaqus_backend.read_result_csv(Path(ab.payload["files"]["csv"]))
            cmp = abaqus_backend.compare_rows(
                self._native_rows(name), rows_ab,
                displacement_scale=self.units.disp_scale)
        except Exception as exc:  # noqa: BLE001 — 比对失败不该让整轮对话崩掉
            return ToolResult(False, {"error": f"读取 Abaqus 结果失败：{exc}"})
        if "error" in cmp:
            return ToolResult(False, {"error": cmp["error"]})
        worst = max((v["e"] for v in cmp["errors"].values() if v["e"] is not None),
                    default=None)
        return ToolResult(True, {
            "case": name, "element": element,
            "nodes_compared": cmp["nodes_compared"],
            "errors": {k: (None if v["e"] is None else float(f"{v['e']:.3e}"))
                       for k, v in cmp["errors"].items()},
            "worst_error": None if worst is None else float(f"{worst:.3e}"),
            "peak_displacement": cmp["peak"],
            "metric": "全场归一化相对误差 sqrt(Σ(a−b)²)/sqrt(Σb²)，以 Abaqus 为参考；"
                      "该分量参考解整体为零时记 null（归一化无意义），不是没算",
            "note": ("B33 与本程序同为 Euler-Bernoulli，偏差应在 1e-5 以下；"
                     "B31 含剪切变形，偏差随杆件越粗越大，那是格式差异不是错误。"),
        })
