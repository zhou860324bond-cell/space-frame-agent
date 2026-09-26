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
    def solve_steps(self, inspect: str | None = None) -> ToolResult:
        """按顺序求解模型里声明的全部分析步。

        荷载与边界条件在步之间**传播**：某一步建的东西自动沿用到后面每一步，
        直到被改写或显式失活。所以"满载 → 撤活载 → 拆支座"这样三步，只需要
        在第一步写全荷载，后两步各写一行变化。

        **每一步都从未变形、无应力的状态重解。** 线弹性下这不是问题——撤掉
        支座之后的平衡态就是重解出来的那个。但它表达不了施工过程：后装的杆件
        不会躲开先前的变形，塑性残余也不会留到下一步。杆件同样不能在步之间
        生灭。返回里带着这两条，转达时不要省掉。

        ``inspect`` 指定哪一步的结果留在会话里供后处理（画图、验算、查内力）；
        默认是最后一步。求解本身每一步都做，只是会话一次只能端着一份结果。
        """
        import steps as step_module

        declared = step_module.from_payload(self.model)
        if not declared:
            return ToolResult(False, {
                "error": "模型里没有分析步",
                "hint": "用 add_step 建一个；只想算一次的话直接用 solve_model"})
        try:
            effective = step_module.resolve(declared, self.model)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc),
                                      "hint": "分析步没有执行，模型保持不变"})

        names = [e.name for e in effective]
        target = inspect if inspect is not None else names[-1]
        if target not in names:
            return ToolResult(False, {
                "error": f"没有名为 {target!r} 的分析步", "steps": names})

        original = self.model
        summary = []
        try:
            for eff in effective:
                self.model = step_module.payload_for(eff, original)
                if eff.analysis == "pdelta":
                    outcome = self.solve_model(
                        analysis="step", amplitudes=dict(eff.loads),
                        increments=eff.increments)
                    case_key = "STEP"
                else:
                    # 线性步：各工况按幅值曲线在 t=1 处的系数叠加成一个组合。
                    # 线性叠加在这里是精确的，所以不必走增量求解。
                    factors = self._amplitude_factors(eff.loads, original)
                    self.model = dict(self.model)
                    self.model["combos"] = [{"name": eff.name,
                                             "factors": factors}]
                    outcome = self.solve_model(analysis="linear")
                    case_key = eff.name
                if not outcome.ok:
                    return ToolResult(False, {
                        "error": f"分析步 {eff.name!r} 求解失败",
                        "detail": outcome.payload,
                        "completed": [s["step"] for s in summary]})
                entry = outcome.payload.get("cases", {}).get(case_key, {})
                summary.append({
                    "step": eff.name, "analysis": eff.analysis,
                    "loads": dict(eff.loads),
                    "supports": sorted(eff.supports),
                    "changes": list(eff.changes),
                    "max_deflection_mm": entry.get("max_deflection_mm"),
                    "max_displacement_mm": entry.get("max_displacement_mm"),
                    "equilibrium_ok": entry.get("equilibrium_ok")})
                if eff.name == target:
                    kept = (self.solution, self.result_db,
                            self.frame, self.compilation)
        finally:
            self.model = original

        self.solution, self.result_db, self.frame, self.compilation = kept
        return ToolResult(True, {
            "steps": summary, "count": len(summary), "inspecting": target,
            "note": f"会话里留的是分析步 {target!r} 的结果，后处理都按它来；"
                    "换一步看结果用 solve_steps(inspect=\"步名\")",
            "limitation": "每一步都从未变形、无应力状态重解，不把上一步的状态"
                          "带进来；杆件也不能在步之间生灭。所以这表达的是"
                          "「同一结构的几种配置」，不是施工过程。"})

    @staticmethod
    def _amplitude_factors(loads: dict, payload: dict) -> dict:
        """线性步里各工况的叠加系数：幅值曲线在伪时间 1.0 处的值。

        **不是一律取 1.0。** 自定义曲线完全可以收在 0.5——那种情况下取 1.0
        会把荷载放大一倍，而结果看着完全正常。
        """
        from frame3d import BUILTIN_AMPLITUDES, Amplitude

        defined = payload.get("amplitudes") or {}
        out = {}
        for case, amp in loads.items():
            if amp in defined:
                curve = Amplitude(amp, tuple((float(t), float(v))
                                             for t, v in defined[amp]))
            else:
                curve = Amplitude(amp, BUILTIN_AMPLITUDES[amp])
            out[case] = curve.at(1.0)
        return out

    def solve_model(self, analysis: str = "linear", increments: int = 10,
                    max_iter: int = 40, tolerance: float = 1e-7,
                    amplitudes: dict | None = None) -> ToolResult:
        errors = self.validation_errors()
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "先按上面的清单修正模型，再重新求解"})
        self.compilation = compile_model(self.model, validated=True)
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
            elif analysis == "step":
                from nonlinear import solve_step
                if not amplitudes:
                    self.solution = None
                    return ToolResult(False, {
                        "error": "分析类型 step 需要 amplitudes："
                                 "把工况名映射到幅值曲线名",
                        "example": {"DL": "STEP", "WX": "RAMP"},
                        "hint": "内置 RAMP（0→1 斜坡）与 STEP（全程为 1）；"
                                "自定义曲线用 define_amplitude"})
                self.solution = solve_step(
                    self.frame, dict(amplitudes), increments=increments,
                    max_iter=max_iter, tolerance=tolerance)
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
        view = self.result_db.solution_view(self.frame)
        for name, res in self.solution.all_results().items():
            node, mag = self._max_displacement(res)
            # 二阶分析的平衡只在变形后位形上成立。拿未变形几何去查，残差
            # 恰好是 P·Δ——那是二阶效应本身，不是误差。一根 1600 kN 轴压、
            # 顶点侧移 54 mm 的柱子会报 5.4% 的"不平衡"而模型完全正确，
            # 用户看到的是一次假警报。
            kind = str(self.solution.analysis.get("type", ""))
            second_order = kind.startswith(("pdelta", "material_nonlinear"))
            # 变形后位形下残差由几何刚度自身的近似定界，落在 1e-4 量级；
            # 线性静力用的 1e-8 会把每一次正确的二阶求解都判成不平衡。
            eq = check_equilibrium(self.frame, self.solution, name,
                                   rtol=1e-3 if second_order else 1e-8,
                                   deformed=second_order)
            entry = {"max_displacement_mm": round(mag * U.disp_scale, 6),
                     "at_node": node,
                     "equilibrium_ok": eq["ok"],
                     "equilibrium_residual": float(f"{eq['relative']:.3e}")}
            if second_order:
                entry["equilibrium_frame"] = "变形后位形"
            # 节点位移**不是**最大位移。满跨均布下挠度全在单元内部，
            # 节点那一栏可能只有真值的一半，简支梁更是直接报 0。
            # 详见 _intra_member_deflection 的注释。
            defl, defl_at = self._intra_member_deflection(view, name, entry)
            governing = max(mag, defl)
            if mag <= 1e-12 and self._applied_load_magnitude(name) > 0.0:
                axis = self._out_of_plane_load_axis(name)
                if axis is not None:
                    entry["warning"] = (
                        f"有荷载但节点位移为零：模型的节点都在一个平面内，而这个"
                        f"工况的荷载几乎全部沿平面法向 {axis}，也就是面外。"
                        "平面刚架（bays 留空）的面外自由度是被自动约束的，"
                        "面外荷载传不到节点上。检查荷载方向。"
                        + (f"（max_deflection_mm 里那 {entry['max_deflection_mm']} mm "
                           "是杆件在面外被两端夹住硬弯出来的，不代表结构真能这样受力。）"
                           if defl > 1e-12 else ""))
                elif defl > 1e-12:
                    entry["note"] = (
                        "节点位移为零是正常的，不是荷载加错了：这个工况的节点"
                        "全被约束住，响应发生在单元内部。真实最大挠度看 "
                        f"max_deflection_mm（{entry['max_deflection_mm']} mm，"
                        f"在{defl_at}），校核挠跨比用它。")
                else:
                    entry["warning"] = (
                        "有荷载但位移为零，且单元内部也没有挠度。常见原因：荷载"
                        "全部作用在被约束死的方向上。检查荷载方向。")
            elif span > 0 and governing > span / 200.0:
                entry["warning"] = (
                    f"最大位移达到最短杆件长度的 1/{max(1, int(span / governing))}，量级异常。"
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

    def _intra_member_deflection(self, view, name: str,
                                 entry: dict[str, Any]) -> tuple[float, str]:
        """把**含单元内部**的最大挠度写进 entry，返回 (挠度, 位置描述)。

        为什么 solve_model 的头条必须带上这个数：`_max_displacement` 只扫节点，
        而满跨均布、梯形这类荷载不会触发 model_compiler 的剖分（它只在集中力
        位置和显式内节点处切分），于是跨中根本没有节点可查。后果实测：

        * 简支梁 6 m、20 kN/m —— 节点位移 0.0 mm，真实跨中挠度 16.38 mm；
        * 87 杆三层框架、各梁 20 kN/m —— 头条报 4.30 mm，真实 9.27 mm，差 2.16 倍。

        两个数都不是错的，错的是只报前一个还管它叫"最大位移"。内力早就做了
        单元内解析恢复（internal_forces.member_deflection 的 docstring 专门
        写了这件事），位移这一路当时没接上来。

        算不出来时**显式说算不出来**，不静默省略——省略会让人以为单元内没有
        挠度，那正是这个函数要消灭的误读。代价实测 47 ms（求解本身 223 ms），
        stations 取 21 到 201 耗时几乎不变，瓶颈在逐单元开销，所以不必省测点。
        """
        try:
            from internal_forces import max_deflection
            best = max_deflection(self.frame, view, name, stations=101,
                                  mapping=self.compilation.mapping)
        except Exception as exc:  # noqa: BLE001 — 异常没被吞掉：它的文本
            # 进了 max_deflection_note，随结果一起返回给用户。挠度是求解的
            # 附加信息，算不出来不该把整次求解拖垮（见模块 docstring 末段）。
            entry["max_deflection_mm"] = None
            entry["max_deflection_note"] = f"单元内挠度算不出来：{exc}"
            return 0.0, ""
        if best is None or best.get("member") is None:
            entry["max_deflection_mm"] = None
            entry["max_deflection_note"] = "模型里没有杆件，无从谈单元内挠度"
            return 0.0, ""
        value = float(best["value"])
        U = self.units
        entry["max_deflection_mm"] = round(value * U.disp_scale, 6)
        entry["at_member"] = best["member"]
        entry["at_x_m"] = round(float(best["x"]) * U.length_to_m, 4)
        where = f"杆件 {best['member']} 距 i 端 {entry['at_x_m']} m 处"
        return value, where

    def _out_of_plane_load_axis(self, name: str) -> str | None:
        """模型是真正的平面结构、且该工况荷载几乎全垂直于这个平面时，
        返回法向的名字（"X"/"Y"/"Z" 或坐标串）；否则返回 None。

        **为什么不能沿用"节点位移为零"做判据。** 两件完全不同的事都会让节点
        位移为零：

        * 简支梁加满跨均布——正常，节点本来就全被约束，响应在单元内部；
        * 平面刚架加面外荷载——建模错误，面外自由度是被自动约束的。

        原来的实现把两者都当成后者，于是对着一根正常的简支梁喊"检查荷载
        方向"。反过来只看单元内挠度又会把后者放过去——面外荷载在杆件内部是
        **真的**会产生弯曲的（实测 My = wL²/12、挠度 14.7 mm），它不是零。

        能分开两者的是几何：平面刚架的节点张成一个二维平面，面外方向唯一；
        简支梁的节点共线，压根没有唯一的"面外"，那种模型不该报这条。
        所以这里先做共面性判断，再看荷载方向。
        """
        pts = np.array([[n.x, n.y, n.z] for n in self.frame.nodes.values()],
                       dtype=float)
        if len(pts) < 3:
            return None
        centered = pts - pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(centered, full_matrices=True)
        if sv[0] <= 0:
            return None
        # sv[2]≈0 ⇒ 共面；sv[1] 太小 ⇒ 共线，没有唯一法向，不适用这条检查
        if sv[2] / sv[0] > 1e-9 or sv[1] / sv[0] < 1e-6:
            return None
        normal = vt[2] / np.linalg.norm(vt[2])

        vectors: list[np.ndarray] = []
        load_case = self.frame.load_cases.get(name)
        if load_case is None:                      # 组合不单独判，交给各工况
            return None
        for load in load_case.nodal_loads.values():
            vectors.append(np.asarray(load, dtype=float)[:3])
        for w in load_case.member_loads.values():
            vectors.append(np.asarray(w, dtype=float)[:3])
        for loads in (load_case.member_spans or {}).values():
            for item in loads:
                vectors.append(np.asarray(item.w1, dtype=float))
                vectors.append(np.asarray(item.w2, dtype=float))
        seen = False
        for vec in vectors:
            norm = float(np.linalg.norm(vec))
            if norm <= 0:
                continue
            seen = True
            if abs(float(np.dot(vec / norm, normal))) < 0.99:
                return None                        # 有一项在面内，就不是纯面外
        if not seen:
            return None
        for axis, label in enumerate("XYZ"):
            if abs(normal[axis]) > 0.999:
                return label
        return "(" + ", ".join(f"{v:.3f}" for v in normal) + ")"

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
                            dry_run: bool = False, levels: int = 1,
                            progress=None) -> ToolResult:
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
                    self, int(node_id), case, anchor_member, mesh_sizes_mm,
                    levels=int(levels), progress=progress)
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
