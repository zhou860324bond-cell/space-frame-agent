"""Session 的校核域：模态、屈曲、强度验算、对称性、节点编号与带宽、支座诊断。

从 agent.py 搬出来的，代码一个字节没改。

这一组是《计算结构力学》讲义逐节对照时补上的那几项，各自对应讲义的一节
（强度验算 §3-9 三、对称性 §3-9 四、编号与带宽 §3-9 八 / §3-10 / §4-6）。
它们的共同点是"算完之后再看一眼"：不改模型，只给判断。

注意边界：强度验算只含正应力，不含剪应力与扭转，**不是规范意义上的承载力
验算**；check_symmetry 通过只说明结果自洽，不等于模型建对了。这两句在系统
提示里也写着，别在任何一处把它说大。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frame3d import diagnose_singularity, solve
from model_compiler import CompilationError, compile_model
from model_io import from_dict, migrate_payload, validate_payload
from session_base import ToolResult


class ChecksMixin:
    """校核：模态、屈曲、强度、对称性、编号、支座诊断。见模块 docstring。"""
    def modal_analysis(self, num_modes: int = 6) -> ToolResult:
        """自振频率与振型。结构固有属性，与荷载无关。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        try:
            from modal import modal
            r = modal(from_dict(self.model), int(num_modes))
        except ImportError as exc:
            return ToolResult(False, {"error": f"模态模块不可用：{exc}"})
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ToolResult(False, {"error": str(exc)})
        share = r.effective_mass.sum(axis=0) / r.total_mass if r.total_mass else None
        return ToolResult(True, {
            "modes": [{"order": k + 1,
                       "frequency_Hz": round(float(f), 6),
                       "period_s": round(float(1.0 / f), 6) if f > 0 else None}
                      for k, f in enumerate(r.frequencies)],
            "total_mass_kg": round(r.total_mass, 6),
            "effective_mass_ratio_xyz": (None if share is None else
                                         [round(float(v), 4) for v in share]),
            "note": "一致质量矩阵，频率略高于精确解；每跨四个单元时误差 0.2% 以内。"
                    "有效质量比接近 1 才说明取的阶数够——差得远就加大 num_modes。",
        })

    def buckling_analysis(self, case: str | None = None,
                          num_modes: int = 4) -> ToolResult:
        """线性屈曲。λ 是该工况荷载的临界放大倍数。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        if self.solution is None:
            solved = self.solve_model()
            if not solved.ok:
                return ToolResult(False, {"error": "静力求解先失败了，无法做屈曲分析",
                                          "detail": solved.payload})
        name = case or self._controlling_case()
        try:
            from buckling import buckling
            r = buckling(self.frame, name, int(num_modes), solution=self.solution)
        except ImportError as exc:
            return ToolResult(False, {"error": f"屈曲模块不可用：{exc}"})
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ToolResult(False, {"error": str(exc)})
        U = self.units
        worst = min(r.axial, key=lambda m: r.axial[m])
        return ToolResult(True, {
            "case": r.case,
            "critical_factor": round(float(r.critical), 6),
            "factors": [round(float(v), 6) for v in r.factors],
            "most_compressed_member": worst,
            "its_axial_kN": round(r.axial[worst] * U.force_scale, 6),
            "note": "λ 是该工况荷载的临界放大倍数：λ=3 表示放大三倍才失稳。"
                    "这是**线性特征值屈曲**，假定失稳前保持线弹性、变形小、"
                    "轴力不随变形改变。真实结构有初始缺陷与残余应力，"
                    "实际承载力低于此值——只能当上限，不能直接当承载力用。",
        })

    def check_strength(self, members: list[int] | None = None,
                       cases: list[str] | None = None,
                       slenderness_limit: float | None = None) -> ToolResult:
        """强度验算 + 逐杆稳定校核（讲义 §3-9 三）。

        按**物理构件**验算：传编译映射进去，剖分过的杆件才会按整根算 Pcr。
        按分段算会让临界力成倍偏大，而且不报错。
        """
        if self.solution is None:
            solved = self.solve_model()
            if not solved.ok:
                return ToolResult(False, {"error": "还没有结果，且求解失败",
                                          "detail": solved.payload})
        try:
            from strength import StrengthUnavailable, check_strength
        except ImportError as exc:                      # pragma: no cover
            return ToolResult(False, {"error": f"强度验算模块不可用：{exc}"})
        try:
            from stress import StressUnavailable
        except ImportError as exc:                      # pragma: no cover
            return ToolResult(False, {"error": f"应力模块不可用：{exc}"})

        try:
            got = check_strength(
                self.frame, self.solution,
                mapping=self.compilation.mapping,
                members=[int(m) for m in members] if members else None,
                cases=[str(c) for c in cases] if cases else None,
                slenderness_limit=(None if slenderness_limit is None
                                   else float(slenderness_limit)))
        except StrengthUnavailable as exc:
            return ToolResult(False, {
                "error": str(exc),
                "hint": "在 define_materials_and_sections 里给材料加 allow_tension"})
        except StressUnavailable as exc:
            return ToolResult(False, {
                "error": str(exc),
                "hint": "截面要给 cy/cz（极端纤维距离），否则算不出弯曲应力"})
        except KeyError as exc:
            return ToolResult(False, {"error": f"杆件不存在：{exc}"})

        U = self.units
        rows = []
        for r in got["members"]:
            row = {
                "member": r["member"], "section": r["section"],
                "length_m": round(r["length"] * U.length_to_m, 4),
                "stress_ratio": round(r["strength"]["ratio"], 4),
                "governs": r["strength"]["governs"],
                "worst_case": r["strength"]["case"],
                "at_x_m": round(r["strength"]["x"] * U.length_to_m, 3),
                "verdict": r["verdict"],
            }
            b = r["buckling"]
            if b is not None:
                row.update({
                    "axial_kN": round(-b["P"] * U.force_scale, 3),
                    "P_cr_kN": round(b["P_cr"] * U.force_scale, 3),
                    "buckling_ratio": round(b["ratio"], 4),
                    "buckling_status": b["status"],
                    "slenderness": round(b["slenderness"], 1),
                    "mu": round(b["axes"][b["critical_axis"]]["mu"], 3),
                    "mu_source": b["axes"][b["critical_axis"]]["mu_source"],
                })
            rows.append(row)

        warnings: list[str] = []
        for r in got["members"]:
            if r["buckling"]:
                for w in r["buckling"]["warnings"]:
                    if w not in warnings:
                        warnings.append(w)
        return ToolResult(True, {
            "cases": got["cases"], "count": got["count"],
            "ok": got["ok"], "failed_members": got["failed"],
            # 「判不了」不是「不合格」。粗短杆的欧拉临界力没有物理意义，
            # 把它算进 failed 会让用户以为结构不安全。
            "inconclusive_members": got["inconclusive"],
            "members": rows,
            "worst_strength": got["worst_strength"] and {
                "member": got["worst_strength"]["member"],
                "ratio": round(got["worst_strength"]["ratio"], 4),
                "case": got["worst_strength"]["case"],
                "governs": got["worst_strength"]["governs"]},
            "worst_buckling": got["worst_buckling"] and {
                "member": got["worst_buckling"]["member"],
                "ratio": round(got["worst_buckling"]["ratio"], 4),
                "slenderness": round(got["worst_buckling"]["slenderness"], 1)},
            "notes": got["notes"],
            "warnings": warnings,
            "reading": "failed_members 是**真的超限**；inconclusive_members 是"
                       "「欧拉公式在这根杆上不适用（λ<λp，中小柔度）」，"
                       "既不是通过也不是不通过，转达时不要说成不安全。",
            "limitation": "只算正应力（轴力 + 双向弯曲的极端纤维应力），"
                          "**不含剪应力与扭转**，因此不是规范意义上的构件承载力验算。"
                          "逐杆欧拉校核回答「这一根会不会先屈」，"
                          "特征值屈曲分析回答「整体什么时候失稳」，"
                          "两者不能互相替代。",
        })

    def check_symmetry(self, case: str | None = None) -> ToolResult:
        """对称性检测 + 把对称性当校核用（讲义 §3-9 四）。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        try:
            from model_io import from_dict, migrate_payload
            from symmetry import check_symmetric_response, detect_symmetry
        except ImportError as exc:                      # pragma: no cover
            return ToolResult(False, {"error": f"对称性模块不可用：{exc}"})

        # 对着**物理模型**查：跨间集中力会生成内节点，剖分点不在对称面上时，
        # 分析模型的几何就不再对称，物理上明明对称的结构会查不出来。
        physical = from_dict(migrate_payload(self.model))
        found = detect_symmetry(physical)
        payload: dict[str, Any] = {
            "symmetric": found["symmetric"],
            "planes": [{"plane": p["plane"], "cases": p["cases"],
                        "usable_cases": p["usable"]} for p in found["planes"]],
            "advice": found["advice"],
            "note": "**不做半结构简化**：对称面上的边界条件写错不报错，"
                    "只会给出一个看着合理的错答案；而整解本来就很快，"
                    "省下的算力不值这个风险。对称性在这里当**校核手段**用。",
        }
        if self.solution is not None:
            name = case or self._controlling_case()
            checked = check_symmetric_response(self.frame, self.solution, name)
            payload["response_check"] = {
                "case": checked["case"], "ok": checked["ok"],
                "note": checked["note"],
                "checks": [{"plane": c["plane"], "kind": c["kind"],
                            "max_relative_difference": float(
                                f"{c['max_relative_difference']:.2e}"),
                            "worst_pair": c["worst_pair"], "ok": c["ok"]}
                           for c in checked["checks"]],
            }
            if checked["checks"]:
                payload["response_check"]["meaning"] = (
                    "对称结构 + 对称荷载 ⇒ 对称位置的位移必须互为镜像。"
                    "这条校核**不需要任何外部参照**，结构自己就是自己的对照组。")
        else:
            payload["response_check"] = {"note": "还没有结果，只做了几何与荷载判断"}
        return ToolResult(True, payload)

    def check_numbering(self, verify: bool = True) -> ToolResult:
        """节点编号与总刚存储方案对比（讲义 §3-9 八、§3-10、§4-6）。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        try:
            from frame3d import assemble, constrained_dofs, solve
            from numbering import (estimated_half_bandwidth, matrix_bandwidth,
                                   node_number_span, permute, rcm_order)
            from skyline import Skyline, factory
        except ImportError as exc:                      # pragma: no cover
            return ToolResult(False, {"error": f"存储模块不可用：{exc}"})

        if self.compilation is None:
            try:
                self.compilation = compile_model(self.model)
            except CompilationError as exc:
                return ToolResult(False, {"errors": list(exc.diagnostics)})
        frame = self.compilation.analysis_model
        K, _, _ = assemble(frame)
        fixed = constrained_dofs(frame)
        free = np.setdiff1d(np.arange(frame.num_dofs), fixed)
        if free.size == 0:
            return ToolResult(False, {"error": "没有自由自由度，无从谈带宽"})
        Kff = K[free][:, free]
        order = rcm_order(Kff)
        before = Skyline.from_matrix(Kff).storage()
        after = Skyline.from_matrix(permute(Kff, order)).storage()

        payload: dict[str, Any] = {
            "dofs": int(Kff.shape[0]),
            "node_number_span": node_number_span(frame),
            "estimated_half_bandwidth": estimated_half_bandwidth(frame),
            "half_bandwidth": {"current": matrix_bandwidth(Kff),
                               "after_rcm": matrix_bandwidth(permute(Kff, order))},
            "storage_entries": {
                "full": before["full"],
                "banded": after["banded"],
                "skyline_current": before["skyline"],
                "skyline_after_rcm": after["skyline"],
                "sparse_nonzeros": int(Kff.nnz)},
            "note": "重编号**只在求解内部发生**，模型里的节点号一个都没变。",
        }
        # 讲义那句结论要按**这个模型的实际数**说，不能背台词。
        # 小问题上稀疏的非零元数可能反而多于变带宽存储量（索引开销另算），
        # 写死"稀疏更省"就会和自己给出的表打架。
        sky, sparse = after["skyline"], int(Kff.nnz)
        payload["lecture"] = (
            "满阵→等带宽（§3-10）→一维变带宽（§4-6）是讲义的三步，"
            "每一步都靠「非零元集中在对角线附近」这一点省下存储。"
            + ("本例中稀疏存储只存非零元，比变带宽还省——变带宽的额外开销"
               "全在「列内的零也要存」。" if sparse < sky else
               "本例规模小，稀疏存储的非零元数反而多于变带宽存储量："
               "刚架的带内几乎填满，变带宽存的零不多，而稀疏还要另付行列"
               "索引的开销。规模一大、带宽相对变窄，稀疏才反超。"))
        if verify:
            try:
                reference = solve(frame)
                other = solve(frame, factorize=factory())
            except np.linalg.LinAlgError as exc:
                return ToolResult(False, {"error": f"求解失败：{exc}"})
            worst = 0.0
            for name in reference.all_results():
                a, b = reference[name].U, other[name].U
                scale = max(float(np.abs(a).max(initial=0.0)), 1e-30)
                worst = max(worst, float(np.max(np.abs(a - b))) / scale)
            payload["verification"] = {
                "max_relative_difference": float(f"{worst:.2e}"),
                "agrees": worst < 1e-8,
                "note": "讲义的一维变带宽 LDLᵀ 与默认的稀疏 LU 解同一个方程组。"
                        "两条路径共用同一份装配、约束消元与反力回算，"
                        "**只有分解这一步不同**，所以这个差值只反映解法本身。"}
        return ToolResult(True, payload)

    def diagnose_supports(self) -> ToolResult:
        if not self.model:
            return ToolResult(False, {"error": "还没有模型"})
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        frame = from_dict(self.model)
        modes = diagnose_singularity(frame)
        if not modes:
            return ToolResult(True, {"modes": [], "note": "约束充分，未发现刚体位移模态"})
        return ToolResult(True, {"modes": [
            {"eigenvalue": m["eigenvalue"],
             "participants": [{"node": p["node"], "direction": p["dof_name"]}
                              for p in m["participants"][:4]]}
            for m in modes
        ], "note": "上列节点与方向缺少约束，结构在该方向可自由运动"})
