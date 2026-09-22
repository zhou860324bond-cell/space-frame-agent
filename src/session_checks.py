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

from frame3d import diagnose_singularity
from model_compiler import CompilationError, compile_model
from model_io import from_dict, validate_payload
from session_base import ToolResult


class ChecksMixin:
    """校核：模态、屈曲、强度、对称性、编号、支座诊断。见模块 docstring。"""
    def _elements_per_span(self, frame=None) -> tuple[int, int]:
        """(最少单元数, 那一跨里的一个代表杆号)。算不出来时返回 (0, 0)。

        数的是**每跨**几个单元，不是"每根构件几个单元"——这两个不一样，
        而按后者判会造出假警告：用户把一根 8 m 梁手工拆成 4 根构件时，
        每根仍只有 1 个单元，但那一跨实际有 4 个，结果误差 0.05%，
        这时候还喊"网格太粗"就是喊狼来了。

        "跨"的定义：沿着**共线、无支座、且只连两根杆**的中间节点把杆件串起来，
        串不下去的地方（拐角、汇交、支座）就是跨的端点。这正好对应形函数
        需要细分的那个尺度——决定一致质量阵/几何刚度阵精度的是两个约束点
        之间放了几个单元。

        模态与屈曲都靠形函数装配，两者都从上方逼近精确解；静力内力是解析
        恢复的、与网格无关，所以这条只管这两项。
        """
        frame = frame if frame is not None else getattr(self, "frame", None)
        if frame is None or not getattr(frame, "members", None):
            return 0, 0
        incident: dict[int, list[int]] = {}
        axis: dict[int, np.ndarray] = {}
        for mid, m in frame.members.items():
            ni, nj = frame.nodes[m.i], frame.nodes[m.j]
            vec = np.array([nj.x - ni.x, nj.y - ni.y, nj.z - ni.z], dtype=float)
            norm = float(np.linalg.norm(vec))
            if norm <= 0:
                continue
            axis[mid] = vec / norm
            incident.setdefault(int(m.i), []).append(mid)
            incident.setdefault(int(m.j), []).append(mid)

        def continues_through(node_id: int) -> tuple[int, int] | None:
            """这个节点是不是"跨内部"的点；是就返回它连的两根杆。"""
            here = incident.get(node_id, [])
            if len(here) != 2 or node_id in getattr(frame, "supports", {}):
                return None
            a, b = here
            if abs(float(np.dot(axis[a], axis[b]))) < 0.999:
                return None                       # 拐了弯，不是同一跨
            return a, b

        seen: set[int] = set()
        smallest = (0, 0)
        for mid in sorted(axis):
            if mid in seen:
                continue
            chain = {mid}
            # 从这根杆往两头走，能穿过去就继续
            frontier = [(int(frame.members[mid].i), mid),
                        (int(frame.members[mid].j), mid)]
            while frontier:
                node_id, came_from = frontier.pop()
                pair = continues_through(node_id)
                if pair is None:
                    continue
                nxt = pair[0] if pair[1] == came_from else pair[1]
                if nxt in chain:
                    continue
                chain.add(nxt)
                other = frame.members[nxt]
                far = other.j if int(other.i) == node_id else other.i
                frontier.append((int(far), nxt))
            seen |= chain
            if smallest[0] == 0 or len(chain) < smallest[0]:
                smallest = (len(chain), min(chain))
        return smallest

    # 实测（逐级加密到收敛，再回看粗网格的偏差）：
    #
    #   单跨门式刚架，每构件 1 个单元 —— λ=1.66，收敛值 0.82，**偏高 102%**。
    #   λ=1.66 是在说「还有 66% 余量」，而真相是 λ<1、它已经失稳了。
    #   两个单元就收敛到 0.8229，再加密不变。
    #
    #   8 m 梁，1 个单元 —— 屈曲简支 +21.6%、悬臂 +0.75%、两端固接解不出来；
    #                      模态简支 +11.0%、悬臂 +0.47%。
    #
    # 偏差方向永远是**偏高**（形函数比真实振型硬），也就是偏不安全那一侧。
    # generate_frame 生成的恰好是每构件一个单元，所以这条会在最常见的
    # 模型上触发——那不是误报，那正是它要拦的情形。
    COARSE_MESH_ELEMENTS = 4

    def _coarse_mesh_note(self, what: str, frame=None) -> dict[str, Any] | None:
        """网格太粗时给一条带**实测数字**的提醒，够密就不啰嗦。"""
        count, mid = self._elements_per_span(frame)
        if not count or count >= self.COARSE_MESH_ELEMENTS:
            return None
        return {
            "elements_in_span": count,
            "member_in_that_span": mid,
            "direction": "偏高",
            "message": (
                f"构件 {mid} 所在的那一跨只有 {count} 个单元。{what}靠形函数装配，"
                "**从上方**逼近精确解——网格越粗报得越高，也就是偏不安全那一侧。"
                "实测单跨门式刚架每构件一个单元时 λ=1.66，而收敛值是 0.82，"
                "偏高 102%：前者在说「还有 66% 余量」，后者意味着它已经失稳。"
                "两个单元就收敛。8 m 梁一个单元则是屈曲 +21.6%、模态 +11.0%（简支）。"
                "要稳妥，每跨至少 4 个单元。"),
            "how_to_fix": "在那一跨里加中间节点，让它至少有 4 个单元，再重算。",
        }

    def modal_analysis(self, num_modes: int = 6) -> ToolResult:
        """自振频率与振型。结构固有属性，与荷载无关。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        try:
            from modal import modal
            physical = from_dict(self.model)
            r = modal(physical, int(num_modes))
        except ImportError as exc:
            return ToolResult(False, {"error": f"模态模块不可用：{exc}"})
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # 参与质量比的分母是**能参与振动的**质量 rᵀM_ff r，不是 Σ ρAL。
        # 压在支座上的那部分永远不参与：拿 Σ ρAL 当分母，一根剖成 4 段的
        # 悬臂柱把振型取满也只报得出 84%，而"加大 num_modes"这条建议在那里
        # 是无效的——差的那 16% 不在振型里，在支座上。网格越粗越明显。
        cumulative = r.cumulative_ratio[-1]
        payload = {
            "modes": [{"order": k + 1,
                       "frequency_Hz": round(float(f), 6),
                       "period_s": round(float(1.0 / f), 6) if f > 0 else None,
                       "participation_xyz": [round(float(v), 6)
                                             for v in r.participation[k]],
                       "mass_ratio_xyz": [round(float(v), 4)
                                          for v in r.mass_ratio[k]]}
                      for k, f in enumerate(r.frequencies)],
            "total_mass_kg": round(r.total_mass, 6),
            "participable_mass_kg": [round(float(v), 6)
                                     for v in r.participable_mass],
            "cumulative_mass_ratio_xyz": [round(float(v), 4)
                                          for v in cumulative],
            "note": "一致质量矩阵，频率略高于精确解；每跨四个单元时误差 0.2% 以内。"
                    "**参与质量比的分母是 participable_mass（能参与振动的质量），"
                    "不是 total_mass**——压在支座上的质量永远不参与，"
                    "两者的差随网格变粗而变大。",
        }
        short = {axis: float(cumulative[k])
                 for k, axis in enumerate("XYZ") if cumulative[k] < 0.9}
        if short:
            payload["mass_ratio_warning"] = (
                "这些方向的累计参与质量比不到 90%（GB 50011 的门槛）："
                + "、".join(f"{axis} {value:.1%}" for axis, value in short.items())
                + "。加大 num_modes 再算。**判据只对要做地震分析的方向适用**"
                  "——竖向柱在轴向的前几阶本来就接近 0，那不是缺陷。")
        if _coarse := self._coarse_mesh_note("一致质量阵", physical):
            payload["mesh_warning"] = _coarse
        return ToolResult(True, payload)

    def response_spectrum_analysis(
            self, direction: str = "x", num_modes: int = 12,
            alpha_max: float | None = None, tg: float = 0.35,
            spectrum_points: list | None = None,
            combination: str = "CQC", damping: float = 0.05,
            gravity: float = 9.81) -> ToolResult:
        """振型分解反应谱法（地震作用）。

        谱二选一：``alpha_max`` + ``tg`` 走 GB 50011 的设计谱（地震影响系数，
        内部乘 g 变成加速度），或者 ``spectrum_points`` 直接给 (周期, 加速度)
        表。

        **返回的内力和位移没有符号。** SRSS 与 CQC 都是平方和开方，出来的
        只有大小。地震往复，把它当普通工况直接叠加到重力上，得到的只是两个
        方向里恰好同号的那一个——必须按 ±  分别组合。
        """
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        if alpha_max is None and not spectrum_points:
            return ToolResult(False, {
                "error": "要么给 alpha_max（走 GB 50011 设计谱），"
                         "要么给 spectrum_points（自定义谱表）",
                "example": {"alpha_max": 0.08, "tg": 0.35}})
        try:
            from spectrum import (gb50011_spectrum, response_spectrum,
                                  table_spectrum)
            physical = from_dict(self.model)
            if spectrum_points:
                curve = table_spectrum(spectrum_points)
            else:
                alpha = gb50011_spectrum(float(alpha_max), float(tg),
                                         damping=float(damping))
                def curve(period, _alpha=alpha, _g=float(gravity)):
                    return _alpha(period) * _g
            result = response_spectrum(
                physical, curve, direction=str(direction),
                num_modes=int(num_modes), combination=str(combination),
                damping=float(damping))
        except ImportError as exc:
            return ToolResult(False, {"error": f"反应谱模块不可用：{exc}"})
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ToolResult(False, {"error": str(exc)})

        payload = {
            "direction": str(direction), "combination": str(combination),
            "base_shear_kN": round(result.base_shear / 1e3, 4),
            "mass_ratio": round(float(result.mass_ratio), 4),
            "modes": [{"order": k + 1,
                       "period_s": round(float(t), 6),
                       "spectral_acceleration": round(float(a), 6),
                       "base_shear_kN": round(float(v) / 1e3, 4)}
                      for k, (t, a, v) in enumerate(zip(
                          result.periods, result.spectral_acceleration,
                          result.modal_base_shear, strict=True))],
            "sign": "**这些量没有符号。** 地震往复，与重力组合时要按 ± 各算一次；"
                    "直接当普通工况叠加，得到的只是两个方向里恰好同号的那一个。",
        }
        if result.mass_ratio < 0.9:
            payload["mass_ratio_warning"] = (
                f"取用的 {num_modes} 阶只覆盖 {result.mass_ratio:.1%} 的参与质量，"
                "GB 50011 要求不小于 90%。加大 num_modes 再算。")
        return ToolResult(True, payload)

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
            # 网格太粗时这里报的是"没有找到正的临界荷载因子，检查荷载方向与
            # 约束"——把人指向荷载和约束，而真实原因常常是**一个单元根本没有
            # 可屈曲的自由度**（两端固接单单元实测就是这样）。照那句话去查
            # 荷载方向永远查不出问题，所以粗网格时要把这一条摆在前面。
            payload: dict[str, Any] = {"error": str(exc)}
            coarse = self._coarse_mesh_note("屈曲的几何刚度阵")
            if coarse is not None:
                payload["likely_cause"] = (
                    f"构件 {coarse['member_in_that_span']} 所在的那一跨只有 "
                    f"{coarse['elements_in_span']} 个单元。单元太少时跨内"
                    "没有可屈曲的自由度，解不出正的临界因子——这比荷载方向或"
                    "约束更可能是原因。先把构件拆细再试。")
                payload["mesh"] = coarse
            return ToolResult(False, payload)
        U = self.units
        worst = min(r.axial, key=lambda m: r.axial[m])
        return ToolResult(True, {
            "case": r.case,
            "critical_factor": round(float(r.critical), 6),
            "factors": [round(float(v), 6) for v in r.factors],
            "most_compressed_member": worst,
            "its_axial_kN": round(r.axial[worst] * U.force_scale, 6),
            **({"mesh_warning": _coarse} if (_coarse := self._coarse_mesh_note(
                "屈曲的几何刚度阵")) else {}),
            "note": "λ 是该工况荷载的临界放大倍数：λ=3 表示放大三倍才失稳。"
                    "这是**线性特征值屈曲**，假定失稳前保持线弹性、变形小、"
                    "轴力不随变形改变。真实结构有初始缺陷与残余应力，"
                    "实际承载力低于此值——只能当上限，不能直接当承载力用。",
        })

    def check_strength(self, members: list[int] | None = None,
                       cases: list[str] | None = None,
                       slenderness_limit: float | None = None,
                       buckling_curve: str = "b") -> ToolResult:
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
                                   else float(slenderness_limit)),
                buckling_curve=str(buckling_curve))
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
            c = r["combined"]
            g = c["gb50017"]
            row.update({
                # 折算应力与上面的 stress_ratio 是**两条独立结论**：
                # 前者把 σ 与 τ 合起来判，后者按拉压分别比许用值。
                # 短深梁上两者能差两倍多（实测 L/h=2.5 时 2.78 倍），
                # 只看正应力会把一根剪切控制的梁判成"安全得很"。
                "combined_stress_MPa": round(g["sigma_r"] * U.stress_scale, 3),
                "combined_ratio": round(g["ratio"], 4),
                "combined_point": g["point"],
                "combined_at_x_m": round(g["x"] * U.length_to_m, 3),
                "equivalent_stress_MPa": {
                    f"σr{t}": round(v["sigma_r"] * U.stress_scale, 3)
                    for t, v in c["theories"].items()},
            })
            if not c["has_shear"]:
                row["combined_note"] = c["note"]
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
                cc = b["code_check"]
                if cc is not None:
                    # 规范法与欧拉并列给出。欧拉在中小柔度段弃权，而实际
                    # 钢柱大多落在那里——实测 λ=24.9 的粗短柱，欧拉
                    # N/Pcr=0.023、规范 0.822，差 35 倍且偏不安全。
                    row.update({
                        "phi": round(cc["phi"], 4),
                        "buckling_curve": cc["curve"],
                        "code_stability_ratio": round(cc["ratio"], 4),
                        "code_stability_ok": cc["ok"],
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
            "worst_combined": got["worst_combined"] and {
                "member": got["worst_combined"]["member"],
                "ratio": round(got["worst_combined"]["ratio"], 4),
                "point": got["worst_combined"]["point"],
                "basis": got["worst_combined"]["basis"]},
            "worst_buckling": got["worst_buckling"] and {
                "member": got["worst_buckling"]["member"],
                "ratio": round(got["worst_buckling"]["ratio"], 4),
                "slenderness": round(got["worst_buckling"]["slenderness"], 1)},
            "notes": got["notes"],
            "warnings": warnings,
            "reading": "failed_members 是**真的超限**。inconclusive_members 现在只在"
                       "**连规范法也给不出结论**时才有内容（缺屈服应力）——"
                       "欧拉公式不适用的中小柔度段已由 GB 50017 的 φ 系数法覆盖，"
                       "不再弃权。",
            "limitation": "正应力校核按拉压分别比许用值；折算应力 √(σ²+3τ²) "
                          "另算一条，在极端纤维、中性轴、腹板边缘三处取最不利。"
                          "**仍不含扭转剪应力**（截面契约里没有扭转常数之外的壁厚分布），"
                          "所以还不是完整的规范承载力验算。"
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
