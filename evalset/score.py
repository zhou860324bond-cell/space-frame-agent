"""评分器：一律读会话状态，不读回复文本。

回复可以说得天花乱坠，状态骗不了人。唯一读文本的是"追问"这一类——它的正确行为
恰恰是不建模，只能看有没有调工具加上回复里有没有问句。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODELLING_TOOLS = {"generate_frame", "set_model", "set_load_cases", "solve_model"}
_QUESTION_MARKS = ("？", "?")


@dataclass
class Verdict:
    case_id: str
    category: str
    checks: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, str] = field(default_factory=dict)
    rounds: int = 0
    seconds: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.checks) and all(self.checks.values())


def _vertical_total(session, case: str | None) -> float | None:
    if session.solution is None:
        return None
    name = case or session._controlling_case()
    try:
        res = session.solution[name]
    except KeyError:
        return None
    return sum(float(res.R[session.frame.node_dofs(nid)[2]])
               for nid in session.frame.supports) / 1e3


def _max_displacement_mm(session, case: str | None) -> float | None:
    if session.solution is None:
        return None
    name = case or session._controlling_case()
    try:
        res = session.solution[name]
    except KeyError:
        return None
    best = 0.0
    for nid in session.frame.order():
        d = session.frame.node_dofs(nid)
        best = max(best, float(np.linalg.norm(res.U[d[:3]])))
    return best * 1000.0


def score(case: dict, out: Any) -> Verdict:
    """按题目声明的期望逐项判定。out 是 run_turn 的返回值。"""
    checks = dict(case["checks"])
    v = Verdict(case_id=case["id"], category=case["category"], rounds=out.rounds)
    session = out.session
    used = [name for name, _ in out.tool_calls]

    if checks.get("must_ask"):
        # 「有没有追问」不能靠问号判——"请补充以下参数：1. 截面 2. 材料" 一个问号都没有，
        # 却是标准的追问。真正该验的是：它没擅自建模，且指出了缺的是什么。
        touched = [t for t in used if t in MODELLING_TOOLS]
        terms = checks.get("missing_terms") or []
        hit = [w for w in terms if w in out.reply]
        marked = any(mark in out.reply for mark in _QUESTION_MARKS)
        v.checks["未擅自建模"] = not touched
        v.checks["指出了缺失项"] = len(hit) >= 2 or (marked and not terms)
        v.detail["追问"] = (f"调用了 {touched or '无建模工具'}；"
                            f"点名的缺失项 {hit or '无'}（期望至少 2 项，共 {len(terms)} 项）；"
                            f"回复{'含' if marked else '不含'}问号")
        v.usage = getattr(out, "usage", {}) or {}
        return v

    if checks.get("must_handle_trap"):
        # 避开陷阱和撞上去再修，都是正确行为——要求"必须先失败"会惩罚有预见性的做法。
        # 判据只看结果：算出来了，而且说清了陷阱在哪。走的哪条路只作观察记录。
        failed = [n for n, _, r in session.tool_log if n == "solve_model" and not r.ok]
        v.checks["最终算了出来"] = session.solution is not None
        v.detail["处理方式"] = ("撞墙后自修复（solve 失败 %d 次）" % len(failed)
                                if failed else "预先避开，未触发失败")

    if checks.get("must_fail_to_solve"):
        v.checks["求解被拦下"] = session.solution is None
        words = checks.get("reply_mentions", [])
        hit = [w for w in words if w in out.reply]
        v.checks["说明了原因"] = bool(hit)
        v.detail["诊断"] = f"命中关键词 {hit or '无'}（期望其一：{words}）"

    if "topology" in checks:
        want = checks["topology"]
        got = {"nodes": len(session.model.get("nodes", [])),
               "members": len(session.model.get("members", []))}
        v.checks["拓扑正确"] = got == want
        v.detail["拓扑"] = f"实得 {got}，期望 {want}"

    for name, spec in (checks.get("numeric") or {}).items():
        key, _, case_name = name.partition("case:")
        key = key.rstrip("_") if key.endswith("_") else key
        target = case_name or None
        if key.startswith("vertical_total_kN"):
            got = _vertical_total(session, target)
        elif key.startswith("max_displacement_mm"):
            got = _max_displacement_mm(session, target)
        else:
            got = None
        if spec is None:                       # 只要求算出来，不比数值
            v.checks[f"{key} 有结果"] = got is not None
            v.detail[key] = f"实得 {got}"
            continue
        expect, rtol = spec
        ok = got is not None and abs(got - expect) <= rtol * max(abs(expect), 1e-12)
        v.checks[f"{key} 数值正确"] = ok
        rel = "—" if got is None else f"{abs(got - expect) / max(abs(expect), 1e-12):.2e}"
        v.detail[key] = f"实得 {got}，期望 {expect}，相对误差 {rel}"

    if checks.get("all_cases_equilibrium"):
        # 组合名由模型自定，不能拿名字做判据；平衡是与命名无关的硬性质
        from frame3d import check_equilibrium
        if session.solution is None:
            v.checks["每个工况都平衡"] = False
            v.detail["平衡"] = "没有求解结果"
        else:
            names = list(session.solution.all_results())
            bad = [n for n in names
                   if not check_equilibrium(session.frame, session.solution, n)["ok"]]
            v.checks["每个工况都平衡"] = not bad
            v.detail["平衡"] = f"{len(names)} 个工况/组合，未通过的：{bad or '无'}"

    for tool in checks.get("tools_required", []):
        v.checks[f"用了 {tool}"] = tool in used
    for tool in checks.get("tools_forbidden", []):
        v.checks[f"未用 {tool}"] = tool not in used
    if "min_load_cases" in checks:
        n = len(session.model.get("load_cases", []))
        v.checks["工况数达标"] = n >= checks["min_load_cases"]
        v.detail["工况"] = f"实得 {n} 个，至少要 {checks['min_load_cases']} 个"
    if checks.get("combos_defined"):
        combos = session.model.get("combos", [])
        v.checks["定义了组合"] = bool(combos)
        v.detail["组合"] = f"{[c.get('name') for c in combos] or '无'}"

    if not v.checks:
        v.checks["有结果"] = session.solution is not None
    return v


def aggregate(verdicts: list[Verdict]) -> dict[str, Any]:
    total = len(verdicts)
    passed = sum(1 for v in verdicts if v.passed)
    by_cat: dict[str, list[Verdict]] = {}
    for v in verdicts:
        by_cat.setdefault(v.category, []).append(v)
    return {
        "题数": total,
        "通过": passed,
        "通过率": round(passed / total, 4) if total else 0.0,
        "平均轮数": round(sum(v.rounds for v in verdicts) / total, 2) if total else 0.0,
        "分类": {c: {"通过": sum(1 for x in vs if x.passed), "题数": len(vs)}
                 for c, vs in by_cat.items()},
        "输入tokens": sum(v.usage.get("prompt_tokens", 0) for v in verdicts),
        "输出tokens": sum(v.usage.get("completion_tokens", 0) for v in verdicts),
        "缓存命中": sum(v.usage.get("cache_hit_tokens", 0) for v in verdicts),
    }
