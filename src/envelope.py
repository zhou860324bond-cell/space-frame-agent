"""内力包络与控制组合。

真实的刚架设计不看单个工况，看的是**所有组合里最不利的那个**。这个模块
沿每根杆件逐点取各组合的上下包线，并记住每一点是**哪个组合控制**的。

两件事必须分开说清楚：

* **包络** 是逐点的上下界。跨中由重力组合控制、支座处由风组合控制，
  这在同一根杆上很常见——所以包络必须逐点算，不能"先各工况取极值再比"。
  后一种做法给出的数偏大且位置错乱，是最常见的偷懒错法。
* **控制组合** 是某一点取到极值时对应的那个组合名。设计时要引用的是它，
  而不只是那个数。

默认拿**组合**做包络（没有定义组合时退回用工况）。这是工程惯例：
组合才是校核对象，单工况只是中间量。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from frame3d import Frame
from internal_forces import (COMPONENTS, member_diagram, member_length,
                             physical_member_diagram)


@dataclass
class MemberEnvelope:
    """一根杆件的内力包络。

    x 是测点；upper / lower 是各分量逐点的上下包线；
    upper_case / lower_case 是每一点取到该极值的组合名。
    """
    member: int
    length: float
    x: np.ndarray
    cases: tuple[str, ...]
    upper: dict[str, np.ndarray] = field(default_factory=dict)
    lower: dict[str, np.ndarray] = field(default_factory=dict)
    upper_case: dict[str, list[str]] = field(default_factory=dict)
    lower_case: dict[str, list[str]] = field(default_factory=dict)

    def extreme(self, component: str) -> dict:
        """该分量绝对值最大的一点：位置、数值、控制组合、是上包线还是下包线。"""
        hi, lo = self.upper[component], self.lower[component]
        k_hi = int(np.argmax(np.abs(hi)))
        k_lo = int(np.argmax(np.abs(lo)))
        if abs(hi[k_hi]) >= abs(lo[k_lo]):
            return {"x": float(self.x[k_hi]), "value": float(hi[k_hi]),
                    "case": self.upper_case[component][k_hi], "side": "upper"}
        return {"x": float(self.x[k_lo]), "value": float(lo[k_lo]),
                "case": self.lower_case[component][k_lo], "side": "lower"}

    def governing(self, component: str) -> list[str]:
        """该分量在这根杆上出现过的全部控制组合，按出现频次降序。"""
        seen: dict[str, int] = {}
        for name in self.upper_case[component] + self.lower_case[component]:
            seen[name] = seen.get(name, 0) + 1
        return sorted(seen, key=lambda k: (-seen[k], k))


def envelope_cases(frame: Frame) -> tuple[str, ...]:
    """默认参与包络的名字：有组合就用组合，没有就用工况。

    组合才是设计校核的对象；把单工况混进来会让包络虚大——
    比如 1.0DL 和 1.3DL 同时在场，前者永远不控制，只是噪声。
    """
    return tuple(frame.combos) if frame.combos else tuple(frame.load_cases)


def member_envelope(frame: Frame, solution, member_id: int,
                    cases: tuple[str, ...] | None = None,
                    stations: int = 21, mapping=None) -> MemberEnvelope:
    """一根杆件的逐点包络。"""
    names = tuple(cases) if cases else envelope_cases(frame)
    if not names:
        raise ValueError("没有可用于包络的工况或组合")
    available = set(solution.all_results())
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"结果里没有 {missing}，现有 {sorted(available)}")

    if mapping is None:
        from internal_forces import diagram_stations
        # 各工况的测点取并集：集中力位置在不同工况可能不同，
        # 只用其中一套会漏掉另一套的跳跃点
        grid = np.unique(np.concatenate(
            [diagram_stations(frame, solution, member_id, n, stations)
             for n in names]))
        diagrams = {n: member_diagram(frame, solution, member_id, n, at_x=grid)
                    for n in names}
        length = member_length(frame, member_id)
    else:
        # 编译后所有工况共用一套分析网格，逐段恢复后自然拥有同一组物理坐标。
        diagrams = {
            n: physical_member_diagram(frame, solution, mapping, member_id, n,
                                       stations)
            for n in names
        }
        grid = diagrams[names[0]].x
        length = diagrams[names[0]].length

    out = MemberEnvelope(member=member_id,
                         length=length,
                         x=grid, cases=names)
    for comp in COMPONENTS:
        stack = np.vstack([diagrams[n].component(comp) for n in names])
        hi = np.argmax(stack, axis=0)
        lo = np.argmin(stack, axis=0)
        out.upper[comp] = stack[hi, np.arange(stack.shape[1])]
        out.lower[comp] = stack[lo, np.arange(stack.shape[1])]
        out.upper_case[comp] = [names[k] for k in hi]
        out.lower_case[comp] = [names[k] for k in lo]
    return out


def all_envelopes(frame: Frame, solution, cases: tuple[str, ...] | None = None,
                  stations: int = 21, mapping=None) -> dict[int, MemberEnvelope]:
    member_ids = (mapping.physical_to_elements if mapping is not None
                  else frame.members)
    return {mid: member_envelope(frame, solution, mid, cases, stations, mapping)
            for mid in sorted(member_ids)}


def governing_summary(frame: Frame, solution, cases: tuple[str, ...] | None = None,
                      stations: int = 21, mapping=None) -> dict:
    """全结构每个分量的最不利点：在哪根杆、哪个位置、哪个组合控制。

    这是设计时真正要看的一张表——"这根梁按哪个组合配筋/选截面"。

    两个"控制"要分清，它们不是一回事：

    * `controls_global_extreme` —— 该组合拿下了几个分量的**全结构**极值；
    * `never_governs_anywhere` —— 该组合在**任何杆件的任何一点**都不控制。

    只按全局极值判断"从不控制"是错的：一个组合完全可能在某根梁的端部
    说了算，却不是全结构最大的那个点。按前者去删组合会删掉真正起作用的。
    这个错我写第一版时就犯了，靠一个三组合的算例才发现。
    """
    envs = all_envelopes(frame, solution, cases, stations, mapping)
    names = tuple(cases) if cases else envelope_cases(frame)
    worst: dict[str, dict] = {}
    global_counts: dict[str, int] = {n: 0 for n in names}
    anywhere: set[str] = set()

    for comp in COMPONENTS:
        best = None
        for mid, env in envs.items():
            anywhere.update(env.upper_case[comp])
            anywhere.update(env.lower_case[comp])
            got = env.extreme(comp)
            if best is None or abs(got["value"]) > abs(best["value"]):
                best = {**got, "member": mid}
        worst[comp] = best
        if best is not None:
            global_counts[best["case"]] = global_counts.get(best["case"], 0) + 1

    return {"cases": list(names), "worst": worst,
            "controls_global_extreme": global_counts,
            "governs_somewhere": sorted(anywhere),
            "never_governs_anywhere": [n for n in names if n not in anywhere]}
