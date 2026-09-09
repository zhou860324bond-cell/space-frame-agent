"""梁截面正应力。

**只算正应力 σ**：轴力与双向弯矩在极端纤维上的组合。

    非圆截面： σ = N/A ± ( |Mz|·cy/Iz + |My|·cz/Iy )
    圆形截面： σ = N/A ± sqrt( (Mz·cy/Iz)² + (My·cz/Iy)² )

两式的差别在于极值点在哪里。矩形的角、工字的翼缘尖，在 (±cy, ±cz) 处**确实
有材料**，所以两个弯曲项直接相加就是真实极值；圆形截面在那个位置没有材料，
极值出现在合弯矩方向上，必须平方和开方合成。把圆形套用前一式会高估，
虽然偏安全，但那是一个说不出来源的数，所以分开处理。

**不含剪应力**：τ = VQ/(Ib) 与扭转 τ = T·r/J 需要截面一次矩和壁厚，
当前截面契约里没有这些几何。因此本模块给出的不是 von Mises 等效应力，
不能替代需要计入剪切的校核。函数名里的 ``normal`` 就是这个限制的声明。

截面若没有 ``cy``/``cz``（即直接以 A/Iy/Iz/J 定义，而不是由 sections.py
的 builder 按尺寸算出），一律抛 :class:`StressUnavailable`，绝不估算——
一个来路不明的应力数比没有应力更危险。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from frame3d import Frame, Section
from internal_forces import member_diagram


class StressUnavailable(ValueError):
    """截面缺少极端纤维距离，正应力无法计算。"""


def _require_geometry(section: Section) -> None:
    if section.cy is None or section.cz is None:
        raise StressUnavailable(
            f"截面 {section.name!r} 只有 A/Iy/Iz/J，没有极端纤维距离 cy/cz，"
            "算不出弯曲应力。请用 sections.py 的 builder 按尺寸定义截面"
            "（矩形/工字形/圆管/实心圆），或直接给出 cy、cz。")


def extreme_normal_stress(section: Section, N, My, Mz):
    """极端纤维正应力，返回 ``(σ_min, σ_max)``。

    受拉为正，与轴力 N 的约定一致。输入可以是标量或等长数组。
    """
    _require_geometry(section)
    N = np.asarray(N, dtype=float)
    My = np.asarray(My, dtype=float)
    Mz = np.asarray(Mz, dtype=float)

    axial = N / section.A
    from_mz = np.abs(Mz) * section.cy / section.Iz     # 绕局部 z：应力沿 y 变化
    from_my = np.abs(My) * section.cz / section.Iy     # 绕局部 y：应力沿 z 变化
    bending = (np.hypot(from_mz, from_my) if section.circular
               else from_mz + from_my)
    return axial - bending, axial + bending


def member_normal_stress(frame: Frame, solution, member_id: int,
                         case: str | None = None, stations: int = 21) -> dict:
    """一根杆件沿长度的极端纤维正应力。"""
    name = case or solution.primary
    member = frame.members[member_id]
    section = frame.sections[member.section]
    diagram = member_diagram(frame, solution, member_id, name, stations=stations)
    low, high = extreme_normal_stress(section, diagram.N, diagram.My, diagram.Mz)
    peak = np.maximum(np.abs(low), np.abs(high))
    k = int(np.argmax(peak))
    return {"member": member_id, "case": name, "section": section.name,
            "x": diagram.x, "sigma_min": low, "sigma_max": high,
            "sigma_abs": peak,
            "worst": {"x": float(diagram.x[k]), "value": float(peak[k]),
                      "sigma_min": float(low[k]), "sigma_max": float(high[k])}}


def max_normal_stress(frame: Frame, solution, case: str | None = None,
                      stations: int = 21,
                      member_ids: Iterable[int] | None = None) -> dict:
    """最大正应力绝对值：在哪根杆、哪个位置、拉还是压。

    ``member_ids`` 留空表示全结构。**范围内任何一根杆的截面缺几何就整体抛
    StressUnavailable**——只统计"算得出的那部分"会给出一个偏小的最大值，
    那正是最不该出现的失效方式。需要缩小范围时应当显式传 member_ids，
    而不是让缺几何的杆件被悄悄跳过。
    """
    name = case or solution.primary
    targets = (sorted(frame.members) if member_ids is None
               else sorted(member_ids))
    best: dict | None = None
    for mid in targets:
        got = member_normal_stress(frame, solution, mid, name, stations)
        w = got["worst"]
        if best is None or w["value"] > best["value"]:
            best = {"member": mid, "case": name, "section": got["section"],
                    "x": w["x"], "value": w["value"],
                    "sigma_min": w["sigma_min"], "sigma_max": w["sigma_max"]}
    if best is None:
        return {"member": None, "case": name, "section": None,
                "x": 0.0, "value": 0.0, "sigma_min": 0.0, "sigma_max": 0.0}
    return best


def worst_normal_stress_all_cases(frame: Frame, solution,
                                  stations: int = 21,
                                  member_ids: Iterable[int] | None = None
                                  ) -> float:
    """全工况下的最大正应力绝对值。截面校核约束用这个。"""
    worst = 0.0
    for name in solution.all_results():
        worst = max(worst, float(max_normal_stress(
            frame, solution, name, stations, member_ids)["value"]))
    return worst
