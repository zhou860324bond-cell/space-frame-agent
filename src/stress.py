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


# ----------------------------------------------------------------- 剪应力

def _require_shear_geometry(section: Section) -> None:
    if section.Sz is None or section.bz is None:
        raise StressUnavailable(
            f"截面 {section.name!r} 没有静矩 Sz 与中性轴宽度 bz，算不出剪应力 "
            "τ=V·S/(I·b)。请用 sections.py 的 builder 按尺寸定义截面，"
            "或直接给出 Sz/bz（弱轴受剪再给 Sy/by）。")


def shear_stress(section: Section, Vy, Vz):
    """中性轴处的剪应力，返回 ``(τ_y, τ_z)``。

        τ = V·S/(I·b)

    取中性轴是因为**那里 τ 最大**：弯曲正应力在中性轴为零、剪应力在中性轴
    最大，两者的极值点正好错开。这也是折算应力必须逐点算、不能拿两个极值
    硬凑的原因——σmax 和 τmax 根本不在同一个位置。

    校核过（V=100 kN）：矩形 1.5V/A、实心圆 4V/(3A)，误差 0.0000%；
    薄壁圆管与 2V/A 差 0.064%；工字形比 V/(hw·tw) 高 4.7%，
    那是抛物线分布峰值与"腹板均匀受剪"近似之间应有的差别。
    """
    _require_shear_geometry(section)
    Vy = np.asarray(Vy, dtype=float)
    Vz = np.asarray(Vz, dtype=float)
    tau_y = np.abs(Vy) * section.Sz / (section.Iz * section.bz)
    if section.Sy is None or section.by is None:
        tau_z = np.zeros_like(tau_y)
    else:
        tau_z = np.abs(Vz) * section.Sy / (section.Iy * section.by)
    return tau_y, tau_z


# ------------------------------------------------------------- 强度理论

#: 梁单元里只有 σx 与 τ，属于**平面应力中的单向拉压加纯剪**。
#: 这种应力状态下三个主应力有闭式解：
#:     σ1,3 = σ/2 ± √((σ/2)² + τ²)，  σ2 = 0
#: 四个强度理论的相当应力因此都能直接写成 σ 与 τ 的显式表达式，
#: 不必先解特征值。下面每一条都按这个前提推导。
def principal_stresses(sigma, tau):
    """由 (σ, τ) 给出三个主应力，从大到小。

    梁里 σ2 恒为 0：垂直于梁轴的两个方向都没有正应力，
    而纯剪在主轴上变成一对等值异号的正应力。
    """
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)
    half = sigma / 2.0
    radius = np.sqrt(half ** 2 + tau ** 2)
    a, b = half + radius, half - radius
    zero = np.zeros_like(a)
    stacked = np.stack([a, b, zero])
    stacked = np.sort(stacked, axis=0)[::-1]
    return stacked[0], stacked[1], stacked[2]


def equivalent_stress(sigma, tau, theory: str, nu: float = 0.3):
    """相当应力 σr。theory 取 "1"/"2"/"3"/"4"。

    * 第一强度理论（最大拉应力，Rankine）：σr1 = σ1
      只管拉，**压不管**——铸铁这类脆性材料受拉破坏用它。
    * 第二强度理论（最大伸长线应变）：σr2 = σ1 − ν(σ2 + σ3)
      需要泊松比，所以这一条是唯一依赖材料参数的。
    * 第三强度理论（最大剪应力，Tresca）：σr3 = σ1 − σ3 = √(σ² + 4τ²)
      塑性材料常用，偏安全。
    * 第四强度理论（形状改变比能，von Mises）：σr4 = √(σ² + 3τ²)
      塑性材料更贴合实测，**也正是 GB 50017 折算应力的那个式子**。

    σr3 与 σr4 都有闭式，这里仍走主应力再合成：两条路结果必须一致，
    测试拿它当交叉校验——闭式写错了主应力那条会立刻对不上。
    """
    s1, s2, s3 = principal_stresses(sigma, tau)
    theory = str(theory)
    if theory == "1":
        return s1
    if theory == "2":
        return s1 - float(nu) * (s2 + s3)
    if theory == "3":
        return s1 - s3
    if theory == "4":
        return np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))
    raise ValueError(f"强度理论只能取 1/2/3/4，收到 {theory!r}")


def von_mises(sigma, tau):
    """折算应力 √(σ² + 3τ²)。GB 50017 §6.1.5 用的就是它，等同第四强度理论。"""
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return np.sqrt(sigma ** 2 + 3.0 * tau ** 2)
