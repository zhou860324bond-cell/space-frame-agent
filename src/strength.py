"""强度验算与逐杆稳定校核（讲义 §3-9 三）。

讲义把「强度验算」拆成两件事，这里一件不少地做出来：

1. **强度**：逐杆算最不利截面的极端纤维正应力，与许用应力比。
   **拉、压许用值分开**——铸铁、砌体、木材的抗拉抗压相差好几倍，
   只用一个 [σ] 去卡，受拉那侧会漏判。
2. **稳定**：受压杆逐根按欧拉公式校核 ``Pcr = π²EI/(μl)²``，
   计算长度系数 μ 按两端约束取 1 / 2 / 0.7 / 0.5。

三个容易出事的地方，都在这里显式处理，而不是留给调用方：

**一、剖分**。杆件被自动剖分成若干分析单元后，如果拿分段长度去算 Pcr，
两等分就会让临界力变成四倍——**结果偏大、不报错、看着还很合理**。
所以传了 ``mapping`` 时一律按**物理构件**整根算，长度是整根的长度。
没传 mapping 就只能把每个单元当一根杆，这一点写在返回值的 ``notes`` 里。

**二、μ 的来源**。由杆端释放只能推出「无侧移」情形下的 μ；有侧移的框架
μ 会大于 1（悬臂柱到 2.0）。推定值一律在输出里标 ``mu_source`` 并附警告，
用户可以在杆件上写 ``mu_y`` / ``mu_z`` 覆盖。**推出来的数不当成用户给的数。**

**三、欧拉公式的适用范围**。σcr = π²E/λ² 只在比例极限以内成立；细长比小到
σcr 超过屈服应力时，欧拉值是个没有物理意义的大数，照着它判「稳定够」是危险的。
给了 ``yield_stress`` 就判 λ 与 λp = π√(E/σs) 的关系，判不了就明说判不了，
不假装通过。

不做的事，同样写清楚：这里只有正应力，没有剪应力与扭转（截面契约里没有
一次矩和壁厚，见 ``stress.py``），因此不是规范意义上的构件承载力验算；
框架整体的失稳请用 ``buckling.py`` 的特征值屈曲，那是另一回事——
逐杆欧拉校核看的是「这一根会不会先屈」，特征值屈曲看的是「整体什么时候失稳」。
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from frame3d import Frame
from internal_forces import member_diagram, physical_member_diagram
from stress import (StressUnavailable, equivalent_stress,
                    extreme_normal_stress)

# 讲义里的四种经典杆端约束 → 计算长度系数
MU_PINNED_PINNED = 1.0      # 两端铰支
MU_FIXED_FREE = 2.0         # 一端固定、一端自由（悬臂）
MU_FIXED_PINNED = 0.7       # 一端固定、一端铰支
MU_FIXED_FIXED = 0.5        # 两端固定

# 判定某端「是不是铰」看的是绕两个主轴的弯矩释放
_BENDING_DOFS = ("ry", "rz")

# 稳定校核的四种结局。「判不了」自成一档：既不是通过也不是超限。
BUCKLING_OK = "通过"
BUCKLING_FAIL = "超限"
BUCKLING_NA = "欧拉公式不适用"
BUCKLING_UNKNOWN = "无法判定"
BUCKLING_SLENDER = "长细比超限"


class StrengthUnavailable(ValueError):
    """缺少验算所需的材料或截面数据。"""


def effective_length_factor(released_i: bool, released_j: bool) -> float:
    """按两端是否释放弯矩给出 μ（无侧移假定）。

    看得出「铰」，看不出「自由端」——自由端是节点没有约束，不是杆端释放。
    所以这里永远给不出 2.0，悬臂柱必须由用户显式写 ``mu_y`` / ``mu_z``。
    """
    if released_i and released_j:
        return MU_PINNED_PINNED
    if released_i or released_j:
        return MU_FIXED_PINNED
    return MU_FIXED_FIXED


def _is_hinged(releases: Iterable[str]) -> bool:
    return any(name in _BENDING_DOFS for name in releases)


def _allowables(material) -> tuple[float, float, str]:
    """返回 (许用拉应力, 许用压应力, 压许用的来源说明)。"""
    ft = getattr(material, "allow_tension", None)
    fc = getattr(material, "allow_compression", None)
    if ft is None and fc is None:
        raise StrengthUnavailable(
            f"材料 {material.name!r} 没有许用应力，无法做强度验算。"
            "请给 allow_tension（受拉许用应力）；抗压不同的材料"
            "（铸铁、砌体、木材）再单独给 allow_compression。")
    if ft is None:
        return float(fc), float(fc), "拉许用未给，取压许用"
    if fc is None:
        # 钢材抗拉抗压相同，这是最常见的情形；但要让用户看见这是个默认
        return float(ft), float(ft), "压许用未给，取拉许用（钢材惯例）"
    return float(ft), float(fc), "用户给定"


def _member_view(frame: Frame, mapping, member_id: int):
    """物理构件的 (代表单元, i 端释放, j 端释放, 单元号列表)。"""
    if mapping is None:
        m = frame.members[member_id]
        return m, m.releases_i, m.releases_j, (member_id,)
    ids = mapping.element_ids(member_id)
    first, last = frame.members[ids[0]], frame.members[ids[-1]]
    return first, first.releases_i, last.releases_j, ids


# GB 50017 §6.1.5：折算应力允许取 β·f，β=1.1。
# 理由是那一点处于局部高应力区、塑性重分布余地大，不是"放宽要求"。
GB_BETA = 1.1

#: 折算应力要在**三个特征点**上分别算，因为 σ 与 τ 的极值根本不在同一处：
#:   极端纤维   σ 最大、τ = 0
#:   中性轴     弯曲 σ = 0（只剩 N/A）、τ 最大
#:   腹板边缘   两者同时都大 —— 控制点往往在这里，GB 50017 §6.1.5 验的就是它
#: 只取 σmax 和 τmax 硬凑成 √(σmax²+3τmax²) 是错的：那个组合在截面上
#: 并不存在，结果虚高，而且高多少说不清。
POINT_FIBRE = "极端纤维"
POINT_NEUTRAL = "中性轴"
POINT_WEB = "腹板边缘"


def _station_points(section, d):
    """逐测点给出三个特征点上的 (σ, τ)，返回 [(名称, σ数组, τ数组), ...]。

    缺剪切几何时只返回极端纤维那一项并把 τ 记为零——那等价于原来的纯正应力
    校核，结论不会因为"少算了 τ"而偏不安全地放松，但会在返回值里标明。
    """
    from stress import shear_stress

    low, high = extreme_normal_stress(section, d.N, d.My, d.Mz)
    # 极端纤维取拉压两侧里绝对值大的那个：相当应力与符号无关的只有
    # σr3/σr4，σr1/σr2 是认符号的，所以两侧都要留着给上层挑。
    points = [(POINT_FIBRE, high, np.zeros_like(high)),
              (POINT_FIBRE, low, np.zeros_like(low))]
    try:
        tau_y, tau_z = shear_stress(section, d.Vy, d.Vz)
    except StressUnavailable:
        return points, False

    # 两个方向的剪应力是截面上同一个剪应力矢量的两个分量，合成用平方和
    # 开方，不是直接相加——相加会虚高最多 41%。
    tau = np.hypot(tau_y, tau_z)
    sigma_axial = d.N / section.A
    points.append((POINT_NEUTRAL, sigma_axial, tau))

    if section.S_flange is not None and section.c_web is not None:
        # 腹板边缘：z = 0，所以 My 在这一点不产生正应力；
        # 中性轴以外只剩翼缘，静矩取 S_flange。
        bend = np.abs(d.Mz) * section.c_web / section.Iz
        tau_web = np.abs(d.Vy) * section.S_flange / (section.Iz * section.bz)
        points.append((POINT_WEB, sigma_axial + bend, tau_web))
        points.append((POINT_WEB, sigma_axial - bend, tau_web))
    return points, True


def combined_stress_check(frame: Frame, solution, member_id: int, *,
                          mapping=None, cases: Iterable[str] | None = None,
                          stations: int = 21) -> dict[str, Any]:
    """一根构件的强度理论校核：材力四理论 + GB 50017 折算应力。

    与既有的正应力校核**并列**，不是替换。两者回答的不是同一个问题：

    * 正应力校核按拉、压分别比许用值——铸铁、砌体、木材抗拉抗压差好几倍，
      必须分开，这是相当应力做不到的（σr3/σr4 都是非负的，丢掉了符号）。
    * 相当应力把 σ 与 τ 合起来判，回答"这一点会不会屈服"。只有正应力时
      腹板受剪那一段完全查不出来。

    所以两条都留着，构件要两条都过才算通过。
    """
    member, _, _, _ = _member_view(frame, mapping, member_id)
    section = frame.sections[member.section]
    material = frame.materials[member.material]
    ft, _fc, _src = _allowables(material)
    names = list(cases) if cases is not None else list(solution.all_results())

    best: dict[str, Any] = {}
    has_shear = True
    for name in names:
        if mapping is None:
            d = member_diagram(frame, solution, member_id, name, stations)
        else:
            d = physical_member_diagram(frame, solution, mapping,
                                        member_id, name, stations)
        points, got_shear = _station_points(section, d)
        has_shear = has_shear and got_shear
        for theory in ("1", "2", "3", "4"):
            for label, sigma, tau in points:
                sr = equivalent_stress(sigma, tau, theory, nu=material.nu)
                k = int(np.argmax(sr))
                value = float(sr[k])
                prev = best.get(theory)
                if prev is None or value > prev["sigma_r"]:
                    best[theory] = {
                        "sigma_r": value, "ratio": value / ft,
                        "case": name, "x": float(d.x[k]), "point": label,
                        "sigma": float(np.asarray(sigma)[k]),
                        "tau": float(np.asarray(tau)[k]),
                    }

    gb = dict(best["4"])
    gb["limit"] = GB_BETA * ft
    gb["ratio"] = gb["sigma_r"] / gb["limit"]
    gb["ok"] = gb["ratio"] <= 1.0
    gb["basis"] = (f"GB 50017 §6.1.5 折算应力 √(σ²+3τ²) ≤ β·f，"
                   f"β={GB_BETA}，f 取材料的 allow_tension。")
    for theory in best:
        best[theory]["ok"] = best[theory]["ratio"] <= 1.0

    return {
        "member": member_id,
        "theories": best,
        "gb50017": gb,
        "has_shear": has_shear,
        "limit_used": ft,
        "note": ("截面缺少静矩/中性轴宽度，τ 按零处理——这一栏此时等价于纯正"
                 "应力校核，不能当作计入剪切的结论。"
                 if not has_shear else
                 "σ 与 τ 在极端纤维、中性轴、腹板边缘三处分别取值后合成；"
                 "不是拿 σmax 与 τmax 硬凑，那个组合在截面上并不存在。"),
    }


def check_member(frame: Frame, solution, member_id: int, *,
                 mapping=None, cases: Iterable[str] | None = None,
                 stations: int = 21,
                 slenderness_limit: float | None = None,
                 buckling_curve: str = "b") -> dict[str, Any]:
    """一根（物理）构件的强度与稳定验算。

    ``buckling_curve`` 是 GB 50017 的截面类别 a/b/c/d，只影响规范法的 φ。
    默认 b 类——热轧工字钢、H 型钢绕强轴的常见归类。**同一个 λ 下
    a 类比 d 类高出一大截**（λ=80 时 0.783 对 0.493），所以这个默认值
    不是可以随便忽略的细节，判不准时应当显式指定。
    """
    member, rel_i, rel_j, _ = _member_view(frame, mapping, member_id)
    section = frame.sections[member.section]
    material = frame.materials[member.material]
    ft, fc, allow_source = _allowables(material)

    names = list(cases) if cases is not None else list(solution.all_results())
    if not names:
        raise ValueError("没有可验算的工况")

    worst: dict[str, Any] | None = None
    axial_min = math.inf          # 最大压力（负得最多）
    axial_max = -math.inf         # 最大拉力
    axial_min_case = axial_max_case = names[0]
    length = 0.0

    for name in names:
        if mapping is None:
            d = member_diagram(frame, solution, member_id, name, stations)
        else:
            d = physical_member_diagram(frame, solution, mapping,
                                        member_id, name, stations)
        length = d.length
        low, high = extreme_normal_stress(section, d.N, d.My, d.Mz)
        # 应力比按拉压分开算，再取大者——不能先取 |σ| 再除一个许用值，
        # 那样拉压许用不同的材料一定判错。
        ratio_t = np.maximum(high, 0.0) / ft
        ratio_c = np.maximum(-low, 0.0) / fc
        ratio = np.maximum(ratio_t, ratio_c)
        k = int(np.argmax(ratio))
        if worst is None or float(ratio[k]) > worst["ratio"]:
            worst = {"case": name, "x": float(d.x[k]),
                     "ratio": float(ratio[k]),
                     "sigma_max": float(high[k]), "sigma_min": float(low[k]),
                     "N": float(d.N[k]), "My": float(d.My[k]),
                     "Mz": float(d.Mz[k]),
                     "governs": "受拉" if ratio_t[k] >= ratio_c[k] else "受压"}
        lo, hi = float(np.min(d.N)), float(np.max(d.N))
        if lo < axial_min:
            axial_min, axial_min_case = lo, name
        if hi > axial_max:
            axial_max, axial_max_case = hi, name

    assert worst is not None
    out: dict[str, Any] = {
        "member": member_id, "section": section.name,
        "material": material.name, "length": length,
        "allow_tension": ft, "allow_compression": fc,
        "allow_source": allow_source,
        "strength": {**worst, "ok": worst["ratio"] <= 1.0},
        "axial": {"max_tension": max(axial_max, 0.0),
                  "max_compression": min(axial_min, 0.0),
                  "tension_case": axial_max_case,
                  "compression_case": axial_min_case},
    }
    out["combined"] = combined_stress_check(
        frame, solution, member_id, mapping=mapping, cases=names,
        stations=stations)
    out["buckling"] = _euler(frame, member, section, material, length,
                             rel_i, rel_j, axial_min, axial_min_case,
                             slenderness_limit, buckling_curve)
    b = out["buckling"]
    failed = []
    if not out["strength"]["ok"]:
        failed.append("强度超限")
    # 折算应力单独成一条结论。它查得出正应力查不出的东西——腹板受剪那一段
    # 正应力可能很小，而 √(σ²+3τ²) 已经超了。两条都过才算通过。
    if not out["combined"]["gb50017"]["ok"]:
        failed.append("折算应力超限")
    if b is not None and not b["ok"]:
        failed.append(b["status"] if b["status"] == BUCKLING_SLENDER
                      else "稳定超限")
    out["ok"] = not failed
    # **规范法给了结论就不再算"判不了"。** 欧拉在 λ<λp 那一段只能弃权，
    # 而实际钢柱大多恰好落在那里——那一档原本意味着最常见的情形没有结论。
    # 实测一根 λ=24.9 的粗短柱：欧拉 N/Pcr=0.023（看着安全到离谱），
    # 规范法 N/(φA)/f=0.822。差 35 倍，而且是偏不安全的方向。
    code = b["code_check"] if b is not None else None
    out["conclusive"] = b is None or b["conclusive"] or code is not None
    reason = {BUCKLING_NA: "λ<λp，欧拉公式不适用",
              BUCKLING_UNKNOWN: "未给屈服应力"}.get(
                  b["status"] if b is not None else None, "")
    if failed:
        out["verdict"] = "、".join(failed)
    elif out["conclusive"] and code is not None and b["status"] == BUCKLING_NA:
        # 欧拉弃权但规范法通过：说清楚结论是谁给的，别让人以为欧拉说了算
        out["verdict"] = (f"通过（稳定按 GB 50017 {code['curve']} 类，"
                          f"φ={code['phi']:.3f}，应力比 {code['ratio']:.3f}；"
                          "欧拉公式在此柔度段不适用）")
    elif out["conclusive"]:
        out["verdict"] = "通过"
    else:
        out["verdict"] = f"强度通过 · 稳定判不了（{reason}）"
    return out


# ------------------------------------------- GB 50017 轴心受压稳定系数 φ

#: GB 50017-2017 附录 D 的 Perry 型拟合系数，按截面类别分。
#: c、d 两类在 λn=1.05 处换一组系数，所以值是 (界限, 低段, 高段)。
_PHI_COEFFS = {
    "a": (None, (0.41, 0.986, 0.152), None),
    "b": (None, (0.65, 0.965, 0.300), None),
    "c": (1.05, (0.73, 0.906, 0.595), (0.73, 1.216, 0.302)),
    "d": (1.05, (1.35, 0.868, 0.915), (1.35, 1.375, 0.432)),
}
BUCKLING_CURVES = tuple(_PHI_COEFFS)


def stability_factor(slenderness: float, yield_stress: float, modulus: float,
                     curve: str = "b") -> float:
    """GB 50017-2017 附录 D 的轴心受压构件稳定系数 φ。

    **这一条补的是欧拉公式的盲区。** λ<λp 时欧拉给出的 σcr 已经超过屈服
    应力，没有物理意义，所以 _euler 在那一段只能弃权——而实际钢柱大多
    恰好落在中小柔度区，弃权等于最常见的情形没有结论。

    φ 覆盖全柔度范围：λ→0 时 φ→1（强度控制），λ→∞ 时 φ→π²E/(λ²fy)
    （退回欧拉）。中间那段由 Perry-Robertson 型公式给出，系数按截面类别
    a/b/c/d 取——类别反映初弯曲与残余应力的严重程度，同一个 λ 下
    a 类比 d 类高出一大截。

    正则化长细比 λn = (λ/π)·√(fy/E)，λn=0.215 是低柔度段的分界。

    校核过：b 类 Q235 在 λ=40/60/80/100/120/150/200 处与规范表值
    逐点吻合（见 tests/test_strength.py）。
    """
    curve = str(curve).lower()
    if curve not in _PHI_COEFFS:
        raise ValueError(f"截面类别只能取 {BUCKLING_CURVES}，收到 {curve!r}")
    if yield_stress <= 0 or modulus <= 0:
        raise ValueError("算 φ 需要正的屈服应力与弹性模量")
    lam = float(slenderness)
    if lam <= 0.0:
        return 1.0
    lam_n = lam / math.pi * math.sqrt(float(yield_stress) / float(modulus))

    split, low, high = _PHI_COEFFS[curve]
    a1, a2, a3 = low if (split is None or lam_n <= split) else high
    if lam_n <= 0.215:
        return 1.0 - a1 * lam_n ** 2
    # φ = [ (α2+α3·λn+λn²) − √((α2+α3·λn+λn²)² − 4λn²) ] / (2λn²)
    t = a2 + a3 * lam_n + lam_n ** 2
    root = math.sqrt(max(t ** 2 - 4.0 * lam_n ** 2, 0.0))
    return (t - root) / (2.0 * lam_n ** 2)


def code_stability_check(P: float, area: float, slenderness: float,
                         material, curve: str = "b") -> dict[str, Any] | None:
    """N/(φ·A) ≤ f。缺屈服应力或许用值时返回 None，不猜。"""
    ft, _fc, _src = _allowables(material)
    if not material.yield_stress:
        return None
    phi = stability_factor(slenderness, float(material.yield_stress),
                           float(material.E), curve)
    sigma = P / (phi * area)
    return {
        "phi": phi,
        "curve": curve,
        "slenderness": slenderness,
        "sigma": sigma,
        "limit": ft,
        "ratio": sigma / ft,
        "ok": sigma <= ft,
        "basis": (f"GB 50017 附录 D，{curve} 类截面：N/(φ·A) ≤ f，"
                  f"φ={phi:.4f}，f 取材料的 allow_tension。"
                  "**覆盖全柔度范围**，欧拉公式弃权的中小柔度段由它给出结论。"),
    }


def _euler(frame, member, section, material, length, rel_i, rel_j,
           axial_min, case, slenderness_limit,
           curve: str = "b") -> dict[str, Any] | None:
    """受压杆的欧拉校核。不受压就返回 None——没有压力就没有屈曲。"""
    if axial_min >= 0.0:
        return None
    P = -axial_min                                   # 压力取正
    E, A = material.E, section.A

    given_y = getattr(member, "mu_y", None)
    given_z = getattr(member, "mu_z", None)
    inferred = effective_length_factor(_is_hinged(rel_i), _is_hinged(rel_j))
    mu_y = float(given_y) if given_y is not None else inferred
    mu_z = float(given_z) if given_z is not None else inferred

    axes = {}
    for tag, I, mu, given in (("y", section.Iy, mu_y, given_y),
                              ("z", section.Iz, mu_z, given_z)):
        le = mu * length
        radius = math.sqrt(I / A)                    # 回转半径 i = √(I/A)
        axes[tag] = {
            "mu": mu,
            "mu_source": "用户给定" if given is not None else "由杆端释放推定",
            "effective_length": le,
            "radius_of_gyration": radius,
            "slenderness": le / radius,
            "P_cr": math.pi ** 2 * E * I / le ** 2,
        }

    critical = min(axes, key=lambda t: axes[t]["P_cr"])
    P_cr = axes[critical]["P_cr"]
    slenderness = axes[critical]["slenderness"]
    sigma_cr = P_cr / A

    # 欧拉公式的适用范围：λ ≥ λp = π√(E/σs)。λ 小于它时公式给出的
    # σcr 已经超过屈服应力，是个没有物理意义的数，不能拿来判「稳定够」。
    valid: bool | None = None
    lambda_p: float | None = None
    if material.yield_stress:
        lambda_p = math.pi * math.sqrt(E / float(material.yield_stress))
        valid = slenderness >= lambda_p

    ratio = P / P_cr
    # 规范校核与欧拉并列。欧拉在 λ<λp 时只能弃权，而实际钢柱大多恰好落在
    # 那一段——弃权等于最常见的情形没有结论。φ 覆盖全柔度范围。
    code = code_stability_check(P, A, slenderness, material, curve)
    warnings: list[str] = []
    if any(axes[t]["mu_source"] != "用户给定" for t in axes):
        warnings.append(
            "μ 由杆端释放推定，只对**无侧移**结构成立。有侧移的框架柱 μ>1"
            "（悬臂柱 2.0），此时推定值偏小、Pcr 偏大、判定偏不安全。"
            "请在杆件上显式给 mu_y / mu_z，或用特征值屈曲分析核对。")

    # 三种结局要分清楚：**通过、超限、判不了**。
    # 把「判不了」算成「超限」会让一整片粗短柱报成不安全，用户会当成结构有问题；
    # 算成「通过」更糟——那是拿一个超过屈服应力的 σcr 给不安全的杆盖章。
    # 所以它自己是一档，ok 既不为真也不算失败，由调用方单独列出来。
    if ratio > 1.0:
        status = BUCKLING_FAIL
    elif valid is False:
        status = BUCKLING_NA
        if code is not None:
            warnings.append(
                f"欧拉公式在这根杆上不适用，但规范法给出了结论："
                f"φ={code['phi']:.4f}（{curve} 类截面），"
                f"N/(φA)={code['sigma'] / 1e6:.1f} MPa，"
                f"应力比 {code['ratio']:.3f} —— 以这一条为准。")
        # 措辞里**不放本杆的 λ 和 N/Pcr**：那两个数每根都不同，会让汇总层
        # 攒出十几条只差小数点的"同一句话"。逐杆的数在表里，警告只说结论。
        warnings.append(
            f"部分杆件的长细比小于 λp={lambda_p:.1f}，属于中小柔度杆，"
            "欧拉公式不适用（算出的 σcr 已超过屈服应力）。"
            "这类杆的稳定由强度或经验公式控制，**不能用这里的 Pcr 判**；"
            "照着算出来的 N/Pcr 往往很小，正是「照着算就会误判通过」的情形。")
    elif valid is None:
        status = BUCKLING_UNKNOWN
        warnings.append(
            "材料未给 yield_stress，无法判断是否落在欧拉公式的适用范围内。"
            "补上 yield_stress 才能确认这个 Pcr 有没有物理意义。")
    else:
        status = BUCKLING_OK
    if code is not None and not code["ok"] and status != BUCKLING_FAIL:
        # 规范法说超限而欧拉说没事：以规范为准。欧拉只在大柔度段可信，
        # 而它偏高的那一段恰恰是初弯曲与残余应力影响最大的地方。
        status = BUCKLING_FAIL
        warnings.append(
            f"欧拉判定未超限，但规范法 N/(φA) 超了（应力比 "
            f"{code['ratio']:.3f}）。欧拉不计初弯曲与残余应力，"
            "这一段它偏高——以规范结论为准。")
    if valid is False and ratio > 1.0:
        # 既超限又不适用：超限的结论仍然成立（真值只会比欧拉值更低）
        warnings.append("λ 小于 λp，实际临界力比欧拉值更低，超限的结论只会更严重。")

    if slenderness_limit is not None and slenderness > slenderness_limit:
        warnings.append(
            f"长细比 λ={slenderness:.1f} 超过限值 {slenderness_limit:g}。")
        status = BUCKLING_SLENDER

    return {"ok": status not in (BUCKLING_FAIL, BUCKLING_SLENDER),
            "status": status, "conclusive": status in (BUCKLING_OK,
                                                       BUCKLING_FAIL,
                                                       BUCKLING_SLENDER),
            "case": case, "P": P, "P_cr": P_cr, "ratio": ratio,
            "sigma_cr": sigma_cr, "critical_axis": critical,
            "slenderness": slenderness, "lambda_p": lambda_p,
            "euler_applicable": valid, "axes": axes,
            # 规范法的结论单独放一格。它和欧拉不是同一套判据，
            # 合并成一个数会让"以哪一条为准"这件事消失。
            "code_check": code, "warnings": warnings}


def check_strength(frame: Frame, solution, *, mapping=None,
                   members: Iterable[int] | None = None,
                   cases: Iterable[str] | None = None,
                   stations: int = 21,
                   slenderness_limit: float | None = None,
                   buckling_curve: str = "b") -> dict[str, Any]:
    """全结构强度 + 稳定验算。

    ``members`` 留空表示全部。传了 ``mapping`` 就按物理构件整根验算，
    这对稳定校核是**必须**的：按剖分后的分段算，Pcr 会成倍偏大。
    """
    if solution is None:
        raise ValueError("先求解再验算")
    if mapping is None:
        targets = sorted(frame.members) if members is None else sorted(members)
    else:
        targets = (sorted(mapping.physical_to_elements)
                   if members is None else sorted(members))

    rows = [check_member(frame, solution, mid, mapping=mapping, cases=cases,
                         stations=stations,
                         slenderness_limit=slenderness_limit,
                         buckling_curve=buckling_curve)
            for mid in targets]

    strength_worst = max(rows, key=lambda r: r["strength"]["ratio"], default=None)
    combined_worst = max(rows, key=lambda r: r["combined"]["gb50017"]["ratio"],
                         default=None)
    compressed = [r for r in rows if r["buckling"] is not None]
    buckling_worst = max(compressed, key=lambda r: r["buckling"]["ratio"],
                         default=None)
    notes: list[str] = []
    if mapping is None:
        notes.append(
            "未提供编译映射，按分析单元逐段验算。若模型发生过自动剖分，"
            "稳定校核用的是分段长度，Pcr 偏大、结果偏不安全。")
    if not compressed:
        notes.append("没有受压杆件，稳定校核未执行。")

    return {
        "members": rows,
        "cases": list(cases) if cases is not None
                 else list(solution.all_results()),
        "count": len(rows),
        "failed": [r["member"] for r in rows if not r["ok"]],
        # 判不了的单独列出来。混进 failed 里会让一整片粗短柱看着像结构不安全。
        "inconclusive": [r["member"] for r in rows
                         if r["ok"] and not r["conclusive"]],
        "worst_strength": (None if strength_worst is None else
                           {"member": strength_worst["member"],
                            **strength_worst["strength"]}),
        "worst_combined": (None if combined_worst is None else
                           {"member": combined_worst["member"],
                            **combined_worst["combined"]["gb50017"]}),
        "worst_buckling": (None if buckling_worst is None else
                           {"member": buckling_worst["member"],
                            **{k: v for k, v in buckling_worst["buckling"].items()
                               if k != "axes"}}),
        "ok": all(r["ok"] for r in rows),
        "notes": notes,
    }
