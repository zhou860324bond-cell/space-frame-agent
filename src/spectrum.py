"""振型分解反应谱法。

地震作用不是一个确定的荷载时程，规范给的是**谱**：每个自振周期对应一个
最大反应。做法是把结构分解成各阶振型，每阶按单自由度体系查谱，再把各阶
的反应组合起来。

这里有两件事最容易出错，都写在前面：

**组合结果没有符号。** SRSS 与 CQC 都是取平方和再开方，出来的只有大小。
"地震下柱底弯矩 120 kN·m"这句话缺了一半——地震往复，实际要按 ±120 与
重力工况分别组合。**不能把反应谱结果当成一个普通工况直接叠加进去**，
那样得到的只是两个方向里恰好同号的那一个。本模块返回的量一律是非负的，
组合时必须自己带上 ±。

**阶数要取够。** 判据是累计参与质量比，而它的分母是**能参与振动的**质量
`rᵀM_ff r`，不是 Σ ρAL——压在支座上的质量永远不参与。拿 Σ ρAL 当分母的话，
粗网格模型把振型取满也到不了 90%，而用户会以为是阶数不够。详见 modal.py。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frame3d import Frame, LoadCase, _member_forces
from modal import ModalResult, modal

#: 组合方式。
COMBINATIONS = ("SRSS", "CQC")

#: 方向名 → 平动自由度序号。
DIRECTIONS = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SpectrumResult:
    """反应谱分析结果。**所有量都是非负的大小，没有符号。**"""

    direction: str
    combination: str
    #: (n_modes,) 各阶周期 s
    periods: np.ndarray
    #: (n_modes,) 各阶查到的谱加速度 m/s²
    spectral_acceleration: np.ndarray
    #: (n_modes,) 各阶的基底剪力 M_eff·Sa，带符号（供检查 Σ 是否等于总量）
    modal_base_shear: np.ndarray
    #: (n_modes, 3) 各阶累计参与质量比
    cumulative_ratio: np.ndarray
    #: (n_dofs,) 组合后的位移大小
    displacement: np.ndarray
    #: 杆件编号 → (12,) 组合后的杆端力大小
    member_forces: dict[int, np.ndarray]
    #: 组合后的基底剪力大小
    base_shear: float
    #: 取用阶数覆盖的参与质量比
    mass_ratio: float


def gb50011_spectrum(alpha_max: float, tg: float,
                     damping: float = 0.05):
    """GB 50011 的设计反应谱 α(T)，返回一个可调用的 α(T)。

    ``alpha_max`` 是水平地震影响系数最大值（多遇地震 7 度 0.08、8 度 0.16
    这一类），``tg`` 是特征周期。返回的是**地震影响系数** α，不是加速度；
    乘以重力加速度才是 m/s²，本模块的 ``spectrum`` 参数要的是后者。

    分段与阻尼调整系数按规范 5.1.5：

    * 0 < T ≤ 0.1：直线上升段
    * 0.1 < T ≤ Tg：平台段 η₂·α_max
    * Tg < T ≤ 5Tg：曲线下降段 (Tg/T)^γ·η₂·α_max
    * 5Tg < T ≤ 6.0：直线下降段
    """
    zeta = float(damping)
    gamma = 0.9 + (0.05 - zeta) / (0.3 + 6 * zeta)
    eta1 = max(0.0, 0.02 + (0.05 - zeta) / (4 + 32 * zeta))
    eta2 = max(0.55, 1.0 + (0.05 - zeta) / (0.08 + 1.6 * zeta))

    def alpha(period: float) -> float:
        t = float(period)
        if t <= 0.1:
            return (0.45 + (eta2 - 0.45) * t / 0.1) * alpha_max
        if t <= tg:
            return eta2 * alpha_max
        if t <= 5 * tg:
            return (tg / t) ** gamma * eta2 * alpha_max
        # 6 秒以上规范没有规定，沿用直线下降段的式子而不是外推成负数
        return max(0.0, (eta2 * 0.2 ** gamma - eta1 * (t - 5 * tg)) * alpha_max)

    return alpha


def table_spectrum(points):
    """把一张 (周期, 谱值) 表变成可调用的谱，线性插值、表外取端点值。

    不外推——表只写到 6 秒而某阶周期是 6.01 秒时，外推会给出一个表外的值，
    那种错很难查。
    """
    table = sorted((float(t), float(v)) for t, v in points)
    if len(table) < 2:
        raise ValueError("谱至少要两个点")

    def value(period: float) -> float:
        t = float(period)
        if t <= table[0][0]:
            return table[0][1]
        if t >= table[-1][0]:
            return table[-1][1]
        for (t0, v0), (t1, v1) in zip(table, table[1:], strict=False):
            if t0 <= t <= t1:
                return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        return table[-1][1]

    return value


def _cqc_correlation(omega: np.ndarray, damping: float) -> np.ndarray:
    """Der Kiureghian 的振型相关系数。

    频率接近的振型不是独立的，SRSS 把它们当独立处理会低估——三维结构里
    平扭耦联的那两阶往往只差百分之几，这时 CQC 与 SRSS 能差十几个百分点。
    """
    zeta = float(damping)
    ratio = omega[:, None] / omega[None, :]
    numerator = 8 * zeta**2 * (1 + ratio) * ratio ** 1.5
    denominator = ((1 - ratio**2) ** 2
                   + 4 * zeta**2 * ratio * (1 + ratio) ** 2)
    return numerator / denominator


def _combine(responses: np.ndarray, omega: np.ndarray,
             combination: str, damping: float) -> np.ndarray:
    """把各阶反应组合成一个大小。responses 的第一维是振型。"""
    if combination == "SRSS":
        return np.sqrt(np.sum(responses**2, axis=0))
    rho = _cqc_correlation(omega, damping)
    flat = responses.reshape(responses.shape[0], -1)
    total = np.einsum("ij,ik,jk->k", rho, flat, flat)
    return np.sqrt(np.maximum(total, 0.0)).reshape(responses.shape[1:])


def response_spectrum(model: Frame, spectrum, direction: str = "x",
                      num_modes: int = 12, combination: str = "CQC",
                      damping: float = 0.05,
                      prepared: ModalResult | None = None) -> SpectrumResult:
    """振型分解反应谱法。

    ``spectrum`` 是可调用的 Sa(T)，单位 m/s²（``gb50011_spectrum`` 给的是
    地震影响系数 α，要自己乘 g）。

    每一阶的反应是 ``Γ_k · φ_k · Sa(T_k) / ω_k²``，各阶再按 SRSS 或 CQC 组合。

    **返回的量没有符号。** 地震往复，用这些数去组合时必须自己带 ±。
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction 只能是 {list(DIRECTIONS)}")
    if combination not in COMBINATIONS:
        raise ValueError(f"combination 只能是 {list(COMBINATIONS)}")
    if not 0 < damping < 1:
        raise ValueError("阻尼比要在 0 与 1 之间")

    result = prepared or modal(model, num_modes=num_modes)
    axis = DIRECTIONS[direction]
    omega = np.asarray(result.omega, dtype=float)
    if np.any(omega <= 0):
        raise ValueError(
            "存在零频振型（刚体位移），反应谱法要求结构已被完全约束")

    periods = np.asarray(result.periods, dtype=float)
    accelerations = np.array([float(spectrum(t)) for t in periods])
    factors = np.asarray(result.participation, dtype=float)[:, axis]
    shapes = np.asarray(result.shapes, dtype=float)

    # 各阶位移：Γ·φ·Sa/ω²。带符号——组合是下一步的事。
    scale = factors * accelerations / omega**2
    modal_displacement = (shapes * scale).T           # (n_modes, n_dofs)

    empty = LoadCase("__spectrum__")
    modal_member = {}
    for index in range(modal_displacement.shape[0]):
        forces = _member_forces(model, modal_displacement[index], empty)
        for member_id, value in forces.items():
            modal_member.setdefault(member_id, []).append(value)

    displacement = _combine(modal_displacement, omega, combination, damping)
    member_forces = {
        member_id: _combine(np.array(values), omega, combination, damping)
        for member_id, values in modal_member.items()
    }

    # 各阶基底剪力就是 M_eff·Sa——这是反应谱法最有名的那条恒等式，
    # 也是检查实现对不对最硬的判据：谱取常量时各阶之和等于可参与质量×Sa。
    effective = np.asarray(result.effective_mass, dtype=float)[:, axis]
    modal_shear = effective * accelerations
    combined_shear = float(_combine(modal_shear[:, None], omega,
                                    combination, damping)[0])
    covered = float(result.cumulative_ratio[-1, axis])

    return SpectrumResult(
        direction=direction, combination=combination, periods=periods,
        spectral_acceleration=accelerations, modal_base_shear=modal_shear,
        cumulative_ratio=np.asarray(result.cumulative_ratio, dtype=float),
        displacement=displacement, member_forces=member_forces,
        base_shear=combined_shear, mass_ratio=covered)


__all__ = ["COMBINATIONS", "DIRECTIONS", "SpectrumResult", "gb50011_spectrum",
           "response_spectrum", "table_spectrum"]
