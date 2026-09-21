"""杆间荷载：梯形分布与任意位置集中力。

原来只有一种杆间荷载——满跨均布。真实的刚架里，吊车梁下的集中力、
挑檐上的三角形荷载、边跨的梯形板传荷都算不了。这里补齐。

**两件事必须用同一份公式，否则弯矩图和求解器会各说各话**：

1. 等效节点荷载（固端力），求解时用；
2. 沿杆长的荷载合力与合力矩，回算内力分布时用。

所以两者都放在这个模块里，`frame3d` 和 `internal_forces` 各取所需，
不各写一遍。均布是梯形的特例（w2 = w1），只实现梯形一套公式，
均布走同一条路——这样两者永远不会漂。

约定与既有代码一致：
* 强度和力都在**全局坐标**下书写，进入公式前先转到杆件局部坐标；
* 均布线荷载使用当前模型的力/长度单位，集中力使用当前模型的力单位；
* 集中力位置 a 从 i 端量起，使用当前模型长度单位。

暂不支持跨间集中力偶——需要的话再加，别为了凑数把没验过的公式放进来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

UNIFORM = "uniform"
TRAPEZOID = "trapezoid"
POINT = "point"
PARTIAL = "partial"
#: 区间内线性变化。它是 partial（w1=w2）与 trapezoid（a=0,b=L）的推广，
#: 补它的直接理由是**双向板导荷**：短边梁的对称三角形、长边梁的对称梯形，
#: 在一根杆上都得靠两三段这种荷载拼出来，前面那两种一条都表达不了。
RAMP = "partial_trapezoid"
KINDS = (UNIFORM, TRAPEZOID, POINT, PARTIAL, RAMP)


@dataclass(frozen=True)
class SpanLoad:
    """一项杆间荷载。全局坐标。

    uniform:    w1 是满跨强度（当前力/长度单位），w2 与 a 不用
    trapezoid:  w1 是 i 端强度、w2 是 j 端强度（当前力/长度单位）
    point:      w1 是集中力（当前力单位），a 是距 i 端的距离（当前长度单位）
    partial:    w1 是强度，荷载只作用在 [a, b] 这一段（当前长度单位）
    partial_trapezoid: [a, b] 段内强度由 w1（在 a）线性变到 w2（在 b）
    """
    kind: str
    w1: tuple[float, float, float]
    w2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    a: float = 0.0
    #: partial 的终点。只有 partial 用得上；其余类型保持 0.0 并被忽略。
    b: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"杆间荷载类型只能取 {KINDS}，收到 {self.kind!r}")

    def scaled(self, factor: float) -> "SpanLoad":
        """按系数缩放。荷载组合按此逐项合成，位置 a、b 不缩放。"""
        return SpanLoad(self.kind,
                        tuple(factor * v for v in self.w1),
                        tuple(factor * v for v in self.w2),
                        self.a, self.b)

    def ends(self) -> tuple[np.ndarray, np.ndarray]:
        """把均布归一成梯形的两端强度——下游只需处理梯形一种。"""
        v1 = np.asarray(self.w1, dtype=float)
        v2 = v1 if self.kind == UNIFORM else np.asarray(self.w2, dtype=float)
        return v1, v2

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "w1": [float(v) for v in self.w1]}
        if self.kind == TRAPEZOID:
            out["w2"] = [float(v) for v in self.w2]
        if self.kind == POINT:
            out["a"] = float(self.a)
        if self.kind in (PARTIAL, RAMP):
            out["a"] = float(self.a)
            out["b"] = float(self.b)
        if self.kind == RAMP:
            out["w2"] = [float(v) for v in self.w2]
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SpanLoad":
        kind = str(data.get("kind", UNIFORM))
        if kind == RAMP and not {"w2", "a", "b"} <= set(data):
            raise ValueError("区间梯形荷载必须同时给出 w1、w2、a、b")
        if kind == TRAPEZOID and "w2" not in data:
            # 缺 w2 就默认成 w1 的话，写漏的梯形会静悄悄变成均布；
            # 默认成 0 又会静悄悄变成三角形。两种都不能接受
            raise ValueError("梯形荷载必须同时给出 w1 与 w2")
        if kind == PARTIAL and ("a" not in data or "b" not in data):
            # 缺哪一头都不能猜：默认成 0 或 L 会把"半跨堆载"静悄悄变成满跨，
            # 合力差一倍而且不报错。
            raise ValueError("部分跨均布必须同时给出起点 a 与终点 b")
        return SpanLoad(kind,
                        tuple(float(v) for v in data["w1"]),
                        tuple(float(v) for v in data.get("w2", (0.0, 0.0, 0.0))),
                        float(data.get("a", 0.0)),
                        float(data.get("b", 0.0)))


def _bending_fixed_end(L: float, q1: float, q2: float) -> tuple[float, float, float, float]:
    """梯形线荷载的 (V_i, V_j, M_i, M_j)。

    拆成"满跨均布 q1"加上"三角形 0→Δ"：
        均布   V = qL/2、qL/2          M = qL²/12、-qL²/12
        三角形 V = 3ΔL/20、7ΔL/20      M = ΔL²/30、-ΔL²/20
    """
    d = q2 - q1
    vi = q1 * L / 2.0 + 3.0 * d * L / 20.0
    vj = q1 * L / 2.0 + 7.0 * d * L / 20.0
    mi = q1 * L ** 2 / 12.0 + d * L ** 2 / 30.0
    mj = -q1 * L ** 2 / 12.0 - d * L ** 2 / 20.0
    return vi, vj, mi, mj


def _point_fixed_end(L: float, P: float, a: float) -> tuple[float, float, float, float]:
    """集中力的 (V_i, V_j, M_i, M_j)。a 从 i 端量起，b = L - a。"""
    b = L - a
    vi = P * b ** 2 * (L + 2.0 * a) / L ** 3
    vj = P * a ** 2 * (L + 2.0 * b) / L ** 3
    mi = P * a * b ** 2 / L ** 2
    mj = -P * a ** 2 * b / L ** 2
    return vi, vj, mi, mj


def _partial_fixed_end(L: float, w: float, a: float,
                       b: float) -> tuple[float, float, float, float]:
    """[a, b] 段均布的 (V_i, V_j, M_i, M_j)。

    做法是把 `_point_fixed_end` 对 x 从 a 积到 b——集中力的固端力本来就是
    位置的多项式，积分有闭式，不必另找公式表：

        V_i = w/L³ · [L³x − Lx³ + x⁴/2]
        V_j = w/L³ · [Lx³ − x⁴/2]
        M_i = w/L² · [L²x²/2 − 2Lx³/3 + x⁴/4]
        M_j = −w/L² · [Lx³/3 − x⁴/4]

    **自校验**：取 a=0、b=L 时四项分别退回 wL/2、wL/2、wL²/12、−wL²/12，
    也就是满跨均布的固端力，一位不差。测试拿这一条当判据。
    """
    def vi_f(x: float) -> float:
        return L ** 3 * x - L * x ** 3 + x ** 4 / 2.0

    def vj_f(x: float) -> float:
        return L * x ** 3 - x ** 4 / 2.0

    def mi_f(x: float) -> float:
        return L ** 2 * x ** 2 / 2.0 - 2.0 * L * x ** 3 / 3.0 + x ** 4 / 4.0

    def mj_f(x: float) -> float:
        return L * x ** 3 / 3.0 - x ** 4 / 4.0

    vi = w * (vi_f(b) - vi_f(a)) / L ** 3
    vj = w * (vj_f(b) - vj_f(a)) / L ** 3
    mi = w * (mi_f(b) - mi_f(a)) / L ** 2
    mj = -w * (mj_f(b) - mj_f(a)) / L ** 2
    return vi, vj, mi, mj


def _point_kernels(L: float) -> tuple[np.polynomial.Polynomial, ...]:
    """集中力固端力对位置 x 的四个核：(V_i, V_j, M_i, M_j) / P。

    直接从 `_point_fixed_end` 里读出来，写成多项式：

        V_i = (L³ − 3Lx² + 2x³) / L³      （由 b²(L+2a)/L³ 展开）
        V_j = (3Lx² − 2x³) / L³           （由 a²(L+2b)/L³ 展开）
        M_i = (L²x − 2Lx² + x³) / L²      （由 a·b²/L²）
        M_j = −(Lx² − x³) / L²            （由 −a²·b/L²）

    **写成多项式而不是手推原函数**，是因为分布荷载的固端力就是"核乘以强度
    再积分"，而强度一旦是线性的，被积函数升到五次——手推八个原函数出错的
    概率远高于让 numpy 去积。测试里另拿数值积分独立对一遍。
    """
    P = np.polynomial.Polynomial
    return (P([L ** 3, 0.0, -3.0 * L, 2.0]) / L ** 3,
            P([0.0, 0.0, 3.0 * L, -2.0]) / L ** 3,
            P([0.0, L ** 2, -2.0 * L, 1.0]) / L ** 2,
            -P([0.0, 0.0, L, -1.0]) / L ** 2)


def _ramp_fixed_end(L: float, w1: float, w2: float, a: float,
                    b: float) -> tuple[float, float, float, float]:
    """[a, b] 上强度由 w1 线性变到 w2 的固端力 (V_i, V_j, M_i, M_j)。

    强度写成 w(x) = (w1 − s·a) + s·x，其中 s = (w2 − w1)/(b − a)；
    于是每一项都是 ∫ₐᵇ w(x)·核(x) dx，两个多项式相乘再积分。

    这一式同时覆盖已有的两种：w1 = w2 时退回 partial，
    a=0、b=L 时退回 trapezoid。测试拿这两条当自校验。
    """
    span = b - a
    if span <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    s = (w2 - w1) / span
    intensity = np.polynomial.Polynomial([w1 - s * a, s])
    out = []
    for kernel in _point_kernels(L):
        anti = (intensity * kernel).integ()
        out.append(float(anti(b) - anti(a)))
    return tuple(out)                                        # type: ignore[return-value]


def fixed_end(load: SpanLoad, L: float, rot: np.ndarray) -> np.ndarray:
    """一项荷载的等效节点荷载向量（12,），局部坐标。

    rot 是全局→局部的方向余弦矩阵。返回的排列与 frame3d 的杆端力一致：
    [Ni,Vyi,Vzi,Ti,Myi,Mzi, Nj,Vyj,Vzj,Tj,Myj,Mzj]。

    局部 y 与局部 z 两个方向的弯矩项符号相反——这一条和 internal_forces 里
    那两项的符号是同一个来源（r × F），改一处必须同时改另一处。
    """
    p = np.zeros(12)
    if load.kind == POINT:
        px, py, pz = rot @ np.asarray(load.w1, dtype=float)
        a = float(load.a)
        b = L - a
        p[0], p[6] = px * b / L, px * a / L
        vi, vj, mi, mj = _point_fixed_end(L, py, a)
        p[1], p[7], p[5], p[11] = vi, vj, mi, mj
        vi, vj, mi, mj = _point_fixed_end(L, pz, a)
        p[2], p[8], p[4], p[10] = vi, vj, -mi, -mj
        return p

    if load.kind == RAMP:
        s1 = rot @ np.asarray(load.w1, dtype=float)
        s2 = rot @ np.asarray(load.w2, dtype=float)
        a, b = float(load.a), float(load.b)
        span = b - a
        slope = (s2 - s1) / span if span > 0 else np.zeros(3)
        # 轴向：∫ₐᵇ w(x)(1−x/L)dx 与 ∫ₐᵇ w(x)·x/L dx
        axial = np.polynomial.Polynomial([s1[0] - slope[0] * a, slope[0]])
        near = (axial * np.polynomial.Polynomial([1.0, -1.0 / L])).integ()
        far = (axial * np.polynomial.Polynomial([0.0, 1.0 / L])).integ()
        p[0] = float(near(b) - near(a))
        p[6] = float(far(b) - far(a))
        vi, vj, mi, mj = _ramp_fixed_end(L, s1[1], s2[1], a, b)
        p[1], p[7], p[5], p[11] = vi, vj, mi, mj
        vi, vj, mi, mj = _ramp_fixed_end(L, s1[2], s2[2], a, b)
        p[2], p[8], p[4], p[10] = vi, vj, -mi, -mj
        return p

    if load.kind == PARTIAL:
        qx, qy, qz = rot @ np.asarray(load.w1, dtype=float)
        a, b = float(load.a), float(load.b)
        # 轴向：∫ₐᵇ w(1−x/L)dx 与 ∫ₐᵇ w·x/L dx
        span, first = b - a, (b ** 2 - a ** 2) / (2.0 * L)
        p[0] = qx * (span - first)
        p[6] = qx * first
        vi, vj, mi, mj = _partial_fixed_end(L, qy, a, b)
        p[1], p[7], p[5], p[11] = vi, vj, mi, mj
        vi, vj, mi, mj = _partial_fixed_end(L, qz, a, b)
        p[2], p[8], p[4], p[10] = vi, vj, -mi, -mj
        return p

    v1, v2 = load.ends()
    q1 = rot @ v1
    q2 = rot @ v2
    # 轴向：∫w(1−x/L) 与 ∫w·x/L，梯形下分别是 qL/2 + ΔL/6 与 qL/2 + ΔL/3
    da = q2[0] - q1[0]
    p[0] = q1[0] * L / 2.0 + da * L / 6.0
    p[6] = q1[0] * L / 2.0 + da * L / 3.0
    vi, vj, mi, mj = _bending_fixed_end(L, q1[1], q2[1])
    p[1], p[7], p[5], p[11] = vi, vj, mi, mj
    vi, vj, mi, mj = _bending_fixed_end(L, q1[2], q2[2])
    p[2], p[8], p[4], p[10] = vi, vj, -mi, -mj
    return p


def span_force(load: SpanLoad, L: float, rot: np.ndarray,
               x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """左段 [0, x] 上该荷载的合力与对截面 x 的合力矩，局部坐标。

    返回 (S, M)，形状都是 (3, len(x))：

        S(x) = ∫₀ˣ w(ξ) dξ                合力
        M(x) = ∫₀ˣ w(ξ)·(x − ξ) dξ        对截面 x 取矩的力臂积分

    内力回算就用这两项——均布时 S = w·x、M = w·x²/2，正是原来那套公式。
    """
    x = np.asarray(x, dtype=float)
    if load.kind == POINT:
        p_local = rot @ np.asarray(load.w1, dtype=float)
        # 集中力处剪力有跳跃。取 x > a 为已跨过，与"截面切在荷载右侧"一致
        past = (x > load.a).astype(float)
        S = np.outer(p_local, past)
        M = np.outer(p_local, past * (x - load.a))
        return S, M

    if load.kind == RAMP:
        q1 = rot @ np.asarray(load.w1, dtype=float)
        q2 = rot @ np.asarray(load.w2, dtype=float)
        a, b = float(load.a), float(load.b)
        span = b - a
        slope = (q2 - q1) / span if span > 0 else np.zeros(3)
        u = np.maximum(np.clip(x, a, b) - a, 0.0)   # 已覆盖长度
        c = x - a
        # S = ∫₀ᵘ (w1 + s·v)dv；M = ∫₀ᵘ (w1 + s·v)(c − v)dv
        S = np.outer(q1, u) + np.outer(slope, u ** 2 / 2.0)
        M = (np.outer(q1, c * u - u ** 2 / 2.0)
             + np.outer(slope, c * u ** 2 / 2.0 - u ** 3 / 3.0))
        return S, M

    if load.kind == PARTIAL:
        q = rot @ np.asarray(load.w1, dtype=float)
        a, b = float(load.a), float(load.b)
        # 截面走到 a 之前什么都没有；越过 b 之后荷载不再增加，所以先把
        # 积分上限夹到 [a, b]，再套均布的那两条式子。
        t = np.clip(x, a, b)
        covered = np.maximum(t - a, 0.0)
        S = np.outer(q, covered)
        # ∫ₐᵗ (x−ξ)dξ = x(t−a) − (t²−a²)/2
        arm = x * covered - (t ** 2 - a ** 2) / 2.0
        M = np.outer(q, np.maximum(arm, 0.0))
        return S, M

    v1, v2 = load.ends()
    q1 = rot @ v1
    d = (rot @ v2) - q1
    S = np.outer(q1, x) + np.outer(d, x ** 2 / (2.0 * L))
    M = np.outer(q1, x ** 2 / 2.0) + np.outer(d, x ** 3 / (6.0 * L))
    return S, M


def resultant(load: SpanLoad, L: float) -> tuple[np.ndarray, float]:
    """整根杆上的荷载合力（全局）与其作用点距 i 端的距离。

    整体平衡校核用。合力为零时（例如反对称的梯形荷载）作用点没有意义，
    返回杆件中点——那种情况下合力矩由调用方按分量单独算，不靠这个。
    """
    if load.kind == POINT:
        return np.asarray(load.w1, dtype=float), float(load.a)
    if load.kind == RAMP:
        v1 = np.asarray(load.w1, dtype=float)
        v2 = np.asarray(load.w2, dtype=float)
        span = float(load.b - load.a)
        total = (v1 + v2) * span / 2.0
        k = int(np.argmax(np.abs(v1 + v2)))
        s = v1[k] + v2[k]
        xbar = (load.a + span * (v1[k] + 2.0 * v2[k]) / (3.0 * s)
                if abs(s) > 0 else (load.a + load.b) / 2.0)
        return total, float(xbar)

    if load.kind == PARTIAL:
        # 合力 = 强度 × 作用长度，作用点在这一段的中点
        return (np.asarray(load.w1, dtype=float) * (load.b - load.a),
                float((load.a + load.b) / 2.0))
    v1, v2 = load.ends()
    total = (v1 + v2) * L / 2.0
    # 梯形形心：x̄ = L(w1 + 2w2) / (3(w1 + w2))，逐分量的形心可能不同，
    # 只有合力共线时才有单一作用点，故按最大分量定形心
    k = int(np.argmax(np.abs(v1 + v2)))
    s = v1[k] + v2[k]
    xbar = L * (v1[k] + 2.0 * v2[k]) / (3.0 * s) if abs(s) > 0 else L / 2.0
    return total, float(xbar)


def moment_about_origin(load: SpanLoad, L: float, pi: np.ndarray,
                        pj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全局坐标下该荷载的合力，以及它对原点的合力矩。

    整体平衡校核直接用这两个量，不必再关心荷载是什么形状。
    """
    axis = (pj - pi) / L
    if load.kind == POINT:
        force = np.asarray(load.w1, dtype=float)
        point = pi + axis * load.a
        return force, np.cross(point, force)

    if load.kind == RAMP:
        v1 = np.asarray(load.w1, dtype=float)
        v2 = np.asarray(load.w2, dtype=float)
        a, b = float(load.a), float(load.b)
        span = b - a
        force = (v1 + v2) * span / 2.0
        slope = (v2 - v1) / span if span > 0 else np.zeros(3)
        # ∫ₐᵇ x·w(x)dx，w(x) = v1 + slope·(x−a)
        first = (v1 * (b ** 2 - a ** 2) / 2.0
                 + slope * ((b ** 3 - a ** 3) / 3.0
                            - a * (b ** 2 - a ** 2) / 2.0))
        return force, np.cross(pi, force) + np.cross(axis, first)

    if load.kind == PARTIAL:
        w = np.asarray(load.w1, dtype=float)
        a, b = float(load.a), float(load.b)
        force = w * (b - a)
        first = w * (b ** 2 - a ** 2) / 2.0        # ∫ₐᵇ s·w ds
        return force, np.cross(pi, force) + np.cross(axis, first)

    v1, v2 = load.ends()
    # 逐分量积分：∫ w_k(s) ds 与 ∫ (pi + axis·s) × w(s) ds
    force = (v1 + v2) * L / 2.0
    # ∫ s·w_k(s) ds = L²(w1/6 + w2/3)
    first = (v1 / 6.0 + v2 / 3.0) * L ** 2
    moment = np.cross(pi, force) + np.cross(axis, first)
    return force, moment
