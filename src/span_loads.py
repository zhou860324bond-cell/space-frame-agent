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
KINDS = (UNIFORM, TRAPEZOID, POINT)


@dataclass(frozen=True)
class SpanLoad:
    """一项杆间荷载。全局坐标。

    uniform:    w1 是满跨强度（当前力/长度单位），w2 与 a 不用
    trapezoid:  w1 是 i 端强度、w2 是 j 端强度（当前力/长度单位）
    point:      w1 是集中力（当前力单位），a 是距 i 端的距离（当前长度单位）
    """
    kind: str
    w1: tuple[float, float, float]
    w2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    a: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"杆间荷载类型只能取 {KINDS}，收到 {self.kind!r}")

    def scaled(self, factor: float) -> "SpanLoad":
        """按系数缩放。荷载组合按此逐项合成，位置 a 不缩放。"""
        return SpanLoad(self.kind,
                        tuple(factor * v for v in self.w1),
                        tuple(factor * v for v in self.w2),
                        self.a)

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
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SpanLoad":
        kind = str(data.get("kind", UNIFORM))
        if kind == TRAPEZOID and "w2" not in data:
            # 缺 w2 就默认成 w1 的话，写漏的梯形会静悄悄变成均布；
            # 默认成 0 又会静悄悄变成三角形。两种都不能接受
            raise ValueError("梯形荷载必须同时给出 w1 与 w2")
        return SpanLoad(kind,
                        tuple(float(v) for v in data["w1"]),
                        tuple(float(v) for v in data.get("w2", (0.0, 0.0, 0.0))),
                        float(data.get("a", 0.0)))


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

    v1, v2 = load.ends()
    # 逐分量积分：∫ w_k(s) ds 与 ∫ (pi + axis·s) × w(s) ds
    force = (v1 + v2) * L / 2.0
    # ∫ s·w_k(s) ds = L²(w1/6 + w2/3)
    first = (v1 / 6.0 + v2 / 3.0) * L ** 2
    moment = np.cross(pi, force) + np.cross(axis, first)
    return force, moment
