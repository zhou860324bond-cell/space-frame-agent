"""线性屈曲：临界荷载因子与失稳模态。

刚架最核心的稳定问题。参照项目 FEM-Python 没有这一块——它只做线性静力，
连几何刚度矩阵都没有。

做法：先按参考荷载做一次静力分析，拿到各杆轴力 N；用这些轴力组装
**几何刚度矩阵** K_g（受压杆件会削弱结构的抗侧能力，这一项就是那个削弱）；
然后解

    (K + λ K_g) φ = 0

λ 就是临界荷载因子：把参考荷载放大 λ 倍，结构失稳。λ = 3 意味着还有三倍余量。

数值上不直接解上式，而是改写成对称正定的广义特征值问题

    K_g φ = μ K φ,   λ = −1/μ

因为静力可解意味着 K 正定，拿它当右端矩阵最稳；K_g 本身是不定的
（有拉有压），直接放右端会让 `eigh` 报错。

**关键前提：这是线性（特征值）屈曲。** 它假定失稳前结构保持线弹性、
变形小、轴力不随变形改变。真实钢结构有初始缺陷和残余应力，实际承载力
低于这个值——所以它给的是上限，不能直接当承载力用。

一致的几何刚度阵与一致质量阵同理，**从上方**逼近精确解，加密时单调下降。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from frame3d import (Frame, _member_matrices, assemble, constrained_dofs,
                     project_secondary_matrix, scatter_blocks, solve)
from modal import DENSE_EIGEN_LIMIT


@dataclass
class BucklingResult:
    """线性屈曲结果。

    factors 是临界荷载因子，升序；shapes 每列一个失稳模态（全自由度）。
    axial 是参考荷载下各杆的轴力，受拉为正——看是哪根压杆先失稳就靠它。
    """
    factors: np.ndarray              # (n_modes,)
    shapes: np.ndarray               # (n_dofs, n_modes)
    case: str
    axial: dict[int, float]

    @property
    def critical(self) -> float:
        return float(self.factors[0])


def geometric_stiffness(L: float, N: float, Ip_over_A: float) -> np.ndarray:
    """一根杆件的几何刚度矩阵（12×12，局部坐标）。

    N 是轴力，**受拉为正**。受拉时 N > 0，K_g 使结构变硬（绷紧的琴弦更难弯）；
    受压时 N < 0，K_g 削弱抗弯能力，压到一定程度整体刚度奇异——那就是失稳。

    弯曲两个平面各一块，转角项符号相反，与刚度阵、质量阵、固端力里那几处
    同源（r × F）。扭转项按 N·Ip/A 取，影响弯扭失稳。
    """
    kg = np.zeros((12, 12))
    c = N / (30.0 * L)
    blk = c * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L],
                        [3.0 * L, 4.0 * L ** 2, -3.0 * L, -L ** 2],
                        [-36.0, -3.0 * L, 36.0, -3.0 * L],
                        [3.0 * L, -L ** 2, -3.0 * L, 4.0 * L ** 2]])

    # (v, θz)：自由度 1, 5, 7, 11
    idx = (1, 5, 7, 11)
    for a in range(4):
        for b in range(4):
            kg[idx[a], idx[b]] += blk[a, b]

    # (w, θy)：自由度 2, 4, 8, 10，转角项符号相反
    idx = (2, 4, 8, 10)
    sign = np.array([1.0, -1.0, 1.0, -1.0])
    for a in range(4):
        for b in range(4):
            kg[idx[a], idx[b]] += sign[a] * sign[b] * blk[a, b]

    # 扭转
    t = N * Ip_over_A / L
    kg[3, 3] += t
    kg[9, 9] += t
    kg[3, 9] -= t
    kg[9, 3] -= t
    return kg


def assemble_geometric(model: Frame, axial: dict[int, float]) -> coo_matrix:
    """按给定的杆件轴力组装整体几何刚度矩阵。"""
    all_dofs, blocks = [], []
    for mid, m in model.members.items():
        N = float(axial.get(mid, 0.0))
        if N == 0.0:
            continue
        L, R, T, B, _, _, cond = _member_matrices(model, m)
        sec = model.sections[m.section]
        kg = geometric_stiffness(L, N, (sec.Iy + sec.Iz) / sec.A)
        kg = project_secondary_matrix(kg, cond)
        all_dofs.append(model.node_dofs(m.i) + model.node_dofs(m.j))
        blocks.append(B.T @ T.T @ kg @ T @ B)
    return scatter_blocks(model.num_dofs, all_dofs, blocks)


def buckling(model: Frame, case: str | None = None,
             num_modes: int = 4, solution=None) -> BucklingResult:
    """线性屈曲分析。case 指定参考荷载工况，λ 是它的放大倍数。"""
    if not model.members:
        raise ValueError("模型里没有杆件，无法做屈曲分析")
    sol = solution if solution is not None else solve(model)
    name = case or sol.primary
    try:
        res = sol[name]
    except KeyError:
        raise ValueError(
            f"没有名为 {name!r} 的工况或组合，"
            f"现有 {sorted(sol.all_results())}") from None

    # 受拉为正：杆端力向量的第 7 个分量（j 端轴力）
    axial = {mid: float(f[6]) for mid, f in res.member_forces.items()}
    if all(N >= 0 for N in axial.values()):
        raise ValueError(
            f"工况 {name!r} 下没有受压杆件，不存在屈曲问题。"
            "检查荷载方向——受拉的结构不会失稳。")

    K, _, _ = assemble(model)
    Kg = assemble_geometric(model, axial)

    n = model.num_dofs
    fixed = constrained_dofs(model)
    free = np.setdiff1d(np.arange(n), fixed)
    if free.size == 0:
        raise ValueError("所有自由度都被约束了，不存在屈曲模态")

    Kff = K[free][:, free]
    Gff = Kg[free][:, free]
    Kff = 0.5 * (Kff + Kff.T)
    Gff = 0.5 * (Gff + Gff.T)

    # K_g φ = μ K φ，λ = −1/μ。K 正定所以放右端，K_g 不定不能放
    if free.size <= DENSE_EIGEN_LIMIT or num_modes >= free.size - 1:
        mu, vecs = eigh(Gff.toarray(), Kff.toarray())
    else:
        # 最小的正 λ 对应最负的 μ，正好是谱的一端，普通广义模式（K 放右端、
        # 内部用 splu 分解）收敛就快，不必移频。K 正定这一要求与稠密路径相同。
        try:
            mu, vecs = eigsh(Gff.tocsc(), k=num_modes, M=Kff.tocsc(), which="SA")
        except RuntimeError as exc:
            raise np.linalg.LinAlgError(
                "屈曲分析失败：刚度矩阵奇异，约束不足或存在机构") from exc
    with np.errstate(divide="ignore"):
        lam = np.where(np.abs(mu) > 0, -1.0 / mu, np.inf)

    # 只要正的 λ：负的对应"把荷载反向才失稳"，不是这个工况的临界值
    order = [k for k in np.argsort(lam) if np.isfinite(lam[k]) and lam[k] > 0]
    if not order:
        raise ValueError("没有找到正的临界荷载因子，检查荷载方向与约束")
    order = order[:num_modes]

    shapes = np.zeros((n, len(order)))
    shapes[free, :] = vecs[:, order]
    return BucklingResult(factors=lam[order], shapes=shapes, case=name,
                          axial=axial)
