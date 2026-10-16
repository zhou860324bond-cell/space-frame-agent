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
from scipy.sparse import block_diag, coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import eigsh

from frame3d import (Frame, _member_matrices, _released_projection, assemble,
                     dof_basis, project_secondary_matrix, scatter_blocks, solve)
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


# --------------------------------------------------------- 弯扭屈曲（翘曲自由度）
#
# 普通梁单元每节点 6 个自由度，扭转只有 GJ，几何刚度只随轴力变化——受弯钢梁的
# 侧向弯扭失稳因此根本出不来。这里按 Vlasov 薄壁梁理论，**只在屈曲分析里**给每根
# 单元两端各加一个翘曲自由度 θ′：
#
# * 扭转改用三次 Hermite 插值，弹性刚度 GJ∫θ′² + EIw∫θ″²；
# * 几何刚度 ½∫[N(v′² + w′² + r0²θ′²) + 2Mz·θ·w″ + 2My·θ·v″]dx（Trahair 1993，
#   荷载作用在剪心）。耦合项由应力 σ = N/A − Mz·y/Iz + My·z/Iy 推出，与内核
#   截面内力符号一致；N、My、Mz 取两端截面内力沿杆线性插值，剖分越密越准。
#
# 适用范围（代码里检查，超出就拒绝）：剪心与形心重合的截面（双轴对称、点对称或
# 闭口圆截面），荷载作用在剪心；扭转自由度不得释放。线性静力仍是每节点 6 个
# 自由度，结果不受影响。

_GAUSS4 = np.polynomial.legendre.leggauss(4)
_V_DOFS, _W_DOFS, _T_DOFS = (1, 5, 7, 11), (2, 4, 8, 10), (3, 12, 9, 13)
# x–z 平面内 w′ = −θy，所以 w 的转角自由度带负号（与 geometric_stiffness 同源）
_W_SIGN = np.array([1.0, -1.0, 1.0, -1.0])


def _hermite(xi: float, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三次 Hermite 形函数及其对 x 的一、二阶导数（自由度：端值、端斜率）。"""
    H = np.array([1 - 3*xi**2 + 2*xi**3, L*(xi - 2*xi**2 + xi**3),
                  3*xi**2 - 2*xi**3, L*(-xi**2 + xi**3)])
    dH = np.array([-6*xi + 6*xi**2, L*(1 - 4*xi + 3*xi**2),
                   6*xi - 6*xi**2, L*(-2*xi + 3*xi**2)]) / L
    d2H = np.array([-6 + 12*xi, L*(-4 + 6*xi), 6 - 12*xi, L*(-2 + 6*xi)]) / L**2
    return H, dH, d2H


def _row(dofs: tuple[int, ...], values: np.ndarray) -> np.ndarray:
    r = np.zeros(14)
    r[list(dofs)] = values
    return r


def thin_walled_matrices(L: float, E: float, G: float, J: float, Iw: float,
                         r0_squared: float, axial: tuple[float, float],
                         moment_y: tuple[float, float],
                         moment_z: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """薄壁梁单元的 14×14 局部阵：(相对普通梁单元的弹性刚度修正, 几何刚度)。

    局部自由度 0–11 与普通梁单元相同，12、13 为 i、j 端翘曲 θ′。``axial``、
    ``moment_y``、``moment_z`` 是 x=0 与 x=L 处的截面内力（轴力受拉为正）。
    弹性修正 = 三次插值的 GJ、EIw 扭转阵减去普通单元的线性 GJ/L 扭转阵。
    """
    dK = np.zeros((14, 14))
    kg = np.zeros((14, 14))
    for t, weight in zip(*_GAUSS4, strict=True):
        xi = 0.5 * (t + 1.0)
        dx = 0.5 * weight * L
        H, dH, d2H = _hermite(xi, L)
        th, dth, d2th = _row(_T_DOFS, H), _row(_T_DOFS, dH), _row(_T_DOFS, d2H)
        dv, d2v = _row(_V_DOFS, dH), _row(_V_DOFS, d2H)
        dw, d2w = _row(_W_DOFS, _W_SIGN*dH), _row(_W_DOFS, _W_SIGN*d2H)
        N = axial[0]*(1 - xi) + axial[1]*xi
        My = moment_y[0]*(1 - xi) + moment_y[1]*xi
        Mz = moment_z[0]*(1 - xi) + moment_z[1]*xi
        dK += dx * (G*J*np.outer(dth, dth) + E*Iw*np.outer(d2th, d2th))
        kg += dx * (N*(np.outer(dv, dv) + np.outer(dw, dw) + r0_squared*np.outer(dth, dth))
                    + Mz*(np.outer(th, d2w) + np.outer(d2w, th))
                    + My*(np.outer(th, d2v) + np.outer(d2v, th)))
    gj = G * J / L
    dK[3, 3] -= gj
    dK[9, 9] -= gj
    dK[3, 9] += gj
    dK[9, 3] += gj
    return dK, kg


def _scatter(n: int, all_dofs: list[list[int]], blocks: list[np.ndarray]) -> csr_matrix:
    if not blocks:
        return csr_matrix((n, n))
    dofs = np.asarray(all_dofs, dtype=np.int64)
    size = dofs.shape[1]
    rows = np.repeat(dofs, size, axis=1).ravel()
    cols = np.tile(dofs, (1, size)).ravel()
    return coo_matrix((np.concatenate([b.ravel() for b in blocks]), (rows, cols)),
                      shape=(n, n)).tocsr()


def continuous_warping_groups(model: Frame,
                              same_member: dict[int, int] | None = None) -> dict[int, int]:
    """按物理条件划分翘曲连续的单元组：同一直杆内翘曲连续，转角与变截面处断开。

    两根单元在共有节点处翘曲连续的条件：轴线共线、局部 y 轴同向（翼缘对齐）、该端
    没有任何释放，并且截面相同或属于同一物理杆件（``same_member`` 给出单元 → 物理
    杆件；变截面杆件的各分段截面名不同，但杆件本身是连续的）。这覆盖了"一根直梁在
    次梁连接处被拆成几段"的常见建模；梁柱节点等方向改变处翘曲按自由处理（叉支座
    假定，偏保守），不同截面的杆件相接处同样按自由处理。
    """
    from frame3d import local_axes, member_endpoints

    parent = {eid: eid for eid in model.members}

    def find(e: int) -> int:
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    axes, ends = {}, {}
    for eid, m in model.members.items():
        pi, pj = member_endpoints(model, m)
        _, R = local_axes(pi, pj, m.ref_vector)
        axes[eid] = R
        for node, end in ((m.i, "i"), (m.j, "j")):
            free_end = not (m.releases_i if end == "i" else m.releases_j) and not (
                m.springs_i if end == "i" else m.springs_j)
            ends.setdefault(node, []).append((eid, free_end))
    for at_node in ends.values():
        for a in range(len(at_node)):
            for b in range(a + 1, len(at_node)):
                (ea, fa), (eb, fb) = at_node[a], at_node[b]
                ma, mb = model.members[ea], model.members[eb]
                same = same_member is not None and same_member.get(ea) == same_member.get(eb)
                if not (fa and fb) or (ma.section != mb.section and not same):
                    continue
                Ra, Rb = axes[ea], axes[eb]
                if abs(abs(Ra[0] @ Rb[0]) - 1.0) > 1e-9 or abs(abs(Ra[1] @ Rb[1]) - 1.0) > 1e-9:
                    continue
                parent[find(ea)] = find(eb)
    return {eid: find(eid) for eid in model.members}


def thin_walled_system(model: Frame, member_forces: dict[int, np.ndarray],
                       warping_constants: dict[str, float],
                       warping_groups: dict[int, object] | None = None,
                       warping_restrained=()) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
    """组装含翘曲自由度的整体 (K, K_g)，返回二者及应约束的翘曲自由度编号。

    翘曲自由度编号排在普通 6 个自由度之后。``warping_groups``（单元 → 组）中同组
    单元在共有节点处共用翘曲自由度；不同组在节点处互不相连，即翘曲自由。不给时按
    ``continuous_warping_groups`` 的物理条件划分。``warping_restrained`` 中的节点上，
    所有翘曲自由度被约束（如嵌固端）。
    """
    n6 = model.num_dofs
    groups = warping_groups if warping_groups is not None else continuous_warping_groups(model)
    warp: dict[tuple[object, int], int] = {}

    def warping_dof(group: object, node: int) -> int:
        return warp.setdefault((group, node), n6 + len(warp))

    all_dofs, dK_blocks, kg_blocks = [], [], []
    for eid, m in model.members.items():
        torsion = {3, 9} & (set(m.released_indices()) | set(m.spring_indices()))
        if torsion:
            raise ValueError(f"杆件 {eid} 的杆端释放了扭转（rx）。弯扭屈曲需要扭转沿杆连续传递，"
                             "暂不支持扭转释放；请去掉该释放，或不做弯扭屈曲。")
        if m.section not in warping_constants:
            raise ValueError(f"截面 {m.section!r} 没有翘曲常数，无法做弯扭屈曲。")
        L, _, T, B, _, _, cond = _member_matrices(model, m)
        sec, mat = model.sections[m.section], model.materials[m.material]
        f = np.asarray(member_forces[eid], dtype=float)
        dK, kg = thin_walled_matrices(
            L, mat.E, mat.G, sec.J, float(warping_constants[m.section]),
            (sec.Iy + sec.Iz) / sec.A, (-f[0], f[6]), (-f[4], f[10]), (-f[5], f[11]))
        P = np.eye(14)
        P[:12, :12] = _released_projection(cond)
        E = np.eye(14)
        E[:12, :12] = T @ B
        group = groups[eid]
        all_dofs.append(list(model.node_dofs(m.i)) + list(model.node_dofs(m.j))
                        + [warping_dof(group, m.i), warping_dof(group, m.j)])
        dK_blocks.append(E.T @ dK @ E)
        kg_blocks.append(E.T @ P.T @ kg @ P @ E)
    n = n6 + len(warp)
    K6, _, _ = assemble(model)
    K = block_diag([K6, csr_matrix((len(warp), len(warp)))], format="csr") \
        + _scatter(n, all_dofs, dK_blocks)
    Kg = _scatter(n, all_dofs, kg_blocks)
    restrained_nodes = {int(v) for v in warping_restrained}
    restrained = np.array(sorted(idx for (_, node), idx in warp.items()
                                 if node in restrained_nodes), dtype=int)
    return K, Kg, restrained


def buckling(model: Frame, case: str | None = None,
             num_modes: int = 4, solution=None, *,
             warping_constants: dict[str, float] | None = None,
             warping_groups: dict[int, object] | None = None,
             warping_restrained=()) -> BucklingResult:
    """线性屈曲分析。case 指定参考荷载工况，λ 是它的放大倍数。

    给出 ``warping_constants``（截面名 → Iw）即做**弯扭屈曲**：加入翘曲自由度和
    弯矩引起的弯扭耦合，见上方 ``thin_walled_system``。不给则与原来逐位一致。
    """
    lateral_torsional = warping_constants is not None
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
    # 弯扭屈曲由弯矩驱动，纯弯梁没有压杆也会失稳；只有轴压与弯矩都没有才拒绝
    bending = lateral_torsional and any(
        np.any(np.abs(np.asarray(f)[[4, 5, 10, 11]]) > 0) for f in res.member_forces.values())
    if all(N >= 0 for N in axial.values()) and not bending:
        raise ValueError(
            f"工况 {name!r} 下没有受压杆件，不存在屈曲问题。"
            "检查荷载方向——受拉的结构不会失稳。")

    basis = dof_basis(model)
    free = basis.free
    if free.size == 0:
        raise ValueError("所有自由度都被约束了，不存在屈曲模态")

    if lateral_torsional:
        K, Kg, restrained = thin_walled_system(
            model, res.member_forces, warping_constants, warping_groups, warping_restrained)
        n6, n = model.num_dofs, K.shape[0]
        warp_free = np.setdiff1d(np.arange(n6, n), restrained)
        P6 = basis.P if basis.P is not None else identity(n6, format="csr")[:, free]
        Pw = identity(n, format="csr")[:, warp_free][n6:]
        P_ext = block_diag([P6, Pw], format="csr")
        Kff = (P_ext.T @ K @ P_ext).tocsr()
        Gff = (P_ext.T @ Kg @ P_ext).tocsr()
    else:
        K, _, _ = assemble(model)
        Kg = assemble_geometric(model, axial)
        Kff = basis.reduce(K)
        Gff = basis.reduce(Kg)
    Kff = 0.5 * (Kff + Kff.T)
    Gff = 0.5 * (Gff + Gff.T)

    # K_g φ = μ K φ，λ = −1/μ。K 正定所以放右端，K_g 不定不能放
    size = Kff.shape[0]
    if size <= DENSE_EIGEN_LIMIT or num_modes >= size - 1:
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

    # 弯扭屈曲的翘曲分量只在屈曲内部使用，模态按普通 6 自由度返回
    shapes = basis.expand(vecs[:free.size, order])
    return BucklingResult(factors=lam[order], shapes=shapes, case=name,
                          axial=axial)
