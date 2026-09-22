"""模态分析：自振频率与振型。

参照项目 FEM-Python 只有线性静力，没有质量矩阵、没有特征值求解。
这一块是我们比它多出来的能力，而且验证起来一点不含糊——
悬臂梁、简支梁的频率都有闭合解。

做法是标准的：解广义特征值问题

    K φ = ω² M φ

只在自由自由度上解（支座行列先划掉），得到圆频率 ω，频率 f = ω / 2π。

**一致质量矩阵**（consistent mass）而不是集中质量：它由与刚度阵同一套
形函数积分得到，收敛快得多。实测每跨四个单元时基频误差 0.2% 以内
（悬臂 +0.007%、简支 +0.026%、两端固接 +0.19%）。
最省的一档是悬臂一个单元 +0.47%，但同样一个单元的简支要差 11%——
边界越"硬"形函数越不够用，所以别一概而论说"一个单元就够"。
代价是矩阵不再是对角阵，但我们本来就在解广义特征值，无所谓。

一致质量阵会**高估**频率（形函数比真实振型硬），集中质量则低估——
测试里对闭合解的偏差方向因此是可预期的，这本身也是一条判据。

质量来自材料密度 ρ 与截面面积 A。密度为零的材料不产生质量，
所以老模型（没写 density）调模态会直接报错而不是给出一个零质量的荒唐结果。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

from frame3d import (DOF_PER_NODE, Frame, _member_matrices, assemble,
                     constrained_dofs, member_endpoints,
                     member_local_displacements, project_secondary_matrix)

# 一致质量矩阵在扭转项上要用极惯性矩 / 面积，这里按 (Iy + Iz) / A 取，
# 对常见截面就是极回转半径的平方
_MIN_MODES = 1


@dataclass
class ModalResult:
    """模态分析结果。frequencies 单位 Hz，shapes 每列是一个振型。"""
    frequencies: np.ndarray          # (n_modes,) Hz
    omega: np.ndarray                # (n_modes,) rad/s
    shapes: np.ndarray               # (n_dofs, n_modes) 全自由度，支座行为 0
    total_mass: float                # Σ ρAL，结构的全部质量
    effective_mass: np.ndarray       # (n_modes, 3) 三个平动方向的有效质量
    #: (3,) 各方向**能参与振动**的质量 rᵀM_ff r。压在支座上的那部分不在内，
    #: 所以它总是小于 total_mass——判"阶数取够没有"要拿它当分母。
    participable_mass: np.ndarray = None      # type: ignore[assignment]
    #: (n_modes, 3) 参与系数 Γ = φᵀMr / φᵀMφ。反应谱用它把谱位移放大成振型贡献。
    participation: np.ndarray = None          # type: ignore[assignment]
    #: (n_modes,) 广义模态质量 φᵀMφ。振型的归一化方式不同它就不同，
    #: 所以任何用到振型幅值的公式都得带上它，不能默认为 1。
    modal_mass: np.ndarray = None             # type: ignore[assignment]

    @property
    def periods(self) -> np.ndarray:
        return 1.0 / self.frequencies

    @property
    def mass_ratio(self) -> np.ndarray:
        """(n_modes, 3) 各阶的参与质量比，分母是**能参与的**质量。

        拿 total_mass 当分母是错的：支座上那部分永远参与不了，比值上不去
        100%，而用户会以为阶数不够。
        """
        return self.effective_mass / self.participable_mass

    @property
    def cumulative_ratio(self) -> np.ndarray:
        """(n_modes, 3) 累计参与质量比。规范看的是这个。"""
        return np.cumsum(self.mass_ratio, axis=0)


def consistent_mass(L: float, rho: float, A: float, Ip_over_A: float) -> np.ndarray:
    """三维梁单元的一致质量矩阵（12×12，局部坐标）。

    轴向与扭转是各自独立的一维问题，弯曲两个方向用 Hermite 形函数积分。
    绕局部 y 弯曲（即 z 向位移）那一块的耦合项符号与绕 z 的相反——
    和刚度阵、和固端力里那两处是同一个来源，改一处要同时改三处。
    """
    m = rho * A * L
    M = np.zeros((12, 12))

    # 轴向：[[2,1],[1,2]] * m/6
    for a, b in ((0, 6),):
        M[a, a] = M[b, b] = 2.0 * m / 6.0
        M[a, b] = M[b, a] = m / 6.0
    # 扭转：同样的形状，惯量用 ρ·Ip·L
    j = rho * A * Ip_over_A * L
    M[3, 3] = M[9, 9] = 2.0 * j / 6.0
    M[3, 9] = M[9, 3] = j / 6.0

    # 弯曲：经典 Hermite 一致质量阵，顺序 (v_i, θ_i, v_j, θ_j)
    blk = (m / 420.0) * np.array(
        [[156.0, 22.0 * L, 54.0, -13.0 * L],
         [22.0 * L, 4.0 * L ** 2, 13.0 * L, -3.0 * L ** 2],
         [54.0, 13.0 * L, 156.0, -22.0 * L],
         [-13.0 * L, -3.0 * L ** 2, -22.0 * L, 4.0 * L ** 2]])

    # (v, θz) 在局部 y 方向：自由度 1, 5, 7, 11
    idx = (1, 5, 7, 11)
    for a in range(4):
        for b in range(4):
            M[idx[a], idx[b]] += blk[a, b]

    # (w, θy) 在局部 z 方向：自由度 2, 4, 8, 10。转角项符号相反
    idx = (2, 4, 8, 10)
    sign = np.array([1.0, -1.0, 1.0, -1.0])
    for a in range(4):
        for b in range(4):
            M[idx[a], idx[b]] += sign[a] * sign[b] * blk[a, b]
    return M


def assemble_mass(model: Frame) -> coo_matrix:
    """整体一致质量矩阵。"""
    n = model.num_dofs
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for m in model.members.values():
        L, R, T, B, _, _, cond = _member_matrices(model, m)
        mat, sec = model.materials[m.material], model.sections[m.section]
        ip_over_a = (sec.Iy + sec.Iz) / sec.A
        Me = consistent_mass(L, mat.density, sec.A, ip_over_a)
        Me = project_secondary_matrix(Me, cond)
        Mg = B.T @ T.T @ Me @ T @ B
        dofs = model.node_dofs(m.i) + model.node_dofs(m.j)
        for a in range(12):
            for b in range(12):
                rows.append(dofs[a]); cols.append(dofs[b]); vals.append(Mg[a, b])
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def modal(model: Frame, num_modes: int = 6) -> ModalResult:
    """解 K φ = ω² M φ，返回前 num_modes 阶。

    自由度不多（几百到几千），直接用稠密 `scipy.linalg.eigh` 的广义形式——
    它对 (K, M) 都对称正定的情形是稳的，比稀疏迭代法少一堆收敛参数要调。
    真到几万自由度再换 `eigsh` 不迟，那时这个函数的签名不用变。
    """
    if not model.members:
        raise ValueError("模型里没有杆件，无法做模态分析")
    if all(not model.materials[m.material].density for m in model.members.values()):
        raise ValueError(
            "所有材料的 density 都是 0，质量矩阵为零，模态分析没有意义。"
            "给材料加上 density（kg/m³，钢约 7850）再算。")
    if num_modes < _MIN_MODES:
        raise ValueError(f"至少要求 {_MIN_MODES} 阶")

    K, _, _ = assemble(model)
    M = assemble_mass(model)

    n = model.num_dofs
    fixed = constrained_dofs(model)
    free = np.setdiff1d(np.arange(n), fixed)
    if free.size == 0:
        raise ValueError("所有自由度都被约束了，没有可振动的自由度")

    Kff = np.asarray(K[free][:, free].todense())
    Mff = np.asarray(M[free][:, free].todense())
    # 对称化：装配时的浮点误差会让两边差 1e-16 量级，eigh 要求严格对称
    Kff = 0.5 * (Kff + Kff.T)
    Mff = 0.5 * (Mff + Mff.T)

    if np.abs(Mff).max() <= 0.0:
        raise ValueError("自由自由度上的质量为零，模态分析没有意义")

    vals, vecs = eigh(Kff, Mff)
    # 数值噪声会让最低几阶出现极小的负值，截到 0 而不是让 sqrt 变 nan
    vals = np.clip(vals, 0.0, None)
    keep = min(num_modes, vals.size)
    omega = np.sqrt(vals[:keep])

    shapes = np.zeros((n, keep))
    shapes[free, :] = vecs[:, :keep]

    total = float(sum(model.materials[m.material].density
                      * model.sections[m.section].A
                      * float(np.linalg.norm(member_endpoints(model, m)[1]
                                             - member_endpoints(model, m)[0]))
                      for m in model.members.values()))

    # 有效质量：振型对刚体平动的参与程度。取够阶数时它们的和有一个**精确**
    # 的归宿，而那个归宿不是 Σ ρAL。
    #
    # 完备性关系给出 Σ_全部振型 (φᵀMr)²/(φᵀMφ) = rᵀ M_ff r，右边只含**自由**
    # 自由度。直接压在支座上的那部分质量不在里面——它永远不会参与振动。
    #
    # **这不是小数。** 一根剖成 4 段的悬臂柱，rᵀM_ff r 只有 Σ ρAL 的 84%：
    # 拿 Σ ρAL 当分母的话，把全部 24 阶取满也只报得出 84%，而用户照
    # GB 50011「参与质量 ≥90%」去判，会以为阶数不够，然后不断加阶数——
    # 加到天荒地老也过不去，因为差的那 16% 在支座上。网格越粗越明显。
    eff = np.zeros((keep, 3))
    factors = np.zeros((keep, 3))
    participable = np.zeros(3)
    for d in range(3):
        r = np.zeros(n)
        r[d::DOF_PER_NODE] = 1.0
        rf = r[free]
        participable[d] = float(rf @ Mff @ rf)
        for k in range(keep):
            phi = vecs[:, k]
            m_k = float(phi @ Mff @ phi)
            if m_k > 0:
                # 参与系数 Γ = φᵀMr / φᵀMφ；有效质量是 Γ²·φᵀMφ。
                # 两个都给：反应谱要 Γ 来放大振型，判"算够没算够"要有效质量。
                factors[k, d] = float(phi @ Mff @ rf) / m_k
                eff[k, d] = factors[k, d] ** 2 * m_k
    return ModalResult(frequencies=omega / (2.0 * np.pi), omega=omega,
                       shapes=shapes, total_mass=total, effective_mass=eff,
                       participable_mass=participable,
                       participation=factors, modal_mass=np.array(
                           [float(vecs[:, k] @ Mff @ vecs[:, k])
                            for k in range(keep)]))


def member_mode_displacement(model: Frame, shapes: np.ndarray, mode: int,
                             member_id: int, stations: int = 21
                             ) -> tuple[np.ndarray, np.ndarray]:
    """沿杆件恢复振型位移，而不是只把两个节点平移连成直线。

    轴向线性插值，两个弯曲方向使用与梁自由度符号一致的 Hermite 插值；
    杆端释放先由内核凝聚关系回算真实杆端转角，因此铰端不会错误继承节点转角。
    返回可变形梁段的未变形坐标与全局振型位移。
    """
    if stations < 2:
        raise ValueError("stations 至少为 2")
    member = model.members[member_id]
    L, R, _, _, _, _, _ = _member_matrices(model, member)
    q = member_local_displacements(model, member, shapes[:, mode])
    s = np.linspace(0.0, 1.0, stations)
    n1 = 1.0 - 3.0 * s**2 + 2.0 * s**3
    n2 = L * (s - 2.0 * s**2 + s**3)
    n3 = 3.0 * s**2 - 2.0 * s**3
    n4 = L * (-s**2 + s**3)
    axial = (1.0 - s) * q[0] + s * q[6]
    v = n1 * q[1] + n2 * q[5] + n3 * q[7] + n4 * q[11]
    # 当前局部约定中 ry 与局部 z 位移的斜率符号相反。
    w = n1 * q[2] - n2 * q[4] + n3 * q[8] - n4 * q[10]
    displacement = (np.outer(axial, R[0]) + np.outer(v, R[1])
                    + np.outer(w, R[2]))
    pi, pj = member_endpoints(model, member)
    points = pi + s[:, None] * (pj - pi)
    return points, displacement
