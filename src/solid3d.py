"""小变形三维线弹性实体有限元内核。

求解器只实现课程作业当前需要的一个单元：10 节点二次四面体（C3D10）。
网格生成属于独立问题，由上层提供节点和连接表；这里负责形函数、积分、
本构、稀疏组装、边界条件、求解和应力恢复。

约定：应变/应力向量为 ``[xx, yy, zz, xy, yz, xz]``，后三项应变采用工程
剪应变。长度和弹性模量可使用任意一致单位；节点实体模块使用 mm / MPa / N。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class Solid3DError(RuntimeError):
    """实体网格或求解条件无效。"""


@dataclass(frozen=True)
class SolidMesh:
    nodes: np.ndarray              # (n, 3)
    elements: np.ndarray           # (m, 10)，零基节点编号

    def __post_init__(self) -> None:
        xyz = np.asarray(self.nodes, dtype=float)
        conn = np.asarray(self.elements, dtype=np.int64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise Solid3DError("实体节点坐标必须是 n×3 数组")
        if conn.ndim != 2 or conn.shape[1] != 10:
            raise Solid3DError("C3D10 连接表必须是 m×10 数组")
        if not np.all(np.isfinite(xyz)):
            raise Solid3DError("实体节点坐标含非有限数")
        if conn.size and (conn.min() < 0 or conn.max() >= len(xyz)):
            raise Solid3DError("实体单元引用了不存在的节点")
        object.__setattr__(self, "nodes", xyz)
        object.__setattr__(self, "elements", conn)


@dataclass(frozen=True)
class SolidResult:
    displacement: np.ndarray       # (n, 3)
    reaction: np.ndarray           # (n, 3)
    element_mises: np.ndarray      # (m,)
    element_abs_principal: np.ndarray
    gauss_stress: np.ndarray       # (m, 4, 6)
    residual_norm: float


# 角点之后依次是边 01, 12, 20, 03, 13, 23；与 VTK quadratic tetra 一致。
EDGE_PAIRS = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))

_A = 0.5854101966249685
_B = 0.1381966011250105
GAUSS_BARYCENTRIC = np.array([
    [_A, _B, _B, _B],
    [_B, _A, _B, _B],
    [_B, _B, _A, _B],
    [_B, _B, _B, _A],
], dtype=float)
GAUSS_WEIGHT = 1.0 / 24.0

# 每个二次三角形面的 3 个角点 + 3 个边中点（局部节点号）。
TET10_FACES = ((1, 2, 3, 5, 9, 8),
               (0, 2, 3, 6, 9, 7),
               (0, 1, 3, 4, 8, 7),
               (0, 1, 2, 4, 5, 6))


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """三维各向同性 Hooke 矩阵（工程剪应变约定）。"""
    E, nu = float(E), float(nu)
    if not math.isfinite(E) or E <= 0.0:
        raise Solid3DError("弹性模量 E 必须为正数")
    if not math.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise Solid3DError("泊松比必须满足 -1 < nu < 0.5")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6), dtype=float)
    D[:3, :3] = lam
    np.fill_diagonal(D[:3, :3], lam + 2.0 * mu)
    D[3:, 3:] = np.eye(3) * mu
    return D


def shape_tet10(barycentric: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """返回形函数 N 和对自然坐标 (r,s,t) 的导数。"""
    L = np.asarray(tuple(barycentric), dtype=float)
    if L.shape != (4,):
        raise Solid3DError("四面体重心坐标必须有四项")
    dL = np.array([[-1.0, -1.0, -1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
    N = np.empty(10, dtype=float)
    dN = np.empty((10, 3), dtype=float)
    for i in range(4):
        N[i] = L[i] * (2.0 * L[i] - 1.0)
        dN[i] = (4.0 * L[i] - 1.0) * dL[i]
    for k, (i, j) in enumerate(EDGE_PAIRS, start=4):
        N[k] = 4.0 * L[i] * L[j]
        dN[k] = 4.0 * (dL[i] * L[j] + L[i] * dL[j])
    return N, dN


def _b_matrix(coords: np.ndarray, barycentric: np.ndarray) -> tuple[np.ndarray, float]:
    _N, dN_nat = shape_tet10(barycentric)
    jacobian = coords.T @ dN_nat
    det_j = float(np.linalg.det(jacobian))
    if not math.isfinite(det_j) or det_j <= 1e-12:
        raise Solid3DError(f"C3D10 Jacobian 非正或退化：detJ={det_j:.6g}")
    dN = dN_nat @ np.linalg.inv(jacobian)
    B = np.zeros((6, 30), dtype=float)
    for i, (dx, dy, dz) in enumerate(dN):
        c = 3 * i
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c:c + 2] = (dy, dx)
        B[4, c + 1:c + 3] = (dz, dy)
        B[5, (c, c + 2)] = (dz, dx)
    return B, det_j


def element_stiffness(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """30×30 C3D10 单元刚度，采用四点二阶精确积分。"""
    xyz = np.asarray(coords, dtype=float)
    if xyz.shape != (10, 3):
        raise Solid3DError("单元坐标必须是 10×3")
    D = elasticity_matrix(E, nu)
    Ke = np.zeros((30, 30), dtype=float)
    for bary in GAUSS_BARYCENTRIC:
        B, det_j = _b_matrix(xyz, bary)
        Ke += (B.T @ D @ B) * det_j * GAUSS_WEIGHT
    return 0.5 * (Ke + Ke.T)


def assemble(mesh: SolidMesh, E: float, nu: float,
             chunk_size: int = 250) -> csr_matrix:
    """分块 COO→CSR 组装，避免一次保存 m×900 个索引。"""
    ndof = 3 * len(mesh.nodes)
    K = csr_matrix((ndof, ndof), dtype=float)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    def flush() -> None:
        nonlocal K
        if not data:
            return
        block = coo_matrix((np.concatenate(data),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(ndof, ndof)).tocsr()
        K = K + block
        rows.clear(); cols.clear(); data.clear()

    for index, conn in enumerate(mesh.elements):
        dofs = np.column_stack((3 * conn, 3 * conn + 1, 3 * conn + 2)).ravel()
        Ke = element_stiffness(mesh.nodes[conn], E, nu)
        rows.append(np.repeat(dofs, 30))
        cols.append(np.tile(dofs, 30))
        data.append(Ke.ravel())
        if (index + 1) % int(chunk_size) == 0:
            flush()
    flush()
    return K


def equivalent_nodal_wrench(points: np.ndarray, force: Iterable[float],
                             moment: Iterable[float],
                             origin: Iterable[float]) -> np.ndarray:
    """将截面六分量合力变成最小二乘意义下最均匀的节点力。

    返回的节点力严格满足 ``sum(f)=F`` 和 ``sum(r×f)=M``。这不是刚性耦合，
    但切割面离热点区足够远时符合 Saint-Venant 原理，并避免引入额外自由度。
    """
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
        raise Solid3DError("切割面至少需要三个不共线节点")
    rel = xyz - np.asarray(tuple(origin), dtype=float)
    A = np.zeros((6, 3 * len(xyz)), dtype=float)
    for i, (x, y, z) in enumerate(rel):
        A[:3, 3 * i:3 * i + 3] = np.eye(3)
        A[3:, 3 * i:3 * i + 3] = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    if np.linalg.matrix_rank(A) < 6:
        raise Solid3DError("切割面节点不能传递完整六分量合力")
    wrench = np.r_[np.asarray(tuple(force), dtype=float),
                   np.asarray(tuple(moment), dtype=float)]
    nodal = A.T @ np.linalg.solve(A @ A.T, wrench)
    return nodal.reshape((-1, 3))


def _stress_measures(stress: np.ndarray) -> tuple[float, float]:
    sx, sy, sz, txy, tyz, txz = stress
    tensor = np.array([[sx, txy, txz],
                       [txy, sy, tyz],
                       [txz, tyz, sz]])
    principal = np.linalg.eigvalsh(tensor)
    mises = math.sqrt(max(0.0, 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2
                                      + (sz - sx) ** 2)
                                + 3.0 * (txy ** 2 + tyz ** 2 + txz ** 2)))
    return float(mises), float(np.max(np.abs(principal)))


def extrapolate_element_nodal_stress(gauss_stress: np.ndarray) -> np.ndarray:
    """将四个积分点应力线性外推到 C3D10 的十个节点。

    四点积分恰好确定重心坐标中的线性应力场；六个边中点取两端角点外推值的
    平均。节点共享时由调用方决定平均或取包络。
    """
    stress = np.asarray(gauss_stress, dtype=float)
    if stress.shape != (4, 6):
        raise Solid3DError("C3D10 积分点应力必须是 4×6")
    corner = np.linalg.solve(GAUSS_BARYCENTRIC, stress)
    nodal = np.empty((10, 6), dtype=float)
    nodal[:4] = corner
    for k, (i, j) in enumerate(EDGE_PAIRS, start=4):
        nodal[k] = 0.5 * (corner[i] + corner[j])
    return nodal


def nodal_abs_principal_envelope(mesh: SolidMesh,
                                 gauss_stress: np.ndarray) -> np.ndarray:
    """共享节点取相邻单元外推绝对主应力的最大值。"""
    field = np.asarray(gauss_stress, dtype=float)
    if field.shape != (len(mesh.elements), 4, 6):
        raise Solid3DError("积分点应力数量与实体网格不一致")
    out = np.full(len(mesh.nodes), np.nan, dtype=float)
    for conn, element_field in zip(mesh.elements, field, strict=True):
        for node, stress in zip(conn, extrapolate_element_nodal_stress(element_field), strict=True):
            _vm, principal = _stress_measures(stress)
            if math.isnan(out[node]) or principal > out[node]:
                out[node] = principal
    return out


def boundary_nodes(mesh: SolidMesh) -> np.ndarray:
    """由只出现一次的角点面识别二次四面体外表面节点。"""
    faces: dict[tuple[int, int, int], tuple[int, ...] | None] = {}
    for conn in mesh.elements:
        for local in TET10_FACES:
            nodes = tuple(int(conn[i]) for i in local)
            key = tuple(sorted(nodes[:3]))
            faces[key] = nodes if key not in faces else None
    boundary: set[int] = set()
    for nodes in faces.values():
        if nodes is not None:
            boundary.update(nodes)
    return np.asarray(sorted(boundary), dtype=np.int64)


def solve(mesh: SolidMesh, E: float, nu: float,
          fixed_nodes: Iterable[int], nodal_loads: np.ndarray,
          *, chunk_size: int = 250) -> SolidResult:
    """组装并求解线弹性实体模型。节点编号均为零基。"""
    loads = np.asarray(nodal_loads, dtype=float)
    if loads.shape != mesh.nodes.shape:
        raise Solid3DError("节点荷载必须与节点坐标同为 n×3")
    fixed = np.unique(np.asarray(tuple(fixed_nodes), dtype=np.int64))
    if fixed.size == 0 or fixed.min() < 0 or fixed.max() >= len(mesh.nodes):
        raise Solid3DError("固定节点集合为空或含无效编号")
    K = assemble(mesh, E, nu, chunk_size=chunk_size)
    f = loads.ravel()
    fixed_dofs = np.column_stack((3 * fixed, 3 * fixed + 1, 3 * fixed + 2)).ravel()
    free = np.setdiff1d(np.arange(len(f), dtype=np.int64), fixed_dofs,
                        assume_unique=False)
    if free.size == 0:
        raise Solid3DError("模型没有自由自由度")
    u = np.zeros_like(f)
    u[free] = spsolve(K[free][:, free].tocsc(), f[free])
    if not np.all(np.isfinite(u)):
        raise Solid3DError("实体刚度矩阵奇异或求解结果非有限")
    residual = K @ u - f
    free_residual = float(np.linalg.norm(residual[free]))

    D = elasticity_matrix(E, nu)
    stresses = np.empty((len(mesh.elements), 4, 6), dtype=float)
    mises = np.empty(len(mesh.elements), dtype=float)
    principal = np.empty(len(mesh.elements), dtype=float)
    for e, conn in enumerate(mesh.elements):
        ue = u[np.column_stack((3 * conn, 3 * conn + 1, 3 * conn + 2)).ravel()]
        em, ep = [], []
        for q, bary in enumerate(GAUSS_BARYCENTRIC):
            B, _det = _b_matrix(mesh.nodes[conn], bary)
            stresses[e, q] = D @ (B @ ue)
            vm, pr = _stress_measures(stresses[e, q])
            em.append(vm); ep.append(pr)
        mises[e] = max(em)
        principal[e] = max(ep)
    return SolidResult(
        displacement=u.reshape((-1, 3)),
        reaction=residual.reshape((-1, 3)),
        element_mises=mises,
        element_abs_principal=principal,
        gauss_stress=stresses,
        residual_norm=free_residual,
    )
