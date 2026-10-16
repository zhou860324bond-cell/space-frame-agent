"""截面特性有限元求解：Saint-Venant 翘曲函数 → 扭转常数 J、翘曲常数 Iw、剪心。

为什么要算：开口薄壁截面的 J 只有近似公式，翘曲常数 Iw 和剪心位置更是没有通用公式；
而弯扭屈曲（钢梁最常见的失稳）离不开这三个量。按真实轮廓解一次二维位势问题，
任意多边形截面（含根部圆角、孔洞）都能得到同一套口径的结果。

理论（Pilkey, *Analysis and Design of Elastic Beams*, 2002, 第 6 章）：

* 以形心为坐标原点，翘曲函数 ω(y, z) 满足 ∇²ω = 0，边界上
  ∂ω/∂n = z·n_y − y·n_z。弱形式 ∫∇ω·∇δω dA = ∫(z ∂δω/∂y − y ∂δω/∂z) dA。
* J = ∫(y² + z² + y ∂ω/∂z − z ∂ω/∂y) dA。位移元给出的 J 从上方逼近真值。
* 换极点 P：ω_P = ω − z_P·y + y_P·z + c。剪心是使 ω_P 与 1、y、z 都正交的极点，
  于是剪心坐标就是 ω 在 {1, y, z} 上投影的系数；Iw = ∫ω_P² dA。

坐标沿用 ``section_geometry``：y 为轮廓第一坐标（沿截面高度 h），z 沿宽度。
单元为直边二次三角形（T6），高斯 6 点积分对本问题的被积函数精确。网格由 Gmsh
划分，刚度、荷载与求解全部在这里完成。几何先按特征长度归一化再划网格，
所以换单位制时网格完全相同，结果只差长度比例的幂次。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from section_geometry import profile

#: 默认网格尺寸 = 最小特征尺寸（壁厚、板厚或实心截面局部尺度）× 该系数。
#: T6 单元下厚度方向约 3 个单元，J 与 Iw 的离散误差在 1e-3 量级以内（见收敛测试）。
DEFAULT_SIZE_FACTOR = 0.35

# 三角形 6 点积分（Dunavant，4 阶精度）：面积坐标与权重（权重和为 1）
_A, _B = 0.445948490915965, 0.091576213509771
_WA, _WB = 0.223381589678011, 0.109951743655322
_QP = np.array([[_A, _A, 1 - 2*_A], [_A, 1 - 2*_A, _A], [1 - 2*_A, _A, _A],
                [_B, _B, 1 - 2*_B], [_B, 1 - 2*_B, _B], [1 - 2*_B, _B, _B]])
_QW = np.array([_WA, _WA, _WA, _WB, _WB, _WB])


class SectionSolverError(RuntimeError):
    """截面网格或求解无法完成。"""


def _shape(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T6 形函数及其对面积坐标 (L1, L2, L3) 的导数。节点顺序：三角点，再 12、23、31 边中点。"""
    L1, L2, L3 = L
    N = np.array([L1*(2*L1 - 1), L2*(2*L2 - 1), L3*(2*L3 - 1), 4*L1*L2, 4*L2*L3, 4*L3*L1])
    dN = np.array([[4*L1 - 1, 0, 0], [0, 4*L2 - 1, 0], [0, 0, 4*L3 - 1],
                   [4*L2, 4*L1, 0], [0, 4*L3, 4*L2], [4*L3, 0, 4*L1]])
    return N, dN


_SHAPES = [_shape(q) for q in _QP]


def _mesh(outer, holes, size: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise SectionSolverError("截面特性求解需要 Gmsh Python SDK 划分截面网格；"
                                 "请执行 pip install gmsh") from exc
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("section")
        geo = gmsh.model.geo

        def loop(points):
            tags = [geo.addPoint(float(y), float(z), 0.0) for y, z in points]
            lines = [geo.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
            return geo.addCurveLoop(lines)

        loops = [loop(outer)] + [loop(h) for h in holes]
        geo.addPlaneSurface(loops)
        geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)   # 边中点取直线中点：直边单元
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        types, _, conn = gmsh.model.mesh.getElements(2)
        if 9 not in list(types):
            raise SectionSolverError("Gmsh 没有生成 6 节点三角形单元")
        raw = np.asarray(conn[list(types).index(9)], dtype=np.int64).reshape(-1, 6)
    except SectionSolverError:
        raise
    except Exception as exc:  # noqa: BLE001  Gmsh 以通用 Exception 报告几何错误，转为工具错误并保留原因。
        raise SectionSolverError(f"截面网格划分失败：{exc}") from exc
    finally:
        gmsh.finalize()
    index = {int(t): k for k, t in enumerate(tags)}
    nodes = np.asarray(xyz, dtype=float).reshape(-1, 3)[:, :2]
    elements = np.vectorize(index.__getitem__)(raw)
    return nodes, elements


def _element_geometry(nodes: np.ndarray, elements: np.ndarray):
    """直边三角形的面积与面积坐标梯度（每单元为常量）。"""
    p = nodes[elements[:, :3]]                         # (ne, 3, 2)
    y, z = p[:, :, 0], p[:, :, 1]
    b = np.stack([z[:, 1] - z[:, 2], z[:, 2] - z[:, 0], z[:, 0] - z[:, 1]], axis=1)
    c = np.stack([y[:, 2] - y[:, 1], y[:, 0] - y[:, 2], y[:, 1] - y[:, 0]], axis=1)
    two_area = (y[:, 1] - y[:, 0])*(z[:, 2] - z[:, 0]) - (y[:, 2] - y[:, 0])*(z[:, 1] - z[:, 0])
    if np.any(two_area <= 0):
        # Gmsh 平面三角形统一逆时针；出现非正面积说明网格退化
        flip = two_area < 0
        if np.any(two_area == 0):
            raise SectionSolverError("截面网格出现零面积单元，请检查轮廓是否有重合边。")
        elements[flip] = elements[flip][:, [0, 2, 1, 5, 4, 3]]
        return _element_geometry(nodes, elements)
    grad = np.stack([b, c], axis=2) / two_area[:, None, None]   # dL_k/d(y,z): (ne, 3, 2)
    return elements, two_area / 2, grad


def solve_section(geometry: dict[str, Any], mesh_size: float | None = None) -> dict[str, Any]:
    """按真实轮廓求 J、Iw 与剪心。长度沿用几何的模型单位。

    返回字段：``J``（长度⁴）、``Iw``（长度⁶，对剪心）、``shear_center``（相对形心的
    [y, z]）、校核用的 ``A``/``Iy``/``Iz``/``Iyz``（由网格积分，应与轮廓精确值一致）、
    网格信息 ``mesh``。圆截面的 J 有闭合解，也可以用这里核对多边形近似的影响。
    """
    p = profile(geometry)
    outer = np.asarray(p["outer"], dtype=float)
    holes = [np.asarray(h, dtype=float) for h in p["holes"]]
    scale = float(np.abs(outer).max())                 # 特征长度：归一化后网格与单位无关
    feature = float(p["feature_size"]) / scale
    size = feature * DEFAULT_SIZE_FACTOR if mesh_size is None else float(mesh_size) / scale
    size = round(size, 12)                              # 理由同下方坐标舍入
    if not math.isfinite(size) or size <= 0:
        raise SectionSolverError("截面网格尺寸必须为正数")
    # 归一化坐标舍入到 12 位：米制与毫米制换算后末位不同，Gmsh 会因此划出不同网格
    nodes, elements = _mesh(np.round(outer / scale, 12),
                            [np.round(h / scale, 12) for h in holes], size)
    elements, area, grad = _element_geometry(nodes, elements)
    n_nodes, n_el = len(nodes), len(elements)

    # 积分点上的物理坐标与形函数梯度
    corner = nodes[elements[:, :3]]                    # (ne, 3, 2)
    K_rows, K_cols, K_vals = [], [], []
    f = np.zeros(n_nodes)
    qp_xy, qp_w, qp_N, qp_G = [], [], [], []
    for (N, dN), w in zip(_SHAPES, _QW, strict=True):
        L = _QP[len(qp_w)]
        xy = np.einsum("k,ekd->ed", L, corner)          # (ne, 2)
        G = np.einsum("ak,ekd->ead", dN, grad)          # (ne, 6, 2) 形函数对 (y, z) 的梯度
        weight = w * area                               # (ne,)
        qp_xy.append(xy), qp_w.append(weight), qp_N.append(N), qp_G.append(G)
        ke = np.einsum("e,ead,ebd->eab", weight, G, G)
        K_rows.append(np.repeat(elements, 6, axis=1).ravel())
        K_cols.append(np.tile(elements, (1, 6)).ravel())
        K_vals.append(ke.ravel())
        fe = weight[:, None] * (xy[:, 1, None]*G[:, :, 0] - xy[:, 0, None]*G[:, :, 1])
        np.add.at(f, elements.ravel(), fe.ravel())
    K = coo_matrix((np.concatenate(K_vals), (np.concatenate(K_rows), np.concatenate(K_cols))),
                   shape=(n_nodes, n_nodes)).tocsr()
    # ω 只确定到常数：钉住一个节点，之后再统一正交化
    keep = np.arange(1, n_nodes)
    omega = np.zeros(n_nodes)
    omega[keep] = spsolve(K[keep][:, keep].tocsc(), f[keep])

    # 网格积分：A、一次矩、二次矩、J 与 ω 的各项矩
    s = dict.fromkeys(("A", "Sy", "Sz", "Iyy", "Izz", "Iyz", "W", "Wy", "Wz", "J0"), 0.0)
    for xy, weight, N, G in zip(qp_xy, qp_w, qp_N, qp_G, strict=True):
        y, z = xy[:, 0], xy[:, 1]
        om = omega[elements] @ N
        dom = np.einsum("ea,ead->ed", omega[elements], G)
        s["A"] += weight.sum()
        s["Sy"] += (weight*y).sum(); s["Sz"] += (weight*z).sum()
        s["Iyy"] += (weight*y*y).sum(); s["Izz"] += (weight*z*z).sum()
        s["Iyz"] += (weight*y*z).sum()
        s["W"] += (weight*om).sum(); s["Wy"] += (weight*om*y).sum(); s["Wz"] += (weight*om*z).sum()
        s["J0"] += (weight*(y*y + z*z + y*dom[:, 1] - z*dom[:, 0])).sum()
    A = s["A"]
    yc, zc = s["Sy"]/A, s["Sz"]/A
    # 网格形心与轮廓形心本应重合（轮廓已移到形心）；仍按网格自身形心换算，口径一致
    Iz = s["Iyy"] - A*yc*yc                            # ∫y²dA，强轴
    Iy = s["Izz"] - A*zc*zc                            # ∫z²dA
    Iyz = s["Iyz"] - A*yc*zc
    # ω 与 J₀ 用同一坐标原点求得；换原点时 ω 只差线性项，J 不变，无需移到形心
    J = s["J0"]
    Wy = s["Wy"] - yc*s["W"]
    Wz = s["Wz"] - zc*s["W"]
    beta, gamma = np.linalg.solve([[Iz, Iyz], [Iyz, Iy]], [Wy, Wz])
    z_sc, y_sc = beta, -gamma                          # ω_P = ω − z_P·y + y_P·z + c
    # Iw = ∫ω_P²：ω_P 是 ω 去掉在 {1, y−yc, z−zc} 上投影后的余量
    Iw = 0.0
    mean = s["W"]/A
    for xy, weight, N in zip(qp_xy, qp_w, qp_N, strict=True):
        om = omega[elements] @ N - mean - beta*(xy[:, 0] - yc) - gamma*(xy[:, 1] - zc)
        Iw += (weight*om*om).sum()
    return {
        "J": float(J * scale**4), "Iw": float(Iw * scale**6),
        "shear_center": [float(y_sc * scale), float(z_sc * scale)],
        "A": float(A * scale**2), "Iy": float(Iy * scale**4), "Iz": float(Iz * scale**4),
        "Iyz": float(Iyz * scale**4),
        "mesh": {"elements": int(n_el), "nodes": int(n_nodes), "size": float(size * scale),
                 "element": "T6"},
        "method": "Saint-Venant 翘曲函数二维有限元（T6）",
    }
