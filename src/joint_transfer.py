"""梁—实体子模型的切面运动恢复与面积加权分布耦合。"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

import solid3d
from frame3d import local_axes, span_loads_of
from internal_forces import _result_local_displacements, member_diagram, member_length
from span_loads import MOMENT, PARTIAL, POINT, RAMP


def cut_motion(session, member_id: int, point: np.ndarray, case: str):
    """在实际切面位置积分整体梁曲率，避免把节点转角直接搬到远处切面。"""
    frame = session.frame
    for eid in session.compilation.mapping.element_ids(member_id):
        member = frame.members[eid]
        pi, pj = frame.nodes[member.i].xyz, frame.nodes[member.j].xyz
        length, rot = local_axes(pi, pj, member.ref_vector)
        position = rot @ (point - pi)
        x = float(position[0])
        tol = max(length * 1e-9, 1e-12)
        if not (-tol <= x <= length + tol) or np.linalg.norm(position[1:]) > tol:
            continue
        x = float(np.clip(x, 0.0, length))
        q = _result_local_displacements(frame, session.solution, member, case)
        # 每段内力是至多三次多项式；四点 Gauss 可精确积分其位移贡献。
        breaks = {0.0, x}
        sources = frame.combos[case].items() if case in frame.combos else [(case, 1.0)]
        for base, factor in sources:
            if factor == 0.0:
                continue
            for load in span_loads_of(frame.load_cases[base], eid):
                breaks.update(v for v in (load.a, load.b) if 0.0 < v < x)
        abscissa, weight = np.polynomial.legendre.leggauss(4)
        stations, weights = [], []
        edges = sorted(breaks)
        for a, b in zip(edges, edges[1:], strict=False):
            stations.extend(0.5 * (a + b) + 0.5 * (b - a) * abscissa)
            weights.extend(0.5 * (b - a) * weight)
        u, theta = q[:3].copy(), q[3:6].copy()
        u[1] += q[5] * x
        u[2] -= q[4] * x
        if stations:
            s, w = np.asarray(stations), np.asarray(weights)
            d = member_diagram(frame, session.solution, eid, case, at_x=s)
            mat, sec = frame.materials[member.material], frame.sections[member.section]
            u[0] += w @ d.N / (mat.E * sec.A)
            theta[0] += w @ d.T / (mat.G * sec.J)
            theta[1] += w @ d.My / (mat.E * sec.Iy)
            theta[2] += w @ d.Mz / (mat.E * sec.Iz)
            u[1] += (w * (x - s)) @ d.Mz / (mat.E * sec.Iz)
            u[2] -= (w * (x - s)) @ d.My / (mat.E * sec.Iy)
            if sec.Ay is not None:
                u[1] += w @ d.Vy / (mat.G * sec.Ay)
            if sec.Az is not None:
                u[2] += w @ d.Vz / (mat.G * sec.Az)
        return rot.T @ u, rot.T @ theta, eid, x
    raise ValueError(f"杆件 {member_id} 的实体切面未落在整体分析单元内")


def validate_local_sources(session, node_id: int, arms, case: str) -> None:
    """拒绝未在实体内重建的区域载荷及本征应变，防止漏载后给出应力。"""
    if session.solution.analysis.get("type", "linear_static") != "linear_static":
        raise ValueError("位移驱动当前需要线性静力整体结果；请先完成线性求解。")
    frame = session.frame
    origin = frame.nodes[node_id].xyz
    scale = session.units.length_to_m * 1000.0
    for arm in arms:
        direction = np.asarray(arm.direction)
        for nid, fixed in frame.supports.items():
            rel = frame.nodes[nid].xyz - origin
            axial = float(rel @ direction)
            if (any(fixed) and -1e-10 <= axial < arm.length_mm / scale - 1e-10
                    and np.linalg.norm(rel - axial * direction) < 1e-9):
                raise ValueError("实体保留区域存在整体支座约束；请先明确实体支承区域，"
                                 "不能只传切面运动而遗漏内部支承。")
    sources = frame.combos[case].items() if case in frame.combos else [(case, 1.0)]
    for base, factor in sources:
        if factor == 0.0:
            continue
        loads = frame.load_cases[base]
        for arm in arms:
            direction = np.asarray(arm.direction)
            cut = arm.length_mm / scale
            for nid, value in loads.nodal_loads.items():
                rel = frame.nodes[nid].xyz - origin
                axial = float(rel @ direction)
                if (np.any(np.asarray(value) != 0.0) and -1e-10 <= axial <= cut + 1e-10
                        and np.linalg.norm(rel - axial * direction) < 1e-9):
                    raise ValueError("实体保留区域内存在直接集中载荷；请先定义实体加载区域，"
                                     "或选择 force 内力加载进行原方案对照。")
            for eid in session.compilation.mapping.element_ids(arm.member_id):
                member = frame.members[eid]
                length = member_length(frame, eid)
                start = float((frame.nodes[member.i].xyz - origin) @ direction)
                end = float((frame.nodes[member.j].xyz - origin) @ direction)
                if min(start, end) >= cut or max(start, end) < 0.0:
                    continue
                if loads.member_strains.get(eid) or any(loads.member_curvatures.get(eid, ())):
                    raise ValueError("实体保留区域存在温度或初应变；请先在实体内重建相应场。")
                for load in span_loads_of(loads, eid):
                    if not any(load.w1) and not any(load.w2):
                        continue
                    if load.kind in (POINT, MOMENT):
                        a = b = load.a
                    elif load.kind in (PARTIAL, RAMP):
                        a, b = load.a, load.b
                    else:
                        a, b = 0.0, length
                    sign = 1.0 if end > start else -1.0
                    lo, hi = sorted((start + sign * a, start + sign * b))
                    overlaps = (max(lo, 0.0) <= min(hi, cut)
                                if load.kind in (POINT, MOMENT)
                                else max(lo, 0.0) < min(hi, cut))
                    if overlaps:
                        raise ValueError("实体保留杆段内存在分布载荷或集中载荷；"
                                         "请先定义实体加载区域，或选择 force 内力加载对照。")


def cut_area_weights(mesh, indices: np.ndarray) -> np.ndarray:
    """将外表面六节点三角形划成四个子三角形，累计正的节点分担面积。"""
    selected = set(map(int, indices))
    faces = {}
    for conn in mesh.elements:
        for local in solid3d.TET10_FACES:
            nodes = tuple(int(conn[i]) for i in local)
            key = tuple(sorted(nodes[:3]))
            faces[key] = nodes if key not in faces else None
    area = dict.fromkeys(selected, 0.0)
    for face in faces.values():
        if face is None or not set(face).issubset(selected):
            continue
        for tri in ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)):
            ids = [face[i] for i in tri]
            p = mesh.nodes[ids]
            share = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / 6.0
            for nid in ids:
                area[nid] += share
    weights = np.asarray([area[int(i)] for i in indices])
    if np.any(weights <= 0.0):
        raise solid3d.Solid3DError("切面节点没有有效表面积；请重新检查切面识别和网格。")
    return weights


def distributing_operator(points, origin, weights):
    """面积加权拟合切面六分量运动，约束均值而保留其余截面变形。"""
    xyz, w = np.asarray(points, dtype=float), np.asarray(weights, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or w.shape != (len(xyz),)
            or len(xyz) < 3 or not np.all(np.isfinite(xyz))
            or not np.all(np.isfinite(w)) or np.any(w <= 0.0)):
        raise solid3d.Solid3DError("分布耦合需要至少三个有效节点及正的面积权重。")
    rel = xyz - np.asarray(origin, dtype=float)
    radius = float(np.sqrt(np.average(np.sum(rel * rel, axis=1), weights=w)))
    if not np.isfinite(radius) or radius <= 0.0:
        raise solid3d.Solid3DError("分布耦合切面尺寸无效。")
    b = np.zeros((3 * len(xyz), 6))
    for i, (x, y, z) in enumerate(rel / radius):
        b[3 * i:3 * i + 3, :3] = np.eye(3)
        b[3 * i:3 * i + 3, 3:] = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
    weighted = b.T * np.repeat(w / w.sum(), 3)
    if np.linalg.matrix_rank(weighted @ b) < 6:
        raise solid3d.Solid3DError("切面节点共线，无法传递完整平移和转角。")
    return np.linalg.solve(weighted @ b, weighted), radius


def joint_constraints(spec, mesh, cut_faces):
    """将每个切面的平均运动约束组装到全局稀疏矩阵。"""
    rows, cols, data, values = [], [], [], []
    for index, arm in enumerate(spec.arms):
        ids = np.asarray(cut_faces[arm.member_id], dtype=np.int64)
        center = np.asarray(arm.direction) * arm.length_mm
        operator, radius = distributing_operator(
            mesh.nodes[ids], center, cut_area_weights(mesh, ids))
        dofs = np.column_stack((3 * ids, 3 * ids + 1, 3 * ids + 2)).ravel()
        rows.extend(np.repeat(np.arange(6 * index, 6 * index + 6), len(dofs)))
        cols.extend(np.tile(dofs, 6))
        data.extend(operator.ravel())
        values.extend(arm.translation_mm)
        values.extend(radius * np.asarray(arm.rotation_rad))
    matrix = coo_matrix((data, (rows, cols)), shape=(len(values), 3 * len(mesh.nodes)))
    return matrix.tocsr(), np.asarray(values)


def evaluate_cut_transfer(spec, row, tolerance=0.05):
    """核对实体切面反力与整体梁内力，揭示位移一致但平均刚度不一致的情况。"""
    expected = {}
    for arm in spec.arms:
        f = np.asarray(arm.force_n)
        center = np.asarray(arm.direction) * arm.length_mm
        m = np.asarray(arm.moment_nmm) - np.cross(center, f)
        expected[str(arm.member_id)] = (f, m)
    force_scale = float(max(np.linalg.norm(f) for f, m in expected.values()))
    moment_scale = float(max(np.linalg.norm(m) for f, m in expected.values()))
    checks = {}
    for mid, (f, m) in expected.items():
        actual = row["cut_resultants"][mid]
        df = float(np.linalg.norm(np.asarray(actual["force_n"]) - f))
        dm = float(np.linalg.norm(np.asarray(actual["moment_at_cut_nmm"]) - m))
        checks[mid] = {"expected_force_n": f.tolist(),
                       "expected_moment_at_cut_nmm": m.tolist(),
                       "force_difference_n": df, "moment_difference_nmm": dm,
                       "force_difference_ratio": df / force_scale if force_scale else None,
                       "moment_difference_ratio": dm / moment_scale if moment_scale else None,
                       "within_tolerance": (df <= 1e-5 + tolerance * force_scale
                                            and dm <= 1e-3 + tolerance * moment_scale)}
    return {"tolerance": tolerance, "normalization": "各切面整体梁合力/合矩的最大范数",
            "cuts": checks, "within_tolerance": all(v["within_tolerance"] for v in checks.values())}
