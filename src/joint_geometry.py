"""真实截面挤出与有限范围连接包络；保留切面轮廓和截面孔洞。"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def _basis(arm):
    d = np.asarray(arm.direction)
    if arm.axis_y is not None:
        return d, np.asarray(arm.axis_y), np.asarray(arm.axis_z)
    candidate = np.array([0., 0., 1.]) if abs(d[2]) < .9 else np.array([0., 1., 0.])
    y = np.cross(candidate, d)
    y /= np.linalg.norm(y)
    return d, y, np.cross(d, y)


def _profile(arm):
    if arm.profile_mm is not None:
        return arm.profile_mm
    from section_geometry import profile
    return profile({"type": "tube" if arm.inner_diameter_mm else "circle",
                    "dimensions": {"d": arm.outer_diameter_mm,
                    **({"t": (arm.outer_diameter_mm-arm.inner_diameter_mm)/2}
                       if arm.inner_diameter_mm else {})}})


def _polygon_surface(occ, coordinates):
    points = [occ.addPoint(*map(float, p)) for p in coordinates]
    lines = [occ.addLine(a, b) for a, b in zip(points, points[1:]+points[:1], strict=True)]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def _extrude_loop(occ, arm, loop, start=0., length=None):
    d, y, z = _basis(arm)
    coordinates = start*d + np.asarray(loop)[:, :1]*y + np.asarray(loop)[:, 1:]*z
    face = _polygon_surface(occ, coordinates)
    vector = d*(arm.length_mm if length is None else length)
    entities = occ.extrude([(2, face)], *map(float, vector))
    return [(dim, tag) for dim, tag in entities if dim == 3]


def _hull_volume(occ, coordinates):
    xyz = np.unique(np.round(coordinates, decimals=9), axis=0)
    hull = ConvexHull(xyz)
    groups = {}
    for triangle, equation in zip(hull.simplices, hull.equations, strict=True):
        key = tuple(np.round(equation, decimals=7))
        groups.setdefault(key, set()).update(map(int, triangle))
    point_tags = {i: occ.addPoint(*map(float, xyz[i])) for i in hull.vertices}
    edges, surfaces = {}, []
    for equation, indices in groups.items():
        normal = np.asarray(equation[:3])
        ids = list(indices)
        center = xyz[ids].mean(axis=0)
        e1 = xyz[ids[0]]-center
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        angles = np.arctan2((xyz[ids]-center) @ e2, (xyz[ids]-center) @ e1)
        ids = [ids[i] for i in np.argsort(angles)]
        lines = []
        for a, b in zip(ids, ids[1:]+ids[:1], strict=True):
            key = (min(a, b), max(a, b))
            if key not in edges:
                edges[key] = occ.addLine(point_tags[key[0]], point_tags[key[1]])
            lines.append(edges[key] if a < b else -edges[key])
        surfaces.append(occ.addPlaneSurface([occ.addCurveLoop(lines)]))
    return [(3, occ.addVolume([occ.addSurfaceLoop(surfaces, sewing=True)]))]


def build_occ_joint(gmsh, spec):
    """构造CAD并返回记录：连接包络仅在节点附近，孔洞继续贯穿对应杆段。"""
    occ = gmsh.model.occ
    solids, holes, sections, hull_points = [], [], [], []
    minimum_length = min(a.length_mm for a in spec.arms)
    depth = 0.35*minimum_length
    for arm in spec.arms:
        p = _profile(arm)
        d, y, z = _basis(arm)
        if p["type"] in ("circle", "tube"):
            radius = arm.outer_diameter_mm/2
            solid = [(3, occ.addCylinder(0, 0, 0, *(d*arm.length_mm), radius))]
            voids = []
            if arm.inner_diameter_mm:
                eps = max(arm.length_mm*1e-5, 1e-5)
                voids = [(3, occ.addCylinder(*(-eps*d), *(d*(arm.length_mm+2*eps)),
                                           arm.inner_diameter_mm/2))]
        else:
            solid = _extrude_loop(occ, arm, p["outer"])
            eps = max(arm.length_mm*1e-5, 1e-5)
            voids = [v for loop in p["holes"] for v in
                     _extrude_loop(occ, arm, loop, -eps, arm.length_mm+2*eps)]
        if spec.geometry_mode == "intersection" and voids:
            solid, _ = occ.cut(solid, voids, removeObject=True, removeTool=True)
        elif voids:
            holes.extend(voids)
        solids.extend(solid)
        contour = np.asarray(p["outer"])
        if p["type"] in ("circle", "tube"):
            contour = contour[::max(1, len(contour)//12)]
        coordinates = contour[:, :1]*y + contour[:, 1:]*z
        hull_points.extend(coordinates)
        hull_points.extend(coordinates+depth*d)
        sections.append({"member_id": arm.member_id, "profile": p["type"]})
    noncollinear = any(np.linalg.norm(np.cross(spec.arms[0].direction, a.direction)) > 1e-8
                      for a in spec.arms[1:])
    if spec.geometry_mode == "transition" and noncollinear:
        solids.extend(_hull_volume(occ, np.asarray(hull_points)))
    if len(solids) > 1:
        solids, _ = occ.fuse(solids[:1], solids[1:], removeObject=True, removeTool=True)
    if holes:
        # 相交杆件的空腔先并集；直接用重叠工具做差会保留空腔碎片为实体体积。
        if len(holes) > 1:
            holes, _ = occ.fuse(holes[:1], holes[1:], removeObject=True, removeTool=True)
        solids, _ = occ.cut(solids, holes, removeObject=True, removeTool=True)
    occ.synchronize()
    if not solids:
        raise ValueError("连接实体布尔运算没有留下有效体积，请检查截面与杆件方向。")
    if len(solids) != 1:
        raise ValueError("截面与孔洞未形成连续节点，请调整杆件方向或节点连接几何。")
    return {"mode": spec.geometry_mode, "connection_depth_mm": depth if spec.geometry_mode == "transition" and noncollinear else 0,
            "volumes": len(solids), "volume_mm3": float(sum(occ.getMass(*v) for v in solids)),
            "sections": sections, "holes_preserved": True}


def export_step(spec, path):
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("local_joint_geometry")
        record = build_occ_joint(gmsh, spec)
        gmsh.write(str(path))
        return record
    finally:
        gmsh.finalize()
