"""截面约束的G2扫掠，以及多臂半空间斜接与自适应滚动球圆角。"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def bezier_values(points, parameters):
    p, t = np.asarray(points), np.asarray(parameters)
    n = len(p)-1
    return sum(math.comb(n, i)*(1-t)[:, None]**(n-i)*t[:, None]**i*p[i]
               for i in range(n+1))


def controls(first, second, ratio):
    """前三点、后三点各自共线且等距，保证端点曲率为零。"""
    return np.array([first, (1-ratio)*first, (1-2*ratio)*first,
                     (1-2*ratio)*second, (1-ratio)*second, second])


def curvature(points, parameters):
    velocity = bezier_values(5*np.diff(points, axis=0), parameters)
    acceleration = bezier_values(20*np.diff(points, n=2, axis=0), parameters)
    return np.linalg.norm(np.cross(velocity, acceleration), axis=1)/np.linalg.norm(velocity, axis=1)**3


def _rotation(first, second):
    axis = np.cross(first, second)
    sine = np.linalg.norm(axis)
    cosine = float(first @ second)
    axis /= sine
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)+sine*skew+(1-cosine)*(skew @ skew)


def sweep_plan(spec):
    """从尺寸与朝向确定曲线；不匹配或空间不足时返回斜接方式的原因。"""
    from joint_geometry import _basis, _profile
    if len(spec.arms) != 2:
        return None, "G2扫掠适用于两个匹配截面的短臂；多臂节点采用斜接裁切。"
    a, b = spec.arms
    d, ay, az = _basis(a)
    e, by, bz = _basis(b)
    if np.linalg.norm(np.cross(d, e)) < 1e-7:
        return None, "共线杆件不需要弯曲扫掠。"
    pa, pb = _profile(a), _profile(b)
    if pa['type'] != pb['type'] or not math.isclose(pa['area'], pb['area'], rel_tol=1e-8):
        return None, "两臂截面不一致，采用真实截面的斜接裁切。"
    if pa['type'] in ('circle', 'tube') and not np.allclose(
            [a.outer_diameter_mm, a.inner_diameter_mm], [b.outer_diameter_mm, b.inner_diameter_mm], rtol=1e-8):
        return None, "两臂直径或壁厚不一致，采用真实截面的斜接裁切。"
    if len(pa['holes']) != len(pb['holes']):
        return None, "两臂孔洞不一致，采用斜接裁切。"
    rotation = _rotation(-d, e)
    if pa['type'] not in ('circle', 'tube'):
        for first, second in zip([pa['outer'], *pa['holes']], [pb['outer'], *pb['holes']], strict=True):
            u, v = np.asarray(first), np.asarray(second)
            initial = (u[:, :1]*ay+u[:, 1:]*az) @ rotation.T
            target = v[:, :1]*by+v[:, 1:]*bz
            if len(u) != len(v) or np.linalg.norm(initial[:, None]-target, axis=2).min(axis=1).max() > 1e-6:
                return None, "两臂截面朝向不匹配，采用斜接裁切以保留切面轮廓。"
    sample = np.linspace(1e-4, 1-1e-4, 401)
    def objective(ratio):
        return float(curvature(controls(d, e, ratio), sample).max())
    result = minimize_scalar(objective, bounds=(.2, .48), method='bounded', options={'xatol': 1e-5})
    if not result.success:
        raise ValueError("连接曲线优化未完成，请选择斜接裁切或原始杆件交汇。")
    unit = controls(d, e, result.x)
    maximum = float(curvature(unit, np.linspace(1e-5, 1-1e-5, 4097)).max())
    length = 1.3*max(a.outer_diameter_mm, b.outer_diameter_mm)/2*maximum
    if length > .8*min(a.length_mm, b.length_mm):
        return None, "曲率约束需要更长的过渡区域，当前短臂采用斜接裁切。"
    return {'algorithm': 'g2-bezier-sweep', 'control_ratio': float(result.x),
            'control_points_mm': (unit*length).tolist(), 'transition_length_mm': length,
            'minimum_centerline_radius_mm': length/maximum,
            'curvature_safety_factor': 1.3}, None


def _section_face(occ, arm):
    from joint_geometry import _basis, _polygon_surface, _profile
    d, y, z = _basis(arm)
    p = _profile(arm)
    end = d*arm.length_mm
    if p['type'] in ('circle', 'tube'):
        face = occ.addDisk(0, 0, 0, arm.outer_diameter_mm/2, arm.outer_diameter_mm/2)
        axis = np.cross([0, 0, 1], d)
        if np.linalg.norm(axis) < 1e-8:
            axis = np.array([1., 0., 0.])
        angle = float(np.arccos(np.clip(d[2], -1, 1)))
        occ.rotate([(2, face)], 0, 0, 0, *axis, angle)
        occ.translate([(2, face)], *end)
        if arm.inner_diameter_mm:
            hole = occ.addDisk(0, 0, 0, arm.inner_diameter_mm/2, arm.inner_diameter_mm/2)
            occ.rotate([(2, hole)], 0, 0, 0, *axis, angle)
            occ.translate([(2, hole)], *end)
            faces, _ = occ.cut([(2, face)], [(2, hole)])
            face = faces[0][1]
        return face
    loop = np.asarray(p['outer'])
    face = _polygon_surface(occ, end+loop[:, :1]*y+loop[:, 1:]*z)
    holes = []
    for loop in map(np.asarray, p['holes']):
        holes.append((2, _polygon_surface(occ, end+loop[:, :1]*y+loop[:, 1:]*z)))
    if holes:
        faces, _ = occ.cut([(2, face)], holes)
        face = faces[0][1]
    return face


def build_sweep(gmsh, spec, plan):
    occ = gmsh.model.occ
    a, b = spec.arms
    tags = [occ.addPoint(*v) for v in plan['control_points_mm']]
    first = occ.addPoint(*(np.asarray(a.direction)*a.length_mm))
    last = occ.addPoint(*(np.asarray(b.direction)*b.length_mm))
    wire = occ.addWire([occ.addLine(first, tags[0]), occ.addBezier(tags), occ.addLine(tags[-1], last)])
    return [v for v in occ.addPipe([(2, _section_face(occ, a))], wire, trihedron='CorrectedFrenet') if v[0] == 3]


def column_host_plan(spec):
    """识别柱顶节点；保持柱截面，梁端接到实际翼缘/腹板面，不对柱做角平分裁切。"""
    from joint_geometry import _basis, _profile
    from section_geometry import contains
    vertical = [a for a in spec.arms if abs(abs(a.direction[2])-1.) < 1e-7]
    if len(spec.arms) < 3 or len(vertical) != 1:
        return None
    host = vertical[0]
    hd, hy, hz = _basis(host)
    beams = [a for a in spec.arms if a is not host]
    if any(abs(np.dot(a.direction, hd)) > 1e-7 for a in beams):
        return None
    profile = _profile(host)
    if profile['type'] in ('circle', 'tube'):
        return None
    loop = np.asarray(profile['outer'])
    starts, head = {}, 0.
    for arm in beams:
        d, y, z = _basis(arm)
        ray = np.array([d @ hy, d @ hz])
        hits = []
        for a, b in zip(loop, np.roll(loop, -1, axis=0), strict=True):
            edge = b-a
            matrix = np.column_stack((ray, -edge))
            if abs(np.linalg.det(matrix)) < 1e-10:
                continue
            station, fraction = np.linalg.solve(matrix, a)
            if station > 1e-7 and -1e-7 <= fraction <= 1+1e-7:
                hits.append((float(station), edge))
        if not hits:
            raise ValueError(f'杆件{arm.member_id}未找到柱材料连接面，请明确梁柱连接细节。')
        start, edge = max(hits, key=lambda hit: hit[0])
        if abs(edge @ ray) > 1e-7*np.linalg.norm(edge) or start >= arm.length_mm:
            raise ValueError(f'杆件{arm.member_id}无法贴合柱的平直材料面，请选择原始杆件交汇或定义连接实体。')
        p = _profile(arm)
        outer = np.asarray(p['outer'])
        # 柱截面沿轴向恒定，端面覆盖等价于梁宽投影落在柱材料线段内。
        # 枚举外轮廓和孔洞的全部交点分段检查，避免网格取样漏掉薄壁或狭窄缺口。
        transverse = np.array([-ray[1], ray[0]])
        width = (outer[:, :1]*y+outer[:, 1:]*z) @ (transverse[0]*hy+transverse[1]*hz)
        limits = [float(width.min()), float(width.max())]
        origin = ray*(start-1e-5)
        breaks = list(limits)
        for contour in [profile['outer'], *profile['holes']]:
            coordinates = np.asarray(contour)
            for a, b in zip(coordinates, np.roll(coordinates, -1, axis=0), strict=True):
                matrix = np.column_stack((transverse, a-b))
                if abs(np.linalg.det(matrix)) < 1e-10:
                    continue
                t, fraction = np.linalg.solve(matrix, a-origin)
                if -1e-7 <= fraction <= 1+1e-7 and limits[0] < t < limits[1]:
                    breaks.append(float(t))
        breaks = np.unique(breaks)
        for low, high in zip(breaks[:-1], breaks[1:], strict=True):
            if high-low < 1e-7:
                continue
            point = origin+transverse*(low+high)/2
            if not contains(profile['outer'], point) or any(contains(h, point) for h in profile['holes']):
                raise ValueError(f'杆件{arm.member_id}端面超出柱材料面，请补充连接板或调整截面朝向。')
        head = max(head, float(np.max((outer[:, :1]*y+outer[:, 1:]*z) @ (-hd))))
        starts[str(arm.member_id)] = start
    if head > .35*min(a.length_mm for a in spec.arms):
        raise ValueError('柱顶延长区接近驱动切面，请增大短臂范围或明确连接尺寸。')
    return {'algorithm': 'column-host-butt-joint', 'host_member_id': host.member_id,
            'head_extension_mm': head, 'beam_start_mm': starts,
            'host_material_preserved': True, 'fillet_status': 'not_requested',
            'applied_fillet_radius_mm': 0.,
            'connection_assumption': '柱顶延长至梁外缘，梁端与柱实际材料面理想粘结；未建立焊缝或连接板。'}


def build_column_host(gmsh, spec, plan):
    """先建立完整柱体，再并入贴合梁端；孔洞仅从所属杆件内扣除。"""
    from joint_geometry import _basis, _extrude_loop, _profile
    occ = gmsh.model.occ
    solids = []
    for arm in spec.arms:
        p = _profile(arm)
        start = (-plan['head_extension_mm'] if arm.member_id == plan['host_member_id']
                 else plan['beam_start_mm'][str(arm.member_id)])
        length = arm.length_mm-start
        if p['type'] in ('circle', 'tube'):
            d = _basis(arm)[0]
            solid = [(3, occ.addCylinder(*(start*d), *(length*d), arm.outer_diameter_mm/2))]
            holes = ([] if not arm.inner_diameter_mm else
                     [(3, occ.addCylinder(*(start*d), *(length*d), arm.inner_diameter_mm/2))])
        else:
            solid = _extrude_loop(occ, arm, p['outer'], start, length)
            holes = [v for loop in p['holes'] for v in _extrude_loop(occ, arm, loop, start, length)]
        if holes:
            solid, _ = occ.cut(solid, holes, removeObject=True, removeTool=True)
        solids.extend(solid)
    return occ.fuse(solids[:1], solids[1:], removeObject=True, removeTool=True)[0]


def build_miter(gmsh, spec):
    """沿角平分半空间分配各杆件实体与孔洞，避免凸包填满翼缘间凹槽。"""
    from joint_geometry import _basis, _extrude_loop, _polygon_surface, _profile
    occ = gmsh.model.occ
    solids, holes = [], []
    back = 2*max(a.outer_diameter_mm for a in spec.arms)
    extent = 3*(max(a.length_mm for a in spec.arms)+back)
    for arm in spec.arms:
        d, y, z = _basis(arm)
        p = _profile(arm)
        contour = np.asarray(p['outer'])
        cut = arm.length_mm*d+contour[:, :1]*y+contour[:, 1:]*z
        for other in spec.arms:
            if np.max(cut @ (np.asarray(other.direction)-d)) > 1e-6:
                raise ValueError(f"杆件{arm.member_id}切面仍位于相贯裁切区，请增大短臂范围或选择原始杆件交汇。")
        if p['type'] in ('circle', 'tube'):
            solid = [(3, occ.addCylinder(*(-back*d), *(d*(arm.length_mm+back)), arm.outer_diameter_mm/2))]
            void = [] if not arm.inner_diameter_mm else [
                (3, occ.addCylinder(*(-back*d), *(d*(arm.length_mm+back)), arm.inner_diameter_mm/2))]
        else:
            solid = _extrude_loop(occ, arm, p['outer'], -back, arm.length_mm+back)
            void = [v for h in p['holes'] for v in _extrude_loop(occ, arm, h, -back, arm.length_mm+back)]
        for other in spec.arms:
            normal = d-np.asarray(other.direction)
            if np.linalg.norm(normal) < 1e-8:
                continue
            normal /= np.linalg.norm(normal)
            u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            face = _polygon_surface(occ, np.array([-u-v, u-v, u+v, -u+v])*extent)
            tool = [t for t in occ.extrude([(2, face)], *(-extent*normal)) if t[0] == 3]
            solid, _ = occ.cut(solid, tool, removeObject=True, removeTool=False)
            if void:
                void, _ = occ.cut(void, tool, removeObject=True, removeTool=False)
            occ.remove(tool, recursive=True)
        solids.extend(solid)
        holes.extend(void)
    if len(solids) > 1:
        solids, _ = occ.fuse(solids[:1], solids[1:], removeObject=True, removeTool=True)
    if holes:
        if len(holes) > 1:
            holes, _ = occ.fuse(holes[:1], holes[1:], removeObject=True, removeTool=True)
        solids, _ = occ.cut(solids, holes, removeObject=True, removeTool=True)
    return solids


def fillet_miter(gmsh, spec, solids):
    """用边法向平面内的材料占比识别凹边，圆角失败只保留有效斜接实体并明确记录。"""
    from joint_geometry import _profile
    gmsh.model.occ.synchronize()
    if len(solids) != 1:
        return solids, {'fillet_status': 'disconnected', 'applied_fillet_radius_mm': 0.}
    volume = solids[0][1]
    curves = sorted({t for _, t in gmsh.model.getBoundary(
        gmsh.model.getBoundary(solids, oriented=False), combined=False, oriented=False)})
    feature = min(_profile(a)['feature_size'] for a in spec.arms)
    reach = .6*min(a.length_mm for a in spec.arms)
    selected = []
    for tag in curves:
        box = np.asarray(gmsh.model.getBoundingBox(1, tag)).reshape(2, 3)
        if np.linalg.norm(box, axis=1).max() > reach:
            continue
        lo, hi = gmsh.model.getParametrizationBounds(1, tag)
        parameter = float((lo[0]+hi[0])/2)
        point = np.asarray(gmsh.model.getValue(1, tag, [parameter]))
        tangent = np.asarray(gmsh.model.getDerivative(1, tag, [parameter]))
        if np.linalg.norm(tangent) < 1e-12:
            continue
        tangent /= np.linalg.norm(tangent)
        u = np.cross(tangent, np.eye(3)[np.argmin(abs(tangent))])
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        phi = (np.arange(24)+.5)*2*np.pi/24
        points = point+feature*1e-3*(np.cos(phi)[:, None]*u+np.sin(phi)[:, None]*v)
        if sum(gmsh.model.isInside(3, volume, p.tolist()) for p in points)/24 > .56:
            selected.append(tag)
    record = {'algorithm': 'halfspace-miter+rolling-ball', 'concave_edges': len(selected),
              'requested_fillet_radius_mm': .45*feature, 'applied_fillet_radius_mm': 0.,
              'fillet_region_radius_mm': reach, 'fillet_status': 'not_required'}
    if not selected:
        return solids, record
    for factor in (1., .5):
        try:
            rounded = gmsh.model.occ.fillet([volume], selected, [.45*feature*factor], removeVolume=False)
        except Exception:  # noqa: BLE001  OCCT圆角失败不改变原实体；记录未形成圆角，不改成凸包。
            continue
        gmsh.model.occ.remove(solids, recursive=True)
        record.update(applied_fillet_radius_mm=.45*feature*factor, fillet_status='rounded')
        return rounded, record
    record['fillet_status'] = 'kernel_unavailable'
    return solids, record
