"""截取保留区域内的实际外载，并按表面积分担传递严格等效的六分量合力。"""
# ruff: noqa: B905
# 表面加载函数原样导出到Abaqus Python 2.7，该环境的zip没有strict参数。

from __future__ import annotations

import math

import numpy as np

from frame3d import span_loads_of
from internal_forces import member_length
from span_loads import MOMENT, PARTIAL, POINT, RAMP


def collect_local_loads(session, node_id, arms, case):
    """使用已编译工况，避免集中力自动剖分成节点荷载后再次计入。"""
    frame = session.frame
    origin = frame.nodes[node_id].xyz
    scale = session.units.length_to_m * 1000
    sources = frame.combos[case].items() if case in frame.combos else [(case, 1.)]
    records = []
    radius = .3 * min(a.outer_diameter_mm for a in arms)
    for base, factor in sources:
        if factor == 0:
            continue
        loads = frame.load_cases[base]
        for nid, value in loads.nodal_loads.items():
            rel = frame.nodes[nid].xyz-origin
            inside = any(-1e-10 <= rel @ a.direction < a.length_mm/scale-1e-10
                         and np.linalg.norm(rel-(rel @ a.direction)*np.asarray(a.direction)) < 1e-9
                         for a in arms)
            value = np.asarray(value)*factor
            if inside and np.any(value != 0):
                records.append(dict(kind='point', source_case=base, source_node=nid,
                                    point_mm=(rel*scale).tolist(), radius_mm=radius,
                                    force_n=value[:3].tolist(), moment_nmm=(value[3:]*scale).tolist()))
        for arm in arms:
            direction = np.asarray(arm.direction)
            cut = arm.length_mm/scale
            for eid in session.compilation.mapping.element_ids(arm.member_id):
                element = frame.members[eid]
                length = member_length(frame, eid)
                start = float((frame.nodes[element.i].xyz-origin) @ direction)
                end = float((frame.nodes[element.j].xyz-origin) @ direction)
                sign = 1 if end > start else -1
                for load in span_loads_of(loads, eid):
                    if load.kind in (POINT, MOMENT):
                        raise ValueError('集中载荷未完成整体剖分，请重新求解整体模型。')
                    a, b = (load.a, load.b) if load.kind in (PARTIAL, RAMP) else (0., length)
                    lo, hi = sorted((start+sign*a, start+sign*b))
                    lo, hi = max(lo, 0.), min(hi, cut)
                    if hi <= lo:
                        continue
                    first, last = load.ends()
                    if load.kind == PARTIAL:
                        last = first
                    if not np.any(first) and not np.any(last):
                        continue
                    # 线性强度逐段解析积分；空间分担细化随截面尺寸，不随网格编号变化。
                    count = max(1, math.ceil((hi-lo)*scale/(.4*arm.outer_diameter_mm)))
                    edges = np.linspace(lo, hi, count+1)
                    for left, right in zip(edges, edges[1:], strict=False):
                        x0, x1 = (left-start)/sign, (right-start)/sign
                        q0 = factor*(first+(last-first)*(x0-a)/(b-a))
                        q1 = factor*(first+(last-first)*(x1-a)/(b-a))
                        force = (q0+q1)*(right-left)/2
                        moment = np.cross(direction, q1-q0)*(right-left)**2*scale/12
                        records.append(dict(kind='line', source_case=base, source_element=eid,
                                            member_id=arm.member_id, direction=direction.tolist(),
                                            axial_range_mm=[left*scale, right*scale],
                                            point_mm=(direction*(left+right)*scale/2).tolist(),
                                            radius_mm=.55*arm.outer_diameter_mm,
                                            force_n=force.tolist(), moment_nmm=moment.tolist()))
    return tuple(records)


def local_load_resultant(records):
    """所有原载荷关于节点原点的合力与合矩，供两个后端及结果追溯共用。"""
    force, moment = np.zeros(3), np.zeros(3)
    for item in records:
        f = np.asarray(item['force_n'])
        force += f
        moment += np.asarray(item['moment_nmm'])+np.cross(item['point_mm'], f)
    return force.tolist(), moment.tolist()


def regional_load_forces(nodes, elements, arms, sources):
    """纯标准库算法可原样导入Abaqus旧版；表面积加权对偶保证合力与合矩。"""
    import math
    try:
        from builtins import any, max, sorted, sum
    except ImportError:
        from __builtin__ import any, max, sorted, sum

    forces = [[0., 0., 0.] for _ in nodes]
    if not sources:
        return forces, []
    faces = {}
    patterns = ((0, 1, 2, 4, 5, 6), (0, 3, 1, 7, 8, 4),
                (1, 3, 2, 8, 9, 5), (2, 3, 0, 9, 7, 6))
    for element in elements:
        for pattern in patterns:
            face = tuple(element[i] for i in pattern)
            key = tuple(sorted(face[:3]))
            faces[key] = face if key not in faces else None
    areas = {}
    for face in faces.values():
        if face is None:
            continue
        for tri in ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)):
            ids = [face[i] for i in tri]
            p, q, r = [nodes[i] for i in ids]
            u, v = [q[k]-p[k] for k in range(3)], [r[k]-p[k] for k in range(3)]
            cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            share = math.sqrt(sum(x*x for x in cross))/6
            for i in ids:
                areas[i] = areas.get(i, 0.)+share
    candidates = [i for i in sorted(areas) if areas[i] > 0 and not any(
        abs(sum(nodes[i][k]*arm['direction'][k] for k in range(3))-arm['length_mm']) < 1e-5
        for arm in arms)]
    records = []
    for source in sources:
        center = source['point_mm']
        radius = source['radius_mm']
        selected, weights = [], []
        for factor in (1., 1.5, 2.):
            selected, weights = [], []
            for i in candidates:
                rel = [nodes[i][k]-center[k] for k in range(3)]
                distance2 = sum(v*v for v in rel)
                if source['kind'] == 'line':
                    direction = source['direction']
                    axial = sum(nodes[i][k]*direction[k] for k in range(3))
                    lo, hi = source['axial_range_mm']
                    transverse2 = sum((nodes[i][k]-axial*direction[k])**2 for k in range(3))
                    inside = lo-1e-8 <= axial <= hi+1e-8 and transverse2 <= radius**2
                    weight = areas[i]
                else:
                    inside = distance2 < (factor*radius)**2
                    weight = areas[i]*max(0., 1-distance2/(factor*radius)**2)**2
                if inside and weight > 0:
                    selected.append(i)
                    weights.append(weight)
            if len(selected) >= 3:
                break
        if len(selected) < 3:
            raise ValueError('Local load region contains fewer than three surface nodes; refine the mesh.')
        region_factor = factor
        total = sum(weights)
        weights = [v/total for v in weights]
        size = math.sqrt(sum(w*sum((nodes[i][k]-center[k])**2 for k in range(3))
                             for i, w in zip(selected, weights)))
        if size <= 1e-12:
            raise ValueError('Local load region has zero radius.')
        operators = []
        gram = [[0. for _ in range(6)] for _ in range(6)]
        for i, w in zip(selected, weights):
            x, y, z = [(nodes[i][k]-center[k])/size for k in range(3)]
            operator = [[1., 0., 0., 0., z, -y], [0., 1., 0., -z, 0., x],
                        [0., 0., 1., y, -x, 0.]]
            operators.append(operator)
            for row in operator:
                for j in range(6):
                    for k in range(6):
                        gram[j][k] += w*row[j]*row[k]
        rhs = list(source['force_n'])+[v/size for v in source['moment_nmm']]
        matrix = [row+[value] for row, value in zip(gram, rhs)]
        for j in range(6):
            pivot = max(range(j, 6), key=lambda k: abs(matrix[k][j]))
            if abs(matrix[pivot][j]) < 1e-12:
                raise ValueError('Local load region cannot transmit all six resultant components.')
            matrix[j], matrix[pivot] = matrix[pivot], matrix[j]
            divisor = matrix[j][j]
            matrix[j] = [v/divisor for v in matrix[j]]
            for k in range(6):
                if k != j:
                    factor = matrix[k][j]
                    matrix[k] = [a-factor*b for a, b in zip(matrix[k], matrix[j])]
        multiplier = [row[-1] for row in matrix]
        for i, w, operator in zip(selected, weights, operators):
            for k in range(3):
                forces[i][k] += w*sum(a*b for a, b in zip(operator[k], multiplier))
        records.append(dict(kind=source['kind'], node_indices=list(selected),
                            requested_radius_mm=radius, effective_radius_mm=radius*region_factor
                            if source['kind'] == 'point' else None))
    return forces, records


def abaqus_load_helper():
    """导出同一数值算法的ASCII源码；无需旧版CAE安装NumPy或SciPy。"""
    import inspect
    return inspect.getsource(regional_load_forces).encode('ascii', errors='backslashreplace').decode('ascii')
