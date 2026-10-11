# -*- coding: utf-8 -*-
"""把图 IV-5 原题编成 B33 输入；在荷载边界与局部切口增设分析节点。"""
from __future__ import division, print_function

import hashlib
import json
import math
import os


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def norm(a):
    return math.sqrt(sum(x*x for x in a))


def prepare():
    with open('model.json', 'rb') as handle:
        raw = handle.read()
    model = json.loads(raw.decode('utf-8-sig'))
    if model['units'] != 'N-m-Pa':
        raise ValueError('This case requires N-m-Pa input units.')
    config = {'local_node': 3, 'cut_length_mm': 200.0, 'mesh_sizes_mm': [20.0, 15.0, 10.0]}
    if os.path.isfile('config.json'):
        with open('config.json') as handle:
            config.update(json.load(handle))
    if config['local_node'] != 3:
        raise ValueError('This verified case is restricted to node 3.')
    nodes = dict((n['id'], [float(n[k]) for k in 'xyz']) for n in model['nodes'])
    expected = {1: [0, 0, 0], 2: [3, 9, 0], 3: [0, 0, 3], 4: [0, 6, 3]}
    if nodes != expected:
        raise ValueError('This example requires the original IV-5 geometry.')
    sections = dict((s['name'], s) for s in model['sections'])
    materials = dict((m['name'], m) for m in model['materials'])
    case = model['load_cases'][0]
    if len(model['load_cases']) != 1:
        raise ValueError('Select exactly one case for this example.')
    cut = config['cut_length_mm']/1000.0
    if not 0.15 <= cut < 1.0:
        raise ValueError('Cut length must be between 150 and 1000 mm.')
    specs = []
    lines = ['*HEADING', 'IV5 original case, B33 Euler-Bernoulli, N-m-Pa',
             '*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO']
    elements = []
    cuts = {}
    point_loads = []
    distributed = []
    next_node = max(nodes)+1
    for member in model['members']:
        mid = member['id']
        pi, pj = nodes[member['i']], nodes[member['j']]
        vector = [pj[k]-pi[k] for k in range(3)]
        length = norm(vector)
        axis = [v/length for v in vector]
        # 圆截面无方向刚度差异，显式给出一个正交截面方向。
        ref = [0.0, 0.0, 1.0] if abs(axis[2]) < 0.99 else [0.0, 1.0, 0.0]
        n1 = cross(ref, axis)
        n1 = [v/norm(n1) for v in n1]
        section = sections[member['section']]
        material = materials[member['material']]
        if section.get('Ay') or section.get('Az'):
            raise ValueError('B33 case must omit shear areas Ay/Az.')
        if abs(section['Iy']-section['Iz']) > 1e-10 or not section.get('circular'):
            raise ValueError('This local case requires a solid circular section.')
        radius = math.sqrt(section['A']/math.pi)
        if abs(section['Iy']-math.pi*radius**4/4.0) > section['Iy']*1e-6:
            raise ValueError('The section must be solid circular, not hollow.')
        spans = [s for s in case.get('member_spans', []) if s['member'] == mid]
        positions = [0.0, length]
        for span in spans:
            if span['kind'] == 'partial':
                positions.extend([float(span['a']), float(span['b'])])
            elif span['kind'] == 'point':
                positions.append(float(span['a']))
            else:
                raise ValueError('This case supports only partial and point spans.')
        local_end = None
        if 3 in (member['i'], member['j']):
            local_end = cut if member['i'] == 3 else length-cut
            positions.append(local_end)
            direction = axis if member['i'] == 3 else [-x for x in axis]
            specs.append({'member_id': mid, 'direction': direction,
                          'outer_diameter_mm': radius*2000.0, 'length_mm': cut*1000.0,
                          'E_mpa': material['E']/1e6, 'nu': material['nu']})
        positions = sorted(set(round(x, 12) for x in positions))
        labels = []
        for distance in positions:
            if distance == 0.0:
                label = member['i']
            elif abs(distance-length) < 1e-9:
                label = member['j']
            else:
                label = next_node
                next_node += 1
                nodes[label] = [pi[k]+axis[k]*distance for k in range(3)]
            labels.append(label)
        member_elements = []
        for index in range(len(positions)-1):
            eid = len(elements)+1
            start, end = positions[index:index+2]
            row = {'id': eid, 'member': mid, 'i': labels[index], 'j': labels[index+1],
                   'a': start, 'b': end, 'n1': n1, 'section': section, 'material': material}
            elements.append(row)
            member_elements.append(row)
            for span in spans:
                if span['kind'] == 'partial' and span['a'] <= (start+end)/2.0 <= span['b']:
                    distributed.append({'element': eid, 'w': span['w1']})
        for span in spans:
            if span['kind'] == 'point':
                index = min(range(len(positions)), key=lambda k: abs(positions[k]-span['a']))
                point_loads.append({'node': labels[index], 'load': list(span['w1'])+[0.0]*3})
        if local_end is not None:
            index = min(range(len(positions)), key=lambda k: abs(positions[k]-local_end))
            outside = member_elements[index] if member['i'] == 3 else member_elements[index-1]
            inside = member_elements[index-1] if member['i'] == 3 else member_elements[index]
            cuts[str(mid)] = {'node': labels[index], 'outer_element': outside['id'],
                             'inner_element': inside['id'], 'position_m': nodes[labels[index]]}
    lines.append('*NODE')
    for label in sorted(nodes):
        lines.append('%d, %.14g, %.14g, %.14g' % tuple([label]+nodes[label]))
    for row in elements:
        lines.extend(['*ELEMENT, TYPE=B33, ELSET=E%d' % row['id'],
                      '%d, %d, %d' % (row['id'], row['i'], row['j'])])
        s, m = row['section'], row['material']
        lines.extend(['*BEAM GENERAL SECTION, ELSET=E%d, SECTION=GENERAL' % row['id'],
                      '%.14g, %.14g, 0., %.14g, %.14g' % (s['A'], s['Iy'], s['Iz'], s['J']),
                      '%.14g, %.14g, %.14g' % tuple(row['n1']),
                      '%.14g, %.14g' % (m['E'], m['E']/(2.0*(1.0+m['nu'])))])
    lines.append('*BOUNDARY')
    for support in model['supports']:
        for index, fixed in enumerate(support['fix']):
            if fixed:
                lines.append('%d, %d, %d, 0.' % (support['node'], index+1, index+1))
    lines.extend(['*STEP, NAME=GLOBAL, NLGEOM=NO', '*STATIC', '1., 1.', '*CLOAD'])
    for load in case.get('nodal_loads', [])+point_loads:
        for index, value in enumerate(load['load']):
            if value:
                lines.append('%d, %d, %.14g' % (load['node'], index+1, value))
    lines.append('*DLOAD')
    for load in distributed:
        for index, value in enumerate(load['w']):
            if value:
                lines.append('E%d, %s, %.14g' % (load['element'], ['PX', 'PY', 'PZ'][index], value))
    lines.extend(['*OUTPUT, FIELD', '*NODE OUTPUT', 'U, UR, RF, RM',
                  '*ELEMENT OUTPUT', 'NFORC, SF, SM', '*END STEP'])
    with open('global_b33.inp', 'w') as handle:
        handle.write('\n'.join(lines)+'\n')
    metadata = {'input_sha256': hashlib.sha256(raw).hexdigest(), 'units_global': 'N-m-Pa',
                'units_local': 'N-mm-MPa', 'case': case['name'], 'config': config,
                'nodes': nodes, 'elements': elements, 'cuts': cuts, 'arms': specs,
                'distributed_loads': distributed, 'nodal_loads': case['nodal_loads']+point_loads,
                'local_origin_m': nodes[3], 'global_element': 'B33',
                'local_element': 'C3D10'}
    with open('case_meta.json', 'w') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True, sort_keys=True)
    print('Prepared %d nodes / %d B33 elements; original 4 nodes / 3 physical members retained.' %
          (len(nodes), len(elements)))
    return metadata


if __name__ == '__main__':
    prepare()
