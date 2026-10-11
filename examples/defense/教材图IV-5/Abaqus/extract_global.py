# -*- coding: utf-8 -*-
"""由 Abaqus 自带 Python 读取整体 ODB；保留切口逐单元 NFORC 原始值。"""
from __future__ import division, print_function

import json
import math

from odbAccess import openOdb


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def nodal_equivalent(meta, eid, node):
    element = next(e for e in meta['elements'] if e['id'] == eid)
    pi, pj = meta['nodes'][str(element['i'])], meta['nodes'][str(element['j'])]
    vector = [pj[k]-pi[k] for k in range(3)]
    length = math.sqrt(sum(x*x for x in vector))
    axis = [x/length for x in vector]
    out = [0.0]*6
    sign = 1.0 if node == element['i'] else -1.0
    for load in meta['distributed_loads']:
        if load['element'] != eid:
            continue
        moment = cross(axis, load['w'])
        for k in range(3):
            out[k] += load['w'][k]*length/2.0
            out[k+3] += sign*moment[k]*length**2/12.0
    return out


def main():
    with open('case_meta.json') as handle:
        meta = json.load(handle)
    odb = openOdb(path='global_b33.odb', readOnly=True)
    frame = odb.steps['GLOBAL'].frames[-1]
    out = {'fields': list(frame.fieldOutputs.keys()), 'nodes': {}, 'cuts': {}}
    for field in ('U', 'UR', 'RF', 'RM'):
        if field not in frame.fieldOutputs:
            raise RuntimeError('Missing global output: '+field)
        for value in frame.fieldOutputs[field].values:
            row = out['nodes'].setdefault(str(value.nodeLabel), {})
            row[field] = [float(x) for x in value.data]
    # Abaqus 6.14 将 NFORC 分为六个标量 FieldOutput，不假定新版向量接口。
    for mid, cut in meta['cuts'].items():
        row = dict(cut)
        row['nforc_inner'] = [0.0]*6
        row['nforc_outer'] = [0.0]*6
        for component in range(6):
            name = 'NFORC%d' % (component+1)
            if name not in frame.fieldOutputs:
                raise RuntimeError('Missing beam end force: '+name)
            matches = {'inner': 0, 'outer': 0}
            for value in frame.fieldOutputs[name].values:
                if value.nodeLabel != cut['node']:
                    continue
                for side in ('inner', 'outer'):
                    if value.elementLabel == cut[side+'_element']:
                        row['nforc_'+side][component] = float(value.data)
                        matches[side] += 1
            if matches != {'inner': 1, 'outer': 1}:
                raise RuntimeError('Ambiguous NFORC at cut '+mid)
        row['U_m'] = out['nodes'][str(cut['node'])]['U']
        row['UR_rad'] = out['nodes'][str(cut['node'])]['UR']
        # NFORC 是单元内力作用，受均布荷载的梁还需计入该单元的一致节点荷载。
        # 在本例真实 ODB 中，直接使用 NFORC 会漏掉外段的 1120 N 和端弯矩。
        row['equivalent_outer_load'] = nodal_equivalent(meta, cut['outer_element'], cut['node'])
        row['cut_wrench_N_Nm'] = [row['nforc_outer'][k]+row['equivalent_outer_load'][k]
                                 for k in range(6)]
        out['cuts'][mid] = row
    out['beam_section_outputs'] = {}
    for name in ('SF', 'SM'):
        if name in frame.fieldOutputs:
            out['beam_section_outputs'][name] = [
                {'element': v.elementLabel, 'ip': v.integrationPoint,
                 'data': [float(x) for x in v.data]} for v in frame.fieldOutputs[name].values]
    with open('global_results.json', 'w') as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    odb.close()
    print('Global ODB extracted. Node 3 Ux = %.9f mm' % (out['nodes']['3']['U'][0]*1000.0))


if __name__ == '__main__':
    main()
