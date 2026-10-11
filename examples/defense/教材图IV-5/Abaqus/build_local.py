# -*- coding: utf-8 -*-
# ruff: noqa: F403, F405
"""Abaqus 6.14 局部建模：保留均布荷载，节点集中力采用明确的分布耦合加载区。"""
from __future__ import division, print_function

import json
import math

from abaqus import mdb
from abaqusConstants import *
import assembly  # noqa: F401
import interaction  # noqa: F401
import job  # noqa: F401
import load  # noqa: F401
import material  # noqa: F401
import mesh
import part  # noqa: F401
import regionToolset
import section  # noqa: F401
import sketch  # noqa: F401
import step  # noqa: F401


def build(index, size, meta, global_result):
    name = 'Node3_C3D10_h%d' % int(size)
    model = mdb.Model(name=name)
    root = model.rootAssembly
    root.DatumCsysByDefault(CARTESIAN)
    length = meta['config']['cut_length_mm']
    radius = meta['arms'][0]['outer_diameter_mm']/2.0
    if abs(radius-50.0) > 1e-6:
        raise ValueError('This local loading region requires D100 solid circular arms.')
    instances = []
    for label, angle in (('Member1', -90.0), ('Member2', 180.0)):
        rod = model.Part(name=label, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        profile = model.ConstrainedSketch(name='profile', sheetSize=400.0)
        profile.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(radius, 0.0))
        rod.BaseSolidExtrude(sketch=profile, depth=length)
        del model.sketches['profile']
        inst = root.Instance(name=label+'-1', part=rod, dependent=ON)
        root.rotate(instanceList=(inst.name,), axisPoint=(0.0, 0.0, 0.0),
                    axisDirection=(1.0, 0.0, 0.0), angle=angle)
        instances.append(inst)
    # 保留交叠边界，便于只向原横杆圆柱域施加体力；共享节点保持连续。
    root.InstanceFromBooleanMerge(name='Joint', instances=tuple(instances),
                                  keepIntersections=ON, originalInstances=DELETE, domain=GEOMETRY)
    solid = model.parts['Joint']
    mat = model.Material(name='E210GPa')
    mat.Elastic(table=((meta['arms'][0]['E_mpa'], meta['arms'][0]['nu']),))
    model.HomogeneousSolidSection(name='SolidSection', material=mat.name)
    solid.SectionAssignment(region=regionToolset.Region(cells=solid.cells), sectionName='SolidSection')

    loaded_indices = []
    cell_info = []
    for cell in solid.cells:
        props = solid.getMassProperties(regions=solid.cells[cell.index:cell.index+1], relativeAccuracy=HIGH)
        x, y, z = props['volumeCentroid']
        cell_info.append({'index': cell.index, 'volume': props['volume'],
                          'centroid': props['volumeCentroid']})
        if -1e-6 <= y <= length+1e-6 and x*x+z*z <= radius*radius+1e-5:
            loaded_indices.append(cell.index)
    loaded = solid.cells[0:0]
    for index_cell in loaded_indices:
        loaded += solid.cells[index_cell:index_cell+1]
    properties = solid.getMassProperties(regions=loaded, relativeAccuracy=HIGH)
    expected_volume = math.pi*radius**2*length
    if abs(properties['volume']/expected_volume-1.0) > 1e-5:
        raise RuntimeError('Loaded cells do not reproduce the original horizontal cylinder.')
    if abs(properties['volumeCentroid'][1]-length/2.0) > 1e-4:
        raise RuntimeError('Uniform-load centroid is incorrect.')
    solid.Set(name='DISTRIBUTED_LOAD_VOLUME', cells=loaded)

    element = mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD)
    solid.setMeshControls(regions=solid.cells, elemShape=TET, technique=FREE)
    solid.setElementType(regions=(solid.cells,), elemTypes=(element,))
    solid.seedPart(size=size, deviationFactor=0.1, minSizeFactor=0.1)
    solid.generateMesh()
    hot_labels = []
    for elem in solid.elements:
        coords = [solid.nodes[k].coordinates for k in elem.connectivity]
        center = [math.fsum([p[k] for p in coords])/float(len(coords)) for k in range(3)]
        if math.fsum([x*x for x in center]) <= 135.0**2:
            hot_labels.append(elem.label)
    solid.Set(name='JOINT_ZONE', elements=solid.elements.sequenceFromLabels(labels=tuple(hot_labels)))
    root.regenerate()
    inst = root.instances['Joint-1']
    model.StaticStep(name='LOCAL', previous='Initial', nlgeom=OFF,
                     initialInc=1.0, maxInc=1.0, maxNumInc=1)
    model.FieldOutputRequest(name='F-Local', createStepName='LOCAL', variables=('S', 'U', 'RF', 'RM'))
    for mid, end, probe in ((1, (0.0, length, 0.0), (20.0, length, 0.0)),
                            (2, (0.0, 0.0, -length), (20.0, 0.0, -length))):
        faces = inst.faces.findAt((probe,))
        surface = root.Surface(name='CUT_MEMBER%d' % mid, side1Faces=faces)
        reference = root.ReferencePoint(point=end)
        region = root.Set(name='RP_CUT%d' % mid, referencePoints=(root.referencePoints[reference.id],))
        model.Coupling(name='CUT_COUPLING%d' % mid, controlPoint=region, surface=surface,
                       influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                       u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
        cut = global_result['cuts'][str(mid)]
        if mid == 2:
            displacement = [v*1000.0 for v in cut['U_m']]
            rotation = cut['UR_rad']
            model.DisplacementBC(name='BEAM_CUT_DISPLACEMENT', createStepName='LOCAL', region=region,
                                 u1=displacement[0], u2=displacement[1], u3=displacement[2],
                                 ur1=rotation[0], ur2=rotation[1], ur3=rotation[2])
        else:
            f = cut['cut_wrench_N_Nm'][:3]
            m = [v*1000.0 for v in cut['cut_wrench_N_Nm'][3:]]
            model.ConcentratedForce(name='BEAM_CUT_FORCE', createStepName='LOCAL', region=region,
                                    cf1=f[0], cf2=f[1], cf3=f[2])
            model.Moment(name='BEAM_CUT_MOMENT', createStepName='LOCAL', region=region,
                         cm1=m[0], cm2=m[1], cm3=m[2])

    # 原 JSON 的节点荷载只指定合力；取 y=0、z>0 的暴露半圆端面作有限加载区。
    # 参考点位于原节点中心，分布耦合传递 2000 N、零力矩，不将加载区刚性化。
    central_faces = inst.faces.findAt(((15.0, 0.0, 25.0),))
    central = root.Surface(name='NODE3_LOAD_PATCH', side1Faces=central_faces)
    reference = root.ReferencePoint(point=(0.0, 0.0, 0.0))
    region = root.Set(name='RP_NODE3_LOAD', referencePoints=(root.referencePoints[reference.id],))
    model.Coupling(name='NODE3_FORCE_DISTRIBUTION', controlPoint=region, surface=central,
                   influenceRadius=WHOLE_SURFACE, couplingType=DISTRIBUTING,
                   weightingMethod=UNIFORM, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    direct = [0.0]*6
    for nodal in meta['nodal_loads']:
        if nodal['node'] == 3:
            direct = [direct[k]+nodal['load'][k] for k in range(6)]
    model.ConcentratedForce(name='NODE3_DIRECT_FORCE', createStepName='LOCAL', region=region,
                            cf1=direct[0], cf2=direct[1], cf3=direct[2])
    if any(direct[3:]):
        raise ValueError('This example load patch does not include a direct nodal moment.')
    # q=-800 N/m=-0.8 N/mm。体力 q/A 在原圆柱域积分后给出 -160 N，形心 y=100 mm。
    q = [0.0]*3
    with open('model.json', 'rb') as handle:
        source = json.loads(handle.read().decode('utf-8-sig'))
    for span in source['load_cases'][0]['member_spans']:
        if span['member'] == 1 and span['kind'] == 'partial':
            if span['a'] != 0 or span['b']*1000.0 < length:
                raise ValueError('Uniform-load interval must cover the local horizontal arm.')
            q = [q[k]+span['w1'][k]/1000.0 for k in range(3)]
    body = [v/(math.pi*radius**2) for v in q]
    model.BodyForce(name='LOCAL_PARTIAL_UNIFORM_LOAD', createStepName='LOCAL',
                    region=inst.sets['DISTRIBUTED_LOAD_VOLUME'],
                    comp1=body[0], comp2=body[1], comp3=body[2])
    jobname = 'local_h%d' % int(size)
    analysis = mdb.Job(name=jobname, model=name, numCpus=1, numDomains=1, memory=4096, memoryUnits=MEGA_BYTES)
    analysis.writeInput(consistencyChecking=ON)
    return {'job': jobname, 'model': name, 'mesh_size_mm': size, 'nodes': len(solid.nodes),
            'elements': len(solid.elements), 'body_force_N_per_mm3': body,
            'loaded_volume_mm3': properties['volume'], 'load_centroid_mm': properties['volumeCentroid'],
            'cell_info': cell_info, 'node3_load_patch': 'exposed y=0 semicircular cap, z>0'}


def main():
    with open('case_meta.json') as handle:
        meta = json.load(handle)
    with open('global_results.json') as handle:
        result = json.load(handle)
    mdb.ModelFromInputFile(name='Global_B33', inputFileName='global_b33.inp')
    # 6.14 不允许先删除仅存的默认模型；已有整体模型后才清理空模型。
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']
    records = []
    for index, size in enumerate(meta['config']['mesh_sizes_mm']):
        records.append(build(index, size, meta, result))
        print('Built '+records[-1]['job'])
    mdb.saveAs(pathName='IV5_global_and_node3.cae')
    with open('local_mesh_meta.json', 'w') as handle:
        json.dump(records, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
