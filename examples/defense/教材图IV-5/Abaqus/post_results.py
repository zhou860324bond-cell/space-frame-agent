# -*- coding: utf-8 -*-
# ruff: noqa: F403, F405
"""实际 ODB 的应力、反力与网格敏感性；输出完整色标的 Abaqus 云图。"""
from __future__ import division, print_function

import json
import math
import re

from abaqus import session
from abaqusConstants import *
from odbAccess import openOdb
import visualization  # noqa: F401


def stress_stats(values):
    mises = sorted([float(v.mises) for v in values])
    if not mises:
        raise RuntimeError('No integration-point stresses were found.')
    peak = values[0]
    for value in values:
        if value.mises > peak.mises:
            peak = value
    return {'maximum_mises_MPa': mises[-1],
            'peak_mises_element': int(peak.elementLabel),
            'peak_mises_integration_point': int(peak.integrationPoint),
            'p99_mises_MPa': mises[int(0.99*(len(mises)-1))],
            'maximum_abs_principal_MPa': max([max(abs(v.maxPrincipal), abs(v.minPrincipal))
                                              for v in values]),
            'integration_point_count': len(mises)}


def vector_sum(values):
    return [math.fsum([float(v.data[k]) for v in values]) for k in range(3)]


def contour(odb, name, variable, refinement):
    if name in session.viewports:
        del session.viewports[name]
    viewport = session.Viewport(name=name, width=200, height=130)
    viewport.setValues(displayedObject=odb)
    viewport.odbDisplay.setPrimaryVariable(variableLabel=variable, outputPosition=INTEGRATION_POINT,
                                           refinement=refinement)
    viewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF,))
    viewport.odbDisplay.contourOptions.setValues(maxAutoCompute=ON, minAutoCompute=ON)
    viewport.viewportAnnotationOptions.setValues(title=OFF, state=OFF, compass=ON)
    viewport.view.setValues(session.views['Iso'])
    viewport.view.setValues(cameraUpVector=(0.0, 0.0, 1.0))
    viewport.view.fitView()
    session.pngOptions.setValues(imageSize=(1600, 1000))
    session.printOptions.setValues(vpDecorations=OFF)
    session.printToFile(fileName=name, format=PNG, canvasObjects=(viewport,))


def main():
    with open('case_meta.json') as handle:
        meta = json.load(handle)
    with open('global_results.json') as handle:
        overall = json.load(handle)
    with open('local_mesh_meta.json') as handle:
        meshes = json.load(handle)
    cut = overall['cuts']['2']['cut_wrench_N_Nm']
    expected_force = cut[:3]
    expected_moment = [v*1000.0 for v in cut[3:]]
    records = []
    for index, mesh in enumerate(meshes):
        odb = openOdb(path=str(mesh['job'])+'.odb', readOnly=True)
        inst = odb.rootAssembly.instances['JOINT-1']
        frame = odb.steps['LOCAL'].frames[-1]
        zone = inst.elementSets['JOINT_ZONE']
        anchor = odb.rootAssembly.nodeSets['RP_CUT2']
        reaction = vector_sum(frame.fieldOutputs['RF'].getSubset(region=anchor).values)
        reaction_moment = vector_sum(frame.fieldOutputs['RM'].getSubset(region=anchor).values)
        force_error = [reaction[k]-expected_force[k] for k in range(3)]
        moment_error = [reaction_moment[k]-expected_moment[k] for k in range(3)]
        if max([abs(x) for x in force_error]) > 0.2:
            raise RuntimeError('Local reaction does not reproduce the overall cut force: '+str(force_error))
        if max([abs(x) for x in moment_error]) > 30.0:
            raise RuntimeError('Local reaction moment mismatch: '+str(moment_error))
        out = dict(mesh)
        with open(str(mesh['job'])+'.dat') as handle:
            log = handle.read()
        quality = re.search(r'(\d+) elements are distorted', log)
        out['distorted_element_warning_count'] = int(quality.group(1)) if quality else 0
        out['solver_warning_note'] = 'Cut coupling notices and distorted-element warning; no numerical singularity.'
        out['whole_model_stress'] = stress_stats(frame.fieldOutputs['S'].values)
        out['joint_zone_stress'] = stress_stats(frame.fieldOutputs['S'].getSubset(region=zone).values)
        out['anchor_RF_N'] = reaction
        out['anchor_RM_Nmm'] = reaction_moment
        out['anchor_force_error_N'] = force_error
        out['anchor_moment_error_Nmm'] = moment_error
        out['max_displacement_mm_including_rigid_motion'] = max([
            math.sqrt(math.fsum([float(x)*float(x) for x in v.data]))
            for v in frame.fieldOutputs['U'].values])
        if index == len(meshes)-1:
            contour(odb, 'node3_mises_MPa_full_range', 'S', (INVARIANT, 'Mises'))
            viewport = session.viewports['node3_mises_MPa_full_range']
            viewport.odbDisplay.display.setValues(plotState=(UNDEFORMED,))
            viewport.odbDisplay.commonOptions.setValues(visibleEdges=ALL)
            session.printToFile(fileName='node3_c3d10_mesh', format=PNG, canvasObjects=(viewport,))
        records.append(out)
        odb.close()
    peak = [r['joint_zone_stress']['maximum_mises_MPa'] for r in records]
    percentiles = [r['joint_zone_stress']['p99_mises_MPa'] for r in records]
    diagnosis = {'joint_zone_peak_relative_changes': [abs(peak[k]/peak[k-1]-1.0) for k in range(1, len(peak))],
                 'joint_zone_p99_relative_changes': [abs(percentiles[k]/percentiles[k-1]-1.0)
                                                      for k in range(1, len(percentiles))],
                 'formal_Kt': None,
                 'convergence_claim': 'Three mesh results only; sharp intersection and load patch require sensitivity assessment.'}
    summary = {'input_sha256': meta['input_sha256'], 'global_units': 'N-m-Pa',
               'local_units': 'N-mm-MPa', 'global_element': 'B33', 'local_element': 'C3D10',
               'local_node': 3, 'cut_length_mm': meta['config']['cut_length_mm'],
               'meshes': records, 'mesh_sensitivity': diagnosis,
               'assumptions': ['Solid D100 cylinders, union without weld/fillet geometry.',
                               'One cut prescribed from Abaqus beam U/UR; other cut loaded by force/moment.',
                               'Cut wrench includes the outer beam consistent distributed nodal load.',
                               'Node-3 2000 N force distributed over exposed y=0, z>0 semicircular cap.',
                               'Local q mapped as body force q/A over the original horizontal cylinder.',
                               'Linear elastic continuum includes shear deformation; B33 global beam does not.',
                               'Joint zone is element-centroid radius 135 mm, not a weld hot-spot extrapolation.',
                               'No formal Kt or engineering safety certification.']}
    with open('local_results.json', 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    odb = openOdb(path='global_b33.odb', readOnly=True)
    viewport = session.Viewport(name='global_beam_deformation', width=200, height=130)
    viewport.setValues(displayedObject=odb)
    viewport.odbDisplay.display.setValues(plotState=(DEFORMED,))
    viewport.view.setValues(session.views['Iso'])
    viewport.view.fitView()
    session.pngOptions.setValues(imageSize=(1600, 1000))
    session.printToFile(fileName='global_beam_deformation', format=PNG, canvasObjects=(viewport,))
    odb.close()
    print('Postprocessing completed; stress in MPa, displacement in mm for local model.')


if __name__ == '__main__':
    main()
