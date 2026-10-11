# -*- coding: utf-8 -*-
# ruff: noqa: F403, F405
"""原软件网格/已解标量导入ODB并原生出图；不重划网格，不冒充Abaqus复算。"""
from __future__ import division, print_function

import json
import os

from abaqus import session
from abaqusConstants import *
from odbAccess import Odb
import visualization  # noqa: F401


def main():
    with open('native_mesh.json') as handle:
        original = json.load(handle)
    filename = 'node3_original_mesh.odb'
    if os.path.isfile(filename):
        os.remove(filename)
    odb = Odb(name='IV5_Node3_OriginalMesh', path=filename,
              analysisTitle='Original Gmsh mesh and native solver results',
              description='Imported results, not a new Abaqus solver run.')
    part = odb.Part(name='JOINT', embeddedSpace=THREE_D, type=DEFORMABLE_BODY)
    nodes = tuple(tuple([index + 1] + xyz) for index, xyz in enumerate(original['nodes']))
    elements = tuple(tuple([index + 1] + [n + 1 for n in conn])
                     for index, conn in enumerate(original['elements']))
    part.addNodes(nodeData=nodes, nodeSetName='ALL_NODES')
    part.addElements(elementData=elements, type='C3D10', elementSetName='ALL_ELEMENTS')
    inst = odb.rootAssembly.Instance(name='JOINT-1', object=part)
    step = odb.Step(name='IMPORTED', description='Original software results, N-mm-MPa',
                    domain=TIME, timePeriod=1.0)
    frame = step.Frame(incrementNumber=1, frameValue=1.0,
                       description='Original element maximum Mises, without averaging')
    mises = frame.FieldOutput(name='Mises_MPa', type=SCALAR,
                              description='Native solver element maximum Mises (MPa)')
    # 6.14主变量不接受任意CENTROID标量；同一单元统计值重复到四点供QUILT显示。
    mises.addData(position=INTEGRATION_POINT, instance=inst,
                   labels=tuple(range(1, len(elements) + 1)),
                   data=tuple((value,) for value in original['element_mises'] for _ in range(4)))
    displacement = frame.FieldOutput(name='U', type=VECTOR,
                                      description='Native solver displacement (mm)')
    displacement.addData(position=NODAL, instance=inst,
                          labels=tuple(range(1, len(nodes) + 1)),
                          data=tuple(tuple(v) for v in original['displacement']))
    odb.save()
    odb.close()
    odb = session.openOdb(name=filename, readOnly=True)
    imported = odb.rootAssembly.instances['JOINT-1']
    assert len(imported.nodes) == len(nodes)
    assert len(imported.elements) == len(elements)
    assert all(tuple(element.connectivity) == elements[index][1:]
               for index, element in enumerate(imported.elements))
    coordinate_error = max([abs(node.coordinates[axis] - original['nodes'][index][axis])
                            for index, node in enumerate(imported.nodes) for axis in range(3)])
    assert coordinate_error < 2e-5
    imported_values = odb.steps['IMPORTED'].frames[-1].fieldOutputs['Mises_MPa'].values
    assert len(imported_values) == 4 * len(elements)
    imported_mises = [float(imported_values[index].data)
                      for index in range(0, len(imported_values), 4)]
    stress_error = max([abs(imported_mises[index] - original['element_mises'][index])
                        for index in range(len(elements))])
    assert stress_error < 2e-6
    defaults = session.defaultGraphicsOptions
    session.graphicsOptions.setValues(backgroundStyle=defaults.backgroundStyle,
                                       backgroundColor=defaults.backgroundColor,
                                       backgroundBottomColor=defaults.backgroundBottomColor)
    viewport = session.Viewport(name='Original mesh', width=240, height=140.5)
    viewport.setValues(displayedObject=odb)
    viewport.odbDisplay.setFrame(step=0, frame=-1)
    viewport.odbDisplay.setPrimaryVariable(variableLabel='Mises_MPa', outputPosition=INTEGRATION_POINT)
    viewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF,))
    viewport.odbDisplay.commonOptions.setValues(visibleEdges=NONE, renderStyle=SHADED)
    session.Spectrum(name='EngineeringMises',
                      colors=('#0000FF', '#004DFF', '#00A0FF', '#00D8FF',
                              '#00FFDA', '#00FF8A', '#32FF00', '#8CFF00',
                              '#DEFF00', '#FFFF00', '#FF8C00', '#FF0000'))
    viewport.odbDisplay.contourOptions.setValues(
        contourType=QUILT, contourStyle=UNIFORM, numIntervals=12,
        spectrum='EngineeringMises', minAutoCompute=OFF, minValue=0.0, maxAutoCompute=ON)
    viewport.viewportAnnotationOptions.setValues(
        title=OFF, state=OFF, compass=OFF, triad=ON, triadLabels=LETTERS,
        triadPosition=(7, 11), triadSize=4, legend=ON, legendBox=OFF,
        legendPosition=(82, 86), legendDecimalPlaces=2)
    viewport.view.setValues(session.views['Iso'])
    viewport.view.setValues(cameraUpVector=(0.0, 0.0, 1.0))
    viewport.view.fitView()
    note = odb.userData.Text(
        name='Source', text='IV-5 | Node 3 | Mises (MPa)\n4415 nodes / 2500 C3D10\n'
        'Source: native solver; imported results',
        anchor=TOP_LEFT, referencePoint=TOP_LEFT, offset=(6.0, -6.0), color='White',
        font='-*-verdana-medium-r-*-*-*-100-*-*-*-*-iso8859-1', box=OFF)
    viewport.plotAnnotation(annotation=note)
    session.pngOptions.setValues(imageSize=(1920, 1080))
    session.printOptions.setValues(vpBackground=ON, vpDecorations=OFF, reduceColors=OFF)
    session.printToFile(fileName='node3_original_mesh_abaqus_default', format=PNG,
                        canvasObjects=(viewport,))
    viewport.odbDisplay.commonOptions.setValues(visibleEdges=ALL)
    session.printToFile(fileName='node3_original_mesh_abaqus_edges', format=PNG,
                        canvasObjects=(viewport,))
    with open('same_mesh_verification.json', 'w') as handle:
        json.dump({'source_npz_sha256': original['source_npz_sha256'],
                   'nodes': len(nodes), 'elements': len(elements),
                   'connectivity_identical': True,
                   'maximum_coordinate_storage_error_mm': coordinate_error,
                   'maximum_scalar_storage_error_MPa': stress_error,
                   'mises_max_MPa': max(original['element_mises']),
                   'background_style': str(defaults.backgroundStyle),
                   'background_color': defaults.backgroundColor,
                   'background_bottom_color': defaults.backgroundBottomColor,
                   'result_source': 'native-c3d10+gmsh; imported, not Abaqus solved',
                   'deformation_scale': 0}, handle, indent=2)
    odb.close()
    print('SAME_MESH_NATIVE_RENDER_COMPLETE')


if __name__ == '__main__':
    main()
