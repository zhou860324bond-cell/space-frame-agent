from __future__ import print_function
import os
import sys
import json
import numpy as np
from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT

stem = sys.argv[-1]
odb = openOdb(stem + '.odb', readOnly=True)
frame = odb.steps[odb.steps.keys()[-1]].frames[-1]
instance = max(odb.rootAssembly.instances.values(), key=lambda item: len(item.nodes))
labels = np.array([node.label for node in instance.nodes], dtype=np.int32)
coordinates = np.array([node.coordinates for node in instance.nodes], dtype=float)
index = dict((int(label), offset) for offset, label in enumerate(labels))
arrays = dict(labels=labels, coordinates=coordinates)
available = list(frame.fieldOutputs.keys())
for name in ('U', 'UR', 'RF', 'RM'):
    values = np.zeros((len(labels), 3))
    if name in available:
        for value in frame.fieldOutputs[name].getSubset(region=instance).values:
            if value.nodeLabel in index:
                values[index[value.nodeLabel], :len(value.data)] = value.data
    arrays[name] = values
if 'S' in available:
    field = frame.fieldOutputs['S'].getSubset(region=instance, position=INTEGRATION_POINT)
    arrays['stress_labels'] = np.array([value.elementLabel for value in field.values], dtype=np.int32)
    arrays['stress_ip'] = np.array([value.integrationPoint for value in field.values], dtype=np.int32)
    arrays['stress'] = np.array([value.data for value in field.values], dtype=float)
    arrays['mises'] = np.array([value.mises for value in field.values], dtype=float)
for name in ('SF', 'SM'):
    if name in available:
        field = frame.fieldOutputs[name].getSubset(region=instance)
        arrays[name] = np.array([value.data for value in field.values], dtype=float)
        arrays[name + '_labels'] = np.array([value.elementLabel for value in field.values], dtype=np.int32)
        arrays[name + '_ip'] = np.array([value.integrationPoint for value in field.values], dtype=np.int32)
np.savez_compressed(stem + '_abaqus.npz', **arrays)
history = {}
for region_name, region in odb.steps[odb.steps.keys()[-1]].historyRegions.items():
    for name, output in region.historyOutputs.items():
        if name.startswith('ALL') and output.data:
            history[region_name + '/' + name] = float(output.data[-1][1])
metadata = dict(odb=os.path.abspath(stem + '.odb'), step=odb.steps.keys()[-1],
                frame=frame.frameId, time=frame.frameValue, instance=instance.name,
                nodes=len(instance.nodes), elements=len(instance.elements),
                fields=list(frame.fieldOutputs.keys()), history=history,
                version=odb.jobData.version)
with open(stem + '_odb_metadata.json', 'w') as handle:
    json.dump(metadata, handle, indent=2)
odb.close()
print(json.dumps(metadata))
