# -*- coding: utf-8 -*-
"""Extract nodal displacements and reactions from an Abaqus ODB into CSV.

MUST be run by Abaqus' own Python 2 interpreter:

    abaqus python extract_odb.py job1.odb job2.odb ...

Writes <job>_abaqus.csv next to each odb with columns:
    node, u1, u2, u3, ur1, ur2, ur3, rf1, rf2, rf3, rm1, rm2, rm3

Python 2 syntax throughout (Abaqus 6.14 ships Python 2.7): no f-strings,
no print(a, b), and 1/2 == 0 so float division is written explicitly.
"""

import os
import sys

from odbAccess import openOdb

FIELDS = ("U", "UR", "RF", "RM")
COLUMNS = ("u1", "u2", "u3", "ur1", "ur2", "ur3",
           "rf1", "rf2", "rf3", "rm1", "rm2", "rm3")
SLOTS = {"U": (0, 3), "UR": (3, 6), "RF": (6, 9), "RM": (9, 12)}


def extract(odb_path):
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step = odb.steps[odb.steps.keys()[-1]]
        frame = step.frames[-1]
        rows = {}

        for name in FIELDS:
            if name not in frame.fieldOutputs.keys():
                continue
            start, stop = SLOTS[name]
            for value in frame.fieldOutputs[name].values:
                label = int(value.nodeLabel)
                row = rows.setdefault(label, [0.0] * 12)
                data = value.data
                for k in range(min(stop - start, len(data))):
                    row[start + k] = float(data[k])

        out_path = os.path.splitext(odb_path)[0] + "_abaqus.csv"
        handle = open(out_path, "w")
        try:
            handle.write("node," + ",".join(COLUMNS) + "\n")
            for label in sorted(rows.keys()):
                cells = ["%.12e" % v for v in rows[label]]
                handle.write("%d,%s\n" % (label, ",".join(cells)))
        finally:
            handle.close()
        print "wrote %s  (%d nodes)" % (out_path, len(rows))
        return out_path
    finally:
        odb.close()


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = [f for f in os.listdir(".") if f.endswith(".odb")]
    if not targets:
        print "no .odb files given or found in this directory"
        return 1
    for path in targets:
        if not os.path.isfile(path):
            print "skip %s (not found)" % path
            continue
        extract(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
