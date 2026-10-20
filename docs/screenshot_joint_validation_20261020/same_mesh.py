"""将存档 C3D10 与切面平均运动约束原样写入 Abaqus，用真实求解核对。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.linalg import qr

ROOT = Path(os.environ.get('SPACE_FRAME_REPO', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / 'src'))
from abaqus_backend import find_abaqus, solver_environment, write_env_file
from solid3d import SolidMesh
from solid_joint import JointArm, JointSpec
from joint_transfer import joint_constraints

OUT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main(name):
    source = (OUT / name / 'native/node_2_LC1' if name == 'two_arm'
              else ROOT / 'results/native_solid_joint/node_2_LC1')
    original = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    arrays = np.load(source / 'native_joint_0.npz')
    raw = original['boundary_spec'].copy()
    raw['arms'] = tuple(JointArm(**arm) for arm in raw['arms'])
    spec = JointSpec(**raw)
    mesh = SolidMesh(arrays['nodes'], arrays['elements'])
    cuts = {}
    for arm in spec.arms:
        direction = np.array(arm.direction)
        axial = mesh.nodes @ direction
        tolerance = max(arm.length_mm * 1e-7, 1e-6)
        radial = np.linalg.norm(mesh.nodes - axial[:, None] * direction, axis=1)
        cuts[arm.member_id] = np.flatnonzero(
            (np.abs(axial - arm.length_mm) <= tolerance)
            & (radial <= .5 * arm.outer_diameter_mm + tolerance))
    C, given = joint_constraints(spec, mesh, cuts)
    active = np.unique(C.indices)
    dense = C[:, active].toarray()
    _, _, order = qr(dense, mode='economic', pivoting=True)
    pivots = active[order[:C.shape[0]]]
    pivot_matrix = C[:, pivots].toarray()
    equivalent = np.linalg.solve(pivot_matrix, dense)
    prescribed = np.linalg.solve(pivot_matrix, given)
    # 等价行变换只改变约束的表达，消除所有其它被消元自由度。
    for row, pivot in enumerate(pivots):
        equivalent[:, np.flatnonzero(active == pivot)[0]] = np.eye(len(pivots))[:, row]
    expected = C @ arrays['displacement'].ravel() - given
    assert np.max(np.abs(expected)) < 1e-8
    directory = OUT / name / 'same_mesh'
    directory.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ['TEMP']) / ('sfshot_20261020_' + name)
    work.mkdir(exist_ok=True)
    inp = work / 'matched.inp'
    with inp.open('w', encoding='ascii', newline='\n') as handle:
        handle.write('*Heading\nHistorical C3D10 and exact mean-motion constraints; N mm MPa\n*Node\n')
        for i, xyz in enumerate(mesh.nodes, 1):
            handle.write('%d, %.17g, %.17g, %.17g\n' % (i, *xyz))
        first_ref = len(mesh.nodes) + 1
        for i in range(len(pivots)):
            handle.write('%d, 0., 0., 0.\n' % (first_ref + i))
        handle.write('*Element, type=C3D10, elset=SOLID\n')
        for i, conn in enumerate(mesh.elements + 1, 1):
            handle.write(', '.join(map(str, [i, *conn])) + '\n')
        handle.write('*Material, name=STEEL\n*Elastic\n%.17g, %.17g\n' %
                     (spec.elastic_modulus_mpa, spec.poisson))
        handle.write('*Solid Section, elset=SOLID, material=STEEL\n,\n')
        pivot_set = set(pivots)
        for row, pivot in enumerate(pivots):
            terms = [(int(pivot // 3 + 1), int(pivot % 3 + 1), 1.),
                     (first_ref + row, 1, -1.)]
            for dof, coefficient in zip(active, equivalent[row], strict=True):
                if dof not in pivot_set and coefficient != 0.:
                    terms.append((int(dof // 3 + 1), int(dof % 3 + 1), float(coefficient)))
            handle.write('*Equation\n%d\n' % len(terms))
            for offset in range(0, len(terms), 4):
                handle.write(', '.join('%d, %d, %.17g' % term for term in terms[offset:offset+4]) + '\n')
        handle.write('*Boundary\n')
        for i in range(len(pivots)):
            handle.write('%d, 2, 3, 0.\n' % (first_ref + i))
        handle.write('*Step, name=Static, nlgeom=NO\n*Static\n1., 1.\n*Boundary\n')
        for i, value in enumerate(prescribed):
            handle.write('%d, 1, 1, %.17g\n' % (first_ref + i, value))
        loads = arrays['applied_load_n']
        if np.any(loads):
            handle.write('*Cload\n')
            for node, axis in np.argwhere(loads != 0.):
                handle.write('%d, %d, %.17g\n' % (node+1, axis+1, loads[node, axis]))
        handle.write('*Output, field\n*Node Output\nU, RF\n*Element Output\nS\n')
        handle.write('*Output, history\n*Energy Output\nALLSE, ALLWK, ALLAE\n*End Step\n')
    write_env_file(work)
    executable = find_abaqus()
    for argv, logfile in [([executable, 'job=matched', 'cpus=2', 'interactive', 'ask_delete=OFF'], 'solve_console.txt'),
                          ([executable, 'python', str(OUT / 'abq_extract.py'), 'matched'], 'post_console.txt')]:
        print(name, argv[1], flush=True)
        with (work / logfile).open('w', encoding='utf-8') as log:
            result = subprocess.run(argv, cwd=work, env=solver_environment(), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(str(work / logfile))
    assert 'COMPLETED SUCCESSFULLY' in (work/'matched.sta').read_text()
    abq = np.load(work / 'matched_abaqus.npz')
    node_order = np.argsort(abq['labels'])
    all_u, all_rf = abq['U'][node_order], abq['RF'][node_order]
    u = all_u[:len(mesh.nodes)]
    ref_rf = all_rf[len(mesh.nodes):, 0]
    recovered_rf = np.zeros(mesh.nodes.size)
    recovered_rf[active] = equivalent.T @ ref_rf
    recovered_rf = recovered_rf.reshape((-1, 3))
    order = np.lexsort((abq['stress_ip'], abq['stress_labels']))
    stress = abq['stress'][order][:, [0, 1, 2, 3, 5, 4]].reshape((-1, 4, 6))
    mises = abq['mises'][order].reshape((-1, 4)).max(axis=1)
    def error(a, b):
        return float(np.linalg.norm(a-b) / np.linalg.norm(b))
    row = dict(case=name, same_nodes=len(mesh.nodes), same_elements=len(mesh.elements),
        constraints=len(given), constraint_pivot_condition=float(np.linalg.cond(pivot_matrix)),
        historical_constraint_residual=float(np.max(np.abs(expected))),
        abaqus_constraint_residual_mm=float(np.max(np.abs(C @ u.ravel() - given))),
        displacement_relative_l2=error(u, arrays['displacement']),
        reaction_relative_l2=error(recovered_rf, arrays['reaction']),
        mean_stress_relative_l2=error(stress.mean(axis=1), arrays['gauss_stress'].mean(axis=1)),
        element_max_mises_relative_l2=error(mises, arrays['element_mises']),
        native_peak_mpa=float(arrays['element_mises'].max()),
        abaqus_peak_mpa=float(mises.max()),
        native_max_u_mm=float(np.linalg.norm(arrays['displacement'],axis=1).max()),
        abaqus_max_u_mm=float(np.linalg.norm(u,axis=1).max()),
        native_energy_nmm=float(.5*np.sum(arrays['reaction']*arrays['displacement'])),
        abaqus_boundary_work_nmm=float(.5*np.dot(ref_rf,prescribed)),
        abaqus_force_balance_n=recovered_rf.sum(axis=0).tolist(),
        abaqus_moment_balance_nmm=np.cross(mesh.nodes,recovered_rf).sum(axis=0).tolist(),
        cuts={str(mid):dict(force_n=recovered_rf[ids].sum(axis=0).tolist(),
             moment_nmm=np.cross(mesh.nodes[ids],recovered_rf[ids]).sum(axis=0).tolist()) for mid,ids in cuts.items()},
        constraint_export='C u = g; QR chooses independent pivots; invertible row transformation; exact Abaqus EQUATION with prescribed auxiliary DOFs')
    metadata=json.loads((work/'matched_odb_metadata.json').read_text())
    row['odb_metadata']=metadata
    save(directory/'comparison.json', row)
    np.savez_compressed(directory/'paired.npz',nodes=mesh.nodes,elements=mesh.elements,
        native_u=arrays['displacement'],abaqus_u=u,native_reaction=arrays['reaction'],
        abaqus_reaction=recovered_rf,native_stress_mean=arrays['gauss_stress'].mean(axis=1),
        abaqus_stress_mean=stress.mean(axis=1),native_mises=arrays['element_mises'],abaqus_mises=mises)
    for path in work.iterdir():
        if path.suffix in ['.inp','.odb','.sta','.dat','.msg','.json','.txt','.npz']:
            shutil.copy2(path,directory/path.name)
    print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
