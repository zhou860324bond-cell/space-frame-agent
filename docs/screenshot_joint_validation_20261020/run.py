"""从历史胶囊恢复截图模型，实测节点实体后端；不修改生产代码。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import traceback

ROOT = Path(os.environ.get('SPACE_FRAME_REPO', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / 'src'))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from agent import Session
from native_joint import run_native_joint_analysis
from solid_joint import run_joint_analysis, prepare_joint_spec, SolidJointError

OUT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    rows = []
    cases = [('frame', '20261016_223639_eb62572f', 5, None),
             ('two_arm', '20261016_225826_060a24b7', 2, 'LC1'),
             ('three_arm', '20261016_230634_6cfadf4e', 2, 'LC1')]
    for name, capsule, node, case in cases:
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        source = ROOT / 'capsules' / (capsule + '.json')
        record = json.loads(source.read_text(encoding='utf-8'))
        model = record['input_model']
        save(directory / 'model.json', model)
        save(directory / 'provenance.json', {'capsule': str(source),
             'timestamp': record['timestamp'], 'input_hash': record['input_hash'],
             'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        if name == 'frame':
            rows.append({'case': name, 'status': 'unloaded',
                         'note': '历史模型没有荷载；不新增假定荷载，不以零场证明精度。'})
            save(OUT / 'status.json', rows)
            continue
        session = Session(model=model)
        solved = session.solve_model()
        save(directory / 'global_solve.json', solved.payload)
        if not solved.ok:
            raise RuntimeError(str(solved.payload))
        spec = prepare_joint_spec(session, node, case, loading_method='displacement',
                                  geometry_mode='transition')
        save(directory / 'boundary_spec.json', spec.to_dict())
        if name == 'two_arm':
            target = directory / 'native' / 'node_2_LC1' / 'summary.json'
            if not target.exists():
                print('NATIVE', name, flush=True)
                run_native_joint_analysis(session, node, case, mesh_sizes_mm=[20.],
                    output_dir=directory / 'native', progress=lambda msg: print(msg, flush=True))
        else:
            target = directory / 'native' / 'node_2_LC1'
            if not target.exists():
                shutil.copytree(ROOT / 'results/native_solid_joint/node_2_LC1', target)
        print('ABAQUS NODE WORKFLOW', name, flush=True)
        try:
            target = directory / 'abaqus' / 'node_2_LC1' / 'summary.json'
            if not target.exists():
                run_joint_analysis(session, node, case, mesh_sizes_mm=[30., 20., 10.],
                    output_dir=directory / 'abaqus', timeout=1800.,
                    loading_method='displacement', geometry_mode='transition')
            rows.append({'case': name, 'status': 'completed'})
        except SolidJointError:
            error = traceback.format_exc()
            print(error, flush=True)
            save(directory / 'failure.json', {'traceback': error})
            rows.append({'case': name, 'status': 'failed', 'error': error})
        save(OUT / 'status.json', rows)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
