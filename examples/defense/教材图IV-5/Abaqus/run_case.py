# -*- coding: utf-8 -*-
"""由 Abaqus Python 2.7 驱动完整算例；每一步检查返回码和实际完成状态。"""
from __future__ import division, print_function

import json
import os
import subprocess
import sys


def run(executable, args, label):
    print('Running: '+label)
    sys.stdout.flush()
    process = subprocess.Popen([executable]+args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    console = process.communicate()[0]
    with open(label+'_console.txt', 'wb') as handle:
        handle.write(console)
    if process.returncode:
        raise RuntimeError(label+' failed; inspect '+label+'_console.txt')
    if args[0].startswith('job='):
        name = args[0][4:]
        with open(name+'.sta', 'rb') as handle:
            status = handle.read().decode('ascii', 'replace')
        if 'COMPLETED SUCCESSFULLY' not in status:
            raise RuntimeError(name+' did not complete; inspect .sta/.msg/.dat')
        for suffix in ('.msg', '.dat'):
            with open(name+suffix, 'rb') as handle:
                log = handle.read().decode('ascii', 'replace').upper()
            if any(x in log for x in ('***ERROR', 'NUMERICAL SINGULARITY', 'ZERO PIVOT')):
                raise RuntimeError(name+' contains a solver error; inspect '+suffix)
    print('Completed: '+label)


def main():
    executable = sys.argv[1] if len(sys.argv) > 1 else 'abaqus'
    env_cpu = os.environ.get('PROCESSOR_IDENTIFIER', '').upper()
    if ('AMD' in env_cpu or 'AUTHENTICAMD' in env_cpu) and 'MKL_DEBUG_CPU_TYPE' not in os.environ:
        os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
        with open('abaqus_v6.env', 'w') as handle:
            handle.write("import os\nos.environ.setdefault('MKL_DEBUG_CPU_TYPE', '5')\n")
    # Abaqus 启动后再清理外部虚拟环境变量，避免它的子进程误加载项目 Python。
    os.environ.pop('PYTHONHOME', None)
    os.environ.pop('PYTHONPATH', None)
    run(executable, ['python', 'prepare_case.py'], 'prepare')
    run(executable, ['job=global_b33', 'input=global_b33.inp', 'cpus=1', 'interactive', 'ask_delete=OFF'],
        'global_solve')
    run(executable, ['python', 'extract_global.py'], 'global_extract')
    run(executable, ['cae', 'noGUI=build_local.py'], 'local_build')
    with open('local_mesh_meta.json') as handle:
        meshes = json.load(handle)
    for mesh in meshes:
        run(executable, ['job='+mesh['job'], 'input='+mesh['job']+'.inp', 'cpus=1',
                         'memory=4096mb', 'interactive', 'ask_delete=OFF'], mesh['job']+'_solve')
    run(executable, ['cae', 'noGUI=post_results.py'], 'post_results')
    print('SUCCESS: open IV5_global_and_node3.cae and local_h10.odb.')


if __name__ == '__main__':
    main()
