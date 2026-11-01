#!/usr/bin/env python3
"""Check scheduler/row bounds; optionally exercise direct copies on two GPUs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=ROOT/'build-community')
    p.add_argument('--rocm',type=Path,default=Path(os.environ.get('ROCM_PATH','/opt/rocm')))
    p.add_argument('--gpus',help='Also run byte-level GPU copy regressions, e.g. 0,1')
    a=p.parse_args()
    b=a.build_dir.resolve()
    tests=b/'community-tests'
    tests.mkdir(exist_ok=True)
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=str(a.rocm.resolve()/'lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    log=[]
    def run(cmd):
        r=subprocess.run(cmd,cwd=b,env=env,capture_output=True,text=True)
        log.append({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        (tests/'results.json').write_text(json.dumps(log,indent=2)+'\n')
        print(r.stdout,r.stderr,flush=True)
        r.check_returncode()
    for name,source in [('batch_groups_test',ROOT/'src/program/batch_groups_test.cpp'),
                        ('row_bounds_test',ROOT/'community/tests/row_bounds_test.cpp')]:
        exe=tests/name
        run(['c++','-std=c++20','-O2','-I'+str(ROOT/'include'),str(source),'-o',str(exe)])
        run([str(exe)])
    if a.gpus:
        flags={}
        for line in (b/'CMakeFiles/strata_engine.dir/flags.make').read_text().splitlines():
            if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
                key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
        cppflags=[v for k in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'] for v in flags[k] if v!='-DNDEBUG']
        link=shlex.split((b/'CMakeFiles/strata.dir/link.txt').read_text())
        libraries=link[link.index('-o')+2:]
        exe=tests/'direct_snapshot_test'
        run(['c++',*cppflags,'-I'+str(ROOT/'src/core'),str(ROOT/'community/tests/direct_snapshot_test.cpp'),
             '-o',str(exe),*libraries,'-pthread'])
        for device in a.gpus.split(','):
            env.update(HIP_VISIBLE_DEVICES=device,GPU_MAX_HW_QUEUES='4',STRATA_CONVERSATION_PINNED_COPY='1')
            run([str(exe)])
    print('All requested checks passed')

if __name__=='__main__':
    main()
