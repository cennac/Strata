#!/usr/bin/env python3
"""Build the complete gfx1100 engine from source, without old objects/archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def dependency(name, spec, directory):
    dest = directory / name
    marker = dest / '.community-source-sha256'
    if marker.exists():
        if marker.read_text().strip() != spec['archive_sha256']:
            raise RuntimeError('Dependency checksum marker differs: ' + name)
        return dest
    if dest.exists():
        raise RuntimeError('Existing dependency has no verified marker: ' + str(dest))
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / (name + '.tar.gz')
    if not archive.exists():
        print('Downloading', spec['url'], flush=True)
        with urllib.request.urlopen(spec['url'], timeout=120) as response, archive.open('wb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != spec['archive_sha256']:
        raise RuntimeError('Dependency archive checksum mismatch: ' + name)
    dest.mkdir()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            parts = member.name.split('/', 1)
            if len(parts) < 2 or not parts[1]:
                continue
            member.name = parts[1]
            tf.extract(member, dest, filter='data')
    marker.write_text(spec['archive_sha256'] + '\n')
    return dest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rocm', type=Path, default=Path(os.environ.get('ROCM_PATH', '/opt/rocm')))
    p.add_argument('--build-dir', type=Path, default=ROOT / 'build-community')
    p.add_argument('--deps-dir', type=Path, default=ROOT / '.community-deps')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--configure-only', action='store_true')
    args = p.parse_args()
    rocm = args.rocm.resolve()
    compiler = rocm / 'llvm/bin/clang++'
    if not compiler.is_file():
        raise SystemExit('ROCm compiler not found. Supply --rocm /path/to/rocm')
    lock = json.loads((ROOT / 'community/dependencies.lock.json').read_text())
    deps = {name:dependency(name,spec,args.deps_dir.resolve()) for name,spec in lock.items()}
    bitcodes = [rocm / 'lib/llvm/amdgcn/bitcode', rocm / 'amdgcn/bitcode', rocm / 'lib/amdgcn/bitcode']
    bitcode = next((p for p in bitcodes if p.is_dir()), None)
    if bitcode is None:
        raise SystemExit('ROCm device bitcode directory not found')
    env = dict(os.environ, HIP_PLATFORM='amd', HIP_COMPILER='clang', HIP_RUNTIME='rocclr',
               ROCM_PATH=str(rocm), HIP_PATH=str(rocm))
    env['PATH'] = str(rocm / 'bin') + ':' + str(rocm / 'llvm/bin') + ':' + env['PATH']
    env['CMAKE_PREFIX_PATH'] = str(rocm) + (';' + env['CMAKE_PREFIX_PATH'] if env.get('CMAKE_PREFIX_PATH') else '')
    args.build_dir.mkdir(parents=True, exist_ok=True)
    command = ['cmake','-S',str(ROOT),'-B',str(args.build_dir),'-G','Unix Makefiles',
        '-DCMAKE_BUILD_TYPE=Release','-DSTRATA_ENABLE_HIP=ON','-DSTRATA_ENABLE_CUDA=OFF',
        '-DSTRATA_BUILD_TESTS=OFF','-DSTRATA_PREFILL_MMQ=ON','-DSTRATA_MMQ_KQUANTS=ON',
        '-DSTRATA_GGML_DIR='+str(deps['llama.cpp']),'-DCMAKE_HIP_ARCHITECTURES=gfx1100',
        '-DCMAKE_HIP_COMPILER='+str(compiler),'-DCMAKE_HIP_COMPILER_ROCM_ROOT='+str(rocm),
        '-DCMAKE_HIP_FLAGS=--rocm-path='+str(rocm)+' --rocm-device-lib-path='+str(bitcode),
        '-DSTRATA_ROCWMMA_INCLUDE_DIR='+str(deps['rocWMMA']/'library/include')]
    print('Configuring full source build', flush=True)
    subprocess.run(command, env=env, check=True)
    if not args.configure_only:
        subprocess.run(['cmake','--build',str(args.build_dir),'--target','strata','-j',str(args.jobs)],env=env,check=True)
        binary = args.build_dir / 'strata'
        record = {'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                  'build':'full source, no reused object/static archive',
                  'dependencies':lock,'configure_command':command}
        (args.build_dir/'community-build.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Built:',binary,record['binary_sha256'],flush=True)

if __name__ == '__main__':
    main()
