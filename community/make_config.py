#!/usr/bin/env python3
"""Write the measured two-request configuration using this PC's model paths."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',required=True,type=Path)
    p.add_argument('--ple-gguf',required=True,type=Path)
    p.add_argument('--pack',required=True,type=Path)
    p.add_argument('--mtp',required=True,type=Path)
    p.add_argument('--rocm',default='/opt/rocm',type=Path)
    p.add_argument('--exe',default=ROOT/'build-community/strata',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--port',default=8100,type=int)
    p.add_argument('--gpus',default='0,1',help='Two HIP device indices, in layer order')
    p.add_argument('--vision-exe',type=Path)
    p.add_argument('--mmproj',type=Path)
    p.add_argument('--vision-gpu',default=0,type=int)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite an existing configuration')
    required = [a.gguf,a.ple_gguf,a.exe,a.pack/'experts.bin',a.pack/'dense.bin',
        a.pack/'native_experts.txt',a.pack/'tokenizer/chat_template.jinja',
        a.mtp/'experts.bin',a.mtp/'dense.bin',a.mtp/'dense.txt',a.mtp/'draft_vocab.bin']
    if bool(a.vision_exe) != bool(a.mmproj):
        raise SystemExit('Supply --vision-exe and --mmproj together')
    if a.vision_exe:
        required += [a.vision_exe,a.mmproj]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit('Required files missing: '+', '.join(missing))
    gpus = [int(n) for n in a.gpus.split(',')]
    if len(gpus)!=2 or len(set(gpus))!=2:
        raise SystemExit('--gpus needs two distinct indices')
    template = json.loads((ROOT/'community/config.template.json').read_text())
    run = a.output.resolve().parent/'community-run'
    run.mkdir(parents=True,exist_ok=True)
    profile = run/'expert-profile.bin'
    if not profile.exists():
        shutil.copy2(ROOT/'community/assets/expert-profile-dual128.bin',profile)
    replacements = {'PACK':str(a.pack.resolve()),'GGUF':str(a.gguf.resolve()),
        'PLE_GGUF':str(a.ple_gguf.resolve()),'MTP':str(a.mtp.resolve()),'PROFILE':str(profile)}
    template['args'] = [replacements.get(v,v) for v in template['args']]
    template.update(exe=str(a.exe.resolve()),cwd=str(ROOT),tokenizer=str(a.pack.resolve()/'tokenizer'),
        log=str(run/'engine.log'),port=a.port,gpu=gpus,lib_dirs=[str(a.rocm.resolve()/'lib')])
    template['env']['STRATA_HIPBLASLT_TUNING']=str(ROOT/'community/assets/gfx1100-hipblaslt-100401.txt')
    if a.vision_exe:
        template['vision']={'exe':str(a.vision_exe.resolve()),'mmproj':str(a.mmproj.resolve()),
            'model':str(a.gguf.resolve()),'gpu':True,'threads':8,'max_tokens':1024,'cuda_device':a.vision_gpu}
    a.output.write_text(json.dumps(template,ensure_ascii=False,indent=2)+'\n')
    print('Wrote',a.output.resolve())

if __name__=='__main__':
    main()
