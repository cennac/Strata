#!/usr/bin/env python3
"""Repeat the published synthetic long-context workload over an OpenAI SSE API."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent

def request(endpoint, model, client, tokens, barrier):
    item = json.loads(gzip.decompress((ROOT / f'fixtures/ctx131072-route{client}.json.gz').read_bytes()))
    assert hashlib.sha256(item['messages'][0]['content'].encode()).hexdigest() == item['sha256']
    body = {'model':model,'messages':item['messages'],'max_tokens':tokens,
        'temperature':0,'seed':42,'stream':True,'stream_options':{'include_usage':True},
        'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False},
        'repetition_penalty':1,'cache_prompt':True}
    req = urllib.request.Request(endpoint.rstrip('/')+'/v1/chat/completions',
        data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    barrier.wait(30)
    start = time.monotonic()
    first = None
    chunks = []
    usage = {}
    timings = {}
    with urllib.request.urlopen(req,timeout=900) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            packet = json.loads(line[6:])
            if packet.get('error'):
                raise RuntimeError(packet['error'])
            for choice in packet.get('choices',[]):
                content = choice.get('delta',{}).get('content') or ''
                if content:
                    first = first or time.monotonic()
                    chunks.append(content)
            usage = packet.get('usage') or usage
            timings = packet.get('timings') or timings
    elapsed = time.monotonic()-start
    text = ''.join(chunks)
    other = json.loads(gzip.decompress((ROOT / f'fixtures/ctx131072-route{1-client}.json.gz').read_bytes()))
    return {'client':client,'input_sha256':item['sha256'],'wall_s':elapsed,
        'ttft_s':first-start if first else None,'usage':usage,'timings':timings,
        'marker_pass':all(v in text for v in item['expected']) and not any(v in text for v in other['expected']),
        'e2e_tps':usage.get('completion_tokens',0)/elapsed}

def burst(args, clients):
    barrier = threading.Barrier(clients)
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(clients) as pool:
        rows = list(pool.map(lambda i:request(args.endpoint,args.model,i,args.tokens,barrier),range(clients)))
    elapsed = time.monotonic()-start
    tokens = sum(r['usage'].get('completion_tokens',0) for r in rows)
    result = {'clients':clients,'wall_s':elapsed,'completion_tokens':tokens,
        'aggregate_e2e_tps':tokens/elapsed,'requests':rows}
    if not all(r['marker_pass'] and r['usage'].get('completion_tokens',0)>0 for r in rows):
        raise RuntimeError('Marker/output check failed: '+json.dumps(result))
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',default='http://127.0.0.1:8100')
    p.add_argument('--model',default='Qwen3.8-Flash-Next-IQ3_XXS')
    p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--singles',type=int,default=3)
    p.add_argument('--dual-seconds',type=float,default=180)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing result')
    result = {'workload':'synthetic near-128K, cached after cold warmup','cold':None,'single':[],'dual':[]}
    def save():
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    result['cold'] = burst(args,2)
    save()
    for _ in range(args.singles):
        result['single'].append(burst(args,1))
        save()
    start = time.monotonic()
    while time.monotonic()-start < args.dual_seconds:
        result['dual'].append(burst(args,2))
        save()
    elapsed = time.monotonic()-start
    result['summary'] = {'solo_mean_e2e_tps':statistics.mean(r['aggregate_e2e_tps'] for r in result['single']),
        'dual_aggregate_e2e_tps':sum(r['completion_tokens'] for r in result['dual'])/elapsed,
        'dual_wall_s':elapsed,'dual_batches':len(result['dual'])}
    save()
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
