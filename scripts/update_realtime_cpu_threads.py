#!/usr/bin/env python3
"""Change one resident TP setting through the existing cooperative admission.

Requires an idle original service. Never starts workers or changes weights.
Evidence is immutable on the NAS; a rejected experiment can restore TP43.
"""
import argparse
import hashlib
import json
import shlex
import time
from pathlib import Path
from cineduplex.cluster.guard import guard
from cineduplex.train.nas_transport import NAS

PREFIX='output/realtime-tp4-20261010-v9'
BASELINE='215e68a6376d2368afcab97480fa84d841c2abdc13b409b0686b7e2a12745d4a'
CANDIDATE='082326fc2ff3464f806d6eaa0be13ab6e1eb87972745a4367013e530c7a0a9f7'

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--restore',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());n=NAS(cfg['nas'])
    guard(cfg['cluster'],heartbeat=True)
    def api(route,data=None):
        args=['curl','-fsS','--max-time','25']
        if data is not None:args+=['-X','POST','-H','Content-Type: application/json','--data-binary','@-']
        return json.loads(n.run(shlex.join(args+['http://127.0.0.1:18092'+route]),data=data,timeout=30))
    state=api('/status');expected=CANDIDATE if a.restore else BASELINE
    if not state.get('ready') or state.get('active') or state.get('tp_runtime_sha256')!=expected:
        raise RuntimeError('original service must be idle, ready, and on the expected source')
    name='tp-runtime-v43.py' if a.restore else 'tp-runtime-v44.py'
    sha=BASELINE if a.restore else CANDIDATE
    source=n.read(PREFIX+'/'+name,timeout=15)
    if hashlib.sha256(source).hexdigest()!=sha:raise ValueError('pinned source differs')
    label='RESTORE-CPU-v44' if a.restore else 'UPDATE-CPU-v44'
    spec={'name':name,'sha256':sha}
    n.put(PREFIX+'/'+label+'-REQUEST.json',json.dumps({'at':time.time(),'before':state,'spec':spec}).encode())
    result=api('/tp_reload',json.dumps(spec).encode())
    after=api('/status')
    n.put(PREFIX+'/'+label+'.json',json.dumps({'at':time.time(),'result':result,'after':after,
        'weights_reloaded':False,'quantization_changed':False,'runtime_call_test_pending':True}).encode())
    if after.get('active') or not after.get('ready') or after.get('tp_runtime_sha256')!=sha:
        raise RuntimeError('source update did not leave the original service ready and idle')
    print(json.dumps({'updated':result.get('updated'),'source':sha,'ranks':result.get('ranks'),'ready':after['ready']}))

if __name__=='__main__':main()
