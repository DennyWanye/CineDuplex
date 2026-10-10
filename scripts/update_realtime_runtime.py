#!/usr/bin/env python3
"""Pinned idle-only hot update under the original Gateway admission."""
import argparse,hashlib,json,re,shlex,time
from pathlib import Path
from cineduplex.cluster.guard import guard
from cineduplex.train.nas_transport import NAS

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--source',required=True);p.add_argument('--sha',required=True);p.add_argument('--expected',required=True);p.add_argument('--record',required=True);p.add_argument('--prefix',default='output/realtime-tp4-20261010-v9');a=p.parse_args()
    if not re.fullmatch(r'tp-runtime-v\d+\.py',a.source) or not re.fullmatch(r'[A-Za-z0-9-]+',a.record):raise ValueError('source/record name')
    if any(not re.fullmatch(r'[0-9a-f]{64}',x) for x in (a.sha,a.expected)):raise ValueError('source SHA')
    cfg=json.loads(Path(a.config).read_text());n=NAS(cfg['nas']);prefix=a.prefix.rstrip('/')+'/'
    if not re.fullmatch(r'output/realtime-tp4-[A-Za-z0-9-]+/',prefix):raise ValueError('NAS prefix')
    guard(cfg['cluster'],heartbeat=True)
    def api(route,data=None):
        cmd=['curl','-fsS','--max-time','25']
        if data is not None:cmd+=['-X','POST','-H','Content-Type: application/json','--data-binary','@-']
        return json.loads(n.run(shlex.join(cmd+['http://127.0.0.1:18092'+route]),data=data,timeout=30))
    before=api('/status')
    if not before.get('ready') or before.get('active') or before.get('tp_runtime_sha256')!=a.expected:raise RuntimeError('original service is not on the expected ready/idle source')
    source=n.read(prefix+a.source,timeout=15)
    if hashlib.sha256(source).hexdigest()!=a.sha:raise ValueError('NAS source SHA differs')
    spec={'name':a.source,'sha256':a.sha};n.put(prefix+a.record+'-REQUEST.json',json.dumps({'at':time.time(),'before':before,'spec':spec}).encode())
    result=api('/tp_reload',json.dumps(spec).encode());after=api('/status')
    n.put(prefix+a.record+'.json',json.dumps({'at':time.time(),'result':result,'after':after,'weights_reloaded':False,'runtime_call_test_pending':True}).encode())
    if not after.get('ready') or after.get('active') or after.get('tp_runtime_sha256')!=a.sha:raise RuntimeError('hot update did not leave expected source ready/idle')
    print(json.dumps({'updated':result['updated'],'ranks':result['ranks'],'ready':after['ready']}))

if __name__=='__main__':main()
