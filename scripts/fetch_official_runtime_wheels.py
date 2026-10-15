#!/usr/bin/env python3
"""Fetch selected official-lock ARM64 wheels into NAS with PyPI SHA checks.

Download bytes remain in RAM until the immutable NAS writer verifies them.
This changes neither network configuration nor the existing serving environment.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from cineduplex.train.nas_transport import NAS

ALLOWED={'sentencepiece','matplotlib','contourpy','cycler','fonttools',
         'kiwisolver','pyparsing','openai','nvidia-ml-py','pyzmq','blake3'}


def curl(url,limit):
    result=subprocess.run(['curl','--http1.1','-fsS','--max-time','120','--max-filesize',str(limit),url],
                          capture_output=True,timeout=125)
    if result.returncode:
        raise RuntimeError('public package transfer failed: '+result.stderr.decode()[-180:])
    if len(result.stdout)>limit:
        raise RuntimeError('package byte budget exceeded')
    return result.stdout


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--record',required=True)
    parser.add_argument('--package', action='append', choices=sorted(ALLOWED),
                        help='Fetch only a diagnosed missing dependency from the official lock')
    args=parser.parse_args()
    n=NAS(json.loads(Path(args.config).read_text())['nas'])
    prefix='output/official-backend-20261013-v1/'
    lock=n.read(prefix+'source/requirements.txt',timeout=15).decode()
    versions=dict(re.findall(r'^([a-zA-Z0-9_-]+)==([^\s]+)$',lock,re.M))
    def fetch(name):
        version=versions[name]
        info=json.loads(curl(f'https://pypi.org/pypi/{name}/{version}/json',2*1024*1024))
        files=[f for f in info['urls'] if f['filename'].endswith('.whl') and
               (((('cp312-cp312' in f['filename']) or
                  re.search(r'cp3(?:[2-9]|1[0-2])-abi3-', f['filename'])) and 'aarch64' in f['filename'])
                or 'py3-none-any' in f['filename'] or 'py2.py3-none-any' in f['filename'])]
        if not files:
            raise RuntimeError('no compatible official-lock wheel: '+name)
        entry=sorted(files,key=lambda f:f['filename'])[0]
        path=prefix+'wheels/'+entry['filename'];digest=entry['digests']['sha256']
        if n.stat(path) is not None:
            existing=n.run('sha256sum -- '+shlex.quote(n.path(path)),timeout=20).decode().split()[0]
            if existing!=digest:
                raise RuntimeError('existing wheel differs from official index: '+name)
            pin={'path':path,'sha256':digest,'bytes':entry['size'],'reused':True}
        else:
            payload=curl(entry['url'],80*1024*1024)
            if len(payload)!=entry['size'] or hashlib.sha256(payload).hexdigest()!=digest:
                raise RuntimeError('wheel differs from official index: '+name)
            pin=n.put(path,payload)
        return {'package':name,'version':version,**pin}
    # Network reads are independent; each immutable NAS destination is unique.
    pins=[]; errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(fetch,name):name for name in sorted(set(args.package or ALLOWED))}
        for future in concurrent.futures.as_completed(pending):
            try:pins.append(future.result())
            except Exception as exc:errors.append({'package':pending[future],'error':str(exc)})
    print(n.put(prefix+args.record,json.dumps({'pins':pins,'errors':errors},indent=2).encode()))
    print(json.dumps(pins))
    if errors:raise RuntimeError(json.dumps(errors))


if __name__=='__main__':
    main()
