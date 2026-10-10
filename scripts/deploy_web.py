#!/usr/bin/env python3
"""Deploy the independent LAN app from an immutable NAS source revision."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from cineduplex.train.nas_transport import NAS

IMAGE='sha256:5ac677d3dba4ac066166919ef9f6fddbc2497f7f9a8eb077f5597dbb18998908'


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());version=sys.argv[2]
    if not version.startswith('v') or not version[1:].isdigit():raise ValueError('version must be vN')
    nas=NAS(cfg['nas']);root=cfg['nas']['root'];run=cfg['run_prefix']
    for name in ('server.py','index.html'):
        if not nas.stat(run+'/web/source-'+version+'/'+name):raise ValueError('source revision missing')
    def call(args,check=True):
        r=subprocess.run(nas.ssh+[shlex.join(args)],capture_output=True,text=True,timeout=40)
        if check and r.returncode:raise RuntimeError(r.stderr[-400:])
        return r
    old=json.loads(call(['docker','inspect','cineduplex-web']).stdout)[0]
    if old['Image']!=IMAGE or root not in [m['Source'] for m in old['Mounts']]:raise ValueError('wrong service identity')
    previous=next(s for s in old['Config']['Cmd'] if '/source-v' in s).split('/source-')[1].split('/')[0]
    if previous==version:raise ValueError('revision already deployed; inspect existing service')
    backup='cineduplex-web-'+previous
    if not call(['docker','inspect',backup],check=False).returncode:raise ValueError('rollback container already exists')
    call(['docker','stop','cineduplex-web']);call(['docker','rename','cineduplex-web',backup])
    launch=['docker','run','-d','--name','cineduplex-web','--restart','unless-stopped','--read-only',
        '--log-driver','none','--user','990:990','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--pids-limit','64','--memory','256m','-p','192.168.12.10:18086:18086',
        '--mount','type=bind,src='+root+',dst='+root,
        '--mount','type=bind,src=/srv/nas/.nas-online,dst=/srv/nas/.nas-online,readonly',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','CINEDUPLEX_NAS_UUID='+cfg['nas']['uuid'],
        '-e','CINEDUPLEX_CONTAINER_NAS=1','-e','TMPDIR='+root+'/cache/web',
        '--entrypoint','/bin/sh',IMAGE,'-c','exec /usr/local/bin/python '+
        shlex.quote(root+'/'+run+'/web/source-'+version+'/server.py')+' --root '+shlex.quote(root)+
        ' --run '+shlex.quote(run)+' --host 0.0.0.0 >> '+shlex.quote(root+'/'+run+'/web/service-'+version+'.log')+' 2>&1']
    r=call(launch,check=False)
    if r.returncode:
        created=call(['docker','inspect','cineduplex-web'],check=False)
        if not created.returncode:call(['docker','rename','cineduplex-web','cineduplex-web-'+version+'-failed'])
        call(['docker','rename',backup,'cineduplex-web']);call(['docker','start','cineduplex-web'])
        raise RuntimeError('launch failed; prior service restored: '+r.stderr[-300:])
    receipt={'at':time.time(),'version':version,'container_id':r.stdout.strip(),'image':IMAGE,
        'url':'http://192.168.12.10:18086/','restart_policy':'unless-stopped',
        'runtime_and_logs':'NAS','rollback_container':backup,'reboot_tested':False}
    print(nas.put(run+'/web/deployment-'+version+'.json',json.dumps(receipt,indent=2).encode()))


if __name__=='__main__':main()
