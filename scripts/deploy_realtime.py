#!/usr/bin/env python3
"""Deploy a saved realtime web revision, preserving the previous container."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from cineduplex.train.nas_transport import NAS

IMAGE='sha256:5ac677d3dba4ac066166919ef9f6fddbc2497f7f9a8eb077f5597dbb18998908'
PREFIX='output/realtime-20261009-v1'
NAME='cineduplex-realtime-web'

def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());version=sys.argv[2]
    if not version.startswith('v') or not version[1:].isdigit():raise ValueError('use vN')
    nas=NAS(cfg['nas']);root=nas.path(PREFIX)
    for name in ['realtime_server.py','realtime.html','realtime.js','capture.worklet.js']:
        if not nas.stat(PREFIX+'/web-source-'+version+'/'+name):raise ValueError('source missing')
    def call(args,check=True):
        r=subprocess.run(nas.ssh+[nas.guarded(shlex.join(args))],capture_output=True,text=True,timeout=40)
        if check and r.returncode:raise RuntimeError(r.stderr[-400:])
        return r
    old=json.loads(call(['docker','inspect',NAME]).stdout)[0]
    if old['Image']!=IMAGE or root not in [m['Source'] for m in old['Mounts']]:raise ValueError('service identity')
    previous=next(x for x in old['Config']['Cmd'] if '/web-source-v' in x).split('/web-source-')[1].split('/')[0]
    if previous==version:raise ValueError('already deployed; inspect existing service')
    backup=NAME+'-'+previous
    if not call(['docker','inspect',backup],False).returncode:raise ValueError('backup exists')
    call(['docker','stop',NAME]);call(['docker','rename',NAME,backup])
    launch=['docker','run','-d','--name',NAME,'--restart','unless-stopped','--read-only','--log-driver','none',
            '--network','host','--user','0:0','--cap-drop','ALL','--cap-add','SETUID','--cap-add','SETGID',
            '--security-opt','no-new-privileges','--pids-limit','96','--memory','256m','--no-healthcheck',
            '--mount','type=bind,src='+root+',dst=/service',
            '--mount','type=bind,src=/srv/nas/.nas-online,dst=/srv/nas/.nas-online,readonly']
    for source,dest in [('certs/current/host.crt','host.crt'),('certs/current/host.key','host.key'),
                        ('public/agentos-household-root.crt','root.crt')]:
        launch+=['--mount','type=bind,src=/var/lib/agentos/'+source+',dst=/run/cineduplex/'+dest+',readonly']
    launch+=['-e','PYTHONDONTWRITEBYTECODE=1','-e','TMPDIR=/service/tmp','--entrypoint','/usr/local/bin/python',
             IMAGE,'/service/web-source-'+version+'/realtime_server.py','--root','/service/web-source-'+version,
             '--cert','/run/cineduplex/host.crt','--key','/run/cineduplex/host.key']
    r=call(launch,False)
    if r.returncode:
        if not call(['docker','inspect',NAME],False).returncode:call(['docker','rename',NAME,NAME+'-'+version+'-failed'])
        call(['docker','rename',backup,NAME]);call(['docker','start',NAME]);raise RuntimeError('previous service restored')
    print(nas.put(PREFIX+'/deployment-'+version+'.json',json.dumps({'at':time.time(),'version':version,
        'container_id':r.stdout.strip(),'rollback':backup,'url':'https://192.168.12.10:18088/',
        'image':IMAGE,'reboot_tested':False,'runtime_logs':'NAS'}).encode()))

if __name__=='__main__':main()
