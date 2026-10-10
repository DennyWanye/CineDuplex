#!/usr/bin/env python3
"""Loopback-only browser candidate; preserves the existing LAN service."""
import json
from pathlib import Path
import shlex
import sys
import time
from cineduplex.cluster.guard import guard
from cineduplex.train.nas_transport import NAS

IMAGE='sha256:5ac677d3dba4ac066166919ef9f6fddbc2497f7f9a8eb077f5597dbb18998908'
PREFIX='output/realtime-tp4-20261010-v9'
NAME='cineduplex-realtime-tp4-shadow-v9-web15'


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());nas=NAS(cfg['nas'])
    guard(cfg['cluster'],heartbeat=True)
    status=json.loads(nas.run('curl -fsS --max-time 5 http://127.0.0.1:18092/status',timeout=10))
    if not status.get('ready') or status.get('active') or status.get('tensor_parallel_size')!=4:
        raise RuntimeError('wait for the original four-rank candidate to be idle/ready')
    old=json.loads(nas.run('docker inspect cineduplex-realtime-tp4-shadow-v9-web14',timeout=15))[0]
    if '/service/web-shadow-v14/realtime_server.py' not in old['Config']['Cmd']:
        raise ValueError('original shadow identity changed')
    nas.run('docker stop cineduplex-realtime-tp4-shadow-v9-web14',timeout=40)
    nas.run('if docker inspect '+NAME+' >/dev/null 2>&1; then exit 1; fi; '
        'test -z "$(ss -H -lnt \'sport = :18093 or sport = :18094\')"',timeout=15)
    for name in ('realtime_server.py','realtime.html','realtime.js','capture.worklet.js'):
        source=(Path('src/cineduplex/web')/name).read_bytes()
        if name=='realtime.html':
            source=source.replace('CineDuplex · 全双工通话'.encode(),'CineDuplex · 四机候选测试'.encode())
        nas.put(PREFIX+'/web-shadow-v15/'+name,source)
    root=nas.path(PREFIX)
    launch=['docker','run','-d','--name',NAME,'--restart','no','--read-only','--log-driver','none',
        '--network','host','--user','0:0','--cap-drop','ALL','--cap-add','SETUID','--cap-add','SETGID',
        '--security-opt','no-new-privileges','--pids-limit','96','--memory','256m','--no-healthcheck',
        '--mount','type=bind,src='+root+',dst=/service',
        '--mount','type=bind,src=/srv/nas/.nas-online,dst=/srv/nas/.nas-online,readonly']
    for source,dest in (('certs/current/host.crt','host.crt'),('certs/current/host.key','host.key'),
            ('public/agentos-household-root.crt','root.crt')):
        launch+=['--mount','type=bind,src=/var/lib/agentos/'+source+',dst=/run/cineduplex/'+dest+',readonly']
    launch+=['-e','PYTHONDONTWRITEBYTECODE=1','-e','TMPDIR=/service/tmp','--entrypoint','/usr/local/bin/python',
        IMAGE,'/service/web-shadow-v15/realtime_server.py','--root','/service/web-shadow-v15',
        '--cert','/run/cineduplex/host.crt','--key','/run/cineduplex/host.key','--host','127.0.0.1',
        '--port','18093','--plain-port','18094','--upstream','http://127.0.0.1:18092']
    container=nas.run(shlex.join(launch),timeout=40).decode().strip()
    served=nas.run('curl -fsS --max-time 5 http://127.0.0.1:18094/realtime.js',timeout=10)
    if served!=(Path('src/cineduplex/web/realtime.js').read_bytes()):raise RuntimeError('served JavaScript differs from candidate')
    print(nas.put(PREFIX+'/SHADOW-WEB-v15.json',json.dumps({'at':time.time(),'container':container,
        'name':NAME,'image':IMAGE,'bind':'loopback-only','upstream':'127.0.0.1:18092',
        'ports':[18093,18094],'gpu_status':status,'formal_lan_service_preserved':True}).encode()))


if __name__=='__main__':main()
