#!/usr/bin/env python3
"""Promote the verified candidate web; preserve the original web and GPU."""
import json
import hashlib
from pathlib import Path
import shlex
import sys
import time
from cineduplex.cluster.guard import guard
from cineduplex.train.nas_transport import NAS

IMAGE='sha256:5ac677d3dba4ac066166919ef9f6fddbc2497f7f9a8eb077f5597dbb18998908'
PREFIX='output/realtime-tp4-20261010-v9'
NAME='cineduplex-realtime-web'
BACKUP=NAME+'-v27-before-latency'


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());n=NAS(cfg['nas'])
    guard(cfg['cluster'],heartbeat=True)
    def command(args,timeout=40):return n.run(shlex.join(args),timeout=timeout)
    def status(port):return json.loads(command(['curl','-fsS','--max-time','5',f'http://127.0.0.1:{port}/status']))
    candidate=status(18092);original=status(18087)
    if not candidate.get('ready') or candidate.get('active') or original.get('active'):
        raise ValueError('promotion requires both existing runtimes idle and candidate ready')
    if candidate.get('tensor_parallel_size')!=4 or candidate.get('decode_precision')!='bf16':
        raise ValueError('candidate runtime identity')
    evidence=json.loads(command(['cat',n.path(PREFIX+'/PROMOTION-LATENCY-v43.json')]))
    if not evidence.get('real_gpu_pass') or not evidence.get('real_browser_pass') or evidence.get('rebuffer_count')!=0 or not evidence.get('latency_improvement_pass'):
        raise ValueError('saved real GPU and browser acceptance required')
    if evidence.get('runtime_source_sha256')!=(candidate.get('tp_runtime_sha256') or candidate.get('source_sha256')):
        raise ValueError('acceptance belongs to a different candidate source')
    if evidence.get('kernel_sha256')!='40b892083c0ce8b8cbf5f005002ab3e4c90314a8d348b7a79ab4657c93cb931b':
        raise ValueError('acceptance must use the admitted production BF16 kernel')
    actual_web={file:hashlib.sha256((Path('src/cineduplex/web')/file).read_bytes()).hexdigest() for file in ('realtime_server.py','realtime.html','realtime.js','capture.worklet.js')}
    if evidence.get('web_source_sha256')!=actual_web:raise ValueError('accepted web source changed')
    old=json.loads(command(['docker','inspect',NAME]))[0]
    if old['Image']!=IMAGE or n.path(PREFIX) not in [m['Source'] for m in old['Mounts']]:
        raise ValueError('original formal web identity')
    if '/service/web-source-v27/realtime_server.py' not in old['Config']['Cmd']:
        raise ValueError('original formal web changed; reconcile before promotion')
    n.run('if docker inspect '+BACKUP+' >/dev/null 2>&1; then exit 1; fi',timeout=10)
    for file in ('realtime_server.py','realtime.html','realtime.js','capture.worklet.js'):
        n.put(PREFIX+'/web-source-v43/'+file,(Path('src/cineduplex/web')/file).read_bytes())
    root=n.path(PREFIX)
    launch=['docker','run','-d','--name',NAME,'--restart','unless-stopped','--read-only','--log-driver','none',
        '--network','host','--user','0:0','--cap-drop','ALL','--cap-add','SETUID','--cap-add','SETGID',
        '--security-opt','no-new-privileges','--pids-limit','96','--memory','256m','--no-healthcheck',
        '--mount','type=bind,src='+root+',dst=/service',
        '--mount','type=bind,src=/srv/nas/.nas-online,dst=/srv/nas/.nas-online,readonly']
    for source,dest in (('certs/current/host.crt','host.crt'),('certs/current/host.key','host.key'),
            ('public/agentos-household-root.crt','root.crt')):
        launch+=['--mount','type=bind,src=/var/lib/agentos/'+source+',dst=/run/cineduplex/'+dest+',readonly']
    launch+=['-e','PYTHONDONTWRITEBYTECODE=1','-e','TMPDIR=/service/tmp','--entrypoint','/usr/local/bin/python',
        IMAGE,'/service/web-source-v43/realtime_server.py','--root','/service/web-source-v43',
        '--cert','/run/cineduplex/host.crt','--key','/run/cineduplex/host.key','--upstream','http://127.0.0.1:18092']
    command(['docker','stop',NAME]);command(['docker','rename',NAME,BACKUP])
    try:
        container=command(launch).decode().strip()
        verified=None
        for _ in range(20):
            try:
                verified=json.loads(command(['curl','-fsS','--max-time','3','http://127.0.0.1:18089/api/status']))
                if verified.get('ready') and verified.get('tp_runtime_sha256')==candidate.get('tp_runtime_sha256'):break
            except RuntimeError:pass
            time.sleep(.5)
        else:raise RuntimeError('promoted web did not report the candidate')
        served=command(['curl','-fsS','--max-time','5','http://127.0.0.1:18089/realtime.js'])
        if hashlib.sha256(served).hexdigest()!=actual_web['realtime.js']:raise RuntimeError('formal web serves different JavaScript')
    except Exception:
        n.run('if docker inspect '+NAME+' >/dev/null 2>&1; then docker stop '+NAME+'; docker rename '+NAME+' '+NAME+'-v43-failed; fi',timeout=40)
        command(['docker','rename',BACKUP,NAME]);command(['docker','start',NAME])
        raise
    print(n.put(PREFIX+'/deployment-v43.json',json.dumps({'at':time.time(),'container':container,
        'rollback':BACKUP,'backend_status':verified,'original_gpu_preserved':True,
        'url':'https://192.168.12.10:18088/','acceptance':evidence}).encode()))


if __name__=='__main__':main()
