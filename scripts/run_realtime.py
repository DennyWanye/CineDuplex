#!/usr/bin/env python3
"""One Gateway-owned realtime worker and loopback NAS-to-DGX tunnel."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from cineduplex.cluster.guard import guard
from cineduplex.train.four_controller import modules,remote
from cineduplex.train.nas_mount import DirectMount
from cineduplex.train.nas_transport import NAS

TUNNEL=r'''
import json,subprocess,sys,time
args=json.loads(sys.stdin.readline())
child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
try:
 time.sleep(1)
 if child.poll() is not None:raise RuntimeError(child.stderr.read().decode()[-300:])
 print('TUNNEL_READY',flush=True)
 sys.stdin.read()
finally:
 if child.poll() is None:child.terminate()
 child.wait(timeout=15)
'''

def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());root=Path('/Volumes/media/CineDuplex/output/realtime-20261009-v1')
    if not Path('/Volumes/media').is_mount():raise RuntimeError('real NAS required')
    with (root/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard(cfg['cluster'],heartbeat=True)
        reuse_mount=len(sys.argv)>2 and sys.argv[2]=='--reuse-inference-mount'
        node=next(n for n in cfg['nodes'] if n['rank']==(3 if reuse_mount else 2))
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',node['ssh_alias'],
            'hostname; whoami; nvidia-smi --query-compute-apps=pid --format=csv,noheader'],capture_output=True,text=True,timeout=15)
        allowed=[]
        if reuse_mount:
            resident=json.loads(NAS(cfg['nas']).read(cfg['run_prefix']+'/web/resident/SERVICE.json',timeout=15))
            if not resident.get('model_ready') or time.time()-resident['heartbeat']>30:raise RuntimeError('existing resident mount is not live')
            allowed=[str(resident['pid'])]
        if r.returncode or r.stdout.splitlines()!=[node['hostname'],node['user']]+allowed:raise RuntimeError('selected GPU has unrelated tasks or identity differs')
        if reuse_mount:
            check=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',node['ssh_alias'],
                'findmnt -rn -o FSTYPE -T "$HOME/cineduplex-work/nas-inference"; awk \'/^MemAvailable:/ {print int($2/1024)}\' /proc/meminfo'],capture_output=True,text=True,timeout=15)
            lines=check.stdout.splitlines()
            if check.returncode or lines[0]!='fuse.sshfs' or int(lines[1])<50000:raise RuntimeError('existing task mount or free GPU memory differs')
        run=Path('/Volumes/media/CineDuplex')/cfg['run_prefix']
        completed=json.loads((run/cfg['training_attempt']/'GPU-TRAINING-COMPLETE.json').read_text())
        original=json.loads((run/cfg['training_attempt']/'RUN-IDENTITY.json').read_text())
        packed=modules()
        service={**cfg,'checkpoint':completed['checkpoint'],'stop_step':completed['steps'],
            'batch_prefix':cfg['run_prefix'],'model':original['model'],
            'mount_name':'nas-inference' if reuse_mount else 'nas-realtime','realtime_prefix':'output/realtime-20261009-v1','realtime_port':18087,
            'source_modules_sha256':hashlib.sha256(json.dumps(packed,sort_keys=True).encode()).hexdigest()}
        nas=NAS(cfg['nas'])
        generation=str(time.time_ns())
        nas.put('output/realtime-20261009-v1/source-'+generation+'.json',json.dumps(packed,sort_keys=True).encode())
        (root/'controller.json').write_text(json.dumps({'pid':os.getpid(),'started_at':time.time(),
            'node':node['hostname'],'generation':cfg['cluster']['lease']['generation'],
            'source_sha256':service['source_modules_sha256'],'source_snapshot':'source-'+generation+'.json'}))
        # Authentication stays in the existing SSH agent. No private key is
        # copied to K6 or DGX. The audio tunnel is K6 loopback -> DGX loopback.
        remote_ssh=['ssh','-N','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ExitOnForwardFailure=yes',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes',
            '-o','UserKnownHostsFile='+nas.path('cache/sshfs/dgx-all-known-hosts'),
            '-L','127.0.0.1:18087:127.0.0.1:18087',node['user']+'@'+node['address']]
        cmd=nas.guarded(shlex.join([cfg['nas_python'],'-B','-u','-c',TUNNEL]))
        outer=nas.ssh[:1]+['-A']+nas.ssh[1:]+[cmd]
        tunnel=subprocess.Popen(outer,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        tunnel.stdin.write(json.dumps(remote_ssh).encode()+b'\n');tunnel.stdin.flush()
        mount=None
        try:
            if tunnel.stdout.readline()!=b'TUNNEL_READY\n':raise RuntimeError('realtime tunnel failed: '+tunnel.stderr.read().decode()[-300:])
            if not reuse_mount:mount=DirectMount(nas,node,cfg['nas_sftp'],cfg['nas_python'],mount_name='nas-realtime')
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                f=pool.submit(remote,node,service,'realtime')
                while not f.done():
                    guard(cfg['cluster'],heartbeat=True)
                    if tunnel.poll() is not None:
                        (root/'STOP.json').write_text(json.dumps({'reason':'tunnel_lost','at':time.time()}))
                        raise RuntimeError('audio tunnel exited; stop worker cooperatively')
                    concurrent.futures.wait([f],timeout=30)
                result=f.result()
            nas.put('output/realtime-20261009-v1/process-'+generation+'.json',json.dumps(result).encode())
            if result['returncode']:raise RuntimeError('realtime worker failed; read saved NAS log')
        finally:
            if mount:mount.close()
            if tunnel.poll() is None:tunnel.stdin.close();tunnel.wait(timeout=25)
        # The existing offline worker shares this lease, so do not release it.

if __name__=='__main__':main()
