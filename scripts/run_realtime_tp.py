#!/usr/bin/env python3
"""Shadow TP service under the existing K6 Gateway cooperative admission.

No independent lease, catalog switch, or unrelated GPU lifecycle operation.
The current single-GPU service remains live until the shadow passes tests.
"""
import argparse
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
from run_realtime import TUNNEL


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--recover-v49',action='store_true',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    prefix=args.prefix
    if not prefix.startswith('output/realtime-tp4-') or '/' in prefix[7:] or '..' in prefix:
        raise ValueError('unique NAS attempt prefix required')
    preserved=Path('/Volumes/media/CineDuplex/output/realtime-tp4-20261010-v9')
    cleanup=json.loads((preserved/'FAILURE-v50-CLEANUP.json').read_text())
    if not cleanup:raise RuntimeError('original failed group cleanup evidence missing')
    root=Path('/Volumes/media/CineDuplex')/prefix
    if not Path('/Volumes/media').is_mount():raise RuntimeError('real NAS required')
    nas=NAS(cfg['nas']);nas.run('sudo -n -u family mkdir -p '+shlex.quote(nas.path(prefix)),timeout=15)
    with (root/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admitted=guard(cfg['cluster'],heartbeat=True)
        if admitted['state']!='ACTIVE':raise RuntimeError('existing Gateway admission not active')
        if (root/'controller.json').exists():raise RuntimeError('reconcile original TP attempt; never duplicate launch')
        live=json.loads(nas.run('curl -fsS --max-time 8 http://127.0.0.1:18087/status',timeout=12))
        if live.get('active'):raise RuntimeError('wait for the user call before loading shadow inference')
        identities=[]
        for node in cfg['nodes']:
            check=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                '-o','StrictHostKeyChecking=yes',node['ssh_alias'],
                'hostname; whoami; nvidia-smi --query-compute-apps=pid --format=csv,noheader; '
                'awk \'/^MemAvailable:/ {print "MEM " int($2/1024)}\' /proc/meminfo'],
                capture_output=True,text=True,timeout=15)
            lines=check.stdout.splitlines()
            if check.returncode or lines[:2]!=[node['hostname'],node['user']]:raise RuntimeError('TP node identity differs')
            pids=lines[2:-1]
            expected=['1371284','2097709'] if node['rank']==3 else []
            if sorted(pids)!=sorted(expected):raise RuntimeError('TP GPU has unrelated or changed processes')
            if not lines[-1].startswith('MEM ') or int(lines[-1][4:])<(41000 if node['rank']==3 else 65000):
                raise RuntimeError('insufficient memory to preserve the current service')
            identities.append({'node':node['hostname'],'existing_gpu_pids':pids,'available_mb':int(lines[-1][4:])})
        completed=json.loads(nas.read(cfg['run_prefix']+'/'+cfg['training_attempt']+'/GPU-TRAINING-COMPLETE.json',timeout=15))
        original=json.loads(nas.read(cfg['run_prefix']+'/'+cfg['training_attempt']+'/RUN-IDENTITY.json',timeout=15))
        packed=modules();fingerprint=hashlib.sha256(json.dumps(packed,sort_keys=True).encode()).hexdigest()
        service={**cfg,'checkpoint':completed['checkpoint'],'stop_step':completed['steps'],
            'model':original['model'],'mount_name':'nas-training','realtime_prefix':prefix,
            'realtime_port':18092,'master_port':29629,'source_modules_sha256':fingerprint,
            'tp_kernel_sha256':hashlib.sha256(Path('src/cineduplex/train/bf16_kernels.py').read_bytes()).hexdigest()}
        runtime=(preserved/'tp-runtime-v49.py').read_bytes()
        import ast
        def peer_loop(source):
            cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='Coordinator')
            return ast.dump(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='peer_loop'))
        if peer_loop(packed['cineduplex.train.tensor_parallel']['source'])!=peer_loop(runtime):
            raise RuntimeError('boot-bound peer loop differs from recovery runtime')
        runtime_sha=hashlib.sha256(runtime).hexdigest()
        if runtime_sha!='4136194fc7bc64d58c9ff21984f893ad5e5bf04689a5bfc103e27856a3773ba4':
            raise RuntimeError('known-good recovery source differs')
        service['tp_runtime_source']={'name':'tp-runtime-v49.py','sha256':runtime_sha}
        generation=str(time.time_ns())
        nas.put(prefix+'/source-'+generation+'.json',json.dumps(packed,sort_keys=True).encode())
        # Kernels are immutable snapshots; current single-GPU helper is untouched.
        for target in ('bf16-kernels-v3.py','bf16-tp-kernels-v7.py',
                'independent-sampling-v2.py','generation-stream-v4.py','stage-meter-v3.py',
                'call-runtime-v42.py','rank-window-profile-v1.py','native-turn-policy-v6.py','audio-embedding-slice-v2.py',
                'bf16-kernels-v2.py'):
            nas.put(prefix+'/'+target,(preserved/target).read_bytes())
        nas.put(prefix+'/tp-runtime-v49.py',runtime)
        voices=nas.read('output/realtime-20261009-v1/voice-presets.json',timeout=15)
        nas.put(prefix+'/voice-presets.json',voices)
        for spec in json.loads(voices).values():
            nas.put(prefix+'/'+spec['tokens'],nas.read('output/realtime-20261009-v1/'+spec['tokens'],timeout=15))
        nas.put(prefix+'/controller.json',json.dumps({'pid':os.getpid(),'started_at':time.time(),
            'generation':cfg['cluster']['lease']['generation'],'source_sha256':fingerprint,
            'source_snapshot':'source-'+generation+'.json','nodes':identities,
            'single_gpu_service_preserved':True}).encode())
        mounts=[];tunnel=None
        try:
            for node in cfg['nodes']:
                mounts.append(DirectMount(nas,node,cfg['nas_sftp'],cfg['nas_python'],mount_name='nas-training'))
            remote_ssh=['ssh','-N','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ExitOnForwardFailure=yes',
                '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes',
                '-o','UserKnownHostsFile='+nas.path('cache/sshfs/dgx-all-known-hosts'),
                '-L','127.0.0.1:18092:127.0.0.1:18092','spark-01@192.168.12.12']
            tunnel=subprocess.Popen(nas.ssh[:1]+['-A']+nas.ssh[1:]+[nas.guarded(shlex.join([cfg['nas_python'],'-B','-u','-c',TUNNEL]))],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tunnel.stdin.write(json.dumps(remote_ssh).encode()+b'\n');tunnel.stdin.flush()
            if tunnel.stdout.readline()!=b'TUNNEL_READY\n':
                raise RuntimeError('TP task tunnel failed: '+tunnel.stderr.read().decode(errors='replace')[-500:])
            guard(cfg['cluster'],heartbeat=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(remote,node,service,'realtime_tp') for node in cfg['nodes']]
                while not all(f.done() for f in futures):
                    guard(cfg['cluster'],heartbeat=True)
                    if tunnel.poll() is not None:
                        nas.put(prefix+'/STOP.json',json.dumps({'reason':'tp_tunnel_lost','at':time.time()}).encode())
                        raise RuntimeError('TP tunnel lost; preserve identity and reconcile workers')
                    concurrent.futures.wait(futures,timeout=30)
                    failures=[f.result() for f in futures if f.done() and f.result()['returncode']]
                    if failures:
                        # Group timeouts stop blocked collectives; do not issue
                        # a second execution or kill other resident services.
                        if nas.stat(prefix+'/STOP.json') is None:
                            nas.put(prefix+'/STOP.json',json.dumps({'reason':'tp_peer_failed','at':time.time()}).encode())
                results=[f.result() for f in futures]
            nas.put(prefix+'/process-'+generation+'.json',json.dumps(results).encode())
            if any(r['returncode'] for r in results):raise RuntimeError('TP workers failed; see original NAS logs')
        finally:
            if tunnel and tunnel.poll() is None:
                tunnel.stdin.close();tunnel.wait(timeout=25)
            for mount in reversed(mounts):mount.close()
        # The original realtime and offline workers still own this admission.


if __name__=='__main__':main()
