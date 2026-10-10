#!/usr/bin/env python3
"""Hand off the finished training process to one resident inference worker.

The existing Gateway lease stays held throughout. Never overlap workers on the
same GPU, restart training, or take ownership from another Gateway job.
"""
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


def atomic(path,value):
    temp=path.with_name(path.name+'.pending-'+str(os.getpid()))
    temp.write_text(json.dumps(value,ensure_ascii=False));temp.replace(path)


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text())
    root=Path('/Volumes/media/CineDuplex')/cfg['run_prefix']
    if not Path('/Volumes/media').is_mount():raise RuntimeError('real NAS required')
    with (root/'inference-controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((root/'controller-identity.json').read_text())
        identity={'pid':os.getpid(),'training_controller':previous,'started_at':time.time(),
                  'policy':'retain existing Gateway ownership; no overlap on selected GPU'}
        atomic(root/'inference-controller.json',identity)
        complete=root/cfg['training_attempt']/'GPU-TRAINING-COMPLETE.json'
        while not complete.exists():
            guard(cfg['cluster'],heartbeat=True)
            if not alive(previous['pid']):raise RuntimeError('training exited without completed checkpoint')
            time.sleep(30)
        # Rank 0 can keep its original service while an exited training rank
        # takes over using a separate NAS mount and queue namespace.
        node=next(n for n in cfg['nodes'] if n['rank']==3)
        deadline=time.monotonic()+180
        while True:
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                '-o','StrictHostKeyChecking=yes',node['ssh_alias'],
                'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],
                capture_output=True,text=True,timeout=20)
            if r.returncode:raise RuntimeError('cannot inspect selected GPU; retain ownership')
            if not r.stdout.strip():break
            if time.monotonic()>deadline:raise RuntimeError('selected GPU remains occupied; retain ownership')
            guard(cfg['cluster'],heartbeat=True);time.sleep(10)
        guard(cfg['cluster'],heartbeat=True)
        prefix=cfg['run_prefix']+'/'+cfg['training_attempt']
        completed=json.loads((root/cfg['training_attempt']/'GPU-TRAINING-COMPLETE.json').read_text())
        original=json.loads((root/cfg['training_attempt']/'RUN-IDENTITY.json').read_text())
        batches=json.loads((root/'batches.json').read_text())
        service={**cfg,'batch_prefix':cfg['run_prefix'],'run_prefix':prefix,
            'checkpoint':completed['checkpoint'],'stop_step':completed['steps'],
            'model':original['model'],'token_ids':batches['token_ids'],
            'inference_service_seconds':0,'mount_name':'nas-inference','service_namespace':'resident'}
        packed=modules()
        service['source_modules_sha256']=hashlib.sha256(json.dumps(packed,sort_keys=True).encode()).hexdigest()
        nas=NAS(cfg['nas'])
        nas.put(cfg['run_prefix']+'/web/resident-source.json',json.dumps(packed,sort_keys=True).encode())
        atomic(root/'inference-controller.json',{**identity,'phase':'LOADING_RESIDENT_MODEL',
            'checkpoint':completed['checkpoint'],'source_sha256':service['source_modules_sha256']})
        mount=DirectMount(nas,node,cfg['nas_sftp'],cfg['nas_python'],mount_name='nas-inference')
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(remote,node,service,'serve')
                while not future.done():
                    guard(cfg['cluster'],heartbeat=True)
                    concurrent.futures.wait([future],timeout=30)
                result=future.result()
            nas.put(cfg['run_prefix']+'/web/resident-process-'+str(time.time_ns())+'.json',json.dumps(result).encode())
            if result['returncode']:raise RuntimeError('resident worker failed; preserve ownership and saved jobs')
        finally:mount.close()
        atomic(root/'inference-controller.json',{**identity,'phase':'STOPPED','at':time.time()})
        while alive(previous['pid']):
            guard(cfg['cluster'],heartbeat=True);time.sleep(30)
        # Cooperative STOP-SERVICE is honored between jobs. Release only after
        # the worker exited and every GPU is empty under the same generation.
        guard(cfg['cluster'])
        for n in cfg['nodes']:
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',n['ssh_alias'],
                'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],
                capture_output=True,text=True,timeout=20)
            if r.returncode or r.stdout.strip():raise RuntimeError('GPU occupied; retain Gateway ownership')
        lease=cfg['cluster']['lease']
        remote_cli=['ssh','-i',lease['ssh_identity'],'-o','BatchMode=yes','-o','ConnectTimeout=8',lease['ssh_host']]
        commands=[['docker','exec',lease['container'],'python',lease['cli'],'--root',lease['root'],
            'release','--owner',lease['owner'],'--token-file',lease['token_file']],
            ['docker','exec','-e','PYTHONPATH=/workspace/DGXGateway',lease['container'],
             'python','-m','dgx_gateway.maintenance','set-mutation-fence',
             '--config','/run/dgx-gateway/gateway.toml','--enabled','false']]
        results=[]
        for command in commands:
            r=subprocess.run(remote_cli+[shlex.join(command)],capture_output=True,text=True,timeout=20)
            if r.returncode:raise RuntimeError('Gateway release incomplete; inspect existing authority')
            results.append(json.loads(r.stdout))
        atomic(root/'RESOURCE-RELEASED.json',{'at':time.time(),'all_four_gpu_process_lists_empty':True,
            'lease':results[0],'admission':results[1]})


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(type(e).__name__+': '+str(e),flush=True)
        sys.exit(1)
