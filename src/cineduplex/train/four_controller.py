"""Mac controller: verified NAS service, memory workers, four-rank train/resume."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from .nas_transport import NAS

PROJECT=Path(__file__).resolve().parents[3]
REMOTE_PYTHON='cineduplex-work/envs/p0-a/bin/python'
MESH={'libnccl.so.2.29.7':'de86e56c99a2c616bfbcb0c65c54380ca9b73abbc6a958383857633d7853626f',
      'libnccl-net.so':'876bd4c662e4d68ded40ed87d22d4ffde737278f77059008774335f60c30a832'}


def modules():
    out={p:{'source':'','package':True} for p in ('cineduplex','cineduplex.train','Models')}
    for name in ('memory_runtime','four_worker','generate','export_memory','export_bulk','decode_generated','codec_config','inference_service','realtime_service','native_turn_policy','audio_embedding_slice','independent_sampling','generation_stream','stage_meter','tensor_parallel'):
        out['cineduplex.train.'+name]={'source':(PROJECT/'src/cineduplex/train'/f'{name}.py').read_text(),'package':False}
    out['cineduplex.guardrail']={'source':'','package':True}
    for name in ('aliyun_guardrail','call_guardrail'):
        out['cineduplex.guardrail.'+name]={'source':(PROJECT/'src/cineduplex/guardrail'/f'{name}.py').read_text(),'package':False}
    manifest=json.loads((PROJECT/'configs/vendor-source-manifest.json').read_text())
    for name in ('LycheeFD','configuration_step_audio_2','modeling_step_audio_2'):
        rel='Training_package/Code/Models/'+name+'.py'
        data=(PROJECT/'vendor/lychee-fd-source'/rel).read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest['files'][rel]:raise ValueError('vendor source changed')
        out['Models.'+name]={'source':data.decode(),'package':False}
    vendor=PROJECT/'vendor/lychee-fd-source'
    for path in (vendor/'third_party/Step-Audio2').rglob('*.py'):
        rel=str(path.relative_to(vendor));data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest['files'].get(rel):raise ValueError('codec source changed: '+rel)
        parts=list(path.relative_to(vendor/'third_party/Step-Audio2').with_suffix('').parts)
        package=parts[-1]=='__init__'
        if package:parts.pop()
        name='.'.join(parts)
        if not name:continue
        for i in range(1,len(parts)):out.setdefault('.'.join(parts[:i]),{'source':'','package':True})
        out[name]={'source':data.decode(),'package':package}
    # The upstream realtime state machine is used unchanged. Strip only unused
    # optional plotting imports when packing, since the runtime has no UI deps.
    for name in ('lychee_fd','lychee_fd.runtime'):
        out[name]={'source':'','package':True}
    for name in ('hf_v9_generation','vllm_generation','hf_v9_realtime'):
        source=(vendor/'lychee_fd/runtime'/f'{name}.py').read_text()
        source=source.replace('import matplotlib.pyplot as plt\n','')
        out['lychee_fd.runtime.'+name]={'source':source,'package':False}
    return out


BOOTSTRAP=r'''
import json,sys,types,signal,os,hashlib,subprocess,tempfile,pathlib,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
payload=json.load(sys.stdin)
signal.alarm(0 if payload['action'] in ('serve','realtime','realtime_tp') else payload['config'].get('worker_timeout_seconds',10800) if payload['action']=='train' and not payload['config'].get('communication_only') else 1800)
cfg=payload['config']
if subprocess.check_output(['hostname'],text=True).strip()!=cfg['hostname']:raise ValueError('wrong host')
if __import__('getpass').getuser()!=cfg['user']:raise ValueError('wrong user')
# Verify the task's mounted NAS before any library may create a cache.
assert cfg.get('mount_name','nas-training') in ('nas-training','nas-inference','nas-realtime')
mount=pathlib.Path.home()/'cineduplex-work'/cfg.get('mount_name','nas-training')
assert subprocess.check_output(['findmnt','-rn','-o','FSTYPE','-T',str(mount)],text=True).strip()=='fuse.sshfs'
cache=mount/'cache/train-pilot10'/cfg['hostname']
cache.mkdir(parents=True,exist_ok=True)
for key,child in {'TMPDIR':'tmp','TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton',
 'HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg','NUMBA_CACHE_DIR':'numba','TORCH_HOME':'torch'}.items():
 path=cache/child;path.mkdir(exist_ok=True);os.environ[key]=str(path)
tempfile.tempdir=os.environ['TMPDIR']
if payload['action'] in ('train','realtime_tp'):
 for name,sha in payload['mesh'].items():
  with open('/srv/h3-ring/mesh/'+name,'rb') as f:
   if hashlib.file_digest(f,'sha256').hexdigest()!=sha:raise ValueError('Mesh library mismatch')
for name in ('cineduplex','cineduplex.train'):
 m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
name='cineduplex.train.memory_runtime';m=types.ModuleType(name);sys.modules[name]=m
exec(compile(payload['modules'][name]['source'],'nas-memory://memory_runtime.py','exec'),m.__dict__)
sys.meta_path.insert(0,m.MemoryImporter(payload['modules']))
if payload['action']=='prepare':
 if cfg.get('input_files'):
  from cineduplex.train.export_bulk import run
 else:
  from cineduplex.train.export_memory import run
elif payload['action']=='audit':
 from cineduplex.train.memory_runtime import audit as run
elif payload['action']=='decode':
 from cineduplex.train.decode_generated import decode
 def run(c):
  import torch
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  decode(m.Store(c['url'],''),c)
elif payload['action']=='serve':
 from cineduplex.train.inference_service import run
elif payload['action']=='realtime':
 from cineduplex.train.realtime_service import run
elif payload['action']=='realtime_tp':
 from cineduplex.train.tensor_parallel import run
else:
 from cineduplex.train.four_worker import run
run(cfg)
'''


def remote(node,cfg,action,log_callback=None):
    env={'PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_DISABLE':'1','NUMBA_DISABLE_JIT':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
         'HF_HOME':'/proc/cineduplex-no-cache','WANDB_MODE':'disabled','WANDB_DISABLED':'true',
         'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    if action in ('train','realtime_tp'):
        env.update(RANK=str(node['rank']),WORLD_SIZE='4',LOCAL_RANK='0',CUDA_VISIBLE_DEVICES='0',
            MASTER_ADDR=cfg['master_addr'],MASTER_PORT=str(cfg['master_port']),
            LD_PRELOAD='/srv/h3-ring/mesh/libnccl.so.2',LD_LIBRARY_PATH='/srv/h3-ring/mesh',
            NCCL_NET_PLUGIN='mesh',NCCL_NET='Mesh',NCCL_ALGO='Ring',NCCL_SOCKET_IFNAME='=wlP9s9',
            NCCL_SOCKET_FAMILY='AF_INET',NCCL_MESH_GID_INDEX='3',NCCL_MESH_DISABLE_RDMA='0',
            GLOO_SOCKET_IFNAME='wlP9s9',
            NCCL_MESH_DEBUG='1',NCCL_NVLS_ENABLE='0',NCCL_RUNTIME_CONNECT='1',
            NCCL_IGNORE_CPU_AFFINITY='1',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    else:env['CUDA_VISIBLE_DEVICES']='0' if action in ('decode','serve','realtime') else ''
    command='env '+' '.join(shlex.quote(k+'='+v) for k,v in env.items())+' "$HOME"/'+REMOTE_PYTHON+' -u -c '+shlex.quote(BOOTSTRAP)
    packed=modules()
    fingerprint=hashlib.sha256(json.dumps(packed,sort_keys=True).encode()).hexdigest()
    if cfg.get('source_modules_sha256',fingerprint)!=fingerprint:raise ValueError('training source changed during run')
    payload={'modules':packed,'config':{**cfg,**node,'url':'file:///home/'+node['user']+'/cineduplex-work/'+cfg.get('mount_name','nas-training'),'token':''},'action':action,'mesh':MESH}
    proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes',
        node['ssh_alias'],command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    proc.stdin.write(json.dumps(payload));proc.stdin.close()
    lines=[];started=time.monotonic()
    def consume():
        for line in proc.stdout:
            lines.append(line);print(node['hostname']+': '+line.rstrip(),flush=True)
    reader=threading.Thread(target=consume,daemon=True);reader.start()
    try:
        while proc.poll() is None:
            limit=cfg.get('worker_timeout_seconds',10800) if action=='train' and not cfg.get('communication_only') else 1800
            if action not in ('serve','realtime','realtime_tp') and time.monotonic()-started>limit:raise TimeoutError('worker time budget exhausted')
            time.sleep(.5)
        reader.join(timeout=10)
        if reader.is_alive():raise TimeoutError('worker output did not close')
        result={'hostname':node['hostname'],'returncode':proc.returncode,'log':''.join(lines)}
        if log_callback:log_callback(result)
        return result
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=15)


def main(cfg,action):
    # Even CPU prepare mounts the DGX hosts: acquire cooperative ownership first.
    from cineduplex.cluster.guard import guard
    guard(cfg['cluster'],heartbeat=True)
    if action in ('train','decode'):
        attempt=cfg['training_attempt']
        if not attempt.startswith('attempt-') or not attempt.removeprefix('attempt-').isdigit():
            raise ValueError('training_attempt must be attempt-N; never overwrite earlier attempts')
        cfg={**cfg,'batch_prefix':cfg['run_prefix'],'run_prefix':cfg['run_prefix']+'/'+attempt}
    nas=NAS(cfg['nas']);common=dict(cfg)
    common['source_modules_sha256']=hashlib.sha256(json.dumps(modules(),sort_keys=True).encode()).hexdigest()
    loader=(PROJECT/'vendor/lychee-fd-source/Training_package/Code/DataLoaders/LycheeFDDataset.py').read_bytes()
    rel='cache/train-code/loader-source.py'
    if nas.stat(rel) is None:nas.put(rel,loader)
    elif nas.read(rel)!=loader:raise ValueError('cached loader source changed')
    mounts=[]
    try:
        from cineduplex.train.nas_mount import DirectMount
        nodes=cfg['nodes'] if action not in ('prepare','audit','decode') else [next(n for n in cfg['nodes'] if n['rank']==(0 if action=='decode' else 3))]
        for node in nodes:mounts.append(DirectMount(nas,node,cfg['nas_sftp'],cfg['nas_python']))
        if action in ('prepare','audit'):
            node=next(n for n in cfg['nodes'] if n['hostname']=='spark-4350')
            result=remote(node,{**common,'existing_tokenizer':cfg['existing_root']},action)
            nas.put(cfg['run_prefix']+'/'+action+'-receipt-'+str(time.time_ns())+'.json',json.dumps(result).encode())
            if result['returncode']:raise RuntimeError('CPU preparation failed; see NAS receipt')
            return
        from cineduplex.cluster.guard import guard
        guard(cfg['cluster'],heartbeat=True)
        if action=='decode':
            for i in range(4):
                trained=json.loads(nas.read(cfg['run_prefix']+f'/resumed/rank{i}-complete.json'))
                recovered=json.loads(nas.read(cfg['run_prefix']+f'/resumed/resume-comparison-rank{i}.json'))
                if not trained['ok'] or trained['last_step']!=16 or not recovered['ok']:
                    raise ValueError('decode-only requires completed four-node training and verified resume')
            result=remote(nodes[0],common,'decode')
            nas.put(cfg['run_prefix']+'/decode-process-'+str(time.time_ns())+'.json',json.dumps(result).encode())
            if result['returncode']:raise RuntimeError('standalone codec stage failed')
            from cineduplex.train.finalize_run import finalize
            finalize(nas,cfg['run_prefix'])
            print('FOUR-NODE TRAIN/RESUME/GENERATED-WAV FLOW COMPLETE; listening remains',flush=True)
            return
        if action=='communicate':
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(remote,n,{**common,'communication_only':True,'phase':cfg['communication_phase']},'train') for n in cfg['nodes']]
                while not all(f.done() for f in futures):
                    guard(cfg['cluster'],heartbeat=True);concurrent.futures.wait(futures,timeout=30)
                results=[f.result() for f in futures]
            nas.put(cfg['run_prefix']+'/'+cfg['communication_phase']+'/processes.json',json.dumps(results).encode())
            if any(r['returncode'] for r in results):raise RuntimeError('four-node collective failed')
            return
        batches=json.loads(nas.read(cfg['batch_prefix']+'/batches.json'))
        data=cfg.get('data_input')
        if data and (batches.get('data_input')!=data or batches['rows']!=data['expected_rows']
                or set(batches['scenes'])!=set(data['expected_scenes'])):
            raise ValueError('prepared batches do not match the audited data configuration')
        generation_scenes=cfg.get('generation_scenes',['S03','S06','S07','S08'])
        if not generation_scenes or len(set(generation_scenes))!=len(generation_scenes) or not set(generation_scenes)<=set(batches['scenes']):
            raise ValueError('generation scenes must be unique audited training scenes')
        common.update(expected_rows=batches['rows'],expected_scenes=batches['scenes'],generation_scenes=generation_scenes)
        common.update(batches_sha256=batches['sha256'],token_ids=batches['token_ids'],sharded=batches.get('sharded',False))
        repo=json.loads(Path(cfg['manifest']).read_text())['repositories'][0]
        shards=[e for e in repo['files'] if e['path'].endswith('.safetensors')]
        for entry in shards:
            if nas.stat('raw/lychee-full-duplex/'+entry['path'])!=entry['size']:raise ValueError('weights not complete')
        common['model']={'shards':[{k:e[k] for k in ('path','sha256','size')} for e in shards],
            'config_sha256':hashlib.sha256(nas.read('raw/lychee-full-duplex/config.json')).hexdigest(),
            'index_sha256':hashlib.sha256(nas.read('raw/lychee-full-duplex/model.safetensors.index.json')).hexdigest()}
        nas.put(cfg['run_prefix']+'/RUN-IDENTITY.json',json.dumps({
            k:common[k] for k in ('source_modules_sha256','batches_sha256','model','seed','learning_rate','expected_rows','expected_scenes','generation_scenes')
        },indent=2).encode())
        if cfg.get('bulk_training'):
            import math
            steps=math.ceil(batches['rows']/4)*cfg.get('epochs',1)
            saves=sorted({2,steps,*range(64,steps+1,64)})
            run={**common,'phase':'training','stop_step':steps,'save_steps':saves,'generate':False}
            def record_bulk(result):
                nas.put(cfg['run_prefix']+'/training/'+result['hostname']+'-process.json',json.dumps(result).encode())
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(remote,n,run,'train',record_bulk) for n in cfg['nodes']]
                while not all(f.done() for f in futures):
                    guard(cfg['cluster'],heartbeat=True)
                    concurrent.futures.wait(futures,timeout=30)
                results=[f.result() for f in futures]
            if any(r['returncode'] for r in results):raise RuntimeError('bulk training worker failed; preserve checkpoint')
            receipts=[json.loads(nas.read(cfg['run_prefix']+f'/training/rank{i}-complete.json')) for i in range(4)]
            observed={i for r in receipts for i in r['observed_source_rows']}
            if observed!=set(range(batches['rows'])) or not all(r['ok'] and r['last_step']==steps for r in receipts):
                raise ValueError('training input coverage incomplete')
            if len({r['adapter_sha256'] for r in receipts})!=1:raise ValueError('four-rank adapter mismatch')
            nas.put(cfg['run_prefix']+'/TRAINING-COMPLETE.json',json.dumps({
                'ok':True,'rows':batches['rows'],'steps':steps,'epochs':cfg.get('epochs',1),
                'checkpoint':cfg['run_prefix']+f'/training/checkpoint-{steps:04}.pt',
                'input_files':cfg['input_files'],'quality_approved':False,'rank_receipts':receipts}).encode())
            print('BULK TRAINING COMPLETE; audio quality not evaluated',flush=True)
            return
        # Two independent SSH/Python launches; no in-process resume substitution.
        for phase,stop,saves in [('initial',9,[8,9]),('resumed',16,[16])]:
            run={**common,'phase':phase,'stop_step':stop,'save_steps':saves,'generate':phase=='resumed'}
            if phase=='resumed':run.update(resume=cfg['run_prefix']+'/initial/checkpoint-0008.pt',
                compare_step=9,compare_checkpoint=cfg['run_prefix']+'/initial/checkpoint-0009.pt')
            def record(result):
                nas.put(cfg['run_prefix']+'/'+phase+'/'+result['hostname']+'-process.json',json.dumps(result).encode())
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(remote,n,run,'train',record) for n in cfg['nodes']]
                while not all(f.done() for f in futures):
                    guard(cfg['cluster'],heartbeat=True)
                    concurrent.futures.wait(futures,timeout=30)
                results=[f.result() for f in futures]
            if any(r['returncode'] for r in results):raise RuntimeError('four-node phase failed: '+phase)
        rank_receipts=[json.loads(nas.read(cfg['run_prefix']+f'/resumed/rank{i}-complete.json')) for i in range(4)]
        resume=[json.loads(nas.read(cfg['run_prefix']+f'/resumed/resume-comparison-rank{i}.json')) for i in range(4)]
        audio=json.loads(nas.read(cfg['run_prefix']+'/audio/summary.json'))
        if not all(r['ok'] and r['last_step']==16 and r['changed'] for r in rank_receipts):raise ValueError('incomplete training')
        if {s for r in rank_receipts for s in r['observed_scenes']}!=set(batches['scenes']):
            raise ValueError('not all audited scenes were trained')
        if {i for r in rank_receipts for i in r['observed_source_rows']}!=set(range(batches['rows'])):
            raise ValueError('missing training rows')
        if not all(r['ok'] for r in resume) or len({r['adapter_sha256'] for r in rank_receipts})!=1:
            raise ValueError('resume or rank consensus missing')
        if len(audio['records'])!=2*len(generation_scenes):raise ValueError('missing audio comparison')
        for record in audio['records']:
            if hashlib.sha256(nas.read(record['path'])).hexdigest()!=record['sha256']:raise ValueError('WAV readback failed')
        from cineduplex.train.finalize_run import finalize
        finalize(nas,cfg['run_prefix'])
        print('FOUR-NODE TRAIN/RESUME/GENERATED-WAV FLOW COMPLETE; listening and quality evaluation remain',flush=True)
    finally:
        for mount in reversed(mounts):mount.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','communicate','audit','decode']);p.add_argument('config')
    args=p.parse_args();main(json.loads(Path(args.config).read_text()),args.action)
