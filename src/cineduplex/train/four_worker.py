"""Four-rank DDP sound-branch LoRA pilot, with NAS-only persistence."""
from __future__ import annotations
import datetime
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from cineduplex.train.memory_runtime import Store,load_model
from collections import OrderedDict


class FeatureReader:
    def __init__(self,store):self.store=store;self.cache=OrderedDict()
    def batch(self,row):
        if 'batch' in row:return row['batch']
        key=row['feature_path']
        if key not in self.cache:
            self.cache[key]=torch.load(io.BytesIO(self.store.get(key,row['feature_sha256'])),map_location='cpu',weights_only=True)['rows']
            if len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key][row['feature_index']]['batch']


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8):
        super().__init__();self.base=base;self.rank=rank;self.enabled=True
        self.lora_a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
    def forward(self,x):
        result=self.base(x)
        if self.enabled:
            result=result+((x.float()@self.lora_a.T)@self.lora_b.T).to(result.dtype)
        return result


def attach(model):
    selected=[]
    for branch in ('stoken_model','merge_model'):
        module=getattr(model,branch)
        if module is None:raise ValueError('required sound branch missing')
        for name,layer in list(module.named_modules()):
            if isinstance(layer,nn.Linear) and name.rsplit('.',1)[-1] in ('q_proj','v_proj'):
                parent,leaf=name.rsplit('.',1)
                setattr(module.get_submodule(parent),leaf,LoRALinear(layer))
                selected.append(branch+'.'+name)
    if len(selected)!=16:raise ValueError('unexpected adapter module count')
    return selected


def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((str(x.dtype),tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def load_adapter(model,state):
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(params)!=set(state):raise ValueError('adapter key mismatch')
    with torch.no_grad():
        for n,p in params.items():
            if p.shape!=state[n].shape:raise ValueError('adapter shape mismatch')
            p.copy_(state[n])


def serialize(obj):
    stream=io.BytesIO();torch.save(obj,stream);return stream.getvalue()


def run(config):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=4 or torch.cuda.device_count()!=1:raise ValueError('requires four single-GPU workers')
    store=Store(config['url'],config['token']);prefix=config['run_prefix']
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180 if config.get('communication_only') else 3600),device_id=torch.device('cuda:0'))
    # A single bounded real collective, not a broad regression suite.
    probe=torch.tensor([rank+1.],device='cuda');dist.all_reduce(probe)
    if probe.item()!=10:raise ValueError('NCCL sum failed')
    print('rank',rank,'four-node collective passed',flush=True)
    if config.get('communication_only'):
        def counters():
            return {str(p):int(p.read_text()) for p in Path('/sys/class/infiniband').glob('*/ports/1/counters/port_*_data')}
        before=counters();durations=[]
        for i in range(3):
            sample=torch.full((16*1024*1024,),float(rank+1),device='cuda',dtype=torch.float32)
            dist.barrier();torch.cuda.synchronize();started=time.perf_counter()
            dist.all_reduce(sample);torch.cuda.synchronize();durations.append(time.perf_counter()-started)
            if not torch.all(sample==10).item():raise ValueError('64 MiB allreduce mismatch')
        after=counters()
        libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libnccl' in line})
        hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libraries}
        if '876bd4c662e4d68ded40ed87d22d4ffde737278f77059008774335f60c30a832' not in hashes.values():
            raise ValueError('Mesh network plugin not loaded')
        receipt={'ok':True,'rank':rank,'world_size':4,'payload_bytes':64*1024*1024,'allreduce_seconds':durations,
            'correct':True,'libraries':hashes,'rdma_counter_delta':{k:after[k]-v for k,v in before.items()},
            'training_tested':False,'scope':'bounded real communication check, not a throughput scaling claim'}
        store.json(prefix+'/'+config['phase']+f'/rank{rank}.json',receipt)
        print(json.dumps(receipt),flush=True);dist.barrier();dist.destroy_process_group();return
    control_group=dist.new_group(backend='gloo',timeout=datetime.timedelta(seconds=3600))
    model=load_model(store,config['model'],control_group);selected=attach(model)
    model.only_training_stoken_and_control=True
    trainable=[p for p in model.parameters() if p.requires_grad]
    optim=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=0.)
    if config.get('sharded'):
        payload=json.loads(store.get(config['batch_prefix']+'/batches-index.json',config['batches_sha256']))
    else:
        payload=torch.load(io.BytesIO(store.get(config['batch_prefix']+'/batches.pt',config['batches_sha256'])),map_location='cpu',weights_only=True)
    rows=payload['rows']
    if (len(rows)!=config.get('expected_rows',14)
        or {x['scene_id'] for x in rows}!=set(config.get('expected_scenes',[f'S{i:02}' for i in range(1,11)]))
        or {x['source_row'] for x in rows}!=set(range(len(rows)))):
        raise ValueError('training rows differ from the audited batch inventory')
    # Group similar lengths to reduce the longest-rank wait at each update.
    features=FeatureReader(store)
    rows=sorted(rows,key=lambda r:(-r.get('sequence_length',r.get('batch',{}).get('input_ids',torch.empty(1,0)).shape[1]),r['source_row']))
    initial=adapter_state(model);step=0
    if config.get('resume'):
        receipt=json.loads(store.get(config['resume']+'.json'))
        saved=torch.load(io.BytesIO(store.get(config['resume'],receipt['sha256'])),map_location='cpu',weights_only=True)
        for key in ('batches_sha256','model','seed','learning_rate','source_modules_sha256'):
            if saved['identity'][key]!=config[key]:raise ValueError('resume identity mismatch: '+key)
        if state_hash(saved['adapter'])!=saved['metric']['adapter_sha256']:raise ValueError('checkpoint adapter hash mismatch')
        load_adapter(model,saved['adapter']);optim.load_state_dict(saved['optimizer'])
        if optimizer_hash(optim.state_dict())!=saved['optimizer_sha256']:
            raise ValueError('restored optimizer differs from saved state')
        step=saved['step'];rng=saved['rng'][rank]
        torch.cuda.set_rng_state(rng['cuda']);torch.set_rng_state(rng['cpu'])
        store.json(prefix+f'/{config["phase"]}/restored-state-rank{rank}.json',
            {'step':step,'optimizer_sha256':optimizer_hash(optim.state_dict()),
             'optimizer_exact':True,'adapter_sha256':state_hash(adapter_state(model)),
             'cuda_rng_exact':torch.equal(torch.cuda.get_rng_state().cpu(),rng['cuda']),
             'cpu_rng_exact':torch.equal(torch.get_rng_state(),rng['cpu']),
             'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
             'attention_backend':'math','cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
        print('rank',rank,'restored step',step,'optimizer states',len(optim.state),flush=True)
        initial=adapter_state(model)
    # All weights were checksum-loaded on every rank; avoid broadcasting 26GB.
    ddp=DDP(model,device_ids=[0],broadcast_buffers=False,init_sync=False)
    def consensus():
        digest=state_hash(adapter_state(model))
        values=[None]*world;dist.all_gather_object(values,digest,group=control_group)
        if len(set(values))!=1:raise ValueError('rank adapters diverged')
        return digest
    previous_digest=consensus()
    metrics=[];epoch_steps=math.ceil(len(rows)/world)
    for step in range(step+1,config['stop_step']+1):
        torch.cuda.synchronize();started=time.perf_counter()
        offset=((step-1)%epoch_steps)*world;valid=min(world,len(rows)-offset)
        epoch=(step-1)//epoch_steps
        index=offset+(rank+epoch)%world;active=index<len(rows);row=rows[index if active else -1]
        batch={k:v.to('cuda') for k,v in features.batch(row).items()}
        optim.zero_grad(set_to_none=True)
        output=ddp(**batch,use_cache=False)
        loss=output.loss_stoken
        if not torch.isfinite(loss):raise ValueError('nonfinite speech loss')
        dist.barrier(group=control_group)
        # DDP averages ranks, so preserve a mean over actual samples in the tail.
        (loss*(world/valid if active else 0.)).backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
        if not torch.isfinite(norm) or norm.item()==0:raise ValueError('invalid or zero synchronized gradient')
        optim.step();torch.cuda.synchronize()
        global_loss=torch.tensor([loss.item() if active else 0.],device='cuda')
        dist.all_reduce(global_loss);global_loss/=valid
        digest=consensus()
        if digest==previous_digest:raise ValueError('optimizer step did not update parameters')
        previous_digest=digest
        metric={'step':step,'rank':rank,'scene':row['scene_id'] if active else None,
            'source_row':row['source_row'] if active else None,'parameters_changed_this_step':True,
            'active_sample':active,'loss_local':loss.item(),'loss_global_mean':global_loss.item(),
            'gradient_norm':norm.item(),'adapter_sha256':digest,'seconds':time.perf_counter()-started,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
        metrics.append(metric);print(json.dumps(metric),flush=True)
        store.json(prefix+f'/{config["phase"]}/rank{rank}-step{step:04}.json',metric)
        if rank==0 and config.get('bulk_training'):
            live=store.file_path(prefix+'/'+config['phase']+'/LIVE.json')
            temp=live.with_suffix('.pending');temp.write_text(json.dumps({**metric,'updated_at':time.time(),
                'total_steps':config['stop_step'],'phase':'TRAINING'}));temp.replace(live)
        if step in config['save_steps']:
            rngs=[None]*world;dist.all_gather_object(rngs,
                {'cuda':torch.cuda.get_rng_state().cpu(),'cpu':torch.get_rng_state()},group=control_group)
            if rank==0:
                saved={'format':'cineduplex-ddp-lora-v2','step':step,'world_size':world,
                    'adapter':adapter_state(model),'optimizer':optim.state_dict(),'rng':rngs,
                    'optimizer_sha256':optimizer_hash(optim.state_dict()),
                    'identity':{k:config[k] for k in ('batches_sha256','model','seed','learning_rate','source_modules_sha256')},
                    'modules':selected,'metric':metric,'quality_approved':False}
                key=prefix+f'/{config["phase"]}/checkpoint-{step:04}.pt'
                receipt=store.put(key,serialize(saved));store.json(key+'.json',receipt)
            dist.barrier(group=control_group)
        if config.get('compare_step')==step:
            receipt=json.loads(store.get(config['compare_checkpoint']+'.json'))
            ref=torch.load(io.BytesIO(store.get(config['compare_checkpoint'],receipt['sha256'])),map_location='cpu',weights_only=True)
            actual=adapter_state(model)
            max_delta=max((actual[k]-ref['adapter'][k]).abs().max().item() for k in actual)
            same=all(torch.allclose(actual[k],ref['adapter'][k],rtol=1e-4,atol=1e-6) for k in actual)
            loss_delta=abs(metric['loss_global_mean']-ref['metric']['loss_global_mean'])
            store.json(prefix+f'/{config["phase"]}/resume-comparison-rank{rank}.json',
                {'ok':same and loss_delta<1e-3,'max_adapter_difference':max_delta,'loss_difference':loss_delta,
                 'reference_step':step,'independent_worker_restart':True})
            if not same or loss_delta>=1e-3:raise ValueError('numerical resume mismatch')
        del output,loss,batch
    final=adapter_state(model)
    changed=any(not torch.equal(initial[k],final[k]) for k in initial)
    if not changed:raise ValueError('training did not change adapters')
    store.json(prefix+f'/{config["phase"]}/rank{rank}-complete.json',
        {'ok':True,'rank':rank,'world_size':world,'last_step':step,'changed':changed,
         'trainable_parameters':sum(p.numel() for p in trainable),'modules':selected,
         'adapter_sha256':consensus(),'quality_improvement':'NOT_EVALUATED',
         'observed_scenes':sorted({m['scene'] for m in metrics if m['active_sample']}),
         'observed_source_rows':sorted({m['source_row'] for m in metrics if m['active_sample']})})
    dist.barrier(group=control_group);del ddp
    if config.get('generate') and rank==0:
        from cineduplex.train.generate import comparisons
        rows=[{**r,'batch':features.batch(r)} for r in rows if r['scene_id'] in config['generation_scenes']]
        comparisons(model,rows,store,config)
    dist.barrier(group=control_group);dist.destroy_process_group()
    if config.get('serve_after_training') and rank==0:
        receipts=[json.loads(store.get(prefix+f'/{config["phase"]}/rank{i}-complete.json')) for i in range(4)]
        if {i for r in receipts for i in r['observed_source_rows']}!=set(range(len(rows))):
            raise ValueError('incomplete training coverage')
        store.json(prefix+'/GPU-TRAINING-COMPLETE.json',{'ok':True,'rows':len(rows),'steps':step,
            'checkpoint':prefix+f'/{config["phase"]}/checkpoint-{step:04}.pt',
            'adapter_sha256':state_hash(final),'quality_approved':False,'rank_receipts':receipts})
        # A small actual holdout forward pass, after all main training steps.
        evaluation=json.loads(store.get(config['batch_prefix']+'/eval-features.json'))['rows']
        model.eval();losses=[]
        with torch.inference_mode():
            for row in evaluation[::max(1,len(evaluation)//16)][:16]:
                batch={k:v.to('cuda') for k,v in features.batch(row).items()}
                loss=model(**batch,use_cache=False).loss_stoken
                if not torch.isfinite(loss):raise ValueError('nonfinite held-out loss')
                losses.append(float(loss));del loss,batch
        store.json(prefix+'/EVALUATION.json',{'rows':len(losses),'mean_speech_loss':sum(losses)/len(losses),
            'split':'project recording holdout','quality_approved':False})
        from cineduplex.train.inference_service import serve
        serve(model,store,config)
