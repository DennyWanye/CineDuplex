"""Same-BF16 tensor parallelism for the native four-branch streaming model.

Decode keeps all attention heads and MLP channels local to each rank.
Prefill retains native full-matrix BF16 projections; decode row projections
reduce FP32 partial sums before a BF16 cast.
Sampling is owned by rank 3 so concurrent codec RNG never desynchronizes peers.
This module does not acquire resources or launch workers.
"""
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
import time
import types
import threading
import struct
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

LEADER=3
TP_GROUP=None
CALL_RUNTIME_SHA='9f03c2235d518dc045d66d66cf2409662be346f2b988442a79424013d37472ed'

def install_call_runtime(root):
    path=Path(root)/'call-runtime-v42.py';source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=CALL_RUNTIME_SHA:raise ValueError('call runtime source identity')
    if dist.get_rank()!=LEADER:return
    import cineduplex.train.realtime_service as runtime
    if getattr(runtime,'_call_runtime_sha',None)==CALL_RUNTIME_SHA:return
    module=types.ModuleType('cineduplex.train.call_runtime_update');module.__package__='cineduplex.train'
    exec(compile(source,str(path),'exec'),module.__dict__)
    module.StageMeter=runtime.StageMeter
    runtime.Call=module.Call;runtime._call_runtime_sha=CALL_RUNTIME_SHA

# Existing Coordinators execute a pinned update module before replacing their
# class. Install Call before the next Application.start constructs a session.
if __name__.startswith('cineduplex_tp_update_'):
    install_call_runtime(Path(install_call_runtime.__code__.co_filename).parent)
TP_KERNEL_NAME='bf16-tp-kernels-v7.py'
TP_KERNEL_SHA256='40b892083c0ce8b8cbf5f005002ab3e4c90314a8d348b7a79ab4657c93cb931b'


def initialize():
    if int(os.environ['WORLD_SIZE'])!=4:raise ValueError('TP requires four admitted ranks')
    torch.cuda.set_device(0)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180),device_id=torch.device('cuda:0'))
    # Shard loading shares NAS bandwidth with the preserved services. Keep
    # its proven budget separate from the interactive command timeout.
    control=dist.new_group(backend='gloo',timeout=datetime.timedelta(seconds=1800))
    probe=torch.tensor([dist.get_rank()+1.],device='cuda');dist.all_reduce(probe)
    if probe.item()!=10:raise ValueError('TP collective failed')
    return control


class ParallelLinear:
    def __init__(self,weight,bias,axis,kernels):
        if weight.dtype!=torch.bfloat16:raise ValueError('TP weight precision must be BF16')
        self.rank=dist.get_rank();self.world=dist.get_world_size();self.axis=axis
        self.n,self.k=weight.shape;self.kernels=kernels;self.bias=bias;self.full_weight=weight.detach()
        self.width=math.ceil(weight.shape[axis]/self.world)
        start=self.rank*self.width;end=min(start+self.width,weight.shape[axis])
        part=weight.detach().narrow(axis,start,max(0,end-start)).contiguous()
        if end-start<self.width:
            shape=list(part.shape);shape[axis]=self.width
            padded=part.new_zeros(shape);padded.narrow(axis,0,end-start).copy_(part);part=padded
        self.weight=part

    def __call__(self,x):
        if x.numel()>x.shape[-1]:
            # Prefill is a GEMM workload. Preserve the native full-matrix
            # arithmetic instead of applying the decode GEMV partition.
            if self.axis==1 and x.shape[-1]==self.width:
                flat=x.reshape(-1,self.width).contiguous()
                gathered=torch.empty((self.world*flat.shape[0],self.width),device=x.device,dtype=x.dtype)
                dist.all_gather_into_tensor(gathered,flat,group=TP_GROUP)
                x=gathered.reshape(self.world,*x.shape[:-1],self.width).movedim(0,-2).reshape(*x.shape[:-1],self.world*self.width)[...,:self.k]
            return torch.nn.functional.linear(x,self.full_weight,self.bias)
        if self.axis==0:
            y=(self.kernels.linear(x.contiguous(),self.weight,None) if x.numel()==self.k else
                torch.nn.functional.linear(x,self.weight))
            flat=y.reshape(-1,self.width).contiguous()
            gathered=torch.empty((self.world*flat.shape[0],self.width),device=x.device,dtype=x.dtype)
            dist.all_gather_into_tensor(gathered,flat,group=TP_GROUP)
            result=gathered.reshape(self.world,*x.shape[:-1],self.width).movedim(0,-2)
            result=result.reshape(*x.shape[:-1],self.world*self.width)[...,:self.n]
        else:
            start=self.rank*self.width;end=min(start+self.width,self.k)
            if x.shape[-1]==self.width:
                part=x.contiguous()
            elif x.shape[-1]==self.k:
                part=x[...,start:end].contiguous()
                if end-start<self.width:part=torch.nn.functional.pad(part,(0,self.width-(end-start)))
            else:raise ValueError('TP row projection input width')
            # No early BF16 rounding of row partials. Prefill uses full FP32
            # multiplication too; it is uncommon and stays numerically explicit.
            result=(self.kernels.linear(part,self.weight,None,output_dtype=torch.float32) if part.numel()==self.width else
                torch.mm(part.reshape(-1,self.width),self.weight.t(),out_dtype=torch.float32).reshape(*part.shape[:-1],self.n))
            dist.all_reduce(result,group=TP_GROUP)
            result=result.to(x.dtype)
        if self.bias is not None:result=result+self.bias
        return result


class PackedColumns:
    """One projection/gather for QKV or gate/up with the same native input."""
    def __init__(self,specs,kernels,local):
        self.world=dist.get_world_size();self.rank=dist.get_rank();self.kernels=kernels
        self.specs=specs;self.local=local;self.widths=[math.ceil(weight.shape[0]/self.world) for weight,_ in specs]
        pieces=[]
        for (weight,_),width in zip(specs,self.widths):
            start=self.rank*width;part=weight[start:min(start+width,weight.shape[0])]
            if part.shape[0]<width:part=torch.nn.functional.pad(part,(0,0,0,width-part.shape[0]))
            pieces.append(part)
        self.weight=torch.cat(pieces,0).contiguous()
        self.bias=(torch.cat([bias[self.rank*width:(self.rank+1)*width] for (_,bias),width in zip(specs,self.widths)])
            if all(bias is not None for _,bias in specs) else None)
        if any(bias is not None for _,bias in specs) and self.bias is None:raise ValueError('mixed TP packed bias')
        self.cached_input=None;self.outputs=None;self.used=set()

    def compute(self,x):
        if x.numel()>x.shape[-1]:
            outputs=[]
            for (weight,bias),width in zip(self.specs,self.widths):
                y=torch.nn.functional.linear(x,weight,bias)
                outputs.append(y[...,self.rank*width:(self.rank+1)*width] if self.local else y)
            return outputs
        y=self.kernels.linear(x.contiguous(),self.weight,self.bias)
        # Attention heads and MLP intermediate channels stay local. Only
        # their output row projections reduce the full hidden state.
        if self.local:grouped=y
        else:
            width=sum(self.widths);flat=y.reshape(-1,width).contiguous()
            gathered=torch.empty((self.world*flat.shape[0],width),device=x.device,dtype=x.dtype)
            dist.all_gather_into_tensor(gathered,flat,group=TP_GROUP)
            grouped=gathered.reshape(self.world,*x.shape[:-1],width).movedim(0,-2)
        result=[];offset=0
        for (weight,bias),width in zip(self.specs,self.widths):
            value=grouped[...,offset:offset+width]
            if not self.local:value=value.reshape(*x.shape[:-1],self.world*width)[...,:weight.shape[0]]
            result.append(value);offset+=width
        return result

    def select(self,index,x):
        # Native attention calls Q/K/V on the exact same tensor, and the
        # native MLP does the same for gate/up. Release cache after the group.
        if x is not self.cached_input or index in self.used:
            self.outputs=self.compute(x);self.cached_input=x;self.used=set()
        result=self.outputs[index];self.used.add(index)
        if len(self.used)==len(self.specs):
            self.cached_input=None;self.outputs=None;self.used=set()
        return result


class Coordinator:
    command_transport="fixed-byte-nccl-active/gloo-idle"
    def __init__(self,control,root,kernel_sha256):
        self.control=control;self.root=root;self.installed=[];self.rank=dist.get_rank()
        self.kernels=None;self.kernel_sha256=kernel_sha256

    def command(self,value=None):
        # Resident peers remain inside their original bound peer_loop. Hook
        # its dynamically dispatched command boundary, not a future loop.
        if self.rank!=LEADER:self.finish_window()
        # Idle/start/update commands keep the admitted CPU GLOO channel.
        # During a call, send the same fixed frame through the already admitted
        # NCCL Mesh group. No socket, route, interface or model changes.
        # Stop returns every rank to GLOO; idle peers do not spin on a GPU.
        if not hasattr(self,'command_lock'):self.command_lock=threading.RLock()
        with self.command_lock:
            if not hasattr(self,'command_storage'):
                self.command_storage=bytearray(16384)
                self.command_tensor=torch.frombuffer(self.command_storage,dtype=torch.uint8)
            while True:
                if self.rank==LEADER:
                    meta=dict(value);pcm=meta.pop('pcm',b'')
                    meta['_pcm_bytes']=len(pcm)
                    header=json.dumps(meta,separators=(',',':')).encode()
                    total=8+len(header)+len(pcm)
                    if total>len(self.command_storage):raise ValueError('TP command exceeds fixed frame')
                    struct.pack_into('<II',self.command_storage,0,len(header),len(pcm))
                    self.command_storage[8:8+len(header)]=header
                    self.command_storage[8+len(header):total]=pcm
                dispatch_started=time.perf_counter()
                if getattr(self,'command_cuda_active',False):
                    if not hasattr(self,'command_cuda_tensor'):
                        # This buffer survives the inference-mode audio loop
                        # and is reused by stop/heartbeat lifecycle code.
                        with torch.inference_mode(False):
                            self.command_cuda_tensor=torch.empty_like(self.command_tensor,device='cuda')
                    if self.rank==LEADER:self.command_cuda_tensor.copy_(self.command_tensor)
                    dist.broadcast(self.command_cuda_tensor,src=LEADER,group=TP_GROUP)
                    # The native peer state machine needs the exact CPU PCM.
                    # Copy only this 16KiB frame after the collective completes.
                    self.command_tensor.copy_(self.command_cuda_tensor)
                else:
                    dist.broadcast(self.command_tensor,src=LEADER,group=self.control)
                dispatch_ms=(time.perf_counter()-dispatch_started)*1000
                header_len,pcm_len=struct.unpack_from('<II',self.command_storage)
                if not 0<header_len or 8+header_len+pcm_len>len(self.command_storage):
                    raise ValueError('invalid TP command frame')
                item=json.loads(bytes(self.command_storage[8:8+header_len]))
                if item.pop('_pcm_bytes')!=pcm_len:raise ValueError('TP PCM frame size mismatch')
                if pcm_len:item['pcm']=bytes(self.command_storage[8+header_len:8+header_len+pcm_len])
                if item['kind']=='start':self.command_cuda_active=True
                elif item['kind']=='stop':self.command_cuda_active=False
                if self.rank!=LEADER:
                    if item['kind']=='start':self.profile_id=item.get('profile_id')
                    elif item['kind']=='audio' and getattr(self,'rank_profile',None) is not None:
                        self.rank_profile.begin(item['profile_seq'])
                    elif item['kind']=='stop' and getattr(self,'rank_profile',None) is not None:
                        self.rank_profile.close();self.rank_profile=None
                profile=getattr(self,'rank_profile',None)
                if item['kind']=='audio' and profile is not None:
                    profile.command_dispatch_ms=dispatch_ms
                if self.rank==LEADER or item['kind']!='heartbeat':return item

    def keep_idle_channel_alive(self):
        if self.rank!=LEADER or hasattr(self,'idle_heartbeat'):return
        if not hasattr(self,'command_lock'):self.command_lock=threading.RLock()
        if not hasattr(self,'in_call'):self.in_call=False
        def loop():
            while not (self.root/'STOP.json').exists():
                time.sleep(10)
                with self.command_lock:
                    if not self.in_call and not (self.root/'STOP.json').exists():
                        try:self.command({'kind':'heartbeat'})
                        except Exception:
                            # A failed command group cannot safely start a
                            # second call. Exit only this candidate worker.
                            os._exit(1)
        self.idle_heartbeat=threading.Thread(target=loop,daemon=True)
        self.idle_heartbeat.start()

    def apply_update(self,spec):
        if getattr(self,'rank_profile',None) is not None:
            self.rank_profile.close();self.rank_profile=None
        name=spec['name']
        if not name.startswith('tp-runtime-v') or not name.endswith('.py') or '/' in name:
            raise ValueError('TP update source path')
        source=(self.root/name).read_bytes()
        if hashlib.sha256(source).hexdigest()!=spec['sha256']:raise ValueError('TP update source identity')
        module=types.ModuleType('cineduplex_tp_update_'+spec['sha256'][:12])
        module.__package__='cineduplex.train'
        exec(compile(source,str(self.root/name),'exec'),module.__dict__)
        module.install_call_runtime(self.root)
        self.__class__=module.Coordinator
        self.install_key=None;self.qualification=None
        states=[None]*self.world
        dist.all_gather_object(states,{'rank':self.rank,'sha256':spec['sha256']},group=self.control)
        if len({s['sha256'] for s in states})!=1:raise ValueError('TP ranks disagree on runtime source')
        return states

    def update(self,spec):
        if not hasattr(self,'command_lock'):self.command_lock=threading.RLock()
        with self.command_lock:
            self.in_call=True
            try:
                self.command({'kind':'update','source':spec})
                return self.apply_update(spec)
            finally:self.in_call=False

    def install(self,model):
        snapshot=[(layer,layer.forward,{name:getattr(layer,name) for name in
            ('_cineduplex_tp_operator','_cineduplex_tp_group','_cineduplex_tp_index') if hasattr(layer,name)})
            for layer in model.modules()]
        old_key=getattr(self,'install_key',None);old_qualification=getattr(self,'qualification',None)
        try:
            record=self._install(model)
            profile_id=getattr(self,'profile_id',None)
            if profile_id is not None:
                if getattr(self,'rank_profile',None) is not None:self.rank_profile.close()
                path=self.root/'rank-window-profile-v2.py';source=path.read_bytes()
                profile_sha='d3ade5ed27d8f3fff7b70f5efb0b932b024ecfda813479c5d44aede783c07c3a'
                if hashlib.sha256(source).hexdigest()!=profile_sha:raise ValueError('rank profiler source identity')
                module=types.ModuleType('cineduplex_rank_window_profile')
                exec(compile(source,str(path),'exec'),module.__dict__)
                self.rank_profile=module.RankWindowProfile(model,self.root,profile_id,self.rank)
                self.profile_seq=0
                record={**record,'rank_profile_call_id':profile_id,'rank_profile_sha256':profile_sha}
            return record
        except ValueError:
            # A rejected numerical candidate must not discard the resident
            # model or leave half of its projections on the new layout.
            for layer,forward,attrs in snapshot:
                layer.forward=forward
                for name in ('_cineduplex_tp_operator','_cineduplex_tp_group','_cineduplex_tp_index'):
                    if hasattr(layer,name):delattr(layer,name)
                for name,value in attrs.items():setattr(layer,name,value)
            self.install_key=old_key;self.qualification=old_qualification
            raise

    def _install(self,model):
        global TP_GROUP
        install_call_runtime(self.root)
        # Update the native state-machine adapter on all admitted ranks,
        # including existing framework objects, without reloading weights.
        policy_path=self.root/'native-turn-policy-v6.py'
        policy_source=policy_path.read_bytes()
        policy_sha='e28a71299b6124f09d34d039da6e210c6dc291bc5ed14fcc48fc1192bb566ecc'
        if hashlib.sha256(policy_source).hexdigest()!=policy_sha:
            raise ValueError('native turn policy source identity')
        policy=types.ModuleType('cineduplex_native_turn_policy')
        exec(compile(policy_source,str(policy_path),'exec'),policy.__dict__)
        policy.install(native_prefix=False)
        sampling_sha='6926cdf5f2804de298dce215f0f30bb57db081b3122d9f2397a4433f89734178'
        if getattr(model,'_cineduplex_sampling_sha',None)!=sampling_sha:
            path=self.root/'independent-sampling-v2.py';source=path.read_bytes()
            if hashlib.sha256(source).hexdigest()!=sampling_sha:raise ValueError('sampling source identity')
            sampling_module=types.ModuleType('cineduplex_independent_sampling')
            exec(compile(source,str(path),'exec'),sampling_module.__dict__)
            model._tp_local_sample,model._cineduplex_sampling_identity=sampling_module.install(model)
            model._cineduplex_sampling_sha=sampling_sha
        stream_sha='7bc18ddd2df5d89f1f3c1d1759fb94117743d4208e0400b67c3cda0b4e900207'
        if getattr(model,'_cineduplex_stream_sha',None)!=stream_sha:
            path=self.root/'generation-stream-v4.py';source=path.read_bytes()
            if hashlib.sha256(source).hexdigest()!=stream_sha:raise ValueError('stream source identity')
            module=types.ModuleType('cineduplex_native_incremental_stream')
            exec(compile(source,str(path),'exec'),module.__dict__)
            module.install(model);model._cineduplex_stream_sha=stream_sha
        stage_sha='eec9a815faae4c0ea1aafbde861782496bf19478e9d22e22900e7eaa67636123'
        if self.rank==LEADER:
            path=self.root/'stage-meter-v3.py';source=path.read_bytes()
            if hashlib.sha256(source).hexdigest()!=stage_sha:raise ValueError('stage meter source identity')
            module=types.ModuleType('cineduplex_sparse_stage_meter')
            exec(compile(source,str(path),'exec'),module.__dict__)
            import cineduplex.train.realtime_service as runtime
            runtime.StageMeter=module.StageMeter
            runtime.Call.loop.__globals__["StageMeter"]=module.StageMeter
            previous_start=runtime.Application.start
            if not getattr(previous_start,'_nccl_command_metadata',False):
                def source_start(app,*args,**kwargs):
                    call=previous_start(app,*args,**kwargs)
                    app.identity['command_transport']=getattr(app.tensor_parallel,'command_transport','fixed-byte-gloo')
                    return call
                source_start._nccl_command_metadata=True
                runtime.Application.start=source_start
            # The resident server predates the generation-loop correction.
            # Refresh descriptive fields for each new call without restarting it.
            if not getattr(runtime.Application,'_native_metadata_patch',False):
                previous=runtime.Application.start
                def start(app,*args,**kwargs):
                    call=previous(app,*args,**kwargs)
                    app.identity['listening_channels']='native-sampled-merge-cache/pad-history'
                    app.identity['generation_loop']='native-channels-incremental'
                    app.identity['stage_profile_window_interval']=25
                    return call
                runtime.Application.start=start;runtime.Application._native_metadata_patch=True
        audio_sha='4649b1371cbf9d317fd8ccffd796b98f727cf1053bd52f939cc3378ae1b3a363'
        if getattr(model,'_cineduplex_audio_slice_sha',None)!=audio_sha:
            path=self.root/'audio-embedding-slice-v2.py';source=path.read_bytes()
            if hashlib.sha256(source).hexdigest()!=audio_sha:raise ValueError('audio slice source identity')
            audio_module=types.ModuleType('cineduplex_audio_slice')
            exec(compile(source,str(path),'exec'),audio_module.__dict__)
            self.audio_qualification=audio_module.install(model)
            model._cineduplex_audio_slice_sha=audio_sha
        if not hasattr(self,'priority_group'):
            options=dist.ProcessGroupNCCL.Options();options.is_high_priority_stream=True
            self.priority_group=dist.new_group(backend='nccl',timeout=datetime.timedelta(seconds=180),pg_options=options)
        TP_GROUP=self.priority_group
        if not getattr(self,'nccl_command_bytes_qualified',False):
            expected=torch.arange(16384,device='cuda',dtype=torch.int32).remainder(256).to(torch.uint8)
            probe=expected.clone() if self.rank==LEADER else torch.zeros_like(expected)
            dist.broadcast(probe,src=LEADER,group=TP_GROUP)
            correct=torch.tensor([int(torch.equal(probe,expected))],device='cuda')
            dist.all_reduce(correct,op=dist.ReduceOp.MIN,group=TP_GROUP)
            if not correct.item():raise ValueError('NCCL command byte identity differs')
            self.nccl_command_bytes_qualified=True
        if not hasattr(self,'model_stream'):self.model_stream=torch.cuda.Stream(priority=-1)
        self.model_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(self.model_stream)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        # A new call can select base or trained LoRA weights. Refresh the
        # admitted rank's shards after the existing inference merge.
        if self.kernels is None or getattr(self,'active_kernel_sha256',None)!=TP_KERNEL_SHA256:
            path=self.root/TP_KERNEL_NAME
            if hashlib.sha256(path.read_bytes()).hexdigest()!=TP_KERNEL_SHA256:
                raise ValueError('TP kernel source identity differs')
            spec=importlib.util.spec_from_file_location('cineduplex_tp_bf16',path)
            self.kernels=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.kernels)
            self.active_kernel_sha256=TP_KERNEL_SHA256
        # A local fused MLP would skip the sharded gate/up collectives. Keep
        # the native module path, while retaining independent RMS fusion.
        for layer in model.modules():
            if layer.__class__.__name__=='Qwen2MLP' and hasattr(layer,'_tp_mlp_original'):
                layer.forward=layer._tp_mlp_original
            if layer.__class__.__name__=='Qwen2MLP' and hasattr(layer,'_realtime_unfused_forward'):
                layer.forward=layer._realtime_unfused_forward
        candidates=[]
        for name,layer in model.named_modules():
            if not name.startswith(('model.','stoken_model.','control_model.','merge_model.','lm_head.original')):continue
            if layer.__class__.__name__=='LoRALinear':
                weight=layer._realtime_effective_weight;bias=layer.base.bias
            elif isinstance(layer,torch.nn.Linear):
                if '.base' in name:continue
                weight=layer.weight;bias=layer.bias
            else:continue
            leaf=name.rsplit('.',1)[-1]
            axis=1 if leaf in ('o_proj','down_proj') else 0
            candidates.append((name,layer,weight,bias,axis))
        install_key=tuple((name,id(weight),id(bias),axis) for name,_,weight,bias,axis in candidates)
        if getattr(self,'install_key',None)==install_key:
            # prepare_adapter restores LoRA forwards each call. Rebind the
            # already qualified projection without compiling/allocating again.
            for _,layer,_,_,_ in candidates:
                if hasattr(layer,'_cineduplex_tp_group'):
                    layer.forward=types.MethodType(lambda m,x:m._cineduplex_tp_group.select(m._cineduplex_tp_index,x),layer)
                else:
                    layer.forward=types.MethodType(lambda m,x:m._cineduplex_tp_operator(x),layer)
            return {**self.qualification,'reused':True}
        # Drop prior shards before constructing this call's groups. The
        # resident full BF16 tensors remain the source of truth.
        for _,layer,_,_,_ in candidates:
            for attr in ('_cineduplex_tp_operator','_cineduplex_tp_group'):
                if hasattr(layer,attr):delattr(layer,attr)
        # Four-way heads are aligned for every branch in this fixed model.
        # Preserve full weights and qualify complete native decoder layers,
        # not only individual local matrix outputs.
        for branch_name in ('model','stoken_model','control_model','merge_model'):
            branch=getattr(model,branch_name)
            cfg=branch.config
            if cfg.num_attention_heads%self.world or cfg.num_key_value_heads%self.world or cfg.intermediate_size%self.world:
                raise ValueError('fixed model heads/channels do not divide across TP ranks')
        for _,layer,weight,bias,_ in candidates:
            layer.forward=types.MethodType(lambda m,x,w=weight,b=bias:torch.nn.functional.linear(x,w,b),layer)
        decoder_checks=[]
        with torch.inference_mode():
            for branch_name in ('model','stoken_model','control_model','merge_model'):
                branch=getattr(model,branch_name);layer=branch.layers[0]
                for length in (1,3):
                    x=torch.sin(torch.arange(length*branch.config.hidden_size,device='cuda').float()*.017).bfloat16().reshape(1,length,-1)
                    positions=torch.arange(length,device='cuda')[None,:]
                    kwargs={'position_embeddings':branch.rotary_emb(x,positions),'attention_mask':None,'use_cache':False,
                        'cache_position':positions[0]}
                    ref=layer(x,**kwargs)
                    decoder_checks.append((branch_name,layer,x,kwargs,ref))
        errors=[];qualified=set();grouped_names=set();groups=[]
        by_name={name:(layer,weight,bias,axis) for name,layer,weight,bias,axis in candidates}
        for name,_,_,_,_ in candidates:
            parent,_,leaf=name.rpartition('.')
            leaves=('q_proj','k_proj','v_proj') if leaf=='q_proj' else ('gate_proj','up_proj') if leaf=='gate_proj' else None
            if not leaves:continue
            names=[parent+'.'+suffix for suffix in leaves]
            if not all(n in by_name for n in names):continue
            items=[by_name[n] for n in names]
            if len({tuple(w.shape)[1] for _,w,_,_ in items})!=1:continue
            groups.append((names,items));grouped_names.update(names)
        with torch.inference_mode():
            for names,items in groups:
                operator=PackedColumns([(w,b) for _,w,b,_ in items],self.kernels,True)
                key=('packed',operator.local,tuple(tuple(w.shape) for _,w,_,_ in items))
                if key not in qualified:
                    for length in (1,3):
                        x=torch.sin(torch.arange(length*operator.weight.shape[1],device='cuda').float()*.017).bfloat16().reshape(1,length,-1)
                        values=operator.compute(x)
                        for index,(_,weight,bias,_) in enumerate(items):
                            width=operator.widths[index]
                            if weight.shape[0]%self.world:raise ValueError('TP local column must divide evenly')
                            start=self.rank*width
                            original=torch.nn.functional.linear(x,weight,bias)
                            if operator.local:original=original[...,start:start+width]
                            new=values[index]
                            relative=float((new.float()-original.float()).norm()/original.float().norm().clamp_min(1e-6))
                            maximum=float((new.float()-original.float()).abs().max()/original.float().abs().max().clamp_min(1e-6))
                            ok=bool(torch.isfinite(new).all()) and relative<=.003 and maximum<=.008
                            consensus=torch.tensor([int(ok)],device='cuda');dist.all_reduce(consensus,op=dist.ReduceOp.MIN,group=TP_GROUP)
                            if not consensus.item():
                                failure={'key':str(key),'names':names,'index':index,'rank':self.rank,'tokens':length,
                                    'relative_error':relative,'normalized_max_error':maximum,'finite':bool(torch.isfinite(new).all())}
                                (self.root/('TP-QUALIFICATION-FAILED-rank'+str(self.rank)+'.json')).write_text(json.dumps(failure))
                                raise ValueError('packed TP numerical qualification failed: '+json.dumps(failure))
                            errors.append({'shape':list(weight.shape),'axis':0,'tokens':length,'packed_group_size':len(items),
                                'relative_error':relative,'normalized_max_error':maximum})
                    qualified.add(key)
                for index,(layer,_,_,_) in enumerate(items):
                    layer._cineduplex_tp_group=operator;layer._cineduplex_tp_index=index
                    layer.forward=types.MethodType(lambda m,x:m._cineduplex_tp_group.select(m._cineduplex_tp_index,x),layer)
            for name,layer,weight,bias,axis in candidates:
                if name in grouped_names:continue
                operator=ParallelLinear(weight,bias,axis,self.kernels)
                key=(tuple(weight.shape),axis,bias is not None)
                if key not in qualified:
                    # All ranks use exactly the same deterministic vectors.
                    for length in (1,3):
                        x=torch.sin(torch.arange(length*weight.shape[1],device='cuda').float()*.017).bfloat16().reshape(1,length,-1)
                        original=torch.nn.functional.linear(x,weight,bias)
                        new=operator(x)
                        relative=float((new.float()-original.float()).norm()/original.float().norm().clamp_min(1e-6))
                        maximum=float((new.float()-original.float()).abs().max()/original.float().abs().max().clamp_min(1e-6))
                        ok=bool(torch.isfinite(new).all()) and relative<=.003 and maximum<=.008
                        consensus=torch.tensor([int(ok)],device='cuda');dist.all_reduce(consensus,op=dist.ReduceOp.MIN,group=TP_GROUP)
                        if not consensus.item():raise ValueError('TP numerical qualification failed: '+str(key))
                        errors.append({'shape':list(weight.shape),'axis':axis,'tokens':length,'relative_error':relative,'normalized_max_error':maximum})
                    qualified.add(key)
                layer._cineduplex_tp_operator=operator
                layer.forward=types.MethodType(lambda m,x:m._cineduplex_tp_operator(x),layer)
            # Preserve native BF16 SiLU and multiply across repeated calls.
            # Experimental kernel v8 was associated with a garbled second turn.
            # Keep the admitted kernel v7 and native MLP baseline for now.
            fused_mlp_count=0
            decoder_errors=[]
            for name,layer,x,kwargs,ref in decoder_checks:
                actual=layer(x,**kwargs)
                relative=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-6))
                maximum=float((actual.float()-ref.float()).abs().max()/ref.float().abs().max().clamp_min(1e-6))
                ok=bool(torch.isfinite(actual).all()) and relative<=.006 and maximum<=.015
                consensus=torch.tensor([int(ok)],device='cuda');dist.all_reduce(consensus,op=dist.ReduceOp.MIN,group=TP_GROUP)
                if not consensus.item():
                    failure={'branch':name,'tokens':x.shape[1],'relative_error':relative,'normalized_max_error':maximum,
                        'rank':self.rank,'finite':bool(torch.isfinite(actual).all())}
                    (self.root/('TP-DECODER-FAILED-rank'+str(self.rank)+'.json')).write_text(json.dumps(failure))
                    raise ValueError('TP complete decoder qualification failed: '+json.dumps(failure))
                decoder_errors.append({'branch':name,'tokens':x.shape[1],'relative_error':relative,'normalized_max_error':maximum})
        # Main/control/speech/merge use the same token chosen by the leader.
        original=getattr(model,'_tp_local_sample',model._sample_from_logits)
        model._tp_local_sample=original
        def sample(m,*args,**kwargs):
            token=torch.empty((1,1),device='cuda',dtype=torch.long)
            if self.rank==LEADER:
                try:token=original(*args,**kwargs)
                except Exception as exc:
                    token.fill_(-1)
                    if not getattr(self,'sampling_fault_probe',False):
                        (self.root/'SAMPLING-ERROR.json').write_text(json.dumps({
                            'at':time.time(),'error':type(exc).__name__+': '+str(exc)[:400]}))
            sampling=kwargs.get('do_sample',args[6] if len(args)>6 else True)
            if sampling:
                # Speech and control are consumed together by the native
                # loop. Publish them in one collective after control sampling.
                self.pending_speech=token
            elif getattr(self,'pending_speech',None) is not None:
                pair=torch.cat((self.pending_speech,token),1)
                dist.broadcast(pair,src=LEADER,group=TP_GROUP)
                self.pending_speech.copy_(pair[:,:1]);token=pair[:,1:]
                self.pending_speech=None
                # The incremental stream validates this pair in its existing
                # triplet host transfer, before the next model step.
            else:
                dist.broadcast(token,src=LEADER,group=TP_GROUP)
                if token.item()<0:raise ValueError('TP sampling failed on leader')
            return token
        model._sample_from_logits=types.MethodType(sample,model)
        self.sampling_fault_probe=True
        try:
            # Exercise the deferred pair sentinel on all actual ranks before
            # admitting a call. No negative token can reach the next step.
            stream_path=self.root/'generation-stream-v4.py'
            stream_module=types.ModuleType('cineduplex_stream_sentinel_probe')
            exec(compile(stream_path.read_bytes(),str(stream_path),'exec'),stream_module.__dict__)
            probe_text=torch.tensor([[2]],device='cuda')
            probe_speech=model._sample_from_logits(torch.zeros((1,4),device='cuda'),None,
                lambda ids,scores:None,0.,None,None,True)
            probe_control=model._sample_from_logits(torch.tensor([[0.,1.,3.,2.]],device='cuda'),
                None,None,0.,None,None,False)
            try:stream_module.validated_triplet(probe_text,probe_speech,probe_control)
            except ValueError as exc:
                if str(exc)!='TP sampling failed on leader':raise
            else:raise ValueError('deferred pair sampling error was not synchronized')
            probe_speech=model._sample_from_logits(torch.tensor([[0.,1.,3.,2.]],device='cuda'),
                None,None,0.,None,None,True)
            probe_control=model._sample_from_logits(torch.tensor([[0.,1.,3.,2.]],device='cuda'),
                None,None,0.,None,None,False)
            if stream_module.validated_triplet(probe_text,probe_speech,probe_control)!=[2,2,2]:
                raise ValueError('deferred pair sampling did not recover')
            try:
                model._sample_from_logits(torch.zeros((1,4),device='cuda'),None,
                    lambda ids,scores:None,0.,None,None,False)
            except ValueError as exc:
                if str(exc)!='TP sampling failed on leader':raise
            else:raise ValueError('TP sampling error was not synchronized')
            resumed=model._sample_from_logits(torch.tensor([[0.,1.,3.,2.]],device='cuda'),
                None,None,0.,None,None,False)
            if resumed.item()!=2:raise ValueError('TP sampling did not recover after synchronized error')
        finally:self.sampling_fault_probe=False
        record={'command_transport':'fixed-byte-nccl-active/gloo-idle',
                'nccl_command_bytes_qualification':'passed',
                'precision':'bf16','quantization':'none','world_size':self.world,'leader_rank':LEADER,
                'sampling_error_broadcast_and_recovery':'passed',
                'deferred_pair_sentinel_and_recovery':'passed',
                'native_turn_policy_sha256':policy_sha,
                'speech_sampling':model._cineduplex_sampling_identity,
                'speech_sampling_sha256':sampling_sha,'generation_stream_sha256':stream_sha,
                'generation_loop':model._cineduplex_generation_loop,'stage_meter_sha256':stage_sha,
                'stage_profile_window_interval':25,'main_control_gpu_window_interval':1,
                'audio_slice_sha256':audio_sha,'audio_slice_qualification':self.audio_qualification,
                'gemv_tiling_benchmarks':self.kernels.benchmarks,
                'fused_bf16_mlp_count':fused_mlp_count,
                'gated_mlp_benchmarks':self.kernels.gated_benchmarks,
                'model_stream_priority':self.model_stream.priority,'collective_priority':'high',
                'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                'layers':len(candidates),'packed_groups':len(groups),'kernel_sha256':self.active_kernel_sha256,
                'qualified_shapes':errors,'decoder_checks':decoder_errors,'kv_layout':'all-branches-local',
                'intermediate_layout':'all-branches-local','prefill_projection':'native-full-bf16','at':time.time()}
        self.install_key=install_key;self.qualification=record
        self.keep_idle_channel_alive()
        return record

    @property
    def world(self):return dist.get_world_size()

    def start(self,voice,language,scale,profile_id=None):
        if not hasattr(self,'command_lock'):self.command_lock=threading.RLock()
        with self.command_lock:
            self.in_call=True
            self.profile_id=profile_id
            self.command({'kind':'start','voice':voice,'language':language,'scale':scale,'profile_id':profile_id})
    def audio(self,pcm):
        profile=getattr(self,'rank_profile',None)
        if profile is not None:profile.begin(self.profile_seq)
        self.command({'kind':'audio','pcm':pcm,'profile_seq':getattr(self,'profile_seq',0)})
        self.profile_seq=getattr(self,'profile_seq',0)+1
    def finish_window(self):
        if getattr(self,'rank_profile',None) is not None:self.rank_profile.finish()
    def stop(self):
        with self.command_lock:
            try:
                self.finish_window()
                self.command({'kind':'stop'})
                if getattr(self,'rank_profile',None) is not None:
                    self.rank_profile.close();self.rank_profile=None
            except Exception as exc:
                # A timed-out collective cannot be reused. Make the public
                # readiness false before clearing the call, then let this
                # task's controller reconcile its original workers.
                marker=self.root/'STOP.json'
                if not marker.exists():
                    marker.write_text(json.dumps({'reason':'tp_stop_failed',
                        'error':type(exc).__name__+': '+str(exc)[:300],'at':time.time()}))
                raise
            self.in_call=False

    def peer_loop(self,model,fw):
        from .realtime_service import prepare_adapter,optimize_bf16,conversation_instruction
        from lychee_fd.runtime import vllm_generation
        while True:
            item=self.command()
            if item['kind']=='shutdown':return
            if item['kind']=='update':
                self.apply_update(item['source']);continue
            if item['kind']!='start':raise ValueError('peer expected start')
            self.profile_id=item.get('profile_id')
            prepare_adapter(model,item['scale']);optimize_bf16(model,self.root)
            try:record=self.install(model)
            except ValueError:
                item=self.command()
                if item['kind']=='shutdown':return
                if item['kind']!='stop':raise ValueError('rejected TP qualification expected call stop')
                continue
            (self.root/('TP-QUALIFICATION-rank'+str(self.rank)+'.json')).write_text(json.dumps(record))
            fw.last_ss_pos=None;fw.model._merge_cache_by_past_id.clear()
            vllm_generation.SYSTEM_MESSAGE_PREFIX='<|BOT|>system\n'+conversation_instruction(item['language'])+'<|EOT|>'
            session=fw.create_incremental_stream_session(start_speak_token_factor=1.0,
                start_listen_token_factor=1.,audio_incremental_mode=True)
            try:
                with torch.inference_mode():
                    while True:
                        item=self.command()
                        if item['kind']=='stop':break
                        if item['kind']!='audio':raise ValueError('peer expected audio')
                        audio=np.frombuffer(item['pcm'],dtype='<i2').astype(np.float32)/32768.
                        try:
                            for event in session.advance_stream(audio,audio_is_incremental=True,emit_generation_complete=False):
                                if event['type']=='speaking_token':fw.runtime_speech_steps+=1
                        except ValueError as exc:
                            if str(exc)!='TP sampling failed on leader':raise
                            self.finish_window()
                            # Every rank consumed the same error sentinel;
                            # control collectives can now finish the call.
                            item=self.command()
                            if item['kind']!='stop':raise ValueError('failed sampling expected call stop')
                            break
                        self.finish_window()
            finally:
                self.finish_window()
                if getattr(self,'rank_profile',None) is not None:
                    self.rank_profile.close();self.rank_profile=None
                session.close();fw.model._merge_cache_by_past_id.clear();torch.cuda.empty_cache()


def run(config):
    from .memory_runtime import Store,load_model
    from .four_worker import attach,load_adapter,adapter_state,state_hash
    from .realtime_service import bind_incremental_model,framework,StreamingCodec,Application,serve,atomic,prepare_adapter,optimize_bf16
    from transformers import AutoTokenizer
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(config['seed'])
    store=Store(config['url'],config['token']);root=store.file_path(config['realtime_prefix']);root.mkdir(exist_ok=True)
    control=initialize();rank=dist.get_rank()
    try:
        model=load_model(store,config['model'],control,cancelled=lambda:(root/'STOP.json').exists())
        receipt=json.loads(store.get(config['checkpoint']+'.json'))
        saved=torch.load(io.BytesIO(store.get(config['checkpoint'],receipt['sha256'])),map_location='cpu',weights_only=True)
        if saved['identity']['model']!=config['model'] or saved['step']!=config['stop_step']:raise ValueError('TP checkpoint identity')
        if state_hash(saved['adapter'])!=saved['metric']['adapter_sha256']:raise ValueError('TP adapter identity')
        if attach(model)!=saved['modules']:raise ValueError('TP adapter structure')
        load_adapter(model,saved['adapter']);del saved
        model.eval();bind_incremental_model(model)
        tokenizer=AutoTokenizer.from_pretrained(str(store.file_path('raw/lychee-tokenizer')),local_files_only=True,use_fast=False)
        fw=framework(model,tokenizer)
        serving_control=dist.new_group(backend='gloo',timeout=datetime.timedelta(seconds=60))
        tp=Coordinator(serving_control,root,config['tp_kernel_sha256'])
        prepare_adapter(model,.25);optimize_bf16(model,root)
        # Boot the already tested projection layout. Experimental layouts
        # are updated only after this resident model is available, so a
        # numerical rejection can retain its weights and accept a correction.
        fallback=config['tp_runtime_source']
        tp.apply_update(fallback)
        # Bind the boot loop after the pinned runtime has replaced the class.
        # A pre-update bound method retains the experimental implementation.
        tp.peer_loop=types.MethodType(type(tp).peer_loop,tp)
        qualification=tp.install(model)
        atomic(root/('TP-QUALIFICATION-rank'+str(rank)+'.json'),qualification)
        if rank!=LEADER:return tp.peer_loop(model,fw)
        identity={'model':f'training-20261009-v1:step-{config["stop_step"]}',
            'adapter_sha256':state_hash(adapter_state(model)),'checkpoint_sha256':receipt['sha256'],
            'control_mode':'native-full-duplex','input_sample_rate':16000,'output_sample_rate':24000,
            'chunk_seconds':fw.window_second,'source_sha256':config['source_modules_sha256'],
            'tensor_parallel_size':4,'decode_precision':'bf16','quantization':'none'}
        identity['tp_runtime_sha256']=fallback['sha256']
        app=Application(model,StreamingCodec(store),fw,root,identity);app.tensor_parallel=tp
        serve(app,config['realtime_port']);tp.command({'kind':'shutdown'})
    except BaseException as exc:
        # Once a rank fails, shutdown of a poisoned NCCL group can itself
        # wait indefinitely. This worker owns only its candidate process;
        # exit it so the original controller can reconcile peers/mounts.
        import traceback
        traceback.print_exc()
        try:atomic(root/('WORKER-FAILED-rank'+str(rank)+'.json'),{
            'at':time.time(),'error':type(exc).__name__+': '+str(exc)[:500]})
        finally:os._exit(1)
    finally:
        dist.destroy_process_group()
