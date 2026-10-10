"""NAS-backed, RAM-only worker utilities. No downloaded code is written on DGX."""
import hashlib
import importlib.abc
import importlib.util
import io
import json
import linecache
import sys
import urllib.request
import os
from pathlib import Path
import secrets


class Store:
    def __init__(self, url, token):self.url,self.token=url.rstrip('/'),token
    def file_path(self,key):
        if key=='loader-source':key='cache/train-code/loader-source.py'
        root=Path(self.url.removeprefix('file://'))
        if not root.is_mount():raise ValueError('NAS mount lost; no local fallback')
        path=root/key
        if Path(key).is_absolute() or '..' in Path(key).parts or not path.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe NAS path')
        return path
    def get(self,key,sha=None):
        if self.url.startswith('file://'):data=self.file_path(key).read_bytes()
        else:
            req=urllib.request.Request(self.url+'/'+key,headers={'Authorization':'Bearer '+self.token})
            with urllib.request.urlopen(req,timeout=900) as r:data=r.read()
        if sha and hashlib.sha256(data).hexdigest()!=sha:raise ValueError('remote checksum mismatch: '+key)
        return data
    def put(self,key,data):
        if self.url.startswith('file://'):
            path=self.file_path(key);path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_name(path.name+'.pending-'+secrets.token_hex(8))
            try:
                with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                os.link(tmp,path);tmp.unlink()
            finally:tmp.unlink(missing_ok=True)
            return {'path':key,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        req=urllib.request.Request(self.url+'/'+key,data=data,method='PUT',headers={'Authorization':'Bearer '+self.token})
        with urllib.request.urlopen(req,timeout=900) as r:receipt=json.load(r)
        if receipt['sha256']!=hashlib.sha256(data).hexdigest():raise ValueError('write receipt mismatch')
        return receipt
    def json(self,key,value):return self.put(key,json.dumps(value,indent=2,ensure_ascii=False).encode())


class MemoryImporter(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def __init__(self,modules):self.modules=modules
    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_loader(fullname,self,is_package=self.modules[fullname]['package'])
    def create_module(self,spec):return None
    def get_source(self,fullname):return self.modules[fullname]['source']
    def exec_module(self,module):
        item=self.modules[module.__name__]
        module.__file__='nas-memory://'+module.__name__.replace('.','/')+'.py'
        linecache.cache[module.__file__]=(len(item['source']),None,item['source'].splitlines(True),module.__file__)
        if item['package']:module.__path__=[]
        exec(compile(item['source'],module.__file__,'exec'),module.__dict__)


def audit(config):
    import torch
    from accelerate import init_empty_weights
    from Models.LycheeFD import LycheeFD,LycheeFDConfig
    store=Store(config['url'],config['token'])
    cfg=json.loads(store.get('raw/lychee-full-duplex/config.json'))
    cfg.pop('auto_map',None);cfg['model_type']='lychee_full_duplex';cfg['architectures']=['LycheeFD']
    cfg=LycheeFDConfig(**cfg)
    cfg._attn_implementation='sdpa'
    for k in ('text_config','stoken_layer_config','control_layer_config','merge_layer_config'):getattr(cfg,k)._attn_implementation='sdpa'
    old=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with init_empty_weights(include_buffers=False):model=LycheeFD(cfg)
    finally:torch.set_default_dtype(old)
    index=json.loads((Path(config['existing_root'])/'model.safetensors.index.json').read_text())
    state=model.state_dict()
    if set(state)!=set(index['weight_map']):raise ValueError('RAM imported class/index keys differ')
    count=sum(p.numel() for p in model.parameters())
    if count!=12812043520 or len(state)!=982:raise ValueError('wrong model structure')
    result={'ok':True,'parameters':count,'state_tensors':len(state),'memory_source_import':True,
            'gpu_loaded':False,'training_tested':False,'weight_values_verified':False,
            'source_modules_sha256':config['source_modules_sha256']}
    store.json(config['run_prefix']+'/memory-source-audit.json',result);print(json.dumps(result),flush=True)


def load_model(store, manifest, control_group, cancelled=None):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load
    import torch.distributed as dist
    from Models.LycheeFD import LycheeFD,LycheeFDConfig
    published=json.loads(store.get('raw/lychee-full-duplex/config.json',manifest['config_sha256']))
    if published['model_type']!='step_audio_2_full_duplex':raise ValueError('architecture mismatch')
    published.pop('auto_map',None);published['model_type']='lychee_full_duplex';published['architectures']=['LycheeFD']
    cfg=LycheeFDConfig(**published);cfg._attn_implementation='sdpa'
    for k in ('text_config','stoken_layer_config','control_layer_config','merge_layer_config'):
        getattr(cfg,k)._attn_implementation='sdpa'
    old_dtype=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with init_empty_weights(include_buffers=False):model=LycheeFD(cfg)
    finally:torch.set_default_dtype(old_dtype)
    expected=model.state_dict();seen=set()
    index=json.loads(store.get('raw/lychee-full-duplex/model.safetensors.index.json',manifest['index_sha256']))
    if set(expected)!=set(index['weight_map']):raise ValueError('model/index key mismatch')
    leader=3
    distributed=control_group is not None
    rank=dist.get_rank() if distributed else leader
    for entry in manifest['shards']:
        if cancelled is not None and cancelled():raise RuntimeError('model load stopped by task controller')
        print('loading verified shard',entry['path'],flush=True)
        if rank==leader:
            # Reuse old, unchanged local files when present. New assets are NAS-only.
            old=Path.home()/'cineduplex-work/models/lychee_full_duplex'/entry['path']
            if old.is_file():
                raw=old.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('old shard hash mismatch')
            else:raw=store.get('raw/lychee-full-duplex/'+entry['path'],entry['sha256'])
            source_state=load(raw);del raw
            if set(source_state)!={k for k,v in index['weight_map'].items() if v==entry['path']}:raise ValueError('shard key mismatch')
        # A NAS read may outlast Mesh's receive-completion timeout. Wait on
        # the CPU control group before posting any GPU receive for this shard.
        if distributed:dist.barrier(group=control_group)
        metadata=[{k:(list(v.shape),str(v.dtype).removeprefix('torch.')) for k,v in source_state.items()} if rank==leader else None]
        if distributed:dist.broadcast_object_list(metadata,src=leader,group=control_group)
        spec=metadata[0]
        state={}
        for name in sorted(k for k,v in index['weight_map'].items() if v==entry['path']):
            shape,dtype=spec[name]
            if shape!=list(expected[name].shape) or dtype not in ('bfloat16','float16','float32','int64','bool'):
                raise ValueError('tensor metadata mismatch')
            if rank==leader:value=source_state.pop(name).to('cuda')
            else:value=torch.empty(shape,device='cuda',dtype=getattr(torch,dtype))
            if name in seen or name not in expected or index['weight_map'][name]!=entry['path']:
                raise ValueError('duplicate or wrong shard tensor: '+name)
            if value.shape!=expected[name].shape:raise ValueError('source tensor shape changed')
            if distributed:dist.broadcast(value,src=leader)
            if not torch.isfinite(value).all():
                raise ValueError('shape or finite check failed: '+name)
            seen.add(name);state[name]=value
        model.load_state_dict(state,strict=False,assign=True);del state
    if seen!=set(expected):raise ValueError('missing model tensors')
    del expected
    model.to('cuda');model.requires_grad_(False);model.eval()
    if any(p.is_meta for p in model.parameters()):raise ValueError('unloaded parameter')
    return model
