"""Append-only BF16 KV storage, preserving the complete native history.

Attention receives only the used prefix, never the reserved capacity. Storage
expands in blocks rather than copying the complete prefix on every decode step.
"""
import torch
from transformers.cache_utils import Cache, DynamicLayer

class AppendLayer(DynamicLayer):
    block_size=2048
    def __init__(self):
        super().__init__()
        self.length=0;self.capacity=0;self.key_storage=None;self.value_storage=None

    def update(self,key_states,value_states,cache_kwargs=None):
        if key_states.shape!=value_states.shape:raise ValueError('KV shape differs')
        # This runtime admits a single stream. Do not silently reinterpret a
        # beam/batch mutation or change dtype, device, head count, or position.
        shape=tuple(key_states.shape)
        if len(shape)!=4 or shape[0]!=1:raise ValueError('single-stream cache required')
        if self.is_initialized and (shape[:2]+shape[3:]!=self.layout or key_states.dtype!=self.dtype or key_states.device!=self.device):
            raise ValueError('KV layout changed')
        added=shape[-2];needed=self.length+added
        if needed>self.capacity:
            capacity=((needed+self.block_size-1)//self.block_size)*self.block_size
            allocation=(*shape[:-2],capacity,shape[-1])
            keys=key_states.new_empty(allocation);values=value_states.new_empty(allocation)
            if self.length:
                keys[...,:self.length,:].copy_(self.keys)
                values[...,:self.length,:].copy_(self.values)
            self.key_storage,self.value_storage=keys,values;self.capacity=capacity
        self.key_storage[...,self.length:needed,:].copy_(key_states)
        self.value_storage[...,self.length:needed,:].copy_(value_states)
        self.length=needed;self.keys=self.key_storage[...,:needed,:];self.values=self.value_storage[...,:needed,:]
        self.dtype,self.device=key_states.dtype,key_states.device
        self.layout=shape[:2]+shape[3:];self.is_initialized=True
        return self.keys,self.values

    def crop(self,max_length):
        if max_length<0:max_length=max(0,self.length-abs(max_length))
        self.length=min(self.length,max_length)
        if self.is_initialized:
            self.keys=self.key_storage[...,:self.length,:];self.values=self.value_storage[...,:self.length,:]

    def reset(self):
        self.crop(0)

    def batch_repeat_interleave(self,repeats):
        if repeats!=1:raise ValueError('beam expansion unsupported by single-stream runtime')

    def batch_select_indices(self,indices):
        if indices.numel()!=1 or int(indices[0])!=0:raise ValueError('batch selection unsupported')

    def reorder_cache(self,beam_idx):
        self.batch_select_indices(beam_idx)

class AppendCache(Cache):
    def __init__(self):
        super().__init__(layer_class_to_replicate=AppendLayer)

    def to_legacy_cache(self):
        return tuple((layer.keys,layer.values) for layer in self.layers)


def qualify():
    """Bounded GPU equality and timing on the already admitted process."""
    from transformers.cache_utils import DynamicCache
    checks=[];original=DynamicCache();candidate=AppendCache()
    with torch.inference_mode():
        for size in (67,1,2048,1):
            k=torch.sin(torch.arange(size*128,device='cuda').float()*.017).bfloat16().reshape(1,1,size,128)
            v=k*.5
            x=original.update(k,v,0);y=candidate.update(k,v,0)
            q=k[...,-1:,:].expand(-1,7,-1,-1)
            left=torch.nn.functional.scaled_dot_product_attention(q,x[0].expand(-1,7,-1,-1),x[1].expand(-1,7,-1,-1))
            right=torch.nn.functional.scaled_dot_product_attention(q,y[0].expand(-1,7,-1,-1),y[1].expand(-1,7,-1,-1))
            if not all(torch.equal(i,j) for i,j in zip(x,y)) or not torch.equal(left,right):
                raise ValueError('append cache GPU equality failed')
            checks.append({'length':candidate.get_seq_length(),'kv_equal':True,'attention_equal':True})
        for crop in (-7,100,0):
            original.crop(crop);candidate.crop(crop)
            if not torch.equal(original.layers[0].keys,candidate.layers[0].keys):raise ValueError('cache crop differs')
        timing={}
        for name,kind in (('dynamic',DynamicCache),('append',AppendCache)):
            cache=kind();cache.update(k,k,0)
            # Warm kernels before measurement. At 6K context, each update
            # adds one token; attention and semantic processing are excluded.
            prefix=k.repeat(1,1,6144,1)[...,:6144,:]
            cache=kind();cache.update(prefix,prefix,0);token=k[...,-1:,:]
            for _ in range(10):cache.update(token,token,0)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(64):cache.update(token,token,0)
            end.record();end.synchronize();timing[name]=begin.elapsed_time(end)/64
    return {'device':'cuda','bf16':True,'checks':checks,'crop_equal':True,'update_ms_at_6k':timing,'full_call_speed_pass':False}
