"""Bounded acoustic-only A/B inside an exclusively owned diagnostic call.

Fresh native modules share immutable resident BF16 base tensors. Optimizers own
only independent adapter tensors. Morning recordings are held out entirely.
No text-assisted inference, device capture/playback or public audio release.
"""
import ast
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import random
import time
import types

import numpy as np
import torch
import torch.nn.functional as F


def run(call, spec):
    from accelerate import init_empty_weights
    from cineduplex.train.memory_runtime import Store
    from cineduplex.train.four_worker import FeatureReader, attach, load_adapter, adapter_state, state_hash, serialize
    resident = call.app.model
    root = Path(call.app.root).parents[1]
    store = Store('file://' + str(root), '')
    prefix = str(call.folder.relative_to(root)) + '/acoustic-ab'
    begun = time.monotonic()
    record = dict(schema=1, call=call.id, kind='native-window-acoustic-ab',
                  precision='bf16', quantization='none', microphone=False, speaker=False,
                  production_weights_changed=False, quality_accepted=False,
                  inference_transcript_injection=False, arms=[], errors=[])
    original_versions = {n:p._version for n,p in resident.named_parameters()}
    original_adapter = {n:p.detach().cpu().clone() for n,p in resident.named_parameters() if '.lora_' in n}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
    threads = torch.get_num_threads()
    shadow = None

    def budget():
        if call.closed.is_set() or time.monotonic()-begun > 900:
            raise RuntimeError('bounded acoustic comparison canceled/timed out')

    def pinned(key):
        item=spec[key];return store.get(item['path'],item['sha256'])

    def progress(phase, **details):
        call.emit(dict(type='diagnostic_progress',phase=phase,**details))
        store.json(prefix+'/progress-'+str(time.time_ns())+'.json',dict(phase=phase,elapsed=time.monotonic()-begun,**details))

    try:
        torch.set_num_threads(4)
        saved=torch.load(io.BytesIO(pinned('checkpoint')),map_location='cpu',weights_only=True)
        if state_hash(original_adapter)!=state_hash(saved['adapter']):raise ValueError('resident/checkpoint adapter differs')
        record['initial_adapter_sha256']=state_hash(saved['adapter'])
        record['checkpoint']=spec['checkpoint']
        record['steps_per_arm']=8;record['learning_rate']=1e-4;record['seed']=20261014
        record['training_scale']=.25;record['training_forward']='differentiable BF16 merged weight; same as serving scale'
        record['held_out_recordings']=spec['cases']
        inputs=json.loads(pinned('small_inputs'))
        receipts={split:store.put(prefix+'/inputs-'+split+'.jsonl',
            ('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode()) for split,rows in inputs.items()}
        if [len(inputs[s]) for s in ('train','eval')]!=[2,1]:raise ValueError('bounded fixed selection')
        contract=types.ModuleType('ab_contract');exec(compile(pinned('audio_contract'),'<contract>','exec'),contract.__dict__)
        tree=ast.parse(pinned('exporter'))
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in
            ('cineduplex.train.memory_runtime','cineduplex.train.native_audio_contract'))]
        exported={}
        for label,window in [('legacy24',24.),('native04',.4)]:
            budget()
            module=types.ModuleType('ab_export_'+label);module.Store=Store
            def selected_contract(config,window=window):
                value=contract.native_audio_contract(config)
                if window==24.:value={**value,'window_second':24.,'window_samples':384000,'diagnostic_legacy_only':True}
                return value
            module.native_audio_contract=selected_contract
            exec(compile(tree,'<pinned-exporter>','exec'),module.__dict__)
            arm_prefix=prefix+'/'+label
            module.run(dict(url='file://'+str(root),seed=20261014,loader_sha256=spec['dataset']['sha256'],
                input_files=receipts,existing_tokenizer=str(root/'raw/lychee-tokenizer'),run_prefix=arm_prefix))
            exported[label]={s:json.loads(store.get(arm_prefix+'/'+s+'-features.json'))['rows'] for s in ('train','eval')}
        reader=FeatureReader(store)
        # Prove the intervention leaves every target and non-audio input fixed.
        record['paired_batches']=[]
        for split in ('train','eval'):
            for i in range(len(inputs[split])):
                a,b=(reader.batch(exported[label][split][i]) for label in ('legacy24','native04'))
                equal={k:bool(torch.equal(a[k],b[k])) for k in a if k not in ('wavs','wav_lens')}
                if not all(equal.values()):raise ValueError('A/B changes non-audio fields: '+str(equal))
                record['paired_batches'].append(dict(split=split,scene_id=inputs[split][i]['scene_id'],
                    equal=equal,legacy_windows=len(a['wav_lens']),native_windows=len(b['wav_lens'])))
        budget();progress('construct-independent-native-training-modules')
        # Same native class/config, no inference monkey patches or TP collectives.
        cfg=copy.deepcopy(resident.config)
        old_dtype=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
        try:
            with init_empty_weights(include_buffers=False):shadow=type(resident)(cfg)
        finally:torch.set_default_dtype(old_dtype)
        base={n.replace('.base.','.'):t.detach() for n,t in resident.state_dict().items()
              if '.lora_' not in n and not n.startswith('lm_head.')}
        # Serving wraps the head with selected-row buffers; training needs the
        # original full projection, not those runtime-only derived buffers.
        base['lm_head.weight']=resident.lm_head.weight.detach()
        shadow.load_state_dict(base,strict=True,assign=True)
        shadow.requires_grad_(False)
        # Empty-weight construction leaves only small nonpersistent buffers on CPU.
        for module in shadow.modules():
            for name,value in list(module._buffers.items()):
                if value is not None:module._buffers[name]=value.to(resident.device)
        if any(p.is_meta or p.dtype!=torch.bfloat16 for p in shadow.parameters()):raise ValueError('native BF16 base contract')
        if any(p.data_ptr()!=base[n].data_ptr() for n,p in shadow.named_parameters()):raise ValueError('base sharing differs')
        del base
        modules=attach(shadow)
        shadow.only_training_stoken_and_control=True;shadow.stream_generation_flag=False;shadow.eval()
        # Preserve the serving rounding point, with gradients only into new LoRA.
        for layer in shadow.modules():
            if layer.__class__.__name__=='LoRALinear':
                def differentiable_merged(m,x):
                    weight=(m.base.weight.float()+.25*(m.lora_b@m.lora_a)).to(torch.bfloat16)
                    return F.linear(x,weight,m.base.bias)
                layer.forward=types.MethodType(differentiable_merged,layer)
        params=[p for p in shadow.parameters() if p.requires_grad]
        if len(params)!=32 or any(p.data_ptr()==q.data_ptr() for p in params for q in resident.parameters()):
            raise ValueError('optimizer must own exactly 32 independent acoustic adapters')
        def batch(label,split,index):
            return {k:v.to(resident.device) for k,v in reader.batch(exported[label][split][index]).items()}
        def evaluate(label):
            with torch.no_grad():
                output=shadow(**batch(label,'eval',0),use_cache=False)
                loss=float(output.loss_stoken)
                if not np.isfinite(loss):raise ValueError('nonfinite held-out loss')
                return loss
        for label in ('legacy24','native04'):
            budget();torch.manual_seed(20261014);load_adapter(shadow,saved['adapter'])
            optim=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
            arm=dict(name=label,steps=[],before={key:evaluate(key) for key in exported})
            record['arms'].append(arm);progress('training',arm=label,before=arm['before'])
            for step in range(8):
                budget();optim.zero_grad(set_to_none=True);t=time.monotonic()
                output=shadow(**batch(label,'train',step%2),use_cache=False)
                loss=output.loss_stoken
                if not torch.isfinite(loss):raise ValueError('nonfinite acoustic loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm) or float(norm)<=0:raise ValueError('nonfinite/zero acoustic gradient')
                value=float(loss);optim.step();del output,loss
                arm['steps'].append(dict(step=step+1,loss=value,gradient_norm=float(norm),seconds=time.monotonic()-t))
                progress('training-step',arm=label,**arm['steps'][-1])
            arm['after']={key:evaluate(key) for key in exported}
            state=adapter_state(shadow);arm['adapter_sha256']=state_hash(state)
            if arm['adapter_sha256']==record['initial_adapter_sha256']:raise ValueError('adapter did not change')
            arm['checkpoint']=store.put(prefix+'/'+label+'/checkpoint-8.pt',serialize(dict(
                format='cineduplex-acoustic-window-comparison-v1',adapter=state,modules=modules,
                steps=8,initial=spec['checkpoint'],scale=.25,seed=20261014,learning_rate=1e-4,
                feature_rows=exported[label],metric=arm,quality_accepted=False)))
            progress('arm-complete',arm=label,after=arm['after'],checkpoint=arm['checkpoint'])
            del optim,state;gc.collect()
        record['passed_training_mechanics']=True
        record['native_eval_delta_vs_legacy']=record['arms'][1]['after']['native04']-record['arms'][0]['after']['native04']
    except Exception as exc:
        record['errors'].append(type(exc).__name__+': '+str(exc)[:600]);raise
    finally:
        record['resident_parameter_versions_unchanged']=original_versions=={n:p._version for n,p in resident.named_parameters()}
        current={n:p.detach().cpu().clone() for n,p in resident.named_parameters() if '.lora_' in n}
        record['resident_adapter_unchanged']=state_hash(current)==state_hash(original_adapter)
        record['elapsed_seconds']=time.monotonic()-begun
        store.json(prefix+'/RESULT.json',record)
        del shadow;gc.collect()
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state(rng[3])
        torch.set_num_threads(threads)
        if not record['resident_parameter_versions_unchanged'] or not record['resident_adapter_unchanged']:
            raise RuntimeError('resident immutability failed')
