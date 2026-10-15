"""All-rank exact adapter/merged/packed-weight audit; no model forward."""
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

import torch


def install(model, root, profile_id, rank, spec):
    """Called after ordinary per-call TP install on each admitted rank.

    Keep failures as evidence, not exceptions into the collective protocol.
    No parameters or inference functions are modified by the audit.
    """
    if profile_id is None:return
    profile_id=str(uuid.UUID(profile_id))
    destination=Path(root)/'calls'/profile_id/('acoustic-weights-rank'+str(rank)+'.json')
    if destination.exists():return
    begun=time.monotonic()
    out=dict(schema=1,rank=rank,call=profile_id,rows=[],errors=[],
             weight_mutation=False,generation=False,quality_accepted=False)
    try:
        nas=Path(root).parents[1]
        cp=spec['checkpoint'];path=nas/cp['path']
        if path.stat().st_size!=cp['bytes'] or cp['bytes']>12000000:raise ValueError('checkpoint size')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cp['sha256']:raise ValueError('checkpoint SHA')
        saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        state=saved['adapter']
        actual={n:p for n,p in model.named_parameters() if n in state}
        expected_keys={n+'.'+suffix for n,m in model.named_modules()
                       if m.__class__.__name__=='LoRALinear' for suffix in ('lora_a','lora_b')}
        if set(state)!=expected_keys or set(actual)!=set(state):raise ValueError('adapter names differ')
        def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        h=hashlib.sha256()
        for name,value in sorted(state.items()):
            h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
        if h.hexdigest()!=saved['metric']['adapter_sha256']:raise ValueError('internal adapter hash')
        out.update(checkpoint=cp,adapter_sha256=h.hexdigest(),step=saved['step'],
                   adapter_values=sum(v.numel() for v in state.values()),adapter_tensors=len(state))
        with torch.inference_mode():
            for name in saved['modules']:
                if time.monotonic()-begun>90:raise ValueError('audit time budget')
                layer=model.get_submodule(name)
                a,b=state[name+'.lora_a'],state[name+'.lora_b']
                matches=bool(torch.equal(a,layer.lora_a.detach().cpu()) and torch.equal(b,layer.lora_b.detach().cpu()))
                scale=layer.realtime_scale
                if scale!=.25:raise ValueError('expected original quarter-scale adapter')
                expected=(layer.base.weight.float()+scale*(b.to(layer.base.weight.device)@a.to(layer.base.weight.device))).bfloat16()
                effective=layer._realtime_effective_weight
                op=layer._cineduplex_tp_group;index=layer._cineduplex_tp_index
                width=op.widths[index];offset=sum(op.widths[:index])
                packed=op.weight[offset:offset+width]
                expected_local=expected[rank*width:(rank+1)*width]
                row=dict(name=name,scale=scale,shape=list(expected.shape),
                         adapter_exact=matches,merged_exact=bool(torch.equal(expected,effective)),
                         packed_exact=bool(torch.equal(packed,expected_local)),
                         prefill_spec_exact=bool(torch.equal(op.specs[index][0],expected)),
                         dtype=str(effective.dtype),finite=bool(torch.isfinite(effective).all()),
                         effective_sha256=digest(effective),packed_sha256=digest(packed),
                         effective_values=effective.numel(),packed_values=packed.numel())
                out['rows'].append(row)
                del expected,expected_local
        out['passed']=len(out['rows'])==16 and all(all(r[k] for k in
             ('adapter_exact','merged_exact','packed_exact','prefill_spec_exact','finite'))
             and r['dtype']=='torch.bfloat16' for r in out['rows'])
    except Exception as exc:
        out['errors'].append(type(exc).__name__+': '+str(exc)[:500]);out['passed']=False
    finally:
        out['elapsed_seconds']=time.monotonic()-begun
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.pending')
        with temporary.open('x') as f:json.dump(out,f)
        temporary.replace(destination)


def run(call,spec):
    from cineduplex.train.realtime_service import prepare_adapter
    tp=call.app.tensor_parallel
    started=False
    try:
        prepare_adapter(call.app.model,.25)
        tp.start(call.voice,call.language,.25,profile_id=call.id);started=True
        tp.install(call.app.model)
    finally:
        if started:tp.stop()
        if getattr(tp,'rank_profile',None) is not None:
            tp.rank_profile.close();tp.rank_profile=None
    rows=[]
    for rank in range(4):
        path=call.folder/('acoustic-weights-rank'+str(rank)+'.json')
        deadline=time.monotonic()+5
        while True:
            try:data=json.loads(path.read_text());break
            except (FileNotFoundError,json.JSONDecodeError):
                if time.monotonic()>=deadline:raise
                time.sleep(.1)
        rows.append(data)
    if not all(r['passed'] for r in rows):raise ValueError('acoustic weight audit failed; inspect rank evidence')
    hashes=[{r['name']:r['effective_sha256'] for r in x['rows']} for x in rows]
    result=dict(schema=1,call=call.id,rank_count=4,all_rank_exact=True,
                effective_weight_consensus=all(h==hashes[0] for h in hashes),
                adapter_tensors_per_rank=rows[0]['adapter_tensors'],
                microphone=False,speaker=False,generated_audio=False,quality_accepted=False)
    with (call.folder/'native-acoustic-weights.json').open('x') as f:json.dump(result,f)
    if not result['effective_weight_consensus']:raise ValueError('effective weights differ between ranks')
