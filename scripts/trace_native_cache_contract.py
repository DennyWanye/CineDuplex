"""One read-only cache/full-history comparison at the observed first error.

Installed on all admitted ranks for a private diagnostic profile only. It does
not replace generation logits, caches, tokens, precision or sampling. No audio
is emitted. The reference rebuilds only the main semantic transformer, using
the exact embeddings previously consumed by the cached path.
"""
import json
import types
import uuid
import torch


def install(model, root, profile_id, rank):
    core = model.model
    previous = core.forward
    if hasattr(core, '_native_cache_trace_original'):
        core.forward = core._native_cache_trace_original
        del core._native_cache_trace_original
        previous = core.forward
    if profile_id is None:
        return
    call_id = str(uuid.UUID(profile_id))
    intent = root / 'calls' / call_id / 'native-input-contract.json'
    if not intent.is_file() or not json.loads(intent.read_text()).get('cache_trace_requested'):
        return
    core._native_cache_trace_original = previous
    chunks = []
    expected_position = 0
    calls = 0
    done = False

    def forward(current, *args, **kw):
        nonlocal expected_position, calls, done
        calls += 1
        positions = kw['cache_position']
        chunks.append(kw['inputs_embeds'].detach().clone())
        contiguous = bool(torch.equal(positions, torch.arange(
            expected_position, expected_position + positions.numel(), device=positions.device)))
        expected_position += positions.numel()
        actual = previous(*args, **kw)
        # 10 listening prefills + three speaking steps in the pinned trace.
        # All ranks execute the same additional forward before resuming.
        if calls == 13 and not done:
            done = True
            full = torch.cat(chunks, dim=1)
            shadow_kw = dict(kw)
            shadow_kw.update(inputs_embeds=full, past_key_values=None,
                cache_position=torch.arange(full.shape[1], device=full.device),
                position_ids=None, attention_mask=None, use_cache=True,
                output_hidden_states=True)
            shadow = previous(*args, **shadow_kw)
            # Independent full resident BF16 projections on each rank. Same
            # effective LoRA weights; remove only tensor-parallel projection
            # execution during this reference, then restore before generation.
            projections = []
            try:
                for name, layer in current.named_modules():
                    if layer.__class__.__name__ == 'LoRALinear':
                        weight, bias = layer._realtime_effective_weight, layer.base.bias
                    elif isinstance(layer, torch.nn.Linear) and '.base' not in name:
                        weight, bias = layer.weight, layer.bias
                    else:
                        continue
                    projections.append((layer, layer.forward))
                    layer.forward = types.MethodType(lambda m,x,w=weight,b=bias:
                        torch.nn.functional.linear(x,w,b),layer)
                unsharded = previous(*args, **shadow_kw)
            finally:
                for layer, method in projections:
                    layer.forward = method
            left, right = actual[0][:, -1:, :], shadow[0][:, -1:, :]
            # Read the existing BF16 text weights directly. The runtime head
            # wrapper cycles text/control/speech phases; calling that wrapper
            # for observation would advance its state and corrupt generation.
            weight = model.lm_head.weight[:151643]
            a = torch.nn.functional.linear(left, weight)[0, -1].float()
            b = torch.nn.functional.linear(right, weight)[0, -1].float()
            c = torch.nn.functional.linear(unsharded[0][:,-1:,:], weight)[0,-1].float()
            def top(logits):
                values, ids = logits.topk(8)
                return [{'token': i, 'logit': v} for i, v in zip(ids.tolist(), values.tolist())]
            def diff(x, y):
                x, y = x.float(), y.float()
                return {'relative_l2': float((x-y).norm()/y.norm().clamp_min(1e-9)),
                    'max_abs': float((x-y).abs().max()), 'equal': bool(torch.equal(x,y)),
                    'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all())}
            cached = actual.past_key_values
            fresh = shadow.past_key_values
            keys = []
            if hasattr(cached, 'layers') and hasattr(fresh, 'layers'):
                pairs = [(a.keys,a.values,b.keys,b.values) for a,b in zip(cached.layers,fresh.layers)]
            else:
                pairs = list(zip(cached.key_cache,cached.value_cache,fresh.key_cache,fresh.value_cache))
            for i, (ck,cv,fk,fv) in enumerate(pairs):
                keys.append({'layer':i, 'key':diff(ck,fk), 'value':diff(cv,fv)})
            record = {'rank':rank, 'forward':calls, 'position':positions.tolist(),
                'contiguous':contiguous, 'total_positions':full.shape[1],
                'hidden':diff(left,right), 'logits':diff(a,b), 'cached_top':top(a),
                'full_history_top':top(b), 'layer_kv':keys,
                'full_bf16_top':top(c), 'tp_vs_full_bf16_logits':diff(b,c),
                'full_bf16_projection_count':len(projections),
                'generation_outputs_replaced':False, 'audio_released':False}
            folder = root / 'calls' / profile_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f'cache-contract-rank{rank}.json').write_text(json.dumps(record))
            # No persistent runtime hook remains after the observation.
            current.forward = previous
            del current._native_cache_trace_original
        return actual

    core.forward = types.MethodType(forward, core)
