"""Bounded shadow computation on recorded inputs; never changes live outputs.

The admitted leader temporarily uses its resident full BF16 weights. Peers
wait at the normal next-token boundary. No shadow KV enters the real session.
This diagnostic is explicitly installed, never enabled by a production default.
"""
from contextlib import contextmanager
import time
import types
import uuid
from pathlib import Path

import torch


def replace_history(previous, start, values):
    """Retain exactly the embeddings used at each absolute cache position."""
    if start < 0 or (previous is None and start) or (previous is not None and start > previous.shape[1]):
        raise ValueError('shadow history has a missing position')
    if start == 0:
        return values.detach().clone()
    return torch.cat((previous[:, :start], values.detach().clone()), 1)


@contextmanager
def dense_native(model):
    """Restore all leader projections and unfused RMS temporarily, then undo."""
    saved = []
    try:
        for name, module in model.named_modules():
            if not name.startswith(('model.', 'stoken_model.', 'control_model.', 'merge_model.', 'lm_head.original')):
                continue
            if module.__class__.__name__ == 'LoRALinear':
                weight, bias = module._realtime_effective_weight, module.base.bias
            elif isinstance(module, torch.nn.Linear) and '.base' not in name:
                weight, bias = module.weight, module.bias
            else:
                if module.__class__.__name__ in ('Qwen2RMSNorm', 'Qwen2MLP'):
                    original = getattr(module, '_realtime_unfused_forward', None)
                    if original is None:
                        original = getattr(module, '_tp_mlp_original', None)
                    if original is not None:
                        saved.append((module, module.forward)); module.forward = original
                continue
            if weight.dtype != torch.bfloat16:
                raise ValueError('shadow language-model weights must remain BF16')
            saved.append((module, module.forward))
            module.forward = types.MethodType(lambda self, x, w=weight, b=bias: torch.nn.functional.linear(x, w, b), module)
        yield
    finally:
        for module, original in reversed(saved):
            module.forward = original


def difference(actual, reference):
    a, b = actual.float(), reference.float()
    delta = a-b
    return {'relative_l2': float(delta.norm()/b.norm().clamp_min(1e-6)),
            'normalized_max': float(delta.abs().max()/b.abs().max().clamp_min(1e-6)),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


class AcousticProbe:
    def __init__(self, model, root):
        self.model = model
        self.root = Path(root)
        self.path = self.root/('acoustic-shadow-'+uuid.uuid4().hex+'.json')
        self.inputs = None; self.conditioning = None
        self.actual = {}; self.shadow = False; self.codes = 0
        self.results = []; self.handles = []
        self.saved = False
        self.bounds = []
        for name in ('model', 'stoken_model', 'control_model', 'merge_model'):
            branch = getattr(model, name)
            self.handles.append(branch.register_forward_hook(self._output(name)))
        self.handles.append(model.model.register_forward_pre_hook(self._input, with_kwargs=True))

    def _input(self, branch, args, kwargs):
        if self.shadow or len(self.results) >= 3:
            return
        pos = kwargs['cache_position']
        start, end = int(pos[0]), int(pos[-1])+1
        self.inputs = replace_history(self.inputs, start, kwargs['inputs_embeds'])
        self.bounds = [(a, min(b, start)) for a, b in self.bounds if a < start]+[(start, end)]

    def _output(self, name):
        def hook(branch, args, output):
            if self.shadow or len(self.results) >= 3:
                return
            self.actual[name] = output[0][:, -1:].detach().clone()
            if name == 'model':
                self.actual['main_layers'] = [h[:, -1:].detach().clone() for h in output.hidden_states]
        return hook

    def merge_tokens(self, start, tokens):
        if len(self.results) < 3:
            self.conditioning = replace_history(self.conditioning, start, tokens)

    def record(self, speech, logits, caches, text_token, control_token, speaking):
        if not speaking or not 151696 <= speech < 158257 or len(self.results) >= 3:
            return
        ordinal = self.codes; self.codes += 1
        if ordinal not in (0, 9, 24):
            return
        if self.inputs is None or self.conditioning is None or self.inputs.shape[1] != self.conditioning.shape[1]:
            raise ValueError('shadow main/merge histories differ')
        length = self.inputs.shape[1]
        if length > 1800:
            raise ValueError('shadow diagnostic prefix budget exceeded')
        # All native token broadcasts have completed before entering here.
        # No live cache, processor or RNG is passed to the shadow computation.
        started = time.perf_counter(); self.shadow = True
        try:
            with torch.inference_mode(), dense_native(self.model):
                main = self.model.model(inputs_embeds=self.inputs, use_cache=False,
                                        output_hidden_states=True, return_dict=True)
                stoken = self.model.stoken_model(inputs_embeds=main.hidden_states[self.model.stoken_model_index],
                                                 use_cache=False, return_dict=True)
                control = self.model.control_model(inputs_embeds=main.hidden_states[self.model.control_model_index],
                                                    use_cache=False, return_dict=True)
                merge = self.model.merge_model(inputs_embeds=stoken[0]+self.model.get_input_embeddings()(self.conditioning),
                                                use_cache=False, return_dict=True)
                index = self.model.lm_head.speech_index
                ref_logits = torch.nn.functional.linear(merge[0][:, -1:], self.model.lm_head.speech_weight)[0, 0]
                act_logits = logits[0].index_select(0, index)
                def top(x):
                    values, indices = x.float().topk(8)
                    return {'ids': index[indices].tolist(), 'logits': values.tolist()}
                branches = {name: difference(self.actual[name], ref[0][:, -1:])
                            for name, ref in (('model', main), ('stoken_model', stoken),
                                              ('control_model', control), ('merge_model', merge))}
                layer_errors = [difference(a, b[:, -1:]) for a, b in zip(self.actual['main_layers'], main.hidden_states)]
                result = {'acoustic_ordinal': ordinal, 'history_positions': length,
                          'live_cache_lengths': {k: v.get_seq_length() if v is not None else None for k, v in caches.items()},
                          'branches': branches, 'main_layer_errors': layer_errors,
                          'speech_logits': difference(act_logits, ref_logits),
                          'actual_top': top(act_logits), 'reference_top': top(ref_logits),
                          'sampled_triplet': [text_token, speech, control_token],
                          'seconds': time.perf_counter()-started}
                # Reproduce the exact live prefill/decode chunk boundaries.
                # All four caches are independent full-head native shadows.
                from transformers.cache_utils import DynamicCache
                ref_caches = {name: DynamicCache() for name in ('model', 'stoken_model', 'control_model', 'merge_model')}
                for a, b in self.bounds:
                    pos = torch.arange(a, b, device=self.inputs.device)
                    cm = self.model.model(inputs_embeds=self.inputs[:, a:b], use_cache=True,
                                          past_key_values=ref_caches['model'], cache_position=pos,
                                          output_hidden_states=True, return_dict=True)
                    cs = self.model.stoken_model(inputs_embeds=cm.hidden_states[self.model.stoken_model_index],
                                                use_cache=True, past_key_values=ref_caches['stoken_model'], cache_position=pos, return_dict=True)
                    cc = self.model.control_model(inputs_embeds=cm.hidden_states[self.model.control_model_index],
                                                  use_cache=True, past_key_values=ref_caches['control_model'], cache_position=pos, return_dict=True)
                    cg = self.model.merge_model(inputs_embeds=cs[0]+self.model.get_input_embeddings()(self.conditioning[:, a:b]),
                                                use_cache=True, past_key_values=ref_caches['merge_model'], cache_position=pos, return_dict=True)
                    for name, value in (('model', cm), ('stoken_model', cs), ('control_model', cc), ('merge_model', cg)):
                        ref_caches[name] = value.past_key_values
                cached_logits = torch.nn.functional.linear(cg[0][:, -1:], self.model.lm_head.speech_weight)[0, 0]
                result.update(native_cached_vs_full={name: difference(c[0][:, -1:], f[0][:, -1:])
                              for name, c, f in (('model', cm, main), ('stoken_model', cs, stoken),
                                                 ('control_model', cc, control), ('merge_model', cg, merge))},
                              live_vs_native_cached={name: difference(self.actual[name], value[0][:, -1:])
                              for name, value in (('model', cm), ('stoken_model', cs), ('control_model', cc), ('merge_model', cg))},
                              cached_speech_logits=difference(act_logits, cached_logits),
                              native_cached_top=top(cached_logits),
                              native_cache_boundaries=self.bounds,
                              total_shadow_seconds=time.perf_counter()-started)
                self.results.append(result)
        finally:
            self.shadow = False
        # Private NAS evidence, not a call event or a public transcript.
        from cineduplex.train.realtime_service import atomic
        atomic(self.path, {'kind': 'captured-exact-input-shadow-full-native-BF16',
                           'sampled_at': time.time(), 'quantization': 'none',
                           'live_outputs_unchanged': True, 'shadow_uses_live_cache': False,
                           'sampling_parameters_changed': False, 'results': self.results})
        self.saved = True
        if len(self.results) == 3:
            for h in self.handles: h.remove()
            self.handles.clear(); self.inputs = None; self.conditioning = None; self.actual.clear()


def install(model, root):
    old = getattr(model, '_cineduplex_acoustic_probe', None)
    if old is not None:
        for handle in old.handles: handle.remove()
    model._cineduplex_acoustic_probe = AcousticProbe(model, root)
