"""Output-row BF16 decode partition with exact admission and eager fallback.

Every dot product retains its complete input dimension and PyTorch F.linear.
Prefill is unchanged. No split-K reduction, custom GEMV, or quantization.
This opt-in candidate still requires recorded generation equality validation.
"""
import types
import torch
import torch.distributed as dist
import torch.nn.functional as F


class RowGroup:
    def __init__(self, items, group):
        self.items = items
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.widths = [w.shape[0]//self.world for _, _, w, _ in items]
        self.parts = [(w[self.rank*k:(self.rank+1)*k],
                       None if b is None else b[self.rank*k:(self.rank+1)*k])
                      for (_, _, w, b), k in zip(items, self.widths)]
        self.enabled = True
        self.checked_actual = False
        self.cached_input = None
        self.cached_outputs = None
        self.used = set()
        self.replays = 0

    def native(self, x):
        return [F.linear(x, w, b) for _, _, w, b in self.items]

    def partition(self, x):
        y = torch.cat([F.linear(x, w, b) for w, b in self.parts], -1)
        width = sum(self.widths)
        flat = y.reshape(1, width).contiguous()
        gathered = torch.empty((self.world, width), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(gathered, flat, group=self.group)
        outputs = []
        offset = 0
        for k in self.widths:
            outputs.append(gathered[:, offset:offset+k].reshape(*x.shape[:-1], self.world*k))
            offset += k
        return outputs

    def same_everywhere(self, a, b):
        equal = torch.tensor([int(all(torch.equal(x, y) for x, y in zip(a, b)))],
                             device=a[0].device)
        dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=self.group)
        return bool(equal.item())

    def qualify(self):
        w = self.items[0][2]
        for phase in (.017, .031):
            x = torch.sin(torch.arange(w.shape[1], device=w.device).float()*phase).bfloat16()[None, None]
            if not self.same_everywhere(self.native(x), self.partition(x)):
                self.enabled = False
                break

    def compute(self, x):
        if x.numel() != x.shape[-1] or not self.enabled:
            return self.native(x)
        actual = self.partition(x)
        if not self.checked_actual:
            expected = self.native(x)
            self.enabled = self.same_everywhere(expected, actual)
            self.checked_actual = True
            if not self.enabled:
                return expected
        self.replays += 1
        return actual

    def select(self, index, x):
        if x is not self.cached_input or index in self.used:
            self.cached_outputs = self.compute(x)
            self.cached_input = x
            self.used = set()
        result = self.cached_outputs[index]
        self.used.add(index)
        if len(self.used) == len(self.items):
            self.cached_input = None
            self.cached_outputs = None
            self.used.clear()
        return result


def install(model, group):
    if hasattr(model, '_cineduplex_native_row_state'):
        raise ValueError('prior row-partition call still owns state')
    world = dist.get_world_size(group)
    selected = {}
    for name, layer in model.named_modules():
        if not name.startswith(('model.', 'stoken_model.', 'control_model.', 'merge_model.')):
            continue
        if layer.__class__.__name__ == 'LoRALinear':
            w, b = layer._realtime_effective_weight, layer.base.bias
        elif isinstance(layer, torch.nn.Linear) and '.base' not in name:
            w, b = layer.weight, layer.bias
        else:
            continue
        if w.dtype != torch.bfloat16:
            raise ValueError('native BF16 weights required')
        # P191 found a data-dependent rounding difference for the long-K
        # down projection; N193's only admitted down_proj later changed the
        # generated codes. Warmup equality is not a proof for this geometry.
        # Preserve all down projections as full native F.linear, even when a
        # particular layer happens to pass the sampled qualification inputs.
        if name.endswith('.down_proj'):
            continue
        if w.shape[0] % world == 0:
            selected[name] = (name, layer, w, b)
    groups = []
    consumed = set()
    for name, item in selected.items():
        if name in consumed:
            continue
        parent, _, leaf = name.rpartition('.')
        leaves = ('q_proj', 'k_proj', 'v_proj') if leaf == 'q_proj' else (
            ('gate_proj', 'up_proj') if leaf == 'gate_proj' else (leaf,))
        names = [parent+'.'+x for x in leaves]
        if not all(x in selected for x in names):
            names = [name]
        groups.append(RowGroup([selected[x] for x in names], group))
        consumed.update(names)
    saved = []
    try:
        with torch.inference_mode():
            for operator in groups:
                operator.qualify()
                for index, (_, layer, _, _) in enumerate(operator.items):
                    saved.append((layer, layer.forward))
                    layer.forward = types.MethodType(
                        lambda m, x, op=operator, i=index: op.select(i, x), layer)
        model._cineduplex_native_row_state = (saved, groups)
        return receipt(groups)
    except Exception:
        for layer, previous in reversed(saved):
            layer.forward = previous
        raise


def receipt(groups):
    return {'groups': len(groups), 'eligible_groups': sum(x.enabled for x in groups),
            'prefill': 'unchanged full native matrix', 'split_k': False,
            'details': [{'names': [x[0] for x in op.items], 'enabled': op.enabled,
                         'checked_actual': op.checked_actual, 'decode_replays': op.replays}
                        for op in groups]}


def close(model):
    saved, groups = model._cineduplex_native_row_state
    torch.cuda.synchronize()
    result = receipt(groups)
    for layer, previous in reversed(saved):
        layer.forward = previous
    del model._cineduplex_native_row_state
    return result
