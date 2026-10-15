"""Read-only native output-row partition qualification on resident BF16 weights.

No module forward, sampler, cache or weight is replaced. This is a numerical
prerequisite, not a distributed performance or voice quality acceptance test.
"""
import json
import time
import torch
import torch.nn.functional as F


def run(call, spec):
    model = call.app.model
    start = time.monotonic()
    result = {'precision': 'bf16', 'quantization': 'none', 'world': 4,
              'microphone': False, 'speaker': False, 'weights_changed': False,
              'quality_accepted': False, 'scope': 'local arithmetic prerequisite; no NCCL timing',
              'checks': [], 'all_bitwise_equal': True}
    seen = set()
    versions = {n: p._version for n, p in model.named_parameters()}
    with torch.inference_mode():
        for name, layer in model.named_modules():
            if not name.startswith(('model.', 'stoken_model.', 'control_model.', 'merge_model.')):
                continue
            if layer.__class__.__name__ == 'LoRALinear':
                weight, bias = layer._realtime_effective_weight, layer.base.bias
            elif isinstance(layer, torch.nn.Linear) and '.base' not in name:
                weight, bias = layer.weight, layer.bias
            else:
                continue
            key = (tuple(weight.shape), bias is not None)
            if key in seen:
                continue
            seen.add(key)
            if weight.dtype != torch.bfloat16 or weight.shape[0] % 4:
                raise ValueError('expected divisible resident BF16 projection')
            for length in (1, 3):
                for phase in (.017, .031):
                    if call.closed.is_set() or time.monotonic()-start > 90:
                        raise RuntimeError('bounded projection diagnostic expired')
                    x = torch.sin(torch.arange(length*weight.shape[1], device=weight.device).float()*phase)
                    x = x.bfloat16().reshape(1, length, -1)
                    expected = F.linear(x, weight, bias)
                    width = weight.shape[0]//4
                    parts = [F.linear(x, weight[i*width:(i+1)*width],
                                      None if bias is None else bias[i*width:(i+1)*width]) for i in range(4)]
                    actual = torch.cat(parts, -1)
                    equal = torch.equal(expected, actual)
                    result['checks'].append({'name': name, 'shape': list(weight.shape),
                        'length': length, 'phase': phase, 'bitwise_equal': equal,
                        'role': 'decode candidate' if length == 1 else 'prefill negative control; retain full matrix',
                        'different_values': int((expected != actual).sum()),
                        'max_absolute_error': float((expected.float()-actual.float()).abs().max())})
                    if not equal and length == 1:
                        result['all_bitwise_equal'] = False
                        break
                if not result['all_bitwise_equal']:
                    break
            if not result['all_bitwise_equal']:
                break
    if versions != {n: p._version for n, p in model.named_parameters()}:
        raise RuntimeError('resident parameter version changed')
    result['elapsed_seconds'] = time.monotonic()-start
    from cineduplex.train.realtime_service import atomic
    atomic(call.folder/'native-projection-contract.json', result)
