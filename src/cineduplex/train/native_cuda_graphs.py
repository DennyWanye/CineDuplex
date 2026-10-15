"""Replay unchanged native BF16 stateless operators with CUDA graphs.

Only single-token MLP/RMS calls are captured. Attention, KV state, sampling,
audio input and safety stay outside the graph. No arithmetic is rewritten.
Every graph must match its eager operator bit for bit before use.
"""
import types


class NativeGraph:
    def __init__(self, forward, sample, name):
        import torch
        self.forward = forward
        self.name = name
        self.shape = tuple(sample.shape)
        self.dtype = sample.dtype
        self.device = sample.device
        self.replays = 0
        self.checked_actual = False
        self.input = sample.clone()
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Warm the exact eager operators before capture on a side stream.
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                forward(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            self.output = forward(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        # Independent pools: branches do not always execute in the same order.
        for value in (sample, -sample + sample.new_tensor(.125)):
            expected = forward(value)
            actual = self.replay(value)
            if not torch.equal(expected, actual):
                raise ValueError('native graph changed BF16 output: ' + name)

    def replay(self, x):
        self.input.copy_(x)
        self.graph.replay()
        # Do not expose reusable graph storage to residuals or future caches.
        return self.output.clone()

    def __call__(self, x):
        import torch
        if tuple(x.shape) != self.shape or x.dtype != self.dtype or x.device != self.device:
            return self.forward(x)
        result = self.replay(x)
        if not self.checked_actual:
            if not torch.equal(result, self.forward(x)):
                raise ValueError('native graph differs on actual model input: ' + self.name)
            self.checked_actual = True
        self.replays += 1
        return result


def install(model):
    import torch
    if getattr(model, '_cineduplex_native_graphs', None) is not None:
        raise ValueError('previous native graphs have not closed')
    installed = []
    try:
        with torch.inference_mode():
            for name, layer in model.named_modules():
                if not name.startswith(('model.', 'stoken_model.', 'control_model.', 'merge_model.')):
                    continue
                kind = layer.__class__.__name__
                if kind == 'Qwen2RMSNorm':
                    width = layer.weight.numel()
                elif kind == 'Qwen2MLP':
                    projection = getattr(layer.gate_proj, 'base', layer.gate_proj)
                    width = projection.in_features
                else:
                    continue
                sample = torch.sin(torch.arange(width, device=model.device).float() * .017)
                sample = sample.to(torch.bfloat16).reshape(1, 1, width)
                original = layer.forward
                graph = NativeGraph(original, sample, name)
                layer.forward = types.MethodType(lambda m, x, op=graph: op(x), layer)
                installed.append((name, layer, original, graph))
        model._cineduplex_native_graphs = installed
        return {'operators': len(installed), 'qualification': 'two inputs bitwise equal',
                'arithmetic': 'unchanged native BF16', 'attention_or_kv_captured': False}
    except Exception:
        torch.cuda.synchronize()
        for _, layer, original, _ in installed:
            layer.forward = original
        raise


def close(model):
    import torch
    installed = getattr(model, '_cineduplex_native_graphs', None)
    if installed is None:
        return []
    torch.cuda.synchronize()
    receipt = []
    for name, layer, original, graph in installed:
        layer.forward = original
        receipt.append({'name': name, 'replays': graph.replays,
                        'actual_input_bitwise_equal': graph.checked_actual})
        graph.graph.reset()
    model._cineduplex_native_graphs = None
    return receipt
