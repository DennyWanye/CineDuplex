"""Causal BF16 input-addition order comparison against training/vLLM.

No token, weight, sampling or audio changes. The hook rebuilds the same four
inputs; it never subtracts rounded embeddings to approximate the reference.
"""
import types


def install(model):
    import torch
    if getattr(model, '_native_embedding_order', None) is not None:
        raise ValueError('embedding comparison already owned')
    previous = model.forward
    pending = {}
    receipt = {'policy': 'text-speech-control-audio', 'checked': 0,
               'different_calls': 0, 'changed_values': 0, 'max_abs': 0.,
               'original_input_verified': True, 'transcript_input': False}

    def observe(current, *args, **kwargs):
        if pending:
            raise ValueError('nested native forward')
        pending.update(kwargs)
        try:
            return previous(*args, **kwargs)
        finally:
            pending.clear()

    def reorder(layer, args, kwargs):
        if not pending or pending.get('inputs_embeds') is not None:
            raise ValueError('expected native cached forward')
        ids = pending['input_ids']
        prefix_len = pending['prefix_input_ids'].shape[1]
        text = model.get_input_embeddings()(ids)
        if model.no_text_label:
            raise ValueError('text channel must remain enabled')
        speech = model.get_input_embeddings()(pending['stoken_ids'])
        control = model.get_input_embeddings()(pending['control_input_ids'])
        cache = pending.get('past_key_values')
        if cache is None or cache.get_seq_length() == 0:
            zeros = text.new_zeros((text.shape[0], prefix_len, text.shape[2]))
            speech = torch.cat((zeros, speech), 1)
            control = torch.cat((zeros, control), 1)
        cached = pending.get('pre_computed_audio')
        if cached is None:
            cached = model._realtime_audio
        if cached is None or text.dtype != torch.bfloat16:
            raise ValueError('cached BF16 audio inputs required')
        # The admitted sliced-audio forward deliberately does not retain the
        # older packed embedding attribute. Rebuild from the native features;
        # actual assembled inputs are checked exactly below before replacement.
        features, lens = cached
        audio_ids = pending['audio_input_ids']
        audio = model.get_input_embeddings()(audio_ids)
        valid = torch.arange(features.shape[1], device=features.device)[None, :] < lens[:, None]
        locations = audio_ids.eq(151690).unsqueeze(-1).expand_as(audio)
        values = features[valid].to(audio.dtype)
        if int(locations.sum().item()) != values.numel():
            raise ValueError('native audio feature count differs')
        audio = audio.masked_scatter(locations, values)
        audio = torch.cat((audio.new_zeros((audio.shape[0], prefix_len, audio.shape[2])), audio), 1)
        audio = audio[:, pending['cache_position']]
        original = ((text + control) + speech) + audio
        expected = ((text + speech) + control) + audio
        if not torch.equal(original, kwargs['inputs_embeds']):
            receipt['original_input_verified'] = False
            raise ValueError('actual native input differs from known assembly')
        if receipt['checked'] < 512:
            delta = (original.float() - expected.float()).abs()
            changed = int((original != expected).sum().item())
            receipt['checked'] += 1
            receipt['different_calls'] += int(changed > 0)
            receipt['changed_values'] += changed
            receipt['max_abs'] = max(receipt['max_abs'], float(delta.max().item()))
        return args, {**kwargs, 'inputs_embeds': expected}

    handle = model.model.register_forward_pre_hook(reorder, with_kwargs=True)
    model.forward = types.MethodType(observe, model)
    model._native_embedding_order = (previous, handle, receipt)
    return receipt


def close(model):
    state = getattr(model, '_native_embedding_order', None)
    if state is None:
        return None
    previous, handle, receipt = state
    handle.remove()
    model.forward = previous
    del model._native_embedding_order
    return receipt
