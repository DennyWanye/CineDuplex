"""Embed only uncached audio positions, retaining native feature alignment."""
import types
import torch


def install_context(session):
    """Reuse the native append-only feature buffer instead of repacking history."""
    original = session._build_cached_audio_context

    def build(current):
        total = len(current._audio_input_id_list)
        if not total or not current._audio_embed_seq_cache:
            return original()
        packed = current._packed_audio_embeds()
        if packed is None:
            return original()
        ids = torch.tensor(current._audio_input_id_list, device=current.fw.device,
                           dtype=torch.long).unsqueeze(0)
        # Precomputed features replace raw encoder inputs. Keep the raw
        # recordings and every feature window in the native session for
        # diagnostics and close(), but avoid copying them every 400ms.
        return ids, None, None, total, None, packed

    # A real first input verifies the packed buffer against native padding.
    # Later append operations use that same native buffer implementation.
    def checked(current):
        result = original()
        packed = current._packed_audio_embeds()
        if packed is None or result[5] is None:
            return result
        native, native_lens = result[5]
        actual, actual_lens = packed
        if not torch.equal(native_lens, actual_lens) or not torch.equal(native, actual):
            raise ValueError('packed audio context differs from native feature layout')
        current._build_cached_audio_context = types.MethodType(build, current)
        return result[0], None, None, result[3], None, packed

    session._build_cached_audio_context = types.MethodType(checked, session)


def selected_audio(model, audio_ids, features, lens, positions, prefix_len, layout=None):
    if layout is None:
        counts = audio_ids.eq(151690).long().cumsum(-1) - 1
        ends = lens.long().cumsum(0)
        if int(ends[-1]) != int(counts[0, -1]) + 1:
            raise ValueError('cached audio alignment differs from model input')
        layout = counts, ends
    counts, ends = layout
    index = (positions - prefix_len).clamp_min(0)
    ids = audio_ids.index_select(1, index)
    audio = model.get_input_embeddings()(ids)
    feature_index = counts[0].index_select(0, index).clamp_min(0)
    window = torch.searchsorted(ends, feature_index, right=True)
    offsets = torch.cat((ends.new_zeros(1), ends[:-1]))
    within = feature_index - offsets[window]
    values = features[window, within].to(audio.dtype).unsqueeze(0)
    audio = torch.where(ids.eq(151690).unsqueeze(-1), values, audio)
    return torch.where((positions >= prefix_len)[None, :, None], audio, torch.zeros_like(audio)), layout


def install(model):
    model._cineduplex_install_audio_context = install_context
    original = getattr(model, '_cineduplex_audio_unsliced_forward', model.forward)
    model._cineduplex_audio_unsliced_forward = original
    model._cineduplex_audio_slice_layout = None

    def forward(current, *args, **kwargs):
        cached = kwargs.get('pre_computed_audio')
        if cached is None:
            cached = current._realtime_audio
        if cached is None or kwargs.get('inputs_embeds') is not None:
            return original(*args, **kwargs)
        ids = kwargs['input_ids']; prefix = kwargs['prefix_input_ids']
        embed = current.get_input_embeddings()(ids)
        if current.no_text_label:
            embed[:, prefix.shape[1]:] = 0
        controls = current.get_input_embeddings()(kwargs['control_input_ids'])
        speech = current.get_input_embeddings()(kwargs['stoken_ids'])
        past = kwargs.get('past_key_values')
        if past is None or past.get_seq_length() == 0:
            zeros = embed.new_zeros((embed.shape[0], prefix.shape[1], embed.shape[2]))
            controls = torch.cat((zeros, controls), 1)
            speech = torch.cat((zeros, speech), 1)
        features, lens = cached; audio_ids = kwargs['audio_input_ids']
        key = (audio_ids.data_ptr(), tuple(audio_ids.shape), features.data_ptr(), tuple(features.shape), lens.data_ptr())
        saved = current._cineduplex_audio_slice_layout
        layout = saved[1] if saved is not None and saved[0] == key else None
        audio, layout = selected_audio(current, audio_ids, features, lens,
                                      kwargs['cache_position'], prefix.shape[1], layout)
        current._cineduplex_audio_slice_layout = key, layout
        kwargs['inputs_embeds'] = embed + controls + speech + audio
        kwargs['wavs'] = None; kwargs['wav_lens'] = None
        return original(*args, **kwargs)

    # Compare both layouts on real GPU embeddings, including variable window
    # lengths and the system prefix. No audio, generation or new model load.
    ids = torch.tensor([[151690, 151691, 151690, 151691, 151690, 151691,
                         151690, 151691, 151690, 151691, 151690, 151691,
                         151690, 151691, 151690, 151691]], device=model.device)
    hidden = model.get_input_embeddings().weight.shape[1]
    features = torch.sin(torch.arange(2*5*hidden, device=model.device).float()).reshape(2, 5, hidden).bfloat16()
    lens = torch.tensor([5, 3], device=model.device)
    full = model.get_input_embeddings()(ids)
    mask = torch.arange(5, device=model.device)[None, :] < lens[:, None]
    full = full.masked_scatter(ids.eq(151690).unsqueeze(-1).expand_as(full), features[mask].to(full.dtype))
    full = torch.cat((full.new_zeros((1, 3, hidden)), full), 1)
    for positions in (torch.arange(19, device=model.device), torch.tensor([0, 2, 3, 8, 18], device=model.device)):
        sliced, _ = selected_audio(model, ids, features, lens, positions, 3)
        if not torch.equal(full[:, positions], sliced):
            raise ValueError('sliced audio embedding differs from native full layout')
    model.forward = types.MethodType(forward, model)
    return {'native_embedding_comparison': 'exact', 'variable_feature_lengths': [5, 3],
            'positions': 'full-prefill-and-scattered-decode', 'history_retained': True}
