"""Observe retained native token histories against consumed cache positions.

No logits, RNG, histories or caches are modified. This is a bounded per-call
diagnostic, not a production quality claim or a generation workaround.
"""
import types


def differences(previous, current, limit=12):
    return [{'position': i, 'cached': a, 'retained': b}
            for i, (a, b) in enumerate(zip(previous, current)) if a != b][:limit]


def install(session):
    fw = session.fw
    original = session._stream_model
    advance = session.advance_stream
    ledger = {}
    receipt = {'observation_only': True, 'windows': 0, 'cache_resets': 0,
               'cached_history_mismatches': [], 'listening_output_mismatches': [],
               'boundaries': [], 'merge_history_mismatches': []}
    names = (('text', 'input_ids'), ('speech', 'stoken_ids'),
             ('control', 'control_input_ids'))
    result_names = dict(text='sequences', speech='stoken_ids', control='control_ids')
    merge_tokens = []

    def snapshot(current, phase, event=None):
        row = {'phase': phase, 'state': current.listening_state,
               'chunk': int(current.next_chunk_pos), 'event': event}
        for label, attr, end in (
            ('text', 'generated_input_ids', fw.eos_token_id),
            ('speech', 'generated_stoken_ids', fw.tts_end_id),
            ('control', 'generated_control_ids', fw.sl_token_id)):
            values = getattr(current, attr)[0].tolist()
            row[label] = {'length': len(values), 'tail': values[-24:],
                          'end_positions': [i for i, v in enumerate(values) if v == int(end)][-12:]}
            if label == 'text':
                row['generated_text'] = fw.tokenizer.decode(
                    [v for v in values[current.prefix_input_ids.shape[1]:] if v < 151643],
                    skip_special_tokens=True)
        row['cache_lengths'] = {key: (int(getattr(current, key).get_seq_length())
                                     if getattr(current, key) is not None else None)
                                for key in ('past_key_values', 'stoken_past_key_values',
                                            'control_past_key_values', 'merge_past_key_values')}
        if len(receipt['boundaries']) < 20:
            receipt['boundaries'].append(row)

    def observed(current, **kwargs):
        nonlocal merge_tokens
        receipt['windows'] += 1
        if receipt['windows'] > 700:
            raise RuntimeError('native history observation budget exceeded')
        cache = kwargs.get('past_key_values')
        start = int(cache.get_seq_length()) if cache is not None else 0
        prefix_len = int(kwargs['prefix_input_ids'].shape[1])
        if not start:
            ledger.clear(); merge_tokens = []; receipt['cache_resets'] += 1
        for label, key in names:
            values = kwargs[key][0].tolist()
            bad = differences(ledger.get(label, []), values)
            if bad and len(receipt['cached_history_mismatches']) < 20:
                receipt['cached_history_mismatches'].append(
                    {'window': receipt['windows'], 'branch': label, 'state': current.listening_state,
                     'cache_length': start, 'differences': bad})
        # Merge consumes the next text token at each cached position, unlike
        # the main language branch. Compare only already-consumed positions.
        bad = differences(merge_tokens, kwargs['input_ids'][0, 1:].tolist())
        if bad and len(receipt['merge_history_mismatches']) < 20:
            receipt['merge_history_mismatches'].append(
                {'window': receipt['windows'], 'state': current.listening_state, 'differences': bad})
        last = None
        for result in original(**kwargs):
            last = result
            n = int(result['past_key_values'].get_seq_length())
            # A native result includes one sampled token beyond the KV cache.
            # Save consumed positions only; the next call may legitimately
            # replace the unconsumed token during a control transition.
            for label, _ in names:
                end = n if label == 'text' else max(0, n-prefix_len)
                values = result[result_names[label]][0, :end].tolist()
                ledger[label] = values
            next_values = result['sequences'][0, start+1:n+1].tolist()
            merge_tokens = merge_tokens[:start] + next_values
            start = n
            yield result
        if last is not None and current.listening_state == 'l':
            selected = [int(last[key][0, -1]) for key in ('sequences', 'stoken_ids')]
            expected = [int(fw.text_pad_token_id), int(fw.stoken_pad_token_id)]
            if selected != expected and len(receipt['listening_output_mismatches']) < 20:
                receipt['listening_output_mismatches'].append(
                    {'window': receipt['windows'], 'selected': selected, 'retained': expected})

    def observed_advance(current, *args, **kwargs):
        boundary = False
        for event in advance(*args, **kwargs):
            if event.get('type') in ('state_change', 'speaking_done'):
                snapshot(current, 'event', dict(event)); boundary = True
            yield event
        if boundary:
            snapshot(current, 'after_window')

    session._stream_model = types.MethodType(observed, session)
    session.advance_stream = types.MethodType(observed_advance, session)
    return receipt
