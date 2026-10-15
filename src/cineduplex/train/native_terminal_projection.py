"""Skip the word head only after the real native text terminal is latched.

The existing processor then permits exactly TTS padding. Decoder hidden states,
speech/control projections, caches, weights and sampling remain unchanged.
An actual full projection qualifies that invariant once before the shortcut.
"""
import types


def terminal_observed(terminal):
    """Use broadcast output, never leader-local logits-processor state."""
    return bool(terminal is not None and getattr(terminal, '_broadcast_eot_seen', False))


def observe_output(terminal, result):
    if terminal is None:
        return
    token = result.get('text_token')
    if type(token) is not int:
        raise ValueError('native stream must publish its validated broadcast text token')
    if token == int(terminal.processor.eos_token_id):
        terminal._broadcast_eot_seen = True
        # Only the leader invokes sampling processors. All ranks must latch
        # the actual emitted token before choosing the next collective path.
        terminal.processor.has_eos = True


def install(session):
    import torch
    model = getattr(session.fw.model, '_model', session.fw.model)
    head = model.lm_head
    if head.__class__.__name__ != 'RealtimeProjection':
        raise ValueError('original BF16 realtime projection required')
    if hasattr(model, '_cineduplex_terminal_projection_restore'):
        raise ValueError('previous terminal projection owner still active')
    previous = head.forward
    stream = session._stream_model
    pad = int(session.fw.tts_pad_id)
    receipt = {'policy': 'native-terminal-pad-only-head', 'qualified': False,
               'skipped_word_projections': 0, 'original_word_projections': 0,
               'weights_changed': False, 'sampling_changed': False}
    state = {'processor': None}

    def forward(current, x):
        terminal = state['processor']
        if current.phase != 'text' or not terminal_observed(terminal):
            if current.phase == 'text':
                receipt['original_word_projections'] += 1
            return previous(x)
        p = terminal.processor
        if p.ngram_size_processor is not None or p.prefix_token_id is not None:
            raise ValueError('terminal word processor gained another constraint')
        if not receipt['qualified']:
            # The terminal is already latched, so this probe cannot change its
            # counters. It runs no sampling and consumes no RNG.
            full = previous(x)[:, -1, :].clone()
            ids = torch.tensor([[pad]], dtype=torch.long, device=x.device)
            masked = p(ids, full)
            if (int(torch.isfinite(masked).sum().item()) != 1
                    or int(masked.argmax(-1).item()) != pad):
                raise ValueError('real full word head does not force native padding')
            receipt['qualified'] = True
            receipt['qualified_token'] = pad
        current.phase = 'unused'  # exactly the original text-head transition
        scores = x.new_full((x.shape[0], 1, current.weight.shape[0]), -float('inf'))
        scores[:, :, pad] = 0
        receipt['skipped_word_projections'] += 1
        return scores

    def observed(current, **kwargs):
        processors = kwargs.get('logits_processor') or ()
        terms = [p for p in processors if p.__class__.__name__ == 'TerminalTextProcessor']
        # Listening and backchannel remain on their original projection path.
        if terms and (len(terms) != 1 or any(p.__class__.__name__ not in
                ('TerminalTextProcessor', 'TextVocabulary', 'SpeakingTextVocabulary') for p in processors)):
            raise ValueError('unknown word processor cannot use terminal shortcut')
        state['processor'] = terms[0] if terms else None
        try:
            for result in stream(**kwargs):
                observe_output(state['processor'], result)
                yield result
        finally:
            state['processor'] = None

    head.forward = types.MethodType(forward, head)
    session._stream_model = types.MethodType(observed, session)
    model._cineduplex_terminal_projection_restore = (head, previous, receipt)
    return receipt


def close(model):
    saved = getattr(model, '_cineduplex_terminal_projection_restore', None)
    if saved is None:
        return None
    head, previous, receipt = saved
    head.forward = previous
    del model._cineduplex_terminal_projection_restore
    return receipt
