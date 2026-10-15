"""Keep the native text channel typed and its terminal state monotonic.

This does not inject a transcript or force an answer. Text still comes from
native audio-conditioned logits; speech/control sampling is unchanged.
"""


class TerminalTextProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, ids, scores):
        p = self.processor
        # Upstream checks the length cap before noticing the previous EOT.
        # Its capped path returns immediately, causing EOT on every later step.
        # Consume that real emitted EOT before entering the original processor.
        if bool((ids[:, -1] == p.eos_token_id).any().item()):
            p.has_eos = True
        return p(ids, scores)


class TextVocabulary:
    def __init__(self, eos, pad):
        self.special = (int(eos), int(pad))

    def __call__(self, ids, scores):
        keep = scores[:, self.special].clone()
        scores[:, 151643:] = -float('inf')
        scores[:, self.special] = keep
        return scores


class ResponseSpeechRepetition:
    """Keep the existing n-gram rule inside one spoken response.

    Delays and the start marker recur at each response. Applying their n-grams
    to prior responses bans the first acoustic code of a repeated opening.
    Context and model caches remain untouched; only the rule's scope changes.
    """
    def __init__(self, original, receipts, prefix_length, enforce=True):
        self.original = original
        self.start = None
        self.receipts = receipts
        self.prefix_length = int(prefix_length)
        self.enforce = bool(enforce)

    def __call__(self, ids, scores):
        if self.start is None:
            # This wrapper is called by the n-gram processor after the native
            # delay/start prefix has been generated. Include that prefix but
            # exclude all earlier responses and listening padding.
            self.start = max(0, ids.shape[1] - self.prefix_length)
        unrestricted_top = scores.argmax(-1)
        full = self.original(ids, scores.clone())
        current = self.original(ids[:, self.start:], scores if self.enforce else scores.clone())
        import torch
        lo, hi = 151696, 158257
        extra = torch.isfinite(current[:, lo:hi]) & ~torch.isfinite(full[:, lo:hi])
        within_changed = (unrestricted_top != current.argmax(-1)).any()
        if bool((extra.any() | within_changed).item()):
            restored = (extra[0].nonzero().flatten() + lo).tolist()
            self.receipts.append({'history_length':int(ids.shape[1]),
                'response_start':int(self.start),'restored_acoustic_ids':restored,
                'before_repetition_top1':int(unrestricted_top.item()),
                'within_answer_changed_top1':bool(within_changed.item()),
                'raw_top1':int(current.argmax(-1).item()),
                'old_top1':int(full.argmax(-1).item()),
                'rule_applied':self.enforce})
        return current if self.enforce else scores


def install(fw):
    if getattr(fw, '_cineduplex_output_contract', False):
        raise ValueError('native output contract already installed')
    original = fw.init_speaking_processor

    def typed(end_factor):
        text, speech = original(end_factor)
        wrapped = []
        found = 0
        for p in text:
            if all(hasattr(p, k) for k in ('eos_token_id','has_eos','max_token_length')):
                wrapped.append(TerminalTextProcessor(p)); found += 1
            else:
                wrapped.append(p)
        if found != 1:
            raise ValueError('native text terminal processor identity')
        wrapped.append(TextVocabulary(fw.eos_token_id, fw.tts_pad_id))
        if getattr(fw, '_cineduplex_response_speech_scope', False):
            receipts = getattr(fw, '_cineduplex_repetition_receipts', None)
            if receipts is None:
                receipts = []; fw._cineduplex_repetition_receipts = receipts
            for p in speech:
                inner = getattr(p, 'ngram_size_processor', None)
                if inner is not None:
                    p.ngram_size_processor = ResponseSpeechRepetition(
                        inner, receipts, fw.stoken_delay_num+1,
                        enforce=not getattr(fw, '_cineduplex_natural_acoustic_repetition', False))
        return type(text)(wrapped), speech

    fw.init_speaking_processor = typed
    fw._cineduplex_output_contract = True
    return {'policy':'native-typed-text-terminal-v1', 'text_vocab_end':151643,
            'eot':int(fw.eos_token_id), 'pad':int(fw.tts_pad_id),
            'speech_sampling_changed':False, 'transcript_input':False}


def validate(fw, device):
    """Exercise the actual upstream processor at its cap, without generation."""
    import torch
    vocab = fw.model.config.text_config.vocab_size
    legal = torch.tensor([[1]], device=device)
    ended = torch.tensor([[fw.eos_token_id]], device=device)

    def scores():
        return torch.zeros((1, vocab), device=device)

    old, _ = fw.init_speaking_processor(1.)
    p = next(p.processor for p in old if isinstance(p, TerminalTextProcessor))
    p.cnt = p.max_token_length - 1
    old_first = int(p(legal, scores()).argmax(-1).item())
    old_second = int(p(ended, scores()).argmax(-1).item())
    new, _ = fw.init_speaking_processor(1.)
    q = next(p.processor for p in new if isinstance(p, TerminalTextProcessor))
    q.cnt = q.max_token_length - 1
    new_first = int(new(legal, scores()).argmax(-1).item())
    new_second = int(new(ended, scores()).argmax(-1).item())
    typed, _ = fw.init_speaking_processor(1.)
    probe = scores(); probe[:, 151696] = 100.; probe[:, 42] = 10.
    selected = int(typed(legal, probe).argmax(-1).item())
    if (old_first,old_second)!=(fw.eos_token_id,fw.eos_token_id):
        raise ValueError('expected upstream terminal defect not reproduced')
    if (new_first,new_second,selected)!=(fw.eos_token_id,fw.tts_pad_id,42):
        raise ValueError('native output contract validation failed')
    return {'upstream_at_cap':[old_first,old_second],
            'fixed_at_cap':[new_first,new_second], 'speech_id_masked':selected==42,
            'scope':'actual processor tensors, not audio quality'}
