"""Keep acoustic token RNG independent of concurrently running codecs."""
import types
import torch


def restore_native_control(model):
    from lychee_fd.runtime import hf_v9_realtime

    def score(processor, ids, scores):
        tokens = [processor.kl_token_id, processor.ss_token_id]
        names = ['kl', 'ss']
        if processor.bc_token_id is not None:
            tokens.append(processor.bc_token_id); names.append('bc')
        values = scores[:, tokens].clone()
        raw = values[0].float()
        scores.fill_(-float('inf')); scores[:, tokens] = values
        scores[:, processor.ss_token_id] *= processor.start_speak_token_factor
        if processor.bc_token_id is not None:
            scores[:, processor.bc_token_id] *= processor.bc_speak_token_factor
        adjusted = scores[0, tokens].float()
        audit = torch.stack((raw, adjusted, adjusted.softmax(0))).tolist()
        processor.last_probs = dict(zip(names, audit[2]))
        model._realtime_control_audit = {'labels': names, 'raw_logits': audit[0],
            'adjusted_logits': audit[1], 'probabilities': audit[2],
            'start_factor': processor.start_speak_token_factor,
            'backchannel_factor': processor.bc_speak_token_factor}
        return scores

    hf_v9_realtime.ListeningControlLogitsProcessor.__call__ = score
    # Exercise the actual installed processor before a distributed call can
    # enter token collectives. A missing return must fail at installation.
    processor = hf_v9_realtime.ListeningControlLogitsProcessor.__new__(
        hf_v9_realtime.ListeningControlLogitsProcessor)
    processor.kl_token_id=0;processor.ss_token_id=1;processor.bc_token_id=2
    processor.start_speak_token_factor=1.;processor.bc_speak_token_factor=1.
    source=torch.tensor([[13.,12.,-2.,99.]],device=model.device)
    result=processor(None,source.clone())
    if not isinstance(result,torch.Tensor) or not torch.equal(result[:,:3],source[:,:3]) or not torch.isneginf(result[:,3]).all():
        raise ValueError('native control processor contract failed')


def install(model):
    seed = int(torch.initial_seed())

    def reset():
        model._cineduplex_sampling_generator = torch.Generator(device=model.device).manual_seed(seed)

    def sample(current, logits, ids, processors=None, temperature=1., top_k=None,
               top_p=None, do_sample=True):
        if processors is not None:
            logits = processors(ids, logits)
        if not do_sample or temperature == 0:
            return logits.argmax(-1, keepdim=True)
        # Preserve the native speech parameters; text/control remain greedy.
        logits = logits.float() / temperature
        if top_k is not None and top_k > 0:
            pivot = logits.topk(min(top_k, logits.shape[-1]))[0][:, -1:]
            logits = logits.masked_fill(logits < pivot, -float('inf'))
        if top_p is not None and 0 < top_p < 1:
            ordered, index = logits.sort(descending=True)
            remove = ordered.softmax(-1).cumsum(-1) > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            logits = logits.masked_fill(remove.scatter(1, index, remove), -float('inf'))
        return torch.multinomial(logits.softmax(-1), 1,
                                 generator=current._cineduplex_sampling_generator)

    model._cineduplex_reset_sampling = reset
    model._cineduplex_restore_native_control = lambda: restore_native_control(model)
    reset()
    bound=types.MethodType(sample,model)
    restore_native_control(model)
    # Global codec randomness must not change a reset model-only sequence.
    state=torch.cuda.get_rng_state(model.device)
    logits=torch.arange(8,device=model.device,dtype=torch.float32).reshape(1,8)*.01
    first=[bound(logits,None).item() for _ in range(8)]
    reset();torch.rand(73,device=model.device)
    second=[bound(logits,None).item() for _ in range(8)]
    torch.cuda.set_rng_state(state,model.device);reset()
    if first!=second:raise ValueError('model sampling depends on global codec RNG')
    return bound, {'seed': seed, 'scope': 'per-call-model-only',
        'processor_contract': 'passed', 'rng_isolation_sequence': first,
        'codec_rng_can_advance_without_changing_model_sampling': True,
        'speech_parameters': 'native-call-arguments'}
