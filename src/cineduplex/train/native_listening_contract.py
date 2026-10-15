"""Match native listening cache conditioning to retained PAD channels.

Only listening predictions are constrained. Original audio, control, speaking
text and speech generation remain native; no recognized text is supplied.
"""
import types


def install(session):
    from lychee_fd.runtime.vllm_generation import SingleTurnGenerationFramework
    original = session._stream_model
    receipt = {'listening_windows': 0, 'speaking_windows': 0,
               'recognized_text_input': False, 'policy': 'official-listening-pad'}

    def aligned(current, **kwargs):
        if current.listening_state == 'l':
            text, speech = SingleTurnGenerationFramework.init_listening_pad_processor(current.fw)
            kwargs = {**kwargs, 'logits_processor': text, 'stoken_logits_processor': speech}
            receipt['listening_windows'] += 1
        else:
            receipt['speaking_windows'] += 1
        return original(**kwargs)

    session._stream_model = types.MethodType(aligned, session)
    session._cineduplex_listening_contract = receipt
    return receipt
