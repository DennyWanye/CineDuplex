"""Translate native V9 turn boundaries into live playback semantics.

An acoustic EOS is normal completion. An S->L control decision while the
acoustic sequence has not ended cancels queued playback. V9's offline event
marks both as non-interrupting, which is unsuitable for a live output queue.
This adapter retains all native input, control decisions and KV history.
"""
import types


def install(native_prefix=False):
    from lychee_fd.runtime.hf_v9_realtime import HFRealtimeV9GenerationFramework
    cls = HFRealtimeV9GenerationFramework
    original = getattr(cls, '_cineduplex_native_create', cls.create_incremental_stream_session)
    cls._cineduplex_native_create = original

    def create(fw, *args, **kwargs):
        # A multiplicative logit factor is not a calibrated probability
        # threshold. With positive SS logits it can reverse native KL/SS
        # ordering before the user finishes their question. Preserve the
        # model's unweighted listening decision on every admitted rank.
        kwargs['start_speak_token_factor'] = 1.0
        if native_prefix:
            from lychee_fd.runtime import vllm_generation
            vllm_generation.SYSTEM_MESSAGE_PREFIX = '<|BOT|>system\nYou are a helpful assistant.<|EOT|>'
        # Use the upstream conditional EOT/TTS_END tail policy, instead of
        # erasing an unfinished response boundary with two PAD tokens.
        fw.is_old_sl_strategy = False
        reset_sampling = getattr(fw.model, '_cineduplex_reset_sampling', None)
        if reset_sampling is not None:
            reset_sampling()
        restore_control = getattr(fw.model, '_cineduplex_restore_native_control', None)
        if restore_control is not None:
            restore_control()
        # Preserve strong native interruptions. A marginal SL decision must
        # persist for a second control window; a single near-tie in silence
        # must not erase an otherwise valid response before its first sound.
        speaking_cls = __import__('lychee_fd.runtime.hf_v9_realtime',fromlist=['SpeakingControlLogitsProcessor']).SpeakingControlLogitsProcessor
        native_score = getattr(speaking_cls, '_cineduplex_native_score', speaking_cls.__call__)
        speaking_cls._cineduplex_native_score = native_score
        def confirmed_sl(processor, ids, scores):
            result = native_score(processor, ids, scores)
            sl,ks = processor.sl_token_id,processor.ks_token_id
            if (ids.shape[1]%processor.control_token_chunk_size==processor.control_token_chunk_size-1
                    and fw.runtime_speech_steps>fw.stoken_delay_num+fw.control_token_chunk_size):
                probability = float(processor.last_probs['sl'])
                count = getattr(processor,'_weak_sl_windows',0)
                count = count+1 if .5 < probability < .8 else 0
                processor._weak_sl_windows=count
                suppressed = .5 < probability < .8 and count < 2
                if suppressed:result[:,sl]=-float('inf')
                fw.model._realtime_interrupt_audit={'native_sl_probability':probability,
                    'weak_confirmation_windows':count,'suppressed_single_weak_sl':suppressed,
                    'strong_threshold':.8,'required_weak_windows':2}
            return result
        speaking_cls.__call__ = confirmed_sl
        session = original(fw, *args, **kwargs)
        context_install = getattr(fw.model, '_cineduplex_install_audio_context', None)
        if context_install is not None:
            context_install(session)
        advance = session.advance_stream

        def stream(current, *args, **kwargs):
            for event in advance(*args, **kwargs):
                if event.get('type') == 'state_change' and event.get('reason') == 'control_sl':
                    speech_tail = int(current.generated_stoken_ids[0, -1].item())
                    event = {**event, 'interrupt': speech_tail != int(fw.tts_end_id),
                             'interrupt_reason': 'native_control_sl',
                             'tail_policy': 'conditional-eot-tts-end'}
                elif event.get('type') == 'speaking_done' and event.get('reason') == 'eos':
                    # EOS ends both response channels. The next pad prefill
                    # must consume a closed text tail, even if acoustic EOS
                    # arrived before the text branch emitted its own EOT.
                    tail = int(current.generated_input_ids[0, -1].item())
                    if tail not in (int(fw.text_pad_token_id), int(fw.tts_pad_id), int(fw.eos_token_id)):
                        current.generated_input_ids = fw._set_last_token_safe(
                            current.generated_input_ids, fw.eos_token_id)
                yield event

        session.advance_stream = types.MethodType(stream, session)
        return session

    cls.create_incremental_stream_session = create
