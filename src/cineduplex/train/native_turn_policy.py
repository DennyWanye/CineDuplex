"""Translate native V9 turn boundaries into live playback semantics.

An acoustic EOS is normal completion. An S->L control decision while the
acoustic sequence has not ended cancels queued playback. V9's offline event
marks both as non-interrupting, which is unsuitable for a live output queue.
This adapter retains all native input, control decisions and KV history.
"""
import math
import types


class TurnBoundary:
    """Remember completion before V9 pads or replaces the channel tails.

    Input energy is diagnostic evidence, not a speech detector or authority
    to ignore a native interruption. An unfinished native SL still cancels.
    """

    def __init__(self, tts_end_id, text_end_id):
        self.tts_end_id = int(tts_end_id)
        self.text_end_id = int(text_end_id)
        self.audio_complete = False
        self.text_complete = False
        self.input_rms_dbfs = None

    def input_window(self, audio):
        count = len(audio)
        if count:
            energy = sum(float(sample) ** 2 for sample in audio) / count
            self.input_rms_dbfs = round(10 * math.log10(max(energy, 1e-12)), 1)

    def observe(self, event, speech_tail=None):
        kind = event.get('type')
        if kind == 'state_change' and event.get('from') == 'L' and event.get('to') in ('S', 'B'):
            self.audio_complete = False
            self.text_complete = False
        elif kind == 'speaking_token':
            self.audio_complete |= event.get('stoken') == self.tts_end_id
            self.text_complete |= event.get('text_token') == self.text_end_id
        elif kind == 'speaking_done' and event.get('reason') == 'eos':
            self.audio_complete = True
        if kind != 'state_change' or event.get('reason') != 'control_sl':
            return event
        # A remembered acoustic EOS is stronger evidence than a tail that
        # may now contain PAD. Text EOT alone does not prove audio finished.
        complete = self.audio_complete or speech_tail == self.tts_end_id
        native_interrupt = bool(event.get('interrupt'))
        interrupt = native_interrupt or not complete
        return {**event, 'interrupt': interrupt,
                'interrupt_reason': 'native_control_sl' if interrupt else None,
                'tail_policy': 'conditional-eot-tts-end',
                'turn_boundary': {'audio_complete': complete,
                                  'text_complete': self.text_complete,
                                  'native_interrupt': native_interrupt,
                                  'input_rms_dbfs': self.input_rms_dbfs,
                                  'decision': 'cancel_unfinished' if interrupt else 'preserve_completed'}}


def install(native_prefix=False, speech_detector=None):
    from lychee_fd.runtime.hf_v9_realtime import HFRealtimeV9GenerationFramework
    cls = HFRealtimeV9GenerationFramework
    original = getattr(cls, '_cineduplex_native_create', cls.create_incremental_stream_session)
    cls._cineduplex_native_create = original

    def create(fw, *args, **kwargs):
        activity = None
        if speech_detector is not None:
            from cineduplex.train.speech_activity import StreamingSpeech
            activity = StreamingSpeech(speech_detector)
        answered_utterance = 0
        response_utterance = 0
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
        # A native SS may fire in the middle of the user's question. Keep
        # feeding the original audio/KV history, but defer SS/BC until a
        # confirmed speech segment has ended. Do not answer silence twice.
        listening_cls = __import__('lychee_fd.runtime.hf_v9_realtime',fromlist=['ListeningControlLogitsProcessor']).ListeningControlLogitsProcessor if activity else None
        if listening_cls is not None:
            listening_score = listening_cls.__call__
            def wait_for_turn(processor, ids, scores):
                result = listening_score(processor, ids, scores)
                if not activity.evidence.can_reply(answered_utterance):
                    result[:, processor.ss_token_id] = -float('inf')
                    if processor.bc_token_id is not None:
                        result[:, processor.bc_token_id] = -float('inf')
                elif activity.evidence.reply_overdue(answered_utterance):
                    result[:, processor.kl_token_id] = -float('inf')
                    if processor.bc_token_id is not None:
                        result[:, processor.bc_token_id] = -float('inf')
                return result
            listening_cls.__call__ = wait_for_turn
        # Preserve strong native interruptions. A marginal SL decision must
        # persist for a second control window; a single near-tie in silence
        # must not erase an otherwise valid response before its first sound.
        speaking_cls = __import__('lychee_fd.runtime.hf_v9_realtime',fromlist=['SpeakingControlLogitsProcessor']).SpeakingControlLogitsProcessor
        native_score = getattr(speaking_cls, '_cineduplex_native_score', speaking_cls.__call__)
        speaking_cls._cineduplex_native_score = native_score
        def confirmed_sl(processor, ids, scores):
            if activity is not None:
                # Apply the decision before the native state machine truncates
                # its token streams. Changing only the final interrupt flag
                # would release an incomplete answer or lose its audio tail.
                pos = ids.shape[1] % processor.control_token_chunk_size
                sl, ks = processor.sl_token_id, processor.ks_token_id
                allowed = ([sl, ks] if pos == processor.control_token_chunk_size - 1
                           else [fw.detect_token_id] if pos == processor.control_token_chunk_size - 2
                           else [fw.sleep_token_id])
                values = scores[:, allowed].clone()
                scores.fill_(-float('inf')); scores[:, allowed] = values
                if len(allowed) == 2:
                    processor.last_probs = dict(zip(('sl', 'ks'), values[0].float().softmax(0).tolist()))
                    interrupted = activity.evidence.can_interrupt(response_utterance)
                    # A confirmed new user utterance can stop a real response;
                    # otherwise let its acoustic EOS finish it normally.
                    scores[:, ks if interrupted else sl] = -float('inf')
                    fw.model._realtime_interrupt_audit = {
                        'native_sl_probability': processor.last_probs['sl'],
                        'speech_probability': activity.evidence.probability,
                        'speech_active': activity.evidence.active,
                        'utterance': activity.evidence.utterance,
                        'response_utterance': response_utterance,
                        'confirmed_new_speech': interrupted,
                        'policy': 'speech-confirmed-barge-in'}
                return scores
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
        boundary = TurnBoundary(fw.tts_end_id, fw.eos_token_id)

        def stream(current, *args, **kwargs):
            nonlocal answered_utterance, response_utterance
            audio = args[0] if args else kwargs.get('audio')
            if audio is not None:
                boundary.input_window(audio)
                if activity is not None:
                    started = activity.feed(audio)
                    yield {'type': 'input_speech', 'started': started,
                           'active': activity.evidence.active,
                           'utterance': activity.evidence.utterance,
                           'speech_probability': round(activity.evidence.probability, 4),
                           'quiet_seconds': round(activity.evidence.quiet_frames * .032, 3),
                           'reply_overdue': activity.evidence.reply_overdue(answered_utterance),
                           'interrupt_playback': bool(started and answered_utterance
                                                      and current.listening_state == 'l')}
            for event in advance(*args, **kwargs):
                if (activity is not None and event.get('type') == 'state_change'
                        and event.get('from') == 'L' and event.get('to') in ('S', 'B')):
                    answered_utterance = response_utterance = activity.evidence.utterance
                if event.get('type') == 'state_change' and event.get('reason') == 'control_sl':
                    speech_tail = int(current.generated_stoken_ids[0, -1].item())
                    event = boundary.observe(event, speech_tail)
                elif event.get('type') == 'speaking_done' and event.get('reason') == 'eos':
                    boundary.observe(event)
                    # EOS ends both response channels. The next pad prefill
                    # must consume a closed text tail, even if acoustic EOS
                    # arrived before the text branch emitted its own EOT.
                    tail = int(current.generated_input_ids[0, -1].item())
                    if tail not in (int(fw.text_pad_token_id), int(fw.tts_pad_id), int(fw.eos_token_id)):
                        current.generated_input_ids = fw._set_last_token_safe(
                            current.generated_input_ids, fw.eos_token_id)
                else:
                    event = boundary.observe(event)
                yield event
            audit = getattr(fw.model, '_realtime_interrupt_audit', None)
            if activity is not None and audit is not None:
                yield {'type': 'interrupt_control', **audit}

        session.advance_stream = types.MethodType(stream, session)
        return session

    cls.create_incremental_stream_session = create
