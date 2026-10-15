"""Translate native V9 turn boundaries into live playback semantics.

An acoustic EOS is normal completion. An S->L control decision while the
acoustic sequence has not ended cancels queued playback. V9's offline event
marks both as non-interrupting, which is unsuitable for a live output queue.
This adapter retains all native input, control decisions and KV history.
"""
import math
import types


class ResponseScopedProcessor:
    """Apply acoustic repetition rules to this response, preserving model KV."""
    def __init__(self, processor):
        self.processor = processor
        self.start = None

    def __call__(self, ids, scores):
        if self.start is None:
            self.start = max(0, ids.shape[1] - 1)
        return self.processor(ids[:, self.start:], scores)


class SpeakingTextVocabulary:
    """The spoken text channel contains text, EOT and TTS padding only.

    Match the existing offline generation contract. Acoustic/control IDs must
    not become the next-text embedding used by the speech merge branch.
    """
    def __init__(self, eos, pad):
        self.special = (int(eos), int(pad))

    def __call__(self, ids, scores):
        keep = scores[:, self.special].clone()
        scores[:, 151643:] = -float('inf')
        scores[:, self.special] = keep
        return scores


def install_response_scope(fw):
    original = getattr(fw, '_cineduplex_unscoped_speaking', fw.init_speaking_processor)
    fw._cineduplex_unscoped_speaking = original
    def scoped(end_factor):
        text, speech = original(end_factor)
        from cineduplex.train.native_output_contract import TerminalTextProcessor
        text = type(text)([TerminalTextProcessor(p) if all(hasattr(p,k) for k in
            ('eos_token_id','has_eos','max_token_length')) else p for p in text])
        text.append(SpeakingTextVocabulary(fw.eos_token_id, fw.tts_pad_id))
        return text, type(speech)([ResponseScopedProcessor(p) for p in speech])
    fw.init_speaking_processor = scoped


def install_listening_alignment(session):
    """Match training and the bundled vLLM runtime's listening channels.

    HF discards sampled listening words, but previously kept those words in
    the speech merge cache. Feed the same PAD channels that are retained in
    the conversation. Audio input and native control decisions stay intact.
    """
    from lychee_fd.runtime.vllm_generation import SingleTurnGenerationFramework
    native_stream = session._stream_model

    def aligned(current, **kwargs):
        if current.listening_state == 'l':
            text, speech = SingleTurnGenerationFramework.init_listening_pad_processor(current.fw)
            kwargs = {**kwargs, 'logits_processor': text, 'stoken_logits_processor': speech}
        return native_stream(**kwargs)

    session._stream_model = types.MethodType(aligned, session)


def ground_latest_request(session, request):
    """Legacy diagnostic helper; deliberately not called by the live wrapper.

    Retained for the earlier unaccepted ASR-grounding experiment only. The
    native product path does not inject reviewed transcripts into its prefix.
    """
    import json
    import torch
    key = (request['index'], request['seq_through'])
    if getattr(session, '_cineduplex_grounded_request', None) == key:
        return
    if session.listening_state != 'l':
        return
    fw = session.fw
    old_length = session.system_input_length
    if not hasattr(session, '_cineduplex_base_instruction'):
        session._cineduplex_base_instruction = fw.tokenizer.decode(
            session.prefix_input_ids[0].tolist(), skip_special_tokens=False).removesuffix('<|EOT|>')
    words = request['text']
    anchor = ('\nLatest reviewed spoken request: '
              + json.dumps(words, ensure_ascii=False).replace('<|', '\\u003c|')
              + '\nAnswer only this request. Earlier tasks ended unless explicitly resumed.') if words else (
        '\nNo reliably transcribed words. Ask one brief clarification question.')
    prefix_ids = fw.tokenizer([session._cineduplex_base_instruction + anchor + '<|EOT|>'],
                              add_special_tokens=False).input_ids[0]
    prefix = torch.tensor([prefix_ids], dtype=session.generated_input_ids.dtype,
                          device=session.generated_input_ids.device)
    suffix = session.generated_input_ids[:, old_length:]
    if len(prefix_ids) + suffix.shape[1] + 256 >= fw.model.config.text_config.max_position_embeddings:
        raise ValueError('grounded request exceeds context capacity')
    mapping = session.stoken_mapping[:, old_length:]
    delta = len(prefix_ids) - old_length
    mapping = torch.where(mapping >= 0, mapping + delta, mapping)
    session.stoken_mapping = torch.cat((mapping.new_full((1, len(prefix_ids)), -1), mapping), 1)
    session.generated_input_ids = torch.cat((prefix, suffix), 1)
    if not torch.equal(session.generated_input_ids[:, len(prefix_ids):], suffix):
        raise ValueError('grounding changed retained token history')
    session.prefix_input_ids = prefix
    session.system_input_length = len(prefix_ids)
    model = getattr(fw.model, '_model', fw.model)
    conditioning = getattr(model, '_cineduplex_merge_conditioning', None)
    if conditioning is not None:
        model._cineduplex_merge_conditioning = torch.cat((
            conditioning.new_full((1, len(prefix_ids)), -1), conditioning[:, old_length:]), 1)
    for name in ('past_key_values', 'stoken_past_key_values', 'control_past_key_values', 'merge_past_key_values'):
        setattr(session, name, None)
    fw.model._merge_cache_by_past_id.clear()
    session._cineduplex_grounded_request = key


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


def install(native_prefix=False, speech_detector=None, turn_control=None):
    from lychee_fd.runtime.hf_v9_realtime import HFRealtimeV9GenerationFramework
    cls = HFRealtimeV9GenerationFramework
    original = getattr(cls, '_cineduplex_native_create', cls.create_incremental_stream_session)
    cls._cineduplex_native_create = original

    def create(fw, *args, **kwargs):
        # One live session owns this model. Do not reuse conditioning from a
        # previous call, while retaining it across utterances in this call.
        getattr(fw.model, '_model', fw.model)._cineduplex_merge_conditioning = None
        install_response_scope(fw)
        activity = None
        if speech_detector is not None:
            from cineduplex.train.speech_activity import StreamingSpeech
            activity = StreamingSpeech(speech_detector)
        answered_utterance = 0
        response_utterance = 0
        consumed_revision = 0
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
                if (not activity.evidence.can_reply(answered_utterance)
                        or (turn_control is not None and not (turn_control.get('caught_up') and turn_control.get('quiet')))):
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
        install_listening_alignment(session)
        context_install = getattr(fw.model, '_cineduplex_install_audio_context', None)
        if context_install is not None:
            context_install(session)
        advance = session.advance_stream
        boundary = TurnBoundary(fw.tts_end_id, fw.eos_token_id)

        def stream(current, *args, **kwargs):
            nonlocal answered_utterance, response_utterance, consumed_revision
            # Finish the in-flight 400ms collective, then take the same native
            # S->L tail transition on every rank before processing backlog.
            # Preserve all audio and KV history; never answer an older queued
            # pause while the latest captured question has not been consumed.
            revision=(turn_control or {}).get('revision',0)
            if revision > consumed_revision:
                consumed_revision=revision
                if current.listening_state in ('s','b'):
                    previous=current.listening_state.upper()
                    current.generated_input_ids,current.generated_stoken_ids=fw._apply_s2l_tail_fallback(
                        current.generated_input_ids,current.generated_stoken_ids)
                    current.generated_control_ids=fw._set_last_token_safe(current.generated_control_ids,fw.sl_token_id)
                    current.listening_state='l'
                    current._set_last_ss_pos(None)
                    yield {'type':'state_change','from':previous,'to':'L','interrupt':True,
                           'reason':'ingress_barge_in','chunk':current.next_chunk_pos,'revision':revision}
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
            # Keep the native waveform/semantic stream. Safety ASR is never
            # injected into the model prefix or used to pause computation;
            # reviewed output still waits at CallGuardrail's release barrier.
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
