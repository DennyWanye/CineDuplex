"""Bounded per-call review workers; model and microphone remain independent.

Whole-response mode holds both channels. Reviewed-prefix mode audits cumulative
native PCM and its actual ASR, then releases only that reviewed byte range. It
does not assume word-to-audio alignment. Input audio is monitored after
ingress, not represented as a pre-model prompt filter. Optional audio detection
submits in-memory WAVs to Aliyun's temporary upload storage.
"""
from __future__ import annotations

import base64
from array import array
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import queue
import hashlib
import json
import re
import threading
import time
import unicodedata

from .aliyun_guardrail import GuardrailError


def audio_text_alignment(expected, transcript, *, prefix=False):
    """Bounded mismatch heuristic, not a sexual-sound classifier/confidence.

    ASR can hallucinate words on tones or noise. A nonempty string alone must
    not make unrelated audio safe. Normalize punctuation, case and digits;
    keep both texts private and reject a substantial spoken/text mismatch.
    """
    def normalized(text):
        text = unicodedata.normalize('NFKC', text).casefold()
        text = text.translate(str.maketrans('0123456789', '零一二三四五六七八九'))
        return re.sub(r'[\W_]+', '', text)
    expected = normalized(expected); transcript = normalized(transcript)
    expected_chars = len(expected)
    if prefix:
        # Native text may be ahead of the codec. Compare only leading text;
        # an unrelated substring elsewhere must not justify this PCM prefix.
        expected = expected[:len(transcript)]
    matcher = SequenceMatcher(None, expected, transcript, autojunk=False)
    ratio = matcher.ratio() if expected and transcript else 0.0
    threshold = .6 if prefix else .8
    return {'consistent': bool(expected and transcript and ratio >= threshold),
            'similarity': round(ratio, 4), 'expected_chars': expected_chars,
            'compared_chars': len(expected), 'transcript_chars': len(transcript),
            'matched_chars': sum(m.size for m in matcher.get_matching_blocks()),
            'prefix_comparison': prefix, 'minimum_similarity': threshold}


def audio_text_consistent(expected, transcript, *, prefix=False):
    return audio_text_alignment(expected, transcript, prefix=prefix)['consistent']


def speech_progress_can_release(pending, alignment, pcm_end, *, final):
    """Hold an expanding PCM prefix when recognized expected words stagnate.

    This is an additional release condition, not a noise classifier. A final
    complete match can contain a genuine trailing pause. Existing cloud and
    text/audio checks must already have passed before this function is used.
    """
    return (final or alignment['matched_chars'] > pending.prefix_matched_chars
            or pcm_end - pending.prefix_progress_end < 48000 * 3.2)


def refusal_observation(text):
    """Text evidence only; the native model has no structured refusal signal.

    A match does not certify a safe answer, and no match does not prove assent.
    Never include source content in this metadata or use it to bypass review.
    """
    patterns = (
        ('zh_explicit', r'(?:不能|无法|不会|不可以|不允许)(?:为你|帮你|向你)?(?:帮助|协助|提供|教授|教你|满足|执行)'),
        ('en_explicit', r"\b(?:i|we)\s+(?:cannot|can't|will not|won't)\s+(?:help|assist|provide|comply)\b"),
    )
    matches = [name for name, pattern in patterns if re.search(pattern, text, re.I)]
    return {'response_origin': 'model', 'refusal_signal': 'expression_detected' if matches else 'unknown',
            'refusal_detection': 'text_heuristic_v1', 'refusal_rule_ids': matches,
            'safety_verdict': 'not_assessed',
            'response_sha256': hashlib.sha256(text.encode()).hexdigest()}


@dataclass
class PendingResponse:
    event_id: str
    epoch: int
    started: float = field(default_factory=time.monotonic)
    events: list = field(default_factory=list)
    text: str = ''
    pcm: bytearray = field(default_factory=bytearray)
    text_done: bool = False
    audio_done: bool = False
    queued: bool = False
    input_barrier: int = 0
    size: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event)
    audio_review_done: bool = False
    audio_review_started: bool = False
    audio_transcript: str = field(default='', repr=False)
    prefix_busy: int = 0
    prefix_checked: int = 0
    prefix_released: int = 0
    prefix_index: int = 0
    prefix_started: bool = False
    prefix_completed: int = 0
    prefix_final_queued: bool = False
    prefix_submitted_text: str = field(default='', repr=False)
    prefix_matched_chars: int = 0
    prefix_progress_end: int = 0


@dataclass
class PrefixTextReview:
    pending: PendingResponse
    text: str = field(repr=False)
    coverage: dict
    queued_at: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False


class CallGuardrail:
    CONTENT = frozenset(('event_start', 'text_delta', 'event_end', 'audio_start', 'audio', 'audio_end', 'response_warning'))
    MAX_PENDING_BYTES = 8 * 1024 * 1024
    MAX_EVENTS = 4096

    def __init__(self, call, client):
        self.call = call; self.client = client; self.config = client.config
        self.prefix_mode = self.config.output_mode == 'reviewed-prefix'
        if self.prefix_mode and not self.config.output_audio:
            raise GuardrailError('prefix_review_requires_audio_detection')
        self.lock = threading.RLock(); self.condition = threading.Condition(self.lock)
        self.responses = {}; self.pending_bytes = 0; self.event_count = 0
        self.output_queue = queue.Queue(maxsize=8)
        self.input_queue = queue.Queue(maxsize=8)
        self.input_buffer = bytearray(); self.input_overlap = b''
        self.input_submitted = 0; self.input_reviewed = 0
        self.input_last_seq = -1
        self.input_context = ''
        self.input_windows = {}
        self.latest_approved_input = None
        self.approved_inputs = deque(maxlen=32)
        self.input_quiet_samples = 0
        self.input_had_activity = False
        self.audio_queue = queue.Queue(maxsize=8)
        self.prefix_text_queue = queue.Queue(maxsize=8)
        self.audit = deque(maxlen=128)
        self.closed = threading.Event(); self.failed = False
        self.output_worker = threading.Thread(target=self._prefix_loop if self.prefix_mode else self._output_loop,
                                             name='guardrail-output', daemon=True)
        self.input_worker = threading.Thread(target=self._input_loop, name='guardrail-input', daemon=True)
        self.audio_worker = threading.Thread(target=self._audio_loop, name='guardrail-audio', daemon=True)
        self.output_worker.start()
        self.prefix_workers = []
        if self.prefix_mode:
            # One bounded text worker per cumulative audio review. Independent
            # provider requests overlap; neither receipt alone permits release.
            for index in range(3):
                worker = threading.Thread(target=self._prefix_text_loop,
                    name='guardrail-prefix-text-'+str(index), daemon=True)
                worker.start(); self.prefix_workers.append(worker)
            for index in range(2):
                worker = threading.Thread(target=self._prefix_loop, name='guardrail-prefix-'+str(index), daemon=True)
                worker.start(); self.prefix_workers.append(worker)
        if self.config.input_audio:
            self.input_worker.start()
        if self.config.output_audio and not self.prefix_mode:
            self.audio_worker.start()

    def _record(self, direction, decision=None, error=None, event_id=None, provider='aliyun', input_index=None, **coverage):
        row = {'type': 'guardrail_result', 'direction': direction, 'event_id': event_id,
               'provider': provider, 'policy': 'pass-only', 'call_id': self.call.id,
               'input_index': input_index,
               **(decision.record() if decision is not None else {'allowed': False, 'error': error}), **coverage}
        if direction.startswith('input'):
            row.update(input_submitted=self.input_submitted,input_reviewed=self.input_reviewed,
                       input_queue_depth=self.input_queue.qsize(),input_buffer_bytes=len(self.input_buffer))
        self.audit.append(row)
        self.call._emit(row)

    def _deny(self, direction, code, event_id=None, provider='application', cause=None, input_index=None):
        with self.lock:
            if self.failed or self.closed.is_set() or self.call.closed.is_set():
                return
            self.failed = True
            self._record(direction, error=code, event_id=event_id, provider=provider, input_index=input_index)
            source = 'aliyun' if cause is not None and not cause.allowed else 'application'
            message = ('阿里云拦截了本次内容，通话已停止。' if source == 'aliyun'
                       else '本地防护因审核异常或一致性检查停止通话，请查看日志。')
            if source == 'application' and code in ('output_audio_text_mismatch', 'output_audio_semantics_unrecognized'):
                message = '这段回答的声音无法与文字核对，已停止本次通话。请重新开始。'
            self.call._emit({'type': 'safety_outcome', 'schema': 1, 'call_id': self.call.id,
                             'event_id': event_id, 'input_index': input_index,
                             'outcome': 'aliyun_block' if source == 'aliyun' else 'application_stop',
                             'decision_source': source, 'direction': direction, 'reason': code,
                             'request_id': cause.request_id if cause is not None else None,
                             'labels': list(cause.labels) if cause is not None else [],
                             'evidence': 'provider_receipt' if source == 'aliyun' else 'local_policy',
                             'message': message})
            for pending in self.responses.values(): pending.cancelled.set()
            self.responses.clear(); self.pending_bytes = 0; self.event_count = 0
            self.input_buffer.clear(); self.input_overlap = b''
            self.call.epoch += 1
            self.call._emit({'type': 'stop_audio', 'epoch': self.call.epoch, 'reason': 'guardrail'})
            self.call._emit({'type': 'guardrail_block', 'direction': direction,
                             'decision_source': source, 'reason': code,
                             'request_id': cause.request_id if cause is not None else None,
                             'message': message})
            self.call.stop('guardrail_blocked' if code == 'risk_detected' else 'guardrail_unavailable')
            self.closed.set(); self.condition.notify_all()

    def _queue_input(self):
        # Only digital zero may be omitted. Quiet/non-speech audio is still
        # reviewed: speech VAD is not a safety classifier.
        if not self.input_buffer:
            return self.input_submitted
        new = bytes(self.input_buffer)
        self.input_buffer.clear()
        self.input_had_activity = False; self.input_quiet_samples = 0
        if not any(new):
            return self.input_submitted
        self.input_submitted += 1
        self.input_windows[self.input_submitted] = self.input_last_seq
        try:
            self.input_queue.put_nowait((self.input_submitted, self.input_overlap + new))
        except queue.Full:
            self._deny('input_audio', 'review_backlog')
            return self.input_submitted
        self.input_overlap = new[-32000:]
        return self.input_submitted

    def approved_input(self, speech_start_seq, speech_end_seq=None):
        """Private model grounding only after audio and request-semantic pass.

        A receipt from the preceding utterance must never authorize a new one.
        Text is deliberately absent from public diagnostic events.
        """
        with self.lock:
            row = self.latest_approved_input
            if self.failed or self.closed.is_set() or row is None:
                return None
            if row['seq_through'] < speech_start_seq:
                return None
            if speech_end_seq is None:
                if self.input_reviewed < self.input_submitted: return None
            else:
                # Later background windows still undergo safety review, but
                # they are not new words belonging to this utterance. Output
                # release retains its separate complete input-review barrier.
                row = next((p for p in self.approved_inputs
                            if p['seq_through'] >= speech_end_seq), None)
                if (row is None and self.latest_approved_input['index'] == self.input_submitted
                        and getattr(self, 'input_last_seq', -1) >= speech_end_seq
                        and hasattr(self, 'input_buffer') and not any(self.input_buffer)):
                    # _queue_input omits only digital zero. Such a quiet tail
                    # has no new words or sound to moderate, but the VAD end
                    # can fall after the last nonzero provider snapshot. Bind
                    # this exact utterance end once, without extending the
                    # cloud receipt or re-grounding on every later zero packet.
                    row = {**self.latest_approved_input, 'seq_through': speech_end_seq,
                           'digital_zero_tail_from': self.latest_approved_input['seq_through'] + 1}
                if row is None: return None
            words = ''
            last_text_index = row['index']
            for part in self.approved_inputs:
                if part['index'] > row['index']: break
                if part['seq_through'] < speech_start_seq or not part['text']:
                    continue
                new = part['text']; last_text_index = part['index']
                # The audio reviewer includes a one-second overlap. Do not
                # duplicate an exactly repeated word boundary; never invent a
                # correction when the two ASR strings disagree.
                overlap = next((n for n in range(min(len(words),len(new)), 0, -1)
                                if words.endswith(new[:n])), 0)
                words += new[overlap:]
            return {**row, 'index': last_text_index, 'text': words}

    def input_audio(self, seq, pcm):
        if not self.config.input_audio:
            return
        with self.lock:
            if self.closed.is_set() or self.failed or self.call.closed.is_set():
                return
            # Upstream retries must not duplicate charged audio detection.
            if seq <= self.input_last_seq:
                return
            self.input_last_seq = seq
            self.input_buffer.extend(pcm)
            # A quiet tail submits early while generation continues. Loudness
            # affects batching only; it never discards unreviewed samples.
            # While a provider job is waiting, keep quiet-tail fragments in
            # the bounded window. Response barriers still flush all bytes.
            samples = array('h'); samples.frombytes(pcm)
            for offset in range(0, len(samples), 320):
                frame = samples[offset:offset + 320]
                active = sum(x*x for x in frame) > len(frame) * 64**2
                if active:
                    self.input_had_activity = True; self.input_quiet_samples = 0
                else:
                    self.input_quiet_samples += len(frame)
            if (len(self.input_buffer) >= self.config.input_window_seconds * 32000
                    or self.input_had_activity and self.input_quiet_samples >= 6400
                    and self.input_queue.empty()):
                self._queue_input()

    def intercept(self, event):
        """Return True when this event has been handled or intentionally held."""
        kind = event.get('type')
        if kind == 'stop_audio':
            with self.lock:
                for key, pending in list(self.responses.items()):
                    if pending.epoch < event['epoch']:
                        self._cancel(key)
            return False
        if kind not in self.CONTENT:
            return False
        with self.lock:
            if self.closed.is_set() or self.failed or self.call.closed.is_set():
                return True
            event_id = event.get('event_id')
            if event_id is None:
                self._deny('output', 'response_identity_missing')
                return True
            key = str(event_id)
            epoch = event.get('epoch', self.call.epoch)
            if epoch != self.call.epoch:
                return True
            pending = self.responses.get(key)
            if pending is None:
                if len(self.responses) >= 8:
                    self._deny('output', 'review_backlog'); return True
                pending = PendingResponse(key, epoch)
                self.responses[key] = pending
                self.call._emit({'type': 'guardrail_pending', 'event_id': event_id,
                                 'phase': 'generating',
                                 'message': ('正在生成回答，每段声音审核通过后播放。' if self.prefix_mode
                                             else '正在生成回答，完成并通过安全审核后播放。')})
            if ((pending.queued and kind not in ('audio_start', 'audio', 'audio_end', 'response_warning'))
                    or (pending.audio_done and kind in ('audio_start', 'audio', 'audio_end', 'response_warning'))):
                self._deny('output', 'content_after_response_end', key); return True
            value = dict(event)
            size = len(str(value).encode())
            pending.size += size; self.pending_bytes += size; self.event_count += 1
            pending.events.append(value)
            if kind in ('text_delta', 'event_end'):
                if isinstance(event.get('snapshot'), str):
                    pending.text = event['snapshot']
                elif kind == 'event_end' and isinstance(event.get('text'), str):
                    pending.text = event['text']
                elif isinstance(event.get('text'), str):
                    pending.text += event['text']
            if kind == 'audio':
                try:
                    chunk = base64.b64decode(event['pcm'], validate=True)
                    if len(chunk) % 2 or event.get('sample_rate') != 24000:
                        raise ValueError()
                    pending.pcm.extend(chunk)
                except (ValueError, KeyError, TypeError):
                    self._deny('output', 'invalid_audio', key); return True
            if kind == 'event_end':
                if event.get('interrupt'):
                    self._cancel(key); return True
                pending.text_done = True
            if kind in ('audio_end', 'response_warning'):
                pending.audio_done = True
                if self.config.output_audio and not self.prefix_mode and pending.pcm and not pending.audio_review_started:
                    pending.audio_review_started = True
                    try: self.audio_queue.put_nowait(pending)
                    except queue.Full:
                        self._deny('output_audio', 'review_backlog', key); return True
                else:
                    pending.audio_review_done = True
                self.condition.notify_all()
            if (len(pending.text) > 2000 or len(pending.pcm) > 48000 * self.config.max_answer_seconds
                    or self.pending_bytes > self.MAX_PENDING_BYTES or self.event_count > self.MAX_EVENTS):
                self._deny('output', 'response_budget_exceeded', key); return True
            if self.prefix_mode:
                self._queue_prefix(pending)
                return True
            if pending.text_done and not pending.queued:
                pending.input_barrier = self._queue_input() if self.config.input_audio else 0
                pending.queued = True
                try:
                    self.call._emit({'type': 'guardrail_pending', 'event_id': event_id,
                                     'phase': 'reviewing',
                                     'message': '整段回答文字已生成，正在进行阿里云安全审核。'})
                    self.output_queue.put_nowait(pending)
                except queue.Full:
                    self._deny('output', 'review_backlog', key)
            return True

    def _queue_prefix(self, pending):
        """At most three reviews in flight; coalesce while cloud is busy.

        Repeated cumulative checks retain preceding semantic/acoustic context.
        Snapshot bytes and text under the lock; never extend a provider receipt
        to audio that arrived after the submitted snapshot.
        """
        final = pending.text_done and pending.audio_done
        if pending.prefix_busy >= 3 or pending.cancelled.is_set() or pending.prefix_final_queued:
            return
        end = len(pending.pcm)
        if final and end == pending.prefix_checked and pending.prefix_busy and pending.text == pending.prefix_submitted_text:
            # The final codec event often has no new bytes. The last in-flight
            # snapshot already covers this complete text/PCM: do not upload the
            # identical WAV a second time. Its worker performs final consistency.
            pending.prefix_final_queued = True
            return
        # Keep the first two reviews small to fill the initial playback runway.
        # Later cumulative uploads coalesce two
        # native blocks, retaining all context while reducing repeated uploads.
        window = self.config.output_window_seconds if pending.prefix_index < 2 else max(2.4, self.config.output_window_seconds)
        if not final and (end - pending.prefix_checked < 48000 * window
                          or not pending.text.strip()):
            return
        if not final and pending.prefix_index >= 48:
            self._deny('output', 'review_backlog', pending.event_id); return
        # Seal the input belonging to this output snapshot before cloud review.
        # Later microphone samples cannot alter already generated PCM.
        pending.input_barrier = self._queue_input() if self.config.input_audio else 0
        input_barrier = pending.input_barrier
        input_seq_through = self.input_last_seq
        pending.prefix_busy += 1; pending.prefix_index += 1
        pending.prefix_checked = end
        pending.prefix_submitted_text = pending.text
        if final: pending.prefix_final_queued = True
        job = (pending, bytes(pending.pcm), pending.text, final, pending.prefix_index, input_barrier, input_seq_through)
        try: self.output_queue.put_nowait(job)
        except queue.Full: self._deny('output', 'review_backlog', pending.event_id)

    def _live(self, pending):
        return (not self.closed.is_set() and not self.call.closed.is_set()
                and self.responses.get(pending.event_id) is pending
                and not pending.cancelled.is_set() and pending.epoch == self.call.epoch)

    def _prefix_order(self, pending, index):
        deadline = time.monotonic() + self.config.audio_deadline
        while self._live(pending) and pending.prefix_completed != index - 1:
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise GuardrailError('prefix_order_timeout')
            self.condition.wait(min(.2, remaining))
        return self._live(pending)

    def _prefix_finished(self, pending, index):
        pending.prefix_busy -= 1; pending.prefix_completed = index
        self.condition.notify_all()
        self._queue_prefix(pending)

    def _prefix_text_loop(self):
        while not self.closed.is_set() and not self.call.closed.is_set():
            try: job = self.prefix_text_queue.get(timeout=.2)
            except queue.Empty: continue
            pending = job.pending
            started = time.monotonic()
            try:
                with self.lock:
                    if not self._live(pending): continue
                decision = self.client.check_text(job.text, direction='output', session_id=self.call.id,
                    chat_id=pending.event_id+'-prefix-'+str(job.coverage['review_index']))
                with self.lock:
                    if not self._live(pending): continue
                    self._record('output_text', decision, event_id=pending.event_id, **job.coverage,
                        review_wall_ms=round((time.monotonic()-started)*1000,1),
                        review_queue_ms=round((started-job.queued_at)*1000,1),
                        independent_reviews_parallel=True)
                    job.allowed = decision.allowed
                    if not decision.allowed:
                        self._deny('output_text', 'risk_detected', pending.event_id, cause=decision)
            except Exception as exc:
                with self.lock:
                    if self._live(pending): self._deny('output_text', self._error_code(exc), pending.event_id)
            finally:
                with self.condition:
                    job.done.set(); self.condition.notify_all()

    def _prefix_text_join(self, job):
        deadline = job.queued_at + self.config.audio_deadline
        with self.condition:
            while self._live(job.pending) and not job.done.is_set():
                remaining = deadline-time.monotonic()
                if remaining <= 0: raise GuardrailError('output_text_review_timeout')
                self.condition.wait(min(.2, remaining))
            return self._live(job.pending) and job.done.is_set() and job.allowed

    def _prefix_loop(self):
        while not self.closed.is_set() and not self.call.closed.is_set():
            try: pending, pcm, text, final, index, barrier, barrier_seq = self.output_queue.get(timeout=.2)
            except queue.Empty:
                self._expire(); continue
            with self.lock:
                if not self._live(pending): continue
            coverage = {'review_index': index, 'pcm_end': len(pcm),
                        'pcm_sha256': hashlib.sha256(pcm).hexdigest(), 'response_complete': final}
            try:
                text_job = PrefixTextReview(pending, text, dict(coverage))
                try: self.prefix_text_queue.put_nowait(text_job)
                except queue.Full: raise GuardrailError('review_backlog')
                transcript = text
                if pcm:
                    audio_started = time.monotonic()
                    audio = self.client.check_audio(pcm, sample_rate=24000,
                        data_id=self.call.id + '-prefix-' + hashlib.sha256(pending.event_id.encode()).hexdigest()[:12] + '-' + str(index),
                        cancelled=pending.cancelled)
                    with self.lock:
                        if not self._live(pending): continue
                        self._record('output_audio', audio, event_id=pending.event_id, **coverage,
                            review_wall_ms=round((time.monotonic()-audio_started)*1000,1),
                            independent_reviews_parallel=True)
                        if not audio.allowed:
                            self._deny('output_audio', 'risk_detected', pending.event_id, cause=audio); continue
                    transcript = audio.transcript
                    if transcript.strip():
                        semantic = self.client.check_text(transcript, direction='output', session_id=self.call.id,
                                                          chat_id=pending.event_id + '-spoken-' + str(index))
                        with self.lock:
                            if not self._live(pending): continue
                            self._record('output_audio_semantic', semantic, event_id=pending.event_id, **coverage)
                            if not semantic.allowed:
                                self._deny('output_audio_semantic', 'risk_detected', pending.event_id, cause=semantic); continue
                    if not self._prefix_text_join(text_job): continue
                    with self.lock:
                        final = final or (pending.text_done and pending.audio_done
                            and len(pcm) == len(pending.pcm) and text == pending.text
                            and index == pending.prefix_index)
                        coverage['response_complete'] = final
                    alignment = audio_text_alignment(text, transcript, prefix=not final)
                    with self.lock:
                        if not self._live(pending): continue
                        self.call._emit({'type': 'audio_text_alignment', 'event_id': pending.event_id,
                            **coverage, **alignment, 'audio_seconds': round(len(pcm)/48000, 3)})
                    if not alignment['consistent']:
                        if final:
                            raise GuardrailError('output_audio_text_mismatch' if transcript.strip()
                                                 else 'output_audio_semantics_unrecognized')
                        # A short/partial word may be untranscribable. No release;
                        # wait for a larger prefix, never waive this protection.
                        with self.condition:
                            if not self._prefix_order(pending, index): continue
                            self.call._emit({'type': 'guardrail_pending', 'event_id': pending.event_id,
                                             'phase': 'coalescing', 'message': '短段声音尚无法确认，继续收集后审核。'})
                            self._prefix_finished(pending, index)
                        continue
                    with self.condition:
                        if not self._prefix_order(pending, index): continue
                        if not speech_progress_can_release(pending, alignment, len(pcm), final=final):
                            self.call._emit({'type':'guardrail_pending','event_id':pending.event_id,
                                'phase':'speech_progress','matched_chars':alignment['matched_chars'],
                                'pcm_end':len(pcm),'message':'正在确认后续声音，暂缓播放。'})
                            self._prefix_finished(pending, index)
                            continue
                elif not self._prefix_text_join(text_job):
                    continue
                deadline = time.monotonic() + self.config.audio_deadline
                with self.condition:
                    if not self._prefix_order(pending, index): continue
                    # This immutable boundary was captured with the PCM.
                    # Review input and output concurrently; do not append
                    # unrelated later microphone samples after cloud review.
                    if self.input_reviewed < barrier:
                        self.call._emit({'type':'guardrail_pending','event_id':pending.event_id,
                            'phase':'input_review','input_barrier':barrier,'input_seq_through':barrier_seq,
                            'message':'正在审核本段回答对应的输入声音。'})
                    while self._live(pending) and self.input_reviewed < barrier:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0: raise GuardrailError('input_review_timeout')
                        self.condition.wait(min(.2, remaining))
                    if not self._live(pending): continue
                    start = pending.prefix_released
                    self.call._emit({'type': 'guardrail_pass', 'event_id': pending.event_id,
                        'reviewed_prefix': True, 'response_complete': final, **coverage,
                        'released_start': start, 'released_end': len(pcm),
                        'held_ms': round((time.monotonic()-pending.started)*1000,1),
                        'input_barrier': barrier, 'input_seq_through': barrier_seq, 'input_reviewed': self.input_reviewed,
                        'message': '本段声音与文字审核通过，继续收音。'})
                    if not pending.prefix_started:
                        for kind in ('event_start', 'audio_start'):
                            original = next((e for e in pending.events if e['type'] == kind),
                                            {'type': kind, 'event_id': pending.event_id, 'epoch': pending.epoch})
                            self.call._emit({**original, 'streaming': True, 'reviewed_prefix': True,
                                             'response_complete': False, 'sample_rate': 24000})
                        pending.prefix_started = True
                    # Captions describe reviewed speech, not unaligned future text.
                    self.call._emit({'type': 'text_delta', 'event_id': pending.event_id, 'epoch': pending.epoch,
                                     'snapshot': transcript, 'reviewed_prefix': True,
                                     'caption_source': 'reviewed_audio_transcript'})
                    if len(pcm) > start:
                        self.call._emit({'type': 'audio', 'event_id': pending.event_id, 'epoch': pending.epoch,
                            'sample_rate': 24000, 'pcm': base64.b64encode(pcm[start:]).decode(),
                            'samples': (len(pcm)-start)//2, 'sample_offset': start//2,
                            'chunk_seq': index, 'final_chunk': final, 'reviewed_prefix': True,
                            'review_index': index, 'approved_pcm_end': len(pcm)})
                    pending.prefix_released = len(pcm)
                    if pcm and alignment['matched_chars'] > pending.prefix_matched_chars:
                        pending.prefix_matched_chars = alignment['matched_chars']
                        pending.prefix_progress_end = len(pcm)
                    if final:
                        observation = refusal_observation(text)
                        self.call._emit({'type': 'safety_outcome', 'schema': 1, 'call_id': self.call.id,
                            'event_id': pending.event_id, 'decision_source': 'model',
                            'outcome': ('model_refusal_expression' if observation['refusal_signal']=='expression_detected'
                                        else 'model_response'), 'delivery': 'approved_for_release',
                            'reviewed_prefix': True, **observation,
                            'message': '本轮累计文字与声音审核通过。'})
                        self.call._emit({'type': 'event_end', 'event_id': pending.event_id,
                                         'epoch': pending.epoch, 'text': transcript,
                                         'caption_source': 'reviewed_audio_transcript'})
                        for event in pending.events:
                            if event['type'] in ('response_warning', 'audio_end'): self.call._emit(event)
                        self._remove(pending.event_id)
                    else:
                        self._prefix_finished(pending, index)
            except Exception as exc:
                with self.lock:
                    if self._live(pending): self._deny('output', self._error_code(exc), pending.event_id)

    def _remove(self, key):
        pending = self.responses.pop(key, None)
        if pending is not None:
            pending.cancelled.set()
            self.pending_bytes -= pending.size; self.event_count -= len(pending.events)
            self.condition.notify_all()

    def _cancel(self, key):
        pending = self.responses.get(key)
        if pending is not None:
            self.call._emit({'type': 'guardrail_cancelled', 'event_id': key,
                             'reason': 'native_interrupt',
                             'held_ms': round((time.monotonic() - pending.started) * 1000, 1),
                             'message': '模型中断了尚未播放的回答，正在继续聆听。'})
            self._remove(key)

    def _input_loop(self):
        while not self.closed.is_set() and not self.call.closed.is_set():
            try:
                index, pcm = self.input_queue.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                decision = self.client.check_audio(pcm, sample_rate=16000,
                        data_id=self.call.id + '-in-' + str(index), cancelled=self.closed)
                with self.lock:
                    if self.closed.is_set() or self.call.closed.is_set():
                        return
                    self._record('input_audio', decision, input_index=index)
                    if not decision.allowed:
                        self._deny('input_audio', 'risk_detected', cause=decision, input_index=index); return
                if decision.transcript.strip():
                    # Preserve recent request context across audio windows; do
                    # not publish ASR text in browser events or safety logs.
                    text = self.input_context + '\n' + decision.transcript
                    if len(text) > 2000:
                        raise GuardrailError('input_semantic_context_exceeds_limit')
                    semantic = self.client.check_text(text, direction='input',
                            session_id=self.call.id, chat_id='in-' + str(index))
                    with self.lock:
                        if self.closed.is_set() or self.call.closed.is_set(): return
                        self._record('input_semantic', semantic, input_index=index)
                        if not semantic.allowed:
                            self._deny('input_semantic', 'risk_detected', cause=semantic, input_index=index); return
                    self.input_context = text[-1000:]
                with self.lock:
                    seq_through = self.input_windows.pop(index, -1)
                    # Empty ASR is not authority to repeat the previous request.
                    self.latest_approved_input = {'index': index, 'seq_through': seq_through,
                        'text': decision.transcript.strip()}
                    self.approved_inputs.append(self.latest_approved_input)
                    archive = getattr(self.call, 'archive', None)
                    if archive is not None:
                        archive.write('input_review',json.dumps(
                            {'kind':'provider_approved',**self.latest_approved_input},ensure_ascii=False)+'\n')
                    self.input_reviewed = index; self.condition.notify_all()
            except Exception as exc:
                self._deny('input_audio', self._error_code(exc)); return

    @staticmethod
    def _error_code(exc):
        # Only our fixed diagnostics can enter user-visible events.
        return str(exc) if isinstance(exc, GuardrailError) else 'review_failed'

    def _audio_loop(self):
        while not self.closed.is_set() and not self.call.closed.is_set():
            try: pending = self.audio_queue.get(timeout=.2)
            except queue.Empty: continue
            if pending.cancelled.is_set(): continue
            try:
                decision = self.client.check_audio(bytes(pending.pcm), sample_rate=24000,
                        data_id=self.call.id + '-out-' + hashlib.sha256(pending.event_id.encode()).hexdigest()[:16],
                        cancelled=pending.cancelled)
                with self.condition:
                    if (self.closed.is_set() or self.call.closed.is_set()
                            or pending.cancelled.is_set() or pending.epoch != self.call.epoch): continue
                    self._record('output_audio', decision, event_id=pending.event_id)
                    if not decision.allowed:
                        self._deny('output_audio', 'risk_detected', pending.event_id, cause=decision); return
                    # Untranscribable generated audio cannot be justified by
                    # a harmless text channel. Hold it even if vendor says none.
                    if not decision.transcript.strip():
                        raise GuardrailError('output_audio_semantics_unrecognized')
                    pending.audio_transcript = decision.transcript
                    pending.audio_review_done = True
                    self.condition.notify_all()
            except Exception as exc:
                with self.lock:
                    if pending.cancelled.is_set() or pending.epoch != self.call.epoch: continue
                    self._deny('output_audio', self._error_code(exc), pending.event_id)
                return

    def _output_loop(self):
        while not self.closed.is_set() and not self.call.closed.is_set():
            try:
                pending = self.output_queue.get(timeout=.2)
            except queue.Empty:
                self._expire(); continue
            try:
                with self.lock:
                    if self.responses.get(pending.event_id) is not pending or pending.epoch != self.call.epoch:
                        continue
                decision = self.client.check_text(pending.text, direction='output',
                        session_id=self.call.id, chat_id=pending.event_id)
                with self.lock:
                    if self.closed.is_set() or self.call.closed.is_set():
                        return
                    if self.responses.get(pending.event_id) is not pending or pending.epoch != self.call.epoch:
                        continue
                    self._record('output_text', decision, event_id=pending.event_id)
                    if not decision.allowed:
                        self._deny('output_text', 'risk_detected', pending.event_id, cause=decision); return
                # Review the complete text while the codec finishes its tail.
                # No text or PCM is released until BOTH channels are complete.
                deadline = time.monotonic() + self.config.max_answer_seconds
                with self.condition:
                    while not (pending.audio_done and pending.audio_review_done):
                        if (self.closed.is_set() or self.call.closed.is_set()
                                or self.responses.get(pending.event_id) is not pending
                                or pending.epoch != self.call.epoch):
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise GuardrailError('audio_completion_timeout')
                        self.condition.wait(min(.2, remaining))
                    if (self.closed.is_set() or self.call.closed.is_set()
                            or self.responses.get(pending.event_id) is not pending
                            or pending.epoch != self.call.epoch):
                        continue
                if (self.config.output_audio and pending.pcm
                        and not audio_text_consistent(pending.text, pending.audio_transcript)):
                    raise GuardrailError('output_audio_text_mismatch')
                deadline = time.monotonic() + self.config.audio_deadline
                with self.condition:
                    while (self.input_reviewed < pending.input_barrier and not self.closed.is_set()
                           and not self.call.closed.is_set() and not pending.cancelled.is_set()):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise GuardrailError('input_review_timeout')
                        self.condition.wait(min(.2, remaining))
                    if self.closed.is_set() or self.call.closed.is_set():
                        return
                    if self.responses.get(pending.event_id) is not pending or pending.epoch != self.call.epoch:
                        continue
                    self.call._emit({'type': 'guardrail_pass', 'event_id': pending.event_id,
                                     'response_complete': True,
                                     'held_ms': round((time.monotonic() - pending.started) * 1000, 1),
                                     'input_barrier': pending.input_barrier,
                                     'input_reviewed': self.input_reviewed,
                                     'message': '回答已通过已启用的安全审核。'})
                    observation = refusal_observation(pending.text)
                    refusal = observation['refusal_signal'] == 'expression_detected'
                    self.call._emit({'type': 'safety_outcome', 'schema': 1, 'call_id': self.call.id,
                                     'event_id': pending.event_id, 'decision_source': 'model',
                                     'outcome': 'model_refusal_expression' if refusal else 'model_response',
                                     'delivery': 'approved_for_release', 'input_barrier': pending.input_barrier,
                                     'input_reviewed': self.input_reviewed, **observation,
                                     'message': ('模型回答含拒绝表达（文字判断）；已启用审核均通过，阿里未拦截本轮。'
                                                 if refusal else '模型生成回答；已启用审核均通过，未识别到明确拒绝表达。')})
                    for event in pending.events:
                        if self.closed.is_set() or self.call.closed.is_set() or pending.epoch != self.call.epoch:
                            break
                        # Both channels are complete before this worker runs.
                        # Tell the browser to collect the full approved PCM,
                        # instead of reserving time for future generation.
                        if event.get('type') == 'audio_start':
                            event = {**event, 'streaming': False, 'response_complete': True}
                        self.call._emit(event)
                    self._remove(pending.event_id)
            except Exception as exc:
                # A stale cancelled response must not terminate the new epoch.
                with self.lock:
                    if self.responses.get(pending.event_id) is not pending or pending.epoch != self.call.epoch:
                        continue
                self._deny('output', self._error_code(exc), pending.event_id); return

    def _expire(self):
        with self.lock:
            if any(time.monotonic() - item.started > self.config.max_answer_seconds + self.config.audio_deadline * 2
                   for item in self.responses.values()):
                self._deny('output', 'response_review_timeout')

    def close(self):
        self.closed.set()
        with self.condition:
            for pending in self.responses.values(): pending.cancelled.set()
            self.responses.clear(); self.pending_bytes = 0; self.event_count = 0
            self.input_buffer.clear(); self.input_overlap = b''
            self.condition.notify_all()
            self.input_context = ''
            for tasks in (self.input_queue, self.output_queue, self.audio_queue, self.prefix_text_queue):
                while True:
                    try:tasks.get_nowait()
                    except queue.Empty:break
        for worker in (self.output_worker, self.input_worker, self.audio_worker, *self.prefix_workers):
            if worker.is_alive() and worker is not threading.current_thread():
                worker.join(timeout=self.config.timeout + .5)
