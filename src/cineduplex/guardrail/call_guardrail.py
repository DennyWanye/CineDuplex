"""Bounded per-call review workers; model and microphone remain independent.

Output is held for a whole native response because native text and audio chunks
do not provide a proven word-to-audio alignment. Input audio is monitored after
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
import re
import threading
import time
import unicodedata

from .aliyun_guardrail import GuardrailError


def audio_text_consistent(expected, transcript):
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
    return bool(expected and transcript and SequenceMatcher(None, expected, transcript, autojunk=False).ratio() >= .6)


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


class CallGuardrail:
    CONTENT = frozenset(('event_start', 'text_delta', 'event_end', 'audio_start', 'audio', 'audio_end', 'response_warning'))
    MAX_PENDING_BYTES = 8 * 1024 * 1024
    MAX_EVENTS = 4096

    def __init__(self, call, client):
        self.call = call; self.client = client; self.config = client.config
        self.lock = threading.RLock(); self.condition = threading.Condition(self.lock)
        self.responses = {}; self.pending_bytes = 0; self.event_count = 0
        self.output_queue = queue.Queue(maxsize=8)
        self.input_queue = queue.Queue(maxsize=8)
        self.input_buffer = bytearray(); self.input_overlap = b''
        self.input_submitted = 0; self.input_reviewed = 0
        self.input_last_seq = -1
        self.input_context = ''
        self.input_quiet_samples = 0
        self.input_had_activity = False
        self.audio_queue = queue.Queue(maxsize=8)
        self.audit = deque(maxlen=128)
        self.closed = threading.Event(); self.failed = False
        self.output_worker = threading.Thread(target=self._output_loop, name='guardrail-output', daemon=True)
        self.input_worker = threading.Thread(target=self._input_loop, name='guardrail-input', daemon=True)
        self.audio_worker = threading.Thread(target=self._audio_loop, name='guardrail-audio', daemon=True)
        self.output_worker.start()
        if self.config.input_audio:
            self.input_worker.start()
        if self.config.output_audio:
            self.audio_worker.start()

    def _record(self, direction, decision=None, error=None, event_id=None, provider='aliyun'):
        row = {'type': 'guardrail_result', 'direction': direction, 'event_id': event_id,
               'provider': provider, 'policy': 'pass-only',
               **(decision.record() if decision is not None else {'allowed': False, 'error': error})}
        self.audit.append(row)
        self.call._emit(row)

    def _deny(self, direction, code, event_id=None, provider='application'):
        with self.lock:
            if self.failed or self.closed.is_set() or self.call.closed.is_set():
                return
            self.failed = True
            self._record(direction, error=code, event_id=event_id, provider=provider)
            for pending in self.responses.values(): pending.cancelled.set()
            self.responses.clear(); self.pending_bytes = 0; self.event_count = 0
            self.input_buffer.clear(); self.input_overlap = b''
            self.call.epoch += 1
            self.call._emit({'type': 'stop_audio', 'epoch': self.call.epoch, 'reason': 'guardrail'})
            self.call._emit({'type': 'guardrail_block', 'direction': direction,
                             'message': ('这次通话的内容未通过安全审核，通话已停止。' if code == 'risk_detected'
                                         else '安全审核暂时无法完成，通话已停止，请稍后重试。')})
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
        try:
            self.input_queue.put_nowait((self.input_submitted, self.input_overlap + new))
        except queue.Full:
            self._deny('input_audio', 'review_backlog')
            return self.input_submitted
        self.input_overlap = new[-32000:]
        return self.input_submitted

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
            samples = array('h'); samples.frombytes(pcm)
            for offset in range(0, len(samples), 320):
                frame = samples[offset:offset + 320]
                active = sum(x*x for x in frame) > len(frame) * 64**2
                if active:
                    self.input_had_activity = True; self.input_quiet_samples = 0
                else:
                    self.input_quiet_samples += len(frame)
            if (len(self.input_buffer) >= self.config.input_window_seconds * 32000
                    or self.input_had_activity and self.input_quiet_samples >= 6400):
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
                                 'message': '正在生成回答，完成并通过安全审核后播放。'})
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
                if self.config.output_audio and pending.pcm and not pending.audio_review_started:
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
                    self._record('input_audio', decision)
                    if not decision.allowed:
                        self._deny('input_audio', 'risk_detected'); return
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
                        self._record('input_semantic', semantic)
                        if not semantic.allowed:
                            self._deny('input_semantic', 'risk_detected'); return
                    self.input_context = text[-1000:]
                with self.lock:
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
                        self._deny('output_audio', 'risk_detected', pending.event_id); return
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
                        self._deny('output_text', 'risk_detected', pending.event_id); return
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
            for tasks in (self.input_queue, self.output_queue, self.audio_queue):
                while True:
                    try:tasks.get_nowait()
                    except queue.Empty:break
        for worker in (self.output_worker, self.input_worker, self.audio_worker):
            if worker.is_alive() and worker is not threading.current_thread():
                worker.join(timeout=self.config.timeout + .5)
