"""Bounded per-call review workers; model and microphone remain independent.

Output is held for a whole native response because native text and audio chunks
do not provide a proven word-to-audio alignment. Input audio is monitored after
ingress, not represented as a pre-model prompt filter. Optional audio detection
submits in-memory WAVs to Aliyun's temporary upload storage.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, field
import queue
import hashlib
import threading
import time

from .aliyun_guardrail import GuardrailError


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
        self.audit = deque(maxlen=128)
        self.closed = threading.Event(); self.failed = False
        self.output_worker = threading.Thread(target=self._output_loop, name='guardrail-output', daemon=True)
        self.input_worker = threading.Thread(target=self._input_loop, name='guardrail-input', daemon=True)
        self.output_worker.start()
        if self.config.input_audio:
            self.input_worker.start()

    def _record(self, direction, decision=None, error=None, event_id=None):
        row = {'type': 'guardrail_result', 'direction': direction, 'event_id': event_id,
               'provider': 'aliyun', 'policy': 'pass-only',
               **(decision.record() if decision is not None else {'allowed': False, 'error': error})}
        self.audit.append(row)
        self.call._emit(row)

    def _deny(self, direction, code, event_id=None):
        with self.lock:
            if self.failed or self.closed.is_set() or self.call.closed.is_set():
                return
            self.failed = True
            self._record(direction, error=code, event_id=event_id)
            self.responses.clear(); self.pending_bytes = 0; self.event_count = 0
            self.input_buffer.clear(); self.input_overlap = b''
            self.call.epoch += 1
            self.call._emit({'type': 'stop_audio', 'epoch': self.call.epoch, 'reason': 'guardrail'})
            self.call._emit({'type': 'guardrail_block', 'direction': direction,
                             'message': '这次通话的内容未通过安全审核，通话已停止。'})
            self.call.stop('guardrail_blocked' if code == 'risk_detected' else 'guardrail_unavailable')
            self.closed.set(); self.condition.notify_all()

    def _queue_input(self):
        # Submit all newly received samples, with one second of overlap to
        # reduce boundary loss. Fixed windows still have limited semantics.
        if not self.input_buffer:
            return self.input_submitted
        new = bytes(self.input_buffer)
        self.input_submitted += 1
        try:
            self.input_queue.put_nowait((self.input_submitted, self.input_overlap + new))
        except queue.Full:
            self._deny('input_audio', 'review_backlog')
            return self.input_submitted
        self.input_overlap = new[-32000:]
        self.input_buffer.clear()
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
            if len(self.input_buffer) >= self.config.input_window_seconds * 32000:
                self._queue_input()

    def intercept(self, event):
        """Return True when this event has been handled or intentionally held."""
        kind = event.get('type')
        if kind == 'stop_audio':
            with self.lock:
                for key, pending in list(self.responses.items()):
                    if pending.epoch < event['epoch']:
                        self._remove(key)
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
                                 'message': '正在审核回答，文字和声音将在通过后一起发送。'})
            if pending.queued:
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
                    self._remove(key); return True
                pending.text_done = True
            if kind in ('audio_end', 'response_warning'):
                pending.audio_done = True
            if (len(pending.text) > 2000 or len(pending.pcm) > 48000 * self.config.max_answer_seconds
                    or self.pending_bytes > self.MAX_PENDING_BYTES or self.event_count > self.MAX_EVENTS):
                self._deny('output', 'response_budget_exceeded', key); return True
            if pending.text_done and pending.audio_done:
                pending.input_barrier = self._queue_input() if self.config.input_audio else 0
                pending.queued = True
                try:
                    self.output_queue.put_nowait(pending)
                except queue.Full:
                    self._deny('output', 'review_backlog', key)
            return True

    def _remove(self, key):
        pending = self.responses.pop(key, None)
        if pending is not None:
            self.pending_bytes -= pending.size; self.event_count -= len(pending.events)

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
                    self.input_reviewed = index; self.condition.notify_all()
            except Exception as exc:
                self._deny('input_audio', self._error_code(exc)); return

    @staticmethod
    def _error_code(exc):
        # Only our fixed diagnostics can enter user-visible events.
        return str(exc) if isinstance(exc, GuardrailError) else 'review_failed'

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
                if self.config.output_audio and pending.pcm:
                    decision = self.client.check_audio(bytes(pending.pcm), sample_rate=24000,
                            data_id=self.call.id + '-out-' + hashlib.sha256(pending.event_id.encode()).hexdigest()[:16], cancelled=self.closed)
                    with self.lock:
                        if self.closed.is_set() or self.call.closed.is_set():
                            return
                        if self.responses.get(pending.event_id) is not pending or pending.epoch != self.call.epoch:
                            continue
                        self._record('output_audio', decision, event_id=pending.event_id)
                        if not decision.allowed:
                            self._deny('output_audio', 'risk_detected', pending.event_id); return
                deadline = time.monotonic() + self.config.audio_deadline
                with self.condition:
                    while (self.input_reviewed < pending.input_barrier and not self.closed.is_set()
                           and not self.call.closed.is_set()):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise GuardrailError('input_review_timeout')
                        self.condition.wait(min(.2, remaining))
                    if self.closed.is_set() or self.call.closed.is_set():
                        return
                    if self.responses.get(pending.event_id) is not pending or pending.epoch != self.call.epoch:
                        continue
                    self.call._emit({'type': 'guardrail_pass', 'event_id': pending.event_id,
                                     'message': '回答已通过已启用的安全审核。'})
                    for event in pending.events:
                        if self.closed.is_set() or self.call.closed.is_set() or pending.epoch != self.call.epoch:
                            break
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
            self.responses.clear(); self.pending_bytes = 0; self.event_count = 0
            self.input_buffer.clear(); self.input_overlap = b''
            self.condition.notify_all()
            for tasks in (self.input_queue, self.output_queue):
                while True:
                    try:tasks.get_nowait()
                    except queue.Empty:break
        for worker in (self.output_worker, self.input_worker):
            if worker.is_alive() and worker is not threading.current_thread():
                worker.join(timeout=self.config.timeout + .5)
