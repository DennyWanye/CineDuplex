"""CPU-only safety checks. No cloud calls, credentials, GPU or audio files."""
import ast
import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import unittest
from urllib.parse import parse_qs
from unittest.mock import patch

from cineduplex.guardrail.aliyun_guardrail import (AliyunGuardrail, Config, Decision,
    GuardrailError, NoRedirect, parse_decision, rpc_signature)
from cineduplex.guardrail.call_guardrail import CallGuardrail, audio_text_consistent, refusal_observation


ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('worker did not finish')
        time.sleep(.005)


class FakeClient:
    def __init__(self, config=None, text_allowed=True, audio_allowed=True, text_error=None):
        self.config = config or Config(enabled=True, timeout=.1, audio_deadline=1)
        self.text_allowed = text_allowed; self.audio_allowed = audio_allowed
        self.text_error = text_error
        self.text_requests = []; self.audio_requests = []
        self.text_entered = threading.Event(); self.text_release = threading.Event(); self.text_release.set()
        self.audio_entered = threading.Event(); self.audio_release = threading.Event(); self.audio_release.set()

    def check_text(self, text, **kwargs):
        self.text_requests.append((text, kwargs)); self.text_entered.set()
        self.text_release.wait(2)
        if self.text_error:
            raise self.text_error
        return Decision(self.text_allowed, 'pass' if self.text_allowed else 'block', 'fake-text-id')

    def check_audio(self, pcm, **kwargs):
        self.audio_requests.append((pcm, kwargs)); self.audio_entered.set()
        self.audio_release.wait(2)
        return Decision(self.audio_allowed, 'pass' if self.audio_allowed else 'block', 'fake-audio-id', transcript='春天天气温暖。')


class FakeCall:
    def __init__(self):
        self.id = 'test-session'; self.epoch = 0; self.closed = threading.Event(); self.events = []; self.reason = None

    def _emit(self, event):
        self.events.append(dict(event))

    def stop(self, reason):
        self.reason = reason; self.closed.set()


def answer(gate, text='春天天气温暖。', event_id='evt-1'):
    for event in ({'type': 'event_start'}, {'type': 'audio_start', 'epoch': 0},
                  {'type': 'text_delta', 'snapshot': text},
                  {'type': 'audio', 'epoch': 0, 'sample_rate': 24000,
                   'pcm': base64.b64encode(b'\x01\x00' * 24).decode()},
                  {'type': 'event_end', 'text': text}, {'type': 'audio_end', 'epoch': 0}):
        gate.intercept({**event, 'event_id': event_id})


class ClientTests(unittest.TestCase):
    def test_audio_pending_code_only_allowed_for_result_query(self):
        client = AliyunGuardrail(Config(enabled=True, access_key_id='id', access_key_secret='key'),
                                lambda *args: b'{"Code":280,"Data":{"RiskLevel":"none"}}')
        self.assertEqual(client.rpc('VoiceModerationResult')['Code'], 280)
        with self.assertRaises(GuardrailError): client.rpc('MultiModalGuard')

    def test_audio_pending_is_not_a_pass_and_transcript_stays_private(self):
        client = AliyunGuardrail(Config(enabled=True, access_key_id='id', access_key_secret='key'))
        responses = iter([{'Code':200,'Data':{'TaskId':'task'}},
            {'Code':280,'Data':{'RiskLevel':'none'}},
            {'Code':200,'Data':{'RiskLevel':'none','SliceDetails':[
                {'RiskLevel':'none','Text':'private spoken request','Result':[
                    {'Label':'sexual_sounds','RiskLevel':'none'}]}]}}])
        client.rpc = lambda *a, **k: next(responses)
        client._upload_wav = lambda *a: {}
        with patch('cineduplex.guardrail.aliyun_guardrail.time.sleep', return_value=None):
            result = client.check_audio(b'\x01\x00', sample_rate=16000, data_id='a')
        self.assertFalse(result.allowed)
        self.assertIn('sexual_sounds', result.labels)
        self.assertEqual(result.transcript, 'private spoken request')
        self.assertNotIn('private spoken request', str(result.record()) + repr(result))

    def test_standard_acoustic_label_blocks_even_without_text(self):
        client = AliyunGuardrail(Config(enabled=True, access_key_id='id', access_key_secret='key'))
        responses = iter([{'Code':200,'Data':{'TaskId':'task'}}, {'Code':200,'Data':{
            'RiskLevel':'none','SliceDetails':[{'RiskLevel':'none','Labels':'sexual_sounds','Text':''}]}}])
        client.rpc = lambda *a, **k: next(responses)
        client._upload_wav = lambda *a: {}
        with patch('cineduplex.guardrail.aliyun_guardrail.time.sleep', return_value=None):
            decision = client.check_audio(b'\x01\x00', sample_rate=16000, data_id='sound')
        self.assertFalse(decision.allowed)
        self.assertIn('sexual_sounds', decision.labels)
        self.assertEqual(decision.transcript, '')
        self.assertEqual(client.config.status()['nonverbal_detection'], 'disabled')
        self.assertEqual(replace(client.config, output_audio=True).status()['nonverbal_detection'], 'configured-unvalidated')

    def test_disabled_needs_no_credentials(self):
        self.assertFalse(Config.from_env({}).enabled)

    def test_enabled_rejects_missing_credentials(self):
        with self.assertRaisesRegex(GuardrailError, 'credentials_missing'):
            Config.from_env({'CINEDUPLEX_GUARDRAIL_ENABLED': 'true'})

    def test_status_and_repr_do_not_expose_credentials(self):
        cfg = Config(enabled=True, access_key_id='secret-id', access_key_secret='secret-value')
        self.assertNotIn('secret', repr(cfg))
        self.assertNotIn('secret', json.dumps(cfg.status()))

    def test_rejects_non_mainland_and_invalid_flags(self):
        with self.assertRaises(GuardrailError):
            Config.from_env({'CINEDUPLEX_GUARDRAIL_ENABLED': 'maybe'})
        with self.assertRaises(GuardrailError):
            Config(enabled=True, region='ap-southeast-1', access_key_id='id', access_key_secret='key').validate()

    def test_only_unambiguous_pass_is_allowed(self):
        for suggestion in ('pass', 'block', 'mask', 'watch'):
            result = parse_decision({'Code': 200, 'Data': {'Suggestion': suggestion, 'Detail': []}})
            self.assertEqual(result.allowed, suggestion == 'pass')
        self.assertFalse(parse_decision({'Code': 200, 'Data': {'Suggestion': 'pass',
            'Detail': [{'Suggestion': 'block', 'Result': []}]}}).allowed)

    def test_bad_schema_never_passes(self):
        for body in ({}, {'Code': 200}, {'Code': 500}, {'Code': 200, 'Data': {}},
                     {'Code': 200, 'Data': {'Suggestion': 'unknown'}},
                     {'Code': 200, 'Data': {'Suggestion': 'pass', 'Detail': [{}]}}):
            with self.assertRaises(GuardrailError):
                parse_decision(body)

    def test_rpc_wire_has_service_json_and_sts_in_signature(self):
        captured = []
        def transport(req, timeout):
            captured.append(req)
            return b'{"Code":200,"RequestId":"probe-id","Data":{"Suggestion":"pass","Detail":[]}}'
        config = Config(enabled=True, access_key_id='dummy-id', access_key_secret='dummy-key', security_token='dummy-sts')
        client = AliyunGuardrail(config, transport)
        result = client.check_text('你好', direction='output', session_id='s', chat_id='c')
        params = {k: v[0] for k, v in parse_qs(captured[0].data.decode()).items()}
        signature = params.pop('Signature')
        self.assertEqual(signature, rpc_signature(params, 'dummy-key'))
        self.assertEqual(params['SecurityToken'], 'dummy-sts')
        self.assertEqual(params['Action'], 'MultiModalGuard')
        self.assertEqual(params['Service'], 'response_security_check_pro')
        self.assertEqual(json.loads(params['ServiceParameters']), {'content': '你好', 'sessionId': 's', 'chatId': 'c', 'done': True})
        self.assertTrue(captured[0].full_url.startswith('https://green-cip.cn-shanghai.aliyuncs.com/'))
        self.assertTrue(result.allowed)

    def test_oversize_text_does_not_submit(self):
        calls = []
        client = AliyunGuardrail(Config(enabled=True, access_key_id='id', access_key_secret='key'),
                                lambda *args: calls.append(args))
        for text in ('', 'x' * 2001):
            with self.assertRaises(GuardrailError):
                client.check_text(text, direction='output', session_id='s', chat_id='c')
        self.assertEqual(calls, [])

    def test_redirect_is_rejected(self):
        with self.assertRaises(GuardrailError):
            NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example')

    def test_audio_upload_wav_and_result_use_fake_vendor_protocol(self):
        requests = []
        def transport(req, timeout):
            requests.append(req)
            if req.method == 'PUT':
                self.assertEqual(req.data[:4], b'RIFF')
                self.assertEqual(req.data[8:12], b'WAVE')
                self.assertIn('dummy-sts', str(req.headers))
                return b''
            action = parse_qs(req.data.decode())['Action'][0]
            if action == 'DescribeUploadToken':
                body = {'Code': 200, 'Data': {'Expiration': time.time() + 3600,
                    'OssInternetEndPoint': 'oss-cn-shanghai.aliyuncs.com', 'BucketName': 'test-bucket',
                    'FileNamePrefix': 'temporary/', 'AccessKeyId': 'dummy-id',
                    'AccessKeySecret': 'dummy-key', 'SecurityToken': 'dummy-sts'}}
            elif action == 'VoiceModeration':
                body = {'Code': 200, 'Data': {'TaskId': 'fake-task'}}
            else:
                body = {'Code': 200, 'RequestId': 'audio-id', 'Data': {'RiskLevel': 'none',
                        'SliceDetails': [{'RiskLevel': 'none'}]}}
            return json.dumps(body).encode()
        client = AliyunGuardrail(Config(enabled=True, access_key_id='id', access_key_secret='key'), transport)
        with patch('cineduplex.guardrail.aliyun_guardrail.time.sleep', return_value=None):
            result = client.check_audio(b'\x01\x00' * 64, sample_rate=16000, data_id='test-audio')
        self.assertTrue(result.allowed)
        self.assertEqual(len(requests), 4)


class GateTests(unittest.TestCase):
    def test_audio_text_consistency_handles_asr_errors_but_rejects_tone_hallucination(self):
        self.assertTrue(audio_text_consistent('你好，我是一个能帮你处理各种任务的人工智能助手。', '你好我是个能帮你处理各种任务的人工智能助手'))
        self.assertTrue(audio_text_consistent('2加2等于4。', '二加二等于四'))
        self.assertFalse(audio_text_consistent('这是一段测试音。', '嗯的嗯的'))
        self.assertFalse(audio_text_consistent('你好', '...'))

    def test_unrelated_audio_transcript_does_not_pass_with_safe_text(self):
        client = FakeClient(Config(enabled=True, output_audio=True))
        client.check_audio = lambda *a, **k: Decision(True, 'pass', 'tone-asr', transcript='嗯的嗯的')
        gate = self.make_gate(client); answer(gate)
        wait_for(self.call.closed.is_set)
        self.assertEqual(self.content(), [])
        self.assertIn('output_audio_text_mismatch', str(self.call.events))

    def test_input_semantics_blocks_harmless_audio_verdict(self):
        cfg = Config(enabled=True, input_audio=True, timeout=.1, audio_deadline=1)
        client = FakeClient(cfg)
        original = client.check_text
        client.check_text = lambda text, **kw: (Decision(False, 'block', 'semantic-risk')
            if kw['direction'] == 'input' else original(text, **kw))
        gate = self.make_gate(client)
        gate.input_audio(0, b'\x01\x00' * 6400); answer(gate)
        wait_for(self.call.closed.is_set)
        self.assertEqual(self.content(), [])
        self.assertTrue(any(e.get('direction') == 'input_semantic' and e.get('allowed') is False
                            for e in self.call.events))

    def test_nonzero_quiet_audio_is_not_discarded_but_digital_silence_is(self):
        cfg = Config(enabled=True, input_audio=True, input_window_seconds=1)
        gate = self.make_gate(FakeClient(cfg))
        gate.input_audio(0, b'\0' * 32000)
        self.assertEqual(gate.input_submitted, 0)
        gate.input_audio(1, b'\x01\x00' * 16000)
        wait_for(lambda: gate.input_reviewed == 1)
        self.assertEqual(len(self.client.audio_requests), 1)

    def test_untranscribable_output_is_never_released(self):
        cfg = Config(enabled=True, output_audio=True, timeout=.1, audio_deadline=1)
        client = FakeClient(cfg)
        client.check_audio = lambda *a, **k: Decision(True, 'pass', 'empty-asr')
        gate = self.make_gate(client); answer(gate)
        wait_for(self.call.closed.is_set)
        self.assertEqual(self.content(), [])
        self.assertIn('output_audio_semantics_unrecognized', str(self.call.events))

    def test_output_audio_review_runs_while_text_waits(self):
        cfg = Config(enabled=True, output_audio=True, timeout=.1, audio_deadline=1)
        client = FakeClient(cfg); client.text_release.clear()
        gate = self.make_gate(client); answer(gate)
        self.assertTrue(client.audio_entered.wait(1))
        self.assertEqual(self.content(), [])
        client.text_release.set()
        wait_for(lambda: any(e['type'] == 'audio_end' for e in self.call.events))

    def test_audio_cancellation_does_not_wait_for_old_cloud_task(self):
        cfg = Config(enabled=True, output_audio=True, timeout=.1, audio_deadline=1)
        client = FakeClient(cfg); entered = threading.Event()
        def audio(pcm, **kwargs):
            entered.set()
            kwargs['cancelled'].wait(1)
            raise GuardrailError('audio_review_cancelled')
        client.check_audio = audio
        gate = self.make_gate(client); answer(gate)
        self.assertTrue(entered.wait(1))
        old = gate.responses['evt-1']
        self.call.epoch = 1; gate.intercept({'type':'stop_audio','epoch':1})
        self.assertTrue(old.cancelled.is_set())
        self.assertEqual(self.content(), [])
        self.assertFalse(self.call.closed.is_set())

    def test_complete_text_review_overlaps_codec_but_release_waits_for_tail(self):
        gate = self.make_gate()
        for e in ({'type':'event_start'}, {'type':'text_delta','text':'你好。'},
                  {'type':'event_end','text':'你好。'}):
            gate.intercept({**e,'event_id':'overlap'})
        wait_for(lambda: bool(self.client.text_requests))
        wait_for(lambda: any(e['type']=='guardrail_result' for e in self.call.events))
        self.assertFalse(any(e['type'] in gate.CONTENT for e in self.call.events))
        for e in ({'type':'audio_start','epoch':0},
                  {'type':'audio','epoch':0,'sample_rate':24000,'pcm':'AAA='},
                  {'type':'audio_end','epoch':0}):
            gate.intercept({**e,'event_id':'overlap'})
        wait_for(lambda: any(e['type']=='audio_end' for e in self.call.events))
        self.assertEqual(len(self.client.text_requests), 1)

    def test_interruption_after_text_pass_before_audio_tail_releases_nothing(self):
        gate = self.make_gate()
        for e in ({'type':'event_start'}, {'type':'event_end','text':'你好。'}):
            gate.intercept({**e,'event_id':'old'})
        wait_for(lambda: any(e['type']=='guardrail_result' for e in self.call.events))
        self.call.epoch = 1
        gate.intercept({'type':'stop_audio','epoch':1})
        gate.intercept({'type':'audio_end','event_id':'old','epoch':0})
        self.assertFalse(any(e['type'] in gate.CONTENT for e in self.call.events))
        self.assertNotIn('old', gate.responses)
        self.assertFalse(self.call.closed.is_set())

    def make_gate(self, client=None):
        self.client = client or FakeClient(); self.call = FakeCall()
        self.gate = CallGuardrail(self.call, self.client)
        self.addCleanup(self.gate.close)
        return self.gate

    def content(self):
        return [event for event in self.call.events if event['type'] in CallGuardrail.CONTENT]

    def test_text_and_audio_not_released_before_decision(self):
        client = FakeClient(); client.text_release.clear(); gate = self.make_gate(client)
        answer(gate); self.assertTrue(client.text_entered.wait(1))
        self.assertEqual(self.content(), [])
        client.text_release.set(); wait_for(lambda: any(e['type'] == 'audio_end' for e in self.call.events))
        self.assertEqual([e['type'] for e in self.content()],
            ['event_start', 'audio_start', 'text_delta', 'audio', 'event_end', 'audio_end'])

    def test_risky_text_clears_audio_and_stops(self):
        gate = self.make_gate(FakeClient(text_allowed=False)); answer(gate)
        wait_for(self.call.closed.is_set)
        self.assertEqual(self.content(), [])
        self.assertEqual(self.call.reason, 'guardrail_blocked')
        self.assertTrue(any(e['type'] == 'stop_audio' for e in self.call.events))

    def test_approved_whole_answer_is_marked_complete_for_playback(self):
        gate = self.make_gate(); answer(gate)
        wait_for(lambda: any(e['type'] == 'audio_end' for e in self.call.events))
        start = next(e for e in self.content() if e['type'] == 'audio_start')
        self.assertFalse(start['streaming'])
        self.assertTrue(start['response_complete'])
        types = [e['type'] for e in self.call.events]
        phases = [e['phase'] for e in self.call.events if e['type'] == 'guardrail_pending']
        self.assertEqual(phases, ['generating', 'reviewing'])
        self.assertLess(types.index('guardrail_result'), types.index('audio_start'))

    def test_native_cancel_is_visible_and_never_releases_held_pcm(self):
        gate = self.make_gate()
        gate.intercept({'type': 'event_start', 'event_id': 'evt-1'})
        gate.intercept({'type': 'audio', 'event_id': 'evt-1', 'epoch': 0, 'sample_rate': 24000,
                        'pcm': base64.b64encode(b'\x01\x00' * 24).decode()})
        gate.intercept({'type': 'event_end', 'event_id': 'evt-1', 'interrupt': True})
        self.call.epoch = 1; gate.intercept({'type': 'stop_audio', 'epoch': 1})
        cancelled = [e for e in self.call.events if e['type'] == 'guardrail_cancelled']
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(self.content(), [])
        self.assertEqual(self.client.text_requests, [])

    def test_timeout_or_unexpected_exception_blocks_without_leaking(self):
        gate = self.make_gate(FakeClient(text_error=RuntimeError('secret-provider-body'))); answer(gate)
        wait_for(self.call.closed.is_set)
        self.assertEqual(self.content(), [])
        self.assertNotIn('secret-provider-body', str(self.call.events))
        self.assertEqual(self.call.reason, 'guardrail_unavailable')

    def test_interrupt_while_reviewing_drops_old_response(self):
        client = FakeClient(); client.text_release.clear(); gate = self.make_gate(client)
        answer(gate); self.assertTrue(client.text_entered.wait(1))
        self.call.epoch = 1; gate.intercept({'type': 'stop_audio', 'epoch': 1})
        client.text_release.set(); wait_for(lambda: not gate.responses)
        self.assertEqual(self.content(), [])
        self.assertFalse(self.call.closed.is_set())

    def test_hangup_during_review_never_releases(self):
        client = FakeClient(); client.text_release.clear(); gate = self.make_gate(client)
        answer(gate); self.assertTrue(client.text_entered.wait(1))
        self.call.stop('hangup'); client.text_release.set()
        wait_for(lambda: not gate.output_worker.is_alive())
        self.assertEqual(self.content(), [])

    def test_audio_must_pass_when_enabled(self):
        cfg = Config(enabled=True, output_audio=True, timeout=.1, audio_deadline=1)
        gate = self.make_gate(FakeClient(cfg, audio_allowed=False)); answer(gate)
        wait_for(self.call.closed.is_set)
        self.assertEqual(self.content(), [])
        self.assertEqual(len(self.client.audio_requests), 1)

    def test_input_audio_retry_not_charged_twice(self):
        cfg = Config(enabled=True, input_audio=True, input_window_seconds=1, timeout=.1, audio_deadline=1)
        gate = self.make_gate(FakeClient(cfg))
        pcm = b'\x01\x00' * 16000
        gate.input_audio(0, pcm); gate.input_audio(0, pcm)
        wait_for(lambda: gate.input_reviewed == 1)
        self.assertEqual(len(self.client.audio_requests), 1)

    def test_input_result_barrier_holds_approved_answer(self):
        cfg = Config(enabled=True, input_audio=True, timeout=.1, audio_deadline=1)
        client = FakeClient(cfg); client.audio_release.clear(); gate = self.make_gate(client)
        gate.input_audio(0, b'\x01\x00' * 6400); answer(gate)
        self.assertTrue(client.audio_entered.wait(1)); self.assertTrue(client.text_entered.wait(1))
        self.assertEqual(self.content(), [])
        client.audio_release.set(); wait_for(lambda: any(e['type'] == 'audio_end' for e in self.call.events))

    def test_risky_input_prevents_answer_release(self):
        cfg = Config(enabled=True, input_audio=True, timeout=.1, audio_deadline=1)
        gate = self.make_gate(FakeClient(cfg, audio_allowed=False))
        gate.input_audio(0, b'\x01\x00' * 6400); answer(gate)
        wait_for(self.call.closed.is_set)
        self.assertEqual(self.content(), [])

    def test_invalid_pcm_and_missing_identity_stop(self):
        gate = self.make_gate()
        gate.intercept({'type': 'text_delta', 'text': 'not identified'})
        self.assertTrue(self.call.closed.is_set())
        self.assertEqual(self.content(), [])

    def test_excess_response_budget_stops(self):
        gate = self.make_gate()
        gate.intercept({'type': 'text_delta', 'event_id': 'evt-1', 'snapshot': 'x' * 2001})
        self.assertTrue(self.call.closed.is_set())
        self.assertEqual(self.content(), [])

    def test_response_without_audio_is_reviewed(self):
        gate = self.make_gate()
        gate.intercept({'type': 'event_end', 'event_id': 'evt-1', 'text': '你好。'})
        gate.intercept({'type': 'response_warning', 'event_id': 'evt-1', 'epoch': 0, 'message': 'no audio'})
        wait_for(lambda: any(e['type'] == 'guardrail_pass' for e in self.call.events))
        self.assertTrue(any(e['type'] == 'event_end' for e in self.content()))

    @unittest.skipUnless((ROOT / "src/cineduplex/train/realtime_service.py").is_file(), "private realtime integration is not in the public package")
    def test_actual_call_emit_hook_blocks_before_archive(self):
        # Execute the actual lightweight hook, without importing GPU libraries.
        tree = ast.parse((ROOT / 'src/cineduplex/train/realtime_service.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Call')
        emit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'emit')
        ns = {}; exec(compile(ast.Module(body=[emit], type_ignores=[]), 'actual-call-emit', 'exec'), ns)
        gate = self.make_gate(); self.call.guardrail = gate
        ns['emit'](self.call, {'type': 'text_delta', 'event_id': 'evt-1', 'snapshot': 'pending'})
        self.assertEqual(self.content(), [])
        self.call.guardrail = None
        ns['emit'](self.call, {'type': 'text_delta', 'event_id': 'evt-2', 'snapshot': 'disabled'})
        self.assertEqual(self.content()[0]['snapshot'], 'disabled')


if __name__ == '__main__':
    unittest.main()


class AttributionTests(unittest.TestCase):
    def test_refusal_is_only_an_observation_never_a_safety_verdict(self):
        row = refusal_observation('抱歉，我无法协助进行非法的入侵行为。')
        self.assertEqual(row['refusal_signal'], 'expression_detected')
        self.assertEqual(row['safety_verdict'], 'not_assessed')
        self.assertNotIn('非法', str(row))
        self.assertEqual(refusal_observation('天气很好')['refusal_signal'], 'unknown')
        # Even a quoted refusal is only a heuristic; it cannot approve safety.
        self.assertEqual(refusal_observation('他说“I cannot help”。')['safety_verdict'], 'not_assessed')

    def test_cloud_block_retains_cloud_cause_not_local_enforcement(self):
        call = FakeCall(); gate = CallGuardrail(call, FakeClient(text_allowed=False))
        try:
            answer(gate)
            wait_for(lambda: call.closed.is_set())
            row = next(x for x in call.events if x['type'] == 'safety_outcome')
            self.assertEqual(row['outcome'], 'aliyun_block')
            self.assertEqual(row['decision_source'], 'aliyun')
            self.assertEqual(row['request_id'], 'fake-text-id')
            self.assertFalse(any(x['type'] == 'event_end' for x in call.events))
        finally: gate.close()

    def test_timeout_is_local_stop_not_cloud_content_block(self):
        call = FakeCall(); gate = CallGuardrail(call, FakeClient(text_error=GuardrailError('timeout')))
        try:
            answer(gate); wait_for(lambda: call.closed.is_set())
            row = next(x for x in call.events if x['type'] == 'safety_outcome')
            self.assertEqual(row['outcome'], 'application_stop')
            self.assertIsNone(row['request_id'])
        finally: gate.close()

    def test_model_refusal_released_only_after_cloud_pass(self):
        call = FakeCall(); gate = CallGuardrail(call, FakeClient())
        try:
            answer(gate, text='抱歉，我无法协助入侵。')
            wait_for(lambda: any(x['type'] == 'event_end' for x in call.events))
            kinds = [x['type'] for x in call.events]
            row = next(x for x in call.events if x['type'] == 'safety_outcome')
            self.assertEqual(row['outcome'], 'model_refusal_expression')
            self.assertLess(kinds.index('guardrail_pass'), kinds.index('safety_outcome'))
            self.assertLess(kinds.index('safety_outcome'), kinds.index('event_end'))
        finally: gate.close()
