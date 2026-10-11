"""Aliyun Guardrails native HTTPS client. No SDK, GPU or disk dependency.

RPC signing follows the vendor's native HTTPS examples (2022-03-02).
Never expose credentials, signed requests, source content or vendor error bodies.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import formatdate
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from urllib.parse import quote, urlencode, urlparse
import urllib.request
import uuid
import wave


class GuardrailError(RuntimeError):
    """A deliberately content-free diagnostic."""


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    region: str = 'cn-shanghai'
    input_audio: bool = False
    output_audio: bool = False
    timeout: float = 5.0
    audio_deadline: float = 45.0
    input_window_seconds: float = 4.0
    max_answer_seconds: float = 60.0
    input_service: str = 'query_security_check_pro'
    output_service: str = 'response_security_check_pro'
    audio_service: str = 'audio_detection_byllm'
    access_key_id: str = field(default='', repr=False)
    access_key_secret: str = field(default='', repr=False)
    security_token: str = field(default='', repr=False)

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        def flag(name, default=False):
            value = env.get(name, 'true' if default else 'false').lower()
            if value not in ('true', 'false', '1', '0'):
                raise GuardrailError('invalid_boolean_configuration')
            return value in ('true', '1')
        enabled = flag('CINEDUPLEX_GUARDRAIL_ENABLED')
        if not enabled:
            return cls()
        credentials = {}
        if env.get('CINEDUPLEX_GUARDRAIL_CREDENTIALS_FILE'):
            path = Path(env['CINEDUPLEX_GUARDRAIL_CREDENTIALS_FILE'])
            try:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 16384:
                    raise GuardrailError('credentials_file_requires_0600')
                credentials = json.loads(path.read_text())
                if not isinstance(credentials, dict):
                    raise ValueError()
            except GuardrailError:
                raise
            except (OSError, ValueError):
                raise GuardrailError('credentials_file_unavailable') from None
        try:
            config = cls(
                enabled=True,
                region=env.get('CINEDUPLEX_GUARDRAIL_REGION', 'cn-shanghai'),
                input_audio=flag('CINEDUPLEX_GUARDRAIL_INPUT_AUDIO'),
                output_audio=flag('CINEDUPLEX_GUARDRAIL_OUTPUT_AUDIO'),
                timeout=float(env.get('CINEDUPLEX_GUARDRAIL_TIMEOUT', '5')),
                audio_deadline=float(env.get('CINEDUPLEX_GUARDRAIL_AUDIO_DEADLINE', '45')),
                input_window_seconds=float(env.get('CINEDUPLEX_GUARDRAIL_INPUT_WINDOW', '4')),
                input_service=env.get('CINEDUPLEX_GUARDRAIL_INPUT_SERVICE', 'query_security_check_pro'),
                output_service=env.get('CINEDUPLEX_GUARDRAIL_OUTPUT_SERVICE', 'response_security_check_pro'),
                audio_service=env.get('CINEDUPLEX_GUARDRAIL_AUDIO_SERVICE', 'audio_detection_byllm'),
                access_key_id=env.get('ALIBABA_CLOUD_ACCESS_KEY_ID') or credentials.get('access_key_id', ''),
                access_key_secret=env.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET') or credentials.get('access_key_secret', ''),
                security_token=env.get('ALIBABA_CLOUD_SECURITY_TOKEN') or credentials.get('security_token', ''),
            )
        except (TypeError, ValueError):
            raise GuardrailError('invalid_numeric_configuration') from None
        config.validate()
        return config

    def validate(self):
        if self.region not in ('cn-shanghai', 'cn-beijing', 'cn-hangzhou', 'cn-shenzhen', 'cn-chengdu'):
            raise GuardrailError('unsupported_mainland_region')
        if not (0.1 <= self.timeout <= 15 and self.timeout <= self.audio_deadline <= 120
                and 1 <= self.input_window_seconds <= 10):
            raise GuardrailError('invalid_timeout_or_window')
        if not all(isinstance(x, str) and x for x in (self.access_key_id, self.access_key_secret)):
            raise GuardrailError('credentials_missing')
        for service in (self.input_service, self.output_service, self.audio_service):
            if not re.fullmatch(r'[A-Za-z0-9_]{1,128}', service):
                raise GuardrailError('invalid_service')

    def status(self):
        return {'enabled': self.enabled, 'provider': 'aliyun' if self.enabled else None,
                'region': self.region if self.enabled else None,
                'output_text': self.enabled,
                'output_text_service': self.output_service if self.enabled else None,
                'input_audio': self.enabled and self.input_audio,
                'output_audio': self.enabled and self.output_audio,
                'mode': 'review-before-release' if self.enabled else 'disabled',
                'input_mode': 'monitor-and-stop' if self.enabled and self.input_audio else 'disabled',
                'cloud_verified': False}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    suggestion: str
    request_id: str = ''
    labels: tuple = ()
    elapsed_ms: float = 0.0
    dimensions: tuple = ()

    def record(self):
        return {'allowed': self.allowed, 'suggestion': self.suggestion,
                'request_id': self.request_id, 'labels': list(self.labels),
                'dimensions': list(self.dimensions),
                'elapsed_ms': round(self.elapsed_ms, 1)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GuardrailError('unexpected_redirect')


def percent(value):
    return quote(str(value), safe='~')


def rpc_signature(parameters, secret):
    canonical = '&'.join(percent(k) + '=' + percent(v) for k, v in sorted(parameters.items()))
    signing = 'POST&%2F&' + percent(canonical)
    return base64.b64encode(hmac.new((secret + '&').encode(), signing.encode(), hashlib.sha1).digest()).decode()


def parse_decision(payload, elapsed_ms=0):
    if not isinstance(payload, dict) or payload.get('Code') != 200:
        raise GuardrailError('cloud_api_rejected')
    data = payload.get('Data')
    if not isinstance(data, dict) or data.get('Suggestion') not in ('pass', 'block', 'watch', 'mask'):
        raise GuardrailError('cloud_response_incomplete')
    labels = set(); dimensions = set()
    allowed = data['Suggestion'] == 'pass'
    details = data.get('Detail', [])
    if not isinstance(details, list):
        raise GuardrailError('cloud_response_incomplete')
    for item in details:
        if not isinstance(item, dict) or item.get('Suggestion') not in ('pass', 'block', 'watch', 'mask'):
            raise GuardrailError('cloud_response_incomplete')
        allowed = allowed and item['Suggestion'] == 'pass'
        dimension = item.get('Type')
        if isinstance(dimension, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', dimension):
            dimensions.add(dimension)
        for result in item.get('Result', []):
            if not isinstance(result, dict):
                raise GuardrailError('cloud_response_incomplete')
            label = result.get('Label')
            if isinstance(label, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', label):
                labels.add(label)
    return Decision(allowed, data['Suggestion'], str(payload.get('RequestId', ''))[:128],
                    tuple(sorted(labels)), elapsed_ms, tuple(sorted(dimensions)))


class AliyunGuardrail:
    def __init__(self, config, transport=None):
        self.config = config
        if config.enabled:
            config.validate()
        # Avoid an unrelated proxy receiving signed requests; retain normal TLS validation.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.transport = transport or self._https
        self.token_lock = threading.Lock()
        self.upload_token = None

    def _https(self, request, timeout):
        try:
            with self.opener.open(request, timeout=timeout) as response:
                data = response.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise GuardrailError('cloud_response_too_large')
                return data
        except GuardrailError:
            raise
        except Exception:
            raise GuardrailError('cloud_transport_failed') from None

    def rpc(self, action, service=None, parameters=None, timeout=None):
        cfg = self.config
        if not cfg.enabled:
            raise GuardrailError('guardrail_disabled')
        params = {'Action': action, 'Version': '2022-03-02', 'Format': 'JSON',
                  'AccessKeyId': cfg.access_key_id, 'SignatureMethod': 'HMAC-SHA1',
                  'SignatureVersion': '1.0', 'SignatureNonce': str(uuid.uuid4()),
                  'Timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        if cfg.security_token:
            params['SecurityToken'] = cfg.security_token
        if service:
            params['Service'] = service
        if parameters is not None:
            params['ServiceParameters'] = json.dumps(parameters, ensure_ascii=False, separators=(',', ':'))
        params['Signature'] = rpc_signature(params, cfg.access_key_secret)
        request = urllib.request.Request('https://green-cip.' + cfg.region + '.aliyuncs.com/',
                    data=urlencode(params).encode(), headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    method='POST')
        try:
            payload = json.loads(self.transport(request, timeout or cfg.timeout))
        except GuardrailError:
            raise
        except Exception:
            raise GuardrailError('cloud_response_invalid') from None
        if not isinstance(payload, dict) or payload.get('Code') != 200:
            raise GuardrailError('cloud_api_rejected')
        return payload

    def check_text(self, text, *, direction, session_id, chat_id):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise GuardrailError('text_empty_or_exceeds_2000_characters')
        if direction not in ('input', 'output'):
            raise GuardrailError('invalid_direction')
        begun = time.monotonic()
        payload = self.rpc('MultiModalGuard', self.config.input_service if direction == 'input'
                          else self.config.output_service,
                          {'content': text, 'sessionId': session_id, 'chatId': chat_id, 'done': True})
        return parse_decision(payload, (time.monotonic() - begun) * 1000)

    def _upload_wav(self, pcm, sample_rate, deadline):
        with self.token_lock:
            if self.upload_token is None or self.upload_token.get('Expiration', 0) <= time.time() + 120:
                self.upload_token = self.rpc('DescribeUploadToken', timeout=self._remaining(deadline))['Data']
            token = dict(self.upload_token)
        try:
            host = urlparse('https://' + token['OssInternetEndPoint'].removeprefix('https://').removeprefix('http://')).hostname
            bucket = token['BucketName']
            if not host or not re.fullmatch(r'oss-[a-z0-9-]+\.aliyuncs\.com', host):
                raise ValueError()
            if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]', bucket):
                raise ValueError()
            object_name = token['FileNamePrefix'] + str(uuid.uuid4()) + '.wav'
            if not re.fullmatch(r'[A-Za-z0-9_./-]+', object_name) or '..' in object_name.split('/'):
                raise ValueError()
            wav = io.BytesIO()
            with wave.open(wav, 'wb') as audio:
                audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(sample_rate); audio.writeframes(pcm)
            body = wav.getvalue()
            md5 = base64.b64encode(hashlib.md5(body).digest()).decode()
            date = formatdate(usegmt=True)
            security = token['SecurityToken']
            canonical = ('PUT\n' + md5 + '\naudio/wav\n' + date + '\nx-oss-security-token:'
                         + security + '\n/' + bucket + '/' + object_name)
            signature = base64.b64encode(hmac.new(token['AccessKeySecret'].encode(), canonical.encode(), hashlib.sha1).digest()).decode()
            request = urllib.request.Request('https://' + bucket + '.' + host + '/' + quote(object_name, safe='/'),
                    data=body, method='PUT', headers={'Content-Type': 'audio/wav', 'Content-MD5': md5,
                    'Date': date, 'x-oss-security-token': security,
                    'Authorization': 'OSS ' + token['AccessKeyId'] + ':' + signature})
        except (KeyError, TypeError, ValueError, AttributeError):
            raise GuardrailError('upload_token_invalid') from None
        self.transport(request, self._remaining(deadline))
        return {'ossBucketName': bucket, 'ossObjectName': object_name}

    def _remaining(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GuardrailError('audio_review_timeout')
        return min(self.config.timeout, remaining)

    def check_audio(self, pcm, *, sample_rate, data_id, cancelled=None):
        if sample_rate not in (16000, 24000) or not pcm or len(pcm) % 2:
            raise GuardrailError('invalid_audio')
        if len(pcm) > sample_rate * 2 * self.config.max_answer_seconds:
            raise GuardrailError('audio_exceeds_limit')
        begun = time.monotonic(); deadline = begun + self.config.audio_deadline
        parameters = self._upload_wav(pcm, sample_rate, deadline)
        parameters['dataId'] = data_id
        submitted = self.rpc('VoiceModeration', self.config.audio_service, parameters,
                             timeout=self._remaining(deadline))
        task = submitted.get('Data', {}).get('TaskId')
        if not isinstance(task, str) or not task:
            raise GuardrailError('audio_task_missing')
        while True:
            if cancelled is not None and cancelled.is_set():
                raise GuardrailError('audio_review_cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GuardrailError('audio_review_timeout')
            # File moderation is asynchronous. No resubmission on an ambiguous response.
            if cancelled is not None:
                if cancelled.wait(min(2, remaining)):
                    raise GuardrailError('audio_review_cancelled')
            else:
                time.sleep(min(2, remaining))
            payload = self.rpc('VoiceModerationResult', self.config.audio_service, {'taskId': task},
                               timeout=self._remaining(deadline))
            data = payload.get('Data', {})
            if not isinstance(data, dict):
                raise GuardrailError('cloud_response_incomplete')
            risk = data.get('RiskLevel')
            if risk is None:
                continue
            if risk not in ('none', 'low', 'medium', 'high'):
                raise GuardrailError('audio_result_invalid')
            slices = data.get('SliceDetails', [])
            if not isinstance(slices, list):
                raise GuardrailError('audio_result_invalid')
            allowed = risk == 'none'
            for part in slices:
                if not isinstance(part, dict) or part.get('RiskLevel') not in ('none', 'low', 'medium', 'high'):
                    raise GuardrailError('audio_result_invalid')
                allowed = allowed and part['RiskLevel'] == 'none'
            return Decision(allowed, 'pass' if allowed else 'block', str(payload.get('RequestId', ''))[:128],
                            (), (time.monotonic() - begun) * 1000)
