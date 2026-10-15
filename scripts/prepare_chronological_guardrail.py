"""Pin only chronological ASR assembly on the original safety bundle.

All risk decisions, thresholds, cancellation and release logic stay byte-identical.
This helper prepares immutable NAS sources; it never starts or updates a service.
"""
import ast
import hashlib
import json
from pathlib import Path

BASE = '308e50d837660b4fab74e7b8a5459d76268745d2b58c89c5c7a22c40fb7b3853'


def prepare(nas, prefix, version, runtime):
    raw = nas.read(prefix+'guardrail-bundle-v15.json', timeout=15)
    if hashlib.sha256(raw).hexdigest() != BASE:
        raise ValueError('original safety bundle identity')
    bundle = json.loads(raw)
    name = 'cineduplex.guardrail.aliyun_guardrail'
    old = bundle[name]['source']
    if hashlib.sha256(old.encode()).hexdigest() != bundle[name]['sha256']:
        raise ValueError('original safety client identity')
    local = (Path(__file__).resolve().parents[1]/'src/cineduplex/guardrail/aliyun_guardrail.py').read_text()
    node = next(n for n in ast.parse(local).body if isinstance(n, ast.FunctionDef) and n.name == 'chronological_slices')
    function = ast.get_source_segment(local, node)
    edits = {
        '    phase_ms: tuple = ()': '    phase_ms: tuple = ()\n    audio_order: tuple = ()',
        "**({'phase_ms': dict(self.phase_ms)} if self.phase_ms else {})}":
            "**({'phase_ms': dict(self.phase_ms)} if self.phase_ms else {}),\n                **({'audio_order': dict(self.audio_order)} if self.audio_order else {})}",
        'class NoRedirect(urllib.request.HTTPRedirectHandler):':
            function+'\n\n\nclass NoRedirect(urllib.request.HTTPRedirectHandler):',
        "            slices = data.get('SliceDetails')\n            if not isinstance(slices, list):\n                raise GuardrailError('audio_result_invalid')":
            "            slices, audio_order = chronological_slices(data.get('SliceDetails'))",
        "('polls', polls)))": "('polls', polls)), audio_order)",
    }
    source = old
    for before, after in edits.items():
        if source.count(before) != 1:
            raise ValueError('safety chronological hook identity: '+before[:70])
        source = source.replace(before, after)
    ast.parse(source)
    bundle[name] = {'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest()}
    pin = nas.put(prefix+f'guardrail-bundle-v{version}.json', json.dumps(bundle).encode())
    before = "GUARDRAIL_BUNDLE_NAME='guardrail-bundle-v15.json'"
    digest = 'GUARDRAIL_BUNDLE_SHA='+repr(BASE)
    if runtime.count(before) != 1 or runtime.count(digest) != 1:
        raise ValueError('runtime safety bundle hook identity')
    runtime = runtime.replace(before, f"GUARDRAIL_BUNDLE_NAME='guardrail-bundle-v{version}.json'")
    runtime = runtime.replace(digest, 'GUARDRAIL_BUNDLE_SHA='+repr(pin['sha256']))
    return runtime, pin
