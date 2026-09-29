"""Targeted CPU comparison with the pinned S3Tokenizer frontend, no training."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from functools import lru_cache
import tarfile
from typing import Optional, Union
import wave

from cineduplex.contracts import digest, write_json
from .nas_source import nas_root
from .codec_cpu import MODEL_SHA, MODEL_BYTES

PACKAGE_SHA = '21826f35fafc9c7edcb6f88dc6b4d7b976d312824e0ef13f84f90a6800d4fca5'


def check(root):
    root = nas_root(root)
    ref = root/'raw/codec-reference'
    archive = ref/'s3tokenizer-0.2.0.tar.gz'
    if digest(archive) != PACKAGE_SHA:
        raise ValueError('reference archive checksum mismatch')
    # Only extract these two known regular files; no package installation/import.
    with tarfile.open(archive) as package:
        for name in ('utils.py', 'assets/mel_filters.npz'):
            member = package.getmember('s3tokenizer-0.2.0/s3tokenizer/'+name)
            if not member.isfile() or member.size > 1024*1024:
                raise ValueError('unexpected reference file')
            target = ref/'s3tokenizer-0.2.0'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.extractfile(member).read())
    source = ref/'s3tokenizer-0.2.0/utils.py'
    tree = ast.parse(source.read_text())
    wanted = ('_mel_filters', 'log_mel_spectrogram')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != set(wanted):
        raise ValueError('reference frontend functions missing')
    print('frontend: importing existing NAS CPU dependencies', flush=True)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import librosa
    import onnxruntime as ort
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    namespace = dict(os=os, np=np, torch=torch, F=F, lru_cache=lru_cache,
                     Optional=Optional, Union=Union, __file__=str(source))
    # Exact upstream function ASTs; tensor input avoids unused audio-file loader.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    print('frontend: loading checksum-verified existing encoder', flush=True)
    model = (root/'raw/codec/speech_tokenizer_v2_25hz.onnx').read_bytes()
    if len(model) != MODEL_BYTES or hashlib.sha256(model).hexdigest() != MODEL_SHA:
        raise ValueError('encoder mismatch')
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model, sess_options=opts, providers=['CPUExecutionProvider'])
    del model
    inputs = session.get_inputs()
    filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128).astype(np.float32)
    reference_filters = namespace['_mel_filters']('cpu', 128).numpy()
    rows = []
    for folder in ('output/pilot10-current', 'output/pilot10-loader-input'):
        for line in (root/folder/'scenes.jsonl').read_text().splitlines():
            scene = json.loads(line)
            for turn in scene['turns']:
                wav = root/turn['audio']['path']
                if digest(wav) != turn['audio']['sha256']:
                    raise ValueError('source audio changed')
                tokens_path = wav.with_suffix('.tokens.json')
                saved = json.loads(tokens_path.read_text())
                if saved['model_sha256'] != MODEL_SHA or saved['input_sha256'] != turn['audio']['sha256']:
                    raise ValueError('token provenance mismatch')
                with wave.open(str(wav)) as stream:
                    if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
                        raise ValueError('expected 16k mono PCM16')
                    samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32)/32768
                reference = namespace['log_mel_spectrogram'](torch.from_numpy(samples), n_mels=128).numpy()
                stft = librosa.stft(samples, n_fft=400, hop_length=160, win_length=400,
                                     window='hann', center=True, pad_mode='reflect')
                mel = filters @ (np.abs(stft[:, :-1])**2)
                log = np.log10(np.maximum(mel, 1e-10))
                actual = (np.maximum(log, log.max()-8)+4)/4
                if actual.shape != reference.shape or not np.isfinite(reference).all():
                    raise ValueError('reference feature shape/finite check failed')
                codes = session.run(None, {inputs[0].name:reference[None].astype(np.float32),
                    inputs[1].name:np.asarray([reference.shape[-1]], dtype=np.int32)})[0].reshape(-1).tolist()
                expected = saved['raw_codes']
                same = codes == expected
                rows.append(dict(scene_id=scene['scene_id'], utterance_id=turn['utterance_id'],
                    input_sha256=turn['audio']['sha256'], tokens_sha256=digest(tokens_path),
                    mel_shape=list(reference.shape), mel_max_abs_difference=float(np.max(np.abs(actual-reference))),
                    token_count=len(codes), exact_codes_match=same,
                    differing_positions=sum(a != b for a,b in zip(codes,expected))+abs(len(codes)-len(expected))))
                print('frontend:', turn['utterance_id'], 'exact codes match:', same, flush=True)
    result = dict(status='PASSED' if len(rows) == 29 and all(x['exact_codes_match'] for x in rows) else 'FAILED',
        scope='29 actual pilot/continuous-input clips; pinned S3Tokenizer 0.2.0 Torch CPU frontend and existing ONNX CPU encoder; no CUDA/model-training or hearing claim',
        package_sha256=PACKAGE_SHA, reference_source_sha256=digest(source),
        reference_functions=list(wanted), upstream_function_asts_unchanged=True,
        reference_filter_sha256=digest(ref/'s3tokenizer-0.2.0/assets/mel_filters.npz'),
        filter_max_abs_difference=float(np.max(np.abs(filters-reference_filters))),
        weights_sha256=MODEL_SHA, clips=len(rows), total_tokens=sum(x['token_count'] for x in rows),
        rows=rows, training_ready=False)
    write_json(root/'output/pilot10-current/frontend-reference-check.json', result)
    if result['status'] != 'PASSED' or len(rows) != 29:
        raise ValueError('reference comparison failed; preserve existing artifacts for investigation')
    print('frontend: PASSED for all 29 clips', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    check(parser.parse_args().root)
