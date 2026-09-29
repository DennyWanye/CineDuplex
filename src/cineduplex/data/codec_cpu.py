"""CPU-only speech token extraction for source-backed pilot utterances.

The ONNX weights are the pinned Step-Audio-2-mini S3 v2 25Hz tokenizer.
The NumPy mel frontend follows Whisper's 128-bin log-mel transform. This
produces real raw codes, but does not certify decoder parity or listening QA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import wave

from cineduplex.contracts import digest, write_json, safe_child
from .nas_source import nas_root

MODEL_SHA='d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71'
MODEL_BYTES=496082973


def encode(root, output_rel="output/pilot10"):
    """Encode one explicit scene manifest, including staged revision manifests."""
    root=nas_root(root)
    out=safe_child(root,output_rel)
    os.environ['NUMBA_CACHE_DIR']=str(root/'cache/numba')
    os.environ['TMPDIR']=str(root/'tmp')
    os.environ['HF_DATASETS_CACHE']=str(root/'cache/datasets')
    print('codec: importing CPU dependencies from NAS environment',flush=True)
    import numpy as np
    import librosa
    import onnxruntime as ort
    model=root/'raw/codec/speech_tokenizer_v2_25hz.onnx'
    print('codec: verifying full encoder checksum through NAS mount',flush=True)
    if model.stat().st_size!=MODEL_BYTES:raise ValueError('codec weights size mismatch')
    # Keep the verified bytes in RAM so ONNX does not repeat a slow SMB read.
    # No temporary model copy is written to the local disk.
    model_blob=model.read_bytes()
    if len(model_blob)!=MODEL_BYTES or hashlib.sha256(model_blob).hexdigest()!=MODEL_SHA:
        raise ValueError('codec weights hash mismatch')
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    print('codec: loading pinned ONNX encoder on CPU',flush=True)
    session=ort.InferenceSession(model_blob,sess_options=opts,providers=['CPUExecutionProvider'])
    del model_blob
    inputs=session.get_inputs()
    if len(inputs)!=2:raise ValueError('unexpected codec input signature')
    write_json(root/'evidence/codec-input-contract.json',dict(
        inputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in inputs],
        outputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_outputs()],
        providers=session.get_providers(),weights_sha256=MODEL_SHA,
        versions=dict(numpy=np.__version__,librosa=librosa.__version__,onnxruntime=ort.__version__)))
    scenes=[json.loads(line) for line in (out/'scenes.jsonl').read_text().splitlines()]
    records=[]
    print('codec: encoder loaded; preparing mel frontend',flush=True)
    filters=librosa.filters.mel(sr=16000,n_fft=400,n_mels=128).astype(np.float32)
    for scene in scenes:
        for turn in scene['turns']:
            path=root/turn['audio']['path']
            if digest(path)!=turn['audio']['sha256']:raise ValueError('input audio changed')
            with wave.open(str(path)) as f:
                if f.getframerate()!=16000 or f.getnchannels()!=1 or f.getsampwidth()!=2:
                    raise ValueError('codec requires 16k mono PCM16')
                samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768.0
            if len(samples)>30*16000:raise ValueError('explicit chunk policy required for >30s utterance')
            # Staged revisions may retain exact audio from the original pilot.
            # Reuse only matching encoder/input identities and valid raw codes.
            if output_rel != 'output/pilot10':
                prior=list((root/'output/pilot10').glob('S*/'+turn['utterance_id']+'.tokens.json'))
                reused=None
                for candidate in prior:
                    saved=json.loads(candidate.read_text())
                    raw=saved.get('raw_codes',[])
                    if (saved.get('input_sha256')==turn['audio']['sha256']
                            and saved.get('model_sha256')==MODEL_SHA
                            and saved.get('offset_applied')==0
                            and saved.get('token_count')==len(raw) and raw
                            and all(type(t)is int and 0<=t<6561 for t in raw)
                            and abs(len(raw)-len(samples)/640)<=2):
                        reused=dict(saved,scene_id=scene['scene_id'],
                            reused_from=str(candidate.relative_to(root)))
                        break
                if reused is not None:
                    write_json(out/scene['scene_id']/(turn['utterance_id']+'.tokens.json'),reused)
                    records.append(reused)
                    print(turn['utterance_id'],len(reused['raw_codes']),'verified existing codes reused',flush=True)
                    continue
            stft=librosa.stft(samples,n_fft=400,hop_length=160,win_length=400,window='hann',center=True,pad_mode='reflect')
            power=np.abs(stft[:,:-1])**2
            mel=filters@power
            log=np.log10(np.maximum(mel,1e-10));log=np.maximum(log,log.max()-8.0)
            feat=((log+4.0)/4.0)[None].astype(np.float32)
            lens=np.asarray([feat.shape[-1]],dtype=np.int32)
            started=time.monotonic()
            codes=session.run(None,{inputs[0].name:feat,inputs[1].name:lens})[0].reshape(-1).tolist()
            if not codes or any(type(t)!=int or not 0<=t<6561 for t in codes):
                raise ValueError('raw S3 v2 code out of vocabulary range')
            rate=len(codes)/(len(samples)/16000)
            if abs(len(codes)-len(samples)/640)>2:raise ValueError('unexpected 25Hz token length')
            result=dict(utterance_id=turn['utterance_id'],scene_id=scene['scene_id'],
                raw_codes=codes,token_count=len(codes),codec_vocab_size=6561,offset_applied=0,
                model_sha256=MODEL_SHA,sample_rate=16000,input_sha256=turn['audio']['sha256'],
                audio_duration_seconds=len(samples)/16000,effective_token_rate=rate,
                runtime='onnxruntime-cpu',frontend='numpy-librosa-whisper128',
                frontend_reference_parity='NOT_VERIFIED',decoded_roundtrip='NOT_PERFORMED',
                elapsed_seconds=time.monotonic()-started,training_ready=False)
            dest=out/scene['scene_id']/(turn['utterance_id']+'.tokens.json')
            write_json(dest,result);records.append(result)
            print(turn['utterance_id'],len(codes),'raw codes',flush=True)
    write_json(out/'codec-summary.json',dict(utterances=len(records),
        total_tokens=sum(x['token_count'] for x in records),weights_sha256=MODEL_SHA,
        token_offset=0,frontend_reference_parity='NOT_VERIFIED',decoded_roundtrip='NOT_PERFORMED'))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument("--output-relative",default="output/pilot10")
    args=parser.parse_args();encode(args.root,args.output_relative)
