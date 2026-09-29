"""CPU adaptation of the pinned Step-Audio2 non-streaming Token2wav path.

Produces reconstruction candidates and objective file checks, not listening QA.
Uses the original flow/vocoder modules with CPU tensors and strict state loads.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import wave

from cineduplex.contracts import digest, write_json, safe_child
from .codec_cpu import MODEL_SHA
from .nas_source import nas_root

WEIGHTS = {
    'campplus.onnx': 'a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73',
    'flow.pt': '15ccff24256ff61537c7f8b51e025116b83405f3fb017b54b008fc97da115446',
    'hift.pt': '3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879',
}
WEIGHT_BYTES = {'campplus.onnx': 28303423, 'flow.pt': 623466603, 'hift.pt': 83390254}


def decode(root, scene_id=None, output_rel="output/pilot10"):
    root=nas_root(root)
    for key, value in {'TMPDIR': root/'tmp', 'NUMBA_CACHE_DIR': root/'cache/numba',
                       'TORCH_HOME': root/'cache/torch', 'HF_HOME': root/'cache/huggingface',
                       'TORCHINDUCTOR_CACHE_DIR': root/'cache/torchinductor'}.items():
        os.environ[key]=str(value)
    print('decode: importing CPU runtime from NAS environment',flush=True)
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    import torch
    import torchaudio
    from hyperpyyaml import load_hyperpyyaml
    print('decode: CPU dependencies imported; preparing pinned architecture',flush=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assets=root/'raw/codec'
    def verified_bytes(name):
        print(f'decode: reading and verifying {name} from NAS',flush=True)
        path=assets/name
        if path.stat().st_size!=WEIGHT_BYTES[name]:raise ValueError(f'codec size mismatch: {name}')
        blob=path.read_bytes()
        if len(blob)!=WEIGHT_BYTES[name] or hashlib.sha256(blob).hexdigest()!=WEIGHTS[name]:
            raise ValueError(f'codec checksum mismatch: {name}')
        return blob
    config=(assets/'flow.yaml').read_bytes()
    if hashlib.sha1(b'blob '+str(len(config)).encode()+b'\0'+config).hexdigest()!='8638dca97881751e50c132498acbfa41ecf74d6f':
        raise ValueError('flow config differs from fixed upstream blob')
    project=Path(__file__).resolve().parents[3]
    vendor=project/'vendor/lychee-fd-source/third_party/Step-Audio2'
    sys.path.insert(0,str(vendor))
    from flashcosyvoice.modules.hifigan import HiFTGenerator
    from flashcosyvoice.utils.audio import mel_spectrogram
    # The pinned YAML instantiates cosyvoice2.flow.flow, not the differently
    # structured flashcosyvoice flow module. Keep its exact architecture.
    flow=load_hyperpyyaml(config.decode())['flow']
    flow.load_state_dict(torch.load(io.BytesIO(verified_bytes('flow.pt')),map_location='cpu',weights_only=True),strict=True)
    flow.cpu().float().eval()
    flow.scatter_cuda_graph(False)
    hift=HiFTGenerator()
    state=torch.load(io.BytesIO(verified_bytes('hift.pt')),map_location='cpu',weights_only=True)
    hift.load_state_dict({k.replace('generator.',''):v for k,v in state.items()},strict=True)
    del state
    hift.cpu().float().eval()
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    spk_model=ort.InferenceSession(verified_bytes('campplus.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    print('decode: all pinned models strictly loaded on CPU',flush=True)
    out=safe_child(root,output_rel)
    scenes=[json.loads(s) for s in (out/'scenes.jsonl').read_text().splitlines()]
    indexed=[(s,t) for s in scenes for t in s['turns']]
    def tokens(scene, turn):
        result=json.loads((out/scene['scene_id']/(turn['utterance_id']+'.tokens.json')).read_text())
        if result['input_sha256']!=turn['audio']['sha256'] or result['model_sha256']!=MODEL_SHA:
            raise ValueError('tokens do not match source audio or pinned encoder')
        codes=result['raw_codes']
        if not codes or any(type(x)is not int or not 0<=x<6561 for x in codes):
            raise ValueError('invalid raw codec codes')
        return codes
    prompt_cache={}
    def prompt(scene, turn):
        key=turn['utterance_id']
        if key in prompt_cache:return prompt_cache[key]
        path=root/turn['audio']['path']
        if digest(path)!=turn['audio']['sha256']:raise ValueError('prompt audio hash mismatch')
        with wave.open(str(path)) as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):raise ValueError('prompt format')
            audio=torch.from_numpy(np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768)
        codes=torch.tensor([tokens(scene,turn)],dtype=torch.int32)
        code_lengths=torch.tensor([codes.shape[1]],dtype=torch.int32)
        features=torchaudio.compliance.kaldi.fbank(audio.unsqueeze(0),num_mel_bins=80,dither=0,sample_frequency=16000)
        features-=features.mean(dim=0,keepdim=True)
        embedding=torch.from_numpy(spk_model.run(None,{spk_model.get_inputs()[0].name:features.unsqueeze(0).numpy()})[0])
        audio24=torchaudio.functional.resample(audio.unsqueeze(0),16000,24000)
        mels=mel_spectrogram(audio24).transpose(1,2)
        mel_lengths=torch.tensor([mels.shape[1]],dtype=torch.int32)
        mels=torch.nn.functional.pad(mels,(0,0,0,codes.shape[1]*flow.up_rate-mels.shape[1]),mode='replicate')
        result=(codes,code_lengths,embedding,mels,mel_lengths)
        prompt_cache[key]=result
        return result
    records=[]
    for scene,turn in indexed:
        if scene_id and scene['scene_id']!=scene_id:continue
        codes=tokens(scene,turn)
        candidates=[(s,t) for s,t in indexed if s['source_revision']==scene['source_revision']
            and t['speaker_id']==turn['speaker_id'] and t['utterance_id']!=turn['utterance_id']
            and s['coverage_requested']==scene['coverage_requested']
            and t['emotion']==turn['emotion']
            and 3 <= (t['end_sample']-t['start_sample'])/16000 <= 12
            and not t.get('overlaps')]
        # Same-utterance conditioning is allowed for a reconstruction diagnostic
        # but is explicitly distinguished from an independent reference.
        ps,pt=max(candidates,key=lambda pair:pair[1]['end_sample']-pair[1]['start_sample']) if candidates else (scene,turn)
        prompt_codes,prompt_lengths,embedding,mels,mel_lengths=prompt(ps,pt)
        torch.manual_seed(0)
        started=time.monotonic()
        with torch.inference_mode():
            decoded_mel=flow.inference(torch.tensor([codes],dtype=torch.int32),torch.tensor([len(codes)],dtype=torch.int32),
                prompt_codes,prompt_lengths,mels,mel_lengths,embedding,10)
            audio,_=hift(decoded_mel)
        pcm=audio.squeeze().cpu().numpy()
        if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('invalid reconstructed waveform')
        duration=len(pcm)/24000
        if abs(duration-len(codes)/25)>0.25:raise ValueError('unexpected decoder duration')
        rms=float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        if rms<1e-5:raise ValueError('decoder produced effectively silent output')
        nas_root(root)
        dest=out/scene['scene_id']/(turn['utterance_id']+'.reconstructed.wav')
        partial=dest.with_name(dest.stem+'.partial.wav')
        sf.write(partial,pcm,24000,subtype='PCM_16')
        with partial.open('rb') as f:os.fsync(f.fileno())
        with wave.open(str(partial)) as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getnframes())!=(24000,1,2,len(pcm)):
                raise ValueError('saved reconstruction WAV differs from expected format or length')
        os.replace(partial,dest)
        result=dict(scene_id=scene['scene_id'],utterance_id=turn['utterance_id'],
            path=str(dest.relative_to(root)),sha256=digest(dest),sample_rate=24000,duration_seconds=duration,
            token_count=len(codes),prompt_utterance_id=pt['utterance_id'],
            prompt_source_emotion=pt['emotion'],prompt_scene_id=ps['scene_id'],
            same_utterance_prompt=pt['utterance_id']==turn['utterance_id'],
            prompt_channel_isolation_verified=False,rms=rms,clipped_fraction=float(np.mean(np.abs(pcm)>=1)),
            elapsed_seconds=time.monotonic()-started,runtime='pytorch-cpu-float32',
            strict_state_load=True,upstream_cuda_output_parity='NOT_VERIFIED',auditory_review='NOT_PERFORMED',training_ready=False)
        write_json(dest.with_suffix('.json'),result);records.append(result)
        write_json(out/'decode-progress.json',dict(reconstructed=len(records),requested_scene=scene_id,training_ready=0))
        print(f"{scene['scene_id']} {turn['utterance_id']}: reconstructed {duration:.2f}s; listening QA pending",flush=True)
    write_json(out/('decode-summary-'+scene_id+'.json' if scene_id else 'decode-summary.json'),dict(
        records=records,weights=WEIGHTS,training_ready=0,scope='actual CPU reconstruction candidates; no auditory or CUDA-parity approval'))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--scene')
    parser.add_argument('--output-relative',default='output/pilot10')
    args=parser.parse_args();decode(args.root,args.scene,args.output_relative)
