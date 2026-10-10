"""Decode newly predicted tokens using fixed independent reference audio."""
import hashlib
import io
import json
from pathlib import PurePosixPath
import wave

WEIGHTS={'campplus.onnx':'a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73',
 'flow.pt':'15ccff24256ff61537c7f8b51e025116b83405f3fb017b54b008fc97da115446',
 'hift.pt':'3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879'}


def decode(store,config):
    import numpy as np
    import torch
    import torchaudio
    import onnxruntime as ort
    from cineduplex.train.codec_config import flow_from_config
    from flashcosyvoice.modules.hifigan import HiFTGenerator
    from flashcosyvoice.utils.audio import mel_spectrogram
    cfg=store.get('raw/codec/flow.yaml')
    if hashlib.sha1(b'blob '+str(len(cfg)).encode()+b'\0'+cfg).hexdigest()!='8638dca97881751e50c132498acbfa41ecf74d6f':
        raise ValueError('codec config differs from pinned source')
    flow=flow_from_config(cfg)
    flow.load_state_dict(torch.load(io.BytesIO(store.get('raw/codec/flow.pt',WEIGHTS['flow.pt'])),map_location='cpu',weights_only=True),strict=True)
    flow.to('cuda').float().eval();flow.scatter_cuda_graph(False)
    hift=HiFTGenerator()
    weights=torch.load(io.BytesIO(store.get('raw/codec/hift.pt',WEIGHTS['hift.pt'])),map_location='cpu',weights_only=True)
    hift.load_state_dict({k.replace('generator.',''):v for k,v in weights.items()},strict=True);del weights
    hift.to('cuda').float().eval()
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    spk=ort.InferenceSession(store.get('raw/codec/campplus.onnx',WEIGHTS['campplus.onnx']),
        sess_options=opts,providers=['CPUExecutionProvider'])
    data=config.get('data_input')
    scenes=[json.loads(line) for line in store.get(data['scenes_path'] if data else 'output/pilot10-current/scenes.jsonl',
        data['scenes_sha256'] if data else None).decode().splitlines()]
    records=[]
    for sid in config.get('generation_scenes',('S03','S06','S07','S08')):
        base=json.loads(store.get(config['run_prefix']+'/generation/'+sid+'-base.json'))
        scene=next(s for s in scenes if s['scene_id']==sid)
        target=next(t for t in scene['turns'] if t['utterance_id']==base['target_utterance_id'])
        candidates=[(s,t) for s in scenes for t in s['turns'] if s['source_revision']==scene['source_revision']
            and s.get('split_group')==scene.get('split_group') and s['scene_id']!=sid
            and s['coverage_requested']!='suppression' and t['speaker_id']==target['speaker_id']
            and t['utterance_id']!=target['utterance_id'] and t['text'].strip()!=target['text'].strip()
            and (t.get('original_emotion') or t.get('emotion'))=='neutral' and not t.get('overlaps')
            and 1<=(t['end_sample']-t['start_sample'])/16000<=12]
        if not candidates:raise ValueError('no independent neutral reference')
        _,ref=max(candidates,key=lambda x:x[1]['end_sample']-x[1]['start_sample'])
        audio_bytes=store.get(ref['audio']['path'],ref['audio']['sha256'])
        with wave.open(io.BytesIO(audio_bytes)) as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):raise ValueError('prompt format')
            audio=torch.from_numpy(np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768)
        code_path=str(PurePosixPath(ref['audio']['path']).with_suffix('.tokens.json'))
        tokens=json.loads(store.get(code_path))
        if (tokens['input_sha256']!=ref['audio']['sha256'] or tokens['offset_applied']!=0
            or tokens['model_sha256']!='d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71'):
            raise ValueError('reference token source changed')
        codes=torch.tensor([tokens['raw_codes']],dtype=torch.int32)
        lens=torch.tensor([codes.shape[1]],dtype=torch.int32)
        features=torchaudio.compliance.kaldi.fbank(audio.unsqueeze(0),num_mel_bins=80,dither=0,sample_frequency=16000)
        features-=features.mean(dim=0,keepdim=True)
        embedding=torch.from_numpy(spk.run(None,{spk.get_inputs()[0].name:features.unsqueeze(0).numpy()})[0])
        audio24=torchaudio.functional.resample(audio.unsqueeze(0),16000,24000)
        mel=mel_spectrogram(audio24).transpose(1,2);mel_lens=torch.tensor([mel.shape[1]],dtype=torch.int32)
        mel=torch.nn.functional.pad(mel,(0,0,0,codes.shape[1]*flow.up_rate-mel.shape[1]),mode='replicate')
        prompt=[x.to('cuda') for x in (codes,lens,mel,mel_lens,embedding)]
        for variant in ('base','trained'):
            pred=json.loads(store.get(config['run_prefix']+'/generation/'+sid+'-'+variant+'.json'))
            raw=pred['raw_codes']
            if not pred['autoregressive'] or pred['target_tokens_in_prompt'] or not raw or any(not 0<=x<6561 for x in raw):
                raise ValueError('not valid model-generated speech')
            torch.manual_seed(config['seed'])
            with torch.inference_mode():
                feat=flow.inference(torch.tensor([raw],device='cuda',dtype=torch.int32),
                    torch.tensor([len(raw)],device='cuda',dtype=torch.int32),*prompt,10)
                audio_out,_=hift(feat)
            pcm=audio_out.squeeze().cpu().numpy()
            if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('invalid waveform')
            rms=float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
            if rms<1e-5 or abs(len(pcm)/24000-len(raw)/25)>.25:raise ValueError('silent or wrong-duration waveform')
            stream=io.BytesIO()
            with wave.open(stream,'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000)
                f.writeframes((np.clip(pcm,-1,1)*32767).round().astype('<i2').tobytes())
            saved=store.put(config['run_prefix']+'/audio/'+sid+'-'+variant+'.wav',stream.getvalue())
            saved.update(scene_id=sid,variant=variant,autoregressive=True,target_tokens_in_prompt=False,
                reference_utterance=ref['utterance_id'],reference_audio_sha256=ref['audio']['sha256'],
                reference_source_label='neutral',same_utterance_prompt=False,duration_seconds=len(pcm)/24000,
                rms=rms,clipped_fraction=float(np.mean(np.abs(pcm)>=1)),listening_review='NOT_PERFORMED')
            store.json(config['run_prefix']+'/audio/'+sid+'-'+variant+'.json',saved);records.append(saved)
            print('WAV saved',sid,variant,saved['duration_seconds'],flush=True)
    store.json(config['run_prefix']+'/audio/summary.json',{'records':records,'quality_approved':False})
