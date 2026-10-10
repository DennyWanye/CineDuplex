"""Resident trained model: real audio input, causal generation, fixed codec output."""
import ast
import hashlib
import io
import json
import os
import sys
import threading
import time
import types
import wave
import torch
from transformers import AutoTokenizer,LogitsProcessorList
from cineduplex.train.generate import Allowed
from cineduplex.train.four_worker import LoRALinear,adapter_state,state_hash
from cineduplex.train.decode_generated import WEIGHTS


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending-'+str(os.getpid()));tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(path)


def loader(store,config,tokenizer):
    source=store.get('loader-source',config['loader_sha256']).decode();tree=ast.parse(source)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in ('peft','training_utils','datasets_utils'))]
    mod=types.ModuleType('cineduplex_inference_audio');sys.modules[mod.__name__]=mod
    mod.rank0_print=lambda *a,**k:None
    exec(compile(tree,'nas-memory://LycheeFDDataset.py','exec'),mod.__dict__)
    args=json.loads(store.get(config['batch_prefix']+'/train-features.json'))['configuration']
    mod.datasets=types.SimpleNamespace(load_from_disk=lambda _:[])
    return mod.LazySupervisedDataset(tokenizer,types.SimpleNamespace(**args))


def prompt_from_audio(dataset,tokenizer,pcm,cfg,steps):
    import numpy as np
    acc={k:[] for k in ('user_audio','input_ids','stoken_ids','stoken_mapping','control_input_ids',
        'text_label_ids','stoken_label_ids','control_label_ids')}
    user={'start_time':0.,'end_time':len(pcm)/16000}
    waveform,n=dataset._load_user_segment(pcm,user)
    dataset._append_initial_user_turn(acc,{'ai_backchannel':[]},user,waveform,n)
    waveform=np.concatenate(acc['user_audio']);audio=dataset.stepaudio_audio_prepreocess(waveform)
    prefix=tokenizer('<|BOT|>system\nYou are a helpful assistant.<|EOT|>',add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
    def tensor(k):return torch.tensor(acc[k][0],dtype=torch.long,device='cuda').unsqueeze(0)
    suffix=len(acc['input_ids'][0]);aids=audio['input_ids']
    if len(aids)==suffix+1:aids=aids[:-1]
    if len(aids)!=suffix:raise ValueError('audio feature/token alignment differs')
    audio_ids=torch.tensor([aids+[cfg.audio_pad_token_id]*(steps+2)],device='cuda')
    prompt={'input_ids':torch.cat((prefix,tensor('input_ids')),1),'prefix_input_ids':prefix,
        'stoken_ids':tensor('stoken_ids'),'control_input_ids':tensor('control_input_ids'),
        'audio_input_ids':audio_ids,'wavs':audio['feats'][0].T.unsqueeze(0).to('cuda'),
        'wav_lens':torch.tensor(audio['feats_lengths'],dtype=torch.int32,device='cuda')}
    if ((prompt['stoken_ids']>=151696)&(prompt['stoken_ids']<158257)).any():raise ValueError('answer tokens in user prompt')
    return prompt,suffix


def predict(model,prompt,suffix,ids,seed,steps,variant):
    cfg=model.config;cut=prompt['input_ids'].shape[1]
    for m in model.modules():
        if isinstance(m,LoRALinear):m.enabled=variant=='trained'
    torch.manual_seed(seed)
    speech=Allowed(list(range(151696,158257))+[ids['tts_end']],
        [cfg.stoken_delay_token_id]*cfg.stoken_delay_num+[ids['tts_start']])
    text=Allowed(list(range(151643))+[ids['eot'],ids['tts_pad']],eos=ids['eot'],pad=ids['tts_pad'])
    control=Allowed([cfg.keep_speaking_token_id],[cfg.start_speaking_token_id])
    with torch.inference_mode():
        pre={**prompt,'input_ids':prompt['input_ids'][:,:-1],'stoken_ids':prompt['stoken_ids'][:,:-1],
             'control_input_ids':prompt['control_input_ids'][:,:-1]}
        positions=torch.arange(cut-1,device='cuda')
        context=model(**pre,use_cache=True,apply_internal_merge=False,cache_position=positions)
        merge_input=context.stoken_hidden_states+model.get_input_embeddings()(prompt['input_ids'][:,1:])
        merge=model.merge_model(inputs_embeds=merge_input,use_cache=True,cache_position=positions)
        result=model.multi_head_generate(**prompt,max_new_tokens=steps,temperature=.8,top_k=40,top_p=.9,
            past_key_values=context.past_key_values,stoken_past_key_values=context.stoken_past_key_values,
            control_past_key_values=context.control_past_key_values,merge_past_key_values=merge.past_key_values,
            logits_processor=LogitsProcessorList([text]),stoken_logits_processor=LogitsProcessorList([speech]),
            control_logits_processor=LogitsProcessorList([control]),stoken_eos_token_id=ids['tts_end'])
    codes=[x-151696 for x in result['stoken_ids'][0,suffix:].tolist() if 151696<=x<158257]
    if not codes:raise ValueError('model produced no speech codes')
    return codes,result['sequences'][0,cut:].tolist()


class Decoder:
    def __init__(self,store):
        import numpy as np
        import torchaudio
        import onnxruntime as ort
        from cineduplex.train.codec_config import flow_from_config
        from flashcosyvoice.modules.hifigan import HiFTGenerator
        from flashcosyvoice.utils.audio import mel_spectrogram
        self.flow=flow_from_config(store.get('raw/codec/flow.yaml'))
        self.flow.load_state_dict(torch.load(io.BytesIO(store.get('raw/codec/flow.pt',WEIGHTS['flow.pt'])),map_location='cpu',weights_only=True),strict=True)
        self.flow.to('cuda').float().eval();self.flow.scatter_cuda_graph(False)
        self.hift=HiFTGenerator()
        weights=torch.load(io.BytesIO(store.get('raw/codec/hift.pt',WEIGHTS['hift.pt'])),map_location='cpu',weights_only=True)
        self.hift.load_state_dict({k.replace('generator.',''):v for k,v in weights.items()},strict=True)
        self.hift.to('cuda').float().eval()
        # Reuse the existing pilot's independent neutral voice reference.
        blob=store.get('output/pilot10/S04/G00002_18_08_003.wav','3f4bc1b96bb01cc38645796ecef1db4f1c56320291f7399f865eca1fb2cf3745')
        with wave.open(io.BytesIO(blob)) as f:
            audio=torch.from_numpy(np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768.)
        token=json.loads(store.get('output/pilot10/S04/G00002_18_08_003.tokens.json'))
        if token['input_sha256']!='3f4bc1b96bb01cc38645796ecef1db4f1c56320291f7399f865eca1fb2cf3745' or token['offset_applied']!=0:raise ValueError('voice reference mismatch')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        spk=ort.InferenceSession(store.get('raw/codec/campplus.onnx',WEIGHTS['campplus.onnx']),sess_options=opts,providers=['CPUExecutionProvider'])
        feat=torchaudio.compliance.kaldi.fbank(audio.unsqueeze(0),num_mel_bins=80,dither=0,sample_frequency=16000);feat-=feat.mean(0,keepdim=True)
        embedding=torch.from_numpy(spk.run(None,{spk.get_inputs()[0].name:feat.unsqueeze(0).numpy()})[0])
        codes=torch.tensor([token['raw_codes']],dtype=torch.int32);lens=torch.tensor([codes.shape[1]],dtype=torch.int32)
        mel=mel_spectrogram(torchaudio.functional.resample(audio.unsqueeze(0),16000,24000)).transpose(1,2)
        mel_lens=torch.tensor([mel.shape[1]],dtype=torch.int32)
        mel=torch.nn.functional.pad(mel,(0,0,0,codes.shape[1]*self.flow.up_rate-mel.shape[1]),mode='replicate')
        self.prompt=[x.to('cuda') for x in (codes,lens,mel,mel_lens,embedding)]
    def wav(self,codes,seed):
        import numpy as np
        torch.manual_seed(seed)
        with torch.inference_mode():
            feat=self.flow.inference(torch.tensor([codes],device='cuda',dtype=torch.int32),
                torch.tensor([len(codes)],device='cuda',dtype=torch.int32),*self.prompt,10)
            audio,_=self.hift(feat)
        pcm=audio.squeeze().cpu().numpy()
        if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all() or np.sqrt(np.mean(pcm**2))<1e-5:raise ValueError('invalid generated waveform')
        stream=io.BytesIO()
        with wave.open(stream,'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes((np.clip(pcm,-1,1)*32767).round().astype('<i2').tobytes())
        return stream.getvalue()


def serve(model,store,config):
    import numpy as np
    base=store.file_path(config['batch_prefix']+'/web');base.mkdir(exist_ok=True)
    namespace=config.get('service_namespace','')
    if namespace not in ('','resident'):raise ValueError('invalid service namespace')
    web=base/namespace;web.mkdir(exist_ok=True)
    for child in ('queue','jobs','claimed'):(web/child).mkdir(exist_ok=True)
    service={'model_ready':False,'model':config['run_prefix'].split('/')[-2]+f':step-{config["stop_step"]}',
        'adapter_sha256':state_hash(adapter_state(model)),'pid':os.getpid(),'quality_approved':False,'namespace':namespace}
    stopped=threading.Event()
    def heartbeat():
        while not stopped.is_set():
            atomic(web/'SERVICE.json',{**service,'heartbeat':time.time()});stopped.wait(10)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        tokenizer=AutoTokenizer.from_pretrained(str(store.file_path('raw/lychee-tokenizer')),local_files_only=True,trust_remote_code=False,use_fast=False)
        dataset=loader(store,config,tokenizer);decoder=Decoder(store)
        model.eval();model.stream_generation_flag=True;service['model_ready']=True
        print('CINEDUPLEX REAL AUDIO INFERENCE READY',flush=True)
        duration=config.get('inference_service_seconds',21600)
        deadline=time.monotonic()+duration if duration else float('inf')
        while time.monotonic()<deadline and not (base/'STOP-SERVICE.json').exists():
            entries=sorted((web/'queue').glob('*.json'))
            if not entries:time.sleep(1);continue
            request=entries[0];jid=request.stem;request.rename(web/'claimed'/request.name)
            folder=base/'jobs'/jid;s=json.loads((folder/'status.json').read_text());s.update(state='RUNNING',started_at=time.time());atomic(folder/'status.json',s)
            try:
                blob=(folder/'input.wav').read_bytes()
                if hashlib.sha256(blob).hexdigest()!=s['input_sha256']:raise ValueError('input changed')
                with wave.open(io.BytesIO(blob)) as f:
                    if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):raise ValueError('invalid input format')
                    pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768.
                prompt,suffix=prompt_from_audio(dataset,tokenizer,pcm,model.config,320)
                for variant in ('base','trained'):
                    codes,text=predict(model,prompt,suffix,config['token_ids'],config['seed'],320,variant)
                    generated=decoder.wav(codes,config['seed']);(folder/(variant+'.wav')).write_bytes(generated)
                    s[variant+'_text']=tokenizer.decode(text,skip_special_tokens=True)
                    s[variant+'_sha256']=hashlib.sha256(generated).hexdigest()
                    atomic(folder/(variant+'.json'),{'raw_codes':codes,'text_ids':text,'autoregressive':True,
                        'target_tokens_in_prompt':False,'input_sha256':s['input_sha256'],'model':service['model'],
                        'adapter_enabled':variant=='trained','adapter_sha256':service['adapter_sha256']})
                s.update(state='SUCCEEDED',finished_at=time.time());atomic(folder/'status.json',s)
            except Exception as e:
                s.update(state='FAILED',error=type(e).__name__+': '+str(e)[:300],finished_at=time.time());atomic(folder/'status.json',s)
                print('audio test failed',jid,s['error'],flush=True)
            finally:
                for m in model.modules():
                    if isinstance(m,LoRALinear):m.enabled=True
                torch.cuda.empty_cache()
    finally:
        service['model_ready']=False;stopped.set();atomic(web/'SERVICE.json',{**service,'heartbeat':time.time()})


def run(config):
    """Load the saved adapter once on one GPU, under the controller's lease."""
    from cineduplex.train.memory_runtime import Store,load_model
    from cineduplex.train.four_worker import attach,load_adapter
    store=Store(config['url'],config['token'])
    receipt=json.loads(store.get(config['checkpoint']+'.json'))
    saved=torch.load(io.BytesIO(store.get(config['checkpoint'],receipt['sha256'])),map_location='cpu',weights_only=True)
    if saved['identity']['model']!=config['model'] or saved['step']!=config['stop_step']:
        raise ValueError('completed checkpoint identity differs')
    if state_hash(saved['adapter'])!=saved['metric']['adapter_sha256']:
        raise ValueError('saved adapter differs from its training receipt')
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    model=load_model(store,config['model'],None)
    if attach(model)!=saved['modules']:raise ValueError('adapter module structure differs')
    load_adapter(model,saved['adapter']);del saved
    model.only_training_stoken_and_control=True
    serve(model,store,config)
