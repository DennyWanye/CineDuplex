"""Continuous Lychee-FD sessions using upstream V9 control and codec caches.

Input and output use independent HTTP channels. Audio is never gated by VAD or
by the playback state. All session history stays in the model's incremental KV
cache; model control events, not an external turn detector, govern listening.
"""
import base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import queue
import threading
import time
import types
import uuid
import wave
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
from transformers import AutoTokenizer, LogitsProcessorList
from cineduplex.train.four_worker import LoRALinear, adapter_state, state_hash
from cineduplex.train.inference_service import Decoder
from cineduplex.guardrail.aliyun_guardrail import Config as GuardrailConfig, AliyunGuardrail
from cineduplex.guardrail.call_guardrail import CallGuardrail


def nas_available(path):
    """Require the actual task mount before creating any data directory."""
    for parent in (path,*path.parents):
        if parent.name in ('nas-inference','nas-realtime','nas-training') and parent.parent.name=='cineduplex-work':
            return os.path.ismount(parent)
    return False


def atomic(path,value):
    # Keep a loaded model alive during transport loss, but never create a
    # local shadow of NAS for its heartbeat. Starts are rejected separately.
    if not nas_available(path):return False
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending-'+str(os.getpid()))
    tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(path)
    return True


def optimize_decode(model):
    """Restore resident original weights; inference never uses weight quantization."""
    restored=0
    for name,module in list(model.named_modules()):
        if module.__class__.__name__=='DecodeInt4Linear':
            parent,_,leaf=name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model,leaf,module.original)
            restored+=1
    record={'precision':'bf16','enabled':False,'restored_layers':restored,
            'weight_quantization':'none','activation_quantization':'none',
            'remaining_quantized_layers':sum(m.__class__.__name__=='DecodeInt4Linear' for m in model.modules())}
    model._realtime_quantization=record
    return record


def prepare_adapter(model,scale):
    """Standard inference-only LoRA merge, retaining original BF16 base weights."""
    count=0
    with torch.inference_mode():
        for layer in model.modules():
            if not isinstance(layer,LoRALinear):continue
            if not hasattr(layer,'_realtime_merged_weights'):
                layer._realtime_merged_weights={0.:layer.base.weight.detach()}
            weights=layer._realtime_merged_weights
            if scale not in weights:
                weights[scale]=(weights[0.].float()+scale*(layer.lora_b@layer.lora_a)).bfloat16()
            layer._realtime_effective_weight=weights[scale]
            layer.enabled=scale>0;layer.realtime_scale=scale
            def merged_forward(m,x):
                return torch.nn.functional.linear(x,m._realtime_effective_weight,m.base.bias)
            layer.forward=types.MethodType(merged_forward,layer);count+=1
    return {'mode':'bf16-merged-lora','layers':count,'scale':scale,'checkpoint_changed':False}


def optimize_bf16(model,root):
    """Qualify same-precision GEMV by shape, retaining original prefill math."""
    if getattr(model,'_realtime_bf16_acceleration',{}).get('recipe')=='v4':return model._realtime_bf16_acceleration
    record={'mode':'original-bf16','shapes':[],'layers':0,'quantization':'none','recipe':'v4','fusions':[]}
    try:
        import importlib.util,sys
        headers=root.parents[1]/'cache/train-pilot10/spark-4350/bf16-headers/unpacked/usr/include'
        if not (headers/'python3.12/Python.h').is_file():raise ValueError('matched Python headers not ready')
        os.environ['C_INCLUDE_PATH']=str(headers/'python3.12')+':'+str(headers)
        path=root/'bf16-kernels-v2.py'
        spec=importlib.util.spec_from_file_location('cineduplex_bf16_kernels_v1',path)
        kernels=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernels;spec.loader.exec_module(kernels)
        accepted={};candidates=[]
        with torch.inference_mode():
            for name,layer in model.named_modules():
                if not isinstance(layer,torch.nn.Linear) or not name.startswith(('model.','stoken_model.','control_model.','merge_model.','lm_head.')):continue
                if layer.weight.dtype!=torch.bfloat16 or not layer.weight.is_contiguous():continue
                shape=tuple(layer.weight.shape)
                if shape not in accepted:
                    n,k=shape;relative=0.;max_error=0.
                    def original(x):return torch.nn.functional.linear(x,layer.weight,layer.bias)
                    def candidate(x):return kernels.linear(x,layer.weight,layer.bias)
                    for phase in (0.,1.,2.):
                        x=torch.sin(torch.arange(k,device='cuda').float()*.017+phase).bfloat16().reshape(1,1,k)
                        ref=original(x);new=candidate(x)
                        if not torch.isfinite(new).all():raise ValueError('nonfinite BF16 kernel')
                        relative=max(relative,float((new.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-6)))
                        max_error=max(max_error,float((new.float()-ref.float()).abs().max()/ref.float().abs().max().clamp_min(1e-6)))
                    def bench(fn):
                        for _ in range(3):fn(x)
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(10):fn(x)
                        end.record();end.synchronize();return begin.elapsed_time(end)/10
                    baseline=bench(original);fast=bench(candidate)
                    ok=relative<=.003 and max_error<=.008 and fast<baseline*.9
                    accepted[shape]=ok
                    record['shapes'].append({'shape':shape,'relative_error':relative,'normalized_max_error':max_error,
                        'original_ms':baseline,'gemv_ms':fast,'enabled':ok})
                if accepted[shape]:candidates.append(layer)
            for layer in candidates:
                def fast_forward(m,x):
                    if x.is_cuda and x.dtype==torch.bfloat16 and x.numel()==m.in_features and x.is_contiguous():
                        return kernels.linear(x,m.weight,m.bias)
                    return torch.nn.functional.linear(x,m.weight,m.bias)
                layer.forward=types.MethodType(fast_forward,layer)
            record['layers']=len(candidates)
            if candidates:record['mode']='bf16-gemv-fp32-accumulation'
            fused={};installed=0
            for name,layer in model.named_modules():
                if not name.startswith(('model.','stoken_model.','control_model.','merge_model.')):continue
                kind=layer.__class__.__name__
                if kind=='Qwen2RMSNorm':
                    shape=('rms',layer.weight.numel(),float(layer.variance_epsilon))
                    fast=lambda x,m=layer:kernels.rms(x,m.weight,m.variance_epsilon)
                    original=getattr(layer,'_realtime_unfused_forward',layer.forward)
                    width=layer.weight.numel()
                elif kind=='Qwen2MLP' and getattr(layer,'act_fn',None).__class__.__name__ in ('SiLU','SiLUActivation'):
                    shape=('gated',tuple(layer.gate_proj.weight.shape))
                    fast=lambda x,m=layer:m.down_proj(kernels.gated(x,m.gate_proj.weight,m.up_proj.weight))
                    original=getattr(layer,'_realtime_unfused_forward',layer.forward)
                    width=layer.gate_proj.in_features
                else:continue
                if shape not in fused:
                    relative=0.;max_error=0.
                    for phase in (0.,1.,2.):
                        x=torch.sin(torch.arange(width,device='cuda').float()*.017+phase).bfloat16().reshape(1,1,width)
                        ref=original(x);new=fast(x)
                        if not torch.isfinite(new).all():raise ValueError('nonfinite BF16 fused output')
                        relative=max(relative,float((new.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-6)))
                        max_error=max(max_error,float((new.float()-ref.float()).abs().max()/ref.float().abs().max().clamp_min(1e-6)))
                    baseline=bench(original);speed=bench(fast)
                    fused[shape]=relative<=.003 and max_error<=.008 and speed<baseline*.9
                    record['fusions'].append({'kind':shape[0],'shape':str(shape[1:]),'relative_error':relative,
                        'normalized_max_error':max_error,'original_ms':baseline,'fused_ms':speed,'enabled':fused[shape]})
                if fused[shape]:
                    layer._realtime_unfused_forward=original
                    def fused_forward(m,x,fn=fast,base=original,size=width):
                        if x.is_cuda and x.dtype==torch.bfloat16 and x.numel()==size and x.is_contiguous():return fn(x)
                        return base(x)
                    layer.forward=types.MethodType(fused_forward,layer);installed+=1
            record['fused_layers']=installed
    except Exception as e:record['fallback_reason']=type(e).__name__+': '+str(e)[:400]
    model._realtime_bf16_acceleration=record
    atomic(root/'BF16-KERNEL-QUALIFICATION-v4.json',record)
    return record


from cineduplex.train.stage_meter import StageMeter



class RealtimeProjection(torch.nn.Module):
    """Project only permitted acoustic/control rows, with original BF16 weights.

    The original implementation projects the entire 158k vocabulary four times
    per speech step, including an unused pre-merge acoustic head. The text
    prefill and decode use the original BF16 head; all weights stay resident.
    """
    def __init__(self, original, config):
        super().__init__();self.original=original;self.phase='speech'
        self.config=config;self.configure(config,[])

    @property
    def weight(self):return self.original.weight

    def configure(self, config, extra):
        controls=[getattr(config,k) for k in ('keep_listening_token_id','start_speaking_token_id',
            'start_listening_token_id','keep_speaking_token_id','start_bc_token_id',
            'sleep_token_id','detect_token_id','keep_bc_token_id','end_bc_token_id')
            if getattr(config,k,None) is not None]
        speech=list(range(151696,158257))+extra+[config.stoken_delay_token_id,config.stoken_pad_token_id]
        for name,ids in (('control',controls),('speech',speech)):
            index=torch.tensor(sorted(set(ids)),device=self.weight.device,dtype=torch.long)
            if name+'_index' not in self._buffers:
                self.register_buffer(name+'_index',index)
                self.register_buffer(name+'_weight',self.weight.detach().index_select(0,index).contiguous())
            else:
                self._buffers[name+'_index']=index
                self._buffers[name+'_weight']=self.weight.detach().index_select(0,index).contiguous()

    def forward(self,x):
        prefill=x.shape[1]>2
        x=x[:,-1:]
        phase=self.phase
        if phase=='text':
            self.phase='unused'
            if getattr(self,'fixed_listening',False):
                return x.new_zeros((x.shape[0],1,self.weight.shape[0]))
            # Hot reload creates a new Python class while resident wrappers
            # retain the old class. Do not let isinstance select an int4 head
            # accidentally. Both prefill and decode retain BF16 weights.
            head=getattr(self.original,'original',self.original)
            return head(x)
        if phase=='unused':
            self.phase='control';return x.new_zeros((x.shape[0],1,self.weight.shape[0]))
        if phase=='control':self.phase='speech'
        name='control' if phase=='control' else 'speech'
        if name=='speech' and getattr(self,'fixed_listening',False):
            return x.new_zeros((x.shape[0],1,self.weight.shape[0]))
        values=torch.nn.functional.linear(x,getattr(self,name+'_weight'))
        scores=x.new_full((x.shape[0],1,self.weight.shape[0]),-float('inf'))
        scores[:,:,getattr(self,name+'_index')]=values
        return scores


def bind_incremental_model(model):
    """Reuse trained weights, adding the public runtime's precomputed-audio API.

    The training class has the same encoder/head weights but its forward path
    re-encodes wavs on every generation step. Assemble those identical input
    embeddings from the upstream session's cached encoder output instead.
    """
    # Hot updates retain the existing HTTP server and heartbeat threads.
    # Replace both old writer globals, too, so neither can create local data.
    import sys
    for name in ('cineduplex.train.realtime_service','cineduplex.train.inference_service'):
        if name in sys.modules:sys.modules[name].atomic=atomic
    forward = getattr(model, '_realtime_original_forward', model.forward)
    generate = getattr(model, '_realtime_original_generate', model.multi_head_generate)
    model._realtime_original_forward=forward
    model._realtime_original_generate=generate
    model._realtime_audio = None
    model._realtime_audio_embeddings = None
    if isinstance(model.lm_head,RealtimeProjection) or model.lm_head.__class__.__name__=='RealtimeProjection':
        model.lm_head.__class__=RealtimeProjection
    else:model.lm_head=RealtimeProjection(model.lm_head,model.config)

    def cached_forward(self, *args, **kw):
        cached = kw.pop('pre_computed_audio', None)
        if cached is None:
            cached = self._realtime_audio
        if cached is None or kw.get('inputs_embeds') is not None:
            self.lm_head.phase='text'
            try:return forward(*args, **kw)
            finally:self.lm_head.phase='speech'

        ids = kw['input_ids']; prefix = kw['prefix_input_ids']; pos = kw['cache_position']
        embeds = self.get_input_embeddings()(ids)
        if self.no_text_label:
            embeds[:, prefix.shape[1]:] = 0
        controls = self.get_input_embeddings()(kw['control_input_ids'])
        speech = self.get_input_embeddings()(kw['stoken_ids'])
        past = kw.get('past_key_values')
        if past is None or past.get_seq_length() == 0:
            zeros = embeds.new_zeros((embeds.shape[0], prefix.shape[1], embeds.shape[2]))
            controls = torch.cat((zeros, controls), 1)
            speech = torch.cat((zeros, speech), 1)
        embeds = embeds + controls + speech
        audio_ids = kw['audio_input_ids']
        features, lens = cached
        key=(audio_ids.data_ptr(),tuple(audio_ids.shape),features.data_ptr(),tuple(features.shape))
        prepared=self._realtime_audio_embeddings
        if prepared is None or prepared[0]!=key:
            audio = self.get_input_embeddings()(audio_ids)
            mask = torch.arange(features.shape[1], device=features.device)[None, :] < lens[:, None]
            values = features[mask].to(audio.dtype)
            locations = (audio_ids == 151690).unsqueeze(-1).expand_as(audio)
            if int(locations.sum()) != values.numel():
                raise ValueError('cached audio alignment differs from the model input')
            audio = audio.masked_scatter(locations, values)
            zeros = audio.new_zeros((embeds.shape[0], prefix.shape[1], audio.shape[2]))
            audio = torch.cat((zeros, audio), 1)
            self._realtime_audio_embeddings=(key,audio)
        else:audio=prepared[1]
        embeds = embeds + audio[:, pos]
        kw['inputs_embeds'] = embeds; kw['wavs'] = None; kw['wav_lens'] = None
        self.lm_head.phase='text'
        try:return forward(*args, **kw)
        finally:self.lm_head.phase='speech'

    from cineduplex.train.independent_sampling import install as install_sampling
    model._sample_from_logits,model._cineduplex_sampling_identity=install_sampling(model)
    model.forward = types.MethodType(cached_forward, model)
    from cineduplex.train.generation_stream import install as install_stream
    install_stream(model)
    model.stream_generation_flag = True
    from cineduplex.train.audio_embedding_slice import install as install_audio_slice
    install_audio_slice(model)


def conversation_instruction(language):
    preference={'en':'English','zh':'Chinese','auto':"the user's language"}[language]
    return ('You are a helpful voice assistant who speaks Chinese and English. '
        f'Initially prefer {preference}; change language when the user asks. '
        'Follow the latest spoken request, including topic changes. Stop the previous task '
        'when asked to do something else. Give the direct answer in ONE short sentence, '
        'normally at most 30 Chinese characters or 20 English words. Expand only when '
        'explicitly asked. Do not repeat the question, add a follow-up offer, or narrate '
        'your voice, acting, reasoning or instructions. If unclear, ask one short question. '
        'Refuse requests for sexual roleplay, sexual moaning or orgasm sounds, including '
        'requests framed as acting or imitation. Refuse assistance facilitating violence, '
        'self-harm, fraud or other serious wrongdoing; offer a brief safe alternative. '
        'For a sexual sound request say: 抱歉，我不能模仿性行为的声音。 '
        'Provide only the spoken answer, without stage directions.')


def context_budget(session, model):
    """Bound real cache positions, rather than disconnecting by wall time."""
    limit=int(model.config.text_config.max_position_embeddings)
    cache=session.past_key_values
    used=max(int(session.generated_input_ids.shape[1]),
             int(cache.get_seq_length()) if cache is not None else int(session.prefix_input_ids.shape[1]))
    # Reserve the next input window and generation/tail bookkeeping.
    return used,limit,max(0,limit-used-256)


def framework(model, tokenizer):
    from lychee_fd.runtime import hf_v9_realtime
    from lychee_fd.runtime import vllm_generation
    from lychee_fd.runtime.hf_v9_realtime import HFRealtimeV9GenerationFramework, _HFV9MultiHeadStreamAdapter
    from transformers import LogitsProcessor
    from cineduplex.train.native_turn_policy import install as install_native_turn_policy
    install_native_turn_policy()
    # Incremental sessions read this prefix when they are created. Keep the
    # native audio/control history; this is an instruction, not canned text.
    vllm_generation.SYSTEM_MESSAGE_PREFIX = (
        '<|BOT|>system\nYou are a helpful voice assistant. You understand and '
        'speak both English and Chinese. Reply in the language the user uses. '
        'Answer the latest question directly, in at most two short sentences. '
        'Do not claim you can only communicate in Chinese. If you cannot '
        'understand the audio, ask one short clarification question.<|EOT|>')
    optimize_decode(model)
    original_control = getattr(hf_v9_realtime, '_cineduplex_control_original', hf_v9_realtime.ListeningControlLogitsProcessor)
    hf_v9_realtime._cineduplex_control_original = original_control
    class MeasuredListeningControl(original_control):
        def __init__(self, *args, **kwargs):
            # Only three allowed rows are needed; avoid constructing the
            # 158k-element exclusion list on each 400ms listening window.
            self.kl_token_id=kwargs['kl_token_id'];self.ss_token_id=kwargs['ss_token_id']
            self.bc_token_id=kwargs.get('bc_token_id')
            self.start_speak_token_factor=kwargs.get('start_speak_token_factor',1.)
            self.bc_speak_token_factor=kwargs.get('bc_speak_token_factor',1.)
        def __call__(self, ids, scores):
            tokens = [self.kl_token_id, self.ss_token_id]
            names = ['kl', 'ss']
            if self.bc_token_id is not None:
                tokens.append(self.bc_token_id); names.append('bc')
            raw=scores[0,tokens].float()
            values=scores[:,tokens].clone();scores.fill_(-float('inf'));scores[:,tokens]=values
            if self.start_speak_token_factor>0:scores[:,self.ss_token_id]*=self.start_speak_token_factor
            if self.bc_token_id is not None and self.bc_speak_token_factor>0:
                # HF V9 uses the same multiplicative factor for BC. At its
                # native default 1.0 it must not add a backchannel bias.
                scores[:,self.bc_token_id]*=self.bc_speak_token_factor
            adjusted=scores[0,tokens].float()
            audit=torch.stack((raw,adjusted,torch.softmax(adjusted,0))).tolist()
            self.last_probs=dict(zip(names,audit[2]))
            model._realtime_control_audit={'labels':names,'raw_logits':audit[0],
                'adjusted_logits':audit[1],'probabilities':audit[2],
                'start_factor':self.start_speak_token_factor}
            return scores
    hf_v9_realtime.ListeningControlLogitsProcessor = MeasuredListeningControl
    fw = HFRealtimeV9GenerationFramework.__new__(HFRealtimeV9GenerationFramework)
    fw.model = _HFV9MultiHeadStreamAdapter(model); fw.tokenizer = tokenizer; fw.device = model.device
    cfg = model.config
    for alias, name in {
        'kl_token_id':'keep_listening_token_id', 'ss_token_id':'start_speaking_token_id',
        'sl_token_id':'start_listening_token_id', 'ks_token_id':'keep_speaking_token_id',
        'start_bc_token_id':'start_bc_token_id', 'detect_token_id':'detect_token_id',
        'sleep_token_id':'sleep_token_id', 'text_pad_token_id':'text_pad_token_id',
        'stoken_pad_token_id':'stoken_pad_token_id', 'audio_pad_token_id':'audio_pad_token_id',
        'stoken_delay_token_id':'stoken_delay_token_id', 'adding_text_hiddenstates':'adding_text_hiddenstates',
        'control_token_chunk_size':'control_token_chunk_size', 'stoken_delay_num':'stoken_delay_num'
    }.items():
        setattr(fw, alias, getattr(cfg, name))
    fw.keep_bc_token_id = getattr(cfg, 'keep_bc_token_id', None) or fw.start_bc_token_id
    fw.end_bc_token_id = getattr(cfg, 'end_bc_token_id', None) or fw.sl_token_id
    fw.bc_token_id = fw.start_bc_token_id
    for alias, token in [('audio_token_id','<audio_patch>'),('tts_pad_id','<tts_pad>'),
                         ('tts_end_id','<tts_end>'),('tts_start_id','<tts_start>'),('eos_token_id','<|EOT|>')]:
        setattr(fw, alias, tokenizer([token]).input_ids[0][0])
    # The trained loader alternates each 80ms encoder feature with an audio
    # pad, preserving the 40ms speech/control grid. Without those pads the
    # listening controller advances at half the trained rate.
    model.lm_head.configure(cfg,[fw.tts_pad_id,fw.tts_end_id,fw.tts_start_id])
    fw.runtime_speech_steps=0
    # The first speech window contains only delay/start tokens. Do not accept
    # a normal SL before that warmup plus one acoustic window has completed.
    # The guard is bounded, and native EOS/explicit interruptions still work.
    original_speaking=getattr(hf_v9_realtime,'_cineduplex_speaking_original',hf_v9_realtime.SpeakingControlLogitsProcessor)
    hf_v9_realtime._cineduplex_speaking_original=original_speaking
    class StreamingControl(original_speaking):
        def __call__(self,ids,scores):
            pos=ids.shape[1]%self.control_token_chunk_size
            allowed=([self.sl_token_id,self.ks_token_id] if pos==self.control_token_chunk_size-1
                else [fw.detect_token_id] if pos==self.control_token_chunk_size-2 else [fw.sleep_token_id])
            values=scores[:,allowed].clone();scores.fill_(-float('inf'));scores[:,allowed]=values
            if len(allowed)==2:
                self.last_probs=dict(zip(('sl','ks'),torch.softmax(values[0].float(),0).tolist()))
                if fw.runtime_speech_steps<=fw.stoken_delay_num+fw.control_token_chunk_size:
                    scores[:,self.sl_token_id]=-float('inf')
            return scores
    hf_v9_realtime.SpeakingControlLogitsProcessor=StreamingControl
    fw.align_audio_input = True; fw.allowing_backchannel = True; fw.is_old_sl_strategy = False
    fw.use_cache = True; fw.last_ss_pos = None; fw.sampling_rate = 16000
    fw.window_second = fw.control_token_chunk_size * .04; fw.stoken_no_repeat_n_gram = 4
    # Keep the pinned 6561-code vocabulary. The upstream processor allows a
    # broader historical interval; retain its delay/EOS logic and mask extras.
    class CodecRange(LogitsProcessor):
        def __call__(self, ids, scores):
            # Delay/padding IDs are above the codec range in this model. They
            # are part of its warmup protocol, not invalid audio codes.
            delay=scores[:, fw.stoken_delay_token_id].clone()
            pad=scores[:, fw.stoken_pad_token_id].clone()
            scores[:, 158257:] = -float('inf')
            scores[:, fw.stoken_delay_token_id]=delay
            scores[:, fw.stoken_pad_token_id]=pad
            return scores
    original = fw.init_speaking_processor
    def speaking(end_factor):
        text, speech = original(end_factor)
        fw.runtime_speech_steps=0
        for processor in (*text,*speech):
            for name,value in list(vars(processor).items()):
                if name.startswith('not_') and isinstance(value,list):
                    setattr(processor,name,torch.tensor(value,device=model.device,dtype=torch.long))
        speech.append(CodecRange())
        return text, speech
    fw.init_speaking_processor = speaking
    return fw


class StreamingCodec(Decoder):
    def prepare_voices(self, root):
        """Reuse resident codec weights; cache two source-backed speaker prompts."""
        if hasattr(self, '_voice_prompts'):
            return
        import hashlib
        import torchaudio
        import onnxruntime as ort
        from flashcosyvoice.utils.audio import mel_spectrogram
        manifest=json.loads((root/'voice-presets.json').read_text())
        nas=root.parent.parent
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        from cineduplex.train.inference_service import WEIGHTS
        blob=(nas/'raw/codec/campplus.onnx').read_bytes()
        if hashlib.sha256(blob).hexdigest()!=WEIGHTS['campplus.onnx']:
            raise ValueError('speaker encoder identity')
        session=ort.InferenceSession(blob,sess_options=opts,providers=['CPUExecutionProvider'])
        prompts={}
        for name, spec in manifest.items():
            blob=(nas/spec['audio']).read_bytes()
            if hashlib.sha256(blob).hexdigest()!=spec['sha256']:
                raise ValueError('voice reference identity')
            token=json.loads((root/spec['tokens']).read_text())
            if token['input_sha256']!=spec['sha256'] or token['offset_applied']!=0:
                raise ValueError('voice tokens identity')
            codes=token['raw_codes']
            if not codes or any(type(c) is not int or not 0<=c<6561 for c in codes):
                raise ValueError('voice token range')
            with wave.open(io.BytesIO(blob)) as f:
                if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):
                    raise ValueError('voice reference format')
                audio=torch.from_numpy(np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768.)
            feat=torchaudio.compliance.kaldi.fbank(audio.unsqueeze(0),num_mel_bins=80,dither=0,sample_frequency=16000)
            feat-=feat.mean(0,keepdim=True)
            embedding=torch.from_numpy(session.run(None,{session.get_inputs()[0].name:feat.unsqueeze(0).numpy()})[0])
            tokens=torch.tensor([codes],dtype=torch.int32)
            mel=mel_spectrogram(torchaudio.functional.resample(audio.unsqueeze(0),16000,24000)).transpose(1,2)
            mel=torch.nn.functional.pad(mel,(0,0,0,len(codes)*self.flow.up_rate-mel.shape[1]),mode='replicate')
            prompts[name]=[x.to('cuda') for x in (tokens,torch.tensor([len(codes)],dtype=torch.int32),
                mel,torch.tensor([mel.shape[1]],dtype=torch.int32),embedding)]
        self._voice_prompts=prompts;self.voice_specs=manifest
        self._stream_voice_caches={}

    def select_voice(self, root, name):
        self.prepare_voices(root)
        self.prompt=self._voice_prompts[name];self.voice=name
        # Streaming comparison uses a separate reference cache per speaker.
        if name in self._stream_voice_caches:
            self._reference_cache=self._stream_voice_caches[name]
        elif hasattr(self,'_reference_cache'):
            del self._reference_cache

    def complete(self, codes):
        if not codes or any(type(x) is not int or not 0<=x<6561 for x in codes):
            raise ValueError('model returned invalid speech codes')
        # The page already buffers a complete reply. Decode its native codes
        # in one pass instead of repeatedly stitching 160ms vocoder overlaps.
        # This preserves context and avoids small-chunk cache discontinuities.
        with torch.inference_mode():
            mel=self.flow.inference(torch.tensor([codes],device='cuda',dtype=torch.int32),
                torch.tensor([len(codes)],device='cuda',dtype=torch.int32),*self.prompt,10)
            audio=self.hift(mel)[0].squeeze().float().cpu().numpy()
        if audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError('invalid decoded waveform')
        return audio

    @staticmethod
    def reduce_noise(audio):
        """Conservative spectral attenuation, never a microphone/VAD gate.

        Estimate stationary floor from the quietest output frames. Bound
        attenuation to 18dB only in noise-dominated frequency bins.
        This does not repair incorrect acoustic codes or certify sound quality.
        """
        size=1024;hop=256
        if len(audio)<size:return audio.copy()
        padded=np.pad(audio,(size,size),mode='reflect')
        count=(len(padded)-size+hop-1)//hop+1
        padded=np.pad(padded,(0,(count-1)*hop+size-len(padded)))
        frames=np.lib.stride_tricks.sliding_window_view(padded,size)[::hop]
        window=np.sqrt(np.hanning(size)+1e-8)
        spectrum=np.fft.rfft(frames*window,axis=1)
        power=np.abs(spectrum)**2
        rms=np.sqrt(np.mean(frames**2,axis=1))
        quiet=rms<=np.quantile(rms,.15)
        floor=np.quantile(power[quiet],.3,axis=0)
        # Avoid erasing voicing in short utterances with no actual pauses.
        floor=np.minimum(floor,np.quantile(power,.1,axis=0))
        gain=np.sqrt(np.maximum(.016,1-2*floor[None,:]/(power+1e-12)))
        # Smooth neighboring bands and frames to reduce musical noise.
        gain=(np.pad(gain,((0,0),(1,1)),mode='edge')[:,:-2]+2*gain+
            np.pad(gain,((0,0),(1,1)),mode='edge')[:,2:])/4
        gain=(np.pad(gain,((1,1),(0,0)),mode='edge')[:-2]+2*gain+
            np.pad(gain,((1,1),(0,0)),mode='edge')[2:])/4
        # A gentle high-pass removes DC/low mechanical rumble only.
        frequencies=np.fft.rfftfreq(size,1/24000)
        gain*=np.minimum(1,frequencies/60)[None,:]
        rebuilt=np.fft.irfft(spectrum*gain,n=size,axis=1)*window
        output=np.zeros(len(padded));weight=np.zeros(len(padded))
        for index,frame in enumerate(rebuilt):
            offset=index*hop
            output[offset:offset+size]+=frame
            weight[offset:offset+size]+=window**2
        return (output/np.maximum(weight,1e-8))[size:size+len(audio)].astype(np.float32)

    @staticmethod
    def encode_pcm(audio,fade=True):
        # Remove DC and retain headroom rather than hard clipping loud samples.
        audio=audio-audio.mean()
        peak=float(np.max(np.abs(audio)))
        if peak>.95:audio=audio*(.95/peak)
        fade=min(120,len(audio)//2) if fade else 0
        if fade:
            ramp=np.linspace(0,1,fade,dtype=np.float32)
            audio[:fade]*=ramp;audio[-fade:]*=ramp[::-1]
        return (np.clip(audio,-1,1)*32767).round().astype('<i2').tobytes()

    def reset(self):
        codes, _, mel, _, embedding = self.prompt
        lookahead = max(1, int(getattr(self.flow, 'pre_lookahead_len', 3)))
        # The reference voice is fixed. The upstream estimator returns views
        # into mutable work buffers, so retain independent prompt tensors and
        # give every utterance its own copies. Never reuse previous speech.
        with torch.inference_mode():
            if not hasattr(self, '_reference_cache'):
                initial = self.flow.setup_cache(torch.cat((codes, codes[:, :lookahead]), 1),
                                               mel, embedding, n_timesteps=10)
                self._reference_cache = {k:v.detach().clone() for k,v in initial.items()}
            self.cache = {k:v.clone() for k,v in self._reference_cache.items()}
            if hasattr(self,'voice'):self._stream_voice_caches[self.voice]=self._reference_cache
        self.hift_cache = {'mel':torch.zeros(1, mel.shape[2], 0, device='cuda'),
                           'source':torch.zeros(1, 1, 0, device='cuda'),
                           'speech':torch.zeros(1, 0, device='cuda')}
        self.overlap = 8 * 480
        self.window = torch.from_numpy(np.hamming(2*self.overlap)).to('cuda')

    def pcm(self, codes, last=False):
        from token2wav import fade_in_out
        if any(type(x) is not int or not 0 <= x < 6561 for x in codes):
            raise ValueError('model returned invalid speech codes')
        with torch.inference_mode():
            mel, self.cache = self.flow.inference_chunk(
                token=torch.tensor([codes], dtype=torch.int32, device='cuda'),
                spk=self.prompt[-1], cache=self.cache, last_chunk=last, n_timesteps=10)
            if self.cache['estimator_att_cache'].shape[4] > self.prompt[2].shape[1]+100:
                self.cache['estimator_att_cache'] = torch.cat((
                    self.cache['estimator_att_cache'][:,:,:,:,:self.prompt[2].shape[1]],
                    self.cache['estimator_att_cache'][:,:,:,:,-100:]), 4)
            mel = torch.cat((self.hift_cache['mel'], mel), 2)
            speech, source = self.hift(mel, self.hift_cache['source'])
            if self.hift_cache['speech'].shape[-1]:
                speech = fade_in_out(speech, self.hift_cache['speech'], self.window)
            self.hift_cache = {'mel':mel[:,:,-8:].detach().clone(),
                               'source':source[:,:,-self.overlap:].detach().clone(),
                               'speech':speech[:,-self.overlap:].detach().clone()}
            if not last:
                speech = speech[:, :-self.overlap]
            audio = speech.cpu().numpy().reshape(-1)
        if not np.isfinite(audio).all():
            raise ValueError('codec returned nonfinite audio')
        self.last_raw_audio=audio.copy()
        return self.encode_pcm(audio,fade=False)


class CallArchive:
    """Ordered, bounded NAS writer; filesystem waits never hold inference locks."""
    def __init__(self, call):
        self.call=call;self.queue=queue.Queue();self.lock=threading.Lock()
        self.pending=0;self.peak=0;self.error=None;self.max_write_ms=0.;self.slow_writes=0
        self.files={'events':call.log,'client':call.client_log,'input':call.input_file,'output':call.output_file}
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def write(self, key, payload):
        size=len(payload.encode() if isinstance(payload,str) else payload)
        with self.lock:
            if self.error:raise RuntimeError('NAS diagnostics writer failed: '+self.error)
            if self.pending+size>16*1024*1024:raise RuntimeError('NAS diagnostics backlog exceeds 16 MiB')
            self.pending+=size;self.peak=max(self.peak,self.pending)
            self.queue.put_nowait((key,payload,size))
    def run(self):
        try:
            while True:
                item=self.queue.get()
                if item is None:break
                key,payload,size=item;begun=time.monotonic()
                if key not in self.files:self.files[key]=(self.call.folder/'codec-raw.pcm').open('ab')
                self.files[key].write(payload)
                elapsed=(time.monotonic()-begun)*1000
                self.max_write_ms=max(self.max_write_ms,elapsed);self.slow_writes+=elapsed>250
                with self.lock:self.pending-=size
            for file in self.files.values():file.close()
        except Exception as exc:
            self.error=type(exc).__name__+': '+str(exc)[:200]
            self.call.app.fatal_error='NAS diagnostics: '+self.error
            self.call.end_reason='diagnostics_write_failed';self.call.closed.set()
    def close(self):
        self.queue.put_nowait(None);self.thread.join(timeout=30)
        if self.thread.is_alive():
            self.call.app.fatal_error='NAS diagnostics writer did not drain within 30 seconds'
            raise RuntimeError(self.call.app.fatal_error)
        if self.error:raise RuntimeError(self.error)
        return {'mode':'bounded-background-writer','max_write_ms':round(self.max_write_ms,1),
                'writes_over_250ms':self.slow_writes,'peak_pending_bytes':self.peak,'pending_bytes':self.pending}


class Call:
    def __init__(self, app, variant, voice='female', probe=False, language='auto'):
        app.identity.update(codec_mode='native-stream-32-40',codec_chunk_tokens=40,codec_first_chunk_tokens=32,command_transport='fixed-byte-gloo',interrupt_policy='speech-confirmed-barge-in' if getattr(app,'tensor_parallel',None) is not None else 'native-control',tp_branch_layout='all-decoder-branches-sharded')
        app.model._realtime_interrupt_audit=None
        self.last_interrupt_audit=None
        self.app=app; self.id=str(uuid.uuid4()); self.variant=variant;self.voice=voice;self.probe=probe;self.language=language
        self.adapter_scale={'base':0.,'trained':.25,'trained-full':1.,
                            'trained-half':.5,'trained-quarter':.25}[variant]
        self.started=time.time(); self.last_input=self.started; self.last_seq=-1; self.received=0; self.processed=0
        self.input=queue.Queue(maxsize=75); self.synth=queue.Queue(maxsize=100)
        self.closed=threading.Event(); self.condition=threading.Condition(); self.events=deque(maxlen=1024)
        self.seq=0; self.epoch=0; self.end_reason=None; self.input_lock=threading.Lock();self.client_lock=threading.Lock();self.client_seq=-1
        self.output_samples=0;self.audio_chunk_seq=0;self.previous_output=None;self.previous_event=None
        self.folder=app.root/'calls'/self.id; self.folder.mkdir()
        self.log=(self.folder/'events.jsonl').open('a',buffering=1)
        self.client_log=(self.folder/'client.jsonl').open('a',buffering=1)
        self.input_file=(self.folder/'input.pcm').open('wb')
        self.output_file=(self.folder/'output.pcm').open('wb'); self.file_lock=threading.Lock()
        self.archive=CallArchive(self)
        self.guardrail=None
        if app.guardrail_client.config.enabled:
            self.guardrail=CallGuardrail(self,app.guardrail_client)
        self.emit({'type':'connected','session_id':self.id,'model':app.identity['model'],
                   'voice':self.voice,'language':self.language,'adapter_enabled':self.adapter_scale>0,'adapter_scale':self.adapter_scale,
                   'adapter_sha256':app.identity['adapter_sha256'],
                   'input_rate':16000,'output_rate':24000,'chunk_seconds':app.fw.window_second,
                   'runtime':dict(app.identity),'diagnostics_schema':2})
        self.worker=threading.Thread(target=self.loop,daemon=True)

    def emit(self, event):
        if self.guardrail is not None and self.guardrail.intercept(event):return
        self._emit(event)

    def _emit(self, event):
        with self.condition:
            self.seq+=1; value={**event,'seq':self.seq,'at':time.time()}
            self.events.append(value)
            recorded={k:v for k,v in value.items() if k!='pcm'}
            if 'pcm' in value:recorded['pcm_bytes']=len(base64.b64decode(value['pcm']))
            self.archive.write('events',json.dumps(recorded,ensure_ascii=False)+'\n')
            self.condition.notify_all()

    def stop(self, reason='hangup'):
        if self.end_reason is None:
            self.end_reason=reason
            # Publish the reason before closed becomes visible to long-poll.
            # Final cleanup may still be joining the codec worker.
            if not self.closed.is_set():self.emit({'type':'session_stopping','reason':reason})
        self.closed.set()
        with self.condition:self.condition.notify_all()

    def feed(self, seq, pcm):
        with self.input_lock:
            if self.closed.is_set():raise ValueError('通话已结束')
            if seq<=self.last_seq:return {'accepted':True,'duplicate':True}
            if seq!=self.last_seq+1:raise ValueError('音频包序号不连续，请重新连接')
            if not 0<len(pcm)<=32000 or len(pcm)%2:raise ValueError('无效PCM音频包')
            if self.input.full():self.stop('input_backlog');raise ValueError('模型处理积压，请结束后重试')
            self.archive.write('input',pcm)
            if self.guardrail is not None:self.guardrail.input_audio(seq,pcm)
            if self.closed.is_set():raise ValueError('安全审核不可用，通话已结束')
            self.input.put((time.time(),pcm));self.last_seq=seq;self.last_input=time.time();self.received+=len(pcm)//2
            return {'accepted':True,'received_seconds':self.received/16000,
                    'pending_seconds':(self.received-self.processed)/16000}

    def client_diagnostics(self,body):
        events=body.get('events',[])
        if not isinstance(events,list) or len(events)>64:raise ValueError('invalid diagnostics batch')
        allowed={'client_config','audio_receive','audio_schedule','audio_played','audio_cancel',
            'initial_buffer','rebuffer','transport_sample','client_error','call_stop','rtc_state','mark_problem'}
        with self.client_lock:
            for entry in events:
                if not isinstance(entry,dict) or entry.get('type') not in allowed:raise ValueError('invalid diagnostic event')
                seq=entry.get('client_seq')
                if type(seq) is not int:raise ValueError('invalid diagnostic sequence')
                if seq<=self.client_seq:continue
                # Bounded structured telemetry only. Never include arbitrary DOM,
                # credentials, device IDs, or signed URLs.
                if len(json.dumps(entry))>4000:raise ValueError('diagnostic event too large')
                self.archive.write('client',json.dumps({**entry,'server_received_at':time.time()},ensure_ascii=False)+'\n')
                self.client_seq=seq
        return {'accepted':True,'last_client_seq':self.client_seq}

    def codec_probe(self):
        """Bounded same-code comparison on the resident decoder, without an LLM call."""
        codec=self.app.codec
        try:
            codec.select_voice(self.app.root,self.voice)
            codes=self.app.codec._voice_prompts['female'][0][0].cpu().tolist()
            # Codec warmup runs only in an explicitly requested idle probe.
            # Cover first/steady/final shapes with real reference codes, preserve
            # every sample/flow step, and retain the resident weights/caches.
            warmed=getattr(codec,'_latency_warm_shapes',set())
            timings=[]
            if self.voice not in warmed:
                lookahead=int(codec.flow.pre_lookahead_len)
                for final_count in range(1,44):
                    begun=time.monotonic();codec.reset()
                    codec.pcm(codes[:32+lookahead])
                    codec.pcm(codes[32:32+40+lookahead])
                    codec.pcm(codes[72:72+final_count],last=True)
                    timings.append({'final_codes':final_count,'wall_ms':round((time.monotonic()-begun)*1000,1)})
                warmed.add(self.voice);codec._latency_warm_shapes=warmed
            codec.reset()
            self.emit({'type':'codec_warmup','voice':self.voice,'shapes':timings,
                'quality_changes':False,'model_weights_reloaded':False})
            atomic(self.folder/'probe.json',{'voice':self.voice,'reference':codec.voice_specs[self.voice],
                'warmup_shapes':timings,'model':self.app.identity,'subjective_listening':False})
        except Exception as e:self.emit({'type':'error','message':str(e)[:400]})
        finally:
            self.closed.set()
            self.emit({'type':'closed','reason':'codec_probe_complete'})
            self.archive.close()
            with self.app.lock:
                if self.app.active is self:self.app.active=None

    def loop(self):
        if self.probe:return self.codec_probe()
        from lychee_fd.runtime import vllm_generation
        from lychee_fd.runtime.vllm_generation import StreamingDecoder
        torch.set_num_threads(4)
        fw=self.app.fw
        session=None;synth=None;meter=None
        tp=getattr(self.app,'tensor_parallel',None);tp_started=False
        try:
            adapter=prepare_adapter(self.app.model,self.adapter_scale)
            acceleration=optimize_bf16(self.app.model,self.app.root)
            self.emit({'type':'runtime_config','adapter':adapter,'bf16_acceleration':acceleration,
                       'quantization':self.app.model._realtime_quantization,'model':dict(self.app.identity)})
            fw.last_ss_pos=None
            if tp is not None:
                tp.start(self.voice,self.language,self.adapter_scale,profile_id=self.id);tp_started=True
                qualification=tp.install(self.app.model)
                atomic(self.app.root/'TP-QUALIFICATION-rank3.json',qualification)
                self.emit({'type':'tensor_parallel_config',**qualification})
            fw.model._merge_cache_by_past_id.clear()
            # Preference is subordinate to the user's later spoken request.
            # Keep native audio/control history throughout this conversation.
            instruction=conversation_instruction(self.language)
            vllm_generation.SYSTEM_MESSAGE_PREFIX='<|BOT|>system\n'+instruction+'<|EOT|>'
            session=fw.create_incremental_stream_session(start_speak_token_factor=1.0,
                    start_listen_token_factor=1.0,audio_incremental_mode=True)
            decoder=StreamingDecoder(fw.tokenizer,fw,tts_chunk_size=10,end_event_on_generation_complete=False)
            synth=threading.Thread(target=self.synthesis,daemon=True);synth.start()
            meter=StageMeter(self.app.model)
            context_warned=False
            with torch.inference_mode():
                while not self.closed.is_set():
                    if time.time()-self.last_input>15 and self.input.empty():self.stop('input_timeout');break
                    used,limit,remaining=context_budget(session,self.app.model)
                    if remaining<fw.control_token_chunk_size*2:
                        self.stop('context_limit');break
                    if remaining<=1536 and not context_warned:
                        context_warned=True
                        self.emit({'type':'session_warning','message':'通话接近模型上下文上限，请结束后开始新通话。',
                                   'context_used':used,'context_limit':limit})
                    try:sent,pcm=self.input.get(timeout=.25)
                    except queue.Empty:continue
                    ingress_begun=time.monotonic()
                    if tp is not None:tp.audio(pcm)
                    dispatch_ms=(time.monotonic()-ingress_begun)*1000
                    audio=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.
                    begun=time.monotonic()
                    for raw in session.advance_stream(audio,audio_is_incremental=True,emit_generation_complete=False):
                        if raw['type']=='speaking_token':fw.runtime_speech_steps+=1
                        # Finish the current collective sequence on every
                        # rank even if the browser hangs up mid-window.
                        if self.closed.is_set():
                            if tp is None:break
                            continue
                        if raw['type']=='input_speech':
                            self.emit({**raw,'input_seq':self.processed//6400})
                            # With whole-answer review, generation may already
                            # be idle while its approved audio is still playing.
                            # A new utterance must also stop that playback.
                            if raw.get('interrupt_playback'):
                                self.epoch+=1
                                self.emit({'type':'stop_audio','epoch':self.epoch,
                                           'reason':'confirmed_user_speech'})
                            continue
                        if raw['type']=='interrupt_control':
                            self.emit({**raw,'input_seq':self.processed//6400})
                            continue
                        if raw['type']=='speaking_done':
                            self.emit({'type':'generation_end','reason':raw.get('reason'),
                                       'chunk':raw.get('chunk')})
                        for event in decoder.feed(raw):
                            kind=event['type']
                            if (kind=='state_change' and event['to']=='L'
                                    and event['from'] in ('S','B') and event.get('interrupt')):
                                # Normal completion also returns to listening.
                                # Preserve its tail; only an explicit native
                                # interruption invalidates already queued PCM.
                                self.epoch+=1
                                self.emit({**event,'type':'stop_audio','epoch':self.epoch})
                            if kind in ('event_start','audio_chunk','event_end'):
                                self.synth.put((self.epoch,event),timeout=2)
                            if kind in ('state_change','control_decision','text_delta','event_start','event_end'):
                                self.emit({k:v for k,v in event.items() if k!='stoken_ids'})
                                if kind=='control_decision' and event.get('state')=='s':
                                    audit=getattr(self.app.model,'_realtime_interrupt_audit',None)
                                    if audit:self.emit({'type':'interrupt_control','chunk':event.get('chunk'),**audit})
                                if kind=='control_decision' and event.get('state')=='l':
                                    audit=getattr(self.app.model,'_realtime_control_audit',None)
                                    if audit:self.emit({'type':'control_scores','chunk':event.get('chunk'),**audit})
                    audit=getattr(self.app.model,'_realtime_interrupt_audit',None)
                    if audit is not None and audit is not getattr(self,'last_interrupt_audit',None):
                        self.emit({'type':'interrupt_control','input_seq':self.processed//6400,**audit})
                        self.last_interrupt_audit=audit
                    if tp is not None:tp.finish_window()
                    self.processed+=len(audio)
                    self.emit({'type':'metrics','received_seconds':self.received/16000,
                        'processed_seconds':self.processed/16000,'pending_seconds':(self.received-self.processed)/16000,
                        'compute_ms':round((time.monotonic()-begun)*1000,1),
                        'tp_dispatch_ms':round(dispatch_ms,1),
                        'input_age_ms':round((time.time()-sent)*1000,1),
                        'input_seq':self.processed//6400-1,'rtf':round((time.monotonic()-begun)/(len(audio)/16000),3),
                        'stages':meter.result(),'synthesis_queue':self.synth.qsize(),
                        'cuda_allocated_mb':round(torch.cuda.memory_allocated()/1048576),
                        'context_used':int(session.generated_input_ids.shape[1]),'context_limit':limit,
                        'state':session.listening_state.upper()})
        except Exception as e:
            self.emit({'type':'error','message':type(e).__name__+': '+str(e)[:400]});self.stop('model_error')
            if tp is not None and not (isinstance(e,ValueError) and str(e)=='TP sampling failed on leader'):
                self.app.fatal_error=type(e).__name__+': '+str(e)[:400]
                atomic(self.app.root/'STOP.json',{'reason':'distributed_model_error',
                    'call_id':self.id,'error':self.app.fatal_error,'at':time.time()})
            import traceback;traceback.print_exc()
        finally:
            if tp_started:
                try:tp.stop()
                except Exception as e:self.emit({'type':'error','message':'TP stop: '+str(e)[:200]})
            self.closed.set()
            if synth:synth.join()
            if self.guardrail is not None:self.guardrail.close()
            if meter:meter.close()
            if session:session.close()
            fw.model._merge_cache_by_past_id.clear()
            try:
                self.emit({'type':'closed','reason':self.end_reason or 'hangup'})
                archive_stats=self.archive.close()
            except RuntimeError as exc:
                self.app.fatal_error='NAS diagnostics: '+str(exc)[:300]
                self.end_reason='diagnostics_write_failed'
                archive_stats={'mode':'bounded-background-writer','error':str(exc)[:300]}
            atomic(self.folder/'summary.json',{'id':self.id,'variant':self.variant,
                'started':self.started,'finished':time.time(),'reason':self.end_reason,
                'received_seconds':self.received/16000,'processed_seconds':self.processed/16000,
                'adapter_scale':self.adapter_scale,'voice':self.voice,'language':self.language,
                'model':self.app.identity,'model_controls_turns':True,'archive':archive_stats})
            with self.app.lock:
                if self.app.active is self:self.app.active=None
            torch.cuda.empty_cache()

    def synthesis(self):
        # Input/model and codec operate on independent tensors. Do not put
        # thousands of codec kernels ahead of the next input/model step on
        # CUDA's default stream. Join before another call reuses codec caches.
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.default_stream())
        try:
            with torch.cuda.stream(stream):self.synthesis_stream()
        finally:stream.synchronize()

    def synthesis_stream(self):
        codec=self.app.codec;tokens=[];all_codes=[];epoch=-1;event_id=None;initialized=False;emitted=0
        try:
            codec.select_voice(self.app.root,self.voice)
            lookahead=int(codec.flow.pre_lookahead_len);block=32
            def publish(codes,last=False):
                nonlocal emitted
                begun=time.monotonic();pcm=codec.pcm(codes,last=last)
                self.output(pcm,epoch,event_id,final_chunk=last,advance_seconds=(len(codes)-lookahead)*.04);emitted+=len(pcm)/48000
                self.emit({'type':'codec_metrics','event_id':event_id,'epoch':epoch,
                    'compute_ms':round((time.monotonic()-begun)*1000,1),
                    'audio_seconds':len(pcm)/48000,'mode':'stream-32-40','voice':self.voice,
                    'noise_reduction':'off-quality-baseline','codes':len(codes),'final_chunk':last,
                    'synthesis_queue':self.synth.qsize()})
            while not self.closed.is_set() or not self.synth.empty():
                try:current,event=self.synth.get(timeout=.2)
                except queue.Empty:continue
                if self.closed.is_set() or current!=self.epoch:continue
                if current!=epoch or event['type']=='event_start':
                    tokens=[];all_codes=[];emitted=0;block=32;epoch=current;initialized=True;codec.reset()
                    event_id=event.get('event_id',str(uuid.uuid4()))
                    self.emit({'type':'audio_start','event_id':event_id,'epoch':epoch,'voice':self.voice,
                               'streaming':True,'buffer_seconds':.8})
                if not initialized:continue
                if event['type']=='audio_chunk':
                    tokens.extend(event['stoken_ids']);all_codes.extend(event['stoken_ids'])
                    if len(all_codes)>1500:raise ValueError('单次回答超过60秒，请重新连接')
                    while len(tokens)>=block+lookahead:
                        publish(tokens[:block+lookahead]);del tokens[:block];block=40
                elif event['type']=='event_end':
                    if not event.get('interrupt'):
                        if tokens:publish(tokens,last=True)
                        atomic(self.folder/(event_id+'-codes.json'),{'raw_codes':all_codes,'voice':self.voice,
                            'text':event.get('text'),'native_model_codes':True})
                        if not all_codes:
                            self.emit({'type':'response_warning','event_id':event_id,'epoch':epoch,
                                'message':'模型这轮生成了文字，但没有生成声音。可以继续说话。'})
                        if epoch==self.epoch and not self.closed.is_set():
                            self.emit({'type':'audio_end','event_id':event_id,'epoch':epoch,'audio_seconds':emitted})
                    tokens=[];initialized=False
        except Exception as e:
            self.emit({'type':'error','message':'语音解码失败：'+str(e)[:300]});self.stop('codec_error')
            import traceback;traceback.print_exc()

    def output(self, pcm, epoch, event_id, *, final_chunk=False, advance_seconds=None):
        if not pcm or epoch!=self.epoch or self.closed.is_set():return
        signal=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.
        offset=self.output_samples;self.output_samples+=len(signal);self.audio_chunk_seq+=1
        boundary=None if self.previous_event!=event_id else float(abs(signal[0]-self.previous_output))
        self.previous_output=float(signal[-1]);self.previous_event=event_id
        self.archive.write('output',pcm)
        raw=getattr(self.app.codec,'last_raw_audio',None)
        if raw is not None:
            self.archive.write('raw',raw.astype('<f4').tobytes())
        self.emit({'type':'audio','pcm':base64.b64encode(pcm).decode(),'sample_rate':24000,
            'epoch':epoch,'event_id':event_id,'chunk_seq':self.audio_chunk_seq,'sample_offset':offset,
            'samples':len(signal),'rms_dbfs':round(float(20*np.log10(max(1e-9,np.sqrt(np.mean(signal**2))))),2),
            'peak':round(float(abs(signal).max()),5),'clipped_samples':int((abs(signal)>=.999).sum()),
            'boundary_jump':boundary,'final_chunk':final_chunk,'codec_advance_seconds':advance_seconds})


class Application:
    def __init__(self, model, codec, fw, root, identity):
        self.model=model;self.codec=codec;self.fw=fw;self.root=root
        self.identity=identity;self.lock=threading.Lock();self.active=None;self.calls={}
        self.guardrail_client=AliyunGuardrail(GuardrailConfig.from_env())
        self.identity['guardrail']=self.guardrail_client.config.status()
        (root/'calls').mkdir(exist_ok=True)

    def start(self, variant):
        # Existing hot-reload objects have not executed this revision's __init__.
        # Never silently skip an explicitly enabled but misconfigured guardrail.
        if not hasattr(self,'guardrail_client'):
            self.guardrail_client=AliyunGuardrail(GuardrailConfig.from_env())
        self.identity['guardrail']=self.guardrail_client.config.status()
        # Existing hot-reload HTTP handlers pass a single variant string.
        # Keep that wire contract backward compatible without restarting GPU.
        parts=variant.split(':')
        if len(parts)>3:raise ValueError('无效通话配置')
        variant=parts[0];voice=parts[1] if len(parts)>1 else 'female'
        language=parts[2] if len(parts)>2 else 'auto'
        if voice not in ('male','female'):raise ValueError('无效声音预设')
        if language not in ('auto','zh','en'):raise ValueError('无效对话语言')
        probe=variant=='codec-probe'
        if probe:variant='base'
        if variant not in ('base','trained','trained-full','trained-half','trained-quarter'):
            raise ValueError('无效模型版本')
        if not nas_available(self.root):raise RuntimeError('存储连接已断开，请稍后重试')
        with self.lock:
            if getattr(self,'fatal_error',None) or (self.root/'STOP.json').exists():
                raise RuntimeError('模型服务正在恢复，暂时不能开始通话')
            if self.active is not None:raise RuntimeError('已有通话正在使用模型，请稍后再试')
            self.identity['decode_precision']=self.model._realtime_quantization['precision']
            self.identity['playback_mode']='stream-measured-adaptive-buffer'
            self.identity['language_instruction']='initial-preference-user-can-switch'
            self.identity['text_projection']='original-bf16-prefill-and-decode'
            self.identity['diagnostics_schema']=2
            self.identity['inference_optimization']='bf16-merged-lora/cached-input/native-stream-loop/restricted-heads/sdpa'
            self.identity['listening_channels']='native-sampled-merge-cache/pad-history'
            self.identity['generation_loop']=getattr(self.model,'_cineduplex_generation_loop','native-channels-incremental')
            self.identity['session_limit']='context-capacity-with-warning'
            self.identity['context_limit']=int(self.model.config.text_config.max_position_embeddings)
            self.identity['codec_mode']='native-stream-32-40'
            self.identity['codec_chunk_tokens']=40;self.identity['codec_first_chunk_tokens']=32
            self.identity['voice_presets']=['female','male']
            self.identity['output_noise_reduction']='off-quality-baseline'
            self.identity['trained_adapter_scale']=.25
            self.identity['speech_control_projection']='original-bf16-selected-rows'
            call=Call(self,variant,voice,probe,language);self.active=call;self.calls[call.id]=call
            call.worker.start()
            if len(self.calls)>20:
                for k,c in list(self.calls.items()):
                    if c is not call and not c.worker.is_alive():del self.calls[k];break
            return call


def post_handler(app):
    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<=length<=32000:raise ValueError('packet size')
            data=self.rfile.read(length);bits=urlparse(self.path).path.strip('/').split('/')
            if bits==['tp_reload']:
                spec=json.loads(data)
                with app.lock:
                    tp=getattr(app,'tensor_parallel',None)
                    if tp is None:raise RuntimeError('this service has no TP group')
                    if app.active is not None:raise RuntimeError('stop active call before updating all ranks')
                    states=tp.update(spec)
                    app.identity['tp_runtime_sha256']=spec['sha256']
                return self.send(200,{'updated':True,'ranks':states,**app.identity})
            if bits==['reload']:
                # Only the controller's loopback tunnel can reach this
                # route; the LAN proxy never exposes it. Reuse loaded
                # weights when fixing this task's pinned runtime source.
                spec=json.loads(data)
                with app.lock:
                    if getattr(app,'tensor_parallel',None) is not None:
                        raise RuntimeError('TP runtime must update all four admitted ranks together')
                    if app.active is not None:raise RuntimeError('stop active call before runtime update')
                    name=spec['name']
                    if not name.startswith('runtime-') or not name.endswith('.py') or '/' in name:raise ValueError('runtime path')
                    source=(app.root/name).read_bytes()
                    import hashlib
                    if hashlib.sha256(source).hexdigest()!=spec['sha256']:raise ValueError('runtime source identity')
                    module=types.ModuleType('cineduplex.train.realtime_service_update')
                    exec(compile(source,str(app.root/name),'exec'),module.__dict__)
                    module.bind_incremental_model(app.model)
                    app.fw=module.framework(app.model,app.fw.tokenizer)
                    app.codec.__class__=module.StreamingCodec
                    app.__class__=module.Application
                    app.identity['runtime_sha256']=spec['sha256']
                return self.send(200,{'updated':True,**app.identity})
            if bits==['calls']:
                call=app.start(json.loads(data or b'{}').get('variant','trained'))
                return self.send(201,{'id':call.id,**app.identity})
            call=app.calls[str(uuid.UUID(bits[1]))]
            if bits[0]!='calls':raise ValueError('invalid route')
            if bits[2]=='diagnostics':return self.send(200,call.client_diagnostics(json.loads(data)))
            if bits[2]=='audio':return self.send(202,call.feed(int(self.headers['X-Audio-Seq']),data))
            if bits[2]=='stop':call.stop();return self.send(200,{'stopped':True})
            raise ValueError('unknown action')
        except RuntimeError as e:return self.send(409,{'error':str(e)})
        except (KeyError,ValueError,IndexError) as e:return self.send(400,{'error':str(e)})
    return do_POST


def serve(app, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def send(self, code, data):
            blob=json.dumps(data,ensure_ascii=False).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(blob)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(blob)
        def do_GET(self):
            path=urlparse(self.path)
            if path.path=='/status':return self.send(200,{**app.identity,
                'ready':nas_available(app.root) and not getattr(app,'fatal_error',None) and not (app.root/'STOP.json').exists(),
                'error':getattr(app,'fatal_error',None),'active':app.active.id if app.active else None})
            try:
                bits=path.path.strip('/').split('/');call=app.calls[str(uuid.UUID(bits[1]))]
                if bits[0]!='calls' or bits[2]!='events':raise ValueError('invalid route')
                after=int(parse_qs(path.query).get('after',['0'])[0]);deadline=time.monotonic()+15
                with call.condition:
                    while call.seq<=after and not call.closed.is_set() and time.monotonic()<deadline:
                        call.condition.wait(timeout=max(.01,deadline-time.monotonic()))
                    if call.events and after<call.events[0]['seq']-1:
                        return self.send(409,{'error':'事件接收中断，请重新连接'})
                    return self.send(200,{'events':[v for v in call.events if v['seq']>after],
                        'closed':call.closed.is_set(),'after':call.seq})
            except (KeyError,ValueError,IndexError):return self.send(404,{'error':'通话不存在'})
        do_POST=post_handler(app)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    stopped=app.root/'STOP.json'
    while not stopped.exists():
        atomic(app.root/'SERVICE.json',{**app.identity,'ready':True,'heartbeat':time.time(),'pid':os.getpid()})
        time.sleep(10)
    if app.active:
        app.active.stop('service_stop');app.active.worker.join(timeout=60)
    server.shutdown()
    atomic(app.root/'SERVICE.json',{**app.identity,'ready':False,'heartbeat':time.time(),'pid':os.getpid()})


def run(config):
    from cineduplex.train.memory_runtime import Store, load_model
    from cineduplex.train.four_worker import attach, load_adapter
    store=Store(config['url'],config['token']);root=store.file_path(config['realtime_prefix']);root.mkdir(exist_ok=True)
    atomic(root/'SERVICE.json',{'ready':False,'phase':'loading','pid':os.getpid(),'heartbeat':time.time()})
    receipt=json.loads(store.get(config['checkpoint']+'.json'))
    saved=torch.load(io.BytesIO(store.get(config['checkpoint'],receipt['sha256'])),map_location='cpu',weights_only=True)
    if saved['identity']['model']!=config['model'] or saved['step']!=config['stop_step']:raise ValueError('checkpoint identity')
    if state_hash(saved['adapter'])!=saved['metric']['adapter_sha256']:raise ValueError('adapter identity')
    # Import runtime dependencies before the expensive weights read.
    from lychee_fd.runtime.hf_v9_realtime import HFRealtimeV9GenerationFramework
    torch.set_num_threads(4);torch.cuda.set_device(0)
    loading_done=threading.Event()
    def loading_heartbeat():
        while not loading_done.is_set():
            atomic(root/'SERVICE.json',{'ready':False,'phase':'loading','pid':os.getpid(),'heartbeat':time.time()})
            loading_done.wait(10)
    loader=threading.Thread(target=loading_heartbeat,daemon=True);loader.start()
    try:model=load_model(store,config['model'],None,cancelled=lambda:(root/'STOP.json').exists())
    except Exception as e:
        loading_done.set();loader.join()
        atomic(root/'SERVICE.json',{'ready':False,'phase':'failed','pid':os.getpid(),'heartbeat':time.time(),'error':str(e)[:300]})
        raise
    finally:loading_done.set();loader.join()
    if attach(model)!=saved['modules']:raise ValueError('adapter structure')
    load_adapter(model,saved['adapter']);del saved
    model.only_training_stoken_and_control=True;model.eval()
    bind_incremental_model(model)
    tokenizer=AutoTokenizer.from_pretrained(str(store.file_path('raw/lychee-tokenizer')),local_files_only=True,use_fast=False)
    fw=framework(model,tokenizer);codec=StreamingCodec(store)
    identity={'model':f'training-20261009-v1:step-{config["stop_step"]}',
        'adapter_sha256':state_hash(adapter_state(model)),'control_mode':'native-full-duplex',
        'input_sample_rate':16000,'output_sample_rate':24000,'chunk_seconds':fw.window_second,
        'checkpoint_sha256':receipt['sha256'],'source_sha256':config['source_modules_sha256']}
    print('CINEDUPLEX REALTIME MODEL READY',flush=True)
    serve(Application(model,codec,fw,root,identity),config.get('realtime_port',18087))
