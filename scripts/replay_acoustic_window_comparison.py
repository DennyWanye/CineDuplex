"""Silent native A/B replay, isolated modules and held-out reviewer audio only."""
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import types
import wave
import numpy as np
import torch
import torch.nn.functional as F


def run(call,spec):
    from accelerate import init_empty_weights
    from cineduplex.train.memory_runtime import Store
    from cineduplex.train.four_worker import attach,load_adapter,state_hash
    from cineduplex.train.realtime_service import bind_incremental_model,framework,prepare_adapter,conversation_instruction
    from cineduplex.train import native_turn_policy
    from lychee_fd.runtime import hf_v9_realtime,vllm_generation
    from cineduplex.train.speech_activity import StreamingSpeech
    resident=call.app.model;root=Path(call.app.root).parents[1]
    store=Store('file://'+str(root),'');prefix=str(call.folder.relative_to(root))+'/recorded-ab'
    begun=time.monotonic();rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
    prior_seed=torch.initial_seed();threads=torch.get_num_threads()
    versions={n:p._version for n,p in resident.named_parameters()}
    globals_before=(hf_v9_realtime.ListeningControlLogitsProcessor,hf_v9_realtime.SpeakingControlLogitsProcessor,
        hf_v9_realtime.HFRealtimeV9GenerationFramework.create_incremental_stream_session,vllm_generation.SYSTEM_MESSAGE_PREFIX)
    result=dict(call=call.id,precision='bf16',quantization='none',microphone=False,speaker=False,
        recognized_text_input=False,quality_accepted=False,public_audio_released=False,
        scope='isolated single-GPU native modules; not production latency or long-call acceptance',cases=[],errors=[])
    result['context_repair_only']=bool(spec.get('context_repair_only'))
    shadow=None;session=None
    def get(item):return store.get(item['path'],item['sha256'])
    def budget():
        if call.closed.is_set() or time.monotonic()-begun>900:raise RuntimeError('recorded A/B time/cancel budget')
    def progress(phase,**kw):call.emit(dict(type='diagnostic_progress',phase=phase,**kw))
    def module(key):
        mod=types.ModuleType('private_'+key);exec(compile(get(spec[key]),'<pinned-'+key+'>','exec'),mod.__dict__);return mod
    def save():
        target=call.folder/'recorded-ab-progress.json';tmp=target.with_suffix('.pending');tmp.write_text(json.dumps(result,ensure_ascii=False));tmp.replace(target)
    try:
        torch.set_num_threads(4);torch.manual_seed(spec['seed'])
        old_dtype=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
        try:
            with init_empty_weights(include_buffers=False):shadow=type(resident)(copy.deepcopy(resident.config))
        finally:torch.set_default_dtype(old_dtype)
        base={n.replace('.base.','.'):v.detach() for n,v in resident.state_dict().items()
              if '.lora_' not in n and not n.startswith('lm_head.')}
        base['lm_head.weight']=resident.lm_head.weight.detach();shadow.load_state_dict(base,strict=True,assign=True)
        shadow.requires_grad_(False)
        for child in shadow.modules():
            for name,value in list(child._buffers.items()):
                if value is not None:child._buffers[name]=value.to(resident.device)
        if any(p.data_ptr()!=base[n].data_ptr() for n,p in shadow.named_parameters()):raise ValueError('base identity')
        del base
        attach(shadow);shadow.eval();bind_incremental_model(shadow)
        fw=framework(shadow,call.app.fw.tokenizer)
        if 'output_contract' in spec:
            contract=module('output_contract')
            result['output_contract']=contract.install(fw)
            result['output_contract_validation']=contract.validate(fw,shadow.device)
        sampling=module('sampling');bound,sampling_receipt=sampling.install(shadow);shadow._sample_from_logits=bound
        module('stream').install(shadow)
        turn={};native_turn_policy.install(native_prefix=False,speech_detector=resident._cineduplex_vad_detector,turn_control=turn)
        result['sampling']=sampling_receipt
        # Independently observe sensitivity to audio features at the previously
        # identified first wrong semantic position. Never replace generated logits.
        chunks=[];acoustic=[];pending={};observing={'enabled':False,'nested':False,'count':0}
        ids={s:fw.tokenizer.encode(s,add_special_tokens=False) for s in ('安','坛','眼')}
        def before_model(m,args,kw):
            if not observing['enabled']:return
            cached=kw.get('pre_computed_audio')
            if cached is None:cached=m._realtime_audio
            features,lens=cached;audio_ids=kw['audio_input_ids'];positions=kw['cache_position']
            audio=torch.zeros((*audio_ids.shape,features.shape[-1]),device=features.device,dtype=features.dtype)
            valid=torch.arange(features.shape[1],device=features.device)[None,:]<lens[:,None]
            mask=audio_ids.eq(fw.audio_token_id).unsqueeze(-1).expand_as(audio)
            audio=audio.masked_scatter(mask,features[valid])
            zero=audio.new_zeros((1,kw['prefix_input_ids'].shape[1],audio.shape[-1]))
            pending['audio']=torch.cat((zero,audio),1)[:,positions].detach().clone()
        def after_core(m,args,kw,out):
            if not observing['enabled'] or observing['nested']:return
            chunks.append(kw['inputs_embeds'].detach().clone());acoustic.append(pending.pop('audio'))
            observing['count']+=1
            if observing['count']!=13:return
            observing['nested']=True
            try:
                full=torch.cat(chunks,1);audio=torch.cat(acoustic,1)
                outcomes={}
                for label,embeds in [('intact',full),('audio_features_zeroed',full-audio)]:
                    logits=F.linear(m(inputs_embeds=embeds,use_cache=False,output_hidden_states=False,return_dict=True)[0][:,-1:,:],shadow.lm_head.weight[:151643])[0,-1].float()
                    values,tokens=logits.topk(10)
                    outcomes[label]=dict(top=[dict(id=i,text=fw.tokenizer.decode([i]),logit=v) for v,i in zip(values.tolist(),tokens.tolist())],
                        selected={s:[float(logits[i]) for i in indexes] for s,indexes in ids.items()})
                result['semantic_audio_ablation']=dict(forward=13,positions=full.shape[1],audio_feature_norm=float(audio.float().norm()),
                    outcomes=outcomes,scope='counterfactual zero-feature sensitivity, not a normal input or fix',generation_modified=False)
            finally:observing['nested']=False;observing['enabled']=False
        # Native streaming calls .forward directly, bypassing Module.__call__
        # and its pre-hooks. Observe that actual entry without altering kwargs.
        original_forward=shadow.forward
        def observed_forward(current,*args,**kwargs):
            before_model(current,args,kwargs)
            return original_forward(*args,**kwargs)
        shadow.forward=types.MethodType(observed_forward,shadow)
        handles=[shadow.model.register_forward_hook(after_core,with_kwargs=True)]
        codec=call.app.codec;old_voice=getattr(codec,'voice','female');codec.select_voice(call.app.root,'female')
        for arm in spec['arms']:
            budget();saved=torch.load(io.BytesIO(get(arm['checkpoint'])),map_location='cpu',weights_only=True)
            # load_adapter selects trainable adapter tensors, so keep only them
            # trainable until after the identity check; inference disables grads.
            load_adapter(shadow,saved['adapter'])
            for child in shadow.modules():
                if child.__class__.__name__=='LoRALinear':
                    child._realtime_merged_weights={0.:child.base.weight.detach()}
            prepare_adapter(shadow,arm.get('scale',.25))
            fw.last_ss_pos=None;fw.model._merge_cache_by_past_id.clear();turn.clear()
            vllm_generation.SYSTEM_MESSAGE_PREFIX='<|BOT|>system\n'+conversation_instruction(spec.get('language','auto'))+'<|EOT|>'
            result['prefix_sha256']=hashlib.sha256(vllm_generation.SYSTEM_MESSAGE_PREFIX.encode()).hexdigest()
            result['policy_capabilities']={k:hasattr(native_turn_policy,k) for k in ('install_response_scope','install_listening_alignment')}
            session=fw.create_incremental_stream_session(start_speak_token_factor=1.,start_listen_token_factor=1.,audio_incremental_mode=True)
            if arm['name'].startswith('context_refresh'):
                result['context_policy']=module('audio_context').install(session,resident._cineduplex_vad_detector)
            ingress=StreamingSpeech(resident._cineduplex_vad_detector);revision=0
            observing['enabled']=arm['name']=='original' and 'audio_context' not in spec;observing['count']=0;chunks.clear();acoustic.clear()
            for case in spec['cases']:
                budget();progress('recorded-native-replay',arm=arm['name'],case=case['id'])
                raw=get(case)
                with wave.open(io.BytesIO(raw)) as wav:
                    if (wav.getframerate(),wav.getsampwidth(),wav.getnchannels())!=(16000,2,1):raise ValueError('recording format')
                    pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768.
                tail=int(spec.get('settling_seconds',24))
                if not 24<=tail<=48:raise ValueError('settling budget')
                audio=np.concatenate((pcm,np.zeros(tail*16000,dtype=np.float32)));events=[];speech=[];words=[];first=None;t=time.monotonic()
                with torch.inference_mode():
                    for offset in range(0,len(audio),6400):
                        budget();packet=audio[offset:offset+6400]
                        if ingress.feed(packet):revision+=1
                        turn.update(revision=revision,caught_up=True,quiet=not ingress.evidence.active and ingress.evidence.quiet_frames>=20)
                        for event in session.advance_stream(packet,audio_is_incremental=True,emit_generation_complete=False):
                            if event['type'] in ('speaking_token','state_change','speaking_done'):events.append(dict(input_end_sample=offset+6400,**event))
                            if event['type']=='speaking_token':
                                fw.runtime_speech_steps+=1
                                token=event.get('stoken',-1);text=event.get('text_token',-1)
                                if 151696<=token<158257:
                                    speech.append(token-151696)
                                    if first is None:first=time.monotonic()-t
                                if 0<=text<151643:words.append(text)
                row=dict(arm=arm['name'],scale=arm.get('scale',.25),case=case['id'],input=case,adapter_sha256=state_hash(saved['adapter']),
                    generated_text=fw.tokenizer.decode(words,skip_special_tokens=True),speech_codes=len(speech),
                    first_code_seconds=first,generation_seconds=time.monotonic()-t,events=events,
                    audio_context=list(getattr(session,'_cineduplex_audio_context_receipts',[])))
                row['settling_seconds']=tail
                row['speech_code_file']=store.put(prefix+'/'+arm['name']+'-'+case['id']+'-codes.json',json.dumps({'speech_codes':speech,'text_ids':words}).encode())
                if speech:
                    torch.manual_seed(spec['seed']);codec.reset();parts=[];raw_parts=[];offset=0;width=36
                    lookahead=int(codec.flow.pre_lookahead_len)
                    # Match normal synthesis exactly: lookahead is supplied
                    # again in the next window, not consumed as emitted audio.
                    while len(speech)-offset>=width+lookahead:
                        parts.append(codec.pcm(speech[offset:offset+width+lookahead],last=False));raw_parts.append(codec.last_raw_audio.copy())
                        offset+=width;width=40
                    if offset<len(speech):
                        parts.append(codec.pcm(speech[offset:],last=True));raw_parts.append(codec.last_raw_audio.copy())
                    row['codec_window_contract']='normal-live-lookahead-36-40'
                    pcm_bytes=b''.join(parts);buf=io.BytesIO()
                    with wave.open(buf,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm_bytes)
                    row['wav']=store.put(prefix+'/'+arm['name']+'-'+case['id']+'.wav',buf.getvalue())
                    floats=np.concatenate(raw_parts).astype('<f4');row['raw']=store.put(prefix+'/'+arm['name']+'-'+case['id']+'.f32',floats.tobytes())
                    row['audio_metrics']=dict(seconds=len(pcm_bytes)/48000,raw_peak=float(np.abs(floats).max()),
                        raw_clipped=int((np.abs(floats)>=1).sum()),raw_rms=float(np.sqrt(np.mean(floats**2))),finite=bool(np.isfinite(floats).all()))
                result['cases'].append(row);save();progress('recorded-case-complete',arm=arm['name'],case=case['id'],text=row['generated_text'],codes=len(speech))
            session.close();session=None
        for h in handles:h.remove()
        codec.select_voice(call.app.root,old_voice);codec.reset()
        result['completed']=True
    except Exception as exc:
        result['errors'].append(type(exc).__name__+': '+str(exc)[:600]);raise
    finally:
        if session is not None:session.close()
        hf_v9_realtime.ListeningControlLogitsProcessor,hf_v9_realtime.SpeakingControlLogitsProcessor,hf_v9_realtime.HFRealtimeV9GenerationFramework.create_incremental_stream_session,vllm_generation.SYSTEM_MESSAGE_PREFIX=globals_before
        result['resident_parameter_versions_unchanged']=versions=={n:p._version for n,p in resident.named_parameters()}
        result['elapsed_seconds']=time.monotonic()-begun
        store.json(prefix+'/RESULT.json',result)
        del shadow;gc.collect();torch.manual_seed(prior_seed);torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);torch.set_num_threads(threads)
