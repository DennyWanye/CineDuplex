"""Isolated native encoder-context causal diagnosis; no decoder or public audio."""
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
    store=Store('file://'+str(root),'');prefix=str(call.folder.relative_to(root))+'/encoder-context'
    begun=time.monotonic();rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
    prior_seed=torch.initial_seed();threads=torch.get_num_threads()
    versions={n:p._version for n,p in resident.named_parameters()}
    globals_before=(hf_v9_realtime.ListeningControlLogitsProcessor,hf_v9_realtime.SpeakingControlLogitsProcessor,
        hf_v9_realtime.HFRealtimeV9GenerationFramework.create_incremental_stream_session,vllm_generation.SYSTEM_MESSAGE_PREFIX)
    result=dict(call=call.id,precision='bf16',quantization='none',microphone=False,speaker=False,
        recognized_text_input=False,quality_accepted=False,public_audio_released=False,
        scope='isolated single-GPU native modules; not production latency or long-call acceptance',cases=[],errors=[])
    shadow=None;session=None
    def get(item):return store.get(item['path'],item['sha256'])
    def budget():
        if call.closed.is_set() or time.monotonic()-begun>900:raise RuntimeError('recorded A/B time/cancel budget')
    def progress(phase,**kw):call.emit(dict(type='diagnostic_progress',phase=phase,**kw))
    def module(key):
        mod=types.ModuleType('private_'+key);exec(compile(get(spec[key]),'<pinned-'+key+'>','exec'),mod.__dict__);return mod
    def save():
        target=call.folder/'encoder-context-progress.json';tmp=target.with_suffix('.pending');tmp.write_text(json.dumps(result,ensure_ascii=False));tmp.replace(target)
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
        sampling=module('sampling');bound,sampling_receipt=sampling.install(shadow);shadow._sample_from_logits=bound
        module('stream').install(shadow)
        turn={};native_turn_policy.install(native_prefix=False,speech_detector=resident._cineduplex_vad_detector,turn_control=turn)
        result['sampling']=sampling_receipt
        saved=torch.load(io.BytesIO(get(spec['checkpoint'])),map_location='cpu',weights_only=True)
        load_adapter(shadow,saved['adapter']);prepare_adapter(shadow,.25)
        def encode(waveform):
            # Call the same pure preprocessing with only its window extent changed.
            view=copy.copy(fw);view.window_second=len(waveform)/16000.
            values=view.stepaudio_audio_prepreocess(waveform)
            if len(values['feats'])!=1:raise ValueError('context must be one encoder window')
            mel=values['feats'][0].t().unsqueeze(0).to(shadow.device,dtype=torch.bfloat16)
            lengths=torch.tensor(values['feats_lengths'],device=shadow.device,dtype=torch.int32)
            encoded,lens=shadow.encoder(mel,lengths);encoded=shadow.adapter(encoded)
            size=int(((lens-1)//2+1)[0]);return encoded[0,:size].detach()
        def frontend_context(waveform):
            # Use the same received waveform/STFT as full-context encoding,
            # but retain independent 400ms encoder attention and positions.
            # Each original window has 40 valid Mel frames plus two framing
            # frames; slicing at stride 40 preserves exactly five output patches.
            view=copy.copy(fw);view.window_second=len(waveform)/16000.
            values=view.stepaudio_audio_prepreocess(waveform)
            mel=values['feats'][0]
            n=len(waveform)//6400
            pieces=[mel[i*40:i*40+42] for i in range(n)]
            if any(len(x)!=42 for x in pieces):raise ValueError('frontend context slice alignment')
            outputs=[]
            # Keep batch size one, as in the baseline, to avoid a numerical
            # batch-size change becoming a second variable.
            for piece in pieces:
                encoded,lens=shadow.encoder(piece.t().unsqueeze(0).to(shadow.device,dtype=torch.bfloat16),
                    torch.tensor([40],device=shadow.device,dtype=torch.int32))
                encoded=shadow.adapter(encoded)
                count=int(((lens-1)//2+1)[0])
                if count!=5:raise ValueError('frontend context patch count')
                outputs.append(encoded[0,:count].detach())
            return torch.cat(outputs)
        arms=([{'name':'original'},{'name':'frontend_refresh'},{'name':'turn_refresh_original'}]
              if spec.get('encoder_stage_comparison') else
              [{'name':'turn_refresh_original','scale':.25},{'name':'turn_refresh_base','scale':0.}]
              if spec.get('turn_refresh') else
              [{'name':'original'},{'name':'whole_recording_oracle'},{'name':'causal_prefix'}])
        for arm in arms:
            prepare_adapter(shadow,arm.get('scale',.25))
            for case in spec['cases']:
                budget();progress('recorded-native-replay',arm=arm['name'],case=case['id'])
                raw=get(case)
                with wave.open(io.BytesIO(raw)) as wav:
                    if (wav.getframerate(),wav.getsampwidth(),wav.getnchannels())!=(16000,2,1):raise ValueError('recording format')
                    pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768.
                torch.manual_seed(spec['seed']);fw.last_ss_pos=None
                fw.model._merge_cache_by_past_id.clear();turn.clear()
                vllm_generation.SYSTEM_MESSAGE_PREFIX='<|BOT|>system\n'+conversation_instruction(spec.get('language','auto'))+'<|EOT|>'
                session=fw.create_incremental_stream_session(start_speak_token_factor=1.,start_listen_token_factor=1.,audio_incremental_mode=True)
                ingress=StreamingSpeech(resident._cineduplex_vad_detector);revision=0
                audio=np.concatenate((pcm,np.zeros(12*16000,dtype=np.float32)))
                events=[];words=[];t=time.monotonic();context_rows=[];state={'end':0,'refreshed':False}
                original_cache=session._cache_precomputed_audio
                with torch.inference_mode():oracle=encode(pcm) if arm['name']=='whole_recording_oracle' else None
                def cache(current,feats,lens):
                    end=state['end'];start=end-6400
                    if len(feats)!=1:raise ValueError('expected one 400ms feature window')
                    if arm['name'].startswith('turn_refresh') or arm['name']=='frontend_refresh':
                        original_cache(feats,lens)
                        if state['refreshed'] or current.listening_state!='l' or not ingress.evidence.can_reply(0):return
                        old=torch.cat(current._audio_embed_seq_cache)
                        complete=(frontend_context(audio[:end]) if arm['name']=='frontend_refresh' else encode(audio[:end]))
                        if len(complete)!=end//1280:raise ValueError('context refresh feature count')
                        current._audio_embed_seq_cache=list(complete.split(5))
                        current._audio_embed_lens_cache=[len(x) for x in current._audio_embed_seq_cache]
                        current._rebuild_packed_audio_embed_buffer()
                        for name in ('past_key_values','stoken_past_key_values','control_past_key_values','merge_past_key_values'):setattr(current,name,None)
                        fw.model._merge_cache_by_past_id.clear()
                        state['refreshed']=True
                        diff=(complete.float()-old.float())
                        context_rows.append(dict(end=end,features=len(complete),future_visible=False,all_received_history_refreshed=True,
                            intervention='frontend-only' if arm['name']=='frontend_refresh' else 'frontend-and-encoder-context',
                            feature_relative_l2=float(diff.norm()/old.float().norm().clamp_min(1e-12)),
                            feature_cosine=float(F.cosine_similarity(complete.float().flatten(),old.float().flatten(),dim=0))))
                        return
                    if arm['name']=='original' or (oracle is not None and start>=len(pcm)):
                        return original_cache(feats,lens)
                    if oracle is not None:
                        seq=oracle[start//1280:end//1280]
                    else:
                        all_features=encode(audio[:end]);seq=all_features[-5:]
                        if len(all_features)!=end//1280:raise ValueError('causal context temporal count')
                    if len(seq)!=5:raise ValueError('feature count must remain five')
                    current._audio_embed_seq_cache.append(seq)
                    current._audio_embed_lens_cache.append(5);current._append_packed_audio_embed(seq,5)
                    context_rows.append(dict(end=end,features=5,future_visible=oracle is not None and end<len(pcm)))
                session._cache_precomputed_audio=types.MethodType(cache,session)
                with torch.inference_mode():
                    for offset in range(0,len(audio),6400):
                        budget();packet=audio[offset:offset+6400];state['end']=offset+6400
                        if ingress.feed(packet):revision+=1
                        turn.update(revision=revision,caught_up=True,quiet=not ingress.evidence.active and ingress.evidence.quiet_frames>=20)
                        for event in session.advance_stream(packet,audio_is_incremental=True,emit_generation_complete=False):
                            if event['type'] in ('speaking_token','state_change','speaking_done'):events.append(dict(input_end_sample=offset+6400,**event))
                            if event['type']=='speaking_token':
                                fw.runtime_speech_steps+=1
                                token=event.get('stoken',-1);text=event.get('text_token',-1)
                                if 0<=text<151643:words.append(text)
                        if len(words)>=12:break
                row=dict(arm=arm['name'],case=case['id'],input=case,adapter_sha256=state_hash(saved['adapter']),
                    generated_text=fw.tokenizer.decode(words,skip_special_tokens=True),
                    generation_seconds=time.monotonic()-t,events=events,context_rows=context_rows,
                    scope='diagnostic only; whole_recording_oracle may see future; causal_prefix never sees future')
                result['cases'].append(row);save();progress('encoder-context-complete',arm=arm['name'],case=case['id'],text=row['generated_text'])
                session.close();session=None
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
