#!/usr/bin/env python3
"""Private silent reference, invoked only by the original admitted call worker.

Never start this over SSH. The parent supplies its actual call identity. Model
weights and the existing LoRA checkpoint remain read-only on the real NAS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = Path(request['environment_root']).resolve()
    nas = root.parents[1]
    call_root = args.request.parent.resolve()
    if not nas.is_mount() or not call_root.is_relative_to(nas/'output'):
        raise RuntimeError('actual NAS reference paths required')
    if os.getppid() != request['parent_pid'] or os.environ.get('CINEDUPLEX_REFERENCE_PARENT') != str(os.getppid()):
        raise RuntimeError('only the admitted call worker may invoke this reference')
    service = json.loads(Path(request['service_root'],'SERVICE.json').read_text())
    if service['pid'] != request['parent_pid'] or service['decode_precision'] != 'bf16' or service['quantization'] != 'none':
        raise RuntimeError('original worker identity/precision changed')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    sys.dont_write_bytecode = True
    for key, relative in [('TMPDIR','tmp'),('HF_HOME','hf-cache'),
            ('HF_MODULES_CACHE','hf-modules'),('XDG_CACHE_HOME','cache'),
            ('TORCH_HOME','torch-cache'),('TRITON_CACHE_DIR','triton-cache'),
            ('CUDA_CACHE_PATH','cuda-cache'),('NUMBA_CACHE_DIR','numba-cache'),
            ('MPLCONFIGDIR','mpl-cache')]:
        os.environ[key]=str(root/relative)
        (root/relative).mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      VLLM_USE_V1='0',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
                      LYCHEEFD_T2W_STRICT_STOKEN_RANGE='1')
    sys.path[:0]=[str(root/'source'),str(root/'source/third_party/Step-Audio2')]
    import numpy as np
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from lychee_fd.vllm_integration.model_lychee import LycheeDuplexForVLLM
    from lychee_fd.runtime.vllm_generation import VLLMGenerationFramework
    from lychee_fd.runtime import vllm_generation
    torch.set_num_threads(4)
    torch.manual_seed(request['seed'])
    torch.cuda.set_device(0)
    result={'at':time.time(),'microphone':False,'speaker':False,
            'dtype':'bfloat16','quantization':'none','quality_accepted':False,
            'official_revision':request['official_revision'],'phases':[]}
    def save():
        p=call_root/'official-reference.json'
        tmp=p.with_suffix('.pending')
        with tmp.open('w') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        os.replace(tmp,p)
    save()
    try:
        # A new native kernel must pass a small mathematical check before
        # allocating/loading the full model. This is not audio acceptance.
        with torch.inference_mode():
            q=torch.randn((16,28,128),device='cuda',dtype=torch.bfloat16)
            k=torch.randn((16,4,128),device='cuda',dtype=torch.bfloat16)
            v=torch.randn_like(k)
            cu=torch.tensor([0,16],device='cuda',dtype=torch.int32)
            out=flash_attn_varlen_func(q,k,v,cu,cu,16,16,causal=True)
            ref=torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0,1)[None],k.repeat_interleave(7,1).transpose(0,1)[None],
                v.repeat_interleave(7,1).transpose(0,1)[None],is_causal=True)
            ref=ref[0].transpose(0,1)
            error=float((out.float()-ref.float()).norm()/ref.float().norm())
            if not torch.isfinite(out).all() or error>.015:
                raise RuntimeError('GB10 BF16 attention failed reference: '+str(error))
            result['phases'].append({'phase':'native_attention','relative_l2':error,'ok':True})
            save()
        if request.get('kernel_only'):
            result['ok']=True
            return
        checkpoint=nas/request['checkpoint']
        payload=checkpoint.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=request['checkpoint_sha256']:
            raise RuntimeError('original adapter checkpoint changed')
        import io
        saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
        del payload
        if saved['step']!=3222:
            raise RuntimeError('wrong original training step')
        adapter=saved['adapter'];merged=set()
        adapter_mode=request.get('adapter_mode','checkpoint')
        if adapter_mode not in ('checkpoint','original-base'):
            raise RuntimeError('unknown controlled adapter comparison')
        original_loader=LycheeDuplexForVLLM.load_weights
        if request.get('native_state_compare'):
            if request.get('recognized_text_input') or request.get('prescribed_text') or request.get('capability_text'):
                raise RuntimeError('native comparison cannot use recognized or prescribed text')
            original_state_loader=torch.nn.Module.load_state_dict
            result['audio_weight_loads']=[]
            def strict_audio_state(module,state,strict=True,**kwargs):
                is_audio=type(module).__name__ in ('AudioEncoder','Adaptor')
                checked=original_state_loader(module,state,strict=True if is_audio else strict,**kwargs)
                if is_audio:
                    result['audio_weight_loads'].append({'class':type(module).__name__,
                        'keys':len(state),'missing':list(checked.missing_keys),'unexpected':list(checked.unexpected_keys)})
                    save()
                return checked
            torch.nn.Module.load_state_dict=strict_audio_state
            original_audio_projection=LycheeDuplexForVLLM._build_audio_embeds_for_positions
            result['audio_position_trace']=[]
            def traced_audio_projection(model,positions,hidden):
                from lychee_fd.vllm_integration.model_lychee import LycheeDuplexState as S
                embeds,ids=S.audio_embeds,S.audio_input_ids
                projected=original_audio_projection(model,positions,hidden)
                if embeds is not None and ids is not None:
                    lens=S.audio_feat_lens
                    count=int(lens.sum().item()) if lens is not None else int(embeds.shape[-2])
                    patches=int((ids==int(S.audio_patch_token_id or model.audio_patch_token_id)).sum().item())
                    row={'case':result.get('active_case'),'prefix':int(S.prefix_input_len or 0),
                        'positions':[int(positions.min().item()),int(positions.max().item())],
                        'audio_input_tokens':int(ids.numel()),'encoded_features':count,'patch_tokens':patches,
                        'projected_nonzero_rows':int((projected.float().abs().sum(-1)>0).sum().item()) if projected is not None else 0}
                    result['audio_position_trace'].append(row)
                    if count!=patches:save();raise RuntimeError('actual audio feature/layout count differs')
                    if len(result['audio_position_trace'])>3000:raise RuntimeError('native audio trace budget')
                return projected
            LycheeDuplexForVLLM._build_audio_embeds_for_positions=traced_audio_projection
            original_text_sample=LycheeDuplexForVLLM._sample_text_token_for_merge
            result['native_semantic_logits']=[]
            semantic_counts={}
            def traced_text_sample(model,hidden):
                chosen=original_text_sample(model,hidden)
                case=result.get('active_case')
                if case is not None and chosen<151643 and semantic_counts.get(case,0)<12:
                    # Observe the already chosen token. Never invoke a stateful
                    # processor twice or replace its decision. Timing here is
                    # diagnostic, not a production latency measurement.
                    raw=model._project_logits(model._select_last_hidden(hidden)).float().reshape(-1)
                    values,indices=raw[:151643].topk(8)
                    result['native_semantic_logits'].append({'case':case,'chosen':chosen,
                        'candidate_ids':indices.tolist(),'candidate_logits':values.tolist()})
                    semantic_counts[case]=semantic_counts.get(case,0)+1
                return chosen
            LycheeDuplexForVLLM._sample_text_token_for_merge=traced_text_sample
        text_contract_trace=[]
        single_text_sample=False
        if request.get('trace_text_sample_contract'):
            from lychee_fd.vllm_integration.model_lychee import LycheeDuplexState
            original_compute=LycheeDuplexForVLLM.compute_logits
            def traced_compute(model, hidden_states, sampling_metadata):
                native=LycheeDuplexState.out_text
                processors=LycheeDuplexState.text_processors
                before=[getattr(p,'cnt',None) for p in processors] if processors is not None else []
                if single_text_sample and native is not None:
                    # The forward pass already applied the text processor and
                    # used this chosen token to condition the acoustic branch.
                    LycheeDuplexState.text_processors=None
                try:
                    logits=original_compute(model,hidden_states,sampling_metadata)
                finally:
                    LycheeDuplexState.text_processors=processors
                if logits is not None and native is not None:
                    last=logits[-1:, :] if logits.dim()==2 else logits[:, -1, :]
                    if single_text_sample:
                        chosen=int(native)
                        if not 0<=chosen<last.shape[-1]:raise RuntimeError('native text token range')
                        value=last[:,chosen].clone()
                        if not torch.isfinite(value).all():raise RuntimeError('native text score not finite')
                        last.fill_(-float('inf'));last[:,chosen]=value
                    emitted=int(last.argmax(-1).item())
                    ids=LycheeDuplexState.text_input_ids
                    text_contract_trace.append({'single_text_sample':single_text_sample,
                        'input_length':int(ids.shape[-1]) if ids is not None else None,
                        'native_merge_text':int(native),'sampler_text':emitted,
                        'processor_count_before':before,
                        'processor_count_after':[getattr(p,'cnt',None) for p in processors] if processors is not None else []})
                    if len(text_contract_trace)>6000:raise RuntimeError('text contract trace budget')
                return logits
            LycheeDuplexForVLLM.compute_logits=traced_compute
        def same_effective_weights(model,weights):
            def merged_weights():
                for name,value in weights:
                    prefix=name.removesuffix('.weight')
                    a,b=prefix+'.lora_a',prefix+'.lora_b'
                    if a in adapter and adapter_mode=='checkpoint':
                        if b not in adapter or value.dtype!=torch.bfloat16:
                            raise RuntimeError('LoRA binding/precision mismatch')
                        with torch.inference_mode():
                            value=(value.to('cuda').float()+.25*(adapter[b].to('cuda')@adapter[a].to('cuda'))).bfloat16()
                        merged.add(prefix)
                    yield name,value
            returned=original_loader(model,merged_weights())
            if len(merged)!=(16 if adapter_mode=='checkpoint' else 0):
                raise RuntimeError('not all 16 original LoRA modules were merged')
            return returned
        LycheeDuplexForVLLM.load_weights=same_effective_weights
        vllm_generation.SYSTEM_MESSAGE_PREFIX=request['system_message_prefix']
        started=time.monotonic()
        fw=VLLMGenerationFramework(model_type='lychee_full_duplex',
            model_path=request['model_overlay'],device='cuda',torch_dtype=torch.bfloat16,
            align_audio_input=request['align_audio_input'],
            gpu_memory_utilization=.30,max_model_len=2048,max_num_seqs=1,
            max_num_batched_tokens=2048,enforce_eager=True,
            enable_prefix_caching=False,tensor_parallel_size=1,pipeline_parallel_size=1)
        if fw.vllm_engine.engine.model_config.quantization is not None:
            raise RuntimeError('quantization is forbidden')
        result['phases'].append({'phase':'load_same_weights','seconds':time.monotonic()-started,
                                'merged_modules':sorted(merged),
                                'adapter_mode':adapter_mode,'scale':.25 if merged else 0.})
        # The engine initializes its own RNG while loading. Bind the inference
        # seed after construction, rather than reporting a seed it overwrote.
        torch.manual_seed(request['seed'])
        if not fw.align_audio_input or fw.stoken_audio_end_id != 151696+6561:
            raise RuntimeError('reference input grid or actual codec vocabulary differs')
        result['phases'].append({'phase':'native_input_contract',
            'align_audio_input':fw.align_audio_input,'control_token_chunk_size':fw.control_token_chunk_size,
            'window_second':fw.window_second,'inference_seed':request['seed'],
            'engine_initialization_seed':fw.vllm_engine.engine.model_config.seed,
            'stoken_audio_end_id':fw.stoken_audio_end_id})
        save()
        original_speaking=fw.init_speaking_processor
        def set_prescribed_text(prescribed):
            fw.init_speaking_processor=original_speaking
            if prescribed is None:return
            if not isinstance(prescribed,str) or not 1<=len(prescribed)<=100:
                raise RuntimeError('bounded prescribed speech diagnostic required')
            from transformers import LogitsProcessorList
            target=fw.tokenizer.encode(prescribed,add_special_tokens=False)+[fw.eos_token_id]
            if any(t>=151643 for t in target[:-1]):
                raise RuntimeError('prescribed text contains protocol tokens')
            def prescribed_speaking(end_factor):
                _,speech=original_speaking(end_factor)
                class KnownText:
                    def __init__(self):
                        self.start=None
                        self.eos_token_id=fw.eos_token_id
                        self.text_pad_token_id=fw.tts_pad_id
                    def __call__(self,ids,scores):
                        length=int(ids.shape[-1])
                        if self.start is None:self.start=length
                        offset=length-self.start
                        if offset<0:raise RuntimeError('prescribed text position regressed')
                        token=target[offset] if offset<len(target) else fw.tts_pad_id
                        scores=scores.clone();scores.fill_(-float('inf'));scores[:,token]=0
                        return scores
                return LogitsProcessorList([KnownText()]),speech
            fw.init_speaking_processor=prescribed_speaking
            result['phases'].append({'phase':'prescribed_text_capability',
                'text':prescribed,'natural_reply_test':False,
                'sampling_parameters_changed':False,'weights_changed':False})
            save()
        set_prescribed_text(request.get('prescribed_text'))
        result['events']=[]
        result['native_case_states']=[]
        def run_cases(cases):
            session=fw.create_incremental_stream_session(audio_incremental_mode=True,
                                                         start_speak_token_factor=1.,start_listen_token_factor=1.)
            try:
                with torch.inference_mode():
                    for case in cases:
                        result['active_case']=case['id']
                        def state(stage):
                            result['native_case_states'].append({'case':case['id'],'stage':stage,
                                'listening_state':session.listening_state,
                                'text_positions':int(session.generated_input_ids.shape[1]),
                                'speech_positions':int(session.generated_stoken_ids.shape[1]),
                                'control_positions':int(session.generated_control_ids.shape[1]),
                                'text_tail':session.generated_input_ids[0,-14:].tolist(),
                                'speech_tail':session.generated_stoken_ids[0,-14:].tolist(),
                                'control_tail':session.generated_control_ids[0,-14:].tolist(),
                                'text_eot_count':int((session.generated_input_ids==fw.eos_token_id).sum().item()),
                                'speech_end_count':int((session.generated_stoken_ids==fw.tts_end_id).sum().item()),
                                'next_chunk_pos':int(session.next_chunk_pos),
                                'audio':session._audio_cache_stats()})
                        if request.get('native_state_compare'):state('before')
                        path=nas/case['path']; blob=path.read_bytes()
                        if hashlib.sha256(blob).hexdigest()!=case['sha256']:
                            raise RuntimeError('original reviewer recording changed')
                        with wave.open(str(path)) as f:
                            if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(1,2,16000):
                                raise RuntimeError('original PCM format changed')
                            pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768.
                        # Fixed recorded question followed by bounded digital silence.
                        settling=request.get('settling_seconds',24)
                        if type(settling) is not int or not 24<=settling<=48:
                            raise RuntimeError('recorded comparison settling budget')
                        audio=np.concatenate([pcm,np.zeros(16000*settling,dtype=np.float32)])
                        for offset in range(0,len(audio),6400):
                            for event in session.advance_stream(audio[offset:offset+6400],
                                    audio_is_incremental=True,emit_generation_complete=False):
                                if event['type'] in ('control_decision','state_change','speaking_token','speaking_done'):
                                    result['events'].append({'case':case['id'],
                                        'input_end_sample':min(offset+6400,len(audio)),**event})
                            if len(result['events'])>6000:
                                raise RuntimeError('reference event budget exceeded')
                        if request.get('native_state_compare'):state('after')
                        save()
            finally:
                session.close()
        if request.get('recognized_text_input'):
            from transformers import LogitsProcessorList
            from lychee_fd.runtime.vllm_generation import FixedTokenLogitsProcessor
            history=''
            result['recognized_text_comparison']=[]
            prefix_ids=fw.tokenizer.encode(request['system_message_prefix'],add_special_tokens=False)
            for case in request['cases']:
                recognized=case.get('recognized_text')
                if not isinstance(recognized,str) or not 1<=len(recognized)<=120 or '<|' in recognized:
                    raise RuntimeError('bounded original recorded transcript required')
                transcript_source=nas/case['transcript_source']
                transcript_blob=transcript_source.read_bytes()
                if hashlib.sha256(transcript_blob).hexdigest()!=case['transcript_source_sha256']:
                    raise RuntimeError('original independent transcription evidence changed')
                if recognized not in transcript_blob.decode():
                    raise RuntimeError('recognized request is not in its actual source evidence')
                # Step-Audio2's actual formatter maps public "user" to the
                # checkpoint's "human" role. Keep that protocol verbatim.
                prompt=request['system_message_prefix']+history+'<|BOT|>human\n'+recognized+'<|EOT|><|BOT|>assistant\n'
                ids=fw.tokenizer.encode(prompt,add_special_tokens=False)
                length=len(ids)-len(prefix_ids)
                if not 1<=length<=1024:raise RuntimeError('private text comparison context budget')
                text_processor,_=original_speaking(1.)
                begun=time.monotonic();tokens=[]
                stream=fw.model.multi_head_generate_stream(
                    input_ids=torch.tensor([ids],device=fw.device),
                    stoken_ids=torch.full((1,length),fw.stoken_pad_token_id,device=fw.device,dtype=torch.long),
                    control_input_ids=torch.full((1,length),fw.sleep_token_id,device=fw.device,dtype=torch.long),
                    prefix_input_ids=torch.tensor([prefix_ids],device=fw.device),
                    audio_input_ids=torch.full((1,length),fw.audio_pad_token_id,device=fw.device,dtype=torch.long),
                    max_new_tokens=80,logits_processor=text_processor,
                    stoken_logits_processor=LogitsProcessorList([FixedTokenLogitsProcessor(fw.stoken_pad_token_id)]),
                    control_logits_processor=LogitsProcessorList([FixedTokenLogitsProcessor(fw.sleep_token_id)]),
                    temperature=.7,top_p=1,eos_token_id=fw.eos_token_id,
                    keep_request_alive=False)
                try:
                    with torch.inference_mode():
                        for step in stream:
                            token=int(step['text_token'])
                            if token==fw.eos_token_id:break
                            tokens.append(token)
                finally:stream.close()
                answer=fw.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                result['recognized_text_comparison'].append({'case':case['id'],
                    'recognized_text':recognized,'generated_text':answer,
                    'seconds':time.monotonic()-begun,'natural_audio_end_to_end':False,
                    'transcript_source_sha256':case['transcript_source_sha256']})
                save()
                if not 1<=len(answer)<=100 or '<|' in answer:
                    raise RuntimeError('generated text outside private speech comparison budget')
                history+='<|BOT|>human\n'+recognized+'<|EOT|><|BOT|>assistant\n'+answer+'<|EOT|>'
                torch.manual_seed(request['seed'])
                set_prescribed_text(answer)
                run_cases([{**case,'id':'recognized-text:'+case['id']}])
        else:
            run_cases(request['cases'])
            if request.get('native_state_compare'):
                if len(request['cases'])!=2:raise RuntimeError('exact two original questions required')
                torch.manual_seed(request['seed'])
                run_cases([{**request['cases'][1],'id':'fresh-native:'+request['cases'][1]['id']}])
                result['native_comparison_scope']='same original second recording, continuous versus fresh native session; no ASR/prescribed answer or sampler change'
        if request.get('trace_text_sample_contract'):
            result['text_contract_baseline']=list(text_contract_trace)
            mismatches=[p for p in text_contract_trace if p['native_merge_text']!=p['sampler_text']]
            result['text_contract_mismatches']=mismatches
            save()
            # Apply the single-source text repair only when its exact contract
            # violation occurred on the original recording. No parameter grid.
            if mismatches:
                single_text_sample=True
                text_contract_trace.clear()
                torch.manual_seed(request['seed'])
                result['phases'].append({'phase':'single_text_sample_repair',
                    'weights_changed':False,'sampling_parameters_changed':False})
                run_cases([{**c,'id':'single-text-sample:'+c['id']} for c in request['cases']])
                result['text_contract_fixed']=text_contract_trace
                if any(p['native_merge_text']!=p['sampler_text'] for p in text_contract_trace):
                    raise RuntimeError('repaired text sample contract differs')
            else:
                result['single_text_sample_repair_not_run']='no actual contract violation on this input'
        if request.get('capability_text') is not None:
            torch.manual_seed(request['seed'])
            set_prescribed_text(request['capability_text'])
            # A fresh private session isolates speech capability from the
            # preceding natural conversation; never concatenate its audio.
            run_cases([{**request['cases'][0],'id':'prescribed-speech-capability'}])
        result['ok']=True
        for row in result.get('native_semantic_logits',[]):
            row['chosen_text']=fw.tokenizer.decode([row['chosen']],skip_special_tokens=True)
            row['candidate_text']=[fw.tokenizer.decode([token],skip_special_tokens=True) for token in row['candidate_ids']]
    except Exception as exc:
        result['ok']=False;result['error']=type(exc).__name__+': '+str(exc)[:800]
        raise
    finally:
        save()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        print(json.dumps({k:v for k,v in result.items() if k!='events'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
