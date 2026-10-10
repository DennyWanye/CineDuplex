"""Autoregressive speech-token comparison; never decode ground truth as inference."""
import hashlib
import json
import torch
from transformers import LogitsProcessorList


class Allowed:
    def __init__(self,ids,prefix=(),eos=None,pad=None):
        self.ids=ids;self.prefix=list(prefix);self.step=0;self.eos=eos;self.pad=pad;self.ended=False
    def __call__(self,history,scores):
        if self.eos is not None and int(history[0,-1])==self.eos:self.ended=True
        if self.ended and self.pad is not None:
            mask=torch.full_like(scores,float('-inf'));mask[:,self.pad]=scores[:,self.pad];return mask
        ids=[self.prefix[self.step]] if self.step<len(self.prefix) else self.ids
        self.step+=1
        mask=torch.full_like(scores,float('-inf'));mask[:,ids]=scores[:,ids];return mask


def comparisons(model,rows,store,config):
    from cineduplex.train.four_worker import LoRALinear,state_hash,adapter_state
    model.eval();model.stream_generation_flag=True
    cfg=model.config
    # Token IDs come from the fixed tokenizer, not hand-invented labels.
    ids=config['token_ids'];steps=config.get('generation_steps',160)
    for sid in config.get('generation_scenes',('S03','S06','S07','S08')):
        candidates=[x for x in rows if x['scene_id']==sid]
        if not candidates:raise ValueError('generation scene has no audited training input: '+sid)
        row=next((x for x in candidates if x.get('original_emotion') not in (None,'neutral')),candidates[0])
        batch={k:v.to('cuda') for k,v in row['batch'].items()}
        prefix_len=batch['prefix_input_ids'].shape[1]
        labels=batch['labels'][0]
        useful=(labels!=-100)&(labels!=cfg.text_pad_token_id)&(labels!=ids['tts_pad'])
        starts=torch.nonzero(useful).flatten()
        if not len(starts):raise ValueError('no speech onset in example')
        cut=int(starts[0])
        if cut<=prefix_len:raise ValueError('no causal user context')
        suffix=cut-prefix_len
        audio=batch['audio_input_ids']
        audio=torch.nn.functional.pad(audio,(0,max(0,suffix+steps+2-audio.shape[1])),value=cfg.audio_pad_token_id)
        prompt={'input_ids':batch['input_ids'][:,:cut],
            'stoken_ids':batch['stoken_ids'][:,:suffix],
            'control_input_ids':batch['control_input_ids'][:,:suffix],
            'prefix_input_ids':batch['prefix_input_ids'],'audio_input_ids':audio,
            'wavs':batch['wavs'],'wav_lens':batch['wav_lens']}
        if ((prompt['stoken_ids']>=151696)&(prompt['stoken_ids']<151696+6561)).any():
            raise ValueError('ground-truth speech leaked into the inference prompt')
        fingerprint=hashlib.sha256()
        for key,value in sorted(prompt.items()):
            fingerprint.update(key.encode());fingerprint.update(str(tuple(value.shape)).encode())
            fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
        prompt_sha=fingerprint.hexdigest()
        # No answer token from any of the three channels is supplied in the prompt.
        for variant in ('base','trained'):
            for module in model.modules():
                if isinstance(module,LoRALinear):module.enabled=variant=='trained'
            torch.manual_seed(config['seed'])
            speech=Allowed(list(range(151696,151696+6561))+[ids['tts_end']],
                [cfg.stoken_delay_token_id]*cfg.stoken_delay_num+[ids['tts_start']])
            text=Allowed(list(range(151643))+[ids['eot'],ids['tts_pad']],eos=ids['eot'],pad=ids['tts_pad'])
            # Same externally requested speaking mode for both variants. This is
            # an expressive response diagnostic, not a turn-taking evaluation.
            control=Allowed([cfg.keep_speaking_token_id],prefix=[cfg.start_speaking_token_id])
            with torch.inference_mode():
                # The pinned generation helper slices the stoken stream at
                # cache_length-prefix_length. An empty cache plus a long
                # listening prompt produces a negative slice. Prefill the known
                # context explicitly, leaving the final prompt token for it.
                pre=dict(prompt)
                pre['input_ids']=prompt['input_ids'][:,:-1]
                pre['stoken_ids']=prompt['stoken_ids'][:,:-1]
                pre['control_input_ids']=prompt['control_input_ids'][:,:-1]
                positions=torch.arange(cut-1,device='cuda')
                context=model(**pre,use_cache=True,apply_internal_merge=False,cache_position=positions)
                # Merge uses the next known context text, never the target answer.
                merge_input=context.stoken_hidden_states+model.get_input_embeddings()(prompt['input_ids'][:,1:])
                merge=model.merge_model(inputs_embeds=merge_input,use_cache=True,cache_position=positions)
                result=model.multi_head_generate(**prompt,max_new_tokens=steps,temperature=.8,top_k=40,top_p=.9,
                    past_key_values=context.past_key_values,stoken_past_key_values=context.stoken_past_key_values,
                    control_past_key_values=context.control_past_key_values,merge_past_key_values=merge.past_key_values,
                    logits_processor=LogitsProcessorList([text]),
                    stoken_logits_processor=LogitsProcessorList([speech]),
                    control_logits_processor=LogitsProcessorList([control]),stoken_eos_token_id=ids['tts_end'])
            predicted=result['stoken_ids'][0,suffix:].tolist()
            raw=[x-151696 for x in predicted if 151696<=x<151696+6561]
            output={'scene_id':sid,'variant':variant,'source_row':row['source_row'],
                'target_utterance_id':row['target_utterance_id'],'raw_codes':raw,'generated_stoken_ids':predicted,
                'generated_text_ids':result['sequences'][0,cut:].tolist(),
                'generated_control_ids':result['control_ids'][0,suffix:].tolist(),
                'prompt_cut':cut,'prompt_sha256':prompt_sha,'target_tokens_in_prompt':False,'autoregressive':True,
                'adapter_sha256':state_hash(adapter_state(model)),'adapter_enabled':variant=='trained',
                'scope':'same-input expressive response with externally requested speaking mode',
                'quality_approved':False,'has_speech_codes':bool(raw),'seed':config['seed']}
            store.json(config['run_prefix']+'/generation/'+sid+'-'+variant+'.json',output)
            print('autoregressive generation',sid,variant,len(raw),'speech codes',flush=True)
            if not raw:raise ValueError('generation produced no speech codes')
    for module in model.modules():
        if isinstance(module,LoRALinear):module.enabled=True
    from cineduplex.train.decode_generated import decode
    decode(store,config)
