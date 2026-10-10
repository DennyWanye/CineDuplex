"""Incremental native generation, including discarded listening channels.

V9 stores PAD in listening histories but retains the native sampled-text merge
KV. Sampling those discarded channels is part of its state, not UI output.
"""
import types
import torch

def validated_triplet(text,speech,control):
    # This host transfer is required by the native Python state machine.
    values=torch.stack((text[0,0],speech[0,0],control[0,0])).tolist()
    if min(values)<0:raise ValueError("TP sampling failed on leader")
    return values


def stream(self, **kw):
    from transformers.cache_utils import DynamicCache
    limit = int(kw.pop('max_new_tokens', 1))
    cached = kw.pop('pre_computed_audio', None)
    kw.pop('audio_embeds', None)
    self._realtime_audio = cached
    self._realtime_audio_embeddings = None
    # V9 discards listening outputs, but its merge cache is conditioned on
    # the sampled text. Forcing PAD here changes future speech state.
    self.lm_head.fixed_listening=False
    try:
        # Preserve all four native KV branches for the entire window.
        # Calling generate(max_new_tokens=1) for each token restarts its
        # prefill path and clones the full history on every 40ms step.
        use_cache=kw.get('use_cache',True)
        if not use_cache:raise ValueError('realtime streaming requires KV caching')
        prefix=kw['prefix_input_ids'];prefix_len=prefix.shape[1]
        ids=kw['input_ids'];speech_ids=kw['stoken_ids'];control_ids=kw['control_input_ids']
        caches={key:kw.get(key) for key in ('past_key_values','stoken_past_key_values',
            'control_past_key_values','merge_past_key_values')}
        if caches['past_key_values'] is None:
            caches={key:DynamicCache() for key in caches}
            if self.merge_model is None:caches['merge_past_key_values']=None
        temperature=kw.get('temperature',1.)
        top_k=kw.get('top_k');top_p=kw.get('top_p')
        for step in range(limit):
            start=caches['past_key_values'].get_seq_length()
            if not 0<=start<ids.shape[1]:raise ValueError('native KV/input alignment differs')
            suffix_start=max(0,start-prefix_len)
            curr_ids=ids[:,start:];curr_speech=speech_ids[:,suffix_start:];curr_control=control_ids[:,suffix_start:]
            pos=torch.arange(start,ids.shape[1],device=ids.device)
            output=self.forward(input_ids=curr_ids,stoken_ids=curr_speech,
                control_input_ids=curr_control,prefix_input_ids=prefix,
                audio_input_ids=kw['audio_input_ids'],stoken_mapping=None,
                wavs=kw.get('wavs'),wav_lens=kw.get('wav_lens'),
                apply_internal_merge=False,use_cache=True,cache_position=pos,**caches)
            next_text=self._sample_from_logits(output.logits[:,-1,:],ids,
                kw.get('logits_processor'),temperature,top_k,top_p,False)
            if self.merge_model is not None:
                hidden=output.stoken_hidden_states;length=hidden.shape[1]
                next_text_ids=next_text if length==1 else torch.cat((ids[:,start+1:start+length],next_text),1)
                merge=self.merge_model(inputs_embeds=hidden+self.get_input_embeddings()(next_text_ids),
                    attention_mask=None,position_ids=None,use_cache=True,
                    past_key_values=caches['merge_past_key_values'],cache_position=pos)
                caches['merge_past_key_values']=merge.past_key_values
                logits=self.lm_head(merge[0])[:,-1,:]
            else:logits=output.stoken_logits[:,-1,:]
            next_speech=self._sample_from_logits(logits,speech_ids,
                kw.get('stoken_logits_processor'),temperature,top_k,top_p,True)
            next_control=self._sample_from_logits(output.control_logits[:,-1,:],control_ids,
                kw.get('control_logits_processor'),temperature,top_k,top_p,False)
            # Return the caches actually produced by each branch. Some
            # cache implementations replace their container on update.
            caches['past_key_values']=output.past_key_values
            caches['stoken_past_key_values']=output.stoken_past_key_values
            caches['control_past_key_values']=output.control_past_key_values
            ids=torch.cat((ids,next_text),1);speech_ids=torch.cat((speech_ids,next_speech),1)
            control_ids=torch.cat((control_ids,next_control),1)
            if ids.shape[1]-prefix_len!=speech_ids.shape[1] or speech_ids.shape[1]!=control_ids.shape[1]:
                raise ValueError('native stream channels are not aligned')
            text,speech,control=validated_triplet(next_text,next_speech,next_control)
            result={'sequences':ids,'stoken_ids':speech_ids,'control_ids':control_ids,**caches}
            yield {**result, 'text_token': text, 'stoken_token': speech,
                   'control_token': control, 'step': step}
            if speech == kw.get('stoken_eos_token_id') or text == kw.get('eos_token_id'):
                break
    finally:
        self._realtime_audio = None
        self.lm_head.fixed_listening=False


def install(model):
    model.multi_head_generate_stream = types.MethodType(stream, model)
    model._cineduplex_generation_loop = 'native-channels-incremental'
