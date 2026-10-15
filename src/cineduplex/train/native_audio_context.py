"""Refresh received acoustic context once per completed utterance.

No transcript or answer is injected. Existing token histories remain intact;
only received waveform features and the four dependent decoder caches change.
The module is opt-in until real recorded-call quality validation passes.
"""
import copy
import math
import time
import types


class ReceivedUtteranceWindows:
    """Keep pending speech, not the silence since the previous answer.

    Two 400ms windows cover the existing VAD's 256ms onset hysteresis,
    pending 32ms frame, and the preceding packet. Original session audio
    features are never removed; this only selects the interval to refresh.
    """
    def __init__(self, capacity):
        if capacity < 2:
            raise ValueError('received context capacity')
        self.capacity = capacity
        self.chunks = []
        self.end = 0
        self.refreshed_utterance = 0

    @property
    def start(self):
        return self.end - len(self.chunks)

    def append(self, packet, utterance):
        self.chunks.append(packet)
        self.end += 1
        limit = self.capacity if utterance > self.refreshed_utterance else 2
        if len(self.chunks) > limit:
            del self.chunks[:-limit]

    def refreshed(self, utterance):
        self.refreshed_utterance = utterance
        self.chunks.clear()


def install(session, detector, *, continuous_received=False):
    import numpy as np
    import torch
    from cineduplex.train.speech_activity import StreamingSpeech

    if getattr(session, '_cineduplex_audio_context_installed', False):
        raise ValueError('audio context already installed')
    if session._is_vllm_backend():
        raise ValueError('native HF context implementation required')
    fw=session.fw
    model=getattr(fw.model, '_model', fw.model)
    window=int(fw.window_second*16000)
    if window!=6400:raise ValueError('expected native 400ms windows')
    # Mel framing adds 479 samples and the encoder downsamples by two.
    capacity=int(model.encoder.positional_embedding.weight.shape[0])*320
    max_windows=(capacity-479)//window
    if max_windows<1:raise ValueError('encoder context capacity')
    activity=StreamingSpeech(detector)
    state=ReceivedUtteranceWindows(max_windows)
    continuous_windows=[]
    receipts=[]
    original_split=session._split_incremental_windows
    original_cache=session._cache_precomputed_audio

    def split(current, audio, flush_audio_tail=False):
        windows=original_split(audio,flush_audio_tail=flush_audio_tail)
        # The public ingress currently supplies exactly one 400ms packet.
        # Reject ambiguous multi-window bookkeeping rather than use future input.
        if len(windows)>1:raise ValueError('audio context requires one native window per advance')
        for packet in windows:
            if len(packet)!=window:raise ValueError('audio context requires complete native window')
            activity.feed(packet)
            state.append(np.asarray(packet,dtype=np.float32).copy(),activity.evidence.utterance)
            if continuous_received:
                continuous_windows.append(np.asarray(packet,dtype=np.float32).copy())
                if len(continuous_windows)>max_windows:
                    del continuous_windows[:-max_windows]
        return windows

    def cache(current, feats, lengths):
        original_cache(feats,lengths)
        speaking_refresh=continuous_received and current.listening_state=='s'
        utterance_refresh=(current.listening_state=='l'
                           and activity.evidence.can_reply(state.refreshed_utterance))
        if not (speaking_refresh or utterance_refresh):return
        if continuous_received and len(receipts)>=90:
            raise RuntimeError('continuous acoustic diagnostic refresh budget exceeded')
        started=time.monotonic()
        windows=continuous_windows if continuous_received else state.chunks
        waveform=np.concatenate(windows)
        view=copy.copy(fw);view.window_second=len(waveform)/16000.
        values=view.stepaudio_audio_prepreocess(waveform)
        if len(values['feats'])!=1:raise ValueError('received context must be one encoder window')
        mel=values['feats'][0].t().unsqueeze(0).to(model.device,dtype=torch.bfloat16)
        lens=torch.tensor(values['feats_lengths'],device=model.device,dtype=torch.int32)
        encoded,out_lens=model.encoder(mel,lens);encoded=model.adapter(encoded)
        count=int(((out_lens-1)//2+1)[0]);expected=5*len(windows)
        if count!=expected or not torch.isfinite(encoded[:,:count]).all():raise ValueError('received audio context feature alignment')
        end=state.end;start=end-len(windows)
        if end!=len(current._audio_embed_seq_cache):raise ValueError('native audio context window cursor')
        if any(x!=5 for x in current._audio_embed_lens_cache[start:end]):raise ValueError('native audio patch counts')
        histories={name:getattr(current,name).clone() for name in
            ('generated_input_ids','generated_stoken_ids','generated_control_ids','prefix_input_ids')}
        current._audio_embed_seq_cache[start:end]=list(encoded[0,:count].detach().split(5))
        current._rebuild_packed_audio_embed_buffer()
        for name in ('past_key_values','stoken_past_key_values','control_past_key_values','merge_past_key_values'):
            setattr(current,name,None)
        fw.model._merge_cache_by_past_id.clear()
        if any(not torch.equal(value,getattr(current,name)) for name,value in histories.items()):
            raise ValueError('audio refresh changed token history')
        receipts.append(dict(utterance=activity.evidence.utterance,start_window=start,end_window=end,
            received_samples=end*window,reencoded_samples=len(waveform),feature_count=count,
            encoder_ms=(time.monotonic()-started)*1000,token_history_unchanged=True,
            future_audio=False,transcript_input=False,phase=current.listening_state))
        # Freeze earlier utterances. Later refreshes may only touch newly received audio.
        state.refreshed(activity.evidence.utterance)

    session._split_incremental_windows=types.MethodType(split,session)
    session._cache_precomputed_audio=types.MethodType(cache,session)
    session._cineduplex_audio_context_installed=True
    session._cineduplex_audio_context_receipts=receipts
    return {'policy':('received-continuous-audio-diagnostic-v1' if continuous_received
                      else 'received-audio-utterance-refresh-v2'),'max_context_windows':max_windows,
            'idle_preroll_windows':2,
            'precision':'bf16','quantization':'none','transcript_input':False,
            'continuous_received':continuous_received}
