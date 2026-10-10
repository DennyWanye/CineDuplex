"""Prepare real vendor Dataset features once from frozen NAS JSONL inputs."""
import ast
import hashlib
import io
import json
import random
import sys
import types
import wave
from cineduplex.train.memory_runtime import Store


def run(config):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    store = Store(config['url'], '')
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    source = store.get('loader-source', config['loader_sha256']).decode()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module in
                  ('peft','training_utils','datasets_utils'))]
    mod = types.ModuleType('cineduplex_bulk_dataset'); sys.modules[mod.__name__] = mod
    mod.rank0_print = lambda *a, **k: None
    exec(compile(tree, 'nas-memory://LycheeFDDataset.py', 'exec'), mod.__dict__)
    def audio(path, sr):
        prefix = '/Volumes/media/CineDuplex/'
        if not path.startswith(prefix): raise ValueError('audio outside NAS')
        blob = store.get(path[len(prefix):])
        with wave.open(io.BytesIO(blob)) as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth()) != (sr,1,2):
                raise ValueError('expected mono PCM16 WAV')
            return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768., sr
    mod.librosa = types.SimpleNamespace(load=audio, filters=mod.librosa.filters)
    tokenizer = AutoTokenizer.from_pretrained(config['existing_tokenizer'],local_files_only=True,
                                             trust_remote_code=False,use_fast=False,padding_side='right')
    published = json.loads(store.get('raw/lychee-tokenizer/config.json'))
    names = ['start_speaking_token_id','keep_listening_token_id','start_listening_token_id',
             'keep_speaking_token_id','detect_token_id','sleep_token_id','start_bc_token_id',
             'text_pad_token_id','stoken_pad_token_id','stoken_delay_token_id','stoken_delay_num',
             'audio_pad_token_id','control_token_chunk_size','adding_text_hiddenstates','no_stoken_label']
    args = {k:published[k] for k in names}
    for key, token in dict(audio_token_id='<audio_patch>',tts_start_id='<tts_start>',
                            tts_pad_id='<tts_pad>',tts_end_id='<tts_end>',eot_id='<|EOT|>').items():
        ids=tokenizer.encode(token,add_special_tokens=False)
        if len(ids)!=1: raise ValueError('non-atomic control token')
        args[key]=ids[0]
    args.update(data_path='frozen-input',window_second=24,align_audio_input=True,max_data_length=10000,
                enable_user_bc=False,enable_ai_bc=True,ai_bc_lead_silence_sec=0.,ai_bc_min_gap_sec=5.,ai_bc_max_num=1)
    files = config['input_files']; descriptors=[]; input_count=0
    for split in ('train','eval'):
        rows=[json.loads(line) for line in store.get(files[split]['path'],files[split]['sha256']).decode().splitlines()]
        mod.datasets=types.SimpleNamespace(load_from_disk=lambda _:rows)
        dataset=mod.LazySupervisedDataset(tokenizer,types.SimpleNamespace(**args))
        collate=mod.DataCollatorForSupervisedDataset(tokenizer)
        split_rows=[]; chunk=[]
        def save_chunk(number):
            stream=io.BytesIO(); torch.save({'rows':chunk},stream)
            key=config['run_prefix']+f'/features/{split}-{number:04}.pt'
            receipt=store.put(key,stream.getvalue())
            for j,r in enumerate(chunk):
                split_rows.append({k:r[k] for k in ('scene_id','source_row','target_utterance_id','sequence_length')}
                                  | {'feature_path':key,'feature_sha256':receipt['sha256'],'feature_index':j})
        for i,row in enumerate(rows):
            item=dataset[i]
            active=item['stoken_label_ids']; active=active[active>=151696]
            if not len(active) or not (active<151696+6561).all(): raise ValueError('missing/invalid speech labels')
            batch={k:v for k,v in collate([item]).items() if torch.is_tensor(v)}
            if not all(torch.isfinite(v).all() for v in batch.values()): raise ValueError('nonfinite feature')
            chunk.append({'scene_id':row['scene_id'],'source_row':i,'target_utterance_id':row['source_target_id'],
                          'sequence_length':batch['input_ids'].shape[1],'batch':batch})
            if len(chunk)==128 or i==len(rows)-1:
                save_chunk(len(split_rows)//128); chunk=[]
                print('CPU features',split,len(split_rows),'/',len(rows),flush=True)
        store.json(config['run_prefix']+f'/{split}-features.json',{'rows':split_rows,'configuration':args})
        if split=='train': descriptors=split_rows
        input_count+=len(rows)
    index={'format':'cineduplex-sharded-features-v1','rows':descriptors,'quality_approved':False}
    receipt=store.json(config['run_prefix']+'/batches-index.json',index)
    receipt.update(rows=len(descriptors),scenes=[r['scene_id'] for r in descriptors],input_count=input_count,
                   input_files=files,quality_approved=False,sharded=True,
                   token_ids={k:tokenizer.encode(t,add_special_tokens=False)[0] for k,t in
                    dict(tts_start='<tts_start>',tts_end='<tts_end>',tts_pad='<tts_pad>',eot='<|EOT|>').items()})
    store.json(config['run_prefix']+'/batches.json',receipt)
    print('CPU PREPARATION COMPLETE',len(descriptors),flush=True)
