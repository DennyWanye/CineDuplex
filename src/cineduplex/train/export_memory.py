"""Run the unchanged source loader on CPU with WAV/Arrow bytes fetched from NAS."""
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
    import pyarrow as pa
    from transformers import AutoTokenizer
    store=Store(config['url'],config['token']);torch.set_num_threads(2)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    data=config.get('data_input')
    base='output/pilot10-loader-input'
    receipt_path=data['receipt_path'] if data else base+'/loader-all-scenes-diagnostic.json'
    hf_path=data['hf_path'] if data else base+'/hf_all_scenes_diagnostic'
    receipt=json.loads(store.get(receipt_path,data['receipt_sha256'] if data else None))
    configuration_receipt=receipt
    if data and data.get('configuration_receipt_path'):
        configuration_receipt=json.loads(store.get(data['configuration_receipt_path'],data['configuration_receipt_sha256']))
    state=json.loads(store.get(hf_path+'/state.json',data['hf_files']['state.json'] if data else None))
    rows=[]
    for item in state['_data_files']:
        rows.extend(pa.ipc.open_stream(io.BytesIO(store.get(hf_path+'/'+item['filename'],data['hf_files'][item['filename']] if data else None))).read_all().to_pylist())
    expected_rows=data['expected_rows'] if data else 14
    expected_scenes=data['expected_scenes'] if data else [f'S{i:02}' for i in range(1,11)]
    if expected_rows<1 or len(rows)!=expected_rows or {r['scene_id'] for r in rows}!=set(expected_scenes):raise ValueError('inventory changed')
    scenes_path=data['scenes_path'] if data else 'output/pilot10-current/scenes.jsonl'
    scenes=[json.loads(line) for line in store.get(scenes_path,data['scenes_sha256'] if data else None).decode().splitlines()]
    hashes={}
    def walk(obj):
        if isinstance(obj,dict):
            if 'path' in obj and 'sha256' in obj:hashes[obj['path']]=obj['sha256']
            for value in obj.values():walk(value)
        elif isinstance(obj,list):
            for value in obj:walk(value)
    walk(scenes)
    def audio(path,sr):
        if not path.startswith('/Volumes/media/CineDuplex/'):raise ValueError('unapproved source path')
        relative=path.removeprefix('/Volumes/media/CineDuplex/')
        if relative not in hashes:raise ValueError('audio source lacks hash: '+relative)
        blob=store.get(relative,hashes[relative])
        with wave.open(io.BytesIO(blob)) as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(sr,1,2):raise ValueError('WAV format changed')
            pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768.
        return pcm,sr
    source=store.get('loader-source',receipt['loader_source_sha256']).decode()
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in ('peft','training_utils','datasets_utils'))]
    mod=types.ModuleType('cineduplex_memory_dataset');sys.modules[mod.__name__]=mod
    mod.rank0_print=lambda *a,**k:None
    exec(compile(tree,'nas-memory://LycheeFDDataset.py','exec'),mod.__dict__)
    # Narrow I/O adapters only; all dataset/collator methods are unmodified.
    mod.datasets=types.SimpleNamespace(load_from_disk=lambda _:rows)
    mod.librosa=types.SimpleNamespace(load=audio,filters=mod.librosa.filters)
    tokenizer=AutoTokenizer.from_pretrained(config['existing_tokenizer'],local_files_only=True,
        trust_remote_code=False,use_fast=False,padding_side='right')
    configuration=dict(configuration_receipt['configuration'])
    configuration['data_path']=hf_path
    dataset=mod.LazySupervisedDataset(tokenizer,types.SimpleNamespace(**configuration))
    collate=mod.DataCollatorForSupervisedDataset(tokenizer);batches=[]
    for i,row in enumerate(rows):
        batch={k:v for k,v in collate([dataset[i]]).items() if torch.is_tensor(v)}
        if not all(torch.isfinite(v).all() for v in batch.values()):raise ValueError('nonfinite batch')
        target=row.get('source_target_id')
        turn=next((t for s in scenes for t in s['turns'] if t['utterance_id']==target),{})
        batches.append({'scene_id':row['scene_id'],'source_row':i,'batch':batch,
            'target_utterance_id':target,'original_emotion':turn.get('original_emotion',turn.get('emotion'))})
        print('CPU export',i,row['scene_id'],list(batch['input_ids'].shape),flush=True)
    payload={'format':'cineduplex-pilot-batches-v1','rows':batches,
        'quality_approved':False,'loader_source_sha256':receipt['loader_source_sha256']}
    buf=io.BytesIO();torch.save(payload,buf);batch_bytes=buf.getvalue()
    saved=store.put(config['run_prefix']+'/batches.pt',batch_bytes)
    saved.update(rows=len(rows),scenes=sorted({r['scene_id'] for r in rows}),quality_approved=False,
        data_input=data,
        token_ids={k:tokenizer.encode(t,add_special_tokens=False)[0] for k,t in
        {'tts_start':'<tts_start>','tts_end':'<tts_end>','tts_pad':'<tts_pad>','eot':'<|EOT|>'}.items()})
    store.json(config['run_prefix']+'/batches.json',saved)
    print(json.dumps(saved),flush=True)
