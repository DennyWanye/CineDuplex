"""Run the real feature exporter, Dataset and collator on a pinned tiny batch.

Private admitted diagnostic only. No optimizer, model generation or devices.
All newly exported features stay under this call's immutable NAS directory.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch


def run(call, spec):
    from cineduplex.train.memory_runtime import Store
    from cineduplex.train.four_worker import FeatureReader
    root = Path(call.app.root).parents[1]
    store = Store('file://' + str(root), '')
    prefix = str(call.folder.relative_to(root)) + '/small-batch'
    out = dict(call=call.id, schema=1, microphone=False, speaker=False,
               training=False, generated_audio=False, quality_accepted=False,
               rows=[], errors=[])
    begun = time.monotonic()
    threads = torch.get_num_threads()
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()

    def pinned(key):
        item=spec[key]
        return store.get(item['path'], item['sha256'])

    def budget():
        if call.closed.is_set() or time.monotonic()-begun > 120:
            raise RuntimeError('small batch time/cancel budget')

    def diff(a,b):
        a,b=a.detach().float().cpu(),b.detach().float().cpu()
        if a.shape!=b.shape:raise ValueError('feature shape mismatch')
        d=a-b
        return dict(shape=list(a.shape),max_abs=float(d.abs().max()),
                    relative_l2=float(d.norm()/b.norm().clamp_min(1e-9)))

    try:
        contract_mod=types.ModuleType('private_native_contract')
        exec(compile(pinned('audio_contract'),'<pinned-audio-contract>','exec'),contract_mod.__dict__)
        # Only replace imports with the same functions injected explicitly;
        # the exporter body, original Dataset and collator run unchanged.
        tree=ast.parse(pinned('exporter'))
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and
                   n.module in ('cineduplex.train.memory_runtime','cineduplex.train.native_audio_contract'))]
        exporter=types.ModuleType('private_small_export')
        exporter.Store=Store;exporter.native_audio_contract=contract_mod.native_audio_contract
        exec(compile(tree,'<pinned-exporter>','exec'),exporter.__dict__)
        inputs=json.loads(pinned('small_inputs'))
        receipts={}
        for split,rows in inputs.items():
            if not 1<=len(rows)<=2:raise ValueError('bounded selection required')
            receipts[split]=store.put(prefix+'/'+split+'.jsonl',
                ('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode())
        cfg=dict(url='file://'+str(root),seed=20261014,
                 loader_sha256=spec['dataset']['sha256'],input_files=receipts,
                 existing_tokenizer=str(root/'raw/lychee-tokenizer'),run_prefix=prefix)
        budget();exporter.run(cfg);budget()
        index=json.loads(store.get(prefix+'/batches-index.json'))
        published=json.loads(pinned('config'))
        out['contract']=contract_mod.require_native_audio_contract(index,published)
        reader=FeatureReader(store)
        mod=sys.modules['cineduplex_bulk_dataset']
        tokenizer=call.app.fw.tokenizer
        for split,rows in inputs.items():
            metadata=json.loads(store.get(prefix+'/'+split+'-features.json'))
            mod.datasets=types.SimpleNamespace(load_from_disk=lambda _,rows=rows:rows)
            dataset=mod.LazySupervisedDataset(tokenizer,types.SimpleNamespace(**metadata['configuration']))
            for i,row in enumerate(rows):
                budget()
                item=dataset[i];batch=reader.batch(metadata['rows'][i])
                fresh=mod.DataCollatorForSupervisedDataset(tokenizer)([item])
                # Control labels use a random mask; validate their allowed
                # values separately rather than claiming bitwise reproduction.
                equal={k:bool(torch.equal(v,fresh[k])) for k,v in batch.items() if k!='control_label_ids'}
                if not all(equal.values()):raise ValueError('export/readback/collator mismatch')
                audio=item['user_wav'];ids=[];mels=[];lens=[]
                for start in range(0,len(audio),6400):
                    rt=call.app.fw.stepaudio_audio_prepreocess(audio[start:start+6400])
                    ids+=rt['input_ids'];mels+=rt['feats'];lens+=rt['feats_lengths']
                actual_ids=item['audio_input_ids'].tolist()
                # Vendor permits dropping a single final pad, never a patch.
                if len(ids)==len(actual_ids)+1 and ids[-1]==dataset.audio_pad_token_id:ids=ids[:-1]
                if ids!=actual_ids or lens!=item['wav_lens'].tolist():raise ValueError('runtime audio positions differ')
                mel_diff=[diff(a,b) for a,b in zip(item['wavs'],mels,strict=True)]
                with torch.inference_mode():
                    enc,n=call.app.model.encoder(batch['wavs'].to(call.app.model.device).bfloat16(),
                                                batch['wav_lens'].to(call.app.model.device))
                    features=call.app.model.adapter(enc)
                    lengths=((n-1)//2+1).tolist()
                patches=int((item['audio_input_ids']==dataset.audio_token_id).sum())
                if sum(lengths)!=patches:raise ValueError('actual encoder features/patch count mismatch')
                labels=batch['stoken_label_ids'];speech=labels[labels>=151696]
                source_codes=row['turns'][0]['assistant']['stoken']
                if not torch.equal(speech.cpu()-151696,torch.tensor(source_codes)):raise ValueError('speech target code identity')
                mapping=batch['stoken_mapping'];valid=mapping[mapping>=0]
                if len(valid) and int(valid.max())>=batch['input_ids'].shape[1]:raise ValueError('text/speech mapping outside sequence')
                if not all(torch.isfinite(v).all() for v in batch.values()):raise ValueError('nonfinite collated tensor')
                out['rows'].append(dict(split=split,scene_id=row['scene_id'],
                    feature=metadata['rows'][i],sequence_length=batch['input_ids'].shape[1],
                    exported_readback_equal=equal,control_mask='stochastic; not compared bitwise',
                    encoder_feature_count=sum(lengths),audio_patch_count=patches,
                    input_waveform_samples=len(audio),windows=len(mels),
                    runtime_ids_equal=True,runtime_lengths_equal=True,
                    max_mel_abs=max(d['max_abs'] for d in mel_diff),
                    max_mel_relative_l2=max(d['relative_l2'] for d in mel_diff),
                    exact_source_speech_labels=len(speech),mapping_in_range=True))
        out['passed']=len(out['rows'])==sum(map(len,inputs.values()))
        out['feature_index']=dict(path=prefix+'/batches-index.json',
            sha256=hashlib.sha256(store.get(prefix+'/batches-index.json')).hexdigest())
    except Exception as exc:
        out['errors'].append(type(exc).__name__+': '+str(exc)[:500]);raise
    finally:
        torch.set_num_threads(threads);random.setstate(py_rng);np.random.set_state(np_rng)
        torch.set_rng_state(torch_rng);torch.cuda.set_rng_state(cuda_rng)
        out['elapsed_seconds']=time.monotonic()-begun
        store.json(str(call.folder.relative_to(root))+'/native-small-batch.json',out)
