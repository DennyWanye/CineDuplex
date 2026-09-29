"""Prepare and execute source-backed CPU loader diagnostics, never training.

The vendor dataset algorithms remain exact. Three unused/training-only imports
are removed in RAM, and the original rank0_print function is loaded on its own.
This is explicitly an import-adapted diagnostic, not the full training runtime.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
from cineduplex.contracts import digest, write_json
from .nas_source import nas_root
from .pilot import read_pcm, save_pcm

VENDOR = Path(__file__).resolve().parents[3]/'vendor/lychee-fd-source/Training_package/Code'


def prepare(root):
    root=nas_root(root)
    source=root/'output/pilot10-revision2'
    scene=json.loads((source/'S09/scene.json').read_text())
    ordered=sorted(scene['turns'],key=lambda t:t['start_sample'])
    a,b,c,d,e=ordered
    assert [t['speaker_id'] for t in ordered]==['A1','B1','B1','A1','B1']
    audit=json.loads((root/'output/source-audit/SmoothConv'/scene['source_scene']/'audit.json').read_text())
    channel=next(t for t in audit['audition_tracks'] if t['channel_index']==1)
    path=root/channel['path']
    if digest(path)!=channel['sha256']:raise ValueError('source track hash changed')
    rate,pcm=read_pcm(path)
    assert rate==16000 and pcm.shape[1]==1
    lo=round(b['source_start_seconds']*rate);hi=round(c['source_end_seconds']*rate)
    out=root/'output/pilot10-loader-input';(out/'S09').mkdir(parents=True,exist_ok=True)
    dest=out/'S09/B1-continuous-first-response.wav'
    save_pcm(dest,rate,pcm[lo:hi,0])
    _,check=read_pcm(dest)
    if not (check==pcm[lo:hi]).all():raise ValueError('continuous slice changed')
    turn=copy.deepcopy(b)
    turn.update(utterance_id='B1-continuous-first-response',
        source_end_seconds=c['source_end_seconds'],end_sample=c['end_sample'],
        text=b['text']+' '+c['text'],original_text=b['original_text']+' '+c['original_text'],
        audio=dict(path=str(dest.relative_to(root)),sha256=digest(dest)),
        component_utterance_ids=[b['utterance_id'],c['utterance_id']],
        grouping='Continuous original channel slice including the original intervening gap')
    scene['turns']=[turn]
    write_json(out/'S09/scene.json',scene)
    (out/'scenes.jsonl').write_text(json.dumps(scene,ensure_ascii=False)+'\n')
    write_json(out/'preparation.json',dict(training_ready=False,source_channel=channel,
        slice_start_sample=lo,slice_end_sample=hi,actual_source_sample_equality=True,
        purpose='One continuous assistant span for diagnostic loader, not a new pilot scene'))
    print('Prepared continuous S09 assistant span',len(check)/rate,'seconds',flush=True)


def execute(root, include_expressive=False):
    root=nas_root(root)
    print('loader: importing CPU dependencies',flush=True)
    import numpy as np
    import torch
    from datasets import Dataset,load_from_disk
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    out=root/'output/pilot10-loader-input';stage=root/'output/pilot10-revision2'
    scenes={sid:json.loads((stage/sid/'scene.json').read_text()) for sid in ('S09','S10')}
    def codes(scene,turn,folder=stage):
        from .codec_cpu import MODEL_SHA
        saved=json.loads((folder/scene['scene_id']/(turn['utterance_id']+'.tokens.json')).read_text())
        raw=saved['raw_codes']
        assert saved['input_sha256']==turn['audio']['sha256'] and saved['model_sha256']==MODEL_SHA
        assert saved['offset_applied']==0 and raw and all(type(x)is int and 0<=x<6561 for x in raw)
        assert digest(root/turn['audio']['path'])==turn['audio']['sha256']
        return raw
    def timing(turn):
        return dict(start_time=turn['start_sample']/16000,end_time=turn['end_sample']/16000)
    def assistant(scene,turn,folder=stage):
        return dict(timing(turn),text=turn['text'],stoken=codes(scene,turn,folder))
    def pair(user,ai,interrupted=False,bc=None):
        return dict(user=timing(user),assistant=ai,is_ai_been_interrupted=interrupted,
                    ai_backchannel=bc or [],user_backchannel=[])
    s9=scenes['S09'];a,b,c,d,e=sorted(s9['turns'],key=lambda t:t['start_sample'])
    grouped=json.loads((out/'S09/scene.json').read_text())['turns'][0]
    s10=scenes['S10'];u,bc,reply=sorted(s10['turns'],key=lambda t:t['start_sample'])
    rows=[]
    for s,user_channel,turns in [
        (s9,0,[pair(a,assistant(s9,grouped,out),True),pair(d,assistant(s9,e))]),
        (s10,1,[pair(u,assistant(s10,reply),bc=[assistant(s10,bc)])])]:
        track=next(x for x in s['source_channels'] if x['channel_index']==user_channel)
        assert digest(root/track['path'])==track['sha256']
        rows.append(dict(scene_id=s['scene_id'],user_audio_path=str(root/track['path']),
            turns=turns,training_ready=False,scope='unreviewed real-source CPU loader diagnostic'))
    # Expressive sources are known mixture excerpts. Feed the actual observed
    # user utterance directly for a causal pair diagnostic; never synthesize or
    # advertise a continuous independent user track from missing source gaps.
    if include_expressive:
        for scene in [json.loads(x) for x in (root/'output/pilot10-current/scenes.jsonl').read_text().splitlines()][:8]:
            scenes[scene['scene_id']]=scene
            turns=sorted(scene['turns'],key=lambda t:t['source_start_seconds'])
            for index,target in enumerate(turns[1:],1):
                earlier=[t for t in turns[:index] if t['speaker_id']!=target['speaker_id']
                    and t['source_end_seconds']<=target['source_start_seconds']]
                if not earlier:raise ValueError('no observed prior opposite-speaker input')
                user=earlier[-1];path=root/user['audio']['path']
                assert digest(path)==user['audio']['sha256']
                rate,pcm=read_pcm(path);assert rate==16000 and pcm.shape[1]==1
                ai=assistant(scene,target,root/'output/pilot10')
                ai['start_time']=target['source_start_seconds']-user['source_start_seconds']
                ai['end_time']=target['source_end_seconds']-user['source_start_seconds']
                conv=dict(user=dict(start_time=0.,end_time=len(pcm)/rate),assistant=ai,
                    is_ai_been_interrupted=False,ai_backchannel=[],user_backchannel=[])
                rows.append(dict(scene_id=scene['scene_id'],user_audio_path=str(path),turns=[conv],
                    training_ready=False,scope='causal utterance-pair diagnostic on actual source mixture excerpt; no independent track or full scene continuity claimed',
                    source_user_id=user['utterance_id'],source_target_id=target['utterance_id']))
    for row in rows:
        row.setdefault('source_user_id',None);row.setdefault('source_target_id',None)
    dataset_path=out/('hf_all_scenes_diagnostic' if include_expressive else 'hf_diagnostic_dataset')
    Dataset.from_list(rows).save_to_disk(str(dataset_path))
    assert list(load_from_disk(str(dataset_path)))==rows
    print('loader:',len(rows),'diagnostic HF rows exactly reloaded',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(str(root/'raw/lychee-tokenizer'),
        local_files_only=True,trust_remote_code=False,use_fast=False,padding_side='right')
    config=json.loads((root/'raw/lychee-tokenizer/config.json').read_text())
    names=['start_speaking_token_id','keep_listening_token_id','start_listening_token_id',
        'keep_speaking_token_id','detect_token_id','sleep_token_id','start_bc_token_id',
        'text_pad_token_id','stoken_pad_token_id','stoken_delay_token_id','stoken_delay_num',
        'audio_pad_token_id','control_token_chunk_size','adding_text_hiddenstates','no_stoken_label']
    args={k:config[k] for k in names}
    for key,token in dict(start_speaking_token_id='<|S-S|>',
        start_listening_token_id='<|S-L|>',keep_listening_token_id='<|K-L|>',
        keep_speaking_token_id='<|K-S|>',start_bc_token_id='<|BackChannel|>').items():
        assert tokenizer.encode(token,add_special_tokens=False)==[args[key]]
    for key,token in dict(audio_token_id='<audio_patch>',tts_start_id='<tts_start>',
        tts_pad_id='<tts_pad>',tts_end_id='<tts_end>',eot_id='<|EOT|>').items():
        ids=tokenizer.encode(token,add_special_tokens=False)
        if len(ids)!=1:raise ValueError('control token is not atomic: '+token)
        args[key]=ids[0]
    args.update(data_path=str(dataset_path),window_second=24,align_audio_input=True,
        max_data_length=10000,enable_user_bc=False,enable_ai_bc=True,
        ai_bc_lead_silence_sec=0.,ai_bc_min_gap_sec=5.,ai_bc_max_num=1)
    # Import adaptation only: preserve every function/class AST exactly.
    file=VENDOR/'DataLoaders/LycheeFDDataset.py';original=file.read_text();tree=ast.parse(original)
    removed=[];kept=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module in ('peft','training_utils','datasets_utils'):
            removed.append(ast.get_source_segment(original,node));continue
        kept.append(node)
    assert len(removed)==3
    original_defs=[ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    assert original_defs==[ast.dump(n) for n in kept if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    tree.body=kept
    module=ModuleType('cineduplex_source_loader_diagnostic')
    sys.modules[module.__name__]=module
    namespace=module.__dict__;namespace['torch']=torch
    logging_tree=ast.parse((VENDOR/'training_utils.py').read_text())
    logging_def=next(n for n in logging_tree.body if isinstance(n,ast.FunctionDef) and n.name=='rank0_print')
    exec(compile(ast.Module(body=[logging_def],type_ignores=[]),str(VENDOR/'training_utils.py'),'exec'),namespace)
    exec(compile(tree,str(file),'exec'),namespace)
    # Dataclasses inspect sys.modules for the module defining a class.
    dataset=namespace['LazySupervisedDataset'](tokenizer,SimpleNamespace(**args))
    interruption_events=[]
    original_append=dataset._append_ai_response_turn
    def observe_append(acc,conv,conv_id,conversation,ai_info,user_speech_wave,state):
        original_append(acc,conv,conv_id,conversation,ai_info,user_speech_wave,state)
        if conv['is_ai_been_interrupted']:
            remainder=state['last_ai_stoken']
            assert remainder is not None and len(remainder)>0
            interruption_events.append(dict(turn_index=conv_id,
                remaining_stoken_steps=len(remainder),
                source_interruption_offset_seconds=conversation[conv_id+1]['user']['start_time']-ai_info['start_time']))
    dataset._append_ai_response_turn=observe_append
    results=[];loaded_items=[]
    for i,row in enumerate(rows):
        item=dataset[i]
        n=len(item['input_ids'])
        prefix_len=len(item['prefix_input_ids'])
        assert all(len(item[k])==n for k in ['labels','stoken_mapping','stoken_label_ids','control_label_ids','attention_mask'])
        # The official model separately prepends zero embeddings to these
        # three streams; the dataset intentionally leaves out system tokens.
        assert all(len(item[k])==n-prefix_len for k in ['stoken_ids','control_input_ids','audio_input_ids'])
        assert all((item[k][:prefix_len]==-100).all() for k in ['labels','stoken_label_ids','control_label_ids'])
        assert all(torch.isfinite(x).all() for x in item['wavs'])
        active=item['stoken_label_ids'][item['stoken_label_ids']>=151696]
        assert len(active)>0 and (active<151696+6561).all()
        bc_count=int((item['control_label_ids']==args['start_bc_token_id']).sum())
        if row['scene_id']=='S09':assert len(interruption_events)==1
        if row['scene_id']=='S10':
            assert bc_count>0,'real backchannel lost in loader'
            expected_bc=[v+151696 for v in row['turns'][0]['ai_backchannel'][0]['stoken']]
            labels=item['stoken_label_ids'].tolist()
            assert any(labels[j:j+len(expected_bc)]==expected_bc for j in range(len(labels)-len(expected_bc)+1)), 'BC speech labels lost'
        results.append(dict(scene_id=row['scene_id'],sequence_tokens=n,
            prefix_tokens=prefix_len,suffix_tokens=n-prefix_len,
            active_raw_codec_labels=len(active),backchannel_control_labels=bc_count,
            emitted_user_audio_seconds=len(item['user_wav'])/16000,
            source_scene_seconds=scenes[row['scene_id']]['duration_samples']/16000))
        print('loader: actual source row passed',row['scene_id'],n,'steps',flush=True)
        loaded_items.append(item)
    batch=namespace['DataCollatorForSupervisedDataset'](tokenizer)(loaded_items)
    assert batch['input_ids'].shape[0]==len(rows) and batch['stoken_ids'].shape[0]==len(rows)
    assert batch['input_ids'].shape[1]-batch['stoken_ids'].shape[1]==batch['prefix_input_ids'].shape[1]
    assert torch.isfinite(batch['wavs']).all()
    write_json(out/('loader-all-scenes-diagnostic.json' if include_expressive else 'loader-diagnostic.json'),dict(status='PASSED',results=results,
        scenes_exercised=sorted(set(row['scene_id'] for row in rows)),
        expressive_scope='causal utterance pairs only, not complete independent duplex scenes' if include_expressive else None,
        collator_actual_batch_size=len(rows),batch_shapes={k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)},
        interruption_events=interruption_events,
        loader_source_sha256=digest(file),
        exact_function_and_class_ast=True,removed_imports=removed,
        original_rank0_print_function=True,configuration=args,
        runtime='CPU import-adapted exact vendor dataset algorithms; no model or training',
        waveform_quality_review='NOT_PERFORMED',time_preservation='NOT_VERIFIED',training_ready=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('action',choices=['prepare','execute','encode-and-execute'])
    parser.add_argument('--include-expressive',action='store_true');args=parser.parse_args()
    try:
        if args.action=='encode-and-execute':
            from .codec_cpu import encode
            encode(args.root,output_rel='output/pilot10-loader-input')
        if args.action=='prepare':prepare(args.root)
        else:execute(args.root,include_expressive=args.include_expressive)
    except Exception as exc:
        if args.action!='prepare':
            root=nas_root(args.root)
            filename='loader-all-scenes-diagnostic.json' if args.include_expressive else 'loader-diagnostic.json'
            write_json(root/'output/pilot10-loader-input'/filename,
                dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),
                     training_ready=False,scope='CPU import-adapted source loader diagnostic'))
        raise
