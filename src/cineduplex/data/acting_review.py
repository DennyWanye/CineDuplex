"""Build a source-verified acting audition packet without approving any sample."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import wave

from cineduplex.contracts import digest, safe_child, write_json
from .nas_source import nas_root

PRIORITY = ['G00002_10_08_002', 'G00002_13_08_006',
            'G00002_15_07_005', 'G00002_05_07_002']


def verified_audio(root, asset, sample_rate):
    path = safe_child(root, asset['path'])
    if digest(path) != asset['sha256']:
        raise ValueError('audio identity mismatch: ' + asset['path'])
    with wave.open(str(path)) as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, sample_rate):
            raise ValueError('unexpected audition WAV format')
        return f.getnframes() / sample_rate


def build(root, output_rel='output/acting-review-v1',
          control_rel='output/acting-neutral-control-v1'):
    root = nas_root(root)
    out = safe_child(root, output_rel)
    # Review decisions stay separate; rerendering never touches accepted feedback.
    if (out/'review-decisions.json').exists():
        raise ValueError('review decisions exist; preserve this packet and choose a new output')
    current = root/'output/pilot10-current'
    manifest = current/'scenes.jsonl'
    scenes = [json.loads(s) for s in manifest.read_text().splitlines()]
    by_id = {t['utterance_id']: (s,t) for s in scenes for t in s['turns']}
    rows = []
    for s in scenes:
        if s['scene_id'] not in {f'S{i:02}' for i in range(1,9)}:
            continue
        for t in s['turns']:
            original = t['audio']
            seconds = verified_audio(root, original, 16000)
            p = safe_child(root, original['path'])
            rec = json.loads(p.with_name(p.stem+'.reconstructed.json').read_text())
            verified_audio(root, rec, 24000)
            token = json.loads(p.with_name(p.stem+'.tokens.json').read_text())
            if token['input_sha256'] != original['sha256']:
                raise ValueError('token input differs from source')
            control = None
            cp = safe_child(root, control_rel+'/'+s['scene_id']+'/'+t['utterance_id']+'.reconstructed.json')
            if cp.exists():
                control = json.loads(cp.read_text())
                if (control['target_audio'] != original or control['same_utterance_prompt']
                        or control['prompt_policy'] != 'independent-neutral'
                        or not control['prompt_text_differs']):
                    raise ValueError('invalid independent-reference receipt')
                import hashlib
                codes_sha = hashlib.sha256(json.dumps(token['raw_codes'],separators=(',',':')).encode()).hexdigest()
                if control['target_codes_sha256'] != codes_sha:
                    raise ValueError('control used different target codes')
                ps, pt = by_id[control['prompt_utterance_id']]
                if (pt['speaker_id'] != t['speaker_id'] or ps['split_group'] != s['split_group']
                        or ps['scene_id'] == s['scene_id'] or ps['coverage_requested'] == 'suppression'
                        or pt.get('original_emotion',pt['emotion']) != 'neutral'
                        or pt['text'].strip() == t['text'].strip() or pt['audio'] != control['prompt_audio']):
                    raise ValueError('reference annotation/identity mismatch')
                verified_audio(root, control, 24000)
                verified_audio(root, control['prompt_audio'], 16000)
            rows.append(dict(scene_id=s['scene_id'], utterance_id=t['utterance_id'],
                coverage_requested=s['coverage_requested'], original_emotion=t.get('original_emotion',t['emotion']),
                speaker_id=t['speaker_id'], text=t['text'], source_captions=t.get('source_captions'),
                seconds=seconds, source=original, reconstruction=rec, neutral_control=control,
                overlaps=t.get('overlaps',[]), source_scene=s['source_scene'],split_group=s['split_group'],
                source_revision=s['source_revision'], rights_ref=s['rights_ref'],
                source_start_seconds=t['source_start_seconds'],source_end_seconds=t['source_end_seconds'],
                role='expressive_audition_only',auditory_review='NOT_PERFORMED',training_ready=False))
    if len(rows) != 20 or len({r['utterance_id'] for r in rows}) != 20:
        raise ValueError('expected the fixed twenty expressive utterances')
    rows.sort(key=lambda r: (PRIORITY.index(r['utterance_id']) if r['utterance_id'] in PRIORITY else 99,
                            r['scene_id'],r['utterance_id']))
    out.mkdir(parents=True,exist_ok=True)
    cards = []
    def player(asset):
        relative = quote(os.path.relpath(safe_child(root,asset['path']),out),safe='/')
        return '<audio controls preload="none" src="'+relative+'"></audio>'
    for i,r in enumerate(rows):
        control = r['neutral_control']
        c = ('<h4>C · 另一句 neutral 标注参考下的重建</h4>'+player(control)
             +'<details><summary>听参考音及查看参考条件</summary>'
             +player(control['prompt_audio'])+'<p>参考原标签 neutral，尚未听审；改变的是整套参考条件，'
             '包含参考音频特征、token 和说话人嵌入，不能将差异全部归因于某一因素。</p></details>') if control else '<p>C · 未制作独立参考对照</p>'
        same = r['reconstruction']['same_utterance_prompt']
        labels = html.escape(json.dumps(r['source_captions'],ensure_ascii=False))
        cards.append('<section><h2>'+str(i+1)+' · '+html.escape(r['scene_id']+' / '+r['utterance_id'])
            +f'</h2><p>{r["seconds"]:.2f} 秒 · 先听原音，再比较两种重建</p>'
            +'<h4>A · 原音</h4>'+player(r['source'])+'<h4>B · 原有重建</h4>'+player(r['reconstruction'])
            +('<p>原句本身作参考：只能作为重建诊断。</p>' if same else '<p>另一话段作参考：不等于中性参考对照。</p>')
            +c+'<details><summary>听完再查看台词、原始标签与候选类别</summary><p>'
            +html.escape(r['text'])+'</p><p>原标签：'+html.escape(r['original_emotion'])
            +'；场景候选：'+html.escape(r['coverage_requested'])+'</p><p>'+labels+'</p></details>'
            +'<p>反馈：原音表达了什么？B/C 是否保留情绪和停顿？文字/声线是否变了？是否有串音或失真？不确定可直说。</p></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CineDuplex 表演资料试听</title><style>body{font:17px/1.65 system-ui;max-width:940px;margin:32px auto;padding:0 20px;background:#f5f3ef;color:#222}section{background:white;padding:24px;margin:22px 0;border-radius:14px}audio{width:100%}summary{cursor:pointer}h4{margin-bottom:6px}</style>
<h1>先确认表演能否保留下来</h1><p>第一阶段优先表演感。前4项为悲伤、愤怒、恐惧、压抑候选的重点对照，其余保留作完整审核。S08原标签仍是neutral，不能先入为主认定压抑。</p>
<p>这是有顺序提示的试听，不是正式盲测，也没有模型微调结果。B/C均使用固定目标token；C更换同说话人的另一句neutral标注参考。听感仍需真实反馈，所有样本目前未批准训练。</p>
<p>请先听前4项，再在当前聊天按“编号：原音情绪；B变化；C变化；问题”反馈。页面不会上传音频、保存浏览器录音或自动写入批准结果。</p>'''+''.join(cards)+'</html>'
    page_path=out/'review.html';page_path.write_text(page)
    packet=dict(schema_version='cineduplex.acting_audition.v1',created_at=datetime.now(timezone.utc).isoformat(),
        priority='expressive_performance_first',source_manifest_sha256=digest(manifest),
        expressive_scenes=8,utterances=len(rows),seconds=sum(r['seconds'] for r in rows),
        original_emotion_counts=dict(Counter(r['original_emotion'] for r in rows)),
        same_utterance_reconstructions=sum(r['reconstruction']['same_utterance_prompt'] for r in rows),
        neutral_reference_controls=sum(r['neutral_control'] is not None for r in rows),
        interaction_candidates=['S09','S10'],interaction_acceptance='NOT_COMPLETED',
        training_ready=0,rows=rows,html_sha256=digest(page_path),
        limitations=['listening_not_performed','neutral_reference_labels_not_heard',
            'same_speaker_reference_bundle_changed_not_single_factor',
            'suppression_not_confirmed','no_model_training','no_blind_evaluation'])
    write_json(out/'packet.json',packet)
    if json.loads((out/'packet.json').read_text()) != packet or page_path.read_text() != page:
        raise ValueError('packet readback mismatch')
    print(json.dumps({k:v for k,v in packet.items() if k not in {'rows'}},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--output-relative',default='output/acting-review-v1')
    p.add_argument('--control-relative',default='output/acting-neutral-control-v1')
    a=p.parse_args();build(a.root,a.output_relative,a.control_relative)
