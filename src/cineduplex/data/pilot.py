"""Produce traceable review scenes without promoting mixed audio to duplex gold.

Commands: python -m cineduplex.data.pilot --root /Volumes/media/CineDuplex
The ten selections are fixed, small training-group windows, not a benchmark.
"""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
import re
import subprocess
import wave

import numpy as np

from cineduplex.contracts import digest, safe_child, write_json
from .nas_source import Source, SparseTar, nas_root

SELECTIONS = [
    ('S01', 'neutral', 'G00002_02', ['G00002_02_07_001', 'G00002_02_08_002']),
    ('S02', 'neutral', 'G00002_09', ['G00002_09_08_004', 'G00002_09_07_005']),
    ('S03', 'sadness', 'G00002_10', ['G00002_10_07_001', 'G00002_10_08_002']),
    ('S04', 'sadness', 'G00002_18', ['G00002_18_08_003', 'G00002_18_07_004', 'G00002_18_07_006']),
    ('S05', 'anger', 'G00002_04', ['G00002_04_08_002', 'G00002_04_07_003', 'G00002_04_08_004', 'G00002_04_07_005']),
    ('S06', 'anger', 'G00002_13', ['G00002_13_07_005', 'G00002_13_08_006']),
    ('S07', 'fear', 'G00002_15', ['G00002_15_08_004', 'G00002_15_07_005']),
    ('S08', 'suppression', 'G00002_05', ['G00002_05_07_002', 'G00002_05_08_003', 'G00002_05_07_004']),
]
EMOTIONS = {'sad': 'sadness', 'angry': 'anger', 'fearful': 'fear'}


def read_pcm(path):
    with wave.open(str(path), 'rb') as f:
        if f.getsampwidth() != 2 or f.getcomptype() != 'NONE':
            raise ValueError('16-bit PCM WAV required')
        rate = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').reshape(-1, f.getnchannels())
    return rate, data


def save_pcm(path, rate, data):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(np.asarray(data, dtype='<i2').tobytes())


def normalize(root, raw, output):
    nas_root(root)
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(raw),
                    '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(output)], check=True,
                   env={**__import__('os').environ, 'TMPDIR': str(root/'tmp')})
    return dict(path=str(output.relative_to(root)), sha256=digest(output))


def stripped(text):
    # Preserve original text separately; markup is not spoken text.
    return re.sub(r'\[[^\]]*\]|【[^】]*】', '', text)


def build(root, annotations='manifests/Emotiontalk-audio-annotations.json'):
    root = nas_root(root)
    rows = json.loads(safe_child(root, annotations).read_text())
    by_id = {Path(x['member']).stem: x for x in rows}
    metadata = json.loads((root/'manifests/Emotiontalk-repo.json').read_text())
    source = Source(root, 'BAAI/Emotiontalk', metadata['sha'])
    archive = SparseTar(source, 'Audio.tar', next(x['size'] for x in metadata['siblings'] if x['rfilename']=='Audio.tar'), root/'raw/Emotiontalk-Audio-prefix64.tar')
    out = root/'output/pilot10'; out.mkdir(parents=True, exist_ok=True)
    rights = dict(source_id='BAAI/Emotiontalk', revision=metadata['sha'], license_id='CC-BY-NC-SA-4.0',
        allowed_uses=['noncommercial_research'], commercial_use=False,
        source_url='https://huggingface.co/datasets/BAAI/Emotiontalk',
        license_url='https://creativecommons.org/licenses/by-nc-sa/4.0/',
        attribution='Sun et al., EmotionTalk: An Interactive Chinese Multimodal Emotion Dataset With Rich Annotations (2025)',
        original_card='raw/Emotiontalk-README.md', original_card_sha256=digest(root/'raw/Emotiontalk-README.md'))
    write_json(out/'rights.json',rights)
    scenes = []
    for sid,category,group,ids in SELECTIONS:
        if group.split('_')[0] in {'G00001','G00003','G00012','G00015'}:
            raise ValueError('official validation/test group cannot enter training pilot')
        selected = [by_id[i]['data'] for i in ids]
        start = min(d['paragraphs']['startTime'] for d in selected)
        end = max(d['paragraphs']['endTime'] for d in selected)
        # Include every annotated utterance that intersects the selected window.
        # Do not move turns, concatenate unrelated scenes, or invent speaker audio.
        selected = sorted([x['data'] for x in rows if x['data']['file_path'].split('/')[1]==group
            and x['data']['paragraphs']['startTime'] < end
            and x['data']['paragraphs']['endTime'] > start], key=lambda d:d['paragraphs']['startTime'])
        start = min(d['paragraphs']['startTime'] for d in selected)
        end = max(d['paragraphs']['endTime'] for d in selected)
        folder = out/sid; folder.mkdir(exist_ok=True)
        turns, native = [], {}
        for d in selected:
            uid = Path(d['file_path']).stem
            raw = archive.extract('Audio/wav/'+d['file_path'])
            rate, pcm = read_pcm(raw)
            expected = d['paragraphs']['endTime']-d['paragraphs']['startTime']
            error = len(pcm)/rate - expected
            if abs(error) > 2/rate:
                raise ValueError(f'original duration mismatch: {uid} {error}')
            native[uid] = (rate, pcm[:,0])
            normalized = normalize(root,raw,folder/(uid+'.wav'))
            turn = dict(utterance_id=uid, speaker_id=d['speaker_id'],
                source_start_seconds=d['paragraphs']['startTime'], source_end_seconds=d['paragraphs']['endTime'],
                start_sample=round((d['paragraphs']['startTime']-start)*16000),
                end_sample=round((d['paragraphs']['endTime']-start)*16000),
                text=stripped(d['content']), original_text=d['content'],
                emotion=EMOTIONS.get(d['emotion_result'],d['emotion_result']),
                original_emotion=d['emotion_result'], source_annotators=d['data'],
                source_captions=d.get('sourceAttr',{}),
                original_audio=dict(path=str(raw.relative_to(root)),sha256=digest(raw),
                    sample_rate=rate, channels=pcm.shape[1], duration_error_seconds=error,
                    dual_mono=bool(pcm.shape[1]==2 and np.array_equal(pcm[:,0],pcm[:,1]))),
                audio=normalized, label_provenance='source_annotation',
                auditory_review='NOT_PERFORMED', overlaps=[])
            turns.append(turn)
        overlap_evidence=[]
        for i,a in enumerate(turns):
            for b in turns[i+1:]:
                lo=max(a['source_start_seconds'],b['source_start_seconds'])
                hi=min(a['source_end_seconds'],b['source_end_seconds'])
                if hi<=lo or a['speaker_id']==b['speaker_id']:continue
                a['overlaps'].append(b['utterance_id']);b['overlaps'].append(a['utterance_id'])
                rate,apcm=native[a['utterance_id']];br,bpcm=native[b['utterance_id']]
                if rate!=br:raise ValueError('source rates differ')
                n=max(0,int((hi-lo)*rate)-2)
                aa=apcm[round((lo-a['source_start_seconds'])*rate):][:n]
                bb=bpcm[round((lo-b['source_start_seconds'])*rate):][:n]
                same=bool(n and len(aa)==len(bb) and np.array_equal(aa,bb))
                overlap_evidence.append(dict(a=a['utterance_id'],b=b['utterance_id'],duration_seconds=hi-lo,
                    samples_compared=n, identical_source_samples=same,
                    interpretation='SAME_MIXTURE_NOT_INDEPENDENT_TRACKS' if same else 'INDEPENDENT_TRACKS_NOT_ESTABLISHED'))
        # A review rendering is useful, but unknown inter-utterance gaps are explicit.
        # Prefer the first observed mixture in overlaps; do not sum duplicated audio.
        length=round((end-start)*16000);master=np.zeros(length,dtype='<i2');known=np.zeros(length,dtype=bool)
        for turn in turns:
            _,pcm=read_pcm(root/turn['audio']['path']);begin=turn['start_sample'];n=min(len(pcm),length-begin)
            mask=~known[begin:begin+n];master[begin:begin+n][mask]=pcm[:n,0][mask];known[begin:begin+n]=True
            turn['clean_target_candidate']=not turn['overlaps'] and 'over' not in turn['original_text'] and 'interrupted' not in turn['original_text']
        save_pcm(folder/'review_mix.wav',16000,master)
        gaps=[];edges=np.flatnonzero(np.diff(np.r_[True,known,True].astype(int)))
        for begin,finish in zip(edges[::2],edges[1::2]):gaps.append([int(begin),int(finish)])
        suppression=[dict(utterance_id=t['utterance_id'],field=k,text=v) for t in turns for k,v in t['source_captions'].items()
            if any(w in v for w in ('压抑','克制','隐忍','强忍','忍住'))]
        blockers=['auditory_review_not_performed','codec_roundtrip_not_verified','official_loader_not_verified']
        if overlap_evidence:blockers.append('independent_speaker_tracks_missing')
        if category=='suppression':blockers.append('suppression_caption_only_not_confirmed')
        if any('interrupted' in t['original_text'] and not t['overlaps'] for t in turns):
            blockers.append('interruption_tag_without_annotated_partner')
        scene=dict(schema_version='cineduplex.review_scene.v1',scene_id=sid,coverage_requested=category,
            source_scene=group,split='train',split_group=group.split('_')[0],
            source_revision=metadata['sha'],rights_ref='output/pilot10/rights.json',
            source_start_seconds=start,source_end_seconds=end,sample_rate=16000,duration_samples=length,
            turns=turns,overlap_evidence=overlap_evidence,suppression_evidence=suppression,
            review_audio=dict(path=str((folder/'review_mix.wav').relative_to(root)),sha256=digest(folder/'review_mix.wav'),
                provenance='source utterance mixture placed on original timeline; unobserved gaps zero-filled for listening only',
                unknown_gap_samples=gaps),
            annotation_status='SOURCE_LABELS_PRESERVED_REVIEW_PENDING',training_ready=False,blockers=blockers)
        write_json(folder/'scene.json',scene);scenes.append(scene)
        write_json(out/'progress.json',dict(completed_review_scenes=len(scenes),total=10,training_ready=0))
        print(f'{sid} {category}: {len(turns)} turns; {length/16000:.2f}s; training_ready=false',flush=True)
    # The conversational cases use actual paired source channels rather than
    # the EmotionTalk mixture cuts. Import lazily to share PCM utilities.
    from .smoothconv import build_candidates
    scenes.extend(build_candidates(root))
    write_json(out/'progress.json',dict(completed_review_scenes=len(scenes),total=10,training_ready=0))
    nas_root(root)
    (out/'scenes.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in scenes))
    write_json(out/'summary.json',dict(count=len(scenes),training_ready=0,
        source_speaker_ids=sorted({str(t['speaker_id']) for s in scenes for t in s['turns']}),
        scene_duration_seconds=sum(s['duration_samples'] for s in scenes)/16000,
        scope='ten source-backed review scenes, not a certified duplex training set',
        coverage=[dict(scene_id=s['scene_id'],requested=s['coverage_requested'],blockers=s['blockers']) for s in scenes]))
    render_review(root,scenes)
    return scenes


def render_review(root,scenes,output_rel="output/pilot10"):
    out=safe_child(root,output_rel)
    def audio_url(path):
        return html.escape(os.path.relpath(path,out),quote=True)
    cards=[]
    for s in scenes:
        rows=[]
        for t in s['turns']:
            original=safe_child(root,t['audio']['path'])
            rebuilt=original.with_name(original.stem+'.reconstructed.wav')
            reconstructed='<span>待重建</span>'
            if rebuilt.is_file() and rebuilt.with_suffix('.json').is_file():
                info=json.loads(rebuilt.with_suffix('.json').read_text())
                if digest(rebuilt)==info['sha256']:
                    reconstructed='<audio controls preload="none" src="'+audio_url(rebuilt)+'"></audio><small>待听审</small>'
            rows.append('<tr><td>'+html.escape(t['utterance_id'])+'</td><td>'+f"{t['start_sample']/16000:.2f}–{t['end_sample']/16000:.2f}"+'</td><td>'+html.escape(t['emotion'])+'</td><td>'+html.escape(t['original_text'])+'</td><td><audio controls preload="none" src="'+audio_url(original)+'"></audio></td><td>'+reconstructed+'</td></tr>')
        rows=''.join(rows)
        mix=audio_url(safe_child(root,s['review_audio']['path']))
        cards.append(f'<section><h2>{s["scene_id"]} · {s["coverage_requested"]}</h2><p>{s["source_scene"]} · 来源场景 · {s["duration_samples"]/16000:.2f} 秒</p><audio controls preload="none" src="{mix}"></audio><p class="warn">'+html.escape('；'.join(s['blockers']))+'</p><table><thead><tr><th>原始片段</th><th>场景秒数</th><th>来源情绪</th><th>原文（保留事件标记）</th><th>逐句音频</th></tr></thead><tbody>'+rows+'</tbody></table></section>')
        if s.get('source_channels'):
            tracks=''.join(f'<p>原始声道 {a["channel_index"]}（隔离程度待听审） <audio controls preload="none" src="{audio_url(safe_child(root,a["path"]))}"></audio></p>' for a in s['source_channels'])
            cards[-1]=cards[-1].replace('</section>',tracks+'</section>')
    page='''<!doctype html><meta charset="utf-8"><title>CineDuplex · 10 场景审阅</title><style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;background:#f5f6f8;color:#19212b}section{padding:24px;background:white;border-radius:12px;margin:24px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}audio{max-width:260px}.warn{color:#9b4b00;font-size:14px}header{line-height:1.7}h2{font-size:22px}</style><header><h1>CineDuplex · 10 场景审阅包</h1><p>真实来源片段已落盘；这些是待验收的候选场景，当前训练就绪数为 0。情绪沿用来源标注，压抑表达仅为文字线索，未冒称听审完成。</p><p>试听混音按原始时间摆放片段，EmotionTalk 缺失间隙用零填充且已标记，重叠区域保留一份原始混音。SmoothConv 保留原声道连续录音，试听混音只是两声道半幅求和，不作为独立训练轨道。S01–S08 来自 EmotionTalk G00002 两位说话人；S09–S10 来自 SmoothConv 真实双声道录音，具体来源见各场景。仅用于工程试制，不代表泛化能力；具体分组和声道验收状态见各 scene.json。</p><p>来源：<a href="https://huggingface.co/datasets/BAAI/Emotiontalk">EmotionTalk / Sun et al. (2025)</a> · CC BY-NC-SA 4.0；<a href="https://huggingface.co/datasets/qualialabsAI/SmoothConv">SmoothConv</a> · CC BY-NC 4.0 · 非商业研究</p></header>'''+''.join(cards)
    page=page.replace('<th>逐句音频</th>','<th>原音</th><th>重建音频</th>')
    page=page.replace('10 场景',str(len(scenes))+' 场景')
    page=page.replace('</header>','<p>听审顺序：逐句对比原音与重建，核对文字、音色、情绪和异常噪声；S08 需单独确认是否确有压抑表达；S09/S10 请分别听两个原始声道，确认串音、背景音乐与打断/附和是否符合预期。音频能播放不等于上述项目已经通过。</p></header>')
    (out/'review.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--annotations', default='manifests/Emotiontalk-audio-annotations.json',
                        help='NAS-relative annotation index; permits a verified reconstruction without overwriting a stale file')
    args=parser.parse_args();build(args.root,args.annotations)
