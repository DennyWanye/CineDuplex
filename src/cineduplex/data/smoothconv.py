"""Inspect real source channels before admitting SmoothConv duplex candidates.

Channel difference is necessary, but does not establish speaker isolation.
No denoising, source separation, or synthetic silence is introduced here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from cineduplex.contracts import digest, safe_child, write_json
from .nas_source import nas_root
from .pilot import read_pcm, save_pcm

REVISION = 'cd74b4fca285a66d6ac8c16228d0953ff1e0cda2'
STEM = '1765799160_cuCZuOweWy_seg59_active'


def metrics(pcm):
    x = pcm.astype(np.float64)
    rms = np.sqrt(np.mean(x*x, axis=0))
    std = x.std(axis=0)
    corr = float(np.corrcoef(x.T)[0, 1]) if np.all(std > 0) else None
    return dict(frames=len(x), rms=rms.tolist(), correlation=corr,
                identical_channels=bool(np.array_equal(pcm[:, 0], pcm[:, 1])),
                channel_0_to_1_db=float(20*np.log10((rms[0]+1e-9)/(rms[1]+1e-9))))


def audit(root, stem):
    root = nas_root(root)
    if Path(stem).name != stem or not stem.replace('_', '').isalnum():
        raise ValueError('source basename required')
    folder = safe_child(root, 'raw/SmoothConv')
    audio, annotation = folder/(stem+'.wav'), folder/(stem+'.json')
    # Require an acquisition receipt and recheck its actual source bytes.
    for path in (audio, annotation):
        receipt = json.loads(path.with_suffix(path.suffix+'.receipt.json').read_text())
        if (receipt['repo'] != 'qualialabsAI/SmoothConv' or receipt['revision'] != REVISION
                or digest(path) != receipt['sha256']):
            raise ValueError('unverified source bytes')
    rate, pcm = read_pcm(audio)
    if pcm.shape[1] != 2 or not len(pcm):
        raise ValueError('this inspector requires a nonempty two-channel PCM recording')
    annotation_data = json.loads(annotation.read_text())
    speech = []
    for row in annotation_data['instances']:
        if not row.get('text', '').strip():
            continue
        channel = row['channelIndex']
        start, end = float(row['start']), float(row['end'])
        if channel not in (0, 1) or not 0 <= start < end <= len(pcm)/rate + 1/rate:
            raise ValueError('annotation channel or timeline outside real audio')
        lo, hi = round(start*rate), min(len(pcm), round(end*rate))
        if hi <= lo:
            raise ValueError('empty annotated interval')
        speech.append(dict(source_annotation=row, channel_metrics=metrics(pcm[lo:hi])))
    # Preserve original recording continuity on each real channel. These are
    # audition assets, not certified clean user/assistant training targets.
    out = root/'output/source-audit/SmoothConv'/stem
    out.mkdir(parents=True, exist_ok=True)
    tracks = []
    for channel in (0, 1):
        nas_root(root)
        dest = out/f'channel-{channel}-16k.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(audio),
                        '-af', f'pan=mono|c0=c{channel}', '-ar', '16000',
                        '-c:a', 'pcm_s16le', str(dest)], check=True, timeout=60)
        tracks.append(dict(channel_index=channel, path=str(dest.relative_to(root)), sha256=digest(dest)))
    report = dict(source_audio=str(audio.relative_to(root)), sha256=digest(audio),
                  source_annotation=str(annotation.relative_to(root)),
                  sample_rate=rate, channels=2, duration_seconds=len(pcm)/rate,
                  full_recording_metrics=metrics(pcm), speech=speech,
                  nontext_annotations=[r for r in annotation_data['instances'] if not r.get('text', '').strip()],
                  audition_tracks=tracks, independent_speaker_tracks='NOT_ESTABLISHED',
                  auditory_review='NOT_PERFORMED', training_ready=False,
                  interpretation='Different channels and energy ratios alone do not prove absence of crosstalk.')
    write_json(out/'audit.json', report)
    print(json.dumps(dict(output=str(out), duration_seconds=report['duration_seconds'],
                          full_recording_metrics=report['full_recording_metrics'],
                          speech_segments=len(speech), training_ready=False), ensure_ascii=False), flush=True)
    return report


def build_candidates(root, selections=None, output_rel="output/pilot10"):
    """Build source-backed S09/S10 review candidates, retaining isolation blockers."""
    root = nas_root(root)
    out = safe_child(root, output_rel)
    out.mkdir(exist_ok=True, parents=True)
    rights_path = out/'SmoothConv-rights.json'
    write_json(rights_path, dict(source_id='qualialabsAI/SmoothConv', revision=REVISION,
        license_id='CC-BY-NC-4.0', allowed_uses=['noncommercial_research'], commercial_use=False,
        source_url='https://huggingface.co/datasets/qualialabsAI/SmoothConv',
        original_card='raw/SmoothConv/README.md',
        original_card_sha256=digest(root/'raw/SmoothConv/README.md')))
    scenes = []
    selections = selections or [('S09', 'interrupt', 18.0, 25.0, STEM), ('S10', 'backchannel', 10.0, 18.0, STEM)]
    for sid, category, start, end, stem in selections:
        if Path(stem).name != stem or not stem.replace('_', '').isalnum():
            raise ValueError('invalid source stem')
        if not sid.startswith('S') or not sid[1:].isdigit() or not 0 <= start < end:
            raise ValueError('invalid scene identity or interval')
        folder = root/'output/source-audit/SmoothConv'/stem
        report = json.loads((folder/'audit.json').read_text())
        if digest(root/report['source_audio']) != report['sha256']:
            raise ValueError('source changed since channel audit')
        tracks = []
        for track in report['audition_tracks']:
            path = root/track['path']
            if digest(path) != track['sha256']:
                raise ValueError('audition channel changed')
            rate, pcm = read_pcm(path)
            if rate != 16000 or pcm.shape[1] != 1:
                raise ValueError('normalized channel format mismatch')
            tracks.append(pcm[:, 0])
        if len(tracks[0]) != len(tracks[1]):
            raise ValueError('channels must retain the same timeline')

        dest = out/sid
        dest.mkdir(exist_ok=True)
        lo, hi = round(start*16000), round(end*16000)
        if hi > len(tracks[0]):
            raise ValueError("scene exceeds verified source audio")
        channel_assets = []
        for channel, data in enumerate(tracks):
            path = dest/f'channel-{channel}.wav'
            save_pcm(path, 16000, data[lo:hi])
            channel_assets.append(dict(channel_index=channel, path=str(path.relative_to(root)), sha256=digest(path)))
        turns = []
        for item in report['speech']:
            row = item['source_annotation']
            if row['end'] <= start or row['start'] >= end:
                continue
            if row['start'] < start or row['end'] > end:
                raise ValueError('candidate window would cut a source utterance')
            uid = row['id']; channel = row['channelIndex']
            # Source IDs are used only as filenames after strict validation.
            if not uid.replace('-', '').isalnum():
                raise ValueError('invalid source utterance id')
            path = dest/(uid+'.wav')
            save_pcm(path, 16000, tracks[channel][round(row['start']*16000):round(row['end']*16000)])
            turns.append(dict(utterance_id=uid, speaker_id=row['attributes']['speaker'],
                channel_index=channel, start_sample=round((row['start']-start)*16000),
                end_sample=round((row['end']-start)*16000), source_start_seconds=row['start'],
                source_end_seconds=row['end'], text=row['text'], original_text=row['text'],
                emotion=row['attributes']['emotion'], source_annotation=row,
                audio=dict(path=str(path.relative_to(root)), sha256=digest(path)),
                auditory_review='NOT_PERFORMED', clean_target_candidate=False, overlaps=[]))
        overlaps = []
        for i, a in enumerate(turns):
            for b in turns[i+1:]:
                overlap = min(a['end_sample'], b['end_sample'])-max(a['start_sample'], b['start_sample'])
                if a['channel_index'] != b['channel_index'] and overlap > 0:
                    a['overlaps'].append(b['utterance_id']); b['overlaps'].append(a['utterance_id'])
                    overlaps.append(dict(a=a['utterance_id'], b=b['utterance_id'],
                                         duration_seconds=overlap/16000, interpretation='DISTINCT_SOURCE_CHANNELS_ISOLATION_REVIEW_PENDING'))
        # Listening mix is explicitly derivative; retain separate original tracks.
        mix = np.rint((tracks[0][lo:hi].astype(float)+tracks[1][lo:hi].astype(float))/2)
        save_pcm(dest/'review_mix.wav', 16000, mix)
        scene = dict(schema_version='cineduplex.review_scene.v1', scene_id=sid,
            coverage_requested=category, source_scene=stem, split='candidate_unassigned', split_group=stem,
            source_revision=REVISION, rights_ref=str(rights_path.relative_to(root)),
            source_start_seconds=start, source_end_seconds=end, sample_rate=16000, duration_samples=hi-lo,
            turns=turns, overlap_evidence=overlaps, suppression_evidence=[], source_channels=channel_assets,
            review_audio=dict(path=str((dest/'review_mix.wav').relative_to(root)),
                sha256=digest(dest/'review_mix.wav'), provenance='half-amplitude sum for audition only; original channels retained', unknown_gap_samples=[]),
            annotation_status='SOURCE_LABELS_PRESERVED_REVIEW_PENDING', training_ready=False,
            blockers=['auditory_review_not_performed', 'channel_isolation_not_verified',
                      'codec_roundtrip_not_verified', 'official_loader_not_verified', 'split_assignment_pending'])
        write_json(dest/'scene.json', scene)
        scenes.append(scene)
        print(f'{sid} {category}: {len(turns)} source-channel turns; training_ready=false', flush=True)
    return scenes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--build-candidates', action='store_true')
    args = parser.parse_args()
    audit(args.root, args.stem)
    if args.build_candidates:
        if args.stem != STEM:
            raise ValueError('candidate selection applies only to the pinned selected recording')
        build_candidates(args.root)
