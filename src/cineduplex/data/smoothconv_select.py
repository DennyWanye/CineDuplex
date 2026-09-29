"""Find source-annotated concurrent backchannels without moving timestamps.

This only nominates sources for waveform inspection; it never approves channels
or emotion labels, and it does not overwrite the ten existing review scenes.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from cineduplex.contracts import write_json
from .nas_source import Source, nas_root
from .smoothconv import REVISION


def select(root, limit=24):
    root = nas_root(root)
    inventory = json.loads((root/'manifests/SmoothConv-tree-none_edu_70h_2ch.json').read_text())
    sizes = {x['path']: x['size'] for x in inventory if x['type'] == 'file'}
    names = sorted((x for x in sizes if x.endswith('.json') and x[:-5]+'.wav' in sizes),
                   key=lambda x: sizes[x[:-5]+'.wav'])[:limit]
    source = Source(root, 'qualialabsAI/SmoothConv', REVISION)
    scanned, candidates, rejected = [], [], []
    for name in names:
        dest = source.fetch(name, 'raw/SmoothConv/'+Path(name).name)
        rows = [x for x in json.loads(dest.read_text())['instances'] if x.get('text', '').strip()]
        for bc in rows:
            if bc.get('attributes', {}).get('turn') != 'backchannel':
                continue
            for speech in rows:
                if speech['channelIndex'] == bc['channelIndex']:
                    continue
                # Preserve natural onset. The loader's six-second lead is a
                # configurable sampling filter, not a source-validity rule.
                if not (speech['start'] <= bc['start'] < bc['end'] <= speech['end']):
                    continue
                if any(x.get('attributes', {}).get('age') != 'adult' for x in (bc, speech)):
                    continue
                if any('<unclear>' in x['text'] for x in (bc, speech)):
                    continue
                hazards = [x for x in rows if x['id'] not in (bc['id'], speech['id'])
                    and max(x['start'], speech['start']) < min(x['end'], speech['end'])
                    and (re.fullmatch(r'\s*<[^>]+>\s*', x['text'])
                         or 'speaker box' in x.get('attributes', {}).get('speech event', []))]
                if hazards:
                    rejected.append(dict(source_path=name, backchannel_id=bc['id'],
                        reason='overlapping source-annotated noise/music/speaker-box event',
                        hazard_annotations=hazards))
                    continue
                candidates.append(dict(source_path=name, audio_path=name[:-5]+'.wav',
                    audio_bytes=sizes[name[:-5]+'.wav'], backchannel=bc, other_speech=speech,
                    source_revision=REVISION,
                    relative_start_seconds=bc['start']-speech['start'],
                    passes_default_six_second_filter=bc['start']-speech['start'] >= 6,
                    auditory_review='NOT_PERFORMED', training_ready=False))
        scanned.append(name)
        write_json(root/'output/source-audit/SmoothConv/backchannel-selection.json',
                   dict(scanned=scanned, candidates=candidates, rejected=rejected,
                        waveform_verified=False, training_ready=False))
        print(f'{len(scanned)} annotation files inspected; {len(candidates)} contained concurrent BC candidates', flush=True)
        if len(candidates) >= 3:
            break
    return candidates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--limit', type=int, default=24)
    args = parser.parse_args()
    select(args.root, args.limit)
