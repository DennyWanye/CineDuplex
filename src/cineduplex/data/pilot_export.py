"""Export review artifacts and fail closed on unreviewed duplex training data."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from cineduplex.contracts import digest, write_json, safe_child
from .nas_source import nas_root


def export_review(root, output_rel='output/pilot10'):
    root=nas_root(root)
    os.environ['HF_DATASETS_CACHE']=str(root/'cache/datasets')
    os.environ['TMPDIR']=str(root/'tmp')
    from datasets import Dataset, load_from_disk
    out=safe_child(root,output_rel)
    scenes=[json.loads(line) for line in (out/'scenes.jsonl').read_text().splitlines()]
    items=[]
    for scene in scenes:
        for turn in scene['turns']:
            codepath=safe_child(root,turn['audio']['path']).with_suffix('.tokens.json')
            codes=json.loads(codepath.read_text()) if codepath.exists() else None
            if codes and codes['input_sha256']!=turn['audio']['sha256']:
                raise ValueError('tokens do not belong to this audio')
            items.append(dict(scene_id=scene['scene_id'],source_scene=scene['source_scene'],
                split_group=scene['split_group'],split=scene['split'],coverage_requested=scene['coverage_requested'],
                utterance_id=turn['utterance_id'],speaker_id=turn['speaker_id'],
                start_sample=turn['start_sample'],end_sample=turn['end_sample'],
                audio_path=str(root/turn['audio']['path']),audio_sha256=turn['audio']['sha256'],
                text=turn['text'],original_text=turn['original_text'],emotion=turn['emotion'],
                speech_tokens=codes['raw_codes'] if codes else None,token_offset=0,
                clean_target_candidate=turn['clean_target_candidate'],
                training_ready=False,review_required=scene['blockers']))
    dest=out/'hf_review_dataset'
    Dataset.from_list(items).save_to_disk(str(dest))
    restored=load_from_disk(str(dest))
    if len(restored)!=len(items):
        raise ValueError('persisted review row count differs')
    for expected, actual in zip(items, restored):
        if expected!=actual:
            raise ValueError('persisted review row differs from source')
    write_json(out/'hf-review-readback.json',dict(
        rows=len(restored),exact_row_comparison=True,training_ready=False,
        scope='Hugging Face save/load integrity only; not Lychee loader validation'))
    # This dataset is deliberately not in the Lychee training schema. No fake
    # empty user tracks, mixed target tracks, or unreviewed gold labels are emitted.
    blockers=sorted({b for scene in scenes for b in scene['blockers']})
    write_json(out/'lychee-export-status.json',dict(
        exported=False,training_records=0,review_records=len(items),blockers=blockers,
        reason='Review dataset retained; training export requires verified causal tracks and labels.'))
    return items


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output-relative',default='output/pilot10')
    args=parser.parse_args()
    print('review rows',len(export_review(args.root,args.output_relative)))
