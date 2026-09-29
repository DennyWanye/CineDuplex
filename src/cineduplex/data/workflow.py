"""Explicit preparation/consolidation commands for the documented pilot recipe.

No automatic training or background scheduling. Invoke stages separately after
checking their receipts. Existing source-backed production modules are reused.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from cineduplex.contracts import digest, safe_child, write_json
from .nas_source import Source, nas_root, tar_prefix_inventory

PROJECT = Path(__file__).resolve().parents[3]
REVISED_SELECTIONS = [
    ('S09', 'interrupt', 12., 28., '1765799160_cuCZuOweWy_seg59_active'),
    ('S10', 'backchannel', 34., 47., '1769939396_AJdvYtW8AA_seg241_active'),
]


def verify_asset(path, spec):
    if path.stat().st_size != spec['size']:
        raise ValueError('asset size differs: '+path.name)
    if 'sha256' in spec and digest(path) != spec['sha256']:
        raise ValueError('asset SHA256 differs: '+path.name)
    if 'git_blob_sha1' in spec:
        blob=path.read_bytes()
        if hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()!=spec['git_blob_sha1']:
            raise ValueError('asset Git blob differs: '+path.name)


def acquire(root, spec):
    dest=safe_child(root,spec['destination'])
    # Existing server-side copies can have a different transfer receipt. Require
    # the pinned full-file identity, never redownload matching material.
    if dest.exists():
        verify_asset(dest,spec)
        return dest
    source=Source(root,spec['repo'],spec['revision'],spec.get('repo_type','dataset'))
    span=(0,spec['size']-1) if spec['size']>4*1024*1024 else None
    path=source.fetch(spec['source_path'],spec['destination'],span)
    verify_asset(path,spec)
    return path


def prepare_inputs(root):
    root=nas_root(root)
    spec=json.loads((PROJECT/'configs/data_sources.json').read_text())
    for name in ('raw','manifests','cache','tmp','output'):
        (root/name).mkdir(exist_ok=True)
    emotion=spec['emotiontalk']
    source=Source(root,emotion['repo'],emotion['revision'])
    source.fetch('README.md','raw/Emotiontalk-README.md')
    # Only metadata fields consumed by SparseTar are needed. The revision and
    # fixed archive size come from the pinned published manifest.
    meta=dict(id=emotion['repo'],sha=emotion['revision'],siblings=[
        dict(rfilename=emotion['archive'],size=emotion['archive_size'])])
    metadata=root/'manifests/Emotiontalk-repo.json'
    if metadata.exists():
        if json.loads(metadata.read_text())['sha']!=emotion['revision']:
            raise ValueError('existing EmotionTalk revision differs')
    else:write_json(metadata,meta)
    prefix=root/'raw/Emotiontalk-Audio-prefix64.tar'
    if not prefix.exists():
        source.fetch(emotion['archive'],str(prefix.relative_to(root)),(0,emotion['prefix_bytes']-1))
    verify_asset(prefix,dict(size=emotion['prefix_bytes'],sha256=emotion['prefix_sha256']))
    # The recovered index is an existing local workaround for a stale NAS file.
    # Never overwrite either index merely to reconstitute identical annotations.
    indexes=[root/'manifests'/n for n in ('Emotiontalk-audio-annotations.recovered.json','Emotiontalk-audio-annotations.json')]
    index=next((p for p in indexes if p.exists()),None)
    if index is None:
        _,annotations=tar_prefix_inventory(prefix,root)
    else:annotations=json.loads(index.read_text())
    if len(annotations)!=19250:raise ValueError('unexpected annotation inventory size')
    smooth=Source(root,'qualialabsAI/SmoothConv',spec['smoothconv_revision'])
    smooth.fetch('README.md','raw/SmoothConv/README.md')
    for asset in spec['assets']:
        acquire(root,asset)
        print('verified:',asset['destination'],flush=True)
    write_json(root/'manifests/pilot-inputs-ready.json',dict(status='PASSED',
        specification_sha256=digest(PROJECT/'configs/data_sources.json'),
        fixed_assets=len(spec['assets']),annotation_records=len(annotations),training_ready=False))


def prepare_reference(root):
    root=nas_root(root)
    expected='21826f35fafc9c7edcb6f88dc6b4d7b976d312824e0ef13f84f90a6800d4fca5'
    out=root/'raw/codec-reference/s3tokenizer-0.2.0.tar.gz'
    if out.exists():
        verify_asset(out,dict(size=225241,sha256=expected));return
    def get(url):
        result=subprocess.run(['curl','-fLsS','--proto','=https','--proto-redir','=https',
            '--connect-timeout','10','--max-time','60','--max-filesize','1048576',url],capture_output=True)
        if result.returncode:raise RuntimeError('public reference fetch failed; curl '+str(result.returncode))
        return result.stdout
    metadata=json.loads(get('https://pypi.org/pypi/s3tokenizer/0.2.0/json'))
    source=next(x for x in metadata['urls'] if x['filename']==out.name)
    if source['digests']['sha256']!=expected:raise ValueError('PyPI reference identity changed')
    blob=get(source['url'])
    if hashlib.sha256(blob).hexdigest()!=expected or len(blob)!=225241:
        raise ValueError('reference payload mismatch')
    nas_root(root);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(blob)
    verify_asset(out,dict(size=225241,sha256=expected))
    write_json(out.parent/'acquisition.json',dict(source='https://pypi.org/project/s3tokenizer/0.2.0/',
        filename=out.name,sha256=expected,bytes=len(blob),installed=False))


def revised(root):
    root=nas_root(root)
    out=root/'output/pilot10-revision2'
    if (out/'scenes.jsonl').exists():
        raise FileExistsError('revision already exists; inspect it, do not overwrite')
    from .smoothconv import audit,build_candidates
    from .pilot import render_review
    for selection in REVISED_SELECTIONS:audit(root,selection[-1])
    scenes=build_candidates(root,REVISED_SELECTIONS,output_rel='output/pilot10-revision2')
    (out/'scenes.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in scenes))
    render_review(root,scenes,output_rel='output/pilot10-revision2')


def consolidate(root):
    root=nas_root(root);out=root/'output/pilot10-current'
    if (out/'scenes.jsonl').exists():
        raise FileExistsError('current package already exists; preserve reviews and inspect status')
    from .pilot import render_review
    from .codec_cpu import MODEL_SHA
    scenes=[];reviews=[];seen=set();total=0
    for i in range(1,11):
        folder='pilot10' if i<=8 else 'pilot10-revision2'
        path=root/f'output/{folder}/S{i:02}/scene.json'
        scene=json.loads(path.read_text());scene['source_manifest']=str(path.relative_to(root))
        for turn in scene['turns']:
            uid=turn['utterance_id']
            if uid in seen:raise ValueError('duplicate utterance identity')
            seen.add(uid);wav=safe_child(root,turn['audio']['path'])
            if digest(wav)!=turn['audio']['sha256']:raise ValueError('source audio changed')
            code=json.loads(wav.with_suffix('.tokens.json').read_text())
            if code['input_sha256']!=turn['audio']['sha256'] or code['model_sha256']!=MODEL_SHA:
                raise ValueError('token identity mismatch')
            raw=code['raw_codes']
            if not raw or any(type(x)is not int or not 0<=x<6561 for x in raw) or code['offset_applied']!=0:
                raise ValueError('invalid raw codes')
            total+=len(raw)
            rebuilt=wav.with_name(wav.stem+'.reconstructed.wav')
            receipt=json.loads(rebuilt.with_suffix('.json').read_text())
            if digest(rebuilt)!=receipt['sha256']:raise ValueError('reconstruction changed')
            reviews.append(dict(scene_id=scene['scene_id'],utterance_id=uid,
                source_audio=turn['audio']['path'],reconstruction=str(rebuilt.relative_to(root)),
                text_correct=None,speaker_match=None,emotion_match=None,noise_or_crosstalk=None,
                reviewer=None,reviewed_at=None,approved=False))
        scene['training_ready']=False
        scene['objective_codec_status']='ENCODED_AND_RECONSTRUCTION_HASH_VERIFIED'
        scene['auditory_codec_status']='NOT_PERFORMED'
        scenes.append(scene)
    if len(seen)!=28:raise ValueError('expected 28 current utterances')
    out.mkdir(parents=True,exist_ok=True)
    (out/'scenes.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in scenes))
    write_json(out/'summary.json',dict(scenes=10,utterances=len(seen),raw_tokens=total,
        scene_seconds=sum(s['duration_samples'] for s in scenes)/16000,training_ready=0,
        delivery_status='REVIEW_CANDIDATES_READY_ACCEPTANCE_PENDING'))
    write_json(out/'auditory-review-queue.json',dict(records=reviews,training_ready=False,
        special_checks=['S08 suppression requires hearing','S09/S10 channel quality and events require hearing']))
    render_review(root,scenes,output_rel='output/pilot10-current')


def status(root):
    root=nas_root(root)
    out=root/'output/pilot10-current'
    for name in ['summary.json','hf-review-readback.json','prosody-readback.json','frontend-reference-check.json']:
        path=out/name
        if path.exists():
            obj=json.loads(path.read_text())
            print(name,json.dumps({k:v for k,v in obj.items() if k not in ('rows','records') or isinstance(v,int)},ensure_ascii=False))
        else:print(name,'MISSING')
    path=out/'auditory-review-queue.json'
    if path.exists():
        q=json.loads(path.read_text())['records']
        print('reviewed',sum(x.get('reviewed_at') is not None for x in q),'approved',sum(x.get('approved') is True for x in q))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare-inputs','prepare-reference','revised','consolidate','status'])
    parser.add_argument('--root',required=True)
    args=parser.parse_args(argv)
    {'prepare-inputs':prepare_inputs,'prepare-reference':prepare_reference,
     'revised':revised,'consolidate':consolidate,'status':status}[args.action](args.root)
    return 0


if __name__=='__main__':raise SystemExit(main())
