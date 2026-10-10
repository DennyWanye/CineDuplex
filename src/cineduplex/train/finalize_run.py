"""Verify persisted training, independent resume, generation and decoded audio."""
import hashlib
import json


def finalize(nas,prefix):
    def read(key):return json.loads(nas.read(prefix+'/'+key))
    identity=read('RUN-IDENTITY.json')
    expected_scenes=set(identity.get('expected_scenes',[f'S{i:02}' for i in range(1,11)]))
    expected_rows=identity.get('expected_rows',14)
    generation_scenes=identity.get('generation_scenes',['S03','S06','S07','S08'])
    if not expected_scenes or expected_rows<1 or not generation_scenes or len(set(generation_scenes))!=len(generation_scenes):
        raise ValueError('invalid run inventory')
    ranks=[read(f'resumed/rank{i}-complete.json') for i in range(4)]
    resume=[read(f'resumed/resume-comparison-rank{i}.json') for i in range(4)]
    restored=[read(f'resumed/restored-state-rank{i}.json') for i in range(4)]
    if not all(r['ok'] and r['last_step']==16 and r['changed'] for r in ranks):
        raise ValueError('training incomplete')
    if {s for r in ranks for s in r['observed_scenes']}!=expected_scenes:
        raise ValueError('missing scene coverage')
    if {s for r in ranks for s in r['observed_source_rows']}!=set(range(expected_rows)):
        raise ValueError('missing row coverage')
    if not all(r['ok'] and r['independent_worker_restart'] for r in resume):
        raise ValueError('independent resume failed')
    if not all(r['optimizer_exact'] and r['cuda_rng_exact'] and r['cpu_rng_exact'] for r in restored):
        raise ValueError('restored state is not exact')
    hashes={r['adapter_sha256'] for r in ranks}
    if len(hashes)!=1:raise ValueError('rank parameters differ')
    adapter_sha=next(iter(hashes))
    audio=read('audio/summary.json')
    wanted={(sid,variant) for sid in generation_scenes for variant in ('base','trained')}
    if len(audio['records'])!=len(wanted) or {(r['scene_id'],r['variant']) for r in audio['records']}!=wanted:
        raise ValueError('missing or duplicate WAV')
    for sid in generation_scenes:
        predictions=[read(f'generation/{sid}-{v}.json') for v in ('base','trained')]
        if len({p['prompt_sha256'] for p in predictions})!=1:raise ValueError('comparison prompts differ')
        for p,v in zip(predictions,('base','trained')):
            if (not p['autoregressive'] or p['target_tokens_in_prompt'] or not p['has_speech_codes']
                or p['adapter_sha256']!=adapter_sha or p['adapter_enabled']!=(v=='trained')):
                raise ValueError('generation provenance mismatch')
    for r in audio['records']:
        expected=f'{prefix}/audio/{r["scene_id"]}-{r["variant"]}.wav'
        if r['path']!=expected or not r['autoregressive'] or r['same_utterance_prompt'] or r['target_tokens_in_prompt']:
            raise ValueError('audio provenance mismatch')
        if hashlib.sha256(nas.read(r['path'])).hexdigest()!=r['sha256']:raise ValueError('WAV readback failed')
    result={'world_size':4,'steps':16,'ranks':ranks,'resume':resume,'restored_state':restored,
        'audio':audio,'technical_flow_complete':True,'quality_improvement_verified':False,
        'listening_review':'NOT_PERFORMED'}
    nas.put(prefix+'/TECHNICAL-COMPLETE.json',json.dumps(result,indent=2).encode())
    return result
