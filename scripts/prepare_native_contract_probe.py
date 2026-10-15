"""Prepare a private resident-model trace; apply through the admitted updater."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from cineduplex.train.nas_transport import NAS

TP73 = '2d9405253fa14781c83fd5db6868e403f8a9b7b8d10f3d9d167f3c2e5c7ecb6a'
CALL53 = '472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214'
BODY = '''    def codec_probe(self):
        import hashlib,importlib.util
        try:
            path=self.app.root.parents[1]/TRACE_RUNNER['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=TRACE_RUNNER['sha256']:
                raise RuntimeError('native trace code identity')
            spec=importlib.util.spec_from_file_location('private_native_input_trace',str(path))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.run(self,TRACE_SPEC)
            atomic(self.folder/'probe.json',{'kind':'native-input-contract',
                'quality_accepted':False,'released_audio_events':0})
        except Exception as exc:
            self.emit({'type':'error','message':str(exc)[:400]})
        finally:
            self.closed.set()
            self.emit({'type':'closed','reason':'native_input_trace_complete'})
            try:self.archive.close()
            finally:
                with self.app.lock:
                    if self.app.active is self:self.app.active=None
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=Path)
    p.add_argument('--prefix', required=True)
    p.add_argument('--tp-version', type=int, required=True)
    p.add_argument('--call-version', type=int, required=True)
    p.add_argument('--frontend-receipt')
    p.add_argument('--cache-trace', action='store_true')
    p.add_argument('--weights-contract', action='store_true')
    p.add_argument('--alignment-contract', action='store_true')
    p.add_argument('--small-batch', action='store_true')
    p.add_argument('--acoustic-weights', action='store_true')
    p.add_argument('--acoustic-training', action='store_true')
    p.add_argument('--acoustic-replay', action='store_true')
    p.add_argument('--encoder-context', action='store_true')
    p.add_argument('--turn-refresh', action='store_true')
    p.add_argument('--encoder-stage-comparison', action='store_true', help='Separate frontend framing from encoder context on the same received audio')
    p.add_argument('--context-replay', action='store_true')
    p.add_argument('--context-repair-only', action='store_true', help='Validate only the revised native context; reuse saved original baseline')
    p.add_argument('--context-base-comparison', action='store_true')
    p.add_argument('--travel-replay', action='store_true')
    p.add_argument('--output-contract-replay', action='store_true')
    p.add_argument('--output-contract-only', action='store_true')
    p.add_argument('--normal-repeat-comparison', action='store_true', help='Match normal Chinese call prefix and output contract on the two recorded repeat questions')
    p.add_argument('--projection-contract', action='store_true', help='Read-only native row-partition arithmetic prerequisite')
    p.add_argument('--training-result', help='Immutable completed acoustic A/B RESULT.json on NAS')
    a = p.parse_args()
    if (a.weights_contract or a.alignment_contract or a.small_batch or a.acoustic_weights or a.acoustic_training or a.acoustic_replay or a.encoder_context) and (a.cache_trace or a.frontend_receipt):
        raise ValueError('weight contract is an independent read-only probe')
    if sum((a.weights_contract,a.alignment_contract,a.small_batch,a.acoustic_weights,a.acoustic_training,a.acoustic_replay,a.encoder_context))>1:
        raise ValueError('select one contract')
    if not re.fullmatch(r'output/realtime-tp4-[a-zA-Z0-9-]+', a.prefix):
        raise ValueError('existing admitted runtime required')
    nas = NAS(json.loads(a.config.read_text())['nas'])
    pre = a.prefix + '/'
    env = 'output/official-backend-20261013-v1/'
    call = nas.read(pre+'call-runtime-v53.py', timeout=15)
    tp = nas.read(pre+'tp-runtime-v73.py', timeout=15)
    if hashlib.sha256(call).hexdigest()!=CALL53 or hashlib.sha256(tp).hexdigest()!=TP73:
        raise ValueError('original runtime changed')
    upstream_path = env+'source/lychee_fd/runtime/vllm_generation.py'
    upstream = nas.read(upstream_path, timeout=15)
    contract = json.loads(nas.read(env+'COMPARE-INPUTS-v10.json', timeout=15))
    spec = {'cases': contract['cases'], 'upstream_runtime': {
        'path': upstream_path, 'sha256': hashlib.sha256(upstream).hexdigest()}}
    if a.weights_contract or a.alignment_contract or a.small_batch or a.acoustic_training:
        config = json.loads(a.config.read_text())
        for key, path in [('config','raw/lychee-full-duplex/config.json'),
                          ('index','raw/lychee-full-duplex/model.safetensors.index.json'),
                          ('dataset','cache/train-code/loader-source.py')]:
            data = nas.read(path, timeout=15)
            spec[key] = {'path':path,'sha256':hashlib.sha256(data).hexdigest()}
        if spec['dataset']['sha256'] != config['loader_sha256']:
            raise ValueError('frozen training Dataset identity differs')
        manifest = json.loads(Path(config['manifest']).read_text())
        repo = next(r for r in manifest['repositories'] if any(
            f['path']=='model-00001-of-00006.safetensors' for f in r['files']))
        spec['shards'] = [{k:f[k] for k in ('path','size','sha256')}
                          for f in repo['files'] if f['path'].endswith('.safetensors')]
        if a.alignment_contract:
            path = pre+'TRAINING-WINDOW-CONTRACT-146.json'
            data = nas.read(path, timeout=15)
            spec['training_receipt'] = {'path':path,'sha256':hashlib.sha256(data).hexdigest()}
        if a.small_batch or a.acoustic_training:
            base = Path(__file__).resolve().parents[1]
            for key,filename in [('exporter','export_bulk.py'),('audio_contract','native_audio_contract.py')]:
                data=(base/'src/cineduplex/train'/filename).read_bytes();ast.parse(data)
                spec[key]=nas.put(env+f'small-batch-{a.tp_version}-{filename}',data)
            batch=json.loads(nas.read('output/training-20261009-v1/batches.json',timeout=15))
            selected={}
            spec['selection_sources']={}
            for split,count in [('train',2),('eval',1)]:
                item=batch['input_files'][split];data=nas.read(item['path'],timeout=20)
                if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('original training input changed')
                selected[split]=[json.loads(line) for line in data.decode().splitlines()[:count]]
                spec['selection_sources'][split]={**item,'selected_indices':list(range(count))}
            spec['small_inputs']=nas.put(env+f'small-batch-{a.tp_version}-inputs.json',json.dumps(selected).encode())
    if a.frontend_receipt:
        if not a.frontend_receipt.startswith(pre+'calls/') or not a.frontend_receipt.endswith('/native-input-contract.json'):
            raise ValueError('frontend evidence must be in this admitted run')
        prior = nas.read(a.frontend_receipt, timeout=15)
        spec['frontend_receipt'] = {'path': a.frontend_receipt,
            'sha256': hashlib.sha256(prior).hexdigest()}
    if a.acoustic_weights or a.acoustic_training or a.acoustic_replay or a.encoder_context:
        path='output/training-20261009-v1/attempt-01/training/checkpoint-3222.pt'
        receipt=json.loads(nas.read(path+'.json',timeout=15))
        spec['checkpoint']={'path':path,'sha256':receipt['sha256'],'bytes':nas.stat(path)}
    if a.context_base_comparison and not a.context_replay:raise ValueError('context base requires context replay')
    if a.context_repair_only and (not a.context_replay or a.context_base_comparison or a.output_contract_replay or a.travel_replay):
        raise ValueError('context repair requires the two original questions and unchanged output contract')
    if a.context_replay:
        if not a.acoustic_replay:raise ValueError('context replay requires acoustic replay')
        raw=Path(__file__).resolve().parents[1].joinpath('src/cineduplex/train/native_audio_context.py').read_bytes();ast.parse(raw)
        spec['audio_context']=nas.put(env+f'native-audio-context-{a.tp_version}.py',raw)
        spec['arms']=([{'name':'context_refresh','checkpoint':spec['checkpoint'],'scale':.25},{'name':'context_refresh_base','checkpoint':spec['checkpoint'],'scale':0.}] if a.context_base_comparison else [{'name':'original','checkpoint':spec['checkpoint']},{'name':'context_refresh','checkpoint':spec['checkpoint']}])
        if a.context_repair_only:
            spec['arms']=[{'name':'context_refresh','checkpoint':spec['checkpoint']}]
            spec['context_repair_only']=True
            spec['language']='zh'
    if a.travel_replay:
        if not a.context_replay or a.context_base_comparison:raise ValueError('travel requires original/context comparison')
        spec['cases']=[]
        for index in range(1,5):
            path=f'output/realtime-tp4-20261013-after-dsv41-v1/morning-asr/travel-q{index}.wav'
            data=nas.read(path,timeout=15)
            spec['cases'].append({'id':f'travel-q{index}','path':path,'sha256':hashlib.sha256(data).hexdigest()})
        spec['settling_seconds']=48
        spec['scope']='four original reviewer excerpts, continuous session with digital settling; not original wall-clock replay'
    if a.output_contract_only and not a.output_contract_replay:raise ValueError('output-only requires output contract replay')
    if a.output_contract_replay:
        if not (a.travel_replay or (a.normal_repeat_comparison and a.context_replay)):raise ValueError('output contract replay requires a pinned failure set')
        data=Path(__file__).resolve().parents[1].joinpath('src/cineduplex/train/native_output_contract.py').read_bytes();ast.parse(data)
        spec['output_contract']=nas.put(env+f'native-output-contract-{a.tp_version}.py',data)
        spec['arms']=[{'name':'original_output' if a.output_contract_only else 'context_refresh_output','checkpoint':spec['checkpoint'],'scale':.25}]
    if a.normal_repeat_comparison:
        if not a.output_contract_replay or a.travel_replay:raise ValueError('normal repeat requires repeat output contract')
        spec['language']='zh'
    if a.acoustic_replay and not a.context_replay:
        result_path=a.training_result
        if not result_path or not result_path.startswith(pre+'calls/') or not result_path.endswith('/acoustic-ab/RESULT.json'):
            raise ValueError('acoustic replay requires a completed training result in this admitted run')
        result_raw=nas.read(result_path,timeout=15);result=json.loads(result_raw)
        if not result.get('passed_training_mechanics') or result['errors'] or not result['resident_adapter_unchanged']:
            raise ValueError('completed acoustic comparison required')
        spec['training_result']={'path':result_path,'sha256':hashlib.sha256(result_raw).hexdigest()}
        spec['arms']=[{'name':'original','checkpoint':spec['checkpoint']}]+[
            {'name':arm['name'],'checkpoint':arm['checkpoint']} for arm in result['arms']]
    if a.turn_refresh and not a.encoder_context:raise ValueError('turn refresh requires encoder context probe')
    if a.encoder_stage_comparison and (not a.encoder_context or a.turn_refresh):
        raise ValueError('encoder stages require independent encoder context probe')
    spec['encoder_stage_comparison']=a.encoder_stage_comparison
    if a.encoder_stage_comparison:spec['language']='zh'
    spec['turn_refresh']=a.turn_refresh
    if a.acoustic_replay or a.encoder_context:
        spec['seed']=20261009
        for key,name in [('sampling','independent-sampling-v2.py'),('stream','generation-stream-v4.py')]:
            raw=nas.read(pre+name,timeout=15)
            spec[key]={'path':pre+name,'sha256':hashlib.sha256(raw).hexdigest()}
    cache_runner = None
    if a.cache_trace:
        cache_source = Path(__file__).with_name('trace_native_cache_contract.py').read_bytes()
        ast.parse(cache_source)
        cache_runner = nas.put(env+f'native-cache-trace-{a.tp_version}.py', cache_source)
        spec['cache_trace'] = cache_runner
    filename = ('trace_native_projection_contract.py' if a.projection_contract else
                'trace_native_encoder_context.py' if a.encoder_context else
                'trace_native_weight_contract.py' if a.weights_contract else
                'trace_native_alignment_contract.py' if a.alignment_contract else
                'trace_native_small_batch.py' if a.small_batch else
                'train_native_window_comparison.py' if a.acoustic_training else
                'replay_acoustic_window_comparison.py' if a.acoustic_replay else
                'trace_native_acoustic_weights.py' if a.acoustic_weights else
                'trace_native_input_contract.py')
    source = Path(__file__).with_name(filename).read_bytes()
    ast.parse(source)
    runner = nas.put(env+f'native-input-trace-{a.tp_version}.py', source)
    tree = ast.parse(call)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='Call')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='codec_probe')
    lines = call.decode().splitlines(True)
    body=BODY
    if a.acoustic_training:
        body=body.replace('            module.run(self,TRACE_SPEC)',
            '            import torch\n            with torch.inference_mode(False), torch.enable_grad():\n                module.run(self,TRACE_SPEC)')
    changed = ('TRACE_RUNNER='+repr(runner)+'\nTRACE_SPEC='+repr(spec)+'\n'+
        ''.join(lines[:method.lineno-1])+body+''.join(lines[method.end_lineno:]))
    ast.parse(changed)
    call_name = f'call-runtime-v{a.call_version}.py'
    cp = nas.put(pre+call_name, changed.encode())
    updated = tp.decode().replace("CALL_RUNTIME_NAME='call-runtime-v53.py'", 'CALL_RUNTIME_NAME='+repr(call_name))
    updated = updated.replace('CALL_RUNTIME_SHA='+repr(CALL53), 'CALL_RUNTIME_SHA='+repr(cp['sha256']))
    if a.acoustic_weights:
        needle='            return record\n        except ValueError:'
        assert updated.count(needle)==1
        hook=f'''            acoustic_path=self.root.parents[1]/{runner['path']!r}
            acoustic_source=acoustic_path.read_bytes()
            if hashlib.sha256(acoustic_source).hexdigest()!={runner['sha256']!r}:raise ValueError('acoustic audit source')
            acoustic=types.ModuleType('private_acoustic_weight_audit')
            exec(compile(acoustic_source,str(acoustic_path),'exec'),acoustic.__dict__)
            acoustic.install(model,self.root,getattr(self,'profile_id',None),self.rank,{spec!r})
'''
        updated=updated.replace(needle,hook+needle)
    if cache_runner:
        needle = '            return record\n        except ValueError:'
        assert updated.count(needle) == 1
        hook = f'''            trace_path=self.root.parents[1]/{cache_runner['path']!r}
            trace_source=trace_path.read_bytes()
            if hashlib.sha256(trace_source).hexdigest()!={cache_runner['sha256']!r}:raise ValueError('cache trace identity')
            trace_module=types.ModuleType('private_native_cache_trace')
            exec(compile(trace_source,str(trace_path),'exec'),trace_module.__dict__)
            trace_module.install(model,self.root,profile_id,self.rank)
'''
        updated = updated.replace(needle, hook+needle)
    ast.parse(updated)
    pins = {'runner':runner, 'call':cp, 'tp':nas.put(pre+f'tp-runtime-v{a.tp_version}.py',updated.encode()),
            'baseline_tp':TP73, 'spec':spec, 'production_quality_release':False}
    nas.put(env+f'PROBE-{a.tp_version}-PINS.json',json.dumps(pins,indent=2).encode())
    print(json.dumps(pins))


if __name__=='__main__':
    main()
