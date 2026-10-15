#!/usr/bin/env python3
"""Prepare an idle-only private call probe under the existing Gateway owner.

No workers are launched here. The normal pinned hot-update tool applies the
prepared source, and the admitted original call worker owns/cancels its probe.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from cineduplex.train.nas_transport import NAS

BODY = '''    def codec_probe(self):
        import os,signal,subprocess,hashlib
        from pathlib import Path
        process=None
        try:
            environment_root=Path(self.app.root).parents[1]/'output/official-backend-20261013-v1'
            checker=environment_root/ENV_RECORD
            evidence=checker.read_bytes()
            if hashlib.sha256(evidence).hexdigest()!=ENV_SHA:
                raise RuntimeError('official environment evidence identity')
            if not json.loads(evidence).get('ok'):
                raise RuntimeError('official environment import check did not pass')
            runner=environment_root/RUNNER_NAME
            if hashlib.sha256(runner.read_bytes()).hexdigest()!=RUNNER_SHA:
                raise RuntimeError('official reference code identity')
            inputs=environment_root/INPUTS_NAME
            raw=inputs.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=INPUTS_SHA:
                raise RuntimeError('same-weight recorded-input evidence identity')
            request={**json.loads(raw),'parent_pid':os.getpid(),
                'environment_root':str(environment_root),'service_root':str(self.app.root),
                'model_overlay':str(environment_root/'model-overlay'),'kernel_only':KERNEL_ONLY}
            request_path=self.folder/'official-request.json'
            atomic(request_path,request)
            env=os.environ.copy()
            for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR',
                        'MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','HF_TOKEN','OPENAI_API_KEY'):
                env.pop(key,None)
            env.update(CINEDUPLEX_REFERENCE_PARENT=str(os.getpid()),CUDA_VISIBLE_DEVICES='0',
                PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(environment_root/'tmp'))
            begun=time.monotonic()
            with (self.folder/'official-reference.log').open('wb') as log:
                process=subprocess.Popen([str(environment_root/'env-arm64/bin/python'),'-B',
                    str(runner),str(request_path)],env=env,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
                atomic(self.folder/'official-process.json',{'pid':process.pid,'parent_pid':os.getpid(),
                    'call':self.id,'scope':'private existing-admission reference','microphone':False,'speaker':False})
                while process.poll() is None:
                    if self.closed.wait(.25):
                        raise RuntimeError('official reference canceled with its owning call')
                    if time.monotonic()-begun>BUDGET:
                        raise TimeoutError('bounded official reference time exceeded')
                if process.returncode:
                    raise RuntimeError('official reference failed; inspect its private NAS log')
            report=json.loads((self.folder/'official-reference.json').read_text())
            if not report.get('ok'):
                raise RuntimeError('official reference did not pass')
            atomic(self.folder/'probe.json',{'kind':'official-reference','seconds':time.monotonic()-begun,
                'kernel_only':KERNEL_ONLY,'quality_accepted':False,'released_audio_events':0})
        except Exception as exc:
            self.emit({'type':'error','message':str(exc)[:400]})
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            self.closed.set()
            self.emit({'type':'closed','reason':'official_reference_complete'})
            try:self.archive.close()
            finally:
                with self.app.lock:
                    if self.app.active is self:self.app.active=None
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--environment-record',required=True)
    parser.add_argument('--tp-version',type=int,required=True)
    parser.add_argument('--call-version',type=int,required=True)
    parser.add_argument('--full-reference',action='store_true')
    parser.add_argument('--prefix',default='output/realtime-tp4-20261014-quality-recovery-v2')
    parser.add_argument('--runner',default='run_reference_v2.py')
    parser.add_argument('--inputs',default='COMPARE-INPUTS-v2.json')
    args=parser.parse_args()
    n=NAS(json.loads(Path(args.config).read_text())['nas'])
    if not re.fullmatch(r'output/realtime-tp4-[a-zA-Z0-9-]+',args.prefix):
        parser.error('unique existing admitted runtime prefix required')
    if not re.fullmatch(r'run_reference(?:_v\d+)?\.py',args.runner) or not re.fullmatch(r'COMPARE-INPUTS(?:-v\d+)?\.json',args.inputs):
        parser.error('fixed reference asset names required')
    pre=args.prefix+'/'
    env='output/official-backend-20261013-v1/'
    source=n.read(pre+'call-runtime-v53.py',timeout=15).decode()
    if hashlib.sha256(source.encode()).hexdigest()!='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214':
        raise RuntimeError('original Call53 changed')
    env_raw=n.read(env+args.environment_record,timeout=15)
    if not json.loads(env_raw).get('ok'):raise RuntimeError('environment is not qualified for a GPU check')
    constants={'ENV_RECORD':args.environment_record,'ENV_SHA':hashlib.sha256(env_raw).hexdigest(),
        'RUNNER_NAME':args.runner,'RUNNER_SHA':hashlib.sha256(n.read(env+args.runner,timeout=15)).hexdigest(),
        'INPUTS_NAME':args.inputs,'INPUTS_SHA':hashlib.sha256(n.read(env+args.inputs,timeout=15)).hexdigest(),
        'KERNEL_ONLY':not args.full_reference,'BUDGET':900 if args.full_reference else 120}
    tree=ast.parse(source)
    call=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Call')
    method=next(node for node in call.body if isinstance(node,ast.FunctionDef) and node.name=='codec_probe')
    lines=source.splitlines(True)
    updated=''.join(lines[:method.lineno-1])+BODY+''.join(lines[method.end_lineno:])
    updated='\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n'+updated
    ast.parse(updated)
    name=f'call-runtime-v{args.call_version}.py'
    pin=n.put(pre+name,updated.encode())
    tp=n.read(pre+'tp-runtime-v73.py',timeout=15).decode()
    if hashlib.sha256(tp.encode()).hexdigest()!='2d9405253fa14781c83fd5db6868e403f8a9b7b8d10f3d9d167f3c2e5c7ecb6a':
        raise RuntimeError('original TP73 changed')
    tp=tp.replace("CALL_RUNTIME_NAME='call-runtime-v53.py'",'CALL_RUNTIME_NAME='+repr(name)).replace(
        "CALL_RUNTIME_SHA='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214'",'CALL_RUNTIME_SHA='+repr(pin['sha256']))
    ast.parse(tp)
    pins={'call':pin,'tp':n.put(pre+f'tp-runtime-v{args.tp_version}.py',tp.encode()),
          'kernel_only':not args.full_reference,'guardrail_bundle':'v15','production_quality_release':False}
    print(n.put(env+f'PROBE-{args.tp_version}-PINS.json',json.dumps(pins,indent=2).encode()))
    print(json.dumps(pins))


if __name__=='__main__':main()
