"""Submit one private diagnostic after verifying original Gateway ownership.

No automatic POST retry. The immutable intent and returned call identity are
kept on NAS so an unknown transport outcome can be reconciled before retrying.
"""
import argparse
import json
from pathlib import Path
import time
import uuid
from cineduplex.cluster.guard import guard
from cineduplex.train.nas_transport import NAS


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path)
    p.add_argument('--pins',required=True);p.add_argument('--prefix',required=True);p.add_argument('--record',required=True)
    p.add_argument('--voice',choices=('female','male'),default='female')
    args=p.parse_args()
    if Path(args.record).name!=args.record:raise ValueError('record basename required')
    config=json.loads(args.config.read_text());nas=NAS(config['nas'])
    guard(config['cluster'],heartbeat=True)
    pins=json.loads(nas.read(args.pins,timeout=15))
    before=json.loads(nas.run('curl -fsS --max-time 10 http://127.0.0.1:18092/status',timeout=15))
    safety=before.get('guardrail',{})
    if not(before.get('ready') and before.get('active') is None and before.get('tp_runtime_sha256')==pins['tp']['sha256']
           and before.get('decode_precision')=='bf16' and before.get('quantization')=='none'
           and all(safety.get(k) for k in ('enabled','input_audio','input_semantic','output_text','output_audio'))):
        raise ValueError('expected idle pinned native service and safety required')
    record=args.prefix.rstrip('/')+'/'+args.record
    nas.put(record+'-INTENT.json',json.dumps(dict(at=time.time(),before=before,pins=args.pins,private=True,public_audio=False,voice=args.voice)).encode())
    result=json.loads(nas.run("curl -fsS --max-time 20 -X POST -H 'Content-Type: application/json' --data-binary @- http://127.0.0.1:18092/calls",
                             data=json.dumps({'variant':'codec-probe:'+args.voice}).encode(),timeout=25))
    call=str(uuid.UUID(result['id']));nas.put(record+'-CALL.json',json.dumps(result).encode())
    print(json.dumps(dict(call=call,record=record+'-CALL.json')))

if __name__=='__main__':main()
