#!/usr/bin/env python3
"""Qualify the already resident TP runtime; never launch or reload GPU work."""
import argparse,json,shlex,time,threading
from pathlib import Path
from cineduplex.train.nas_transport import NAS
from cineduplex.cluster.guard import guard

def main():
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--prefix',default='output/realtime-tp4-20261010-v9');p.add_argument('--version',required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
 if not a.version.isalnum():raise ValueError('version')
 cfg=json.loads(Path(a.config).read_text());n=NAS(cfg['nas']);pre=a.prefix;name=pre+'/QUALIFY-'+a.version
 if n.stat(name+'-REQUEST.json') is not None:raise RuntimeError('preserve existing attempt')
 guard(cfg['cluster'],heartbeat=True)
 def api(route,data=None,headers=None):
  cmd=['curl','-fsS','--max-time','25']
  if data is not None:
   cmd+=['-X','POST','--data-binary','@-']
   for h in headers or ['Content-Type: application/json']:cmd+=['-H',h]
  return json.loads(n.run(shlex.join(cmd+['http://127.0.0.1:18092'+route]),data=data,timeout=30))
 s=api('/status')
 if not s['ready'] or s['active'] is not None or s['tp_runtime_sha256']!=a.source_sha:raise RuntimeError('resident identity/idle differs')
 r=api('/calls',b'{"variant":"trained:male:zh"}');cid=r['id'];print(cid,flush=True);n.put(name+'-REQUEST.json',json.dumps(r).encode());done=threading.Event();errors=[]
 def feeder():
  seq=0
  try:
   while not done.is_set():
    api('/calls/'+cid+'/audio',bytes(12800),['Content-Type: application/octet-stream','X-Audio-Seq:'+str(seq)]);seq+=1;done.wait(4)
  except Exception as e:errors.append(str(e))
 t=threading.Thread(target=feeder,daemon=True);t.start();rows=[];after=0;deadline=time.monotonic()+120
 try:
  while time.monotonic()<deadline:
   e=api('/calls/'+cid+'/events?after='+str(after));after=e['after'];rows+=e['events']
   if any(x['type'] in ('error','closed') for x in rows):break
   if any(x['type']=='tensor_parallel_config' for x in rows) and any(x['type']=='metrics' for x in rows):break
 finally:
  done.set();t.join(30);api('/calls/'+cid+'/stop',b'{}')
  lifecycle_deadline=time.monotonic()+35;post_stop=None
  while time.monotonic()<lifecycle_deadline:
   e=api('/calls/'+cid+'/events?after='+str(after));after=e['after'];rows+=e['events']
   post_stop=api('/status')
   if post_stop.get('active') is None:
    # Cleanup can finish between events polling and the status read. Drain
    # the final closed/error events before deciding the lifecycle result.
    e=api('/calls/'+cid+'/events?after='+str(after));after=e['after'];rows+=e['events']
    break
   time.sleep(.5)
  n.put(name+'.json',json.dumps({'call':cid,'events':rows,'feeder_errors':errors,'post_stop':post_stop}).encode())
 if not post_stop or not post_stop.get('ready') or post_stop.get('active') is not None or post_stop.get('tp_runtime_sha256')!=a.source_sha:
  raise RuntimeError('post-stop lifecycle did not leave original runtime ready/idle')
 if errors or any(x['type']=='error' for x in rows) or not any(x['type']=='closed' for x in rows) or not any(x['type']=='metrics' for x in rows):
  raise RuntimeError('qualification or stop failed; diagnostic preserved')
 q=next((x for x in rows if x['type']=='tensor_parallel_config'),None)
 if q is None:raise RuntimeError('qualification did not pass; preserved diagnostic')
 print(json.dumps({'decoder_checks':len(q['decoder_checks']),'selected_graphs':sum(x['selected'] for x in q.get('fixed_shape_mlp_graphs',[])),'graph_count':len(q.get('fixed_shape_mlp_graphs',[]))}),flush=True)
if __name__=='__main__':main()
