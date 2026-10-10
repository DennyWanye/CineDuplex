#!/usr/bin/env python3
"""Rebind the preserved formal web to its recovered backend's NAS directory."""
import argparse,hashlib,json,shlex,sys,time
from pathlib import Path
from cineduplex.cluster.guard import guard
from cineduplex.train.nas_transport import NAS
IMAGE='sha256:5ac677d3dba4ac066166919ef9f6fddbc2497f7f9a8eb077f5597dbb18998908'
NAME='cineduplex-realtime-web'
def main():
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--from-prefix',required=True);p.add_argument('--to-prefix',required=True);p.add_argument('--web-version',required=True,type=int);a=p.parse_args()
 OLD=a.from_prefix;NEW=a.to_prefix;WEB='web-source-v'+str(a.web_version);BACKUP=NAME+'-v'+str(a.web_version)+'-before-'+NEW.rsplit('-',1)[-1]
 for pre in (OLD,NEW):
  if not pre.startswith('output/realtime-tp4-') or '/' in pre[7:] or '..' in pre:raise ValueError('NAS prefix')
 cfg=json.loads(Path(a.config).read_text());n=NAS(cfg['nas']);guard(cfg['cluster'],heartbeat=True)
 def command(args,timeout=40):return n.run(shlex.join(args),timeout=timeout)
 status=json.loads(command(['curl','-fsS','--max-time','5','http://127.0.0.1:18092/status']))
 if not status.get('ready') or status.get('active'):raise ValueError('recovered backend must be idle/ready')
 old=json.loads(command(['docker','inspect',NAME]))[0]
 if old['Image']!=IMAGE or n.path(OLD) not in [m['Source'] for m in old['Mounts']] or '/service/'+WEB+'/realtime_server.py' not in old['Config']['Cmd']:
  raise ValueError('formal web identity differs')
 n.run('if docker inspect '+BACKUP+' >/dev/null 2>&1; then exit 1; fi',timeout=10)
 pins={}
 for file in ('realtime_server.py','realtime.html','realtime.js','capture.worklet.js'):
  data=n.read(OLD+'/'+WEB+'/'+file,timeout=15)
  pins[file]=n.put(NEW+'/'+WEB+'/'+file,data)
 n.put(NEW+'/FORMAL-STORAGE-RESTORE-REQUEST.json',json.dumps({'at':time.time(),'status':status,'pins':pins}).encode())
 launch=['docker','run','-d','--name',NAME,'--restart','unless-stopped','--read-only','--log-driver','none','--network','host','--user','0:0','--cap-drop','ALL','--cap-add','SETUID','--cap-add','SETGID','--security-opt','no-new-privileges','--pids-limit','96','--memory','256m','--no-healthcheck','--mount','type=bind,src='+n.path(NEW)+',dst=/service','--mount','type=bind,src=/srv/nas/.nas-online,dst=/srv/nas/.nas-online,readonly']
 for src,dst in (('certs/current/host.crt','host.crt'),('certs/current/host.key','host.key'),('public/agentos-household-root.crt','root.crt')):
  launch+=['--mount','type=bind,src=/var/lib/agentos/'+src+',dst=/run/cineduplex/'+dst+',readonly']
 launch+=['-e','PYTHONDONTWRITEBYTECODE=1','-e','TMPDIR=/service/tmp','--entrypoint','/usr/local/bin/python',IMAGE,'/service/'+WEB+'/realtime_server.py','--root','/service/'+WEB,'--cert','/run/cineduplex/host.crt','--key','/run/cineduplex/host.key','--upstream','http://127.0.0.1:18092']
 command(['docker','stop',NAME]);command(['docker','rename',NAME,BACKUP])
 try:
  cid=command(launch).decode().strip()
  for _ in range(20):
   try:
    verified=json.loads(command(['curl','-fsS','--max-time','3','http://127.0.0.1:18089/api/status']))
    if verified.get('ready') and verified.get('tp_runtime_sha256')==status['tp_runtime_sha256']:break
   except RuntimeError:pass
   time.sleep(.5)
  else:raise RuntimeError('restored proxy unavailable')
  served=command(['curl','-fsS','--max-time','3','http://127.0.0.1:18089/realtime.js'])
  if hashlib.sha256(served).hexdigest()!=pins['realtime.js']['sha256']:raise RuntimeError('formal frontend changed')
 except Exception:
  n.run('if docker inspect '+NAME+' >/dev/null 2>&1; then docker stop '+NAME+'; docker rename '+NAME+' '+NAME+'-restore-failed; fi',timeout=40)
  command(['docker','rename',BACKUP,NAME]);command(['docker','start',NAME]);raise
 print(n.put(NEW+'/FORMAL-STORAGE-RESTORED.json',json.dumps({'at':time.time(),'container':cid,'status':verified,'web_pins':pins,'backup':BACKUP,'frontend_changed':False}).encode()))
if __name__=='__main__':main()
