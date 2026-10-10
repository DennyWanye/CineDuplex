#!/usr/bin/env python3
"""Restore only this run's lost NAS transport, retaining the loaded GPU worker."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from cineduplex.cluster.guard import guard
from cineduplex.train.nas_mount import DirectMount
from cineduplex.train.nas_transport import NAS

cfg=json.loads(Path('configs/train-overnight.local.json').read_text())
root=Path('/Volumes/media/CineDuplex/output/realtime-20261009-v1')
assert Path('/Volumes/media').is_mount()
with (root/'mount-controller.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guard(cfg['cluster'],heartbeat=True)
    nas=NAS(cfg['nas']);node=next(n for n in cfg['nodes'] if n['rank']==3)
    stamp=str(time.time_ns())
    # A lost mount let two old heartbeat writers create metadata locally.
    # Preserve these exact files on NAS; never adopt that directory as storage.
    code='''import os,pathlib,json,time
p=pathlib.Path.home()/'cineduplex-work/nas-inference'
assert not os.path.ismount(p) and not p.is_symlink()
files=list(p.rglob('*'))
expected={'output/realtime-20261009-v1/SERVICE.json','output/training-20261009-v1/web/resident/SERVICE.json'}
assert {str(x.relative_to(p)) for x in files if x.is_file()}==expected
assert all(x.stat().st_size<4096 for x in files if x.is_file())
data={str(x.relative_to(p)):x.read_text() for x in files if x.is_file()}
print(json.dumps({'files':data}))
'''
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',node['ssh_alias'],
        'python3 -c '+__import__('shlex').quote(code)],capture_output=True,text=True,timeout=15)
    if r.returncode:raise RuntimeError(r.stderr[-300:])
    evidence=json.loads(r.stdout)
    nas.put('output/realtime-20261009-v1/disconnected-metadata-'+stamp+'.json',json.dumps(evidence).encode())
    mount=DirectMount(nas,node,cfg['nas_sftp'],cfg['nas_python'],mount_name='nas-inference',restore_empty_metadata=True)
    # Remove only the two preserved small heartbeat files and empty own dirs.
    cleanup='''import pathlib,json
root=pathlib.Path.home()/'cineduplex-work'
saved={}
for p in root.glob('nas-inference-disconnected-*'):
 assert not p.is_symlink()
 files=[x for x in p.rglob('*') if x.is_file()]
 assert {str(x.relative_to(p)) for x in files}<={'output/realtime-20261009-v1/SERVICE.json','output/training-20261009-v1/web/resident/SERVICE.json'}
 assert all(x.stat().st_size<4096 and not x.is_symlink() for x in files)
 saved[p.name]={str(x.relative_to(p)):x.read_text() for x in files}
dest=root/'nas-inference/output/realtime-20261009-v1/disconnected-preserved.json'
assert not dest.exists()
dest.write_text(json.dumps(saved))
for name,files in saved.items():
 p=root/name
 for name,text in files.items():
  x=p/name
  assert x.read_text()==text
  x.unlink()
 for x in sorted(p.rglob('*'),key=lambda x:len(x.parts),reverse=True):x.rmdir()
 p.rmdir()
'''
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',node['ssh_alias'],
        'python3 -c '+__import__('shlex').quote(cleanup)],input=json.dumps(evidence),capture_output=True,text=True,timeout=15)
    if r.returncode:raise RuntimeError('preserved metadata cleanup failed: '+r.stderr[-300:])
    nas.put('output/realtime-20261009-v1/mount-restored-'+stamp+'.json',json.dumps({
        'at':time.time(),'pid':os.getpid(),'worker_pid':1371284,'generation':cfg['cluster']['lease']['generation'],
        'policy':'NAS transport only; existing loaded model retained','mount':'nas-inference'}).encode())
    while True:
        guard(cfg['cluster'],heartbeat=True)
        if mount.proc.poll() is not None:raise RuntimeError('NAS transport exited; do not fall back locally')
        time.sleep(30)
