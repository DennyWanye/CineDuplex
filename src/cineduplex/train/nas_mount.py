"""Unprivileged task-owned SSHFS using an SSH/SFTP pipe, without copied keys.

Only the mountpoint is local. The verified executable is an anonymous memfd;
all cache/temp file contents live on NAS. No root mount or network changes.
"""
import base64
import hashlib
import json
import shlex
import subprocess

BOOT=r'''
import os,sys,json,base64,hashlib,pathlib,socket,getpass,time
line=bytearray()
while not line.endswith(b'\n'):line.extend(os.read(0,1))
c=json.loads(line)
assert (socket.gethostname(),getpass.getuser())==(c['hostname'],c['user'])
root=pathlib.Path.home()/'cineduplex-work'
assert not root.is_symlink() and (root/'.cineduplex-owner.json').is_file()
assert c.get('mount_name','nas-training') in ('nas-training','nas-inference','nas-realtime')
mount=root/c.get('mount_name','nas-training')
mount.mkdir(exist_ok=True)
assert not mount.is_symlink() and not os.path.ismount(mount)
binary=base64.b64decode(c['binary'])
assert hashlib.sha256(binary).hexdigest()==c['sha256']
fd=os.memfd_create('cineduplex-sshfs',0);os.write(fd,binary);os.fchmod(fd,0o500)
if c.get('restore_empty_metadata') and list(mount.iterdir()):
 assert c['mount_name']=='nas-inference'
 files=[x for x in mount.rglob('*') if x.is_file()]
 assert {str(x.relative_to(mount)) for x in files}<={'output/realtime-20261009-v1/SERVICE.json','output/training-20261009-v1/web/resident/SERVICE.json'}
 assert files and all(not x.is_symlink() and x.stat().st_size<4096 for x in files)
 mount.rename(mount.with_name('nas-inference-disconnected-'+str(time.time_ns())))
 mount.mkdir()
assert not list(mount.iterdir())
print('READY',flush=True)
os.execve('/proc/self/fd/'+str(fd),['sshfs',':'+c['root'],str(mount),'-f','-o','passive,idmap=user,auto_cache,fsname=cineduplex-nas'],os.environ)
'''


# Keep bulk SFTP traffic between NAS and DGX. The Mac only forwards its
# existing authentication agent to the trusted NAS for this control session.
NAS_SUPERVISOR=r'''
import sys,json,subprocess,threading,shlex,time
c=json.loads(sys.stdin.readline())
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15',
 '-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+c['known_hosts'],c['target']]
client=subprocess.Popen(ssh+['python3 -u -c '+shlex.quote(c['boot'])],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
client.stdin.write(json.dumps(c['payload']).encode()+b'\n');client.stdin.flush()
if client.stdout.readline()!=b'READY\n':raise RuntimeError('DGX SSHFS bootstrap failed')
server=subprocess.Popen(['sudo','-n','-u','family',c['sftp']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,dst):
 try:
  while data:=src.read1(65536):dst.write(data);dst.flush()
 except (OSError,ValueError):pass
 finally:
  try:dst.close()
  except (OSError,ValueError):pass
for a,b in ((client,server),(server,client)):
 threading.Thread(target=pump,args=(a.stdout,b.stdin),daemon=True).start()
ready=False
try:
 for _ in range(20):
  check=subprocess.run(ssh+['findmnt -rn -t fuse.sshfs -o TARGET | grep -Fx "$HOME/cineduplex-work/'+c['payload'].get('mount_name','nas-training')+'"'],capture_output=True,timeout=30)
  if check.returncode==0:ready=True;break
  if client.poll() is not None:raise RuntimeError(client.stderr.read().decode()[-500:])
  time.sleep(.5)
 if not ready:raise RuntimeError('NAS mount readiness failed')
 print('MOUNT_READY',flush=True)
 sys.stdin.read()
finally:
 if ready:
  result=subprocess.run(ssh+['fusermount3 -u "$HOME/cineduplex-work/'+c['payload'].get('mount_name','nas-training')+'"'],capture_output=True,timeout=30)
  if result.returncode:raise RuntimeError('mount is busy; no force unmount')
 for p in (client,server):
  if p.poll() is None:p.terminate()
  p.wait(timeout=10)
'''


class DirectMount:
    def __init__(self,nas,node,sftp,python,mount_name='nas-training',restore_empty_metadata=False):
        self.node=node
        if mount_name not in ('nas-training','nas-inference','nas-realtime'):raise ValueError('invalid task mount')
        binary=nas.read('cache/sshfs/sshfs-arm64')
        if hashlib.sha256(binary).hexdigest()!='d42d5b5dc641c15ed40c8ea8148ff18d53d8e7c53b4034a366ce4ae4b2da30ef':
            raise ValueError('SSHFS executable changed')
        payload={**node,'root':nas.config['root'],'mount_name':mount_name,'restore_empty_metadata':restore_empty_metadata,'binary':base64.b64encode(binary).decode(),
            'sha256':hashlib.sha256(binary).hexdigest()}
        cfg={'payload':payload,'boot':BOOT,'known_hosts':nas.path('cache/sshfs/dgx-all-known-hosts'),
             'target':node['user']+'@'+node['address'],'sftp':sftp}
        cmd=nas.guarded('env PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(python)+' -u -c '+shlex.quote(NAS_SUPERVISOR))
        self.proc=subprocess.Popen(nas.ssh[:1]+['-A']+nas.ssh[1:]+[cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.proc.stdin.write(json.dumps(cfg)+'\n');self.proc.stdin.flush()
        if self.proc.stdout.readline()!='MOUNT_READY\n':
            raise RuntimeError('direct NAS mount failed: '+self.proc.stderr.read()[-500:])
        print(node['hostname'],'direct NAS SSHFS mounted; keys remain on Mac',flush=True)
    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=45)
        if self.proc.returncode:raise RuntimeError('NAS mount cleanup failed: '+self.proc.stderr.read()[-500:])
