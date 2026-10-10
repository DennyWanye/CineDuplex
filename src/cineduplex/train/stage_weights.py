"""Reuse verified existing DGX shards, download missing ones directly to NAS."""
import argparse
import hashlib
import json
import shlex
import subprocess
import re
from pathlib import Path
from .nas_transport import NAS


def stage(config):
    nas = NAS(config['nas'])
    repo = json.loads(Path(config['manifest']).read_text())['repositories'][0]
    assert repo['repo_id'] == 'HIT-TMG/Lychee-FD'
    assert repo['revision'] == 'ac06087bef70b33d4dfebf63a3eadadb4e024f33'
    entries=repo['files']
    if config.get('start_at'):
        entries=entries[next(i for i,e in enumerate(entries) if e['path']==config['start_at']):]
    for entry in entries:
        name = entry['path']
        if '/' in name: raise ValueError('expected flat checkpoint')
        rel = 'raw/lychee-full-duplex/'+name
        expected = entry.get('sha256')
        if nas.stat(rel) is not None:
            alg = 'sha256sum' if expected else 'sha1sum'
            command = alg+' '+shlex.quote(nas.path(rel)) if expected else (
                '{ printf '+shlex.quote('blob '+str(entry['size'])+'\\0')+'; cat '
                +shlex.quote(nas.path(rel))+'; } | sha1sum')
            actual = nas.run(command, timeout=600).decode().split()[0]
            if actual != (expected or entry['git_blob_oid']): raise ValueError('existing NAS file mismatch: '+name)
            print('REUSED NAS', name, flush=True); continue
        if not expected:
            r = subprocess.run(['/usr/bin/curl','-fsSL','--max-time','45',entry['url']],capture_output=True,timeout=50)
            if r.returncode: raise RuntimeError('metadata download failed: '+name)
            if hashlib.sha1(b'blob '+str(len(r.stdout)).encode()+b'\0'+r.stdout).hexdigest()!=entry['git_blob_oid']:
                raise ValueError('git blob mismatch')
            nas.put(rel, r.stdout); print('SAVED metadata',name,flush=True); continue
        source = config['existing_root'].rstrip('/')+'/'+name
        probe = subprocess.run(['ssh','-o','BatchMode=yes',config['existing_host'],
            'test -f '+shlex.quote(source)+' && stat -c %s '+shlex.quote(source)],capture_output=True,text=True,timeout=20)
        if not config.get('force_download',False) and probe.returncode == 0 and int(probe.stdout)==entry['size']:
            print('COPY existing shard to NAS',name,flush=True)
            src = subprocess.Popen(['ssh','-o','BatchMode=yes',config['existing_host'],'cat '+shlex.quote(source)],stdout=subprocess.PIPE)
            try:
                dest = subprocess.run(nas.ssh+[nas.writer_command(rel,expected)],stdin=src.stdout,timeout=3600)
                src.stdout.close(); src.wait(timeout=15)
                if dest.returncode or src.returncode: raise RuntimeError('copy failed')
            finally:
                if src.poll() is None: src.terminate()
        else:
            print('DOWNLOAD missing shard to NAS',name,flush=True)
            # Resolve the CDN on Mac; signed URL travels only on SSH stdin.
            resolved = subprocess.run(['/usr/bin/curl','-fsSIL','--max-time','40','-o','/dev/null','-w','%{url_effective}',entry['url']+'?download=true'],capture_output=True,text=True,timeout=45)
            if resolved.returncode or not resolved.stdout.startswith('https://'): raise RuntimeError('CDN resolution failed')
            if name in config.get('segmented_files',[]):
                source=Path(__file__).with_name('segmented_fetch.py').read_text()
                command=nas.guarded('sudo -n -u family env PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(config['nas_python'])+' -u -c '+shlex.quote(source))
                payload={'root':nas.config['root'],'uuid':nas.config['uuid'],'name':name,'size':entry['size'],
                         'sha256':expected,'url':resolved.stdout.strip()}
                r=subprocess.run(nas.ssh+[command],input=json.dumps(payload).encode(),timeout=1900)
                if r.returncode:raise RuntimeError('segmented NAS download failed: '+name)
                print('VERIFIED shard',name,flush=True);continue
            dest = nas.path(rel); partial=dest+'.partial'
            script = ('set -eu; test ! -e '+shlex.quote(dest)+'; mkdir -p '+shlex.quote(str(Path(dest).parent))+'; '
                'curl -4 -fsSL --http1.1 --proto =https --proto-redir =https --connect-timeout 30 --max-time 1800 '
                '--speed-time 60 --speed-limit 1024 --continue-at - --output '+shlex.quote(partial)+' --config -; '
                'test "$(sha256sum '+shlex.quote(partial)+' | cut -d " " -f 1)" = '+shlex.quote(expected)+'; '
                'sync -f '+shlex.quote(partial)+'; mv -n '+shlex.quote(partial)+' '+shlex.quote(dest))
            command=nas.guarded('sudo -n -u family sh -c '+shlex.quote(script))
            result=subprocess.run(nas.ssh+[command],input=('url = '+json.dumps(resolved.stdout.strip())+'\n').encode(),capture_output=True,timeout=1820)
            if result.returncode:
                message=re.sub(r'https?://\S+','<redacted-url>',result.stderr.decode(errors='replace'))[-500:]
                raise RuntimeError('NAS download failed: '+name+' exit='+str(result.returncode)+' '+message)
        print('VERIFIED shard',name,flush=True)
    print('SELECTED FILES VERIFIED ON NAS' if config.get('start_at') else 'ALL WEIGHTS VERIFIED ON NAS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args()
    stage(json.loads(Path(a.config).read_text()))
