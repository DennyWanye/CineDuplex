"""Bounded HTTPS ranges committed directly on NAS; interrupted blocks discarded."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def run(c):
    root=Path(c['root']);dest=root/'raw/lychee-full-duplex'/c['name'];partial=dest.with_name(dest.name+'.partial')
    if dest.exists():raise ValueError('final already exists')
    if subprocess.check_output(['findmnt','-n','-o','UUID','-T',str(root)],text=True).strip()!=c['uuid']:
        raise ValueError('real NAS missing')
    if not Path('/srv/nas/.nas-online').exists():raise ValueError('NAS marker missing')
    with dest.with_name(dest.name+'.download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>c['size']:raise ValueError('oversized partial')
        started=time.monotonic();last_report=offset
        def fetch(start):
            end=min(start+8*1024*1024,c['size'])-1;length=end-start+1
            for attempt in range(3):
                r=subprocess.run(['curl','-4','-fsSL','--http1.1','--proto','=https','--proto-redir','=https',
                    '--connect-timeout','15','--max-time','55','--max-filesize',str(length),
                    '--range',f'{start}-{end}','--dump-header','/dev/stderr','--write-out','%{http_code}',
                    '--config','-'],input=('url = '+json.dumps(c['url'])+'\n').encode(),capture_output=True,timeout=60)
                status=r.stdout[-3:] if len(r.stdout)>=3 else b''
                if status in (b'401',b'403'):raise PermissionError('dataset/CDN access denied; hard pause')
                ranges=re.findall(rb'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',r.stderr,re.I)
                if r.returncode==0 and status==b'206' and len(r.stdout)-3==length and ranges and tuple(map(int,ranges[-1]))==(start,end,c['size']):
                    return r.stdout[:-3]
                if attempt==2:raise RuntimeError('range failed without committing: '+str(start)+' curl='+str(r.returncode)+' status='+status.decode(errors='replace'))
                time.sleep(2)
        # Four independent bounded ranges, committed strictly in source order.
        # Each response is capped at 8 MiB; there are no sparse/guessed file holes.
        with partial.open('ab') as out, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            while offset<c['size']:
                if time.monotonic()-started>1800:raise TimeoutError('bounded range session exhausted; resume retained')
                starts=list(range(offset,min(offset+4*8*1024*1024,c['size']),8*1024*1024))
                futures=[pool.submit(fetch,start) for start in starts]
                for start,future in zip(starts,futures):
                    data=future.result()
                    if start!=offset:raise ValueError('noncontiguous block')
                    out.write(data);out.flush();os.fsync(out.fileno());offset+=len(data)
                if offset-last_report>=128*1024*1024 or offset==c['size']:
                    print(json.dumps({'file':c['name'],'committed_bytes':offset,'total_bytes':c['size'],
                        'seconds':round(time.monotonic()-started,2)}),flush=True);last_report=offset
        with partial.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=c['sha256']:raise ValueError('full shard checksum failed; partial retained')
        os.link(partial,dest);partial.unlink()
        print('VERIFIED '+c['name'],flush=True)


if __name__=='__main__':run(json.load(sys.stdin))
