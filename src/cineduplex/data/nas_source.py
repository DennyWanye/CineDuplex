"""Pinned, credential-safe, bounded Hugging Face reads on the mounted NAS."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

from cineduplex.contracts import digest, safe_child, write_json


def nas_root(root):
    root = Path(root).absolute()
    mount = Path('/Volumes/media')
    mounts = subprocess.check_output(['mount'], text=True)
    if not any(' on /Volumes/media (smbfs' in line for line in mounts.splitlines()):
        raise RuntimeError('NAS SMB mount absent; refusing local fallback')
    if not root.is_relative_to(mount):
        raise ValueError('output must remain under the NAS mount')
    # A listed SMB mount can be disconnected and hang inside filesystem syscalls.
    # Probe in a child process so a stale mount cannot silently trigger more work.
    probe = subprocess.Popen(['/usr/bin/stat', '-f', '%d', str(root)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        probe.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        probe.terminate()
        raise RuntimeError('NAS I/O unavailable after 8s; stop and retain NAS partials') from None
    if probe.returncode:
        raise RuntimeError('NAS filesystem probe failed')
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError('output must remain under the NAS mount without symlinks')
    return root


def token_config():
    token = os.environ.get('HF_TOKEN')
    if not token:
        path = Path(os.environ.get('HF_TOKEN_PATH', str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'token')))
        token = path.read_text().strip()
    if not token or any(c in token for c in '\r\n"\\'):
        raise ValueError('invalid token format')
    return 'header = "Authorization: Bearer ' + token + '"\n'


class Source:
    def __init__(self, root, repo, revision, repo_type='dataset'):
        self.root = nas_root(root)
        if len(repo.split('/')) != 2 or not all(x.replace('-','').replace('_','').isalnum() for x in repo.split('/')):
            raise ValueError('invalid repository')
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('fixed revision required')
        if repo_type not in {'dataset','model'}:
            raise ValueError('unsupported repository type')
        self.repo, self.revision = repo, revision
        self.repo_type = repo_type

    def fetch(self, name, destination, byte_range=None):
        nas_root(self.root)
        if any(x in name for x in ['..', '?', '#', '\\']) or name.startswith('/'):
            raise ValueError('invalid source path')
        dest = safe_child(self.root, destination)
        receipt = dest.with_suffix(dest.suffix + '.receipt.json')
        identity = dict(repo=self.repo, revision=self.revision, source_path=name,
                        byte_range=list(byte_range) if byte_range else None)
        if self.repo_type == 'model':
            identity['repo_type'] = 'model'
        if dest.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            if all(saved.get(k) == v for k,v in identity.items()) and digest(dest) == saved['sha256']:
                return dest
            raise ValueError('existing asset or source identity mismatch')
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Short completed network transfers are persisted serially to the NAS.
        # This separates network stalls from SMB stalls, bounds RAM use, and keeps
        # a verified checkpoint after each chunk instead of restarting a 4MiB range.
        temp = dest.with_suffix(dest.suffix + '.segments.partial')
        progress = dest.with_suffix(dest.suffix + '.progress.json')
        prefix = 'datasets/' if self.repo_type == 'dataset' else ''
        url = f'https://huggingface.co/{prefix}{self.repo}/resolve/{self.revision}/{name}'
        done=0
        hasher=hashlib.sha256()
        resumed=False
        if temp.exists() and progress.exists():
            saved=json.loads(progress.read_text())
            if any(saved.get(k)!=v for k,v in identity.items()):
                raise ValueError('partial source identity mismatch')
            done=saved['bytes']
            if type(done) is not int or done<0:raise ValueError('invalid committed prefix length')
            resumed=True
            # A terminated write may leave uncommitted trailing bytes. Verify the
            # committed prefix before trimming only this task's partial file.
            remaining=done
            with temp.open('rb') as f:
                while remaining:
                    block=f.read(min(1024*1024,remaining))
                    if not block:break
                    hasher.update(block);remaining-=len(block)
            if remaining or hasher.hexdigest()!=saved['sha256']:
                raise ValueError('partial committed prefix corrupt')
            with temp.open('r+b') as f:f.truncate(done)
        elif temp.exists():
            raise ValueError('orphan partial needs inspection before reuse')
        total=byte_range[1]-byte_range[0]+1 if byte_range else (done if resumed else None)
        if byte_range and (byte_range[0]<0 or total<=0):raise ValueError('invalid byte range')
        if done<0 or (total is not None and done>total):raise ValueError('invalid committed prefix length')
        while total is None or done<total:
            nas_root(self.root)
            args=['curl','--http1.1','--config','-','-fLsS','--retry','1',
                  '--connect-timeout','10','--max-time','45','--write-out','%{http_code}']
            expected=None;request_url=url
            if byte_range:
                begin=byte_range[0]+done;end=min(begin+512*1024-1,byte_range[1]);expected=end-begin+1
                args+=['--range',f'{begin}-{end}','--max-filesize',str(expected)]
                request_url+=f'?range={begin}-{end}'
            else:args+=['--max-filesize',str(64*1024*1024)]
            result=subprocess.run(args+[request_url],input=token_config().encode(),capture_output=True)
            status=result.stdout[-3:].decode(errors='replace');payload=result.stdout[:-3]
            if result.returncode or status!=('206' if byte_range else '200'):
                raise RuntimeError(f'source fetch failed: curl={result.returncode}, HTTP={status[:3]}')
            if expected is not None and len(payload)!=expected:raise ValueError('incomplete network range')
            nas_root(self.root)
            with temp.open('ab') as f:
                f.write(payload);f.flush();os.fsync(f.fileno())
            done+=len(payload)
            hasher.update(payload)
            write_json(progress,dict(identity,bytes=done,sha256=hasher.hexdigest()))
            if total is None:break
        nas_root(self.root)
        sha = digest(temp)
        if sha!=hasher.hexdigest():raise ValueError('NAS readback differs from committed download bytes')
        os.replace(temp, dest)
        write_json(receipt, dict(identity, sha256=sha, bytes=dest.stat().st_size, acquired_unix=time.time()))
        return dest


def tar_prefix_inventory(path, output_root):
    """Read complete entries from a partial TAR; never extract paths supplied by it."""
    root = nas_root(output_root)
    entries, annotations = [], []
    with tarfile.open(path, mode='r:') as archive:
        try:
            for member in archive:
                if member.offset_data + member.size > Path(path).stat().st_size:
                    break
                if not member.isfile():
                    continue
                entries.append(dict(name=member.name, offset=member.offset_data, size=member.size))
                if member.name.endswith('.json'):
                    annotations.append(dict(member=member.name, data=json.load(archive.extractfile(member))))
        except tarfile.ReadError:
            pass  # Expected only at the explicitly bounded end of the prefix.
    write_json(root/'manifests/Emotiontalk-prefix-index.json', entries)
    write_json(root/'manifests/Emotiontalk-audio-annotations.json', annotations)
    return entries, annotations


class SparseTar:
    """Locate members in a sorted uncompressed TAR, checking every header checksum.

    Binary search is an optimization only. A successful result must match the exact
    requested member; archives with other ordering fail instead of returning data.
    """
    def __init__(self, source, name, size, prefix):
        self.source, self.name, self.size = source, name, size
        self.prefix = Path(prefix)
        self.index_path = source.root/'manifests/Emotiontalk-sparse-index.json'
        self.entries = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def headers_after(self, offset):
        start = (offset//512)*512
        def scan(block, base):
            found=[]
            for i in range(0,len(block)-511,512):
                header=block[i:i+512]
                if not header.startswith(b'Audio/'):
                    continue
                try:
                    member=tarfile.TarInfo.frombuf(header,'utf-8','strict')
                except (tarfile.HeaderError, UnicodeDecodeError, ValueError):
                    continue
                if member.isfile() and member.name.endswith('.wav'):
                    entry=dict(name=member.name, offset=base+i+512, size=member.size)
                    self.entries[member.name]=entry;found.append(entry)
            if found:
                write_json(self.index_path,self.entries)
            return found
        # 4MiB windows generally include at least one complete WAV header.
        for attempt in range(8):
            end = min(start + 4*1024*1024-1, self.size-1)
            if start >= self.size:
                return []
            if end < self.prefix.stat().st_size:
                with self.prefix.open('rb') as f:
                    f.seek(start); block=f.read(end-start+1)
            else:
                # Binary search can move by only a few TAR blocks. Reuse the
                # overlapping suffix of completed ranges before fetching an
                # almost identical 4MiB window. A checksum-valid exact TAR
                # header is still required; a cache miss uses the normal path.
                cached_ranges=[]
                for cached in (self.source.root/'cache/tar-ranges').glob('*.bin'):
                    base,finish=map(int,cached.stem.split('-'))
                    if base<=start and finish>=start+511:
                        cached_ranges.append((finish,base,cached))
                for finish,base,cached in sorted(cached_ranges,reverse=True):
                    with cached.open('rb') as f:
                        f.seek(start-base);cached_block=f.read(min(end,finish)-start+1)
                    found=scan(cached_block,start)
                    if found:return found
                path=self.source.fetch(self.name, f'cache/tar-ranges/{start}-{end}.bin', (start,end))
                block=path.read_bytes()
            found=scan(block,start)
            if found:
                return found
            start=end+1
        raise RuntimeError('no verified TAR header in bounded search window')

    def locate(self, target):
        if target in self.entries:
            return self.entries[target]
        low,high=0,self.size
        for _ in range(40):
            # Already observed neighbors narrow searches for nearby utterances.
            lower=[v['offset']+((v['size']+511)//512)*512 for k,v in self.entries.items() if k<target]
            upper=[v['offset']-512 for k,v in self.entries.items() if k>target]
            if lower:low=max(low,max(lower))
            if upper:high=min(high,min(upper))
            if target in self.entries:return self.entries[target]
            if low>high:break
            mid=(low+high)//1024*512
            found=self.headers_after(mid)
            if target in self.entries:return self.entries[target]
            if not found:high=mid-512;continue
            if found[0]['name']>target:high=mid-512
            else:low=found[0]['offset']+((found[0]['size']+511)//512)*512
        raise RuntimeError(f'exact archive member not found: {target}')

    def extract(self, target):
        entry=self.locate(target)
        # Actual member bytes only; never tar.extract() a supplied filesystem path.
        relative=PurePosixPath(target).relative_to('Audio/wav')
        destination='raw/Emotiontalk/wav/'+str(relative)
        start,end=entry['offset'],entry['offset']+entry['size']-1
        # Reuse payload bytes already acquired while locating TAR headers.
        candidates=[(0,self.prefix)]
        for cached in (self.source.root/'cache/tar-ranges').glob('*.bin'):
            candidates.append((int(cached.stem.split('-')[0]),cached))
        for base,cached in candidates:
            if base<=start and end<base+cached.stat().st_size:
                with cached.open('rb') as f:f.seek(start-base);payload=f.read(entry['size'])
                nas_root(self.source.root)
                path=safe_child(self.source.root,destination);path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(payload)
                write_json(path.with_suffix('.wav.receipt.json'),dict(repo=self.source.repo,
                    revision=self.source.revision,source_path=self.name,byte_range=[start,end],
                    bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                    reused_range=str(cached.relative_to(self.source.root))))
                return path
        path=self.source.fetch(self.name,destination,(start,end))
        return path
