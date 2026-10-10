"""NAS SSH storage and a loopback-only bridge. No DGX filesystem staging.

The deployment supplies identities/paths; secrets never enter the repository.
All writes are immutable, hash-checked and atomically renamed on the NAS.
"""
from __future__ import annotations
import hashlib
import json
import secrets
import shlex
import subprocess
from pathlib import PurePosixPath


class NAS:
    def __init__(self, config):
        self.config = config
        self.ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    '-o', 'StrictHostKeyChecking=yes', '-i', config['identity'], config['host']]

    def path(self, relative):
        p = PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts or not p.parts:
            raise ValueError('unsafe NAS relative path')
        return self.config['root'].rstrip('/') + '/' + str(p)

    def guarded(self, command):
        root = shlex.quote(self.config['root'])
        uuid = shlex.quote(self.config['uuid'])
        # The existing marker and filesystem UUID distinguish the real NAS.
        return ('set -eu; test "$(findmnt -n -o UUID -T ' + root + ')" = ' + uuid
                + '; test -f /srv/nas/.nas-online; ' + command)

    def run(self, command, *, data=None, timeout=60):
        r = subprocess.run(self.ssh + [self.guarded(command)], input=data,
                           capture_output=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError('NAS operation failed: ' + r.stderr.decode(errors='replace')[-400:])
        return r.stdout

    def read(self, relative, timeout=600):
        return self.run('cat -- ' + shlex.quote(self.path(relative)), timeout=timeout)

    def stat(self, relative):
        q = shlex.quote(self.path(relative))
        out = self.run('if test -f '+q+'; then stat -c %s '+q+'; else echo absent; fi')
        return None if out.strip() == b'absent' else int(out)

    def writer_command(self, relative, expected):
        dest = self.path(relative)
        parent = str(PurePosixPath(dest).parent)
        # unique staging file belongs only to this call; never clobber a final file
        tmp = dest + '.pending-' + secrets.token_hex(8)
        q, t = shlex.quote(dest), shlex.quote(tmp)
        script = ('set -eu; umask 002; mkdir -p '+shlex.quote(parent)+'; '
                  'test ! -e '+q+'; test ! -L '+q+'; '
                  'trap '+shlex.quote('rm -f -- '+t)+' EXIT; '
                  'cat > '+t+'; test "$(sha256sum '+t+' | cut -d " " -f 1)" = '
                  +shlex.quote(expected)+'; sync -f '+t+'; '
                  'ln '+t+' '+q+'; rm '+t+'; sync -f '+shlex.quote(parent))
        return self.guarded('sudo -n -u family sh -c '+shlex.quote(script))

    def put(self, relative, payload):
        sha = hashlib.sha256(payload).hexdigest()
        r = subprocess.run(self.ssh + [self.writer_command(relative, sha)], input=payload,
                           capture_output=True, timeout=600)
        if r.returncode:
            raise RuntimeError('NAS write failed: '+r.stderr.decode(errors='replace')[-400:])
        return {'path': relative, 'sha256': sha, 'bytes': len(payload)}
