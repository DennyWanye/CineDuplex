#!/usr/bin/env python3
"""Release this run's Gateway ownership only after its GPU workers have exited."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from cineduplex.cluster.guard import guard


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text())
    root=Path('/Volumes/media/CineDuplex')/cfg['run_prefix']
    identity=json.loads((root/'controller-identity.json').read_text())
    pid=identity['pid']
    while True:
        try:os.kill(pid,0)
        except ProcessLookupError:break
        time.sleep(20)
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='TRAINING_COMPLETE':
        (root/'cleanup-pending.json').write_text(json.dumps({'reason':'run requires recovery','status':status}))
        return 1
    guard(cfg['cluster'])
    for node in cfg['nodes']:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
            '-o','StrictHostKeyChecking=yes',node['ssh_alias'],
            'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],
            capture_output=True,text=True,timeout=20)
        if r.returncode or r.stdout.strip():raise RuntimeError('GPU still occupied; retain ownership')
    lease=cfg['cluster']['lease']
    remote=['ssh','-i',lease['ssh_identity'],'-o','BatchMode=yes','-o','ConnectTimeout=8',
        '-o','StrictHostKeyChecking=yes',lease['ssh_host']]
    cmd=['docker','exec',lease['container'],'python',lease['cli'],'--root',lease['root'],
         'release','--owner',lease['owner'],'--token-file',lease['token_file']]
    released=subprocess.run(remote+[shlex.join(cmd)],capture_output=True,text=True,timeout=20)
    if released.returncode:raise RuntimeError('Gateway lease release failed')
    cmd=['docker','exec','-e','PYTHONPATH=/workspace/DGXGateway',lease['container'],
        'python','-m','dgx_gateway.maintenance','set-mutation-fence',
        '--config','/run/dgx-gateway/gateway.toml','--enabled','false']
    opened=subprocess.run(remote+[shlex.join(cmd)],capture_output=True,text=True,timeout=20)
    if opened.returncode:raise RuntimeError('Gateway fence remains; inspect maintenance result')
    (root/'RESOURCE-RELEASED.json').write_text(json.dumps({'at':time.time(),
        'lease':json.loads(released.stdout),'admission':json.loads(opened.stdout),
        'all_four_gpu_process_lists_empty':True},indent=2))
    return 0


if __name__=='__main__':sys.exit(main())
