#!/usr/bin/env python3
"""Close the exact orphaned voice group through its NAS STOP protocol.

Only storage transport is reattached. No GPU process is launched or killed.
Retirement uses the existing Gateway signed safe-state verifier and lock CLI.
All four original PIDs must match before storage can expose the stop marker.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

from cineduplex.train.nas_mount import DirectMount
from cineduplex.train.nas_transport import NAS


def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--prefix',required=True)
    p.add_argument('--original-pids',required=True);p.add_argument('--retire-only',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());n=NAS(cfg['nas']);lease=cfg['cluster']['lease']
    expected=json.loads(a.original_pids)
    if set(expected)!={str(i) for i in range(4)}:raise ValueError('four original identities required')
    root=Path('/Volumes/media/CineDuplex')/a.prefix
    if not Path('/Volumes/media').is_mount() or root.parent.name!='output':raise ValueError('existing NAS root required')
    def remote(node,command):
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes',
            node['ssh_alias'],command],capture_output=True,text=True,timeout=15)
        if r.returncode:raise RuntimeError('read-only node check failed: '+node['hostname'])
        return r.stdout.splitlines()
    def authority(args,code=None):
        cmd=['docker','exec']+(['-i'] if code else [])+[lease['container'],'python']+args
        return n.run(shlex.join(cmd),data=code,timeout=30)
    with (root/'recovery-stop.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=json.loads(n.read(a.prefix+'/controller.json',timeout=15))
        if identity['generation']!=lease['generation']:raise RuntimeError('original generation differs')
        try: __import__('os').kill(identity['pid'],0)
        except ProcessLookupError: pass
        else:raise RuntimeError('original controller still exists; do not duplicate ownership')
        status=json.loads(authority([lease['cli'],'--root',lease['root'],'status']))
        if (status['state'],status['owner'],status['lease_generation'])!=('STALE',lease['owner'],lease['generation']):
            raise RuntimeError('only the exact expired original admission can be reconciled')
        # The old capability is verified in the authority container, never read
        # or copied by this process. This does not authorize new GPU work.
        check="import importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('lease_cli',"+repr(lease['cli'])+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.validate_owner(m.read_metadata(Path("+repr(lease['root'])+")),"+repr(lease['owner'])+",Path("+repr(lease['token_file'])+"));print('ORIGINAL_CAPABILITY_VALID')"
        authority(['-c',check])
        for node in cfg['nodes']:
            rows=remote(node,'hostname; whoami; nvidia-smi --query-compute-apps=pid --format=csv,noheader')
            wanted=[node['hostname'],node['user']]+([] if a.retire_only else [str(expected[str(node['rank'])])])
            if rows!=wanted:
                raise RuntimeError('GPU ownership changed: '+node['hostname'])
            code="from pathlib import Path;import os;p=Path.home()/'cineduplex-work/nas-training';assert not p.is_symlink() and not os.path.ismount(p) and not list(p.iterdir());print('EMPTY_DISCONNECTED_MOUNT')"
            if remote(node,'python3 -B -c '+shlex.quote(code))!=['EMPTY_DISCONNECTED_MOUNT']:
                raise RuntimeError('preserve unexpected disconnected files')
        if not a.retire_only:
            n.put(a.prefix+'/NATIVE-RECOVERY-BEFORE.json',json.dumps({'at':time.time(),'original_pids':expected,'admission':status,'reason':'controller absent, original four GPU processes alive, NAS mounts absent'}).encode())
        elif not json.loads(n.read(a.prefix+'/NATIVE-RECOVERY-ALL-STOPPED.json',timeout=15))['all_four_gpu_processes_empty']:
            raise RuntimeError('preserved original stop proof absent')
        if n.stat(a.prefix+'/STOP.json') is None:
            n.put(a.prefix+'/STOP.json',json.dumps({'at':time.time(),'reason':'expired_original_controller_storage_loss','recovery':'cooperative original STOP, no SSH GPU kill'}).encode())
        mounts=[]
        try:
            if not a.retire_only:
                for node in cfg['nodes']:
                    mounts.append(DirectMount(n,node,cfg['nas_sftp'],cfg['nas_python'],mount_name='nas-training'))
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                rows=[remote(node,'hostname; whoami; nvidia-smi --query-compute-apps=pid --format=csv,noheader') for node in cfg['nodes']]
                if all(row==[node['hostname'],node['user']] for row,node in zip(rows,cfg['nodes'])):break
                time.sleep(2)
            else:raise RuntimeError('original STOP has not released GPUs; no kill or takeover')
            if not a.retire_only:
                n.put(a.prefix+'/NATIVE-RECOVERY-ALL-STOPPED.json',json.dumps({'at':time.time(),'original_pids':expected,'all_four_gpu_processes_empty':True,'stop_method':'original NAS marker and state machine','microphone':False,'speaker':False}).encode())
        finally:
            for mount in reversed(mounts):mount.close()
        # Gateway's own trusted verifier observes its catalog/attestation and
        # signs the exact expired capability. Extra four-node GPU-empty proof
        # above covers the cooperative model, absent from the model catalog.
        code="""import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,'/workspace/DGXGateway')
from dgx_gateway.config import load_config
from dgx_gateway.recovery import _atomic_private_json,_verified_safe_proof
c=json.load(sys.stdin);cfg=load_config(Path('/run/dgx-gateway/gateway.toml'))
from dgx_gateway.ssh import FixedSSH
FixedSSH.default_config_file=cfg.ssh_config_file
lease=c['lease'];status=json.loads(subprocess.check_output(['python',lease['cli'],'--root',lease['root'],'status']))
assert status['state']=='STALE' and status['owner']==lease['owner'] and status['lease_generation']==lease['generation']
op='cineduplex-native-recovery-20261014'
request=cfg.state_dir/('recovery-request-'+op+'.json')
_atomic_private_json(request,{'schema':'dgx-gateway-recovery-request-v1','operation_id':op})
proof=_verified_safe_proof(cfg,op,status,request)
r=subprocess.run(['python',lease['cli'],'--root',lease['root'],'release','--owner',lease['owner'],'--token-file',lease['token_file'],'--operation-id',op,'--stale-retirement-proof',str(proof),'--stale-retirement-verifier-key',str(cfg.authority_credential)],capture_output=True,text=True,timeout=30)
if r.returncode:raise RuntimeError('Gateway retirement failed: '+r.stderr[-300:])
print(r.stdout)
"""
        out=n.run(shlex.join(['docker','exec','-i',lease['container'],'python','-c',code]),data=json.dumps({'lease':lease}).encode(),timeout=55)
        record=json.loads(out)
        if record['state']!='UNLOCKED':raise RuntimeError('Gateway retirement is not complete')
        print(n.put(a.prefix+'/NATIVE-RECOVERY-RETIREMENT.json',json.dumps(record).encode()))
        print('ORIGINAL_GROUP_STOPPED_AND_GATEWAY_CAPABILITY_RETIRED',flush=True)


if __name__=='__main__':main()
