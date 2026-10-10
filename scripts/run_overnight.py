#!/usr/bin/env python3
"""One new run, with durable NAS status and no duplicate controller launch."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path('/Volumes/media/CineDuplex')


def main():
    config_path=Path(sys.argv[1]).resolve();cfg=json.loads(config_path.read_text())
    if not Path('/Volumes/media').is_mount():raise RuntimeError('real NAS mount required')
    run=ROOT/cfg['run_prefix'];run.mkdir(exist_ok=True)
    with (run/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'controller-identity.json').exists():raise RuntimeError('run already launched; inspect original identity')
        (run/'controller-identity.json').write_text(json.dumps({'pid':os.getpid(),'started_at':time.time(),
            'config':str(config_path),'run_prefix':cfg['run_prefix']},indent=2))
        env={**os.environ,'PYTHONPATH':str(PROJECT/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
        def status(phase,**extra):
            temp=run/'status.pending.json'
            temp.write_text(json.dumps({'phase':phase,'updated_at':time.time(),'run_prefix':cfg['run_prefix'],
                'controller_pid':os.getpid(),'train_rows':cfg['input_files']['train']['rows'],
                'eval_rows':cfg['input_files']['eval']['rows'],'quality_approved':False,**extra},indent=2))
            temp.replace(run/'status.json')
        for action in ('prepare','train'):
            status('PREPARING' if action=='prepare' else 'LOADING_MODEL')
            with (run/(action+'.log')).open('x') as log:
                p=subprocess.Popen([sys.executable,'-m','cineduplex.train.four_controller',action,str(config_path)],
                    cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                status('PREPARING' if action=='prepare' else 'LOADING_MODEL',child_pid=p.pid)
                rc=p.wait()
            if rc:
                status('FAILED',stage=action,returncode=rc,log=action+'.log')
                return rc
        status('TRAINING_COMPLETE',model_ready=False,web_inference='PENDING')
    return 0


if __name__=='__main__':sys.exit(main())
