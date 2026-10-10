"""Use the existing deployment lease; never create a second authority."""
import json
import sqlite3
import subprocess
import shlex
from pathlib import Path

from cineduplex.contracts import Blocked


def guard(config, heartbeat=False):
    lease = config["lease"]
    if lease.get('authority') == 'k6':
        remote = ['ssh', '-i', lease['ssh_identity'], '-o', 'BatchMode=yes',
                  '-o', 'ConnectTimeout=8', '-o', 'StrictHostKeyChecking=yes', lease['ssh_host']]
        args = ['docker', 'exec', lease['container'], 'python', lease['cli'], '--root',
                lease['root'], 'heartbeat' if heartbeat else 'guard', '--owner',
                lease['owner'], '--token-file', lease['token_file']]
        r = subprocess.run(remote + [shlex.join(args)], capture_output=True, text=True, timeout=20)
        if r.returncode:
            raise Blocked('K6 Gateway lease is unavailable; do not launch GPU work')
        info = json.loads(r.stdout)
        if info.get('lease_generation') != lease['generation']:
            raise Blocked('K6 lease generation changed')
        # Read the live authority's database, never the retired Mac database.
        query = ('import sqlite3; c=sqlite3.connect("file:'+lease['gateway_database']+'?mode=ro",uri=True); '
                 'r=c.execute("SELECT value FROM metadata WHERE key=\'mutation_fenced\'").fetchone(); '
                 'print(r[0] if r else "missing")')
        r = subprocess.run(remote + [shlex.join(['docker','exec',lease['container'],'python','-c',query])],
                           capture_output=True, text=True, timeout=20)
        if r.returncode or r.stdout.strip() != '1':
            raise Blocked('K6 Gateway admission is not fenced')
        return {**info, 'mutation_fenced': True, 'authority': 'k6'}
    cmd = [lease["cli"], "--root", lease["root"], "heartbeat" if heartbeat else "guard",
           "--owner", lease["owner"], "--token-file", lease["token_file"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if r.returncode:
        raise Blocked("existing DGX lease is unavailable; do not launch GPU work")
    info = json.loads(r.stdout)
    if info.get("lease_generation") != lease["generation"]:
        raise Blocked("lease generation changed")
    uri = Path(lease["gateway_database"]).absolute().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='mutation_fenced'").fetchone()
    if row is None or row[0] != "1":
        raise Blocked("Gateway mutation admission is not fenced")
    return {**info, "mutation_fenced": True, "scope": "Gateway/cooperative deployment admission"}
