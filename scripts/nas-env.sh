# Source from the repository checkout: source scripts/nas-env.sh
# Sets task caches only; never changes HOME or CODEX_HOME.
export CINEDUPLEX_ROOT="${CINEDUPLEX_ROOT:-/Volumes/media/CineDuplex}"
export CINEDUPLEX_PROJECT="$(pwd)"
if [ ! -f "$CINEDUPLEX_PROJECT/src/cineduplex/data/nas_source.py" ]; then
  echo 'Run from the CineDuplex repository root.' >&2
  return 1
fi
python3 -B - <<'PY' || return 1
import os,subprocess
from pathlib import Path
p=Path(os.environ['CINEDUPLEX_ROOT']).absolute()
m=Path('/Volumes/media')
if not any(' on /Volumes/media (smbfs' in line for line in subprocess.check_output(['mount'],text=True).splitlines()):
    raise SystemExit('Real SMB mount /Volumes/media is required; no local fallback.')
if not p.is_relative_to(m) or p==m or any(x.is_symlink() for x in [p,*p.parents]):
    raise SystemExit('An explicit non-symlink task directory on NAS is required.')
probe=subprocess.Popen(['/usr/bin/stat','-f','%d',str(m)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:probe.communicate(timeout=8)
except subprocess.TimeoutExpired:
    probe.kill();raise SystemExit('NAS metadata access timed out.')
if probe.returncode:raise SystemExit('NAS mount is not readable.')
for name in ['tmp','cache/uv-codec-v2','cache/numba','cache/huggingface','cache/datasets','cache/torch','cache/torchinductor']:
    d=p/name
    if any(x.is_symlink() for x in [d,*d.parents]):raise SystemExit('Cache path cannot redirect away from NAS.')
    d.mkdir(parents=True,exist_ok=True)
PY
export PYTHONPATH="$CINEDUPLEX_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TMPDIR="$CINEDUPLEX_ROOT/tmp"
export XDG_CACHE_HOME="$CINEDUPLEX_ROOT/cache"
export UV_CACHE_DIR="$CINEDUPLEX_ROOT/cache/uv-codec-v2"
export UV_PYTHON_DOWNLOADS=never
export HF_HOME="$CINEDUPLEX_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$CINEDUPLEX_ROOT/cache/datasets"
export NUMBA_CACHE_DIR="$CINEDUPLEX_ROOT/cache/numba"
export TORCH_HOME="$CINEDUPLEX_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$CINEDUPLEX_ROOT/cache/torchinductor"
export CINE_PY="$CINEDUPLEX_ROOT/cache/codec-env-linked/bin/python"
