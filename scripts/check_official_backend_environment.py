#!/usr/bin/env python3
"""CPU-only import/ABI preflight for the isolated Lychee GB10 reference.

Run on the admitted DGX against the actual NAS environment. This never loads
weights, runs inference, or promotes the reference to production.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    nas = root.parents[1]
    if not nas.is_mount() or not str(root).startswith(str(nas)+'/output/'):
        raise RuntimeError('actual task NAS mount required')
    if Path(sys.prefix).resolve() != root/'env-arm64':
        raise RuntimeError('wrong isolated interpreter')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only preflight requires GPUs hidden')
    for key, relative in [('TMPDIR','tmp'), ('NUMBA_CACHE_DIR','tmp/numba'),
                          ('MPLCONFIGDIR','tmp/mpl'), ('HF_HOME','hf-cache')]:
        os.environ[key] = str(root/relative)
        (root/relative).mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['VLLM_USE_V1'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root/'source'),str(root/'source/third_party/Step-Audio2')]
    record = {'at':time.time(),'kind':'CPU import and native ABI',
              'weights_loaded':False,'gpu_inference_tested':False,
              'quality_accepted':False,'microphone':False,'speaker':False}
    try:
        import torch
        import vllm
        from lychee_fd.vllm_integration.engine import LycheeVLLMEngine
        from lychee_fd.runtime.vllm_generation import VLLMGenerationFramework
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        from vllm.platforms import current_platform
        # Cover imports reached during real worker startup, not only the public
        # engine class. Neither module starts a worker or initializes CUDA.
        from vllm.worker.worker import Worker
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        import blake3
        if current_platform.device_type != 'cuda':
            raise RuntimeError('official backend cannot identify the NVIDIA platform')
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
        resolved = Path(vllm.__file__).resolve()
        if not resolved.is_relative_to(root/'vllm-arm64'):
            raise RuntimeError('ordinary vLLM loaded instead of official patched tree')
        native = [root/'vllm-arm64/vllm/_C.abi3.so',
                  root/'vllm-arm64/vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so']
        record.update({'ok':True,'resolved_vllm':str(resolved),
                       'engine':LycheeVLLMEngine.__name__,
                       'framework':VLLMGenerationFramework.__name__,
                       'platform':current_platform.device_type,
                       'versions':{name:importlib.metadata.version(name) for name in
                                   ('torch','transformers','vllm','numpy','scipy')},
                       'native':[{'path':str(p),'bytes':p.stat().st_size,
                                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                                 for p in native]})
    except Exception as exc:
        record.update({'ok':False,'error':type(exc).__name__+': '+str(exc)[:500]})
        raise
    finally:
        destination = root/args.record
        if destination.parent != root or destination.exists():
            raise RuntimeError('preflight record must be a fresh root filename')
        with destination.open('x') as output:
            json.dump(record,output,ensure_ascii=False,indent=2)
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
