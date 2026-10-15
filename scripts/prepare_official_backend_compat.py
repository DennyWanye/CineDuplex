#!/usr/bin/env python3
"""Apply narrowly scoped GB10 build compatibility to an isolated NAS copy.

This does not install a model, initialize CUDA, change sampling, or modify the
official source snapshot. Compiled kernels still require actual BF16 checks.
"""
import argparse
import hashlib
import json
import shlex
from pathlib import Path

from cineduplex.train.nas_transport import NAS


PATCH = r'''
import hashlib, json, pathlib, re, time
root = pathlib.Path(ROOT)
assert root.is_dir() and (root/'source/.git').is_dir()
changes = []
def replace(relative, old, new, count):
    path = root/relative
    text = path.read_text()
    if text.count(old) != count:
        raise ValueError('unexpected official source: '+relative)
    before = hashlib.sha256(text.encode()).hexdigest()
    result = text.replace(old, new)
    path.write_text(result)
    changes.append({'file':relative,'before':before,
                    'after':hashlib.sha256(result.encode()).hexdigest(),
                    'replacements':count})
replace('vllm-arm64/CMakeLists.txt',
        'set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")',
        'set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;12.1")',1)
replace('flash-attention/CMakeLists.txt',
        'set(CUDA_SUPPORTED_ARCHS "8.0;8.6;8.9;9.0")',
        'set(CUDA_SUPPORTED_ARCHS "8.0;8.6;8.9;9.0;12.1")',1)
replace('flash-attention/csrc/flash_attn/flash_api.cpp',
        'TORCH_CHECK(is_sm90 || is_sm8x,',
        'TORCH_CHECK(is_sm90 || is_sm8x || (dprops->major == 12 && dprops->minor == 1),',6)
record = {'at':time.time(),'official_revision':'7bb1068c59b578e1f8e65fb762cd26708c14cc6d',
          'changes':changes,'scope':'GB10 architecture admission only',
          'dtype':'bfloat16','quantization':'none',
          'model_weights_changed':False,'sampling_changed':False,
          'compiled':False,'gpu_tested':False,'quality_accepted':False}
dest = root/'GB10-COMPATIBILITY-PATCH.json'
with dest.open('x') as f: json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps(record))
'''

SPECIALIZE = r"""
import hashlib,json,pathlib,re,time
root=pathlib.Path(ROOT)
assert (root/'GB10-COMPATIBILITY-PATCH.json').is_file()
changes=[]
def write(relative,content):
    p=root/relative; before=p.read_bytes()
    p.write_text(content)
    changes.append({'file':relative,'before':hashlib.sha256(before).hexdigest(),
                    'after':hashlib.sha256(content.encode()).hexdigest()})
p=root/'vllm-arm64/CMakeLists.txt'; text=p.read_text()
anchor='message(STATUS "Enabling C extension.")'
assert text.count(anchor)==1
text=text.replace(anchor,
    'list(FILTER VLLM_EXT_SRC EXCLUDE REGEX "quantization/|mamba/|permute_cols|layernorm_quant_kernels")\n'+anchor)
write('vllm-arm64/CMakeLists.txt',text)
# Keep the upstream schemas/implementations for the actual dense Qwen path.
# Unsupported quantization and Mamba operators are not registered or stubbed.
p=root/'vllm-arm64/csrc/torch_bindings.cpp'; text=p.read_text()
start=text.index('TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {')
end=text.index('TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops)')
allowed={'weak_ref_tensor','paged_attention_v1','paged_attention_v2',
    'silu_and_mul','gelu_and_mul','gelu_tanh_and_mul','fatrelu_and_mul',
    'gelu_new','gelu_fast','gelu_quick','advance_step_flashattn',
    'advance_step_flashinfer','rms_norm','fused_add_rms_norm',
    'rotary_embedding','batched_rotary_embedding'}
statements=[]; found=set()
for match in re.finditer(r'ops\.def\(\s*"([a-zA-Z0-9_]+)[\s\S]*?;\s*ops\.impl\([\s\S]*?;',text[start:end]):
    if match[1] in allowed:
        statements.append(match[0]); found.add(match[1])
assert found==allowed, sorted(allowed-found)
write('vllm-arm64/csrc/torch_bindings.cpp',text[:start]+
      'TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {\n'+
      '\n'.join(statements)+'\n}\n\n'+text[end:])
# The fixed checkpoint has head dimension 128 on every vLLM branch.
# Build that exact BF16 specialization; reject other requests explicitly.
p=root/'flash-attention/CMakeLists.txt'; text=p.read_text()
old='file(GLOB FLASH_ATTN_GEN_SRCS "csrc/flash_attn/src/flash_fwd_*.cu")'
assert text.count(old)==1
write('flash-attention/CMakeLists.txt',text.replace(old,
    'file(GLOB FLASH_ATTN_GEN_SRCS "csrc/flash_attn/src/flash_fwd_hdim128_bf16*_sm80.cu" '
    '"csrc/flash_attn/src/flash_fwd_split_hdim128_bf16*_sm80.cu")'))
p=root/'flash-attention/csrc/flash_attn/src/static_switch.h'; text=p.read_text()
start=text.index('#define FP16_SWITCH('); end=text.index('#define HEADDIM_SWITCH(',start)
tail=text.find('\n#define ',end+1)
if tail<0:tail=len(text)
macros='''#define FP16_SWITCH(COND, ...) \\
  [&] { \\
    TORCH_CHECK(!(COND), "Lychee GB10 reference requires BF16"); \\
    using elem_type = cutlass::bfloat16_t; \\
    return __VA_ARGS__(); \\
  }()

#define HEADDIM_SWITCH(HEADDIM, ...) \\
  [&] { \\
    TORCH_CHECK((HEADDIM) == 128, "Lychee GB10 reference requires head_dim=128"); \\
    constexpr static int kHeadDim = 128; \\
    return __VA_ARGS__(); \\
  }()
'''
write('flash-attention/csrc/flash_attn/src/static_switch.h',text[:start]+macros+text[tail:])
record={'at':time.time(),'scope':'dense Qwen BF16 head_dim128 build specialization',
        'changes':changes,'exported_dense_ops':sorted(found),
        'quantized_model_supported':False,'sampling_changed':False,
        'model_weights_changed':False,'gpu_tested':False,'quality_accepted':False}
with (root/'BF16-BUILD-SPECIALIZATION.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
"""

CUDA13_API = r'''
import hashlib,json,pathlib,time
root=pathlib.Path(ROOT)
p=root/'vllm-arm64/csrc/layernorm_kernels.cu'
before=p.read_bytes(); text=before.decode()
assert text.count('cub::Sum{}')==3
text='#include <cuda/std/functional>\n'+text.replace('cub::Sum{}','cuda::std::plus<>{}')
p.write_text(text)
record={'at':time.time(),'file':str(p.relative_to(root)),
        'before':hashlib.sha256(before).hexdigest(),
        'after':hashlib.sha256(text.encode()).hexdigest(),
        'reason':'CUDA13 CCCL3 removed cub::Sum; use documented cuda::std::plus',
        'reference':'https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html',
        'changed_reductions':3,'sampling_changed':False,'gpu_tested':False}
with (root/'CUDA13-API-PATCH.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--prefix', default='output/official-backend-20261013-v1')
    parser.add_argument('--specialize-bf16', action='store_true')
    parser.add_argument('--cuda13-api', action='store_true')
    args = parser.parse_args()
    nas = NAS(json.loads(Path(args.config).read_text())['nas'])
    if args.specialize_bf16 and args.cuda13_api:
        parser.error('one compatibility change per invocation')
    patch = CUDA13_API if args.cuda13_api else SPECIALIZE if args.specialize_bf16 else PATCH
    program = 'ROOT='+repr(nas.path(args.prefix))+'\n'+patch
    # The existing CPU container has no GPU runtime; all changes remain on NAS.
    result = nas.run('docker exec -i --user "$(id -u family):$(id -g family)" '
                     'cineduplex-web python3 -B -', data=program.encode(), timeout=20)
    print(result.decode())


if __name__ == '__main__':
    main()
