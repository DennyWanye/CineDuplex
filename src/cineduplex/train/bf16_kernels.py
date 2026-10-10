"""BF16 GEMV, FP32 accumulation; no compressed/quantized weights."""
import torch
import triton
import triton.language as tl

@triton.jit
def gemv_kernel(X,W,B,Y,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,BLOCK)
    xv=tl.load(X+k,k<K,other=0).to(tl.float32)
    wv=tl.load(W+row*K+k,k<K,other=0).to(tl.float32)
    out=tl.sum(xv*wv,0)
    if HAS_BIAS:out+=tl.load(B+row).to(tl.float32)
    tl.store(Y+row,out)

def linear(x,weight,bias,output_dtype=None):
    n,k=weight.shape
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=output_dtype or x.dtype)
    gemv_kernel[(n,)](x,weight,bias if bias is not None else weight,y,n,k,bias is not None,
        triton.next_power_of_2(k),num_warps=8 if k>8192 else 4)
    return y

@triton.jit
def rms_kernel(X,W,Y,K:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    k=tl.arange(0,BLOCK)
    value=tl.load(X+k,k<K,other=0).to(tl.float32)
    variance=tl.sum(value*value,0)/K
    # Qwen casts the normalized values back before multiplying its weight.
    normalized=(value*tl.rsqrt(variance+EPS)).to(Y.dtype.element_ty)
    weight=tl.load(W+k,k<K,other=0)
    tl.store(Y+k,normalized*weight,k<K)

def rms(x,weight,eps):
    y=torch.empty_like(x)
    rms_kernel[(1,)](x,weight,y,x.shape[-1],eps,triton.next_power_of_2(x.shape[-1]))
    return y

@triton.jit
def gated_kernel(X,G,U,Y,N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);k=tl.arange(0,BLOCK)
    value=tl.load(X+k,k<K,other=0).to(tl.float32)
    gate=tl.sum(value*tl.load(G+row*K+k,k<K,other=0).to(tl.float32),0).to(Y.dtype.element_ty).to(tl.float32)
    up=tl.sum(value*tl.load(U+row*K+k,k<K,other=0).to(tl.float32),0).to(Y.dtype.element_ty)
    activated=(gate*tl.sigmoid(gate)).to(Y.dtype.element_ty)
    tl.store(Y+row,activated*up)

def gated(x,gate,up):
    n,k=gate.shape
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    gated_kernel[(n,)](x,gate,up,y,n,k,triton.next_power_of_2(k))
    return y
