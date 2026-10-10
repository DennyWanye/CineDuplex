"""BF16 GEMV with measured row tiling and unchanged FP32 accumulation."""
import torch
import triton
import triton.language as tl


@triton.jit
def gemv(X,W,B,Y,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr,ROWS:tl.constexpr):
    rows=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    k=tl.arange(0,BLOCK)
    x=tl.load(X+k,k<K,other=0).to(tl.float32)
    w=tl.load(W+rows[:,None]*K+k[None,:],(rows[:,None]<N)&(k[None,:]<K),other=0).to(tl.float32)
    out=tl.sum(w*x[None,:],1)
    if HAS_BIAS:out+=tl.load(B+rows,rows<N,other=0).to(tl.float32)
    tl.store(Y+rows,out,rows<N)


tiling={}
benchmarks=[]
gated_selection={}
gated_benchmarks=[]


@triton.jit
def gated_gemv(G,U,W,Y,N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr,ROWS:tl.constexpr):
    rows=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    k=tl.arange(0,BLOCK)
    gate=tl.load(G+k,k<K,other=0).to(tl.float32)
    up=tl.load(U+k,k<K,other=0).to(tl.float32)
    # Match native BF16 silu output, then native BF16 multiplication.
    activated=(gate*tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    x=(activated*up).to(tl.bfloat16).to(tl.float32)
    w=tl.load(W+rows[:,None]*K+k[None,:],(rows[:,None]<N)&(k[None,:]<K),other=0).to(tl.float32)
    tl.store(Y+rows,tl.sum(w*x[None,:],1),rows<N)


def gated_linear(gate,up,weight):
    n,k=weight.shape
    output=torch.empty((*gate.shape[:-1],n),device=gate.device,dtype=torch.float32)
    # Same row tiling selected for the corresponding FP32 down projection.
    rows=tiling.get((n,k,False,str(torch.float32)),4)
    def fused():
        gated_gemv[(triton.cdiv(n,rows),)](gate,up,weight,output,n,k,
            triton.next_power_of_2(k),rows,num_warps=8 if k>4096 else 4)
        return output
    def native():
        value=torch.nn.functional.silu(gate)*up
        return linear(value,weight,None,output_dtype=torch.float32)
    key=(n,k)
    if key not in gated_selection:
        times={}
        for name,run in (('native',native),('fused',fused)):
            for _ in range(3):run()
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):run()
            end.record();end.synchronize()
            times[name]=begin.elapsed_time(end)/10
        gated_selection[key]=times['fused']<times['native']*.9
        gated_benchmarks.append({'shape':[n,k],'milliseconds':times,
            'selected':'fused' if gated_selection[key] else 'native'})
    return fused() if gated_selection[key] else native()


def linear(x,weight,bias,output_dtype=None):
    n,k=weight.shape
    y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=output_dtype or x.dtype)
    def run(rows):
        gemv[(triton.cdiv(n,rows),)](x,weight,bias if bias is not None else weight,y,n,k,bias is not None,
            triton.next_power_of_2(k),rows,num_warps=8 if k>4096 else 4)
    key=(n,k,bias is not None,str(y.dtype))
    if key not in tiling:
        times={}
        for rows in (4,8,16):
            try:
                for _ in range(3):run(rows)
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(10):run(rows)
                end.record();end.synchronize()
                times[rows]=begin.elapsed_time(end)/10
            except Exception:
                if rows==4:raise
                # A larger tile can exceed a kernel's register/shared-memory
                # limit. Retain the working four-row tile in that shape.
                continue
        tiling[key]=min(times,key=times.get)
        benchmarks.append({'shape':[n,k],'output_dtype':str(y.dtype),
                           'milliseconds':times,'selected_rows':tiling[key]})
    run(tiling[key])
    return y
