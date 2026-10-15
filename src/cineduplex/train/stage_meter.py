"""Low-overhead stage timing for continuous audio inference."""
import time
import gc
import resource
import torch

class StageMeter:
    """Record CPU wall time every window; sample all GPU branches one in 25.

    Avoid injecting four CUDA events per native token into the realtime path.
    Unsampled windows explicitly contain zero completed GPU measurements.
    """
    def __init__(self,model):
        self.handles=[];self.gc_start=None;self.gc_ms=0.;self.reset()
        def gc_event(phase,info):
            if phase=="start":self.gc_start=time.perf_counter()
            elif self.gc_start is not None:
                self.gc_ms+=(time.perf_counter()-self.gc_start)*1000;self.gc_start=None
        self.gc_callback=gc_event;gc.callbacks.append(gc_event)
        for name in ('encoder','adapter','model','stoken_model','control_model','merge_model','lm_head'):
            module=getattr(model,name,None)
            if module is None:continue
            def before(m,args,key=name):
                begin=None
                if self.enabled:
                    begin=torch.cuda.Event(enable_timing=True);begin.record()
                self.active[key]=(time.perf_counter(),begin)
            def after(m,args,output,key=name):
                cpu,begin=self.active.pop(key);end=None
                if begin is not None:
                    end=torch.cuda.Event(enable_timing=True);end.record()
                self.samples.append((key,(time.perf_counter()-cpu)*1000,begin,end))
            self.handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    def reset(self):
        self.samples=[];self.active={}
        self.host_usage=resource.getrusage(resource.RUSAGE_SELF)
        self.window=getattr(self,'window',-1)+1
        self.enabled=self.window%25==0
    def result(self):
        result={}
        for key,cpu,begin,end in self.samples:
            value=result.setdefault(key,{'calls':0,'cpu_dispatch_ms':0.,'gpu_ms':0.,'gpu_complete_calls':0})
            value['calls']+=1;value['cpu_dispatch_ms']+=cpu
            if end is not None and end.query():value['gpu_ms']+=begin.elapsed_time(end);value['gpu_complete_calls']+=1
        usage=resource.getrusage(resource.RUSAGE_SELF);previous=self.host_usage
        result["host_process"]={"minor_faults":usage.ru_minflt-previous.ru_minflt,"major_faults":usage.ru_majflt-previous.ru_majflt,"cpu_user_ms":(usage.ru_utime-previous.ru_utime)*1000,"cpu_system_ms":(usage.ru_stime-previous.ru_stime)*1000}
        result["host_gc"]={"pause_ms":self.gc_ms};self.gc_ms=0.
        self.reset()
        return result
    def close(self):
        for handle in self.handles:handle.remove()
        if self.gc_callback in gc.callbacks:gc.callbacks.remove(self.gc_callback)
