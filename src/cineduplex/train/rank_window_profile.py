"""Bounded per-rank window diagnostics; no synchronization or model changes."""
import functools
import json
import queue
import resource
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist


class RankWindowProfile:
    def __init__(self, model, root, call_id, rank):
        import re
        if not re.fullmatch(r'[0-9a-f-]{36}', call_id):
            raise ValueError('rank profile call identity')
        self.path = Path(root) / 'rank-profile' / call_id / ('rank%d.jsonl' % rank)
        self.rank = rank
        self.active = False
        self.in_main = False
        self.closed = False
        self.handles = []
        self.originals = {}
        self.queue = queue.Queue(maxsize=128)
        self.error = None
        self.write_max_ms = 0.
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()
        # Forward hooks observe the admitted modules; they return no replacement.
        def before_main(module, args):
            if not self.active or not self.detailed:
                return
            self.in_main = True
            self.main_start = time.perf_counter()
            self.main_event = torch.cuda.Event(enable_timing=True)
            self.main_event.record()
        def after_main(module, args, result):
            if not self.active or not self.detailed:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._sample('main', 'main', time.perf_counter()-self.main_start,
                         self.main_event, end)
            self.in_main = False
        self.handles.extend((model.model.register_forward_pre_hook(before_main),
                             model.model.register_forward_hook(after_main)))
        for name, module in model.model.named_modules():
            if '.base' in name or not name.endswith('_proj'):
                continue
            state = {}
            def before(module, args, state=state):
                if self.active and self.detailed:
                    state['start'] = time.perf_counter()
            def after(module, args, result, state=state, name=name):
                if self.active and self.detailed:
                    self._sample('projection_'+name.rsplit('.', 1)[-1], name,
                                 time.perf_counter()-state.pop('start'))
            self.handles.extend((module.register_forward_pre_hook(before),
                                 module.register_forward_hook(after)))
        for name in ('all_reduce', 'all_gather_into_tensor'):
            original = getattr(dist, name)
            @functools.wraps(original)
            def measured(*args, _name=name, _fn=original, **kwargs):
                if not self.active or not self.in_main:
                    return _fn(*args, **kwargs)
                begin = None
                if self.window % 25 == 0:
                    begin = torch.cuda.Event(enable_timing=True)
                    begin.record()
                start = time.perf_counter()
                result = _fn(*args, **kwargs)
                elapsed = time.perf_counter()-start
                end = None
                if begin is not None:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                self._sample(_name, _name, elapsed, begin, end)
                return result
            self.originals[name] = (original, measured)
            setattr(dist, name, measured)

    def _sample(self, key, label, elapsed, begin=None, end=None):
        self.samples.append((key, label, elapsed*1000, begin, end))

    @staticmethod
    def _schedule():
        # This is kernel telemetry, not model/data storage or a GPU sync.
        try:
            return [int(x) for x in Path('/proc/thread-self/schedstat').read_text().split()]
        except OSError:
            return None

    @staticmethod
    def _allocation():
        stats = torch.cuda.memory_stats()
        return {key:stats.get(key, 0) for key in
                ('num_alloc_retries', 'num_ooms', 'num_sync_all_streams',
                 'num_device_alloc', 'num_device_free')}

    def begin(self, window):
        if self.closed or self.active:
            raise RuntimeError('rank profiler window lifecycle')
        if self.error:
            raise RuntimeError('rank diagnostic writer: '+self.error)
        begun=time.perf_counter()
        self.command_dispatch_ms=None
        self.window = window
        self.detailed = window % 25 == 0
        self.samples = []
        self.usage = resource.getrusage(resource.RUSAGE_SELF)
        self.schedule = self._schedule()
        self.allocation = self._allocation()
        self.begun = time.perf_counter()
        self.setup_ms=(self.begun-begun)*1000
        self.active = True

    def finish(self):
        if not self.active:
            return
        self.active = False
        self.in_main = False
        totals = {}
        for key, label, cpu, begin, end in self.samples:
            row = totals.setdefault(key, {'calls':0, 'cpu_ms':0., 'gpu_ms':0.,
                                         'gpu_recorded_calls':0, 'gpu_complete_calls':0})
            row['calls'] += 1
            row['cpu_ms'] += cpu
            if end is not None:
                row['gpu_recorded_calls'] += 1
                if end.query():
                    row['gpu_ms'] += begin.elapsed_time(end)
                    row['gpu_complete_calls'] += 1
        usage = resource.getrusage(resource.RUSAGE_SELF)
        schedule = self._schedule()
        allocation = self._allocation()
        record = {'at':time.time(), 'rank':self.rank, 'input_seq':self.window,
                  'detailed':self.detailed,
                  'window_wall_ms':(time.perf_counter()-self.begun)*1000,
                  'setup_ms':self.setup_ms, 'command_dispatch_ms':self.command_dispatch_ms,
                  'stages':totals,
                  'slow_cpu_calls':[{'stage':key, 'module':label, 'cpu_ms':cpu}
                      for key, label, cpu, _, _ in sorted(self.samples, key=lambda x:x[2], reverse=True)[:6]],
                  'cpu_user_ms':(usage.ru_utime-self.usage.ru_utime)*1000,
                  'cpu_system_ms':(usage.ru_stime-self.usage.ru_stime)*1000,
                  'major_faults':usage.ru_majflt-self.usage.ru_majflt,
                  'voluntary_switches':usage.ru_nvcsw-self.usage.ru_nvcsw,
                  'involuntary_switches':usage.ru_nivcsw-self.usage.ru_nivcsw,
                  'thread_schedule_delta_runtime_ns_wait_ns_count':([a-b for a,b in zip(schedule,self.schedule)]
                      if schedule is not None and self.schedule is not None else None),
                  'allocator_delta':{k:allocation[k]-v for k,v in self.allocation.items()},
                  'timing_caveat':'CUDA spans include execution and waits; nested stages overlap; no extra synchronize'}
        self.samples = []
        payload = json.dumps(record)+'\n'
        if len(payload)>32768:
            raise RuntimeError('rank diagnostic record exceeds budget')
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            raise RuntimeError('rank diagnostic backlog exceeds 128 records')

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('x') as file:
                while True:
                    payload = self.queue.get()
                    if payload is None:
                        break
                    begun = time.perf_counter()
                    file.write(payload)
                    self.write_max_ms = max(self.write_max_ms, (time.perf_counter()-begun)*1000)
        except Exception as exc:
            self.error = type(exc).__name__+': '+str(exc)[:200]

    def close(self):
        if self.closed:
            return
        self.finish()
        self.closed = True
        for handle in self.handles:
            handle.remove()
        for name, (original, measured) in self.originals.items():
            if getattr(dist, name) is measured:
                setattr(dist, name, original)
        self.queue.put_nowait(None)
        self.writer.join(timeout=30)
        if self.writer.is_alive() or self.error:
            raise RuntimeError('rank diagnostics did not close cleanly: '+str(self.error))
