"""Opt-in markers; event intervals include host-fed GPU idle, not pure compute."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import time
import torch

_ACTIVE = ContextVar("moe_recorder", default=None)


class Recorder:
    def __init__(self, events=False, nvtx=True, record_functions=False):
        self.events = events
        self.nvtx = nvtx
        self.record_functions = record_functions
        self.records = []

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    @contextmanager
    def region(self, name):
        begin = torch.cuda.Event(enable_timing=True) if self.events else None
        end = torch.cuda.Event(enable_timing=True) if self.events else None
        if self.nvtx:
            torch.cuda.nvtx.range_push(name)
        tick = time.perf_counter()
        if begin:
            begin.record()
        try:
            with torch.profiler.record_function(name) if self.record_functions else nullcontext():
                yield
        finally:
            if end:
                end.record()
            wall_us = (time.perf_counter()-tick)*1e6
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            self.records.append((name, begin, end, wall_us))

    def resolved(self):
        torch.cuda.synchronize()
        return [dict(stage=name, cuda_interval_us=begin.elapsed_time(end)*1000 if begin else None,
                     cpu_region_us=wall_us) for name, begin, end, wall_us in self.records]


def stage(name):
    active = _ACTIVE.get()
    return active.region(name) if active is not None else nullcontext()


def trace(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate
