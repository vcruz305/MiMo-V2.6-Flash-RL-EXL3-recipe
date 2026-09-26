"""Shared CPU fakes for the profiler overlay tests.

These are explicit test adapters around real worker lifecycle code: the profiler is
fake (CPU only) and the generator boundary is a CPU fixture. They say nothing about
CUDA/CUPTI availability, real trace volume, or numerical correctness.
"""
import json
import threading
import time
import types
from pathlib import Path


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError('worker condition timed out')


class FakeProfiler:
    def __init__(self, owner, options):
        self.owner = owner
        owner.options.append(options)
        owner.note('construct')
        if owner.fail == 'construct':
            raise RuntimeError('fake construct failure')

    def start(self):
        self.owner.note('start')
        self.owner.live += 1
        self.owner.peak = max(self.owner.peak, self.owner.live)
        if self.owner.fail == 'start':
            raise RuntimeError('fake start failure')

    def stop(self):
        self.owner.note('stop')
        self.owner.live -= 1
        if self.owner.fail == 'stop':
            raise RuntimeError('fake stop failure')

    def export_chrome_trace(self, path):
        self.owner.note('export')
        if self.owner.fail == 'export':
            raise OSError('fake export failure')
        Path(path).write_text(json.dumps({'traceEvents': [
            {'cat': 'kernel', 'ph': 'X', 'name': 'FAKE_kernel_A', 'dur': 2},
            {'cat': 'kernel', 'ph': 'X', 'name': 'FAKE_kernel_A', 'dur': 3},
            {'cat': 'cpu_op', 'ph': 'X', 'name': 'FAKE_operator', 'dur': 900},
        ]}), encoding='utf-8')


class FakeTorch(types.ModuleType):
    def __init__(self, fail=None):
        super().__init__('torch')
        self.fail = fail
        self.options, self.calls = [], []
        self.live = self.peak = 0
        self.__version__ = 'FAKE-not-torch'
        self.profiler = types.SimpleNamespace(
            profile=lambda **kw: FakeProfiler(self, kw),
            ProfilerActivity=types.SimpleNamespace(CPU='FAKE_CPU', CUDA='FAKE_CUDA'))

    def note(self, name):
        self.calls.append((name, threading.get_ident()))


class GeneratorAdapter:
    """CPU fixture implementing the native generator boundary, not inference."""
    def __init__(self, mode='eos', batch=1):
        self.mode = mode
        self.batch = batch
        self.jobs, self.cancelled, self.calls = [], [], []
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def enqueue(self, job):
        self.calls.append(('enqueue', threading.get_ident()))
        self.jobs.append(job)
        if self.mode == 'enqueue_error':
            raise RuntimeError('fake enqueue failure')

    def cancel(self, job):
        self.calls.append(('cancel', threading.get_ident()))
        self.cancelled.append(job)
        if job in self.jobs:
            self.jobs.remove(job)
        if self.mode == 'cancel_error':
            raise RuntimeError('fake cancel failure')

    def iterate(self):
        self.calls.append(('iterate', threading.get_ident()))
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError('test barrier timeout')
        if self.mode == 'iterate_error':
            raise RuntimeError('fake iterate failure')
        if self.mode in ('hold', 'cancel_error'):
            time.sleep(0.002)
            return []
        if self.mode == 'flood':
            return [dict(identifier=self.jobs[0].identifier, stage='streaming', text='FAKE')
                    for _ in range(200)]
        job = self.jobs[0]
        self.jobs.remove(job)
        return [dict(identifier=job.identifier, stage='streaming', eos=True,
                     text='FAKE', new_tokens=1, logits=object(), job=job)]
        # logits/job are deliberately non-serializable to prove they never reach HTTP.

    def num_remaining_jobs(self):
        return len(self.jobs)
