"""Actual worker lifecycle, NumPy input shapes, explicitly fake Torch profiler.
No model, sampler, generator method replacement, CUDA, or inference is exercised.
"""
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(os.environ.get('PROF_TEST_SERVER', ROOT / 'server'))
sys.path.insert(0, str(SERVER))
from worker import GenerationWorker
from protocol import APIError
sys.path.insert(0, str(ROOT))
import summarize


from fake_runtime import GeneratorAdapter, FakeTorch, until


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'test-work')
        self.directory = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'EXL3_PROF_DIR': str(self.directory)})
        self.env.start()
        self.fake = FakeTorch()
        self.modules = patch.dict(sys.modules, {'torch': self.fake})
        self.modules.start()
        import bounded_profiler
        self.guard = patch.object(bounded_profiler, '_PROFILE_LOCK', threading.Lock())
        self.guard.start()
        self.workers = []
        self.factories = []

    def tearDown(self):
        for worker in self.workers:
            worker.generator.release.set()
            worker.close()
            self.assertFalse(worker.thread.is_alive())
        self.guard.stop()
        self.modules.stop()
        self.env.stop()
        self.tmp.cleanup()

    def make_worker(self, mode='eos', factory=None, **kw):
        gen = GeneratorAdapter(mode)
        def create(**kwargs):
            self.factories.append((kwargs, threading.get_ident()))
            return types.SimpleNamespace(**kwargs)
        worker = GenerationWorker(gen, factory or create, **kw)
        self.workers.append(worker)
        return worker

    def submit(self, worker, length=12, max_new=2):
        return worker.submit(input_ids=np.zeros((1, length), dtype=np.int64), max_new_tokens=max_new)

    def done(self, worker, ticket):
        event = ticket.events.get(timeout=3)
        until(lambda: worker.status()['requests'] == 0)
        return event

    def test_12_summary_is_derived_once_and_never_overwritten(self):
        job = self.directory / 'jobs' / 'job-1'
        job.mkdir(parents=True)
        events = [{'cat': 'kernel', 'ph': 'X', 'name': 'FAKE_kernel_A', 'dur': 2},
                  {'cat': 'kernel', 'ph': 'X', 'name': 'FAKE_kernel_B', 'dur': 5},
                  {'cat': 'kernel', 'ph': 'X', 'name': 'FAKE_kernel_A', 'dur': 3},
                  {'cat': 'cpu_op', 'ph': 'X', 'name': 'FAKE_operator', 'dur': 5000},
                  {'cat': 'kernel', 'ph': 'M', 'name': 'FAKE_marker_not_an_event'}]
        (job / 'trace.json').write_text('{\n "traceEvents": ' + json.dumps(events) + '\n}\n', encoding='utf-8')
        self.assertEqual(summarize.main([str(job)]), 0)
        summary, before = job / 'summary.json', None
        data = json.loads(summary.read_text())
        self.assertEqual(data['kernel_event_count'], 3)
        self.assertEqual(data['kernel_duration_us_total'], 10)
        self.assertEqual(data['kernels'], [{'name': 'FAKE_kernel_A', 'count': 2, 'duration_us': 5},
                                          {'name': 'FAKE_kernel_B', 'count': 1, 'duration_us': 5}])
        self.assertTrue(any('wall latency' in note for note in data['caveats']))
        before = summary.read_bytes()
        summary.write_text('{"keep": true}', encoding='utf-8')
        self.assertEqual(summarize.main([str(job)]), 0)
        self.assertEqual(summary.read_text(), '{"keep": true}')
        self.assertNotEqual(summary.read_bytes(), before)
        empty = self.directory / 'jobs' / 'job-2'
        empty.mkdir()
        self.assertEqual(summarize.main([str(empty)]), 3)
        self.assertEqual(summarize.main([str(self.directory / 'jobs' / 'absent')]), 3)
        self.assertFalse((empty / 'summary.json').exists())

    def test_11_unusable_profiler_directory_refuses_before_factory(self):
        os.environ['EXL3_PROF_DIR'] = str(self.directory / 'file-not-dir')
        (self.directory / 'file-not-dir').write_text('not a directory', encoding='utf-8')
        worker = self.make_worker()
        for _ in range(3):
            with self.assertRaises(APIError) as cm:
                self.submit(worker)
            self.assertEqual((cm.exception.status, cm.exception.code), (503, 'profiler_unavailable'))
        self.assertEqual(self.factories, [])
        self.assertEqual(self.fake.calls, [])
        self.assertIsNone(worker.profiler.current)
        missing = self.directory / 'created-on-demand' / 'nested'
        os.environ['EXL3_PROF_DIR'] = str(missing)
        other = self.make_worker()
        self.assertIsInstance(self.done(other, self.submit(other)), dict)
        self.assertTrue(list(missing.rglob('manifest.json')))

    def test_10_failed_stop_blocks_any_later_profiler(self):
        self.fake.fail = 'stop'
        first = self.make_worker()
        self.assertIsInstance(self.done(first, self.submit(first)), APIError)
        until(lambda: not first.thread.is_alive())
        self.fake.fail = None
        second = self.make_worker()
        self.assertIsInstance(self.done(second, self.submit(second)), APIError)
        self.assertEqual([n for n, _ in self.fake.calls].count('construct'), 1)
        self.assertEqual(len(self.factories), 1)

    def test_09_process_guard_prevents_concurrent_profilers(self):
        first = self.make_worker(mode='hold')
        ticket = self.submit(first)
        self.assertTrue(first.generator.entered.wait(3))
        second = self.make_worker()
        other = self.submit(second)
        event = self.done(second, other)
        self.assertIsInstance(event, APIError)
        self.assertEqual(self.fake.peak, 1)
        self.assertEqual(len(self.factories), 1)
        first.cancel(ticket)
        until(lambda: first.status()['requests'] == 0)
        self.assertEqual(self.fake.live, 0)

    def test_08_profiler_failures_deliver_worker_error_once(self):
        for phase in ('construct', 'start', 'export', 'stop'):
            with self.subTest(phase=phase):
                self.fake.fail = phase
                self.fake.calls.clear()
                self.fake.live = 0
                os.environ['EXL3_PROF_DIR'] = str(self.directory / phase)
                worker = self.make_worker()
                ticket = self.submit(worker)
                until(lambda: not worker.thread.is_alive())
                self.assertEqual(worker.status()['requests'], 0)
                self.assertFalse(ticket.events.empty(), 'profiler failure lost HTTP error')
                event = ticket.events.get_nowait()
                self.assertIsInstance(event, APIError)
                self.assertEqual(event.code, 'engine_unavailable')
                self.assertNotIn('fake', str(event))
                self.assertIsNone(worker.profiler.current)
                names = [name for name, _ in self.fake.calls]
                self.assertEqual(names.count('stop'), 0 if phase == 'construct' else 1)
                if phase == 'stop':
                    self.assertNotIn('export', names)
                manifests = list((self.directory / phase).rglob('manifest.json'))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text())
                self.assertTrue(manifest['errors'])
                self.assertFalse(manifest['complete'])
                self.assertTrue(all(ident == worker.thread.ident for _, ident in self.fake.calls))

    def test_07_terminal_failures_and_close_stop_and_cleanup(self):
        for mode in ('factory_value', 'factory_assert', 'factory_runtime',
                     'enqueue_error', 'iterate_error', 'cancel_error', 'close'):
            with self.subTest(mode=mode):
                self.fake.calls.clear()
                self.fake.live = 0
                os.environ['EXL3_PROF_DIR'] = str(self.directory / mode)
                def fail_factory(**kwargs):
                    exc = {'factory_value': ValueError, 'factory_assert': AssertionError,
                           'factory_runtime': RuntimeError}[mode]
                    raise exc('FAKE private input')
                worker = self.make_worker(mode='hold' if mode == 'close' else mode,
                    factory=fail_factory if mode.startswith('factory_') else None)
                ticket = self.submit(worker)
                if mode in ('close', 'cancel_error'):
                    self.assertTrue(worker.generator.entered.wait(3))
                    if mode == 'close':
                        worker.close()
                    else:
                        worker.cancel(ticket)
                event = self.done(worker, ticket)
                self.assertIsInstance(event, APIError)
                self.assertEqual([n for n, _ in self.fake.calls], ['construct', 'start', 'stop', 'export'])
                self.assertEqual(self.fake.live, 0)
                self.assertIsNone(worker.profiler.current)
                self.assertTrue(all(ident == worker.thread.ident for _, ident in self.fake.calls))
                if mode in ('factory_value', 'factory_assert'):
                    self.assertEqual(event.status, 400)
                    self.assertTrue(worker.status()['healthy'])
                else:
                    self.assertEqual(event.code, 'engine_unavailable')
                    self.assertNotIn('FAKE private input', str(event))
                if mode in ('enqueue_error', 'iterate_error', 'cancel_error', 'close'):
                    self.assertIn(ticket.job, worker.generator.cancelled)
                worker.close()

    def test_06_cancel_stops_on_owner_and_queued_cancel_uses_budget(self):
        worker = self.make_worker(mode='hold')
        active = self.submit(worker)
        self.assertTrue(worker.generator.entered.wait(3))
        queued = self.submit(worker)
        worker.cancel(queued)
        worker.cancel(active)
        until(lambda: worker.status()['requests'] == 0)
        self.assertEqual([name for name, _ in self.fake.calls], ['construct', 'start', 'stop', 'export'])
        self.assertEqual(worker.generator.cancelled, [active.job])
        self.assertEqual(len(self.factories), 1)
        self.assertTrue(all(ident == worker.thread.ident for _, ident in self.fake.calls))
        manifests = list(self.directory.rglob('manifest.json'))
        self.assertEqual(json.loads(manifests[0].read_text())['outcome'], 'cancelled')
        with self.assertRaises(APIError):
            self.submit(worker)

    def test_05_eos_profiles_whole_job_on_owner_thread(self):
        def factory(**kwargs):
            self.assertEqual([name for name, _ in self.fake.calls], ['construct', 'start'])
            self.factories.append((kwargs, threading.get_ident()))
            return types.SimpleNamespace(**kwargs)
        worker = self.make_worker(factory=factory)
        ticket = self.submit(worker)
        event = self.done(worker, ticket)
        self.assertIsInstance(event, dict)
        self.assertEqual([name for name, _ in self.fake.calls], ['construct', 'start', 'stop', 'export'])
        self.assertTrue(all(ident == worker.thread.ident for _, ident in self.fake.calls))
        self.assertEqual(self.fake.options, [dict(activities=['FAKE_CPU', 'FAKE_CUDA'],
            record_shapes=False, profile_memory=False, with_stack=False, with_flops=False)])
        manifests = list(self.directory.rglob('manifest.json'))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text())
        self.assertEqual(manifest['outcome'], 'eos')
        self.assertFalse(manifest['request_tps_valid'])
        self.assertEqual(manifest['prompt_tokens'], 12)
        self.assertEqual(manifest['max_new_tokens'], 2)
        self.assertTrue((manifests[0].parent / 'trace.json').is_file())
        self.assertNotIn('logits', event)
        self.assertNotIn('job', event)
        self.assertNotIn('trace', event)
        self.assertNotIn('manifest', event)
        self.assertEqual(self.fake.live, 0)

    def test_04_enabled_worker_requires_serial_configuration(self):
        with self.assertRaises(ValueError):
            self.make_worker(max_active=2)
        self.assertEqual(self.fake.calls, [])

    def test_03_accepts_at_most_two_jobs_not_seven(self):
        worker = self.make_worker()
        for _ in range(2):
            ticket = self.submit(worker, length=3600, max_new=24)
            self.assertTrue(self.done(worker, ticket)['eos'])
            self.assertEqual(ticket.kwargs['input_ids'].shape, (1, 3600))
            self.assertEqual(ticket.kwargs['max_new_tokens'], 24)
        for _ in range(5):
            with self.assertRaises(APIError) as cm:
                self.submit(worker)
            self.assertEqual((cm.exception.status, cm.exception.code), (429, 'profiler_job_limit'))
        self.assertEqual(len(self.factories), 2)

    def test_02_requires_batch_one_and_explicit_small_output(self):
        worker = self.make_worker()
        invalid = [
            {}, {'max_new_tokens': None}, {'max_new_tokens': 25},
            {'max_new_tokens': 0}, {'max_new_tokens': -1},
            {'max_new_tokens': True}, {'max_new_tokens': 1.5},
            {'input_ids': np.zeros((2, 2)), 'max_new_tokens': 2},
            {'input_ids': np.zeros((2,)), 'max_new_tokens': 2},
            {'input_ids': np.zeros((1, 0)), 'max_new_tokens': 2},
            {'input_ids': None, 'max_new_tokens': 2}]
        for extra in invalid:
            with self.subTest(extra=repr(extra)):
                kwargs = {'input_ids': np.zeros((1, 12))}
                kwargs.update(extra)
                with self.assertRaises(APIError) as cm:
                    worker.submit(**kwargs)
                self.assertEqual(cm.exception.code, 'profiler_request_limit')
        self.assertEqual(self.factories, [])

    def test_01_oversized_prompt_rejected_before_factory(self):
        worker = self.make_worker()
        with self.assertRaises(APIError) as cm:
            self.submit(worker, length=3601)
        self.assertEqual((cm.exception.status, cm.exception.code), (400, 'profiler_request_limit'))
        self.assertEqual(self.factories, [])
        self.assertEqual(self.fake.calls, [])


(ROOT / 'test-work').mkdir(exist_ok=True)
if __name__ == '__main__':
    unittest.main()
