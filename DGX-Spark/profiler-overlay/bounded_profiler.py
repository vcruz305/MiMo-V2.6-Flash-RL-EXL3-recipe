"""Opt-in bounded worker profiling. Torch is imported only on worker start."""
import os
import json
import threading
import time
from pathlib import Path
from protocol import APIError


_PROFILE_LOCK = threading.Lock()


class BoundedProfiler:
    def __init__(self, max_active):
        self.directory = os.environ.get('EXL3_PROF_DIR', '').strip()
        self.accepted = 0
        self.current = None
        self.usable = None
        if self.directory and max_active != 1:
            raise ValueError('EXL3_PROF_DIR requires max_active=1')

    def start(self, ticket):
        if not self.directory:
            return
        if not _PROFILE_LOCK.acquire(blocking=False):
            raise RuntimeError('Another overlay profiler is active or unsafe to restart')
        path = Path(self.directory) / ticket.identifier
        try:
            path.mkdir(parents=True, exist_ok=False)
        except Exception:
            _PROFILE_LOCK.release()
            raise
        manifest = dict(request_id=ticket.identifier,
                        prompt_tokens=int(ticket.kwargs['input_ids'].shape[-1]),
                        max_new_tokens=ticket.kwargs['max_new_tokens'],
                        request_tps_valid=False, errors=[], complete=False,
                        worker_thread=threading.get_ident())
        state = dict(profiler=None, path=path, manifest=manifest, started=time.perf_counter())
        self.current = state
        try:
            import torch
            manifest['torch_version'] = str(torch.__version__)
            state['profiler'] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False, with_flops=False)
            state['profiler'].start()
        except Exception as exc:
            manifest['errors'].append(dict(phase='start', type=type(exc).__name__))
            raise

    def finish(self, outcome):
        if self.current is None:
            return
        state, self.current = self.current, None  # exactly one stop attempt, even on failure
        profiler, path, manifest = state['profiler'], state['path'], state['manifest']
        stopped = False
        if profiler is not None:
            try:
                profiler.stop()
                stopped = True
            except Exception as exc:
                manifest['errors'].append(dict(phase='stop', type=type(exc).__name__))
        # A failed stop leaves backend state unknown: poison this process guard.
        if stopped or profiler is None:
            _PROFILE_LOCK.release()
        manifest['profiled_lifecycle_wall_seconds'] = time.perf_counter() - state['started']
        manifest['outcome'] = outcome
        if stopped:
            try:
                profiler.export_chrome_trace(str(path / 'trace.json'))
            except Exception as exc:
                manifest['errors'].append(dict(phase='export', type=type(exc).__name__))
        manifest['complete'] = stopped and not manifest['errors']
        (path / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        if manifest['errors']:
            raise RuntimeError('Profiler failed; inspect local manifest')

    def accepted_job(self):
        if self.directory:
            self.accepted += 1

    def ensure_directory(self):
        """Fail the request, not the engine, when the trace directory is unusable."""
        if self.usable is None:
            try:
                path = Path(self.directory)
                path.mkdir(parents=True, exist_ok=True)
                probe = path / '.profiler-write-probe'
                probe.write_text('ok', encoding='utf-8')
                probe.unlink()
                self.usable = True
            except OSError:
                self.usable = False
        return self.usable

    def validate(self, kwargs):
        if not self.directory:
            return
        if not self.ensure_directory():
            raise APIError('EXL3_PROF_DIR is not a writable directory', 503, 'profiler_unavailable')
        if self.accepted >= 2:
            raise APIError('Profiler two-job lifetime budget exhausted', 429, 'profiler_job_limit')
        shape = getattr(kwargs.get('input_ids'), 'shape', ())
        output = kwargs.get('max_new_tokens')
        if (len(shape) != 2 or shape[0] != 1 or not 1 <= shape[1] <= 3600
                or type(output) is not int or not 1 <= output <= 24):
            raise APIError('Profiler requires batch=1, 1..3600 prompt tokens and explicit '
                           'max_new_tokens=1..24; no clipping', 400, 'profiler_request_limit')
