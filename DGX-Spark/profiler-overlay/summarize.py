"""Derive a small summary from one saved Chrome trace; never overwrites it.

Run:  python summarize.py <job-directory>

The trace is scanned incrementally, so a large torch profiler export does not have
to be loaded as one in-memory object. Kernel entries are the raw Chrome events with
cat=="kernel" and a numeric dur (microseconds). Those durations are device kernel
time; they are NOT wall latency, and operator device time from cpu_op/gpu_op
entries must never be added to them (that would double count).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

NAME_LIMIT = 120
CAVEATS = [
    "kernel entries are raw Chrome profiler events (cat=kernel, numeric dur in microseconds)",
    "kernel_duration_us_total is summed device kernel time, not wall latency; a profiled "
    "request is much slower than the unprofiled one",
    "do not add operator device time (cpu_op/gpu_op device_time) to raw kernel time: they "
    "describe the same work and would be double counted",
    "the trace reflects one job on the generation worker thread only; HTTP/queue time is not "
    "inside the profile unless it happened on that thread",
    "paths are written next to the accepted job, outside any HTTP response queue",
    "this summary is derived text only; it retains no tensors or profiler objects",
]


def kernel_events(text):
    """Yield raw kernel events from the traceEvents array without loading it whole."""
    marker = '"traceEvents"'
    start = text.find(marker)
    if start < 0:
        raise ValueError('trace has no traceEvents array')
    start = text.find('[', start + len(marker))
    if start < 0:
        raise ValueError('trace has no traceEvents array')
    decoder, index, length = json.JSONDecoder(), start + 1, len(text)
    while True:
        while index < length and text[index] in ' \t\r\n,[]':
            if text[index] == ']':
                return
            index += 1
        if index >= length:
            raise ValueError('traceEvents array is not terminated')
        event, index = decoder.raw_decode(text, index)
        if isinstance(event, dict) and event.get('cat') == 'kernel':
            yield event


def summarize(job_dir):
    job = Path(job_dir)
    trace = job / 'trace.json'
    summary = job / 'summary.json'
    if not trace.is_file():
        raise FileNotFoundError(f'no trace.json in {job}')
    raw = trace.read_text(encoding='utf-8')
    counts = {}
    total, seen = 0, 0
    for event in kernel_events(raw):
        duration = event.get('dur')
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            continue
        name, seen, total = str(event.get('name', '')), seen + 1, total + duration
        entry = counts.setdefault(name, {'count': 0, 'duration_us': 0})
        entry['count'] += 1
        entry['duration_us'] += duration
    kernels = []
    truncated = False
    for name in sorted(counts, key=lambda key: (-counts[key]['duration_us'], key)):
        short = name[:NAME_LIMIT]
        truncated = truncated or short != name
        kernels.append({'name': short, 'count': counts[name]['count'],
                        'duration_us': counts[name]['duration_us']})
    return {
        'trace': trace.name, 'trace_bytes': len(raw),
        'trace_sha256': hashlib.sha256(raw.encode()).hexdigest(),
        'kernel_event_count': seen, 'kernel_duration_us_total': total,
        'kernels': kernels, 'kernel_names_truncated': truncated,
        'named_kernel_kinds': len(kernels), 'caveats': CAVEATS}


def main(argv):
    if len(argv) != 1:
        print('usage: python summarize.py <job-directory>', file=sys.stderr)
        return 2
    summary = Path(argv[0]) / 'summary.json'
    if summary.exists():
        print(f'SUMMARY_EXISTS {summary}')
        return 0
    try:
        payload = summarize(argv[0])
    except FileNotFoundError as exc:
        print(f'summarize: {exc}', file=sys.stderr)
        return 3
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'summarize: cannot read trace: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    try:
        with summary.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(payload, indent=2) + '\n')
    except FileExistsError:
        print(f'SUMMARY_EXISTS {summary}')
        return 0
    print(f'SUMMARY_CREATED {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
