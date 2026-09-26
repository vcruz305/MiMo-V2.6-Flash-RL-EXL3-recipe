# Optional bounded profiler overlay for the native server

A default-off diagnostic used once on the GB10 host to answer one question — *which
kernels does this profile actually spend its time in?* — and then removed. It is here
because the answer is useful and the tool is reusable, not because the server needs it.

**It is not a serving path.** Profiling inflates every request, so a profiled run must
never be quoted as throughput. Kernel durations in the summary are device time, not wall
latency, and operator device time (`cpu_op`/`gpu_op`) must not be added to them.

## What it does

- `server-overlay.patch` — the only change to the server: `server/worker.py` imports
  `BoundedProfiler` and drives it for the lifetime of a generation job.
- `bounded_profiler.py` — the helper the patch adds to `server/`.
- `summarize.py` — derives a small kernel summary from one saved Chrome trace.
- `tests/` — CPU regressions for the gate, the job lifecycle and the summarizer.

Apply from the repository root, run the suite, then put the server back:

```bash
git apply -p1 DGX-Spark/profiler-overlay/server-overlay.patch
cp DGX-Spark/profiler-overlay/bounded_profiler.py server/
PROF_TEST_SERVER=server python DGX-Spark/profiler-overlay/tests/test_profiler.py   # 12 tests
rm server/bounded_profiler.py
git apply -R -p1 DGX-Spark/profiler-overlay/server-overlay.patch
```

`DGX-Spark/profiler-overlay/tests/test_profiler.py` drives the real worker lifecycle against a fake Torch profiler
and NumPy input shapes; `PROF_TEST_SERVER` points it at the patched `server/` directory
(it defaults to a sibling `server/`), and it writes its temporary jobs under its own
`test-work/`. The candidate's separate control suite — byte-identity against captured
fixtures, patch-scope and snapshot hashes — is not shipped, because it is bound to that
capture tree rather than to this repository.

On the host the overlay was deployed for one capture and the server file was restored to
its exact committed bytes afterwards (`sha256 3dbcb86c…` for `server/worker.py`).

## Caps (deliberate, and the reason it is safe to run)

- `EXL3_PROF_DIR` must be set: without it the overlay is inert.
- At most **2 jobs per process** (429 after that); one capture per process lifetime, so a
  second capture means a deliberate reload.
- **Batch 1, prompt ≤ 3600 tokens, explicit `max_new_tokens` 1..24, no clipping** — a
  request outside that window is rejected before the job factory.
- The profiler is constructed, started, stopped and exported **only on the generation
  worker thread**, behind a process-wide lock; a failed stop poisons the lock.
- `record_shapes`, `profile_memory`, `with_stack` and `with_flops` are all off.
- An unwritable `EXL3_PROF_DIR` fails the request (503), not the engine.

The caps bound *inference work*, not CUPTI/Chrome event buffering or host RAM. Check the
trace size and host headroom while collecting; on the GB10 capture the two traces were
108 MB (3527-token prefill, one chunk) and 42 MB (16-token decode) with
`MemAvailable` still at 16.6 GiB after each job and both OOM counters at zero.

## What one capture showed

Same profile as [the host notes](../README.md) — DFlash 7, adaptive drafting, Q4 KV,
4096-token chunks, 4K context:

- **Decode-heavy job:** the unified mixed-K expert kernel (`exl3_moe_mixedk_kernel<256,2,16,3,0>`)
  alone is **~62%** of summed device kernel time (165 calls, 559 ms of 901 ms); the
  cooperative MoE kernels, the draft model's GEMVs and small cutlass GEMMs split the rest.
- **Prefill-heavy job (3527 tokens, one chunk):** no single kernel dominates — mixed-K MoE
  shapes total **~30%**, trellis reconstruction (`reconstruct_had_batch_kernel` +
  `reconstruct_kernel`) **~14%**, cutlass GEMMs ~9%, and `_paged_attn_prefill_kernel`
  only **~3.5%**.

So decode is MoE-expert bound and prefill is MoE-plus-dequant bound; attention is not the
bottleneck at these shapes. Raw captures stayed on the host; the summarized kernel tables
are in [`bench/france-draft-window.json`](../../bench/france-draft-window.json).
