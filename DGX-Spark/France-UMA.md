# 2.50bpw UMA rental profiles

Local deployment record for the France rental; this does not replace the separately measured 2.20bpw/DFlash results in README.md. These changes have not been published upstream.

## Current default: adaptive DFlash 7, confidence target 0.6

`PROFILE=dynamic-balanced` is now the installed default: `-ndt 7 -dds -dc 0.6`,
Q4 paged KV, chunk1024, 4K context, one active request, serial verification,
GPU split106, and the unchanged 8-GiB UMA reserve / cache-aware supervisor.
The native diffusion block remains eight rows; adaptation shortens target
verification, not the drafter's fixed-size forward.

```bash
# Stop the existing owned process cleanly before launching any profile.
bash DGX-Spark/serve-uma-rental.sh
# Highest measured warmed code throughput in this small comparison:
PROFILE=dynamic-code bash DGX-Spark/serve-uma-rental.sh
# Static-window fallback with no online calibration:
PROFILE=dflash4 bash DGX-Spark/serve-uma-rental.sh
```

`dynamic-code` uses the same ceiling7 with target0.4. Target0.5 was tested but was
less consistent. Static4 and dynamic4/0.4 controls were also measured. Each arm
used the same request bytes/order/caps; two separate 0.6 loads reproduced the
code/prose response text, token counts and acceptance counters in this protocol.

- Balanced selected repeat, after adaptation: code **34.82–34.85 tok/s**, acceptance
  **0.779116**; long prose **23.74–23.75 tok/s**, acceptance **0.780899**.
  Versus static4 medians30.41/19.605, these are **+14.55% / +21.12%** observations.
- The first code request in the selected ordered sequence measured **33.74 tok/s**.
  Earlier 0.6 run: initial code33.55/34.73, warmed34.83/34.86; prose23.71–23.79.
- Code-focused target0.4: warmed code **37.44–37.64 tok/s**, acceptance **0.737410**,
  but prose **22.07–22.09 tok/s**, acceptance **0.521583**; its first code was27.83.
- Short cold explanation regressed: roughly21.4 tok/s at ceiling7 versus25.0 at
  static4. There is no universally fastest profile; retain the static fallback.
- Selected prefill: **424.65 prompt tok/s** at1244 tokens; **468.59** at3527 tokens.
  The long MAPLE recall passed. These are measured rates, not a new prefill speedup.
- Complete generated merge_intervals finished in291 tokens and passed207 offline
  function cases (7 hand-picked +200 deterministic random). This tests one generated
  function, not overall model quality or compliance with every prompt-format detail.
- Nine GPU page-boundary requests and subsequent arithmetic passed. Selected run's
  captured minimum availability **18.06 GiB**, no recorded host/container OOM.
  No guard threshold or memory budget was relaxed.

**Calibration and correctness limitations:** the confidence calibrator belongs to
the generator and learns across requests (64 decayed labels of burn-in and sparse-bin
checks). Request order and previous traffic matter; warmed throughput is not startup
throughput. No generator-method wrapper or in-place hot mutation was used. Dynamic
verification changes output text versus static4, so these are not token-equivalent
performance comparisons. The LARCH -> LAPINE exact-copy miss persists even when its
draft acceptance reaches1.0: acceptance is agreement with the target, not accuracy.
No profile is quality-qualified, and the optimum across kernels/prefill settings is
not established. The supervisor mitigates OOM risk but cannot guarantee safety.

Raw data, complete prompts/responses, launch flags, memory summaries and all arms:
[adaptive comparison JSON](../bench/france-dynamic-draft.json). Runtime/recipe
changes are local/rental-only; nothing has been published upstream.

## Additional kernel launch tests: retain BPS1 and automatic N

Two supported env overrides were measured one at a time, then removed:
`EXL3_MK_BPS=2` (more blocks per SM in unified mixed-K) and
`EXL3_MOE_TILE_N=128` (force the narrower fused-MoE column tile, with BPS1).
The compiled kernel's startup diagnostics confirmed both selections. No shared-memory
size override, group-count override, source edit, binary rebuild or memory-budget
increase was made. Q4/chunk1024 and adaptive DFlash7/0.6 remained fixed.

- Before control: warmed code median **34.865 tok/s**, prose **23.745**.
- BPS2: code **31.600**, prose **22.235**, fresh3527-token prefill **465.35 tok/s**.
- N128: code **27.765**, prose **19.290**, fresh3527-token prefill **428.90 tok/s**.
- Restored original: code **34.820**, prose **23.655**, prefill **471.13 tok/s**.
  Restored warmed code samples were34.92/34.72; acceptance remained0.779116.
- N128 and the restored arm matched all14 before-control texts, output counts and
  acceptance counters. BPS2 changed10/14 texts, so its timing difference is not a
  kernel-only speed comparison. Numerical correctness is not certified by text matches.
- All long MAPLE controls passed. The LARCH failure remains. Nine final GPU boundary
  checks and subsequent arithmetic passed after restoration.
- There are42 newly measured comparison requests plus14 reused before-control requests.
  The saved capture minimum across the new arms was **18.085 GiB available**; recorded
  host/container OOM counters remained zero. Guard/reserve unchanged.

Both overrides are rejected. The default remains unified, BPS1, automatic N, with
`PROFILE=dynamic-balanced`; `dynamic-code` remains the separately measured code-focused
option. Raw evidence: [kernel launch comparison](../bench/france-mixedk-kernels.json).
These are native-server end-to-end measurements, not a kernel profiler trace or an
established global optimum. No upstream publication.

## Mixed-K hybrid follow-up: keep the unified default

A controlled unified / threshold16 / threshold64 / unified sequence kept adaptive
DFlash7/0.6, Q4 KV, chunk1024, serial verification and the memory guard fixed.
`EXL3_MOE_MIXEDK_MIN_ROWS=16` selects legacy per-K-group dispatch only below
16 routed assignments; `64` does so below64. Whole-run legacy stayed disabled.
Eight experts/token bounds these experimental calls below the128-row fused cap;
normal1024-row measured loading was retained to initialize unified buffers first.
No source or CUDA binary was changed, and no group/buffer geometry override was used.

- Unified warmed code medians: **34.895 tok/s** before and **34.865** after.
- Hybrid16: **31.820 tok/s** code and **21.915** prose.
- Hybrid64: **31.985 tok/s** code and **21.995** prose.
- Unified prose medians: **23.770** before and **23.745** after.
- Fresh3527-token prefill: **474.85 / 471.40 / 466.57 / 472.14 tok/s** in arm order.
  No prefill gain is established; long MAPLE recall passed in all arms.
- Restoring unified reproduced all14 control texts, output counts and acceptance
  counters. The hybrid arms changed9/14 texts; their acceptance and completion
  lengths also varied. These figures do not isolate kernel time or certify
  numerical equivalence. Neither hybrid setting is selected or promoted.
- All56 comparison requests are preserved. Minimum sampled host availability
  across collected arm captures was **18.009 GiB**, with zero recorded host/container
  OOM events. The8-GiB reserve and existing supervisor remain unchanged.

The current default remains `dynamic-balanced` with unified mixed-K. The known
LARCH-copy miss remains; no quality qualification is claimed. Raw prompts, responses,
configuration and comparison: [hybrid experiment](../bench/france-mixedk-hybrid.json).

## Earlier sampler follow-up: batching did not improve throughput

The runtime now includes the executed-step requirement fix and a fail-closed batch
capability. Only a single exact Argmax/fused-greedy step may batch. Stateful Adaptive-P,
stochastic/multinomial, active penalties and unknown sampler steps remain serial.
The HTTP API does not expose Adaptive-P, but the underlying gate is now constrained.

The static fallback `PROFILE=dflash4` selects **serial verification** explicitly
(`EXL3_BATCH_VERIFY=0`). `PROFILE=dflash4-batch` selects the tested greedy-only batch
path; stochastic requests remain serial in that profile. Both keep Q4 paged KV,
1024-token chunks, 4K context, batch one, the 8-GiB UMA reserve and memory supervisor.

- CPU: **210 tests passed** (12 capability/gate, 11 requirement, 187 page tests).
- GPU: all **13** serial-versus-greedy-batch response texts, token counts, finish
  reasons and acceptance counters matched, including three stochastic request seeds.
  Nine page-boundary smoke requests also passed with the greedy-only batch gate.
- Code serial **30.41 / 30.32 tok/s**, greedy-batch **30.31 / 30.29**. No demonstrated
  throughput gain from batching; this is a correctness fix, not a new speed claim.
- First 1209-token prefills after independent restarts: **423.89 / 426.29 tok/s**.
  Do not use the earlier warmed replay's inflated prefill rate for comparison.
- Final selected serial restart: **30.00 / 30.37 tok/s** code, **0.664286** acceptance;
  1209-token prefill **421.59 tok/s**. All 13 responses matched the serial control.
- Final load/replay minimum availability **18.22 GiB**, no recorded host/container
  OOM. The guard mitigates risk; it is not a kernel-enforced guarantee.

The LARCH exact-copy failure remains, so no profile is quality-qualified. Historical
measurements below retain their original runtime scope. The runtime patches apply
also to the fallback launchers; they are not byte-identical historical runtimes.
Raw evidence: [sampler comparison JSON](../bench/france-sampler-greedy.json).
Nothing has been published upstream.

## DFlash ceiling 4 / native-page fix comparison

The installed profile now defaults to `PROFILE=dflash4`, retaining all Q4/1024,
4K context, single-request and UMA reserve/supervisor settings. `PROFILE=dflash7`
remains available, as do `q4-chunk1024` and `baseline`. These are small prompt-set
measurements, not a universally optimal configuration.

```bash
# Stop the existing owned server cleanly first.
bash DGX-Spark/serve-uma-rental.sh
```

A local two-file runtime patch separates the DFlash native write reservation from
the verification ceiling. It reserves the full native block at request end and
requeue boundaries, including implicit output limits and tiny requeue budgets.
The pinned original reproduced 60 CPU failures; the candidate and deployed sources
passed **187/187 CPU tests**. At ceiling 7, all six replayed response texts, output
token counts and acceptance counters matched the pre-patch baseline.

At ceiling 4, **nine real HTTP/GPU page-boundary smoke requests** completed at prompt
lengths 250, 253, 255, 256, 257, 509, 511, 512, 513 with one output token each, followed
by a correct arithmetic check. This is useful integration coverage, not exhaustive
GPU memory safety, streaming/requeue/CFG integration, or quality certification.

- Code, identical prompts and 256-token cap: **30.13 / 30.00 tok/s** with ceiling 4,
  versus **29.03 / 29.01** with patched ceiling 7; median increase **3.60%**.
  Relative to the earlier no-draft median, **67.26% higher** in this prompt set.
- Code acceptance: **0.664286**, versus **0.466667**. A shorter window changes the
  denominator and rejects fewer late guesses; this is not evidence the drafter
  itself became more accurate.
- Explanation: **24.60 tok/s**, acceptance **0.511364**, versus **21.17**, **0.312925**.
- 1209-token prefill: **454.44 prompt tok/s** in this trial. Small-sample timing only.
- Short MAPLE reply slowed: **33.34 tok/s**, versus **57.35** at ceiling 7; a short
  output that fits one larger draft round can lose from a shorter window.
- All six comparison response texts and output token counts matched between 4 and 7.
  The LARCH copy failure therefore remains; no model-quality qualification is claimed.
- Minimum sampled availability in the ceiling-4 load/tests: **18.18 GiB**; host and
  container OOM counters remained zero; the same guard and reserve remain active.

At the time of this page-fix comparison, the sampler-requirements patch was not
deployed. The follow-up above supersedes that deployment state. Dynamic drafting is off. Evidence and full configuration:
[page-fix and ceiling comparison JSON](../bench/france-dflash-pages.json).
The original native-seven evidence below predates the page patch; its warning about
unpatched short windows still applies to any unmodified runtime, not this patched one.

## Initial DFlash comparison: native ceiling 7, Q4 / chunk 1024

The initial DFlash profile is `PROFILE=dflash7`. It retains the Q4
paged cache, 1024-token chunk, 4K context, batch one, GPU split 106 GiB and all UMA
reserve/supervisor settings. Recurrent SWA buffers remain FP16. CPU affinity is
unchanged; the A/B/A fast-core experiment did not improve throughput.

```bash
# Stop the existing owned server cleanly first; never launch two copies.
PROFILE=dflash7 bash DGX-Spark/serve-uma-rental.sh
# Explicit no-draft comparison:
PROFILE=q4-chunk1024 bash DGX-Spark/serve-uma-rental.sh
```

Drafter source: XiaomiMiMo/MiMo-V2.6-Flash-RL revision
`5711b268169967567844e1e560e8a3966da959b1`, five files totaling 2,936,151,148 bytes,
verified against Hub sizes and LFS SHA256/Git-blob hashes. The corrected COPY has
`tap_shift=0`, taps `[0,11,23,35,47]`, and the BF16 `[4096]` learned mask embedding,
bit-identical to the source payload. CPU staging used restricted `weights_only=True`
loading; the source was verified unchanged. This is not a drafter trained for a new pack.

Measured from native server usage counters, one request at a time:

- Same code prompt bytes and 256-token output cap: **28.93 tok/s on both repeats**;
  all-core no-draft control median **17.975 tok/s**, a **1.609x / 60.95% increase**.
  Code draft acceptance was **0.466667** on both requests. Output text differed;
  this is a throughput comparison, not token-equivalence or a passing code test.
- Database explanation: **21.10 tok/s**, acceptance **0.312925**, 67 output tokens.
- 1209-token prefill: **420.67 prompt tok/s**, versus **469.12** for the identical
  prompt in the first no-draft affinity-control request. Drafting adds prefill work;
  do not carry the no-draft prefill numbers over to this profile.
- Arithmetic returned `323`; the short MAPLE control copied correctly. The long
  `LARCH-7426` request still produced `LAPINE-7426`, so quality remains unresolved.
- Minimum sampled available memory during load and initial inference: **18.19 GiB**;
  final available **18.91 GiB**; no recorded host/container OOM kills. The supervisor
  remains active, not a kernel-enforced guarantee.

Use native `-ndt 7`; dynamic drafting is off in this first comparison. Do NOT shorten
the ceiling until the native-block page-reservation fix is reviewed and GPU tested.
The sampler-requirement fix and page-reservation fix are NOT deployed in this run.
Raw measurements, config and affinity comparison: [DFlash JSON](../bench/france-dflash7-q4-chunk1024.json).
This is a working, faster-decode experimental profile, not the final optimized or
quality-qualified result. Other profiles and original evidence remain below.

## No-draft comparison: Q4 KV / 1024-token chunks

The no-draft `PROFILE=q4-chunk1024` uses `-cq 4 -chunk_size 1024`, as requested.
Q4 quantizes the paged K/V cache; recurrent sliding-window buffers remain FP16.
All other sizing stays at 4096-token context, one active request, GPU split 106 GiB,
no DFlash, OMP_NUM_THREADS=8, and the same opt-in UMA budget and memory supervisor.
CPU affinity has NOT been changed in this experiment.

```bash
# Only when the current owned server has been stopped:
PROFILE=q4-chunk1024 bash DGX-Spark/serve-uma-rental.sh
# Explicit fallback to the earlier FP16-cache / chunk-256 configuration:
PROFILE=baseline bash DGX-Spark/serve-uma-rental.sh
# Show the selected launcher without starting a model:
DRY_RUN=1 bash DGX-Spark/serve-uma-rental.sh
```

Measured through the same native beta API, one request at a time:

- Code decode, 256 output tokens: **18.02 tok/s**; previous profile **17.85 tok/s**.
- Explanation decode, 64 output tokens: **18.11 tok/s**; previous profile **17.94 tok/s** with 66 output tokens. These are small samples, not a robust decode speedup.
- First 1207-token prefill: **419.36 tok/s**, versus **264.66 tok/s** in the earlier profile (**58.45% higher** in that sample).
- Two follow-up 1209-token prefills: **453.15 / 449.98 tok/s**.
- Longer 3488-token prefill: **493.66 tok/s**; correctly returned `MAPLE-9362`.
- Arithmetic smoke returned `323`. The `LARCH-7426` check still failed: `LAPINE-7426` initially, then `LAPTOP-7426` on both repeats.
- Minimum sampled host availability through these tests: **20.87 GiB**; minimum physical free **1.14 GiB**; host/container OOM counters remained zero. Physical free and available memory are different on UMA; the cache-aware reserve/pressure guard remains active.

Raw timings, responses, repeated-test prompts and active argv are in [the Q4/chunk-1024 JSON](../bench/france-q4-chunk1024.json).
Both KV precision and chunk size changed, so these measurements do not isolate either knob's contribution.
Early unique markers prevent material prefix reuse across the prefill requests. No quality score is inherited from the FP16-cache profile, and the exact-copy failures remain unresolved.
The original baseline below is retained, not overwritten. Deployment changes are local-only, not pushed upstream.

## Original FP16-cache / chunk-256 baseline

- Pack: `vcruz305/MiMo-V2.6-Flash-RL-EXL3`, folder `2.50bpw`; 26 files, 13 shards, 98,475,960,404 bytes verified against the Hub tree.
- Runtime: exllamav3 `4c4502dbbf07c8a30c82d5ccba3e54d08eb693ba`, plus the local UMA budget patch.
- Torch 2.11.0+cu130, Python 3.12.3, CUDA toolkit 13.0.88; restored sm_121 extension.
- Recipe base: `49e438ac4deae89fc18bb576482a2b326b48c995`.
- Native beta `/v1` wrapper on loopback port 8096; no draft model.
- One active request, 4096-token cache/context, 256-token prefill chunks, GPU split 106 GiB, recurrent CPU checkpoint cache 0.25 GiB.
- Explicit `EXL3_UMA=1 EXL3_UMA_RESERVE_MB=8192`.

The modified budgeting code lives in exllamav3/util/memory.py and exllamav3/model/model_ls.py in the runtime checkout, NOT this recipe checkout. It uses conservative host availability and visible cgroup headroom rather than treating CUDA free as an absolute capacity limit. It retains an OS reserve, measured autosplit forwards, and the post-load allocation cap. It is opt-in and restricted to Linux with one visible GB10.

To reproduce this original baseline on the installed rental:

```bash
PROFILE=baseline bash DGX-Spark/serve-uma-rental.sh
```

The launcher refuses a second model process and occupied port. It requires the existing verified runtime, pack, and deployed patch; it does not run setup or download anything. It starts the supervised process detached and records the run directory in /workspace/mimo-tune/france-active-run.txt. The supervisor has a 12-hour lifetime. Inspect status with:

```bash
python3 /workspace/mimo-tune/france_load_status.py
curl --fail http://127.0.0.1:8096/health
```

Do not rerun `setup.sh` on this runtime: its default torch/ref choices differ. Matching Python development headers are required for Triton JIT even when the extension and torch import successfully. A 128-token prefill chunk fails this fork's 256-token page-alignment check.

## Measured results and caveats

Raw server responses, timings, memory summary, and configuration are in [the baseline JSON](../bench/france-uma-baseline.json).

- Database explanation: 66 generated tokens, **17.94 tok/s**, coherent two-sentence response.
- Code-generation timing: 256 generated tokens, **17.85 tok/s**; stopped at the requested length. This is throughput evidence, not a passing code test.
- First long-prefill request: 1207 prompt tokens in 4.560594 seconds, **264.7 prompt tok/s**, computed from the server's own prefill time; not a concurrent benchmark or a quality pass.
- Arithmetic smoke test returned the correct integer.
- **Exact-copy/recall tests failed:** requested `LARCH-7426`, observed `LAPINE-7426` on two long-prompt trials and `LARGE-7426` on a short copying prompt. A `MAPLE-9362` control copied correctly. Cause is not established; do not label this profile quality-qualified or attribute the miss to quantization, the memory patch, or a kernel without an A/B.
- DFlash is disabled, and no acceptance measurement or optimized speed claim is made.

## Memory behavior

The real load reclaimed cache: cached RAM declined from approximately 111 GiB to 19 GiB while model allocations grew. Torch reserved 91.24 GiB at the end-of-load budget reset. Availability remained above 21 GiB in the captured load/inference samples; the host and container OOM counters remained zero.

GPU allocation is not fully reflected in this container's memory.current, so the cgroup check alone is insufficient. The supervisor independently monitors host MemAvailable, cgroup headroom, pressure and OOM counters. Its 8-GiB availability/headroom floors remain active. Low physical free (<2 GiB) only stops UMA work when available host/cgroup headroom is also below 16 GiB; a physical-only floor otherwise aborts ordinary cache reclamation. Telemetry errors stop the owned child group. These are safeguards, not a kernel-enforced guarantee against OOM.

The deployed CPU regression suite passed 54 tests (44 budgeting and 10 supervisor/default tests); a real Triton kernel and CUDA operation passed before the full load. Raw evidence is retained under the active run directory. Runtime changes, the guard and launcher are preserved in the local working bundle as well.

## Q4 prefill staging comparison (local/rental-only)

A clean staged/direct/staged comparison kept Q4 KV, 1024-token chunks, dynamic7/0.6
and serial verification fixed. Direct means `EXL3_QC_STAGING=0`; the retained
source default is 1. The active implementation uses transient per-call FP16
window buffers for staging, not a persistent full-cache mirror.

- Warmed code medians: **34.925 / 34.760 / 34.735 tok/s**.
- Warmed prose medians: **23.835 / 23.755 / 23.695 tok/s**.
- Fresh 3527-token prefill: **467.80 / 467.95 / 471.61 tok/s**.
- Fresh 1244-token prefill: **416.36 / 428.74 / 430.97 tok/s**.

The direct arm did not beat the restored control. It changed one already-wrong
LARCH response (`LAPINE-7426` to `LAPTOP-7426`) and reduced the long MAPLE request's
reported draft-acceptance ratio from 1.0 to 0.666667 despite identical final text.
Both LARCH answers are failures. This is not a kernel-only timing or numerical
accuracy comparison. No speed/memory benefit is established; **retain staging=1**.

All 14 restored texts, completion counts and API acceptance ratios matched the
initial control. The 42 full requests, three configurations and comparisons are
in [the raw staging comparison](../bench/france-q4-prefill-staging.json). Nine
restored page-boundary GPU smokes passed. Minimum sampled host availability in
this comparison was **17.99 GiB**; recorded host/cgroup OOM counters stayed zero.
The existing guard is risk mitigation, not an absolute OOM guarantee. No runtime
source or extension was changed, and nothing was published upstream.

## Prefill chunk-size comparison (local/rental-only)

One variable at a time on a fresh load: `-chunk_size` 1024, 2048 or 4096 with Q4 paged
KV, `-cs 4096`, native draft ceiling 7 with `-dds -dc 0.6`, serial verification,
unified mixed-K and the 8 GiB UMA reserve held fixed. Each arm replayed the same
ordered workload (eight cold-sequence requests, four warmed repeats, one long-prefill
and one complete-code request) and the 3527-token prompt bytes were identical in
every arm.

| Arm | 3527-token prefill | 1244-token prefill | Warmed code | Warmed prose | Min host available |
|---|---:|---:|---:|---:|---:|
| chunk1024 (restored control) | 471.6 | 416.4 | 34.74 | 23.70 | 17.99 GiB |
| chunk1024 (second load) | 473.5 | 438.1 | 34.73 | 23.64 | 18.40 GiB |
| chunk2048 | 569.1 | 422.3 | 34.65 | 23.64 | 17.72 GiB |
| chunk4096 (first load) | 714.5 | 427.4 | 34.79 | 23.73 | 17.05 GiB |
| chunk4096 (second load) | 712.4 | 426.1 | 34.77 | 23.74 | 17.10 GiB |
| **chunk4096 (adopted default)** | **717.5** | 428.0 | **34.87** | **23.74** | 17.31 GiB |

Decode columns are medians of the two warmed code or prose repeats, in tok/s; prefill
columns are the server's own `prompt_tokens / time_prefill` for that request.

All five arms returned identical response text and identical completion counts for
every case, including the failing `LAPINE-7426` copy answer and the passing
`MAPLE-9362` control. Chunk size therefore changes prefill scheduling, not the
sampled result on this workload. Thirty-six page-boundary smoke requests (cap 1) and
the arithmetic check passed on the chunk4096 arms. Diagnostic and profiling
instrumentation were off in every arm.

Limitations: one rental, one prompt set, single stream, batch one. The 1244-token
prompt is dominated by fixed per-request overhead and shows no trend, and a larger
chunk means a longer first-token wait before streaming begins. A bigger chunk also
enlarges the load-time measuring cache: minimum physical-free dropped to about
1.45 GiB while host availability stayed above 17 GiB, and the sample watchdog is risk
mitigation, not a kernel-enforced OOM guarantee. Recorded host and cgroup OOM counters
remained zero. `EXL3_QC_PREFILL_NS`, `EXL3_MK_*`, the mixed-K arm and the sampling
mode were untouched: this is a supported CLI flag comparison, not a kernel change.

### Near-full-context prefill on the adopted default

A single prompt of **3959 chat-template tokens** (max_tokens 1,
so the whole prompt is one chunk) measured **794.7 tok/s** prefill
(4.98 s on the server's own timer) and returned the expected
one-word answer. Sampled host availability after the request was
**17.62 GiB** with host and cgroup OOM counters at
zero. At this length chunk1024 would split the prompt across four chunks; the
chunk1024 arm's 3527-token measurement was 471.6-473.5 tok/s. One request, one stream:
read as a boundary check for the adopted default, not a benchmark campaign.

## Draft-window and profiler follow-up (local/rental-only)

- `-dc 0.9`: acceptance 0.87-1.0, warmed decode **15.03/15.02 code, 15.09/15.07 prose**
  tok/s versus **34.90/34.84** and **23.74/23.74** on the adopted `-dc 0.6`. Rejected.
  The draft window is not the limiting factor for throughput; verified tokens per step is.
- `-gs 128`: **34.93/34.92 code, 23.81/23.82 prose**, acceptance 0.765/0.781/0.779/0.781
  (identical to the default arm), fresh 3527-token prefill 715.4 tok/s. No measurable
  change; reverted to `-gs 106`.
- `-ndt 12`: not supported - `draft_ids_pinned` is allocated for 7 draft tokens
  (`RuntimeError ... tensor a (12) ... b (7)`), engine reported `engine_unavailable`, host
  OOM counters zero, logs preserved. **Maximum supported `-ndt` is 7.**
- Profiler capture (opt-in overlay, removed afterwards; `worker.py` restored to LF sha256
  `3dbcb86c...`): decode is dominated by the unified mixed-K MoE kernel (~62% of summed
  device kernel time), prefill by mixed-K MoE (~30%) plus trellis reconstruction (~14%),
  with paged attention at ~3.5%. Both traces completed with zero profiler errors and host
  OOM counters at zero; sampled `MemAvailable` after each job stayed at 16.6-16.7 GiB.
  Profiled timings are not throughput measurements.
