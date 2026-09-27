# Parent review — round 9 three-stage kernel trial on France GB10 (`deleg_efb269c6` / `sa-0-1bc0103e`)

**Verdict: numeric trial EXECUTED and PASSES; retained serve RESTORED and independently re-verified by the
parent. No speed result exists yet.** Not a deployment: the live runtime still loads the original DSO.

## What the operator did (and what the parent re-checked)

| Item | Operator claim | Parent check |
|---|---|---|
| Trial run | `--mode numeric`, rc 0, 319.5 s of a 1800 s budget | `remote-evidence/trial-receipt/guard/result.json` + custody `remaining_children=[]` |
| Numerics | 52 records; 25 flagged `five_three_bitwise`/`off_original_bitwise`; all true; max abs 2.01e-06 | Recomputed from `trial-result/result.json`: 52 records, 25 flagged, **all true**, max abs 1.7099e-05 (`duplicates/cap64/lo1`); nrmse ≈ 8e-4; tolerances untouched |
| Allocations | peak_alloc 886 786 048 B, peak_reserved 929 038 336 B | Matched byte-for-byte in the result header |
| Builds | 3 private fresh caches, DSOs `56caa20f…`, `94c85573…`, `6e5a6e2a…` (`-DEXL3_THREE_STAGE_BINDING`, poison off) | 3 build manifests fetched with effective `build.ninja`; local copies hash-verified |
| Memory | OOM 0 across 1273 samples, min available 112.05 GiB | Live readback: OOM/oom_kill/oom_group_kill all 0 |
| Evidence transfer | 57 files | 55/55 manifest entries re-hashed OK locally (2 additional files arrived after manifest generation) |
| Restoration | new run `…1790469050287027155`, guard 60108/78971117, server 60109/78971119 | Live readback PASSED (below) |

## Parent live readback (`parent_verify_round9_live.py`, read-only, no CUDA init, no inference)

- Pointer → `runs/france-round6-quant-readback-1790469050287027155`; guard **60108/starttime 78971117**,
  server **60109/starttime 78971119** (checked from `/proc/<pid>/stat`, not from the receipt).
- `pid.json` command, the live `/proc/<pid>/cmdline` and the pre-trial argv captured in
  `pre-stop-state.json` are **identical**: `-m 2.50bpw`, `-dm dflash-EXL3-4.0bpw`, `-ndt 7 -dds -dc 0.6
  -gs 106 -cs 4096 -cq 4 -ambs 1 -chunk_size 4096 -ccs 0 -rcs 0.25`, `--max-active-requests 1`.
- `launch.json == france-uma.json`; env exactly the retained set (`ELIDE_HANDLED=1`, counts 0,
  `EXL3_BATCH_VERIFY=0`, `EXL3_UMA_RESERVE_MB=8192`); **no** `EXL3_MK_THREE_STAGE` / `ROUND*` diagnostic
  env leaked into the serving config.
- Health 200 `{healthy:true, requests:0, max_active_requests:1}`; `nvidia-smi` compute-app set ==
  {server pid}.
- 678 original runtime/control hashes, 13 drafter hashes (block_size 8), 3 server source hashes, guard
  sha `eef5706b…`, shadow sha `61653605…`, loaded DSO `02b0ae5b…0207` — all verified.
- Old pair (56011/56012) gone; failed width run identities gone; **zero** round9 harness/controller
  processes remain; `MemAvailable` 19.47 GiB, OOM counters 0.
- Remote `trial-result/result.json` sha256 == `3152ddfa98c527be9762411cf5a3c0b2a162779ff6af0aca467a70d567d10701`.

Receipt: `round9-three-stage-gpu/parent01-live-readback.json`.

## Honest limits

- **No throughput measurement.** Numeric mode only; kernel micro-timings and model tokens/s are both absent.
  The only speed statements remain the round7/round6 numbers (~41 tok/s code, ~24 tok/s prose).
- Two controller starts failed closed **before** any stop (own-launcher false positive, then owned-pair false
  positive). Baseline provably untouched and preserved under `remote-evidence/attempts/`; the fixes were
  operator-side check defects, not kernel findings.
- The guard is a polling watchdog plus an allocator cap — not a kernel-hard OOM guarantee; cgroup ancestors
  above the container namespace root are invisible, so the retained sampler is the authority.
- The trial ran bounded synthetic ABI fixtures only: no model weights, no pages, no KV, no end-to-end path.
  Correctness of the kernel inside a real decode is still unproven, and 43 of 52 records being bitwise
  identical to the original kernel does **not** license swapping the runtime DSO.

## Consequence / next step

The three-stage kernel is now device-numerically validated (bitwise identity on complete-K geometries,
in-tolerance on different-geometry cases) but has **no measured speed advantage**. The deciding experiment is
a guarded `--mode timing` run at every observed q width (q1..q8) — dispatched as `deleg_2104d5b8` /
`sa-0-a3114d16`, same exactly-once stop/restore envelope, 1800 s cap, no poison/sanitizer/profiler, no
end-to-end A/B/A. Only if that shows a real kernel win is an end-to-end A/B/A worth the rental time.
