# Parent review — round 9b three-stage kernel timing trial (`deleg_2104d5b8` / `sa-0-a3114d16`)

**Verdict: the three-stage mixed-K kernel is a REAL REGRESSION at the width that matters. Kernel path
closed. No deployment, no end-to-end A/B/A. Retained serve restored and parent-verified.**

## Independent recomputation (parent, from the raw 400 records — not the operator's table)

Parsed `remote-evidence/trial-result/result.json` (sha256 `deb6b199…12d9`, 400 records, `status=executed`,
`poison=false`, arms `original` 160 / `off` 80 / `five` 80 / `three` 80, 16 labels q1…q8 × {hot,spread},
5 reps × 20 iterations, event-timed per call):

| | three/orig | five/orig | off/orig (noise control) |
|---|---|---|---|
| q1-hot | 0.904 | 0.863 | 0.983 |
| q1-spread | 1.008 | 0.959 | 1.005 |
| q8-hot | **1.153** | 1.296 | 0.981 |
| q8-spread | **1.131** | 1.165 | 0.997 |
| median over 16 labels | **1.096** | 1.13 | 0.32 % median, 1.91 % max |

- **13 of 16 labels are slower** with three-stage; the q8 deltas are ~30–45× the noise floor and hold in
  every repetition. My numbers agree with the operator's paired-ratio table to within 1–2 %.
- q1 is a wash (0.90 hot / 1.01 spread) — and at q1 the five-stage control is faster than both, so the
  reduction bought nothing where the kernel is cheap.
- The claimed "beats five-stage at 11/16" is real but worthless: five-stage is itself 1.30×/1.17× the
  original at q8, i.e. an improvement inside a regression.

## Parent live readback (`parent_verify_round9b_live.py`, read-only, no CUDA init, no inference)

**PASSED.** Pointer → `runs/france-round6-quant-readback-1790470171171732232`; guard **64051/starttime
79083205**, server **64052/starttime 79083207** (read from `/proc/<pid>/stat`); `pid.json` command == live
`/proc/cmdline` == pre-trial retained argv; `launch.json == france-uma.json`; retained env intact with **no**
`EXL3_MK_THREE_STAGE` / `ROUND*` leak; health 200 `{healthy, requests:0, max_active_requests:1}`; sole CUDA
process = server pid; 678 original + 13 drafter + 3 server-source hashes OK; loaded DSO unchanged
`02b0ae5b…0207`; round9's numeric `result.json` still `3152ddfa…0701` (prior artifacts untouched); old pair
60108/60109 gone; no round9/round9b controller processes; MemAvailable 20.13 GiB, host + cgroup OOM
counters all 0. Receipt: `round9b-three-stage-timing/parent01-live-readback-recomputa…` → see
`parent-verify-round9b.log`.

## Operator disclosures the parent accepts (both disclosed, neither invalidates the result)

1. Protected-root inventory was 10 roots (15 after runner expansion) versus round9's per-file enumeration —
   protection is instead evidenced by fresh/disjoint write-path admission plus post-run re-hashing of all
   named identities. Disclosed rather than hidden; recorded in the skill as a rule.
2. Non-default `--samples 5 --iterations 20` (harness bounds) chosen for a real variance estimate.

Harness coverage limit: this revision has no timing branch in the second fixture loop, so the 9 adversarial
fixtures are numeric-only and were not timed. The event-timed region is the whole fixture call path, not pure
kernel time — valid for A/B ordering, not comparable to profiler µs/call. **No tokens/s claim anywhere.**

## Consequence for the 60 TPS goal

- The staged-kernel direction (five→three) is **exhausted**: numerically exact and measurably slower.
  Do not spend more rental time on it.
- Two candidate follow-ons remain, both currently only *nominated* by this harness and unmeasured
  end-to-end: the five-stage control's q1 behaviour (0.863/0.959 vs original) — note q1 dominates prose
  (~93/387 retained warm rounds) while code is q8-dominated — and the DFlash draft-cost selector, whose
  live gate is still attestation + device correctness, not speed.
- Nothing here changes the standing measured numbers: ~41.3 tok/s code / ~24.3 tok/s warm prose single
  stream, ~703 tok/s fresh prefill. **60 TPS is still unmet.**
