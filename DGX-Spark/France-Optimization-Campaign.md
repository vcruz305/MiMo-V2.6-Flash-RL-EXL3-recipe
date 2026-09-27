# France GB10 — optimization campaign handoff

Record for continuing the single-stream work on the **2.50 bpw** France path (GB10-class
rental, sm_121, 128 GB unified, aarch64, CUDA 13.0). Written 2026-09-26 at the end of the
kernel-experiment phase. Numbers are measured, not modelled; every speed figure below is
single-stream decode unless it says prefill.

## Where the profile stands

Served config (unchanged through every experiment below):

| | |
|---|---|
| target | `vcruz305/MiMo-V2.6-Flash-RL-EXL3` **2.50 bpw** |
| drafter | `MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw`, rev `d50ead3c6a3dec221e9a595fbdc103ef60db594e` |
| KV | **Q4**, `-cq 4`, `-chunk_size 4096`, `-cs 4096` |
| draft | native block 8, `-ndt 7`, `-dds -dc 0.6`, `-gs 106`, `-ambs 1`, `-ccs 0`, `-rcs 0.25` |
| runtime env | `EXL3_UMA=1`, `EXL3_UMA_RESERVE_MB=8192`, `EXL3_BATCH_VERIFY=0`, `EXL3_MOE_MIXEDK_ELIDE_HANDLED=1` |

Measured, warm, max-320-token ordered workload: **41.293333 code / 24.293333 prose tok/s**
(round 6 A/B/A bracket 40.79 / 23.8967). Fresh prefill sits near **703 tok/s** at chunk 4096
(chunk 4096 is ~51 % better than 1024 on the same prompt). **60 tok/s is not reached.** The
gap is prose acceptance: warm code emits ≈4.4 tokens per target pass (3.38 accepted), warm
prose only ≈2.0 (0.99 accepted), while every round pays a full target verification — measured
at ≈100.5 ms code / ≈76.2 ms prose per round against a ≈3.6 ms draft step. **The target
forward, not the drafter, is the cost centre.**

## Experiments and their verdicts

### 1. Three-stage mixed-K MoE pipeline — measured, rejected

Cut the five staged MoE stages to three (same-tile gate, in-CTA Hadamard) behind
`EXL3_MK_THREE_STAGE=1`, default off, q1..q8 only, fallback otherwise.

- **Numerics: exact.** 52 device records; all 25 identity-carrying cases true
  (`five_three_bitwise`, `off_original_bitwise`); 22 of them bitwise identical to the original
  mixed-K output across every `q1..q8 × {hot,spread}` geometry; the three different-geometry
  cases inside the untouched tolerances (max |Δ| 1.71e-05). Peak allocation 0.84 GB.
  Artifact: `trials/kernel-three-stage/round9-numeric-result.json`.
- **Speed: a regression.** 400 timed records, 5 reps × 20 iterations, 16 q-widths:

| | three/original | five-stage/original | off-vs-original control |
|---|---|---|---|
| q1-hot | 0.904 | 0.863 | 0.983 |
| q8-hot | **1.153** | 1.296 | 0.981 |
| q8-spread | **1.131** | 1.165 | 0.997 |
| median, 16 labels | **1.096** | ~1.13 | 0.32 % |

**13 of 16 widths are slower**; the q8 deltas are ~40× the control noise. It beats the
five-stage control at 11/16 labels, but that control is itself ≥1.16× the original at q8 — an
improvement inside a regression. Full table `trials/kernel-three-stage/round9b-timing-table.md`,
raw records `round9b-timing-result.json`.

**Do not deploy it and do not spend more rental time on staged-kernel variants.** The source
still exists as fork branch `exp/mixedk-three-stage` because the device numerics result is
worth keeping; treat it as closed.

### 2. Draft-window cost selector (DFlash) — CPU-verified, gate unmet

Per-round choice of the draft window from a measured cost profile, expected-emitted-tokens
objective, default off. Its first version silently rejected the real retained profile: the
guard said `recurrent_cache is not None` and MiMo's default SWA **is** recurrent (no
`-swa_full`) — its own constructor fixture had used a non-recurrent model. Repaired by
replacing that blanket exclusion with an exact class-identity + geometry allow-list
(`MiMoV2Model`/`SWAState`/`SlidingAttention`/`SWALayerState`/`RecurrentCache`, 48 layers,
global set `{0,5,11,17,23,29,35,41,47}`, window 127/overp 512/ring 768, intervals 2048/32768)
plus per-round fallback at checkpoint proximity, requeue, rewind, stop strings and loop
detectors. 254 CPU tests per tree; 3219 + 956 sealed inputs unchanged; both patches apply
byte-exactly.

**Still blocked, on purpose:** the profile's `attest_same_round8_weights_and_runtime` is
`false`, so it must not run live. Before opting in it needs independent weight/runtime
attestation plus device ring/paged-cache/restore/EOS correctness and an ordered cold/warm
A/B/A. Source: fork branch `feat/dflash-draft-cost-selector`.

### 3. Native 16-wide draft block — the live lever, not yet device-validated

Widening the drafter's native block from 8 to 16 (15 mask positions, right bound 15, taps
`[0,11,23,35,47]`, `tap_shift 0`) is the one lever that attacks the real bottleneck: the same
single target pass would verify twice as many proposals. Source/config study and CPU tests
pass (20 new, 187 original, expected native8/request12 RED preserved). The one device attempt
**aborted in its own diagnostic** — a CPU-vs-CUDA `torch.equal` before the QKV path — and
needed an exact-PID guard STOP; that harness is repaired and the recovery path has a locally
verified Linux adapter. Neither the repair nor the adapter has ever been run against the live
service, so the round-8 width trial is **not runnable as it stands**: the repaired harness, the
recovery core and the retained-launcher rollback are being composed into one fail-closed
package before any further device run. Do not hand-run the old controller.

## The unpublished runtime overlay (why 41 tok/s needs more than the pinned source)

The served runtime is **not** a clean checkout of the pinned source: it loads a python shadow
tree (`EXL3_ROOT`) whose only difference from pin `ca4a880e` is **one file**,
`exllamav3/modules/block_sparse_mlp.py` (sha256
`61653605b432350db3cccddbaa00b12f121e38fa2b072fc9de2032adf9e2a706` vs the pin's
`269bc2d994e0232ef994eb29cf71edab14d9317a110b24479315ee7585c09a31`). It is 7 lines: with
`EXL3_MOE_MIXEDK_ELIDE_HANDLED=1` it suppresses only the unconsumed `mixedk_handled`
nonzero/list readback that runs when `expert_count_list is None` (where the fallback loop has
zero iterations). Counts, active count, launch geometry, MTILE tiering, scratch/gather tables
and all kernels are untouched, and the default keeps both original readbacks. That one change
is the entire measured round-6 win: **+1.23396 % code / +1.65992 % prose** with all 8
text/count/acceptance pairs equal.

To reproduce this profile on another box you need that overlay — the pinned source alone will
not give you these numbers. It is now on fork branch `feat/mixedk-handled-readback-elision`
together with the offline source-review package. The separate, larger `counts_fused` elision
was deliberately excluded, and the count-readback experiment was not repeatable: it is **not**
part of any retained profile.

## Continuing on another Spark

1. Clone the fork and check out the branch you need (both default-off; nothing in them
   changes a default): `feat/mixedk-handled-readback-elision` (the measured runtime win),
   `feat/dflash-draft-cost-selector` (blocked, CPU-verified), `exp/mixedk-three-stage`
   (closed, kept for the numerics).
2. Build the extension for `TORCH_CUDA_ARCH_LIST=12.1` with `ninja` on PATH and
   `MAX_JOBS` bounded; the shadow overlay is a python-path override, not a build.
3. Serve with the profile table above and the UMA guard, and watch **both** host and cgroup
   OOM counters: `MemAvailable` around 19 GiB is normal for this profile, and unified memory
   makes a naive free-memory reading misleading.
4. Re-measure with the ordered warm workload before believing any change, and keep the
   no-drafter path as a control: identical no-drafter numbers with a moving drafter path means
   acceptance changed, not throughput.
5. Before any bounded device experiment, stop the owned serve through its guard (never a bare
   kill), verify the exact PID/starttime pair is gone, and relaunch the retained launcher
   afterwards, then re-verify identity, health, hashes and the loaded extension hash. Both
   kernel trials in this repo followed that sequence; the receipts are in
   `trials/kernel-three-stage/`.

### What is not claimed

No 60 tok/s. No aggregate or batched throughput. No SixCat result for this path. No claim that
the kernel variants are deployable, that the cost selector is correct on device, or that a
wider draft block actually pays off — that is exactly the experiment that has not run yet. The
timing trials are event-timed per-call wall times of a synthetic ABI harness, useful for A/B
ordering only, not comparable to profiler µs/call and not model tokens/s.
