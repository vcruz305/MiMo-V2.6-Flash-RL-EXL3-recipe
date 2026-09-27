# France: quantized drafter and handled-readback measurements

Completed native `/v1` single-stream measurements on the France GB10 rental,
2026-09-26. These use the **2.50 bpw target**, not the separate RTX workload/pack.
They are **not SixCat**, not aggregate throughput, and not broad quality certification.
**60 tok/s was not reached.** The earlier ~35 tok/s BF16 results remain a dated
profile record in [France-UMA.md](France-UMA.md), not the new drafter's result.

## Operator replay, not a portable bootstrap

[serve-uma-rental.sh](serve-uma-rental.sh) selects **preexisting rental launchers**.
A fresh clone does not install the runtime, guard, entrypoints or these launchers.
The native server remains primary; this does not change Tabby, RTX pack selection,
or root serving defaults.

- `quantized-draft` is this operator wrapper's default: the round5 EXL3 drafter
  on the original rental runtime. It is also the **original-quant rollback**.
- `quantized-draft-readback` is the measured faster **validated opt-in**, and was
  retained on the rental at closeout. It requires an **isolated Python shadow with
  an unmerged runtime change**, the original compiled extension, and the round6
  entrypoint/launcher. It is **not** made the recipe default while that dependency
  remains unpublished; there is no invented runtime pin or bootstrap here.
- `dynamic-balanced` explicitly opts out to the old corrected **BF16** drafter.
  All eight old profile names remain available. Missing dependencies fail; there
  is **no silent BF16 fallback**.

```bash
# Local, non-executing preview; requires no rental files:
DRY_RUN=1 bash DGX-Spark/serve-uma-rental.sh
DRY_RUN=1 PROFILE=quantized-draft-readback bash DGX-Spark/serve-uma-rental.sh

# Installed rental only: stop the owned server cleanly and ensure requests are idle.
PROFILE=quantized-draft bash DGX-Spark/serve-uma-rental.sh
# Only after verifying the isolated runtime dependency described below:
PROFILE=quantized-draft-readback bash DGX-Spark/serve-uma-rental.sh
# Explicit old-BF16 opt-out:
PROFILE=dynamic-balanced bash DGX-Spark/serve-uma-rental.sh
```

Deployment artifact mappings (operator records, not files provided by this clone):

```text
quantized-draft:
  /workspace/mimo-tune/start_france_quant_dflash4_dynamic06_chunk4096.py
quantized-draft-readback:
  /workspace/mimo-tune/round6-quant-readback/start_quant_readback.py
dynamic-balanced (BF16):
  /workspace/mimo-tune/start_france_dflash7_dynamic06_chunk4096.py
```

The readback wrapper checks the shadow source and original DSO hashes before
executing the installed launcher. These identify measured deployment bytes, not a
published runtime revision; other deployed dependencies must already be installed.
No runtime patch is shipped in this recipe. The runtime change belongs in the fork.

| Dependency | Measured SHA256 |
|---|---|
| Isolated block-sparse Python source | `61653605b432350db3cccddbaa00b12f121e38fa2b072fc9de2032adf9e2a706` |
| Original compiled extension (no rebuild) | `02b0ae5bc8414d335facca41083f561cf24f41d56ef6e1e73a8b41fb16f80207` |

## Fixed setup and workload

Target 2.50 bpw; Q4 paged KV; context 4096; prefill chunk 4096; GPU split 106;
`-ndt 7 -dds -dc 0.6`; batch/active requests 1; serial verification
(`EXL3_BATCH_VERIFY=0`). The native diffusion block remains eight rows. Each clean
load received the same **ordered cold code/prose pair plus three warm pairs**,
greedy temperature 0, seed 42, max 320 tokens. The adaptive state depends on request
history; do not compare unordered traffic to these means.

Decode and acceptance are native `usage.decode_tok_s` and `usage.draft_accept`.
Prefill is prompt tokens divided by native `usage.time_prefill`, not client TTFT.
Full prompts, complete response text, usage counters, cold/warm order, launch flags,
and source-file checksums are in the tracked JSON records linked below. No target
Hub revision was established by this export; deployment identity is not a new
fresh-clone reproducibility claim.

## Round5: only the drafter path changed

[Tracked round5 record](../bench/france-round5-quant-drafter.json).
Original Python/runtime/kernel, target weights, Q4 cache and flags stayed fixed;
only `-dm` changed. This was **A/B, not A/B/A**.

| Drafter | Workload | Cold tok/s | Three warm tok/s | Warm mean tok/s | Mean warm acceptance |
|---|---|---:|---|---:|---:|
| Corrected BF16 | Code | 34.00 | 37.29, 37.25, 37.14 | 37.226667 | 0.841216 |
| EXL3 4.0 bpw | Code | 29.50 | 40.62, 40.54, 40.54 | 40.566667 | 0.854671 |
| Corrected BF16 | Prose | 20.68 | 19.68, 19.60, 19.55 | 19.610000 | 0.590000 |
| EXL3 4.0 bpw | Prose | 24.57 | 23.84, 23.85, 23.75 | 23.813333 | 0.649746 |

Warm code **+8.9721%**, prose **+21.4346%**. Cold code was slower.
**Only 3/8 final texts match: all three warm-code responses** (same completion
counts); cold code and all four prose texts differ. Acceptance differs in all eight
pairs. Code is bounded same-output evidence; prose is a **same-prompt operating
profile comparison**, not identical-work evidence. Proposal token IDs and target
logits were not captured, so the numerical cause is not established. This does not
verify a universal greedy-identity promise.

The requested drafter is
[vcruz305/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw/tree/d50ead3c6a3dec221e9a595fbdc103ef60db594e),
pinned to **d50ead3c6a3dec221e9a595fbdc103ef60db594e**. The saved download/loader
checks verified **13 files**, **735,141,944 weight bytes**, and already-correct
`tap_shift=0` and mask contents. **Do not reapply the BF16 repair** or execute the
repository's remote `auto_map` code. The installed native loader was used, with
no config/weight edits. This is a published component, not a BF16 substitution.

## Round6: quantized drafter fixed, handled-readback A/B/A

[Tracked round6 record](../bench/france-round6-quant-readback.json).
A and A2 are original-quant; B uses the isolated shadow and
`EXL3_MOE_MIXEDK_ELIDE_HANDLED=1`. Only unused handled-set host bookkeeping was
elided. The original DSO, target/drafter weights, Q4, all serving flags and serial
verification stayed fixed. No changed kernels or phased-dispatch promotion.

| Arm | Workload | Cold tok/s | Three warm tok/s | Warm mean tok/s |
|---|---|---:|---|---:|
| A | Code | 35.02 | 40.92, 40.82, 40.80 | 40.846667 |
| B | Code | 35.39 | 41.39, 41.24, 41.25 | 41.293333 |
| A2 | Code | 35.13 | 40.75, 40.76, 40.69 | 40.733333 |
| A | Prose | 24.76 | 24.00, 23.80, 23.97 | 23.923333 |
| B | Prose | 25.14 | 24.41, 24.17, 24.30 | 24.293333 |
| A2 | Prose | 24.69 | 23.89, 23.85, 23.87 | 23.870000 |

B versus the A/A2 mean (**40.790000 code / 23.896667 prose**) gives
**+1.23396% code / +1.65992% prose**. These are small observed deltas on this ordered
workload, not a general speed guarantee. **All 8 corresponding texts, completion
counts and acceptance ratios match across A/B/A2.** This bounded equality does not
certify overall quality or universal identity. Warm code hits the intentional
320-token cap; cold code contains malformed quotation marks in example assertions.
Neither is claimed as a complete valid-code quality pass.

## Fresh prefill and bounded controls

One fresh **3527-token** MAPLE recall request per load, all correct. No cached
repeat or 4096-token chunk-boundary claim; **no prefill improvement established**
from this small set, and no broad prefill gain from the round5 pair.

| Round / arm | Native prefill tok/s |
|---|---:|
| Round5 BF16 | 694.5062 |
| Round5 quantized | 703.3778 |
| Round6 A | 709.7169 |
| Round6 B | 702.8642 |
| Round6 A2 | 708.9306 |
| Round6 final retained profile | 675.7882 |

Arithmetic controls returned 323 for 17 times 19; the quantized round5 profile also
passed exact string-copy and parsed-JSON controls. These do not erase the historical
LARCH-copy miss or establish broad quality. No pack fidelity or RTX SixCat score is
inherited by these Q4-KV profiles.

## Diagnostics and memory caveats

The separate round6 diagnostic load **failed on missing `telemetry_core` before
any diagnostic request or profiler capture**. Restoration succeeded. Dependency
closure was subsequently staged and CPU-preflighted, **but never retried on the
model**. There is no new target/drafter cost split, native-q counter, host-wait,
kernel-time or DRAM attribution. Older BF16 profiler results are historical, not
measurements of this quantized profile. CPU-only count-readback candidates are not
working profiles and are not promoted here.

The same guard retained the **8 GiB host/cgroup reserve**, compound physical-pressure
stop and **12-hour cap**. It polls: mitigation, **not an OOM guarantee**. The round6
captured minimum availability was 19.3046 GiB and physical free 1.2913 GiB; recorded
host/cgroup OOM counters were zero. Historical closeout found one healthy server,
zero requests, all 678 original runtime/guard/launcher/extension file hashes
unchanged, and all 13 drafter-file hashes unchanged. These are saved observations,
not a new live check performed by this local recipe update.

## Offline verification

The export checked all **90 round5 / 65 round6 source-manifest entries** before
selecting the public record fields. Only serving measurements and configuration
are retained, not model payloads, runtime implementation or internal build inputs.
JSON is tracked under bench, not the ignored results/JSONL locations. The records
preserve exact prompt/text/usage values; summaries can be recomputed without a GPU:

```bash
python DGX-Spark/test_quant_records.py -v
python DGX-Spark/test_operator_wrapper.py -v
python tools/check_repo.py
```

These checks validate records, wrapper dry runs and repository hygiene only. They
do **not** claim a hosted test, a fresh deployment, or runtime portability.
