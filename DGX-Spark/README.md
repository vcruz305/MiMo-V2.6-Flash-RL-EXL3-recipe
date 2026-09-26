# DGX Spark / GB10 — aarch64, unified memory

One GB10-class box, measured 2026-09-25, serving the model this repo is for. This is the record for
the second row of [Cards tested on](../README.md#cards-tested-on); the numbers on the 96 GB card
live in the [main README](../README.md).

| | |
|---|---|
| Host | single GB10-class box — 121.7 GiB **unified**, sm_121, aarch64, CUDA 13.0 |
| Runtime | source build at the `EXL3_REF` default (`master`); no wheel is published for that row |
| Pack | **2.20 bpw**, the same rung as the card's numbers |
| Profile | [configs/spark-with-draft.env](../configs/spark-with-draft.env) |

```bash
# from a clone of this repo, on the box
bash setup.sh                                # runtime, drafter fix, pack
PROFILE=spark-with-draft bash serve.sh       # with-draft sizing + GPU_SPLIT=112
bash chat.sh                                 # prints finish_reason, decode tok/s, draft_accept
```

## Measured (native `/v1`, server-reported counters)

> **Harness:** this repo's own server and [chat.sh](../chat.sh) — one stream, one request at a
> time, greedy (`temperature` 0), seven prompts. `draft_accept` and `decode_tok_s` are the
> **server's own `usage` counters**, not client timing.

| Measurement | Result |
|---|---:|
| Decode without dflash, single stream | 19.5 – 19.8 tok/s (flat across every prompt) |
| Decode with dflash, single stream | 17.4 – 43.4 tok/s |
| Ratio | **1.0× – 2.2×**, prompt-dependent |
| `draft_accept` | 0.211 – 0.675 (mean ≈ 0.42) |

| Prompt (drafter attached) | Completion tokens | `draft_accept` | Decode tok/s |
|---|---:|---:|---:|
| "What is the capital of France? Answer in one short sentence." | 17 | 0.667 | 32.2 – 40.5 |
| "Write one sentence about the ocean." | 39 | 0.229 | 18.1 – 19.8 |
| "Explain why the sky is blue, in three sentences, for a ten year old." | 109 | 0.211 | 17.4 |
| "Summarize the following in two sentences: the Tal Revolt …" | 166 | 0.675 | 40.3 |
| "Write a Python function that returns the n-th Fibonacci number iteratively, then explain it." | 39 – 59 | 0.229 – 0.279 | 19.5 – 21.1 |
| "Return JSON only: a list of the five largest planets by radius …" | 59 | 0.279 | 21.1 |
| "Count from 1 to 30 in words, one per line, then say how many were even." | 512 | 0.457 – 0.627 | 30.2 – 43.4 |

Every request finished coherently (`finish_reason` `stop` or `length`) with the drafter attached, so
this is a **working** drafter — not the degenerate case described in
[The DFlash fix](../README.md#the-dflash-fix-two-edits-no-code-change).

Two things follow from the spread. The ratio moves more between **prompts** than between the two
configurations, so quote it with the prompt class attached or not at all. And a single prompt is not
a verdict: one drafter, one box, one greedy sampler, seven prompts is what this table is.

## The config that makes it load: `GPU_SPLIT=112`

`serve.sh` evicts the page cache before launching on a unified-memory host. That is necessary but
not sufficient — the load budget must also cover the loader's own measuring forward, a **~983 MiB**
transient at layer 39, plus 256 MiB of headroom:

- `GPU_SPLIT=119` → the load dies at layer 39 with **1220 MiB** free where **1239 MiB** is required;
- `GPU_SPLIT=112` → loads, reporting **~112 GiB free** on a 121.7 GiB box.

Without `-gs` at all, `model.py` applies its 0.5 GB reserve default and freezes the process at
`(mem_get_info free − 0.5) / total` — and CUDA's free **excludes the reclaimable page cache that is
holding the pack**, so that reads as a fraction of the memory the box actually has.

## What is *not* attributed

**The ratio here is below the 96 GB card's, and the cause has not been measured.** The experiment
that would attribute it — one lever at a time — was started and not completed, so these are
candidates, not findings:

- the **pack rung**: the **2.50 bpw** pack has never been measured with a drafter here;
- the **sm_121 kernel path**: the legacy grouped-coop mixed-K kernel and the graph-captured attention
  are the aarch64 side of the fork, and the drafter consumes the target's hidden states at five tap
  layers, so a different kernel path changes what it is fed.

The source **ref** is *not* a candidate. The ref the card's numbers were measured on and this host's
`master` build differ by 16 commits whose files are CI, the main README, the version module, the
*unified* mixed-K row floor, encode-side quantize hooks, and three removed editor-backup files —
nothing in the fork's DFlash port, its drafter architecture module, or its generator package. Check
that before rebuilding a runtime to chase an acceptance difference:

```bash
git log --oneline <ref>..master -- '*dflash*' 'exllamav3/generator/*' 'exllamav3/architecture/*'
```

## Measuring a drafter on this host

**Use a server, not a harness that wraps the drafter's generate generator.** A probe that wrapped
`iterate_draftmodel_dflash_gen` to dump drafted blocks reported acceptance around **0.04 – 0.16**
with garbage drafts **on a drafter `chat.sh` measures working** — the instrumentation corrupted the
path it was observing and sent a day into chasing the port.

- Quote `draft_accept` from the server's `usage` field (`bash chat.sh` prints it per prompt).
- `draft_accept` is `accepted / (accepted + rejected)` over the whole request, so it is only
  comparable between runs with the same prompt.
- If acceptance looks near zero, `python ../tools/verify_dflash.py --dir "$DRAFT_DIR"` names which of
  the two drafter edits is missing before you touch anything else.
- **One server, many requests** beats a script that loads the pack repeatedly: on a 121.7 GiB
  unified box, a loop that reloads the model per test thrashes the machine hard enough to make it
  stop answering SSH, which looks exactly like a dead remote-access tunnel.


## 2.50bpw UMA rental profiles (local deployment)

The [UMA rental profile](France-UMA.md) loaded the 2.50bpw pack on the France rental
without a host cache flush, using an opt-in host/cgroup-aware runtime budget and
an independent memory supervisor. The no-draft baseline measured 17.85–17.94 tok/s
and 264.7 prompt tok/s on a 1207-token request, with no recorded OOM kills.
Exact-copy/recall checks **failed**, so this is a working load/performance baseline,
not a quality-qualified or tuned profile. It does not replace the 2.20bpw/DFlash
measurements above. Runtime patch and launch details, raw responses, and limitations
are linked in that profile. Changes are local to this deployment, not published.

The no-draft comparison uses **Q4 paged KV cache / 1024-token prefill chunks**, still
4K context, one request, no DFlash. It measured **18.02 tok/s** for the 256-token code
request, **419.36 tok/s** for 1207-token prefill, **449.98–453.15 tok/s** on two
1209-token repeats, and **493.66 tok/s** on a 3488-token request. Minimum sampled
host availability was **20.87 GiB**, with no recorded OOM kills. Exact-copy failures
remain unresolved; these are speed measurements, not a quality pass. Both cache
precision and chunk size changed. The profile preserves the original configuration
as `PROFILE=baseline`; the no-draft Q4 profile is `PROFILE=q4-chunk1024`.

The initial DFlash profile is `PROFILE=dflash7`: corrected drafter, native ceiling 7,
Q4 paged KV and chunk 1024. Identical code prompts/caps measured **28.93 tok/s** on
both repeats, vs **17.975 tok/s** no-draft median (**+60.95%**); acceptance **0.466667**.
The explanation request measured **21.10 tok/s**, acceptance **0.312925**. Initial
1209-token prefill was **420.67 tok/s**. Available memory stayed above **18.19 GiB**
in captured load/inference samples; OOM counters remained zero. Exact-copy failures
remain unresolved. See the profile for raw data, caveats and the no-draft fallback.

After a locally deployed native-block page-reservation patch (187 CPU tests and
nine real page-boundary smoke requests), the then-default was `PROFILE=dflash4`.
Code decode measured **30.00–30.13 tok/s**, acceptance **0.664286**; explanation
**24.60 tok/s**, acceptance **0.511364**. Prefill measured **454.44 tok/s** on 1209
tokens. Q4 cache and 1024-token chunks are unchanged. Available memory stayed above
**18.18 GiB** in the captured tests, with no recorded OOM kills. Six response texts
matched ceiling 7, including the unresolved LARCH error. Short replies can be slower
with ceiling 4; the higher acceptance ratio partly reflects the shorter window.
Full evidence and fallbacks are in [the rental profile](France-UMA.md).

The sampler follow-up retained **serial verification** as the `dflash4` default:
removing stale history requirements and restricting batch eligibility passed 210
CPU tests and 13 matched HTTP/GPU replay cases, but greedy batching did not improve
speed (30.30 vs 30.365 tok/s median). `PROFILE=dflash4-batch` is the restricted
comparison option. Final serial code measured 30.00 / 30.37 tok/s; Q4 KV and
1024-token chunks remain unchanged. See [the rental profile](France-UMA.md) and
[raw sampler evidence](../bench/france-sampler-greedy.json). No quality pass or
statistically established new speedup is claimed.

### Current rental default: adaptive drafting

The installed rental wrapper now defaults to `PROFILE=dynamic-balanced`
(`-ndt 7 -dds -dc 0.6`, serial verify, **Q4 / chunk1024** unchanged).
After adaptation, the selected repeat measured code **34.82–34.85 tok/s** and
prose **23.74–23.75 tok/s**, with acceptance about **0.779 / 0.781**.
`PROFILE=dynamic-code` (target0.4) reached **37.44–37.64 tok/s** on warmed code,
but was slower on prose. Static4 remains available for cold/short requests.
Long3527-token prefill measured **468.59 tok/s** and passed MAPLE recall.
The selected run's sampled availability stayed above **18.06 GiB**, OOM counters0.
These dynamic responses differ from static output, calibration depends on earlier
requests, and the LARCH copy error remains. No token-equivalence or model-quality
claim is made. [Details and caveats](France-UMA.md),
[raw adaptive comparison](../bench/france-dynamic-draft.json).

### Q4 prefill staging follow-up

Direct in-kernel Q4 dequantization did not establish a speed gain over staged
prefill in a clean three-load comparison. The original staging default remains.
The restored profile measured **34.735 tok/s** warmed code median and **471.61
prompt tok/s** at 3527 tokens; these are workload-specific speed observations,
not a new quality pass. See [the comparison](France-UMA.md#q4-prefill-staging-comparison-localrental-only)
and [raw requests](../bench/france-q4-prefill-staging.json).

### Prefill chunk size: 4096 is the measured fast path

A clean chunk comparison (Q4 KV cache, draft7 adaptive0.6, serial verification,
unified mixed-K all fixed) found that a 4096-token prefill chunk roughly halves
long-prompt prefill time on this pack. Fresh 3527-token prefill measured
**471.6-473.5 tok/s at chunk1024**, **569.1 tok/s at chunk2048** and
**712.4-717.5 tok/s at chunk4096** (about **+51%** versus chunk1024). Warmed decode
was unchanged across every arm (code **34.6-34.9 tok/s**, prose **23.6-23.7 tok/s**),
and all five arms produced byte-identical responses and completion counts, so the
knob changes scheduling rather than results. The 1244-token prompt showed no trend
(416-438 tok/s in every arm). The `dynamic-balanced` default now selects
`-chunk_size 4096`; `PROFILE=chunk1024` preserves the previously measured path.
Minimum sampled host availability was **17.05 GiB** with host and cgroup OOM counters
at zero. These are speed observations on one rental and one prompt set: not a quality
pass, and the LARCH copy error persists in every arm.
See [the rental profile](France-UMA.md#prefill-chunk-size-comparison-localrental-only)
and [the raw comparison](../bench/france-chunk-size.json).

### Acceptance is not the objective, and the draft window has a hard cap

Two follow-ups on the adaptive draft window, both single-variable against the adopted
default:

- `-dc 0.9` (draft longer only when very confident) raised measured acceptance to
  **0.87-1.0** but cut warmed decode to **~15 tok/s** from **34.9** - a 2.3x loss. Higher
  acceptance percentage is not better throughput: with shorter windows each verify step
  confirms fewer tokens.
- `-gs 106 -> 128` changed nothing measurable (warm code 34.92-34.93 vs 34.84-34.90, prose
  23.81-23.82 vs 23.74, fresh 3527-token prefill 715.4 vs 714.5-717.5, acceptance
  identical) and was reverted.
- `-ndt 12` is **not supported** by this DFlash drafter: the pinned draft buffer is sized
  for 7 draft tokens, so the window copy raised `RuntimeError: The size of tensor a (12)
  must match the size of tensor b (7)`. The engine reported `engine_unavailable`; the
  guard and server stayed up with host OOM counters at zero. **7 is the maximum `-ndt`.**

A bounded Torch-profiler overlay (default off, two jobs per process, batch 1, prompt
<=3600, output <=24) was deployed for one capture and then removed, with the server's
`worker.py` restored byte-identically. On the adopted default the device-kernel picture is:

- **decode-heavy job:** `exl3_moe_mixedk_kernel<256,2,16,3,0>` alone accounts for
  **~62%** of summed device kernel time (165 calls, 559 ms of 901 ms); the cooperative MoE
  kernels, draft-model GEMVs and small cutlass GEMMs split the rest.
- **prefill-heavy job (3527 tokens, one chunk):** no single kernel dominates - mixed-K MoE
  shapes total **~30%**, trellis reconstruction/dequant (`reconstruct_had_batch_kernel` +
  `reconstruct_kernel`) **~14%**, cutlass GEMMs ~9%, and `_paged_attn_prefill_kernel` only
  **~3.5%**.

So decode is MoE-expert bound and prefill is MoE-plus-dequant bound; attention is not the
bottleneck at these shapes. Kernel times are device time, not wall latency, and profiling
inflates the request, so these traces identify work, not throughput. Raw record:
[`france-draft-window.json`](../bench/france-draft-window.json).
