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
