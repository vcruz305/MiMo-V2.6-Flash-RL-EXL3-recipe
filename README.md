# MiMo-V2.6-Flash-RL in EXL3 — serving recipe

Serves the **MiMo-V2.6-Flash-RL** EXL3 pack with the
[vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) fork as the runtime. Needs a single
GPU with about 96 GB of VRAM; see [Cards tested on](#cards-tested-on) for what this has actually
been verified on. The pack is 48 layers (9 full attention, 39 sliding-window at window 128), 256
routed experts, top-8 sigmoid routing.

**Two routes.** The **native `/v1` server** in [`server/`](server/) is what the quick start
below runs. The **TabbyAPI route** in [`exllamav3-tabby/`](exllamav3-tabby/README.md) is the
same fork and a second server. Speed in this README is SixCat speed on the **2.20 bpw** pack
(unscored, synthetic prompts); see [Measured results](#measured-results-native-v1-2026-09-25).
The Tabby folder's table is a different unpublished pack and is not this pack's speed.

> **Agents and automation:** hand the agent the [Don't](#dont-the-short-list) list and the
> [Troubleshooting](#troubleshooting) table below before it touches the card. The scripts check
> their own preconditions and will refuse rather than half-load a model.

## Status (2026-09-25)

| | |
|---|---|
| Model | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) |
| Pack | [vcruz305/MiMo-V2.6-Flash-RL-EXL3](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3), an EXL3 trellis build |
| Card requirement | one GPU with at least ~96 GB of VRAM, x86_64 host |
| Runtime | `vcruz305/exllamav3` release **`v1.5.1.post1`** (prebuilt wheel; source build documented) |
| Server | [`server/serve_native.py`](server/serve_native.py) — native `/v1/chat/completions`, no engine patch |
| Pack served for the numbers | **2.20 bpw**, 86.94 GB — [2.20bpw](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3/tree/main/2.20bpw) |
| Larger rung | **2.50 bpw**, 98.48 GB — **does not fit a 96 GB card** |
| Context served | 65,536 tokens per request, 65,536-token KV pool, 16 requests generating at once |
| Decode without dflash | **49.57 tok/s** p50 |
| Decode with dflash | **184.11 tok/s** p50 |
| SixCat eval, strict, 120-item default | **68.7** overall — provisional, see [SixCat eval](#sixcat-eval-default-120) |
| VRAM after load | — |
| Max usable concurrency | 8 without dflash · 8 with dflash |
| TabbyAPI route | not measured on this 2.20 bpw pack |

## Cards tested on

This recipe serves the model, not a card. It needs a single GPU with about 96 GB of VRAM, or a
unified-memory host with at least that much addressable memory. Two hosts have been served on it:

| Host | Memory | Status |
|---|---|---|
| NVIDIA RTX 6000 | 96 GB | **Verified** — every measured number in the sections below comes from this card |
| NVIDIA DGX Spark / GB10 (aarch64, sm_121) | 121.7 GiB unified | **Measured** — the drafter works, but the with/without ratio is far lower than this card's. See [Measured on a unified-memory host](#measured-on-a-unified-memory-host-aarch64) |

This table is the only place in the repository where a host model is named; each row's figures live
in the section it links to. **No other host has been measured**: on anything else, treat every
figure here as unverified until it has been served and benchmarked there, and add it to this table
when it has.

## Repository layout

| Path | What it is |
|---|---|
| [`env.sh`](env.sh) | Every path, pin and check. Sourced by the other scripts; the imported-runtime check lives here |
| [`setup.sh`](setup.sh) | Runtime (released wheel, or `--from-source`), pack download, DFlash fix. `--check` verifies an existing install |
| [`serve.sh`](serve.sh) | **The serve command.** `PROFILE=with-draft` (default) or `no-draft`, `DRY_RUN=1`, sizing sanity check |
| [`chat.sh`](chat.sh) | Readiness poll plus two sanity prompts; prints `finish_reason`, decode tok/s and `draft_accept` |
| [`preflight.sh`](preflight.sh) | Read-only: card, disk, runtime, pack, drafter wiring, port |
| [`server/`](server/) | The native `/v1` server (`serve_native.py`, `protocol.py`, `worker.py`) |
| [`configs/`](configs/) | The context/batch values actually used, one file per profile |
| [`tools/`](tools/) | [`fix_dflash.py`](tools/fix_dflash.py), [`verify_dflash.py`](tools/verify_dflash.py), [`sixcat_speed.sh`](tools/sixcat_speed.sh), [`check_repo.py`](tools/check_repo.py) |
| [`exllamav3-tabby/`](exllamav3-tabby/README.md) | **Second route:** the same fork under TabbyAPI. Own env/setup/serve/chat and config |
| [`DGX-Spark/`](DGX-Spark/README.md) | **Host notes:** the unified-memory (aarch64) host — its measured numbers, and the profile it needs to load |
| [Cards tested on](#cards-tested-on) | The one card this recipe has been verified on — the only place here a card model is named |
| [Quants](#quants) | The pack rungs, their sizes and their measured fidelity (`bpw`, top-1, KLD) |

## Quick start

On the box that owns the card, from a clone of this repo:

```bash
# 1. Runtime + drafter + pack (pack download needs PACK_DIR to exist, or see Downloads)
bash setup.sh

# 2. Read-only check: card free, pack intact, drafter wired, port free
bash preflight.sh

# 3. Serve the 2.20 bpw pack. OpenAI-compatible /v1 on 127.0.0.1:8096
bash serve.sh
```

Then:

```bash
bash chat.sh                                        # two sanity prompts against the server
curl -s http://127.0.0.1:8096/v1/models
curl -s http://127.0.0.1:8096/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "MiMo-V2.6-Flash-RL-EXL3",
  "messages": [{"role": "user", "content": "Write a haiku about a 96 GB card."}],
  "max_tokens": 256}'
```

`serve.sh` renders nothing and patches nothing: it exports the paths, checks the imported
runtime is the fork with MiMo support, checks the pack, checks the drafter, checks the card is
free and the port is open, then `exec`s the server with the measured arguments. `DRY_RUN=1 bash
serve.sh` prints the exact command and exits.

**What `setup.sh` installs, and what it refuses.** One venv under `$RECIPE_HOME` (default
`~/mimo-exl3`), the fork's released wheel if it matches your python/torch row, otherwise a
source build of the fork, and then the pack plus the DFlash drafter. Every launcher checks the
*imported* `exllamav3` before loading anything: the MiMo-V2 architecture module must import,
the DFlash port must expose `tap_shift`, and the launcher must offer `-dm`/`-cs`/`-ambs`. A
stock exllamav3 fails all three and the scripts exit with an error instead of loading half a
model.

### Sizing: context and concurrency

`-cs` (`CACHE_SIZE`) is **one KV pool shared by every concurrent request**, not a per-request
allowance — the same pool backs all 16 generating requests. `--max-model-len` caps one request.
Both are 65,536 in the measured configuration. Arithmetic from the pack's `config.json` (an
estimate, not a measurement): the 9 full-attention layers cost about 22.5 KiB per token in fp16
(4 KV heads × (192 K + 128 V) × 2 bytes), so a 65,536-token pool is ~1.4 GiB; the 39
sliding-window layers are bounded by their 128-token window. `serve.sh` prints the sizing and
refuses combinations that cannot work (`CACHE_SIZE < MAX_SEQ_LEN`, `MAX_ACTIVE_REQUESTS >
AUTOSPLIT_MAX_BATCH`).

Do not raise `MAX_SEQ_LEN` past 65,536 expecting a measured result. The checkpoint's own
position limit is 1,048,576, but nothing above 65,536 has been validated for this recipe.

### Running it in the background

```bash
screen -dmS mimoserve bash -c 'PORT=8096 bash serve.sh > ~/mimo-exl3/state/serve.log 2>&1'
screen -ls
tail -f ~/mimo-exl3/state/serve.log      # "Native API listening 127.0.0.1:8096; max_active=16"
```

The server binds the port *before* it loads weights, so an occupied port aborts instead of
displacing a running server.

## Don't (the short list)

- **Don't install a stock exllamav3** (PyPI or upstream) as the runtime. It has no MiMo-V2
  architecture and no DFlash port. `source env.sh && verify_runtime` must print
  `exllamav3 1.5.1.post1 (fork) at ...`; if it does not, the scripts exit rather than load.
- **Don't serve the 2.50 bpw pack on a 96 GB card.** It is 98.48 GB and does not fit. The pack
  this recipe serves is **2.20 bpw**, 86.94 GB.
- **Don't attach the drafter without `tools/fix_dflash.py`.** An unfixed drafter is rejected
  and is slower than no drafter.
- **Don't edit the original `dflash/` folder.** The fixer copies first and refuses `--src == --dst`.
- **Don't loosen the template check beyond whitespace normalization.** See
  [Template gotcha](#template-gotcha-it-bites-everyone).
- **Don't raise the context past 65,536** and quote it as measured, and don't set
  `MAX_ACTIVE_REQUESTS` above `AUTOSPLIT_MAX_BATCH` (the server refuses that combination).
- **Don't start either route while another process holds the card.** Over ~2000 MiB resident,
  stop and find out whose run it is.
- **Don't add a number to this README you did not measure** on the pack named in the table header.
- **Don't treat the Tabby folder's speed table as this 2.20 bpw pack.** That table is a different
  unpublished pack.

## Quants

These are the [MiMo-V2.6-Flash-RL EXL3 packs](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3). Pick one bitrate before downloading.

The following table mirrors the [Hugging Face model README](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3/blob/main/README.md). Sizes are whole folders, not just model shards; the smallest-card column is the pack's own estimate, not a promise that every card like it can serve the pack.

| bpw | size | smallest card | top-1 vs original | mean KLD | p99 KLD | download |
| --- | --- | --- | --- | --- | --- | --- |
| **2.50** | 98.48 GB | >96 GB — does not fit | 83.76% (8,577 / 10,240) | 0.2055 | 3.380 | [2.50bpw](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3/tree/main/2.50bpw) |
| **2.20** | 86.94 GB | 96 GB | 82.16% (8,413 / 10,240) | 0.19265 | 2.241 | [2.20bpw](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3/tree/main/2.20bpw) |

**Top-1** is next-token agreement with the original checkpoint on the held-out evaluation set. **KLD** is the mean/99th-percentile KL(reference ‖ pack) on those positions; lower is closer. These are quantization-fidelity measures, not task-accuracy scores. Evaluation text was not used for calibration.

On a separate independent set, not the held-out cell above: **2.20 bpw** is 82.64% (8,462 / 10,240) top-1, mean KLD 0.19861, p99 KLD 2.498. The 2.50 bpw independent figures already recorded for that rung are 87.30% top-1 and mean KLD 0.1086.

Every measured result in this repository is tied to one card, one pack and one set of server settings — see [Cards tested on](#cards-tested-on) — and does not predict another card or another pack.

## Downloads

| Artifact | Where | Size | Fits 96 GB |
|---|---|---|---|
| EXL3 pack, **2.20 bpw** — the servable rung | [vcruz305/MiMo-V2.6-Flash-RL-EXL3](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3) (`2.20bpw/`) | 86.94 GB, 24 files | **yes** |
| EXL3 pack, **2.50 bpw** | same repo (`2.50bpw/`) | 98.48 GB, 26 files | **no** |
| DFlash drafter | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (`dflash/`) | 2.94 GB + `mask_embedding.pt` + config | — |
| Runtime wheel | [vcruz305/exllamav3 releases](https://github.com/vcruz305/exllamav3/releases/tag/v1.5.1.post1) | per python/torch row | — |

```bash
# the pack that fits one 96 GB card (default PACK_SUBDIR=2.20bpw)
hf download vcruz305/MiMo-V2.6-Flash-RL-EXL3 --include "2.20bpw/*" \
  --local-dir "$RECIPE_HOME/models/MiMo-V2.6-Flash-RL-EXL3"

# 2.50 bpw does not fit a 96 GB card
hf download vcruz305/MiMo-V2.6-Flash-RL-EXL3 --include "2.50bpw/*" \
  --local-dir "$RECIPE_HOME/models/MiMo-V2.6-Flash-RL-EXL3"

# the drafter comes from the ORIGINAL checkpoint, not from the pack
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --include "dflash/*" \
  --local-dir "$RECIPE_HOME/models/MiMo-V2.6-Flash-RL"
```

The **2.20 bpw rung is the one that fits one 96 GB card** and is the one the speed table below
was measured on. `setup.sh` fetches `PACK_SUBDIR` (default `2.20bpw`).

## Measured results (native `/v1`, 2026-09-25)

SixCat speed (unscored, synthetic prompts). Not a quality score.

> **Hardware:** the card in [Cards tested on](#cards-tested-on), x86_64 host.
> **Pack:** 2.20 bpw. The served model id in the run was `MiMo-V2.6-Flash-RL-EXL3-2.20`.
> **Harness:** SixCat 0.7.0 `speed` (schema `sixcat-speed-v2`). [`tools/sixcat_speed.sh`](tools/sixcat_speed.sh)
> is the command: `--max-seconds 7200 --curve-seconds 1500`, strict policy (temperature 0, thinking off, seed 1).
> **Context:** 65,536. The server advertised `max_model_len` 65536 and `cache_size` 65536.
> **Timing:** these are **client-observed** figures. SixCat reported no per-request provider timing
> on this route, so they are not the server's own counters. Decode, prefill, and TTFT are
> separate rows so the two configurations are not crammed into one cell.

| Measurement | Result |
|---|---:|
| Decode without dflash | **49.57 tok/s** p50 (max 49.64), single stream |
| Decode with dflash | **184.11 tok/s** p50 (max 191.02), single stream |
| Prefill without dflash | **2,370.2 tok/s** p50 (max 2,375.8), single stream |
| Prefill with dflash | **2,257.7 tok/s** p50 (max 2,260.1), single stream |
| TTFT without dflash | **4,685 ms** p50 (p95 4,707 ms), SixCat summary, balanced profile |
| TTFT with dflash | **4,851 ms** p50 (p95 5,058 ms), SixCat summary, balanced profile |
| TTFT without dflash, single stream | **0.360 s** p50, decode profile, C=1 |
| TTFT with dflash, single stream | **0.400 s** p50, decode profile, C=1 |

The with-dflash / without-dflash decode ratio is **3.71×**, arithmetic on the two single-stream
p50s, not a third measurement. Max usable concurrency is **8** in both runs.

**Throughput under load**, decode profile only (32-word synthetic prompt, 512 max tokens,
aggregate output tok/s, prefill and queueing included, client-observed):

| Profile | C=1 | C=2 | C=4 | C=8 | confirmation |
|---|---:|---:|---:|---:|---:|
| Decode without dflash | 47.9 | 77.8 | 138.9 | 236.2 | 222.5 at C=8 |
| Decode with dflash | 161.7 | 170.3 | 251.9 | 330.6 | 321.9 at C=8 |

The eight-stream per-stream decode p50 on that profile is 34.95 tok/s without the drafter and
54.46 tok/s with it, so the aggregate is queue throughput rather than eight interactive sessions.
SixCat's summary TTFT is the balanced-profile confirmation, not the single-stream decode TTFT;
both are in the table above. p99 is not quoted: each confirmation has fewer than 100 requests.

## Measured on a unified-memory host (aarch64)

The second row of [Cards tested on](#cards-tested-on) has been served and measured: the drafter
works, but the with/without ratio (**1.0× – 2.2×**, prompt-dependent; `draft_accept` 0.211 – 0.675)
is far below this card's 3.71×, and the cause is **not yet attributed**. Everything for that host —
the launch profile it needs (`GPU_SPLIT=112`), the per-prompt table, and the candidates for the gap —
is in [`DGX-Spark/README.md`](DGX-Spark/README.md).

## SixCat eval (default 120)

This is the quality run, not the speed suite. Same pack, same default server (`bash serve.sh`, corrected DFlash drafter on), model id `MiMo-V2.6-Flash-RL-EXL3-2.20`.

SixCat 0.7.0, schema `sixcat-v2`, policy `strict` (temperature 0, thinking off), `--limit 20` (20 items in each of six categories, 120 total), concurrency 1. The only flag off the built-in default was `--max-minutes 0`, so the 30-minute deadline could not cut the set short. Finished 1:36 AM PDT.

| Category | Score | n |
|---|---:|---:|
| knowledge | 47.37 | 19 / 20 |
| math | 95.0 | 20 |
| truth | 80.0 | 20 |
| instruct | 50.0 | 20 |
| code | 45.0 | 20 |
| tools | 95.0 | 20 |
| **overall[strict]** | **68.7** | 119 / 120 |

SixCat printed `PARTIAL / PROVISIONAL` and said not to treat this as a complete score. Flags: `truncated:instruct`, `loop-failures:instruct`, `incomplete-scope`. The missing item is `knowledge/mmlu:4`: the generation hit the default 768-token budget (`finish_reason=length`) and the grader raised `TypeError`, so that item is unscored. Instruct has one length truncation and two loop failures at its default budget; those rows are scored and counted as fails. A second pass with the knowledge budget raised to 2048 still truncated that same item and did not change the overall.

The summary SixCat wrote is [`bench/sixcat-eval-2.20-dflash.json`](bench/sixcat-eval-2.20-dflash.json).

## The DFlash fix (two edits, no code change)

`XiaomiMiMo/MiMo-V2.6-Flash-RL` ships a DFlash draft model in `dflash/`. exllamav3's port of it
needs two corrections, and both are checkpoint/config edits, so nothing in the runtime is
patched and the fix survives a fork update that keeps the same reader keys.
[`tools/fix_dflash.py`](tools/fix_dflash.py) applies them **to a copy** — never the original
folder — and [`tools/verify_dflash.py`](tools/verify_dflash.py) checks the result on CPU.

**(a) `tap_shift`.** The port reads `dflash_config->tap_shift` (or a top-level `tap_shift`) and
adds it to `target_layer_ids` *before* the target decides which layers export a hidden state, and
its class default is **1**. The reference implementation conditions on
`hidden_states[target_layer_ids[i] + 1]` — the **output** of layer `target_layer_ids[i]` — and
exllamav3's export index `j` already denotes the output of layer `j`, so the raw ids
`[0, 11, 23, 35, 47]` are the correct taps. With the default `1` the target exports layers
`1/12/24/36/48`, the drafter's context taps sit one block too deep, and acceptance collapses.
The shipped `config.json` does not pin the key, so the wrong default applies. **Add
`"tap_shift": 0` inside `dflash_config`.**

**(b) the mask embedding.** The port loads the drafter's learned mask embedding only if the
tensor collection contains a tensor named `mask_embedding`; otherwise it embeds mask positions
from the target's own untrained embedding row for the mask token. The checkpoint ships that
vector only as the side file `mask_embedding.pt` (a dict with `mask_token_id` and a bf16
`[4096]` embedding), and the shipped safetensors index does not mention it. The collection is a
glob of `*.safetensors` in the drafter directory, so **writing the vector into a new one-tensor
shard `mask_embedding.safetensors` in the same folder is enough** — ~8 KB, and the 2.94 GB
drafter shard is not touched.

```bash
# whole flow, from the pristine dflash/ folder to a served drafter
python tools/fix_dflash.py --src "$DRAFT_SRC" --dst "$DRAFT_DIR"
python tools/verify_dflash.py --dir "$DRAFT_DIR"
PROFILE=with-draft bash serve.sh
```

**Confirming it worked.** `chat.sh` prints `draft_accept` for every prompt. The SixCat speed
runs behind [Measured results](#measured-results-native-v1-2026-09-25) did not record acceptance,
so this README does not quote an acceptance figure for the 2.20 bpw pack. If acceptance comes
back near zero, `tools/verify_dflash.py` names which of the two edits is missing.

## Template gotcha (it bites everyone)

The pack's `chat_template.jinja` and the `chat_template` embedded in its
`tokenizer_config.json` differ by **one blank line**, so a strict equality check between the two
raises

```
RuntimeError: Model file and embedded tokenizer templates disagree
```

and the server aborts at startup, before any weights load. **This is inherited from the original
checkpoint's tokenizer files, not caused by the quantization.** The workaround, exactly as used
here: **serve through this recipe's `serve.sh`**, whose server normalizes whitespace for that
comparison only. The rendered template is unchanged — the whitespace-normalized text hashes
identically on both sides (`sha256 0ba145010cf2695eb35fa4c151e824b9269c417b39b37041de5c5ba3b7919a56`),
and the server prints `Canonical chat template sha256=853650be…` (the template file's own bytes)
at startup, which is the same hash `GET /v1/models` reports as `chat_template_sha256`. A template
that is genuinely different still fails the check.

## Known limitations

- **Greedy output is not expected to be bit-identical between the draft and draft-free configurations.**
  Speculative verification uses different kernels than 1-row decode steps, so the two
  configurations can diverge even when each reproduces itself. This recipe has not re-checked
  that divergence on the 2.20 bpw pack.
- **Only 65,536 tokens of context are validated**, on a model whose position limit is 1,048,576.
  No long-context retrieval test has been run for this recipe.
- **The drafter changes prefill.** Single-stream prefill p50 is 2,257.7 tok/s with it and
  2,370.2 tok/s without, both client-observed. Resident memory with the drafter attached was
  not in the SixCat files.
- **The TabbyAPI route has not been remeasured on the 2.20 bpw pack.** The table in
  [`exllamav3-tabby/README.md`](exllamav3-tabby/README.md) is a different unpublished pack.
  Do not use it as this pack's speed.
- Single runs, one card: each speed row is one SixCat suite, not a campaign. Treat differences
  below a few percent as noise.
- One card, no tensor parallelism: this recipe is TP1 by construction.
- **2.50 bpw does not fit a 96 GB card.** It is published; it is not the pack `serve.sh` loads
  by default.
- **On the unified-memory host the with/without-drafter ratio is 1.0× - 2.2×**, not this card's
  3.71×, and it moves more between prompts than between configurations. The cause is not yet
  attributed; see [Measured on a unified-memory host](#measured-on-a-unified-memory-host-aarch64)
  for the candidates.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error: ... is not the vcruz305 fork runtime with MiMo-V2 support` | a stock exllamav3 is being imported (or an old venv). Re-run `bash setup.sh`, then `bash setup.sh --check` |
| `RuntimeError: Model file and embedded tokenizer templates disagree` | something is serving this pack with a strict template gate; use `serve.sh` (see [Template gotcha](#template-gotcha-it-bites-everyone)) |
| `error: port 8096 is taken` | another server is up. `screen -ls`, or `PORT=8097 bash serve.sh` |
| `error: N MiB already resident on the GPU` | this recipe wants the whole card; stop the other process (make sure it is not someone else's run) |
| `error: no pack at ...` | set `PACK_DIR=/path/to/pack`, or see [Downloads](#downloads) |
| `error: .../config.json has tap_shift=None` | the drafter copy is unfixed; `python tools/fix_dflash.py` |
| acceptance near zero | the drafter is attached but unfixed; `python tools/verify_dflash.py --dir "$DRAFT_DIR"` names the missing edit |
| `error: MAX_ACTIVE_REQUESTS (16) > AUTOSPLIT_MAX_BATCH (8)` | lower `MAX_ACTIVE_REQUESTS` or raise `AUTOSPLIT_MAX_BATCH`; the server refuses this combination |
| load refuses for memory | confirm the rung: the 2.50 bpw pack does not fit. Check `nvidia-smi` for another process |
| load refuses for memory on a unified-memory host (GB10 / Grace-Blackwell), well under the card size | CUDA's free excludes reclaimable page cache and `model.py` freezes the budget at process start. Use `PROFILE=spark-with-draft` (`GPU_SPLIT=112`) — see [Measured on a unified-memory host](#measured-on-a-unified-memory-host-aarch64) |
| drafter looks broken: acceptance near 0.04 with garbage drafts | if the number came from a custom harness, it may be the harness — one that wraps the drafter's generate generator corrupts the path it observes. Re-measure with `bash chat.sh`, which reads the server's own `draft_accept` |
| replies truncated | the client set `max_tokens`; the server honours it |

## Related repositories

| Repo | Role |
|---|---|
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | the runtime fork: MiMo-V2 architecture, the DFlash draft port, released wheels |
| [vcruz305/MiMo-V2.6-Flash-RL-EXL3](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3) | the pack (2.20 bpw fits a 96 GB card; 2.50 bpw does not) |
| [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | the original checkpoint, including the DFlash drafter this recipe fixes |
| [Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) | sibling recipe this one is modeled on |

## Credits and upstream work

**ExLlamaV3 and the EXL3 format** by [turboderp](https://github.com/turboderp-org/exllamav3):
the trellis format, the MCG codebook, the kernels and the quantization method are theirs. MIT,
Copyright (c) 2025 Turboderp. **TabbyAPI** by
[theroyallab](https://github.com/theroyallab/tabbyAPI) is the server the second route runs.
**MiMo-V2.6-Flash-RL** is Xiaomi's model, and its DFlash drafter is theirs.

## License

MIT for the scripts and notes in this repo (see [LICENSE](LICENSE)). Weights are **not**
redistributed here: pull them from Hugging Face and respect the upstream licenses (the pack is
MIT; the base model keeps its own).
