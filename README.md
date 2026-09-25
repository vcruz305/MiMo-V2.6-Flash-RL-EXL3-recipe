# MiMo-V2.6-Flash-RL EXL3 on one 96 GB card

Serves **MiMo-V2.6-Flash-RL** in EXL3 on a single 96 GB-class GPU, with the
[vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) fork as the runtime. The pack is 48
layers (9 full attention, 39 sliding-window at window 128), 256 routed experts, top-8 sigmoid
routing.

**Tested on:** NVIDIA RTX 6000 (96 GB) — this is one of the cards the recipe was tested on, and
every measured number below comes from that card.

**Two routes, one measured.** The **native `/v1` server** in [`server/`](server/) is the path
whose numbers are measured here and it is what the quick start below runs:
**184.46 tok/s p50 single-stream decode** with the corrected DFlash drafter, **47.63 tok/s**
without it. The **TabbyAPI route** is a first-class second path in
[`exllamav3-tabby/`](exllamav3-tabby/README.md) — same fork runtime, same pack, TabbyAPI at the
tip of `main` — and **its measurement is in progress**: that folder's results table says
"measurement in progress, numbers to be filled in" and nothing there is quoted as if it had
been run here.

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
| Pack served for the numbers | **2.22 bpw**, 87.87 GB, 24 files — **publication pending** |
| Pack published today | **2.50 bpw**, 98.48 GB, 26 files — **does not fit a 96 GB card** |
| Context served | 65,536 tokens per request, 65,536-token KV pool, 16 requests generating at once |
| Decode without dflash | **47.63 tok/s** p50 |
| Decode with dflash | **184.46 tok/s** p50 (3.87×) — from the corrected DFlash drafter fix |
| VRAM after load | 90,627 MiB (no drafter) · 94,723 MiB (with drafter) |
| Max usable concurrency | 8 |
| TabbyAPI route | [measurement in progress](exllamav3-tabby/README.md) |

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
| [`exllamav3-tabby/`](exllamav3-tabby/README.md) | **Second, first-class route:** the same fork under TabbyAPI. Own env/setup/serve/chat and config; measurement in progress |

## Quick start

On the box that owns the card, from a clone of this repo:

```bash
# 1. Runtime + drafter + pack (pack download needs PACK_DIR to exist, or see Downloads)
bash setup.sh

# 2. Read-only check: card free, pack intact, drafter wired, port free
bash preflight.sh

# 3. Serve. OpenAI-compatible /v1 on 127.0.0.1:8096 (184.46 tok/s p50 decode)
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
sliding-window layers are bounded by their 128-token window. The authority on card fit is the
measured resident total above. `serve.sh` prints the sizing and refuses combinations that cannot
work (`CACHE_SIZE < MAX_SEQ_LEN`, `MAX_ACTIVE_REQUESTS > AUTOSPLIT_MAX_BATCH`).

Do not raise `MAX_SEQ_LEN` past 65,536 expecting a measured result. The checkpoint's own
position limit is 1,048,576, but nothing above 65,536 has been validated for this recipe.

### Running it in the background

```bash
screen -dmS mimoserve bash -c 'PORT=8096 bash serve.sh > ~/mimo-exl3/state/serve.log 2>&1'
screen -ls
tail -f ~/mimo-exl3/state/serve.log      # "Native API listening 127.0.0.1:8096; max_active=16"
```

The server binds the port *before* it loads weights, so an occupied port aborts instead of
displacing a running server. The first load takes about 90 seconds.

## Don't (the short list)

- **Don't install a stock exllamav3** (PyPI or upstream) as the runtime. It has no MiMo-V2
  architecture and no DFlash port. `source env.sh && verify_runtime` must print
  `exllamav3 1.5.1.post1 (fork) at ...`; if it does not, the scripts exit rather than load.
- **Don't serve the 2.50 bpw pack on a 96 GB card.** It is 98.48 GB; the smaller rung already
  residents 90,627 MiB.
- **Don't attach the drafter without `tools/fix_dflash.py`.** An unfixed drafter measures ~0.04
  acceptance and is slower than no drafter.
- **Don't edit the original `dflash/` folder.** The fixer copies first and refuses `--src == --dst`.
- **Don't loosen the template check beyond whitespace normalization.** See
  [Template gotcha](#template-gotcha-it-bites-everyone).
- **Don't raise the context past 65,536** and quote it as measured, and don't set
  `MAX_ACTIVE_REQUESTS` above `AUTOSPLIT_MAX_BATCH` (the server refuses that combination).
- **Don't start either route while another process holds the card.** Over ~2000 MiB resident,
  stop and find out whose run it is.
- **Don't add a number to this README you did not measure** on the pack named in the table header.

## Downloads

| Artifact | Where | Size | Fits 96 GB |
|---|---|---|---|
| EXL3 pack, **2.50 bpw** | [vcruz305/MiMo-V2.6-Flash-RL-EXL3](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3) (`2.50bpw/`) | 98.48 GB, 26 files | **no** (measured load of the 87.87 GB rung is already 90.6 GiB) |
| EXL3 pack, **2.22 bpw** — the servable rung | same repo, not published yet | 87.87 GB, 24 files | **yes** |
| DFlash drafter | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (`dflash/`) | 2.94 GB + `mask_embedding.pt` + config | — |
| Runtime wheel | [vcruz305/exllamav3 releases](https://github.com/vcruz305/exllamav3/releases/tag/v1.5.1.post1) | per python/torch row | — |

```bash
# published pack (2.50 bpw) - a >96 GB card or a second machine
hf download vcruz305/MiMo-V2.6-Flash-RL-EXL3 --include "2.50bpw/*" \
  --local-dir "$RECIPE_HOME/models/MiMo-V2.6-Flash-RL-EXL3"

# the drafter comes from the ORIGINAL checkpoint, not from the pack
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --include "dflash/*" \
  --local-dir "$RECIPE_HOME/models/MiMo-V2.6-Flash-RL"
```

The **2.22 bpw rung is the one that fits one 96 GB card** and is the one every number in this
README was measured on. Its publication to the public pack repo is still pending, so this repo
does not link it: point `PACK_DIR` at it (or set `PACK_SUBDIR=2.22bpw` once it lands and let
`setup.sh` fetch it).

**Quality of the two rungs.** Both were scored with the same harness on held-out text and on an
independent text set (top-1 agreement with the full-precision model, and KL divergence):

| Rung | Size | Files | Held-out top-1 | Held-out KLD | Independent top-1 | Independent KLD |
|---|---:|---:|---:|---:|---:|---:|
| 2.50 bpw (published) | 98.48 GB | 26 | 83.76% | 0.2055 | 87.30% | 0.1086 |
| 2.22 bpw (servable) | 87.87 GB | 24 | 79.07% | 0.3010 | 84.35% | 0.1595 |

The quantization method is **SAGE**, which stores different parts of the model at different bit
widths instead of one width everywhere; the published 2.50 bpw rung carries 6 head bits.
Evaluation text was not used for calibration. The method itself, its knobs and its per-layer maps
are not published here.

## Measured results (native `/v1`, 2026-09-25)

> **Hardware:** the NVIDIA RTX 6000 (96 GB) we measured on, x86_64 host.
> **Software:** [`server/serve_native.py`](server/serve_native.py) on `vcruz305/exllamav3`
> (`93e58ca`, which is in the `v1.5.1.post1` lineage), pack 2.22 bpw, 65,536-token pool,
> 16 concurrent, greedy (temperature 0).
> **Harness:** SixCat 0.7.0 `speed` — [`tools/sixcat_speed.sh`](tools/sixcat_speed.sh) is the
> command. Each column is one curve of 4 levels plus a 32-request confirmation run.

| Measurement | No drafter | Corrected DFlash drafter |
|---|---:|---:|
| Decode, single stream (`C=1`) | **47.63 tok/s** p50 (max 47.68) | **184.46 tok/s** p50 (max 184.91) — **3.87×** |
| Prefill, single stream | **2,521.4 tok/s** p50 (max 2,526.1) | 2,404.4 tok/s p50 (max 2,445.6), −4.6% |
| TTFT, idle, ~126-token prompt | ~0.33 s | ~0.36 s |
| TTFT, idle, ~388-token prompt | ~0.59 s | ~0.62 s |
| Balanced-profile TTFT (`C=8` / `C=4`) | 4,356 ms p50 · 4,376 p95 · 4,499 p99 | 2,009 ms p50 · 2,458 p95 |
| Draft acceptance | — | **0.945 mean**, 6.62 accepted tokens per 7-token block (n=209) |
| VRAM after load | 90,627 MiB | 94,723 MiB |

**Throughput under load** (decode profile: 32-word prompt, 512 max tokens, aggregate output
across the level, prefill and queueing included):

| Concurrency | 1 | 2 | 4 | 8 | confirmation `C=8` |
|---|---:|---:|---:|---:|---:|
| No drafter | 46.3 | 73.7 | 131.7 | 208.2 | 213.7 |
| Corrected DFlash drafter | 163.0 | 171.5 | 252.4 | 336.1 | 335.1 |

Max usable concurrency is **8** in both configurations (the level where ≥95% of requests
succeed). The eight-stream per-stream rate is ~26 tok/s without the drafter and ~48 tok/s with
it, so the aggregate is queue throughput rather than eight interactive sessions.

**What the drafter is worth.** With the wiring fixed, acceptance is 0.945 and single-stream
decode is 3.87× the draft-free figure; prefill pays 4.6% for the drafter's extra forward pass,
and TTFT at load drops because the drafter shortens generation. Before the fix the same pack and
the same arguments measured a mean acceptance of **0.0418** (n=202) and single-stream decode of
~7 tok/s — the drafter was drafting, being rejected, and paying for both.

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

**Confirming it worked.** `chat.sh` prints `draft_accept` for every prompt, and it is the
network-visible form of the fix:

| Wiring | Mean `draft_accept` | Accepted per 7-token block | Single-stream decode |
|---|---:|---:|---:|
| shipped `config.json`, no `mask_embedding` shard | 0.0418 (n=202, min 0.0095, max 0.0946) | 0.29 | ~7 tok/s p50 |
| `tap_shift: 0` + the shard in place | **0.9452** (n=209, min 0.310, max 1.0) | **6.62** | **184.46 tok/s p50** |

If acceptance comes back near 0.04 rather than near 0.94, `tools/verify_dflash.py` names which
of the two edits is missing.

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

- **Greedy output is not bit-identical between the draft and draft-free configurations.** With
  the drafter attached, a greedy (temperature 0) 200-token generation diverges from the
  draft-free run at character 59, even though each configuration reproduces itself exactly
  (repeat runs are byte-identical within a configuration). The reason is mechanical: the
  speculative verification window takes different kernels than 1-row decode steps, so the
  result is distributionally equivalent but not bit-identical. Nothing was changed to hide it.
- **The servable pack rung is not published yet.** 2.22 bpw is what fits the card and what was
  measured; today's public download is 2.50 bpw, which does not fit one 96 GB card.
- **Only 65,536 tokens of context are validated**, on a model whose position limit is 1,048,576.
  No long-context retrieval test has been run for this recipe.
- **The drafter costs prefill** (−4.6%) and adds ~4 GiB resident. On a short-prompt workload it
  is a large win; on a prefill-heavy one it is not.
- **The TabbyAPI route is unmeasured here.** Its README says so, and its config has not been
  validated against TabbyAPI `main` on this box yet.
- Single runs, one card: the tables are one curve plus one confirmation run per configuration,
  not a campaign. Treat differences below a few percent as noise.
- One card, no tensor parallelism: this recipe is TP1 by construction.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error: ... is not the vcruz305 fork runtime with MiMo-V2 support` | a stock exllamav3 is being imported (or an old venv). Re-run `bash setup.sh`, then `bash setup.sh --check` |
| `RuntimeError: Model file and embedded tokenizer templates disagree` | something is serving this pack with a strict template gate; use `serve.sh` (see [Template gotcha](#template-gotcha-it-bites-everyone)) |
| `error: port 8096 is taken` | another server is up. `screen -ls`, or `PORT=8097 bash serve.sh` |
| `error: N MiB already resident on the GPU` | this recipe wants the whole card; stop the other process (make sure it is not someone else's run) |
| `error: no pack at ...` | set `PACK_DIR=/path/to/pack`, or see [Downloads](#downloads) |
| `error: .../config.json has tap_shift=None` | the drafter copy is unfixed; `python tools/fix_dflash.py` |
| acceptance near 0.04 instead of 0.94 | the drafter is attached but unfixed; `python tools/verify_dflash.py --dir "$DRAFT_DIR"` names the missing edit |
| `error: MAX_ACTIVE_REQUESTS (16) > AUTOSPLIT_MAX_BATCH (8)` | lower `MAX_ACTIVE_REQUESTS` or raise `AUTOSPLIT_MAX_BATCH`; the server refuses this combination |
| load refuses for memory | confirm the rung: the 2.50 bpw pack does not fit. Check `nvidia-smi` for another process |
| replies truncated | the client set `max_tokens`; the server honours it |

## Related repositories

| Repo | Role |
|---|---|
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | the runtime fork: MiMo-V2 architecture, the DFlash draft port, released wheels |
| [vcruz305/MiMo-V2.6-Flash-RL-EXL3](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3) | the pack (2.50 bpw published today) |
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
