# TabbyAPI route: MiMo-V2.6-Flash-RL in EXL3 — serving recipe

The second, self-contained route in this repository: the **same fork runtime**
([vcruz305/exllamav3](https://github.com/vcruz305/exllamav3)) and the **same pack** as the native
`/v1` route at the repository root, with **[TabbyAPI](https://github.com/theroyallab/tabbyAPI)**
at the tip of `main` as the server, in the same venv. Nothing here is a rewrite of the native
route: one venv, one runtime, one pack, two servers. Which pack to download is the root README's
[Quants](../README.md#quants) section; a re-quantization is not this route's business.

> ## Measurement in progress
>
> **The numbers for this route are not in yet.** The table below is deliberately empty. What is
> known and measured lives in the [root README](../README.md) and belongs to the *native* route:
> 47.63 tok/s p50 decode without a drafter and 184.46 tok/s p50 with the corrected DFlash drafter.
> Do not copy those figures into the table below, and do not describe this route as measured
> until it has been served and benchmarked on the card.
>
> | Measurement | TabbyAPI route | (native route, for orientation only) |
> |---|---:|---:|
> | Decode without dflash | *measurement in progress — to be filled in* | 47.63 tok/s p50 (no draft) |
> | Decode with dflash | *measurement in progress — to be filled in* | 184.46 (draft fixed, the 3.87× drafter fix) |
> | Prefill, single stream | *measurement in progress — to be filled in* | 2,521.4 tok/s p50 |
> | TTFT, idle, ~126-token prompt | *measurement in progress — to be filled in* | ~0.33 s |
> | Balanced-profile TTFT | *measurement in progress — to be filled in* | 4,356 ms p50 (no draft) / 2,009 ms (draft fixed) |
> | Aggregate output at `C=8` | *measurement in progress — to be filled in* | 208.2 tok/s (no draft) / 336.1 (draft fixed) |
> | VRAM after load | *measurement in progress — to be filled in* | 90,627 MiB / 94,723 MiB |
> | Load time | *measurement in progress — to be filled in* | ~90 s |
> | TabbyAPI commit | *to be filled in* | — |
> | Pack rung used | *to be filled in* (expect 2.22 bpw, 87.87 GB) | 2.22 bpw |

## Quick start

```bash
bash exllamav3-tabby/setup.sh                 # runtime + pack + drafter (root setup.sh), then TabbyAPI main
bash exllamav3-tabby/setup.sh --check          # runtime, TabbyAPI commit, pack, drafter, config validation
bash exllamav3-tabby/serve.sh                  # OpenAI-compatible API on 127.0.0.1:8096/v1
bash exllamav3-tabby/chat.sh                   # readiness + the same two prompts the native route uses
```

`DRY_RUN=1 bash exllamav3-tabby/serve.sh` prints the command line and the rendered config path
without starting anything. Only one of the two routes should be serving the card at a time — they
use the same port, 8096, on purpose.

## Files

| Path | What it is |
|---|---|
| [`env.sh`](env.sh) | Paths and pins from the root [`env.sh`](../env.sh), TabbyAPI's repo/ref/dir, context+batch values, `check_config` |
| [`setup.sh`](setup.sh) | Runs the root `setup.sh` (runtime, pack, drafter), then clones and installs **TabbyAPI `main`, unpinned**, in the same venv. `--check`, `--check-config`, `--from-source` |
| [`serve.sh`](serve.sh) | Renders [`tabby-config.yml`](tabby-config.yml) into the state dir, validates it, and execs `TabbyAPI/main.py --config` |
| [`chat.sh`](chat.sh) | Readiness poll plus the native route's two prompts, so the two are comparable |
| [`tabby-config.yml`](tabby-config.yml) | The config, with every unvalidated or unmeasured key marked |

## What is pinned and what is not

- **TabbyAPI is deliberately unpinned** (`TABBY_REF=main`), the same way the sibling
  Qwen3.8-Flash-Next recipe tracks it. Re-run `setup.sh` to update.
- **TabbyAPI's GPU extras are not installed.** Its `cu12`/`cu13` extras pull exllamav3 and torch
  wheels that would shadow the fork; `setup.sh` installs TabbyAPI's base package into the recipe
  venv and re-verifies the imported `exllamav3` afterwards, reinstalling the fork wheel if
  TabbyAPI dragged a stock one back in.
- **The config is not yet validated against TabbyAPI `main` on this box.** `setup.sh
  --check-config` loads the rendered YAML with TabbyAPI's own schema class; the folder README
  says "measurement in progress" until that passes and a run has been recorded.
- **Fork knobs with no TabbyAPI config key** are exported from the environment by
  [`env.sh`](env.sh). That block is empty on purpose: the measured native numbers were taken with
  no `EXL3_*` override, so leaving them unset keeps this route comparable to the native one.
  Candidates to try while tuning are listed there as comments, with the note that none of them is
  measured for this pack on this card.
- **The drafter is not wired into this route yet.** The MiMo drafter is a separate DFlash model
  directory rather than an MTP head, and both the key name and whether TabbyAPI applies the
  `tap_shift`/`mask_embedding` corrections the same way are open. Until it is confirmed,
  `tabby-config.yml` has the `draft_model:` block commented out and this route runs draft-free.

## Why this route exists

The native route is the measured one and is what almost everyone should run. TabbyAPI is worth
having for what it brings on top: streaming with a fuller OpenAI surface (including
`/v1/completions`), admin and model-management endpoints, sampler overrides, its own
tool/reasoning handling, and clients that assume TabbyAPI. Same weights, same runtime, different
server.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no TabbyAPI at ...` | run `bash exllamav3-tabby/setup.sh` |
| `is not the vcruz305 fork runtime with MiMo-V2 support` | a stock `exllamav3` is imported; re-run `setup.sh` (it reinstalls the fork wheel when TabbyAPI drags one in) |
| `could not locate TabbyAPI's config schema` | TabbyAPI moved its config module; extend the candidate list in [`env.sh`](env.sh) and re-run `--check-config` |
| `schema mismatch: ...` | a key in `tabby-config.yml` is not in TabbyAPI main's schema. Check the `UNVERIFIED` markers in the file — that is exactly what they are for |
| `error: port 8096 is taken` | the native route is probably up; only one route at a time |
| `error: N MiB already resident on the GPU` | stop the other server first; this recipe wants the whole card |
| replies truncated | the client set `max_tokens` |
