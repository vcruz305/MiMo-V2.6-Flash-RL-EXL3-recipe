# TabbyAPI route: MiMo-V2.6-Flash-RL in EXL3 — serving recipe

The second, self-contained route in this repository: the **same fork runtime**
([vcruz305/exllamav3](https://github.com/vcruz305/exllamav3)) and the **same pack** as the native
`/v1` route at the repository root, with **[TabbyAPI](https://github.com/theroyallab/tabbyAPI)**
at the tip of `main` as the server, in the same venv. Nothing here is a rewrite of the native
route: one venv, one runtime, one pack, two servers. Which pack to download is the root README's
[Quants](../README.md#quants) section; a re-quantization is not this route's business.

> Both routes are measured, on the same pack (2.22 bpw), the same context (65,536), and the same
> SixCat 0.7.0 `speed` suite. SixCat's speed runs are `unscored: true` (synthetic prompts, speed
> only). TabbyAPI's streamed responses did not include the engine's own timing fields, so the
> Tabby prefill and TTFT figures below are client-observed. Acceptance is from TabbyAPI's
> per-request `draft N/M` log lines, not from SixCat.
>
> | Measurement | TabbyAPI route | native `/v1` route |
> |---|---:|---:|
> | Decode without dflash | **46.62 tok/s** p50 | 47.63 tok/s p50 |
> | Decode with dflash | **185.52 tok/s** p50 | 184.46 tok/s p50 |
> | Prefill without dflash | **2,051.7 tok/s** p50 (client-observed) | 2,521.4 tok/s p50 |
> | Prefill with dflash | **1,906.9 tok/s** p50 (client-observed) | 2,404.4 tok/s p50 |
> | Balanced TTFT without dflash | **4,574 ms** p50 @ C=8 | 4,357 ms p50 @ C=8 |
> | Balanced TTFT with dflash | **2,613 ms** p50 @ C=4 | 2,009 ms p50 @ C=4 |
> | Decode aggregate @ C=8, without dflash | **208.6 tok/s** | 208.2 tok/s |
> | Decode aggregate @ C=8, with dflash | **318.0 tok/s** | 336.1 tok/s |
> | Draft acceptance | **0.946** mean (n=199; 6.62 accepted per 7-token block) | 0.945 mean (n=209) |
> | Load time | 12.4 s without dflash · 13.0 s with dflash | ~90 s |
> | TabbyAPI commit | `f07131cd8fe34e449fe87cdd3a066b52b96d3cac` (`main` at measurement) | — |
> | Pack rung used | 2.22 bpw | 2.22 bpw |
>
> Schema check on that commit: `TabbyConfigModel.model_validate` accepted the rendered config,
> unknown keys none. Greedy (temperature 0, seed 7, 200 tokens) through TabbyAPI with the
> corrected drafter was byte-identical to the native draft run (sha256 `6832e8992b88746b`) and,
> like that run, diverges from the no-draft baseline at character 59. The no-draft TabbyAPI run
> was byte-identical to the native no-draft baseline (sha256 `9ad711c7af0fcb22`). Resident VRAM
> while serving, from `nvidia-smi`: 89,095 MiB without dflash, 93,217–93,931 MiB with dflash.

## Quick start

```bash
bash exllamav3-tabby/setup.sh                 # runtime + pack + drafter, then TabbyAPI main
bash exllamav3-tabby/setup.sh --check          # runtime, TabbyAPI commit, pack, drafter, config validation
bash exllamav3-tabby/serve.sh                  # DRAFT=1: 185.52 tok/s p50, OpenAI API on 127.0.0.1:8096/v1
DRAFT=0 bash exllamav3-tabby/serve.sh          # the no-draft row: 46.62 tok/s p50
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

- **TabbyAPI is deliberately unpinned** (`TABBY_REF=main`). The run above was commit
  `f07131cd8fe34e449fe87cdd3a066b52b96d3cac`. Re-run `setup.sh` to update; a newer `main` can
  move a config key, which `--check-config` is there to catch.
- **TabbyAPI's GPU extras are not installed.** Its `cu12`/`cu13` extras pull exllamav3 and torch
  wheels that would shadow the fork; `setup.sh` installs TabbyAPI's base package into the recipe
  venv and re-verifies the imported `exllamav3` afterwards, reinstalling the fork wheel if
  TabbyAPI dragged a stock one back in. It does install `uvloop` on Linux: `main.py` imports it
  unconditionally, and a plain install of the base package does not, which aborts startup.
- **The rendered config validated** against `common.config_models.TabbyConfigModel` on that
  commit (`model_validate` accepted every value; unknown keys none). `serve.sh` runs the same
  check before it execs.
- **Fork knobs with no TabbyAPI config key** are exported from the environment by
  [`env.sh`](env.sh). That block is empty on purpose: both measured routes were taken with no
  `EXL3_*` override, so leaving them unset keeps this route comparable to the native one.
- **The drafter is wired.** `DRAFT=1` (the default) sets `draft_mode: model`, depth 7, pointing
  at the corrected drafter copy. `DRAFT=0` sets `draft_mode: disabled`. Acceptance on the
  with-dflash run was 0.946 mean (6.62 of 7 drafted tokens), the same band as the native route.

## Why this route exists

The native route and this one land on the same decode: 46.62 vs 47.63 tok/s without dflash,
185.52 vs 184.46 with it. TabbyAPI is the route to run when you want its OpenAI surface
(including `/v1/completions`), admin endpoints, sampler overrides, and clients that assume
TabbyAPI. Prefill through TabbyAPI was slower on this measurement (client-observed 2,051.7 vs
2,521.4 tok/s without dflash). Same weights, same runtime, different server.

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
