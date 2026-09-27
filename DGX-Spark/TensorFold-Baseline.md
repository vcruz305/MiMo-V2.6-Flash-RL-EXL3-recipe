# TensorFold baseline on spark-724a

The first TensorFold measurement on our own GB10 box, taken 2026-09-26 (23:48–23:56 local) on
spark-724a right after the [Second-Spark-Port.md](Second-Spark-Port.md) bring-up. It had two
purposes. The first was to show that TensorFold's published single-Spark numbers reproduce on this box
without NVIDIA's container. The second was to put MiMo, served by our exllamav3 stack, through the same
client, so that a later MiMo port to TensorFold has a fixed number to beat. The raw records are in
[trials/tensorfold-baseline/](trials/tensorfold-baseline/MANIFEST.json).

**Result:**

- Qwen3.8-27B + DFlash2 on TensorFold measured **49.9 / 45.8 / 49.5 / 46.1 tok/s** (code sampled /
  chat sampled / code greedy / chat greedy). TensorFold's README gives 49.6 / 45.8 / 49.2 / 45.9 for one
  Spark, so every cell is within 0.7 %.
- MiMo on our exllamav3 serve, through the same client, measured **32.9 / 24.6 / 32.0 / 24.6 tok/s**.
- **TensorFold cannot load MiMo yet.** The blockers are listed [below](#what-blocks-mimo-on-tensorfold).

## Setup

| | |
|---|---|
| Host | spark-724a, as in [Second-Spark-Port.md](Second-Spark-Port.md#the-box) (GB10, driver 580.178.04, CUDA 13.0, kernel `7.0.0-1019-nvidia`) |
| TensorFold | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) **v0.3.4, `2f8e514`** (MIT). Our fork `vcruz305/TensorFold` had the same head when this was measured |
| Python stack | TensorFold's own venv (system Python 3.12.3), **reusing the MiMo venv's torch 2.11.0+cu130 and triton 3.6.0** |
| Target model | `Vontra/Qwen3.8-27B-MLX-4bit` rev `70ae7fac` |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` rev `50307d4c`, picked up by TensorFold's default `--drafter auto` |
| Client | TensorFold's [tools/bench_openai.py](https://github.com/ashhart/TensorFold/blob/2f8e514b0b7d615df7c971627ce3c0fb7e55d93a/tools/bench_openai.py) at `2f8e514` (sha256 `39149b94…ceafa4`) |

### Native, without NVIDIA's container

TensorFold's README says to run the CUDA engine inside `nvcr.io/nvidia/pytorch:26.07-py3`. On this box
it ran natively instead. It got its own venv, with an editable install of the checkout and a `.pth` file
that adds the MiMo venv's `site-packages` to its path. The CUDA toolkit, torch, triton, numpy,
tokenizers, safetensors, jinja2 and the HF client all come from the stack that was already built and
tested for MiMo. The venv's own `site-packages` holds only pip, TensorFold and the `.pth` files.
The equivalent commands:

```bash
git clone https://github.com/ashhart/TensorFold.git ~/tensorfold/src
git -C ~/tensorfold/src checkout 2f8e514
python3 -m venv ~/tensorfold/venv
echo "$HOME/mimo-exl3/venv/lib/python3.12/site-packages" \
  > ~/tensorfold/venv/lib/python3.12/site-packages/zz_mimo_torch.pth
~/tensorfold/venv/bin/pip install -e ~/tensorfold/src     # its dependencies are already satisfied via the .pth
```

Then pull and serve with the two scripts in [spark724a/](spark724a/README.md):

- [tensorfold_pull.sh](spark724a/tensorfold_pull.sh): `HF_HUB_DISABLE_XET=1 hf download --max-workers 8`.
  xet stalls on this box; see [Downloads](Second-Spark-Port.md#downloads-and-the-network).
- [tensorfold_serve27b.sh](spark724a/tensorfold_serve27b.sh): `tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit
  --backend cuda --host 127.0.0.1 --port 8080` with `TORCH_CUDA_ARCH_LIST=12.1`. It also sets a private
  `TORCH_EXTENSIONS_DIR` for the one extension TensorFold compiles at first start, and turns off the
  update check.

The server loaded in 32.8 s with drafts on and TensorFold's default sampling (T 1.0, top-k 20, top-p
0.95). It ran under the same UMA guard as MiMo. Only one model was resident at a time, because the
runs were sequenced: the MiMo serve was stopped before this one started and relaunched after it. The
guard's lock did not enforce that, since the two configs name different lock files. The guard stopped
TensorFold on request after 175.7 s. The lowest `MemAvailable` it saw was 91.8 GiB, and both OOM-kill
deltas were 0.

The first guard start failed, because the guard takes a JSON config rather than a command line. The
config it then ran with is [qwen27b-guard-config.json](trials/tensorfold-baseline/qwen27b-guard-config.json).

## Protocol

The TensorFold protocol, which is how its README's DGX Spark table was measured:

- one stream, 64-token replies with `ignore_eos`, thinking off for chat;
- one warm-up request per cell, then 5 requests with seeds 1234–1238, and the **median** is reported;
- sampled means T 1.0, top-k 20, top-p 0.95; greedy means T 0;
- decode tok/s = (completion tokens − 1) / (time of the last streamed token − time of the first),
  measured by the client;
- two prompts. The code prompt ("Write a short Python function that computes the Fibonacci sequence and
  explain it.") is sent raw to `/v1/completions`. The chat prompt ("Explain how matrix multiplication
  uses a GPU …") goes to `/v1/chat/completions`.

The Qwen run used `bench_openai.py` unchanged, against `http://127.0.0.1:8080`.

The MiMo run used [tf_bench_mimo.py](spark724a/tf_bench_mimo.py). It imports `bench_openai.py`'s own
`stream()` and `PROMPTS` (a byte-identical copy after LF normalization) and runs the same cells against
this repo's [server/serve_native.py](../server/serve_native.py), running the retained France profile at
`127.0.0.1:8096`. There is **one forced deviation**: `serve_native.py` has no `/v1/completions`, so
the code prompt goes through chat with thinking off, and its row is labelled `fibonacci-raw-as-chat`.
For each cell the script also sends 3 non-streamed requests and records the server's own
`usage.decode_tok_s` and `usage.draft_accept`. The server was the attempt-5 load from the port, already
warm from the France-protocol bench.

## Results

### Qwen3.8-27B + DFlash2 on TensorFold

| Cell | spark-724a (median of 5) | the 5 requests | TensorFold README, 1 Spark | Δ |
|---|---:|---|---:|---:|
| Code, sampled | **49.92** | 49.92, 87.15, 57.47, 49.19, 46.09 | 49.6 | +0.65 % |
| Chat, sampled | **45.77** | 45.77, 46.03, 34.39, 53.02, 40.52 | 45.8 | −0.06 % |
| Code, greedy | **49.48** | 49.42, 49.53, 49.48, 49.60, 49.47 | 49.2 | +0.57 % |
| Chat, greedy | **46.06** | 46.06, 46.15, 46.03, 46.23, 46.01 | 45.9 | +0.34 % |

The median TTFT was 0.087–0.110 s. The sampled cells vary widely between seeds because each seed drafts
a different text. The greedy cells are within ±0.25 %.

**Exactness:** TensorFold claims that each drafted number is byte-identical to its own serial
decoding. This session checked that claim for this model by comparing the SHA-256 of the drafted and
serial outputs, and they were equal. That comparison's raw output is not among the committed records.

This run does **not** reproduce the README's "vs vLLM" column. vLLM was not run on this box.

### MiMo on the exllamav3 serve, same client

| Cell | client decode tok/s (median of 5) | the 5 requests | server `decode_tok_s` (3 requests) | server `draft_accept` | TTFT median |
|---|---:|---|---|---|---:|
| Code (as chat), sampled | **32.89** | 37.58, 32.73, 32.14, 32.89, 39.51 | 33.95, 34.09, 31.61 | 0.521, 0.588, 0.595 | 0.351 s |
| Chat, sampled | **24.56** | 22.95, 24.56, 28.26, 25.47, 22.66 | 25.41, 25.41, 27.91 | 0.547, 0.547, 0.519 | 0.433 s |
| Code (as chat), greedy | **32.00** | 31.90, 32.00, 33.80, 30.50, 41.28 | 32.28, 24.78, 33.47 | 0.592, 0.456, 0.627 | 0.355 s |
| Chat, greedy | **24.61** | 24.64, 24.60, 24.61, 24.57, 24.77 | 22.82, 23.19, 23.26 | 0.339, 0.353, 0.353 | 0.476 s |

How to read the two tables together:

- **They measure different models.** One is a dense 27B at MLX 4-bit, the other a 48-layer MoE at 2.50
  bpw. What is shared is the box, the client, the prompts and the arithmetic. The MiMo row is the
  number a MiMo port to TensorFold has to beat under this protocol.
- **The MiMo code row is not comparable with the France-protocol 43.95 tok/s.** That figure used a
  different code prompt, up to 320 tokens and greedy decoding without `ignore_eos`, with warm-code
  acceptance of 0.851. Here the reply is forced to 64 tokens of a Fibonacci answer sent as chat, and
  acceptance was 0.46–0.63.

## What blocks MiMo on TensorFold

A header scan of the served pack (`vcruz305/MiMo-V2.6-Flash-RL-EXL3` `2.50bpw/`), read-only, compared
with TensorFold at `2f8e514`:

1. **No family.** TensorFold chooses a family package by `config.json` `model_type` and refuses any
   type it has no recipe for before it reads a weight. At `2f8e514` the families are `qwen3_5`,
   `qwen4_exp`, `nemotron_h` and `glm5_next`. MiMo is `model_type: mimo_v2` (`MiMoV2ForCausalLM`),
   with 48 layers: one dense-MLP layer and 47 MoE layers of 256 routed experts each.
2. **Its EXL3 reader is GLM-only.** EXL3 is read solely by the `glm5_next` family, for
   `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`. That family pins the variant to
   `{"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}` and refuses anything else.
   Its decoder implements the mcg codebook, and every other tensor is expected in BF16. Our pack
   differs on all three counts:
   - **codebook `mul1`**, in every trellis tensor;
   - **mixed 2–8 bits** across tensors, and mixed within a single layer's expert set;
   - **every linear is quantized**: routed experts, attention q/k/v/o, the dense MLP and the
     6-bit `lm_head`. Only the router gates and `embed_tokens` are unquantized.
3. **The drafter is not DFlash2-as-published.** Our drafter is a 5-layer DFlash model quantized to
   EXL3 4.0 bpw (`mul1`) with a separate learned mask-embedding shard, block 8, `tap_shift` 0. It reads
   target hidden states after layers 0, 11, 23, 35 and 47. TensorFold's DFlash path serves z-lab's
   DFlash2 checkpoints.

## Plan

In order. Each step lands as a PR to the TensorFold fork first, and upstream where it applies:

1. **A generic EXL3 reader and kernel.** It should cover the 3inst, mcg and mul1 codebooks, 1–8 bits
   set separately for each tensor, and any linear rather than only routed experts. It needs a
   grouped-expert path that takes a different bitrate for each expert in one launch. It should be
   checked numerically against exllamav3's own reference on real tensors from this pack, and offered
   upstream as a replacement for the GLM-only variant check. This change is not specific to MiMo.
2. **A `mimo_v2` family**, meaning the forward pass and weight loading through that reader. The gate
   is token identity with the exllamav3 serve on a fixed greedy prompt set, before any speed number
   is quoted.
3. **A DFlash port.** This means loading our EXL3 4.0 bpw drafter and its mask shard in TensorFold's
   drafter interface, fed with the target's hidden states at the five taps for every committed token.
   The gate is TensorFold's own exactness rule: drafted output byte-identical to serial.
4. **Measure** with `bench_openai.py` against the MiMo row above (32.9 / 24.6 / 32.0 / 24.6), and with the
   France protocol against 43.95 / 23.72 tok/s and 653 tok/s prefill
   ([Second-Spark-Port.md](Second-Spark-Port.md#result-france-protocol)).

## Not claimed

- MiMo on TensorFold. Nothing in the plan is built or measured here.
- A model-to-model speed comparison. The two tables use different models and are a harness baseline.
- The README's vLLM figures, and any result for two Sparks.
- Anything beyond one load per server and one 5-seed pass per cell.
