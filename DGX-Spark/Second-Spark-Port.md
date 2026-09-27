# Second Spark: porting the France profile to spark-724a

Record of bringing the retained France profile (**2.50 bpw** target, EXL3 4.0 bpw drafter, Q4 KV,
chunk 4096, handled-readback elision) up on a second, owned GB10 box, written 2026-09-27. The
operator scripts that ran are in [spark724a/](spark724a/README.md); the raw records are in
[trials/spark724a-port/](trials/spark724a-port/MANIFEST.json). Numbers are single-stream, native
`/v1`, server-reported counters unless a line says otherwise.

**Result:** on the France protocol this box measured **43.95 tok/s warm code / 23.72 tok/s warm
prose**, fresh prefill **653 tok/s** at 3527 tokens (recall correct), against France's retained
41.29 / 24.29 / 703. The greedy texts are **not** the same as France's (see
[Result](#result-france-protocol)), so this is a same-prompt operating-profile comparison, not
identical work. **60 tok/s is not reached here either.**

## The box

| | |
|---|---|
| Host | `spark-724a`: GB10, 121.7 GiB unified (`MemTotal` 127,598,836 kB), sm_121 (CC 12.1), aarch64, 20 cores |
| OS / driver | Ubuntu 24.04.5 LTS, kernel `7.0.0-1019-nvidia`, NVIDIA open module 580.178.04, CUDA 13.0 (`nvcc` V13.0.88) |
| Disk | NVMe root, 3.5 TB (3.2 TiB) free |
| Python stack | system Python 3.12.3; venv with torch **2.11.0+cu130**, triton 3.6.0 — the same torch row as the France rental |
| Runtime | `vcruz305/exllamav3` branch `feat/mixedk-handled-readback-elision` at **`249f22a`**, source build for `TORCH_CUDA_ARCH_LIST=12.1`, plus one local budget-code patch ([below](#2-exl3_uma-cannot-read-the-root-cgroup)) |
| Server | this repo's [server/serve_native.py](../server/serve_native.py) at `79a9b5c` |
| Target | `vcruz305/MiMo-V2.6-Flash-RL-EXL3` `2.50bpw/`, fetched from `main` on 2026-09-26 (Hub `main` is `ad3a4f0b` at the time of writing; its last change predates the fetch). 26 files, 13 shards, 98.40 GB, every file sha256-checked against the Hub tree API |
| Drafter | `vcruz305/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw` rev `d50ead3c`, 13 files, `tap_shift` 0, mask embedding present — the France drafter |

Served flags and env are the France retained profile unchanged:

```text
serve_native.py -m <pack> -dm <drafter> -ndt 7 -dds -dc 0.6 -gs 106 -cs 4096 -cq 4 -ambs 1
  -chunk_size 4096 -ccs 0 -rcs 0.25 --max-active-requests 1 --max-pending-requests 2
  --max-model-len 4096 --host 127.0.0.1 --port 8096 --request-timeout 600 -lv
EXL3_UMA=1 EXL3_UMA_RESERVE_MB=8192 TORCH_CUDA_ARCH_LIST=12.1 OMP_NUM_THREADS=8
EXL3_BATCH_VERIFY=0 EXL3_MOE_MIXEDK_ELIDE_HANDLED=1
```

One difference from France: there the elision was a Python shadow tree over pin `ca4a880e` with the
original compiled extension; here it is the installed tree of the fork branch, built on this box, so
no shadow and no France extension hash apply.

## Bring-up runbook

Order that worked, everything under `~/mimo-exl3`, no sudo except where step 3 says so.

1. **Provision** — [provision_spark724a.sh](spark724a/provision_spark724a.sh): venv, torch
   2.11.0 from the cu130 index, `transformers`, `safetensors`, hf client, fork clone, extension
   build. The fork clone failed once on an HTTP/2 stream reset (`curl 92`); a re-run cloned it.
   Also `pip install jsonschema` into the same venv: [server/protocol.py](../server/protocol.py)
   imports it and neither this script nor the recipe's `setup.sh` installs it (launch attempt 1
   below).
2. **Build the extension without root** — see [the build section](#building-without-root).
3. **Python headers for Triton's JIT** — Triton compiles a small C helper at first use with
   `-I/usr/include/python3.12`, the *system* include path, so the venv header copy from step 2 does
   not reach it. `libpython3.12-dev` and `python3.12-dev` were installed with apt (root) at 22:38
   (the box's dpkg log); that cleared launch attempt 3. Untested root-free alternative: export
   `CPATH` pointing at the venv's include dir in the server env.
4. **Fetch the pack and drafter** — [fetch_spark724a.sh](spark724a/fetch_spark724a.sh), not
   `hf download` (see [Downloads](#downloads-and-the-network)).
5. **Patch the UMA budget for a bare-metal cgroup root** —
   [memory.py.root-cgroup-fix.patch](spark724a/memory.py.root-cgroup-fix.patch), applied to the
   fork checkout (`git apply`): +18 / −1 lines in the fork's [util/memory.py](https://github.com/vcruz305/exllamav3/blob/249f22a153ce0fec9711ec7400ee8fd870958972/exllamav3/util/memory.py).
6. **Guard** — [guard_uma.py](spark724a/guard_uma.py), the France guard plus a portability overlay
   ([diff](spark724a/guard_uma.spark724a.diff)).
7. **Evict the pack from the page cache**, then launch through
   [wait_and_launch_spark724a_v2.sh](spark724a/wait_and_launch_spark724a_v2.sh) /
   [launch_spark724a.py](spark724a/launch_spark724a.py). The eviction was a one-off
   `posix_fadvise(DONTNEED)` over the weight files, equivalent to:

   ```bash
   ~/mimo-exl3/venv/bin/python - <<'PY'
   import glob, os
   for p in glob.glob(os.path.expanduser("~/mimo-exl3/models/*/**/*.safetensors"), recursive=True):
       fd = os.open(p, os.O_RDONLY)
       try:
           os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
       finally:
           os.close(fd)
   PY
   grep -E 'MemFree|MemAvailable|^Cached' /proc/meminfo   # Cached should now be a few GiB
   ```

8. **Benchmark** with [bench_spark724a.py](spark724a/bench_spark724a.py) (the France ordered
   workload and the 3527-token fixture [dynamic-long-fixture.json](spark724a/dynamic-long-fixture.json)).

### Building without root

At build time the box had no `Python.h` and no sudo. The extension build needed three fixes, one
failed build each (build logs on the box, 19:56–20:02):

| build | failure | fix |
|---|---|---|
| 1 | `Python.h: No such file or directory`, and torch's `cpp_extension.py` printed *"Attempted to use ninja … could not find ninja … Falling back to using the slow distutils backend"* | headers into the venv; **put the venv's `bin/` on `PATH`** — `ninja` was installed in the venv but not on the build's `PATH`, so the build silently used the slow backend |
| 2 | `cpython/pymem.h: No such file or directory` | copy the whole `python3.12` header tree, including `cpython/` and `internal/` |
| 3 | `aarch64-linux-gnu/python3.12/pyconfig.h: No such file or directory` | add the multiarch `pyconfig.h` under the venv include dir |
| 4 | `Successfully built exllamav3` (`1.5.1.post1`, 185 ninja steps) | — |

Equivalent commands (the headers come from the Ubuntu package, extracted, never installed):

```bash
V="$HOME/mimo-exl3/venv"
cd "$(mktemp -d)"
apt-get download libpython3.12-dev            # download only; no root
dpkg-deb -x libpython3.12-dev_*.deb x
cp -r x/usr/include/python3.12/. "$V/include/"    # Python.h, cpython/, internal/
mkdir -p "$V/include/aarch64-linux-gnu/python3.12"
cp x/usr/include/aarch64-linux-gnu/python3.12/pyconfig.h "$V/include/aarch64-linux-gnu/python3.12/"

cd ~/mimo-exl3/exllamav3
PATH="$V/bin:$PATH" CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=16 \
  "$V/bin/python" -m pip install -e . --no-build-isolation
```

The first line of the build log should show ninja steps (`[1/185] …`); a build that prints the
distutils fallback warning is the slow path even if it eventually succeeds.

## Launch attempts: what failed and why

Five launches through the guard, each with its own run directory. Every one had host and cgroup
OOM-kill deltas of **0**; there was no kernel OOM at any point. The launch config (identical for all
five apart from the run directory), each attempt's guard result and server log, the full memory trace
of attempt 4 and the first 180 s of attempt 5's are in
[trials/spark724a-port/](trials/spark724a-port/MANIFEST.json).

| # | stopped by | elapsed | cause | fix |
|---|---|---:|---|---|
| 1 | server exit 1 | 0.25 s | `ModuleNotFoundError: No module named 'jsonschema'` | `pip install jsonschema` (4.26.0) |
| 2 | server exit 1 | 3.8 s | `EXL3_UMA: required telemetry unavailable: /sys/fs/cgroup/memory.current` | the root-cgroup patch below |
| 3 | server exit 1 | 4.5 s | Triton's `cuda_utils.c`: `fatal error: Python.h: No such file or directory` (gcc with `-I/usr/include/python3.12`) | system `python3.12-dev` (step 3 above) |
| 4 | guard `memory_pressure` | 19.6 s | PSI `full avg10` 10.55 > 10 while the load allocated with the pack still in page cache | evict the pack before launch |
| 5 | — (loaded, health 200) | — | — | — |

### 2. `EXL3_UMA` cannot read the root cgroup

On this kernel the process's cgroup-v2 walk ends at the hierarchy root, and the root has no
`memory.current` or `memory.max` — the kernel does not create them there, because the root cannot be
limited. The fork's UMA budget treated every level of the walk as required telemetry and failed
closed before loading a byte. The patch treats **only the last level, and only when both files are
absent**, as unlimited; every other level and any partial or garbled telemetry still fails closed.
It touches the budget code only — no kernel, no model code. Attempt 5's log shows it in effect:

```text
EXL3_UMA device=0 mode=use request_bytes=113816633344 current=23068672 host_available=126448357376
  cgroup_headroom=unlimited device_headroom=130638139392 os_reserve=8589934592 ... fraction=0.8712586
```

This is a local patch, not on any fork branch yet. It belongs in the fork with a CPU test that
stubs the root layout.

### 4. The pack in page cache plus the load's allocations trip the pressure stop

After a fresh download the pack sits in page cache (`Cached` 100.4 GiB at launch; `MemAvailable`
still 118.6 GiB because clean cache is reclaimable). The load then allocates its ~98 GB of
weights, and the kernel has to reclaim that cache while the allocations proceed. In 18.3 s the
guard saw `MemAvailable` 118.6 → 35.2 GiB, `Cached` 100.4 → 33.3 GiB and PSI `full avg10`
0 → **10.55**, and its `psi_full10 > 10` rule stopped the load. Nothing was killed by the kernel.

With the weight files evicted first (step 7), attempt 5 launched at `Cached` 1.6 GiB and loaded
with PSI `full avg10` peaking at **0.28** (`MemAvailable` 118.9 -> 21.8 GiB by 40 s, 20.2 GiB by 60 s,
where it stayed), 9/9 warm-up passes in 24.35 s, then `/health` 200. The allocator then held 98.86 GB. Across the
57-minute session the minimum sampled `MemAvailable` was 19.68 GiB. A later relaunch (after the
TensorFold baseline) started with 20.7 GiB of unrelated files in cache and also loaded cleanly: peak PSI
0.32, minimum `MemAvailable` 19.53 GiB, no OOM kills (`relaunch-guard-result.json`).

France never hit this because its launcher ran long after the download; on a fresh box the
eviction is part of the launch.

### The guard's portability overlay

The France guard (`guard_uma.py`, sha256 `eef5706b…d24d`) assumed a cgroup-v2 root with
`memory.current`/`memory.max`/`memory.stat` and a lock file under `/workspace/mimo-tune`. The
spark-724a copy (sha256 `321cfb6d…1626`) changes two things, both recorded in the box's
`PROVENANCE.txt`:

1. the lock path comes from the launch config (`lock_path`);
2. `cgroup_dir()` walks from the process's own cgroup up to the first directory that carries memory
   accounting. If none does, the sample is labelled `cgroup_accounting: false` and the host's
   `MemAvailable` is used as the headroom budget; the floors stay active and no limit is invented.

In every run here the walk found an accounted ancestor (`cgroup_accounting: true` in all samples),
so the host fallback never engaged. Floors, the PSI rule and the 12-hour cap are unchanged.

## Downloads and the network

| path | measured |
|---|---|
| `hf download` with hf_xet (first pack pull, unauthenticated) | froze at 6/26 files, 0 B in 30 s after ~6.3 GB; killed |
| `hf download` with hf_xet (later probes, authenticated, `HF_XET_HIGH_PERFORMANCE=1`) | 0 bytes in 90 s, empty xet debug log; one A/B with hf_xet 1.6.1a0 finished the 735 MB drafter weights at 6.5 MB/s while wget on the same link held 11.4 MB/s |
| [fetch_spark724a.sh](spark724a/fetch_spark724a.sh): `wget -c`, 4 streams, size + sha256 gate | 98.4 GB in 2 h 31 min, **~10.5–10.8 MB/s aggregate**, ~2.3–2.6 MB/s per stream |
| `HF_HUB_DISABLE_XET=1 hf download --max-workers 8` (TensorFold's Qwen control model, later) | **59 MB/s** measured live; the pull log's wall times give 64–68 MB/s per repo (3.85 GB in 59 s, 16.08 GB in 3 min 54 s) |

What the network actually is, measured afterwards:

- Ethernet `enP7s7` (r8127, 1000 Mb/s full duplex) is the primary default route (metric 100); WiFi
  `wlP9s9` is the backup (metric 600). Per-NIC byte counters during a 100 MB download: ethernet
  103 MB, WiFi 0 MB.
- Single-stream HTTPS download **26.5 MB/s (~212 Mbit/s)**. Ookla: 152.9 Mbit/s down / 12.4 Mbit/s
  up — below what HTTPS downloads achieved, so it is not the ceiling. HTTPS POST upload ~18 Mbit/s.
- **The pack fetch ran over WiFi.** The ethernet cable came up at 22:29 but got no DHCP lease until
  22:43, when NetworkManager made it the default route (journal); the fetch finished at 22:33. The
  ~10.5 MB/s aggregate is a WiFi-path figure, and the per-stream ~2.5 MB/s was recorded at the time
  as an HF CDN per-connection limit — not re-tested over ethernet.
- The ethernet link dropped for ~3 s twice after coming up (22:52, 23:02; kernel `link down` /
  `link up`). Long transfers should stay resumable.

### Correction

A working note during the port put the slow fetch down to an **"80 Mbit/s WAN cap". That was
wrong.** The same line later carried 212 Mbit/s on one HTTPS stream and ~470 Mbit/s (59 MB/s) with
8 workers. The slow fetch was the WiFi path at the time plus a single-digit number of streams, and
the xet stalls were the xet client, not the line.

## Result (France protocol)

[bench_spark724a.py](spark724a/bench_spark724a.py) replays the France round-6 workload exactly: the
same two prompts (byte-identical to [the France record](../bench/france-round6-quant-readback.json)),
greedy, seed 42, max 320 tokens, thinking off, one load, ordered cold code / cold prose then three
warm pairs; decode and acceptance are the server's `usage.decode_tok_s` / `usage.draft_accept`,
prefill is `prompt_tokens / time_prefill`. France reference is round-6 arm B, the retained profile.

| Measurement | spark-724a | France (round 6, B) | Δ |
|---|---:|---:|---:|
| Warm code, mean of 3 | **43.953333** (44.06, 43.92, 43.88) | 41.293333 | +6.44 % |
| Warm prose, mean of 3 | **23.720000** (23.73, 23.73, 23.70) | 24.293333 | −2.36 % |
| Cold code | 21.97 | 35.39 | — |
| Cold prose | 23.81 | 25.14 | — |
| Warm code acceptance | 0.851 | 0.855 | — |
| Warm prose acceptance | 0.591 | 0.650 | — |
| Fresh prefill, 3527 tokens | **653.25 tok/s**, `MAPLE-9362` correct | 702.86 tok/s | −7.06 % |
| Arithmetic control (17 × 19) | 323, correct | 323 | — |
| Minimum sampled `MemAvailable` during the bench | 20.19 GiB | 19.30 GiB | — |

Read it with these caveats:

- **Different text, different work.** Warm code here stops at **289** tokens (`stop`) and starts
  with a fenced block; France's warm code runs to the **320** cap (`length`) with different text from
  the first token. Warm prose is 252 vs 257 tokens and diverges after ~160 characters. Same prompt,
  same flags, different runtime build (fork branch build on this box vs France's pin + shadow +
  original extension) — so the ±% above compare operating profiles, not identical requests.
- **Warm prose is slower because acceptance is lower** (0.591 vs 0.650) on a different completion;
  warm code is faster at nearly the same acceptance on a shorter completion.
- **Cold code is slow here** (21.97 tok/s, TTFT 5.0 s of which 3.2 s is prefill of a 58-token prompt)
  — the first request after load pays one-time compilation. Warm reps are the comparable figure.
- One load, one ordered workload. No A/B/A on this box, no quality pass, no SixCat.

The raw per-request records (full prompts, full texts, usage, memory samples) are
[trials/spark724a-port/](trials/spark724a-port/MANIFEST.json); `summary.json` carries the SHA-256 of
the responses file it was computed from.

## Not claimed

- No 60 tok/s. No aggregate or batched throughput. No token-identity with France.
- The root-cgroup `memory.py` patch and the guard overlay are local to this box; neither is on a
  published branch yet.
- The recipe's own `setup.sh`/`serve.sh` were **not** used or validated on this box; the scripts in
  [spark724a/](spark724a/README.md) are records of what ran, with this box's layout baked in.
- The root-free `CPATH` route for Triton's JIT headers is untested.
