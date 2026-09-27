# spark-724a operator scripts (as installed)

The scripts that brought the retained France profile up on the second Spark, committed as
records of what ran — **not a portable bootstrap**. They assume the layout
`$HOME/mimo-exl3/{venv,exllamav3,repo,models,guard,runs,logs,state}` and a system Python 3.12.
The runbook that explains the order, the failures and the fixes is
[Second-Spark-Port.md](../Second-Spark-Port.md).

| file | role |
|---|---|
| [provision_spark724a.sh](provision_spark724a.sh) | venv, torch 2.11.0 (cu130), hf client, fork clone at `feat/mixedk-handled-readback-elision`, `sm_121` extension build. No sudo. |
| [fetch_spark724a.sh](fetch_spark724a.sh) | resumable per-file `wget -c`, 4 streams, size + sha256 gate against the Hub tree API, for the 2.50 bpw pack and the EXL3 4.0 bpw drafter. |
| [wait_and_launch_spark724a_v2.sh](wait_and_launch_spark724a_v2.sh) | waits for the fetcher's sha256 verdicts, re-checks every size against the Hub, launches, waits for health, one smoke request. |
| [launch_spark724a.py](launch_spark724a.py) | the France launcher adapted to this box: same server, flags and env, fail-closed preconditions, runs the server under the guard. |
| [guard_uma.py](guard_uma.py) | the UMA memory guard with the spark-724a portability overlay; [guard_uma.spark724a.diff](guard_uma.spark724a.diff) is the diff from the France original. |
| [memory.py.root-cgroup-fix.patch](memory.py.root-cgroup-fix.patch) | the local exllamav3 change (budget code only) that lets `EXL3_UMA` run on a bare-metal cgroup-v2 root. Applies cleanly to fork commit `249f22a`. |
| [bench_spark724a.py](bench_spark724a.py) | the France ordered workload and 3527-token prefill probe, so the numbers compare with France. |
| [dynamic-long-fixture.json](dynamic-long-fixture.json) | the prefill probe's prompt and expected answer (`MAPLE-9362`); the same prompt as the France records in `bench/`. |
| [tf_bench_mimo.py](tf_bench_mimo.py) | TensorFold's `bench_openai.py` client pointed at our serve (see [TensorFold-Baseline.md](../TensorFold-Baseline.md)). Needs TensorFold's [tools/bench_openai.py](https://github.com/ashhart/TensorFold/blob/2f8e514b0b7d615df7c971627ce3c0fb7e55d93a/tools/bench_openai.py) at `2f8e514` next to it as `tf_bench_openai.py` (not vendored here). |
| [tensorfold_pull.sh](tensorfold_pull.sh) | pulls TensorFold's Qwen3.8-27B control model and its DFlash2 drafter with xet disabled and 8 workers (deployed as `~/tensorfold/pull.sh`). |
| [tensorfold_serve27b.sh](tensorfold_serve27b.sh) | starts `tensorfold serve` natively on CUDA for that model (deployed as `~/tensorfold/serve27b.sh`). |

## Differences from the deployed bytes

[MANIFEST.json](MANIFEST.json) lists, per file, the sha256 of the deployed copy and of the committed
copy. All deployed copies were already LF. The hashes differ for four files, and only by these
edits:

- `provision_spark724a.sh`: the box's LAN address in the header comment replaced by `<lan-ip>`;
- `launch_spark724a.py`, `memory.py.root-cgroup-fix.patch`, `wait_and_launch_spark724a_v2.sh`: the
  operator's tool name removed from comments (one each in the two launch scripts, two in the patch; in
  `launch_spark724a.py` the marker now reads `SPARK724A_DRAFT_NAME_FIX`). No code line changed.

The two TensorFold scripts are byte-identical to the deployed copies and have only been renamed.
`guard_uma.spark724a.diff` is derived from the two guard versions and was never deployed.

The patched `memory.py` that actually ran has sha256
`9d8f95257c1d01c1d074e3cdf0d95d627baa4e329f2e63e3726c7261eca01d6b` (fork `249f22a` plus the deployed
patch); the committed patch applied to `249f22a` gives
`78314795f9e967b03c99b9d5d4284fca42a08618ca64a85fc9cf3f6c6158b8b1`, differing only in the two marker
comments.
