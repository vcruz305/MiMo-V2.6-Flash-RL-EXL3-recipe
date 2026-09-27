# Device-trial records

Verbatim artifacts from the bounded, guarded trials run on the France GB10 rental, plus the
parent (reviewer) readback and review for each. Nothing here is a serving recipe; these are
the evidence behind the verdicts in [France-Optimization-Campaign.md](../France-Optimization-Campaign.md).

Each trial stopped the owned serve through its guard, ran one bounded workload, restored the
retained launcher, and was re-verified read-only afterwards (pointer, exact PID/starttime pair,
health, 678 runtime + 13 drafter + 3 server-source hashes, loaded extension hash, OOM
counters). The readback records below are those verifications.

| directory | what it records |
|---|---|
| `kernel-three-stage/` | the opt-in three-stage mixed-K kernel: numeric trial (exact) and timing trial (**regression — not deployed**). |

## Second Spark (spark-724a)

Records from the owned second box. Each directory has a `MANIFEST.json` with the same fields: sha256,
bytes and lines of each committed file. They also carry the sha256 of the copy read back from the box,
and say what was changed, if anything. These are bring-up and baseline runs, and not the
stop/restore trials described above.

| directory | what it records |
|---|---|
| `spark724a-port/` | the bring-up: five guarded launch attempts (four failures, their fixes) and the France-protocol bench on the fifth. See [Second-Spark-Port.md](../Second-Spark-Port.md). |
| `tensorfold-baseline/` | the first TensorFold baseline: TensorFold's `bench_openai.py` against TensorFold (Qwen3.8-27B + DFlash2) and against our MiMo serve. See [TensorFold-Baseline.md](../TensorFold-Baseline.md). |
