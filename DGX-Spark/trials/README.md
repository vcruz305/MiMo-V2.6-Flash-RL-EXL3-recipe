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
