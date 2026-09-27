"""Measure spark-724a with the France protocol, so the numbers are comparable.

Same prompts, same caps, same 4 reps per case (rep 0 cold, 1-3 warm), same long-prefill probe
(3527 prompt tokens, expected MAPLE-9362) and the same aggregation: mean of the server's
`decode_tok_s` over the warm reps. `decode_tok_s` and `time_prefill` are server-reported, and
the server here is the recipe's own server/serve_native.py, so the metric definition matches
the France measurements rather than being re-invented client-side.
"""
import argparse, hashlib, json, os, statistics, sys, time, urllib.request
from pathlib import Path

H = Path(os.environ.get("RECIPE_HOME", Path.home() / "mimo-exl3"))
sys.path.insert(0, str(H / "guard"))
from guard_uma import memory_sample  # noqa: E402

URL = "http://127.0.0.1:8096"
FIXTURE = H / "dynamic-long-fixture.json"

CASES = [
    ("code", "Write a Python function merge_intervals(intervals) that merges overlapping closed "
             "intervals. Include a docstring explaining inputs, sorted output and time complexity, "
             "and three example assertions. Output only Python code.", 320),
    ("prose", "Explain why database indexes speed up reads but slow down writes. Use a concrete "
              "library-catalog analogy, discuss selectivity and give one practical rule for choosing "
              "a compound index. Write about 220 words.", 320),
]


def req(prompt, max_tokens=256, timeout=900):
    body = {"model": "MiMo-V2.6-Flash-RL-EXL3", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    t = time.monotonic(); first = None; text = ""; stats = {}; finish = None
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            obj = json.loads(raw)
            if "error" in obj:
                raise RuntimeError(obj["error"])
            if obj.get("usage"):
                stats = obj["usage"]
            for c in obj.get("choices", []):
                d = c.get("delta", {})
                frag = d.get("content", "") or ""
                if frag and first is None:
                    first = time.monotonic() - t
                text += frag
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    wall = time.monotonic() - t
    if not stats or finish is None:
        raise RuntimeError("no final usage/finish")
    return {"wall_s": wall, "ttft_s": first, "usage": stats, "finish": finish, "text": text,
            "prefill_tok_s": (stats["prompt_tokens"] / stats["time_prefill"]) if stats.get("time_prefill", 0) > 0 else None}


def health():
    with urllib.request.urlopen(URL + "/health", timeout=10) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    out = H / "runs" / f"bench-{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "responses.jsonl"
    assert not rpath.exists(), f"refusing to overwrite {rpath}"
    assert health().get("healthy"), "server is not healthy"
    fixture = json.loads(FIXTURE.read_text())
    rows = []
    with rpath.open("a", buffering=1) as f:
        for rep in range(a.reps):
            for label, prompt, cap in CASES:
                p = f"Round2 paired single-stream workload; ignore this marker.\n{prompt}"
                before = memory_sample()
                assert before["available_gib"] > 16, f"headroom fell to {before['available_gib']:.1f} GiB"
                rr = req(p, cap)
                rr.update(tag=a.tag, case=label, rep=rep, phase="cold" if rep == 0 else "warm",
                          prompt=p, cap=cap, mem_before=before, mem_after=memory_sample())
                rows.append(rr); f.write(json.dumps(rr) + "\n")
                u = rr["usage"]
                print(json.dumps({"case": label, "rep": rep, "decode_tok_s": u.get("decode_tok_s"),
                                  "completion_tokens": u.get("completion_tokens"),
                                  "draft_accept": u.get("draft_accept"), "ttft_s": round(rr["ttft_s"], 3),
                                  "wall_s": round(rr["wall_s"], 2)}), flush=True)
        # fresh-prefill probe (same fixture as France: 3527 prompt tokens)
        before = memory_sample()
        pf = req(fixture["prompt"], 24)
        pf.update(tag=a.tag, case="long-prefill", prompt=fixture["prompt"], expected=fixture["expected"],
                  correct=pf["text"].strip() == fixture["expected"], mem_before=before, mem_after=memory_sample())
        (out / "prefill.json").write_text(json.dumps(pf, indent=2)); f.write(json.dumps(pf) + "\n")
        print(json.dumps({"case": "long-prefill", "prompt_tokens": pf["usage"].get("prompt_tokens"),
                          "prefill_tok_s": pf["prefill_tok_s"], "correct": pf["correct"]}), flush=True)
        # arithmetic sanity: cheap and catches a broken serve
        sn = req("What is 17 multiplied by 19? Reply with only the integer.", 16)
        sn["correct"] = sn["text"].strip() == str(17 * 19)
        (out / "sanity.json").write_text(json.dumps(sn, indent=2)); f.write(json.dumps(sn) + "\n")
        print(json.dumps({"case": "sanity", "text": sn["text"].strip(), "correct": sn["correct"]}), flush=True)

    warm = [r for r in rows if r["rep"] > 0]
    means = {c: statistics.mean(r["usage"]["decode_tok_s"] for r in warm if r["case"] == c) for c, _, _ in CASES}
    allmean = {c: statistics.mean(r["usage"]["decode_tok_s"] for r in rows if r["case"] == c) for c, _, _ in CASES}
    accept = {c: statistics.mean(r["usage"].get("draft_accept", 0) for r in warm if r["case"] == c) for c, _, _ in CASES}
    summary = {
        "tag": a.tag, "reps": a.reps, "url": URL,
        "warm_mean_decode_tok_s": means, "all_mean_decode_tok_s": allmean, "warm_mean_draft_accept": accept,
        "per_rep": [{"case": r["case"], "rep": r["rep"], "decode_tok_s": r["usage"].get("decode_tok_s"),
                     "completion_tokens": r["usage"].get("completion_tokens"),
                     "ttft_s": r["ttft_s"], "wall_s": r["wall_s"]} for r in rows],
        "prefill": {"prompt_tokens": pf["usage"].get("prompt_tokens"), "tok_s": pf["prefill_tok_s"], "correct": pf["correct"]},
        "sanity_correct": sn["correct"],
        "min_available_gib": min(r["mem_after"]["available_gib"] for r in rows),
        "server": health(),
        "responses_sha256": hashlib.sha256(rpath.read_bytes()).hexdigest(),
        "france_retained_reference": {"code": 41.293333, "prose": 24.293333, "prefill_tok_s": 703},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "per_rep"}), flush=True)


if __name__ == "__main__":
    main()
