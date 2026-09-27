"""TensorFold's bench_openai protocol against our MiMo EXL3 serve (serve_native.py).

Same client code (tf_bench_openai.stream, copied verbatim from ashhart/TensorFold tools/bench_openai.py @ 2f8e514),
same prompts, 64-token replies, ignore_eos, seeds 1234-1238, one warm-up per cell, median of 5, temperatures 1.0
(top-k 20, top-p 0.95) and 0. Decode tok/s = (completion tokens - 1) / (last - first streamed token), client side.

One deviation, forced by the server: serve_native.py has no /v1/completions, so the "fibonacci-raw" prompt goes
through /v1/chat/completions (thinking off) instead of as a raw completion. Its row is labelled "-as-chat".
Also records the server's own decode_tok_s / draft_accept from the final usage chunk.
"""
import json, statistics, sys, time, urllib.request
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import tf_bench_openai as tf

BASE, MODEL = "http://127.0.0.1:8096", "MiMo-V2.6-Flash-RL-EXL3"
out = sys.argv[1] if len(sys.argv) > 1 else "tf-bench-mimo.json"


def usage_of(item, tokens, temp, seed):
    body = {"model": MODEL, "max_tokens": tokens, "temperature": temp, "stream": False, "ignore_eos": True,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None:
        body["seed"] = seed
    if temp > 0:
        body.update(top_k=20, top_p=0.95)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()).get("usage", {})


items = [dict(tf.PROMPTS[0], kind="chat", name="fibonacci-raw-as-chat"), tf.PROMPTS[1]]
rows = []
for temp in (1.0, 0.0):
    for item in items:
        seeds = [1234 + i for i in range(5)]
        tf.stream(BASE, MODEL, item, 64, temp, seeds[0])  # warm-up, as bench_openai does
        runs = [tf.stream(BASE, MODEL, item, 64, temp, s) for s in seeds]
        tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
        srv = [usage_of(item, 64, temp, s) for s in seeds[:3]]
        row = {"prompt": item["name"], "temperature": temp, "decode_tps_median": round(statistics.median(tps), 2),
               "decode_tps_all": [round(x, 2) for x in tps],
               "ttft_s_median": round(statistics.median(r["ttft_s"] for r in runs), 3),
               "tokens": [r["tokens"] for r in runs],
               "server_decode_tok_s": [u.get("decode_tok_s") for u in srv],
               "server_draft_accept": [round(u.get("draft_accept") or 0, 3) for u in srv],
               "sample": runs[0]["text"][:120]}
        print(json.dumps({k: row[k] for k in ("prompt", "temperature", "decode_tps_median", "decode_tps_all",
                                              "ttft_s_median", "server_decode_tok_s", "server_draft_accept")}),
              flush=True)
        rows.append(row)
json.dump({"engine": "exllamav3 fork 249f22a + serve_native (retained Q4/chunk4096/dflash4.0bpw/elision)",
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}, open(out, "w"), indent=1)
print("WROTE", out)
