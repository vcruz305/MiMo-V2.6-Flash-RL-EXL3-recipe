#!/usr/bin/env bash
# v2 (2026-09-26) — replaces wait_and_launch_spark724a.sh (v1 kept, untouched).
# Same flow as v1; fixes:
#   * the published drafter repo ships its weights as model.safetensors, not
#     dflash_draft_model.safetensors (v1's pack-verify would crash and refuse to launch);
#   * v1 trusted DOWNLOADS_DONE, which the fetcher prints even if its verify rounds never pass.
#     v2 requires the fetcher's own sha256 "COMPLETE" line for BOTH repos and re-checks every
#     file's exact size against the Hub tree before launching;
#   * after HEALTH_OK, one short greedy chat request as a coherence smoke test.
set -uo pipefail
H="$HOME/mimo-exl3"
LOG="$H/logs/wait_and_launch.log"
exec >> "$LOG" 2>&1
echo "=== $(date -Is) v2 waiter pid $$ (replaces v1); waiting for DOWNLOADS_DONE ==="
for i in $(seq 1 400); do
  grep -q DOWNLOADS_DONE "$H/logs/fetch3.log" 2>/dev/null && break
  if ! pgrep -f "[f]etch_spark724a.sh" >/dev/null 2>&1; then
    sleep 5
    grep -q DOWNLOADS_DONE "$H/logs/fetch3.log" 2>/dev/null && break
    echo "fetcher is gone and never reported done -> not launching"; exit 1
  fi
  sleep 30
done
grep -q DOWNLOADS_DONE "$H/logs/fetch3.log" 2>/dev/null || { echo "timed out waiting -> not launching"; exit 1; }
c26=$(grep -c "COMPLETE: all 26 files verified" "$H/logs/fetch3.log")
c13=$(grep -c "COMPLETE: all 13 files verified" "$H/logs/fetch3.log")
echo "=== $(date -Is) fetcher sha256 verdicts: pack(26 files)=$c26 drafter(13 files)=$c13 ==="
if [ "$c26" -lt 1 ] || [ "$c13" -lt 1 ]; then
  echo "fetcher did not sha256-verify both repos -> not launching"; exit 1
fi
echo "=== $(date -Is) independent size check vs Hub + pack/drafter layout ==="
VER="$H/logs/pack-verify.log"
"$H/venv/bin/python" - > "$VER" 2>&1 <<'PY'
import glob, json, os, struct, sys, urllib.request
tokp = os.path.expanduser("~/.cache/huggingface/token")
hdrs = {"Authorization": "Bearer " + open(tokp).read().strip()} if os.path.isfile(tokp) else {}
M = os.path.expanduser("~/mimo-exl3/models")
d = M + "/MiMo-V2.6-Flash-RL-EXL3/2.50bpw"
dd = M + "/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0"
checks = [("https://huggingface.co/api/models/vcruz305/MiMo-V2.6-Flash-RL-EXL3/tree/main/2.50bpw?expand=true", d),
          ("https://huggingface.co/api/models/vcruz305/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw/tree/main?expand=true", dd)]
bad = 0
for url, dest in checks:
    files = [x for x in json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdrs), timeout=60))
             if x.get("type") == "file"]
    for x in files:
        p = os.path.join(dest, x["path"].rsplit("/", 1)[-1])
        want = x.get("size") or (x.get("lfs") or {}).get("size")
        got = os.path.getsize(p) if os.path.isfile(p) else None
        if got != want:
            print("SIZE_MISMATCH", p, got, want); bad += 1
    print(f"hub-size check: {len(files)} files in {dest}")
sh = sorted(glob.glob(os.path.join(d, "*.safetensors")))
tot = sum(os.path.getsize(p) for p in sh)
print(f"pack: {len(sh)} shards {tot/1e9:.2f} GB, {len(os.listdir(d))} files")
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "quantization_config.json", "model.safetensors.index.json"):
    if not os.path.isfile(os.path.join(d, f)):
        print("MISSING", f); bad += 1
w = [p for p in ("dflash_draft_model.safetensors", "model.safetensors") if os.path.isfile(os.path.join(dd, p))]
if not w:
    print("DRAFTER_WEIGHTS_MISSING"); sys.exit(1)
with open(os.path.join(dd, w[0]), "rb") as f:
    n, = struct.unpack("<Q", f.read(8)); hdr = json.loads(f.read(n))
layers = sorted({k.split(".")[1] for k in hdr if k.startswith("layers.")})
print(f"drafter weights: {w[0]} {os.path.getsize(os.path.join(dd, w[0]))/1e9:.3f} GB, layers {layers}, "
      f"mask_embedding in weights: {'mask_embedding' in hdr}, "
      f"mask_embedding.safetensors: {os.path.isfile(os.path.join(dd, 'mask_embedding.safetensors'))}")
cfg = json.load(open(os.path.join(dd, "config.json")))
shift = (cfg.get("dflash_config") or {}).get("tap_shift", cfg.get("tap_shift"))
if not layers or shift != 0:
    print("DRAFTER_BAD layers/tap_shift", layers, shift); bad += 1
if len(sh) != 13 or tot < 98e9:
    print("PACK_INCOMPLETE"); bad += 1
print("PACK_OK" if bad == 0 else f"VERIFY_FAILED bad={bad}")
PY
cat "$VER"
grep -q '^PACK_OK' "$VER" || { echo "pack verification failed -> not launching"; exit 1; }
echo "=== $(date -Is) meminfo before launch ==="
grep -E "MemTotal|MemFree|MemAvailable|Cached" /proc/meminfo
echo "=== $(date -Is) launch ==="
"$H/venv/bin/python" "$H/launch_spark724a.py" || { echo "launcher failed -> see above"; exit 1; }
echo "=== $(date -Is) waiting for health ==="
ok=0
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8096/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "HEALTH_OK after $((i*10))s"; ok=1; break; fi
  sleep 10
done
code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8096/v1/models 2>/dev/null || echo 000)
echo "final health=$code"
if [ "$ok" = "1" ]; then
  echo "=== $(date -Is) smoke test ==="
  "$H/venv/bin/python" - <<'PY'
import json, time, urllib.request
base = "http://127.0.0.1:8096/v1"
mid = json.load(urllib.request.urlopen(base + "/models", timeout=30))["data"][0]["id"]
body = {"model": mid, "temperature": 0, "max_tokens": 48,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
dt = time.time() - t
msg = r["choices"][0]["message"]
print("SMOKE model", mid, "secs", round(dt, 2), "usage", r.get("usage"))
print("SMOKE content", repr((msg.get("content") or "")[:400]))
if msg.get("reasoning_content"):
    print("SMOKE reasoning", repr(msg["reasoning_content"][:300]))
PY
fi
echo "=== $(date -Is) serve state ==="
grep -E "LOAD_PREFLIGHT|LAUNCHED" "$LOG" | tail -2
tail -5 "$H/guard.log" 2>/dev/null
echo "LAUNCH_SEQUENCE_DONE"
