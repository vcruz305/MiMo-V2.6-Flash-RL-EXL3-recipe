#!/usr/bin/env bash
# Sanity-check the TabbyAPI route: readiness, then the same two prompts the native route uses,
# so the two are comparable. Exits non-zero if the server never comes up.
#
#   bash exllamav3-tabby/chat.sh
#   PORT=8097 bash exllamav3-tabby/chat.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

WAIT="${WAIT:-900}"     # TabbyAPI loads the model itself, so allow longer than the native route
PY="$(render_py)"

say "waiting for http://$HOST:$PORT/v1/models (up to ${WAIT}s)"
"$PY" - "$HOST" "$PORT" "$WAIT" <<'PY'
import json, sys, time, urllib.request
host, port, wait = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + wait
while True:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = json.load(response)
        print(json.dumps(body, indent=1))
        break
    except Exception as exc:
        if time.time() > deadline:
            print(f"error: {url} not ready after {wait:.0f}s ({exc})", file=sys.stderr)
            sys.exit(1)
        time.sleep(5)
PY

ask() {
  local prompt="$1" max_tokens="$2"
  say "prompt (${max_tokens} max tokens): $prompt"
  "$PY" - "$HOST" "$PORT" "$SERVED_NAME" "$prompt" "$max_tokens" <<'PY'
import json, sys, urllib.request
host, port, model, prompt, max_tokens = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": max_tokens, "temperature": 0}
req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as response:
    body = json.load(response)
choice = body["choices"][0]
usage = body.get("usage") or {}
finish = choice.get("finish_reason")
text = (choice["message"].get("content") or choice["message"].get("reasoning_content") or "")
print(f"finish={finish} tokens={usage.get('completion_tokens')} "
      f"prompt_tokens={usage.get('prompt_tokens')} decode={usage.get('decode_tok_s')} "
      f"draft_accept={usage.get('draft_accept')}")
print(f"text: ...{text[-220:]!r}")
if finish not in ("stop", "length"):
    sys.exit(1)
PY
}

ask "What is the capital of France? Answer in one short sentence." 64
ask "Write one sentence about the ocean." 200
say "ok - record the numbers you see here in this folder's README results table"
