#!/usr/bin/env bash
# Fetch the MiMo EXL3 pack + drafter with direct, resumable per-file wget.
# Why not `hf download`: on this link it froze (tqdm stuck at 6/26, 0 B/30 s at ~6.3 GB),
# while plain wget holds ~9 MB/s. Files land flat in the pack dir and are checked by size
# and sha256 against the Hub tree API before the script reports done.
set -uo pipefail
H="$HOME/mimo-exl3"
ST="$H/state"
JOBS=4

fetch_repo() {           # $1 tree API url (?expand=true)   $2 resolve base   $3 dest dir
  export API="$1" BASE="$2" DEST="$3"
  mkdir -p "$DEST" "$ST"
  # Parser as a file: `curl | python - <<'PY'` would send the program on stdin and drop the JSON.
  cat > "$ST/parse_tree.py" <<'PY'
import json, sys
for x in json.load(sys.stdin):
    if x.get("type") != "file":
        continue
    lfs = x.get("lfs") or {}
    print(x["path"].rsplit("/", 1)[-1], x.get("size") or lfs.get("size") or 0,
          (lfs.get("oid") or "").replace("sha256:", ""))
PY
  curl -s "$API" | "$H/venv/bin/python" "$ST/parse_tree.py" > "$ST/manifest.tsv" || true
  local n; n="$(wc -l < "$ST/manifest.tsv")"
  echo "manifest: $n files"
  [ "$n" -gt 0 ] || { echo "ABORT: empty manifest for $API"; return 1; }

  dl() {   # $1 = file name
    local b="$1"
    local want; want="$(awk -v p="$b" '$1==p{print $2}' "$ST/manifest.tsv" | head -1)"
    local have=0; [ -f "$DEST/$b" ] && have="$(stat -c %s "$DEST/$b")"
    if [ "$have" = "$want" ]; then echo "  have $b"; return 0; fi
    echo "  get  $b ($have/$want)"
    wget -c -q --tries=25 --timeout=30 --waitretry=5 --retry-connrefused -O "$DEST/$b" "$BASE/$b"
    echo "  got  $b $(stat -c %s "$DEST/$b" 2>/dev/null || echo 0)/$want"
  }
  export -f dl; export ST BASE DEST

  local round=0
  while (( round < 30 )); do
    round=$((round+1))
    echo "round $round"
    awk '{print $1}' "$ST/manifest.tsv" | xargs -P "$JOBS" -I{} bash -c 'dl "$@"' _ {}
    # completeness: size + sha256
    local bad=0
    while read -r b size sha; do
      [ -z "$b" ] && continue
      if [ ! -f "$DEST/$b" ]; then echo "MISSING $b"; bad=$((bad+1)); continue; fi
      local got; got="$(stat -c %s "$DEST/$b")"
      if [ "$got" != "$size" ]; then echo "SIZE $b $got != $size"; bad=$((bad+1)); continue; fi
      if [ -n "$sha" ]; then
        local calc; calc="$(sha256sum "$DEST/$b" | cut -d' ' -f1)"
        if [ "$calc" != "$sha" ]; then echo "SHA $b"; bad=$((bad+1)); continue; fi
      fi
    done < "$ST/manifest.tsv"
    echo "round $round verify: $bad bad of $n"
    (( bad == 0 )) && { echo "COMPLETE: all $n files verified"; break; }
  done
}

echo "=== $(date -Is) pack 2.50bpw (98.48 GB) ==="
fetch_repo "https://huggingface.co/api/models/vcruz305/MiMo-V2.6-Flash-RL-EXL3/tree/main/2.50bpw?expand=true" \
           "https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3/resolve/main/2.50bpw" \
           "$H/models/MiMo-V2.6-Flash-RL-EXL3/2.50bpw"
echo "=== $(date -Is) drafter (0.75 GB) ==="
fetch_repo "https://huggingface.co/api/models/vcruz305/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw/tree/main?expand=true" \
           "https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw/resolve/main" \
           "$H/models/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0"
rm -rf "$H/models/MiMo-V2.6-Flash-RL-EXL3/.cache" "$H/models/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0/.cache"
echo "=== $(date -Is) DOWNLOADS_DONE ==="
du -sh "$H/models"/*
