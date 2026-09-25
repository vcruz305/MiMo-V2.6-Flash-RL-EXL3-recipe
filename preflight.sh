#!/usr/bin/env bash
# Read-only preflight: card, disk, runtime, pack integrity, drafter wiring, port.
# Nothing is loaded, nothing is started. Run it before serve.sh.
#
#   bash preflight.sh
#   PROFILE=no-draft bash preflight.sh
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PROFILE="${PROFILE:-with-draft}"
[[ -f "$RECIPE_DIR/configs/$PROFILE.env" ]] && source "$RECIPE_DIR/configs/$PROFILE.env"
PORT="${PORT:-8096}"
MODEL_DIR="${MODEL_DIR:-$PACK_DIR}"
FAILED=0
step() { echo; echo "=== $* ==="; }
warn() { echo "warning: $*" >&2; }

step "card"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap \
    --format=csv,noheader
  used="$(vram_used_mib)"
  if [[ -n "$used" ]] && (( used > 2000 )); then
    warn "${used} MiB already resident; this recipe wants the whole card"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
    FAILED=1
  fi
else
  warn "nvidia-smi not found"
  FAILED=1
fi

step "paths"
printf '%s\n' "RECIPE_HOME=$RECIPE_HOME" "VENV=$VENV" "STATE_DIR=$STATE_DIR" \
              "PACK_DIR=$MODEL_DIR" "DRAFT_DIR=$DRAFT_DIR (drafter used: ${DRAFT:-1})" "PORT=$PORT"

step "disk"
if [[ -d "$MODELS_DIR" ]]; then
  df -h "$MODELS_DIR" | tail -n +1
else
  warn "$MODELS_DIR does not exist yet (setup.sh creates it)"
fi

step "runtime"
if [[ -x "$VENV/bin/python" ]]; then
  verify_runtime || FAILED=1
else
  warn "no venv at $VENV; run: bash setup.sh"
  FAILED=1
fi

step "pack"
if [[ -f "$MODEL_DIR/config.json" ]]; then
  verify_pack "$MODEL_DIR" || FAILED=1
else
  warn "no pack at $MODEL_DIR; see README 'Downloads'"
  FAILED=1
fi

step "drafter"
if [[ "${DRAFT:-1}" == "1" ]]; then
  if [[ -d "$DRAFT_DIR" ]]; then
    verify_draft "$DRAFT_DIR" || FAILED=1
  else
    warn "no corrected drafter at $DRAFT_DIR; run: python tools/fix_dflash.py"
    FAILED=1
  fi
  if [[ -d "$DRAFT_SRC" ]]; then
    echo "original drafter source: $DRAFT_SRC (must be untouched; tools/fix_dflash.py works on a copy)"
  fi
else
  echo "PROFILE=$PROFILE: no drafter needed"
fi

step "port"
if port_free "$PORT"; then echo "port $PORT is free"; else FAILED=1; fi

step "result"
if (( FAILED )); then
  echo "preflight FAILED - fix the warnings above before serving"
  exit 1
fi
echo "preflight ok. Next: PROFILE=$PROFILE bash serve.sh   (or DRY_RUN=1 bash serve.sh to see the command)"
echo "note: sizing numbers are printed by serve.sh; cache_size is ONE pool shared by all concurrent requests."
