#!/usr/bin/env bash
# Serve MiMo-V2.6-Flash-RL EXL3 as an OpenAI-compatible /v1 API. Needs one ~96 GB GPU; the card
# the numbers in this repo were measured on is named in README > Cards tested on.
# This is the route whose numbers are measured (README "Measured results").
#
#   bash serve.sh                       # PROFILE=with-draft  (184.46 tok/s p50 decode)
#   PROFILE=no-draft bash serve.sh      # no drafter          ( 47.63 tok/s p50 decode)
#   DRY_RUN=1 bash serve.sh             # print the exact command line and exit
#   PORT=8097 bash serve.sh             # any value can be overridden like this
#
# Profiles (configs/*.env; every value can still be overridden individually):
#   with-draft  CTX=65536  CACHE_SIZE=65536  AUTOSPLIT_MAX_BATCH=16  MAX_ACTIVE_REQUESTS=16  -dm $DRAFT_DIR
#   no-draft    same sizing, no drafter
#
# Profiles of the other knobs: MODEL_DIR=$PACK_DIR, HOST=127.0.0.1, PORT=8096,
# SERVED_NAME=MiMo-V2.6-Flash-RL-EXL3.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PROFILE="${PROFILE:-with-draft}"
[[ -f "$RECIPE_DIR/configs/$PROFILE.env" ]] \
  || die "unknown PROFILE '$PROFILE' (available: $(cd "$RECIPE_DIR/configs" && ls *.env | sed 's/\.env//' | tr '\n' ' '))"
# shellcheck source=/dev/null
source "$RECIPE_DIR/configs/$PROFILE.env"

CTX="${CTX:-65536}"
CACHE_SIZE="${CACHE_SIZE:-65536}"
AUTOSPLIT_MAX_BATCH="${AUTOSPLIT_MAX_BATCH:-16}"
MAX_ACTIVE_REQUESTS="${MAX_ACTIVE_REQUESTS:-16}"
MAX_PENDING_REQUESTS="${MAX_PENDING_REQUESTS:-64}"
DRAFT="${DRAFT:-1}"
MODEL_DIR="${MODEL_DIR:-$PACK_DIR}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8096}"
SERVED_NAME="${SERVED_NAME:-MiMo-V2.6-Flash-RL-EXL3}"

check_sizing "$CACHE_SIZE" "$CTX" "$MAX_ACTIVE_REQUESTS" "$AUTOSPLIT_MAX_BATCH"
port_free "$PORT"

# An install that was built from source records its checkout here.
# shellcheck source=/dev/null
[[ -f "$STATE_DIR/runtime.env" ]] && source "$STATE_DIR/runtime.env"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
[[ -n "${EXL3_ROOT:-}" ]] && export PYTHONPATH="${EXL3_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CMD=( "$(render_py)" "$RECIPE_DIR/server/serve_native.py"
      -m "$MODEL_DIR" -cs "$CACHE_SIZE" -ambs "$AUTOSPLIT_MAX_BATCH"
      --max-active-requests "$MAX_ACTIVE_REQUESTS"
      --max-pending-requests "$MAX_PENDING_REQUESTS"
      --max-model-len "$CTX"
      --served-model-name "$SERVED_NAME"
      --host "$HOST" --port "$PORT" )
if [[ "$DRAFT" == "1" ]]; then
  CMD+=( -dm "$DRAFT_DIR" )
fi

say "profile $PROFILE: ${CTX} tokens per request, ${CACHE_SIZE}-token pool, ${MAX_ACTIVE_REQUESTS} concurrent jobs, drafter: $DRAFT"
say "pack:   $MODEL_DIR"
[[ "$DRAFT" == "1" ]] && say "drafter: $DRAFT_DIR (tap_shift 0 + mask_embedding.safetensors; verify with tools/verify_dflash.py)"

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "EXL3_ROOT=${EXL3_ROOT:-<unset: wheel install>} CUDA_HOME=$CUDA_HOME"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

verify_runtime
verify_pack "$MODEL_DIR"
[[ "$DRAFT" == "1" ]] && verify_draft "$DRAFT_DIR"
check_card_free

say "native /v1 API on http://$HOST:$PORT/v1, model id: $SERVED_NAME (first load ~90 s)"
say "log it with: screen -dmS mimoserve bash -c 'PORT=$PORT bash serve.sh > $STATE_DIR/serve-$PROFILE.log 2>&1'"
exec "${CMD[@]}"
