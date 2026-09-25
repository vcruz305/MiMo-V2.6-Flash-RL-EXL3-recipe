#!/usr/bin/env bash
# The measurement command behind every number in README "Measured results".
#
# SixCat 0.7.0, run against the native /v1 server, exact command line used on 2026-09-25:
#   python -m sixcat speed --base-url http://127.0.0.1:8096/v1 --model <served name> \
#     --max-seconds 7200 --curve-seconds 1500 --out <json>
#
# SixCat is the harness these numbers came from; point SIXCAT_PY at an environment where
# `python -m sixcat` resolves. Start the server first (serve.sh) - this script only measures.
#
#   bash tools/sixcat_speed.sh
#   SIXCAT_PY=~/sixcat-venv/bin/python bash tools/sixcat_speed.sh
#   PORT=8097 LABEL=no-draft PROFILE=no-draft bash tools/sixcat_speed.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

PORT="${PORT:-8096}"
HOST="${HOST:-127.0.0.1}"
SERVED_NAME="${SERVED_NAME:-MiMo-V2.6-Flash-RL-EXL3}"
LABEL="${LABEL:-measured}"
SIXCAT_PY="${SIXCAT_PY:-$PYTHON_BIN}"
MAX_SECONDS="${MAX_SECONDS:-7200}"
CURVE_SECONDS="${CURVE_SECONDS:-1500}"
RESULTS="${RESULTS:-$STATE_DIR/results}"
OUT="${OUT:-$RESULTS/sixcat-$LABEL-speed.json}"

mkdir -p "$RESULTS"
say "sixcat: $SIXCAT_PY -m sixcat speed  ->  $OUT"
"$SIXCAT_PY" -m sixcat speed \
  --base-url "http://$HOST:$PORT/v1" \
  --model "$SERVED_NAME" \
  --max-seconds "$MAX_SECONDS" \
  --curve-seconds "$CURVE_SECONDS" \
  --out "$OUT"
say "done: $OUT"
echo
echo "Compare a fresh run against the numbers in README 'Measured results' (2.20 bpw, client-observed):"
echo "  decode single-stream p50   49.57 tok/s (no draft)   184.11 tok/s (with the corrected drafter)"
echo "  prefill single-stream p50  2,370.2 tok/s            2,257.7 tok/s"
echo "  max usable concurrency     8                       8"
