#!/usr/bin/env bash
set -euo pipefail
# Installed rental only; every profile keeps the UMA memory supervisor.
case "${PROFILE:-dynamic-balanced}" in
  dynamic-balanced) launcher=/workspace/mimo-tune/start_france_dflash7_dynamic06_chunk4096.py ;;
  chunk1024) launcher=/workspace/mimo-tune/start_france_dflash7_dynamic06.py ;;
  dynamic-code) launcher=/workspace/mimo-tune/start_france_dflash7_dynamic04.py ;;
  dflash4) launcher=/workspace/mimo-tune/start_france_dflash4_serial.py ;;
  dflash4-batch) launcher=/workspace/mimo-tune/start_france_dflash4_greedy.py ;;
  dflash7) launcher=/workspace/mimo-tune/start_france_dflash7.py ;;
  q4-chunk1024) launcher=/workspace/mimo-tune/start_france_q4_1024.py ;;
  baseline) launcher=/workspace/mimo-tune/start_france_uma.py ;;
  *) printf '%s\n' 'Unknown PROFILE; use dynamic-balanced, chunk1024, dynamic-code, dflash4, dflash4-batch, dflash7, q4-chunk1024 or baseline.' >&2; exit 2 ;;
esac
if [[ ! -f "$launcher" ]]; then
  printf '%s\n' 'Requires the installed France UMA runtime/profile; see DGX-Spark/France-UMA.md.' >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '/usr/bin/python3 %s\n' "$launcher"
  exit 0
fi
exec /usr/bin/python3 "$launcher"
