#!/usr/bin/env bash
set -euo pipefail
# Operator-only replay of preexisting France deployment artifacts, not a bootstrap.
# Default: published EXL3 4.0 bpw drafter on the original rental runtime (round5).
# Faster readback arm stays opt-in: it needs an isolated Python shadow with an
# unmerged runtime patch and the original DSO. This recipe does not install either.
# All profiles keep the UMA supervisor; missing dependencies never select BF16.
profile=${PROFILE:-quantized-draft}
case "$profile" in
  quantized-draft) launcher=/workspace/mimo-tune/start_france_quant_dflash4_dynamic06_chunk4096.py ;;
  quantized-draft-readback) launcher=/workspace/mimo-tune/round6-quant-readback/start_quant_readback.py ;;
  dynamic-balanced) launcher=/workspace/mimo-tune/start_france_dflash7_dynamic06_chunk4096.py ;;
  chunk1024) launcher=/workspace/mimo-tune/start_france_dflash7_dynamic06.py ;;
  dynamic-code) launcher=/workspace/mimo-tune/start_france_dflash7_dynamic04.py ;;
  dflash4) launcher=/workspace/mimo-tune/start_france_dflash4_serial.py ;;
  dflash4-batch) launcher=/workspace/mimo-tune/start_france_dflash4_greedy.py ;;
  dflash7) launcher=/workspace/mimo-tune/start_france_dflash7.py ;;
  q4-chunk1024) launcher=/workspace/mimo-tune/start_france_q4_1024.py ;;
  baseline) launcher=/workspace/mimo-tune/start_france_uma.py ;;
  *) printf '%s\n' 'Unknown PROFILE; use quantized-draft, quantized-draft-readback, dynamic-balanced, chunk1024, dynamic-code, dflash4, dflash4-batch, dflash7, q4-chunk1024 or baseline.' >&2; exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '/usr/bin/python3 %s\n' "$launcher"
  exit 0
fi
if [[ "$profile" == quantized-draft-readback ]]; then
  shadow=/workspace/mimo-tune/60tps-round3/python-shadow
  source_file=$shadow/exllamav3/modules/block_sparse_mlp.py
  extension=$shadow/exllamav3_ext.cpython-312-aarch64-linux-gnu.so
  entrypoint=/workspace/mimo-tune/round6-quant-readback/serve_native.py
  if [[ ! -f "$source_file" || ! -f "$extension" || ! -f "$entrypoint" || ! -f "$launcher" ]]; then
    printf '%s\n' 'Readback requires the installed isolated Python shadow, original DSO and round6 entrypoint/launcher; unmerged runtime dependency, no BF16 fallback. See DGX-Spark/France-Quant-Readback.md.' >&2
    exit 1
  fi
  # Hash files only; never import the model/runtime or build a replacement here.
  if ! printf '%s  %s\n' \
    61653605b432350db3cccddbaa00b12f121e38fa2b072fc9de2032adf9e2a706 "$source_file" \
    02b0ae5bc8414d335facca41083f561cf24f41d56ef6e1e73a8b41fb16f80207 "$extension" \
    | sha256sum --check --status; then
    printf '%s\n' 'Readback dependency hash mismatch; requires the measured isolated Python shadow and original DSO, no BF16 fallback.' >&2
    exit 1
  fi
fi
if [[ ! -f "$launcher" ]]; then
  printf '%s\n' 'Requires the installed France UMA runtime/profile; see DGX-Spark/France-UMA.md.' >&2
  exit 1
fi
exec /usr/bin/python3 "$launcher"
