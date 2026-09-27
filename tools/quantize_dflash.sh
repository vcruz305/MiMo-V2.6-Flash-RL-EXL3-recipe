#!/bin/bash
# Build the EXL3 4.0 bpw dflash drafter from the corrected BF16 drafter.
#
# Why: the drafter runs one forward per decode step. Quantizing it to EXL3 4.0 bpw cuts
# its weights from 2.94 GB to ~735 MB and measured +3.2% decode (193.66 -> 199.86 tok/s p50
# on the SixCat speed suite) at an identical draft acceptance rate. The target verifies
# every drafted token, so the quantized drafter cannot change the served output.
#
# It also frees ~2.1 GB of VRAM, which raises the with-draft context ceilings
# (FP16 KV 139,264 -> 196,608; Q4 KV 524,288 -> 655,360).
#
# Usage:  bash tools/quantize_dflash.sh [DRAFT_DIR] [OUT_DIR] [BITS]
# Requires the fork (exllamav3) importable and the pack dir present for its tokenizer.
set -euo pipefail

DRAFT_DIR="${1:-${DRAFT_DIR:-$HOME/models/MiMo-V2.6-Flash-RL-dflash-fixed}}"
OUT_DIR="${2:-${OUT_DIR:-$HOME/models/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0}}"
BITS="${3:-4.0}"

FORK="${EXL3_ROOT:-$HOME/mimo/exllamav3-mimo-git}"
PY="${PY:-python}"
PACK="${PACK_DIR:-$HOME/models/MiMo-V2.6-Flash-RL-EXL3-2.20}"
STAGE="${STAGE:-$HOME/models/dflash-exl3-stage}"
WORK="${WORK:-$HOME/mimo/dflash-exl3-work}"

[ -f "$DRAFT_DIR/config.json" ] || { echo "no drafter at $DRAFT_DIR" >&2; exit 2; }
[ -d "$FORK" ] || { echo "no fork at $FORK (set EXL3_ROOT)" >&2; exit 2; }

# 1. Stage: the converter needs a tokenizer, and the drafter folder carries none (it embeds
#    through the target's embedding). Copy the drafter and borrow the pack's tokenizer files.
rm -rf "$STAGE"
cp -a "$DRAFT_DIR" "$STAGE"
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja special_tokens_map.json; do
  [ -f "$PACK/$f" ] && cp -a "$PACK/$f" "$STAGE/" || true
done

export PYTHONPATH="$FORK"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PATH="$CUDA_HOME/bin:$PATH"

# 2. Convert. The drafter has no embedding table, so the converter reports
#    "Performing uncalibrated quantization" and quantizes it as a side model -- there is no
#    calibration forward pass to run, and no cal data is needed.
rm -rf "$OUT_DIR" "$WORK"
cd "$FORK"
"$PY" convert.py -i "$STAGE" -w "$WORK" -o "$OUT_DIR" -b "$BITS" -d 0 -ss 8192

# 3. The converted folder keeps tap_shift 0 and carries the mask embedding as a tensor inside
#    the shard, which is where the runtime looks for it (stc.has_tensor("mask_embedding")).
du -sh "$OUT_DIR"
echo "built $OUT_DIR at $BITS bpw"
echo "serve it with: DRAFT_DIR=$OUT_DIR bash serve.sh"
