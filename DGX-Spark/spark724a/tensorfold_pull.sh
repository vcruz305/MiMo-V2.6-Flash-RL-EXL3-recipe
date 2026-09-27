#!/bin/bash
# control model for the TensorFold CUDA baseline: Qwen3.8-27B MLX 4-bit + its DFlash2 drafter (~20 GB).
# xet hangs on this box (measured), so use the classic HTTP path with 8 file workers.
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
for r in z-lab/Qwen3.8-27B-DFlash2 Vontra/Qwen3.8-27B-MLX-4bit; do
  for try in 1 2 3 4 5 6; do
    echo "=== $(date -Is) $r try $try ==="
    timeout 5400 $HOME/mimo-exl3/venv/bin/hf download "$r" --max-workers 8 && { echo "OK $r"; break; }
    sleep 5
  done
done
echo "=== $(date -Is) PULL_DONE ==="
