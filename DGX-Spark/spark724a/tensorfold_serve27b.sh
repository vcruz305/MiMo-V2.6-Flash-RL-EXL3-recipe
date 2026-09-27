#!/bin/bash
export PATH=$HOME/tensorfold/venv/bin:$HOME/mimo-exl3/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=12.1 TENSORFOLD_NO_UPDATE_CHECK=1 TORCH_EXTENSIONS_DIR=$HOME/tensorfold/torch_ext
exec tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --backend cuda --host 127.0.0.1 --port 8080 --no-update-check
