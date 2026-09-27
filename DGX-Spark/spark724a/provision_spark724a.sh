#!/usr/bin/env bash
# Provision spark-724a (<lan-ip>, GB10 / sm_121) to continue the France MiMo EXL3 work.
# Nothing here uses sudo. Everything lands under $HOME/mimo-exl3.
set -uo pipefail
H="$HOME/mimo-exl3"
mkdir -p "$H"/{logs,state,models,runs,repo}
VENV="$H/venv"
PY="$VENV/bin/python"
step() { echo; echo "=== $(date -Is) $* ==="; }
fail() { echo "!!! FAILED: $*"; exit 1; }

step "recipe repo (public) -> $H/repo"
if [ ! -d "$H/repo/.git" ]; then
  git clone -q https://github.com/vcruz305/MiMo-V2.6-Flash-RL-EXL3-recipe.git "$H/repo" || fail "recipe clone"
fi
git -C "$H/repo" log --oneline -1

step "venv + build tooling"
[ -x "$PY" ] || python3 -m venv "$VENV" || fail "venv"
"$PY" -m pip install -q --upgrade pip setuptools wheel || fail "pip bootstrap"
"$PY" -m pip install -q ninja packaging || fail "ninja/packaging"

step "torch 2.11.0 (cu130 first, cu128 fallback) - matches the France runtime"
"$PY" -m pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130 \
  || "$PY" -m pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 \
  || fail "torch install"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_capability() if torch.cuda.is_available() else '')" || fail "torch import"

step "huggingface client + hf_xet"
"$PY" -m pip install -q "transformers>=5.3" safetensors hf_xet "huggingface_hub[cli]" || fail "hf deps"
"$VENV/bin/hf" version || fail "hf cli"

step "start the 99 GB model download in the background (public repos, no token)"
nohup bash -c '
  H="$HOME/mimo-exl3"; HF="$H/venv/bin/hf"
  echo "=== $(date -Is) pack 2.50bpw (98.48 GB) ==="
  "$HF" download vcruz305/MiMo-V2.6-Flash-RL-EXL3 --include "2.50bpw/*" --local-dir "$H/models/MiMo-V2.6-Flash-RL-EXL3"
  echo "=== $(date -Is) pack rc=$? ==="
  echo "=== $(date -Is) drafter EXL3 4.0bpw (0.75 GB) ==="
  "$HF" download vcruz305/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw --local-dir "$H/models/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0"
  echo "=== $(date -Is) drafter rc=$? ==="
  echo DOWNLOADS_DONE
' > "$H/logs/download.log" 2>&1 &
echo "download pid $!"

step "fork clone at the measured runtime branch"
if [ ! -d "$H/exllamav3/.git" ]; then
  git clone -q -b feat/mixedk-handled-readback-elision https://github.com/vcruz305/exllamav3.git "$H/exllamav3" || fail "fork clone"
fi
git -C "$H/exllamav3" log --oneline -1
grep -n "ELIDE_HANDLED" "$H/exllamav3/exllamav3/modules/block_sparse_mlp.py" | head -3 || fail "elision code missing in the branch"

step "build the sm_121 extension (slow step, ~10-30 min)"
cd "$H/exllamav3" || fail "cd"
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=16 \
  "$PY" -m pip install -e . --no-build-isolation || fail "extension build"

step "verify the runtime imports as the fork with the MiMo-V2 + DFlash ports"
"$PY" - <<'PYEOF' || fail "runtime verify"
import exllamav3
from exllamav3.version import __version__ as v
from exllamav3.architecture import mimo_v2
from exllamav3.architecture.dflash import DFlashConfig
assert hasattr(DFlashConfig, "tap_shift")
print("exllamav3", v, "at", exllamav3.__file__)
PYEOF

echo
echo "RUNTIME_DONE (download continues in the background; see $H/logs/download.log)"
