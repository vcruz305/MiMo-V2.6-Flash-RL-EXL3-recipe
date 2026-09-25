#!/usr/bin/env bash
# Install the runtime and stage the pack for MiMo-V2.6-Flash-RL EXL3 on one RTX PRO 6000.
#
#   bash setup.sh                 released wheel (recommended) + pack + DFlash fix
#   bash setup.sh --from-source   build vcruz305/exllamav3 from a checkout instead
#   bash setup.sh --check         verify an existing install and stop
#   bash setup.sh --no-pack       runtime + drafter only (pack already on disk)
#
# Everything lands under $RECIPE_HOME (default ~/mimo-exl3); the models under
# $MODELS_DIR. Nothing outside those directories is touched, and no process is started.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

FROM_SOURCE=0
DO_PACK=1
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --from-source) FROM_SOURCE=1 ;;
    --no-pack) DO_PACK=0 ;;
    --check) CHECK_ONLY=1 ;;
    *) die "unknown argument '$arg' (see the header of setup.sh)" ;;
  esac
done

mkdir -p "$RECIPE_HOME" "$STATE_DIR" "$MODELS_DIR"

# ---------------------------------------------------------------------------
# 1. Runtime
# ---------------------------------------------------------------------------
install_wheel() {
  say "creating venv at $VENV"
  [[ -x "$VENV/bin/python" ]] || "$PYTHON_BIN" -m venv "$VENV"
  say "installing torch (the wheel does not pull it) and the fork wheel"
  # The released wheel requires one exact torch version; the wheel name pins it.
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
  local url="https://github.com/vcruz305/exllamav3/releases/download/$EXL3_WHEEL_TAG/$EXL3_WHEEL"
  say "wheel: $url"
  "$VENV/bin/python" -m pip install -q --force-reinstall "$url" \
    || die "wheel install failed. Check the release page for a row matching your python/torch/CUDA, set EXL3_WHEEL=..., or use: bash setup.sh --from-source"
  "$VENV/bin/python" -m pip install -q "transformers>=5.3" safetensors packaging huggingface_hub
  # A stock exllamav3 in the same venv would shadow the fork depending on sys.path order.
  "$VENV/bin/python" -m pip uninstall -y -q exllamav3 2>/dev/null || true
  "$VENV/bin/python" -m pip install -q --force-reinstall --no-deps \
    "https://github.com/vcruz305/exllamav3/releases/download/$EXL3_WHEEL_TAG/$EXL3_WHEEL"
  rm -f "$STATE_DIR/runtime.env"
  echo "$EXL3_WHEEL_TAG ($EXL3_WHEEL)" > "$MARKER"
}

install_source() {
  say "building vcruz305/exllamav3 from source at $EXL3_REF (needs a CUDA toolkit at $CUDA_HOME)"
  [[ -x "$VENV/bin/python" ]] || "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/python" -m pip install -q "transformers>=5.3" safetensors packaging huggingface_hub
  if [[ -d "$EXL3_SRC/.git" ]]; then
    git -C "$EXL3_SRC" fetch -q origin
  else
    git clone -q "$EXL3_REPO" "$EXL3_SRC"
  fi
  git -C "$EXL3_SRC" checkout -q "$EXL3_REF"
  say "building the extension (this is the slow step: several minutes)"
  ( cd "$EXL3_SRC" && CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
      "$VENV/bin/python" -m pip install -q -e . ) \
    || die "source build failed; the log is above, and $CUDA_HOME/bin/nvcc must exist"
  local rev; rev="$(git -C "$EXL3_SRC" rev-parse --short HEAD)"
  printf 'export EXL3_ROOT=%q\n' "$EXL3_SRC" > "$STATE_DIR/runtime.env"
  echo "$rev" > "$MARKER"
}

# ---------------------------------------------------------------------------
# 2. Pack (and the drafter that ships with the ORIGINAL checkpoint)
# ---------------------------------------------------------------------------
fetch_pack() {
  command -v hf >/dev/null || die "the 'hf' CLI is missing. pip install 'huggingface_hub[cli]' (or use huggingface-cli)"
  if [[ -f "$PACK_DIR/config.json" ]]; then
    say "pack already present at $PACK_DIR"
    return 0
  fi
  say "downloading $PACK_REPO subset '$PACK_SUBDIR/*' into $PACK_HOME"
  if ! hf download "$PACK_REPO" --include "$PACK_SUBDIR/*" --local-dir "$PACK_HOME"; then
    if [[ "$PACK_SUBDIR" == "2.22bpw" ]]; then
      die "the 2.22 bpw rung is not on the Hub yet (its publication is pending). Use the published 2.50 bpw rung on a >96 GB card with PACK_SUBDIR=2.50bpw, or point PACK_DIR at your copy of the servable rung."
    fi
    die "pack download failed; check the repo/revision in README 'Downloads'"
  fi
}

fetch_draft() {
  command -v hf >/dev/null || die "the 'hf' CLI is missing"
  if [[ -f "$DRAFT_SRC/config.json" ]]; then
    say "drafter source already present at $DRAFT_SRC"
  else
    say "downloading the DFlash drafter from $BASE_MODEL_REPO ($DRAFT_SUBDIR/)"
    hf download "$BASE_MODEL_REPO" --include "$DRAFT_SUBDIR/*" \
      --local-dir "$MODELS_DIR/MiMo-V2.6-Flash-RL" \
      || die "drafter download failed; it is required for the default (with-draft) profile"
  fi
  say "staging the corrected copy at $DRAFT_DIR"
  "$VENV/bin/python" "$RECIPE_DIR/tools/fix_dflash.py" --src "$DRAFT_SRC" --dst "$DRAFT_DIR"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if (( CHECK_ONLY )); then
  verify_runtime
  verify_pack
  verify_draft
  show_card
  exit 0
fi

if (( FROM_SOURCE )); then install_source; else install_wheel; fi
say "runtime: $("$(render_py)" -c 'import exllamav3; from exllamav3.version import __version__; print(__version__, exllamav3.__file__)' 2>/dev/null || echo 'import failed')"
verify_runtime

if (( DO_PACK )); then fetch_pack; fi
if [[ -f "$PACK_DIR/config.json" ]]; then verify_pack; fi
fetch_draft
verify_draft

show_card
cat >&2 <<EOF

==> done. Next:
      bash preflight.sh        # card + pack + drafter, read-only
      PROFILE=with-draft bash serve.sh      # OpenAI API on 127.0.0.1:8096/v1  (184.5 tok/s p50 decode)
      bash chat.sh             # sanity prompts against a running server

    The TabbyAPI route is a separate, second path: see exllamav3-tabby/README.md
    (its measured numbers are still being filled in).
EOF
