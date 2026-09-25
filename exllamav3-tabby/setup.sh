#!/usr/bin/env bash
# One-time setup for the TabbyAPI route: the fork runtime, the pack, the DFlash fix (all from the
# repository root's setup.sh, in the shared venv) plus TabbyAPI at the tip of main in that same venv.
#
#   bash exllamav3-tabby/setup.sh                 # everything, then validate the config
#   bash exllamav3-tabby/setup.sh --check         # verify an existing install
#   bash exllamav3-tabby/setup.sh --check-config  # only validate the rendered config
#   bash exllamav3-tabby/setup.sh --from-source   # build the fork from a checkout
#
# Idempotent: re-running pulls the latest TabbyAPI main and leaves the runtime alone unless you
# ask for --from-source. TabbyAPI's GPU extras (cu12/cu13) are deliberately NOT installed: they
# pull exllamav3 and torch wheels that would shadow the fork.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODE="install"
for arg in "$@"; do
  case "$arg" in
    --check) MODE="check" ;;
    --check-config) MODE="config" ;;
    --from-source) export FROM_SOURCE=1 ;;
    *) die "unknown argument '$arg' (see the header of setup.sh)" ;;
  esac
done

render_config() {
  mkdir -p "$STATE_DIR"
  # Serve from an isolated view directory holding one symlink: TabbyAPI's /v1/models lists every
  # directory under model_dir, so pointing it at the models folder would advertise other packs.
  MODEL_NAME="$SERVED_NAME"
  MODEL_PARENT="$STATE_DIR/models-tabby"
  mkdir -p "$MODEL_PARENT"
  find "$MODEL_PARENT" -mindepth 1 -maxdepth 1 -type l -delete
  ln -sfn "$(cd "$PACK_DIR" && pwd)" "$MODEL_PARENT/$MODEL_NAME"
  export MODEL_PARENT MODEL_NAME HOST PORT MAX_SEQ_LEN CACHE_SIZE MAX_BATCH_SIZE
  DISABLE_AUTH="${DISABLE_AUTH:-true}"
  case "$HOST" in
    127.0.0.1|localhost|::1) ;;
    *) DISABLE_AUTH=false
       echo "warning: auth enabled because HOST=$HOST; TabbyAPI writes api_tokens.yml into $TABBY_DIR" >&2 ;;
  esac
  export DISABLE_AUTH
  "$(render_py)" - "$RECIPE_DIR/exllamav3-tabby/tabby-config.yml" "$STATE_DIR/tabby-config.yml" <<'PY'
import os, string, sys
src, dst = sys.argv[1], sys.argv[2]
text = string.Template(open(src, encoding="utf-8").read()).substitute(os.environ)
open(dst, "w", encoding="utf-8").write(text)
print(f"rendered {dst}", file=sys.stderr)
PY
}

if [[ "$MODE" == "config" ]]; then
  render_config
  check_config "$STATE_DIR/tabby-config.yml"
  exit 0
fi

if [[ "$MODE" == "check" ]]; then
  verify_runtime
  [[ -f "$TABBY_DIR/main.py" ]] || die "no TabbyAPI at $TABBY_DIR; run: bash exllamav3-tabby/setup.sh"
  say "TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD) at $TABBY_DIR"
  verify_pack
  [[ "${DRAFT:-0}" == "1" ]] && verify_draft
  render_config
  check_config "$STATE_DIR/tabby-config.yml"
  exit 0
fi

command -v git >/dev/null || die "git not found"

# 1. Runtime + pack + drafter: the root script owns all of it, in the shared venv.
say "step 1/3: runtime, pack and drafter (repository root's setup.sh)"
if [[ "${FROM_SOURCE:-0}" == "1" ]]; then
  bash "$RECIPE_DIR/setup.sh" --from-source
else
  bash "$RECIPE_DIR/setup.sh"
fi

# 2. TabbyAPI at the tip of main (unpinned on purpose).
say "step 2/3: TabbyAPI"
if [[ ! -d "$TABBY_DIR/.git" ]]; then
  git clone "$TABBY_REPO" "$TABBY_DIR"
fi
git -C "$TABBY_DIR" fetch -q origin
if [[ -n "$(git -C "$TABBY_DIR" status --porcelain --untracked-files=no)" ]]; then
  echo "warning: $TABBY_DIR has local changes; leaving its checkout as is" >&2
else
  git -C "$TABBY_DIR" checkout -q --detach "origin/$TABBY_REF" 2>/dev/null \
    || git -C "$TABBY_DIR" checkout -q --detach "$TABBY_REF"
fi
say "TabbyAPI at $(git -C "$TABBY_DIR" log -1 --format='%h %cs %s' | cut -c1-90)"
( cd "$TABBY_DIR" && "$VENV/bin/python" -m pip install -q . )
# main.py imports uvloop unconditionally, but the package metadata only pulls it in on
# Linux x86_64 extras. A plain `pip install .` leaves the server dying at startup
# with ModuleNotFoundError. Install it explicitly on Linux.
if [[ "$(uname -s)" == Linux ]]; then
  "$VENV/bin/python" -m pip install -q 'uvloop==0.22.1'
fi

# TabbyAPI's dependency set can drag a stock exllamav3 back into the venv on some platforms.
if "$VENV/bin/python" -m pip show exllamav3 2>/dev/null | grep -q "^Location:.*site-packages$" \
   && ! "$VENV/bin/python" -m pip show exllamav3 2>/dev/null | grep -q "Editable project location"; then
  say "a stock exllamav3 was pulled in by TabbyAPI; reinstalling the fork wheel"
  bash "$RECIPE_DIR/setup.sh" --no-pack
fi

# 3. Verify the imported runtime and the config.
say "step 3/3: verify"
verify_runtime
verify_pack
if [[ "${DRAFT:-0}" == "1" ]]; then verify_draft; fi
render_config
check_config "$STATE_DIR/tabby-config.yml" || echo "warning: config validation did not pass; fix it before serving (see exllamav3-tabby/README.md)" >&2

cat >&2 <<EOF

Setup complete.
  runtime:  $("$VENV/bin/python" -c 'import exllamav3; from exllamav3.version import __version__; print(__version__, exllamav3.__file__)')
  server:   TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD)  ($TABBY_DIR)
  venv:     $VENV
  config:   $STATE_DIR/tabby-config.yml

Next:
  bash exllamav3-tabby/serve.sh
  # the measured numbers for this route are still being filled in: see exllamav3-tabby/README.md
EOF
