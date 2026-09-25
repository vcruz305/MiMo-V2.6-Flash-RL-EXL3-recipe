#!/usr/bin/env bash
# Serve MiMo-V2.6-Flash-RL EXL3 through TabbyAPI: the vcruz305/exllamav3 fork runtime, the same
# pack, the same card as serve.sh at the repository root. This route's measured numbers are still
# being filled in - see README.md in this folder and do not quote native-route numbers for it.
#
#   bash exllamav3-tabby/serve.sh              # OpenAI-compatible API on 127.0.0.1:8096
#   DRY_RUN=1 bash exllamav3-tabby/serve.sh    # print the command and the rendered config path
#   PORT=8097 bash exllamav3-tabby/serve.sh
#
# TabbyAPI takes a YAML config, rendered from tabby-config.yml with the values in env.sh.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODEL_NAME="$SERVED_NAME"
MODEL_PARENT="$STATE_DIR/models-tabby"

check_sizing "$CACHE_SIZE" "$MAX_SEQ_LEN" "$MAX_BATCH_SIZE" "$MAX_BATCH_SIZE"
port_free "$PORT"

# Env-var knobs the fork reads and TabbyAPI has no key for (empty by default; see env.sh).
if [[ -n "${EXL3_ENV_DEFAULTS:-}" ]]; then
  # shellcheck disable=SC2086
  export $EXL3_ENV_DEFAULTS
fi
# A source install records its checkout here (written by the root setup.sh).
# shellcheck source=/dev/null
[[ -f "$STATE_DIR/runtime.env" ]] && source "$STATE_DIR/runtime.env"
[[ -n "${EXL3_ROOT:-}" ]] && export PYTHONPATH="${EXL3_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$VENV/bin:$PATH"

# TabbyAPI's /v1/models lists every directory under model_dir, so serve from a one-symlink view.
mkdir -p "$MODEL_PARENT"
find "$MODEL_PARENT" -mindepth 1 -maxdepth 1 -type l -delete
if [[ -d "$PACK_DIR" ]]; then
  ln -sfn "$(cd "$PACK_DIR" && pwd)" "$MODEL_PARENT/$MODEL_NAME"
fi

DISABLE_AUTH="${DISABLE_AUTH:-true}"
case "$HOST" in
  127.0.0.1|localhost|::1) ;;
  *)
    DISABLE_AUTH=false
    echo "warning: auth enabled because HOST=$HOST; the key is written to $TABBY_DIR/api_tokens.yml on first start" >&2
    ;;
esac
export MODEL_PARENT MODEL_NAME HOST PORT DISABLE_AUTH MAX_SEQ_LEN CACHE_SIZE MAX_BATCH_SIZE

CONFIG="$STATE_DIR/tabby-config.yml"
PY="$(render_py)"
"$PY" - "$RECIPE_DIR/exllamav3-tabby/tabby-config.yml" "$CONFIG" <<'PY'
import os, string, sys
src, dst = sys.argv[1], sys.argv[2]
open(dst, "w", encoding="utf-8").write(string.Template(open(src, encoding="utf-8").read()).substitute(os.environ))
print(f"config: {dst}", file=sys.stderr)
PY
say "config: $CONFIG"

CMD=( "$VENV/bin/python" "$TABBY_DIR/main.py" --config "$CONFIG" )
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "cd $TABBY_DIR && ${CMD[*]}"
  exit 0
fi

verify_runtime
[[ -f "$TABBY_DIR/main.py" ]] || die "no TabbyAPI at $TABBY_DIR. Run: bash exllamav3-tabby/setup.sh"
verify_pack
[[ "${DRAFT:-0}" == "1" ]] && verify_draft
check_card_free
check_config "$CONFIG" \
  || die "the rendered config did not validate against TabbyAPI's schema; see exllamav3-tabby/README.md"

say "TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD) on http://$HOST:$PORT/v1, model id: $MODEL_NAME"
say "log it with: screen -dmS mimotabby bash -c 'bash exllamav3-tabby/serve.sh > $STATE_DIR/serve-tabby.log 2>&1'"
# TabbyAPI resolves templates/, sampler_overrides/ and api_tokens.yml relative to its own cwd.
cd "$TABBY_DIR"
exec "${CMD[@]}"
