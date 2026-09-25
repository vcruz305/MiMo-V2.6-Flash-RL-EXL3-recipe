# Shared defaults for the TabbyAPI route. Sourced by setup.sh, serve.sh and chat.sh here.
# Everything path- and pin-related comes from the repository root's env.sh, so both routes use
# one venv, one runtime checkout and one pack; only the server differs.

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$RECIPE_DIR/env.sh"

# The API server: latest theroyallab/tabbyAPI main, deliberately not pinned.
TABBY_REPO="${TABBY_REPO:-https://github.com/theroyallab/tabbyAPI.git}"
TABBY_REF="${TABBY_REF:-main}"
TABBY_DIR="${TABBY_DIR:-$RECIPE_HOME/tabbyAPI}"

# The API surface. Same port as the native route on purpose: only one of the two should be
# serving the card at a time, and identical ports make that obvious.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8096}"
SERVED_NAME="${SERVED_NAME:-MiMo-V2.6-Flash-RL-EXL3}"

# Context and concurrency: the measured native configuration (65,536 / 65,536 / 16), so the two
# routes can be compared on the same workload.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
CACHE_SIZE="${CACHE_SIZE:-65536}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"

# Drafter. Default on: that is the measured fast path (DRAFT=0 is the no-draft row).
# The directory must be the corrected copy (tap_shift 0 + mask_embedding shard).
DRAFT="${DRAFT:-1}"
DRAFT_NAME="${DRAFT_NAME:-MiMo-V2.6-Flash-RL-dflash-fixed}"
DRAFT_NUM_TOKENS="${DRAFT_NUM_TOKENS:-7}"

# Fork knobs that TabbyAPI has no config key for are read from the environment, which is what
# this block is for. It is intentionally EMPTY: the measured native numbers were taken with no
# EXL3_* override set, so leaving them unset keeps this route comparable to the native one.
# Candidates to try when tuning this route (see the sibling Qwen3.8 recipe for what they were
# worth on the box that recipe was measured on - none of it is measured for this pack on this
# card):
#   export EXL3_MOE_COOP_WIDE=1
#   export EXL3_GR_INT8=1
#   export EXL3_INT8_GEMV=0
#   export EXL3_DRAFT_CONFIDENCE=0.6
EXL3_ENV_DEFAULTS=""

# Validate the rendered TabbyAPI config against TabbyAPI's own schema. TabbyAPI has moved this
# module before; the function reports which one it found, and fails loudly rather than passing
# silently. Run it after setup.sh has installed TabbyAPI.
check_config() {
  local cfg="$1"
  [[ -f "$TABBY_DIR/main.py" ]] || die "no TabbyAPI at $TABBY_DIR; run: bash exllamav3-tabby/setup.sh"
  "$(render_py)" - "$cfg" "$TABBY_DIR" <<'PY'
import importlib, os, sys
import yaml

cfg_path, tabby = sys.argv[1], sys.argv[2]
sys.path.insert(0, tabby)
os.chdir(tabby)  # TabbyAPI resolves its own relative paths from its directory
loaded = yaml.safe_load(open(cfg_path, encoding="utf-8"))

candidates = [
    ("common.config_models", "TabbyConfigModel"),
    ("common.config", "Config"),
    ("util.config", "Config"),
    ("config", "Config"),
    ("tabby.config", "Config"),
]
found = None
for module_name, attribute in candidates:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        continue
    if hasattr(module, attribute):
        found = (module_name, attribute, getattr(module, attribute))
        break

if not found:
    print("could not locate TabbyAPI's config schema in any of: "
          + ", ".join(f"{m}.{a}" for m, a in candidates) + "\n"
          "  -> the config was NOT validated. Check TabbyAPI's own docs/args for the current "
          "entry point and extend the candidate list in exllamav3-tabby/env.sh.", file=sys.stderr)
    sys.exit(3)

module_name, attribute, model_cls = found
print(f"validating against {module_name}.{attribute}")
try:
    if hasattr(model_cls, "model_validate"):
        model_cls.model_validate(loaded)
    else:
        model_cls(**loaded)
except TypeError as exc:
    print(f"schema mismatch: {exc}", file=sys.stderr)
    sys.exit(4)
except Exception as exc:
    print(f"validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(4)
print("config is valid against TabbyAPI's own schema")
PY
}
