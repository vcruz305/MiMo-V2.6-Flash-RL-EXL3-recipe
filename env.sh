# Shared defaults for the MiMo-V2.6-Flash-RL EXL3 serving recipe (needs a single ~96 GB-class GPU).
# Sourced by setup.sh, serve.sh, chat.sh, preflight.sh and exllamav3-tabby/env.sh.
# Every value can be overridden from the environment before calling those scripts.

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------
# Runtime: vcruz305/exllamav3. The fork is required - upstream/stock exllamav3 has no
# MiMo-V2 architecture and no DFlash draft port, so the model will not load on it.
EXL3_REPO="${EXL3_REPO:-https://github.com/vcruz305/exllamav3.git}"
EXL3_MIN_VERSION="${EXL3_MIN_VERSION:-1.5.1.post1}"
EXL3_WHEEL_TAG="${EXL3_WHEEL_TAG:-v1.5.1.post1}"   # released wheel: the preferred install
EXL3_WHEEL="${EXL3_WHEEL:-exllamav3-1.5.1.post1+cu128.torch2.11.0-cp312-cp312-linux_x86_64.whl}"
EXL3_REF="${EXL3_REF:-master}"                     # source fallback, used with --from-source

# The pack. 2.20 bpw is the servable rung for one 96 GB card (86.94 GB);
# 2.50 bpw is published but does not fit the card (98.48 GB, 26 files). See README.
PACK_REPO="${PACK_REPO:-vcruz305/MiMo-V2.6-Flash-RL-EXL3}"
PACK_SUBDIR="${PACK_SUBDIR:-2.20bpw}"
# The DFlash drafter ships in the ORIGINAL checkpoint, not in the pack.
BASE_MODEL_REPO="${BASE_MODEL_REPO:-XiaomiMiMo/MiMo-V2.6-Flash-RL}"
DRAFT_SUBDIR="${DRAFT_SUBDIR:-dflash}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RECIPE_HOME="${RECIPE_HOME:-$HOME/mimo-exl3}"
VENV="${VENV:-$RECIPE_HOME/venv}"
EXL3_SRC="${EXL3_SRC:-$RECIPE_HOME/exllamav3}"
STATE_DIR="${STATE_DIR:-$RECIPE_HOME/state}"
MODELS_DIR="${MODELS_DIR:-$RECIPE_HOME/models}"
PACK_HOME="${PACK_HOME:-$MODELS_DIR/MiMo-V2.6-Flash-RL-EXL3}"
PACK_DIR="${PACK_DIR:-$PACK_HOME/$PACK_SUBDIR}"
DRAFT_SRC="${DRAFT_SRC:-$MODELS_DIR/MiMo-V2.6-Flash-RL/$DRAFT_SUBDIR}"
DRAFT_FIXED="${DRAFT_FIXED:-$MODELS_DIR/MiMo-V2.6-Flash-RL-dflash-fixed}"
DRAFT_EXL3="${DRAFT_EXL3:-$MODELS_DIR/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0}"
# Default to the EXL3 4.0 bpw drafter when it has been built (tools/quantize_dflash.sh): same
# draft acceptance as the BF16 copy, ~3% faster decode, ~2.1 GB less VRAM, which is what raises
# the with-draft context ceiling. Falls back to the corrected BF16 drafter when it is absent.
if [[ -z "${DRAFT_DIR:-}" ]]; then
  if [[ -d "$DRAFT_EXL3" ]]; then DRAFT_DIR="$DRAFT_EXL3"; else DRAFT_DIR="$DRAFT_FIXED"; fi
fi

# Toolchain. Only the source build needs these; the released wheel ships a fat arch list
# (8.0 8.6 8.9 9.0 10.0 12.0+PTX) and needs a driver, not a toolkit.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="$VENV/.mimo-recipe-runtime"

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*" >&2; }

render_py() {  # prefer the recipe venv, fall back to the system interpreter
  if [[ -x "$VENV/bin/python" ]]; then echo "$VENV/bin/python"; else echo "$PYTHON_BIN"; fi
}

# ---------------------------------------------------------------------------
# Runtime check: the IMPORTED exllamav3 must be the fork, with this model's port.
# Agents and users most often go wrong here by picking up a stock wheel or the wrong venv,
# so this inspects the module that actually gets imported rather than a directory name.
# ---------------------------------------------------------------------------
verify_runtime() {
  local py; py="$(render_py)"
  command -v "$py" >/dev/null || die "no python at $py. Run: bash setup.sh"
  "$py" - "$EXL3_MIN_VERSION" <<'PY' || die "the exllamav3 in this environment is not the vcruz305 fork runtime with MiMo-V2 support. Run: bash setup.sh"
import sys
import exllamav3
from packaging.version import Version
from exllamav3.version import __version__ as v

try:
    from exllamav3.architecture import mimo_v2  # MiMo-V2 port: fork only
except Exception as exc:
    print(f"exllamav3 {v} at {exllamav3.__file__} has no MiMo-V2 architecture ({exc})", file=sys.stderr)
    sys.exit(1)

try:
    from exllamav3.architecture.dflash import DFlashConfig
except Exception as exc:
    print(f"exllamav3 {v} has no DFlash draft port ({exc})", file=sys.stderr)
    sys.exit(1)
if not hasattr(DFlashConfig, "tap_shift"):
    print("DFlash port has no tap_shift key: too old for the MiMo drafter", file=sys.stderr)
    sys.exit(1)

from argparse import ArgumentParser
from exllamav3.model_init import add_args
parser = ArgumentParser(add_help=False)
add_args(parser, cache=True, add_sampling_args=False, add_draft_model_args=True)
opts = set(parser._option_string_actions)
missing = [o for o in ("-cs", "--cache_size", "-ambs", "--autosplit_max_batch_size",
                       "-dm", "--draft_model_dir") if o not in opts]
if missing:
    print(f"exllamav3 {v} is missing launcher options: {missing}", file=sys.stderr)
    sys.exit(1)

if Version(v.split("+")[0]) < Version(sys.argv[1]):
    print(f"exllamav3 {v} < {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)
print(f"exllamav3 {v} (fork) at {exllamav3.__file__}", file=sys.stderr)
PY
  if [[ -f "$MARKER" ]]; then
    local built; built="$(cat "$MARKER")"
    [[ "$built" == *"$EXL3_REF"* || "$EXL3_REF" == *"$built"* ]] \
      || echo "warning: runtime was built from $built, the recipe default is $EXL3_REF." >&2
  fi
}

# ---------------------------------------------------------------------------
# Pack check: layout, architecture string, file count, size, template consistency.
# Deliberately no tensor-level detail: the pack's quantization_config.json carries a full
# map of how the model was quantized and none of it belongs in this repo's output.
# ---------------------------------------------------------------------------
verify_pack() {
  local dir="${1:-$PACK_DIR}"
  [[ -f "$dir/config.json" ]] || die "no pack at $dir (no config.json). Download it (README, Downloads) or set PACK_DIR."
  local py; py="$(render_py)"
  "$py" - "$dir" <<'PY'
import glob, json, os, sys, hashlib
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
arch = (cfg.get("architectures") or ["?"])[0]
if cfg.get("model_type") != "mimo_v2" or arch != "MiMoV2ForCausalLM":
    print(f"error: {d} is model_type={cfg.get('model_type')} architectures={arch}, expected the MiMo-V2 EXL3 pack", file=sys.stderr)
    sys.exit(1)
shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
total = sum(os.path.getsize(p) for p in shards)
files = len(os.listdir(d))
print(f"pack: {files} files, {len(shards)} safetensors, {total/1e9:.2f} GB, {cfg.get('num_hidden_layers')} layers", file=sys.stderr)

need = ["config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]
missing = [n for n in need if not os.path.isfile(os.path.join(d, n))]
if missing:
    print(f"error: pack is missing {missing}; a partial download cannot be served", file=sys.stderr)
    sys.exit(1)

# The template gotcha, checked here so it is visible before a 90-second load.
tpl = open(os.path.join(d, "chat_template.jinja"), encoding="utf-8").read()
emb = json.load(open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8")).get("chat_template")
norm = " ".join(tpl.split())
same_raw = isinstance(emb, str) and emb == tpl
same_norm = isinstance(emb, str) and " ".join(emb.split()) == norm
print(f"template: raw-identical={same_raw} whitespace-normalized-identical={same_norm} "
      f"normalized-sha256={hashlib.sha256(norm.encode()).hexdigest()}", file=sys.stderr)
if isinstance(emb, str) and not same_norm:
    print("error: chat_template.jinja and the embedded tokenizer template really differ; "
          "the server would (correctly) refuse this pack", file=sys.stderr)
    sys.exit(1)
if isinstance(emb, str) and not same_raw:
    print("note: they differ by whitespace only - expected for this pack, see README 'Template gotcha'", file=sys.stderr)

# Card fit. The 2.20 bpw pack is 86.94 GB and is the rung that fits a 96 GB card.
# The 2.50 bpw pack is 98.48 GB and does not. Resident MiB for 2.20 bpw was not
# in the SixCat speed files, so this check uses folder size only.
if total > 95e9:
    print(f"warning: this pack is {total/1e9:.2f} GB; the 96 GB card fits the 2.20 bpw rung "
          f"(86.94 GB), not this one", file=sys.stderr)
PY
}

# ---------------------------------------------------------------------------
# Drafter check: the two fixes from README "The DFlash fix" must be in effect.
# ---------------------------------------------------------------------------
verify_draft() {
  local dir="${1:-$DRAFT_DIR}"
  [[ -d "$dir" ]] || die "no draft folder at $dir. Build it: bash setup.sh (or python tools/fix_dflash.py)"
  local py; py="$(render_py)"
  "$py" - "$dir" <<'PY'
import json, os, struct, sys
d = sys.argv[1]
cfg_path = os.path.join(d, "config.json")
if not os.path.isfile(cfg_path):
    print(f"error: no config.json in {d}", file=sys.stderr); sys.exit(1)
cfg = json.load(open(cfg_path))
shift = (cfg.get("dflash_config") or {}).get("tap_shift", cfg.get("tap_shift"))
if shift != 0:
    print(f"error: {d}/config.json has tap_shift={shift!r}; expected 0 for this drafter "
          f"(a missing key defaults to 1 and collapses acceptance). Fix: python tools/fix_dflash.py", file=sys.stderr)
    sys.exit(1)
ids = (cfg.get("dflash_config") or {}).get("target_layer_ids")
print(f"drafter: tap_shift=0 target_layer_ids={ids}", file=sys.stderr)
shard = os.path.join(d, "mask_embedding.safetensors")
if not os.path.isfile(shard):
    print(f"error: no mask_embedding.safetensors in {d}; the loader looks for a tensor named "
          f"'mask_embedding' and will silently fall back to the untrained embedding row. "
          f"Fix: python tools/fix_dflash.py", file=sys.stderr)
    sys.exit(1)
with open(shard, "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(n))
entry = header.get("mask_embedding")
if not entry:
    print(f"error: {shard} has no 'mask_embedding' tensor", file=sys.stderr); sys.exit(1)
if entry.get("dtype") != "BF16" or entry.get("shape") != [4096]:
    print(f"error: unexpected mask_embedding {entry.get('dtype')} {entry.get('shape')}", file=sys.stderr); sys.exit(1)
if not os.path.isfile(os.path.join(d, "dflash_draft_model.safetensors")):
    print(f"error: {d} has no dflash_draft_model.safetensors", file=sys.stderr); sys.exit(1)
print(f"drafter: mask_embedding {entry['dtype']} {entry['shape']} in mask_embedding.safetensors", file=sys.stderr)
PY
}

# ---------------------------------------------------------------------------
# Card check: nothing else may be resident, and the pack must fit.
# ---------------------------------------------------------------------------
vram_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

check_card_free() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found; this recipe needs an NVIDIA driver"
  local used; used="$(vram_used_mib)"
  [[ -n "$used" ]] || die "nvidia-smi gave no GPU memory reading"
  if (( used > 2000 )); then
    echo "error: ${used} MiB already resident on the GPU. This recipe wants the whole card." >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader >&2
    exit 3
  fi
  say "card free (${used} MiB in use)"
}

show_card() {
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
    --format=csv,noheader 2>/dev/null || true
}

# Sizing sanity. cache_size is ONE pool shared by every concurrent request, not a
# per-request allowance; that mismatch is what produces "my context is tiny" reports.
check_sizing() {
  local cache="$1" max_seq="$2" batch="$3" ambs="$4"
  (( cache % 256 == 0 )) || die "CACHE_SIZE must be a multiple of 256"
  (( cache >= max_seq )) || die "CACHE_SIZE ($cache) < MAX_SEQ_LEN ($max_seq): one full-length request would not fit"
  (( batch >= 1 )) || die "MAX_ACTIVE_REQUESTS must be >= 1"
  (( batch <= ambs )) || die "MAX_ACTIVE_REQUESTS ($batch) > AUTOSPLIT_MAX_BATCH ($ambs): the server refuses this"
  (( max_seq <= 65536 )) || echo "warning: MAX_SEQ_LEN > 65536 is past what has been measured for this recipe (the checkpoint's own position limit is 1048576, but nothing beyond 65536 is validated here)" >&2
  say "sizing: ${max_seq} tokens per request, ${cache}-token pool shared by ${batch} concurrent requests"
}

port_free() {
  local port="$1"
  local py; py="$(render_py)"
  "$py" - "$port" <<'PY' || die "port $1 is already in use"
import socket, sys
port = int(sys.argv[1])
with socket.socket() as s:
    try:
        s.bind(("127.0.0.1", port))
    except OSError as exc:
        print(f"error: port {port} is taken ({exc})", file=sys.stderr)
        sys.exit(1)
PY
}
