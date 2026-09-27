"""Launch the retained France profile on spark-724a (GB10, sm_121).

Adapted from the France `start_quant_readback.py`: same server, same args, same env and the
same UMA guard contract. Differences, all forced by the box:

  * everything lives under $HOME/mimo-exl3 instead of /workspace,
  * the runtime is a source build of the fork branch `feat/mixedk-handled-readback-elision`,
    so EXL3_ROOT/PYTHONPATH shadowing is unnecessary (the elision code is in the installed
    tree) and the France DSO hash does not apply,
  * the France "deployed patch" marker files are replaced by inline checks of the same facts,
  * the guard's lock path is configurable (France hardcodes /workspace/mimo-tune/server.lock).
"""
import json, os, socket, subprocess, sys, time
from pathlib import Path

H = Path(os.environ.get("RECIPE_HOME", Path.home() / "mimo-exl3"))
VENV = H / "venv"
PY = VENV / "bin" / "python"
BASE = H
sys.path.insert(0, str(H / "guard"))

from guard_uma import memory_sample, require_load_headroom  # noqa: E402

PACK = H / "models" / "MiMo-V2.6-Flash-RL-EXL3" / "2.50bpw"
DRAFT = H / "models" / "MiMo-V2.6-Flash-RL-dflash-EXL3-4.0"
SERVER = H / "repo" / "server" / "serve_native.py"
GUARD = H / "guard" / "guard_uma.py"

# ---- preconditions (fail closed; never fall back to a different profile) -------------
assert SERVER.is_file(), f"server missing: {SERVER}"
assert GUARD.is_file(), f"guard missing: {GUARD}"
cfg_pack = json.loads((PACK / "config.json").read_text())
assert cfg_pack.get("model_type") == "mimo_v2", "pack is not the MiMo-V2 EXL3 pack"
shards = sorted(PACK.glob("*.safetensors"))
assert len(shards) == 13, f"expected 13 shards, found {len(shards)}"
total = sum(p.stat().st_size for p in shards)
assert total > 98e9, f"shards look incomplete: {total/1e9:.2f} GB"
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "quantization_config.json", "model.safetensors.index.json"):
    assert (PACK / f).is_file(), f"pack missing {f}"
draft_cfg = json.loads((DRAFT / "config.json").read_text())
shift = (draft_cfg.get("dflash_config") or {}).get("tap_shift", draft_cfg.get("tap_shift"))
assert shift == 0, f"drafter tap_shift={shift!r}, expected 0"
assert (DRAFT / "mask_embedding.safetensors").is_file(), "drafter mask_embedding missing"
# SPARK724A_DRAFT_NAME_FIX: the published EXL3 drafter repo ships model.safetensors; the local
# fix_dflash build is dflash_draft_model.safetensors. The loader globs *.safetensors, so either works.
_draft_w = [p for p in (DRAFT / "dflash_draft_model.safetensors", DRAFT / "model.safetensors") if p.is_file()]
assert _draft_w, "drafter weights missing (no dflash_draft_model.safetensors or model.safetensors)"
import struct as _struct
with open(_draft_w[0], "rb") as _f:
    _n, = _struct.unpack("<Q", _f.read(8))
    _hdr = json.loads(_f.read(_n))
assert any(k.startswith("layers.") for k in _hdr), f"{_draft_w[0].name} has no draft layers"

for p in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        cmd = p.read_bytes()
        if b"serve_native.py" in cmd:
            raise SystemExit(f"a serve is already running: pid {p.parent.name}")
    except FileNotFoundError:
        pass
assert "ELIDE_HANDLED" in (H / "exllamav3" / "exllamav3" / "modules" / "block_sparse_mlp.py").read_text(), \
    "the installed runtime does not carry the handled-readback elision"

deadline = time.monotonic() + 90
while True:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 8096))
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(1)

sample = memory_sample()
print("LOAD_PREFLIGHT", json.dumps(sample), flush=True)
require_load_headroom(sample, memory_policy="uma")

# ---- launch ------------------------------------------------------------------------
run = BASE / "runs" / ("spark724a-retained-" + str(time.time_ns()))
cfg = {
    "outdir": str(run),
    "memory_policy": "uma",
    "max_seconds": 43200,
    "lock_path": str(BASE / "server.lock"),
    "env": {
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": f"{VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED": "1",
        "EXL3_UMA": "1",
        "EXL3_UMA_RESERVE_MB": "8192",
        "TORCH_CUDA_ARCH_LIST": "12.1",
        "OMP_NUM_THREADS": "8",
        "EXL3_BATCH_VERIFY": "0",
        "EXL3_MOE_MIXEDK_ELIDE_HANDLED": "1",
    },
    "command": [
        str(PY), str(SERVER),
        "-m", str(PACK),
        "-dm", str(DRAFT),
        "-ndt", "7", "-dds", "-dc", "0.6", "-gs", "106",
        "-cs", "4096", "-cq", "4", "-ambs", "1", "-chunk_size", "4096",
        "-ccs", "0", "-rcs", "0.25",
        "--max-active-requests", "1", "--max-pending-requests", "2",
        "--max-model-len", "4096", "--host", "127.0.0.1", "--port", "8096",
        "--request-timeout", "600", "-lv",
    ],
}
run.mkdir(parents=True, exist_ok=True)
(run / "launch.json").write_text(json.dumps(cfg, indent=2))
active = BASE / "active-run.txt"
active.write_text(str(run))
with (BASE / "guard.log").open("w") as log:
    child = subprocess.Popen(["/usr/bin/python3", str(GUARD), str(run / "launch.json")],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
time.sleep(2)
print("LAUNCHED", json.dumps({"guard_pid": child.pid, "running": child.poll() is None, "run": str(run)}), flush=True)
if child.poll() is not None:
    print((BASE / "guard.log").read_text(), flush=True)
    raise SystemExit(1)
