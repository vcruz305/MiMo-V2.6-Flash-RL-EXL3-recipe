#!/usr/bin/env python3
"""Stage the DFlash drafter fix into a COPY of the drafter folder (CPU only, no CUDA).

Two faults in the shipped MiMo-V2.6-Flash-RL drafter are corrected here. Both are
checkpoint/config edits: no exllamav3 code is changed, so the fix survives a runtime
update that keeps the same reader keys.

  1. config.json: add "tap_shift": 0 inside dflash_config.
     The drafter's own reference implementation conditions on
     hidden_states[target_layer_ids[i] + 1], i.e. the OUTPUT of target layer
     target_layer_ids[i], and exllamav3's export index j is the output of layer j, so the
     raw ids [0, 11, 23, 35, 47] are the correct taps. exllamav3's DFlash port defaults to
     tap_shift = 1 (right for the original z-lab drafters), which feeds the drafter layers
     1, 12, 24, 36, 48 instead; the shipped config.json does not pin the key, so the wrong
     default applies and acceptance collapses (measured mean draft_accept 0.0418 before,
     0.9452 after).

  2. mask_embedding.safetensors: a new one-tensor shard holding the learned mask
     embedding that ships only as mask_embedding.pt. The port loads the target's own
     embedding row for the mask token unless a tensor literally named "mask_embedding" is
     found in the tensor collection, and that collection globs *.safetensors in the
     drafter directory - so dropping the shard in is enough. The shard is small: the
     vector is bf16, shape [4096], 8192 bytes of data.

Usage:
  python tools/fix_dflash.py                      # $DRAFT_SRC -> $DRAFT_DIR (see env.sh)
  python tools/fix_dflash.py --src DIR --dst DIR
  python tools/fix_dflash.py --src DIR --dst DIR --force   # overwrite an existing copy

The source folder is only ever read. The 2.94 GB drafter shard is never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

# This is a CPU-only staging tool on purpose: hide any GPU before torch is imported, so a
# copy of the drafter can be prepared while a server owns the card.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

KEY = "mask_embedding"
NEW_SHARD = "mask_embedding.safetensors"


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    recipe = os.path.dirname(here)
    default_src = os.environ.get("DRAFT_SRC", "")
    default_dst = os.environ.get("DRAFT_DIR", "")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=default_src, help="pristine drafter folder (dflash/ from the original checkpoint)")
    parser.add_argument("--dst", default=default_dst, help="where to build the corrected copy")
    parser.add_argument("--force", action="store_true", help="replace an existing --dst")
    args = parser.parse_args()

    if not args.src or not args.dst:
        print("error: pass --src/--dst, or export DRAFT_SRC/DRAFT_DIR (see env.sh)", file=sys.stderr)
        return 2
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if not os.path.isfile(os.path.join(src, "config.json")):
        print(f"error: no config.json in {src}", file=sys.stderr)
        return 2
    if src == dst or dst.startswith(src + os.sep):
        print("error: --dst must be a different folder from --src (never edit the original)", file=sys.stderr)
        return 2
    if os.path.exists(dst):
        if not args.force:
            print(f"note: {dst} already exists; checking it instead of rebuilding (use --force to rebuild)")
            return verify(dst, src)
        shutil.rmtree(dst)

    # --- 0. copy -----------------------------------------------------------
    shutil.copytree(src, dst, symlinks=True)
    report: dict[str, object] = {"src": src, "dst": dst, "recipe": recipe}
    report["src_files"] = {f: os.path.getsize(os.path.join(src, f)) for f in sorted(os.listdir(src))}
    src_md5 = {f: md5(os.path.join(src, f)) for f in sorted(os.listdir(src))
               if os.path.isfile(os.path.join(src, f))}

    # --- 1. tap_shift ------------------------------------------------------
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    dflash = cfg.setdefault("dflash_config", {})
    report["target_layer_ids"] = dflash.get("target_layer_ids")
    report["tap_shift_before"] = dflash.get("tap_shift")
    dflash["tap_shift"] = 0
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    report["tap_shift_after"] = 0

    # --- 2. mask embedding shard ------------------------------------------
    import torch
    from safetensors.torch import save_file

    pt_path = os.path.join(src, "mask_embedding.pt")
    if not os.path.isfile(pt_path):
        print(f"error: {pt_path} is missing; the drafter ships that vector as a side file", file=sys.stderr)
        return 2
    with torch.no_grad():
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "embedding" not in payload:
        print(f"error: {pt_path} is not the expected {{'mask_token_id', 'embedding'}} dict", file=sys.stderr)
        return 2
    emb = payload["embedding"]
    if not torch.is_tensor(emb) or emb.dtype != torch.bfloat16 or emb.dim() != 1:
        print(f"error: unexpected embedding {getattr(emb, 'dtype', None)} shape {getattr(emb, 'shape', None)}", file=sys.stderr)
        return 2
    emb = emb.contiguous()
    shard_path = os.path.join(dst, NEW_SHARD)
    save_file({KEY: emb}, shard_path)
    report["mask_token_id"] = int(payload.get("mask_token_id", -1))
    report["embedding_dtype"] = str(emb.dtype)
    report["embedding_shape"] = list(emb.shape)
    report["shard_bytes"] = os.path.getsize(shard_path)

    # --- 3. index (belt and braces; the runtime loader globs *.safetensors) -
    idx_path = os.path.join(dst, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        if KEY not in idx.get("weight_map", {}):
            idx["weight_map"][KEY] = NEW_SHARD
            if isinstance(idx.get("metadata"), dict) and "total_size" in idx["metadata"]:
                idx["metadata"]["total_size"] += emb.numel() * emb.element_size()
            with open(idx_path, "w", encoding="utf-8") as f:
                json.dump(idx, f, indent=4)
                f.write("\n")
            report["index_updated"] = True

    # --- 4. prove the original was only read -------------------------------
    unchanged = {f: md5(os.path.join(src, f)) == h for f, h in src_md5.items()}
    report["src_unchanged"] = all(unchanged.values())
    if not report["src_unchanged"]:
        print(f"error: the source folder changed during the run: {unchanged}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    sidecar = os.path.join(os.path.dirname(dst), "dflash-fix.report.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({"src": src, "dst": dst, "src_md5": src_md5, "report": report}, f, indent=2)
        f.write("\n")
    print(f"\nprovenance written to {sidecar} (sha-free record of the original's md5s, read by tools/verify_dflash.py)")
    print(f"staged. Serve it with: PROFILE=with-draft bash serve.sh   (-dm {dst})")
    print(f"Check it with:         python tools/verify_dflash.py --dir {dst}")
    return 0


def verify(dst: str, src: str) -> int:
    cfg = json.load(open(os.path.join(dst, "config.json"), encoding="utf-8"))
    shift = (cfg.get("dflash_config") or {}).get("tap_shift", cfg.get("tap_shift"))
    shard = os.path.join(dst, NEW_SHARD)
    ok = shift == 0 and os.path.isfile(shard)
    print(f"{dst}: tap_shift={shift!r} mask_embedding.safetensors={os.path.isfile(shard)} -> {'ok' if ok else 'NEEDS FIX (re-run with --force)'}")
    if src and os.path.isdir(src):
        same = sorted(os.listdir(src)) == sorted(os.listdir(dst))
        print(f"file lists match the original folder: {same}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
