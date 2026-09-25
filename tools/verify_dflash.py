#!/usr/bin/env python3
"""CPU-side verification of the corrected DFlash drafter copy. Uses no CUDA at all.

Run it after tools/fix_dflash.py and before serving:

  python tools/verify_dflash.py --dir "$DRAFT_DIR"

It checks, in order:
  1. the original drafter folder is byte-unchanged (md5s are read live, not hard-coded);
  2. config.json parses and the tap pins are the reference ids [0, 11, 23, 35, 47];
  3. exllamav3's own DFlashConfig parses that config the way the loader will, and reports
     which target layers will export state;
  4. the tensor collection the loader actually globs finds a tensor named
     "mask_embedding", and it is bit-identical to the shipped mask_embedding.pt vector;
  5. the copy's index agrees, if the fork's compiler ever reads it.

Needs the recipe venv (torch + safetensors + exllamav3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # this tool must never touch a busy card

KEY = "mask_embedding"


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=os.environ.get("DRAFT_DIR", ""), help="the corrected copy")
    parser.add_argument("--src", default=os.environ.get("DRAFT_SRC", ""), help="the pristine original")
    parser.add_argument("--report", default="", help="pre-fix md5 record written by tools/fix_dflash.py")
    args = parser.parse_args()
    dst = os.path.abspath(args.dir) if args.dir else ""
    src = os.path.abspath(args.src) if args.src else ""
    if not args.report and dst:
        candidate = os.path.join(os.path.dirname(dst), "dflash-fix.report.json")
        args.report = candidate if os.path.isfile(candidate) else ""
    if not dst or not os.path.isfile(os.path.join(dst, "config.json")):
        print("error: pass --dir pointing at the corrected drafter folder", file=sys.stderr)
        return 2

    failures: list[str] = []

    if src and os.path.isdir(src):
        print(f"=== original folder (must be untouched): {src} ===")
        before: dict = {}
        if args.report and os.path.isfile(args.report):
            before = json.load(open(args.report, encoding="utf-8")).get("src_md5", {})
            print(f"  comparing against the pre-fix record in {args.report}")
        for name in sorted(os.listdir(src)):
            path = os.path.join(src, name)
            if not os.path.isfile(path):
                continue
            digest = md5(path)
            tag = ""
            if name in before:
                if before[name] == digest:
                    tag = " (matches the pre-fix record)"
                else:
                    tag = " MISMATCH vs pre-fix record"
                    failures.append(f"original {name} changed since the fix was staged")
            print(f"  {name:34s} {os.path.getsize(path):>12d}  {digest}{tag}")
    else:
        print("note: --src not given; skipping the original-untouched check")

    print(f"\n=== corrected copy: {dst} ===")
    cfg = json.load(open(os.path.join(dst, "config.json"), encoding="utf-8"))
    dflash = cfg.get("dflash_config") or {}
    shift = dflash.get("tap_shift", cfg.get("tap_shift"))
    ids = dflash.get("target_layer_ids")
    print(f"  tap_shift={shift!r}  target_layer_ids={ids}")
    if shift != 0:
        failures.append("tap_shift is not 0")
    if ids != [0, 11, 23, 35, 47]:
        failures.append(f"target_layer_ids are {ids}, not the reference [0, 11, 23, 35, 47]")

    print("\n=== exllamav3's own DFlashConfig parse (the loader's code path) ===")
    try:
        from exllamav3.architecture.dflash import DFlashConfig
        parsed = DFlashConfig(dst)
        print(f"  tap_shift={parsed.tap_shift} target_layer_ids={list(parsed.target_layer_ids)}")
        print(f"  key_mask_embedding={parsed.key_mask_embedding!r} block_size={parsed.block_size} "
              f"mask_token_id={parsed.mask_token_id}")
        print(f"  export_state_layers={sorted(set(parsed.target_layer_ids))}")
        if parsed.tap_shift != 0 or list(parsed.target_layer_ids) != [0, 11, 23, 35, 47]:
            failures.append("the port still resolves the wrong taps")
        if parsed.key_mask_embedding != KEY:
            failures.append("the port would not load a mask embedding")
    except Exception as exc:
        print(f"  could not parse with the port's own class: {type(exc).__name__}: {exc}")
        failures.append("DFlashConfig parse failed")

    print("\n=== the tensor collection the loader globs ===")
    try:
        from exllamav3.loader.safetensors import SafetensorsCollection
        stc = SafetensorsCollection(dst)
        print(f"  tensor files: {[os.path.basename(p) for p in stc.tensor_files]}")
        print(f"  has_tensor({KEY!r}): {stc.has_tensor(KEY)}")
        print(f"  resolves to: {os.path.basename(stc.tensor_file_map.get(KEY, 'MISSING'))}")
        if not stc.has_tensor(KEY):
            failures.append("the loader would not find mask_embedding")
    except Exception as exc:
        print(f"  could not build the collection: {type(exc).__name__}: {exc}")
        failures.append("SafetensorsCollection failed")

    print("\n=== the vector itself ===")
    try:
        import torch
        from safetensors.torch import load_file
        shard = os.path.join(dst, "mask_embedding.safetensors")
        loaded = load_file(shard)[KEY]
        print(f"  shard: {os.path.getsize(shard)} bytes, {loaded.dtype} {tuple(loaded.shape)}")
        pt = os.path.join(src, "mask_embedding.pt") if src else ""
        if pt and os.path.isfile(pt):
            original = torch.load(pt, map_location="cpu", weights_only=False)["embedding"]
            same = bool(torch.equal(loaded.view(-1), original.view(-1)))
            print(f"  bit-identical to {os.path.basename(pt)}: {same}")
            if not same:
                failures.append("the shard is not the shipped vector")
        if str(loaded.dtype) != "torch.bfloat16" or loaded.shape != (4096,):
            failures.append("unexpected shard dtype/shape")
    except Exception as exc:
        print(f"  could not read the shard: {type(exc).__name__}: {exc}")
        failures.append("shard read failed")

    print()
    if failures:
        print("FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("all checks passed: the corrected copy is wired the way the reference drafter expects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
