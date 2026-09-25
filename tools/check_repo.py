#!/usr/bin/env python3
"""Repository hygiene check for this recipe. Run it before every push:

    python3 tools/check_repo.py

Checks, all of them static (nothing is executed except `bash -n`):
  1. every file is LF-only (no CR bytes) and has no trailing whitespace runs at EOL;
  2. every referenced repo path resolves - markdown links and backticked paths in the docs;
  3. every .sh passes `bash -n`, every .py compiles;
  4. every .sh/.py carries the git executable bit in the index (when run inside a repo);
  5. no internal absolute paths and no quantization-method internals in any file.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEXT_EXT = {".md", ".sh", ".py", ".yml", ".yaml", ".json", ".env", ".example", ".gitattributes", ".gitignore"}
DIR_PREFIXES = ("server/", "tools/", "configs/", "exllamav3-tabby/")
LINK_EXT = {".md", ".sh", ".py", ".yml", ".yaml", ".json", ".env", ".jinja", ".txt", ".pdf", ".example"}

# Substrings that must never appear. Absolute paths from the machine this recipe was
# developed on, and anything that would describe how the pack was quantized.
FORBIDDEN = [
    "/home/", "10.80.10.", "id_ed25519", "frosty", "C:\\Users",
    "SAGE-EXL3", "sage_", "sentinel", "k_map", "K-hist", "K histogram",
    "wave budget", "layer budget", "calibration source", "calibration corpus",
    "calibration set", "calibration volume", "solver", "allocation rule", "per-tensor",
    "replay procedure", "measurement procedure",
]

LINK_RX = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
BACKTICK_RX = re.compile(r"`([^`\n]+)`")
PATHY_RX = re.compile(r"^[A-Za-z0-9_./-]+$")


def walk_files() -> list[str]:
    out = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for name in sorted(names):
            out.append(os.path.relpath(os.path.join(base, name), ROOT))
    return sorted(out)


def check_line_endings(files: list[str], problems: list[str]) -> None:
    for rel in files:
        ext = os.path.splitext(rel)[1]
        if ext not in TEXT_EXT and os.path.basename(rel) not in (".gitattributes", ".gitignore", ".env.example"):
            continue
        with open(os.path.join(ROOT, rel), "rb") as f:
            raw = f.read()
        if b"\r" in raw:
            problems.append(f"{rel}: contains CR bytes (must be LF-only)")
        for i, line in enumerate(raw.split(b"\n"), 1):
            if line.rstrip() != line.rstrip(b" \t"):
                problems.append(f"{rel}:{i}: trailing whitespace")


def check_paths(files: list[str], problems: list[str]) -> int:
    checked = 0
    for rel in files:
        if not rel.endswith(".md"):
            continue
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        here = os.path.dirname(rel)

        for target in LINK_RX.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")) or target.startswith("/"):
                continue
            target = target.split("#")[0]
            if not target:
                continue
            checked += 1
            for candidate in (os.path.join(here, target), target):
                if os.path.exists(os.path.join(ROOT, candidate)):
                    break
            else:
                problems.append(f"{rel}: link target does not exist: {target}")

        for token in BACKTICK_RX.findall(text):
            token = token.strip()
            if not PATHY_RX.match(token) or "/" not in token:
                continue
            if "$" in token or "*" in token or "~" in token or token.startswith("-"):
                continue
            looks_like_path = os.path.splitext(token)[1] in LINK_EXT or token.startswith(DIR_PREFIXES)
            if not looks_like_path:
                continue
            checked += 1
            target = token.rstrip("/")
            if not os.path.exists(os.path.join(ROOT, target)):
                problems.append(f"{rel}: backticked path does not exist: {token}")
    return checked


def check_syntax(files: list[str], problems: list[str]) -> None:
    has_bash = subprocess.run(["bash", "-c", "true"], capture_output=True).returncode == 0
    if not has_bash:
        problems.append("bash not found: .sh syntax was NOT checked")
    for rel in files:
        path = os.path.join(ROOT, rel)
        if rel.endswith(".sh"):
            if has_bash:
                done = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
                if done.returncode != 0:
                    problems.append(f"{rel}: bash -n failed: {done.stderr.strip()}")
        elif rel.endswith(".py"):
            try:
                compile(open(path, encoding="utf-8").read(), rel, "exec")
            except SyntaxError as exc:
                problems.append(f"{rel}: python syntax error: {exc}")


def check_exec_bits(files: list[str], problems: list[str]) -> None:
    done = subprocess.run(["git", "-C", ROOT, "ls-files", "-s"], capture_output=True, text=True)
    if done.returncode != 0:
        print("note: not a git repo (or git missing); exec-bit check skipped")
        return
    modes = {}
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            modes[parts[3].replace("\\", "/")] = parts[0]
    for rel in files:
        if not (rel.endswith(".sh") or rel.endswith(".py")):
            continue
        key = rel.replace(os.sep, "/")
        mode = modes.get(key, "100644")
        if mode != "100755":
            problems.append(f"{rel}: not executable in the git index (mode {mode}); git update-index --chmod=+x {key}")


def check_forbidden(files: list[str], problems: list[str]) -> None:
    for rel in files:
        if rel.replace(os.sep, "/") == "tools/check_repo.py":
            continue  # this file has to spell the needles out
        ext = os.path.splitext(rel)[1]
        if ext not in TEXT_EXT and os.path.basename(rel) not in (".gitattributes", ".gitignore", ".env.example"):
            continue
        text = open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace").read()
        for needle in FORBIDDEN:
            if needle in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if needle in line:
                        problems.append(f"{rel}:{i}: forbidden string {needle!r}")


def main() -> int:
    files = walk_files()
    problems: list[str] = []
    check_line_endings(files, problems)
    checked = check_paths(files, problems)
    check_syntax(files, problems)
    check_exec_bits(files, problems)
    check_forbidden(files, problems)

    print(f"files: {len(files)}")
    for rel in files:
        print(f"  {os.path.getsize(os.path.join(ROOT, rel)):>9d}  {rel}")
    print(f"paths checked in docs: {checked}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("\nall checks passed (LF-only, links resolve, syntax ok, exec bits set, no internal paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
