"""Build the verifier binary (verifier/src/cev.c) with the first C compiler found.

    python verifier/build.py [--cc PATH] [--out verifier/build/cev[.exe]]

Compiler search order: --cc, the environment variable CEV_CC, then gcc, clang, cc on PATH. The flags
-frounding-math -ffp-contract=off are required (directed rounding, no fused multiply-add); the build fails
loudly when no compiler is found. Standard library only; this script shares no code with certify_ed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src", "cev.c")
FLAGS = ["-std=c11", "-O1", "-frounding-math", "-ffp-contract=off", "-fno-builtin", "-Wall", "-Wextra",
         "-Wno-unknown-pragmas"]


def find_cc(explicit: str | None = None) -> str | None:
    for cand in [explicit, os.environ.get("CEV_CC"), "gcc", "clang", "cc"]:
        if not cand:
            continue
        path = cand if os.path.isabs(cand) and os.path.exists(cand) else shutil.which(cand)
        if path:
            return path
    return None


def default_out() -> str:
    return os.path.join(HERE, "build", "cev.exe" if os.name == "nt" else "cev")


def build(cc: str | None = None, out: str | None = None, quiet: bool = False) -> str:
    cc_path = find_cc(cc)
    if cc_path is None:
        raise RuntimeError("no C compiler found (set CEV_CC or put gcc/clang/cc on PATH)")
    out = out or default_out()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(cc_path) + os.pathsep + env.get("PATH", "")   # MinGW gcc needs its own bin dir
    cmd = [cc_path, *FLAGS, "-o", out, SRC, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"compilation failed ({cc_path}):\n{r.stderr}")
    if not quiet:
        print(f"built {out} with {cc_path}")
        if r.stderr.strip():
            print(r.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc")
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        build(a.cc, a.out)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
