"""Run `cecheck` on every exported certificate directory and collect the reports.

    python export/run_cecheck.py --exported work/core/export --out results/core/genuine [--workers 2]

Writes <out>/<case>.json (the checker's report) and <out>/index.json (case, exit code, verdict, counts).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, ".lake", "build", "bin", "cecheck.exe" if os.name == "nt" else "cecheck")


def one(exe: str, case_dir: str, out: str):
    case = os.path.basename(case_dir)
    js = os.path.join(out, case + ".json")
    t = time.time()
    r = subprocess.run([exe, case_dir, "--json", js, "--quiet"], capture_output=True, text=True)
    rep = json.load(open(js, encoding="utf-8")) if os.path.exists(js) and r.returncode in (0, 1) else None
    return {"case": case, "exit_code": r.returncode, "verdict": rep["verdict"] if rep else "ERROR",
            "counts": rep["counts"] if rep else None, "seconds": round(time.time() - t, 1),
            "fails": [c["id"] for c in rep["checks"] if c["status"] == "FAIL"] if rep else [],
            "stderr": r.stderr.strip()[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exported", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exe", default=EXE, help="the cecheck binary (default: .lake/build/bin)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = sorted(d for d in os.listdir(a.exported) if os.path.isdir(os.path.join(a.exported, d)))
    if a.only:
        cases = [c for c in cases if c in a.only]
    rows = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(lambda c: one(a.exe, os.path.join(a.exported, c), a.out), cases):
            rows.append(row)
            print(f"{row['case']:40s} exit {row['exit_code']} {row['verdict']} {row['seconds']} s {row['fails']}", flush=True)
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
