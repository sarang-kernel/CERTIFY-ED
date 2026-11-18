"""Run the C verifier of certify-ed (`verifier/src/cev.c`, built with `verifier/build.py --out ...`) on the
validation certificates and on the forgeries, the way `verifier/validation/run_validation.py` runs it (the
producer bundle's `operator.npz` is passed with --operator when the bundle has one), and collect the results
for the agreement table.

    python export/run_cev.py --cev work/core/bin/cev.exe --index work/core/certs/index.json --certs work/core/certs \
        --out work/core/cev/genuine --summary results/core/cev_genuine.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cev", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--certs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for row in json.load(open(a.index, encoding="utf-8")):
        if row.get("refused"):
            continue
        cert = os.path.join(a.certs, row["certificate"])
        op = os.path.join(row["bundle"], "operator.npz")
        js = os.path.join(a.out, row["case"] + ".json")
        cmd = [a.cev, cert, "--quiet", "--json", js] + (["--operator", op] if os.path.exists(op) else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        summ = json.load(open(js)) if r.returncode in (0, 1) and os.path.exists(js) else None
        rows.append({"case": row["case"], "exit_code": r.returncode,
                     "fails": [c["id"] for c in summ["checks"] if c["status"] == "FAIL"] if summ else [],
                     "reproduced": {c["id"]: c["status"] for c in summ["checks"]
                                    if c["status"] in ("REPRODUCED", "NOT_REPRODUCED")} if summ else {},
                     "operator_given": os.path.exists(op)})
        print(row["case"], r.returncode, rows[-1]["fails"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.summary)), exist_ok=True)
    with open(a.summary, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
