"""Write the 80 forged certificates of the C verifier's validation (verifier/validation/run_validation.py) for
the core-Lean checker, without touching the code repository.

The forgeries are made by the functions of `verifier/validation/run_validation.py` itself (imported read-only:
`MUTATIONS`, `read_npz`, `write_npz`, `canonical`, `reseal`), applied to the same candidates in the same order
as its `main()`: the first N certificates (default 20) whose S2 or S3 status is neither CERTIFIED nor
NOT_CHECKED, four forgeries each (`radius`, `array_hash`, `body_hash`, `status`). The output directory gets
<case>__<mutation>.json and .arrays.npz, plus `index.json` in the format `export_certs.py` reads (the operator
comes from the original bundle, as the C verifier is given it).

    python export/make_mutants.py --code <certify-ed checkout> --certs work/core/certs --out work/core/mutants
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="the certify-ed checkout (read only)")
    ap.add_argument("--certs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutants", type=int, default=20)
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.join(a.code, "verifier", "validation"))
    import run_validation as rv  # noqa: E402  (the C verifier's own harness; only its functions are used)

    index = json.load(open(os.path.join(a.certs, "index.json"), encoding="utf-8"))
    candidates = [r for r in index if not r["refused"] and any(
        r["statuses"].get(s) not in ("CERTIFIED", "NOT_CHECKED", None) for s in ("S2.eigenpairs", "S3.completeness"))]
    candidates = candidates[:a.mutants]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for row in candidates:
        cert = os.path.join(a.certs, row["certificate"])
        body = json.load(open(cert, encoding="utf-8"))
        arrays = rv.read_npz(os.path.join(a.certs, body["arrays"]["sidecar"]))
        for mname, fn in rv.MUTATIONS:
            res = fn(body, arrays)
            if res is None:
                continue
            mbody, marrays, what = res
            stem = os.path.join(a.out, f"{row['case']}__{mname}")
            mbody["arrays"]["sidecar"] = os.path.basename(stem) + ".arrays.npz"
            mbody = rv.reseal(mbody) if mname != "body_hash" else mbody
            with open(stem + ".json", "w", encoding="utf-8") as f:
                f.write(rv.canonical(mbody))
            rv.write_npz(stem + ".arrays.npz", marrays)
            rows.append({"case": f"{row['case']}__{mname}", "origin_case": row["case"], "mutation": mname,
                         "what": what, "certificate": os.path.basename(stem) + ".json", "bundle": row["bundle"],
                         "refused": False, "statuses": row["statuses"]})
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(rows)} forgeries from {len(candidates)} certificates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
