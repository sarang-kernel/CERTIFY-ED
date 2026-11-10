"""certify-ed's own integrity check (`Certificate.load(path, verify=True)`: body hash, array hashes, scope block)
on the forged certificates, for the agreement table. Imports certify_ed (read only).

    python export/python_integrity.py --index work/core/mutants/index.json --certs work/core/mutants --out results/core/python_integrity_mutants.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--certs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from certify_ed.certificate_v2 import Certificate
    rows = []
    for row in json.load(open(a.index, encoding="utf-8")):
        if row.get("refused"):
            continue
        path = os.path.join(a.certs, row["certificate"])
        try:
            Certificate.load(path, verify=True, validate_schema=False)
            rows.append({"case": row["case"], "integrity_ok": True, "error": ""})
        except Exception as e:  # noqa: BLE001 - the refusal is the result
            rows.append({"case": row["case"], "integrity_ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
        f.write("\n")
    print(sum(r["integrity_ok"] for r in rows), "of", len(rows), "pass the integrity check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
