"""Produce the certificates the verifier is validated against (verifier/README.md, section Validation).

Runs certify-ed of this checkout on every corpus case (corpus/cases) and every conformance fixture
(conformance/reference) with Options(store_eigenvectors=True), so that the sidecar carries the claimed vectors,
and writes <out>/<case>.json + <case>.arrays.npz plus <out>/index.json (case, the expected.json verdicts,
certify-ed's statuses). A case certify-ed refuses (non-finite input) is recorded with "refused": true and no
certificate. This is the producer side of the validation and the only script under verifier/ that imports
certify_ed; the verifier itself shares no code with it.

    python verifier/make_validation_set.py --out verifier/validation/certs
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("verifier", "validation", "certs"))
    ap.add_argument("--cases", default=os.path.join("corpus", "cases"))
    ap.add_argument("--conformance", default=os.path.join("conformance", "reference"))
    ap.add_argument("--no-vectors", action="store_true", help="do not store the eigenvectors in the sidecar")
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    from certify_ed.certify import Options, certify
    from certify_ed.importers import claim_from_bundle

    os.makedirs(a.out, exist_ok=True)
    index = []
    dirs = [(os.path.join(a.cases, d), "corpus") for d in sorted(os.listdir(a.cases))
            if os.path.isdir(os.path.join(a.cases, d))]
    dirs += [(os.path.join(a.conformance, d), "conformance") for d in sorted(os.listdir(a.conformance))
             if os.path.isdir(os.path.join(a.conformance, d))]
    for src, origin in dirs:
        name = os.path.basename(src)
        row = {"case": name, "origin": origin, "bundle": src.replace(os.sep, "/")}
        exp_path = os.path.join(src, "expected.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                exp = json.load(f)
            row["expected"] = {k: exp.get(k) for k in ("error_class", "must_not_certify", "must_certify",
                                                        "may_certify", "refusal_ok", "conditional")}
        try:
            claim, spec, meta = claim_from_bundle(src)
            cert = certify(claim, spec, Options(store_eigenvectors=not a.no_vectors))
            out = cert.save(os.path.join(a.out, name))
            row["certificate"] = os.path.basename(out)
            row["statuses"] = {s["id"]: s["status"] for s in cert.statements}
            row["refused"] = False
        except Exception as e:  # noqa: BLE001 - a refusal is a result of this run, not an error of the script
            row["refused"] = True
            row["error"] = f"{type(e).__name__}: {e}"
        index.append(row)
        print(f"{name:40s} {'refused' if row['refused'] else ' '.join(v[0] for v in row['statuses'].values())}")
    with open(os.path.join(a.out, "index.json"), "w", newline="\n") as f:
        json.dump(index, f, indent=1, sort_keys=True)
    print(f"{len(index)} cases, {sum(r['refused'] for r in index)} refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
