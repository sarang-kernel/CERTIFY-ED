"""Runner and scorer of the non-Hermitian cases: python corpus/nonhermitian/run_cases.py

For each case: certify_ed.nonhermitian.certify_nonhermitian on the stored arrays, then (i) the expectation of
cases/<name>.json (must_certify, must_not_certify) and (ii) a soundness check of every CERTIFIED N3 and N4 statement
against the exact eigenvalues of the case, in exact rational arithmetic: every eigenvalue lies in a disc, and each group
of m discs holds exactly m eigenvalues. Verdicts: clean (control, every must_certify CERTIFIED), detected (planted, no
must_not_certify CERTIFIED and at least one of them INCONCLUSIVE or NOT_CHECKED), sound (planted, nothing false certified),
unsound (a must_not_certify statement CERTIFIED, or a CERTIFIED conclusion false), alarm (control not clean). The exit
code is 0 when no case is unsound and every control is clean.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from certify_ed.nonhermitian import certify_nonhermitian  # noqa: E402


def _cf(z):
    z = complex(z)
    return (Fraction(z.real), Fraction(z.imag))


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def check_conclusions(rec, mu, exact):
    """None when every CERTIFIED N3/N4 conclusion holds for the exact eigenvalues, else a message."""
    st = {s["id"]: s for s in rec["statements"]}
    n = len(mu)
    if st.get("N3.inclusion", {}).get("status") == "CERTIFIED":
        r2 = Fraction(rec["radius"]["rho"]) ** 2
        for lam in exact:
            if not any(_d2(lam, _cf(mu[i])) <= r2 for i in range(n)):
                return f"N3: eigenvalue {lam} outside every disc"
    if st.get("N4.counting", {}).get("status") == "CERTIFIED":
        r2 = Fraction(rec["radius"]["rho"]) ** 2
        gof = [d["group"] for d in rec["discs"]]
        size, count = {}, {}
        for g in gof:
            size[g] = size.get(g, 0) + 1
            count[g] = 0
        for lam in exact:
            inside = {gof[i] for i in range(n) if _d2(lam, _cf(mu[i])) <= r2}
            if len(inside) != 1:
                return f"N4: eigenvalue {lam} in {len(inside)} groups"
            count[inside.pop()] += 1
        if count != size:
            return f"N4: counts {count} differ from group sizes {size}"
    return None


def run(only=None, verbose=True):
    cases_dir = os.path.join(HERE, "cases")
    index = json.load(open(os.path.join(cases_dir, "index.json"), encoding="utf-8"))
    rows = []
    for name in sorted(index):
        if only and name not in only:
            continue
        exp = json.load(open(os.path.join(cases_dir, name + ".json"), encoding="utf-8"))
        z = np.load(os.path.join(cases_dir, name + ".npz"))
        left = z["left_eigenvectors"] if "left_eigenvectors" in z.files else None
        rec = certify_nonhermitian(z["A"], z["eigenvalues"], z["eigenvectors"], left_eigenvectors=left)
        status = {s["id"]: s["status"] for s in rec["statements"]}
        exact = [(Fraction(a), Fraction(b)) for a, b in exp["exact_eigenvalues"]]
        problem = check_conclusions(rec, z["eigenvalues"], exact)
        bad = [s for s in exp["must_not_certify"] if status.get(s) == "CERTIFIED"]
        if problem or (bad and not exp.get("conditional")):
            verdict = "unsound"
        elif bad:
            verdict = "conditional"
        elif exp["error_class"] == "correct":
            verdict = "clean" if all(status.get(s) == "CERTIFIED" for s in exp["must_certify"]) else "alarm"
        elif exp["must_not_certify"]:
            verdict = "detected"
        else:
            verdict = "sound"
        if exp["error_class"] != "correct" and any(status.get(s) != "CERTIFIED" for s in exp["must_certify"]):
            verdict += " (weak alarm)"
        rows.append({"case": name, "class": exp["error_class"], "statuses": status, "verdict": verdict,
                     "problem": problem})
        if verbose:
            print(f"{name:28s} {verdict:22s} " + " ".join(f"{k.split('.')[0]}={v}" for k, v in status.items()))
    unsound = [r for r in rows if r["verdict"].startswith("unsound")]
    alarms = [r for r in rows if r["verdict"].startswith("alarm")]
    if verbose:
        print(f"{len(rows)} cases: unsound {len(unsound)}, control alarms {len(alarms)}")
    return rows


if __name__ == "__main__":
    rows = run()
    sys.exit(1 if any(r["verdict"].startswith(("unsound", "alarm")) for r in rows) else 0)
