"""Build results/core/agreement.csv: the core-Lean checker, certify-ed (Python) and the C verifier on the 55
validation certificates and the 80 forgeries.

    python export/agreement.py --work work/core --results results/core

Inputs: <work>/certs/index.json and <work>/mutants/index.json (certify-ed's statuses and the forgery list),
<results>/genuine/*.json and <results>/mutants/*.json (cecheck reports), <results>/cev_genuine.json and
<results>/cev_mutants.json (C verifier), <results>/python_integrity_*.json (certify-ed's Certificate.load).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

SIDS = ["S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces", "S8.degeneracy"]


def lean_row(path):
    rep = json.load(open(path, encoding="utf-8"))
    st = {c["id"]: c["status"] for c in rep["checks"]}
    failed = {k: v for k, v in st.items() if k in ("S2.failed", "S2.failed.refuted", "S3.failed")}
    return rep, [c["id"] for c in rep["checks"] if c["status"] == "FAIL"], failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    R = a.results
    cev = {r["case"]: r for r in json.load(open(os.path.join(R, "cev_genuine.json")))}
    cev.update({r["case"]: r for r in json.load(open(os.path.join(R, "cev_mutants.json")))})
    pyi = {r["case"]: r for r in json.load(open(os.path.join(R, "python_integrity_genuine.json")))}
    pyi.update({r["case"]: r for r in json.load(open(os.path.join(R, "python_integrity_mutants.json")))})
    rows = []
    gen = [r for r in json.load(open(os.path.join(a.work, "certs", "index.json"))) if not r["refused"]]
    mut = json.load(open(os.path.join(a.work, "mutants", "index.json")))
    stats = {"genuine": 0, "genuine_lean_pass": 0, "genuine_c_pass": 0, "failed_statuses": 0,
             "failed_lean_reproduced": 0, "failed_lean_refuted": 0, "failed_c_reproduced": 0,
             "forgeries": 0, "forgeries_lean_rejected": 0, "forgeries_c_rejected": 0,
             "forgeries_python_rejected": 0, "by_mutation": {}}
    for kind, items, sub in (("genuine", gen, "genuine"), ("forgery", mut, "mutants")):
        for r in items:
            case = r["case"]
            rep, lfails, lfailed = lean_row(os.path.join(R, sub, case + ".json"))
            c = cev[case]
            st = r["statuses"]
            failed_sids = [s for s in ("S2.eigenpairs", "S3.completeness") if st.get(s) == "FAILED"] if kind == "genuine" else []
            lean_rep = []
            c_rep = []
            for s in failed_sids:
                key = s.split(".")[0] + ".failed"
                lv = lfailed.get(key, "no record")
                if s.startswith("S2") and lfailed.get("S2.failed.refuted") == "REFUTED":
                    lv = "REFUTED"
                lean_rep.append(f"{s.split('.')[0]}={lv}")
                c_rep.append(f"{s.split('.')[0]}={c['reproduced'].get(key, 'no record')}")
                stats["failed_statuses"] += 1
                stats["failed_lean_reproduced"] += lv in ("REPRODUCED", "REFUTED")
                stats["failed_lean_refuted"] += lv == "REFUTED"
                stats["failed_c_reproduced"] += c["reproduced"].get(key) == "REPRODUCED"
            lean_exit = 1 if lfails else 0
            if kind == "genuine":
                stats["genuine"] += 1
                stats["genuine_lean_pass"] += lean_exit == 0
                stats["genuine_c_pass"] += c["exit_code"] == 0
                agree = lean_exit == 0 and c["exit_code"] == 0 and all("REPRODUCED" in x or "REFUTED" in x for x in lean_rep) \
                    and all("=REPRODUCED" in x for x in c_rep)
            else:
                m = r["mutation"]
                stats["forgeries"] += 1
                stats["forgeries_lean_rejected"] += lean_exit == 1
                stats["forgeries_c_rejected"] += c["exit_code"] == 1
                stats["forgeries_python_rejected"] += not pyi[case]["integrity_ok"]
                bm = stats["by_mutation"].setdefault(m, {"total": 0, "lean": 0, "c": 0, "python_integrity": 0})
                bm["total"] += 1
                bm["lean"] += lean_exit == 1
                bm["c"] += c["exit_code"] == 1
                bm["python_integrity"] += not pyi[case]["integrity_ok"]
                agree = lean_exit == 1 and c["exit_code"] == 1
            rows.append({"case": case, "kind": kind if kind == "genuine" else f"forgery:{r['mutation']}",
                         "certify_ed_S1_S2_S3_S4_S8": " ".join((st.get(s) or "-") for s in SIDS),
                         "python_integrity": "accepted" if pyi[case]["integrity_ok"] else "rejected",
                         "c_exit": c["exit_code"], "c_failing_checks": ";".join(c["fails"]),
                         "c_failed_statuses": ";".join(c_rep),
                         "lean_exit": lean_exit, "lean_failing_checks": ";".join(lfails),
                         "lean_failed_statuses": ";".join(lean_rep),
                         "lean_exact_counts": len(rep["exact_counts"]), "lean_ms": rep["milliseconds"],
                         "agree": "yes" if agree else "no"})
    with open(os.path.join(R, "agreement.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    stats["rows"] = len(rows)
    stats["agree"] = sum(r["agree"] == "yes" for r in rows)
    with open(os.path.join(R, "agreement_stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(stats, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
