"""Build results/core/agreement.csv: the core-Lean checker, certify-ed (Python) and the C verifier on the 55
validation certificates and the 80 forgeries.

Column meanings and the categories REPRODUCED and REFUTED are in DEFINITIONS below and are copied into
agreement_stats.json.  agree compares the two checkers on the same question (exit code, and reproduction of every
recorded FAILED status); the exact refutation by cecheck is reported in its own column.

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


DEFINITIONS = {
    "REPRODUCED": "the recorded FAILED status is reproduced: the checker finds, on the stored numbers, the reason "
                  "that certify-ed recorded (for S2 a pair whose residual radius plus the model shift exceeds the "
                  "accuracy target, for S3 more eigenvalues below the witness shift, or in the window, than claimed)",
    "REFUTED": "REPRODUCED, and in addition an exact inertia count of the stored operator contradicts the claim "
               "(for S2 no eigenvalue within target plus model shift of the claimed value, for S3 an eigenvalue "
               "among the lowest, or in the window, that no claim covers); only cecheck computes exact counts, so "
               "the C verifier reports REPRODUCED or NOT_REPRODUCED only",
    "failed_reproduced_both": "yes when cecheck and the C verifier both reproduce every recorded FAILED status of "
                              "the certificate (the same question asked of both), no otherwise, empty when there "
                              "is none",
    "lean_exact_refutation": "per FAILED statement, yes when cecheck's exact inertia count contradicts the claim; "
                             "this has no counterpart in the C verifier and does not enter agree",
    "agree": "genuine certificate: both checkers exit 0 and failed_reproduced_both is yes or empty; forgery: both "
             "checkers exit 1",
}


def lean_row(path):
    rep = json.load(open(path, encoding="utf-8"))
    st = {c["id"]: c["status"] for c in rep["checks"]}
    failed = {k: v for k, v in st.items() if k in ("S2.failed", "S2.failed.refuted", "S3.failed")}
    return rep, [c["id"] for c in rep["checks"] if c["status"] == "FAIL"], failed


def lean_failed(lfailed, sid):
    """(reproduced, refuted) for a recorded FAILED statement, from cecheck's checks.

    S2: check S2.failed is REPRODUCED (exact residual radius plus model shift above the target); the refutation is
    the separate check S2.failed.refuted (exact counts at mu -/+ (target + shift)).  S3: certify-ed's reason is a
    count at the witness shift (or the window ends), and cecheck reproduces it with an exact count, so the
    reproduction is itself an exact refutation; cecheck reports it as S3.failed = REFUTED.
    """
    if sid == "S2.eigenpairs":
        rep = lfailed.get("S2.failed") in ("REPRODUCED", "REFUTED")
        return rep, rep and lfailed.get("S2.failed.refuted") == "REFUTED"
    v = lfailed.get("S3.failed")
    return v in ("REPRODUCED", "REFUTED"), v == "REFUTED"


def yn(b):
    return "yes" if b else "no"


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
    stats = {"definitions": DEFINITIONS, "genuine": 0, "genuine_lean_pass": 0, "genuine_c_pass": 0,
             "certificates_with_failed_status": 0, "failed_statuses": 0, "failed_statuses_by_statement": {},
             "failed_lean_reproduced": 0, "failed_c_reproduced": 0, "failed_reproduced_by_both": 0,
             "failed_lean_exact_refutation": 0, "failed_lean_exact_refutation_by_statement": {},
             "failed_lean_category_REFUTED": 0, "failed_lean_category_REPRODUCED": 0,
             "failed_c_category_REPRODUCED": 0,
             "forgeries": 0, "forgeries_lean_rejected": 0, "forgeries_c_rejected": 0,
             "forgeries_python_rejected": 0, "by_mutation": {}}
    for kind, items, sub in (("genuine", gen, "genuine"), ("forgery", mut, "mutants")):
        for r in items:
            case = r["case"]
            rep, lfails, lfailed = lean_row(os.path.join(R, sub, case + ".json"))
            c = cev[case]
            st = r["statuses"]
            failed_sids = [s for s in ("S2.eigenpairs", "S3.completeness") if st.get(s) == "FAILED"] \
                if kind == "genuine" else []
            short = [s.split(".")[0] for s in failed_sids]
            l_rep, c_rep, l_ref, l_cat, c_cat = [], [], [], [], []
            for s, sh in zip(failed_sids, short):
                rp, rf = lean_failed(lfailed, s)
                cv = c["reproduced"].get(sh + ".failed", "NOT_REPRODUCED")
                crp = cv == "REPRODUCED"
                lcat = "REFUTED" if rf else ("REPRODUCED" if rp else "NOT_REPRODUCED")
                l_rep.append(f"{sh}={yn(rp)}")
                c_rep.append(f"{sh}={yn(crp)}")
                l_ref.append(f"{sh}={yn(rf)}")
                l_cat.append(f"{sh}={lcat}")
                c_cat.append(f"{sh}={cv}")
                stats["failed_statuses"] += 1
                bs = stats["failed_statuses_by_statement"]
                bs[sh] = bs.get(sh, 0) + 1
                stats["failed_lean_reproduced"] += rp
                stats["failed_c_reproduced"] += crp
                stats["failed_reproduced_by_both"] += rp and crp
                stats["failed_lean_exact_refutation"] += rf
                br = stats["failed_lean_exact_refutation_by_statement"]
                br[sh] = br.get(sh, 0) + rf
                stats["failed_lean_category_" + lcat] = stats.get("failed_lean_category_" + lcat, 0) + 1
                stats["failed_c_category_" + cv] = stats.get("failed_c_category_" + cv, 0) + 1
            both = "" if not failed_sids else yn(all(x.endswith("=yes") for x in l_rep + c_rep))
            lean_exit = 1 if lfails else 0
            if kind == "genuine":
                stats["genuine"] += 1
                stats["genuine_lean_pass"] += lean_exit == 0
                stats["genuine_c_pass"] += c["exit_code"] == 0
                stats["certificates_with_failed_status"] += bool(failed_sids)
                agree = lean_exit == 0 and c["exit_code"] == 0 and both in ("", "yes")
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
                         "lean_exit": lean_exit, "lean_failing_checks": ";".join(lfails),
                         "failed_statements": ";".join(short),
                         "c_failed_reproduced": ";".join(c_rep), "lean_failed_reproduced": ";".join(l_rep),
                         "failed_reproduced_both": both,
                         "lean_exact_refutation": ";".join(l_ref),
                         "c_failed_category": ";".join(c_cat), "lean_failed_category": ";".join(l_cat),
                         "lean_exact_counts": len(rep["exact_counts"]), "lean_ms": rep["milliseconds"],
                         "agree": yn(agree)})
    with open(os.path.join(R, "agreement.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    stats["rows"] = len(rows)
    stats["agree"] = sum(r["agree"] == "yes" for r in rows)
    stats["agree_genuine"] = sum(r["agree"] == "yes" for r in rows if r["kind"] == "genuine")
    stats["agree_forgeries"] = sum(r["agree"] == "yes" for r in rows if r["kind"] != "genuine")
    with open(os.path.join(R, "agreement_stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "definitions"}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
