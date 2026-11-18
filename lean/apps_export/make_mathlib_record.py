"""Write results/apps/mathlib_apps.json, the record of the Mathlib module CertifyMathlib.Apps.

Inputs (all from lean/): the module files (hashed and searched for forbidden tokens),
results/apps/mathlib_axioms.txt (output of `lake env lean apps_export/PrintAxiomsApps.lean`),
results/apps/nanoda_apps.jsonl (output of apps_export/nanoda_apps.py) and results/apps/leanchecker_apps.json
(written by apps_export/run_leanchecker_apps.py). Trust levels as in README.md: K = kernel-checked by the
build with the standard axioms only; K+ = the module was also replayed by leanchecker; K2 = the theorem, exported
with all its dependencies, was also accepted by nanoda.

usage: python apps_export/make_mathlib_record.py [--build-seconds S]
"""
import argparse
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "apps")
FILES = ["CertifyMathlib/Apps.lean", "CertifyMathlib/Apps/Bordered.lean", "CertifyMathlib/Apps/Kernel.lean",
         "CertifyMathlib/Apps/Criterion.lean"]
STD = ["Classical.choice", "Quot.sound", "propext"]

# main theorems: what each states and which step of the applications it covers
MAIN = {
    "bordered_inertia": ("Haynsworth inertia additivity for the bordered matrix [[X, -q C], [-q C^H, q G]] with G > 0: "
                         "its negative eigenvalues are those of the Schur complement X - q C G^-1 C^H",
                         "E10, the bordered matrix of eta.py"),
    "bordered_pencil": ("the bordered matrix with X = p G_A has as many negative eigenvalues as (p/q) G_A - C G_B^-1 C^H, "
                        "the number of pencil eigenvalues above p/q", "E10, the bordered-pencil lemma"),
    "qf_inv_shift": ("G^-1 <= (1 + beta/mu) (G + beta I)^-1 in the quadratic form for G >= mu > 0, beta >= 0",
                     "E10, the inequality (G_B + beta)^-1 >= G_B^-1 / (1 + beta/mu)"),
    "eta_count": ("a symmetric N' within q beta of the bordered matrix has at least as many negative eigenvalues as there "
                  "are pencil eigenvalues above t' = (1 + beta/mu)(p/q + beta/nu)", "E10, the certified count of eta.py"),
    "pencil_count_eq": ("for G_A = F F^H and K = F^-1 C G_B^-1 C^H F^-H, the negative eigenvalues of t G_A - C G_B^-1 C^H "
                        "are the eigenvalues of K above t", "E10, pencil count as count of squared cosines"),
    "eig_le_of_count": ("if at most u values exceed t' < 1 and at least u equal 1, every value below 1 is at most t'",
                        "E10, unit cosines"),
    "e10_eta_bound": ("composed: residual bound, pivot count u and at least u unit cosines give eta^2 <= t'",
                      "E10, eta bound of each lattice"),
    "e10_criterion": ("composed: with t' < 1/z^2 and z >= 1 every cosine other than 1 is below 1/z",
                      "E10, criterion of star, cross and the decorated lattices"),
    "criterion_of_eta_sq": ("eta^2 <= t' and t' < 1/z^2 give eta < 1/z", "E10, rational criterion test"),
    "sqrt_le_isqrt_div": ("sqrt(num/den) <= (isqrt(num den) + 1)/den", "E10, square-root bound of __init__.py"),
    "square_octagon_criterion": ("eta_perp^2 <= a, eta_par^2 <= b, a <= s_perp^2, b <= s_par^2, 4 s_perp + 2 s_par < 1 "
                                 "give 4 eta_perp + 2 eta_par < 1", "E10, square-octagon criterion"),
    "linearIndependent_int_of_zmod": ("integer vectors independent modulo a prime are independent over Z",
                                      "E10, ranks modulo p"),
    "rank_zmod_le_rank_rat": ("the rank of an integer matrix modulo a prime is at most its rank over Q",
                              "E10, ranks modulo p (unit cosines, kernel check)"),
    "psd_sum_mulVec_eq_zero": ("a vector killed by a sum of positive semidefinite matrices is killed by every term",
                               "E10, kernel of a projector Hamiltonian"),
    "aklt_kernel_certificate": ("T_e = D^-1 P_e D with P_e >= 0, T_e U = 0, rank_p U >= r and nullity_p of the stacked T_e "
                                "<= r give ker (sum T_e) = range U of dimension r over Q", "E10, the kernel equality"),
}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-seconds", type=float, default=None)
    a = ap.parse_args()
    files = [{"path": "lean/" + f, "sha256": sha(os.path.join(ROOT, f))} for f in FILES]
    words = {"sorry": 0, "admit": 0, "native_decide": 0, "axiom": 0}
    for f in FILES:
        text = open(os.path.join(ROOT, f), encoding="utf8").read()
        for w in words:
            words[w] += len(re.findall(r"\b" + w + r"\b", text))
    ax = {}
    for line in open(os.path.join(RES, "mathlib_axioms.txt"), encoding="utf8"):
        m = re.match(r"^'(.+)' depends on axioms: \[(.*)\]", line.strip())
        if m:
            ax[m.group(1)] = sorted(x.strip() for x in m.group(2).split(",") if x.strip())
        m = re.match(r"^'(.+)' does not depend on any axioms", line.strip())
        if m:
            ax[m.group(1)] = []
    nan = {}
    npath = os.path.join(RES, "nanoda_apps.jsonl")
    if os.path.exists(npath):
        for line in open(npath, encoding="utf8"):
            r = json.loads(line)
            nan[r["theorem"]] = {k: r.get(k) for k in ("result", "declarations", "export_seconds", "check_seconds",
                                                       "export_peak_private_mib", "check_peak_private_mib",
                                                       "export_bytes", "reason")}
    lc = None
    lpath = os.path.join(RES, "leanchecker_apps.json")
    if os.path.exists(lpath):
        lc = json.load(open(lpath))
    lc_ok = bool(lc) and lc.get("exit_code") == 0
    theorems = []
    for name, axioms in ax.items():
        short = name.split(".")[-1]
        level = "K" if set(axioms) <= set(STD) else "not K"
        if level == "K" and lc_ok:
            level = "K+"
        if level != "not K" and nan.get(name, {}).get("result") == "accepted":
            level = "K2"
        row = {"name": name, "axioms": axioms, "trust_level": level, "main": short in MAIN}
        if short in MAIN:
            row["statement"], row["covers"] = MAIN[short]
            row["nanoda"] = nan.get(name, {"result": "not run"})
        theorems.append(row)
    rec = {"module": "CertifyMathlib.Apps", "root_import_note": "the root CertifyMathlib.lean does not import it yet; "
           "the lead adds it", "date": "10 November 2026", "lean": "4.35.0", "mathlib": "v4.35.0",
           "files": files, "forbidden_tokens": words,
           "build": {"command": "lake build CertifyMathlib.Apps", "seconds": a.build_seconds},
           "axioms": {"command": "lake env lean apps_export/PrintAxiomsApps.lean", "theorems": len(ax),
                      "all_standard": all(set(v) <= set(STD) for v in ax.values()), "standard": STD},
           "nanoda": {"command": "python apps_export/nanoda_apps.py --theorems apps_export/nanoda_apps_theorems.txt",
                      "accepted": sum(1 for v in nan.values() if v.get("result") == "accepted"),
                      "rejected": sum(1 for v in nan.values() if v.get("result") == "rejected"),
                      "not_run": sum(1 for v in nan.values() if v.get("result") not in ("accepted", "rejected")),
                      "main_theorems": len(MAIN)},
           "leanchecker": lc,
           "theorems": theorems,
           "counts": {"theorems": len(theorems), "main": sum(1 for t in theorems if t["main"]),
                      "K2": sum(1 for t in theorems if t["trust_level"] == "K2"),
                      "K+": sum(1 for t in theorems if t["trust_level"] == "K+"),
                      "K": sum(1 for t in theorems if t["trust_level"] == "K")}}
    with open(os.path.join(RES, "mathlib_apps.json"), "w", encoding="utf8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps(rec["counts"]), rec["axioms"]["all_standard"], words)


if __name__ == "__main__":
    main()
