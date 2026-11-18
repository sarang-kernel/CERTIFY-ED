"""Drafts/NonHermitian/make_results.py

Writes lean/results/mathlib_nonhermitian.json from the logs of run_checks.ps1:
check_axioms.log (one line per theorem from #print axioms), check_textsearch.log, run_checks_summary.txt,
check_build.log and nanoda_results.jsonl. Run from lean/: python Drafts/NonHermitian/make_results.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "Drafts", "NonHermitian")
NS = "CertifyMathlib.NonHermitian."
DATE = "11 November 2026"


def read(name):
    with open(os.path.join(D, name), encoding="utf-8-sig") as f:
        return f.read()


def summary_steps():
    steps = {}
    for line in read("run_checks_summary.txt").splitlines():
        m = re.match(r"(\w+) exit=(-?\d+) seconds=(\d+)", line)
        if m:
            steps[m.group(1)] = dict(exit_code=int(m.group(2)), seconds=int(m.group(3)))
    return steps


# exports whose proof terms cite each theorem (leanexport writes the closure of the cited constants)
CONTAINED_IN = {
    "bauerFike_core": ["bauerFike_matrix"],
    "bauerFike_residual": ["bauerFike_residual_matrix"],
    "similarity_inclusion": ["disc_inclusion_matrix", "disc_inclusion_certified", "bauerFike_matrix_sharp"],
    "neumann_bound": ["disc_inclusion_certified"],
    "neumann_inverse": ["disc_inclusion_certified"],
    "diag_lower": ["bauerFike_matrix", "bauerFike_residual_matrix", "disc_inclusion_matrix", "disc_inclusion_certified"],
    "bauerFike": ["bauerFike_matrix"],
    "toEuclideanLin_diagonal_apply": ["bauerFike_matrix", "disc_inclusion_certified"],
    "toEuclideanLin_mul_apply": ["bauerFike_matrix", "disc_inclusion_certified"],
    "toEuclideanLin_one_apply": ["bauerFike_matrix", "disc_inclusion_matrix"],
    "disc_inclusion_matrix": ["bauerFike_matrix_sharp"],
}

STATEMENTS = [
    dict(statement="Lemma 1, backward error of a right residual", source="docs/proofs/E-02_non_hermitian.md section 1",
         theorems=["backward_error"], status="partial",
         missing="equality of the norm of E (Lean has the upper bound), the singular value form, the left residual form "
                 "and the two by two example"),
    dict(statement="Theorem 2, Bauer-Fike", source="docs/proofs/E-02_non_hermitian.md section 1",
         theorems=["bauerFike_core", "bauerFike", "bauerFike_matrix", "bauerFike_matrix_sharp"], status="proved"),
    dict(statement="Theorem 2, residual form", source="docs/proofs/E-02_non_hermitian.md section 1",
         theorems=["bauerFike_residual", "bauerFike_residual_matrix"], status="proved"),
    dict(statement="Lemma 3, the inverse from a computed inverse", source="docs/proofs/E-02_non_hermitian.md section 1",
         theorems=["neumann_bound", "neumann_inverse"], status="partial",
         missing="invertibility of W and the formula for the inverse of V (the code uses only the bound)"),
    dict(statement="Theorem 4 (a), similarity", source="docs/proofs/E-02_non_hermitian.md section 1", theorems=[],
         status="not attempted", missing="not needed by the Lean proof of (b)"),
    dict(statement="Theorem 4 (b), discs", source="docs/proofs/E-02_non_hermitian.md section 1",
         theorems=["similarity_inclusion", "disc_inclusion_matrix"], status="proved"),
    dict(statement="Theorem 4 (b) with Lemma 3, statements N2 and N3 of the code",
         source="docs/proofs/E-02_non_hermitian.md section 1", theorems=["disc_inclusion_certified"],
         status="proved"),
    dict(statement="Theorem 4 (c) to (e), counting", source="docs/proofs/E-02_non_hermitian.md section 1", theorems=[],
         status="not attempted", missing="continuity of polynomial roots in a real parameter with multiplicity"),
    dict(statement="Propositions 5 to 7, floating-point accounting",
         source="docs/proofs/E-02_non_hermitian.md sections 1 and 2", theorems=[], status="not attempted",
         missing="floating-point model of ENCLOSURE.md not yet applied"),
]


def main():
    steps = summary_steps()
    axioms = {}
    for line in read("check_axioms.log").splitlines():
        m = re.match(r"'([^']+)' depends on axioms: \[(.*)\]", line.strip())
        if m:
            axioms[m.group(1)] = [a.strip() for a in m.group(2).split(",") if a.strip()]
    text = dict(re.findall(r"^(\S+) (\d+)\s*$", read("check_textsearch.log"), flags=re.M))
    runs = []
    if os.path.exists(os.path.join(D, "nanoda_results.jsonl")):
        seen = {}
        for line in read("nanoda_results.jsonl").splitlines():
            if line.strip():
                r = json.loads(line)
                seen[r["theorem"]] = r
        runs = list(seen.values())
    accepted = {r["theorem"][len(NS):] for r in runs if r.get("result") == "accepted"}
    build_log = read("check_build.log")
    module_seconds = {m.group(1).split(".")[-1]: int(m.group(2))
                      for m in re.finditer(r"Built (CertifyMathlib\.NonHermitian(?:\.\w+)?) \((\d+)s\)", build_log)}
    jobs = re.search(r"Build completed successfully \((\d+) jobs\)", build_log)
    theorems = []
    for full, ax in axioms.items():
        short = full[len(NS):]
        lev = "K+" if steps.get("leanchecker", {}).get("exit_code") == 0 else "K"
        nan = None
        if short in accepted:
            lev, nan = "K2", "accepted (own export)"
        else:
            hosts = [h for h in CONTAINED_IN.get(short, []) if h in accepted]
            if hosts:
                lev, nan = "K2", "accepted as a dependency in the export of " + ", ".join(NS + h for h in hosts)
        theorems.append(dict(name=full, axioms=ax, level=lev, nanoda=nan or "not run"))
    level_of = {t["name"][len(NS):]: t["level"] for t in theorems}
    order = {"K": 0, "K+": 1, "K2": 2}
    for s in STATEMENTS:
        s["theorems"] = [NS + t for t in s["theorems"]]
        levs = [level_of[t[len(NS):]] for t in s["theorems"]]
        s["level"] = min(levs, key=lambda x: order[x]) if levs else "none"
    out = {
        "part": "Mathlib formalization: Bauer-Fike and the residual forms of E-02 (non-Hermitian eigenproblems)",
        "document": "docs/lean/NONHERMITIAN.md",
        "root_module": "CertifyMathlib.NonHermitian",
        "root_file": "lean/CertifyMathlib/NonHermitian.lean",
        "modules": ["CertifyMathlib.NonHermitian.BauerFike", "CertifyMathlib.NonHermitian"],
        "toolchain": "leanprover/lean4:v4.35.0",
        "mathlib": "v4.35.0",
        "imports_from_other_parts": [],
        "statement_source": "docs/proofs/E-02_non_hermitian.md of the code repository, commit 167940a on branch "
                            "track/non-hermitian (from dev/3.2 at fa86456); not yet in archive_v5",
        "build": {
            "command": "lake build CertifyMathlib.NonHermitian",
            "exit_code": steps.get("build", {}).get("exit_code"),
            "result": ("Build completed successfully (%s jobs)" % jobs.group(1)) if jobs else "see check_build.log",
            "date": DATE,
            "module_seconds": module_seconds,
            "step_seconds": steps.get("build", {}).get("seconds"),
            "warnings": "two notes that BauerFike.lean imports Mathlib modules designed for the module system; "
                        "no other warning",
            "log": "lean/Drafts/NonHermitian/check_build.log",
        },
        "level_K_check": {
            "files_searched": int(text.get("files", 0)),
            "sorry": int(text.get("sorry", -1)),
            "admit": int(text.get("admit", -1)),
            "native_decide": int(text.get("native_decide", -1)),
            "axiom_declarations": int(text.get("^\\s*axiom\\s", -1)),
            "set_option": int(text.get("set_option", -1)),
            "method": "text search of CertifyMathlib/NonHermitian/BauerFike.lean, CertifyMathlib/NonHermitian.lean "
                      "and Drafts/NonHermitian/PrintAxioms.lean (run_checks.ps1, step textsearch)",
        },
        "axioms_check": {
            "script": "lean/Drafts/NonHermitian/PrintAxioms.lean",
            "command": "lake env lean Drafts/NonHermitian/PrintAxioms.lean",
            "output": "lean/Drafts/NonHermitian/check_axioms.log",
            "theorems": len(axioms),
            "all_depend_on": sorted({a for ax in axioms.values() for a in ax}),
            "sorryAx": sum(1 for ax in axioms.values() if "sorryAx" in ax),
            "seconds": steps.get("axioms", {}).get("seconds"),
            "date": DATE,
        },
        "second_checker": [
            {
                "tool": "leanchecker (toolchain lean-4.35.0-windows)",
                "command": "lake env leanchecker CertifyMathlib.NonHermitian.BauerFike CertifyMathlib.NonHermitian",
                "modules": ["CertifyMathlib.NonHermitian.BauerFike", "CertifyMathlib.NonHermitian"],
                "exit_code": steps.get("leanchecker", {}).get("exit_code"),
                "seconds": steps.get("leanchecker", {}).get("seconds"),
                "output": "none" if not read("check_leanchecker.log").strip() else read("check_leanchecker.log")[-1500:],
                "date": DATE,
                "trust_level": "K+",
            },
            {
                "tool": "nanoda (nanoda_bin of toolchain lean-4.35.0-windows, input from leanexport)",
                "command": "leanexport <module> -- <theorem> > export.ndjson; nanoda_bin config.json (permitted_axioms "
                           "propext, Classical.choice, Quot.sound; unpermitted_axiom_hard_error, nat_extension, "
                           "string_extension true)",
                "driver": "lean/Drafts/Counting/nanoda_counting.py",
                "theorem_list": "lean/Drafts/NonHermitian/nanoda_theorems.txt",
                "procedure": "one theorem per run; the export contains the theorem and every declaration it depends on, "
                             "including the Lean core and Mathlib declarations; a run is stopped when a process "
                             "exceeds 4096 MiB of private memory or 1800 s; the export file is removed after each run",
                "theorems": len(runs),
                "accepted": sum(1 for r in runs if r.get("result") == "accepted"),
                "rejected": sum(1 for r in runs if r.get("result") == "rejected"),
                "not_run": sum(1 for r in runs if r.get("result") == "not run"),
                "seconds": round(sum((r.get("export_seconds") or 0) + (r.get("check_seconds") or 0) for r in runs), 1),
                "date": DATE,
                "runs": runs,
            },
        ],
        "statements": STATEMENTS,
        "theorems": theorems,
    }
    path = os.path.join(ROOT, "results", "mathlib_nonhermitian.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(path, len(theorems), out["second_checker"][1]["accepted"], [s["level"] for s in STATEMENTS])


if __name__ == "__main__":
    main()
