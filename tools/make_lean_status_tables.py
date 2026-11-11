"""make_lean_status_tables.py: insert the Lean status of every result into docs/math/theorems.md and
docs/math/derivation_documents.md.

The status and level of each Lean statement are read from reverification_lean/results/mathlib_enclosure.json and
reverification_lean/results/mathlib_counting.json; the mapping from a result of the documents to the Lean statement
ids is the table MAP below. The text between the markers `<!-- lean-status begin -->` and `<!-- lean-status end -->`
is replaced. Run from the repository root: python tools/make_lean_status_tables.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "reverification_lean", "results")
BEGIN, END = "<!-- lean-status begin -->", "<!-- lean-status end -->"
WORD = {"proved": "complete", "partial": "partial", "not attempted": "not formalized"}

# result of the documents -> (Lean statement ids, findings ids of VERIFICATION_REPORT.md, note)
THEOREMS = [
    ("Theorem 1 (residual bound)", ["T1", "T1-family", "T1-shift"], "", ""),
    ("Theorem 1-fl (floating-point radius)", ["FP-model", "T1-fl", "T1-fl-complex"], "L-1, L-2, L-3",
     "guards of CertifyCore at level K2"),
    ("Theorem 1 under underflow (INT-05)", ["T1-fl-int05"], "L-7", "real data only"),
    ("polar-factor lemma", ["polar"], "", ""),
    ("Theorem 2 (cluster radius)", ["T2", "Kahan"], "", ""),
    ("Theorem 2'", ["T2prime"], "", ""),
    ("Theorem 3 (Kato-Temple)", ["T3"], "", ""),
    ("Lemma 4 (inertia)", ["SYL", "MINMAX", "L4", "L4-2x2", "NORM"], "", ""),
    ("Corollary 5 and sorted pairing", ["COR5", "SORTED"], "", ""),
    ("Theorem 6 (projector)", ["THM6"], "L-11", ""),
    ("Proposition 7 (quantum numbers)", [], "", "not taken up"),
    ("Theorem 8 (sector assembly)", ["BLOCK", "THM8"], "", ""),
    ("Proposition 9 (whole spectrum)", [], "", "not taken up"),
    ("Lemmas 10 to 12, Proposition 13 (exact layer)", [], "", "no theorem; `cecheck` recomputes the counts (level N)"),
    ("Proposition 14 (bracketing count)", ["P14"], "L-5", ""),
    ("Proposition 15 (summed bounds)", ["P15"], "", ""),
    ("Proposition 16 (thermodynamics)", ["P16"], "", ""),
    ("Proposition 16' (C-06)", ["C-06"], "", ""),
    ("Lemma 17 (dyadic exp and ln)", ["L17"], "L-4, L-8", "constants also in CertifyCore at level K2"),
    ("Appendix A (model shift)", ["A1-A2", "A3", "A4", "A5", "Weyl", "Gershgorin"], "L-9", ""),
]
DERIVATIONS = [
    ("`thm1_fp-accounting.md`", ["FP-model", "T1-fl", "T1-fl-complex"], "L-1, L-2"),
    ("`int05_underflow.md`", ["T1-fl-int05"], "L-7"),
    ("`appA_model-shift.md`", ["A1-A2", "A3", "A4", "A5"], "L-9"),
    ("`L17_dyadic_exp_ln.md`", ["L17"], "L-4, L-8"),
    ("`thm2_polar-factor.md`", ["polar", "T2"], ""),
    ("`thm2prime_eps-star.md`", ["T2prime"], ""),
    ("`cor5_pairing.md`", ["COR5", "SORTED"], ""),
    ("`thm8_rho.md`", ["THM8"], ""),
    ("`P14_degeneracy_bracketing.md`", ["P14"], "L-5"),
    ("`P15_summed_bounds.md`", ["P15"], ""),
    ("`P16_thermo_intervals.md`", ["P16"], ""),
    ("`C06_truncated_thermodynamics.md`", ["C-06"], ""),
    ("`C16_exact_bracket.md`", ["C-16"], ""),
    ("`C-03_entanglement.md`", [], ""),
    ("`C-04_correlation_lift.md`", ["C-04"], ""),
    ("`C-07_moments.md`", ["C-07"], ""),
    ("`E-03_generalized.md`", ["E-03"], ""),
    ("`E-04_parameter_intervals.md`", ["E-04"], ""),
    ("`E-05_variational.md`", ["E-05"], ""),
    ("`E-07_davis_kahan.md`", ["E-07-L1", "E-07-L2", "E-07-P1", "E-07-L3", "C-12", "C-13", "E-07-L5"], "L-11"),
    ("`D-01_sparse-residual.md`", ["D-01"], "L-10"),
    ("`D-02_matrix-free.md`", ["D-02"], "L-6"),
]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def index():
    idx = {}
    for s in load("mathlib_enclosure.json")["statements"]:
        idx[s["id"]] = (s["status"], s.get("trust_level"))
    for s in load("mathlib_counting.json")["statements"]:
        idx[s["id"]] = (s["status"], s.get("level"))
    return idx


def summary(ids, idx):
    if not ids:
        return "not formalized", "none"
    sts = [idx[i] for i in ids]
    words = {WORD[s] for s, _ in sts}
    if words == {"complete"}:
        status = "complete"
    elif words == {"not formalized"}:
        status = "not formalized"
    else:
        status = "partial"
    levels = sorted({lv for _, lv in sts if lv})
    return status, ", ".join(levels) if levels else "none"


def cell(ids, idx):
    return ", ".join(f"{i} ({WORD[idx[i][0]]})" for i in ids) if ids else "none"


def replace(path, block):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if BEGIN in text:
        a = text.index(BEGIN)
        b = text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    idx = index()
    intro = ("The Lean re-verification in `reverification_lean/` (repository certify-ed-project) states the "
             "results below as Lean theorems on Mathlib. The columns give the Lean statement ids of "
             "`reverification_lean/VERIFICATION_REPORT.md`, whether the whole result has a Lean counterpart, its level "
             "(K2: accepted by the Lean kernel and by the independent kernel nanoda; K+: accepted by the Lean kernel "
             "and replayed by leanchecker) and the findings of that report that concern it. A Lean level is not the "
             "check by a person that rule D-12 asks for; the column \"Sarang's check\" above is unchanged.")
    rows = ["| result | Lean statements | Lean | level | findings |", "|---|---|---|---|---|"]
    for name, ids, finds, note in THEOREMS:
        st, lv = summary(ids, idx)
        extra = f"; {note}" if note else ""
        rows.append(f"| {name} | {cell(ids, idx)} | {st}{extra} | {lv} | {finds or 'none'} |")
    block = BEGIN + "\n\n## Lean status of every result\n\n" + intro + "\n\n" + "\n".join(rows) + "\n\n" + END
    replace(os.path.join(ROOT, "docs", "math", "theorems.md"), block)

    rows = ["| derivation document | Lean statements | Lean | level | findings |", "|---|---|---|---|---|"]
    for name, ids, finds in DERIVATIONS:
        st, lv = summary(ids, idx)
        rows.append(f"| {name} | {cell(ids, idx)} | {st} | {lv} | {finds or 'none'} |")
    intro2 = intro.replace("the results below", "the statements of the derivation documents").replace(
        " above is unchanged", " of the summary table is unchanged").replace(
        "the column \"Sarang's check\" of the summary table is unchanged",
        "every document keeps the status \"Sarang's check pending\"")
    block = BEGIN + "\n\n## Lean status of every derivation document\n\n" + intro2 + "\n\n" + "\n".join(rows) + "\n\n" + END
    replace(os.path.join(ROOT, "docs", "math", "derivation_documents.md"), block)
    print("ok")


if __name__ == "__main__":
    main()
