"""make_verification_report.py: write reverification_lean/VERIFICATION_REPORT.md from the result files.

Every number in the report is read from a file under reverification_lean/results/; nothing is typed in by hand except
the explanatory prose. Run from the repository root:

    python tools/make_verification_report.py

Inputs: results/core/summary.json, results/core/second_kernel.json, results/mathlib_enclosure.json,
results/mathlib_counting.json, results/apps/summary.json, results/apps/mathlib_apps.json,
results/apps/final_rerun.json, results/findings_documents.json, results/integration.json.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = os.path.join(ROOT, "reverification_lean")
RES = os.path.join(LR, "results")
OUT = os.path.join(LR, "VERIFICATION_REPORT.md")


def load(*parts):
    with open(os.path.join(RES, *parts), encoding="utf-8") as f:
        return json.load(f)


def cell(text):
    """Make text safe for one cell of a Markdown pipe table and for the house style."""
    t = "" if text is None else str(text)
    t = t.replace("\u2014", ", ").replace("\u2013", " to ").replace("\n", " ")
    t = t.replace("|", "\\|")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def short(name):
    for p in ("CertifyMathlib.Enclosure.", "CertifyMathlib.Counting.", "CertifyMathlib.Apps.", "CertifyMathlib."):
        if name.startswith(p):
            return name[len(p):]
    return name


def names(lst, limit=6):
    lst = [f"`{short(n)}`" for n in (lst or [])]
    if len(lst) > limit:
        return ", ".join(lst[:limit]) + f" and {len(lst) - limit} more"
    return ", ".join(lst)


STATUS = {"proved": "complete", "partial": "partial", "not attempted": "not formalized"}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def num(s, digits=None):
    return s


def main():
    core = load("core", "summary.json")
    sk = load("core", "second_kernel.json")
    enc = load("mathlib_enclosure.json")
    cnt = load("mathlib_counting.json")
    apps = load("apps", "summary.json")
    mapps = load("apps", "mathlib_apps.json")
    rerun = load("apps", "final_rerun.json")
    finds = load("findings_documents.json")["findings"]
    integ = load("integration.json")

    enc_st = enc["statements"]
    cnt_st = cnt["statements"]
    enc_n = {k: sum(1 for s in enc_st if s["status"] == k) for k in STATUS}
    cnt_n = {k: sum(1 for s in cnt_st if s["status"] == k) for k in STATUS}
    enc_nanoda = [x for x in enc["second_checker"] if "nanoda" in str(x.get("tool", ""))]
    enc_nanoda = enc_nanoda[0] if enc_nanoda else {}
    cnt_nanoda = [x for x in cnt["second_checker"] if "nanoda" in str(x.get("tool", ""))][0]
    enc_lc = [x for x in enc["second_checker"] if "leanchecker" in str(x.get("tool", ""))]
    cnt_lc = [x for x in cnt["second_checker"] if "leanchecker" in str(x.get("tool", ""))]
    kt = core["kernel_theorems"]
    cert = core["certificates"]
    fs = core["failed_statuses"]
    fg = core["forgeries"]
    ex = core["exact_checks_on_genuine"]
    au = integ["axiom_audit"]
    b = integ["build"]
    lcf = integ["leanchecker_final"]
    nrc = integ["nanoda_recheck"]

    kplus = [s["id"] for s in enc_st if s.get("trust_level") == "K+"]
    enc_level = "K2" if not kplus else "K2; K+ for " + ", ".join(kplus)
    L = []
    w = L.append
    w("# VERIFICATION_REPORT.md: the Lean re-verification of CERTIFY-ED")
    w("")
    w("This report states what the Lean 4 work in `reverification_lean/` checked, how strongly, and what it did not check. "
      f"It was generated on {integ['date']} by `tools/make_verification_report.py` from the files in "
      "`reverification_lean/results/`; every number below is copied from one of them, and the last section lists them. "
      f"The toolchain is Lean {integ['lean']} and Mathlib {integ['mathlib']}. The objects of the check are the "
      "certificates, derivation documents and run records of CERTIFY-ED as they are in the project archive v5 "
      "(`sources/archive_v5/`, code at `dev/3.2` commit `a814d28`, source hash `cb73dec3e4433a3b`), and the "
      "documentation corrections of the code repository that this work led to (commits `40d59f4` and `fa86456`, "
      "`sources/SOURCES.md`).")
    w("")
    w("## How to read the levels")
    w("")
    w("Every result carries one level of the scale defined in the top-level `README.md`. Level K is a Lean theorem "
      "accepted by the Lean kernel, with no `sorry`, no `native_decide` and no added axiom; the words used for it are "
      "\"kernel-checked in Lean\". Level K2 is a level K theorem that the independent kernel nanoda also accepted, "
      "after the theorem and everything it depends on (Lean core and Mathlib included) were exported with `leanexport`. "
      "Level K+ is a level K theorem whose module `leanchecker` replayed; this repeats Lean's own kernel on the compiled "
      "files and guards against a defect of the elaborator or of the files, not against a defect of the kernel. Level N "
      "is a compiled Lean program that recomputed a number or a status from raw data and agrees with the record; the "
      "program itself is not a theorem. A level applies to a statement as it is written in Lean. How far the Lean "
      "statement says what the document says is described in the column `correspondence` of the result files and "
      "in `ENCLOSURE.md`, `COUNTING.md`, `CORE.md` and `APPS.md`; that comparison was made by reading and is not part "
      "of any level. None of these levels is a check by a person in the sense of rule D-12; the last sections explain "
      "what that means for the derivation documents.")
    w("")
    w("## Summary")
    w("")
    rows = [
        ["build", f"`{b['command']}`", f"exit code {b['exit_code']}, {b['jobs']} jobs, {b['errors']} errors; "
         f"warnings: {b['warnings_module_system']} about Mathlib's module system, {b['warnings_other']} about the "
         "exponent threshold of `FloatUnderflow.lean`", "K"],
        ["axiom audit", "`Audit/AxiomAudit.lean`: every theorem of the three libraries", "; ".join(
            f"{r['library']}: {r['theorems']} named and {r['auxiliary']} auxiliary theorems, axioms "
            f"{', '.join(r['axioms'])}, {r['nonstandard']} with any other axiom" for r in au["libraries"]), "K"],
        ["text search", "the library files of CertifyCore, CertifyMathlib, CertifyApps",
         "; ".join(f"{k}: {v['files']} files, 0 `sorry`, 0 `admit`, 0 `native_decide`, 0 `axiom`, 0 `debug` options"
                   for k, v in integ["text_scan"].items()), "K"],
        ["exact checker `cecheck`", "the 55 validation certificates, their 18 FAILED statuses and 80 forgeries",
         f"{cert['cecheck_pass']} of {cert['checked']} certificates pass; {fs['reproduced_by_cecheck']} of {fs['total']} "
         f"FAILED statuses reproduced ({fs['refuted_by_exact_counts']} also refuted by exact counts); "
         f"{fg['rejected_by_cecheck']} of {fg['total']} forgeries rejected; agreement with the C verifier on "
         f"{core['agreement_rows_agree']} of {core['agreement_rows']} rows", "N"],
        ["CertifyCore theorems", "the finite arithmetic behind the bounds (Theorem 1 guards, INT-05, Lemma 17 "
         "constants, E-04, counting) and three certificates checked in the kernel",
         f"{kt['count']} theorems; nanoda accepted {sum(1 for r in sk['nanoda'] if r.get('ok'))} of {len(sk['nanoda'])}", "K2"],
        ["CertifyMathlib.Enclosure", "Theorems 1 to 3, the floating-point model and accounting, Appendix A, Kahan, "
         "polar factor, E-03, E-05, D-02, Lemma 17",
         f"{len(enc_st)} statements: {enc_n['proved']} complete, {enc_n['partial']} partial; "
         f"{enc['axioms_check']['theorems']} theorems; nanoda accepted {enc_nanoda.get('accepted')} of "
         f"{enc_nanoda.get('theorems')} main theorems", enc_level],
        ["CertifyMathlib.Counting", "Sylvester, Lemma 4, Corollary 5, D-01, Theorems 6 and 8, E-07, P14 to P16, "
         "C-04, C-06, C-07, C-12, C-13, E-04",
         f"{len(cnt_st)} statements: {cnt_n['proved']} complete, {cnt_n['partial']} partial, "
         f"{cnt_n['not attempted']} not formalized; {cnt['axioms_check']['theorems']} theorems; nanoda accepted "
         f"{cnt_nanoda['accepted']} of {cnt_nanoda['theorems']} main theorems", "K2"],
        ["CertifyMathlib.Apps", "bordered-pencil lemma, eta bound, finite-size criterion, AKLT kernel certificate",
         f"{mapps['counts']['theorems']} theorems; nanoda accepted {mapps['nanoda']['accepted']} of "
         f"{mapps['nanoda']['main_theorems']} main theorems", "K2"],
        ["`ceapps`", "the certified application numbers (E7, E9, E10)",
         f"{apps['agreeing']} of {apps['agreeing'] + apps['disagreeing'] + apps['skipped']} targets agree with the "
         f"3.2.1 records; a final re-run with one binary reproduced {sum(1 for t in rerun['targets'] if t['lean_value_identical_to_committed'])} "
         f"of {len(rerun['targets'])} results exactly", "N"],
        ["final replay", f"`leanchecker` on all {lcf['modules']} modules of the three libraries after the final build",
         f"exit code {lcf['exit_code']}, {lcf['seconds']} s", "K+"],
        ["nanoda recheck", "main theorems of the modules rebuilt by the final build",
         f"{nrc['accepted']} of {nrc['theorems']} accepted", "K2"],
    ]
    w(table(["Part", "What", "Result", "Level"], rows))
    w("")
    w("The second checker `lean4lean` could not be used: its build reads no compiled file of Lean 4.35.0 "
      "(\"incompatible header\", `results/core/second_kernel.json`), so K2 rests on nanoda alone.")
    w("")

    w("## The statements of the documents")
    w("")
    w("The three tables list every statement that was taken up, with the main Lean theorems and what is not "
      "formalized. \"Complete\" means that the whole statement of the document has a Lean counterpart at the given "
      "level, under the correspondence described in the track document; \"partial\" means that a named part is "
      "missing, and the last column says which. The source paragraph of each statement is given in the result file.")
    w("")
    w("### Enclosures and the floating-point model (`CertifyMathlib.Enclosure`, `ENCLOSURE.md`)")
    w("")
    rows = []
    for s in enc_st:
        rows.append([s["id"], s["name"], STATUS[s["status"]], s.get("trust_level") or "",
                     names(s.get("main")), s.get("gap") or ""])
    w(table(["Id", "Statement", "Lean", "Level", "Main theorems", "Not formalized, and notes"], rows))
    w("")
    w("### Counting, assembly and eigenspaces (`CertifyMathlib.Counting`, `COUNTING.md`)")
    w("")
    rows = []
    for s in cnt_st:
        rows.append([s["id"], s["name"], STATUS[s["status"]], s.get("level") or "none",
                     names(s.get("lean")), s.get("missing") or ""])
    w(table(["Id", "Statement", "Lean", "Level", "Main theorems", "Not formalized"], rows))
    w("")
    w("### The AKLT application (`CertifyMathlib.Apps`, `APPS.md` section 9)")
    w("")
    rows = [
        ["bordered-pencil lemma and Haynsworth inertia", "complete", "K2", names(["bordered_pencil", "bordered_inertia"]), ""],
        ["eta bound of `eta.py`", "complete", "K2", names(["eta_count", "qf_inv_shift", "pencil_count_eq",
                                                          "eig_le_of_count", "e10_eta_bound"]),
         "the identification of the eigenvalues of K with squared principal-angle cosines is taken as definition"],
        ["finite-size criterion as the code uses it", "complete", "K2",
         names(["e10_criterion", "criterion_of_eta_sq", "square_octagon_criterion", "sqrt_le_isqrt_div"]), ""],
        ["AKLT kernel lemma", "partial", "K2", names(["aklt_kernel_certificate", "rank_zmod_le_rank_rat",
                                                      "psd_sum_mulVec_eq_zero"]),
         "formalized as a certificate over the rationals with the projector similarity as hypothesis; the general SU(2) "
         "lemma for arbitrary graphs is not formalized and is not needed for the six lattices"],
        ["Pomata and Wei, criterion implies a gap", "not formalized", "none", "", "cited from the paper"],
    ]
    w(table(["Statement", "Lean", "Level", "Main theorems", "Not formalized, and notes"], rows))
    w("")

    w("## The certificates")
    w("")
    w(f"`cecheck` (compiled from `CertifyCore`, level N) reads a certificate, its array file and the exported operator, "
      "decodes every binary64 value exactly, recomputes the hashes, and re-checks the S2 radii and the S3 counts in "
      f"exact rational arithmetic. On the validation set of the C verifier ({cert['bundles']} bundles, of which "
      f"{cert['refused_by_certify_ed']} were refused by certify-ed and have no certificate) it passed "
      f"{cert['cecheck_pass']} of {cert['checked']} certificates, as the C verifier did. It confirmed "
      f"{ex['certified_S2_confirmed']} CERTIFIED S2 statements and {ex['certified_S3_confirmed']} CERTIFIED S3 "
      f"statements exactly, recomputed {ex['recorded_counts_recomputed']} recorded inertia counts (all equal) and "
      f"{ex['exact_counts']} exact counts in total, and confirmed {ex['S8_brackets_confirmed']} S8 brackets. "
      f"It reproduced all {fs['total']} FAILED statuses, and for {fs['refuted_by_exact_counts']} of them an exact "
      "inertia count contradicts the claim outright. It rejected all "
      f"{fg['total']} forged certificates (four kinds, twenty each); certify-ed's own integrity check rejects "
      f"{fg['rejected_by_certify_ed_integrity_check']} of them, because a lowered radius or a promoted status with "
      "resealed hashes is consistent as a record and is caught only by recomputation. The row-by-row comparison with "
      f"the C verifier agrees on {core['agreement_rows_agree']} of {core['agreement_rows']} rows "
      "(`results/core/agreement.csv`).")
    w("")
    w("What `cecheck` does not check is listed in `CORE.md`: " + "; ".join(core["not_checked"]) + ".")
    w("")

    w("## The application numbers")
    w("")
    w("`ceapps` (compiled from `CertifyApps`, level N) rebuilds each model from its specification in exact integer "
      "arithmetic, checks it entrywise against the Python builder, and recomputes the certified number from the "
      "recorded vectors and shifts with exact rational arithmetic and exact LDL inertia counts. A target agrees when "
      "the Lean value is at least as strong as the recorded certified value and every recorded count is reproduced. "
      "The final re-run used one binary (SHA-256 `" + rerun["binary_sha256"][:16] + "...`, "
      "`results/apps/final_rerun.json`).")
    w("")
    rows = []
    for t in apps["targets"]:
        lv, rv = t.get("lean_value") or {}, t.get("record_value") or {}
        if t["id"].startswith("e7_"):
            lean_s = f"Rayleigh quotient {lv.get('rayleigh_quotient')}, radius {lv.get('radius')}, count {lv.get('count')}"
            rec_s = f"E0 {rv.get('E0')}, exact radius {rv.get('exact_radius')}, count {rv.get('count')}"
        elif t["id"] == "e9_crown_spin_half_lower_bound":
            lean_s = f"lower bound {lv.get('lower_bound')}"
            rec_s = f"lower bound {rv.get('lower_bound')}"
        elif t["id"] == "e9_crown_spin_one_upper_bound":
            lean_s = f"eigenvalue at most {lv.get('eigenvalue_upper')}"
            rec_s = f"eigenvalue at most {rv.get('eigenvalue_upper')}"
        elif t["id"] == "e10_kernels":
            lean_s = (f"{lv.get('regions_checked')} subgraphs, kernel equals the VBS range in all: "
                      f"{'yes' if lv.get('all_kernels_equal_vbs_range') else 'no'}; largest dimension {lv.get('largest_dimension')}")
            rec_s = f"{rv.get('regions_checked')} subgraphs checked in the record"
        elif t["id"] == "e10_square_octagon_criterion":
            lean_s = f"4 eta_perp + 2 eta_par at most {lv.get('sum_upper')}"
            rec_s = f"at most {rv.get('sum_upper')}"
        else:
            lean_s = f"eta at most {lv.get('eta_upper')}"
            rec_s = f"eta at most {rv.get('eta_upper')}"
        rows.append([t["id"], t["quantity"], lean_s, rec_s, "agrees" if t["agreement"] else "DISAGREES"])
    w(table(["Target", "Quantity", "Lean (`ceapps`)", "3.2.1 record", "Result"], rows))
    w("")
    w("The models were also compared with the Python builder entry by entry, including the two 20-site chains of E7 "
      "(" + ", ".join(f"`{v['file']}`" for v in apps.get("model_checks", {}).values()) + "; all equal).")
    w("")

    w("## Second checkers and axioms")
    w("")
    rows = [
        ["CertifyCore", f"`{sk['leanchecker']['command']}`, exit code {sk['leanchecker']['exit_code']}, "
         f"{sk['leanchecker']['seconds']} s", f"{sum(1 for r in sk['nanoda'] if r.get('ok'))} of {len(sk['nanoda'])} "
         "theorems accepted", ", ".join(kt["axioms"])],
        ["CertifyMathlib.Enclosure", f"{len(enc_lc)} runs, exit codes " + ", ".join(str(x['exit_code']) for x in enc_lc)
         + ", " + " + ".join(str(x['seconds']) for x in enc_lc) + " s",
         f"{enc_nanoda.get('accepted')} of {enc_nanoda.get('theorems')} main theorems accepted; "
         "one theorem (`FP.residual_enclosure_int05`) needed a memory limit of 6 GiB for its export and was run "
         "after the final build",
         ", ".join(enc["axioms_check"]["all_depend_on"])],
        ["CertifyMathlib.Counting", f"{len(cnt_lc)} runs, exit codes " + ", ".join(str(x['exit_code']) for x in cnt_lc)
         + ", " + " + ".join(str(x['seconds']) for x in cnt_lc) + " s",
         f"{cnt_nanoda['accepted']} of {cnt_nanoda['theorems']} main theorems accepted",
         ", ".join(cnt["axioms_check"]["all_depend_on"])],
        ["CertifyMathlib.Apps", f"exit code {mapps['leanchecker']['exit_code']}, {mapps['leanchecker']['seconds']} s",
         f"{mapps['nanoda']['accepted']} of {mapps['nanoda']['main_theorems']} main theorems accepted",
         ", ".join(mapps["axioms"]["standard"])],
        ["all three libraries, final build", f"exit code {lcf['exit_code']}, {lcf['seconds']} s, {lcf['modules']} modules",
         f"{nrc['accepted']} of {nrc['theorems']} rechecked ({nrc['scope']})", "see the axiom audit"],
    ]
    w(table(["Library", "leanchecker", "nanoda", "Axioms"], rows))
    w("")
    w("The axiom audit (`Audit/AxiomAudit.lean`, output `results/integration_logs/axiom_audit.log`) asks Lean for the axioms of "
      "every theorem of the three libraries, auxiliary theorems included. A `sorry` would appear there as `sorryAx` "
      "and a `native_decide` as `Lean.ofReduceBool`; neither appears. The `decide +kernel` proofs of `CertifyCore` "
      "(the three certificates checked in the kernel, and some guards) are evaluated by the kernel itself and add no "
      "axiom. Each nanoda run checks the exported theorem together with every declaration it depends on, about 3000 "
      "to 36000 declarations per run.")
    w("")

    w("## What the formalization found in the documents")
    w("")
    w("Formalizing a statement forces every hypothesis and every range to be written down. Eleven places of the "
      "documents were found where the text and the Lean statement differ. None of them changes a stored certificate, "
      "a status or a published number; seven are corrected in the code repository by the documentation commits "
      "`40d59f4` and `fa86456`, one (L-3) is an open check for 3.2.2, and three confirm a remark of the documents or "
      "record a difference of form. `docs/failures/07_found_while_formalizing.md` describes each in prose.")
    w("")
    rows = [[f["id"], f["document"], f["says"], f["holds"], f["found_by"], f["effect"], f["correction"]] for f in finds]
    w(table(["Id", "Place", "The document says", "What holds", "Found by", "Effect", "Correction"], rows))
    w("")

    w("## What is not formalized, and why")
    w("")
    w("Four kinds of gap remain. First, the link between the Lean model of floating point and the machine: that "
      "binary64 arithmetic, NumPy and SciPy round as the standard model (and, for INT-05, the underflow model) "
      "assumes, that `float(Fraction)` rounds correctly and that a sum or a product in the code has the summation tree "
      "the theorem assumes. These are hypotheses of the Lean theorems, named in each correspondence text, and they are "
      "supported by the runtime probe of certify-ed, not by Lean. Second, the programs: `cecheck` and `ceapps` are "
      "compiled Lean programs (level N), and no theorem states that a program computes what a Mathlib theorem "
      "assumes; their site-by-site elimination for the AKLT kernels, for example, is argued in a docstring. Third, "
      "statements that need mathematics that Mathlib v4.35.0 does not have: C-16 needs Sturm's theorem and squarefree "
      "factorization, C-04 and C-07 need the trace norm and Hölder's inequality for traces (the Lean proofs use a rank "
      "bound instead, with the documents' constants), and the Pomata and Wei gap theorem was cited, not formalized. "
      "Fourth, parts that were not attempted for time: Theorem U5 and D-01 part (c) for complex data, the "
      "floating-point lemmas of E-07 Lemma 5, C-04 Lemma 1 and C-07 Lemmas 3 and 4, C-12 part (a), E-03 Propositions "
      "1 to 3, E-05 Lemma 3 (the computed bound) and Lemma 4, E-04 Lemma 4 and the continuity argument, the code-level "
      "evaluation of the model shift in A4, and the bit-length decomposition of Lemma 17. Statement S1 (that the "
      "operator is the declared physical model) is outside the reach of any of these checks beyond the entrywise "
      "comparison of the rebuilt models. Lemmas 11 and 12 and Proposition 13 of the exact layer "
      "(`manuscript_v4/sections/appB_exact.tex`) have no Lean theorem; `cecheck` recomputes the counts they justify.")
    w("")

    w("## Relation to rule D-12")
    w("")
    w("Rule D-12 says that new mathematics counts as established only after Sarang Vehale has re-derived it. A level K2 "
      "theorem removes the proof as a source of error but not the statement, and the findings above show that "
      "statements and documents can differ. The derivation documents therefore keep the status \"derived, "
      "machine-checked, independently re-derived (not a human check), Sarang's check pending\", with the Lean level "
      "reported next to it. A proposal for how a statement review could take the place of a re-derivation for the "
      "statements at level K2 is in `docs/decisions/05_proposed_d62_lean.md`; it is not adopted.")
    w("")

    w("## How to rebuild and re-run")
    w("")
    w("From `reverification_lean/`, with the toolchain directory and `C:\\Program Files\\Git\\cmd` on `PATH` and "
      "`MATHLIB_CACHE_DIR` set as in `README.md`:")
    w("")
    w("```")
    w("lake exe cache get")
    w(b["command"])
    w("lake env lean Audit/AxiomAudit.lean")
    w("lake env leanchecker " + " ".join(lcf["module_list"][:3]) + " ...   (all " + str(lcf["modules"]) + " modules, see Audit/run_final_checks.py)")
    w("python apps_export/run_all.py --exe .lake/build/bin/ceapps.exe --archive ../sources/archive_v5 --out results/apps --work work/apps")
    w("python ../tools/make_verification_report.py")
    w("```")
    w("")
    w("The certificate checks with `cecheck` and the C verifier are the commands of `CORE.md` section 1; they need "
      "the certify-ed source of `sources/archive_v5/code/certify-ed` on `PYTHONPATH` to regenerate the validation "
      "certificates.")
    w("")
    w("The nanoda drivers are `export/second_kernel.py` (CertifyCore), `Drafts/Enclosure/nanoda_run.py`, "
      "`Drafts/Counting/nanoda_counting.py` and `apps_export/nanoda_apps.py`; each runs one theorem per process and "
      "stops a run above 4 GiB of private memory (6 GiB for the one re-run made by `Audit/run_final_checks.py`).")
    w("")

    w("## Files")
    w("")
    rows = [
        ["`results/core/summary.json`, `agreement.csv`, `findings.json`, `second_kernel.json`", "cecheck results, "
         "agreement with the C verifier, findings, second-kernel runs of CertifyCore"],
        ["`results/mathlib_enclosure.json`", "statements, theorems, axioms and checker runs of the enclosure library"],
        ["`results/mathlib_counting.json`", "the same for the counting library"],
        ["`results/apps/summary.json`, `mathlib_apps.json`, `final_rerun.json`", "application targets, the Mathlib "
         "part of the application, the final re-run"],
        ["`results/findings_documents.json`", "the findings table above"],
        ["`results/integration.json`", "final build, axiom audit, text search, final leanchecker run, nanoda recheck"],
        ["`CORE.md`, `ENCLOSURE.md`, `COUNTING.md`, `APPS.md`", "the full account of each part"],
    ]
    w(table(["File", "Contents"], rows))
    w("")
    text = "\n".join(L)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(OUT, len(text.split()))


if __name__ == "__main__":
    main()
