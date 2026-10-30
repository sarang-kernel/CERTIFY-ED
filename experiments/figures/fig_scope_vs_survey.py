"""E6 fig_scope_vs_survey: what ED papers report (practice survey) against what a certificate states.

Reads experiments/survey/survey_for_scope.json (quantities reported by the n papers of the analysis set,
with Wilson 95 % intervals; model-coded, see RESULT_practice_survey.md for the hand-check agreement) and
experiments/survey/survey_coded.csv (per-paper quantity lists, for the coverage count). Each bar is one
reported quantity; its colour is the status of the corresponding row of docs/SCOPE_MATRIX.md: a certified
statement (with the statement id), a derived record that carries no status, nothing (not certified), or
out of scope by design (reason code). The mapping is the table STATUS_OF below, written from the scope
matrix on dev/3.2; the "other" category of the survey is not classifiable.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_scope_vs_survey"
CERT, DERIVED, NOTCERT, OUT, NA = ("certified statement", "derived record, no status", "nothing certified",
                                   "out of scope by design", "not classifiable")
STATUS_OF = {  # survey quantity: (status, what the scope matrix says)
    "ground energy": (CERT, "S2 + S3 (+ S6 exact, S1 model)"),
    "excited levels": (CERT, "S2 per pair, S3 in the window"),
    "gaps": (CERT, "S3 gap bound; gaps record"),
    "degeneracies": (CERT, "S8 (Prop 14, UNVERIFIED)"),
    "quantum numbers/labels": (CERT, "S4 labels, S1 sector audit"),
    "correlation functions": (CERT, "S4 extension (Prop 15)"),
    "structure factor": (CERT, "S4 extension (Prop 15)"),
    "order parameter": (CERT, "S4 observables"),
    "thermodynamics (full spectrum)": (CERT, "S7 from a whole spectrum"),
    "level statistics": (DERIVED, "levelstats (C-09), kind all"),
    "finite-size extrapolation": (OUT, "OUT_OF_SCOPE_EXTRAPOLATION; report only"),
    "entanglement entropy/spectrum": (NOTCERT, "NOT_IMPLEMENTED"),
    "fidelity/susceptibility": (NOTCERT, "no row in the scope matrix"),
    "dynamical response/spectral function": (OUT, "OUT_OF_SCOPE_DYNAMICS"),
    "time evolution": (OUT, "OUT_OF_SCOPE_DYNAMICS"),
    "other": (NA, "mixed, not coded to a row"),
}
COLORS = {CERT: C.BLUE, DERIVED: C.TEAL, NOTCERT: C.ORANGE, OUT: "#b8b7b3", NA: "white"}


def coverage(csv_path):
    """Papers of the analysis set whose every coded quantity (other than 'other') is a certified statement."""
    n = full = 0
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("in_analysis_set") != "1":
                continue
            qs = [q.strip() for q in r["quantities"].split(";") if q.strip() and q.strip() != "other"]
            n += 1
            if qs and all(STATUS_OF.get(q, (NA,))[0] == CERT for q in qs):
                full += 1
    return n, full


def draw(survey, cov):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    C.style()
    n = survey["n"]
    q = survey["quantities"]
    order = sorted(q, key=lambda k: -q[k]["fraction"])
    fig, ax = plt.subplots(figsize=(7.1, 0.23 * len(order) + 1.0), gridspec_kw=dict(left=0.3, right=0.72, bottom=0.14, top=0.9))
    for i, k in enumerate(order):
        st, what = STATUS_OF.get(k, (NA, ""))
        frac = 100 * q[k]["fraction"]
        lo, hi = (100 * v for v in q[k]["wilson95"])
        ax.barh(i, frac, color=COLORS[st], edgecolor=C.INK2 if st == NA else "none", lw=0.6, height=0.7,
                hatch="////" if st == NA else None)
        ax.errorbar(frac, i, xerr=[[frac - lo], [hi - frac]], fmt="none", ecolor=C.INK2, elinewidth=0.7, capsize=1.5)
        ax.text(hi + 1.5, i, what, va="center", fontsize=C.SIZES[2], color=C.INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"papers reporting the quantity (%), n = {n}, Wilson 95 % interval")
    ax.set_title("the frequently reported quantities have a statement; dynamics and extrapolation do not")
    handles = [Patch(fc=COLORS[s], ec=C.INK2 if s == NA else "none", hatch="////" if s == NA else None, label=s)
               for s in (CERT, DERIVED, NOTCERT, OUT, NA)]
    ax.legend(handles=handles, loc="lower right", fontsize=C.SIZES[2], handlelength=1.2, title="scope-matrix row status",
              title_fontsize=C.SIZES[2])
    npap, full = cov
    ax.text(0.99, 0.31, f"{full} of {npap} papers report only\nquantities a statement covers", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=C.SIZES[2], color=C.INK2, linespacing=1.15)
    numbers = {"n": n, "coded_papers": npap, "papers_fully_covered": full,
               "quantities": {k: {"k": q[k]["k"], "percent": round(100 * q[k]["fraction"], 1),
                                  "wilson95_percent": [round(100 * v, 1) for v in q[k]["wilson95"]],
                                  "status": STATUS_OF.get(k, (NA, ""))[0]} for k in order},
               "status_counts": {s: sum(1 for k in order if STATUS_OF.get(k, (NA,))[0] == s)
                                 for s in (CERT, DERIVED, NOTCERT, OUT, NA)}}
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    sj = C.require(os.path.join(C.ROOT, "experiments", "survey", "survey_for_scope.json"))
    sc = C.require(os.path.join(C.ROOT, "experiments", "survey", "survey_coded.csv"))
    survey = json.load(open(sj, encoding="utf-8"))
    fig, numbers = draw(survey, coverage(sc))
    numbers["source"] = ["experiments/survey/survey_for_scope.json", "experiments/survey/survey_coded.csv",
                         "docs/SCOPE_MATRIX.md"]
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, numbers)
    for f in findings:
        print("layout:", f)
    print(numbers)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
