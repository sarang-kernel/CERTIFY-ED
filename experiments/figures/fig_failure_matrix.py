"""E1 fig_failure_matrix: planted error class x check -> outcome (E1 record of the evidence run).

Reads e1_failure_matrix_*/data.json. Columns: P1 (residual check with the producer's matrix), P2
(recomputation from the intended model), P3 (certify-ed's statement outcome) and the statement that P3's
diagnosis names. The record stores one outcome string per case (`P3_certify_ed`) and a diagnosis text
(`P3_detail`); per-statement statuses are stored for the correct result only, so the "statement named"
column is derived from the diagnosis text by the keyword rules in STATEMENT_OF (printed in the cell), not
read from a status field.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figcommon as C  # noqa: E402

NAME = "fig_failure_matrix"

STATEMENT_OF = [  # (substring of P3_detail, statement named)
    ("S1.model_fidelity=CERTIFIED; S2.eigenpairs=CERTIFIED", "all"),
    ("fit", "S1"), ("scale factor", "S1"), ("H-invariant subspaces", "S1"), ("sectors cover", "S1"),
    ("missing from the claim", "S3"), ("Ritz value", "S2"), ("radius", "S2"), ("tr(P O)", "S4"),
]


def statement_named(detail):
    for key, sid in STATEMENT_OF:
        if key in detail:
            return sid
    return "?"


def p3_class(outcome):
    if outcome == "all CERTIFIED" or outcome == "CERTIFIED":
        return "CERTIFIED"
    if "INCONCLUSIVE" in outcome:
        return "INCONCLUSIVE"
    return "FAILED"


def draw(data):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    C.style()
    rows = data["rows"]
    n = len(rows)
    cols = ["P1\nresidual check", "P2\nrecomputation", "P3\ncertify-ed", "statement\nnamed by P3", "P3 localizes\nthe error"]
    fig, ax = plt.subplots(figsize=(7.1, 0.26 * n + 1.1))
    ax.set_xlim(0, 7.1)
    ax.set_ylim(-0.5, n + 0.3)
    ax.set_axis_off()
    x_case, x_mech, xc = 0.02, 0.5, [2.75, 3.6, 4.45, 5.35, 6.25]
    cw = 0.78
    ax.text(x_case, n + 0.05, "case", fontsize=C.SIZES[1], va="bottom", fontweight="bold")
    ax.text(x_mech, n + 0.05, "planted error", fontsize=C.SIZES[1], va="bottom", fontweight="bold")
    for x, c in zip(xc, cols):
        ax.text(x + cw / 2, n + 0.05, c, fontsize=C.SIZES[2], va="bottom", ha="center", linespacing=1.1)
    flag_c, pass_c = C.ORANGE, "#f1f0ec"
    counts = {"P1": 0, "P2": 0, "P3": 0, "localizes": 0, "wrong": 0}
    for i, r in enumerate(rows):
        y = n - 1 - i
        wrong = not r["case"].startswith("F0") and not r["case"].startswith("F12")
        counts["wrong"] += wrong
        ax.text(x_case, y + 0.5, r["case"].split(" ")[0], fontsize=C.SIZES[2], va="center")
        ax.text(x_mech, y + 0.5, r["case"].split(" ", 1)[1], fontsize=C.SIZES[2], va="center", color=C.INK2)
        # P1, P2: True = passes (no flag), False = flags, None = not applicable
        for x, key, cname in ((xc[0], "P1_residual_only", "P1"), (xc[1], "P2_recomputation", "P2")):
            v = r[key]
            if v is None:
                fc, txt, tc = "white", "n/a", C.INK2
            elif v:
                fc, txt, tc = pass_c, "pass", C.INK2
            else:
                fc, txt, tc = flag_c, "flag", "white"
                counts[cname] += 1
            ax.add_patch(Rectangle((x, y + 0.05), cw, 0.9, fc=fc, ec="white", lw=0.8))
            ax.text(x + cw / 2, y + 0.5, txt, fontsize=C.SIZES[2], ha="center", va="center", color=tc)
        cls = p3_class(r["P3_certify_ed"])
        fc = C.STATUS_COLORS[cls]
        tc = "white" if cls in ("CERTIFIED", "FAILED") else C.INK
        if cls != "CERTIFIED":
            counts["P3"] += 1
        ax.add_patch(Rectangle((xc[2], y + 0.05), cw, 0.9, fc=fc, ec="white", lw=0.8))
        ax.text(xc[2] + cw / 2, y + 0.5, cls if cls != "INCONCLUSIVE" else "INCONCL.", fontsize=C.SIZES[2] - 0.5,
                ha="center", va="center", color=tc)
        sid = statement_named(r["P3_detail"])
        ax.text(xc[3] + cw / 2, y + 0.5, sid, fontsize=C.SIZES[2], ha="center", va="center", color=C.INK)
        loc = r["P3_localizes"]
        if loc is True:
            counts["localizes"] += 1
        ax.text(xc[4] + cw / 2, y + 0.5, "—" if loc is None else ("yes" if loc else "no"), fontsize=C.SIZES[2],
                ha="center", va="center", color=C.INK)
    # legend
    lx = 0.02
    for fc, lab, dx in ((pass_c, "pass: no problem reported", 1.85), (flag_c, "flag: a problem reported", 1.8),
                        (C.STATUS_COLORS["FAILED"], "FAILED", 0.9), (C.STATUS_COLORS["INCONCLUSIVE"], "INCONCLUSIVE", 1.25),
                        (C.STATUS_COLORS["CERTIFIED"], "CERTIFIED", 0)):
        ax.add_patch(Rectangle((lx, -0.45), 0.18, 0.3, fc=fc, ec="none"))
        ax.text(lx + 0.23, -0.3, lab, fontsize=C.SIZES[2], va="center", color=C.INK2)
        lx += dx
    s = data["settings"]
    numbers = {"cases": n, "wrong_results": counts["wrong"], "P1_flags": counts["P1"], "P2_flags": counts["P2"],
               "P3_not_certified": counts["P3"], "P3_localizes": counts["localizes"], "N": s["N"], "K": s["K"],
               "residual_tolerance": s["residual_tolerance"], "agreement_tolerance": s["agreement_tolerance"]}
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0], needs=("e1_failure_matrix",))
    data, rec = C.load_record(a.results, "e1_failure_matrix")
    fig, numbers = draw(data)
    numbers["record"] = rec
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, numbers)
    for f in findings:
        print("layout:", f)
    print(numbers)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
