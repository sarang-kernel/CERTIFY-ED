"""E5 fig_corpus_detection: planted-error class x statement -> outcomes of certify-ed on the corpus.

Reads corpus/results/certify-ed/report.json (schema certify-ed/corpus-report/1.0). Rows are the corpus's
error classes (controls included); each cell shows, for the cases of that class, how many certificates
gave each status to the statement (stacked squares: CERTIFIED, INCONCLUSIVE, NOT_CHECKED, FAILED); a
hatched cell means the checker refused the input (exit code 1, no certificate written). A thick frame marks
the statements the corpus expects not to be certified for that class (`must_not_certify`). The right
column is the corpus verdict per case: detected, refused (counts as detected by refusal), control.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figcommon as C  # noqa: E402

NAME = "fig_corpus_detection"
SIDS = ["S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces", "S8.degeneracy"]
VERDICT_COLORS = {"detected": C.TEAL, "refused": C.PURPLE, "control": "white"}


def draw(report):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    C.style()
    rows = report["rows"]
    classes = sorted({r["error_class"] for r in rows}, key=lambda c: (c != "correct", c))
    n = len(classes)
    fig, ax = plt.subplots(figsize=(7.1, 0.27 * n + 1.5))
    ax.set_axis_off()
    ax.set_xlim(0, 7.1)
    ax.set_ylim(-1.4, n + 0.75)
    x_lab, xc, cw, ch = 0.02, [2.1, 2.9, 3.7, 4.5, 5.3], 0.72, 0.8
    x_verdict = 6.2
    for x, sid in zip(xc, SIDS):
        ax.text(x + cw / 2, n + 0.08, sid.replace(".", "\n"), ha="center", va="bottom", fontsize=C.SIZES[2],
                linespacing=1.1)
    ax.text(x_verdict + 0.4, n + 0.08, "corpus\nverdict", ha="center", va="bottom", fontsize=C.SIZES[2], linespacing=1.1)
    ax.text(x_lab, n + 0.08, "planted error class  (cases)", va="bottom", fontsize=C.SIZES[1], fontweight="bold")
    for i, cls in enumerate(classes):
        y = n - 1 - i
        cases = [r for r in rows if r["error_class"] == cls]
        ax.text(x_lab, y + ch / 2, f"{cls.replace('_', ' ')}  ({len(cases)})", va="center", fontsize=C.SIZES[2],
                color=C.INK if cls != "correct" else C.INK2)
        mnc = set()
        for r in cases:
            mnc.update(r["must_not_certify"])
        for x, sid in zip(xc, SIDS):
            counts = {}
            refused = 0
            for r in cases:
                if r["statuses"] is None:
                    refused += 1
                else:
                    st = r["statuses"].get(sid, "not run")
                    counts[st] = counts.get(st, 0) + 1
            total = len(cases)
            left = x
            for st in ("CERTIFIED", "INCONCLUSIVE", "NOT_CHECKED", "FAILED"):
                k = counts.get(st, 0)
                if k:
                    w = cw * k / total
                    ax.add_patch(Rectangle((left, y), w, ch, fc=C.STATUS_COLORS[st], ec="white", lw=0.5))
                    if w > 0.17:
                        ax.text(left + w / 2, y + ch / 2, str(k), ha="center", va="center", fontsize=C.SIZES[2] - 0.5,
                                color="white" if st in ("CERTIFIED", "FAILED") else C.INK)
                    left += w
            if refused:
                w = cw * refused / total
                ax.add_patch(Rectangle((left, y), w, ch, fc="white", ec=C.INK2, lw=0.5, hatch="////"))
            if sid in mnc:
                ax.add_patch(Rectangle((x, y), cw, ch, fc="none", ec=C.INK, lw=1.4))
        vc = {}
        for r in cases:
            vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
        left = x_verdict
        for v in ("detected", "refused", "control"):
            k = vc.get(v, 0)
            if k:
                w = 0.8 * k / len(cases)
                ax.add_patch(Rectangle((left, y), w, ch, fc=VERDICT_COLORS[v], ec=C.INK2 if v == "control" else "white",
                                       lw=0.5))
                ax.text(left + w / 2, y + ch / 2, str(k), ha="center", va="center", fontsize=C.SIZES[2] - 0.5,
                        color="white" if v != "control" else C.INK)
                left += w
    # legend
    lx = 0.02
    for fc, lab, dx, hatch in ((C.STATUS_COLORS["CERTIFIED"], "CERTIFIED", 0.9, None),
                               (C.STATUS_COLORS["INCONCLUSIVE"], "INCONCLUSIVE", 1.2, None),
                               (C.STATUS_COLORS["NOT_CHECKED"], "NOT_CHECKED", 1.2, None),
                               (C.STATUS_COLORS["FAILED"], "FAILED", 0.8, None),
                               ("white", "refused: no certificate written (exit code 1)", 0.0, "////")):
        ax.add_patch(Rectangle((lx, -0.45), 0.18, 0.3, fc=fc, ec=C.INK2 if hatch else "none", lw=0.5, hatch=hatch))
        ax.text(lx + 0.23, -0.3, lab, fontsize=C.SIZES[2], va="center", color=C.INK2)
        lx += dx
    lx = 0.02
    for fc, lab, dx in ((VERDICT_COLORS["detected"], "verdict: detected", 1.3), (VERDICT_COLORS["refused"], "refused", 0.8),
                        (VERDICT_COLORS["control"], "control (no statement is expected to fail)", 0)):
        ax.add_patch(Rectangle((lx, -0.88), 0.18, 0.3, fc=fc, ec=C.INK2, lw=0.5))
        ax.text(lx + 0.23, -0.73, lab, fontsize=C.SIZES[2], va="center", color=C.INK2)
        lx += dx
    ax.add_patch(Rectangle((0.02, -1.31), 0.18, 0.3, fc="none", ec=C.INK, lw=1.4))
    ax.text(0.25, -1.16, "thick frame: the corpus expects this statement not to be certified for the class "
            "(must_not_certify)", fontsize=C.SIZES[2], va="center", color=C.INK2)
    s = report["summary"]
    numbers = {k: s[k] for k in ("planted", "controls", "controls_clean", "detected", "detected_by_refusal", "missed",
                                 "unsound", "silent", "error", "false_alarms", "weak_alarms", "conditional")}
    numbers["classes"] = len(classes)
    numbers["line"] = s["line"]
    numbers["started_utc"] = report["started_utc"]
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    path = C.require(os.path.join(C.ROOT, "corpus", "results", "certify-ed", "report.json"))
    report = json.load(open(path, encoding="utf-8"))
    fig, numbers = draw(report)
    numbers["report"] = "corpus/results/certify-ed/report.json"
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, numbers)
    for f in findings:
        print("layout:", f)
    print(numbers)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
