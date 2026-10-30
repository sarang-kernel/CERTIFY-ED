"""C1 fig_certificate_anatomy: what a certificate contains and how its statements depend on each other.

Concept figure, no data. Block names follow schema/certificate-2.1.schema.json (top-level keys and the
statement ids of certificate_v2.STATEMENT_IDS); the dependencies follow docs/theory.md: S1 fixes the
matrix the enclosures refer to (model shift added to every radius), S2 encloses, S3 counts below a shift
and orders the enclosures, S4 and S8 read clusters and brackets off S2 + S3, S6 rechecks S2/S3 in exact
arithmetic from the claim alone, S7 needs the whole-spectrum enclosure (a kind 'all' claim with S2 + S3),
S5 is corroboration by a second solver (evidence, not a certified statement).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_certificate_anatomy"

STATEMENTS = [
    ("S1", "model fidelity", "producer matrix = declared model", "CERTIFIED"),
    ("S2", "eigenpairs", "|λ − μ| ≤ r for each claimed pair", "CERTIFIED"),
    ("S3", "completeness", "exactly k eigenvalues below σ", "CERTIFIED"),
    ("S4", "eigenspaces", "projector, labels, observables", "CERTIFIED"),
    ("S5", "corroboration", "a second solver agrees (evidence)", "CORROBORATED"),
    ("S6", "exact recheck", "S2 and S3 without floating point", "CERTIFIED"),
    ("S7", "thermodynamics", "Z, F, U, C from the whole spectrum", "NOT_CHECKED"),
    ("S8", "degeneracy", "exactly g eigenvalues in a bracket", "INCONCLUSIVE"),
]
COL = dict(C.STATUS_COLORS)
COL["CORROBORATED"] = C.TEAL
LINE = 0.125        # inches between 6-pt lines


def box(ax, x, y, w, h, title, lines, fc="#f6f5f2", ec=C.INK2, lw=0.8):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw))
    ax.text(x + 0.07, y + h - 0.06, title, fontsize=C.SIZES[1], fontweight="bold", va="top", ha="left")
    for i, ln in enumerate(lines):
        ax.text(x + 0.07, y + h - 0.24 - LINE * i, ln, fontsize=C.SIZES[2], va="top", ha="left", color=C.INK2)


def chip(ax, x, y, status, w=0.7, h=0.15):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=COL[status], ec="none"))
    txt = "white" if status in ("CERTIFIED", "FAILED", "CORROBORATED") else C.INK
    ax.text(x + w / 2, y + h / 2, status, fontsize=5.5, ha="center", va="center", color=txt)


def draw():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    C.style()
    W, H = 7.1, 4.0
    WL = 4.0
    fig = plt.figure(figsize=(W, H))
    axL = fig.add_axes([0, 0, WL / W, 1])
    axR = fig.add_axes([WL / W, 0, (W - WL) / W, 1])
    for ax, w in ((axL, WL), (axR, W - WL)):
        ax.set_axis_off()
        ax.set_xlim(0, w)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")

    # ---- (a) the document --------------------------------------------------------------------------
    axL.text(0.08, H - 0.08, "a", fontweight="bold", fontsize=C.SIZES[0] + 1, va="top")
    axL.text(0.3, H - 0.1, "The document and its blocks",
             fontsize=C.SIZES[0], va="top")
    axL.add_patch(FancyBboxPatch((0.1, H - 0.55 - 3.28), WL - 0.25, 3.28 + 0.1, boxstyle="round,pad=0,rounding_size=0.08",
                                 fc="white", ec=C.INK2, lw=1.0))
    x0, w0 = 0.2, 1.52
    top = H - 0.55
    box(axL, x0, top - 0.78, w0, 0.72, "subject + claim",
        ["model spec (hashed), sector", "kind: lowest | window | subset | all",
         "k eigenvalues μ, vectors (hashed)", "accuracy target, producer record"])
    box(axL, x0, top - 1.62, w0, 0.76, "scope",
        ["preset, selection, budgets", "in_scope: what ran, with its status",
         "narrowed: removed by the user", "out_of_scope: reason codes"])
    box(axL, x0, top - 2.46, w0, 0.76, "integrity + environment",
        ["body_sha256 of canonical JSON", "numpy / scipy / BLAS, FP probe",
         "source hash of the code that ran", "arithmetic model: u = 2⁻⁵³"])
    box(axL, x0, top - 3.16, w0, 0.62, "arrays (NPZ sidecar)",
        ["radius, existence_radius", "lower, upper, gap bounds", "sector counts; each hashed"])
    # statements column
    sx, sy, sw, sh = 1.82, top - 3.16, 1.95, 3.1
    axL.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=0.05",
                                 fc="#f6f5f2", ec=C.INK2, lw=0.8))
    axL.text(sx + 0.07, sy + sh - 0.06, "statements (one record each)", fontsize=C.SIZES[1],
             fontweight="bold", va="top")
    row_h = 0.345
    for i, (sid, name, what, status) in enumerate(STATEMENTS):
        yt = sy + sh - 0.3 - row_h * i          # top of the row
        axL.text(sx + 0.07, yt, f"{sid}  {name}", fontsize=C.SIZES[1], fontweight="bold", va="top")
        axL.text(sx + 0.07, yt - 0.16, what, fontsize=C.SIZES[2], va="top", color=C.INK2)
        chip(axL, sx + sw - 0.77, yt - 0.14, status)

    # ---- (b) dependency graph -----------------------------------------------------------------------
    WR = W - WL
    axR.text(0.1, H - 0.08, "b", fontweight="bold", fontsize=C.SIZES[0] + 1, va="top")
    axR.text(0.32, H - 0.1, "What each statement rests on", fontsize=C.SIZES[0], va="top")
    xs = (0.55, 1.55, 2.55)
    nodes = {  # id: (x, y, label, sublabel)
        "S1": (xs[0], 3.05, "S1", "model = spec\n(shift σ)"),
        "S2": (xs[1], 3.05, "S2", "enclosures\nThm 1–3"),
        "S3": (xs[2], 3.05, "S3", "count below σ\nLemma 4, Cor 5"),
        "S6": (xs[0], 1.85, "S6", "exact recheck\nrationals, no FP"),
        "S4": (xs[1], 1.85, "S4", "eigenspaces\nlabels, ⟨O⟩"),
        "S8": (xs[2], 1.85, "S8", "degeneracy\ncount in bracket"),
        "S5": (xs[0], 0.62, "S5", "corroboration\nsecond solver"),
        "WS": (xs[1], 0.62, "", "whole-spectrum\nenclosure\n(kind all)"),
        "S7": (xs[2], 0.62, "S7", "thermodynamics\ninterval sums"),
    }
    r = 0.24
    for nid, (x, y, lab, sub) in nodes.items():
        if nid == "WS":
            axR.add_patch(FancyBboxPatch((x - 0.42, y - 0.24), 0.84, 0.48, boxstyle="round,pad=0,rounding_size=0.05",
                                         fc="white", ec=C.INK2, lw=0.8, ls="--"))
            axR.text(x, y, sub, fontsize=C.SIZES[2], ha="center", va="center", color=C.INK2, linespacing=1.1)
            continue
        fc = C.PURPLE if nid == "S6" else (C.TEAL if nid == "S5" else C.BLUE)
        axR.add_patch(plt.Circle((x, y), r, fc=fc, ec="none"))
        axR.text(x, y, lab, fontsize=C.SIZES[0], ha="center", va="center", color="white", fontweight="bold")
        axR.text(x, y - r - 0.04, sub, fontsize=C.SIZES[2], ha="center", va="top", color=C.INK2, linespacing=1.1)

    def arrow(p, q, rad=0.0):
        axR.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, shrinkA=0, shrinkB=0,
                                      color=C.INK2, lw=0.9, connectionstyle=f"arc3,rad={rad}"))

    # horizontal chain S1 -> S2 -> S3
    for a, b in (("S1", "S2"), ("S2", "S3")):
        xa, ya = nodes[a][:2]
        xb, yb = nodes[b][:2]
        arrow((xa + r, ya), (xb - r, yb))
    axR.text((xs[0] + xs[1]) / 2, 3.05 + 0.05, "σ added\nto radii", fontsize=C.SIZES[2], ha="center", va="bottom",
             color=C.INK2, linespacing=1.1)
    axR.text((xs[1] + xs[2]) / 2, 3.05 + 0.05, "orders the\nenclosures", fontsize=C.SIZES[2], ha="center",
             va="bottom", color=C.INK2, linespacing=1.1)
    # S2, S3 -> S4 ; S3 -> S8 (start below the sublabels of the top row)
    ytop = 3.05 - r - 0.3
    arrow((xs[1], ytop), (xs[1], 1.85 + r))
    arrow((xs[2] - 0.08, ytop), (xs[1] + 0.12, 1.85 + r + 0.05), rad=0.15)
    arrow((xs[2] + 0.06, ytop), (xs[2] + 0.06, 1.85 + r))
    # whole-spectrum enclosure -> S7 ; the enclosure is S2 + S3 of every eigenvalue
    arrow((xs[1] + 0.42, 0.62), (xs[2] - r, 0.62))
    axR.text(xs[1] + 0.06, 1.85 - r - 0.3, "S2 + S3 of\nevery eigenvalue", fontsize=C.SIZES[2], ha="left",
             va="top", color=C.INK2, linespacing=1.1)
    arrow((xs[1], 1.85 - r - 0.3), (xs[1], 0.62 + 0.24))
    # S6 and S5 are independent of the floating-point chain
    axR.text(xs[0], 1.85 - r - 0.32, "from the claim\nand the spec alone", fontsize=C.SIZES[2], ha="center",
             va="top", color=C.INK2, linespacing=1.1)
    axR.add_patch(FancyBboxPatch((0.18, 3.05 - r - 0.34), WR - 0.36, 0.95, boxstyle="round,pad=0,rounding_size=0.06",
                                 fc="none", ec=C.GRID, lw=1.0))
    axR.text(WR / 2, 3.05 + r + 0.08, "floating-point chain (Thm 1 model, outward rounding)",
             fontsize=C.SIZES[2], ha="center", va="bottom", color=C.INK2)
    return fig


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    fig = draw()
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, {"data": "concept figure, no data"})
    for f in findings:
        print("layout:", f)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
