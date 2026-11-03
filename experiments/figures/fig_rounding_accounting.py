"""C4 fig_rounding_accounting: the floating-point accounting of the Theorem 1 radius as a flow.

Concept figure, no data. The six steps and their factors are those of docs/theory.md section 2
("Floating-point evaluation of the Theorem 1 radius", code-facing list, enclosure.residuals): residual
vectors; bound on |A||x| with (1 + 4u); entrywise rounding bound E with (1 + 4u); norms with the guards
(1 ± γ_{n+2}) and (1 + 8u); the radius with (1 + 4u); and the model shift σ from S1, added with add_up
(one ulp outward) where the radius enters S3, S4 and the exported arrays — added, not multiplied.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_rounding_accounting"

STEPS = [  # (title, formula lines, guard badge, "what it bounds")
    ("1  residual vectors", ["R = fl(A x − fl(μ x))", "lines 355–358"], None,
     "|Rᵢ − (Ax − μx)ᵢ| ≤ γ_{w+2}((|A||x|)ᵢ + |μ xᵢ|) + η-terms (INT-05)"),
    ("2  bound on |A||x|", ["absHX_up = fl(|A||x|) / (1 − γ_{w+1}) · (1 + 4u)"], "(1 + 4u)",
     "absHX_upᵢ ≥ (|A||x|)ᵢ   (complex data: γ_{w+3})"),
    ("3  entrywise bound E", ["E = g · (absHX_up + |x||μ|) · (1 + 4u),  g = γ̇_{w+2}"], "(1 + 4u)",
     "Eᵢ ≥ |Rᵢ − (Ax − μx)ᵢ| − (underflow part)"),
    ("4  norms", ["upper = fl(‖R‖↑ + ‖E‖↑) + T_N,  xl = ‖x‖↓"], "(1±γ_{n+2})(1+8u) ± η",
     "‖·‖↑ = s(1 + γ_{n+2})(1 + 8u) + η_N ≥ ‖·‖\n‖·‖↓ = s(1 − γ_{n+2})/(1 + 8u) − η_N ≤ ‖·‖ (0 if Σx² overflows)"),
    ("5  radius", ["rel = upper / xl · (1 + 4u) + 2 ulp_min"], "(1 + 4u) + 2 ulp_min",
     "rel ≥ ‖Ax − μx‖ / ‖x‖:\nthe Theorem 1 radius for the stored numbers"),
    ("6  model shift", ["radius = add_up(rel, σ): fl(rel + σ) one ulp towards +∞"], "σ added, not ×",
     "the radius refers to the declared model:\nS3 enclosures, S4 separations, exported arrays"),
]


def draw():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    C.style()
    W, H = 7.1, 4.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.text(0.12, H - 0.1, "Theorem 1 radius in binary64 (3.2): three factors (1 + 4u), two norm guards (1 + 8u), additive underflow terms η, "
            "one shift added at the end", fontsize=C.SIZES[0], va="top")
    ax.text(0.12, H - 0.3, "u = 2⁻⁵³,  γₘ = mu/(1 − mu),  w = nonzeros per row of A,  n = dimension.  Every line "
            "rounds to nearest; the factor named on it covers those roundings; η-terms (INT-05, docs/proofs/int05_underflow.md) cover subnormal results.", fontsize=C.SIZES[2], va="top",
            color=C.INK2)
    ax.text(0.12, H - 0.62, "computed in enclosure.residuals (and certify.py for step 6)", fontsize=C.SIZES[1],
            va="top", fontweight="bold")
    ax.text(4.85, H - 0.62, "what the stored number then bounds", fontsize=C.SIZES[1], va="top", fontweight="bold")
    # inputs line
    ax.text(0.12, H - 0.82, "inputs: stored A, x, μ (the claim as written), w, n", fontsize=C.SIZES[2], va="top",
            color=C.INK2)
    bw, bh, gap = 3.6, 0.4, 0.1
    x0 = 0.12
    y = H - 1.0 - bh
    for i, (title, lines, badge, bounds) in enumerate(STEPS):
        fc = "#f6f5f2" if i < 5 else "#eef3fb"
        ax.add_patch(FancyBboxPatch((x0, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=C.INK2,
                                    lw=0.8))
        ax.text(x0 + 0.06, y + bh / 2, title, fontsize=C.SIZES[1], fontweight="bold", va="center")
        ax.text(x0 + 1.25, y + bh / 2, "   ".join(lines), fontsize=5.8, va="center", color=C.INK)
        if badge:
            col = C.ORANGE if "4u" in badge else (C.PURPLE if "8u" in badge else C.BLUE)
            ax.add_patch(FancyBboxPatch((x0 + bw + 0.08, y + 0.1), 0.92, 0.2,
                                        boxstyle="round,pad=0,rounding_size=0.04", fc=col, ec="none"))
            ax.text(x0 + bw + 0.08 + 0.46, y + 0.2, badge, fontsize=5.2 if "8u" in badge else 5.6, ha="center", va="center", color="white")
        ax.text(4.85, y + bh / 2, bounds, fontsize=C.SIZES[2], va="center", color=C.INK2, linespacing=1.15)
        if i < len(STEPS) - 1:
            ax.add_patch(FancyArrowPatch((x0 + 0.5, y), (x0 + 0.5, y - gap), arrowstyle="-|>", mutation_scale=7,
                                         color=C.INK2, lw=0.8, shrinkA=0, shrinkB=0))
        y -= bh + gap
    # the S1 shift enters step 6 from the right side of the chain
    y6 = y + bh + gap
    ax.add_patch(FancyBboxPatch((x0 + 1.25, y6 - 0.38), 2.1, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#eef3fb", ec=C.BLUE, lw=0.8))
    ax.text(x0 + 1.25 + 1.05, y6 - 0.25, "from S1: ‖H_used − H_spec‖₂ ≤ σ  (0 without S1)", fontsize=5.6,
            ha="center", va="center", color=C.INK)
    ax.add_patch(FancyArrowPatch((x0 + 2.2, y6 - 0.12), (x0 + 2.2, y6), arrowstyle="-|>", mutation_scale=7,
                                 color=C.BLUE, lw=0.8, shrinkA=0, shrinkB=0))
    # footer
    yf = y6 - 0.5
    ax.text(0.12, yf, "Slack where a line has more roundings than its factor covers: γ_{w+1} over γ_w (step 2), "
            "γ_{w+2} over the entrywise error (steps 1, 3), γ_{n+2} over γₙ/2 + u (4).",
            fontsize=C.SIZES[2], va="top", color=C.INK2)
    ax.text(0.12, yf - 0.16, "tests/test_theorem1_accounting.py pins each stage on a 3 × 3 case in exact rational "
            "arithmetic. Endpoints μ ∓ radius: sub_down / add_up (one ulp outward).",
            fontsize=C.SIZES[2], va="top", color=C.INK2)
    lx = 0.12
    for col, lab, dx in ((C.ORANGE, "(1 + 4u): covers ≤ 3 roundings on its line", 2.25),
                         (C.PURPLE, "norm guards (1 ± γ_{n+2}), (1 + 8u) = fp._GUARD", 2.45),
                         (C.BLUE, "shift σ: added, not multiplied; no (1 + 8u) on it", 0)):
        ax.add_patch(FancyBboxPatch((lx, yf - 0.45), 0.16, 0.1, boxstyle="round,pad=0,rounding_size=0.02", fc=col,
                                    ec="none"))
        ax.text(lx + 0.2, yf - 0.4, lab, fontsize=C.SIZES[2], va="center", color=C.INK2)
        lx += dx
    return fig


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    fig = draw()
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, {"data": "concept figure, no data (docs/theory.md section 2)"})
    for f in findings:
        print("layout:", f)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
