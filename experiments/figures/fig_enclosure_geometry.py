"""C2 fig_enclosure_geometry: the enclosures on the real line.

Concept figure, no data (the positions are illustrative). Three rows on one number line, following
docs/theory.md sections 2 and 3: Theorem 1 residual radii around the claimed values; a cluster whose
Theorem 1 intervals overlap and receives one Theorem 2 / 2' radius ε under which its members are matched
to distinct eigenvalues; the Kato-Temple sharpening (Theorem 3) of an isolated value, whose isolating
interval (α, β) comes from the neighbouring enclosures ordered by S3; and the S3 inertia count at the
shift σ with its backward-error margin β (Lemma 4): #{λ < σ − β} ≤ ν ≤ #{λ ≤ σ + β}, so ν = k with k
distinct enclosures below σ − β gives Corollary 5, λ_{k+1} ≥ σ − β.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_enclosure_geometry"

# illustrative claimed values and Theorem 1 radii (not data)
MU = [1.0, 2.6, 4.05, 4.35, 6.3]
R1 = [0.22, 0.25, 0.28, 0.30, 0.24]
LAM = [1.08, 2.52, 4.12, 4.30, 6.22]   # where the eigenvalues "really" are, for the ticks
SIGMA, BETA = 7.4, 0.25


def interval(ax, lo, hi, y, color, lw=1.6, cap=0.08, ls="-"):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    for x in (lo, hi):
        ax.plot([x, x], [y - cap, y + cap], color=color, lw=lw)


def draw():
    import matplotlib.pyplot as plt
    C.style()
    H = 4.6
    fig, ax = plt.subplots(figsize=(7.1, 4.4))
    ax.set_xlim(0.2, 8.6)
    ax.set_ylim(-0.5, H)
    ax.set_axis_off()

    def span(lo, hi, y, color, alpha, half=0.22):
        ax.axvspan(lo, hi, ymin=(y - half + 0.5) / (H + 0.5), ymax=(y + half + 0.5) / (H + 0.5),
                   color=color, alpha=alpha, lw=0)

    # baseline: the (unknown) eigenvalues and the claimed values
    y0 = 0.0
    ax.annotate("", xy=(8.55, y0), xytext=(0.3, y0), arrowprops=dict(arrowstyle="-|>", color=C.INK, lw=0.9))
    for lam in LAM:
        ax.plot([lam, lam], [y0 - 0.08, y0 + 0.08], color=C.GREY, lw=1.4)
    ax.text(LAM[-1] + 0.15, y0 + 0.08, "dots: claimed μ;  grey ticks: eigenvalues λ (unknown)",
            fontsize=C.SIZES[2], color=C.INK2, va="bottom")
    for i, mu in enumerate(MU):
        ax.plot(mu, y0, marker="o", ms=4, color=C.INK, zorder=4)
        ax.text(mu, y0 - 0.14, f"μ{i + 1}", ha="center", va="top", fontsize=C.SIZES[2], color=C.INK)

    # ---- row 1: Theorem 1 residual radii ---------------------------------------------------------
    y1 = 4.15
    ax.text(0.3, y1 + 0.2, "Theorem 1: an eigenvalue lies within r = ‖Ax − μx‖ / ‖x‖ of each μ (existence only)",
            fontsize=C.SIZES[1], va="bottom", color=C.INK)
    for mu, r in zip(MU, R1):
        interval(ax, mu - r, mu + r, y1, C.BLUE)
        ax.plot(mu, y1, marker="o", ms=3, color=C.INK, zorder=4)
    ax.annotate("", xy=(MU[0] + R1[0], y1 - 0.18), xytext=(MU[0], y1 - 0.18),
                arrowprops=dict(arrowstyle="<->", color=C.BLUE, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(MU[0] + R1[0] / 2, y1 - 0.24, "r₁", ha="center", va="top", fontsize=C.SIZES[2], color=C.BLUE)
    ax.annotate("intervals of μ₃ and μ₄ overlap: a cluster (next row)", xy=(MU[3] - R1[3] + 0.1, y1 - 0.06),
                xytext=(4.9, y1 - 0.3), fontsize=C.SIZES[2], color=C.INK2, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=C.INK2, lw=0.6))

    # ---- row 2: cluster radius -------------------------------------------------------------------
    y2 = 3.1
    ax.text(0.3, y2 + 0.2, "Theorem 2 / 2′: one cluster radius ε under which the members are matched to "
            "distinct eigenvalues", fontsize=C.SIZES[1], va="bottom", color=C.INK)
    eps = 0.45
    span(MU[2] - eps, MU[3] + eps, y2, C.BLUE, 0.12, half=0.18)
    for mu in MU[2:4]:
        interval(ax, mu - eps, mu + eps, y2, C.BLUE, lw=1.6)
        ax.plot(mu, y2, marker="o", ms=3, color=C.INK, zorder=4)
    ax.annotate("", xy=(MU[3] + eps, y2 - 0.18), xytext=(MU[3], y2 - 0.18),
                arrowprops=dict(arrowstyle="<->", color=C.BLUE, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(MU[3] + eps / 2, y2 - 0.24, "ε", ha="center", va="top", fontsize=C.SIZES[2], color=C.BLUE)
    ax.text(MU[3] + eps + 0.15, y2 + 0.02,
            "ε = min(Thm 2, Thm 2′) from ‖R‖, η = ‖XᴴX − I‖ < 1 and spread(Λ)\n"
            "2 distinct eigenvalues inside; members keep their Thm 1 radii",
            ha="left", va="center", fontsize=C.SIZES[2], color=C.INK2, linespacing=1.15)

    # ---- row 3: Kato-Temple ----------------------------------------------------------------------
    y3 = 2.05
    ax.text(0.3, y3 + 0.2, "Theorem 3 (Kato–Temple): an isolated value is sharpened inside its isolating "
            "interval (α, β)", fontsize=C.SIZES[1], va="bottom", color=C.INK)
    alpha, beta = MU[0] + R1[0], MU[2] - eps
    theta, r = MU[1], R1[1]
    kt_lo, kt_hi = theta - r * r / (beta - theta), theta + r * r / (theta - alpha)
    interval(ax, alpha, beta, y3, C.GREY, lw=0.9, cap=0.12, ls="--")
    ax.text(alpha, y3 - 0.14, "α", ha="center", va="top", fontsize=C.SIZES[2], color=C.INK2)
    ax.text(beta, y3 - 0.14, "β", ha="center", va="top", fontsize=C.SIZES[2], color=C.INK2)
    interval(ax, theta - r, theta + r, y3, C.BLUE, lw=1.2)
    interval(ax, kt_lo, kt_hi, y3 + 0.1, C.ORANGE, lw=2.6, cap=0.07)
    ax.plot(theta, y3, marker="o", ms=3, color=C.INK, zorder=5)
    ax.text(beta + 0.25, y3 + 0.02,
            "θ − r²/(β − θ) ≤ λ ≤ θ + r²/(θ − α)  (θ = xᴴAx, ‖x‖ = 1)\n"
            "α, β: ends of the neighbouring enclosures, ordered by S3; (α, β) holds one eigenvalue",
            ha="left", va="center", fontsize=C.SIZES[2], color=C.INK2, linespacing=1.15)
    ax.text(theta, y3 - 0.3, "Kato–Temple interval (orange) inside the Theorem 1 interval (blue)",
            ha="left", va="top", fontsize=C.SIZES[2], color=C.ORANGE)

    # ---- row 4: inertia count at the shift ---------------------------------------------------------
    y4 = 0.95
    ax.text(0.3, y4 + 0.2, "S3 (Lemma 4): the inertia ν at the shift σ, with backward error β, satisfies "
            "#{λ < σ − β} ≤ ν ≤ #{λ ≤ σ + β}", fontsize=C.SIZES[1], va="bottom", color=C.INK)
    span(SIGMA - BETA, SIGMA + BETA, y4, C.ORANGE, 0.25, half=0.2)
    ax.plot([SIGMA, SIGMA], [y4 - 0.22, y4 + 0.22], color=C.INK, lw=1.2)
    ax.text(SIGMA, y4 - 0.26, "σ", ha="center", va="top", fontsize=C.SIZES[1], color=C.INK)
    ax.text(SIGMA, y4 - 0.42, "collar σ ± β", ha="center", va="top", fontsize=C.SIZES[2], color=C.INK2)
    for mu, r in zip(MU, R1):
        if mu not in MU[2:4]:
            interval(ax, mu - r, mu + r, y4, C.BLUE, lw=1.2, cap=0.06)
    for mu in MU[2:4]:
        interval(ax, mu - eps, mu + eps, y4, C.BLUE, lw=1.2, cap=0.06)
    ax.annotate("", xy=(SIGMA - BETA, y4 - 0.14), xytext=(0.35, y4 - 0.14),
                arrowprops=dict(arrowstyle="-|>", color=C.INK2, lw=0.7))
    ax.text(3.7, y4 - 0.2, "ν = k = 5 distinct enclosures below σ − β  ⇒  they enclose λ₁ … λ₅ (Cor 5)",
            ha="center", va="top", fontsize=C.SIZES[2], color=C.INK2)
    ax.text(SIGMA + BETA + 0.08, y4, "λ₆ ≥ σ − β\n(gap lower bound)", ha="left", va="center",
            fontsize=C.SIZES[2], color=C.INK2, linespacing=1.1)
    return fig


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    fig = draw()
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, {"data": "concept figure, no data (illustrative positions)"})
    for f in findings:
        print("layout:", f)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
