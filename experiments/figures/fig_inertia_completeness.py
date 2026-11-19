"""C3 fig_inertia_completeness: how inertia counts certify "exactly k eigenvalues below σ" and the gap bound.

Concept figure, no data (illustrative spectra). The counting function N(x) = #{λ < x} is a staircase.
(a) 'lowest' claim: k distinct enclosures below σ − β give N(σ − β) ≥ k from below; the inertia ν of the
LDLᴴ factorization of A − σI with backward error β gives N(σ − β) ≤ ν (Lemma 4); ν = k closes the sandwich:
the enclosures hold λ₁ … λ_k and λ_{k+1} ≥ σ − β (Corollary 5), the certified gap lower bound.
(b) window claim without eigenvectors (C-02): two inertia counts at the edges give ν_b − ν_a = #{λ ∈ (a, b)}
once each edge's collar is shown to be eigenvalue-free by two more counts at e ± δ.
(c) what the count does not certify: ν > k means an eigenvalue below σ is missing from the claim (FAILED);
an eigenvalue inside the collar leaves the count undecided (INCONCLUSIVE).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_inertia_completeness"


def staircase(ax, lams, x0, x1, color=C.INK, lw=1.2):
    xs = [x0] + sorted(lams) + [x1]
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i + 1]], [i, i], color=color, lw=lw, solid_capstyle="butt")
        if i < len(xs) - 2:
            ax.plot([xs[i + 1], xs[i + 1]], [i, i + 1], color=color, lw=0.5, ls=":")
    for lam in lams:
        ax.plot([lam, lam], [-0.35, -0.1], color=C.GREY, lw=1.2)


def collar(ax, sigma, beta, ymax, color=C.ORANGE):
    ax.axvspan(sigma - beta, sigma + beta, 0, 1, color=color, alpha=0.22, lw=0)
    ax.plot([sigma, sigma], [-0.4, ymax], color=C.INK, lw=1.0)


def draw():
    import matplotlib.pyplot as plt
    C.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.1), gridspec_kw=dict(wspace=0.3, left=0.06, right=0.99,
                                                                        bottom=0.12, top=0.9))
    k = 4
    lams = [1.0, 2.1, 2.9, 3.6, 5.4, 6.1]
    mus = [1.05, 2.0, 2.95, 3.5]
    rad = [0.18, 0.2, 0.17, 0.22]
    sigma, beta = 4.6, 0.3

    # ---- (a) lowest: enclosures from below, inertia from above --------------------------------------
    ax = axes[0]
    C.panel_letter(ax, "a", dx=-0.2)
    ax.set_title("ν = k closes the sandwich")
    staircase(ax, lams, 0.2, 7.0)
    collar(ax, sigma, beta, 5.0)
    for mu, r in zip(mus, rad):
        ax.plot([mu - r, mu + r], [-0.75, -0.75], color=C.BLUE, lw=1.8, solid_capstyle="butt")
        ax.plot(mu, -0.75, marker="o", ms=2.5, color=C.INK)
    ax.text(0.3, -1.05, f"{k} distinct enclosures below σ − β ⇒ N(σ − β) ≥ {k}", fontsize=C.SIZES[2],
            color=C.BLUE, va="top")
    ax.plot([0.2, sigma + beta], [k, k], color=C.ORANGE, lw=0.8, ls="--")
    ax.text(0.3, k - 0.15, f"inertia ν = {k} of A − σI ⇒ N(σ − β) ≤ {k}", fontsize=C.SIZES[2],
            color=C.ORANGE, va="top")
    ax.annotate("λ₅ ≥ σ − β\n(gap lower bound)", xy=(sigma - beta, k + 0.05), xytext=(1.2, 5.2), fontsize=C.SIZES[2],
                ha="left", va="bottom", color=C.INK2, linespacing=1.1, arrowprops=dict(arrowstyle="-|>", color=C.INK2, lw=0.6))
    ax.text(sigma, -0.55, "σ", ha="center", va="top", fontsize=C.SIZES[1])
    ax.set_xlim(0.2, 7.0)
    ax.set_ylim(-1.5, 6.6)
    ax.set_ylabel("N(x) = #{λ < x}")
    ax.set_yticks(range(0, 7))
    ax.set_xticks([])

    # ---- (b) window without vectors: two counts ----------------------------------------------------
    ax = axes[1]
    C.panel_letter(ax, "b", dx=-0.2)
    ax.set_title("window: count = ν_b − ν_a")
    a_, b_ = 1.6, 4.1
    ba = 0.18
    staircase(ax, lams, 0.2, 7.0)
    collar(ax, a_, ba, 5.0)
    collar(ax, b_, ba, 5.0)
    na, nb = sum(l < a_ for l in lams), sum(l < b_ for l in lams)
    ax.plot([0.2, a_], [na, na], color=C.ORANGE, lw=0.8, ls="--")
    ax.plot([0.2, b_], [nb, nb], color=C.ORANGE, lw=0.8, ls="--")
    ax.text(0.3, na + 0.12, f"ν_a = {na}", fontsize=C.SIZES[2], color=C.ORANGE, va="bottom")
    ax.text(0.3, nb + 0.12, f"ν_b = {nb}", fontsize=C.SIZES[2], color=C.ORANGE, va="bottom")
    ax.annotate("", xy=(b_ + 0.5, nb), xytext=(b_ + 0.5, na), arrowprops=dict(arrowstyle="<->", color=C.INK2, lw=0.8))
    ax.text(b_ + 0.65, (na + nb) / 2, f"ν_b − ν_a = {nb - na}\n= #{{λ ∈ (a, b)}}", fontsize=C.SIZES[2], color=C.INK2,
            va="center", linespacing=1.1)
    for e in (a_, b_):
        for d in (-0.45, 0.45):
            ax.plot([e + d, e + d], [-0.4, -0.15], color=C.INK2, lw=0.8)
    ax.text(a_, -0.55, "a", ha="center", va="top", fontsize=C.SIZES[1])
    ax.text(b_, -0.55, "b", ha="center", va="top", fontsize=C.SIZES[1])
    ax.text(0.3, -1.05, "counts at e ± δ (ticks) equal ⇒ collar e ± β empty",
            fontsize=C.SIZES[2], color=C.INK2, va="top")
    ax.set_xlim(0.2, 7.0)
    ax.set_ylim(-1.5, 6.6)
    ax.set_yticks(range(0, 7))
    ax.set_yticklabels([])
    ax.set_xticks([])

    # ---- (c) what fails ------------------------------------------------------------------------------
    ax = axes[2]
    C.panel_letter(ax, "c", dx=-0.2)
    ax.set_title("ν > k or a collar hit: refused")
    lams_c = [1.0, 2.1, 2.9, 3.6, 4.25, 5.4, 6.1]       # one eigenvalue the claim missed, below sigma
    staircase(ax, lams_c, 0.2, 7.0)
    collar(ax, sigma, beta, 6.0)
    for mu, r in zip(mus, rad):
        ax.plot([mu - r, mu + r], [-0.75, -0.75], color=C.BLUE, lw=1.8, solid_capstyle="butt")
    ax.plot([0.2, sigma + beta], [k + 1, k + 1], color=C.STATUS_COLORS["FAILED"], lw=0.8, ls="--")
    ax.text(0.3, k + 1 + 0.12, f"ν = {k + 1} > k = {k}: a missed eigenvalue\nbelow σ → FAILED",
            fontsize=C.SIZES[2], color=C.STATUS_COLORS["FAILED"], va="bottom", linespacing=1.1)
    ax.plot([4.25, 4.25], [-0.35, -0.1], color=C.STATUS_COLORS["FAILED"], lw=1.6)
    # an eigenvalue in the collar: undecided
    ax.plot([sigma + 0.12, sigma + 0.12], [-0.35, -0.1], color=C.STATUS_COLORS["INCONCLUSIVE"], lw=1.6)
    ax.annotate("eigenvalue inside\nσ ± β: undecided\n→ INCONCLUSIVE", xy=(sigma + 0.14, -0.1),
                xytext=(5.05, 1.0), fontsize=C.SIZES[2], color="#9a6f00", ha="left", va="bottom", linespacing=1.1,
                arrowprops=dict(arrowstyle="-|>", color="#9a6f00", lw=0.6))
    ax.text(sigma, -0.55, "σ", ha="center", va="top", fontsize=C.SIZES[1])
    ax.set_xlim(0.2, 7.0)
    ax.set_ylim(-1.5, 7.8)
    ax.set_yticks(range(0, 8))
    ax.set_yticklabels([])
    ax.set_xticks([])
    return fig


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    fig = draw()
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, {"data": "concept figure, no data (illustrative spectra)"})
    for f in findings:
        print("layout:", f)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
