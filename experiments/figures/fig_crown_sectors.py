"""A1 fig_crown_sectors: the pyrochlore-crown S = 1 pair-spin sector landscape and the named sectors.

Reads e9_pyrochlore_*/data.json. (a) `spin1_landscape`: the lowest eigenvalue of every pair-spin class of
the S^z = 0 block (92 classes), sorted by energy, at the record's re-optimised couplings
(`spin1_landscape.couplings`); marker area grows with the class dimension; the minimising class, the
all-singlet class and the largest class are labelled. The record holds the landscape at these couplings
only, not at the printed ones. (b) `spin1_printed`: at the couplings printed in the paper (c1, c3, c17 as
exact rationals), the certified upper bound on the all-singlet-sector eigenvalue (`eigenvalue_upper`,
exact re-check, radius `E0_radius_exact`) against the published lower bound (`published_bound`), and the
two named pair-spin sectors (`named_sectors`), whose energies are floating-point values without a
certified interval in the record and are drawn as such.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_crown_sectors"


def label(cfg):
    return "(" + ",".join(str(t // 2) for t in cfg) + ")"


def draw(e9):
    import matplotlib.pyplot as plt
    C.style()
    land = e9["spin1_landscape"]
    cls = land["classes"]
    p = e9["spin1_printed"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.45,
                                                                              left=0.1, right=0.98, bottom=0.17, top=0.86))
    es = [c["energy"] for c in cls]
    dims = [c["dim"] for c in cls]
    sizes = [4 + 3 * math.log10(d) ** 1.6 for d in dims]
    axa.scatter(range(len(cls)), es, s=sizes, fc="none", ec=C.INK2, lw=0.6, zorder=3)
    i_min = 0
    i_sing = next(i for i, c in enumerate(cls) if all(t == 0 for t in c["pair_spins2"]))
    i_big = max(range(len(cls)), key=lambda i: dims[i])
    from matplotlib.lines import Line2D
    handles = []
    for i, col, txt in ((i_min, C.BLUE, f"lowest class {label(cls[i_min]['pair_spins2'])}, dim {dims[i_min]:,}, index {i_min}"),
                        (i_sing, C.ORANGE, f"all-singlet class {label(cls[i_sing]['pair_spins2'])}, dim {dims[i_sing]}, index {i_sing}"),
                        (i_big, C.PURPLE, f"largest class {label(cls[i_big]['pair_spins2'])}, dim {dims[i_big]:,}, index {i_big}")):
        axa.scatter([i], [es[i]], s=18, fc=col, ec="white", lw=0.5, zorder=5)
        handles.append(Line2D([], [], marker="o", ls="none", color=col, label=txt))
    handles.append(Line2D([], [], marker="o", ls="none", mfc="none", mec=C.INK2, label="other classes; marker area grows with dim"))
    axa.legend(handles=handles, loc="upper left", fontsize=C.SIZES[2], handletextpad=0.3)
    spread = max(es[i_min], es[i_sing], es[i_big]) - min(es[i_min], es[i_sing], es[i_big])
    axa.text(0.98, 0.04, f"the three labelled classes lie within {spread:.1e} of each other", transform=axa.transAxes,
             ha="right", va="bottom", fontsize=C.SIZES[2], color=C.INK2)
    axa.set_xlabel("pair-spin class, sorted by lowest eigenvalue")
    axa.set_ylabel("lowest eigenvalue of the class")
    c1, c3 = land["couplings"]["c1"], land["couplings"]["c3"]
    axa.set_title(f"{len(cls)} pair-spin classes at the re-optimised couplings")
    axa.margins(x=0.05, y=0.15)
    C.panel_letter(axa, "a", dx=-0.2)
    # (b) printed couplings: the named sectors and the certified all-singlet bound
    pub = p["published_bound"]
    up = p["eigenvalue_upper"]
    named = p.get("named_sectors", [])
    axb.axhline(pub, color=C.INK2, lw=0.8, ls="--")
    axb.text(-0.55, pub - 0.0002, f"published lower\nbound {pub:.6f}", fontsize=C.SIZES[2], color=C.INK2, va="top",
             linespacing=1.1)
    xs_named = [1.0, 2.4]
    for j, s in enumerate(named):
        axb.plot(xs_named[j], s["energy"], marker="o", mfc="white", mec=C.INK, ms=5, ls="none", zorder=3)
        axb.text(xs_named[j], s["energy"] + 0.0009, f"{label(s['pair_spins2'])}\ndim {s['dim']:,}\n{s['energy']:.7f} (FP)",
                 ha="center", va="bottom", fontsize=C.SIZES[2], color=C.INK, linespacing=1.1)
    axb.plot(0, up, marker="v", color=C.BLUE, ms=6, ls="none", zorder=3)
    axb.text(0, up - 0.0004, f"all-singlet sector\n≤ {up:.7f}\n(exact re-check, r = {p['E0_radius_exact']:.0e})",
             ha="center", va="top", fontsize=C.SIZES[2], color=C.BLUE, linespacing=1.1)
    axb.annotate("", xy=(3.1, up), xytext=(3.1, pub), arrowprops=dict(arrowstyle="<->", color=C.BLUE, lw=0.8))
    axb.text(3.18, (up + pub) / 2, f"margin\n{p['margin']:.1e}", fontsize=C.SIZES[2], color=C.BLUE, va="center",
             linespacing=1.1)
    axb.set_xlim(-0.6, 3.9)
    axb.set_xticks([0, 1.0, 2.4])
    axb.set_xticklabels(["all-singlet", "named A", "named B"])
    axb.set_ylim(up - 0.003, pub + 0.0028)
    axb.set_ylabel("eigenvalue (S^z = 0 block)")
    axb.set_title("printed couplings: certified ≤ published")
    C.panel_letter(axb, "b", dx=-0.42)
    numbers = {"landscape": {"classes": len(cls), "couplings": land["couplings"], "min_class": cls[i_min],
                             "all_singlet_class": cls[i_sing], "largest_class": cls[i_big]},
               "printed": {k: p[k] for k in ("c1", "c3", "c17", "eigenvalue_upper", "published_bound", "margin",
                                            "E0_radius_exact", "E0_radius_standard_model", "exact_recheck",
                                            "no_floating_point_assumption", "contradiction") if k in p},
               "named_sectors": named, "named_sectors_have_intervals": any("radius" in s or "interval" in s for s in named)}
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0], needs=("e9_pyrochlore",))
    e9, rec = C.load_record(a.results, "e9_pyrochlore")
    fig, numbers = draw(e9)
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
