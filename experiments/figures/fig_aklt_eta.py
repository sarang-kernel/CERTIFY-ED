"""A2 fig_aklt_eta: certified eta upper bounds for the Pomata-Wei AKLT lattices against the published values.

Reads e10_aklt_*/data.json. (a) per lattice: the certified upper bound `eta_upper` (bar), the published
value `eta_published` (open diamond; the two differ in the tenth decimal, so the bar end and the diamond
coincide at this scale and the values are printed), and the gap-criterion threshold 1/z for the lattices
whose criterion is eta < 1/z (`criterion_z`); the two square-octagon rows share the combined criterion
4 eta_perp + 2 eta_par < 1 (`square_octagon.sum_upper`). (b) the certified criterion value (z * eta_upper,
or the square-octagon sum) against 1: the margin by which each lattice satisfies its criterion.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figcommon as C  # noqa: E402

NAME = "fig_aklt_eta"
LABEL = {"star": "star [3.12.12]", "square_octagon_par": "square-octagon, η∥", "square_octagon_perp": "square-octagon, η⊥",
         "cross": "cross [4.6.12]", "decorated_honeycomb": "decorated honeycomb", "decorated_square": "decorated square"}
ORDER = ["star", "cross", "decorated_honeycomb", "decorated_square", "square_octagon_perp", "square_octagon_par"]


def draw(e10):
    import matplotlib.pyplot as plt
    C.style()
    lats = e10["lattices"]
    names = [n for n in ORDER if n in lats] + [n for n in lats if n not in ORDER]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.12,
                                                                              left=0.2, right=0.98, bottom=0.17, top=0.86))
    numbers = {"lattices": {}}
    crit_rows = []
    for i, n in enumerate(names):
        lat = lats[n]
        up, pub = lat["eta_upper"], lat["eta_published"]
        axa.barh(i, up, color=C.BLUE, height=0.6)
        axa.plot(pub, i, marker="D", mfc="white", mec=C.INK, ms=4, ls="none", zorder=4)
        z = lat.get("criterion_z")
        if z:
            axa.plot([1 / z, 1 / z], [i - 0.4, i + 0.4], color=C.ORANGE, lw=1.4)
            crit_rows.append((LABEL.get(n, n), f"{z} η", z * up, f"η < 1/{z}"))
        axa.text(0.375, i, f"≤ {up:.10f}   published {pub:.10f}", va="center", fontsize=C.SIZES[2] - 0.5,
                 color=C.INK2)
        numbers["lattices"][n] = {"eta_upper": up, "eta_published": pub, "eta_float": lat["eta_float"],
                                  "criterion_z": z, "criterion_holds": lat.get("criterion_holds"),
                                  "largest_block": lat["largest_block"], "seconds": lat["seconds"]}
    so = e10.get("square_octagon")
    if so and "square_octagon_perp" in lats and "square_octagon_par" in lats:
        crit_rows.append(("square-octagon", "4 η⊥ + 2 η∥", so["sum_upper"], "4 η⊥ + 2 η∥ < 1"))
        numbers["square_octagon"] = so
        i1, i2 = names.index("square_octagon_perp"), names.index("square_octagon_par")
        axa.plot([0.235, 0.235], [i1 - 0.3, i2 + 0.3], color=C.ORANGE, lw=1.0)
        axa.text(0.245, (i1 + i2) / 2, "joint criterion\n4 η⊥ + 2 η∥ < 1", va="center", ha="left", fontsize=C.SIZES[2],
                 color=C.ORANGE, linespacing=1.1)
    axa.set_yticks(range(len(names)))
    axa.set_yticklabels([LABEL.get(n, n) for n in names])
    axa.invert_yaxis()
    axa.set_xlim(0, 0.78)
    axa.set_xticks([0, 0.1, 0.2, 0.3])
    axa.set_xlabel("η  (bar: certified ≤; diamond: published; orange: threshold 1/z)")
    axa.set_title("certified η agrees with the published values to 2e-9")
    C.panel_letter(axa, "a", dx=-0.5)
    # (b) criterion values against 1
    for j, (_lab, _expr, val, _crit) in enumerate(crit_rows):
        axb.barh(j, val, color=C.TEAL, height=0.6)
        axb.text(val - 0.02, j, f"{val:.5f}", va="center", ha="right", fontsize=C.SIZES[2], color="white")
    axb.axvline(1.0, color=C.ORANGE, lw=1.2)
    axb.text(1.03, -0.6, "criterion: value < 1", ha="left", va="center", fontsize=C.SIZES[2], color=C.ORANGE)
    axb.set_yticks(range(len(crit_rows)))
    axb.set_yticklabels([f"{lab.split(' [')[0].replace('decorated', 'dec.')}: {expr}" for lab, expr, _, _ in crit_rows])
    axb.yaxis.tick_right()
    axb.set_xlim(0, 1.4)
    axb.set_ylim(len(crit_rows) - 0.4, -0.95)
    axb.set_xlabel("certified criterion value")
    axb.set_title("every criterion holds with margin")
    C.panel_letter(axb, "b", dx=-0.08)
    numbers["criterion_values"] = {lab + " " + expr: val for lab, expr, val, _ in crit_rows}
    numbers["max_criterion_value"] = max(v for _, _, v, _ in crit_rows)
    numbers["max_abs_diff_upper_minus_published"] = max(abs(lats[n]["eta_upper"] - lats[n]["eta_published"]) for n in names)
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0], needs=("e10_aklt",))
    e10, rec = C.load_record(a.results, "e10_aklt")
    fig, numbers = draw(e10)
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
