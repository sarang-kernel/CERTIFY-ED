"""Numbers and tables for the applications and the exact layer, read from the latest E6 (if run), E7, E8, E9 and E10
records (CERTIFY_RESULTS or experiments/results). Writes MANUSCRIPT_DIR/numbers_apps.tex and
MANUSCRIPT_DIR/tables/apps_{aklt,e6,e7,e8}.tex and MANUSCRIPT_DIR/figures/fig_widths.pdf, and checks the statements the text makes. Sections whose
record is missing are skipped and reported.

usage: python3 make_numbers_apps.py [MANUSCRIPT_DIR]   (default: <results dir>/paper; `certify-ed numbers` and run.py pass it)"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("CERTIFY_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), "paper"))
RES = os.environ.get("CERTIFY_RESULTS", os.path.join(HERE, "results"))
os.makedirs(os.path.join(MS, "tables"), exist_ok=True)
M, CHECKS = {}, []
macro = lambda k, v: M.__setitem__(k, str(v))
check = lambda c, s: CHECKS.append((bool(c), s))


ENVS = {}


def latest(prefix):
    c = sorted(d for d in glob.glob(os.path.join(RES, prefix + "_2*")) if os.path.isdir(d))
    if not c:
        return None
    try:
        with open(os.path.join(c[-1], "environment.json")) as f:
            ENVS[prefix] = json.load(f)
    except OSError:
        ENVS[prefix] = {}
    with open(os.path.join(c[-1], "data.json")) as f:
        return json.load(f)


SKIPPED = []

# ---------------------------------------------------------------- AKLT (E10)
e10 = latest("e10_aklt")
if e10 is None:
    SKIPPED.append("E10 (AKLT)")
else:
    lat = e10["lattices"]
    label = {"star": "star $[3.12.12]$", "square_octagon_perp": "square-octagon, $\\eta_\\perp$",
             "square_octagon_par": "square-octagon, $\\eta_\\parallel$", "cross": "cross $[4.6.12]$",
             "decorated_honeycomb": "decorated honeycomb", "decorated_square": "decorated square"}
    crit = {"star": "$\\eta<1/4$", "square_octagon_perp": "$4\\eta_\\perp+2\\eta_\\parallel<1$",
            "square_octagon_par": "", "cross": "$\\eta<1/3$", "decorated_honeycomb": "$\\eta<1/4$",
            "decorated_square": "$\\eta<1/4$"}
    def crit_value(name):
        """Left-hand side of the criterion, normalised so that the criterion reads 'value < 1'."""
        r = lat[name]
        if name == "square_octagon_perp" and "square_octagon" in e10:
            return e10["square_octagon"]["sum_upper"]
        return r["criterion_z"] * r["eta_upper"] if r.get("criterion_z") else None
    with open(os.path.join(MS, "tables", "apps_aklt.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l l r r r r@{}}\n\\toprule\nLattice & criterion & $\\eta$ (published) & "
                "$\\eta$ (our float) & certified upper bound & criterion value (certified)\\\\\n\\midrule\n")
        for name in label:
            if name not in lat:
                continue
            r = lat[name]
            cv = crit_value(name)
            f.write(f"{label[name]} & {crit[name]} & {r['eta_published']:.10f} & {r['eta_float']:.10f} & "
                    f"{r['eta_upper']:.10f} & {'' if cv is None else f'{cv:.5f}'}\\\\\n")
            # published values have 10 decimals (rounding error <= 5e-11)
            check(r["eta_upper"] >= r["eta_published"] - 5e-11, f"AKLT {name}: bound not below the published value")
            check(r["eta_upper"] - r["eta_published"] < 2e-10,
                  f"AKLT {name}: certified bound agrees with the published value to the 10th decimal (within 2 units)")
            check(r["eta_upper"] >= r["eta_float"], f"AKLT {name}: certified bound above our floating-point eta")
            check(abs(r["eta_float"] - r["eta_published"]) < 1e-8, f"AKLT {name}: geometry reproduces published eta")
            if r.get("criterion_holds") is not None:
                check(r["criterion_holds"], f"AKLT {name}: criterion holds (exact comparison)")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if "square_octagon" in e10:
        check(e10["square_octagon"]["criterion_holds"], "square-octagon: 4 eta_perp + 2 eta_par < 1")
        macro("SOsum", f"{e10['square_octagon']['sum_upper']:.5f}")
    for key, name in (("Star", "star"), ("Cross", "cross"), ("DecHoney", "decorated_honeycomb"),
                      ("DecSquare", "decorated_square")):
        if name in lat:
            macro(f"Eta{key}", f"{lat[name]['eta_upper']:.10f}")
    if "decorated_square" in lat:
        macro("DecSquareLargestBlock", lat["decorated_square"]["largest_block"])
    gaps = [lat[n]["eta_upper"] - lat[n]["eta_published"] for n in label if n in lat]
    macro("AKLTMaxExcess", f"{max(gaps):.0e}")
    cvs = [crit_value(n) for n in label if n in lat and crit_value(n) is not None]
    macro("AKLTCritMin", f"{min(cvs):.2f}")
    macro("AKLTCritMax", f"{max(cvs):.2f}")
    check(max(cvs) < 1, "AKLT: every criterion value is below 1")
    for name, ks in e10.get("kernel", {}).items():
        check(all(k["kernel_equals_vbs_range"] for k in ks), f"kernel lemma confirmed exactly for {name}")

# ---------------------------------------------------------------- pyrochlore (E9)
e9 = latest("e9_pyrochlore")
if e9 is None:
    SKIPPED.append("E9 (pyrochlore)")
else:
    p = e9["spin1_printed"]
    macro("CrownOneEig", f"{p['eigenvalue_upper']:.7f}")
    macro("CrownOnePublished", f"{p['published_bound']:.6f}")
    macro("CrownOneMargin", f"{p['margin']:.1e}")
    check(p["contradiction"], "spin-1 printed couplings: certified eigenvalue below the published bound")
    if "no_floating_point_assumption" in p:
        check(p["no_floating_point_assumption"], "spin-1 printed-coupling eigenvalue certified in exact arithmetic")
    macro("CrownOneLandscapeMin", f"{e9['spin1_landscape']['classes'][0]['energy']:.7f}")
    if "spin_half" in e9:
        macro("CrownHalfBound", e9["spin_half"]["lower_bound_rounded_down"])
        check(e9["spin_half"]["no_floating_point_assumption"], "crown S=1/2 bound certified in exact arithmetic")
        sh = e9["spin_half"]
        check(sh["lower_bound"] <= sh["e0_estimate"], "crown S=1/2: certified bound below the Lanczos estimate")
        check(sh["e0_estimate"] - sh["lower_bound"] < 2e-9, "crown S=1/2: certified bound within 2e-9 of lambda_min")
        check(sh["lower_bound"] >= -0.550158313, "crown S=1/2: at least as sharp as the 10-17 certificate (-0.550158313)")
        macro("CrownHalfEstimate", f"{sh['e0_estimate']:.10f}")
    else:
        SKIPPED.append("E9 S=1/2 crown (full mode)")

# ---------------------------------------------------------------- exact benchmarks (E7)
e7 = latest("e7_exact_benchmarks")
if e7 is None:
    SKIPPED.append("E7 (exact benchmarks)")
else:
    with open(os.path.join(MS, "tables", "apps_e7.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l r r r r r@{}}\n\\toprule\nSystem & block dim & sectors & $E_0$ & exact radius "
                "& time (s)\\\\\n\\midrule\n")
        for r in e7["E7"]:
            f.write(f"{r['system']} & {r['block_dim']} & {len(r['sectors'])} & {r['E0']:.12f} & "
                    f"{r['exact_radius']:.1e} & {r['exact_check_seconds']:.0f}\\\\\n")
            check(r["status"] == "CERTIFIED", f"E7 {r['system']} certified")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- symveig comparison (E8)
e8 = latest("e8_symveig_comparison")
if e8 is None:
    SKIPPED.append("E8 (symveig comparison)")
else:
    rows = e8["E8"]
    with open(os.path.join(MS, "tables", "apps_e8.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l r r r r r r@{}}\n\\toprule\nModel & $L$ & dim & symveig max $h$ & ours max $h$ & "
                "symveig median & ours median\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['L']} & {r['dim']} & {r['symveig_max_halfwidth']:.1e} & "
                    f"{r['certify_max_halfwidth']:.1e} & {r['symveig_median_halfwidth']:.1e} & "
                    f"{r['certify_median_halfwidth']:.1e}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    check(all(r["certify_complete"] for r in rows), "E8: every certify_ed whole-spectrum enclosure is complete")
    big = [r for r in rows if r["L"] == 12]
    if big:
        ratios = [r["symveig_median_halfwidth"] / r["certify_median_halfwidth"] for r in big]
        macro("EeightMedianRatioMin", f"{min(ratios):.0f}")
        macro("EeightMedianRatioMax", f"{max(ratios):.0f}")

    # figure: how the half-widths scale with the dimension, for the two rounding-bound forms
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # palette: reference categorical slots 1-2 (validated all-pairs, light mode); text in ink tokens
    BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                         "legend.frameon": False, "pdf.fonttype": 42})
    os.makedirs(os.path.join(MS, "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), sharey=True)
    dims = np.array([r["dim"] for r in rows], dtype=float)
    slopes = {}
    for ax, stat, title in ((axes[0], "median", "median half-width"), (axes[1], "max", "largest half-width")):
        for tool, key, color, mk, name in (("symveig", f"symveig_{stat}_halfwidth", BLUE, "o",
                                            "symveig 1.0.0 ($\\gamma_{4n}\\|A\\|_F\\|x\\|$)"),
                                           ("certify", f"certify_{stat}_halfwidth", ORANGE, "s",
                                            "certify-ed ($|A||x|$, counts certified)")):
            h = np.array([r[key] for r in rows])
            ax.plot(dims, h, mk, color=color, ms=4.5, ls="none", label=name, mec="white", mew=0.6)
            sel = dims >= 64       # the smallest blocks sit at the floor of a few ulps
            a, b = np.polyfit(np.log10(dims[sel]), np.log10(h[sel]), 1)
            slopes[(tool, stat)] = a
            xs = np.array([dims[sel].min(), dims.max()])
            ax.plot(xs, 10 ** (a * np.log10(xs) + b), "-", color=color, lw=1.2, alpha=0.7)
            ax.annotate(f"$\\propto d^{{{a:.2f}}}$", (xs[-1], 10 ** (a * np.log10(xs[-1]) + b)),
                        xytext=(6, -6 if tool == "certify" else 6), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("dimension of the matrix $d$")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(10, 4096 * 3)
    axes[0].set_ylabel("eigenvalue half-width")
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(MS, "figures", "fig_widths.pdf"))
    plt.close(fig)
    macro("EeightSlopeSymMedian", f"{slopes[('symveig', 'median')]:.2f}")
    macro("EeightSlopeOursMedian", f"{slopes[('certify', 'median')]:.2f}")
    macro("EeightSlopeSymMax", f"{slopes[('symveig', 'max')]:.2f}")
    macro("EeightSlopeOursMax", f"{slopes[('certify', 'max')]:.2f}")
    check(slopes[("certify", "median")] < slopes[("symveig", "median")],
          "E8: median half-widths grow more slowly with d for the |A||x| bound than for the gamma_4n ||A||_F form")
    ts = [r["certify_seconds"] / r["symveig_seconds"] for r in big] if big else []
    if ts:
        macro("EeightSlowdownMin", f"{min(ts):.1f}")
        macro("EeightSlowdownMax", f"{max(ts):.1f}")

# ---------------------------------------------------------------- mutation analysis (E6, opt-in)
e6 = latest("e6_mutation")
if e6 is None:
    SKIPPED.append("E6 (mutation analysis; opt-in: --with-mutation)")
else:
    e6dir = sorted(d for d in glob.glob(os.path.join(RES, "e6_mutation_2*")) if os.path.isdir(d))[-1]
    sm = e6["summary"]
    macro("MutN", sm["n"])
    macro("MutKilled", sm["killed"])
    macro("MutTimeouts", sm["timeouts"])
    macro("MutSurvived", sm["survived"])
    macro("MutScore", f"{sm['score']:.2f}")
    env6 = ENVS.get("e6_mutation", {})
    if "tests_dirty" in env6:
        check(not env6["tests_dirty"], "E6 ran against the committed test suite (no uncommitted test changes)")
    cls_path = os.path.join(e6dir, "classification.json")
    insp_path = os.path.join(e6dir, "inspection.json")
    cats = {}
    if os.path.exists(cls_path):
        with open(cls_path) as f:
            cls = json.load(f)
        check(cls["run"] == os.path.basename(e6dir), "E6 classification belongs to this run")
        check(sum(cls["survivor_categories"].values()) == sm["survived"], "E6: every survivor is classified")
        cats = dict(cls["survivor_categories"])
    else:
        SKIPPED.append("E6 classification (run e6_classify.py)")
    verdicts = {}
    if os.path.exists(insp_path):
        with open(insp_path) as f:
            insp = json.load(f)
        verdicts = insp["summary"]
        check(insp["inspected"] == cats.get("needs inspection", -1),
              "E6: every survivor that needed inspection has a recorded verdict")
    elif cats.get("needs inspection"):
        SKIPPED.append(f"E6 inspection ({cats['needs inspection']} survivors await a human verdict)")
    macro("MutNeedInspection", cats.get("needs inspection", 0))
    macro("MutGapsClosed", verdicts.get("test added, mutant now killed", 0))
    with open(os.path.join(MS, "tables", "apps_e6.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l r@{}}\n\\toprule\nOutcome & mutants\\\\\n\\midrule\n")
        f.write(f"sampled & {sm['n']}\\\\\nkilled by the test suite & {sm['killed']}\\\\\n"
                f"timed out & {sm['timeouts']}\\\\\nsurvived & {sm['survived']}\\\\\n\\midrule\n")
        for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
            if c == "needs inspection":
                continue
            f.write(f"\\quad survivor: {c} & {n}\\\\\n")
        if cats.get("needs inspection"):
            f.write(f"\\quad survivor: needed inspection & {cats['needs inspection']}\\\\\n")
            for v, n in sorted(verdicts.items(), key=lambda kv: -kv[1]):
                f.write(f"\\qquad verdict: {v} & {n}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- provenance
_src = {k: (e.get("source_hash"), e.get("source_dirty")) for k, e in ENVS.items()}
if _src:
    check(len({h for h, _ in _src.values()}) == 1 and None not in {h for h, _ in _src.values()},
          f"E6-E10 ran with the same computational source ({sorted({str(h) for h, _ in _src.values()})})")
    check(not any(d for _, d in _src.values()), "E6-E10 ran without uncommitted changes to certify_ed/ or experiments/")
    macro("SourceHash", next(iter(_src.values()))[0])

with open(os.path.join(MS, "numbers_apps.tex"), "w") as f:
    for k, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
bad = [s for ok, s in CHECKS if not ok]
for ok, s in CHECKS:
    print(("ok    " if ok else "CHECK FAILED ") + s)
for s in SKIPPED:
    print("skipped " + s)
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
sys.exit(1 if bad else 0)
