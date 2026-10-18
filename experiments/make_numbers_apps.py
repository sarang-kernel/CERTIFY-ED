"""Numbers and tables for the applications and the exact layer, read from the latest E7, E8, E9 and E10
records (CERTIFY_RESULTS or experiments/results). Writes MANUSCRIPT_DIR/numbers_apps.tex and
MANUSCRIPT_DIR/tables/apps_{aklt,e7,e8}.tex and checks the statements the text makes. Sections whose
record is missing are skipped and reported.

usage: python3 make_numbers_apps.py [MANUSCRIPT_DIR]   (default: ../../../ms_v2)"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "..", "ms_v2"))
RES = os.environ.get("CERTIFY_RESULTS", os.path.join(HERE, "results"))
os.makedirs(os.path.join(MS, "tables"), exist_ok=True)
M, CHECKS = {}, []
macro = lambda k, v: M.__setitem__(k, str(v))
check = lambda c, s: CHECKS.append((bool(c), s))


def latest(prefix):
    c = sorted(d for d in glob.glob(os.path.join(RES, prefix + "_2*")) if os.path.isdir(d))
    if not c:
        return None
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
    with open(os.path.join(MS, "tables", "apps_aklt.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l l r r r@{}}\n\\toprule\nLattice & criterion & $\\eta$ (published) & "
                "$\\eta$ (our float) & certified upper bound\\\\\n\\midrule\n")
        for name in label:
            if name not in lat:
                continue
            r = lat[name]
            f.write(f"{label[name]} & {crit[name]} & {r['eta_published']:.10f} & {r['eta_float']:.10f} & "
                    f"{r['eta_upper']:.10f}\\\\\n")
            check(r["eta_upper"] >= r["eta_published"], f"AKLT {name}: bound not below the published value")
            check(r["eta_upper"] - r["eta_published"] < 1e-4, f"AKLT {name}: bound within 1e-4 of published")
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
    macro("CrownOneLandscapeMin", f"{e9['spin1_landscape']['classes'][0]['energy']:.7f}")
    if "spin_half" in e9:
        macro("CrownHalfBound", e9["spin_half"]["lower_bound_rounded_down"])
        check(e9["spin_half"]["no_floating_point_assumption"], "crown S=1/2 bound certified in exact arithmetic")
        check(abs(e9["spin_half"]["lower_bound"] - (-0.550158313)) < 1e-8,
              "crown S=1/2 bound agrees with the earlier certificate (-0.550158313)")
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
