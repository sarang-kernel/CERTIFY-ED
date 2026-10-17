"""Numbers and tables for the application section (proof lemmas, exact re-checks), read from the
certificate files of applications/ and the E7 record. Writes MANUSCRIPT_DIR/numbers_apps.tex and
MANUSCRIPT_DIR/tables/apps_{aklt,e7,crown}.tex, and checks the statements the text makes.

usage: python3 make_numbers_apps.py [MANUSCRIPT_DIR]   (default: ../../../ms_v2)"""
import glob, json, os, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "..", "ms_v2"))
APP = os.path.join(ROOT, "applications")
os.makedirs(os.path.join(MS, "tables"), exist_ok=True)
M, CHECKS = {}, []
macro = lambda k, v: M.__setitem__(k, str(v))
check = lambda c, s: CHECKS.append((bool(c), s))
J = lambda *p: json.load(open(os.path.join(APP, *p)))


def best(th):
    """Smallest certified eta upper bound over thresholds (sharp threshold)."""
    vals = []
    for k, v in th.items():
        if v.get("certified") or v.get("at_most_32_above_tprime"):
            vals.append(v["t_prime"] ** 0.5)
    return min(vals)


# ---------------------------------------------------------------- AKLT
star = J("aklt_gap", "star_certificate.json")
so = J("aklt_gap", "square_octagon_certificate.json")
cr = J("aklt_gap", "cross_certificate.json")
dec = J("aklt_gap", "decorated_certificate.json")
rows = [
    ("star $[3.12.12]$", "$\\eta<1/4$", 0.1110430220, star["eta_upper"], 0.25),
    ("square-octagon, $\\eta_\\perp$", "$4\\eta_\\perp+2\\eta_\\parallel<1$", 0.1589663310, best(so["perp"]["thresholds"]), None),
    ("square-octagon, $\\eta_\\parallel$", "", 0.1061446858, best(so["par"]["thresholds"]), None),
    ("cross $[4.6.12]$", "$\\eta<1/3$", 0.1997384500, best(cr["thresholds"]), 1 / 3),
    ("decorated honeycomb", "$\\eta<1/4$", 0.1530329085, best(dec["decorated_honeycomb"]["thresholds"]), 0.25),
    ("decorated square", "$\\eta<1/4$", 0.2203543174, best(dec["decorated_square"]["thresholds"]), 0.25),
]
so_sum = 4 * rows[1][3] + 2 * rows[2][3]
with open(os.path.join(MS, "tables", "apps_aklt.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}l l r r@{}}\n\\toprule\nLattice & criterion & $\\eta$ (Pomata--Wei, float) & certified upper bound\\\\\n\\midrule\n")
    for name, crit, paper, up, lim in rows:
        f.write(f"{name} & {crit} & {paper:.10f} & {up:.10f}\\\\\n")
        check(up >= paper, f"certified bound for {name} is not below the published float value")
        check(up - paper < 1e-4, f"certified bound for {name} within 1e-4 of the published value")
        if lim is not None:
            check(up < lim, f"criterion holds for {name}")
    f.write("\\bottomrule\n\\end{tabular}\n")
check(so_sum < 1, "square-octagon: 4 eta_perp + 2 eta_par < 1")
macro("SOsum", f"{so_sum:.5f}")
for key, name in (("Star", 0), ("Cross", 3), ("DecHoney", 4), ("DecSquare", 5)):
    macro(f"Eta{key}", f"{rows[name][3]:.10f}")
macro("DecSquareLargestBlock", dec["decorated_square"].get("largest_block", ""))
macro("DecSquareSeconds", f"{dec['decorated_square']['seconds'] / 60:.0f}")
neg = dec["decorated_honeycomb"]["thresholds"].get("117/5000")
check(neg is not None and not neg["certified"], "negative control (t below eta^2) is not certified")
kern = J("aklt_gap", "kernel_certificate.json")
check(all(x["kernel_equals_vbs_range"] for v in kern.values() for x in v), "exact kernel checks all pass")

# ---------------------------------------------------------------- crown
c12 = J("pyrochlore_crown", "crown_381414_817957.json")
s1p = J("pyrochlore_crown", "crown_s1_printed.json")
s1o = J("pyrochlore_crown", "crown_s1_opt3.json")
macro("CrownHalfBound", c12["lower_bound_rounded_down"])
macro("CrownOneEig", f"{s1p['eigenvalue_upper']:.7f}")
macro("CrownOnePublished", f"{s1p['published_bound']:.6f}")
macro("CrownOneMargin", f"{s1p['margin']:.1e}")
macro("CrownOneMaxMin", f"{s1o['maxmin']:.7f}")
check(s1p["contradiction"], "spin-1 printed couplings: certified eigenvalue below the published bound")

# ---------------------------------------------------------------- E7
e7dirs = sorted(glob.glob(os.path.join(HERE, "results", "e7_exact_benchmarks_2*")))
e7 = json.load(open(os.path.join(e7dirs[-1], "data.json")))["E7"]
with open(os.path.join(MS, "tables", "apps_e7.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}l r r r r r@{}}\n\\toprule\nSystem & block dim & sectors & $E_0$ & exact radius & time (s)\\\\\n\\midrule\n")
    for r in e7:
        f.write(f"{r['system']} & {r['block_dim']} & {len(r['sectors'])} & {r['E0']:.12f} & "
                f"{r['exact_radius']:.1e} & {r['exact_check_seconds']:.0f}\\\\\n")
        check(r["status"] == "CERTIFIED", f"E7 {r['system']} certified")
    f.write("\\bottomrule\n\\end{tabular}\n")
macro("EsevenAllCertified", "yes" if all(r["status"] == "CERTIFIED" for r in e7) else "no")

with open(os.path.join(MS, "numbers_apps.tex"), "w") as f:
    for k, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
bad = [s for ok, s in CHECKS if not ok]
for ok, s in CHECKS:
    print(("ok    " if ok else "CHECK FAILED ") + s)
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
sys.exit(1 if bad else 0)
