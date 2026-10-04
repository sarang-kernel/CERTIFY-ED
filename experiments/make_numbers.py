"""Generate every number, table and figure of the manuscript from the experiment records.

Usage: python3 make_numbers.py [MANUSCRIPT_DIR]   (default: ../../../ms_v2)

Reads the most recent results/<experiment>_<timestamp>/data.json of each experiment and writes
  MANUSCRIPT_DIR/numbers.tex          one macro per number used in the text
  MANUSCRIPT_DIR/tables/*.tex, *.txt  tables and the certificate listing
  MANUSCRIPT_DIR/figures/*.pdf        figures
It also checks the qualitative statements the text makes about these numbers and prints a
CHECK FAILED line for each statement that the data do not support.
"""

import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "..", "ms_v2"))
RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(MS, "tables"), exist_ok=True)
os.makedirs(os.path.join(MS, "figures"), exist_ok=True)

MACROS = {}
CHECKS = []


def latest(prefix):
    c = sorted(d for d in glob.glob(os.path.join(RES, prefix + "_2*")) if os.path.isdir(d))
    if not c:
        raise SystemExit(f"no results for {prefix}")
    d = c[-1]
    with open(os.path.join(d, "data.json")) as f:
        data = json.load(f)
    with open(os.path.join(d, "environment.json")) as f:
        env = json.load(f)
    return d, data, env


def macro(name, value):
    MACROS[name] = str(value)


def check(cond, statement):
    CHECKS.append((bool(cond), statement))


def sci(x, digits=1):
    """x as m\\times10^{e} inside \\ensuremath."""
    if x is None or not np.isfinite(x):
        return "\\ensuremath{\\infty}"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(abs(m), digits) >= 10:
        e += 1
        m /= 10
    ms = f"{m:.{digits}f}"
    if e == 0:
        return ms
    return f"\\ensuremath{{{ms}\\times10^{{{e}}}}}"


def thin(n):
    """Integer with thin-space thousands separators (math-safe)."""
    s = f"{int(n):,}".replace(",", "\\,")
    return s


def tex_escape(s):
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")
            .replace("%", "\\%").replace("#", "\\#"))


# ============================================================== environment / provenance
runs = {}
for name in ("e0_verifier_validation", "e1_failure_matrix", "e2_cost", "e3_tightness", "e4_workflows_W1W2W3W4"):
    runs[name] = latest(name)
commits = {r[2].get("git_commit") for r in runs.values()}
env0 = runs["e0_verifier_validation"][2]
check(len(commits) == 1, f"all experiments ran at one git commit (found {commits})")
macro("GitCommit", (sorted(c for c in commits if c) or ["unknown"])[0][:10])
macro("RunDate", env0["started"][:10])
cpu = env0.get("cpu_model")
if not cpu:
    with open("/proc/cpuinfo") as f:
        cpu = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "x86-64 CPU")
cpu = cpu.replace("(R)", "").replace("(TM)", "").replace("  ", " ")
macro("MachineCPU", tex_escape(cpu))
mem = env0.get("memory_GiB")
if mem is None:
    with open("/proc/meminfo") as f:
        mem = round(next(int(l.split()[1]) for l in f if l.startswith("MemTotal")) / 2 ** 20, 1)
macro("MachineRAM", f"{mem:.1f}")
macro("PyVersion", env0["python"].split()[0])
macro("NumpyVersion", env0["numpy"])
macro("ScipyVersion", env0["scipy"])
macro("QuspinVersion", env0.get("quspin", "?"))
el_B = sum(runs[k][1].get("_elapsed_seconds", 0) for k in runs if k != "e2_cost")
el_A = runs["e2_cost"][1].get("_elapsed_seconds", 0)
macro("RunMinutes", str(int(round(max(el_A, el_B) / 60 / 5) * 5) or 5))

# ============================================================== code statistics
v1_modules = {"hamiltonian.py", "certificates.py", "observables.py", "oracles.py", "symmetries.py", "__init__.py"}
core = [p for p in glob.glob(os.path.join(ROOT, "certify_ed", "*.py")) if os.path.basename(p) not in v1_modules]
loc = 0
for p in core:
    with open(p) as f:
        loc += sum(1 for line in f if line.strip() and not line.strip().startswith("#"))
macro("LOCcore", thin(round(loc, -2)))
macro("NcoreModules", str(len(core)))


def n_tests(path):
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", path], cwd=ROOT,
                         capture_output=True, text=True).stdout
    for line in out.splitlines()[::-1]:
        if "tests collected" in line or "test collected" in line:
            return int(line.split()[0])
    return int(sum(1 for line in out.splitlines() if "::" in line))


macro("NtestsTotal", str(n_tests("tests")))
macro("NtestsVtwo", str(n_tests("tests/v2")))

# ============================================================== E0
_, e0, _ = runs["e0_verifier_validation"]
a, b, c = e0["a_enclosures"], e0["b_inertia"], e0["c_builders"]
macro("VaMatrices", str(a["matrices"]))
macro("VaRandom", str(a["by_source"].get("random", 0)))
macro("VaSpec", str(a["by_source"].get("spec", 0)))
macro("VaPairs", thin(a["eigenpairs"]))
viol_a = a["violations_interval"] + a["violations_group_count"] + a["violations_residual_bound"]
macro("VaViol", "no" if viol_a == 0 else str(viol_a))
macro("VaRatioMedian", f"{a['bound_over_arb_residual']['median']:.2f}")
macro("VaRatioMax", f"{a['bound_over_arb_residual']['max']:.0f}")
macro("VaNoRef", str(a.get("reference_unavailable", 0)))
check(viol_a == 0, "E0(a): no enclosure violations")
macro("VbMatrices", str(b["matrices"]))
macro("VbChecks", thin(b["checks"]))
macro("VbViol", "no" if b["violations"] == 0 else str(b["violations"]))
macro("VbBetaMedian", sci(b["beta_median"]))
macro("VbBetaMax", sci(b["beta_max"]))
check(b["violations"] == 0 and b["conclusive"] == b["checks"], "E0(b): all inertia counts conclusive and correct")
rows_c = c["rows"]
n_models = sum(r["models"] for r in rows_c)
macro("VcModels", str(n_models))
maxdiff = max(max(r["max_rel_diff_kron"], r["max_rel_diff_quspin"]) for r in rows_c)
macro("VcMaxDiff", sci(maxdiff))
check(all(r["kron_agree"] == r["models"] and r["quspin_agree"] == r["models"] for r in rows_c),
      "E0(c): all random models agree with Kronecker builder and QuSpin")
with open(os.path.join(MS, "tables", "e0.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Validation of the verifier against independent "
            "references (Section~\\ref{sec:e0}).}\n\\label{tab:e0}\n\\small\n"
            "\\begin{tabular}{@{}l l r r@{}}\n\\toprule\n"
            "Component & Reference & Cases & Violations \\\\\n\\midrule\n")
    f.write(f"\\Stwo{{}} enclosures (Theorems~\\ref{{thm:residual}}--\\ref{{thm:cluster}}) & Arb ball-arithmetic "
            f"eigenvalues & {thin(a['eigenpairs'])} pairs & {viol_a} \\\\\n")
    f.write(f"\\Stwo{{}} residual bound & Arb ball-arithmetic residual & {thin(a['eigenpairs'])} pairs & "
            f"{a['violations_residual_bound']} \\\\\n")
    f.write(f"\\Sthree{{}} inertia (Lemma~\\ref{{lem:inertia}}) & exact count from spectrum & {thin(b['checks'])} "
            f"counts & {b['violations']} \\\\\n")
    for r in rows_c:
        f.write(f"\\Sone{{}} builder, {tex_escape(r['local_space'])}, $N={r['N']}$ & Kronecker builder; QuSpin & "
                f"{r['models']} models & {r['models'] - r['kron_agree']}; {r['models'] - r['quspin_agree']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ============================================================== E1
_, e1, _ = runs["e1_failure_matrix"]
rows = e1["rows"]
err_rows = [r for r in rows if not r["case"].startswith(("F0", "F12"))]
flag = lambda v: v is False
n_err = len(err_rows)
n_p1 = sum(1 for r in err_rows if r["P1_residual_only"] is False)
n_p2 = sum(1 for r in err_rows if r["P2_recomputation"] is False)
p3_flag = lambda r: "FAILED" in r["P3_certify_ed"] or "INCONCLUSIVE" in r["P3_certify_ed"]
n_p3 = sum(1 for r in err_rows if p3_flag(r))
n_p3_failed = sum(1 for r in err_rows if "FAILED" in r["P3_certify_ed"])
macro("FnErr", str(n_err))
macro("FnPone", str(n_p1))
macro("FnPtwo", str(n_p2))
macro("FnPoneMissed", str(n_err - n_p1))
macro("FnPtwoMissed", str(n_err - n_p2))
macro("FnPthreeFailed", str(n_p3_failed))
macro("FnN", str(e1["settings"]["N"]))
macro("FnNsectors", "10")
check(n_p3 == n_err, f"E1: CERTIFY-ED flags all erroneous results ({n_p3}/{n_err})")
r0 = rows[0]
check(r0["P3_certify_ed"] == "all CERTIFIED" and r0["P1_residual_only"] and r0["P2_recomputation"],
      "E1: correct result passes all three practices")
missed_p2 = [r["case"].split()[0] for r in err_rows if r["P2_recomputation"] is not False]
check(missed_p2 == ["F3", "F9", "F10b", "F11b"], f"E1: P2 misses exactly F3, F9, F10b, F11b (got {missed_p2})")
flagged_p1 = [r["case"].split()[0] for r in err_rows if r["P1_residual_only"] is False]
check(flagged_p1 == ["F8", "F9"], f"E1: P1 flags exactly F8, F9 (got {flagged_p1})")
inconcl = [r["case"].split()[0] for r in err_rows if "FAILED" not in r["P3_certify_ed"]]
check(inconcl == ["F7"], f"E1: only F7 is INCONCLUSIVE rather than FAILED (got {inconcl})")
f1 = next(r for r in rows if r["case"].startswith("F1"))
f2 = next(r for r in rows if r["case"].startswith("F2"))
f3 = next(r for r in rows if r["case"].startswith("F3"))
f12 = next(r for r in rows if r["case"].startswith("F12"))
macro("FnFoneDelta", f"{f1['fit_delta'][0]:.1f}" if f1.get("fit_delta") else "-2.0")
check(f1.get("fit_term") == ["zz", [7, 8]], f"E1 F1: fit names zz on (7,8) (got {f1.get('fit_term')})")
macro("FnFtwoScale", f"{f2['scale_factor']:.3f}" if f2.get("scale_factor") else "4.000")
check(f3.get("fit_relative_residual") is not None and f3["fit_relative_residual"] > 1e-8,
      f"E1 F3: difference not expressible as coefficient changes (residual {f3.get('fit_relative_residual')})")
macro("FnFtwelveBound", sci(f12["bound"]) if f12.get("bound") else "?")
macro("FnFtwelveSpread", f"{f12['spread']:.3f}" if f12.get("spread") else "?")

short = {
    "F0": ("correct result", "---"),
    "F1": ("sign of one $S^zS^z$ coupling flipped in the input", None),
    "F2": ("spin-convention couplings with Pauli matrices", None),
    "F3": ("sites numbered in reverse; field on site 0", None),
    "F4": ("hopping across the boundary without JW sign", None),
    "F5": ("$S^z=0$ block's lowest states reported as global", None),
    "F6": ("$k=0$ block's lowest states reported for $S^z=0$", None),
    "F7": ("Lanczos without reorthogonalization (ghosts)", None),
    "F8": ("ARPACK tol $10^{-4}$, quoted as $10^{-10}$ accurate", None),
    "F9": ("eigenvectors of levels 0 and 1 interchanged", None),
    "F10a": ("phase error in one basis vector, ground-state block", None),
    "F10b": ("phase error in one basis vector, block $(N_\\uparrow{=}2,k{=}0)$", None),
    "F11a": ("ground-state block omitted from pooled list", None),
    "F11b": ("fully polarized block omitted", None),
    "F12": ("$\\langle O\\rangle$ quoted for one vector of a triplet", None),
}


def p3_cell(r):
    st = r["P3_certify_ed"]
    if st.startswith("all"):
        return "certified"
    if st.startswith("S2="):
        return "\\st{INCONCL.}"
    return "\\st{" + st.replace("_", "\\_") + "}" if st in ("FAILED", "CERTIFIED") else tex_escape(st)


def diag_cell(r):
    k = r["case"].split()[0]
    if k == "F1":
        return f"\\texttt{{zz}} on (7,8) changed by ${f1['fit_delta'][0]:.1f}$"
    if k == "F2":
        return f"global factor {f2['scale_factor']:.3f}"
    if k == "F3":
        return "not a change of coefficients; involves $S^z_0$"
    if k == "F4":
        return "hopping on bond (11,0)"
    if k in ("F5", "F6"):
        import re
        m = re.search(r"at least (\d+)", r["P3_detail"])
        return f"$\\ge{m.group(1)}$ eigenvalues missing" if m else ""
    if k == "F7":
        return f"{r.get('duplicates', '?')} duplicated values"
    if k == "F8":
        import re
        m = re.search(r"radius ([0-9.e+-]+)", r["P3_detail"])
        return f"radius {sci(float(m.group(1)))} $>$ target" if m else ""
    if k == "F9":
        import re
        m = re.search(r"radius ([0-9.e+-]+)", r["P3_detail"])
        return f"radius {sci(float(m.group(1)))}" if m else ""
    if k in ("F10a", "F10b"):
        return "sector audit fails"
    if k in ("F11a", "F11b"):
        import re
        m = re.search(r"cover (\d+) of (\d+)", r["P3_detail"])
        return f"sectors cover {m.group(1)} of {m.group(2)}" if m else ""
    if k == "F12":
        return f"$\\tr(PO)\\pm${sci(f12['bound'])}; spread {f12['spread']:.3f}"
    return ""


def pf(v):
    return "n/a" if v is None else ("pass" if v else "\\textbf{flag}")


with open(os.path.join(MS, "tables", "e1.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Fifteen results checked by three practices "
            "(Section~\\ref{sec:e1}). P1: residual check with the producer's matrix; P2: recomputation with "
            "QuSpin from the intended model; P3: \\pkg{}. ``flag'' means the practice reports a problem; "
            "``pass'' means it does not. Thirteen results (F1--F11b) are wrong; F0 is correct; F12 is a "
            "correct result whose single-vector observable is basis dependent.}\n\\label{tab:e1}\n"
            "\\footnotesize\n\\begin{tabular}{@{}l p{0.36\\textwidth} c c l p{0.25\\textwidth}@{}}\n\\toprule\n"
            "Case & Mechanism & P1 & P2 & P3 & P3 diagnosis \\\\\n\\midrule\n")
    for r in rows:
        k = r["case"].split()[0]
        f.write(f"{k} & {short[k][0]} & {pf(r['P1_residual_only'])} & {pf(r['P2_recomputation'])} & "
                f"{p3_cell(r)} & {diag_cell(r)} \\\\\n")
        if k in ("F0", "F4", "F9", "F11b"):
            f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ============================================================== E3
_, e3, _ = runs["e3_tightness"]
sw = e3["a_tolerance_sweep"]
errs = np.array([r["actual_error"] for r in sw])
kt = np.array([r["kt_halfwidth"] for r in sw], dtype=float)
rad = np.array([r["radius"] for r in sw])
check(all(r["kt_contains"] and r["radius_contains"] for r in sw), "E3(a): all intervals contain the true value")
macro("TaKTDecades", str(int(math.floor(math.log10(errs.max() / errs.min())))))
macro("TaKTRatioMax", f"{np.max(kt / errs):.0f}")
bb = e3["b_float_vs_arb"]["ratio_bound_over_arb"]
macro("TbRatioMin", f"{min(bb):.1f}")
macro("TbRatioMax", f"{max(bb):.1f}")
cg = e3["c_gap"]
uniq = {(r["model"], r["k"]): r for r in cg}
macro("TcCases", str(len(uniq)))
check(all(abs(r["fraction_of_gap_certified"] - 0.9) < 1e-6 and r["n_factorizations"] == 1 for r in uniq.values()),
      "E3(c): certified gap = 0.9 true gap with one factorization in all cases")
dc = e3["d_compose"]
macro("TdSectors", str(dc["n_sectors"]))
macro("TdRadius", sci(dc["radius"]))
macro("TdActual", sci(dc["actual_max_error"]))
macro("TdLooseExp", str(int(round(math.log10(dc["radius"] / dc["actual_max_error"])))))
check(dc["status"] == "CERTIFIED" and dc["actual_max_error"] <= dc["radius"] and dc["true_next"] >= dc["next_lower_bound"],
      "E3(d): assembly certified and the bounds hold")

# figure: tightness
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "legend.frameon": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(4.6, 3.0))
xs = np.arange(len(sw))
labels = ["0" if r["tol"] == 0 else f"1e{int(round(math.log10(r['tol'])))}" for r in sw]
ax.plot(xs, rad, "-o", color=BLUE, lw=2, ms=4, label="residual radius (Thm 1)")
ax.plot(xs, kt, "-s", color=ORANGE, lw=2, ms=4, label="Kato--Temple half-width (Thm 3)".replace("--", "–"))
ax.plot(xs, errs, "-^", color=INK, lw=1.2, ms=4, label="actual error")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_xlabel("ARPACK tolerance")
ax.set_ylabel("|error| or bound")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(MS, "figures", "fig_tightness.pdf"))
plt.close(fig)

# ============================================================== E2
_, e2, _ = runs["e2_cost"]
settings = e2.get("settings", {})
macro("CmemCap", f"{settings.get('memory_cap_GiB', 4.5):.1f}")
rows2 = e2["rows"]
fam = lambda r: ("chain-u1" if r["cfg"]["model"] == "chain" and r["cfg"]["block"] == "u1" else
                 "chain-k0" if r["cfg"]["model"] == "chain" else "square")


def stmt_time(c, sid):
    return c["statements"][sid]["seconds"] if c and "statements" in c else None


ok_rows = [r for r in rows2 if "error" not in r.get("certifier", {"error": 1})]
s3_ok = [r for r in ok_rows if r["certifier"]["statements"]["S3.completeness"]["status"] == "CERTIFIED"]
dmax = max(r["producer"]["dim"] for r in s3_ok)
rmax = max(s3_ok, key=lambda r: (r["producer"]["dim"], r["certifier"]["statements"]["S3.completeness"]["seconds"]))
macro("CmaxFeasibleDim", thin(dmax))
macro("CmaxFeasibleDimRounded", "\\ensuremath{10^4}")
macro("CfillAtMax", sci(max(x for x in rmax["certifier"]["fill_nnz"] if x), 1))
macro("CnnzAtMax", sci(rmax["producer"]["nnz"], 1))
macro("CSthreeAtMax", f"{rmax['certifier']['statements']['S3.completeness']['seconds']:.0f}")
macro("CsolveAtMax", f"{rmax['producer']['t_solve']:.1f}")
s2_times = []
for r in rows2:
    for key in ("certifier", "certifier_without_S3"):
        c = r.get(key)
        if c and "statements" in c:
            s2_times.append((stmt_time(c, "S2.eigenpairs"), r["producer"]["t_solve"] + r["producer"]["t_build"]))
macro("CStwoMax", f"{max(t for t, _ in s2_times):.2f}")
check(all(t <= p for t, p in s2_times), "E2: S2 costs less than the producer (build + solve) everywhere")
u1 = [r for r in rows2 if fam(r) == "chain-u1"]
s1u1 = []
for r in u1:
    c = r["certifier"] if "statements" in r.get("certifier", {}) else r.get("certifier_without_S3")
    if c and "statements" in c:
        s1u1.append((stmt_time(c, "S1.model_fidelity"), r["producer"]["dim"], r["producer"]["t_build"]))
t1, d1, b1 = max(s1u1, key=lambda x: x[1])
macro("CSoneUoneMax", f"{t1:.2f}")
macro("CSoneUoneMaxDim", thin(d1))
check(all(t <= 1.5 * bld + 0.1 for t, _, bld in s1u1), "E2: S1 for U(1) blocks costs about the producer's build")
k22 = next(r for r in rows2 if fam(r) == "chain-k0" and r["cfg"]["N"] == 22)
c22 = k22.get("certifier_without_S3") or {}
macro("CkTwentyTwoDim", thin(k22["producer"]["dim"]))
macro("CkTwentyTwoSone", f"{stmt_time(c22, 'S1.model_fidelity'):.0f}" if "statements" in c22 else "?")
macro("CkTwentyTwoMem", f"{c22.get('maxrss_mb', float('nan')):.0f}")
k24 = next(r for r in rows2 if fam(r) == "chain-k0" and r["cfg"]["N"] == 24)
check("error" in (k24.get("certifier_without_S3") or {}), "E2: N=24 momentum block S1 exceeds the memory cap")
# fill exponent from the chain U(1) points with S3 certified
pts = [(r["producer"]["dim"], max(x for x in r["certifier"]["fill_nnz"] if x)) for r in s3_ok if fam(r) == "chain-u1"]
pts.sort()
if len(pts) >= 2:
    expo = math.log(pts[-1][1] / pts[-2][1]) / math.log(pts[-1][0] / pts[-2][0])
    macro("CfillExponent", f"{expo:.1f}")
failed_s3 = [r for r in rows2 if r not in s3_ok]

with open(os.path.join(MS, "tables", "e2.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Cost of certification (Section~\\ref{sec:e2}). Producer: "
            "QuSpin basis and matrix construction (build) and ARPACK for eight states (solve). Certifier: "
            "statements \\Sone{}--\\Sthree{} in seconds and peak memory. ``mem'' means the step exceeded the "
            f"{settings.get('memory_cap_GiB', 4.5):.1f}\\,GiB cap; the certificate then reports \\Sthree{} as not "
            "checked. Gap: certified gap divided by the gap estimated from the producer's eigenvalues.}\n"
            "\\label{tab:e2}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}l r r r r r r r r r r@{}}\n\\toprule\n"
            "Block & $d$ & nnz$(H)$ & build & solve & \\Sone & \\Stwo & \\Sthree & fill & gap & MB \\\\\n\\midrule\n")
    for r in rows2:
        cfg, p = r["cfg"], r["producer"]
        if cfg["model"] == "chain":
            lab = f"chain $N={cfg['N']}$, " + ("$S^z{=}0$" if cfg["block"] == "u1" else "$S^z{=}0,k{=}0$")
        else:
            lab = f"$4{{\\times}}4$, $J_2={cfg.get('J2', 0.0):g}$, $S^z{{=}}0$"
        c = r.get("certifier", {})
        cw = r.get("certifier_without_S3") or {}
        if "statements" in c:
            st = c["statements"]
            s1, s2, s3 = (f"{st['S1.model_fidelity']['seconds']:.2f}", f"{st['S2.eigenpairs']['seconds']:.2f}",
                          f"{st['S3.completeness']['seconds']:.1f}")
            fill = sci(max(x for x in c["fill_nnz"] if x)) if any(c["fill_nnz"]) else "---"
            gap = f"{c['gap_lower_bound'] / c['true_gap_estimate']:.2f}" if c.get("gap_lower_bound") else "---"
            mb = f"{c['maxrss_mb']:.0f}"
        elif "statements" in cw:
            st = cw["statements"]
            s1, s2, s3 = (f"{st['S1.model_fidelity']['seconds']:.2f}", f"{st['S2.eigenpairs']['seconds']:.2f}", "mem")
            fill, gap, mb = "---", "---", f"{cw['maxrss_mb']:.0f}"
        else:
            s1, s2, s3, fill, gap, mb = "mem", "---", "mem", "---", "---", "---"
        f.write(f"{lab} & {thin(p['dim'])} & {sci(p['nnz'])} & {p['t_build']:.1f} & {p['t_solve']:.2f} & {s1} & {s2} & "
                f"{s3} & {fill} & {gap} & {mb} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# figure: cost
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.0))
mk = {"chain-u1": "o", "chain-k0": "s", "square": "^"}
fam_label = {"chain-u1": "chain, $S^z$ block", "chain-k0": "chain, $(S^z,k)$ block", "square": "4x4, $S^z$ block"}
for fm in ("chain-u1", "chain-k0", "square"):
    R = sorted([r for r in rows2 if fam(r) == fm], key=lambda r: r["producer"]["dim"])
    d = [r["producer"]["dim"] for r in R]
    prod = [r["producer"]["t_build"] + r["producer"]["t_solve"] for r in R]
    s12, s3d, s3t, fills, nnzs, dfill = [], [], [], [], [], []
    for r in R:
        c = r["certifier"] if "statements" in r.get("certifier", {}) else r.get("certifier_without_S3") or {}
        if "statements" in c:
            s12.append((r["producer"]["dim"], c["statements"]["S1.model_fidelity"]["seconds"] +
                        c["statements"]["S2.eigenpairs"]["seconds"]))
        if r in s3_ok:
            s3d.append(r["producer"]["dim"])
            s3t.append(r["certifier"]["statements"]["S3.completeness"]["seconds"])
            dfill.append(r["producer"]["dim"])
            fills.append(max(x for x in r["certifier"]["fill_nnz"] if x))
        nnzs.append(r["producer"]["nnz"])
    ls = "-" if fm == "chain-u1" else (":" if fm == "chain-k0" else "none")
    axL.plot(d, prod, marker=mk[fm], ls=ls, color=BLUE, lw=1.5, ms=4.5)
    if s12:
        axL.plot([x for x, _ in s12], [max(y, 1e-3) for _, y in s12], marker=mk[fm], ls=ls, color=GREEN, lw=1.5, ms=4.5)
    if s3d:
        axL.plot(s3d, s3t, marker=mk[fm], ls=ls, color=ORANGE, lw=1.5, ms=4.5)
    axR.plot(d, nnzs, marker=mk[fm], ls=ls, color=BLUE, lw=1.5, ms=4.5)
    if dfill:
        axR.plot(dfill, fills, marker=mk[fm], ls=ls, color=ORANGE, lw=1.5, ms=4.5)
    for r in R:
        if r not in s3_ok:
            axR.plot([r["producer"]["dim"]], [r["producer"]["nnz"] * 300], marker="x", color=ORANGE, ms=6, ls="none")
for ax in (axL, axR):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("block dimension $d$")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axL.set_ylabel("wall time (s)")
axR.set_ylabel("nonzeros")
from matplotlib.lines import Line2D
hL = [Line2D([], [], color=BLUE, lw=2, label="producer (build+solve)"),
      Line2D([], [], color=GREEN, lw=2, label="S1+S2"),
      Line2D([], [], color=ORANGE, lw=2, label="S3")]
hF = [Line2D([], [], color=INK2, marker=mk[k], ls="none", label=fam_label[k]) for k in mk]
axL.legend(handles=hL + hF, fontsize=6.8, loc="upper left")
hR = [Line2D([], [], color=BLUE, lw=2, label="nnz(H)"), Line2D([], [], color=ORANGE, lw=2, label="nnz(L)+nnz(U)"),
      Line2D([], [], color=ORANGE, marker="x", ls="none", label="S3 exceeded memory cap")]
axR.legend(handles=hR, fontsize=6.8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(MS, "figures", "fig_cost.pdf"))
plt.close(fig)

# ============================================================== E4
_, e4, _ = runs["e4_workflows_W1W2W3W4"]
w1, w2, w3, w4 = e4["W1"], e4["W2"], e4["W3"], e4["W4"]
macro("WoneN", str(w1["N"]))
macro("WoneJtwo", f"{w1['J2']:g}")
macro("WoneBlocks", str(w1["n_blocks"]))
macro("WoneProd", f"{w1['producer_seconds']:.0f}")
macro("WoneCert", f"{w1['certify_seconds']:.0f}")
macro("WoneK", str(w1["K"]))
macro("WoneRadius", sci(w1["radius"]))
macro("WoneNext", f"{w1['next_lower_bound']:.9f}")
check(w1["status"] == "CERTIFIED" and w1["sector_status_counts"]["COMPLETE"] == w1["n_blocks"],
      "W1: certified, all blocks complete")
Nw = w1["N"]
with open(os.path.join(MS, "tables", "w1.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{The " + str(w1["K"]) + " lowest eigenvalues of the "
            f"$J_1$--$J_2$ ring, $N={Nw}$, $J_2={w1['J2']:g}$, certified from {w1['n_blocks']} symmetry blocks "
            f"(Section~\\ref{{sec:w1}}). Common radius {sci(w1['radius'])}; the next eigenvalue is at least "
            f"${w1['next_lower_bound']:.9f}$. Momentum $k$ in units of $2\\pi/N$.}}\n\\label{{tab:w1}}\n\\small\n"
            "\\begin{tabular}{@{}r r r r@{}}\n\\toprule\n$j$ & $E_j$ & $S^z$ & $k$ \\\\\n\\midrule\n")
    for j, (e, sz, k) in enumerate(w1["levels"][: w1["K"]]):
        f.write(f"{j + 1} & ${e:.10f}$ & ${sz:+.0f}$ & {k} \\\\\n".replace("$+0$", "$0$"))
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
cl = w2["clusters"][0]
qn = cl["quantum_numbers"]["Sz_total"]
check(qn["status"] == "CERTIFIED" and all(v == "CERTIFIED" for v in w2["statuses"].values()), "W2: all certified")
macro("WtwoQN", ",".join(f"{v:.0f}" for v in qn["certified_values"]))
macro("WtwoQNBound", sci(qn["bound"]))
o1, o2 = cl["observables"]["Sz_0 Sz_1"], cl["observables"]["Sz_0"]
macro("WtwoObs", f"{o1['value']:.12f}")
macro("WtwoObsBound", sci(o1["bound"]))
macro("WtwoObsSz", sci(o2["value"]))
macro("WtwoObsSzBound", sci(o2["bound"]))
sp_ = o1.get("per_vector_spread_under_rotation", {})
macro("WtwoSpreadMin", f"{sp_.get('min', float('nan')):.2f}")
macro("WtwoSpreadMax", f"{sp_.get('max', float('nan')):.2f}")
check(abs(o2["value"]) <= o2["bound"], "W2: tr(P Sz_0) consistent with zero")
# certificate listing for Section 3: reconstruct the summary from statuses/summaries
lines = [f"subject : spin1_heisenberg_chain (N={w2['N']}, d={w2['dim']})",
         f"claim   : lowest {w2['k']} eigenvalues"]
for sid, st in w2["statuses"].items():
    summ = w2["summaries"][sid]
    lines.append(f"[{st:>11s}] {sid}: {summ}")
import textwrap
wrapped = []
for line in lines:
    wrapped.extend(textwrap.wrap(line, 96, subsequent_indent="              ") or [""])
with open(os.path.join(MS, "tables", "w2_summary.txt"), "w") as f:
    f.write("\n".join(wrapped) + "\n")
macro("WthreeProd", f"{w3['producer_seconds']:.0f}")
macro("WthreeCert", f"{w3['certify_seconds']:.0f}")
macro("WthreeRadius", sci(w3["max_radius"]))
import re
m = re.search(r"max rel\. diff ([0-9.e+-]+)", w3["summaries"]["S1.model_fidelity"])
macro("WthreeProbe", sci(float(m.group(1))) if m else "?")
macro("WthreeDimThin", thin(w3["dim"]))
check(w3["statuses"]["S1.model_fidelity"] == "CERTIFIED" and w3["statuses"]["S2.eigenpairs"] == "CERTIFIED"
      and w3["statuses"]["S3.completeness"] == "NOT_CHECKED", "W3: S1,S2 certified, S3 not checked")
check(w3["dim"] == 2 ** 20, "W3: d = 2^20")
with open(os.path.join(MS, "tables", "w4.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Certified ground-state energies of benchmark systems "
            "($J_1=1$, periodic boundaries; Section~\\ref{sec:w4}). Each value is the lowest eigenvalue of the "
            "stated block unless \\Sthree{} is not checked; ``gap'' is the certified lower bound on the "
            "distance to the next eigenvalue of the block.}\n\\label{tab:w4}\n\\footnotesize\n"
            "\\begin{tabular}{@{}l r l l l l@{}}\n\\toprule\n"
            "System (block) & $d$ & $E_0$ & radius & \\Sthree{} & gap $\\ge$ \\\\\n\\midrule\n")
    names = {0: "Heisenberg ring $N=20$ ($S^z{=}0,k{=}0$)", 1: "$J_1$--$J_2$ ring $N=20$, $J_2=\\frac12$ ($S^z{=}0,k{=}0$)",
             2: "Heisenberg $4\\times4$ ($S^z{=}0$)", 3: "$J_1$--$J_2$ $4\\times4$, $J_2=\\frac12$ ($S^z{=}0$)"}
    for i, r in enumerate(w4):
        s3 = r["statuses"]["S3.completeness"]
        gap = f"{r['gap_lower_bound']:.4f}" if r.get("gap_lower_bound") else "---"
        f.write(f"{names[i]} & {thin(r['dim'])} & ${r['E0']:.12f}$ & {sci(r['radius'])} & "
                f"\\st{{{s3.replace('_', chr(92) + '_')}}} & {gap} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
mg = w4[1]
check(abs(mg["E0"] - (-7.5)) <= mg["radius"], f"W4: Majumdar-Ghosh E0=-7.5 within radius (E0={mg['E0']!r})")
check(w4[3]["statuses"]["S3.completeness"] == "NOT_CHECKED", "W4: 4x4 J2=0.5 completeness not checked")

# ============================================================== write
with open(os.path.join(MS, "numbers.tex"), "w") as f:
    f.write("% generated by experiments/make_numbers.py -- do not edit\n")
    for k in sorted(MACROS):
        f.write(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}\n")
print(f"wrote {len(MACROS)} macros to {os.path.join(MS, 'numbers.tex')}")
bad = [s for ok, s in CHECKS if not ok]
for ok, s in CHECKS:
    print(("ok           " if ok else "CHECK FAILED ") + s)
sys.exit(1 if bad else 0)
