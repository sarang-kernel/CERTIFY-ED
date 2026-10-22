"""Generate every number, table and figure of the manuscript from the experiment records.

Usage: python3 make_numbers.py [MANUSCRIPT_DIR]   (default: <results dir>/paper; `certify-ed numbers` and run.py pass it)

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
sys.path.insert(0, ROOT)
MS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("CERTIFY_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), "paper"))
RES = os.environ.get("CERTIFY_RESULTS", os.path.join(HERE, "results"))
os.makedirs(os.path.join(MS, "tables"), exist_ok=True)
os.makedirs(os.path.join(MS, "figures"), exist_ok=True)

MACROS = {}
NOTES = {}      # macro name -> provenance note, written as a LaTeX comment after the definition
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


def macro(name, value, note=None):
    """Define \\name as value. `note` names the record or field the value comes from; it is written as a
    LaTeX comment after the definition (v4: every macro of the manuscript carries its source)."""
    MACROS[name] = str(value)
    if note:
        NOTES[name] = " ".join(str(note).split())


def check(cond, statement):
    CHECKS.append((bool(cond), statement))


def code_lines(path):
    """Lines of a Python file that carry code: not blank, not a comment, not part of a docstring (a
    docstring is the string-constant first statement of a module, class or function). Counted from the
    token stream, so a statement or string continuing over several lines counts every line it occupies
    (MS-N11)."""
    import ast
    import io
    import tokenize
    with open(path, encoding="utf-8") as f:
        src = f.read()
    doc = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                doc.update(range(first.lineno, first.end_lineno + 1))
    skip = {tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
            tokenize.ENDMARKER}
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in skip:
            continue
        lines.update(ln for ln in range(tok.start[0], tok.end[0] + 1) if ln not in doc)
    return len(lines)


def parse_collected(stdout):
    """Number of tests in the output of `pytest --collect-only -q` ('510 tests collected in 14s',
    '3/5 tests collected (2 deselected)', '1 test collected'); None if the line is missing."""
    import re
    m = re.search(r"(\d+)(?:/\d+)? tests? collected", stdout or "")
    return int(m.group(1)) if m else None


def duplicate_macros(tex_files):
    """Macro names defined by \\newcommand in more than one of the files (LaTeX stops on the second
    definition); \\providecommand does not count, that is how a file defines a fallback."""
    import re
    where = {}
    for path in tex_files:
        with open(path, encoding="utf-8") as f:
            for name in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read()):
                where.setdefault(name, []).append(os.path.basename(path))
    return {k: v for k, v in sorted(where.items()) if len(v) > 1}


def flagged_by_either(row):
    """A result flagged by P1 or P2 (review MS-H02); None when neither practice applies (F12: P2 n/a)."""
    p1, p2 = row["P1_residual_only"], row["P2_recomputation"]
    if p1 is None and p2 is None:
        return None
    if p2 is None and p1:
        return None
    return (p1 is False) or (p2 is False)


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


def dec_dir(x, d, up):
    """x with d decimals, rounded toward +inf (up=True) or -inf, computed exactly: for printing bounds."""
    from fractions import Fraction as _F
    q = _F(x) * 10 ** d
    n = math.ceil(q) if up else math.floor(q)
    sign = "-" if n < 0 else ""
    n = abs(n)
    return f"{sign}{n // 10 ** d}.{n % 10 ** d:0{d}d}" if d > 0 else f"{sign}{n}"


def sci_dir(x, digits=1, up=True):
    """Like sci(), with the mantissa rounded toward +inf (up) or -inf, exactly: for printing bounds."""
    from fractions import Fraction as _F
    if x is None or not np.isfinite(x):
        return "\\ensuremath{\\infty}"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    for _ in range(2):
        q = _F(x) / _F(10) ** e * 10 ** digits
        n = math.ceil(q) if up else math.floor(q)
        if abs(n) >= 10 ** (digits + 1):
            e += 1
            continue
        break
    sign = "-" if n < 0 else ""
    n = abs(n)
    ms = f"{sign}{n // 10 ** digits}.{n % 10 ** digits:0{digits}d}" if digits > 0 else f"{sign}{n}"
    return ms if e == 0 else f"\\ensuremath{{{ms}\\times10^{{{e}}}}}"


def thin(n):
    """Integer with thin-space thousands separators (math-safe)."""
    s = f"{int(n):,}".replace(",", "\\,")
    return s


def tex_escape(s):
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")
            .replace("%", "\\%").replace("#", "\\#"))


# ============================================================== environment / provenance
runs = {}
for name in ("e0_verifier_validation", "e1_failure_matrix", "e2_cost", "e3_tightness", "e4_workflows_W1W2W3W4",
             "e5_arpack_multiplet"):
    runs[name] = latest(name)
commits = {r[2].get("git_commit") for r in runs.values()}
env0 = runs["e0_verifier_validation"][2]


def _same_source(c1, c2):
    """No computational difference between two commits in certify_ed/ and experiments/ (the command line
    and test helpers excluded: they cannot change a number)."""
    from certify_ed.provenance import NON_COMPUTATIONAL
    excl = [f":(exclude){f}" for f in NON_COMPUTATIONAL]
    p = subprocess.run(["git", "diff", "--quiet", c1, c2, "--", "certify_ed", "experiments", *excl], cwd=ROOT)
    return p.returncode == 0


def _tree(commit, path="certify_ed"):
    out = subprocess.run(["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or None


_c = sorted(c for c in commits if c)
trees = {_tree(c) for c in _c}
_hashes = {r[2].get("source_hash") for r in runs.values() if r[2].get("source_hash")}
_known = all(subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=ROOT, capture_output=True).returncode == 0
             for c in _c)
if _known:
    check(bool(_c) and len(commits) == len(_c) and all(_same_source(_c[0], c) for c in _c[1:])
          and len(_hashes) <= 1,
          f"all experiments ran with the same computational source (commits {[c[:7] for c in _c]})")
else:
    # the recorded commits are not in this repository (rewritten history, or a checkout of an archive): the
    # computational source hash of the records is the evidence (certify_ed.provenance)
    check(bool(_c) and len(commits) == len(_c) and len(_hashes) == 1 and all(r[2].get("source_hash") for r in runs.values()),
          f"all experiments ran with the same computational source (source_hash {sorted(_hashes)}; recorded commits "
          f"{[c[:7] for c in _c]} are not in this repository)")
check(not any(r[2].get("source_dirty") for r in runs.values()),
      "no experiment ran with uncommitted changes to certify_ed/ or experiments/")
macro("GitCommit", ", ".join(sorted(c[:7] for c in commits if c)))
macro("GitTree", (sorted(t for t in trees if t) or ["unknown"])[0][:10])
_dates = {k: r[2]["started"][:10] for k, r in runs.items()}
_main_date = max(set(_dates.values()), key=list(_dates.values()).count)
_other = sorted((d, f"E{k[1]}") for k, d in _dates.items() if d != _main_date)
macro("RunDate", _main_date + (" (" + "; ".join(f"{e} on {d}" for d, e in _other) + ")" if _other else ""))
cpu = env0.get("cpu_model")
if not cpu:                          # records written before the platform layer: probe this machine
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import portable
    cpu = portable.cpu_model() or "x86-64 CPU"
cpu = cpu.replace("(R)", "").replace("(TM)", "").replace("  ", " ")
macro("MachineCPU", tex_escape(cpu))
mem = env0.get("memory_GiB")
if mem is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import portable
    mem = portable.memory_gib()
macro("MachineRAM", f"{mem:.1f}")
macro("PyVersion", env0["python"].split()[0])
macro("NumpyVersion", env0["numpy"])
macro("ScipyVersion", env0["scipy"])
macro("QuspinVersion", env0.get("quspin", "?"))
macro("PythonFlintVersion", str(env0.get("flint", "?")))
macro("FlintCVersion", str(env0.get("flint_c_library", "unknown")))     # MS-N12: the FLINT C library
# MS-N12: the name the v4 text uses for the same number; 'unknown' when the harness did not record it (runs before 3.1.0)
macro("FlintLibVersion", str(env0.get("flint_c_library", "unknown")),
      note="E0 environment.json flint_c_library (FLINT C library behind python-flint); 'unknown' if not recorded")
macro("CertifyEdVersion", str(env0.get("certify_ed", "unknown")))
QUEUE_A = ("e2_cost", "e4_workflows_W1W2W3W4", "e5_arpack_multiplet")
el_B = sum(runs[k][1].get("_elapsed_seconds", 0) for k in runs if k not in QUEUE_A)
el_A = sum(runs[k][1].get("_elapsed_seconds", 0) for k in QUEUE_A)
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
# MS-N11: size of the exact layer, code lines only (no blanks, comments or docstrings), counted from the
# token stream of the repository that regenerates the numbers (a code fact, not a run record)
_exact_files = ("checker.py", "exact_sectors.py")
_exact_lines = {f: code_lines(os.path.join(ROOT, "certify_ed", f)) for f in _exact_files}
macro("ExactCheckerLines", thin(sum(_exact_lines.values())),
      note="code lines of certify_ed/checker.py + exact_sectors.py (" + " + ".join(f"{f} {n}" for f, n in _exact_lines.items())
      + "), counted by code_lines() in this repository checkout")
macro("ExactCheckerLinesChecker", thin(_exact_lines["checker.py"]), note="code lines of certify_ed/checker.py")
macro("ExactCheckerLinesSectors", thin(_exact_lines["exact_sectors.py"]), note="code lines of certify_ed/exact_sectors.py")


def run_test_counts():
    """Test counts of the RUN (manifest steps.tests.summary, e.g. '363 passed, 2 skipped, 1 warning in 79s'),
    not of a live collection on the regeneration host (IMP-07 / MS-N02). None when RES is not a run directory."""
    import re
    man = os.path.join(RES, "manifest.json")
    if not os.path.exists(man):
        return None
    with open(man) as f:
        summary = json.load(f).get("steps", {}).get("tests", {}).get("summary", "")
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)", summary)}
    if not counts:
        return None
    return {"summary": summary, "passed": counts.get("passed", 0), "failed": counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0),
            "skipped": counts.get("skipped", 0), "total": sum(counts.values())}


def run_tests_hash():
    """tests_hash the run recorded (manifest source.tests_hash, else the E0 environment); None if absent."""
    man = os.path.join(RES, "manifest.json")
    if os.path.exists(man):
        with open(man) as f:
            h = (json.load(f).get("source") or {}).get("tests_hash")
        if h:
            return h
    return env0.get("tests_hash")


def repo_test_count():
    """Tests the repository's suite contains now: `pytest --collect-only -q` in ROOT (a live collection; a
    module skipped at import, e.g. tests/test_label_audit.py without QuSpin, is not counted, so the number
    can be smaller than on the run host). None if pytest is unavailable or the summary line is missing."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
                           cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_collected(p.stdout)


_tc = run_test_counts()
if _tc is None:
    for k in ("NtestsTotal", "NtestsVtwo", "NtestsPassed", "NtestsSkipped", "NtestsRun", "NtestsAddedAfterRun"):
        macro(k, "?")
    check(False, "tests: the run directory records the test-suite outcome (manifest steps.tests)")
else:
    macro("NtestsTotal", str(_tc["total"]))           # tests the run executed (passed + failed + skipped)
    macro("NtestsVtwo", str(_tc["total"]))
    macro("NtestsPassed", str(_tc["passed"]))
    macro("NtestsSkipped", str(_tc["skipped"]))
    macro("NtestsRun", str(_tc["passed"]), note="manifest steps.tests.summary: tests passed in the evidence run (MS-N02 / RA-01)")
    check(_tc["failed"] == 0, f"tests: the run's test suite passed ({_tc['summary'].split(' in ')[0]})")
    # MS-N02: tests added to the repository after the evidence run. Zero by identity when the run's tests_hash
    # is the repository's (no collection needed); otherwise the repository's suite is collected live and the
    # run's count subtracted, which depends on what the regeneration host can import (see repo_test_count)
    from certify_ed.provenance import source_fingerprint
    _fp_now = source_fingerprint(ROOT)
    _h_run = run_tests_hash()
    if _h_run and _h_run == _fp_now.get("tests_hash") and not _fp_now.get("tests_dirty"):
        macro("NtestsAddedAfterRun", "0", note=f"the run's tests_hash {_h_run} is the repository's: no test was added after the run")
    else:
        _n_now = repo_test_count()
        if _n_now is None:
            macro("NtestsAddedAfterRun", "?", note="the repository's suite could not be collected on this host")
            check(False, "tests: the repository's test suite can be collected to count the tests added after the run")
        else:
            macro("NtestsAddedAfterRun", str(_n_now - _tc["total"]),
                  note=f"pytest --collect-only on this checkout ({_n_now} tests; tests_hash {_fp_now.get('tests_hash')}) minus the "
                       f"run's {_tc['total']} (tests_hash {_h_run}); a live collection on the regeneration host, modules "
                       f"skipped at import are not counted")

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
# MS-N13: 'matrices' counts the cases with an Arb reference; the ones without a reference are excluded from the pairs
macro("VaMatricesGenerated", str(a["matrices"] + a.get("reference_unavailable", 0)))
macro("VaBallW", sci(a.get("arb_ball_max_width", float("nan"))))
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
dd = e0["d_jw_convention_episode"]
macro("VdModels", str(dd["models"]))
macro("VdDisagree", str(dd["disagree_with_quspin"]))
macro("VdConserving", str(dd["number_conserving_models"]))
check(dd["disagree_with_quspin"] == dd["disagreeing_with_number_changing_string"]
      and dd["number_conserving_agree"] == dd["number_conserving_models"],
      "E0(d): with the old JW ordering only models with number-changing strings disagree with QuSpin")
with open(os.path.join(MS, "tables", "e0.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Validation of the verifier against independent "
            "references (Section~\\ref{sec:e0}).}\n\\label{tab:e0}\n\\footnotesize\n"
            "\\begin{tabular}{@{}l l r r@{}}\n\\toprule\n"
            "Component & Reference & Cases & Violations \\\\\n\\midrule\n")
    f.write(f"\\Stwo{{}} enclosures (Theorems~\\ref{{thm:residual}}--\\ref{{thm:cluster}}) & Arb eigenvalues (ball "
            f"arithmetic) & {thin(a['eigenpairs'])} pairs & {viol_a} \\\\\n")
    f.write(f"\\Stwo{{}} residual bound & Arb residual (ball arithmetic) & {thin(a['eigenpairs'])} pairs & "
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
# MS-H02: the two standard practices together (what Referee 1 proposed) and the errors that escape both
_either = [r for r in err_rows if flagged_by_either(r)]
_both_missed = [r for r in err_rows if not flagged_by_either(r)]
macro("FnPoneOrPtwo", str(len(_either)), note="E1 data.json rows: wrong results with P1_residual_only false or P2_recomputation false")
macro("FnBothMissed", str(len(_both_missed)), note="E1 data.json rows: wrong results flagged by neither P1 nor P2")
macro("FnBothMissedList", ", ".join(r["case"].split()[0] for r in _both_missed), note="E1 data.json: the cases missed by both practices")
check(len(_either) + len(_both_missed) == n_err, "E1: every wrong result is either flagged by P1 or P2 or missed by both")
check(len(_either) == len({r["case"] for r in err_rows if r["P1_residual_only"] is False} | {r["case"] for r in err_rows if r["P2_recomputation"] is False}),
      "E1: FnPoneOrPtwo is the size of the union of the P1 and P2 flag sets")
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
macro("FnFtwelveBound", sci_dir(f12["bound"]) if f12.get("bound") else "?")
macro("FnFtwelveSpread", f"{f12['spread']:.3f}" if f12.get("spread") else "?")
_words = {2: "twofold", 3: "threefold", 4: "fourfold"}
macro("FnFtwelveMult", _words.get(f12.get("multiplicity"), f"{f12.get('multiplicity')}-fold"))

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
    "F12": ("$\\langle O\\rangle$ quoted for one vector of a " + _words.get(f12.get("multiplicity"), "degenerate")
            + " level", None),
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
        return f"radius {sci_dir(float(m.group(1)))} $>$ target" if m else ""
    if k == "F9":
        import re
        m = re.search(r"radius ([0-9.e+-]+)", r["P3_detail"])
        return f"radius {sci_dir(float(m.group(1)))}" if m else ""
    if k in ("F10a", "F10b"):
        return "sector audit fails"
    if k in ("F11a", "F11b"):
        import re
        m = re.search(r"cover (\d+) of (\d+)", r["P3_detail"])
        return f"sectors cover {m.group(1)} of {m.group(2)}" if m else ""
    if k == "F12":
        return f"$\\tr(PO)\\pm${sci_dir(f12['bound'])}; spread {f12['spread']:.3f}"
    return ""


def pf(v):
    return "n/a" if v is None else ("pass" if v else "\\textbf{flag}")


with open(os.path.join(MS, "tables", "e1.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Fifteen results checked by three practices "
            "(Section~\\ref{sec:e1}). P1: residual check with the producer's matrix; P2: recomputation with "
            "QuSpin from the intended model; P3: \\pkg{}. ``P1 or P2'': flagged by at least one of the two "
            "standard practices (review MS-H02). ``flag'' means the practice reports a problem; "
            "``pass'' means it does not. Thirteen results (F1--F11b) are wrong; F0 is correct; F12 is a "
            "correct result whose single-vector observable is basis dependent.}\n\\label{tab:e1}\n"
            "\\footnotesize\n\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{0.33\\textwidth} c c c l >{\\raggedright\\arraybackslash}p{0.25\\textwidth}@{}}\n\\toprule\n"
            "Case & Mechanism & P1 & P2 & P1 or P2 & P3 & P3 diagnosis \\\\\n\\midrule\n")
    for r in rows:
        k = r["case"].split()[0]
        _e = flagged_by_either(r)
        f.write(f"{k} & {short[k][0]} & {pf(r['P1_residual_only'])} & {pf(r['P2_recomputation'])} & "
                f"{'n/a' if _e is None else pf(not _e)} & {p3_cell(r)} & {diag_cell(r)} \\\\\n")
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
macro("TbN", str(len(bb)))                                   # MS-N13: the ratios come from n pairs
macro("TbRatioMin", dec_dir(min(bb), 1, False))              # range rounded outward
macro("TbRatioMax", dec_dir(max(bb), 1, True))
check(all(float(dec_dir(min(bb), 1, False)) <= x <= float(dec_dir(max(bb), 1, True)) for x in bb),
      "E3(b): the printed ratio range contains every ratio")
# MS-N13: the names the v4 text uses for the outward-rounded range and the number of pairs behind it
macro("TbRatioMinDown", dec_dir(min(bb), 1, False), note="E3 b_float_vs_arb.ratio_bound_over_arb minimum, rounded down")
macro("TbRatioMaxUp", dec_dir(max(bb), 1, True), note="E3 b_float_vs_arb.ratio_bound_over_arb maximum, rounded up")
macro("TbPairs", str(len(bb)), note="E3 b_float_vs_arb: number of eigenpairs compared with Arb")
cg = e3["c_gap"]
uniq = {(r["model"], r["k"]): r for r in cg}
macro("TcCases", str(len(uniq)))
macro("TcCasesRecorded", str(len(cg)))
macro("TcCasesUnique", str(len(uniq)), note="E3 c_gap: distinct (model, k) cases; the record may hold a duplicated row (MS-N07)")
# MS-N07: 0.9 is the fraction of the estimated gap at which the first shift is placed (a parameter of the
# probe order, Options.gap_fractions[0]), not a tightness result; the macro says so
_fracs = sorted({round(r["fraction_of_gap_certified"], 6) for r in uniq.values()})
macro("TcShiftFraction", f"{_fracs[0]:g}" if len(_fracs) == 1 else ", ".join(f"{x:g}" for x in _fracs))
check(len(_fracs) == 1 and all(r["n_factorizations"] == 1 for r in uniq.values()),
      f"E3(c): the shift at {_fracs[0]:g} of the estimated gap was certified with one factorization in all cases")
dc = e3["d_compose"]
macro("TdSectors", str(dc["n_sectors"]))
macro("TdRadius", sci_dir(dc["radius"]))
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


def s_status(c, sid):
    return c["statements"][sid]["status"] if c and "statements" in c else None


def s1_ok(c):
    """S1 ran to completion (the reference fitted in memory)."""
    return c is not None and "statements" in c and s_status(c, "S1.model_fidelity") != "NOT_CHECKED"


def table_cert(r):
    """The certificate whose S1/S2 times Table e2 prints for row r: the first run when it has statements,
    else the run without S3. Text macros take their numbers from the same source (MS-N04 / MS-N05)."""
    c = r.get("certifier") or {}
    return c if "statements" in c else (r.get("certifier_without_S3") or {})


def table_mem_cert(r):
    """The certificate whose peak memory Table e2 prints: the run without S3 when S3 did not fit."""
    c = r.get("certifier") or {}
    cw = r.get("certifier_without_S3") or {}
    if "statements" in c:
        return cw if (s_status(c, "S3.completeness") == "NOT_CHECKED" and "statements" in cw) else c
    return cw


ok_rows = [r for r in rows2 if "error" not in r.get("certifier", {"error": 1})]
check(len(ok_rows) == len(rows2), "E2: every certifier run produced a certificate (no crash)")
s3_ok = [r for r in ok_rows if r["certifier"]["statements"]["S3.completeness"]["status"] == "CERTIFIED"]
dmax = max(r["producer"]["dim"] for r in s3_ok)
rmax = max(s3_ok, key=lambda r: (r["producer"]["dim"], r["certifier"]["statements"]["S3.completeness"]["seconds"]))
macro("CmaxFeasibleDim", thin(dmax))
macro("CmaxFeasibleDimRounded", "\\ensuremath{10^4}")
macro("CfillAtMax", sci(max(x for x in rmax["certifier"]["fill_nnz"] if x), 1))
macro("CnnzAtMax", sci(rmax["producer"]["nnz"], 1))
# MS-N06: S3 feasibility is set by nnz and fill, not by d: name the blocks of dimension <= dmax whose S3 did not fit
_s3_notfit = [r for r in ok_rows if r["certifier"]["statements"]["S3.completeness"]["status"] == "NOT_CHECKED"
              and r["producer"]["dim"] <= dmax]
macro("CnotFitAtOrBelowMaxDim", str(len(_s3_notfit)))
macro("CnotFitNnzMin", sci(min(r["producer"]["nnz"] for r in _s3_notfit), 1) if _s3_notfit else "---")
macro("CnotFitDim", ", ".join(thin(r["producer"]["dim"]) for r in _s3_notfit) if _s3_notfit else "---")
check(all(r["producer"]["nnz"] > rmax["producer"]["nnz"] for r in _s3_notfit),
      "E2: every block of dimension <= CmaxFeasibleDim whose S3 did not fit has more nonzeros than the largest certified block")
macro("CSthreeAtMax", f"{rmax['certifier']['statements']['S3.completeness']['seconds']:.0f}")
macro("CsolveAtMax", f"{rmax['producer']['t_solve']:.1f}")
s2_times = []
for r in rows2:
    for key in ("certifier", "certifier_without_S3"):
        c = r.get(key)
        if c and "statements" in c:
            s2_times.append((stmt_time(c, "S2.eigenpairs"), r["producer"]["t_solve"] + r["producer"]["t_build"]))
# the text macro is the largest S2 cell of the table (same source as the table, MS-N05); the check covers both runs
_s2_table = [stmt_time(table_cert(r), "S2.eigenpairs") for r in rows2 if "statements" in table_cert(r)]
macro("CStwoMax", f"{max(_s2_table):.2f}")
check(all(t <= p for t, p in s2_times), "E2: S2 costs less than the producer (build + solve) everywhere")
u1 = [r for r in rows2 if r["cfg"]["block"] == "u1"]          # every S^z block of the table: chains and 4x4 squares
s1u1 = []
for r in u1:
    c = table_cert(r)
    if s1_ok(c):
        s1u1.append((stmt_time(c, "S1.model_fidelity"), r["producer"]["dim"], r["producer"]["t_build"]))
# MS-N04: 'at most' is the maximum over all S^z rows of the table (including the 4x4 blocks), not the time of
# the largest block; both are printed
t1max = max(t for t, _, _ in s1u1)
t1, d1, b1 = max(s1u1, key=lambda x: x[1])
macro("CSoneUoneMax", f"{t1max:.2f}")
macro("CSoneUoneMaxDim", thin(d1))
macro("CSoneUoneAtMaxDim", f"{t1:.2f}")
macro("CSoneSzMaxTable", f"{t1max:.2f}",
      note="E2 data.json: largest S1.model_fidelity seconds over the S^z rows of Table e2, from the certificate the table prints (MS-N04)")
# MS-N06: the 4x4 J2 block of the largest certified dimension whose S3 exceeded the memory cap
_sq_j2_notfit = [r for r in ok_rows if fam(r) == "square" and r["cfg"].get("J2")
                 and r["certifier"]["statements"]["S3.completeness"]["status"] == "NOT_CHECKED"]
if _sq_j2_notfit:
    _sqj = max(_sq_j2_notfit, key=lambda r: r["producer"]["dim"])
    macro("CnnzSquareJtwoNotChecked", sci(_sqj["producer"]["nnz"], 1),
          note=f"E2 data.json producer.nnz of the 4x4 J2={_sqj['cfg']['J2']:g} S^z=0 block (d={_sqj['producer']['dim']}) with S3 NOT_CHECKED")
    check(_sqj["producer"]["dim"] == dmax,
          "E2: the 4x4 J2 block whose S3 did not fit has the dimension of the largest block with S3 certified (CmaxFeasibleDim)")
else:
    macro("CnnzSquareJtwoNotChecked", "---", note="E2: no 4x4 J2 block with S3 NOT_CHECKED in this run")
check(all(t <= float(f"{t1max:.2f}") + 5e-3 for t, _, _ in s1u1), "E2: CSoneUoneMax is the largest S1 time of the S^z rows")
check(all(t <= 1.5 * bld + 0.1 for t, _, bld in s1u1), "E2: S1 for U(1) blocks costs about the producer's build")
k22 = next(r for r in rows2 if fam(r) == "chain-k0" and r["cfg"]["N"] == 22)
c22 = table_cert(k22)
macro("CkTwentyTwoDim", thin(k22["producer"]["dim"]))
macro("CkTwentyTwoSone", f"{stmt_time(c22, 'S1.model_fidelity'):.0f}" if "statements" in c22 else "?")
macro("CkTwentyTwoSoneTable", f"{stmt_time(c22, 'S1.model_fidelity'):.1f}" if "statements" in c22 else "?",
      note="E2 data.json: S1.model_fidelity seconds of the chain N=22 (S^z=0, k=0) row, from the certificate Table e2 prints (MS-N05)")
macro("CkTwentyTwoMem", f"{table_mem_cert(k22).get('maxrss_mb', float('nan')):.0f}")
_s1_rows = [(stmt_time(c, "S1.model_fidelity"), r) for r in rows2 for c in (r.get("certifier"), r.get("certifier_without_S3"))
            if s1_ok(c)]
check(max(_s1_rows, key=lambda x: x[0])[1] is k22, "E2: the N=22 momentum block has the largest S1 time")
k24 = next(r for r in rows2 if fam(r) == "chain-k0" and r["cfg"]["N"] == 24)
check(s_status(k24["certifier"], "S1.model_fidelity") == "NOT_CHECKED"
      and "memory" in k24["certifier"]["statements"]["S1.model_fidelity"]["summary"],
      "E2: N=24 momentum block: the S1 reference exceeds the memory cap and S1 is NOT_CHECKED")
s3_mem = [r for r in rows2 if s_status(r["certifier"], "S3.completeness") == "NOT_CHECKED"]
check(all("could not be completed" in r["certifier"]["statements"]["S3.completeness"]["summary"] for r in s3_mem),
      "E2: every S3 NOT_CHECKED is a factorization that did not fit")
if s3_mem:
    worst_fail = max(s3_mem, key=lambda r: r["certifier"]["statements"]["S3.completeness"]["seconds"])
    macro("CfailSeconds", f"{worst_fail['certifier']['statements']['S3.completeness']['seconds']:.0f}")
    macro("CfailDim", thin(worst_fail["producer"]["dim"]))
sq = next(r for r in ok_rows if fam(r) == "square" and not r["cfg"].get("J2"))
ch16 = next(r for r in s3_ok if fam(r) == "chain-u1" and r["cfg"]["N"] == 16)


def per_fact(r):
    c = r["certifier"]
    n = max(1, len([x for x in c["fill_nnz"] if x]))
    return c["statements"]["S3.completeness"]["seconds"] / n, n


macro("CfillSquare", sci(max(x for x in sq["certifier"]["fill_nnz"] if x)))
macro("CnnzSquare", sci(sq["producer"]["nnz"]))
macro("CfactSquare", f"{per_fact(sq)[0]:.0f}")
macro("CsolveSquare", f"{sq['producer']['t_solve']:.1f}")
macro("CfactChain", f"{per_fact(ch16)[0]:.0f}")
sq_status = sq["certifier"]["statements"]["S3.completeness"]["status"]
macro("CSthreeSquareStatus", "\\st{" + sq_status.replace("_", "\\_") + "}")
macro("CSthreeSquareNfact", str(per_fact(sq)[1]))
ch14 = next(r for r in s3_ok if fam(r) == "chain-u1" and r["cfg"]["N"] == 14)
growth = max(x for x in ch16["certifier"]["fill_nnz"] if x) / max(x for x in ch14["certifier"]["fill_nnz"] if x)
macro("CfillGrowth", f"{growth:.0f}")
# fill exponent from the chain U(1) points with S3 certified
pts = [(r["producer"]["dim"], max(x for x in r["certifier"]["fill_nnz"] if x)) for r in s3_ok if fam(r) == "chain-u1"]
pts.sort()
if len(pts) >= 2:
    # least-squares slope of log(fill) vs log(d) over all chain S^z=0 points with S3 certified
    xs_, ys_ = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    expo = float(np.polyfit(xs_, ys_, 1)[0])
    macro("CfillExponent", f"{expo:.1f}")
    macro("CfillNpoints", str(len(pts)))
failed_s3 = [r for r in rows2 if r not in s3_ok]

with open(os.path.join(MS, "tables", "e2.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Cost of certification (Section~\\ref{sec:e2}). Producer: "
            "QuSpin basis and matrix construction (build) and ARPACK for eight states (solve). Certifier: "
            "statements \\Sone{}--\\Sthree{} in seconds and peak memory (MB). ``mem'' means the step did not fit "
            f"in the {settings.get('memory_cap_GiB', 4.5):.1f}\\,GiB address-space limit and the certificate reports "
            "the statement as not checked; MB is then from a second run without \\Sthree{}. Gap: certified gap divided by the gap estimated from the producer's eigenvalues."
            + (" $^\\dagger$\\Sthree{} inconclusive: the claim cuts a degenerate level (Section~\\ref{sec:e5})."
               if any(r.get("certifier", {}).get("statements", {}).get("S3.completeness", {}).get("status")
                      == "INCONCLUSIVE" for r in rows2) else "") + "}\n"
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
            if st["S1.model_fidelity"]["status"] == "NOT_CHECKED":
                s1 = "mem"
            if st["S3.completeness"]["status"] == "INCONCLUSIVE":
                s3 += "$^\\dagger$"
            if st["S3.completeness"]["status"] == "NOT_CHECKED":
                s3 = "mem"
            fill = sci(max(x for x in c["fill_nnz"] if x)) if any(c["fill_nnz"]) else "---"
            gap = f"{c['gap_lower_bound'] / c['true_gap_estimate']:.2f}" if c.get("gap_lower_bound") else "---"
            mb = f"{(cw if s3 == 'mem' and 'statements' in cw else c)['maxrss_mb']:.0f}"
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
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.3))
mk = {"chain-u1": "o", "chain-k0": "s", "square": "^"}
fam_label = {"chain-u1": "chain, $S^z$ block", "chain-k0": "chain, $(S^z,k)$ block", "square": "4x4, $S^z$ block"}
for fm in ("chain-u1", "chain-k0", "square"):
    R = sorted([r for r in rows2 if fam(r) == fm], key=lambda r: r["producer"]["dim"])
    d = [r["producer"]["dim"] for r in R]
    prod = [r["producer"]["t_build"] + r["producer"]["t_solve"] for r in R]
    s12, s3d, s3t, fills, nnzs, dfill = [], [], [], [], [], []
    for r in R:
        c = r["certifier"]
        if s1_ok(c):
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
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(axR.transData, axR.transAxes)
    for r in R:
        if r not in s3_ok:   # failure rug: x = block dimension, drawn at the top edge (no y value)
            axR.plot([r["producer"]["dim"]], [0.96], marker=mk[fm], mfc="none", color=ORANGE, ms=6, ls="none",
                     transform=trans, clip_on=False)
for ax in (axL, axR):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("block dimension $d$")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
# explicit limits on the log axis: the failure rug sits at the top edge and must not set the scale
ymax_R = max(max(x for x in r["certifier"]["fill_nnz"] if x) for r in s3_ok)
ymin_R = min(r["producer"]["nnz"] for r in rows2)
axR.set_ylim(ymin_R / 3, ymax_R * 12)
axL.set_ylabel("wall time (s)")
axR.set_ylabel("nonzeros")
from matplotlib.lines import Line2D
hL = [Line2D([], [], color=BLUE, lw=2, label="producer (build+solve)"),
      Line2D([], [], color=GREEN, lw=2, label="S1+S2"),
      Line2D([], [], color=ORANGE, lw=2, label="S3")]
hF = [Line2D([], [], color=INK2, marker=mk[k], ls="none", label=fam_label[k]) for k in mk]
axL.legend(handles=hL, fontsize=7, loc="upper left")
hR = [Line2D([], [], color=BLUE, lw=2, label="nnz(H)"), Line2D([], [], color=ORANGE, lw=2, label="nnz(L)+nnz(U)"),
      Line2D([], [], color=ORANGE, marker="o", mfc="none", ls="none", label="S3 exceeded memory cap (top edge)")]
axR.legend(handles=hR, fontsize=7, loc="lower right")
fig.legend(handles=hF, fontsize=7, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(MS, "figures", "fig_cost.pdf"))
plt.close(fig)

# ============================================================== E4
_, e4, _ = runs["e4_workflows_W1W2W3W4"]
w1, w2, w3, w4 = e4["W1"], e4["W2"], e4["W3"], e4["W4"]
macro("WcapGiB", f"{e4.get('settings', {}).get('memory_cap_GiB', 5.0):.1f}")
macro("WoneN", str(w1["N"]))
macro("WoneJtwo", f"{w1['J2']:g}")
macro("WoneBlocks", str(w1["n_blocks"]))
macro("WoneProd", f"{w1['producer_seconds']:.0f}")
macro("WoneCert", f"{w1['certify_seconds']:.0f}")
macro("WoneK", str(w1["K"]))
macro("WoneRadius", sci_dir(w1["radius"]))
macro("WoneNext", dec_dir(w1["next_lower_bound"], 9, up=False))
check(w1["status"] == "CERTIFIED" and w1["sector_status_counts"]["COMPLETE"] == w1["n_blocks"],
      "W1: certified, all blocks complete")
Nw = w1["N"]
with open(os.path.join(MS, "tables", "w1.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{The " + str(w1["K"]) + " lowest eigenvalues of the "
            f"$J_1$--$J_2$ ring, $N={Nw}$, $J_2={w1['J2']:g}$, certified from {w1['n_blocks']} symmetry blocks "
            f"(Section~\\ref{{sec:w1}}). Common radius {sci_dir(w1['radius'])}; the next eigenvalue is at least "
            f"${dec_dir(w1['next_lower_bound'], 9, up=False)}$. Momentum $k$ in units of $2\\pi/N$.}}\n\\label{{tab:w1}}\n\\small\n"
            "\\begin{tabular}{@{}r r r r@{}}\n\\toprule\n$j$ & $E_j$ & $S^z$ & $k$ \\\\\n\\midrule\n")
    for j, (e, sz, k) in enumerate(w1["levels"][: w1["K"]]):
        f.write(f"{j + 1} & ${e:.10f}$ & ${sz:+.0f}$ & {k} \\\\\n".replace("$+0$", "$0$"))
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
cl = w2["clusters"][0]
qn = cl["quantum_numbers"]["Sz_total"]
check(qn["status"] == "CERTIFIED" and all(v == "CERTIFIED" for v in w2["statuses"].values()), "W2: all certified")
macro("WtwoQN", ",".join(f"{v:.0f}" for v in qn["certified_values"]))
macro("WtwoQNBound", sci_dir(qn["bound"]))
o1, o2 = cl["observables"]["Sz_0 Sz_1"], cl["observables"]["Sz_0"]
macro("WtwoObs", f"{o1['value']:.12f}")
macro("WtwoObsBound", sci_dir(o1["bound"]))
macro("WtwoObsSz", sci(o2["value"]))
macro("WtwoObsSzBound", sci_dir(o2["bound"]))
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
    wrapped.extend(textwrap.wrap(line, 84, subsequent_indent="              ") or [""])
with open(os.path.join(MS, "tables", "w2_summary.txt"), "w") as f:
    f.write("\n".join(wrapped) + "\n")
macro("WthreeProd", f"{w3['producer_seconds']:.0f}")
macro("WthreeCert", f"{w3['certify_seconds']:.0f}")
macro("WthreeRadius", sci_dir(w3["max_radius"]))
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
            "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.30\\textwidth} r l l l l@{}}\n\\toprule\n"
            "System (block) & $d$ & $E_0$ & radius & \\Sthree{} & gap $\\ge$ \\\\\n\\midrule\n")
    names = {0: "Heisenberg ring $N=20$ ($S^z{=}0,k{=}0$)", 1: "$J_1$--$J_2$ ring $N=20$, $J_2=\\frac12$ ($S^z{=}0,k{=}0$)",
             2: "Heisenberg $4\\times4$ ($S^z{=}0$)", 3: "$J_1$--$J_2$ $4\\times4$, $J_2=\\frac12$ ($S^z{=}0$)"}
    for i, r in enumerate(w4):
        s3 = r["statuses"]["S3.completeness"]
        gap = dec_dir(r["gap_lower_bound"], 4, up=False) if r.get("gap_lower_bound") else "---"
        f.write(f"{names[i]} & {thin(r['dim'])} & ${r['E0']:.12f}$ & {sci_dir(r['radius'])} & "
                f"\\st{{{s3.replace('_', chr(92) + '_')}}} & {gap} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
mg = w4[1]
check(abs(mg["E0"] - (-7.5)) <= mg["radius"], f"W4: Majumdar-Ghosh E0=-7.5 within radius (E0={mg['E0']!r})")
# 4x4 J2=0.5: whether the S3 factorization fits under the memory cap depends on what the cap counts (RLIMIT_AS on
# POSIX counts address space, a Windows job object counts committed memory; E4 settings.memory_cap names it):
# NOT_CHECKED on the Linux run of 2026-10-21, CERTIFIED on the Windows run of 2026-10-22. The check is that
# the record is consistent either way, and the manuscript reads the status from the macro (run-stack).
_sq = w4[3]
_st3 = _sq["statuses"]["S3.completeness"]
macro("WfourSquareSthree", "\\st{" + _st3.replace("_", "\\_") + "}",
      note="E4 W4 J1-J2 4x4 J2=0.5 statuses['S3.completeness'] (run-stack: depends on what the memory cap counts)")
macro("WfourCapMechanism", tex_escape(str(e4.get("settings", {}).get("memory_cap", "RLIMIT_AS"))),
      note="E4 settings.memory_cap (records before 3.1.0: RLIMIT_AS)")
if _st3 == "NOT_CHECKED":
    _why = (_sq.get("s3_note") or "").lower()
    check("memory" in _why or "factorization could not be completed" in _why,
          f"W4: 4x4 J2=0.5 S3 not checked, and only for memory ({_why[:80]!r})")
    macro("WfourSquareGap", "--", note="E4 W4 4x4 J2=0.5: no gap bound (S3 not checked)")
else:
    _g = _sq.get("gap_lower_bound")
    check(_st3 == "CERTIFIED" and _g is not None and 0 < _g <= _sq["estimated_gap"],
          f"W4: 4x4 J2=0.5 S3 certified with 0 < gap bound <= estimated gap ({_g!r} vs {_sq['estimated_gap']!r})")
    macro("WfourSquareGap", f"{math.floor(_g * 1e4) / 1e4:.4f}",
          note="E4 W4 4x4 J2=0.5 gap_lower_bound rounded down to 4 decimals")

# ============================================================== E5
_, e5, _ = runs["e5_arpack_multiplet"]
su, rows5 = e5["setup"], e5["rows"]
lev = su["levels"]
macro("EfDim", thin(su["dim"]))
macro("EfDense", f"{su['dense_seconds'] / 60:.0f}")
macro("EfLevel", f"{lev[3][0]:.8f}")
macro("EfLevelMult", {2: "twofold", 3: "threefold", 4: "fourfold", 6: "sixfold"}.get(lev[3][1], f"{lev[3][1]}-fold"))
macro("EfNextLevel", f"{lev[4][0]:.8f}")
macro("EfSeeds", str(len(rows5)))
trunc = [r for r in rows5 if not r["correct"]]
corr = [r for r in rows5 if r["correct"]]
macro("EfTrunc", str(len(trunc)))
macro("EfCorrect", str(len(corr)))
check(len(trunc) >= 1, f"E5: ARPACK truncated the multiplet in at least one start ({len(trunc)}/{len(rows5)})")
check(all(r["members_of_fourfold_level"] < lev[3][1] for r in trunc),
      "E5: every incorrect list misses members of the degenerate level")
macro("EfMembersSeen", ", ".join(str(m) for m in sorted({r["members_of_fourfold_level"] for r in trunc})))
n_p1 = sum(r["P1_flag"] for r in trunc)
n_p2t = sum(r["P2_flag"] for r in trunc)
n_p2c = sum(r["P2_flag"] for r in corr)
macro("EfPoneFlag", "none" if n_p1 == 0 else str(n_p1))
macro("EfPtwoFlagTrunc", str(n_p2t))
macro("EfPtwoFlagCorrect", str(n_p2c))
check(n_p1 == 0, "E5: the residual check passes every truncated list")
p3_fail = [r for r in trunc if r["P3_statuses"]["S3.completeness"] == "FAILED"
           and (r.get("P3_missing_at_least") or 0) >= 1]
macro("EfPthreeFailed", str(len(p3_fail)))
check(len(p3_fail) == len(trunc), "E5: CERTIFY-ED reports FAILED (missing eigenvalue) for every truncated list")
check(all(r["P3_statuses"]["S2.eigenpairs"] == "CERTIFIED" for r in rows5),
      "E5: S2 certifies the returned pairs in every start (they are accurate eigenpairs)")
check(all(r["P3_statuses"]["S3.completeness"] != "FAILED" for r in corr),
      "E5: no correct list is reported as FAILED")
corr_s3 = sorted({r["P3_statuses"]["S3.completeness"] for r in corr})
macro("EfCorrectSthree", " or ".join("\\st{" + x.replace("_", "\\_") + "}" for x in corr_s3) or "---")
macro("EfFailFact", str(max(r["P3_factorizations"] for r in p3_fail)) if p3_fail else "?")
macro("EfFailSeconds", f"{max(r['P3_seconds'] for r in p3_fail):.0f}" if p3_fail else "?")
check(all(r["P3k_S3"] == "CERTIFIED" for r in corr) and all(r["P3k_S3"] == "INCONCLUSIVE" for r in trunc),
      "E5: claim to the first gap: CERTIFIED for correct lists, INCONCLUSIVE for truncated ones")
check(all("ill-posed" in r["P3k_S3_summary"] and "at least 4 eigenvalues" in r["P3k_S3_summary"]
          and "only 3 are claimed" in r["P3k_S3_summary"] for r in trunc),
      "E5: truncated claims to the first gap get the top-multiplet diagnosis (at least 4, 3 claimed)")
macro("EfKcorrect", ", ".join(str(k) for k in sorted({r["P3k_k"] for r in corr})) or "---")
macro("EfKtrunc", ", ".join(str(k) for k in sorted({r["P3k_k"] for r in trunc})) or "---")
with open(os.path.join(MS, "tables", "e5.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{ARPACK asked for the eight lowest states of the "
            f"$4\\times4$ Heisenberg $S^z=0$ block ($d={thin(su['dim'])}$) from {len(rows5)} random start vectors "
            "(Section~\\ref{sec:e5}). Rows: lists that contain all members of the "
            + {2: "twofold", 3: "threefold", 4: "fourfold", 6: "sixfold"}.get(lev[3][1], "degenerate")
            + f" level at $E={lev[3][0]:.3f}$, and lists that do not. P1, P2: number of starts flagged. P3, "
            "P3$'$: status of \\Sthree{} for the claim ``these are the eight lowest'' (P3) and for the claim of "
            "the states up to the first gap in the returned values (P3$'$).}\n"
            "\\label{tab:e5}\n\\footnotesize\n"
            "\\begin{tabular}{@{}l r r r l l@{}}\n\\toprule\n"
            "Returned list & starts & P1 flags & P2 flags & P3 & P3$'$ \\\\\n\\midrule\n")

    abbrev = {"CERTIFIED": "certified", "FAILED": "\\st{FAILED}", "INCONCLUSIVE": "\\st{INCONCL.}",
              "NOT_CHECKED": "\\st{NOT\\_CHECKED}"}

    def stat_cell(rs, key):
        from collections import Counter
        cnt = Counter(r[key] if isinstance(r[key], str) else r[key]["S3.completeness"] for r in rs)
        return ", ".join(f"{n} {abbrev.get(k, k)}" for k, n in sorted(cnt.items())) or "---"

    for label, rs in ((f"correct ({lev[3][1]} of {lev[3][1]} members)", corr),
                      ("incorrect (fewer members)", trunc)):
        f.write(f"{label} & {len(rs)} & {sum(r['P1_flag'] for r in rs)} & {sum(r['P2_flag'] for r in rs)} & "
                f"{stat_cell(rs, 'P3_statuses')} & {stat_cell(rs, 'P3k_S3')} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ============================================================== write
with open(os.path.join(MS, "numbers.tex"), "w") as f:
    f.write("% generated by experiments/make_numbers.py -- do not edit\n")
    for k in sorted(MACROS):
        f.write(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}" + (f"  % {NOTES[k]}" if k in NOTES else "") + "\n")
print(f"wrote {len(MACROS)} macros to {os.path.join(MS, 'numbers.tex')}")
# the manuscript inputs every generated/*.tex: a macro defined twice stops LaTeX ('command already defined');
# make_numbers_apps.py runs first (`certify-ed numbers`), so its files are present here
_dups = duplicate_macros(sorted(glob.glob(os.path.join(MS, "*.tex"))))
_dup_txt = ", ".join(k + " in " + "/".join(v) for k, v in _dups.items())
check(not _dups, "numbers: no macro is defined by \\newcommand in two generated files"
      + (f" (duplicates: {_dup_txt})" if _dups else ""))
bad = [s for ok, s in CHECKS if not ok]
for ok, s in CHECKS:
    print(("ok           " if ok else "CHECK FAILED ") + s)
sys.exit(1 if bad else 0)
