"""certify-ed command line: one entry point for using the package and for reproducing the paper.

    certify-ed env                                   floating-point environment and versions
    certify-ed certify CLAIM.npz --spec SPEC.json [--kind lowest] [--exact] [-o CERT]
    certify-ed spectrum MATRIX.npz [--sectors KEY]   enclose a whole spectrum (the former symveig)
    certify-ed list                                  the experiments and what they establish
    certify-ed test [PYTEST ARGS...]                 the test suite (repository checkout)
    certify-ed reproduce [--full] [--only E0,E9] [--skip E2] [--with-mutation] [--out DIR]
                                                     tests + experiments + numbers into one run dir
    certify-ed numbers [--run DIR] [--manuscript DIR]

`python run.py` in the repository is `certify-ed reproduce` (quick mode; `--full` for everything).
A run directory holds manifest.json (commit, environment, per-step status and time, record paths),
SUMMARY.md, the test log, one sub-directory per experiment record, and the generated numbers.
"""
import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time

# ------------------------------------------------------------------------------------ registry
# name: (script, quick args or None to skip in quick mode, full args, (minutes quick, minutes full) on one
#        core, purpose)
EXPERIMENTS = {
    "E0": ("e0_verifier_validation.py", [], [], (5, 5), "verifier vs Arb balls, exact counts, QuSpin construction"),
    "E1": ("e1_failure_matrix.py", [], [], (1, 1), "planted errors: which statement catches which failure"),
    "E2": ("e2_cost.py", None, [], (0, 21), "cost and memory of S2/S3 against problem size"),
    "E3": ("e3_tightness.py", [], [], (1, 1), "radii against Arb reference enclosures"),
    "E4": ("e4_workflows.py", None, ["W1", "W2", "W3", "W4", "N=20"], (0, 5),
           "workflows: global spectra from sectors, multiplets, matrix-free, benchmark values"),
    "E5": ("e5_arpack_multiplet.py", None, ["12"], (0, 27), "unplanted error: ARPACK misses a multiplet member"),
    "E6": ("e6_mutation.py", None, ["64", "2"], (0, 15), "mutation analysis of the certifier (opt-in)"),
    "E7": ("e7_exact_benchmarks.py", ["square_J2=0"], [], (2, 12),
           "benchmark E0 values re-derived with no floating-point assumption"),
    "E8": ("e8_symveig_comparison.py", [], [], (2, 2), "symveig 1.0.0 vs certify_ed.spectrum on symveig's benchmarks"),
    "E9": ("e9_pyrochlore.py", ["quick"], ["full"], (1, 25), "pyrochlore crown: certified S=1/2 bound; S=1 findings"),
    "E10": ("e10_aklt.py", ["quick"], ["full"], (2, 75), "Pomata-Wei AKLT gap criteria in exact arithmetic"),
}
OPT_IN = {"E6"}
NEEDS = {"E0": ["flint", "quspin"], "E1": ["quspin"], "E3": ["flint"], "E4": ["quspin"], "E5": ["quspin"],
         "E7": ["flint"], "E9": ["flint"], "E10": ["flint"]}


def repo_root():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return here if os.path.isdir(os.path.join(here, "experiments")) else None


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _git(root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def environment():
    import numpy as np
    import scipy
    from . import __version__
    from .fp import probe_float_environment
    env = {"certify_ed": __version__, "python": sys.version.split()[0], "numpy": np.__version__,
           "scipy": scipy.__version__, "platform": sys.platform}
    for mod in ("flint", "quspin", "h5py", "matplotlib", "mpmath"):
        env[mod] = getattr(__import__(mod), "__version__", "installed") if _have(mod) else "not installed"
    env["float_environment"] = probe_float_environment()
    return env


# ------------------------------------------------------------------------------------ commands
def cmd_env(a):
    print(json.dumps(environment(), indent=2, default=str))
    return 0


def cmd_certify(a):
    from .certify import Options, certify
    from .importers import claim_from_npz
    from .spec import ModelSpec
    spec = None
    if a.spec:
        with open(a.spec) as f:
            spec = ModelSpec.from_dict(json.load(f))
    claim = claim_from_npz(a.claim, kind=a.kind)
    cert = certify(claim, spec, Options(exact_recheck=a.exact))
    out = cert.save(a.output or os.path.splitext(a.claim)[0] + ".certificate")
    for s in cert.statements:
        print(f"{s['id']:22s} {s['status']:13s} {s['summary']}")
    print(f"certificate: {out}")
    return 0 if all(s["status"] != "FAILED" for s in cert.statements) else 1


def cmd_spectrum(a):
    import numpy as np
    import scipy.sparse as sp
    from .spectrum import enclose_spectrum
    z = np.load(a.matrix, allow_pickle=False)
    if "H" in z.files:
        H = z["H"]
    else:
        H = sp.csr_matrix((z["H_data"], z["H_indices"], z["H_indptr"]), shape=tuple(z["H_shape"]))
    S = z[a.sectors] if a.sectors else None
    enc = enclose_spectrum(H, S)
    rows = [{"lo": iv.lo, "hi": iv.hi, "count": iv.count, "counted": iv.counted, "method": iv.method}
            for iv in enc.intervals]
    res = {"dimension": enc.dimension, "complete": enc.complete, "max_halfwidth": enc.max_halfwidth,
           "sector_dims": {str(k): v for k, v in enc.sector_dims.items()}, "intervals": rows}
    if a.output:
        with open(a.output, "w") as f:
            json.dump(res, f, indent=1)
    print(f"{len(rows)} intervals, complete: {enc.complete}, max half-width {enc.max_halfwidth:.3e}")
    for r in rows[: a.show]:
        print(f"  [{r['lo']:.15g}, {r['hi']:.15g}]  count {r['count']}{'' if r['counted'] else ' (not counted)'}")
    return 0 if enc.complete else 2


def cmd_list(a):
    print(f"{'name':5s} {'quick (min)':>12s} {'full (min)':>11s}  purpose")
    for k, (script, q, f, (mq, mf), purpose) in EXPERIMENTS.items():
        quick = "opt-in" if k in OPT_IN else (str(mq) if q is not None else "-")
        print(f"{k:5s} {quick:>12s} {mf:11d}  {purpose}")
    return 0


def _pytest(root, extra, log_path=None):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", *extra]
    t = time.time()
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if log_path:
        with open(log_path, "w") as f:
            f.write(p.stdout + p.stderr)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    return p.returncode, tail, time.time() - t


def cmd_test(a):
    root = repo_root()
    if root is None:
        print("the test suite needs a repository checkout")
        return 2
    code, tail, secs = _pytest(root, a.pytest_args)
    print(tail)
    return code


def _run_experiment(root, name, args, out, log):
    script = EXPERIMENTS[name][0]
    env = dict(os.environ, CERTIFY_RESULTS=out, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONPATH=root)
    before = set(os.listdir(out))
    t = time.time()
    with open(os.path.join(out, f"{name}.log"), "w") as f:
        p = subprocess.run([sys.executable, script, *args], cwd=os.path.join(root, "experiments"),
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    new = sorted(set(os.listdir(out)) - before - {f"{name}.log"})
    return {"script": script, "args": args, "returncode": p.returncode, "seconds": round(time.time() - t, 1),
            "records": new, "log": f"{name}.log"}


def _numbers(root, run_dir, manuscript, log):
    """Application numbers always; the full manuscript numbers when E0-E5 records are present."""
    env = dict(os.environ, CERTIFY_RESULTS=run_dir, PYTHONPATH=root)
    out = {}
    for script in ("make_numbers_apps.py", "make_numbers.py"):
        if script == "make_numbers.py":
            have = {d.split("_")[0] for d in os.listdir(run_dir)}
            missing = [e for e in ("e0", "e1", "e2", "e3", "e4", "e5") if e not in have]
            if missing:
                out[script] = {"status": "skipped", "reason": f"needs records of {', '.join(missing)} (full run)"}
                continue
        p = subprocess.run([sys.executable, script, manuscript], cwd=os.path.join(root, "experiments"),
                           capture_output=True, text=True, env=env)
        checks = [l for l in p.stdout.splitlines() if "checks pass" in l or l.startswith("CHECK FAILED")]
        out[script] = {"status": "ok" if p.returncode == 0 else "failed", "returncode": p.returncode,
                       "checks": checks[-5:]}
        with open(os.path.join(run_dir, script.replace(".py", ".log")), "w") as f:
            f.write(p.stdout + p.stderr)
    return out


def cmd_reproduce(a):
    root = repo_root()
    if root is None:
        print("reproduce needs a repository checkout (experiments/ and tests/)")
        return 2
    mode = "full" if a.full else "quick"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.abspath(a.out or os.path.join(root, "results", f"run_{mode}_{stamp}"))
    os.makedirs(out, exist_ok=True)
    manifest = {"mode": mode, "started": _dt.datetime.now().isoformat(timespec="seconds"),
                "git_commit": _git(root, "rev-parse", "HEAD"), "git_dirty": bool(_git(root, "status", "--porcelain")),
                "environment": environment(), "steps": {}}
    path = os.path.join(out, "manifest.json")
    if os.path.exists(path):                     # resume / extend an earlier run in the same directory
        with open(path) as f:
            prev = json.load(f)
        if prev.get("mode") != mode:
            print(f"{out} holds a run in mode {prev.get('mode')}; use a new --out")
            return 2
        manifest["steps"] = prev.get("steps", {})
        manifest["started"] = prev.get("started", manifest["started"])
        manifest["extended"] = prev.get("extended", []) + [{
            "at": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": manifest["git_commit"], "git_dirty": manifest["git_dirty"]}]
        manifest["git_commit"], manifest["git_dirty"] = prev.get("git_commit"), prev.get("git_dirty")

    def save():
        with open(path, "w") as f:
            json.dump(manifest, f, indent=1, default=str)

    def log(msg):
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(out, "run.log"), "a") as f:
            f.write(line + "\n")

    log(f"certify-ed reproduce ({mode}); output {out}")
    if not manifest["environment"]["float_environment"]["standard_model_hypotheses_hold"]:
        log("WARNING: the floating-point environment violates the standard model; certificates will say so")
    save()
    if not a.no_tests:
        log("tests ...")
        code, tail, secs = _pytest(root, [], os.path.join(out, "tests.log"))
        manifest["steps"]["tests"] = {"returncode": code, "summary": tail, "seconds": round(secs, 1)}
        log(f"tests: {tail} ({secs:.0f}s)")
        save()
    names = [n.strip().upper() for n in a.only.split(",")] if a.only else list(EXPERIMENTS)
    skip = {n.strip().upper() for n in a.skip.split(",")} if a.skip else set()
    for name in names:
        script, qargs, fargs, (mq, mf), purpose = EXPERIMENTS[name]
        if name in skip or (name in OPT_IN and not a.with_mutation and not a.only):
            continue
        args = fargs if mode == "full" else qargs
        if args is None:
            manifest["steps"][name] = {"status": "skipped", "reason": "full mode only"}
            continue
        missing = [m for m in NEEDS.get(name, []) if not _have(m)]
        if missing:
            manifest["steps"][name] = {"status": "skipped", "reason": f"needs {', '.join(missing)}"}
            log(f"{name}: skipped (needs {', '.join(missing)})")
            save()
            continue
        log(f"{name} ({purpose}; about {mf if mode == 'full' else mq} min) ...")
        r = _run_experiment(root, name, args, out, log)
        r["status"] = "ok" if r["returncode"] == 0 else "failed"
        r["git_commit"] = _git(root, "rev-parse", "HEAD")
        r["git_dirty"] = bool(_git(root, "status", "--porcelain"))
        manifest["steps"][name] = r
        log(f"{name}: {r['status']} in {r['seconds']:.0f}s -> {', '.join(r['records']) or 'no record'}")
        save()
    if not a.no_numbers:
        ms = os.path.abspath(a.manuscript or os.path.join(out, "paper"))
        manifest["steps"]["numbers"] = _numbers(root, out, ms, log)
        log(f"numbers: {json.dumps(manifest['steps']['numbers'])}")
    manifest["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    save()
    _summary(out, manifest)
    bad = [k for k, v in manifest["steps"].items()
           if (isinstance(v, dict) and (v.get("status") == "failed" or v.get("returncode", 0) not in (0, None)))]
    bad += [s for s, r in manifest["steps"].get("numbers", {}).items() if r.get("status") == "failed"]
    log(f"done; {'all steps ok' if not bad else 'FAILED: ' + ', '.join(bad)}; see {os.path.join(out, 'SUMMARY.md')}")
    return 1 if bad else 0


def _summary(out, m):
    lines = [f"# certify-ed reproduce ({m['mode']})", "",
             f"* commit `{m['git_commit']}`{' (with uncommitted changes)' if m['git_dirty'] else ''}",
             *[f"* extended at {e['at']} from commit `{e['git_commit']}`" for e in m.get("extended", [])],
             f"* started {m['started']}, finished {m.get('finished')}",
             f"* floating-point standard model holds: "
             f"{m['environment']['float_environment']['standard_model_hypotheses_hold']}", "",
             "| step | status | seconds | records |", "|---|---|---|---|"]
    for k, v in m["steps"].items():
        if k == "numbers":
            for s, r in v.items():
                lines.append(f"| {s} | {r.get('status')} | | {'; '.join(r.get('checks', [])) or r.get('reason', '')} |")
            continue
        status = v.get("status", "ok" if v.get("returncode") == 0 else "failed")
        lines.append(f"| {k} | {status}{(' (' + v['reason'] + ')') if v.get('reason') else ''} | "
                     f"{v.get('seconds', '')} | {', '.join(v.get('records', [])) or v.get('summary', '')} |")
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_numbers(a):
    root = repo_root()
    if root is None:
        print("numbers needs a repository checkout")
        return 2
    run_dir = os.path.abspath(a.run or os.path.join(root, "experiments", "results"))
    ms = os.path.abspath(a.manuscript or os.path.join(run_dir, "paper"))
    res = _numbers(root, run_dir, ms, print)
    print(json.dumps(res, indent=1))
    return 0 if all(r.get("status") != "failed" for r in res.values()) else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="certify-ed", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="floating-point environment probe and versions").set_defaults(fn=cmd_env)
    c = sub.add_parser("certify", help="certify a claim stored in an .npz file")
    c.add_argument("claim"); c.add_argument("--spec", help="model specification (JSON, ModelSpec.to_dict)")
    c.add_argument("--kind", default="subset", choices=["subset", "lowest", "window", "all"])
    c.add_argument("--exact", action="store_true", help="add S6: exact re-check (needs python-flint)")
    c.add_argument("-o", "--output"); c.set_defaults(fn=cmd_certify)
    s = sub.add_parser("spectrum", help="verified enclosure of a whole spectrum (.npz with H)")
    s.add_argument("matrix"); s.add_argument("--sectors", help="key of a conserved diagonal in the .npz")
    s.add_argument("-o", "--output"); s.add_argument("--show", type=int, default=10); s.set_defaults(fn=cmd_spectrum)
    sub.add_parser("list", help="experiments of the paper").set_defaults(fn=cmd_list)
    t = sub.add_parser("test", help="run the test suite"); t.add_argument("pytest_args", nargs="*")
    t.set_defaults(fn=cmd_test)
    r = sub.add_parser("reproduce", help="tests + experiments + numbers into one run directory")
    r.add_argument("--full", action="store_true", help="all experiments at paper size (hours)")
    r.add_argument("--only", help="comma-separated experiment names, e.g. E0,E9")
    r.add_argument("--skip", help="comma-separated experiment names to skip")
    r.add_argument("--with-mutation", action="store_true", help="include E6 (mutation analysis)")
    r.add_argument("--no-tests", action="store_true"); r.add_argument("--no-numbers", action="store_true")
    r.add_argument("--out", help="run directory; an existing one of the same mode is extended (commits recorded per step)")
    r.add_argument("--manuscript"); r.set_defaults(fn=cmd_reproduce)
    n = sub.add_parser("numbers", help="regenerate numbers and tables from a run directory")
    n.add_argument("--run"); n.add_argument("--manuscript"); n.set_defaults(fn=cmd_numbers)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
