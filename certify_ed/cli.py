"""certify-ed command line: one entry point for using the package and for reproducing the paper.

    certify-ed env                                   floating-point environment and versions
    certify-ed certify CLAIM.npz --spec SPEC.json [--kind lowest|window|all] [--window A B] [--exact] [-o CERT]
    certify-ed certify --from BUNDLE_DIR [--preset NAME] [--only S1,S2] [--skip S3]
                       [--budget-seconds S | --budget auto] [--budget-mb MB] [--reason TEXT]
                       [--temperatures T1,T2] [--progress]
    certify-ed import BUNDLE_DIR [same flags]        the same, bundle first (docs/producer_bundle.md)
    certify-ed statement CERT.json [--style latex] [-o FILE]   methods paragraph + BibTeX from a certificate
    certify-ed view CERT.json [-o CERT.html]         offline HTML page of a certificate (docs/viewer.md)
    certify-ed verify CERT.json [--external CMD] [--bundle DIR] [--operator FILE] [--record FILE]
                                                     integrity check; optionally an external verifier, whose
                                                     verdict is recorded apart and changes no status
    certify-ed presets                               the presets and statements table (docs/presets.md)
    certify-ed plan CLAIM.npz [--spec SPEC.json] [--kind ...] [--preset ...]   predicted time and memory per statement
    certify-ed spectrum MATRIX.npz [--sectors KEY]   enclose a whole spectrum (the former symveig)
    certify-ed list                                  the experiments and what they establish
    certify-ed test [PYTEST ARGS...]                 the test suite (repository checkout)
    certify-ed reproduce [--full] [--only E0,E9] [--skip E2] [--with-mutation] [--jobs N]
                         [--out DIR [--resume]]      tests + experiments + numbers into one run dir
    certify-ed numbers [--run DIR] [--manuscript DIR]

`python run.py` in the repository is `certify-ed reproduce` (quick mode; `--full` for everything).
A run directory holds manifest.json (commit, source fingerprint, environment, per-step status, time,
record paths and fingerprint), SUMMARY.md, the test log, one sub-directory per experiment record, and
the generated numbers. `--jobs N` runs experiments in parallel; `--resume` keeps finished steps.
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
    "E0": ("e0_verifier_validation.py", [], [], (4, 4), "verifier vs Arb balls, exact counts, QuSpin construction"),
    "E1": ("e1_failure_matrix.py", [], [], (1, 1), "planted errors: which statement catches which failure"),
    "E2": ("e2_cost.py", None, [], (0, 22), "cost and memory of S2/S3 against problem size"),
    "E3": ("e3_tightness.py", [], [], (1, 1), "radii against Arb reference enclosures"),
    "E4": ("e4_workflows.py", None, ["W1", "W2", "W3", "W4", "N=20"], (0, 6),
           "workflows: global spectra from sectors, multiplets, matrix-free, benchmark values"),
    "E5": ("e5_arpack_multiplet.py", None, ["12"], (0, 38), "unplanted error: ARPACK misses a multiplet member"),
    "E6": ("e6_mutation.py", None, ["64", "2"], (0, 19), "mutation analysis of the certifier (opt-in)"),
    "E7": ("e7_exact_benchmarks.py", ["square_J2=0"], [], (2, 11),
           "benchmark E0 values re-derived with no floating-point assumption"),
    "E8": ("e8_symveig_comparison.py", [], [], (3, 3), "symveig 1.0.0 vs certify_ed.spectrum on symveig's benchmarks"),
    "E9": ("e9_pyrochlore.py", ["quick"], ["full"], (1, 17), "pyrochlore crown: certified S=1/2 bound; S=1 findings"),
    "E10": ("e10_aklt.py", ["quick"], ["full"], (2, 40), "Pomata-Wei AKLT gap criteria in exact arithmetic"),
}
OPT_IN = {"E6"}
# timing or memory is part of what these measure: never run them next to another experiment
EXCLUSIVE = {"E2", "E6", "E8"}
# experiments whose result depends on the test suite (the mutation score): --resume also needs equal tests
TEST_DEPENDENT = {"E6"}
NEEDS = {"E0": ["flint", "quspin"], "E1": ["quspin"], "E3": ["flint", "quspin"], "E4": ["quspin"], "E5": ["quspin"],
         "E7": ["flint"], "E9": ["flint"], "E10": ["flint"]}   # E3 part (d) imports quspin (SS-04)


from .provenance import repo_root, _git, source_fingerprint  # noqa: E402


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def environment():
    import numpy as np
    import scipy
    from . import __version__
    from .fp import probe_float_environment
    env = {"certify_ed": __version__, "python": sys.version.split()[0], "numpy": np.__version__,
           "scipy": scipy.__version__, "platform": sys.platform}
    for mod in ("flint", "quspin", "h5py", "matplotlib", "mpmath"):
        env[mod] = getattr(__import__(mod), "__version__", "installed") if _have(mod) else "not installed"
    env["flint_c_library"] = flint_c_version()            # MS-N12: the FLINT C library, not python-flint
    env["float_environment"] = probe_float_environment()
    env.update({k: v for k, v in source_fingerprint().items() if k in ("git_commit", "source_hash", "source_dirty")})
    return env


def flint_c_version():
    """Version of the FLINT C library behind python-flint (MS-N12), or 'unknown' / 'not installed'."""
    try:
        import flint
    except Exception:
        return "not installed"
    for attr in ("__FLINT_VERSION__", "flint_version", "FLINT_VERSION"):
        v = getattr(flint, attr, None)
        if v:
            return str(v)
    return "unknown"


# ------------------------------------------------------------------------------------ commands
def cmd_env(a):
    print(json.dumps(environment(), indent=2, default=str))
    return 0


def _options_from_args(a):
    from .certify import Options
    from .presets import parse_statement_list
    kw = {}
    if getattr(a, "exact", False):
        kw["exact_recheck"] = True
    if getattr(a, "only", None):
        kw["only"] = parse_statement_list(a.only)
    if getattr(a, "skip", None):
        kw["skip"] = parse_statement_list(a.skip)
    if getattr(a, "reason", None):
        kw["narrowing_reason"] = a.reason
    if getattr(a, "budget_seconds", None) is not None:
        kw["budget_seconds"] = float(a.budget_seconds)
    if getattr(a, "budget", None):
        kw["budget_seconds"] = "auto" if str(a.budget).strip().lower() == "auto" else float(a.budget)
    if getattr(a, "progress", False):
        kw["progress"] = True
    if getattr(a, "budget_mb", None) is not None:
        kw["budget_mb"] = float(a.budget_mb)
    if getattr(a, "temperatures", None):
        kw["temperatures"] = [float(t) for t in a.temperatures.replace(";", ",").split(",") if t.strip()]
    if getattr(a, "assume_no_underflow", False):
        kw["underflow_check"] = False
    if getattr(a, "plan", False):
        kw["plan"] = True
    if getattr(a, "block_mb", None) is not None:
        kw["block_mb"] = float(a.block_mb)
    if getattr(a, "preset", None):
        return Options.from_preset(a.preset, **kw)
    return Options(**kw)


def _meta_defaults(a, meta):
    """Scope fields the bundle's meta.json suggests, used where the command line says nothing."""
    for key in ("preset", "only", "skip", "budget_seconds", "budget_mb"):
        if getattr(a, key, None) in (None, "", False) and meta.get(key) not in (None, ""):
            val = meta[key]
            setattr(a, key, ",".join(val) if isinstance(val, list) else val)
    if not getattr(a, "temperatures", None) and meta.get("temperatures"):
        a.temperatures = ",".join(str(t) for t in meta["temperatures"])


def cmd_certify(a):
    from .certify import certify
    from .spec import ModelSpec
    spec = None
    if a.spec:
        with open(a.spec) as f:
            spec = ModelSpec.from_dict(json.load(f))
    src = getattr(a, "from_bundle", None) or getattr(a, "bundle", None)
    window = None
    if getattr(a, "window", None):
        from .certify import parse_window
        window = parse_window(",".join(a.window) if isinstance(a.window, (list, tuple)) else a.window)
    if src:
        from .importers import claim_from_bundle
        claim, bspec, meta = claim_from_bundle(src, allow_code=bool(getattr(a, "allow_code", False)),
                                               kind=a.kind if a.kind != "auto" else None, window=window)
        spec = spec or bspec
        _meta_defaults(a, meta)
        default_out = os.path.join(src, "certificate")
    else:
        if not a.claim:
            print("give a claim file (.npz/.h5/...) or --from BUNDLE_DIR")
            return 2
        from .importers import claim_from_file, claim_from_npz
        kind = "subset" if a.kind == "auto" else a.kind
        if a.claim.lower().endswith(".npz") and not getattr(a, "operator", None):
            claim = claim_from_npz(a.claim, kind=kind, window=window)
        else:
            claim = claim_from_file(a.claim, getattr(a, "eigenvectors", None), getattr(a, "operator", None),
                                    getattr(a, "basis", None), kind=kind, window=window,
                                    allow_code=bool(getattr(a, "allow_code", False)))
        default_out = os.path.splitext(a.claim)[0] + ".certificate"
    if (claim.kind == "all" and not getattr(a, "preset", None) and not getattr(a, "only", None)
            and not getattr(a, "skip", None)):
        a.preset = "whole-spectrum"          # the whole-spectrum path (C-17), recorded in the scope block
    opts = _options_from_args(a)
    from .precision import PrecisionError
    try:
        cert = certify(claim, spec, opts)
    except PrecisionError as e:          # data binary64 cannot hold exactly: no certificate (precision.py)
        print(f"certify: refused: {e}")
        return 2
    out = cert.save(a.output or default_out)
    for s_ in cert.statements:
        print(f"{s_['id']:22s} {s_['status']:13s} {s_['summary']}")
    sc = cert.body["scope"]
    print(f"scope: preset {sc['preset'] or '-'}; narrowed: "
          f"{', '.join(x['id'].split('.')[0] + ' (' + x['by'] + ')' for x in sc['narrowed']) or 'none'}")
    print(f"certificate: {out}")
    return 0 if all(s_["status"] != "FAILED" for s_ in cert.statements) else 1


def _claim_from_args(a):
    """(claim, spec, default_out) from the certify/import/plan arguments (shared by cmd_certify and cmd_plan)."""
    from .spec import ModelSpec
    spec = None
    if a.spec:
        with open(a.spec) as f:
            spec = ModelSpec.from_dict(json.load(f))
    src = getattr(a, "from_bundle", None) or getattr(a, "bundle", None)
    window = None
    if getattr(a, "window", None):
        from .certify import parse_window
        window = parse_window(",".join(a.window) if isinstance(a.window, (list, tuple)) else a.window)
    if src:
        from .importers import claim_from_bundle
        claim, bspec, meta = claim_from_bundle(src, allow_code=bool(getattr(a, "allow_code", False)),
                                               kind=a.kind if a.kind != "auto" else None, window=window)
        spec = spec or bspec
        _meta_defaults(a, meta)
        return claim, spec, os.path.join(src, "certificate")
    if not a.claim:
        return None, None, None
    from .importers import claim_from_file, claim_from_npz
    kind = "subset" if a.kind == "auto" else a.kind
    if a.claim.lower().endswith(".npz") and not getattr(a, "operator", None):
        claim = claim_from_npz(a.claim, kind=kind, window=window)
    else:
        claim = claim_from_file(a.claim, getattr(a, "eigenvectors", None), getattr(a, "operator", None),
                                getattr(a, "basis", None), kind=kind, window=window,
                                allow_code=bool(getattr(a, "allow_code", False)))
    return claim, spec, os.path.splitext(a.claim)[0] + ".certificate"


def cmd_plan(a):
    """Print the predicted seconds and peak memory per statement (certify_ed.cost.plan_for_claim); nothing runs."""
    from .cost import plan_for_claim
    claim, spec, _ = _claim_from_args(a)
    if claim is None:
        print("give a claim file (.npz/.h5/...) or --from BUNDLE_DIR")
        return 2
    if claim.kind == "all" and not getattr(a, "preset", None) and not getattr(a, "only", None) and not getattr(a, "skip", None):
        a.preset = "whole-spectrum"
    opts = _options_from_args(a)
    plan = plan_for_claim(claim, spec, opts)
    print(plan.format())
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=1)
        print(f"plan: {a.output}")
    return 0


def cmd_statement(a):
    from .certificate_v2 import Certificate
    from .report import methods_paragraph
    cert = Certificate.load(a.certificate, verify=not a.no_verify)
    text = methods_paragraph(cert, style=a.style)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"statement: {a.output}")
    else:
        print(text)
    return 0


def cmd_crossing(a):
    """C-11: order the rank-`level` eigenvalue intervals of two certificates (certify_ed.crossings)."""
    from .certificate_v2 import Certificate
    from .crossings import crossing_main, format_crossing
    ca = Certificate.load(a.a, verify=not a.no_verify)
    cb = Certificate.load(a.b, verify=not a.no_verify)
    rec = crossing_main(ca, cb, level=a.level, json_path=a.json)
    print(format_crossing(rec))
    return 0 if rec["relation"] != "not separated" else 2


def cmd_sweep(a):
    """E-04: certified eigenvalue tubes over a coupling interval (certify_ed.sweep) for a built-in model family."""
    from . import spec as _spec
    from .sweep import certified_crossing_interval, certified_sweep, format_sweep
    factory = getattr(_spec, a.model, None)
    if factory is None or not callable(factory):
        raise SystemExit(f"unknown model factory {a.model!r}; known: heisenberg_chain, tfim_chain, heisenberg_square, "
                         "spinless_fermion_chain, bose_hubbard_chain, spin1_heisenberg_chain")
    fixed = {}
    for item in (a.fixed.split(",") if a.fixed else []):
        key, _, val = item.partition("=")
        try:
            fixed[key.strip()] = float(val)
        except ValueError:
            fixed[key.strip()] = val.strip()
    first = [int(x) for x in a.sizes.split(",")]
    values = [float(x) for x in a.values.split(",")]

    def spec_fn(c):
        return factory(*first, **dict(fixed, **{a.param: c}))

    states_fn = None
    if a.two_sz is not None:
        from .basis import magnetization_state_set
        states_fn = lambda spec: magnetization_state_set(spec, a.two_sz)  # noqa: E731
    sector_fn = None
    if a.momentum is not None:
        from .basis import cyclic_isometry, translation
        sector_fn = lambda spec, st: cyclic_isometry(st, translation(spec), a.momentum)  # noqa: E731
    from .certify import Options
    opts = Options(eigenspaces="none", degeneracy=False) if a.fast else Options()
    rec = certified_sweep(spec_fn, values, k=a.k, options=opts, states_fn=states_fn, sector_fn=sector_fn,
                          parameter_name=a.param, save_dir=a.save_dir)
    print(format_sweep(rec))
    if a.crossing is not None:
        la_, lb_ = (int(x) for x in a.crossing.split(","))
        x = certified_crossing_interval(rec, la_, rec, lb_)
        print(x["statement"])
        rec["crossing"] = x
    if a.json:
        from .certificate_v2 import _jsonable
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(_jsonable({k: v for k, v in rec.items() if not k.startswith("_")}), f, indent=1)
    return 0


def cmd_variational(a):
    """E-05: variational statements for a trial state (certify_ed.variational)."""
    from .io.formats import read_array, read_operator
    from .variational import certify_variational, format_variational
    spec = None
    if a.spec:
        from .spec import ModelSpec
        with open(a.spec, encoding="utf-8") as f:
            spec = ModelSpec.from_dict(json.load(f))
    operator = read_operator(a.operator) if a.operator else None
    if operator is None and spec is None:
        raise SystemExit("give --operator (a stored matrix file) or --spec (a model spec JSON)")
    if a.mps:
        import numpy as np
        data = np.load(a.mps)
        keys = sorted(data.files, key=lambda k: (len(k), k))
        psi = {"mps": [data[k] for k in keys]}
    elif a.ci_indices:
        n = int(a.dim) if a.dim else (operator.shape[0] if operator is not None else spec.dim)
        psi = {"indices": read_array(a.ci_indices), "amplitudes": read_array(a.state, role="eigenvectors"), "n": n}
    else:
        psi = read_array(a.state, role="eigenvectors") if a.state else None
    gap_cert = None
    if a.gap_certificate:
        from .certificate_v2 import Certificate
        gap_cert = Certificate.load(a.gap_certificate)
    declared = None
    if a.declared:
        declared = {}
        for item in a.declared.split(","):
            key, _, val = item.partition("=")
            try:
                declared[key.strip()] = float(val)
            except ValueError:
                declared[key.strip()] = val.strip()
    from .precision import PrecisionError
    try:
        rec = certify_variational(operator, psi, spec=spec, gap_hypothesis=a.gap, gap_source=a.gap_source or "",
                                  gap_certificate=gap_cert, declared=declared, underflow_check=not a.assume_no_underflow,
                                  description=a.description or "")
    except PrecisionError as e:          # data binary64 cannot hold exactly: no record (precision.py)
        print(f"variational: refused: {e}")
        return 2
    print(format_variational(rec))
    if a.json:
        from .certificate_v2 import _jsonable
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(_jsonable({k: v for k, v in rec.items() if not k.startswith("_")}), f, indent=1)
    return 0


def cmd_nonhermitian(a):
    """E-02: backward errors, Bauer-Fike discs and their counts for a non-Hermitian matrix (certify_ed.nonhermitian;
    every statement UNVERIFIED)."""
    import numpy as np
    from .io.formats import read_array, read_operator
    from .nonhermitian import certify_nonhermitian, format_nonhermitian

    def _left(path):
        if path.lower().endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                if "left_eigenvectors" in z.files:
                    return read_array(path, key="left_eigenvectors", role="eigenvectors")
        return read_array(path, role="eigenvectors")
    A = read_operator(a.operator)
    mu = read_array(a.eigenvalues or a.operator, role="eigenvalues")
    X = read_array(a.eigenvectors or a.eigenvalues or a.operator, role="eigenvectors")
    Y = _left(a.left_eigenvectors) if a.left_eigenvectors else None
    try:
        rec = certify_nonhermitian(A, mu, X, kind=a.kind, left_eigenvectors=Y, max_dim=a.max_dim,
                                   underflow_check=not a.assume_no_underflow, description=a.description or "")
    except ValueError as e:
        print(f"nonhermitian: refused: {e}")
        return 2
    print(format_nonhermitian(rec))
    if a.json:
        from .certificate_v2 import _jsonable
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(_jsonable({k: v for k, v in rec.items() if not k.startswith("_")}), f, indent=1)
    return 0


def cmd_levelstats(a):
    """C-09: interval r-ratios and spacing histogram of a complete, counted spectrum (certify_ed.levelstats)."""
    from .certificate_v2 import Certificate
    from .levelstats import format_level_statistics, level_statistics
    cert = Certificate.load(a.certificate, verify=not a.no_verify)
    unfold = a.unfold
    if unfold not in (None, "none"):
        try:
            unfold = int(unfold.split(":")[-1])
        except ValueError:
            raise SystemExit(f"--unfold expects 'none' or 'poly:D' (got {a.unfold!r})") from None
    else:
        unfold = "none"
    try:
        res = level_statistics(cert, unfolding=unfold, bins=a.bins, bin_range=(0.0, a.smax))
    except ValueError as e:
        print(f"levelstats: refused: {e}")
        return 2
    print(format_level_statistics(res))
    if a.json:
        from .certificate_v2 import _jsonable
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(_jsonable(res), f, indent=1)
    return 0


def cmd_extrapolate(a):
    """C-10: finite-size extrapolation report from per-size certificates (certify_ed.extrapolation; not certified)."""
    from .certificate_v2 import Certificate
    from .extrapolation import extrapolation_report
    certs = [Certificate.load(p, verify=not a.no_verify) for p in a.certificates]
    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else None
    rep = extrapolation_report(certs, model=a.model, sizes=sizes, per_site=bool(a.per_site), level=a.level)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(rep["markdown"])
        print(f"report: {a.output}")
    else:
        print(rep["markdown"])
    if a.json:
        from .certificate_v2 import _jsonable
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(_jsonable({k: v for k, v in rep.items() if k != "markdown"}), f, indent=1)
    return 0
def cmd_view(a):
    """Offline HTML view of a saved certificate (certify_ed.viewer). The file is written even when the
    integrity check fails (the page then opens with a failure banner); the exit code is 1 in that case."""
    from .viewer import view_file
    rep = view_file(a.certificate, a.output, max_pairs=a.max_pairs)
    integ = rep["integrity"]
    print(f"view: {rep['html']}")
    print(f"integrity: {'ok' if integ['ok'] else 'FAILED'} (body hash {'ok' if integ['body_hash_ok'] else 'MISMATCH'}, "
          f"arrays {'ok' if integ['arrays_ok'] else 'BAD: ' + ', '.join(integ['bad_arrays'])}, "
          f"scope {'ok' if integ['scope_ok'] else 'n/a' if integ['scope_ok'] is None else 'PROBLEMS'})")
    sch = rep["schema"]["ok"]
    print(f"schema: {'ok' if sch else 'structural checks only' if sch is None else 'FAILED'}")
    return 0 if integ["ok"] else 1


def cmd_presets(a):
    from .presets import format_presets
    print(format_presets(markdown=bool(a.markdown)))
    return 0


def cmd_conform(a):
    """Conformance of producer bundles: schema, import, certify, schema of the result (conformance/README.md)."""
    from .conformance import check_bundle, format_report
    rc = 0
    reports = []
    for path in a.bundles:
        rep = check_bundle(path, allow_code=bool(a.allow_code), save=a.output if len(a.bundles) == 1 else None,
                           require_jsonschema=bool(a.require_jsonschema))
        reports.append(rep)
        print(format_report(rep))
        rc |= 0 if rep["ok"] else 1
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(reports if len(reports) > 1 else reports[0], f, indent=1, default=str)
    return rc


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
    for k, (_script, q, _f, (mq, mf), purpose) in EXPERIMENTS.items():
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


def _run_experiment(root, name, args, out):
    """Run one experiment script with one BLAS thread; its records are the new directories of `out`
    whose name starts with the script's experiment number (safe when experiments run concurrently)."""
    script = EXPERIMENTS[name][0]
    env = dict(os.environ, CERTIFY_RESULTS=out, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONPATH=root)
    prefix = name.lower() + "_"
    before = {d for d in os.listdir(out) if d.startswith(prefix)}
    fp = source_fingerprint(root)
    t = time.time()
    with open(os.path.join(out, f"{name}.log"), "w") as f:
        p = subprocess.run([sys.executable, script, *args], cwd=os.path.join(root, "experiments"),
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    new = sorted(d for d in os.listdir(out) if d.startswith(prefix) and d not in before
                 and os.path.isdir(os.path.join(out, d)))
    return {"script": script, "args": args, "returncode": p.returncode, "seconds": round(time.time() - t, 1),
            "records": new, "log": f"{name}.log", "status": "ok" if p.returncode == 0 else "failed", **fp}


def _numbers(root, run_dir, manuscript, log):
    """Application numbers always; the full manuscript numbers when E0-E5 records are present. The step
    records its own provenance (commit, blob hashes of the number scripts, source and tests hashes; IMP-06)."""
    from .provenance import numbers_fingerprint
    env = dict(os.environ, CERTIFY_RESULTS=run_dir, PYTHONPATH=root)
    out = {"provenance": numbers_fingerprint(root)}
    for script in ("make_numbers_apps.py", "make_numbers.py"):
        if script == "make_numbers.py":
            have = {d.split("_")[0] for d in os.listdir(run_dir)}
            missing = [e for e in ("e0", "e1", "e2", "e3", "e4", "e5") if e not in have]
            if missing:
                out[script] = {"status": "skipped", "reason": f"needs records of {', '.join(missing)} (full run)"}
                continue
        p = subprocess.run([sys.executable, script, manuscript], cwd=os.path.join(root, "experiments"),
                           capture_output=True, text=True, env=env)
        lines = p.stdout.splitlines()
        n_ok = sum(1 for l in lines if l.startswith("ok "))
        failed = [l for l in lines if l.startswith("CHECK FAILED")]
        skipped = [l for l in lines if l.startswith("skipped ")]
        checks = [f"{n_ok}/{n_ok + len(failed)} checks pass"] + failed[:5] + skipped[:5]
        out[script] = {"status": "ok" if p.returncode == 0 else "failed", "returncode": p.returncode,
                       "checks": checks}
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
                "source": source_fingerprint(root),
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

    import threading
    _lock = threading.Lock()

    def log(msg):                                # called from worker threads with --jobs
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        with _lock:
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
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        log(f"unknown experiment(s): {', '.join(unknown)}")
        return 2
    skip = {n.strip().upper() for n in a.skip.split(",")} if a.skip else set()
    current = source_fingerprint(root)
    todo = []
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
            continue
        prev = manifest["steps"].get(name, {})
        if (a.resume and prev.get("status") == "ok" and prev.get("args") == args
                and prev.get("source_hash") == current["source_hash"] and not prev.get("source_dirty")
                and not current["source_dirty"]
                and (name not in TEST_DEPENDENT or (prev.get("tests_hash") == current["tests_hash"]
                                                    and not prev.get("tests_dirty") and not current["tests_dirty"]))):
            log(f"{name}: kept from the earlier run (same computational source {current['source_hash']})")
            continue
        todo.append((name, args, mf if mode == "full" else mq, purpose))
    save()

    def finish(name, r):
        if name == "E6" and r["status"] == "ok":
            # rule-based classification of the survivors (the human inspection stays a separate step)
            for rec in r["records"]:
                subprocess.run([sys.executable, "e6_classify.py", os.path.join(out, rec)],
                               cwd=os.path.join(root, "experiments"), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, env=dict(os.environ, PYTHONPATH=root))
        manifest["steps"][name] = r
        log(f"{name}: {r['status']} in {r['seconds']:.0f}s -> {', '.join(r['records']) or 'no record'}")
        save()

    parallel = [t for t in todo if t[0] not in EXCLUSIVE]
    exclusive = [t for t in todo if t[0] in EXCLUSIVE]
    jobs = max(1, int(a.jobs or 1))
    if jobs > 1 and len(parallel) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        log(f"running {len(parallel)} experiments on {jobs} workers, then {len(exclusive)} alone")
        with ThreadPoolExecutor(jobs) as ex:
            futs = {}

            def job(name, args, mins, purpose):
                log(f"{name} ({purpose}; about {mins} min) started")
                return _run_experiment(root, name, args, out)
            for name, args, mins, purpose in sorted(parallel, key=lambda t: -t[2]):   # longest first
                futs[ex.submit(job, name, args, mins, purpose)] = name
            for fu in as_completed(futs):
                finish(futs[fu], fu.result())
    else:
        exclusive = parallel + exclusive
    for name, args, mins, purpose in exclusive:
        log(f"{name} ({purpose}; about {mins} min) ...")
        finish(name, _run_experiment(root, name, args, out))
    if not a.no_numbers:
        ms = os.path.abspath(a.manuscript or os.path.join(out, "paper"))
        manifest["steps"]["numbers"] = _numbers(root, out, ms, log)
        log(f"numbers: {json.dumps(manifest['steps']['numbers'])}")
    manifest["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    save()
    _summary(out, manifest)
    bad = [k for k, v in manifest["steps"].items()
           if (isinstance(v, dict) and (v.get("status") == "failed" or v.get("returncode", 0) not in (0, None)))]
    bad += [s for s, r in manifest["steps"].get("numbers", {}).items()
            if isinstance(r, dict) and r.get("status") == "failed"]
    log(f"done; {'all steps ok' if not bad else 'FAILED: ' + ', '.join(bad)}; see {os.path.join(out, 'SUMMARY.md')}")
    return 1 if bad else 0


def _summary(out, m):
    lines = [f"# certify-ed reproduce ({m['mode']})", "",
             f"* commit `{m['git_commit']}`{' (with uncommitted changes)' if m['git_dirty'] else ''}",
             *[f"* extended at {e['at']} from commit `{e['git_commit']}`" for e in m.get("extended", [])],
             f"* started {m['started']}, finished {m.get('finished')}",
             f"* computational source (certify_ed/ + experiments/, see certify_ed.provenance): "
             f"{', '.join(sorted({v.get('source_hash') for v in m['steps'].values() if isinstance(v, dict) and v.get('source_hash')})) or 'n/a'}"
             f"{' (UNCOMMITTED CHANGES in some step)' if any(isinstance(v, dict) and v.get('source_dirty') for v in m['steps'].values()) else ''}",
             f"* floating-point standard model holds: "
             f"{m['environment']['float_environment']['standard_model_hypotheses_hold']}", "",
             "| step | status | seconds | records |", "|---|---|---|---|"]
    for k, v in m["steps"].items():
        if k == "numbers":
            for s, r in v.items():
                if s == "provenance":
                    blobs = ", ".join(f"{os.path.basename(p)} {d.get('blob')}" for p, d in r.get("scripts", {}).items())
                    lines.append(f"| numbers (provenance) | commit `{r.get('git_commit')}` | | {blobs} |")
                    continue
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
    mpath = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mpath):                    # a reproduce run directory: record the new numbers step
        with open(mpath) as f:
            manifest = json.load(f)
        manifest["steps"]["numbers"] = res
        manifest.setdefault("numbers_regenerated", []).append(_dt.datetime.now().isoformat(timespec="seconds"))
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=1, default=str)
        _summary(run_dir, manifest)
    return 0 if all(r.get("status") != "failed" for k, r in res.items() if k != "provenance") else 1


def cmd_verify(a):
    """Integrity check of a saved certificate and, optionally, an external verifier (certify_ed.external, N-04).

    Exit code 0: the integrity check passed (or was skipped) and the external verifier, if one ran, reported PASS;
    1: the integrity check failed; 2: the external verifier reported FAIL, ERROR or TIMEOUT. The external verdict
    is written to its own record; the certificate and its statuses are never changed."""
    from .certificate_v2 import Certificate
    from .external import ENV_VAR, verify_external
    if a.no_verify:
        print("integrity: not checked (--no-verify)")
    else:
        try:
            cert = Certificate.load(a.certificate, verify=True)
        except ValueError as e:
            print(f"integrity: FAILED ({e})")
            return 1
        print("integrity: OK (body hash, array hashes, scope block)")
        print(cert.summary())
    command = a.external or os.environ.get(ENV_VAR)
    if not command:
        return 0
    record = a.record or ((a.certificate[:-5] if a.certificate.endswith(".json") else a.certificate) + ".external.json")
    rec = verify_external(a.certificate, command, bundle=a.bundle, operator=a.operator, sidecar=a.sidecar,
                          report=a.report, record=record, timeout=a.timeout)
    run = rec["run"]
    print(f"external verifier: {rec['verdict']} (exit code {run['exit_code']}, {run['seconds']} s"
          + (f", {run['error']}" if run["error"] else "") + ")")
    rep = rec.get("report") or {}
    st = rep.get("statements")
    if isinstance(st, dict):
        for sid in sorted(st):
            v = st[sid]
            print(f"  {sid}: {v.get('verdict') if isinstance(v, dict) else v}")
    print(f"external record: {record} (recorded only: no status of the certificate depends on it)")
    return 0 if rec["verdict"] == "PASS" else 2


def main(argv=None):
    p = argparse.ArgumentParser(prog="certify-ed", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="floating-point environment probe and versions").set_defaults(fn=cmd_env)
    scope_flags = []

    def add_scope_flags(cp):
        cp.add_argument("--spec", help="model specification (JSON, ModelSpec.to_dict); overrides the bundle's spec.json")
        cp.add_argument("--kind", default="auto", choices=["auto", "subset", "lowest", "window", "all"],
                        help="completeness claim (auto: the bundle's meta.json, else subset)")
        cp.add_argument("--window", nargs="+", metavar="A B",
                        help="interior window for --kind window: two numbers 'a b' (or 'a,b'); the claim is that the "
                             "file holds every eigenvalue in (a, b)")
        cp.add_argument("--exact", action="store_true", help="add S6: exact re-check (needs python-flint)")
        cp.add_argument("--preset", help="named statement set and options (certify-ed presets)")
        cp.add_argument("--only", help="comma-separated statements to run, e.g. S1,S2 (dependencies are added)")
        cp.add_argument("--skip", help="comma-separated statements to leave out, e.g. S3")
        cp.add_argument("--reason", help="why the scope was narrowed; recorded in the certificate")
        cp.add_argument("--budget-seconds", type=float, dest="budget_seconds",
                        help="statements not started within this wall time become NOT CHECKED (reason BUDGET)")
        cp.add_argument("--budget", help="time budget in seconds, or 'auto' for the E2 cost model (certify_ed.cost)")
        cp.add_argument("--progress", action="store_true", help="one line per statement with elapsed time (stderr)")
        cp.add_argument("--budget-mb", type=float, dest="budget_mb",
                        help="statements whose estimated memory exceeds this become NOT CHECKED (reason BUDGET)")
        cp.add_argument("--temperatures", help="comma-separated T for S7 (whole-spectrum claims, k_B = 1)")
        cp.add_argument("--assume-no-underflow", action="store_true", dest="assume_no_underflow",
                        help="skip the no-underflow scan (Options(underflow_check=False)); recorded in the certificate")
        cp.add_argument("--allow-code", action="store_true", dest="allow_code",
                        help="allow operator.py (matrix-free producer code) to be executed")
        cp.add_argument("--plan", action="store_true", help="print the predicted time and memory per statement first "
                        "(certify_ed.cost; advisory) and record it in the certificate")
        cp.add_argument("--block-mb", type=float, dest="block_mb",
                        help="memory for one column block of the residuals (streamed or sparse-operator vectors)")
        cp.add_argument("-o", "--output")
        scope_flags.append(cp)

    c = sub.add_parser("certify", help="certify a claim: a file (.npz/.h5/.csv/...) or --from BUNDLE_DIR")
    c.add_argument("claim", nargs="?", help="eigenvalue file (any format; .npz may hold vectors and H too)")
    c.add_argument("--from", dest="from_bundle", help="producer bundle directory (docs/producer_bundle.md)")
    c.add_argument("--eigenvectors", help="eigenvector file (with a separate eigenvalue file)")
    c.add_argument("--operator", help="operator file (with a separate eigenvalue file)")
    c.add_argument("--basis", help="basis file: state list or isometry")
    add_scope_flags(c)
    c.set_defaults(fn=cmd_certify)
    i = sub.add_parser("import", help="certify a producer bundle directory (same as certify --from)")
    i.add_argument("bundle")
    add_scope_flags(i)
    i.set_defaults(fn=cmd_certify, claim=None)
    pl = sub.add_parser("plan", help="predicted seconds and peak memory per statement, without running (certify_ed.cost)")
    pl.add_argument("claim", nargs="?", help="eigenvalue file (any format; .npz may hold vectors and H too)")
    pl.add_argument("--from", dest="from_bundle", help="producer bundle directory")
    pl.add_argument("--eigenvectors"); pl.add_argument("--operator"); pl.add_argument("--basis")
    add_scope_flags(pl)
    pl.set_defaults(fn=cmd_plan)
    st = sub.add_parser("statement", help="methods-section paragraph and BibTeX entry from a saved certificate")
    st.add_argument("certificate", help="certificate JSON written by Certificate.save / certify-ed certify")
    st.add_argument("--style", default="plain", choices=["plain", "latex"])
    st.add_argument("--no-verify", action="store_true", dest="no_verify", help="skip the integrity check on load")
    st.add_argument("-o", "--output")
    st.set_defaults(fn=cmd_statement)
    cx = sub.add_parser("crossing", help="order the rank-LEVEL eigenvalue intervals of two certificates (C-11)")
    cx.add_argument("a", help="certificate JSON A"); cx.add_argument("b", help="certificate JSON B")
    cx.add_argument("--level", type=int, default=0, help="eigenvalue rank, 0 = lowest (default 0)")
    cx.add_argument("--no-verify", action="store_true", dest="no_verify", help="skip the integrity check on load")
    cx.add_argument("--json", help="write the crossing record to this file")
    cx.set_defaults(fn=cmd_crossing)
    sw = sub.add_parser("sweep", help="certified eigenvalue tubes over a coupling interval for a built-in model family (E-04)")
    sw.add_argument("--model", required=True, help="spec factory in certify_ed.spec, e.g. heisenberg_chain, tfim_chain")
    sw.add_argument("--sizes", required=True, help="positional size arguments of the factory, e.g. 8 or 4,4")
    sw.add_argument("--param", required=True, help="the keyword parameter swept, e.g. J2, Delta, h")
    sw.add_argument("--values", required=True, help="increasing grid of parameter values, e.g. 0.3,0.4,0.5")
    sw.add_argument("--fixed", help="other keyword parameters, e.g. J=1.0,boundary=periodic")
    sw.add_argument("--k", type=int, default=1, help="number of lowest eigenpairs per point")
    sw.add_argument("--two-sz", type=int, default=None, help="restrict to the U(1) block with this 2 S^z")
    sw.add_argument("--momentum", type=int, default=None, help="translation sector j (needs --two-sz): tubes in the block")
    sw.add_argument("--crossing", help="two level indices 'a,b' to separate within this scan")
    sw.add_argument("--fast", action="store_true", help="Options(eigenspaces='none', degeneracy=False) per point")
    sw.add_argument("--save-dir", help="save the per-point certificates here")
    sw.add_argument("--json", help="write the scan record to this file")
    sw.set_defaults(fn=cmd_sweep)
    va = sub.add_parser("variational", help="Rayleigh upper bound, existence radius and Temple lower bound for a trial state (E-05)")
    va.add_argument("--operator", help="stored Hermitian matrix (npy/npz/HDF5/Matrix Market/...; docs/producer_bundle.md)")
    va.add_argument("--spec", help="model spec JSON (the operator is then built from it; model shift recorded)")
    va.add_argument("--state", help="trial vector file (dense), or the amplitudes with --ci-indices")
    va.add_argument("--mps", help="npz of MPS tensors (D_left, d, D_right), contracted to a dense vector (<= 2^22)")
    va.add_argument("--ci-indices", dest="ci_indices", help="indices file of a sparse CI vector (with --state = amplitudes)")
    va.add_argument("--dim", help="dimension of the sparse CI vector (default: the operator's)")
    va.add_argument("--gap", type=float, default=None, help="DECLARED lower bound rho on lambda_2 (Temple; recorded as a hypothesis)")
    va.add_argument("--gap-source", dest="gap_source", help="where the declared gap comes from (recorded verbatim)")
    va.add_argument("--gap-certificate", dest="gap_certificate", help="certificate of a 'lowest' claim on the same operator: certified rho")
    va.add_argument("--declared", help="producer estimates, e.g. energy=-3.65,variance=1e-4,samples=10000,method=VMC (DECLARED)")
    va.add_argument("--assume-no-underflow", action="store_true", dest="assume_no_underflow")
    va.add_argument("--description")
    va.add_argument("--json", help="write the record to this file")
    va.set_defaults(fn=cmd_variational)
    nh = sub.add_parser("nonhermitian", help="backward errors, Bauer-Fike discs and counts for a non-Hermitian matrix "
                                              "(E-02, UNVERIFIED)")
    nh.add_argument("--operator", required=True, help="the stored square matrix (npy/npz/HDF5/Matrix Market/...)")
    nh.add_argument("--eigenvalues", help="claimed eigenvalues (complex allowed; default: the operator file)")
    nh.add_argument("--eigenvectors", help="claimed right eigenvectors, one column per value (default: the eigenvalue file)")
    nh.add_argument("--left-eigenvectors", dest="left_eigenvectors", help="optional left eigenvectors (y^H A = mu y^H), "
                    "one column per value; an npz may hold them under the key left_eigenvectors")
    nh.add_argument("--kind", default="auto", choices=["auto", "all", "subset"],
                    help="'all': n pairs (N1 to N4); 'subset': backward errors only (N1); auto: 'all' when k = n")
    nh.add_argument("--max-dim", type=int, dest="max_dim", default=2000, help="dense size limit (default 2000)")
    nh.add_argument("--assume-no-underflow", action="store_true", dest="assume_no_underflow")
    nh.add_argument("--description")
    nh.add_argument("--json", help="write the record to this file")
    nh.set_defaults(fn=cmd_nonhermitian)
    ls = sub.add_parser("levelstats", help="interval r-ratios and spacing histogram of a complete counted spectrum (C-09)")
    ls.add_argument("certificate", help="certificate JSON of a kind 'all' claim with S3 CERTIFIED")
    ls.add_argument("--unfold", default="none", help="'none' (default) or 'poly:D': polynomial unfolding of degree D "
                                                     "(declared, not certified)")
    ls.add_argument("--bins", type=int, default=12, help="spacing histogram bins (default 12)")
    ls.add_argument("--smax", type=float, default=3.0, help="upper edge of the spacing histogram, in mean spacings (default 3)")
    ls.add_argument("--no-verify", action="store_true", dest="no_verify", help="skip the integrity check on load")
    ls.add_argument("--json", help="write the full record to this file")
    ls.set_defaults(fn=cmd_levelstats)
    ex = sub.add_parser("extrapolate", help="finite-size extrapolation report from per-size certificates (C-10; not certified)")
    ex.add_argument("certificates", nargs="+", help="certificate JSON files, one per system size")
    ex.add_argument("--model", default="1/N", help="'1/N' (default) or '1/N^2'")
    ex.add_argument("--sizes", help="comma-separated N per certificate (default: the spec's n_sites)")
    ex.add_argument("--per-site", action="store_true", dest="per_site", help="divide the intervals by N")
    ex.add_argument("--level", type=int, default=0, help="eigenvalue rank, 0 = lowest (default 0)")
    ex.add_argument("--no-verify", action="store_true", dest="no_verify", help="skip the integrity check on load")
    ex.add_argument("-o", "--output", help="write the Markdown report here (default: stdout)")
    ex.add_argument("--json", help="write the JSON record here")
    ex.set_defaults(fn=cmd_extrapolate)
    vw = sub.add_parser("view", help="offline HTML page of a saved certificate (self-contained, no scripts)")
    vw.add_argument("certificate", help="certificate JSON written by Certificate.save / certify-ed certify")
    vw.add_argument("-o", "--output", help="HTML file (default: the certificate's stem + .html)")
    vw.add_argument("--max-pairs", type=int, dest="max_pairs", default=None,
                    help="show at most this many rows of the enclosure table (default: all)")
    vw.set_defaults(fn=cmd_view)
    vf = sub.add_parser("verify", help="integrity check of a saved certificate; optionally run an external verifier "
                        "and record its verdict apart (no status changes)")
    vf.add_argument("certificate", help="certificate JSON written by Certificate.save / certify-ed certify")
    vf.add_argument("--no-verify", action="store_true", dest="no_verify", help="skip the integrity check")
    vf.add_argument("--external", help="external verifier command template (placeholders {certificate}, {sidecar}, "
                    "{bundle}, {operator}, {report}; default: $CERTIFY_ED_EXTERNAL_VERIFIER)")
    vf.add_argument("--bundle", help="producer bundle directory, for {bundle}")
    vf.add_argument("--operator", help="operator file, for {operator}")
    vf.add_argument("--sidecar", help="array sidecar, for {sidecar} (default: the certificate's own)")
    vf.add_argument("--report", help="file for the verifier's JSON report, for {report}")
    vf.add_argument("--record", help="external-verification record (default: CERT.external.json)")
    vf.add_argument("--timeout", type=float, help="seconds before the external verifier is stopped (TIMEOUT)")
    vf.set_defaults(fn=cmd_verify)
    pr = sub.add_parser("presets", help="the presets and statements table")
    pr.add_argument("--markdown", action="store_true")
    pr.set_defaults(fn=cmd_presets)
    cf = sub.add_parser("conform", help="conformance of producer bundles: schema, import, certify, certificate schema")
    cf.add_argument("bundles", nargs="+", help="bundle directories (conformance/reference/* are the references)")
    cf.add_argument("--allow-code", action="store_true", dest="allow_code", help="allow operator.py to be executed")
    cf.add_argument("--require-jsonschema", action="store_true", dest="require_jsonschema",
                    help="fail instead of falling back when the jsonschema package is missing")
    cf.add_argument("-o", "--output", help="save the certificate here (single bundle)")
    cf.add_argument("--json", help="write the report(s) as JSON to this file")
    cf.set_defaults(fn=cmd_conform)
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
    r.add_argument("--resume", action="store_true",
                   help="with --out: keep steps that already succeeded with the same computational source")
    r.add_argument("--jobs", type=int, default=1,
                   help="experiments to run at once (E2, E6, E8 always run alone); one BLAS thread each")
    r.add_argument("--manuscript"); r.set_defaults(fn=cmd_reproduce)
    n = sub.add_parser("numbers", help="regenerate numbers and tables from a run directory")
    n.add_argument("--run"); n.add_argument("--manuscript"); n.set_defaults(fn=cmd_numbers)
    a = p.parse_args(argv)
    from .precision import PrecisionError
    try:
        return a.fn(a)
    except PrecisionError as e:              # one rule for every command: data binary64 cannot hold exactly, or not
        print(f"{a.cmd}: refused: {e}")      # numeric data; nothing is written (precision.py)
        return 2


if __name__ == "__main__":
    sys.exit(main())
