"""Regression tests for the fixes of the external review of 2026-10-21 (ids IMP-xx, SS-xx, MS-Nxx refer to
REVIEW_CERTIFY-ED_2026-10-21). Each test names the finding it closes; the attack cases of the review
are in test_adversarial.py."""
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from conftest import needs_flint, eigh_dense


def _st(cert):
    return {s["id"]: s["status"] for s in cert.statements}


def _stmt(cert, sid):
    return next(s for s in cert.statements if s["id"] == sid)


@pytest.fixture(scope="module")
def heis6():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


# ------------------------------------------------------------------ IMP-18: zero test vector in the exact layer
def test_imp18_radius_exact_returns_none_for_zero_vector(heis6):
    from certify_ed.checker import exact_matrix, radius_exact, _float_vector_to_int
    spec, H, w, V = heis6
    ex = exact_matrix(spec)
    yr, yi, _ = _float_vector_to_int(np.zeros(H.shape[0]))
    assert radius_exact(ex, yr, yi, w[0]) is None
    yr, yi, _ = _float_vector_to_int(V[:, 0])
    assert radius_exact(ex, yr, yi, w[0]) is not None


def test_imp18_exact_radii_report_zero_vectors(heis6):
    spec, H, w, V = heis6
    X = np.column_stack([V[:, 0], np.zeros(H.shape[0])])
    r = ce.exact_eigenpair_radii(spec, [w[0], w[0]], X)
    assert r["zero_test_vectors"] == [1]
    assert math.isinf(r["radii"][1]) and r["radii"][0] < 1e-12


def test_imp18_exact_recheck_lowest_inconclusive_for_zero_vector(heis6):
    spec, H, w, V = heis6
    r = ce.exact_recheck_lowest(spec, np.array([w[0] + 0.05]), np.zeros((H.shape[0], 1)))
    assert r["status"] == "INCONCLUSIVE" and "x != 0" in r["reason"]


@needs_flint
def test_imp18_certify_S6_inconclusive_for_zero_vector(heis6):
    spec, H, w, V = heis6
    c = ce.certify(ce.Claim(np.array([w[0] + 0.05]), np.zeros((H.shape[0], 1)), operator=H, kind="lowest"),
                   spec, ce.Options(exact_recheck=True, inertia=False))
    s6 = _stmt(c, "S6.exact_recheck")
    assert s6["status"] == "INCONCLUSIVE" and "zero test vector" in s6["summary"]


def test_imp18_exact_sector_recheck_inconclusive_for_wrong_sector_vector():
    """A k=pi eigenvector claimed in the k=0 sector: its exact projection onto k=0 is zero."""
    from certify_ed.basis import translation, cyclic_isometry
    spec = ce.heisenberg_chain(10)
    st = ce.u1_state_set(spec, 5)
    H = ce.build_bitwise(spec, st)
    T = translation(spec)
    V0, Vpi = cyclic_isometry(st, T, 0), cyclic_isometry(st, T, 5)
    w0 = la.eigvalsh((V0.conj().T @ H @ V0).toarray())
    wpi, Ypi = la.eigh((Vpi.conj().T @ H @ Vpi).toarray())
    x = np.asarray(np.real_if_close(Vpi @ Ypi[:, :1]))
    r = ce.exact_recheck_sector_lowest(spec, st, [T], {0: 0}, np.array([w0[0] + 0.01]), x, next_hint=w0[1])
    assert r["status"] == "INCONCLUSIVE" and r.get("zero_test_vectors") == [0]


# ------------------------------------------------------------------ IMP-21 / IMP-22: non-finite claimed values
def test_imp22_nan_eigenvalue_in_certify_is_never_certified(heis6):
    spec, H, w, V = heis6
    mu = w.copy()
    mu[0] = np.nan
    c = ce.certify(ce.Claim(mu, V, operator=H, kind="all"), spec)
    s = _st(c)
    assert s["S2.eigenpairs"] == "INCONCLUSIVE" and s["S3.completeness"] == "INCONCLUSIVE"
    assert _stmt(c, "S2.eigenpairs")["evidence"]["non_finite_eigenvalues"] == [0]
    c = ce.certify(ce.Claim(mu[:2], V[:, :2], operator=H, kind="lowest"), spec, ce.Options(exact_recheck=True))
    assert all(v != "CERTIFIED" for k, v in _st(c).items() if k != "S1.model_fidelity")


def test_imp22_nan_eigenvector_in_certify_is_never_certified(heis6):
    spec, H, w, V = heis6
    X = V[:, :3].copy()
    X[5, 1] = np.inf
    c = ce.certify(ce.Claim(w[:3], X, operator=H, kind="lowest"), spec)
    s = _st(c)
    assert s["S2.eigenpairs"] == "INCONCLUSIVE" and s["S3.completeness"] == "INCONCLUSIVE"
    assert _stmt(c, "S2.eigenpairs")["evidence"]["non_finite_eigenvector_columns"] == [1]


def test_imp21_enclose_and_enclose_spectrum_reject_non_finite(heis6):
    spec, H, w, V = heis6
    mu = w.copy()
    mu[0] = np.nan
    with pytest.raises(ValueError):
        ce.enclose(ce.OperatorAdapter(H), mu, V)
    with pytest.raises(ValueError):
        ce.enclose_spectrum(H.toarray(), eigh_result=(mu, V))
    Xb = V.copy()
    Xb[0, 0] = np.nan
    with pytest.raises(ValueError):
        ce.enclose(ce.OperatorAdapter(H), w, Xb)
    Hb = H.toarray().copy()
    Hb[0, 0] = np.nan
    with pytest.raises(ValueError):
        ce.enclose_spectrum(Hb)


def test_imp22_accuracy_comparison_is_fail_safe(heis6, monkeypatch):
    """A NaN radius reaching the accuracy comparison must not leave S2 CERTIFIED (the comparison is written
    as `not (radius <= target)`)."""
    cmod = sys.modules["certify_ed.certify"]
    spec, H, w, V = heis6
    real_enclose = cmod.enclose

    def nan_enclose(adapter, mu, X, **kw):
        enc = real_enclose(adapter, mu, X, **kw)
        enc.radius[0] = np.nan
        return enc
    monkeypatch.setattr(cmod, "enclose", nan_enclose)
    c = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="subset"), spec)
    assert _st(c)["S2.eigenpairs"] != "CERTIFIED"


def test_imp23_zero_eigenvector_in_lowest_claim_is_inconclusive_not_a_crash(heis6):
    spec, H, w, V = heis6
    X = V[:, :2].copy()
    X[:, 1] = 0.0
    c = ce.certify(ce.Claim(w[:2], X, operator=H, kind="lowest"), spec)
    s = _st(c)
    assert s["S2.eigenpairs"] != "CERTIFIED" and s["S3.completeness"] == "INCONCLUSIVE"
    assert _stmt(c, "S3.completeness")["evidence"]["pairs_without_enclosure"] == [1]


# ------------------------------------------------------------------ IMP-13: sharper cluster radius (Theorem 2') and S2 status
def _degenerate_pair(heis6):
    spec, H, w, V = heis6
    k = int(np.argmax(np.abs(np.diff(w)) < 1e-10))
    return spec, H, w, V, k


def test_imp13_valid_nonorthogonal_degenerate_vectors_certified(heis6):
    """Two exact eigenvectors of one degenerate level with overlap 1/2 (eta ~ 0.71): Theorem 2' gives a cluster
    radius of the order of the residual where Theorem 2 gave ~5; the claim is CERTIFIED (was FAILED)."""
    spec, H, w, V, k = _degenerate_pair(heis6)
    X = np.column_stack([V[:, k], 0.5 * V[:, k] + np.sqrt(0.75) * V[:, k + 1]])
    c = ce.certify(ce.Claim(np.array([w[k], w[k + 1]]), X, operator=H, kind="subset"), spec)
    s2 = _stmt(c, "S2.eigenpairs")
    assert s2["status"] == "CERTIFIED", s2["summary"]
    g = s2["evidence"]["groups"][0]
    assert g["bound"]["used"] == "eps_star" and g["bound"]["eps_star"] < 1e-12 < g["bound"]["eps"]
    assert s2["evidence"]["max_radius"] < 1e-12


def test_imp13_cluster_radius_is_min_of_both_bounds(heis6):
    spec, H, w, V = heis6
    enc = ce.enclose(ce.OperatorAdapter(H), w, V)
    assert enc.all_distinct
    for g in enc.groups:
        if g.bound is not None:
            assert g.radius == min(g.bound["eps"], g.bound["eps_star"])
            assert g.bound["eps_star"] >= 0 and g.bound["eps"] >= 0
    assert np.all(enc.existence_radius <= enc.radius)
    recs = enc.to_records()
    assert all("existence_radius" in r for r in recs)


def test_imp13_theorem2_prime_radius_contains_distinct_eigenvalues():
    """Random Hermitian matrices, perturbed non-orthonormal vectors of a near-degenerate cluster: g distinct
    eigenvalues lie within the cluster radius of the claimed values (bottleneck matching)."""
    import itertools
    rng = np.random.default_rng(11)
    viol = 0
    for _ in range(60):
        n, g = 7, 3
        A = rng.normal(size=(n, n)); A = (A + A.T) / 2
        w, V = la.eigh(A)
        w[1:g] = w[0] + rng.choice([0.0, 1e-6, 1e-3]) * rng.normal(size=g - 1)
        A = V @ np.diag(w) @ V.T; A = (A + A.T) / 2
        w, V = la.eigh(A)
        C = np.eye(g) + rng.choice([1e-2, 0.2, 0.4]) * rng.normal(size=(g, g))
        X = V[:, :g] @ C
        X /= np.linalg.norm(X, axis=0)
        mu = w[:g] + rng.choice([0.0, 1e-7, 1e-4]) * rng.normal(size=g)
        enc = ce.enclose(ce.OperatorAdapter(A), mu, X)
        for grp in enc.groups:
            if not grp.distinct or grp.size < 2:
                continue
            best = min(max(abs(mu[m] - w[j]) for m, j in zip(grp.members, inj))
                       for inj in itertools.permutations(range(n), grp.size))
            viol += best > grp.radius
    assert viol == 0


def test_imp13_inconclusive_not_failed_when_only_the_defect_misses_the_target(heis6):
    """Existence radii within the target, cluster radius beyond it (overlap 1/2 and a 6e-8 offset): S2 is
    INCONCLUSIVE with the diagnosis, not FAILED."""
    spec, H, w, V, k = _degenerate_pair(heis6)
    X = np.column_stack([V[:, k], 0.5 * V[:, k] + np.sqrt(0.75) * V[:, k + 1]])
    mu = np.array([w[k], w[k + 1] + 6e-8])
    c = ce.certify(ce.Claim(mu, X, operator=H, kind="subset", accuracy_target=1e-7), spec)
    s2 = _stmt(c, "S2.eigenpairs")
    assert s2["status"] == "INCONCLUSIVE", s2["summary"]
    assert "orthonormality defect" in s2["evidence"]["diagnosis"]
    assert s2["evidence"]["max_existence_radius"] <= 1e-7 < s2["evidence"]["max_radius"]
    # an inaccurate claim (existence radius beyond the target) is still FAILED
    mu = np.array([w[k], w[k + 1] + 3e-7])
    c = ce.certify(ce.Claim(mu, X, operator=H, kind="subset", accuracy_target=1e-7), spec)
    assert _st(c)["S2.eigenpairs"] == "FAILED"


def test_imp12_compose_uses_rho_for_square_W():
    """compose.certify_low_energy: eps_decomposition = rho + builder shift (Weyl for the square W), no longer
    (1 + sqrt 5)/2 * rho; the certified statement still holds."""
    from certify_ed.compose import SectorResult, certify_low_energy
    from certify_ed.construct import full_state_set
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    wf = la.eigvalsh(H.toarray())
    E_cut = float((wf[4] + wf[5]) / 2)
    full = full_state_set(spec)
    secs = []
    for nup in range(spec.n_sites + 1):
        st = ce.u1_state_set(spec, nup)
        idx, found = full.index_of(st.states_in_order())
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        Hs = (Vs.T @ H @ Vs).tocsr()
        ws, Ys = eigh_dense(Hs.toarray())
        sel = np.where(ws < E_cut)[0]
        secs.append(SectorResult(f"nup={nup}", Vs, Hs, ws[sel], Ys[:, sel]))
    res = certify_low_energy(spec, secs, E_cut)
    assert res["status"] == "CERTIFIED"
    eta = res["eta"]
    delta = 1 - math.sqrt(1 - eta)
    # eps is at most rho + builder shift with a rounding guard, well below the golden-ratio form
    assert res["eps_decomposition"] <= 1.001 * (res["invariance_total"] + 20 * delta + res["builder_shift"]) + 1e-300
    K = res["K"]
    assert np.max(np.abs(np.array(res["lowest"]) - wf[:K])) <= res["radius"]
    assert res["next_lower_bound"] <= wf[K]


# ------------------------------------------------------------------ IMP-08: inward rounding of the certified window
def test_imp08_certified_window_endpoints_inside_exact_range():
    from fractions import Fraction
    rng = np.random.default_rng(7)
    N = 6
    from certify_ed.localops import LocalSpace
    from certify_ed.spec import ModelSpec
    static = [["zz", [[float(rng.normal()), i, (i + 1) % N] for i in range(N)]],
              ["x", [[float(rng.normal()), i] for i in range(N)]],
              ["z", [[float(0.3 * rng.normal()), i] for i in range(N)]]]
    spec = ModelSpec.from_static(N, LocalSpace.spin_space("1/2", "spin"), static)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    n_cert, bad = 0, []
    for t in np.linspace(0.11, 0.89, 15):
        a = float(w[0] + t * (w[1] - w[0]))
        b = float(w[2] + t * (w[3] - w[2]))
        c = ce.certify(ce.Claim(w[1:3], V[:, 1:3], operator=H, kind="window", window=(a, b)), spec)
        if _st(c)["S3.completeness"] != "CERTIFIED":
            continue
        n_cert += 1
        ev = _stmt(c, "S3.completeness")["evidence"]
        lo, hi = ev["certified_window"]
        if Fraction(lo) < Fraction(a) + Fraction(ev["beta_a"]) or Fraction(hi) > Fraction(b) - Fraction(ev["beta_b"]):
            bad.append((a, b, lo, hi))
        for k in c.body["kato_temple"]:
            assert k["isolation_interval"][0] >= lo and k["isolation_interval"][1] <= hi
    assert n_cert > 0 and not bad


# ------------------------------------------------------------------ IMP-14: triangularity of the dense LDL factor
def test_imp14_inertia_dense_rejects_a_non_triangular_factor(monkeypatch):
    import certify_ed.inertia as im
    A = np.diag([1.0, 2.0, -3.0])
    assert im.inertia_dense(A, 0.0).negative == 1
    real_ldl = im.la.ldl

    def bad_ldl(S, **kw):
        lu, d, perm = real_ldl(S, **kw)
        lu = lu.copy()
        lu[0, 2] = 0.5                     # an entry above the diagonal: not a triangular factor
        return lu, d, perm
    monkeypatch.setattr(im.la, "ldl", bad_ldl)
    r = im.inertia_dense(A, 0.0)
    assert not r.conclusive and "not lower triangular" in r.note


# ------------------------------------------------------------------ IMP-20: compose records symmetrised producer blocks
def test_imp20_compose_records_symmetrised_blocks():
    from certify_ed.compose import SectorResult, certify_low_energy
    from certify_ed.construct import full_state_set
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec)
    wf = la.eigvalsh(H.toarray())
    E_cut = float((wf[0] + wf[1]) / 2)          # between levels ((wf[2] + wf[3]) / 2 is the triplet level itself)
    full = full_state_set(spec)
    secs = []
    for nup in range(spec.n_sites + 1):
        st = ce.u1_state_set(spec, nup)
        idx, found = full.index_of(st.states_in_order())
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        Hs = (Vs.T @ H @ Vs).tolil()
        if nup == 2:
            Hs[0, 1] = Hs[0, 1] + 1e-13        # a producer block that is not exactly Hermitian
        Hs = Hs.tocsr()
        Hh = ((Hs + Hs.conj().T) / 2).toarray()
        ws, Ys = eigh_dense(Hh)
        sel = np.where(ws < E_cut)[0]
        secs.append(SectorResult(f"nup={nup}", Vs, Hs, ws[sel], Ys[:, sel]))
    res = certify_low_energy(spec, secs, E_cut)
    assert res["symmetrised_blocks"] == ["nup=2"]
    rec = next(p for p in res["sectors"] if p["label"] == "nup=2")
    assert rec["hermitian_part"]["max_abs_asymmetry"] > 0
    assert all(p["hermitian_part"] is None for p in res["sectors"] if p["label"] != "nup=2")


# ------------------------------------------------------------------ IMP-05 / SS-02: dirtiness ignores the executable bit
def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _git_scratch(tmp_path):
    """A directory git can work in: tmp_path, or (environments whose temp dir git cannot stat) one next to the repo."""
    import tempfile
    if subprocess.run(["git", "init", "-q"], cwd=tmp_path, capture_output=True).returncode == 0:
        shutil.rmtree(tmp_path / ".git")
        return tmp_path, None
    base = tempfile.mkdtemp(prefix="gitscratch_", dir=os.path.dirname(os.path.dirname(os.path.abspath(ce.__file__))))
    if subprocess.run(["git", "init", "-q"], cwd=base, capture_output=True).returncode != 0:
        # 3.2.3: a copy of the repository inside the temp directory (E6 of the 3.2.2 run) leaves git no place to work
        shutil.rmtree(base, ignore_errors=True)
        pytest.skip("git cannot create a repository in the temp directory nor next to the package")
    shutil.rmtree(os.path.join(base, ".git"))
    return base, base


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_imp05_source_fingerprint_ignores_lost_executable_bit(tmp_path):
    from certify_ed.provenance import source_fingerprint
    from pathlib import Path
    base, cleanup = _git_scratch(tmp_path)
    try:
        _imp05_body(Path(base) / "r", source_fingerprint)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


def _imp05_body(root, source_fingerprint):
    (root / "experiments").mkdir(parents=True)
    (root / "certify_ed").mkdir()
    (root / "experiments" / "run_all.sh").write_text("#!/bin/sh\necho hi\n")
    (root / "certify_ed" / "x.py").write_text("x = 1\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.org")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "core.filemode", "true")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "a")
    # the committed mode says executable, the working tree (or a Windows copy) does not: content unchanged
    _git(root, "update-index", "--chmod=+x", "experiments/run_all.sh")
    _git(root, "commit", "-q", "-m", "b")
    try:
        os.chmod(root / "experiments" / "run_all.sh", 0o644)
    except OSError:
        pass
    plain = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
    fp = source_fingerprint(str(root))
    assert fp["source_dirty"] is False, (plain, fp)
    (root / "certify_ed" / "x.py").write_text("x = 2\n")
    assert source_fingerprint(str(root))["source_dirty"] is True


# ------------------------------------------------------------------ IMP-06: the numbers step records its provenance
def test_imp06_numbers_fingerprint_records_commit_and_script_blobs():
    from certify_ed.provenance import numbers_fingerprint, repo_root, NUMBER_SCRIPTS
    if repo_root() is None or not os.path.isdir(os.path.join(repo_root(), ".git")):
        pytest.skip("needs a git checkout")
    fp = numbers_fingerprint()
    assert fp["git_commit"] and len(fp["git_commit"]) == 40
    for rel in NUMBER_SCRIPTS:
        s = fp["scripts"][rel]
        assert s["blob"] and len(s["blob"]) == 40 and s["sha256"] and len(s["sha256"]) == 64
        assert s["modified"] in (True, False)
    assert fp["source_hash"] and fp["tests_hash"]


# ------------------------------------------------------------------ IMP-26: version and source hash in every certificate
def test_imp26_version_and_source_hash(heis6):
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(ce.__file__)))
    with open(os.path.join(root, "pyproject.toml")) as f:
        m = re.search(r'^version = "([^"]+)"', f.read(), re.M)
    assert m and m.group(1) == ce.__version__          # one version, in pyproject and the package
    assert re.match(r"^3\.(1|2|3)\.", ce.__version__)
    spec, H, w, V = heis6
    c = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="lowest"), spec)
    env = c.body["environment"]
    assert env["certify_ed"] == ce.__version__
    assert "source_hash" in env and "git_commit" in env
    from certify_ed.provenance import source_fingerprint
    assert env["source_hash"] == source_fingerprint()["source_hash"]


# ------------------------------------------------------------------ SS-04: E3 needs QuSpin; the harness logs an abort
def test_ss04_needs_e3_includes_quspin():
    from certify_ed.cli import NEEDS
    assert "quspin" in NEEDS["E3"] and "flint" in NEEDS["E3"]


def test_ss04_harness_logs_aborted_on_exception(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(ce.__file__)))
    sys.path.insert(0, os.path.join(root, "experiments"))
    try:
        from harness import Run
    finally:
        sys.path.pop(0)
    with pytest.raises(RuntimeError):
        with Run("unit_abort", outdir=str(tmp_path)) as run:
            run.record("x", 1)
            raise RuntimeError("boom")
    log = open(os.path.join(run.dir, "run.log")).read()
    assert "ABORTED" in log and "finished" not in log
    import json
    d = json.load(open(os.path.join(run.dir, "data.json")))
    assert d["_aborted"].startswith("RuntimeError") and d["x"] == 1
    with Run("unit_ok", outdir=str(tmp_path)) as run:
        run.record("x", 1)
    assert "finished" in open(os.path.join(run.dir, "run.log")).read()
    env = json.load(open(os.path.join(run.dir, "environment.json")))
    assert "flint_c_library" in env and env["certify_ed"] == ce.__version__      # MS-N12


# ------------------------------------------------------------------ MS-N12: FLINT C-library version
def test_msn12_flint_c_version_recorded():
    from certify_ed.cli import flint_c_version, environment
    from certify_ed.certificate_v2 import environment_record
    pytest.importorskip("flint")
    v = flint_c_version()
    assert v not in ("not installed",) and (v == "unknown" or v[0].isdigit())
    assert environment()["flint_c_library"] == v
    assert environment_record()["flint_c_library"] == v


# ------------------------------------------------------------------ MS-N03: provenance stamps of the E6 post-run files
def test_msn03_recheck_stamp_has_hashes():
    root = os.path.dirname(os.path.dirname(os.path.abspath(ce.__file__)))
    sys.path.insert(0, os.path.join(root, "experiments"))
    try:
        from e6_recheck import provenance_stamp
    finally:
        sys.path.pop(0)
    st = provenance_stamp()
    assert set(st) >= {"git_commit", "source_hash", "tests_hash", "written"}


# ------------------------------------------------------------------ number scripts: helpers (IMP-07, IMP-27, IMP-28)
def _script_functions(name, *funcs):
    """Load selected top-level functions of an experiments/ script without running it."""
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(ce.__file__)))
    src = open(os.path.join(root, "experiments", name)).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
            and (not isinstance(n, ast.FunctionDef) or n.name in funcs)]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), name, "exec"), ns)
    return ns


def test_imp27_printed_excess_equals_checked_value():
    ns = _script_functions("make_numbers_apps.py", "sci_dir", "sci_dir_value", "_sci_dir_parts", "dec_dir")
    from fractions import Fraction
    for x in (1.9073e-9, 2.3386e-11, 4.9e-9, 5.0e-9, 9.96e-10):
        s, v = ns["sci_dir"](x, 0, True), ns["sci_dir_value"](x, 0, True)
        assert Fraction(x) <= v                       # the printed bound is an upper bound
        m, e = s.replace("\\ensuremath{", "").rstrip("}").split("\\times10^{")
        assert Fraction(m) * Fraction(10) ** int(e) == v


def test_imp28_range_rounded_outward():
    ns = _script_functions("make_numbers_apps.py", "dec_dir")
    cvs = [0.44417208809354597, 0.84816, 0.59922, 0.61214, 0.8814172772291874]
    lo, hi = ns["dec_dir"](min(cvs), 2, False), ns["dec_dir"](max(cvs), 2, True)
    assert (lo, hi) == ("0.44", "0.89")
    assert all(float(lo) <= v <= float(hi) for v in cvs)


def test_imp07_test_counts_come_from_the_run_manifest(tmp_path, monkeypatch):
    import json
    pytest.importorskip("matplotlib")     # make_numbers.py imports it at top level (the 'plots' extra)
    ns = _script_functions("make_numbers.py", "run_test_counts")
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"steps": {"tests": {"summary": "363 passed, 2 skipped, 1 warning in 79.43s (0:01:19)"}}}))
    ns["RES"] = str(tmp_path)
    ns["os"], ns["json"] = os, json
    tc = ns["run_test_counts"]()
    assert tc == {"summary": "363 passed, 2 skipped, 1 warning in 79.43s (0:01:19)", "passed": 363, "failed": 0,
                  "skipped": 2, "total": 365}
    ns["RES"] = str(tmp_path / "nowhere")
    assert ns["run_test_counts"]() is None
