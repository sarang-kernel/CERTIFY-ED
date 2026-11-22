"""Non-finite bounds of the floating-point accounting (3.3.0.dev0; UNVERIFIED, D-12).

A product, a sum or a modulus of the accounting can overflow for operator entries near the largest float: the bound is
inf, or nan where an inf meets a zero (inf * 0) or another inf (inf - inf). Each entry point below must then state
nothing: no CERTIFIED statement on such a bound and no nan in a record. The inputs overflow with np.abs as well: the real
entries 1.7e308 and one step below the largest float, and the complex entry 1.5e308 (1 + i), whose modulus exceeds the
largest float. A claimed vector whose norm binary64 does not resolve (1e-170 or 1e170 times a unit vector) has its own
reason code. Controls with an entry of 1e100 are certified.
"""
import types

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.diagnostics import next_action
from certify_ed.eigenspace import EigenspaceCertificate, certify_observable
from certify_ed.inertia import inertia_dense, inertia_sparse
from certify_ed.moments import certified_moments
from certify_ed.variational import certify_variational
from conftest import eigh_dense

BIG = 1.7e308
NMAX = float(np.nextafter(np.finfo(float).max, 0))
Z_OVF = complex(1.5e308, 1.5e308)
E0 = np.eye(3)[:, :1]
CASES = [("real", BIG), ("near-max", NMAX), ("complex", Z_OVF)]


def _operator(z):
    """diag(1, 2, 3) with the entry z at (1, 2) and its conjugate at (2, 1), and a unit vector that meets z (the upper
    eigenvector of the lower 2 x 2 block; e0 for the complex entry, whose block eigenvectors are not finite)."""
    H = np.diag([1.0, 2.0, 3.0]).astype(complex if isinstance(z, complex) else float)
    H[1, 2], H[2, 1] = z, np.conj(z)
    if isinstance(z, complex):
        return H, E0.astype(complex)
    _w, V = np.linalg.eigh(H[1:, 1:])
    x = np.zeros((3, 1))
    x[1:, 0] = V[:, 1]
    return H, x


def _by_id(statements):
    return {s["id"]: s for s in statements}


def _no_nan(cert):
    for key, v in cert.arrays.items():
        a = np.asarray(v)
        assert not (a.dtype.kind in "fc" and np.isnan(a).any()), key


def test_a_sum_rule_verdict_on_a_nan_endpoint_is_inconclusive():
    """moments._verdict: overlap -> CERTIFIED, disjoint -> FAILED, a nan endpoint -> INCONCLUSIVE (a disjointness found
    on a nan is not a finding)."""
    from certify_ed.moments import _verdict
    assert _verdict(True, 0.0, 1.0) == "CERTIFIED" and _verdict(False, 0.0, 1.0) == "FAILED"
    assert _verdict(False, float("nan"), 1.0) == _verdict(True, 0.0, float("nan")) == "INCONCLUSIVE"


def test_sum_rule_texts_do_not_report_a_disjointness_on_nan():
    from certify_ed.moments import _rule_text
    assert _rule_text(True, "CERTIFIED") == "overlap (consistent)"
    assert _rule_text(False, "FAILED").startswith("are DISJOINT")
    assert "nan" in _rule_text(False, "INCONCLUSIVE") and "DISJOINT" not in _rule_text(False, "INCONCLUSIVE")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name,z", CASES)
def test_variational_moment_and_observable_records_state_nothing(name, z):
    """V1 to V3 INCONCLUSIVE with NONFINITE_BOUND (before: V1 and V2 CERTIFIED with a nan bound); no moment statement
    CERTIFIED on a nan interval and no sum rule FAILED on one; an observable bound of inf, not nan."""
    H, x = _operator(z)
    st = _by_id(certify_variational(sp.csr_matrix(H), x[:, 0])["statements"])
    for sid in ("V1.rayleigh_upper", "V2.existence", "V3.temple_lower"):
        assert st[sid]["status"] == "INCONCLUSIVE" and st[sid]["evidence"]["reason_code"] == "NONFINITE_BOUND", name
    rec = certified_moments(sp.csr_matrix(np.diag([1.0, 2.0, 3.0])), sp.csr_matrix(H), x, 0.0, 0.0, K=1,
                            level=(0.5, 3.5))
    st = _by_id(rec["statements"])
    for sid in ("M2.sum_rule_zeroth", "M3.sum_rule_first"):
        assert st[sid]["status"] == "INCONCLUSIVE" and st[sid]["evidence"]["reason_code"] == "NONFINITE_BOUND", name
    m1 = st["M1.moments"]
    if m1["status"] == "CERTIFIED":
        assert all(np.isfinite(m["interval"]).all() for m in m1["evidence"]["moments"]), name
    else:
        assert m1["status"] == "INCONCLUSIVE" and m1["evidence"]["reason_code"] == "NONFINITE_BOUND", name
    ec = EigenspaceCertificate([0], [1.0], 1.0, 0.0, 0.0, 0.0, 0.0)
    assert certify_observable(ec, x, sp.csr_matrix(H), "O")["bound"] == np.inf


def test_controls_with_a_large_finite_entry_are_certified():
    H, x = _operator(1e100)
    st = _by_id(certify_variational(sp.csr_matrix(H), x[:, 0])["statements"])
    assert st["V1.rayleigh_upper"]["status"] == st["V2.existence"]["status"] == "CERTIFIED"
    r = inertia_dense(H, 0.5)
    assert r.conclusive and np.isfinite(r.beta) and r.negative == 1          # eigenvalues near -1e100, 1, 1e100
    claim = ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=E0, operator=sp.csr_matrix(H), kind="subset")
    assert ce.certify(claim, None, ce.Options(seed=1)).statement("S2.eigenpairs")["status"] == "CERTIFIED"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_s2_and_s3_say_that_the_radius_is_not_finite():
    """S2 and S3 on an overflowed bound: INCONCLUSIVE with NONFINITE_BOUND, a summary that says the radius is not
    finite (not that an eigenvalue lies within it), the advice to rescale the operator, and no nan in the certificate
    (before: a nan radius, and for kind='lowest' a ValueError of the factorization on the real entries)."""
    H, x = _operator(BIG)
    cert = ce.certify(ce.Claim(eigenvalues=np.array([np.linalg.eigvalsh(H[1:, 1:])[1]]), eigenvectors=x,
                               operator=sp.csr_matrix(H), kind="subset"), None, ce.Options(seed=1))
    s2 = cert.statement("S2.eigenpairs")
    assert s2["status"] == "INCONCLUSIVE" and s2["evidence"]["reason_code"] == "NONFINITE_BOUND"
    assert "is not finite" in s2["summary"] and "within its radius" not in s2["summary"]
    assert "rescale the operator" in next_action(s2)
    Hc, e0 = _operator(Z_OVF)
    cert = ce.certify(ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=e0, operator=sp.csr_matrix(Hc),
                               kind="lowest"), None, ce.Options(seed=1))
    s2, s3 = cert.statement("S2.eigenpairs"), cert.statement("S3.completeness")
    assert s2["status"] == s3["status"] == "INCONCLUSIVE"
    assert s2["evidence"]["reason_code"] == s3["evidence"]["reason_code"] == "NONFINITE_BOUND"
    assert "overflowed" in s3["summary"] and s2["evidence"]["max_radius"] == np.inf
    _no_nan(cert)
    cert = ce.certify(ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=E0, operator=sp.csr_matrix(H),
                               kind="lowest"), None, ce.Options(seed=1))
    assert cert.statement("S3.completeness")["status"] == "INCONCLUSIVE"
    zero = ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=np.zeros((3, 1)), operator=sp.csr_matrix(np.eye(3)),
                    kind="lowest")
    s3z = ce.certify(zero, None, ce.Options(seed=1)).statement("S3.completeness")
    assert s3z["summary"].startswith("some claimed pair has no finite enclosure (zero eigenvector)")
    assert "reason_code" not in s3z["evidence"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("scale", [1e-170, 1e170])
def test_an_unresolvable_vector_norm_is_not_an_overflow(scale):
    """A nonzero claimed vector whose norm binary64 does not resolve (its squared norm underflows or overflows, so
    fp.norm2_lower gives 0) has an infinite radius for any operator: UNRESOLVED_VECTOR_NORM and the advice to normalise
    the vectors, not NONFINITE_BOUND and the advice to rescale the operator; the unit vector is certified."""
    H = sp.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    cert = ce.certify(ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=E0 * scale, operator=H, kind="lowest"),
                      None, ce.Options(seed=1))
    for sid in ("S2.eigenpairs", "S3.completeness"):
        s = cert.statement(sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "UNRESOLVED_VECTOR_NORM", sid
        assert "not resolvable" in s["summary"] and "overflowed" not in s["summary"], sid
        assert "normalise" in next_action(s) and "rescale the operator" not in next_action(s), sid
    assert cert.statement("S2.eigenpairs")["evidence"]["pairs_with_unresolved_vector_norm"] == [0]
    cert = ce.certify(ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=E0, operator=H, kind="lowest"), None,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == cert.statement("S3.completeness")["status"] == "CERTIFIED"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name,z", CASES)
def test_inertia_dense_gives_no_count_on_a_non_finite_backward_error(name, z):
    H, _x = _operator(z)
    r = inertia_dense(H, 0.5)
    assert r.negative is None and r.beta == np.inf and not r.conclusive and "not finite" in r.note, name


def test_inertia_sparse_gives_no_count_on_a_non_finite_backward_error(monkeypatch):
    """The guard of inertia_sparse, reached through a stand-in residual bound with beta = nan (no input of the test
    suite gives a non-finite sparse beta: the factorization fails before, 'Factor is exactly singular' or 'non-finite
    entries in the stored factors')."""
    import certify_ed.ldl_residual as ldl
    A = sp.csr_matrix(np.diag([1.0, 2.0, 3.0]))
    assert inertia_sparse(A, 0.5).conclusive
    stand_in = types.SimpleNamespace(beta=float("nan"), to_dict=lambda: {"beta": None})
    monkeypatch.setattr(ldl, "sparse_ldl_residual_bound", lambda *a, **k: stand_in)
    r = inertia_sparse(A, 0.5)
    assert r.negative is None and r.beta == np.inf and not r.conclusive and "not finite" in r.note


def _sz_sectors(spec, E_cut):
    from certify_ed.compose import SectorResult
    from certify_ed.construct import full_state_set
    H = ce.build_bitwise(spec)
    full = full_state_set(spec)
    secs = []
    for nup in range(spec.n_sites + 1):
        st = ce.u1_state_set(spec, nup)
        idx, _found = full.index_of(st.states_in_order())
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        Hs = (Vs.T @ H @ Vs).tocsr()
        ws, Ys = eigh_dense(Hs.toarray())
        sel = np.where(ws < E_cut)[0]
        secs.append(SectorResult(f"nup={nup}", Vs, Hs, ws[sel], Ys[:, sel]))
    return H, secs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("J,certified", [(1e160, False), (1.0, True)])
def test_sector_audits_pass_nothing_on_a_non_finite_defect(J, certified):
    """Couplings of 1e160: the invariance defect of the sector audit overflows (inf <= tol * inf held before).
    audit_sector does not pass, S1 on a sector isometry is INCONCLUSIVE (NONFINITE_BOUND), S2 makes no accuracy
    statement (NONFINITE_BOUND: the sector lift shift is infinite), and the sector composition is INCONCLUSIVE, not
    FAILED; with J = 1 all of them are certified."""
    from certify_ed.compose import certify_low_energy
    spec = ce.heisenberg_chain(4, J=J)
    H = ce.build_bitwise(spec).tocsr()
    w, X = np.linalg.eigh(H.toarray())
    claim = ce.Claim(eigenvalues=w[:2], eigenvectors=X[:, :2], operator=H, kind="lowest",
                     sector_isometry=sp.identity(16, format="csr"))
    cert = ce.certify(claim, spec, ce.Options(seed=1))
    s1, s2 = cert.statement("S1.model_fidelity"), cert.statement("S2.eigenpairs")
    assert s1["status"] == ("CERTIFIED" if certified else "INCONCLUSIVE")
    assert s1["evidence"]["sector_audit"]["passed"] == certified
    assert (s1["evidence"].get("reason_code") == "NONFINITE_BOUND") == (not certified)
    assert s2["status"] == ("CERTIFIED" if certified else "INCONCLUSIVE")
    assert (s2["evidence"].get("reason_code") == "NONFINITE_BOUND") == (not certified)
    # between levels (w[3] - w[2] is 0 here: the cut is half a gap above the ground state, not on the triplet at -|J|)
    E_cut = float((w[2] + w[3]) / 2) if w[3] - w[2] > 1e-9 * abs(J) else float(w[0] + abs(J) / 2)
    _, secs = _sz_sectors(spec, E_cut)
    res = certify_low_energy(spec, secs, E_cut)
    if certified:
        assert res["status"] == "CERTIFIED", res["summary"]
    else:
        assert res["status"] == "INCONCLUSIVE" and "not finite" in res["summary"]
