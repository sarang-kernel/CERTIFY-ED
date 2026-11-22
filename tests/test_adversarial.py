"""Adversarial attack cases against certify_ed (external review of 2026-10-21, test_adversarial_review.py).

Each test asserts the SAFE behaviour: a false claim must not be CERTIFIED, a correct claim must not be
FAILED, and every reported endpoint must lie on the safe side of the exact value. On the reviewed HEAD
(0507885b) 7 of the 14 cases failed (A1, A1b, A2: IMP-18; A3, A3b: IMP-21/22; A5: IMP-08; A8: IMP-13).
Adapted to the 3.1.0 behaviour in two places: A3b expects enclose_spectrum to refuse a non-finite claim
(ValueError) rather than return an enclosure; A8 additionally asserts the fixed status (CERTIFIED or
INCONCLUSIVE). The other 12 cases are unchanged.
"""
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.basis import translation, cyclic_isometry
from certify_ed.checker import exact_recheck_lowest
from certify_ed.exact_sectors import exact_recheck_sector_lowest
from certify_ed.spectrum import enclose_spectrum
from certify_ed.compose import SectorResult, certify_low_energy
from certify_ed.construct import full_state_set
from certify_ed.localops import LocalSpace
from certify_ed.spec import ModelSpec
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


@pytest.fixture(scope="module")
def rand_nondeg():
    """A small real model (random-coupling Ising chain in transverse and longitudinal fields) with a
    non-degenerate low spectrum."""
    rng = np.random.default_rng(7)
    N = 6
    static = [["zz", [[float(rng.normal()), i, (i + 1) % N] for i in range(N)]],
              ["x", [[float(rng.normal()), i] for i in range(N)]],
              ["z", [[float(0.3 * rng.normal()), i] for i in range(N)]]]
    spec = ModelSpec.from_static(N, LocalSpace.spin_space("1/2", "spin"), static)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    assert np.min(np.diff(w[:6])) > 1e-2
    return spec, H, w, V


# ------------------------------------------------------------------ A1: exact layer, zero test vector
def test_A1_exact_recheck_zero_vector_wrong_value_not_certified(heis6):
    """Krylov-Bogolyubov needs x != 0. A zero claimed vector must not give radius 0."""
    spec, H, w, V = heis6
    lam = np.array([w[0] + 0.05])                      # wrong by 0.05
    r = exact_recheck_lowest(spec, lam, np.zeros((H.shape[0], 1)))
    assert r["status"] != "CERTIFIED", r


def test_A1b_certify_S6_zero_vector_not_certified(heis6):
    spec, H, w, V = heis6
    c = ce.certify(ce.Claim(np.array([w[0] + 0.05]), np.zeros((H.shape[0], 1)), operator=H, kind="lowest"),
                   spec, ce.Options(exact_recheck=True, inertia=False))
    assert _st(c)["S6.exact_recheck"] != "CERTIFIED", _stmt(c, "S6.exact_recheck")["summary"]


# ------------------------------------------------------------------ A2: exact sector layer, wrong-sector vector
def test_A2_wrong_sector_vector_with_shifted_value_not_certified():
    """A k=pi eigenvector claimed as the k=0 ground state, value shifted by 0.01 from the true k=0
    ground energy: the exact projection onto k=0 is the zero vector."""
    spec = ce.heisenberg_chain(10)
    st = ce.u1_state_set(spec, 5)
    H = ce.build_bitwise(spec, st)
    T = translation(spec)
    V0, Vpi = cyclic_isometry(st, T, 0), cyclic_isometry(st, T, 5)
    w0 = la.eigvalsh((V0.conj().T @ H @ V0).toarray())
    wpi, Ypi = la.eigh((Vpi.conj().T @ H @ Vpi).toarray())
    x = np.asarray(np.real_if_close(Vpi @ Ypi[:, :1]))
    lam = np.array([w0[0] + 0.01])
    r = exact_recheck_sector_lowest(spec, st, [T], {0: 0}, lam, x, next_hint=w0[1])
    if r["status"] == "CERTIFIED":
        e = r["enclosures"][0]
        assert e["lower"] <= w0[0] <= e["upper"], (r["status"], e, w0[0])


# ------------------------------------------------------------------ A3: non-finite claimed value
def test_A3_nan_eigenvalue_full_spectrum_claim_not_certified(heis6):
    spec, H, w, V = heis6
    mu = w.copy()
    mu[0] = np.nan                                      # the (non-degenerate) ground state replaced by NaN
    c = ce.certify(ce.Claim(mu, V, operator=H, kind="all"), spec)
    s = _st(c)
    assert s["S2.eigenpairs"] != "CERTIFIED" and s["S3.completeness"] != "CERTIFIED", s


def test_A3b_nan_eigenvalue_spectrum_enclosure_not_complete(heis6):
    """3.1.0: a non-finite claimed value is refused at the entry (IMP-21); before, complete=True was returned
    with the ground state in no interval."""
    spec, H, w, V = heis6
    mu = w.copy()
    mu[0] = np.nan
    with pytest.raises(ValueError):
        enclose_spectrum(H.toarray(), eigh_result=(mu, V))


# ------------------------------------------------------------------ A4: duplicated vectors (FP and exact layers)
def test_A4_duplicated_pair_replacing_missing_level_fp(rand_nondeg):
    spec, H, w, V = rand_nondeg
    mu = np.array([w[0], w[1], w[1]])                   # w[2] missing, (w1, v1) duplicated
    X = V[:, [0, 1, 1]]
    c = ce.certify(ce.Claim(mu, X, operator=H, kind="lowest"), spec)
    s = _st(c)
    assert s["S2.eigenpairs"] != "CERTIFIED" and s["S3.completeness"] != "CERTIFIED", s


@needs_flint
def test_A4b_duplicated_pair_exact_layer(rand_nondeg):
    spec, H, w, V = rand_nondeg
    mu = np.array([w[0], w[1], w[1]])
    r = exact_recheck_lowest(spec, mu, V[:, [0, 1, 1]], next_hint=w[3])
    assert r["status"] != "CERTIFIED", r["status"]


# ------------------------------------------------------------------ A5: window endpoints (outward/inward rounding)
def test_A5_window_certified_endpoints_rounded_inward(rand_nondeg):
    """S3 'window' returns (a + beta_a, b - beta_b); both must be on the safe (inner) side of the exact
    values. Several windows are tried so that at least one sum rounds the unsafe way."""
    spec, H, w, V = rand_nondeg
    bad = []
    for t in np.linspace(0.11, 0.89, 15):
        a = float(w[0] + t * (w[1] - w[0]))
        b = float(w[2] + t * (w[3] - w[2]))
        c = ce.certify(ce.Claim(w[1:3], V[:, 1:3], operator=H, kind="window", window=(a, b)), spec)
        ev = _stmt(c, "S3.completeness")["evidence"]
        if _st(c)["S3.completeness"] != "CERTIFIED":
            continue
        lo, hi = ev["certified_window"]
        if Fraction(lo) < Fraction(a) + Fraction(ev["beta_a"]) or Fraction(hi) > Fraction(b) - Fraction(ev["beta_b"]):
            bad.append((a, b, lo, hi, ev["beta_a"], ev["beta_b"]))
    assert not bad, f"{len(bad)} certified windows extend beyond the exact certified range, e.g. {bad[0]}"


# ------------------------------------------------------------------ A6: model shift (values from a nearby matrix)
def test_A6_pairs_of_perturbed_matrix_not_certified_for_declared_model(rand_nondeg):
    spec, H, w, V = rand_nondeg
    rng = np.random.default_rng(3)
    P = rng.standard_normal(H.shape)
    Hp = H.toarray() + 1e-6 * (P + P.T)
    wp, Vp = la.eigh(Hp)
    c = ce.certify(ce.Claim(wp[:3], Vp[:, :3], operator=None, kind="lowest"), spec)
    s = _st(c)
    assert s["S2.eigenpairs"] != "CERTIFIED", s          # radius ~1e-6 > target 1e-8 ||H||
    c2 = ce.certify(ce.Claim(wp[:3], Vp[:, :3], operator=sp.csr_matrix(Hp), kind="lowest"), spec)
    assert _st(c2)["S1.model_fidelity"] == "FAILED", _st(c2)


# ------------------------------------------------------------------ A7: Kato-Temple with a missing neighbour
def test_A7_kato_temple_not_applied_when_a_level_between_claims_is_missing(rand_nondeg):
    spec, H, w, V = rand_nondeg
    idx = [0, 2, 3]                                     # w[1] missing: the isolation interval of w[0] is wrong
    c = ce.certify(ce.Claim(w[idx], V[:, idx], operator=H, kind="lowest"), spec)
    assert _st(c)["S3.completeness"] != "CERTIFIED"
    body = getattr(c, "body", None) or {}
    for k in body.get("kato_temple") or []:
        lo, hi = k["lower"], k["upper"]
        assert any(lo <= x <= hi for x in w), k


# ------------------------------------------------------------------ A8: correct claim, non-orthogonal degenerate vectors
def test_A8_valid_nonorthogonal_degenerate_vectors_not_failed(heis6):
    """Two exact eigenvectors of a degenerate level with overlap 1/2 (Frobenius eta ~ 0.71 < 1): every
    claimed pair is accurate, so S2 must not be FAILED."""
    spec, H, w, V = heis6
    k = int(np.argmax(np.abs(np.diff(w)) < 1e-10))      # first degenerate pair (k, k+1)
    x1 = V[:, k]
    x2 = 0.5 * V[:, k] + np.sqrt(0.75) * V[:, k + 1]
    mu = np.array([w[k], w[k + 1]])
    c = ce.certify(ce.Claim(mu, np.column_stack([x1, x2]), operator=H, kind="subset"), spec)
    assert _st(c)["S2.eigenpairs"] != "FAILED", _stmt(c, "S2.eigenpairs")["summary"]
    assert _st(c)["S2.eigenpairs"] in ("CERTIFIED", "INCONCLUSIVE")


# ------------------------------------------------------------------ A9: sector assembly, missing / duplicated sector, cut ties
def _sz_sectors(spec, E_cut, drop=None, dup=None):
    H = ce.build_bitwise(spec)
    full = full_state_set(spec)
    secs = []
    N = spec.n_sites
    nups = list(range(N + 1))
    if drop is not None:
        nups.remove(drop)
    if dup is not None:
        nups.append(dup)
    for nup in nups:
        st = ce.u1_state_set(spec, nup)
        idx, found = full.index_of(st.states_in_order())
        assert np.all(found)
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        Hs = (Vs.T @ H @ Vs).tocsr()
        ws, Ys = eigh_dense(Hs.toarray())
        sel = np.where(ws < E_cut)[0]
        secs.append(SectorResult(f"nup={nup}", Vs, Hs, ws[sel], Ys[:, sel]))
    return H, secs


def test_A9_missing_or_duplicated_sector_not_certified():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    wf = la.eigvalsh(H.toarray())
    E_cut = float((wf[4] + wf[5]) / 2) if wf[5] - wf[4] > 1e-6 else float(wf[0] + 1.0)
    _, secs = _sz_sectors(spec, E_cut, drop=3)
    assert certify_low_energy(spec, secs, E_cut)["status"] != "CERTIFIED"
    _, secs = _sz_sectors(spec, E_cut, drop=0, dup=6)  # same total dimension as the full space
    assert certify_low_energy(spec, secs, E_cut)["status"] != "CERTIFIED"


def test_A9b_cut_exactly_at_an_eigenvalue():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    wf = la.eigvalsh(H.toarray())
    for E_cut in (float(wf[1]), float(wf[4]), float(wf[8])):
        _, secs = _sz_sectors(spec, E_cut)
        res = certify_low_energy(spec, secs, E_cut)
        if res["status"] == "CERTIFIED":
            K = res["K"]
            assert np.max(np.abs(np.array(res["lowest"]) - wf[:K])) <= res["radius"]
            assert res["next_lower_bound"] <= wf[K]


# ------------------------------------------------------------------ A10: complex (Gaussian) sector, omitted member
@needs_flint
def test_A10_complex_sector_omitted_degenerate_member_not_certified():
    """k = pi/2 sector of an 8-site ring (character i): drop one member of a degenerate level."""
    spec = ce.heisenberg_chain(8)
    st = full_state_set(spec)
    H = ce.build_bitwise(spec, st)
    T = translation(spec)
    V2 = cyclic_isometry(st, T, 2)
    w2, Y2 = la.eigh((V2.conj().T @ H @ V2).toarray())
    X = V2 @ Y2
    k = int(np.argmax(np.abs(np.diff(w2)) < 1e-9))     # first degenerate level in the sector
    idx = [i for i in range(k + 1)]                    # keep the first member only, omit the rest
    r = exact_recheck_sector_lowest(spec, st, [T], {0: 2}, w2[idx], np.asarray(X[:, idx]), next_hint=w2[k + 1])
    assert r["status"] != "CERTIFIED", r["status"]
