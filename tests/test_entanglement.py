"""C-03 (3.2): certified entanglement entropy and spectrum (certify_ed.entanglement; docs/proofs/C-03_entanglement.md).

Positive cases compare the certified interval with a high-precision reference (mpmath, 40 digits) of the exact
eigenvector's reduced state; perturbation cases plant a known rotation of the claimed vector and check that the
interval still contains the entropy of the exact eigenvector; negative controls exercise every NOT_CHECKED path."""
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed import entanglement as ent
from certify_ed.eigenspace import EigenspaceCertificate
from certify_ed.localops import LocalSpace
from certify_ed.spec import ModelSpec

try:
    import mpmath as mp
except ImportError:  # optional: the 40-digit references need mpmath (test extra)
    mp = None
else:
    mp.mp.dps = 40

needs_mpmath = pytest.mark.skipif(mp is None, reason="mpmath not installed")


def _s4(cert):
    s4 = [s for s in cert.statements if s["id"] == "S4.eigenspaces"][0]
    assert s4["status"] == "CERTIFIED", s4["summary"]
    return s4


def _mp_entropy_of_projector(P_cols, d_A, d_B):
    """S(tr_B(P)/g) at 40 digits from the columns (mpmath matrix, n x g) of an orthonormal basis of range P."""
    g = P_cols.cols
    rho = mp.zeros(d_A, d_A)
    for j in range(g):
        Y = mp.matrix(d_A, d_B)
        for a in range(d_A):
            for b in range(d_B):
                Y[a, b] = P_cols[a * d_B + b, j]
        rho += Y * Y.transpose_conj()
    rho = rho / g
    ev = mp.eigsy(rho, eigvals_only=True)
    return -sum(p * mp.log(p) for p in ev if p > mp.mpf(10) ** (-35))


def _mp_ground_state(Hd, k):
    """Eigenvectors k (list of indices) of the dense symmetric matrix Hd at 40 digits (mpmath eigsy)."""
    E, Q = mp.eigsy(mp.matrix(Hd.tolist()))
    idx = sorted(range(len(E)), key=lambda i: E[i])
    cols = [idx[i] for i in k]
    out = mp.matrix(Q.rows, len(cols))
    for j, c in enumerate(cols):
        for i in range(Q.rows):
            out[i, j] = Q[i, c]
    return out


@pytest.fixture(scope="module")
def heis6():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


@needs_mpmath
def test_ground_state_entropy_contains_high_precision_reference(heis6):
    spec, H, w, V = heis6
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec,
                      ce.Options(eigenspaces="all", entanglement={"half": {"sites": [0, 1, 2]}}))
    cl = _s4(cert)["evidence"]["clusters"][0]
    r = cl["entanglement"]["half"]
    assert r["status"] == "CERTIFIED" and r["d_A"] == 8
    lo, hi = r["certified_interval"]
    # site 0 is the most significant digit: the vector index is a * d_B + b with a the A pattern
    S_ref = _mp_entropy_of_projector(_mp_ground_state(H.toarray(), [0]), 8, 8)
    assert lo <= float(S_ref) <= hi
    assert hi - lo < 1e-10
    assert abs(r["value"] - float(S_ref)) < 1e-12
    # the spectrum intervals contain the exact Schmidt weights
    Y = V[:, 0].reshape(8, 8)
    p_ref = np.sort(np.linalg.eigvalsh(Y @ Y.T))
    for k, s in enumerate(r["entanglement_spectrum"]):
        a, b = s["certified_interval"]
        assert a - 1e-13 <= p_ref[k] <= b + 1e-13
    # not listed as out of scope when requested
    assert not any(q["quantity"].startswith("entanglement") for q in cert.body["scope"]["out_of_scope"])


def test_single_site_entropy_of_a_singlet_is_one_bit(heis6):
    spec, H, w, V = heis6
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec,
                      ce.Options(eigenspaces="all", entanglement={"one": {"sites": [3], "log_base": 2}}))
    r = _s4(cert)["evidence"]["clusters"][0]["entanglement"]["one"]
    assert r["status"] == "CERTIFIED" and r["log_base"] == 2
    lo, hi = r["certified_interval"]
    assert lo <= 1.0 <= hi and hi - lo < 1e-9


@needs_mpmath
def test_cluster_uses_the_normalised_projector(heis6):
    """The first excited level of the 6-site ring is a triplet (g = 3): the record is about tr_B(P)/3."""
    spec, H, w, V = heis6
    assert np.allclose(w[1:4], w[1]) and w[4] - w[3] > 0.1
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec,
                      ce.Options(eigenspaces="all", entanglement={"half": {"sites": [0, 1, 2]}}))
    cls = _s4(cert)["evidence"]["clusters"]
    tr = [c for c in cls if len(c["members"]) == 3][0]
    r = tr["entanglement"]["half"]
    assert r["status"] == "CERTIFIED" and r["cluster_size"] == 3 and "normalised projector" in r["object"]
    assert r["trace_distance_bound_formula"] == "s + eta/(1 - eta)"
    S_ref = _mp_entropy_of_projector(_mp_ground_state(H.toarray(), [1, 2, 3]), 8, 8)
    lo, hi = r["certified_interval"]
    assert lo <= float(S_ref) <= hi and hi - lo < 1e-9
    # a rotated basis of the same eigenspace gives an interval containing the same value (basis independence)
    rng = np.random.default_rng(3)
    Qr, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    V2 = V.copy()
    V2[:, 1:4] = V[:, 1:4] @ Qr
    cert2 = ce.certify(ce.Claim(w[:4], V2[:, :4], operator=H, kind="lowest"), spec,
                       ce.Options(eigenspaces="all", entanglement={"half": {"sites": [0, 1, 2]}}))
    r2 = [c for c in _s4(cert2)["evidence"]["clusters"] if len(c["members"]) == 3][0]["entanglement"]["half"]
    lo2, hi2 = r2["certified_interval"]
    assert lo2 <= float(S_ref) <= hi2


@needs_mpmath
def test_u1_block_claim_and_complex_phase_give_the_same_entropy(heis6):
    spec, H, w, V = heis6
    st = ce.magnetization_state_set(spec, 0)
    Hb = ce.build_bitwise(spec, st)
    wb, Vb = la.eigh(Hb.toarray())
    S_ref = _mp_entropy_of_projector(_mp_ground_state(H.toarray(), [0]), 8, 8)
    x = Vb[:, :1] * np.exp(0.7j)                       # complex vectors: the imaginary path of the exact stage
    cert = ce.certify(ce.Claim(wb[:1], x, operator=Hb.astype(complex), kind="lowest", states=st), spec,
                      ce.Options(eigenspaces="all", entanglement={"half": {"sites": [0, 1, 2]}}))
    r = _s4(cert)["evidence"]["clusters"][0]["entanglement"]["half"]
    assert r["status"] == "CERTIFIED"
    lo, hi = r["certified_interval"]
    assert lo <= float(S_ref) <= hi and hi - lo < 1e-9


def _diag3():
    local = LocalSpace.spin_space("1/2", "pauli")
    spec = ModelSpec.from_static(3, local, [["z", [[-1.0, 0], [-2.5, 1], [-4.0, 2]]]], name="diag3")
    H = ce.build_bitwise(spec)
    st = ce.full_state_set(spec).states_in_order()
    i0 = int(np.argmin(H.diagonal()))
    i1 = int(np.where(st == (st[i0] ^ (1 << 2)))[0][0])         # the same state with site 0 flipped: energy +2
    return spec, H, i0, i1


@pytest.mark.parametrize("sin_phi", [0.55, 0.6])
def test_negative_control_trace_distance_above_fannes_audenaert_range(sin_phi):
    """A claimed vector rotated by sin(phi) > 1/2 relative to the exact eigenvector in a d_A = 2 cut: S2, S3 and
    S4 can still be CERTIFIED (accuracy target waived; the window (-12, -5.6) keeps the other eigenvalues 1.2 away so
    that the sin(theta) bound stays below 1), but the entanglement record must be NOT_CHECKED with the range reason
    and no interval, never a false interval."""
    spec, H, i0, i1 = _diag3()
    x = np.zeros(8)
    x[i0], x[i1] = math.sqrt(1 - sin_phi ** 2), sin_phi
    mu = float(x @ H @ x)
    cert = ce.certify(ce.Claim([mu], x[:, None], operator=H, kind="window", window=(-12.0, -5.6),
                               accuracy_target=float("inf")), spec,
                      ce.Options(eigenspaces="all", entanglement={"one": {"sites": [0]}}))
    cl = _s4(cert)["evidence"]["clusters"][0]
    assert sin_phi <= cl["sin_theta_bound"] < 1                 # the bound covers the planted rotation
    r = cl["entanglement"]["one"]
    assert r["status"] == "NOT_CHECKED" and r["reason_code"] == "NOT_APPLICABLE"
    assert "Fannes-Audenaert range" in r["reason"] and "certified_interval" not in r
    assert r["next_action"] and "1 - 1/d_A" in r["next_action"]
    assert r["trace_distance_bound"] > 0.5


def test_planted_rotation_inside_the_range_keeps_the_exact_entropy_inside_the_interval():
    """sin(phi) = 0.15: s < 1/2, the interval must contain the exact eigenvector's entropy (0: a product state)
    although the computed state's value differs from it."""
    spec, H, i0, i1 = _diag3()
    s_ = 0.15
    x = np.zeros(8)
    x[i0], x[i1] = math.sqrt(1 - s_ ** 2), s_
    mu = float(x @ H @ x)
    cert = ce.certify(ce.Claim([mu], x[:, None], operator=H, kind="window", window=(-9.0, -6.0),
                               accuracy_target=float("inf")), spec,
                      ce.Options(eigenspaces="all", entanglement={"one": {"sites": [1]}}))
    cl = _s4(cert)["evidence"]["clusters"][0]
    r = cl["entanglement"]["one"]
    assert r["status"] == "CERTIFIED"
    assert cl["sin_theta_bound"] >= s_ and r["trace_distance_bound"] == pytest.approx(cl["sin_theta_bound"])
    lo, hi = r["certified_interval"]
    assert lo <= 0.0 <= hi                                      # exact eigenvector |uuu> is a product state
    assert hi >= float(ent.ivl.to_float_down(ent.binary_entropy_upper(Fraction(0.15)))) * 0.99


@needs_mpmath
def test_heisenberg_perturbed_vector_interval_contains_exact_entropy(heis6):
    """Random rotations of the ground state towards an excited state, with S2/S3/S4 certified through a window
    claim; the exact entropy of the eigenvector lies in every interval and the trace-distance bound covers sin(phi)."""
    spec, H, w, V = heis6
    S_ref = float(_mp_entropy_of_projector(_mp_ground_state(H.toarray(), [0]), 8, 8))
    rng = np.random.default_rng(7)
    a, b = -3.0, -2.3                                        # only the ground state (-2.803) lies inside; the
    assert a < w[0] < b < w[1]                               # triplet is at -2.118
    for _ in range(4):
        phi = rng.uniform(0.005, 0.04)
        k = int(rng.integers(4, 12))
        x = math.cos(phi) * V[:, 0] + math.sin(phi) * V[:, k]
        mu = float(x @ H @ x)
        cert = ce.certify(ce.Claim([mu], x[:, None], operator=H, kind="window", window=(a, b),
                                   accuracy_target=float("inf")), spec,
                          ce.Options(eigenspaces="all", entanglement={"half": {"sites": [0, 1, 2]}}))
        cl = _s4(cert)["evidence"]["clusters"][0]
        r = cl["entanglement"]["half"]
        assert r["status"] == "CERTIFIED", (r.get("reason"), cl["sin_theta_bound"], phi, k)
        assert r["trace_distance_bound"] >= math.sin(phi)
        lo, hi = r["certified_interval"]
        assert lo <= S_ref <= hi


def test_not_checked_paths_have_reasons_and_next_actions(heis6):
    spec, H, w, V = heis6
    ec = EigenspaceCertificate([0], [float(w[0])], 1.0, 1e-14, 1e-16, 1e-13, 1e-13)
    st = ce.full_state_set(spec).states_in_order()
    # size limit
    r = ent.certify_entanglement(ec, V[:, :1], st, 6, 2, [0, 1, 2], max_dim_a=4)
    assert r["status"] == "NOT_CHECKED" and r["reason_code"] == "SIZE_LIMIT" and "certified_interval" not in r
    # A must be a proper, non-empty subset
    for bad in ([], [0, 1, 2, 3, 4, 5], [7]):
        r = ent.certify_entanglement(ec, V[:, :1], st, 6, 2, bad)
        assert r["status"] == "NOT_CHECKED" and r["reason_code"] == "NOT_APPLICABLE"
    # through the S4 hook: no spec, and a symmetry-adapted claim
    claim = ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest")
    recs = ent.entanglement_records(ec, V, None, claim, {"c": {"sites": [0]}})
    assert recs["c"]["status"] == "NOT_CHECKED" and recs["c"]["reason_code"] == "NEEDS_SPEC" and recs["c"]["next_action"]
    import scipy.sparse as sp
    claim_s = ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest", sector_isometry=sp.identity(64, format="csr"))
    recs = ent.entanglement_records(ec, V, spec, claim_s, {"c": {"sites": [0]}})
    assert recs["c"]["status"] == "NOT_CHECKED" and recs["c"]["reason_code"] == "UNSUPPORTED_CLAIM" and recs["c"]["next_action"]
    recs = ent.entanglement_records(ec, V, spec, claim, {"c": {"sites": [0], "max_dim_a": 1}})
    assert recs["c"]["status"] == "NOT_CHECKED" and recs["c"]["reason_code"] == "SIZE_LIMIT" and "max_dim_a" in recs["c"]["next_action"]
    # the default scope block still lists the quantity as not computed
    cert = ce.certify(claim, spec, ce.Options(eigenspaces="all"))
    out = [q for q in cert.body["scope"]["out_of_scope"] if q["quantity"].startswith("entanglement")]
    assert len(out) == 1 and out[0]["reason_code"] == "DEFAULT_SELECTION"


def test_interval_functions_are_outward_and_monotone():
    """f(p) = -p ln p enclosures contain the true values; the Fannes-Audenaert function is increasing on its range
    and None outside; h(T) at a few rational points."""
    for p in (Fraction(1, 3), Fraction(1, 7), Fraction(9, 10), Fraction(1, 2)):
        iv = ent._f_point(p)
        lo, hi = iv.floats()
        exact = -float(p) * math.log(float(p))
        assert lo <= exact <= hi and hi - lo < 1e-15
    # the range enclosure over an interval straddling 1/e reaches 1/e
    iv = ent._f_interval(Fraction(1, 4), Fraction(1, 2))
    assert iv.floats()[1] >= 1 / math.e and iv.floats()[0] <= min(0.25 * math.log(4), 0.5 * math.log(2))
    assert ent.fannes_audenaert_bound(Fraction(3, 4), 2) is None
    assert ent.fannes_audenaert_bound(Fraction(1, 2), 2) is not None
    assert ent.fannes_audenaert_bound(Fraction(0), 5) == 0
    vals = [ent.ivl.to_float_up(ent.fannes_audenaert_bound(Fraction(k, 20), 4)) for k in range(0, 16)]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    h = ent.ivl.to_float_up(ent.binary_entropy_upper(Fraction(1, 4)))
    assert abs(h - (-(0.25 * math.log(0.25) + 0.75 * math.log(0.75)))) < 1e-15 and h >= -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))


@needs_mpmath
def test_exact_eigenvalue_enclosure_contains_mpmath_eigenvalues():
    rng = np.random.default_rng(11)
    for cplx in (False, True):
        Y = rng.standard_normal((5, 9)) + (1j * rng.standard_normal((5, 9)) if cplx else 0)
        x = Y.ravel() / np.linalg.norm(Y)
        a_idx, b_idx, dA, dB = np.repeat(np.arange(5), 9), np.tile(np.arange(9), 5), 5, 9
        Mr, Mi, t = ent.reduced_density_exact(x, a_idx, b_idx, dA, dB)
        encl = ent.enclose_psd_eigenvalues(Mr, Mi, t)
        M = mp.matrix(5, 5)
        for i in range(5):
            for j in range(5):
                M[i, j] = mp.mpf(Mr[i, j].numerator) / Mr[i, j].denominator / (mp.mpf(t.numerator) / t.denominator)
                if Mi is not None:
                    M[i, j] += 1j * mp.mpf(Mi[i, j].numerator) / Mi[i, j].denominator / (mp.mpf(t.numerator) / t.denominator)
        ev = sorted(mp.eigh(M, eigvals_only=True)) if cplx else sorted(mp.eigsy(M, eigvals_only=True))
        for k in range(5):
            assert encl["lo"][k] <= Fraction(str(mp.nstr(ev[k], 35))) <= encl["hi"][k]
            assert encl["hi"][k] - encl["lo"][k] < Fraction(1, 10 ** 13)
