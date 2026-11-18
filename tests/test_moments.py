"""C-07 first part (certify_ed.moments): certified moments tr(P O H^k O)/g on a certified cluster, the two sum-rule
consistency statements, negative controls (understated s, non-eigenvector, wrong level), record schema."""
import json
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed.certify import Claim, Options, certify
from certify_ed.construct import build_bitwise
from certify_ed.correlations import site_operator
from certify_ed.moments import MOMENTS_SCHEMA, certified_moments, format_moments, moments_from_certificate
from certify_ed.schema import validate_record
from certify_ed.spec import heisenberg_chain


@pytest.fixture(scope="module")
def heis():
    spec = heisenberg_chain(8)
    H = build_bitwise(spec)
    w, V = np.linalg.eigh(H.toarray())
    cert = certify(Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, Options(store_eigenvectors=True, eigenspaces="all"))
    O = sp.csr_matrix(site_operator(spec, "z", [0]) + site_operator(spec, "z", [1]))
    return spec, H, w, V, cert, O


def stmt(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)


def exact_moments(H, O, P, g, K):
    Hd, Od = H.toarray() if sp.issparse(H) else H, O.toarray() if sp.issparse(O) else O
    out, M = [], Od
    for k in range(K + 1):
        if k > 0:
            M = Hd @ M
        out.append(np.trace(P @ Od @ M).real / g)
    return out


def test_ground_state_moments_from_certificate(heis):
    spec, H, w, V, cert, O = heis
    rec = moments_from_certificate(cert, H, O, K=3, members=[0])
    assert rec["schema"] == MOMENTS_SCHEMA and rec["certificate"]["operator_hash_matches"]
    ex = exact_moments(H, O, np.outer(V[:, 0], V[:, 0]), 1, 3)
    for m, e in zip(rec["moments"], ex):
        assert m["interval"][0] <= e <= m["interval"][1]
        assert m["bound"] < 1e-9
    assert stmt(rec, "M1.moments")["status"] == "CERTIFIED"
    assert stmt(rec, "M2.sum_rule_zeroth")["status"] == "CERTIFIED" and stmt(rec, "M2.sum_rule_zeroth")["evidence"]["overlap"]
    assert stmt(rec, "M3.sum_rule_first")["status"] == "CERTIFIED" and stmt(rec, "M3.sum_rule_first")["evidence"]["overlap"]
    m4 = stmt(rec, "M4.continued_fraction")
    assert m4["status"] == "NOT_CHECKED" and m4["evidence"]["reason_code"] == "NOT_IMPLEMENTED" and "next_action" in m4["evidence"]
    assert validate_record(rec) in (True, None)
    json.dumps(rec)
    assert "m_0" in format_moments(rec)


def test_triplet_cluster_basis_independent(heis):
    spec, H, w, V, cert, O = heis
    rec = moments_from_certificate(cert, H, O.toarray(), K=2, members=[1, 2, 3])
    P = V[:, 1:4] @ V[:, 1:4].T
    ex = exact_moments(H, O, P, 3, 2)
    for m, e in zip(rec["moments"], ex):
        assert m["interval"][0] <= e <= m["interval"][1]
    # a rotated basis of the same cluster gives the same exact moments: both records enclose them
    rng = np.random.default_rng(0)
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    X = V[:, 1:4] @ R
    cl = next(c for c in cert.statement("S4.eigenspaces")["evidence"]["clusters"] if c["members"] == [1, 2, 3])
    rec2 = certified_moments(H, O, X, cl["sin_theta_bound"] + 1e-12, cl["eta"] + 1e-12, K=2, level=(w[1] - 1e-10, w[3] + 1e-10))
    for m, e in zip(rec2["moments"], ex):
        assert m["interval"][0] <= e <= m["interval"][1]
    assert all(stmt(rec2, s)["status"] == "CERTIFIED" for s in ("M1.moments", "M2.sum_rule_zeroth", "M3.sum_rule_first"))


def test_negative_control_understated_s_and_non_eigenvector(heis):
    """A vector tilted out of the eigenspace by eps along another eigenvector: s = O(eps) carries the whole first-order
    error of the moments while eta = O(eps^2). With the exact s the enclosure holds; with s / 10^4 it breaks."""
    spec, H, w, V, cert, O = heis
    Hd = H.toarray()
    eps = 1e-3
    Od = O.toarray()
    j = 1 + int(np.argmax(np.abs(V[:, 1:].T @ (Od @ Od @ V[:, 0]))))      # a direction with a nonzero <v_j|O^2|v_0>
    x = V[:, 0] + eps * V[:, j]
    ex = exact_moments(H, O, np.outer(V[:, 0], V[:, 0]), 1, 2)
    s_true = np.linalg.norm(np.outer(x, x) / (x @ x) - np.outer(V[:, 0], V[:, 0]), 2)
    eta_true = abs(x @ x - 1)
    assert 0.5 * eps < s_true < 2 * eps and eta_true < 2 * eps ** 2
    good = certified_moments(Hd, O, x, s_true * (1 + 1e-9), eta_true * (1 + 1e-9), K=2, sum_rules=False)
    for m, e in zip(good["moments"], ex):
        assert m["interval"][0] <= e <= m["interval"][1]
    bad = certified_moments(Hd, O, x, s_true / 1e4, eta_true * (1 + 1e-9), K=2, sum_rules=False)
    assert any(not (m["interval"][0] <= e <= m["interval"][1]) for m, e in zip(bad["moments"], ex))
    rng = np.random.default_rng(1)
    y = rng.standard_normal(256)
    y /= np.linalg.norm(y)
    bad2 = certified_moments(Hd, O, y, 0.0, 0.0, K=2, sum_rules=False)
    assert all(not (m["interval"][0] <= e <= m["interval"][1]) for m, e in zip(bad2["moments"], ex))
    with pytest.raises(ValueError):
        certified_moments(Hd, O, y, 1.5, 0.0)


def test_negative_control_wrong_level_fails_the_sum_rule(heis):
    spec, H, w, V, cert, O = heis
    rec = certified_moments(H, O, V[:, 0], 1e-12, 1e-14, K=1, level=(w[0] + 0.5, w[0] + 0.5 + 1e-9))
    m3 = stmt(rec, "M3.sum_rule_first")
    assert m3["status"] == "FAILED" and not m3["evidence"]["overlap"]
    assert validate_record(rec) in (True, None)
    rec = certified_moments(H, O, V[:, 0], 1e-12, 1e-14, K=0)
    assert stmt(rec, "M3.sum_rule_first")["status"] == "NOT_CHECKED"
    rec = certified_moments(H, O, V[:, 0], 1e-12, 1e-14, K=1)
    assert stmt(rec, "M3.sum_rule_first")["evidence"]["reason_code"] == "NEEDS_DEPENDENCY"


def test_complex_operator_and_exact_rational_rounding():
    rng = np.random.default_rng(2)
    n = 10
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = H + H.conj().T
    O = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    O = O + O.conj().T
    w, V = np.linalg.eigh(H)
    x = V[:, 0]
    rec = certified_moments(H, O, x, 1e-14, 1e-15, K=3, level=(w[0] - 1e-12, w[0] + 1e-12))
    ex = exact_moments(H, O, np.outer(x, x.conj()), 1, 3)
    for m, e in zip(rec["moments"], ex):
        assert m["interval"][0] <= e <= m["interval"][1]
    assert stmt(rec, "M3.sum_rule_first")["status"] == "CERTIFIED"
    # the rounding term alone covers the exact rational tr(x^H O H^k O x) of the stored data
    Hr, Hi, Or, Oi = np.real(H), np.imag(H), np.real(O), np.imag(O)

    def fv(v):
        return [(Fraction(float(np.real(z))), Fraction(float(np.imag(z)))) for z in v]

    def mv(Mr, Mi, v):
        out = []
        for i in range(n):
            re = Fraction(0)
            im = Fraction(0)
            for j in range(n):
                ar, ai = Fraction(float(Mr[i, j])), Fraction(float(Mi[i, j]))
                re += ar * v[j][0] - ai * v[j][1]
                im += ar * v[j][1] + ai * v[j][0]
            out.append((re, im))
        return out

    y = mv(Or, Oi, fv(x))
    z = y
    for k in range(4):
        if k:
            z = mv(Hr, Hi, z)
        exact = sum(y[i][0] * z[i][0] + y[i][1] * z[i][1] for i in range(n))
        m = rec["moments"][k]
        assert abs(Fraction(m["value"]) - exact) <= Fraction(m["terms"]["rounding"]) + Fraction(abs(m["value"])) * Fraction(2 ** -52)


def test_underflow_and_input_checks(heis):
    spec, H, w, V, cert, O = heis
    rec = certified_moments(H, O, V[:, 0] * 2.0 ** -950, 1e-12, 1e-14, K=1)
    assert all(s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "UNDERFLOW_RISK" for s in rec["statements"][:3])
    with pytest.raises(ValueError, match="Hermitian"):
        certified_moments(H, sp.csr_matrix(np.triu(np.ones((256, 256)))), V[:, 0], 1e-12, 1e-14)
    with pytest.raises(ValueError, match="CERTIFIED S4 cluster"):
        moments_from_certificate(cert, H, O, members=[7])
