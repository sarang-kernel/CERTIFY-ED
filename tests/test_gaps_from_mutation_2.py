"""Tests closing gaps found by inspecting the survivors of the second E6 mutation run
(experiments/results/e6_mutation_20261017_081613, classification.json, 'needs inspection').
Each test names the mutant(s) it targets; every test checks a property of the result (containment of an
exact value, agreement with an independent computation), not a re-implementation of the code."""
import sys
from fractions import Fraction

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.fp import exact_dot, exact_gram
from certify_ed.spec import ModelSpec

from conftest import needs_flint, needs_quspin

CERT = sys.modules["certify_ed.certify"]


def _stmt(c, sid):
    return [s for s in c.statements if s["id"] == sid][0]


# ------------------------------------------------ S1: small non-conservation must fail (certify.py:261)
def test_model_fidelity_fails_for_small_leak():
    base = ce.heisenberg_chain(6)
    h = 0.05                                          # small transverse field: leak well below 1
    static = [[t.op, [[c] + list(s) for c, s in t.couplings]] for t in base.terms]
    static += [["+", [[h, i] for i in range(6)]], ["-", [[h, i] for i in range(6)]]]
    spec = ModelSpec.from_static(6, base.local, static, name="heisenberg_plus_field")
    st = ce.u1_state_set(base, 3)
    H = ce.build_bitwise(base, st)
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="lowest", states=st), spec)
    s1 = _stmt(c, "S1.model_fidelity")
    assert s1["status"] == "FAILED", s1


# ------------------------------------------------ S4: partial cluster failure is INCONCLUSIVE (certify.py:800)
def test_s4_partial_cluster_failure_is_inconclusive(monkeypatch):
    spec = ce.heisenberg_chain(8)
    st = ce.u1_state_set(spec, 4)
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    # claim enough lowest pairs to contain at least two degenerate clusters, ending at a gap
    k, nclust, i = 0, 0, 0
    while i < len(w):
        j = i
        while j + 1 < len(w) and w[j + 1] - w[i] < 1e-8:
            j += 1
        nclust += j > i
        i = j + 1
        if nclust >= 2:
            k = i
            break
    assert k > 0
    real = CERT.certify_eigenspace
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        return None if calls["n"] == 2 else real(*a, **kw)
    monkeypatch.setattr(CERT, "certify_eigenspace", flaky)
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest", states=st), spec)
    s4 = _stmt(c, "S4.eigenspaces")
    assert calls["n"] >= 2
    assert s4["status"] == "INCONCLUSIVE", s4["summary"] if "summary" in s4 else s4


# ------------------------------------------------ Kato-Temple uses the adjacent level (certify.py:708)
def test_kato_temple_bounds_contain_exact_eigenvalue():
    rng = np.random.default_rng(11)
    lam = np.r_[0.0, 0.01, np.arange(1.0, 18.0), 17.01]
    Q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    H = (Q * lam) @ Q.T
    H = (H + H.T) / 2
    w, V = la.eigh(H)
    X = V.copy()
    X[:, 0] = V[:, 0] + 1e-3 * V[:, 1]               # error towards the close neighbour
    X[:, 0] /= np.linalg.norm(X[:, 0])
    X[:, -2] = V[:, -2] + 1e-3 * V[:, -1]            # second-to-last towards the last level
    X[:, -2] /= np.linalg.norm(X[:, -2])
    c = ce.certify(ce.Claim(w, X, operator=H, kind="all"))
    kt = c.to_dict()["evidence"]["kato_temple"] if hasattr(c, "to_dict") and "evidence" in c.to_dict() else None
    if kt is None:
        kt = c.body["kato_temple"] if hasattr(c, "body") else c.kato_temple
    entry = [e for e in kt if e["index"] == 0]
    assert entry, "Kato-Temple bound expected for the perturbed vector"
    e = entry[0]
    assert e["lower"] <= w[0] <= e["upper"], e
    assert any(e["index"] == len(w) - 2 for e in kt)
    for e in kt:
        assert e["lower"] <= w[e["index"]] <= e["upper"], e


# ------------------------------------------------ S4 separation uses enclosure edges (certify.py:765)
def test_s4_separation_lower_bound_below_true_separation():
    spec = ce.heisenberg_chain(8)
    st = ce.u1_state_set(spec, 4)
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    # first degenerate cluster [i0, i1] followed by a simple level i1+1
    i = 0
    while True:
        j = i
        while w[j + 1] - w[i] < 1e-8:
            j += 1
        if j > i and w[j + 2] - w[j + 1] > 1e-6:
            break
        i = j + 1
    i0, i1, nxt = i, j, j + 1
    k = nxt + 1
    while k < len(w) and w[k] - w[k - 1] < 1e-6:      # end the claim at a gap
        k += 1
    X = V[:, :k].copy()
    far = V[:, -1]                                    # not claimed: keeps the claim nearly orthonormal
    X[:, nxt] = V[:, nxt] + 3e-5 * far
    X[:, nxt] /= np.linalg.norm(X[:, nxt])
    c = ce.certify(ce.Claim(w[:k], X, operator=H, kind="lowest", states=st), spec)
    s4 = _stmt(c, "S4.eigenspaces")
    recs = [r for r in s4["evidence"]["clusters"] if r["members"][0] == i0 and "separation_lower_bound" in r]
    assert recs, s4
    sep = recs[0]["separation_lower_bound"]
    true_sep = min(w[i0] - (w[i0 - 1] if i0 > 0 else -np.inf), w[nxt] - w[i1])
    assert sep <= true_sep, (sep, true_sep)


# ------------------------------------------------ producer Hermitian part (certify.py:280)
def test_model_shift_with_slightly_nonhermitian_producer():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).toarray()
    rng = np.random.default_rng(5)
    A = rng.standard_normal(H.shape) * (np.abs(H) > 0)
    Hp = H + 1e-15 * (A - A.T)                         # tiny anti-Hermitian part on the pattern of H
    w, V = la.eigh((Hp + Hp.T) / 2)
    c = ce.certify(ce.Claim(w[:3], V[:, :3], operator=sp.csr_matrix(Hp), kind="lowest"), spec)
    s1 = _stmt(c, "S1.model_fidelity")
    assert s1["evidence"].get("hermiticity_max_abs", 0) > 0
    assert s1["evidence"]["model_shift_bound"] < 1e-10, s1["evidence"]["model_shift_bound"]


# ------------------------------------------------ momentum-label threshold (sectors.py:253)
@needs_quspin
def test_label_audit_leakage_threshold():
    from quspin.basis import spin_basis_1d
    from certify_ed.sectors import audit_label
    N = 8
    spec = ce.heisenberg_chain(N)
    v2 = spin_basis_1d(N, Nup=4, kblock=2, pauli=0).get_proj(np.complex128)[:, [0]].toarray()
    v3 = spin_basis_1d(N, Nup=4, kblock=3, pauli=0).get_proj(np.complex128)[:, [0]].toarray()
    for s, expect in ((0.6, False), (0.3, True)):
        x = np.sqrt(1 - s * s) * v2 + s * v3
        r = audit_label(spec, sp.csr_matrix(x), "translation")
        assert abs(r["leakage_bound"] - s) < 0.05, r
        assert r["certified"] is expect, r


# ------------------------------------------------ exact inner products and Gram matrices (fp.py:185, 212)
def test_exact_dot_mixed_real_complex():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50)
    y = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    s, err = exact_dot(x, y)
    ref = sum(Fraction(a) * Fraction(b) for a, b in zip(x, y.real)), sum(Fraction(a) * Fraction(b) for a, b in zip(x, y.imag))
    assert abs(s.real - float(ref[0])) <= err + 1e-300 and abs(s.imag - float(ref[1])) <= err + 1e-300
    s2, _ = exact_dot(y, x)
    assert abs(s2 - np.conj(s)) <= 4 * err


def test_exact_gram_matches_xhx():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 5)) + 1j * rng.standard_normal((40, 5))
    G, E = exact_gram(X)
    ref = X.conj().T @ X
    assert np.allclose(G, ref, rtol=1e-13, atol=1e-13)
    assert np.array_equal(G, G.conj().T)


# ------------------------------------------------ exact residual bound is an upper bound (checker.py:112)
@needs_flint
def test_exact_residual_bound_dominates_exact_residual():
    from certify_ed.checker import residual_inf_norm_exact
    rng = np.random.default_rng(7)
    n = 10
    B = rng.integers(-5, 6, (n, n)).astype(float)
    A = B + B.T + 0.5 * np.diag(rng.integers(-3, 4, n))
    A = np.rint(A)
    lu, d, _ = la.ldl(A, lower=True, hermitian=True)
    ue, Dt = residual_inf_norm_exact(A, lu, d)
    Lt = np.rint(np.ldexp(lu, 52))
    F = lambda M, s=0: [[Fraction(int(x)) / Fraction(2) ** s for x in row] for row in M]
    L = F(Lt, 52)
    D = [[Fraction(float(x)) for x in row] for row in Dt]
    Aq = [[Fraction(int(x)) for x in row] for row in A]
    LD = [[sum(L[i][k] * D[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    E = [[sum(LD[i][k] * L[j][k] for k in range(n)) - Aq[i][j] for j in range(n)] for i in range(n)]
    exact = max(sum(abs(x) for x in row) for row in E)
    assert Fraction(ue) >= exact
    assert ue <= float(exact) * (1 + 1e-12) + 1e-300


# ------------------------------------------------ model constructors (spec.py:278, 306)
def test_spin1_chain_scales_with_J():
    H1 = ce.build_bitwise(ce.spin1_heisenberg_chain(4)).toarray()
    H2 = ce.build_bitwise(ce.spin1_heisenberg_chain(4, J=2.0)).toarray()
    assert np.allclose(H2, 2 * H1)


def test_spinless_fermion_default_interaction_is_zero():
    a = ce.build_bitwise(ce.spinless_fermion_chain(6)).toarray()
    b = ce.build_bitwise(ce.spinless_fermion_chain(6, V=0.0)).toarray()
    assert np.array_equal(a, b)


# ------------------------------------------------ builder rounding allowance present (certify.py:190)
def test_builder_shift_positive_for_nonempty_spec():
    spec = ce.heisenberg_chain(6, J2=0.3)                 # 0.3 is not dyadic: rounding is possible
    b = CERT._builder_shift(spec, None)
    assert b > 0


# ------------------------------------------------ run 3 (e6_mutation_20261017_112726) survivors
def test_check_semidefinite_lower_bound_reports_lambda_min_only_when_definite():
    from certify_ed.checker import check_semidefinite_lower_bound
    rng = np.random.default_rng(9)
    B = rng.standard_normal((20, 20))
    A = B @ B.T + 0.5 * np.eye(20)
    r = check_semidefinite_lower_bound(A)
    assert r["nu"] == 0 and r["lambda_min_lower"] <= la.eigvalsh(A)[0]
    r2 = check_semidefinite_lower_bound(A - 3 * np.eye(20) * la.eigvalsh(A)[1])
    assert r2["nu"] >= 1 and "lambda_min_lower" not in r2


@needs_quspin
def test_sector_producer_hermiticity_reported_as_zero():
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian
    N = 8
    spec = ce.heisenberg_chain(N)
    b = spin_basis_1d(N, Nup=N // 2, kblock=0, pauli=0)
    H = hamiltonian(spec.to_quspin_static(), [], basis=b, dtype=np.float64,
                    check_symm=False, check_herm=False, check_pcon=False).tocsr()
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest",
                            sector_isometry=sp.csr_matrix(b.get_proj(np.float64))), spec)
    s1 = _stmt(c, "S1.model_fidelity")
    assert s1["status"] == "CERTIFIED", s1
    assert s1["evidence"]["hermiticity_max_abs"] == 0.0
