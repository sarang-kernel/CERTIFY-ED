"""Tests of track/backlog-proofs: the implemented bounds pinned to the derived formulas of docs/proofs/ (Theorem 2 with
Kahan's constant rho, Theorem 2', Corollary 5, Theorem 8, Appendix A, the Theorem 1 accounting), the INT-05 underflow
terms (radii never shrink; tiny inputs), and the mutation survivors of RESULT_integration_3.2.md (sectors.py:240,
certify.py Kato-Temple neighbour, checker.py mantissa scale)."""
from decimal import Decimal, getcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.certify import Options, _builder_shift, _certify_lowest, _subspace_shift
from certify_ed.checker import _float_vector_to_int
from certify_ed.construct import SpecOperator, build_bitwise, full_state_set, hermitian_part
from certify_ed.enclosure import OperatorAdapter, _cluster_bound, enclose, residuals
from certify_ed.fp import (U, ULP_MIN, TINY_BOUND, _GUARD, exact_dot, fro_upper, gamma, gamma_dot, inf_norm_upper,
                           norm2_lower, norm2_upper, norm_underflow_term, residual_additive_term, sub_up, up)
from certify_ed.sectors import audit_label, decomposition_shift

getcontext().prec = 120
u = Fraction(U)


def gam(m):
    return m * u / (1 - m * u)


def _sqrt(q, up_):
    getcontext().rounding = ROUND_CEILING if up_ else ROUND_FLOOR
    d = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
    return Fraction(d) * (1 + Fraction(1 if up_ else -1, 10 ** 100))


@pytest.fixture(scope="module")
def heis6():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


# ------------------------------------------------------------------ Theorem 1 accounting: the scalar lemmas
@pytest.mark.parametrize("k", [1, 2, 10, 1000, 10 ** 6, 2 ** 26])
def test_theorem1_accounting_slack_constants(k):
    """(L4)-(L6) of docs/proofs/thm1_fp-accounting.md at a few n, w (the verification script covers the grid)."""
    g0, g1, g2 = gam(k), gam(k + 1), gam(k + 2)
    assert g1 >= g0 + u * (1 + g0)
    for N in (k, 2 * k):
        gN = gam(N)
        lo = _sqrt(1 - gN, False) * (1 - u) * (1 + g2 * (1 - u) ** 2) * (1 + 8 * u) * (1 - u) ** 4
        hi = _sqrt(1 + gN, True) * (1 + u) * (1 - g2 * (1 + u) ** 2) * (1 + u) ** 3 / (1 + 8 * u)
        assert lo >= 1 + 3 * u and hi <= 1 - 3 * u
    assert (1 - g0) * (1 - u) ** 3 * (1 + 4 * u) / (1 - g1 * (1 - u) ** 2) >= 1 + u
    assert (1 - u) ** 6 * (1 + 4 * u) * (1 + u / g2) >= 1


# ------------------------------------------------------------------ Theorem 2 / 2': the cluster bound
def _cluster_case(seed=0, n=7, g=3, mix=0.3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A = (A + A.T) / 2
    w, V = la.eigh(A)
    C = np.eye(g) + mix * rng.normal(size=(g, g))
    X = V[:, :g] @ C
    X /= np.linalg.norm(X, axis=0)
    mu = w[:g] + 1e-6 * rng.normal(size=g)
    return A, X, mu


def test_cluster_bound_is_min_of_rho_and_eps_star_and_rho_matches_the_formula():
    A, X, mu = _cluster_case()
    ad = OperatorAdapter(A)
    res = residuals(ad, mu, X)
    radius, eta, rho, info = _cluster_bound(ad, mu, X, res, [0, 1, 2])
    assert info["eps"] == info["rho"] == rho and radius == min(rho, info["eps_star"])
    assert info["used"] == ("eps_star" if info["eps_star"] < rho else "rho")
    # rho = up(||R||_F-bound + (||A|| + ||Lambda||) delta) + ULP_MIN with the document's ingredients
    delta = up(1.0 - np.sqrt(1.0 - eta)) + 4 * U
    Rg = norm2_upper(np.asarray(res.upper[[0, 1, 2]], dtype=float))
    assert rho == up(Rg + (inf_norm_upper(A) + float(np.max(np.abs(mu)))) * delta) + ULP_MIN
    # Kahan's constant: the old eps = (w + sqrt(w^2 + 4 rho^2))/2 >= rho for every w >= 0
    for w_ in (0.0, 0.5 * rho, rho):
        assert 0.5 * (w_ + np.sqrt(w_ * w_ + 4 * rho * rho)) >= rho


def test_eps_star_matches_the_formula_and_terms_are_upper_bounds():
    A, X, mu = _cluster_case(seed=3, mix=0.2)
    ad = OperatorAdapter(A)
    res = residuals(ad, mu, X)
    radius, eta, rho, info = _cluster_bound(ad, mu, X, res, [0, 1, 2])
    inv_sqrt, inv = up(1.0 / np.sqrt(1.0 - eta)), up(1.0 / (1.0 - eta))
    delta = up(1.0 - np.sqrt(1.0 - eta)) + 4 * U
    spread = up(float(sub_up(float(mu.max()), float(mu.min()))))
    Rg = norm2_upper(np.asarray(res.upper[[0, 1, 2]], dtype=float))
    # the X^H R term is recovered from eps_star; it must dominate the exact ||X^H R||_2
    XR_term = (info["eps_star"] - 4 * ULP_MIN) / _GUARD - Rg * inv_sqrt - delta * spread * inv_sqrt
    R = A @ X - X * mu[None, :]
    assert XR_term * inv >= 0 and XR_term >= la.norm(X.conj().T @ R, 2) * (1 - 1e-12)
    assert inv_sqrt >= 1 / np.sqrt(1 - eta) and inv >= 1 / (1 - eta) and delta >= 1 - np.sqrt(1 - eta)
    assert eta >= la.norm(X.conj().T @ X - np.eye(3), 2) and Rg >= la.norm(R, 2)


def test_duplicate_columns_give_no_cluster_bound():
    """Negative control: eta >= 1 (two identical columns) -> no Theorem 2 / 2' radius, existence only."""
    A, X, mu = _cluster_case()
    X2 = X.copy()
    X2[:, 1] = X2[:, 0]
    ad = OperatorAdapter(A)
    res = residuals(ad, mu, X2)
    assert _cluster_bound(ad, mu, X2, res, [0, 1, 2]) is None
    enc = enclose(ad, mu, X2)
    assert not enc.all_distinct


def test_theorem2_radius_contains_distinct_eigenvalues_random():
    import itertools
    rng = np.random.default_rng(5)
    for _ in range(40):
        n, g = 6, 3
        A = rng.normal(size=(n, n))
        A = (A + A.T) / 2
        w, V = la.eigh(A)
        C = np.eye(g) + rng.choice([0.1, 0.5, 0.8]) * rng.normal(size=(g, g))
        X = V[:, :g] @ C
        mu = w[:g] + rng.choice([0.0, 1e-5]) * rng.normal(size=g)
        ad = OperatorAdapter(A)
        res = residuals(ad, mu, X)
        cb = _cluster_bound(ad, mu, X, res, list(range(g)))
        if cb is None:
            continue
        radius, eta, rho, info = cb
        best = min(max(abs(mu[j] - w[i]) for j, i in enumerate(inj)) for inj in itertools.permutations(range(n), g))
        assert best <= rho and best <= info["eps_star"] and best <= radius


def test_subspace_and_decomposition_shift_return_rho():
    iso, inv, Hn, Mn = 1e-12, 1e-11, 3.0, 2.0
    delta = up(1 - np.sqrt(1 - iso)) + 4 * U
    rho = up(inv + (Hn + Mn) * delta) + ULP_MIN
    assert _subspace_shift(iso, inv, Hn, Mn) == rho
    assert _subspace_shift(1.0, inv, Hn, Mn) is None
    # decomposition_shift: eta from the block Frobenius bound, D from the invariance defects, rho as above
    from certify_ed.sectors import DecompositionAudit, SectorAudit
    secs = [SectorAudit("a", 2, 1e-13, 2e-13, 0.0, True), SectorAudit("b", 3, 3e-13, 1e-13, 0.0, True)]
    aud = DecompositionAudit(secs, 4e-13, 5, 5, None, True)
    eta = norm2_upper(np.array([1e-13, 3e-13, 4e-13, 4e-13]))
    D = norm2_upper(np.array([2e-13, 1e-13]))
    d2 = up(1 - np.sqrt(1 - eta)) + 4 * U
    assert decomposition_shift(aud, Hn, Mn) == up(D + (Hn + Mn) * d2) + ULP_MIN


# ------------------------------------------------------------------ Corollary 5 and Theorem 8
def test_certify_lowest_outputs_are_pinned_to_corollary_5(heis6):
    spec, H, w, V = heis6
    k = 5
    enc = enclose(OperatorAdapter(H), w[:k], V[:, :k])
    Delta = 1e-13
    st, ev, window = _certify_lowest(H, np.sort(w[:k]), enc, Options(), eps_shift=Delta)
    assert st == "CERTIFIED"
    last = ev["counts"][-1]
    beta = float(ce.fp.add_up(last["beta"], Delta))
    s_minus_b = float(ce.fp.sub_down(last["sigma"], beta))
    assert ev["next_eigenvalue_lower_bound"] == s_minus_b and window == (float("-inf"), s_minus_b)
    rad = ce.fp.add_up(enc.radius, Delta)
    hi_k = float(ce.fp.add_up(np.sort(w[:k]), rad).max())
    assert hi_k < s_minus_b and ev["gap_lower_bound"] == float(ce.fp.sub_down(s_minus_b, hi_k))
    # the conclusion for a perturbed model H + E with ||E|| <= Delta (Weyl transfer, Step 4)
    rng = np.random.default_rng(1)
    E = rng.normal(size=H.shape)
    E = (E + E.T) / 2
    E *= Delta / la.norm(E, 2)
    lam = la.eigvalsh(H.toarray() + E)
    assert np.all(np.abs(lam[:k] - np.sort(w[:k])) <= rad + 1e-15) and lam[k] >= s_minus_b
    assert np.sum(lam < s_minus_b) == k


def test_certify_lowest_negative_controls(heis6):
    spec, H, w, V = heis6
    # a missing eigenvalue below the top claim: the second eigenvalue is skipped
    keep = [0, 2, 3]
    enc = enclose(OperatorAdapter(H), w[keep], V[:, keep])
    st, ev, _ = _certify_lowest(H, np.sort(w[keep]), enc, Options())
    assert st != "CERTIFIED"
    # a duplicated vector: not all distinct, never CERTIFIED
    X = V[:, :3].copy()
    X[:, 1] = X[:, 0]
    mu = w[:3].copy()
    mu[1] = mu[0]
    enc = enclose(OperatorAdapter(H), mu, X)
    st, ev, _ = _certify_lowest(H, np.sort(mu), enc, Options())
    assert st != "CERTIFIED"


def test_compose_eps_is_rho_plus_builder_shift_and_the_statement_holds(heis6):
    from certify_ed.compose import SectorResult, certify_low_energy
    spec, H, w, V = heis6
    full = full_state_set(spec)
    secs = []
    for nup in range(spec.n_sites + 1):
        st = ce.u1_state_set(spec, nup)
        idx, found = full.index_of(st.states_in_order())
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        Hs = (Vs.T @ H @ Vs).toarray()
        ws, Vw = la.eigh(Hs)
        secs.append(SectorResult(f"nup={nup}", Vs, sp.csr_matrix(Hs), ws, Vw))
    E_cut = float((w[4] + w[5]) / 2)
    out = certify_low_energy(spec, secs, E_cut, H_full=H)
    assert out["status"] == "CERTIFIED", out["summary"]
    Hn = inf_norm_upper(hermitian_part(H))
    Bn = max(inf_norm_upper(s.H) for s in secs)
    delta = up(1 - np.sqrt(1 - out["eta"])) + 4 * U
    rho = up(out["invariance_total"] + (Hn + Bn) * delta) + ULP_MIN
    assert out["eps_decomposition"] == up(rho + out["builder_shift"])
    assert out["radius"] == float(ce.fp.add_up(out["max_sector_radius"], out["eps_decomposition"]))
    K = out["K"]
    assert np.all(np.abs(np.sort(w)[:K] - np.array(out["lowest"])) <= out["radius"])
    assert w[K] >= out["next_lower_bound"]


# ------------------------------------------------------------------ Appendix A
def test_hermitian_part_symmetrises_only_when_needed():
    spec = ce.heisenberg_chain(4)
    H = build_bitwise(spec)
    assert hermitian_part(H) is not None and (hermitian_part(H) - H).nnz == 0
    M = sp.csr_matrix(np.array([[1.0, 2.0], [2.0 + 1e-16, 3.0]]))
    Hs = hermitian_part(M)
    assert (Hs - Hs.conj().T).nnz == 0 or abs(Hs - Hs.conj().T).max() == 0.0
    assert Hs[0, 1] == (2.0 + (2.0 + 1e-16)) / 2


def test_spec_adapter_width_counts_amplitude_roundings():
    spec = ce.spin1_heisenberg_chain(3)
    ad = OperatorAdapter(SpecOperator(spec))
    max_len = max(len(t.op) for t in spec.terms)
    assert ad.width == spec.n_terms * 2 ** max_len + 2 * max_len + 2


def test_builder_shift_dominates_the_build_error_for_a_spin1_chain():
    from certify_ed.checker import exact_matrix
    spec = ce.spin1_heisenberg_chain(3)
    ex = exact_matrix(spec)
    n = len(ex.weights)
    d = np.sqrt(np.array([float(x) for x in ex.weights]))
    # H_spec = D K D^-1 in extended precision through Fractions -> float at the end (entries exact up to 1 ulp)
    Hspec = np.zeros((n, n), dtype=complex)
    for r, c, a, b in zip(ex.rows, ex.cols, ex.re, ex.im):
        Hspec[r, c] += (float(Fraction(int(a), int(ex.den))) + 1j * float(Fraction(int(b), int(ex.den)))) * d[r] / d[c]
    Href = build_bitwise(spec).toarray()
    assert la.norm(Href - Hspec, 2) <= _builder_shift(spec) + 4 * n * U * np.abs(Hspec).max()
    assert la.norm(hermitian_part(build_bitwise(spec)).toarray() - Hspec, 2) <= _builder_shift(spec) + 4 * n * U * np.abs(Hspec).max()


# ------------------------------------------------------------------ INT-05: underflow terms
def test_norm2_upper_covers_underflowed_squares():
    v = np.full(1000, 2.0 ** -540)                 # every square flushes to zero
    assert np.linalg.norm(v) == 0.0
    exact = np.sqrt(1000.0) * 2.0 ** -540
    assert norm2_upper(v) >= exact and norm2_lower(v) == 0.0
    vc = np.full(1000, 2.0 ** -540 * (1 + 1j))
    assert norm2_upper(vc) >= np.sqrt(2000.0) * 2.0 ** -540
    # the term is invisible for normal vectors
    x = np.array([1.0, -2.0, 0.5])
    assert norm2_upper(x) == float(np.linalg.norm(x)) * (1 + gamma(5)) * _GUARD
    assert norm_underflow_term(1) >= np.sqrt(0.5) * 2.0 ** -537


def test_exact_dot_with_tiny_and_subnormal_products():
    x = np.array([1e-300, 1.0, 2.0 ** -1030, 3e-200])
    y = np.array([1e-300, 3.0, 2.0 ** 70, 2e-200])
    s, err = exact_dot(x, y)
    exact = sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(x, y))
    assert abs(Fraction(s) - exact) <= Fraction(err)
    assert err >= 2 * TINY_BOUND                   # two products below 2^-967 were left out
    s2, err2 = exact_dot(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert s2 == 11.0 and err2 == 2 * U * 11.0    # no tiny products: the pre-branch bound


def test_scaled_operator_radius_encloses_the_eigenvalues():
    """The INT-05 case of the Theorem 2 verification: an operator scaled by 1e-150 gave radius 0 before the fix."""
    rng = np.random.default_rng(2)
    n, g = 5, 2
    A = rng.normal(size=(n, n))
    A = (A + A.T) / 2
    w, V = la.eigh(A)
    s = 1e-150
    X = V[:, :g] @ (np.eye(g) + 0.2 * rng.normal(size=(g, g)))
    mu = w[:g] * s
    enc = enclose(OperatorAdapter(A * s), mu, X)
    assert np.all(np.isfinite(enc.radius)) and np.all(enc.radius > 0)
    lam = la.eigvalsh(A * s)
    for i in range(g):
        assert np.min(np.abs(lam - mu[i])) <= enc.radius[i]


def test_residual_radius_never_shrinks_against_the_pre_branch_formula(heis6):
    """rel_new >= rel_old on every pair, with rel_new - rel_old <= D of docs/proofs/int05_underflow.md (Theorem U6);
    bit identity for the normal-sized radii of the fixture."""
    spec, H, w, V = heis6
    ad = OperatorAdapter(H)
    res = residuals(ad, w, V)
    n = H.shape[0]
    for i in range(0, w.size, 7):
        Rc, Ec, x = res.R[:, i], res.E[:, i], V[:, i]
        R_old = float(np.linalg.norm(Rc)) * (1 + gamma(n + 2)) * _GUARD
        E_old = float(np.linalg.norm(Ec)) * (1 + gamma(n + 2)) * _GUARD
        xl_old = float(np.linalg.norm(x)) * (1 - gamma(n + 2)) / _GUARD
        rel_old = (R_old + E_old) / xl_old * (1 + 4 * U)
        rel_new = float(res.rel_bound[i])
        T = norm_underflow_term(n)
        A_ = residual_additive_term(n, ad.width, False)
        D = 2 * _GUARD * ((2 * T + A_) / res.xnorm_lower[i] + (R_old + E_old) * T / (res.xnorm_lower[i] * xl_old)) + 4 * ULP_MIN
        assert rel_old <= rel_new <= rel_old + D
        assert rel_new == rel_old                  # normal-sized: bit identical


def test_tiny_vector_radius_is_a_bound_in_exact_arithmetic():
    A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    x = np.array([1e-160, 1.5e-160, 1e-160])        # the squares inside the norms are subnormal (inexact)
    mu = 0.5
    res = residuals(OperatorAdapter(A), np.array([mu]), x[:, None])
    r = [sum(Fraction(A[i, j]) * Fraction(x[j]) for j in range(3)) - Fraction(mu) * Fraction(x[i]) for i in range(3)]
    r2 = sum(t ** 2 for t in r)
    x2 = sum(Fraction(t) ** 2 for t in x)
    rel = float(res.rel_bound[0])
    assert np.isfinite(rel) and Fraction(rel) ** 2 * x2 >= r2
    # below the resolution of the norm term the lower bound on ||x|| is 0 and the radius is +inf (a valid bound)
    res2 = residuals(OperatorAdapter(A), np.array([mu]), (x * 1e-110)[:, None])
    assert res2.rel_bound[0] == np.inf and res2.xnorm_lower[0] == 0.0


# ------------------------------------------------------------------ mutation survivors (RESULT_integration_3.2.md)
def _label_vector(spec, eps):
    """One column in the one-up sector of a 2-site chain: cos * symmetric + eps * antisymmetric (translation
    eigenvalues +1 and -1 exactly), so that the label defect is 2 eps up to rounding."""
    from certify_ed.basis import digit_sum
    full = full_state_set(spec)
    states = full.states_in_order()
    ds = digit_sum(states.astype(np.int64), spec.q, spec.n_sites)
    rows = np.nonzero(ds == 1)[0]
    c = np.sqrt(max(0.0, 1.0 - eps * eps))
    v = c * np.array([1.0, 1.0]) / np.sqrt(2.0) + eps * np.array([1.0, -1.0]) / np.sqrt(2.0)
    return sp.csr_matrix((v, (rows, np.zeros(2, int))), shape=(spec.dim, 1)), rows


def test_label_audit_defect_within_rounding_allowance_of_the_threshold():
    """sectors.py:240 `delta = up(defects[k] + rnd)`: a planted defect 2 eps with eps two ulps below 1/2 puts the
    leakage bound within the rounding allowance rnd of the decision threshold 1/2 (leak = delta/((1 - iso) sep),
    sep = 2). With the allowance added the label is not certified; a mutant subtracting rnd would certify it
    (leak 0.49999999999999966 < 0.5, found by bisection over binary64 eps)."""
    spec = ce.heisenberg_chain(2)
    V, rows = _label_vector(spec, 0.4999999999999998)
    r = audit_label(spec, V, "translation")
    assert r["j_physics"] == 0 and not r["certified"] and r["leakage_bound"] >= 0.5
    # the allowance is present: delta >= computed defect + rnd
    from certify_ed.basis import translation
    from certify_ed.sectors import symmetry_matrix
    st = ce.u1_state_set(spec, 1)
    Vd = sp.csr_matrix(V.toarray()[rows])
    G = symmetry_matrix(st, translation(spec))
    defect = fro_upper((G @ Vd - 1.0 * Vd).toarray())
    rnd = gamma_dot(4, True) * 2 * fro_upper(np.abs(Vd.data))
    assert r["defect_upper"] >= defect + rnd * (1 - 1e-10) > defect
    # clearly inside: certified; clearly outside: not
    assert audit_label(spec, _label_vector(spec, 0.49)[0], "translation")["certified"]
    assert not audit_label(spec, _label_vector(spec, 0.51)[0], "translation")["certified"]


def test_kato_temple_second_lowest_pair_uses_its_lower_neighbour_as_isolation_end():
    """certify.py `j_lo = order[pos - 1] if pos > 0 else None`: for the second-lowest pair (pos = 1) the lower end of the
    isolation interval is the upper end of the lowest enclosure. A mutant `pos > 1` would use -inf there, and
    Kato-Temple's upper bound would collapse to the Rayleigh quotient theta < lambda_2 for a vector contaminated
    with the ground state: an unsound interval. Exact eigenvalues 0 and 1 (diagonal A)."""
    A = np.diag([0.0, 1.0, 5.0, 7.0, 9.0])
    t = 1e-4
    x2 = np.array([t, 1.0, 0.0, 0.0, 0.0])
    x2 /= np.linalg.norm(x2)
    X = np.column_stack([np.eye(5)[:, 0], x2])
    claim = ce.Claim(np.array([0.0, 1.0]), X, operator=A, kind="lowest", accuracy_target=1e-3)
    cert = ce.certify(claim, None, Options())
    st = {s["id"]: s["status"] for s in cert.statements}
    assert st["S2.eigenpairs"] == "CERTIFIED" and st["S3.completeness"] == "CERTIFIED"
    kt = {r["index"]: r for r in cert.body["kato_temple"]}
    assert 1 in kt, cert.body["kato_temple"]
    rec = kt[1]
    theta = float(x2 @ A @ x2)
    assert theta < 1.0                                  # the Rayleigh quotient alone would be unsound as an upper end
    assert rec["lower"] <= 1.0 <= rec["upper"]
    assert rec["isolation_interval"][0] > -np.inf and rec["isolation_interval"][0] >= 0.0


def test_float_vector_to_int_reconstructs_the_vector_exactly():
    """checker.py `(m * 2.0 ** 53).astype(np.int64)` with frexp's m in [1/2, 1): the integer mantissa is m 2^53 and the
    exponent e - 53. A mutant scale 2^54 doubles every entry (the radius of the exact layer is scale invariant, so
    only a reconstruction test sees it)."""
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.standard_normal(5) * 10.0 ** rng.integers(-200, 200, 5), [0.0, 2.0 ** -1030, -1.0]])
    xr, xi, E = _float_vector_to_int(x)
    for k in range(x.size):
        assert Fraction(int(xr[k])) * Fraction(2) ** E == Fraction(float(x[k])) and int(xi[k]) == 0
    z = x + 1j * rng.standard_normal(x.size)
    zr, zi, E2 = _float_vector_to_int(z)
    for k in range(z.size):
        assert Fraction(int(zr[k])) * Fraction(2) ** E2 == Fraction(float(z[k].real))
        assert Fraction(int(zi[k])) * Fraction(2) ** E2 == Fraction(float(z[k].imag))
