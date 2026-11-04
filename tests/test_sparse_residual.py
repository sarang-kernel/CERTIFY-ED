"""D-01: a-posteriori residual bound of the sparse S3 path (docs/proofs/D-01_sparse-residual.md).

Pins: the sandwich of Lemma 4 with the a-posteriori beta on planted-inertia and random sparse matrices (ground
truth: dense eigh); the entrywise inequality against an exact rational residual; determinism across thread counts
and block sizes; the memory gate; the adversarial cases; and the negative controls (a corrupted factor or pivot
changes the residual and never produces a wrong count with a small beta)."""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fractions import Fraction

import certify_ed as ce
from certify_ed.inertia import inertia, inertia_sparse
from certify_ed.ldl_residual import sparse_ldl_residual_bound, _row_blocks


def _rand_sparse_herm(rng, n, cplx, density=0.05, scale=1.0):
    k = max(2 * n, int(density * n * n))
    r, c = rng.integers(0, n, k), rng.integers(0, n, k)
    v = rng.standard_normal(k) * scale
    if cplx:
        v = v + 1j * rng.standard_normal(k) * scale
    A = sp.coo_matrix((v, (r, c)), shape=(n, n)).tocsr()
    A = ((A + A.conj().T) * 0.5 + sp.diags(rng.standard_normal(n) * scale)).tocsr()
    A.sort_indices()
    return A


def _factor(S):
    return spla.splu(sp.csc_matrix(S), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options={"SymmetricMode": True, "Equil": False})


def _sandwich_holds(A, sigma, res):
    w = np.linalg.eigvalsh(A.toarray())
    return int(np.sum(w < sigma - res.beta)) <= res.negative <= int(np.sum(w <= sigma + res.beta))


@pytest.mark.parametrize("seed", range(30))
def test_sandwich_random_sparse(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 160))
    A = _rand_sparse_herm(rng, n, bool(seed % 2), density=[0.02, 0.05, 0.1][seed % 3])
    w = np.linalg.eigvalsh(A.toarray())
    for sigma in (float(np.median(w)), float(w[n // 4] + 1e-3), float(w.min() - 0.5), float(w.max() + 0.5)):
        res = inertia_sparse(A, sigma)
        assert res.method == "sparse-superlu-residual"
        if res.conclusive:
            assert _sandwich_holds(A, sigma, res), (seed, sigma, res)
            assert res.beta < 1e-5 * max(1.0, float(np.abs(A.data).max()))   # sanity only (no pivoting: growth)


@pytest.mark.parametrize("n_neg", [0, 3, 17, 40])
def test_planted_inertia(n_neg):
    """Q diag(d) Q^T with a sparse orthogonal Q (product of a few Givens rotations) and a planted sign pattern."""
    rng = np.random.default_rng(n_neg)
    n = 80
    d = np.abs(rng.standard_normal(n)) + 0.1
    d[:n_neg] *= -1
    A = sp.diags(d).tocsr()
    for _ in range(60):
        i, j = rng.choice(n, 2, replace=False)
        th = rng.uniform(0, 2 * np.pi)
        G = sp.identity(n, format="lil")
        G[i, i] = G[j, j] = np.cos(th)
        G[i, j] = np.sin(th)
        G[j, i] = -np.sin(th)
        G = G.tocsr()
        A = (G @ A @ G.T).tocsr()
    A = ((A + A.T) * 0.5).tocsr()
    res = inertia_sparse(A, 0.0)
    assert res.conclusive and res.negative == n_neg
    assert res.beta < 0.1       # no eigenvalue within 0.1 of 0 by construction, so the count is exactly the planted one


def test_entrywise_bound_against_exact_rational_residual():
    rng = np.random.default_rng(7)
    A = _rand_sparse_herm(rng, 14, False, density=0.3)
    f = _factor(A)
    L = f.L.tocsr()
    D = f.U.diagonal()
    rb = sparse_ldl_residual_bound(L, D, A, f.perm_c, block_mb=0.001, threads=1, min_block_rows=1)
    ip = np.argsort(f.perm_c)
    SP = sp.csr_matrix(A)[ip][:, ip].toarray()
    Ld = L.toarray()
    n = 14
    rows = []
    for i in range(n):
        s = Fraction(0)
        for j in range(n):
            acc = Fraction(0)
            for k in range(min(i, j) + 1):
                acc += Fraction(float(Ld[i, k])) * Fraction(float(np.real(D[k]))) * Fraction(float(Ld[j, k]))
            s += abs(acc - Fraction(float(SP[i, j])))
        rows.append(s)
    assert max(rows) <= Fraction(rb.beta)
    assert rb.blocks > 1          # the tiny block budget exercised the mirrored column sums


def test_deterministic_across_threads_and_equal_through_inertia():
    rng = np.random.default_rng(11)
    A = _rand_sparse_herm(rng, 400, True, density=0.02)
    r1 = inertia_sparse(A, 0.1, threads=1)
    r8 = inertia_sparse(A, 0.1, threads=8)
    assert r1.beta == r8.beta and r1.negative == r8.negative
    assert r1.residual["blocks"] == r8.residual["blocks"]
    r = inertia(A, 0.1, method="sparse")
    assert r.beta == r1.beta and r.method == "sparse-superlu-residual"
    # a different block size changes the partition, not the sandwich
    rs = inertia_sparse(A, 0.1, block_mb=0.01)
    assert rs.residual["blocks"] > r1.residual["blocks"] and rs.negative == r1.negative
    assert _sandwich_holds(A, 0.1, rs)


def test_row_blocks_cover_every_row_once():
    rng = np.random.default_rng(3)
    A = _rand_sparse_herm(rng, 300, False, density=0.03)
    f = _factor(A)
    L = f.L.tocsr()
    for mb in (0.001, 0.01, 1.0, 64.0):
        blocks = _row_blocks(L.shape[0], L, 8, mb)
        assert blocks[0][0] == 0 and blocks[-1][1] == L.shape[0]
        assert all(b0 == a1 for (_, a1), (b0, _) in zip(blocks[:-1], blocks[1:]))


def test_apriori_method_still_available_and_never_tighter_than_measured_residual_on_chain():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec, ce.u1_state_set(spec, 5))
    post = inertia(H, -4.0, method="sparse")
    pri = inertia(H, -4.0, method="sparse-apriori")
    assert post.conclusive and pri.conclusive and post.negative == pri.negative
    assert post.method == "sparse-superlu-residual" and pri.method == "sparse-superlu-apriori"
    assert post.beta <= pri.beta
    assert post.residual["residual_inf"] <= post.beta and post.residual["rounding_inf"] <= post.beta


def test_memory_gate_reports_resource_failure_not_a_count():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec, ce.u1_state_set(spec, 5))
    r = inertia(H, -4.0, method="sparse", budget_mb=0.5)
    assert not r.conclusive and r.resource_failure and r.negative is None and "not attempted" in r.note
    ok = inertia(H, -4.0, method="sparse", budget_mb=10 ** 6)
    assert ok.conclusive


def test_adversarial_zero_pivot_and_exact_eigenvalue_are_inconclusive():
    assert not inertia_sparse(sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])), 0.0).conclusive
    assert not inertia_sparse(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]])), 0.0).conclusive   # eigenvalue at sigma
    r = inertia_sparse(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0 + 2.0 ** -52]])), 0.0)
    if r.conclusive:        # eigenvalue 2^-52 from sigma: inside the collar
        assert r.beta >= 2.0 ** -53


def test_adversarial_ill_conditioned_pivot_growth_shows_in_beta():
    for eps in (1e-6, 1e-12):
        B = sp.csr_matrix(np.array([[eps, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]))
        r = inertia_sparse(B, 0.0)
        assert r.conclusive and _sandwich_holds(B, 0.0, r)
        assert r.beta >= 1e-16 / eps / 4     # growth 1/eps without pivoting is measured, not assumed away


def test_huge_scale_sandwich_or_inconclusive_and_nonfinite_factors_refused():
    rng = np.random.default_rng(5)
    for scale in (1e160, 1e300):
        A = _rand_sparse_herm(rng, 8, False, density=0.6, scale=scale)
        r = inertia_sparse(A, 0.0)
        assert (not r.conclusive) or _sandwich_holds(A, 0.0, r)
    A = _rand_sparse_herm(rng, 8, False, density=0.6)
    f = _factor(A)
    L = f.L.tolil()
    L[5, 2] = np.inf
    with pytest.raises(FloatingPointError):
        sparse_ldl_residual_bound(L.tocsr(), f.U.diagonal(), A, f.perm_c)


# ---------------------------------------------------------------- negative controls

def test_negative_control_corrupted_factor_entry_raises_beta():
    """A wrong stored factor must show up in the residual: beta grows by the size of the corruption."""
    rng = np.random.default_rng(21)
    A = _rand_sparse_herm(rng, 60, False, density=0.08)
    f = _factor(A)
    L = f.L.tocsr()
    D = f.U.diagonal()
    good = sparse_ldl_residual_bound(L, D, A, f.perm_c)
    Lc = L.tolil()
    Lc[40, 3] = Lc[40, 3] + 1e-3          # one wrong entry in the strict lower triangle
    bad = sparse_ldl_residual_bound(Lc.tocsr(), D, A, f.perm_c)
    assert bad.beta > good.beta + 1e-4, (good.beta, bad.beta)


def test_negative_control_flipped_pivot_sign_is_caught_by_the_residual():
    """Flipping the sign of one pivot changes the count by one; the residual then exceeds the gap so the
    sandwich still holds with the (large) measured beta -- the wrong count is never certified with a small collar."""
    rng = np.random.default_rng(22)
    A = _rand_sparse_herm(rng, 50, True, density=0.1)
    f = _factor(A)
    L = f.L.tocsr()
    D = f.U.diagonal().copy()
    good = sparse_ldl_residual_bound(L, D, A, f.perm_c)
    Dbad = D.copy()
    Dbad[10] = -Dbad[10]
    bad = sparse_ldl_residual_bound(L, Dbad, A, f.perm_c)
    assert bad.beta > good.beta * 1e3
    assert bad.beta >= 2 * abs(float(np.real(D[10]))) * (1 - 1e-12)   # the perturbation is at least the 2x2 block norm


def test_negative_control_wrong_permutation_raises_beta():
    rng = np.random.default_rng(23)
    A = _rand_sparse_herm(rng, 40, False, density=0.1)
    f = _factor(A)
    L = f.L.tocsr()
    D = f.U.diagonal()
    good = sparse_ldl_residual_bound(L, D, A, f.perm_c)
    perm = f.perm_c.copy()
    perm[[0, 1]] = perm[[1, 0]]
    bad = sparse_ldl_residual_bound(L, D, A, perm)
    assert bad.beta > good.beta * 1e3


def test_negative_control_planted_wrong_count_in_certify_is_not_certified():
    """A claim that omits the lowest eigenvalue: S3 must not be CERTIFIED (FAILED: the sparse count finds one more)."""
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec, ce.u1_state_set(spec, 5))
    w, V = np.linalg.eigh(H.toarray())
    c = ce.quick(H, w[1:6], V[:, 1:6], spec, inertia_method="sparse")
    s3 = [s for s in c.statements if s["id"] == "S3.completeness"][0]
    assert s3["status"] in ("FAILED", "INCONCLUSIVE")
    assert all(cnt["method"] == "sparse-superlu-residual" for cnt in s3["evidence"].get("counts", []))
    ok = ce.quick(H, w[:5], V[:, :5], spec, inertia_method="sparse")   # k = 5 closes the lowest multiplet
    assert [s for s in ok.statements if s["id"] == "S3.completeness"][0]["status"] == "CERTIFIED"


def test_residual_threads_default_to_one_and_opt_in_by_environment(monkeypatch):
    """3.2.1: the a-posteriori residual runs its block products on one thread unless CERTIFY_ED_RESIDUAL_THREADS asks
    for more (under a committed-memory cap, concurrent sparse products crashed the process in the 3.2.0 run, E4 W4)."""
    from certify_ed.cost import residual_threads
    monkeypatch.delenv("CERTIFY_ED_RESIDUAL_THREADS", raising=False)
    assert residual_threads() == 1
    monkeypatch.setenv("CERTIFY_ED_RESIDUAL_THREADS", "4")
    assert residual_threads() == max(1, min(4, 8, os.cpu_count() or 1))
    monkeypatch.setenv("CERTIFY_ED_RESIDUAL_THREADS", "not-a-number")
    assert residual_threads() == 1


def test_residual_bound_does_not_depend_on_the_default_thread_count(monkeypatch):
    """The recorded bound is identical whether the default (1) or an opt-in count is used."""
    import numpy as np
    import scipy.sparse as sp
    from certify_ed.inertia import inertia_sparse
    rng = np.random.default_rng(3)
    n = 300
    A = sp.random(n, n, density=0.02, random_state=rng, format="csr"); A = (A + A.T).tocsc() + sp.identity(n) * 5
    monkeypatch.delenv("CERTIFY_ED_RESIDUAL_THREADS", raising=False)
    r1 = inertia_sparse(A, 0.1)
    monkeypatch.setenv("CERTIFY_ED_RESIDUAL_THREADS", "8")
    r8 = inertia_sparse(A, 0.1)
    assert r1.beta == r8.beta and r1.negative == r8.negative
