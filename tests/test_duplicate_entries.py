"""Repeated entries of a sparse matrix (CSR or CSC with repeated indices, COO with repeated coordinates) are summed
on a copy before any product (precision.summed), so that every product, norm and modulus of a call refers to one
matrix, the summed one, and the caller's matrix stays as given.

Before, the direct calls formed their products from the stored entries and took the moduli of the summed matrix
(abs_up sums in place): with entries that cancel, the rounding of the products was not covered. Example below:
commutator_bound gave 64.0 for a commutator whose exact row-sum norm is 93.87. fro_upper read the stored values:
two entries 1 at one position gave sqrt(2) for a norm of 2.
"""
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.fp import fro_upper, inf_norm_upper
from certify_ed.precision import PrecisionError, summed


def _arrays(M):
    return tuple(np.array(x, copy=True) for x in (M.data, M.indices, M.indptr))


def _unchanged(M, before):
    return all(np.array_equal(x, y) for x, y in zip(before, (M.data, M.indices, M.indptr)))


def _cancelling():
    """[[0, 256], [256, 0]] with each off-diagonal entry stored as 2^60 and -2^60 + 256 (exact; the sum is 256)."""
    t = 2.0 ** 60
    return sp.csr_matrix((np.array([t, -t + 256.0, t, -t + 256.0]), np.array([1, 1, 0, 0]), np.array([0, 2, 4])),
                         shape=(2, 2))


def _split(H):
    """H with every entry stored twice, as two exact halves (the sum is H exactly)."""
    C = sp.csr_matrix(H).tocoo()
    rows, cols, vals = np.r_[C.row, C.row], np.r_[C.col, C.col], np.r_[C.data, C.data] * 0.5
    o = np.lexsort((cols, rows))
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=H.shape[0]))]
    return sp.csr_matrix((vals[o], cols[o], indptr), shape=H.shape)


def _exact_commutator_inf(H, S):
    He = [[Fraction(x) for x in r] for r in H.toarray()]
    Se = [[Fraction(x) for x in r] for r in S.toarray()]
    n = len(He)
    C = [[sum(He[i][k] * Se[k][j] - Se[i][k] * He[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return max(sum(abs(x) for x in r) for r in C)


def test_commutator_bound_covers_entries_that_cancel():
    H = _cancelling()
    S = sp.csr_matrix(np.array([[1.0 / 3, 0.1], [0.1, 0.7]]))
    before = _arrays(H)
    exact = _exact_commutator_inf(summed(H), S)
    assert float(exact) > 93.86
    from certify_ed.davis_kahan import commutator_bound
    r = commutator_bound(H, S)
    assert Fraction(r["bound"]) >= exact, (r, float(exact))
    assert r == commutator_bound(summed(H), S)               # the bound of the summed matrix
    assert _unchanged(H, before)
    # negative control: the products of the stored entries are not covered by the moduli of the summed matrix
    stored = (H @ S - S @ H).toarray()                       # products formed from the stored entries
    assert np.abs(stored).sum(axis=1).max() < float(exact) - 20


def test_fro_upper_reads_the_summed_matrix():
    D = sp.csr_matrix((np.array([1.0, 1.0]), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 2))
    before = _arrays(D)
    assert fro_upper(D) >= 2.0 and np.sqrt(np.sum(D.data ** 2)) < 1.5      # the stored values give sqrt(2)
    assert _unchanged(D, before)


def test_inf_norm_upper_leaves_the_matrix_as_given():
    H = _cancelling()
    before = _arrays(H)
    assert inf_norm_upper(H) >= 256.0 and inf_norm_upper(H) == inf_norm_upper(summed(H))
    assert _unchanged(H, before)


def test_summed_keeps_the_caller_matrix_and_is_the_identity_on_canonical_input():
    H = _cancelling()
    before = _arrays(H)
    S = summed(H)
    assert S is not H and S.has_canonical_format and S.nnz == 2 and np.array_equal(S.toarray(), [[0, 256], [256, 0]])
    assert _unchanged(H, before) and summed(S) is S
    H.indices, H.indptr = H.indices.astype(np.int64), H.indptr.astype(np.int64)
    assert summed(H).indices.dtype == summed(H).indptr.dtype == np.int64
    C = sp.coo_matrix((np.array([1.0, 2.0]), (np.array([0, 0]), np.array([1, 1]))), shape=(2, 2))
    assert summed(C).format == "coo" and summed(C).toarray()[0, 1] == 3.0 and C.nnz == 2
    for M in (np.eye(2), sp.dia_matrix(np.eye(2)), sp.lil_matrix(np.eye(2))):
        assert summed(M) is M


def test_summed_refuses_what_it_cannot_sum_exactly():
    big = sp.csr_matrix((np.array([2 ** 53, 1], dtype=np.int64), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 2))
    with pytest.raises(PrecisionError, match="not exactly representable"):
        summed(big, "op")
    wide = sp.csr_matrix((np.array([2 ** 62, 2 ** 62], dtype=np.int64), np.array([0, 0]), np.array([0, 2, 2])),
                         shape=(2, 2))
    with pytest.raises(PrecisionError, match="overflow"):
        summed(wide, "op")
    ok = sp.csr_matrix((np.array([3, 4], dtype=np.int64), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 2))
    assert summed(ok).dtype == np.int64 and summed(ok)[0, 0] == 7
    narrow = sp.csr_matrix((np.array([1.0, 2.0 ** -30], dtype=np.float32), np.array([0, 0]), np.array([0, 2, 2])),
                           shape=(2, 2))
    with pytest.raises(PrecisionError, match="did not convert"):
        summed(narrow)                                       # a narrower format would be summed in its own precision


def _chain(n=8):
    spec = ce.heisenberg_chain(n)
    return spec, ce.build_bitwise(spec).tocsr()


def test_direct_calls_give_the_results_of_the_summed_matrix():
    """Each entry point, called with a matrix whose entries are stored twice (two exact halves), gives exactly the
    result for the canonical matrix, and leaves the caller's matrix as given."""
    from certify_ed.construct import hermitian_part, matrix_difference
    from certify_ed.correlations import compress_operator
    from certify_ed.davis_kahan import commutator_bound
    from certify_ed.enclosure import OperatorAdapter
    from certify_ed.sectors import audit_sector
    from certify_ed.spectrum import enclose_spectrum
    spec, H = _chain(8)
    rng = np.random.default_rng(3)
    Z = sp.diags(rng.standard_normal(H.shape[0])).tocsr()
    V = sp.identity(H.shape[0], format="csr")
    X = rng.standard_normal((H.shape[0], 3))
    calls = {
        "commutator_bound": lambda A: commutator_bound(A, Z),
        "audit_sector": lambda A: audit_sector(A, V, sp.csr_matrix(H), "full").__dict__,
        "compress_operator": lambda A: (lambda r: (r[0].toarray().tolist(), r[1], r[2]))(compress_operator(V, A)),
        "hermitian_part": lambda A: hermitian_part(A).toarray().tolist(),
        "matrix_difference": lambda A: str(matrix_difference(A, H)),
        "enclose_spectrum": lambda A: str(enclose_spectrum(A)),
        "OperatorAdapter": lambda A: (lambda ad: (ad.width, (ad.matvec(X)).tolist(), ad.abs_matvec(np.abs(X)).tolist(),
                                                  ad.norm_upper()))(OperatorAdapter(A)),
        "fro_upper": fro_upper,
        "inf_norm_upper": inf_norm_upper,
    }
    for name, f in calls.items():
        D = _split(H)
        assert not D.has_canonical_format
        before = _arrays(D)
        got, ref = f(D), f(H)
        assert got == ref, name
        assert _unchanged(D, before), name


def test_certify_gives_the_certificate_of_the_summed_operator():
    spec, H = _chain(8)
    w, V = np.linalg.eigh(H.toarray())
    D = _split(H)
    before = _arrays(D)

    def run(A, obs):
        cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=A, kind="lowest"), spec,
                          ce.Options(seed=1, observables={"H": obs}))
        return [(s["id"], s["status"]) for s in cert.statements], cert.body["enclosures"], cert.body.get("eigenspaces")

    assert run(D, D) == run(H, H)
    assert _unchanged(D, before)


def test_the_documented_example_of_cancelling_entries():
    """docs/proofs/fp_modulus-up.md, section 1: (0, 1) and (1, 0) each hold 2^53 and -2^53 + 2 (summing to 2), S =
    diag(0, 0.7481628719198741). The exact ||[H, S]||_2 is 1.4963257438397481; before the summation at the start the
    bound was 1.0000000000000024."""
    from certify_ed.davis_kahan import commutator_bound
    t = 2.0 ** 53
    H = sp.csr_matrix((np.array([t, -t + 2.0, t, -t + 2.0]), np.array([1, 1, 0, 0]), np.array([0, 2, 4])), shape=(2, 2))
    S = sp.csr_matrix(np.diag([0.0, 0.7481628719198741]))
    b = commutator_bound(H, S)["bound"]
    assert b >= 1.4963257438397481 and b == commutator_bound(summed(H), S)["bound"]
