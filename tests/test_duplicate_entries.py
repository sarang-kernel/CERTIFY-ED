"""Repeated entries of a sparse matrix (CSR, CSC or BSR with repeated indices, COO with repeated coordinates, DIA with a
repeated offset) are summed on a copy before any product and before any format conversion (precision.summed), so that
every product, norm and modulus of a call refers to one matrix, the summed one, and the caller's matrix stays as given.
The sum at a position is the correctly rounded exact sum of its stored entries (the exact sum for integers), whatever
the format, the stored order or the entry point.

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
    with pytest.raises(PrecisionError, match="outside the range of int64"):
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


def _parts(vals, fmt, shape=(2, 2)):
    """A matrix whose (0, 0) entry is stored as the parts `vals`, in the order given, in format `fmt`."""
    vals = np.asarray(vals)
    k = vals.size
    if fmt == "coo":
        return sp.coo_matrix((vals, (np.zeros(k, int), np.zeros(k, int))), shape=shape)
    indptr = np.r_[0, k, np.full(shape[0] - 1, k)] if fmt != "csc" else np.r_[0, k, np.full(shape[1] - 1, k)]
    if fmt == "csr":
        return sp.csr_matrix((vals, np.zeros(k, int), indptr), shape=shape)
    if fmt == "csc":
        return sp.csc_matrix((vals, np.zeros(k, int), indptr), shape=shape)
    return sp.bsr_matrix((vals.reshape(k, 1, 1), np.zeros(k, int), indptr), shape=shape)


def test_the_sum_does_not_depend_on_format_or_order():
    """1 and 2^-53 twice: in stored order 1 + 2^-53 rounds to 1 (ties to even) and the next part is lost as well; the
    exact sum 1 + 2^-52 is a float. Every order and format gives it, real and complex, and so does matrix_sha256."""
    import itertools
    import math
    from certify_ed.certificate_v2 import matrix_sha256
    t = 2.0 ** -53
    exact = 1.0 + 2.0 ** -52
    assert (1.0 + t) + t == 1.0 and math.fsum([1.0, t, t]) == exact
    ref = sp.csr_matrix(np.array([[exact, 0.0], [0.0, 0.0]]))
    for perm in itertools.permutations([1.0, t, t]):
        for fmt in ("coo", "csr", "csc", "bsr"):
            for z in (1.0, 1.0 + 1.0j):
                M = _parts(np.array(perm) * z, fmt)
                S = summed(M)
                assert S.format == fmt and S.toarray()[0, 0] == exact * z, (perm, fmt, z)
                assert matrix_sha256(M) == matrix_sha256(ref * z)


def test_audit_decomposition_sees_one_summed_isometry():
    """Every check of audit_decomposition works on one summed copy of each isometry (a COO isometry with three parts at
    a position gives the audit of its summed copy)."""
    from certify_ed.sectors import audit_decomposition
    t = 2.0 ** -53
    H = sp.diags([1.0, 2.0]).tocsr()
    V0 = sp.coo_matrix((np.array([1.0, t, t]), (np.zeros(3, int), np.zeros(3, int))), shape=(2, 1))
    V1 = sp.coo_matrix((np.array([1.0]), (np.array([1]), np.array([0]))), shape=(2, 1))
    Hss = [sp.csr_matrix(np.array([[1.0]])), sp.csr_matrix(np.array([[2.0]]))]
    a = audit_decomposition(H, [V0, V1], Hss).to_dict()
    assert a == audit_decomposition(H, [summed(V0), V1], Hss).to_dict()
    assert V0.nnz == 3


def test_the_certificate_hash_is_the_hash_of_the_callers_matrix(tmp_path):
    """A COO operator with every entry stored as three parts (2^30, v, -2^30): certify() works on the summed matrix,
    which is the operator itself (the exact sum is v), and the certificate records the hash of the caller's matrix,
    of the operator, and of the file write_operator writes."""
    from certify_ed.certificate_v2 import matrix_sha256
    from certify_ed.io.formats import read_operator, write_operator
    spec, H = _chain(6)
    C = H.tocoo()
    big = np.full(C.nnz, 2.0 ** 30)
    rep = sp.coo_matrix((np.r_[big, C.data, -big], (np.tile(C.row, 3), np.tile(C.col, 3))), shape=H.shape)
    w, V = np.linalg.eigh(H.toarray())
    cert = ce.certify(ce.Claim(eigenvalues=w[:2], eigenvectors=V[:, :2], operator=rep, kind="lowest"), spec,
                      ce.Options(seed=1))
    ref = ce.certify(ce.Claim(eigenvalues=w[:2], eigenvectors=V[:, :2], operator=H, kind="lowest"), spec,
                     ce.Options(seed=1))
    h = cert.body["subject"]["operator"]["matrix_sha256"]
    assert h == matrix_sha256(rep) == matrix_sha256(H) == ref.body["subject"]["operator"]["matrix_sha256"]
    assert [(s["id"], s["status"]) for s in cert.statements] == [(s["id"], s["status"]) for s in ref.statements]
    assert cert.body["enclosures"] == ref.body["enclosures"]
    path = str(tmp_path / "op.npz")
    write_operator(path, rep)
    assert matrix_sha256(read_operator(path)) == h and rep.nnz == 3 * C.nnz


def test_integer_matrices_are_summed_exactly_and_kept_when_nothing_repeats():
    """No repeated entry: an integer matrix in COO or with unsorted indices is accepted as it is (before: refused by a
    guard on the sum of all its entries). Repeated entries: the exact sum, in int64 when it leaves the type
    (int8 100 + 100 = 200, not -56)."""
    C = sp.coo_matrix((np.array([100, 27, 100], dtype=np.int8), (np.array([0, 1, 1]), np.array([1, 0, 1]))),
                      shape=(2, 2))
    assert np.array_equal(summed(C).toarray(), C.toarray()) and summed(C).dtype == np.int8
    U = sp.csr_matrix((np.array([7, 9], dtype=np.int16), np.array([1, 0]), np.array([0, 2, 2])), shape=(2, 2))
    assert not U.has_sorted_indices and np.array_equal(summed(U).toarray(), U.toarray())
    R = sp.coo_matrix((np.array([100, 100, 5], dtype=np.int8), (np.array([0, 0, 1]), np.array([1, 1, 0]))),
                      shape=(2, 2))
    S = summed(R)
    assert S.dtype == np.int64 and S.toarray()[0, 1] == 200 and S.toarray()[1, 0] == 5
    from certify_ed.construct import hermitian_part, matrix_difference
    Rh = sp.coo_matrix((np.array([100, 100, 100, 100], dtype=np.int8), (np.array([0, 0, 1, 1]),
                                                                      np.array([1, 1, 0, 0]))), shape=(2, 2))
    assert hermitian_part(Rh).toarray()[0, 1] == 200
    assert matrix_difference(Rh, sp.csr_matrix(np.array([[0.0, 200.0], [200.0, 0.0]])))["max_abs_diff"] == 0.0


def test_the_file_readers_sum_as_the_direct_path(tmp_path):
    """A COO file with repeated integer or boolean entries is read with the sums of precision.summed (before: summed
    by tocsr in the type of the file: int8 100 + 100 = -56, True + True = True)."""
    from certify_ed.io.formats import read_operator
    for data, want in ((np.array([100, 100, 1], dtype=np.int8), 200), (np.array([True, True, True]), 2.0)):
        R = sp.coo_matrix((data, (np.array([0, 0, 1]), np.array([1, 1, 1]))), shape=(2, 2))
        path = str(tmp_path / f"op_{data.dtype.name}.npz")
        sp.save_npz(path, R)
        A = read_operator(path)
        assert A.toarray()[0, 1] == want, (data.dtype, A.toarray())


def test_single_precision_and_coo_index_dtypes():
    """fro_upper widens a single-precision sparse matrix before summing (before: refused unless canonical); a COO
    copy keeps the caller's int64 coordinates."""
    C = sp.coo_matrix((np.array([1.0, 2.0], dtype=np.float32), (np.array([0, 1]), np.array([1, 0]))), shape=(2, 2))
    assert fro_upper(C) >= np.sqrt(5.0)
    R = sp.coo_matrix((np.array([1.0, 1.0]), (np.array([0, 0], dtype=np.int64), np.array([1, 1], dtype=np.int64))),
                      shape=(2, 2))
    if hasattr(R, "coords"):
        R.coords = (R.row.astype(np.int64), R.col.astype(np.int64))
    assert R.row.dtype == R.col.dtype == np.int64
    S = summed(R)
    assert S.row.dtype == S.col.dtype == np.int64 and S.toarray()[0, 1] == 2.0
    D = sp.dia_matrix((np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1])), shape=(2, 2))
    assert summed(D) is D                         # distinct offsets (scipy refuses repeated ones at construction)


def test_the_sum_is_math_fsum_for_random_groups():
    """Groups of 1 to 12 parts with cancellation, ties at half an ulp, tiny and subnormal parts, in random order and
    in COO and CSR: every summed entry equals math.fsum of its parts (the vectorized TwoSum passes and the check that
    decides when they suffice agree with the reference)."""
    import math
    rng = np.random.default_rng(7)
    rows, vals = [], []
    for k in range(6000):
        m = int(rng.integers(1, 13))
        kind = k % 4
        if kind == 0:
            p = rng.standard_normal(m) * 10.0 ** rng.integers(-5, 6, m)
        elif kind == 1:                                   # cancellation: big parts that sum to a small one
            b = rng.standard_normal() * 2.0 ** 60
            p = np.r_[b, -b, rng.standard_normal(max(m - 2, 0)) * 2.0 ** -int(rng.integers(0, 60))][:max(m, 2)]
        elif kind == 2:                                   # a tie: 1 and two halves of an ulp
            p = np.r_[1.0 + rng.integers(0, 8) * 2.0 ** -52, 2.0 ** -53, 2.0 ** -53 * rng.choice([1, -1]),
                      np.zeros(max(m - 3, 0))]
        else:                                             # tiny and subnormal parts next to normal ones
            p = np.r_[rng.standard_normal(m) * 2.0 ** -1070, rng.standard_normal()]
        p = rng.permutation(p)
        rows.extend([k] * p.size)
        vals.extend(p.tolist())
    rows, vals = np.array(rows), np.array(vals)
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    ref = np.array([math.fsum(vals[s:e].tolist()) for s, e in zip(starts, np.r_[starts[1:], rows.size])])
    n = ref.size
    C = sp.coo_matrix((vals, (rows, np.zeros(rows.size, int))), shape=(n, 1))
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    R = sp.csr_matrix((vals, np.zeros(rows.size, int), indptr), shape=(n, 1))
    for M in (C, R):
        got = summed(M).toarray().ravel()
        assert np.array_equal(got, ref), np.flatnonzero(got != ref)[:5]


def _correct_rounding(parts) -> float:
    """The correctly rounded exact sum of binary64 values, by rational arithmetic (inf beyond the range)."""
    s = sum(Fraction(x) for x in parts)
    try:
        return float(s)
    except OverflowError:
        return float("inf") if s > 0 else float("-inf")


def _midpoint_groups(rng, count):
    """Groups whose exact sum is a midpoint between two floats, or next to one, with two or more nonzero parts left
    after the leading value: x + half an ulp (+- a part far below the ulp of the half), x + two quarters, a power of
    two minus a quarter of its ulp (its lower gap is half the upper one), the same behind one or several cancelling
    pairs of large parts (after the passes large parts can remain, and the remainder sum then rounds), and the largest
    float plus half its ulp (the overflow threshold)."""
    big = np.finfo(float).max
    out = []
    for k in range(count):
        e = int(rng.integers(-60, 61))
        x = (1.0 + float(rng.integers(0, 2 ** 52)) * 2.0 ** -52) * 2.0 ** e * float(rng.choice([1, -1]))
        half = np.spacing(abs(x)) / 2.0 * np.sign(x)
        tiny = half * 2.0 ** -int(rng.integers(54, 200)) * float(rng.choice([1, -1]))
        kind = k % 10
        if kind == 0:
            g = [x, half, tiny]
        elif kind == 1:
            g = [x, half / 2, half / 2]
        elif kind == 2:
            g = [x, half / 2, half / 4, half / 4, tiny]
        elif kind == 3:
            p = 2.0 ** e * float(rng.choice([1, -1]))
            q = -np.spacing(abs(p)) / 4.0 * np.sign(p)
            g = [p, q, q * 2.0 ** -int(rng.integers(54, 200)) * float(rng.choice([1, -1]))]
        elif kind == 4:
            b = 2.0 ** int(rng.integers(100, 400)) * float(rng.choice([1, -1]))
            g = [b, -b, x, half, tiny]
        elif kind == 5:
            g = [x, half, tiny, -tiny / 2, tiny / 4]
        elif kind == 6:
            u = big - np.nextafter(big, 0.0)              # the ulp of the largest float
            g = [big, u / 4, u / 4] + ([] if k % 20 == 6 else [-u * 2.0 ** -60])
        else:                                             # several large cancelling pairs: large parts can remain
            bs = [2.0 ** int(rng.integers(60, 400)) * float(rng.choice([1, -1]))
                  for _ in range(int(rng.integers(2, 5)))]
            g = [x, half, tiny] + bs + [-b for b in bs]
        out.append([g[i] for i in rng.permutation(len(g))])
    return out


def test_the_sum_is_the_correct_rounding_next_to_a_midpoint():
    """Groups at or next to a rounding midpoint, with several parts left after the leading value (where accepting the
    vectorized candidate too early gives the wrong neighbour), in COO and CSR: every summed entry equals the correctly
    rounded exact rational sum. Pins the check that decides when the TwoSum candidate is the correct rounding (its
    slack, its strict comparisons with both half-gaps, the single-remainder case and the gap at the largest float)."""
    rng = np.random.default_rng(11)
    groups = _midpoint_groups(rng, 10000)
    ref = np.array([_correct_rounding(g) for g in groups])
    rows = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    vals = np.array([x for g in groups for x in g])
    n = len(groups)
    C = sp.coo_matrix((vals, (rows, np.zeros(rows.size, int))), shape=(n, 1))
    indptr = np.r_[0, np.cumsum([len(g) for g in groups])]
    R = sp.csr_matrix((vals, np.zeros(rows.size, int), indptr), shape=(n, 1))
    for M in (C, R):
        with np.errstate(over="ignore"):
            got = summed(M).toarray().ravel()
        bad = np.flatnonzero(got != ref)
        assert bad.size == 0, [(groups[i], got[i], ref[i]) for i in bad[:3]]


def test_one_large_group_among_many_small_ones_is_linear():
    """One position with 2000 parts and one with 100000 among 20000 positions of three parts: the sums are the exact
    roundings and the memory stays near the size of the values (before: a buffer of positions x largest group, 3 GB
    for the 2000 parts and a MemoryError for the 100000)."""
    import math
    import tracemalloc
    rng = np.random.default_rng(5)
    sizes = np.r_[np.full(20000, 3), 2000, 100000]
    rows = np.repeat(np.arange(sizes.size), sizes)
    vals = rng.standard_normal(rows.size) * 10.0 ** rng.integers(-8, 9, rows.size)
    perm = rng.permutation(rows.size)
    C = sp.coo_matrix((vals[perm], (rows[perm], np.zeros(rows.size, int))), shape=(sizes.size, 1))
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    ref = np.array([math.fsum(vals[s:s + k].tolist()) for s, k in zip(starts, sizes)])
    tracemalloc.start()
    try:
        got = summed(C).toarray().ravel()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert np.array_equal(got, ref)
    assert peak < 64 * 2 ** 20, peak / 2 ** 20


def test_every_group_size_across_block_boundaries(monkeypatch):
    """Groups of every size from 3 to 70 (both sides of the 64-value limit), groups of 101 to 199 values, and larger
    sizes shared by many groups, with the limits patched small (blocks of 257 values, so that the sizes from 22 up span
    several blocks; vectorized above 64 values when 64 groups share a size, up to 200 values): 64 groups of 65 and 70 of
    100 values take the vectorized path, 63 of 66 (below the shared count) and 64 of 250 (above the cap) math.fsum.
    Every summed entry is the exact rational sum's correct rounding, and the route of each size is checked by counting
    what reaches the math.fsum path. Each group holds a cancelling pair of large parts, so a sum in stored order would
    lose the small parts (pins the cut of a size into blocks, the limits and the math.fsum path)."""
    from certify_ed import precision
    monkeypatch.setattr(precision, "_GROUP_CHUNK", 257)
    monkeypatch.setattr(precision, "_GROUP_VEC_SHARED", 64)
    monkeypatch.setattr(precision, "_GROUP_VEC_CAP", 200)
    calls = []
    inner = precision._fsum_groups

    def spy(v, starts, k):
        calls.append((k, int(starts.size)))
        return inner(v, starts, k)
    monkeypatch.setattr(precision, "_fsum_groups", spy)
    rng = np.random.default_rng(13)
    sizes = [k for k in range(3, 71) for _ in range(12)] + [int(x) for x in rng.integers(101, 200, 8)]
    sizes += [65] * 52 + [100] * 70 + [66] * 51 + [250] * 64   # with the 12 above: 64 of 65, 63 of 66
    groups = []
    for k in sizes:
        b = 2.0 ** int(rng.integers(70, 90))
        g = [b] + (rng.standard_normal(k - 2) * 2.0 ** -int(rng.integers(0, 20))).tolist() + [-b]
        groups.append(g if k % 2 else [g[i] for i in rng.permutation(k)])
    ref = np.array([_correct_rounding(g) for g in groups])
    rows = np.repeat(np.arange(len(groups)), sizes)
    vals = np.array([x for g in groups for x in g])
    R = sp.csr_matrix((vals, np.zeros(rows.size, int), np.r_[0, np.cumsum(sizes)]), shape=(len(groups), 1))
    got = summed(R).toarray().ravel()
    assert np.array_equal(got, ref), np.flatnonzero(got != ref)[:5]
    naive = np.add.reduceat(vals, np.r_[0, np.cumsum(sizes)[:-1]])
    assert np.count_nonzero(naive != ref) > len(groups) // 2         # the stored-order sums are wrong
    whole = {k for k, n in calls if n >= 63}                        # a size sent whole to math.fsum
    assert {66, 250} <= whole and not whole & {65, 100}, calls

@pytest.mark.parametrize("size", [3, 5, 70])
def test_non_finite_groups_of_three_or_more(size):
    """Groups of 3, 5 (vectorized) and 70 values (math.fsum one by one) with values that are not finite, in every
    order: nan for a nan or for both infinities, else the infinity, whatever the order and the path."""
    rng = np.random.default_rng(size)
    cases = [([np.inf, -np.inf], np.nan), ([np.nan], np.nan), ([np.inf], np.inf), ([-np.inf, -np.inf], -np.inf),
             ([np.inf, np.nan], np.nan)]
    groups, want = [], []
    for special, w in cases:
        for _ in range(4):
            g = list(special) + rng.standard_normal(size - len(special)).tolist()
            groups.append([g[i] for i in rng.permutation(size)])
            want.append(w)
    rows = np.repeat(np.arange(len(groups)), size)
    C = sp.coo_matrix((np.array([x for g in groups for x in g]), (rows, np.zeros(rows.size, int))),
                      shape=(len(groups), 1))
    with np.errstate(invalid="ignore", over="ignore"):
        got = summed(C).toarray().ravel()
    want = np.array(want)
    assert np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(got[~np.isnan(want)], want[~np.isnan(want)])


def test_complex_sums_keep_each_part():
    """The real and the imaginary sums are stored as they are: an infinite imaginary sum leaves the real sum 6 (before:
    1j * inf made it nan), and two parts -0.0 + 1j keep the real sign -0.0."""
    C = sp.coo_matrix((np.array([1 + 0j, 2 + 1j, complex(3.0, np.inf), complex(-0.0, 1.0), complex(-0.0, 1.0)]),
                       (np.array([0, 0, 0, 1, 1]), np.zeros(5, int))), shape=(2, 1))
    with np.errstate(invalid="ignore"):
        S = summed(C)
    got = S.data[np.argsort(S.row)]                          # stored values (toarray would add them to +0.0)
    assert got[0].real == 6.0 and got[0].imag == np.inf
    assert got[1].real == 0.0 and np.signbit(got[1].real) and got[1].imag == 2.0


def test_split_rows_sums_and_leaves_the_caller_as_given():
    """ozaki.split_rows of a COO with three parts per entry equals that of the summed matrix, and an unsorted CSR
    keeps its arrays (before: sort_indices on the caller's matrix)."""
    from certify_ed.ozaki import split_rows
    rng = np.random.default_rng(3)
    A = sp.random(30, 30, density=0.2, random_state=4, format="coo")
    parts = [A.data * 0.5 + 2.0 ** 30, A.data * 0.5, -(2.0 ** 30) + 0.0 * A.data]
    C = sp.coo_matrix((np.concatenate(parts), (np.tile(A.row, 3), np.tile(A.col, 3))), shape=A.shape)
    a, b = split_rows(C, 20), split_rows(summed(C), 20)
    assert a.k == b.k and all((x != y).nnz == 0 for x, y in zip(a.real, b.real))
    U = sp.csr_matrix(A)
    order = np.concatenate([rng.permutation(np.arange(s, e)) for s, e in zip(U.indptr[:-1], U.indptr[1:])])
    U = sp.csr_matrix((U.data[order], U.indices[order], U.indptr.copy()), shape=U.shape)
    U.has_sorted_indices = False
    before = _arrays(U)
    split_rows(U, 20)
    assert _unchanged(U, before)


def test_claim_from_quspin_sums_a_scipy_operator():
    """claim_from_quspin given a scipy COO with three parts per entry stores the summed matrix (the exact rounding at
    each position), not one added in stored order, and leaves the caller's arrays as given."""
    pytest.importorskip("quspin")
    from quspin.basis import spin_basis_1d

    from certify_ed.importers import claim_from_quspin
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).tocoo()
    big = 2.0 ** 60
    C = sp.coo_matrix((np.r_[np.full(H.nnz, big), H.data, np.full(H.nnz, -big)],
                       (np.tile(H.row, 3), np.tile(H.col, 3))), shape=H.shape)
    before = tuple(np.array(x, copy=True) for x in (C.data, C.row, C.col))
    w, V = np.linalg.eigh(H.toarray())
    claim, _ = claim_from_quspin(C, spin_basis_1d(6, pauli=0), spec.to_quspin_static(), w[:2], V[:, :2])
    assert (sp.csr_matrix(claim.operator) != summed(C).tocsr()).nnz == 0
    assert all(np.array_equal(x, y) for x, y in zip(before, (C.data, C.row, C.col)))


def test_binary64_sums_sparse_integers_exactly_on_every_route():
    """Integer sparse data reach binary64 through the integer sum: [2^53, 1] at one position (exact sum 2^53 + 1, not a
    binary64 number) is refused on the integer-keeping and on the converting routes alike (before: the converting
    routes summed the converted values, 2^53); [2^53, 1, -2^53] gives 1 on both."""
    from certify_ed.certificate_v2 import matrix_sha256
    from certify_ed.enclosure import OperatorAdapter
    from certify_ed.precision import binary64

    def op(parts):
        k = len(parts)
        return sp.csr_matrix((np.r_[np.array(parts, dtype=np.int64), np.array([2, 3], dtype=np.int64)],
                              np.r_[np.zeros(k, int), [1, 2]], np.array([0, k, k + 1, k + 2])), shape=(3, 3))
    bad = op([2 ** 53, 1])
    for call in (lambda A: summed(A), lambda A: binary64(A), lambda A: OperatorAdapter(A),
                 lambda A: matrix_sha256(A)):
        with pytest.raises(PrecisionError):
            call(bad)
    good = op([2 ** 53, 1, -(2 ** 53)])
    assert binary64(good).toarray()[0, 0] == 1.0 and summed(good).toarray()[0, 0] == 1
    assert OperatorAdapter(good).A.toarray()[0, 0] == 1.0


def test_summed_groups_positions_of_any_shape():
    """No combined row-column key: positions of a COO of shape (2^33, 2^33) stay apart (a key row * ncol + col would
    overflow int64 and merge (0, 5) with (2^31, 5))."""
    n = 2 ** 33
    C = sp.coo_matrix((np.array([1.0, 1.0, 7.0]), (np.array([0, 2 ** 31, 0], dtype=np.int64),
                                                   np.array([5, 5, 5], dtype=np.int64))), shape=(n, n))
    S = summed(C)
    got = sorted(zip(S.row.tolist(), S.col.tolist(), S.data.tolist()))
    assert got == [(0, 5, 8.0), (2 ** 31, 5, 1.0)]


def test_commutes_diagonal_reads_the_summed_matrix():
    """A cross-sector position stored as [x, -x] sums to zero: the matrix commutes with diag(s) (before: the stored
    parts counted as a nonzero entry and the check said no). The caller's arrays stay as given."""
    from certify_ed.eigenspace import commutes_diagonal
    A = sp.csr_matrix((np.array([1.0, 0.75, -0.75, 2.0]), np.array([0, 1, 1, 1]), np.array([0, 3, 4])), shape=(2, 2))
    before = _arrays(A)
    assert commutes_diagonal(A, np.array([0.0, 1.0])) and _unchanged(A, before)
    B = sp.csr_matrix((np.array([1.0, 0.75, 2.0]), np.array([0, 1, 1]), np.array([0, 2, 3])), shape=(2, 2))
    assert not commutes_diagonal(B, np.array([0.0, 1.0]))


def test_the_sparse_residual_bound_reads_the_summed_matrix():
    """The residual of a sparse factorization against S stored as three parts per entry equals the residual against
    the summed S, bit for bit (the factors are those of the summed S)."""
    import scipy.sparse.linalg as spla
    from certify_ed.ldl_residual import sparse_ldl_residual_bound
    rng = np.random.default_rng(5)
    n = 12
    M = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.4)
    S = sp.csr_matrix(M + M.T + 8.0 * np.eye(n))
    S.sum_duplicates()
    C = S.tocoo()
    t = np.full(C.nnz, 2.0 ** 30)
    rep = sp.coo_matrix((np.r_[t, C.data, -t], (np.tile(C.row, 3), np.tile(C.col, 3))), shape=S.shape)
    f = spla.splu(S.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True, "Equil": False})
    L, D = f.L.tocsr(), f.U.diagonal()
    a = sparse_ldl_residual_bound(L, D, rep, f.perm_c, threads=1)
    b = sparse_ldl_residual_bound(L, D, S, f.perm_c, threads=1)
    da, db = dict(a.__dict__), dict(b.__dict__)
    for d in (da, db):
        for k in [k for k in d if "second" in k or "time" in k]:
            d.pop(k)
    assert da == db and rep.nnz == 3 * C.nnz


# ------------------------------------------------------------------------------------- the registry entry points
def _three_parts(M, fmt="coo"):
    """The summed M with every stored entry v kept as three parts (2^30, v, -2^30), as a COO or as a CSR with repeated
    indices: the correctly rounded exact sum at each position is v, while the sum in stored order rounds v to a
    multiple of 2^-22."""
    C = summed(sp.coo_matrix(M)) if sp.issparse(M) else sp.coo_matrix(M)
    C = sp.coo_matrix(C)
    t = np.full(C.nnz, 2.0 ** 30, dtype=C.dtype)
    data = np.concatenate([t, C.data, -t])
    rows, cols = np.tile(C.row, 3), np.tile(C.col, 3)
    if fmt == "coo":
        return sp.coo_matrix((data, (rows, cols)), shape=C.shape)
    o = np.argsort(rows, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=C.shape[0]))]
    return sp.csr_matrix((data[o], cols[o], indptr), shape=C.shape)


_SPLIT = ("A", "S", "O", "B", "V", "Hs", "Vu", "Anh")


def _variants(dtype, fmt="coo"):
    import tempfile
    from test_precision_registry import Data, _data, _ec
    W = _data(dtype)["wide"]
    out = {}
    for kind in ("repeated", "summed"):
        D = Data()
        D.__dict__.update(W.__dict__)
        for k in _SPLIT:
            rep = _three_parts(getattr(W, k), fmt)
            setattr(D, k, rep if kind == "repeated" else summed(rep))
        D.tmp = tempfile.mkdtemp(prefix=f"dup_{dtype}_{fmt}_{kind}_")
        D.ec = lambda D=D: _ec(D)
        out[kind] = D
    return out


def _snapshot(D):
    def arrays(M):
        return (M.data, M.row, M.col) if M.format == "coo" else (M.data, M.indices, M.indptr)
    return {k: tuple(np.array(x, copy=True) for x in arrays(getattr(D, k))) for k in _SPLIT}


def _registry_names():
    from test_precision_registry import R
    return sorted(R)


# Registered functions that read the stored entries by design: rotation_spread is summed now; max_nnz_per_row counts
# the stored entries of a row (a width at least that of the summed matrix, so its gamma factor stays an upper bound).
_STORED_ENTRIES_BY_DESIGN = {"certify_ed.fp.max_nnz_per_row"}
# Recipes that convert or assign into a matrix themselves before the call (tocsr, toarray, csr_matrix or item
# assignment in the test, which add the parts in stored order): the two inputs then differ before the library sees them.
_RECIPE_CONVERTS = {"certify_ed.construct.matrix_difference", "certify_ed.fp.fro_upper",
                    "certify_ed.moments.moments_from_certificate", "certify_ed.compat.symveig.enclose_sectors"}


@pytest.mark.parametrize("name", _registry_names())
@pytest.mark.parametrize("dtype,fmt", [("float32", "coo"), ("complex64", "coo"), ("float32", "csr")])
def test_every_registered_entry_point_sees_the_summed_matrix(dtype, fmt, name):
    """Every public function of the precision registry, given matrices whose entries are stored as three parts that
    cancel, gives bit for bit the result of the summed matrices (the correctly rounded exact sum at each position,
    whatever the stored order) and leaves the caller's arrays as given. A recipe that raises for both inputs (a
    dense-only function given a sparse matrix) is skipped."""
    from test_precision_registry import R, _first_difference, canon
    if name in _STORED_ENTRIES_BY_DESIGN:
        pytest.skip("reads the stored entries by design")
    if name in _RECIPE_CONVERTS:
        pytest.skip("the recipe converts the matrix itself before the call")
    V = _variants(dtype, fmt)
    before = _snapshot(V["repeated"])
    errors = []
    results = []
    for kind in ("repeated", "summed"):
        try:
            results.append(canon(R[name](V[kind])))
            errors.append(None)
        except Exception as e:                    # noqa: BLE001 - compared below
            results.append(None)
            errors.append(f"{type(e).__name__}: {e}")
    if errors[0] is not None and errors[0] == errors[1]:
        pytest.skip(f"raises {errors[0]} for both inputs")
    assert errors == [None, None], f"{name}: {errors}"
    a, b = results
    assert a == b, f"{name}: first difference at {_first_difference(a, b)}"
    after = _snapshot(V["repeated"])
    assert all(all(np.array_equal(x, y) for x, y in zip(before[k], after[k])) for k in _SPLIT), name
