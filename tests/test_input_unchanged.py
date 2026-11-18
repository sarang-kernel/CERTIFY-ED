"""certify() leaves the caller's sparse matrix as given, so that certifying the same claim twice gives the same
certificate. A CSR matrix with duplicate entries used to be summed in place: by the adapter's absolute value without
a model, and by the S1 comparison and the sector audit with one."""
import collections
import copy
import json

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce


def _with_duplicates(H):
    """H with every stored entry split into two exact halves (unsorted columns within a row are not needed)."""
    C = sp.csr_matrix(H).tocoo()
    n = H.shape[0]
    rows, cols, vals = np.r_[C.row, C.row], np.r_[C.col, C.col], np.r_[C.data, C.data] * 0.5
    o = np.lexsort((cols, rows))
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    return sp.csr_matrix((vals[o], cols[o], indptr), shape=H.shape)


def _random_claim():
    n = 200
    A = sp.random(n, n, density=0.05, format="csr", random_state=np.random.RandomState(5))
    H = ((A + A.T) * 0.5 + sp.diags(np.arange(n) * 0.01)).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    return ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=_with_duplicates(H), kind="lowest"), None


def _model_claim(isometry, split_isometry=False):
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = np.linalg.eigh(H.toarray())
    iso = sp.identity(spec.dim, format="csr") if isometry else None
    if split_isometry:
        iso = _with_duplicates(iso)
    claim = ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=_with_duplicates(H), kind="lowest",
                     sector_isometry=iso)
    return claim, spec


def _without_timings(obj):
    if isinstance(obj, dict):
        return {k: _without_timings(v) for k, v in obj.items() if not any(w in k for w in ("second", "time", "wall"))}
    if isinstance(obj, (list, tuple)):
        return [_without_timings(v) for v in obj]
    return obj


def _same(a, b):
    """The two certificates agree in every field of the body except the environment and the times, and in every array bit
    for bit."""
    body = lambda c: json.dumps(_without_timings({k: v for k, v in c.body.items() if k not in ("environment", "timing")}),
                                sort_keys=True, default=repr)
    assert body(a) == body(b)
    assert sorted(a.arrays) == sorted(b.arrays)
    for key in a.arrays:
        x, y = np.asarray(a.arrays[key]), np.asarray(b.arrays[key])
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), key


def _arrays(M):
    return M.indptr.copy(), M.indices.copy(), M.data.copy()


def _unchanged(M, before):
    return all(np.array_equal(x, y) for x, y in zip(before, (M.indptr, M.indices, M.data))) and \
        M.nnz == len(M.data) == len(M.indices) and M.tocoo().nnz == M.nnz


@pytest.mark.parametrize("case,device", [("no model", "cpu"), ("model", "cpu"), ("sector", "cpu"),
                                         ("split isometry", "cpu"), ("no model", "gpu"), ("model", "gpu")])
def test_certify_leaves_a_non_canonical_matrix_as_given_and_repeats_exactly(case, device):
    claim, spec = _random_claim() if case == "no model" else _model_claim(case != "model", case == "split isometry")
    D, iso = claim.operator, claim.sector_isometry
    assert not D.has_canonical_format
    before = _arrays(D)
    before_iso = _arrays(iso) if iso is not None else None
    opts = ce.Options(seed=1, device=device)
    a = ce.certify(claim, spec, opts)
    assert _unchanged(D, before)
    if iso is not None:
        assert _unchanged(iso, before_iso)
    b = ce.certify(claim, spec, opts)
    for key in ("statements", "kato_temple", "enclosures", "gaps"):
        assert json.dumps(_without_timings(a.body.get(key)), sort_keys=True, default=repr) == \
            json.dumps(_without_timings(b.body.get(key)), sort_keys=True, default=repr), key
    assert sorted(a.arrays) == sorted(b.arrays)
    for key in a.arrays:
        x, y = np.asarray(a.arrays[key]), np.asarray(b.arrays[key])
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), key
    if spec is not None:
        assert a.statement("S1.model_fidelity")["status"] == "CERTIFIED", a.summary()


SymmetryEntry = collections.namedtuple("SymmetryEntry", "S lattice")


@pytest.mark.parametrize("shape", ["tuple", "namedtuple", "object array", "list", "deque", "UserList"])
def test_certify_leaves_matrices_of_the_options_as_given(shape):
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = np.linalg.eigh(H.toarray())
    claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
    n_sites = int(round(np.log2(spec.dim)))
    z = np.array([sum(0.5 - ((b >> i) & 1) for i in range(n_sites)) for b in range(spec.dim)])   # total Sz, up to sign
    Sz = _with_duplicates(sp.diags(z).tocsr())
    Obs = _with_duplicates(sp.diags(z * z).tocsr())
    before = {"Sz": _arrays(Sz), "Obs": _arrays(Obs)}
    opts = ce.Options(seed=1, eigenspaces="all", observables={"SzSz": Obs},
                      approximate_symmetries={"Sz": {"tuple": lambda: (Sz, 1.0),
                                                     "namedtuple": lambda: SymmetryEntry(Sz, 1.0),
                                                     "object array": lambda: np.array([Sz, 1.0], dtype=object),
                                                     "list": lambda: [Sz, 1.0],
                                                     "deque": lambda: collections.deque([Sz, 1.0]),
                                                     "UserList": lambda: collections.UserList([Sz, 1.0])}[shape]()})
    a = ce.certify(claim, spec, opts)
    assert _unchanged(Sz, before["Sz"]) and _unchanged(Obs, before["Obs"])
    b = ce.certify(claim, spec, opts)
    for key in ("statements", "kato_temple", "enclosures", "gaps"):
        assert json.dumps(_without_timings(a.body.get(key)), sort_keys=True, default=repr) == \
            json.dumps(_without_timings(b.body.get(key)), sort_keys=True, default=repr), key


def test_entries_absorbed_by_the_sum_are_not_scanned_for_underflow():
    """Each entry stored as v and 2^-950: summed, the tiny parts vanish (v + 2^-950 == v), so the stored operator is H
    and the no-underflow hypothesis holds. The scan must read the summed entries, not stale ones past nnz."""
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec).tocsr()
    C = H.tocoo()
    n = H.shape[0]
    rows, cols = np.r_[C.row, C.row], np.r_[C.col, C.col]
    vals = np.r_[C.data, np.full(C.nnz, 2.0 ** -950)]
    o = np.lexsort((cols, rows))
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    D = sp.csr_matrix((vals[o], cols[o], indptr), shape=H.shape)
    assert not D.has_canonical_format               # the flag is read (and cached) before certify()
    w, V = np.linalg.eigh(H.toarray())
    cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=D, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED", cert.summary()


def _total_sz(spec):
    n_sites = int(round(np.log2(spec.dim)))
    return np.array([sum(0.5 - ((b >> i) & 1) for i in range(n_sites)) for b in range(spec.dim)])


@pytest.mark.parametrize("case", ["unsorted operator as observable", "split operator as observable",
                                  "split operator as observable, model", "split matrix as observable and symmetry"])
def test_a_matrix_given_in_two_places_gets_one_copy(case):
    """A matrix given in two places is one object; a summation done through one place was seen through the other.
    certify() must give the certificate of a run on the matrices themselves (here on a deep copy, which keeps one
    object one object), not of a run with a separate copy for each place."""
    from certify_ed.certify import _certify
    if case == "split matrix as observable and symmetry":
        spec = ce.heisenberg_chain(8)
        H = ce.build_bitwise(spec)
        w, V = np.linalg.eigh(H.toarray())
        claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
        Sz = _with_duplicates(sp.diags(_total_sz(spec)).tocsr())
        opts = ce.Options(seed=1, eigenspaces="all", observables={"Sz": Sz}, approximate_symmetries={"Sz": (Sz, 1.0)})
        given = [Sz]
    else:
        claim, spec = _model_claim(False) if case.endswith("model") else _random_claim()
        if case.startswith("unsorted"):
            D = claim.operator.copy()
            D.sum_duplicates()
            for r in range(D.shape[0]):                          # reverse the columns within each row
                sl = slice(D.indptr[r], D.indptr[r + 1])
                D.indices[sl], D.data[sl] = D.indices[sl][::-1].copy(), D.data[sl][::-1].copy()
            D = sp.csr_matrix((D.data, D.indices, D.indptr), shape=D.shape)
            claim = ce.Claim(eigenvalues=claim.eigenvalues, eigenvectors=claim.eigenvectors, operator=D, kind="lowest")
        opts = ce.Options(seed=1, eigenspaces="all", observables={"H": claim.operator})
        given = [claim.operator]
    assert not any(M.has_canonical_format for M in given)
    before = [_arrays(M) for M in given]
    ref_claim, ref_opts = copy.deepcopy((claim, opts))
    ref = _certify(ref_claim, spec, ref_opts)               # in place, on the deep copy
    a = ce.certify(claim, spec, opts)
    assert all(_unchanged(M, x) for M, x in zip(given, before))
    _same(ref, a)
    _same(a, ce.certify(claim, spec, opts))


@pytest.mark.parametrize("cls", [sp.csr_matrix, sp.csr_array])
def test_a_private_copy_keeps_the_index_dtypes_and_one_copy_per_object(cls):
    from certify_ed.certify import _private_copies
    D = cls(_with_duplicates(_random_claim()[0].operator))
    D.indices, D.indptr = D.indices.astype(np.int64), D.indptr.astype(np.int64)
    own = _private_copies()
    C = own(D)
    assert C is not D and type(C) is type(D) and own(D) is C
    assert C.indices.dtype == C.indptr.dtype == np.int64
    assert all(np.array_equal(x, y) for x, y in zip(_arrays(C), _arrays(D)))
    H = _random_claim()[0].operator.copy()
    H.sum_duplicates()
    assert own(H) is H and own(None) is None
