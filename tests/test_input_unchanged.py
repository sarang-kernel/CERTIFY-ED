"""certify() leaves the caller's sparse matrix as given, so that certifying the same claim twice gives the same
certificate. A CSR matrix with duplicate entries used to be summed in place: by the adapter's absolute value without
a model, and by the S1 comparison and the sector audit with one."""
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
