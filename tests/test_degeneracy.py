"""S8 degeneracy counts by bracketing inertia (Proposition 14, UNVERIFIED): the implementation against the statement."""
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.degeneracy import degeneracy_counts
from certify_ed.enclosure import OperatorAdapter, enclose


def _s8(cert):
    return [s for s in cert.statements if s["id"] == "S8.degeneracy"][0]


def test_diagonal_matrix_known_degeneracies():
    d = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.5, 3.5, 3.5, 3.5])
    A = sp.diags(d).tocsr()
    X = np.eye(d.size)
    enc = enclose(OperatorAdapter(A), d, X)
    clusters = [[0, 1, 2], [3, 4], [5], [6, 7, 8, 9]]
    recs = degeneracy_counts(A, enc, clusters, method="dense")
    assert [r["status"] for r in recs] == ["CERTIFIED"] * 4
    assert [r["degeneracy"] for r in recs] == [3, 2, 1, 4]
    for r in recs:
        lo, hi = r["bracket_inner"]
        assert lo < r["level_enclosure"][0] and r["level_enclosure"][1] < hi
        assert r["count_difference"] == r["degeneracy"]
        assert len(r["counts"]) == 2


def test_partial_cluster_is_inconclusive_with_the_true_count_as_lower_bound():
    d = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    A = sp.diags(d).tocsr()
    X = np.eye(5)
    enc = enclose(OperatorAdapter(A), d[:3], X[:, :3])      # claims 0 and two of the three members at 1
    recs = degeneracy_counts(A, enc, [[1, 2]], method="dense")
    r = recs[0]
    assert r["status"] == "INCONCLUSIVE" and r["degeneracy_at_least"] == 2 and r["count_difference"] == 3
    assert "further eigenvalue" in r["reason"]


def test_heisenberg_triplet_and_singlet_through_certify():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, ce.Options(eigenspaces="all"))
    s8 = _s8(cert)
    assert s8["status"] == "CERTIFIED", s8["summary"]
    recs = s8["evidence"]["clusters"]
    degs = sorted(r["degeneracy"] for r in recs)
    assert degs == [1, 3]
    for r in recs:
        assert r["width"] < 1e-12
        assert "UNVERIFIED" in r["statement"]
    assert any("UNVERIFIED" in a for a in s8["assumptions"])
    # default (clusters only): just the triplet
    cert2 = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec)
    assert [r["degeneracy"] for r in _s8(cert2)["evidence"]["clusters"]] == [3]


def test_truncated_multiplet_is_not_certified_as_a_smaller_level():
    """Claiming two of the three triplet members (grouped by cluster_tol) must not yield 'degeneracy 2'."""
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    idx = [0, 1, 2]
    cert = ce.certify(ce.Claim(w[idx], V[:, idx], operator=H, kind="subset"), spec,
                      ce.Options(cluster_tol=1e-9, eigenspaces="all"))
    s8 = _s8(cert)
    assert s8["status"] == "INCONCLUSIVE"
    pair = [r for r in s8["evidence"]["clusters"] if r["g_claimed"] == 2][0]
    assert pair["status"] == "INCONCLUSIVE" and pair["degeneracy_at_least"] == 2 and pair["count_difference"] == 3


def test_s8_not_checked_reasons_and_narrowing():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim(w[:4], None, operator=H, kind="lowest"), spec)
    assert _s8(c)["status"] == "NOT_CHECKED" and _s8(c)["evidence"]["reason_code"] == "NEEDS_EIGENVECTORS"
    c = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, ce.Options(degeneracy=False))
    assert not [s for s in c.statements if s["id"] == "S8.degeneracy"]       # optional statements appear when selected
    assert any(x["id"] == "S8.degeneracy" for x in c.body["scope"]["narrowed"])   # ... and the scope block says so
    c = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec)
    assert _s8(c)["status"] == "NOT_CHECKED" and _s8(c)["evidence"]["reason_code"] == "NO_CLUSTERS"


def test_free_fermion_degeneracies():
    """Spinless fermions on a 4-site ring at half filling have exact degeneracies from the k = +-pi/2 pair."""
    spec = ce.spinless_fermion_chain(4, t=1.0, boundary="periodic")
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    # claim the full spectrum and count every level
    cert = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec, ce.Options(eigenspaces="all"))
    s8 = _s8(cert)
    assert s8["status"] == "CERTIFIED", s8["summary"]
    degs = [r["degeneracy"] for r in s8["evidence"]["clusters"]]
    assert sum(degs) == 16
    levels, counts = np.unique(np.round(w, 9), return_counts=True)
    assert sorted(degs) == sorted(counts.tolist())


@pytest.mark.parametrize("method", ["dense", "sparse"])
def test_both_factorization_paths(method):
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, ce.Options(inertia_method=method))
    assert _s8(cert)["status"] == "CERTIFIED"
    assert all(c["method"].startswith(method) or method in c["method"] for r in _s8(cert)["evidence"]["clusters"]
               for c in r["counts"])
