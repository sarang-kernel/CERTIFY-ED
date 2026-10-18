"""Code verification with manufactured Hamiltonians whose spectra are known exactly.

H = Q diag(lam) Q^T with Q a product of permutations and Householder reflections
I - v v^T / 2 for v in {+-1}^4 (exact dyadic orthogonal matrices) and dyadic lam with designed
exact degeneracies and near-degeneracies down to 2^-44. H is formed exactly in floating point (checked
against rational arithmetic), so the true eigenvalues are exactly lam: the oracle problem is removed
by construction (method of manufactured solutions, Roache; Salari & Knupp).
"""
from fractions import Fraction

import numpy as np
import pytest

import certify_ed as ce


def manufactured(n, lam, rng, n_reflections=6):
    Q = np.eye(n)[rng.permutation(n)]
    for _ in range(n_reflections):
        idx = rng.choice(n, 4, replace=False)
        v = rng.choice([-1.0, 1.0], 4)
        R = np.eye(n)
        R[np.ix_(idx, idx)] -= np.outer(v, v) / 2
        Q = Q @ R
        Q = Q[:, rng.permutation(n)]
    H = Q @ np.diag(lam) @ Q.T
    # exactness: compare with rational arithmetic on a few rows
    for i in rng.choice(n, 3, replace=False):
        for j in range(n):
            exact = sum(Fraction(Q[i, k]) * Fraction(lam[k]) * Fraction(Q[j, k]) for k in range(n))
            assert Fraction(H[i, j]) == exact
    return H, Q


SPECTRA = {
    "generic": lambda rng, n: np.sort(rng.integers(-2 ** 20, 2 ** 20, n)) / 2.0 ** 18,
    "exact_degenerate": lambda rng, n: np.sort(np.repeat(rng.integers(-64, 64, n // 4), 4)
                                               [:n] / 8.0),
    "near_degenerate": lambda rng, n: np.sort(np.concatenate(
        [np.array([-3.0, -3.0 + 2 ** -44, -3.0 + 2 ** -30, -1.0, -1.0, -1.0 + 2 ** -40]),
         rng.integers(0, 2 ** 16, n - 6) / 2.0 ** 14])),
}


@pytest.mark.parametrize("kind", list(SPECTRA))
@pytest.mark.parametrize("seed", range(4))
def test_enclosures_and_counts_contain_exact_spectrum(kind, seed):
    rng = np.random.default_rng(seed)
    n = 40
    lam = SPECTRA[kind](rng, n)
    H, Q = manufactured(n, lam, rng)
    w, V = np.linalg.eigh(H)
    for k in (1, 3, 6, 9):
        while k < n and lam[k] == lam[k - 1]:
            k += 1                     # a well-posed 'lowest' claim ends at an exact gap
        c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), None,
                       ce.Options(model_fidelity=False, inertia_method="dense"))
        st = {s["id"]: s for s in c.statements}
        rad = c.arrays["radius"]
        # every enclosure contains the exact eigenvalue of the same rank
        assert np.all(np.abs(w[:k] - lam[:k]) <= rad), (kind, k)
        s3 = st["S3.completeness"]
        if s3["status"] == "CERTIFIED":
            lb = s3["evidence"]["next_eigenvalue_lower_bound"]
            assert lb <= lam[k], (kind, k, lb, lam[k])
        else:
            # allowed only when the claim ends inside a near-degenerate level the enclosures cannot split
            assert s3["status"] == "INCONCLUSIVE" and lam[k] - lam[k - 1] < 1e-8, (kind, k, s3["summary"])


@pytest.mark.parametrize("seed", range(3))
def test_projector_bound_against_exact_projector(seed):
    rng = np.random.default_rng(100 + seed)
    n = 32
    lam = np.sort(np.concatenate([[-2.0] * 3, [-1.0] * 2, rng.integers(0, 2 ** 10, n - 5) / 2.0 ** 8]))
    H, Q = manufactured(n, lam, rng)
    w, V = np.linalg.eigh(H)
    O = np.diag(rng.normal(size=n))
    c = ce.certify(ce.Claim(w[:5], V[:, :5], operator=H, kind="lowest"), None,
                   ce.Options(model_fidelity=False, inertia_method="dense", observables={"O": O}))
    s4 = [s for s in c.statements if s["id"] == "S4.eigenspaces"][0]
    assert s4["status"] == "CERTIFIED"
    order = np.argsort(lam, kind="stable")
    for cl in s4["evidence"]["clusters"]:
        members = cl["members"]
        cols = order[members]
        P = Q[:, cols] @ Q[:, cols].T                 # exact projector onto the exact level
        exact = float(np.trace(P @ O))
        rec = cl["observables"]["O"]
        assert abs(rec["value"] - exact) <= rec["bound"], (rec, exact)
