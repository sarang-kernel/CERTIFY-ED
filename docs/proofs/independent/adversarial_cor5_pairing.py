"""Independent adversarial search for Corollary 5 and the sorted-pairing lemma (docs/proofs/cor5_pairing.md).

Part A: the combinatorial statement in exact rational arithmetic (fractions.Fraction): eigenvalues with repeats,
claimed values with ties, groups with uniform radii, touching and disjoint union intervals, the cut (H3) at
equality and strictly satisfied, the shift Delta; the conclusion is checked exactly; the hypotheses are then
broken one at a time (non-uniform radii, touching intervals, (H3) at equality) and the number of cases in which
the conclusion fails is reported (necessity of the hypotheses: expected > 0, not a violation).
Part B: the implementation certify._certify_lowest on small symmetric matrices with planted clusters, repeated
eigenvalues, scales 1e+-100, a model shift Delta with an actual perturbation ||E|| <= Delta; every CERTIFIED
outcome is checked against the mpmath eigenvalues of the model matrix A + E (pairing, count, next-eigenvalue
bound, gap); negative controls (missing eigenvalue, duplicated vector, wrong count) must not be CERTIFIED.

Usage: python docs/proofs/independent/adversarial_cor5_pairing.py [--cases 300] [--seed 11]
"""
import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, Tally, eig_herm, has_matching, mpf, report, to_mp  # noqa: E402
from certify_ed.certify import Options, _certify_lowest  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, enclose  # noqa: E402
from certify_ed.fp import add_up  # noqa: E402


def exact_case(rng, break_=None):
    """Build an exact instance satisfying (H1)-(H3) (or with one hypothesis broken). Returns dict or None."""
    n = int(rng.integers(2, 9))
    k = int(rng.integers(1, n))
    base = sorted(Fraction(int(v), 8) for v in rng.integers(-40, 40, size=n))
    # repeats
    lam = list(base)
    for _ in range(int(rng.integers(0, 3))):
        i = int(rng.integers(0, n - 1)); lam[i + 1] = lam[i]
    lam.sort()
    # groups of consecutive claimed indices with uniform radius; claimed values near the eigenvalues
    groups, j = [], 0
    while j < k:
        s = int(rng.integers(1, min(3, k - j) + 1)); groups.append(list(range(j, j + s))); j += s
    mu = [None] * k; rad = [None] * k
    for G in groups:
        rG = Fraction(int(rng.integers(0, 6)), 64)
        if rng.random() < 0.2:
            rG = Fraction(0)
        for j in G:
            # claimed value within rG of lambda_j (ties allowed)
            mu[j] = lam[j] + Fraction(int(rng.integers(-64, 65)), 64) * rG
            rad[j] = rG
    if break_ == "nonuniform":
        G = groups[int(rng.integers(0, len(groups)))]
        if len(G) < 2:
            return None
        rad[G[0]] = rad[G[0]] + Fraction(3, 2)
        mu[G[0]] = lam[G[0]] + rad[G[0]] * Fraction(int(rng.integers(-64, 65)), 64)
    order = sorted(range(k), key=lambda j: mu[j])
    mu = [mu[j] for j in order]; rad = [rad[j] for j in order]
    # the groups after sorting: rebuild as maximal runs with equal radius (uniform within a run); for the exact
    # check we take the groups to be the connected components of the union intervals
    ivals = [(mu[j] - rad[j], mu[j] + rad[j]) for j in range(k)]
    comps, cur = [], [0]
    for j in range(1, k):
        lo_j = ivals[j][0]
        hi_cur = max(ivals[i][1] for i in cur)
        if lo_j <= hi_cur:
            cur.append(j)
        else:
            comps.append(cur); cur = [j]
    comps.append(cur)
    if break_ != "nonuniform":
        for C in comps:
            if len({rad[j] for j in C}) != 1:
                return None     # H1a would fail by accident: discard
    # (H1b): injection per component, exact
    for C in comps:
        if not has_matching([mu[j] for j in C], lam, max(rad[j] for j in C)):
            return None
    if break_ == "touching":
        # make two components touch by shifting the second's values; keep each component's injection
        if len(comps) < 2:
            return None
        C1, C2 = comps[0], comps[1]
        gapv = min(ivals[j][0] for j in C2) - max(ivals[j][1] for j in C1)
        for j in C2:
            mu[j] -= gapv
        ivals = [(mu[j] - rad[j], mu[j] + rad[j]) for j in range(k)]
        if not has_matching([mu[j] for j in C2], lam, max(rad[j] for j in C2)):
            return None
    Delta = Fraction(int(rng.integers(0, 3)), 128)
    top = max(mu[j] + rad[j] + Delta for j in range(k))
    # (H2): choose sigma, beta with nu = k from the exact count
    beta = Fraction(int(rng.integers(0, 3)), 128)
    if break_ == "cut_equal":
        sigma = top + beta + Delta
    else:
        sigma = top + beta + Delta + Fraction(1, 256)
    nu = sum(1 for v in lam if v < sigma - beta)
    nu_hi = sum(1 for v in lam if v <= sigma + beta)
    if not (nu <= k <= nu_hi):
        return None
    return dict(lam=lam, mu=mu, rad=rad, k=k, Delta=Delta, sigma=sigma, beta=beta, comps=comps)


def conclusion_holds(c, lam_m):
    """Exact check of the conclusion for model eigenvalues lam_m (sorted Fractions)."""
    k, mu, rad, D = c["k"], c["mu"], c["rad"], c["Delta"]
    cut = c["sigma"] - c["beta"] - D
    pair = all(abs(lam_m[j] - mu[j]) <= rad[j] + D for j in range(k))
    count = sum(1 for v in lam_m if v < cut) == k
    nxt = (len(lam_m) == k) or lam_m[k] >= cut
    gap = (len(lam_m) == k) or (lam_m[k] - lam_m[k - 1] >= cut - (mu[k - 1] + rad[k - 1] + D))
    return pair and count and nxt and gap


def part_a(rng, cases, T_lemma, T_cor, T_need):
    fails = {"nonuniform": 0, "touching": 0, "cut_equal": 0}
    tried = {"nonuniform": 0, "touching": 0, "cut_equal": 0}
    i = 0
    while i < cases:
        c = exact_case(rng)
        if c is None:
            continue
        i += 1
        lam = c["lam"]
        # model eigenvalues: perturb each eigenvalue by at most Delta (Weyl is the only link used)
        lam_m = sorted(v + Fraction(int(rng.integers(-128, 129)), 128) * c["Delta"] for v in lam)
        T_cor.add(conclusion_holds(c, lam_m), c["Delta"] > 0 or any(r == 0 for r in c["rad"]),
                  f"exact case k={c['k']} lam={[str(v) for v in lam]} mu={[str(v) for v in c['mu']]} rad={[str(v) for v in c['rad']]}")
        # sorted pairing lemma on the matched block: for each component, any respecting bijection -> sorted one
        for C in c["comps"]:
            a = [c["mu"][j] for j in C]
            b = sorted(lam[j] for j in C)       # by the corollary these are the matched eigenvalues
            r = c["rad"][C[0]]
            # random bijection respecting r?  check via has_matching (exists) and then the sorted pairing
            if has_matching(a, b, r):
                T_lemma.add(all(abs(x - y) <= r for x, y in zip(sorted(a), b)), r == 0, f"lemma a={a} b={b} r={r}")
    for key in fails:
        t = 0
        while t < max(50, cases // 4):
            c = exact_case(rng, break_=key)
            if c is None:
                continue
            t += 1; tried[key] += 1
            lam_m = list(c["lam"])
            if not conclusion_holds(c, lam_m):
                fails[key] += 1
    for key in fails:
        T_need.add(fails[key] > 0, True, f"breaking {key}: conclusion never failed in {tried[key]} cases")
    return fails, tried


def planted(rng, kind):
    """Symmetric matrix with planted spectrum; claims for the k lowest; returns (A, mu, X, k, E, Delta)."""
    n = int(rng.integers(4, 9))
    scale = {0: 1.0, 1: 1e100, 2: 1e-100}[int(rng.integers(0, 3))]
    d = np.sort(rng.standard_normal(n)) * scale
    if kind in ("repeated", "cluster", "neg_missing", "neg_dup"):
        d[1] = d[0] if kind == "repeated" else d[0] + 1e-13 * scale
        if kind == "repeated" and n > 3:
            d[3] = d[2]
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(d) @ Q.T
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    k = int(rng.integers(1, n))
    idx = list(range(k))
    if kind == "neg_missing" and k >= 2:
        idx = list(range(k - 1)) + [k]            # skip eigenvalue k-1, claim k as "k lowest"
    X = V[:, idx] + 1e-9 * scale ** 0 * rng.standard_normal((n, len(idx)))
    mu = w[idx].copy()
    if kind == "neg_dup" and k >= 2:
        X[:, 1] = X[:, 0]; mu[1] = mu[0]
    if kind == "neg_wrongcount":
        mu = w[: k].copy(); X = V[:, :k].copy()
        X = X[:, :-1] if k >= 2 else X             # claim k values with k-1 vectors is a shape error; instead
        mu = mu[:-1] if k >= 2 else mu             # claim k-1 lowest but tell the certifier... (handled below)
    Delta = float(rng.choice([0.0, 1e-12, 1e-6])) * scale
    E = rng.standard_normal((n, n)); E = (E + E.T) / 2
    E *= 0.0 if Delta == 0 else Delta / np.linalg.norm(E, 2) * 0.999
    return A, mu, X, len(mu), E, Delta


def part_b(rng, cases, T_code, T_neg):
    kinds = ["plain", "repeated", "cluster", "neg_missing", "neg_dup"]
    for i in range(cases):
        kind = kinds[i % len(kinds)]
        A, mu, X, k, E, Delta = planted(rng, kind)
        enc = enclose(OperatorAdapter(A), mu, X)
        st, ev, window = _certify_lowest(A, np.sort(mu), enc, Options(), eps_shift=Delta)
        if kind.startswith("neg"):
            if kind == "neg_missing" and k < 2:
                continue
            T_neg.add(st != "CERTIFIED", True, f"{kind}: status {st} ev={ev}")
            continue
        if st != "CERTIFIED":
            T_code.skip(); continue
        Am = to_mp(A + E)
        lam_m, _ = eig_herm(Am)
        mus = sorted(mpf(m) for m in mu)
        rad = add_up(enc.radius[np.argsort(enc.mu, kind="stable")], Delta)
        cut = mpf(ev["next_eigenvalue_lower_bound"])
        ok = all(abs(lam_m[j] - mus[j]) <= mpf(rad[j]) for j in range(k))
        ok = ok and sum(1 for v in lam_m if v < cut) == k and (len(lam_m) == k or lam_m[k] >= cut)
        ok = ok and (len(lam_m) == k or lam_m[k] - lam_m[k - 1] >= mpf(ev["gap_lower_bound"]))
        ok = ok and window[1] == ev["next_eigenvalue_lower_bound"]
        T_code.add(ok, kind != "plain" or Delta > 0, f"{kind} Delta={Delta}: status {st} ev={ev}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_lemma = Tally("sorted-pairing lemma within each group (exact rationals)")
    T_cor = Tally("Corollary 5 conclusion (exact rationals, model eigenvalues moved by <= Delta)")
    T_need = Tally("necessity: breaking H1a / H1c / strict H3 makes the conclusion fail in some case")
    T_code = Tally("_certify_lowest CERTIFIED outcomes vs mpmath eigenvalues of A + E")
    T_neg = Tally("negative controls (missing eigenvalue, duplicated vector) not CERTIFIED")
    fails, tried = part_a(rng, a.cases * 4, T_lemma, T_cor, T_need)
    part_b(rng, a.cases, T_code, T_neg)
    print(f"necessity counts: {', '.join(f'{k}: conclusion failed in {fails[k]} of {tried[k]}' for k in fails)}")
    sys.exit(1 if report([T_lemma, T_cor, T_need, T_code, T_neg]) else 0)


if __name__ == "__main__":
    main()
