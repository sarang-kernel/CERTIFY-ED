"""Computational verification of Proposition 14 (docs/proofs/P14_degeneracy_bracketing.md).

Part A (exact model of the statement): random rational spectra, two shifts with backward errors, and
counts nu = #{perturbed eigenvalues < sigma} for an adversarially chosen perturbation of size <= beta (each
eigenvalue moved by +-beta or left in place, including moves that land exactly on sigma). The two
inequalities #I_in <= nu_hi - nu_lo <= #I_out and the three decision rules are checked against the exact
counts. Adversarial cases put eigenvalues exactly at sigma +- beta, at the ends of J, and make beta equal to
the distance from the shift to J (the hypothesis boundary J inside I_in).

Part B (the implementation): diagonal matrices (exact spectrum = diagonal) through
certify_ed.degeneracy.degeneracy_counts: every CERTIFIED statement "exactly g eigenvalues in I_in" is
checked against the exact diagonal; planted near-degenerate pairs that are not degenerate; truncated
multiplets (INCONCLUSIVE with the true count as lower bound).

Usage: python docs/proofs/verify_P14.py [--random N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from certify_ed.degeneracy import degeneracy_counts  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, enclose  # noqa: E402


# ------------------------------------------------------------------------------------ part A: the statement
def _count_below(vals, s):
    return sum(1 for v in vals if v < s)


def _in_open(vals, a, b):
    return sum(1 for v in vals if a < v < b)


def _in_closed(vals, a, b):
    return sum(1 for v in vals if a <= v <= b)


def _perturb(rng: random.Random, lam, beta, sigma):
    """Eigenvalues of a Hermitian perturbation of norm <= beta: each moved by at most beta (Weyl). The
    adversary prefers moves that land exactly on sigma, or at +-beta."""
    out = []
    for v in lam:
        r = rng.random()
        if r < 0.2 and abs(v - sigma) <= beta:
            out.append(sigma)                       # lands exactly on the shift: not counted (strict <)
        elif r < 0.5:
            out.append(v + beta)
        elif r < 0.8:
            out.append(v - beta)
        else:
            out.append(v + beta * Fraction(rng.randrange(-1000, 1001), 1000))
    return out


def model_case(rng: random.Random, adversarial: bool):
    """One case of the statement; returns (inequalities hold, decision rules hold)."""
    n = rng.randrange(2, 12)
    lam = sorted(Fraction(rng.randrange(-1000, 1000), rng.randrange(1, 50)) for _ in range(n))
    # a cluster: g eigenvalues given by Theorem 2 (distinct, inside J); J = their hull widened a little
    g = rng.randrange(1, min(4, n) + 1)
    start = rng.randrange(0, n - g + 1)
    members = lam[start:start + g]
    w = Fraction(rng.randrange(0, 100), 1000)
    JL, JU = members[0] - w, members[-1] + w
    beta_lo = Fraction(rng.randrange(0, 50), 1000)
    beta_hi = Fraction(rng.randrange(0, 50), 1000)
    if adversarial:
        # shifts at exactly beta from J (the boundary J inside I_in fails by equality on one side) or with an
        # eigenvalue exactly at sigma +- beta
        k = rng.randrange(4)
        if k == 0:
            s_lo, s_hi = JL - beta_lo, JU + beta_hi + Fraction(1, 10 ** 6)
        elif k == 1:
            s_lo, s_hi = JL - beta_lo - Fraction(1, 10 ** 6), JU + beta_hi
        elif k == 2 and start > 0:
            s_lo, s_hi = lam[start - 1] + beta_lo, JU + beta_hi + Fraction(1, 10 ** 3)    # neighbour at sigma - beta
        else:
            top = lam[start + g] if start + g < n else JU + 1
            s_lo, s_hi = JL - beta_lo - Fraction(1, 10 ** 3), top - beta_hi            # neighbour at sigma + beta
    else:
        s_lo = JL - beta_lo - Fraction(rng.randrange(1, 500), 1000)
        s_hi = JU + beta_hi + Fraction(rng.randrange(1, 500), 1000)
    if not (s_lo < JL and JU < s_hi):
        s_lo, s_hi = JL - beta_lo - Fraction(1, 10 ** 6), JU + beta_hi + Fraction(1, 10 ** 6)
    nu_lo = _count_below(_perturb(rng, lam, beta_lo, s_lo), s_lo)
    nu_hi = _count_below(_perturb(rng, lam, beta_hi, s_hi), s_hi)
    # Lemma 4 for each count
    lemma4 = (_count_below(lam, s_lo - beta_lo) <= nu_lo <= sum(1 for v in lam if v <= s_lo + beta_lo)
              and _count_below(lam, s_hi - beta_hi) <= nu_hi <= sum(1 for v in lam if v <= s_hi + beta_hi))
    a_in, b_in = s_lo + beta_lo, s_hi - beta_hi
    a_out, b_out = s_lo - beta_lo, s_hi + beta_hi
    diff = nu_hi - nu_lo
    n_in, n_out = _in_open(lam, a_in, b_in), _in_closed(lam, a_out, b_out)
    nonempty = a_in < b_in
    ineq = (not nonempty or n_in <= diff) and diff <= n_out
    # decision rules when J is inside I_in (the members are g distinct eigenvalues in J by construction)
    rules = True
    if a_in < JL and JU < b_in:
        if diff == g:
            rules &= n_in == g
        elif diff > g:
            rules &= n_out > g                      # at least one further eigenvalue in I_out
        else:
            rules = False                           # diff < g contradicts Theorem 2 + Lemma 4: cannot happen
    return lemma4 and ineq, rules


def check_model(rng: random.Random, n: int, adversarial: bool):
    cases = viol = 0
    for _ in range(n):
        ok1, ok2 = model_case(rng, adversarial)
        cases += 1
        viol += not (ok1 and ok2)
    return cases, viol


# ------------------------------------------------------------------------------------ part B: the code
def _run_diag(d, clusters, claim_idx=None, eps=0.0):
    d = np.asarray(d, dtype=float)
    A = sp.diags(d).tocsr()
    X = np.eye(d.size)
    idx = list(range(d.size)) if claim_idx is None else list(claim_idx)
    enc = enclose(OperatorAdapter(A), d[idx], X[:, idx])
    return degeneracy_counts(A, enc, clusters, method="dense", eps_shift=eps)


def code_case(rng: random.Random):
    """A diagonal matrix with planted levels; every CERTIFIED record must name the exact count in I_in."""
    levels = sorted(set(round(rng.uniform(-5, 5), rng.randrange(1, 4)) for _ in range(rng.randrange(2, 6))))
    d = []
    for lv in levels:
        d += [lv] * rng.randrange(1, 4)
    # a planted near-degenerate pair that is NOT degenerate
    if rng.random() < 0.5:
        base = rng.uniform(-5, 5)
        d += [base, base + 10.0 ** rng.randrange(-12, -6)]
    d = np.array(sorted(d))
    # clusters by a tolerance (as cluster_groups would do)
    tol = 10.0 ** rng.randrange(-9, -3)
    clusters, cur = [], [0]
    for i in range(1, d.size):
        if d[i] - d[i - 1] <= tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    recs = _run_diag(d, clusters)
    ok = True
    for C, r in zip(clusters, recs):
        if r["status"] == "CERTIFIED":
            lo, hi = r["bracket_inner"]
            exact = int(np.sum((d > lo) & (d < hi)))
            ok &= exact == r["degeneracy"] == len(C)
            ok &= lo < r["level_enclosure"][0] and r["level_enclosure"][1] < hi
        elif r["status"] == "INCONCLUSIVE" and "degeneracy_at_least" in r:
            lo, hi = r["bracket_outer"]
            ok &= int(np.sum((d >= lo) & (d <= hi))) >= r["count_difference"] > len(C)
        elif r["status"] == "FAILED":
            ok = False                               # cannot happen for a true claim
    return ok, [r["status"] for r in recs]


def check_code(rng: random.Random, n: int):
    cases = viol = 0
    statuses = {}
    for _ in range(n):
        ok, st = code_case(rng)
        cases += 1
        viol += not ok
        for s in st:
            statuses[s] = statuses.get(s, 0) + 1
    return cases, viol, statuses


def code_adversarial():
    """Fixed cases: a truncated multiplet; a non-degenerate pair at 1e-9 resolved as two levels when the
    tolerance allows and reported INCONCLUSIVE (count 2 in I_in with g = 1 each is impossible, so the pair must
    either be two CERTIFIED singlets or one cluster of two); a model shift eps that widens every bracket."""
    out = []
    # truncated triplet: claim two of three members at 1.0
    d = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    r = _run_diag(d, [[1, 2]], claim_idx=[0, 1, 2])[0]
    out.append(("truncated multiplet", r["status"] == "INCONCLUSIVE" and r.get("degeneracy_at_least") == 2
                and r["count_difference"] == 3))
    # near-degenerate but distinct pair as one cluster: the certified statement is a count of 2 in a bracket,
    # which is TRUE (two eigenvalues in the bracket) and says nothing about equality
    d = np.array([-1.0, 0.0, 1e-9, 3.0])
    r = _run_diag(d, [[1, 2]])[0]
    lo, hi = r["bracket_inner"]
    out.append(("non-degenerate pair as a cluster", r["status"] == "CERTIFIED" and r["degeneracy"] == 2
                and int(np.sum((d > lo) & (d < hi))) == 2 and r["width"] >= 1e-9))
    # the same pair as two singlets: the minimal shift margin (1e-8 x scale) exceeds the splitting 1e-9, so
    # each bracket holds both eigenvalues: INCONCLUSIVE with count 2, never 'degeneracy 1'
    recs = _run_diag(d, [[1], [2]])
    out.append(("pair as two singlets inside the margin", all(
        r["status"] == "INCONCLUSIVE" and r["count_difference"] == 2 and r.get("degeneracy_at_least") == 1
        for r in recs)))
    # splitting above the margin: two CERTIFIED singlets, each bracket holding exactly one eigenvalue
    d2 = np.array([-1.0, 0.0, 1e-6, 3.0])
    recs = _run_diag(d2, [[1], [2]])
    out.append(("pair as two singlets outside the margin", all(
        r["status"] == "CERTIFIED" and r["degeneracy"] == 1
        and int(np.sum((d2 > r["bracket_inner"][0]) & (d2 < r["bracket_inner"][1]))) == 1 for r in recs)))
    # model shift: eps added to beta on both sides and to the radii
    d = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    recs = _run_diag(d, [[0, 1], [3, 4, 5]], eps=1e-10)
    out.append(("model shift eps", [r["status"] for r in recs] == ["CERTIFIED"] * 2
                and all(min(r["beta"]) >= 1e-10 for r in recs)))
    # a wrong count planted: if the lower count were too small by one (as if an eigenvalue were missed), the
    # difference exceeds g and the record must not be CERTIFIED (simulated on the statement, part A covers it);
    # here: a level whose claimed cluster omits one member while the omitted one lies inside J
    d = np.array([0.0, 1.0, 1.0 + 1e-13, 1.0, 5.0])
    r = _run_diag(d, [[1, 2]], claim_idx=[0, 1, 2])[0]
    out.append(("omitted member inside J", r["status"] != "CERTIFIED" or r["degeneracy"] == 3))
    return out


def run(n_random: int = 3000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = {}
    out["model_random"] = check_model(rng, n_random, False)
    out["model_adversarial"] = check_model(rng, n_random // 3, True)
    c, v, st = check_code(rng, max(20, n_random // 50))
    out["code_random"] = (c, v)
    out["code_statuses"] = st
    adv = code_adversarial()
    out["code_adversarial"] = (len(adv), sum(1 for _, ok in adv if not ok))
    out["code_adversarial_detail"] = [(name, ok) for name, ok in adv]
    out["random_cases"] = out["model_random"][0] + out["code_random"][0]
    out["adversarial_cases"] = out["model_adversarial"][0] + len(adv)
    out["violations"] = out["model_random"][1] + out["model_adversarial"][1] + v + out["code_adversarial"][1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = run(a.random, a.seed)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0 if r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
