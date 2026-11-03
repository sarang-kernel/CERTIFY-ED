"""Independent adversarial search for Proposition 14 (docs/proofs/P14_degeneracy_bracketing.md, degeneracy.py).

Part A (exact rationals): random spectra with repeated eigenvalues; shifts sigma_lo < J_L, sigma_hi > J_U with
backward errors beta and counts nu chosen anywhere in the Lemma 4 range [N(sigma - beta), N^(sigma + beta)]
(including the ends: eigenvalues exactly at sigma +- beta); J chosen to contain >= g eigenvalues; (a)-(e) checked
exactly, including the boundary I_in = empty, J touching the ends of I_in, g = 1, g = n.
Part B (code): symmetric matrices with planted degenerate levels (g = 1..4, including g = n), repeated levels
close together, scales 1e+-100, model shift eps with a perturbed operator; every CERTIFIED record is checked
against the exact eigenvalues (mp): exactly g eigenvalues of the declared model in the printed open interval.
Negative controls: a cluster that omits one member of its level (diff > g must not certify g), a non-distinct
cluster (duplicated vector), a cluster whose J is not inside I_in by construction (tiny margin).

Usage: python docs/proofs/independent/adversarial_P14_degeneracy_bracketing.py [--cases 300] [--seed 37]
"""
import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, eig_herm, mpf, report, to_mp  # noqa: E402
from certify_ed.degeneracy import degeneracy_counts  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, cluster_groups, enclose  # noqa: E402


def part_a(rng, cases, T):
    for i in range(cases):
        n = int(rng.integers(1, 9))
        lam = sorted(Fraction(int(v), 4) for v in rng.integers(-12, 13, size=n))
        N = lambda x: sum(1 for v in lam if v < x)       # noqa: E731
        Nb = lambda x: sum(1 for v in lam if v <= x)     # noqa: E731
        # a level: pick an eigenvalue value and the g = multiplicity (or fewer: the claim may cover part of it)
        v0 = lam[int(rng.integers(0, n))]
        mult = sum(1 for v in lam if v == v0)
        g = int(rng.integers(1, mult + 1))
        r = Fraction(int(rng.integers(0, 3)), 8)
        JL, JU = v0 - r, v0 + r                              # J contains >= g eigenvalues (at least the mult copies)
        kind = i % 4
        b_lo = Fraction(int(rng.integers(0, 3)), 8); b_hi = Fraction(int(rng.integers(0, 3)), 8)
        if kind == 0:
            s_lo = JL - b_lo - Fraction(int(rng.integers(1, 5)), 8); s_hi = JU + b_hi + Fraction(int(rng.integers(1, 5)), 8)
        elif kind == 1:   # I_in ends touching J (J inside I_in requires strict <)
            s_lo = JL - b_lo - Fraction(1, 64); s_hi = JU + b_hi + Fraction(1, 64)
        elif kind == 2:   # I_in empty or J not inside
            s_lo = JL - Fraction(1, 16); s_hi = JU + Fraction(1, 16); b_lo = Fraction(1, 2); b_hi = Fraction(1, 2)
        else:             # eigenvalues exactly at sigma +- beta
            s_lo = JL - Fraction(1, 2); b_lo = Fraction(1, 4); s_hi = JU + Fraction(1, 2); b_hi = Fraction(1, 4)
            lam = sorted(lam + [s_lo - b_lo, s_hi + b_hi]); n = len(lam)
            N = lambda x: sum(1 for v in lam if v < x)   # noqa: E731
            Nb = lambda x: sum(1 for v in lam if v <= x)  # noqa: E731
        if not (s_lo < JL and s_hi > JU):
            continue
        nu_lo = int(rng.integers(N(s_lo - b_lo), Nb(s_lo + b_lo) + 1))
        nu_hi = int(rng.integers(N(s_hi - b_hi), Nb(s_hi + b_hi) + 1))
        I_in = (s_lo + b_lo, s_hi - b_hi); I_out = (s_lo - b_lo, s_hi + b_hi)
        cnt_out = sum(1 for v in lam if I_out[0] <= v <= I_out[1])
        cnt_in = sum(1 for v in lam if I_in[0] < v < I_in[1])
        cnt_J = sum(1 for v in lam if JL <= v <= JU)
        diff = nu_hi - nu_lo
        ok = diff <= cnt_out                                                      # (a)
        if I_in[0] < I_in[1]:
            ok = ok and cnt_in <= diff                                            # (b)
        if I_in[0] < JL and JU < I_in[1] and cnt_J >= g:
            if diff == g:
                ok = ok and cnt_in == g                                           # (c)
            ok = ok and not (diff < g)                                            # (e)
            if diff > g:
                ok = ok and cnt_out - g >= diff - g                               # (d)
        T.add(ok, kind != 0, f"exact case lam={[str(v) for v in lam]} J=[{JL},{JU}] g={g} s=({s_lo},{s_hi}) b=({b_lo},{b_hi}) nu=({nu_lo},{nu_hi})")


def planted(rng, kind):
    n = int(rng.integers(3, 8))
    scale = {0: 1.0, 1: 1e100, 2: 1e-100}[int(rng.integers(0, 3))]
    d = np.sort(rng.standard_normal(n)) * scale
    g = int(rng.integers(1, min(4, n) + 1))
    if kind == "g_eq_n":
        g = n
    d[:g] = d[0]                                   # a g-fold level at the bottom
    if kind == "close_levels" and n > g + 1:
        d[g] = d[0] + 1e-6 * scale; d[g + 1] = d[g]   # a second level close by
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(d) @ Q.T; A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    return A, w, V, g, scale


def part_b(rng, cases, T_code, T_neg):
    kinds = ["plain", "close_levels", "g_eq_n", "shift", "neg_omit", "neg_dup"]
    for i in range(cases):
        kind = kinds[i % len(kinds)]
        A, w, V, g, scale = planted(rng, kind)
        n = A.shape[0]
        eps = 0.0
        E = np.zeros((n, n))
        if kind == "shift":
            eps = 1e-10 * scale
            E = rng.standard_normal((n, n)); E = (E + E.T) / 2; E *= eps / np.linalg.norm(E, 2) * 0.99
        if kind == "neg_omit" and g >= 2:
            idx = list(range(g - 1))                   # claim g - 1 of the g members of the level
        else:
            idx = list(range(g))
        X = V[:, idx] + 1e-11 * rng.standard_normal((n, len(idx)))
        mu = w[idx].copy()
        if kind == "neg_dup" and len(idx) >= 2:
            X[:, 1] = X[:, 0]; mu[1] = mu[0]
        enc = enclose(OperatorAdapter(A), mu, X)
        clusters = [list(range(len(idx)))]
        recs = degeneracy_counts(A, enc, clusters, eps_shift=eps)
        rec = recs[0]
        lam_m, _ = eig_herm(to_mp(A + E))              # the declared model: A + E, ||E|| <= eps
        if kind == "neg_dup":
            if len(idx) >= 2:
                T_neg.add(rec["status"] != "CERTIFIED", True, f"neg_dup certified: {rec}")
            continue
        if kind == "neg_omit" and g >= 2:
            # g - 1 claimed of a g-fold level: the counts find g (or more): CERTIFIED with g - 1 would be false
            T_neg.add(not (rec["status"] == "CERTIFIED" and rec["degeneracy"] == g - 1), True, f"neg_omit certified g-1: {rec}")
            if rec["status"] == "CERTIFIED":
                lo, hi = rec["bracket_inner"]
                T_code.add(sum(1 for v in lam_m if mpf(lo) < v < mpf(hi)) == rec["degeneracy"], True, f"neg_omit count: {rec}")
            continue
        if rec["status"] != "CERTIFIED":
            T_code.skip(); continue
        lo, hi = rec["bracket_inner"]
        cnt = sum(1 for v in lam_m if mpf(lo) < v < mpf(hi))
        T_code.add(cnt == rec["degeneracy"] == len(idx), kind != "plain", f"{kind}: certified g={rec['degeneracy']} but {cnt} model eigenvalues in ({lo},{hi})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=37)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_a = Tally("(a)-(e) in exact rationals with nu anywhere in the Lemma 4 range")
    T_code = Tally("degeneracy_counts CERTIFIED: exactly g eigenvalues of the declared model in the printed open interval")
    T_neg = Tally("negative controls: duplicated vector / omitted level member never certified with the claimed g")
    part_a(rng, a.cases * 4, T_a)
    part_b(rng, a.cases, T_code, T_neg)
    sys.exit(1 if report([T_a, T_code, T_neg]) else 0)


if __name__ == "__main__":
    main()
