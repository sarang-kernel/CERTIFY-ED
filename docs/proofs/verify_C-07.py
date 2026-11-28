"""Computational verification of docs/proofs/C-07_moments.md (certified moments and sum rules).

Parts:
  A  floating-point accounting: the exact rational tr(X^H O H^k O X)/g of the stored binary64 data lies within the
     recorded rounding term of the computed value (Fractions; real and complex; k = 0..4; scales 1e-2..1e2; orthonormal
     columns, and the same columns perturbed by 1e-3 with eta declared as an upper bound on their Gram defect)
  B  full statement: random Hermitian H with a planted cluster of g eigenvalues, exact P (mpmath 60 digits), claimed
     vectors = rotated exact eigenvectors + perturbation, exact s = ||Qhat Qhat^H - P||_2 and eta fed as (rounded up)
     inputs: m_k = tr(P O H^k O)/g lies in every interval; both sum rules are consistent (CERTIFIED); vectors tilted out
     of the eigenspace with the exact s; small Heisenberg chains (N = 4, 6) with O = S^z_0 S^z_1 and the exact
     ground-state projector
  C  negative controls: s understated by 10^4 x on tilted vectors breaks the enclosure; a random orthonormal X with
     s = eta = 0 claimed breaks it; a wrong level interval (shifted beyond the interval widths) makes the first-moment sum rule FAILED (never CERTIFIED)
Run from the repository root: python docs/proofs/verify_C-07.py [--cases N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _common import Counter, eigh, herm, norm2, to_mp  # noqa: E402

from certify_ed.moments import certified_moments  # noqa: E402

mp.mp.dps = 60


def rand_herm(rng, n, cplx, scale=1.0):
    A = rng.standard_normal((n, n)) * scale
    if cplx:
        A = A + 1j * rng.standard_normal((n, n)) * scale
    return A + A.conj().T


def fr(z, cplx):
    return (Fraction(float(np.real(z))), Fraction(float(np.imag(z)))) if cplx else (Fraction(float(z)), Fraction(0))


def exact_trace_moments(H, O, X, K):
    """tr(X^H O H^k O X)/g, k = 0..K, exactly in Fractions (real parts)."""
    n, g = X.shape
    cplx = np.iscomplexobj(H) or np.iscomplexobj(O) or np.iscomplexobj(X)
    Hf = [[fr(H[i, j], cplx) for j in range(n)] for i in range(n)]
    Of = [[fr(O[i, j], cplx) for j in range(n)] for i in range(n)]

    def cmul(a, b):
        return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

    def matvec(M, v):
        out = []
        for i in range(n):
            re = Fraction(0)
            im = Fraction(0)
            for j in range(n):
                p = cmul(M[i][j], v[j])
                re += p[0]
                im += p[1]
            out.append((re, im))
        return out

    def inner(a, b):
        s = Fraction(0)
        for i in range(n):
            s += a[i][0] * b[i][0] + a[i][1] * b[i][1]
        return s

    out = [Fraction(0)] * (K + 1)
    for c in range(g):
        y = matvec(Of, [fr(X[i, c], cplx) for i in range(n)])
        z = y
        for k in range(K + 1):
            if k > 0:
                z = matvec(Hf, z)
            out[k] += inner(y, z)
    return [v / g for v in out]


def stmt(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)


def planted(rng, n, g, cplx, gap=1.0):
    """Hermitian H with eigenvalues: a cluster of g values within 1e-3 at the bottom, the rest above by >= gap."""
    vals = np.concatenate([rng.uniform(0, 1e-3, g), gap + rng.uniform(0, 3, n - g)])
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
    H = (Q * vals) @ Q.conj().T
    return (H + H.conj().T) / 2


def exact_cluster(H, g):
    Hm = to_mp(H)
    lam, Q = eigh(Hm)
    Qg = Q[:, 0:g]
    P = Qg * herm(Qg)
    return Hm, lam, P


def exact_moments_mp(Hm, Om, P, g, K):
    out = []
    M = Om
    for k in range(K + 1):
        if k > 0:
            M = Hm * M
        T = P * Om * M                                      # P O H^k O  (M = H^k O)
        out.append(mp.re(sum(T[i, i] for i in range(T.rows))) / g)
    return out


def s_eta_exact(X, P):
    Xm = to_mp(X)
    G = herm(Xm) * Xm
    g = G.rows
    eta = norm2(G - mp.eye(g))
    # polar factor Qhat = X (X^H X)^{-1/2}
    vals, V = eigh(G)
    Ginvhalf = V * mp.diag([1 / mp.sqrt(v) for v in vals]) * herm(V)
    Qh = Xm * Ginvhalf
    s = norm2(Qh * herm(Qh) - P)
    return s, eta


def header(args):
    """One line naming the run: arguments, commit, tracked changes and versions (None where git cannot tell)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                 capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        head, changes = None, None
    return (f"# verify_C-07.py --cases {args.cases} --seed {args.seed}; commit {head}; tracked changes {changes}; "
            f"Python {platform.python_version()}, NumPy {np.__version__}, mpmath {mp.__version__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    print(header(a))
    rng = np.random.default_rng(a.seed)
    rng_p = np.random.default_rng(a.seed + 1)
    cA, cB, cC = Counter(), Counter(), Counter()
    neg_s_broken = neg_s_total = 0
    neg_x_broken = neg_x_total = 0

    # ---- A: rounding accounting in exact rationals
    for case in range(a.cases):
        n = int(rng.integers(2, 6))
        g = int(rng.integers(1, 3))
        cplx = bool(case % 2)
        sc = float(10.0 ** rng.integers(-2, 3))
        H = rand_herm(rng, n, cplx, sc)
        O = rand_herm(rng, n, cplx, 1.0 / sc)
        X = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0)
        # orthonormal columns: since the moments check eta against the vectors, s = eta = 0 is refused for vectors far
        # from orthonormal (the projector term would be understated); this part checks the rounding term only
        X = np.linalg.qr(X)[0]
        K = 4
        # the same check on columns perturbed by 1e-3 with eta declared as an upper bound on their Gram defect (their
        # own generator, so that parts B and C draw the same numbers as before)
        Xp = X + 1e-3 * (rng_p.standard_normal((n, g)) + (1j * rng_p.standard_normal((n, g)) if cplx else 0))
        Xm = to_mp(Xp)
        eta_p = float(norm2(herm(Xm) * Xm - mp.eye(g))) * 1.01 + 1e-18
        for label, Xv, eta in (("orthonormal", X, 0.0), ("perturbed", Xp, eta_p)):
            rec = certified_moments(H, O, Xv, 0.0, eta, K=K, sum_rules=False)
            ex = exact_trace_moments(H, O, Xv, K)
            for m in rec["moments"]:
                k = m["k"]
                ok = abs(Fraction(m["value"]) - ex[k]) <= Fraction(m["terms"]["rounding"]) + Fraction(abs(m["value"])) * Fraction(2 ** -52)
                cA.add(False, ok, f"A case {case} ({label}) k={k}: |m_hat - exact| = {float(abs(Fraction(m['value']) - ex[k]))} > rounding {m['terms']['rounding']}")

    # ---- B: planted clusters, exact P
    for case in range(a.cases):
        n = int(rng.integers(4, 8))
        g = int(rng.integers(1, min(3, n // 2) + 1))
        cplx = bool(case % 3 == 1)
        H = planted(rng, n, g, cplx)
        O = rand_herm(rng, n, cplx)
        Hm, lam, P = exact_cluster(H, g)
        Om = to_mp(O)
        K = int(rng.integers(1, 5))
        _, Q = eigh(Hm)
        X0 = np.array([[complex(Q[i, j]) if cplx else float(mp.re(Q[i, j])) for j in range(g)] for i in range(n)])
        # rotate within the cluster and perturb
        R, _ = np.linalg.qr(rng.standard_normal((g, g)) + (1j * rng.standard_normal((g, g)) if cplx else 0))
        eps = float(10.0 ** rng.uniform(-9, -2))
        X = X0 @ R + eps * (rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0))
        s_ex, eta_ex = s_eta_exact(X, P)
        if not (s_ex < 1 and eta_ex < 1):
            cB.skipped += 1
            continue
        s_in = float(s_ex) * (1 + 1e-12) + 1e-300
        eta_in = float(eta_ex) * (1 + 1e-12) + 1e-300
        level = (float(lam[0]) - 1e-12, float(lam[g - 1]) + 1e-12)
        rec = certified_moments(H, O, X, s_in, eta_in, K=K, level=level)
        exm = exact_moments_mp(Hm, Om, P, g, K)
        for m in rec["moments"]:
            k = m["k"]
            ok = mp.mpf(m["interval"][0]) <= exm[k] <= mp.mpf(m["interval"][1])
            cB.add(False, ok, f"B case {case} k={k}: exact {float(exm[k])} not in {m['interval']}")
        for sid in ("M2.sum_rule_zeroth", "M3.sum_rule_first"):
            cB.add(False, stmt(rec, sid)["status"] == "CERTIFIED", f"B case {case}: {sid} is {stmt(rec, sid)['status']}")
        # negative control 1: s understated 10^4 x on vectors tilted out of the eigenspace by a perturbation orthogonal to
        # range P (then eta = O(eps^2) is tiny while s = O(eps) carries the whole first-order error)
        W = np.array([[complex(Q[i, j]) if cplx else float(mp.re(Q[i, j])) for j in range(g, 2 * g)] for i in range(n)])
        eps_t = 1e-3
        Xt = X0 @ R + eps_t * W @ R
        s_t, eta_t = s_eta_exact(Xt, P)
        rec_bad = certified_moments(H, O, Xt, float(s_t) / 1e4, float(eta_t) * (1 + 1e-12) + 1e-300, K=K, sum_rules=False)
        for m in rec_bad["moments"]:
            neg_s_total += 1
            if not (mp.mpf(m["interval"][0]) <= exm[m["k"]] <= mp.mpf(m["interval"][1])):
                neg_s_broken += 1
        rec_ok = certified_moments(H, O, Xt, float(s_t) * (1 + 1e-12) + 1e-300, float(eta_t) * (1 + 1e-12) + 1e-300, K=K, sum_rules=False)
        for m in rec_ok["moments"]:
            cB.add(True, mp.mpf(m["interval"][0]) <= exm[m["k"]] <= mp.mpf(m["interval"][1]),
                   f"B case {case} tilted k={m['k']}: exact {float(exm[m['k']])} not in {m['interval']}")
        # negative control 2: a random vector with s = eta = 0 claimed
        Xr = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0)
        Xr, _ = np.linalg.qr(Xr)
        rec_bad2 = certified_moments(H, O, Xr, 0.0, 0.0, K=K, sum_rules=False)
        for m in rec_bad2["moments"]:
            neg_x_total += 1
            if not (mp.mpf(m["interval"][0]) <= exm[m["k"]] <= mp.mpf(m["interval"][1])):
                neg_x_broken += 1
        # negative control 3: a wrong level interval, shifted by more than the combined widths of the two certified
        # intervals divided by m_0 (so that the two enclosures cannot overlap any more) -> M3 must be FAILED
        m3 = stmt(rec, "M3.sum_rule_first")["evidence"]
        w_lhs = m3["lhs_interval"][1] - m3["lhs_interval"][0]
        w_rhs = m3["rhs_interval"][1] - m3["rhs_interval"][0]
        m0_lo = max(rec["moments"][0]["interval"][0], 1e-12)
        shift = 1.0 + 2.0 * (w_lhs + w_rhs) / m0_lo
        rec_bad3 = certified_moments(H, O, X, s_in, eta_in, K=K, level=(level[0] + shift, level[1] + shift))
        cC.add(True, stmt(rec_bad3, "M3.sum_rule_first")["status"] == "FAILED",
               f"C case {case}: wrong level interval (shift {shift:.3g}) gave {stmt(rec_bad3, 'M3.sum_rule_first')['status']}")

    # ---- B': small Heisenberg chains, O = S^z_0 S^z_1, exact ground-state projector
    from certify_ed.construct import build_bitwise
    from certify_ed.correlations import site_operator
    from certify_ed.spec import heisenberg_chain
    for N in (4, 6):
        spec = heisenberg_chain(N)
        H = build_bitwise(spec).toarray()
        O = site_operator(spec, "zz", [0, 1]).toarray().real
        w, V = np.linalg.eigh(H)
        g = 1
        Hm, lam, P = exact_cluster(H, g)
        X = V[:, :g]
        s_ex, eta_ex = s_eta_exact(X, P)
        rec = certified_moments(H, O, X, float(s_ex) * 1.01 + 1e-20, float(eta_ex) * 1.01 + 1e-20, K=4,
                                level=(float(lam[0]) - 1e-13, float(lam[g - 1]) + 1e-13))
        exm = exact_moments_mp(Hm, to_mp(O), P, g, 4)
        for m in rec["moments"]:
            cB.add(True, mp.mpf(m["interval"][0]) <= exm[m["k"]] <= mp.mpf(m["interval"][1]),
                   f"chain N={N} k={m['k']}: exact {float(exm[m['k']])} not in {m['interval']}")
        for sid in ("M2.sum_rule_zeroth", "M3.sum_rule_first"):
            cB.add(True, stmt(rec, sid)["status"] == "CERTIFIED", f"chain N={N}: {sid} is {stmt(rec, sid)['status']}")

    ok = True
    ok &= cA.report("A rounding term covers the exact rational tr(X^H O H^k O X)/g")
    ok &= cB.report("B exact tr(P O H^k O)/g inside the intervals, sum rules consistent (planted clusters, tilted vectors, Heisenberg chains)")
    ok &= cC.report("C wrong level interval refuted by the first-moment sum rule")
    print(f"   negative control: s understated 10^4 x breaks the enclosure in {neg_s_broken}/{neg_s_total} moments; a random "
          f"orthonormal X with s = eta = 0 claimed breaks it in {neg_x_broken}/{neg_x_total} moments")
    fired = neg_s_broken > 0 and neg_x_broken > 0
    total_r = cA.random + cB.random + cC.random
    total_a = cA.adversarial + cB.adversarial + cC.adversarial
    print(f"TOTAL: {total_r} random + {total_a} adversarial cases; violations: {0 if ok else 'see above'}"
          + ("" if fired else "; a negative control did not fire"))
    return 0 if ok and fired else 1


if __name__ == "__main__":
    sys.exit(main())
