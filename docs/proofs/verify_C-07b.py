"""Computational verification of docs/proofs/C-07b_spectral_bounds.md (Chebyshev moments M5, distribution bounds M6).

Parts:
  A  Lemmas 5-7 (M5): random Hermitian H (real and complex, scales 1e-2..1e2, shifted spectra), the cluster of the g
     lowest eigenvalues with its exact projector P (mpmath 60 digits), stored vectors = eigenvectors + a tilt, exact
     s = sin theta and eta fed in (rounded up); the exact tau_j = tr(P O T_j(A) O)/g with the record's c and s lies in
     every interval, and the spectral interval holds every exact eigenvalue and maps into [-1, 1] (exact rationals);
     Heisenberg chains N = 4, 6 with the staggered S^z
  B  Proposition 3 and Lemma 8 (M6): atomic measures on [-1, 1] (random, clustered, atoms at +-1 and at the queried
     point) with exact moments widened to intervals; the exact F(xi) = sum of the weights of the atoms <= xi lies in
     [lower, upper] for the Radau-Hermite polynomials and for the linear program (forced)
  C  Lemma 8 alone: random integer polynomials and majorant candidates; whenever a proof succeeds, no sign violation at
     8193 exact rational points and at rational points next to the real critical points; polynomials with a dip of
     one unit below a touching point are never proved
  N  negative controls: s understated (0 claimed for tilted vectors) and a spectral lower bound above the lowest
     eigenvalue break the M5 enclosure; moment intervals shifted beyond their widths move the M6 bounds off the exact F
Run from the repository root: python docs/proofs/verify_C-07b.py [--cases N] [--seed S]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _common import TOL, Counter, eigh, herm, to_mp  # noqa: E402

from certify_ed import spectral_bounds as sb  # noqa: E402

mp.mp.dps = 60


def rand_herm(rng, n, cplx, scale):
    A = rng.standard_normal((n, n)) * scale
    if cplx:
        A = A + 1j * rng.standard_normal((n, n)) * scale
    return A + A.conj().T


def exact_cluster_weights(Hm, Om, g):
    """Exact eigenvalues of Hm, the projector P onto the g lowest, and w_i = (1/g) sum_p |q_p^H O q_i|^2."""
    vals, Q = eigh(Hm)
    n = Hm.rows
    OQ = Om * Q
    w = []
    for i in range(n):
        tot = mp.mpf(0)
        for p in range(g):
            z = sum(mp.conj(Q[k, p]) * OQ[k, i] for k in range(n))
            tot += abs(z) ** 2
        w.append(tot / g)
    P = mp.matrix(n, n)
    for p in range(g):
        for a in range(n):
            for b in range(n):
                P[a, b] += Q[a, p] * mp.conj(Q[b, p])
    return vals, Q, w, P


def s_eta_exact(X, P):
    """sin theta = ||(I - P) Qx||_2 for an orthonormal basis Qx of range(X), and eta = ||X^H X - I||_2 (exact)."""
    Xm = to_mp(X)
    G = herm(Xm) * Xm
    g = G.rows
    evals, evecs = eigh(G)
    eta = max(abs(e - 1) for e in evals)
    # Qx = X G^{-1/2}
    Gm12 = evecs * mp.diag([1 / mp.sqrt(e) for e in evals]) * herm(evecs)
    Qx = Xm * Gm12
    n = P.rows
    R = (mp.eye(n) - P) * Qx
    S = herm(R) * R
    sv, _ = eigh(S)
    s = mp.sqrt(max(sv[-1], mp.mpf(0)))
    return s, eta, g


def up_float(x):
    f = float(x)
    return float(np.nextafter(f, np.inf)) if mp.mpf(f) < x else f


def part_a(rng, cases, cA, neg):
    from certify_ed.construct import build_bitwise
    from certify_ed.correlations import site_operator
    from certify_ed.spec import heisenberg_chain
    work = []
    for case in range(cases):
        adv = case % 3 == 2
        n = int(rng.integers(5, 9))
        cplx = bool(rng.integers(0, 2))
        scale = float(10.0 ** rng.integers(-2, 3))
        H = rand_herm(rng, n, cplx, scale)
        if adv:
            H = H + np.eye(n) * scale * float(rng.choice([-1e3, 1e3, 37.5]))    # spectrum far from 0
        O = rand_herm(rng, n, cplx and bool(rng.integers(0, 2)), 1.0)
        g = int(rng.integers(1, 3))
        work.append((f"random case {case}", H, O, g, adv, cplx))
    for N in (4, 6):
        spec = heisenberg_chain(N)
        H = build_bitwise(spec).toarray()
        O = (sum(np.cos(np.pi * j) * site_operator(spec, "z", [j]) for j in range(N)) * np.sqrt(2.0 / N)).toarray().real
        work.append((f"chain N={N}", H, O, 1, True, False))
    for name, H, O, g, adv, _cplx in work:
        Hm, Om = to_mp(H), to_mp(O)
        vals, Q, w, P = exact_cluster_weights(Hm, Om, g)
        if vals[g] - vals[g - 1] < mp.mpf(1e-6) * max(abs(vals[0]), abs(vals[-1]), 1):
            cA.skipped += 1
            continue
        _, V = np.linalg.eigh(H)
        X = V[:, :g] + 1e-9 * (rng.standard_normal((H.shape[0], g)) + (1j * rng.standard_normal((H.shape[0], g))
                                                                        if np.iscomplexobj(V) else 0))
        s_ex, eta_ex, _ = s_eta_exact(X, P)
        s_in, eta_in = up_float(s_ex * mp.mpf("1.01") + mp.mpf("1e-20")), up_float(eta_ex * mp.mpf("1.01") + mp.mpf("1e-20"))
        lower = None
        if adv or rng.integers(0, 2):
            lower = float(vals[0]) if mp.mpf(float(vals[0])) <= vals[0] else float(np.nextafter(float(vals[0]), -np.inf))
        K = int(rng.integers(8, 25))
        rec = sb.certified_chebyshev_moments(H, O, X, s_in, eta_in, K=K, spectrum_lower=lower,
                                            spectrum_source="exact (verification)" if lower is not None else "")
        st = rec["statements"][-1]
        cA.add(adv, st["status"] == "CERTIFIED", f"{name}: M5 is {st['status']}: {st['summary'][:120]}")
        if st["status"] != "CERTIFIED":
            continue
        ch = rec["chebyshev"]
        c, s = mp.mpf(ch["center"]), mp.mpf(ch["scale"])
        lo, hi = ch["spectrum_interval"]
        fc, fs = Fraction(ch["center"]), Fraction(ch["scale"])
        cA.add(adv, all(mp.mpf(lo) <= v <= mp.mpf(hi) for v in vals), f"{name}: an eigenvalue outside [{lo}, {hi}]")
        cA.add(adv, (Fraction(lo) - fc) * fs >= -1 and (Fraction(hi) - fc) * fs <= 1, f"{name}: map not into [-1, 1]")
        a = [(v - c) * s for v in vals]
        for m in ch["moments"]:
            j = m["j"]
            tau = sum(wi * mp.cos(j * mp.acos(max(min(ai, mp.mpf(1)), mp.mpf(-1)))) for wi, ai in zip(w, a))
            slack = TOL * (1 + abs(tau))
            ok = mp.mpf(m["interval"][0]) - slack <= tau <= mp.mpf(m["interval"][1]) + slack
            cA.add(adv, ok, f"{name} j={j}: exact {mp.nstr(tau, 17)} not in {m['interval']}")
        # negative controls on the same data: s = eta = 0 claimed for the tilted vectors; a lower bound too high
        rec0 = sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=K, spectrum_lower=lower)
        bad = 0
        for m in rec0["chebyshev"].get("moments", []):
            j = m["j"]
            cc, ss = mp.mpf(rec0["chebyshev"]["center"]), mp.mpf(rec0["chebyshev"]["scale"])
            tau = sum(wi * mp.cos(j * mp.acos(max(min((v - cc) * ss, mp.mpf(1)), mp.mpf(-1)))) for wi, v in zip(w, vals))
            bad += not (mp.mpf(m["interval"][0]) <= tau <= mp.mpf(m["interval"][1]))
        neg["s0_broken"] += bad
        neg["s0_total"] += len(rec0["chebyshev"].get("moments", []))
        span = float(vals[-1] - vals[0])
        wrong = float(vals[0]) + 0.2 * span
        try:
            recw = sb.certified_chebyshev_moments(H, O, X, s_in, eta_in, K=K, spectrum_lower=wrong)
        except ValueError:
            continue
        cw, sw = mp.mpf(recw["chebyshev"]["center"]), mp.mpf(recw["chebyshev"]["scale"])
        aw = [(v - cw) * sw for v in vals]
        for m in recw["chebyshev"].get("moments", []):
            j = m["j"]
            tau = sum(wi * mp.cosh(j * mp.acosh(-ai)) * (-1) ** j if ai < -1 else
                      wi * mp.cos(j * mp.acos(max(min(ai, mp.mpf(1)), mp.mpf(-1)))) for wi, ai in zip(w, aw))
            neg["low_broken"] += not (mp.mpf(m["interval"][0]) <= tau <= mp.mpf(m["interval"][1]))
            neg["low_total"] += 1


def exact_F(atoms, weights, xi):
    return sum((wt for a, wt in zip(atoms, weights) if Fraction(a) <= xi), Fraction(0))


def part_b(rng, cases, cB, neg):
    for case in range(cases):
        adv = case % 3 == 2
        m = int(rng.integers(1, 40))
        atoms = list(np.sort(rng.uniform(-1, 1, m)))
        if adv:
            kind = case % 4
            if kind == 0:
                atoms = sorted(atoms[: m // 2] + [atoms[0] + k * 2.0 ** -30 for k in range(1, 4)])   # a tight cluster
            elif kind == 1:
                atoms = sorted(set(atoms + [-1.0, 1.0]))
            else:
                atoms = sorted(set(atoms + [0.0, 0.5, -0.5]))
        weights = [Fraction(int(rng.integers(1, 1000)), 1000) for _ in atoms]
        K = int(rng.integers(6, 41))
        tau = []
        for j in range(K + 1):
            t = sum(mp.mpf(wt.numerator) / wt.denominator * mp.cos(j * mp.acos(mp.mpf(a))) for a, wt in zip(atoms, weights))
            tau.append(t)
        rho = float(rng.choice([1e-14, 1e-10, 1e-6]))
        intervals = [(float(np.nextafter(float(t - rho), -np.inf)), float(np.nextafter(float(t + rho), np.inf))) for t in tau]
        tau_hat = np.array([float(t) for t in tau])
        T = sb._cheb_monomials(K)
        xis = [float(x) for x in rng.uniform(-1, 1, 3)]
        if adv:
            xis += [float(atoms[0]), float(np.nextafter(atoms[0], -np.inf)), -1.0, float(atoms[-1])]
        force_lp = case % 5 == 4 and K <= 24
        saved = sb._radau_hermite
        if force_lp:
            sb._radau_hermite = lambda *a, **k: None
        try:
            for xi in xis:
                if not (-1 <= xi < 1):
                    continue
                xf = Fraction(xi)
                F = exact_F(atoms, weights, xf)
                up_v, ui = sb._one_side(tau_hat, intervals, T, xf, True, None)
                lo_v, li = sb._one_side(tau_hat, intervals, T, xf, False, None)
                if up_v is not None:
                    cB.add(adv, Fraction(up_v) >= F, f"case {case} xi={xi}: upper {up_v} < F {float(F)} ({ui})")
                if lo_v is not None:
                    cB.add(adv, Fraction(lo_v) <= F, f"case {case} xi={xi}: lower {lo_v} > F {float(F)} ({li})")
                # negative control: moment intervals shifted by 1000 rho + 1e-3 at one j
                if up_v is not None and case % 2 == 0:
                    j = int(rng.integers(0, K + 1))
                    sh = 1000 * rho + 1e-3
                    iv2 = list(intervals)
                    iv2[j] = (intervals[j][0] - sh, intervals[j][0] - sh * 0.5)
                    th2 = tau_hat.copy()
                    th2[j] = 0.5 * (iv2[j][0] + iv2[j][1])
                    u2, _ = sb._one_side(th2, iv2, T, xf, True, None)
                    l2, _ = sb._one_side(th2, iv2, T, xf, False, None)
                    neg["shift_total"] += 1
                    neg["shift_moved"] += int((u2 is not None and Fraction(u2) != Fraction(up_v)) or
                                              (l2 is not None and lo_v is not None and Fraction(l2) != Fraction(lo_v)))
                    neg["shift_excludes"] += int((u2 is not None and Fraction(u2) < F) or
                                                 (l2 is not None and Fraction(l2) > F))
        finally:
            sb._radau_hermite = saved


def eval_frac(A, x):
    v = Fraction(0)
    for c in reversed(A):
        v = v * x + c
    return v


def critical_points(A):
    d = np.polynomial.polynomial.polyder(np.array(A, dtype=float))
    if len(d) == 0 or not np.any(d):
        return []
    r = np.roots(d[::-1]) if len(d) > 1 else []
    out = []
    for z in r:
        if abs(z.imag) < 1e-9 and -1 <= z.real <= 1:
            for dx in (0.0, 1e-12, -1e-12, 1e-9, -1e-9):
                x = min(1.0, max(-1.0, z.real + dx))
                out.append(Fraction(x))
    return out


def part_c(rng, cases, cC):
    grid = [Fraction(k, 4096) for k in range(-4096, 4097)]
    for case in range(cases):
        adv = case % 2 == 1
        m = int(rng.integers(1, 11))
        if adv:
            # (2^k x - p)^2 B(x) + e with B > 0 on [-1, 1]: e = 0 touches, e = -1 dips by one unit of the leading scale
            k = int(rng.integers(1, 8))
            p = int(rng.integers(-(2 ** k) + 1, 2 ** k))
            B = [int(rng.integers(1, 50))] + [0] * int(rng.integers(0, 3))
            sq = [p * p, -2 * p * 2 ** k, 2 ** (2 * k)]
            A = [0] * (len(B) + 2)
            for i, bi in enumerate(B):
                for jj, sj in enumerate(sq):
                    A[i + jj] += bi * sj
            e = int(rng.choice([0, -1]))
            A[0] += e
        else:
            A = [int(rng.integers(-2 ** 20, 2 ** 20)) for _ in range(m + 1)]
            pts = np.linspace(-1, 1, 257)
            vmin = float(np.min(np.polynomial.polynomial.polyval(pts, np.array(A, dtype=float))))
            A[0] -= int(math.floor(vmin)) - int(rng.choice([2, 0, -1, -100]))
        proved = sb._nonneg_bernstein(sb._bernstein_int(A), max_depth=30)
        if proved:
            ok = all(eval_frac(A, x) >= 0 for x in grid) and all(eval_frac(A, x) >= 0 for x in critical_points(A))
            cC.add(adv, ok, f"case {case}: proved but negative somewhere, A = {A}")
        if adv and e == -1:
            # the dip: (2^k x - p)^2 B - 1 is -1 at x = p / 2^k, which is in [-1, 1]
            cC.add(True, not proved, f"case {case}: a dip of one unit was proved nonnegative, A = {A}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cA, cB, cC = Counter(), Counter(), Counter()
    neg = {"s0_broken": 0, "s0_total": 0, "low_broken": 0, "low_total": 0, "shift_total": 0, "shift_moved": 0,
           "shift_excludes": 0}
    part_a(rng, args.cases, cA, neg)
    part_b(rng, args.cases * 2, cB, neg)
    part_c(rng, args.cases * 4, cC)
    ok = True
    ok &= cA.report("A exact tau_j = tr(P O T_j(A) O)/g inside the M5 intervals; spectral interval holds the spectrum")
    ok &= cB.report("B exact F(xi) between the M6 bounds (Radau-Hermite and linear program; atoms at the query point)")
    ok &= cC.report("C Bernstein proofs: no sign violation where proved; one-unit dips never proved")
    print(f"   negative control: s = eta = 0 claimed for tilted vectors breaks the M5 enclosure in "
          f"{neg['s0_broken']}/{neg['s0_total']} moments; a spectral lower bound 20% of the span too high breaks it in "
          f"{neg['low_broken']}/{neg['low_total']} moments")
    print(f"   negative control: one moment interval shifted beyond its width moves the M6 bound in "
          f"{neg['shift_moved']}/{neg['shift_total']} queries and puts the exact F outside in {neg['shift_excludes']}")
    ok &= neg["s0_broken"] > 0 and neg["low_broken"] > 0 and neg["shift_moved"] > 0
    total_r = cA.random + cB.random + cC.random
    total_a = cA.adversarial + cB.adversarial + cC.adversarial
    print(f"TOTAL: {total_r} random + {total_a} adversarial cases; violations: {0 if ok else 'see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
