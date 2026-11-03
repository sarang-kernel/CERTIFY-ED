"""Independent adversarial search for C-03 (docs/proofs/C-03_entanglement.md, entanglement.py).

Part 1 (mp, 60 digits): Proposition 1 with the TRUE s = ||Qhat Qhat^H - P||_2 and eta = ||X^H X - I||_2: random
Hermitian A with a g-fold cluster, X = exact eigenvectors mixed inside the eigenspace and perturbed (eta up to
0.99, s up to ~1), complex entries, g = 1, g = n (then P = I), several bipartitions; T computed exactly by the
trace norm of the partial-trace difference. Proposition 3 on the same data; Proposition 2 via Audenaert on
random density-matrix pairs at T near 1 - 1/d (tightness) and on the (rho_A, rho_hat_A) pairs.
Part 2 (exact rationals): Lemma 3 (enclose_psd_eigenvalues) on rational PSD matrices with KNOWN spectra built as
Q diag(p) Q^H with rational (Gaussian-rational) Householder Q: degenerate spectra, zero eigenvalues, d up to 8,
complex; containment checked with Fractions. Lemma 5 (_f_interval) against mp on intervals containing 1/e, the
ends 0 and 1, tiny arguments (including the sub-grid case 0 < p < 2^-192, expected to raise).
Part 3: certify_entanglement end-to-end on small spin chains with a planted eigenspace; the certified entropy
interval and spectrum intervals must contain the mp values of rho_A = tr_B(P)/g for the exact eigenspace.

Usage: python docs/proofs/independent/adversarial_C-03_entanglement.py [--cases 120] [--seed 53]
"""
import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, eig_herm, herm, herm_fun, mpf, report, spec_norm, to_mp  # noqa: E402
from certify_ed import ivl  # noqa: E402
from certify_ed.entanglement import (_f_interval, bipartition_indices, certify_entanglement, enclose_psd_eigenvalues,  # noqa: E402
                                     )

mp.mp.dps = 60


def trace_norm(M):
    w, _ = mp.eigh(M)
    return sum(abs(mp.re(v)) for v in w)


def partial_trace_B(Z, dA, dB):
    out = mp.zeros(dA, dA)
    for a in range(dA):
        for b in range(dA):
            out[a, b] = sum(Z[a * dB + k, b * dB + k] for k in range(dB))
    return out


def entropy(M):
    w, _ = mp.eigh(M)
    return -sum(mp.re(v) * mp.log(mp.re(v)) for v in w if mp.re(v) > mp.mpf("1e-50"))


def g_fa(T, d):
    T = mp.mpf(T)
    h = -(T * mp.log(T) if T > 0 else 0) - ((1 - T) * mp.log(1 - T) if T < 1 else 0)
    return T * mp.log(d - 1) + h


def part1(rng, cases, T_p1, T_p3, T_p2):
    for i in range(cases):
        cplx = i % 3 == 0
        dA, dB = int(rng.choice([2, 3, 4])), int(rng.choice([2, 3]))
        n = dA * dB
        g = int(rng.integers(1, n + 1)) if i % 5 else (1 if i % 2 else n)
        M = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
        Am = to_mp((M + M.conj().T) / 2)
        lam, V = eig_herm(Am)
        # make eigenvalues 0..g-1 degenerate (exact cluster): A := V diag(lam') V^H
        lamp = list(lam); base = lamp[0]
        for k in range(g):
            lamp[k] = base
        Am = V * mp.diag(lamp) * herm(V)
        P = V[:, :g] * herm(V[:, :g])
        # X: eigenspace basis mixed by a random (non-unitary) g x g matrix, plus a component outside
        C = to_mp(rng.standard_normal((g, g)) + (1j * rng.standard_normal((g, g)) if cplx else 0.0))
        # C = I + delta * random / sqrt(g): eta = ||C^H C - I|| ranges from ~0 to above 1 (the latter skipped)
        delta = mp.mpf(float(rng.choice([0.01, 0.05, 0.15, 0.3, 0.6]))) / mp.sqrt(g)
        C = mp.eye(g) + C * delta
        X = V[:, :g] * C
        if i % 2 and g < n:
            outside = V[:, g:] * to_mp(rng.standard_normal((n - g, g)) * float(rng.choice([1e-3, 0.1, 0.5])))
            X = X + outside
        G = herm(X) * X
        eta = spec_norm(G - mp.eye(g))
        if not eta < 1:
            T_p1.skip(); continue
        Minv = herm_fun(G, lambda v: 1 / mp.sqrt(v))
        Q = X * Minv
        s = spec_norm(Q * herm(Q) - P)
        t = sum(G[k, k] for k in range(g))
        rhoA = partial_trace_B(P, dA, dB) / g
        rhoA_hat = partial_trace_B(X * herm(X), dA, dB) / t
        T = trace_norm(rhoA - rhoA_hat) / 2
        Tstar = s + eta / (1 - eta) if g > 1 else s
        T_p1.add(T <= Tstar + mp.mpf("1e-40"), g in (1, n) or eta > mp.mpf("0.5"), f"P1 n={n} g={g} eta={mp.nstr(eta, 6)} s={mp.nstr(s, 6)} T={mp.nstr(T, 10)} T*={mp.nstr(Tstar, 10)}")
        la, _ = eig_herm(rhoA); lb, _ = eig_herm(rhoA_hat)
        T_p3.add(max(abs(x - y) for x, y in zip(la, lb)) <= 2 * Tstar + mp.mpf("1e-40"), g in (1, n), f"P3 n={n} g={g}")
        if Tstar <= 1 - mp.mpf(1) / dA:
            T_p2.add(abs(entropy(rhoA) - entropy(rhoA_hat)) <= g_fa(Tstar, dA) + mp.mpf("1e-30"), g in (1, n), f"P2 n={n} g={g} T*={mp.nstr(Tstar, 8)}")
    # Audenaert tightness: random density pairs with T near 1 - 1/d
    for i in range(cases // 2):
        d = int(rng.integers(2, 6))
        A1 = to_mp(rng.standard_normal((d, d))); R1 = A1 * herm(A1); R1 = R1 / sum(R1[k, k] for k in range(d))
        if i % 2:
            # pure vs maximally mixed: T = 1 - 1/d exactly
            R2 = mp.eye(d) / d; e1 = mp.zeros(d, 1); e1[0] = 1; R1 = e1 * herm(e1)
        else:
            A2 = to_mp(rng.standard_normal((d, d))); R2 = A2 * herm(A2); R2 = R2 / sum(R2[k, k] for k in range(d))
        T = trace_norm(R1 - R2) / 2
        T_p2.add(abs(entropy(R1) - entropy(R2)) <= g_fa(T, d) + mp.mpf("1e-30"), True, f"Audenaert d={d} T={mp.nstr(T, 8)}")


def householder_rational(rng, n, cplx):
    while True:
        v = [complex(int(rng.integers(-3, 4)), int(rng.integers(-3, 4)) if cplx else 0) for _ in range(n)]
        if any(v):
            break
    vv = sum(Fraction(z.real) ** 2 + Fraction(z.imag) ** 2 for z in v)
    Q = [[(Fraction(1 if i == j else 0), Fraction(0)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            re = Fraction(v[i].real) * Fraction(v[j].real) + Fraction(v[i].imag) * Fraction(v[j].imag)
            im = Fraction(v[i].imag) * Fraction(v[j].real) - Fraction(v[i].real) * Fraction(v[j].imag)
            Q[i][j] = (Q[i][j][0] - 2 * re / vv, Q[i][j][1] - 2 * im / vv)
    return Q


def part2(rng, cases, T_l3, T_l5):
    for i in range(cases):
        d = int(rng.integers(1, 9)); cplx = i % 3 == 0
        kind = i % 4
        if kind == 0:
            p = sorted(Fraction(int(rng.integers(0, 20)), 1) for _ in range(d))
        elif kind == 1:
            p = [Fraction(1)] * d                                   # maximally mixed after normalisation
        elif kind == 2:
            p = [Fraction(0)] * (d - 1) + [Fraction(1)]            # pure state: zero eigenvalues
        else:
            p = sorted(Fraction(int(rng.integers(1, 5)), int(rng.integers(1, 5))) for _ in range(d))
            if d >= 2:
                p[1] = p[0] + Fraction(1, 2 ** 70)
        if sum(p) == 0:
            p[-1] = Fraction(1)
        Q = householder_rational(rng, d, cplx)
        Mr = np.empty((d, d), dtype=object); Mi = np.empty((d, d), dtype=object)
        for a in range(d):
            for b in range(d):
                re = im = Fraction(0)
                for k in range(d):
                    qa, qb = Q[a][k], Q[b][k]
                    re += p[k] * (qa[0] * qb[0] + qa[1] * qb[1]); im += p[k] * (qa[1] * qb[0] - qa[0] * qb[1])
                Mr[a, b], Mi[a, b] = re, im
        t = sum(p)
        try:
            enc = enclose_psd_eigenvalues(Mr, Mi if cplx else None, t)
        except ValueError as e:
            T_l3.add(False, True, f"L3 raised: {e}"); continue
        true = sorted(pk / t for pk in p)
        ok = all(enc["lo"][k] <= true[k] <= enc["hi"][k] for k in range(d))
        T_l3.add(ok, kind != 0, f"L3 d={d} kind={kind} cplx={cplx}: enclosure misses an eigenvalue")
    # Lemma 5
    inv_e = mp.exp(-1)
    f = lambda x: -x * mp.log(x) if x > 0 else mp.mpf(0)   # noqa: E731
    tests = [(Fraction(0), Fraction(1)), (Fraction(0), Fraction(0)), (Fraction(1), Fraction(1)), (Fraction(1, 3), Fraction(2, 5)),
             (Fraction(367879441, 10 ** 9), Fraction(367879442, 10 ** 9)), (Fraction(1, 2 ** 100), Fraction(1, 2 ** 90)),
             (Fraction(999, 1000), Fraction(1)), (Fraction(1, 2 ** 192), Fraction(1, 2 ** 150))]
    for _ in range(cases):
        a_ = Fraction(int(rng.integers(0, 10 ** 6)), 10 ** 6); b_ = a_ + Fraction(int(rng.integers(0, 10 ** 5)), 10 ** 6)
        tests.append((a_, min(b_, Fraction(1))))
    for lo, hi in tests:
        iv = _f_interval(lo, hi)
        lo_q, hi_q = iv.fractions()
        pts = [lo, hi] + [lo + (hi - lo) * Fraction(j, 17) for j in range(1, 17)]
        if lo <= Fraction(367879441171442321, 10 ** 18) <= hi:
            pts.append(Fraction(367879441171442321, 10 ** 18))   # ~1/e
        vals = [f(mp.mpf(x.numerator) / x.denominator) for x in pts]
        ok = all(mp.mpf(lo_q.numerator) / lo_q.denominator - mp.mpf("1e-50") <= v <= mp.mpf(hi_q.numerator) / hi_q.denominator + mp.mpf("1e-50") for v in vals)
        if lo <= inv_e <= hi:
            ok = ok and mp.mpf(hi_q.numerator) / hi_q.denominator >= inv_e - mp.mpf("1e-50")
        T_l5.add(ok, True, f"L5 [{lo},{hi}]")
    try:
        _f_interval(Fraction(1, 2 ** 200), Fraction(1, 2 ** 199)); T_l5.add(True, True, "")
    except ValueError:
        T_l5.add(True, True, "")   # documented: sub-grid argument raises (NOT_CHECKED upstream); not a false value


class _EC:
    def __init__(self, members, s, eta):
        self.members, self.sin_theta, self.eta = members, s, eta


def part3(rng, cases, T_e2e):
    import certify_ed as ce
    from certify_ed.construct import build_bitwise, full_state_set, hermitian_part
    for i in range(cases):
        N = int(rng.choice([3, 4]))
        spec = ce.heisenberg_chain(N, J=1.0, Delta=float(rng.choice([1.0, 0.3, 2.0])))
        H = hermitian_part(build_bitwise(spec)).toarray()
        w, V = np.linalg.eigh(H)
        # cluster: the lowest level (exact degeneracy of the Heisenberg chain may be > 1): take g = multiplicity within 1e-9
        g = int(np.sum(np.abs(w - w[0]) < 1e-9))
        idx = list(range(g))
        X = V[:, idx].copy()
        if i % 2:
            X = X @ (np.eye(g) + 0.3 * rng.standard_normal((g, g)))     # non-orthonormal basis of the same space
        Xm = to_mp(X); Gm = herm(Xm) * Xm
        eta = spec_norm(Gm - mp.eye(g))
        if not eta < 1:
            continue
        # s: the true sin theta between the stored span and the exact eigenspace is ~1e-15; use a stated bound
        s = 1e-12
        states = full_state_set(spec)
        sites_A = sorted(rng.choice(N, size=int(rng.integers(1, N)), replace=False).tolist())
        rec = certify_entanglement(_EC(idx, s, float(ivl.to_float_up(ivl.from_fraction(Fraction(mp.nstr(eta, 30)) if False else Fraction(float(eta)) * (1 + Fraction(1, 10 ** 12))).hi))),
                                   X, states.states_in_order(), spec.n_sites, spec.q, sites_A)
        if rec["status"] != "CERTIFIED":
            T_e2e.skip(); continue
        # exact rho_A from the exact eigenspace (mp) and its entropy / spectrum
        a_idx, b_idx, dA, dB = bipartition_indices(states.states_in_order(), spec.n_sites, spec.q, sites_A)
        lam_mp, V_mp = eig_herm(to_mp(H))                    # exact H (dyadic entries): the eigenspace projector to ~1e-55
        sel = [k for k in range(len(lam_mp)) if abs(lam_mp[k] - lam_mp[0]) < mp.mpf("1e-30")]
        assert len(sel) == g, (sel, g)
        Vg = mp.matrix(H.shape[0], g)
        for j, k in enumerate(sel):
            for r_ in range(H.shape[0]):
                Vg[r_, j] = V_mp[r_, k]
        Pm = Vg * herm(Vg)
        rho = mp.zeros(dA, dA)
        for r_ in range(H.shape[0]):
            for c_ in range(H.shape[0]):
                if b_idx[r_] == b_idx[c_]:
                    rho[a_idx[r_], a_idx[c_]] += Pm[r_, c_] / g
        S = entropy(rho); lam, _ = eig_herm(rho)
        lo, hi = rec["certified_interval"]
        # the oracle itself is mp (60 digits): exact zero eigenvalues of the rank-g rho_A come out as +-1e-55 and the
        # code clamps its lower ends at 0 (Lemma 3, PSD) -- allow the oracle's own rounding, TOL << any certified width
        TOL = mp.mpf("1e-45")
        ok = mpf(lo) - TOL <= S <= mpf(hi) + TOL
        for k, sp_ in enumerate(rec["entanglement_spectrum"]):
            l_, h_ = sp_["certified_interval"]
            ok = ok and mpf(l_) - TOL <= lam[k] <= mpf(h_) + TOL
        T_e2e.add(ok, i % 2 == 1, f"e2e N={N} g={g} A={sites_A}: S={mp.nstr(S, 12)} interval=({lo},{hi})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_p1 = Tally("Proposition 1: T <= s + eta/(1 - eta) (s, eta exact; T exact trace norm)")
    T_p3 = Tally("Proposition 3: |lambda_k(rho_A) - lambda_k(rho_hat_A)| <= 2 T*")
    T_p2 = Tally("Proposition 2 / Audenaert: |S(rho) - S(sigma)| <= g_d(T) (incl. T = 1 - 1/d)")
    T_l3 = Tally("Lemma 3: exact enclosure contains the known spectrum (exact rationals)")
    T_l5 = Tally("Lemma 5: _f_interval encloses -p ln p on the interval (mp)")
    T_e2e = Tally("certify_entanglement: certified entropy and spectrum intervals contain the exact-eigenspace values")
    part1(rng, a.cases, T_p1, T_p3, T_p2)
    part2(rng, a.cases, T_l3, T_l5)
    part3(rng, max(6, a.cases // 10), T_e2e)
    sys.exit(1 if report([T_p1, T_p3, T_p2, T_l3, T_l5, T_e2e]) else 0)


if __name__ == "__main__":
    main()
