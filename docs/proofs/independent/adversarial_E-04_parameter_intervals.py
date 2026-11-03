"""Independent adversarial check of E-04 (eigenvalue tubes over a coupling interval), written from the statement only.

Part A (mp, 60 digits): Propositions 1 and 2 on random Hermitian families with a non-zero affinity defect (H(c_{k+1})
= F_k(c_{k+1}) + E, ||E|| = d_k), the tube of `tube_at` built from exact eigenvalue enclosures (outward binary64)
against the exact eigenvalues of F_k(c) at random c (incl. c at and next to both grid points), B the exact ||H1||_2;
equality cases H1 = I, rank-one H1, degenerate H0; the count form at random thresholds and the gap line.
Part B (exact rationals): Lemma 3 / `certified_crossing_interval` on synthetic scalar scans (piecewise affine
branches with random affinity defects at the grid points, random certificate radii, two different grids): every
sample interval labelled '<' or '>' must have the true segment branches in that relation at 64 rational points of
the interval. This is the check that failed on the pre-fix code (the last sample interval before a grid point with
d_k > 0); the negative control is tests/test_independent_check.py.
Part C: Lemma 4 for every letter of every local space (spin 1/2 .. 7/2, three conventions; fermions; bosons n_max
<= 10) and random strings of 1..3 letters with a Jordan-Wigner-like sign string: exact spectral norm (mp) of the
Kronecker product <= term_norm_bound.
Part D (end to end): certified_sweep on TFIM N = 3 (grid 0.3/0.8/1.3, defects > 0) and N = 4 (grid 0.5/1/1.5,
exactly affine): tube_at at random couplings contains the mp eigenvalue of the exact F_k(c) (dyadic matrices);
negative control B/20.

Usage: python docs/proofs/independent/adversarial_E-04_parameter_intervals.py [--cases 120] [--seed 71]
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import mpf, to_mp, eig_herm, spec_norm, Tally, report  # noqa: E402
import certify_ed as ce  # noqa: E402
from certify_ed import sweep  # noqa: E402
from certify_ed.construct import build_bitwise  # noqa: E402
from certify_ed.fp import add_up, sub_down  # noqa: E402
from certify_ed.localops import LocalSpace  # noqa: E402

mp.mp.dps = 60


def out_lo(x):
    return float(np.nextafter(float(mp.mpf(x)), -np.inf))


def out_hi(x):
    return float(np.nextafter(float(mp.mpf(x)), np.inf))


def rand_herm(rng, n, cplx, scale=1.0):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    return (A + A.conj().T) / 2 * scale


# ------------------------------------------------------------------------------------------- Part A: Props 1, 2
def part_A(rng, cases, T1, T2):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 7))
        kind = i % 6
        H0 = rand_herm(rng, n, cplx)
        if kind == 3:                                   # degenerate H0
            H0 = np.diag(np.repeat(rng.standard_normal(2), n)[:n]).astype(complex if cplx else float)
        scale1 = float(rng.choice([1e-3, 1.0, 1e3]))
        H1 = rand_herm(rng, n, cplx, scale1)
        if kind == 1:
            H1 = np.eye(n) * scale1                     # equality in Weyl
        if kind == 2:
            v = rng.standard_normal((n, 1)); H1 = (v @ v.T) * scale1
        c0, c1 = sorted(float(x) for x in rng.uniform(-3, 3, size=2)); c1 = max(c1, c0 + 1e-3)
        d_true = float(rng.choice([0.0, 1e-12, 1e-3, 0.5]))
        E = rand_herm(rng, n, cplx); E = E / max(np.linalg.norm(E, 2), 1e-300) * d_true
        H0m, H1m, Em = to_mp(H0), to_mp(H1), to_mp(E)
        Hk1 = H0m + mpf(c1 - c0) * H1m + Em                 # H(c_{k+1}) = F_k(c_{k+1}) + E
        B = spec_norm(H1m)
        d = spec_norm(Em)
        lam0, _ = eig_herm(H0m); lam1, _ = eig_herm(Hk1)
        # synthetic scan record with exact (outward) enclosures and exact B, d (rounded up)
        rad = [float(rng.choice([0.0, 1e-10, 1e-2])) for _ in range(n)]
        pts = []
        for c, lam in ((c0, lam0), (c1, lam1)):
            pts.append({"index": len(pts), "c": c, "rank_certified": True, "next_eigenvalue_lower_bound": None,
                        "lower": [float(sub_down(out_lo(lam[j]), rad[j])) for j in range(n)],
                        "upper": [float(add_up(out_hi(lam[j]), rad[j])) for j in range(n)]})
        rec = {"parameter": "c", "couplings": [c0, c1], "B": out_hi(B), "affinity_defects": [out_hi(d)], "points": pts,
               "record_sha256": "synthetic"}
        cs = [c0, c1, float(np.nextafter(c0, np.inf)), float(np.nextafter(c1, -np.inf))] + [float(x) for x in rng.uniform(c0, c1, size=4)]
        for c in cs:
            Fc = H0m + mpf(c - c0) * H1m
            lam, _ = eig_herm(Fc)
            ok = True
            for j in range(n):
                tube = sweep.tube_at(rec, j, c)["interval"]
                ok = ok and (mpf(tube[0]) <= lam[j] <= mpf(tube[1]))
            T1.add(ok, kind in (1, 2, 3) or c in cs[:4], f"P1 n={n} kind={kind} c={c!r} d={d_true} tube/lam mismatch")
            # Proposition 2: counts at random thresholds (exact eigenvalues, B exact)
            eps = B * abs(c - c0)
            for sigma in rng.uniform(float(min(lam0)) - 1, float(max(lam0)) + 1, size=3):
                s_ = mpf(sigma)
                left = sum(1 for x in lam if x < s_ - eps)
                mid = sum(1 for x in lam0 if x < s_)
                right = sum(1 for x in lam if x <= s_ + eps)
                T2.add(left <= mid <= right, kind == 1, f"P2 n={n} kind={kind} c={c!r} sigma={sigma} {left},{mid},{right}")


# --------------------------------------------------------------------------------------- Part B: Lemma 3 / crossings
def scalar_scan(cs, B, defects, lows, ups, param="c"):
    pts = [{"index": i, param: c, "lower": [lo], "upper": [hi], "rank_certified": True, "next_eigenvalue_lower_bound": None}
           for i, (c, lo, hi) in enumerate(zip(cs, lows, ups))]
    return {"parameter": param, "couplings": cs, "B": B, "affinity_defects": defects, "points": pts, "record_sha256": "synthetic"}


def random_scalar_family(rng, grid):
    """Piecewise affine scalar family: on [c_k, c_{k+1}] F_k(c) = v_k + s (c - c_k); the next grid value v_{k+1}
    deviates from F_k(c_{k+1}) by a random defect. Returns (values v_k, slope s, defects d_k, exact Fractions)."""
    s = Fraction(rng.choice([-2, -1, 0, 1, 2]).item()) * Fraction(int(rng.integers(1, 5)), 4)
    v = [Fraction(int(rng.integers(-40, 40)), 16)]
    defects = []
    for a, b in zip(grid, grid[1:]):
        dev = Fraction(int(rng.integers(-8, 9)), 64) if rng.random() < 0.6 else Fraction(0)
        v.append(v[-1] + s * (b - a) + dev)
        defects.append(abs(dev))
    return v, s, defects


def targeted_family(rng, grid, nseg_samples):
    """Branches with opposite slopes +-s whose difference crosses zero inside the last sample interval before the
    first interior grid point, and a defect at that grid point large enough to hide the crossing from a criterion
    that reads the grid-point margin from the next segment (the pre-fix code). Returns (va, sa, da, vb, sb, db)."""
    s = Fraction(int(rng.integers(1, 5)), 2)
    D = (grid[1] - grid[0]) / nseg_samples
    eps = D * s * Fraction(int(rng.integers(1, 20)), 100)                  # 0 < eps < 0.2 B D
    c_i = grid[1] - D
    # g(c) = a(c) - b(c) = g0 + 2 s (c - c_i) with g(c_i) = -(s D + eps): zero at c_i + (sD + eps)/(2s) < c_i + D
    g0 = -(s * D + eps)
    vb0 = Fraction(0)
    va0 = g0 + vb0 - 2 * s * (c_i - grid[0])                               # a(c_i) - b(c_i) = g0
    # a: slope +s, b: slope -s; defect at grid[1] on scan a pulls H_a(grid[1]) down by d > 2 s D - eps, which makes
    # the grid-point margin read from the next segment, (s D - eps - d), pass the test m + B D < 0
    d = 2 * s * D - eps + D * s * Fraction(int(rng.integers(1, 11)), 10)
    va = [va0, va0 + s * (grid[1] - grid[0]) - d]
    vb = [vb0, vb0 - s * (grid[1] - grid[0])]
    for a_, b_ in zip(grid[1:], grid[2:]):
        va.append(va[-1] + s * (b_ - a_)); vb.append(vb[-1] - s * (b_ - a_))
    da = [d] + [Fraction(0)] * (len(grid) - 2)
    db = [Fraction(0)] * (len(grid) - 1)
    return va, s, da, vb, -s, db


def part_B(rng, cases, T):
    for i in range(cases):
        grid_a = [Fraction(k, 2) for k in range(0, int(rng.integers(2, 5)) * 2 + 1, 2)]       # 0, 1, 2, ...
        grid_b = grid_a if i % 3 else [Fraction(k, 4) for k in range(0, len(grid_a) * 4 - 3, 4 // 2 * 1)][: len(grid_a) * 2 - 1]
        grid_b = [g for g in grid_b if g <= grid_a[-1]]
        nss = int(rng.choice([8, 40, 200]))
        if i % 2:
            grid_b = grid_a
            va, sa, da, vb, sb, db = targeted_family(rng, grid_a, nss)
            ra_rad = [Fraction(0)] * len(grid_a); rb_rad = ra_rad
        else:
            va, sa, da = random_scalar_family(rng, grid_a)
            vb, sb, db = random_scalar_family(rng, grid_b)
            ra_rad = [Fraction(int(rng.integers(0, 3)), 128) for _ in grid_a]
            rb_rad = [Fraction(int(rng.integers(0, 3)), 128) for _ in grid_b]
        float(max(abs(sa), abs(sb)))
        ra = scalar_scan([float(g) for g in grid_a], float(abs(sa)), [float(d) for d in da],
                         [float(v - r) for v, r in zip(va, ra_rad)], [float(v + r) for v, r in zip(va, ra_rad)])
        rb = scalar_scan([float(g) for g in grid_b], float(abs(sb)), [float(d) for d in db],
                         [float(v - r) for v, r in zip(vb, rb_rad)], [float(v + r) for v, r in zip(vb, rb_rad)])
        out = sweep.certified_crossing_interval(ra, 0, rb, 0, samples_per_segment=nss)

        def branch(v, s, grid, c):
            k = max(j for j in range(len(grid) - 1) if grid[j] <= c) if c < grid[-1] else len(grid) - 2
            return v[k] + s * (c - grid[k])

        viol = 0; checked = 0
        for run in out["runs"]:
            if run["relation"] == "not separated":
                continue
            lo, hi = Fraction(run["interval"][0]), Fraction(run["interval"][1])
            # check inside every segment piece of the run (the family changes at grid points)
            cuts = sorted({lo, hi} | {g for g in grid_a + grid_b if lo < g < hi})
            for p, q in zip(cuts, cuts[1:]):
                for t in range(65):
                    c = p + (q - p) * Fraction(t, 64)
                    # the segment family on [p, q]: evaluate with the segment that contains the open piece
                    cc = c if c < q else q - (q - p) / 10 ** 9      # at q itself the next family starts
                    a_ = branch(va, sa, grid_a, cc) + 0 * c; b_ = branch(vb, sb, grid_b, cc)
                    # value at c itself with the segment-of-[p,q] family:
                    ka = max(j for j in range(len(grid_a) - 1) if grid_a[j] <= cc)
                    kb = max(j for j in range(len(grid_b) - 1) if grid_b[j] <= cc)
                    a_ = va[ka] + sa * (c - grid_a[ka]); b_ = vb[kb] + sb * (c - grid_b[kb])
                    checked += 1
                    if run["relation"] == "<" and not a_ < b_:
                        viol += 1
                    if run["relation"] == ">" and not a_ > b_:
                        viol += 1
        T.add(viol == 0, i % 2 == 1, f"L3 case {i} ({'targeted' if i % 2 else 'random'}): {viol} of {checked} checked points contradict a certified run; defects a={[float(d) for d in da]} b={[float(d) for d in db]}")


# ----------------------------------------------------------------------------------------------- Part C: Lemma 4
def part_C(rng, T):
    spaces = [LocalSpace.spin_space("1/2", conv) for conv in ("spin", "pauli", "pauli_pm1")]
    spaces += [LocalSpace.spin_space(s) for s in ("1", "3/2", "2", "5/2", "7/2")]
    spaces += [LocalSpace.fermion_space(j) for j in ("higher", "lower")]
    spaces += [LocalSpace.boson_space(n) for n in (1, 2, 3, 6, 10)]
    for L in spaces:
        fake = types.SimpleNamespace(local=L)
        for let in L.letters:
            M = to_mp(L.matrix(let))
            T.add(spec_norm(M) <= mpf(sweep.term_norm_bound(fake, let)), True, f"L4 {L.kind} {L.spin}{L.convention} n_max={L.n_max} letter {let}")
        for _ in range(6):
            m = int(rng.integers(1, 4))
            op = "".join(rng.choice(list(L.letters), size=m))
            K = mp.matrix([[1]])
            for let in op:
                K = mp_kron(K, to_mp(L.matrix(let)))
                if rng.random() < 0.5 and K.rows * L.q <= 81:          # a sign string on an intermediate site
                    sgn = mp.diag([mp.mpf(1) if rng.random() < 0.5 else mp.mpf(-1) for _ in range(L.q)])
                    K = mp_kron(K, sgn)
            if K.rows > 81:                                             # mp eigh is too slow beyond this
                continue
            T.add(spec_norm(K) <= mpf(sweep.term_norm_bound(fake, op)), False, f"L4 {L.kind} string {op}")


def mp_kron(A, B):
    ra, ca, rb, cb = A.rows, A.cols, B.rows, B.cols
    K = mp.matrix(ra * rb, ca * cb)
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    K[i * rb + k, j * cb + l] = A[i, j] * B[k, l]
    return K


# ------------------------------------------------------------------------------------------------- Part D: end to end
def part_D(rng, T, T_neg):
    FAST = ce.Options(eigenspaces="none", degeneracy=False)
    for N, grid, k in ((3, [0.3, 0.8, 1.3], 2), (4, [0.5, 1.0, 1.5], 1)):
        spec_fn = lambda h, N=N: ce.tfim_chain(N, J=1.0, h=h)    # noqa: E731
        rec = sweep.certified_sweep(spec_fn, grid, k=k, options=FAST, parameter_name="h")
        H1 = build_bitwise(sweep.derivative_spec(spec_fn, grid[0], grid[1])).toarray()
        Hs = [build_bitwise(spec_fn(c)).toarray() for c in grid]       # dyadic entries: exact spec operators
        for _ in range(40):
            seg = int(rng.integers(0, len(grid) - 1))
            c = float(rng.uniform(grid[seg], grid[seg + 1]))
            Fc = to_mp(Hs[seg]) + (mpf(c) - mpf(grid[seg])) * to_mp(H1)
            lam, _ = eig_herm(Fc)
            ok = True
            for lvl in range(k):
                tube = sweep.tube_at(rec, lvl, c)["interval"]
                ok = ok and mpf(tube[0]) <= lam[lvl] <= mpf(tube[1])
            nlb = sweep.next_level_lower_bound_at(rec, c)
            if nlb is not None:
                ok = ok and mpf(nlb) <= lam[k]
            T.add(ok, N == 3, f"E2E TFIM N={N} h={c} tubes vs mp eigenvalues; defects={rec['affinity_defects']}")
        # negative control: B / 20
        bad = dict(rec); bad["B"] = rec["B"] / 20
        hits = 0
        for _ in range(30):
            seg = int(rng.integers(0, len(grid) - 1)); c = float(rng.uniform(grid[seg], grid[seg + 1]))
            Fc = to_mp(Hs[seg]) + (mpf(c) - mpf(grid[seg])) * to_mp(H1); lam, _ = eig_herm(Fc)
            try:
                tube = sweep.tube_at(bad, 0, c)["interval"]
                if not (mpf(tube[0]) <= lam[0] <= mpf(tube[1])):
                    hits += 1
            except ValueError:
                hits += 1
        T_neg.add(hits > 0, True, f"negative control B/20 TFIM N={N}: {hits} of 30 detected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--seed", type=int, default=71)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T1 = Tally("Proposition 1: tube_at (exact enclosures, exact B and d_k) contains lambda_j(F_k(c)) (mp)")
    T2 = Tally("Proposition 2: count form at random thresholds (mp)")
    T3 = Tally("Lemma 3 / certified_crossing_interval: certified '<' / '>' runs agree with the exact segment branches (rationals)")
    T4 = Tally("Lemma 4: ||o_1 x ... x o_m||_2 <= term_norm_bound for all letters and random strings (mp)")
    T5 = Tally("certified_sweep end to end (TFIM): tubes and gap line vs mp eigenvalues of the exact F_k(c)")
    T6 = Tally("negative control: B/20 detected")
    part_A(rng, a.cases, T1, T2)
    part_B(rng, a.cases, T3)
    part_C(rng, T4)
    part_D(rng, T5, T6)
    report([T1, T2, T3, T4, T5, T6])


if __name__ == "__main__":
    main()
