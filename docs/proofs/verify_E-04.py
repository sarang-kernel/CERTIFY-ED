"""Computational verification of docs/proofs/E-04_parameter_intervals.md (certify_ed.sweep).

Parts:
  A  Weyl per sorted index: |lambda_j(H0 + c H1) - lambda_j(H0)| <= |c| B with B the exact ||H1||_2 and with the
     row-sum bound of a rational H1 (exact rationals), eigenvalues at 50 mpmath digits
  B  counts below a shifting threshold: #{lambda(c) < s - B|c|} <= #{lambda(0) < s} <= #{lambda(c) <= s + B|c|}
  C  Lemma 3 (sampling criterion): on exact two-grid-point tubes (intersections with a kink inside the segment),
     max(m_i, m_{i+1}) + B D < 0 implies m(c) < 0 on the whole sample interval (m evaluated at 1001 points including
     both ends); adversarial: the kink at the midpoint with the criterion holding by a relative 1e-9, and failing
  D  term norm bound: ||o_1 ... o_m||_2 (exact, mpmath) <= product of the local infinity norms, all local spaces
  E  end to end: TFIM N = 4 and XXZ N = 6 (S^z = 0 block) scans; the tubes contain the dense eigenvalues at 100
     random intermediate couplings each (binary64 LAPACK eigenvalues with a 1e-12 allowance)
  F  negative controls: a wrong B (B/20) produces violations or inconsistent tubes; a wrong declared H1 is
     reported through the affinity defect (exactly_affine False)

Run:  python docs/proofs/verify_E-04.py [seed]
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import certify_ed as ce  # noqa: E402
from certify_ed import sweep  # noqa: E402
from certify_ed.localops import LocalSpace  # noqa: E402

mp.mp.dps = 50
MARGIN = mp.mpf(10) ** -40
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261030
rng = np.random.default_rng(SEED)


def to_mp(F):
    F = np.asarray(F)
    M = mp.matrix(F.shape[0], F.shape[1])
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            v = F[i, j]
            M[i, j] = mp.mpc(float(np.real(v)), float(np.imag(v))) if np.iscomplexobj(F) else mp.mpf(float(v))
    return M


def eigs(M):
    cplx = any(isinstance(M[i, j], mp.mpc) and M[i, j].imag != 0 for i in range(M.rows) for j in range(M.cols))
    if cplx:
        return sorted(mp.eigh(M, eigvals_only=True))
    R = mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            R[i, j] = mp.re(M[i, j])
    return sorted(mp.eigsy(R, eigvals_only=True))


def herm(n, cplx, scale=1.0):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    A = (A + A.conj().T) / 2 * scale
    return A


# ---------------------------------------------------------------------------------------------- A, B: Weyl and counts
def part_AB(n_random=80):
    cases = viol = 0
    ccases = cviol = 0
    adversarial = 0

    def check(H0, H1, c, label=None):
        nonlocal cases, viol, ccases, cviol
        H0m, H1m = to_mp(H0), to_mp(H1)
        e0, ec = eigs(H0m), eigs(H0m + c * H1m)
        B_exact = max(abs(x) for x in eigs(H1m))
        # row-sum bound in exact rationals (H1 entries are binary64)
        rows = [sum(abs(Fraction(float(np.real(H1[i, j])))) + abs(Fraction(float(np.imag(H1[i, j])))) for j in range(H1.shape[1]))
                for i in range(H1.shape[0])]
        B_row = mp.mpf(max(rows).numerator) / max(rows).denominator          # >= ||H1||_1 >= ||H1||_2 (Hermitian: ||.||_inf)
        cases += 1
        dev = max(abs(a - b) for a, b in zip(e0, ec))
        if dev > abs(c) * B_exact + MARGIN or dev > abs(c) * B_row + MARGIN:
            viol += 1
            print("  VIOLATION A", label, mp.nstr(dev, 12), mp.nstr(abs(c) * B_exact, 12))
        # counts below a shifting threshold
        for _ in range(3):
            s = mp.mpf(float(rng.uniform(float(min(e0)) - 1, float(max(e0)) + 1)))
            n_mid = sum(1 for x in e0 if x < s)
            n_lo = sum(1 for x in ec if x < s - abs(c) * B_exact)
            n_hi = sum(1 for x in ec if x <= s + abs(c) * B_exact)
            ccases += 1
            if not (n_lo <= n_mid <= n_hi):
                cviol += 1
                print("  VIOLATION B", label, n_lo, n_mid, n_hi)

    for _ in range(n_random):
        n = int(rng.integers(2, 9))
        cplx = bool(rng.integers(0, 2))
        check(herm(n, cplx), herm(n, cplx, 10.0 ** rng.uniform(-3, 3)), float(rng.uniform(-5, 5)))
    # adversarial: H1 = I (equality in Weyl), rank-one H1, degenerate H0, a coupling where eigenvalues of H0 + c H1
    # cross (sorted labels swap), huge c
    n = 5
    check(herm(n, False), np.eye(n), 2.5, "H1 = I")
    v = rng.standard_normal((n, 1))
    check(herm(n, False), v @ v.T, -1.3, "rank one")
    check(np.diag([1.0, 1.0, 1.0, 2.0, 2.0]), herm(n, True), 0.7, "degenerate H0")
    H0 = np.diag([0.0, 1.0])
    H1 = np.array([[1.0, 0.0], [0.0, -1.0]])
    for c in (0.4, 0.5, 0.6, 50.0):
        check(H0, H1, c, f"crossing c={c}")
    adversarial += 7
    print(f"A  Weyl per sorted index: {cases} cases ({cases - adversarial} random + {adversarial} adversarial), {viol} violations")
    print(f"B  counts below a shifting threshold: {ccases} cases, {cviol} violations")
    return cases + ccases, viol + cviol


# ---------------------------------------------------------------------------------------------- C: sampling criterion
def part_C(n_random=300):
    """Tubes from two grid points cL < cR (as tube_at forms them: intersection of the two one-sided tubes, so the
    ends have a kink inside the segment); the criterion is evaluated on sample intervals [c_i, c_{i+1}] and, where it
    holds, m(c) < 0 is checked at 1001 points of the sample interval (both ends included)."""
    cases = viol = 0
    adversarial = 0
    checked = 0

    def tube(lo, hi, B, cL, cR, c):
        """Intersection of the tubes of the two grid points: lo = (lo_L, lo_R), hi = (hi_L, hi_R)."""
        return (max(lo[0] - B * abs(c - cL), lo[1] - B * abs(c - cR)), min(hi[0] + B * abs(c - cL), hi[1] + B * abs(c - cR)))

    def check(ta, tb, B, cL, cR, c_i, c_ip1, label=None):
        nonlocal cases, viol, checked

        def m(c):
            return tube(*ta, B, cL, cR, c)[1] - tube(*tb, B, cL, cR, c)[0]

        D = c_ip1 - c_i
        crit = max(m(c_i), m(c_ip1)) + B * D < 0
        cases += 1
        if crit:
            checked += 1
            for t in np.linspace(c_i, c_ip1, 1001):
                if m(float(t)) >= 0:
                    viol += 1
                    print("  VIOLATION C", label)
                    break
        return crit

    for _ in range(n_random):
        B = 10.0 ** rng.uniform(-2, 1)
        cL, cR = 0.0, 10.0 ** rng.uniform(-2, 0.5)
        # grid-point enclosures of a and b (a below b at both grid points in most draws)
        loA = rng.uniform(-3, 3, size=2)
        hiA = loA + rng.uniform(0, 0.3, size=2)
        loB = hiA + rng.uniform(-0.5, 2.0, size=2)
        hiB = loB + rng.uniform(0, 0.3, size=2)
        # a sample interval inside the segment (random width, may contain the kink of the intersected tubes)
        c_i = rng.uniform(cL, cR)
        c_ip1 = min(cR, c_i + rng.uniform(0, cR - cL))
        if c_ip1 <= c_i:
            continue
        check(((loA[0], loA[1]), (hiA[0], hiA[1])), ((loB[0], loB[1]), (hiB[0], hiB[1])), B, cL, cR, c_i, c_ip1)
    # adversarial: (i) the sample interval is the whole segment, so the kink of both tubes (at the midpoint for equal
    # enclosures) lies inside; the margin at the ends is m0 = -(gap) and the criterion holds iff gap > B (cR - cL):
    # take gap = B D (1 + 1e-9), the criterion holds by a relative 1e-9 and m at the kink equals -gap + B D < 0 by 1e-9 B D;
    # (ii) the same with gap = B D (1 - 1e-9): the criterion must then fail (no check runs; counted separately)
    held = failed = 0
    for _ in range(50):
        B = 10.0 ** rng.uniform(-1, 1)
        D = 10.0 ** rng.uniform(-2, 0)
        gap = B * D * (1 + 1e-9)
        ta = ((0.0, 0.0), (0.0, 0.0))
        tb = ((gap, gap), (gap, gap))
        held += int(check(ta, tb, B, 0.0, D, 0.0, D, "kink, near-equality"))
        adversarial += 1
        gap2 = B * D * (1 - 1e-9)
        failed += int(not check(ta, ((gap2, gap2), (gap2, gap2)), B, 0.0, D, 0.0, D, "kink, criterion fails"))
        adversarial += 1
    print(f"C  Lemma 3 (sampling criterion): {cases} cases ({cases - adversarial} random + {adversarial} adversarial), criterion held in "
          f"{checked} of them (each checked at 1001 points), {viol} violations; near-equality kink cases: criterion held {held}/50 "
          f"(expected 50), failed as expected {failed}/50 (expected 50)")
    return cases, viol + (50 - held) + (50 - failed)


# ---------------------------------------------------------------------------------------------- D: term norm bound
def part_D():
    cases = viol = 0
    spaces = [LocalSpace.spin_space("1/2", "spin"), LocalSpace.spin_space("1/2", "pauli"), LocalSpace.spin_space("1", "spin"),
              LocalSpace.spin_space("3/2", "spin"), LocalSpace.fermion_space(), LocalSpace.boson_space(3)]
    for loc in spaces:
        letters = [x for x in loc.letters if x != "I"]
        for m_ in (1, 2, 3):
            for _ in range(6):
                op = "".join(rng.choice(letters, size=m_))
                spec = ce.ModelSpec(m_, loc, [ce.spec.Term(op, [(1.0, tuple(range(m_)))])])
                M = np.eye(1, dtype=complex)
                for letter in op:
                    M = np.kron(M, loc.matrix(letter))
                exact = max(mp.svd_c(to_mp(M), compute_uv=False))
                bound = sweep.term_norm_bound(spec, op)
                cases += 1
                if exact > bound + MARGIN:
                    viol += 1
                    print("  VIOLATION D", loc.kind, op, mp.nstr(exact, 12), bound)
    print(f"D  term norm bound (all local spaces, strings of 1-3 letters): {cases} cases, {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- E, F: end to end
def part_EF():
    FAST = ce.Options(eigenspaces="none", degeneracy=False)
    cases = viol = 0
    spec_fn = lambda h: ce.tfim_chain(4, J=1.0, h=h)  # noqa: E731
    rec = sweep.certified_sweep(spec_fn, [0.3, 0.8, 1.3], k=2, options=FAST, parameter_name="h")
    for c in rng.uniform(0.3, 1.3, 100):
        w = la.eigvalsh(ce.build_bitwise(spec_fn(float(c))).toarray())
        for j in range(2):
            lo, hi = sweep.tube_at(rec, j, float(c))["interval"]
            cases += 1
            if not (lo - 1e-12 <= w[j] <= hi + 1e-12):
                viol += 1
    xxz = lambda D: ce.heisenberg_chain(6, Delta=D)  # noqa: E731
    stf = lambda spec: ce.magnetization_state_set(spec, 0)  # noqa: E731
    rec2 = sweep.certified_sweep(xxz, [0.5, 1.0, 2.0], k=1, options=FAST, states_fn=stf, parameter_name="Delta")
    for c in rng.uniform(0.5, 2.0, 100):
        w = la.eigvalsh(ce.build_bitwise(xxz(float(c)), stf(xxz(float(c)))).toarray())
        lo, hi = sweep.tube_at(rec2, 0, float(c))["interval"]
        cases += 1
        if not (lo - 1e-12 <= w[0] <= hi + 1e-12):
            viol += 1
    print(f"E  end to end (TFIM N=4 k=2, XXZ N=6 S^z=0 k=1; exactly affine: {rec['exactly_affine']}, {rec2['exactly_affine']}): "
          f"{cases} tube evaluations at random couplings, {viol} violations")
    # F: wrong B
    bad = dict(rec)
    bad["B"] = rec["B"] / 20
    detected = 0
    for c in rng.uniform(0.35, 0.75, 50):
        w = la.eigvalsh(ce.build_bitwise(spec_fn(float(c))).toarray())
        try:
            lo, hi = sweep.tube_at(bad, 0, float(c))["interval"]
        except ValueError:
            detected += 1
            continue
        detected += int(not (lo <= w[0] <= hi))
    base = spec_fn(0.3)
    h1_zero = ce.ModelSpec(base.n_sites, base.local, [ce.spec.Term(t.op, [(0.0, s) for _, s in t.couplings]) for t in base.terms])
    rec3 = sweep.certified_sweep(spec_fn, [0.3, 0.8], k=1, options=FAST, h1_spec=h1_zero, parameter_name="h")
    print(f"F  negative controls: wrong B (B/20): {detected} of 50 random couplings violated or inconsistent (expected > 0); "
          f"wrong declared H1 = 0: exactly_affine = {rec3['exactly_affine']}, affinity defect = {rec3['affinity_defects'][0]:.4g} "
          f"(||H(0.8) - H(0.3)||_2 = 2)")
    return cases, viol, detected, rec3["exactly_affine"]


if __name__ == "__main__":
    print(f"verify_E-04: seed {SEED}, mpmath dps {mp.mp.dps}")
    tot_c = tot_v = 0
    for part in (part_AB, part_C, part_D):
        c, v = part()
        tot_c += c
        tot_v += v
    c, v, det, aff = part_EF()
    tot_c += c
    tot_v += v
    print(f"TOTAL: {tot_c} cases, {tot_v} violations; negative controls: wrong B detected at {det}/50 points, wrong H1 flagged: {not aff}")
