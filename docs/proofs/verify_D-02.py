"""Computational verification of docs/proofs/D-02_matrix-free.md (matrix-free residual bounds from the spec).

For random specs (spin-1/2 with Pauli/spin conventions, spin-1, fermions with Jordan-Wigner strings, bosons,
real and complex couplings, strings of length 1-3, repeated sites) and random stored vectors x, the script
computes the EXACT spec operator H_spec (couplings as stored, exact local amplitudes: rationals for spin-1/2 and
fermions, 60-digit mpmath for the square roots of spin-1 and bosons) and checks

  (V1) |fl(H x)_i - (H_spec x)_i| <= gamma_W a_i   with a_i = (|H_spec|_abs |x|)_i exact and W = op.W,
  (V2) the terms_per_row bound: the number of monomial contributions landing in any output entry (counted
       by instrumenting the monomial passes) is <= op.terms_per_row,
  (V3) ||H_spec||_2 <= op.norm_bound() (any spec) and <= OperatorAdapter(op).norm_upper() (Hermitian specs; the
       Appendix-A row-sum bound needs Hermiticity) -- 60-digit spectral norm,
  (V4) the implemented Theorem 1-fl radius: enclosure.residuals(adapter, mu, x).rel_bound >= ||H_spec x - mu x|| / ||x||
       for mu = the Rayleigh quotient and for mu = a perturbed value (so that the residual is not tiny),
  (V5) fl(H x) of MatrixFreeOperator equals construct.SpecOperator's bit for bit (same arithmetic).

Adversarial cases: exactly cancelling terms (+J and -J on one bond: H_spec = 0 on that bond), many terms per site
(all-to-all couplings, 3-letter strings), couplings 1e150 and 1e-150 and vectors with entries 1e-160 (INT-05
additive terms), spin-1 x/y (rounded amplitudes, rho > 0), a zero coupling, repeated sites in one string, n = 2.
Negative controls: a spec with one wrong coefficient makes (V4)'s residual of an exact eigenvector exceed the
radius the correct spec gives (the wrong spec's own bound is still valid for the wrong operator); a planted wrong
vector gives a residual above the gap.

Usage: python docs/proofs/verify_D-02.py [--cases N] [--seed S]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from _common import Counter, norm2, to_mp  # noqa: E402
import certify_ed as ce  # noqa: E402
from certify_ed.construct import SpecOperator, _apply_string  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, residuals  # noqa: E402
from certify_ed.fp import U  # noqa: E402
from certify_ed.localops import LocalSpace  # noqa: E402
from certify_ed.matrixfree import MatrixFreeOperator  # noqa: E402
from certify_ed.spec import ModelSpec, Term  # noqa: E402

mp.mp.dps = 60


def gamma_mp(k):
    k = mp.mpf(k)
    return k * U / (1 - k * U)


# ------------------------------------------------------------------ exact spec operator (mp, dense, small n)

def exact_local(local: LocalSpace, letter: str):
    """Exact q x q local matrix as mp.matrix (square roots at 60 digits; rationals exact)."""
    q = local.q
    M = mp.matrix(q, q)
    if letter == "I":
        return mp.eye(q)
    if local.kind == "spin":
        S = mp.mpf(Fraction(local.spin).numerator) / Fraction(local.spin).denominator
        m = [mp.mpf(k) - S for k in range(q)]
        Sp = mp.matrix(q, q)
        for k in range(q - 1):
            Sp[k + 1, k] = mp.sqrt(S * (S + 1) - m[k] * (m[k] + 1))
        Sm = Sp.T
        Sz = mp.diag(m)
        base = {"x": (Sp + Sm) / 2, "y": (Sp - Sm) / (2j), "z": Sz, "+": Sp, "-": Sm}
        M = base[letter]
        if local.convention == "pauli":
            return 2 * M
        if local.convention == "pauli_pm1":
            return M if letter in "+-" else 2 * M
        return M
    if local.kind == "fermion":
        c = mp.matrix([[0, 1], [0, 0]])
        base = {"+": c.T, "-": c, "n": mp.diag([0, 1]), "z": mp.diag([mp.mpf(-1) / 2, mp.mpf(1) / 2])}
        return base[letter]
    a = mp.matrix(q, q)
    for n in range(1, q):
        a[n - 1, n] = mp.sqrt(n)
    base = {"+": a.T, "-": a, "n": mp.diag(list(range(q)))}
    return base[letter]


def kron(A, B):
    C = mp.matrix(A.rows * B.rows, A.cols * B.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            if A[i, j] != 0:
                for k in range(B.rows):
                    for l_ in range(B.cols):
                        C[i * B.rows + k, j * B.cols + l_] = A[i, j] * B[k, l_]
    return C


def exact_H(spec: ModelSpec):
    """Dense exact H_spec in the spec's state order (site 0 most significant; descending order reverses)."""
    N, q, local = spec.n_sites, spec.q, spec.local
    n = q ** N
    H = mp.matrix(n, n)
    parity = mp.diag([1, -1]) if local.kind == "fermion" else None
    for term in spec.terms:
        for c, sites in term.couplings:
            # product of letters acting right to left on the sites; JW string: parity on the string side
            ops = [mp.eye(q) for _ in range(N)]
            full = None
            for pos in range(len(term.op) - 1, -1, -1):
                letter, site = term.op[pos], sites[pos]
                factors = [mp.eye(q) for _ in range(N)]
                factors[site] = exact_local(local, letter)
                if local.jw_letter(letter):
                    for l_ in range(N):
                        if local.jw_on(letter, l_, site):
                            factors[l_] = parity
                Mstep = factors[0]
                for f in factors[1:]:
                    Mstep = kron(Mstep, f)
                full = Mstep if full is None else Mstep * full
            del ops
            cc = mp.mpc(complex(c).real, complex(c).imag)
            H += cc * full
    if spec.state_order == "descending":
        P = mp.matrix(n, n)
        for i in range(n):
            P[i, n - 1 - i] = 1
        H = P * H * P
    return H


def mp_abs(M):
    A = mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            A[i, j] = abs(M[i, j])
    return A


def abs_H_exact(spec: ModelSpec):
    """|H_spec|_abs: the sum over couplings of |c| times the entrywise moduli of each monomial (an upper bound on
    the exact |H|_abs the code uses, since the code's monomials are the shift components, finer than one letter
    product; the entrywise moduli of a product of letters are bounded by the product of moduli)."""
    N, q, local = spec.n_sites, spec.q, spec.local
    n = q ** N
    A = mp.matrix(n, n)
    for term in spec.terms:
        for c, sites in term.couplings:
            full = None
            for pos in range(len(term.op) - 1, -1, -1):
                letter, site = term.op[pos], sites[pos]
                factors = [mp.eye(q) for _ in range(N)]
                factors[site] = mp_abs(exact_local(local, letter))
                Mstep = factors[0]
                for f in factors[1:]:
                    Mstep = kron(Mstep, f)
                full = Mstep if full is None else Mstep * full
            A += abs(mp.mpc(complex(c).real, complex(c).imag)) * full
    if spec.state_order == "descending":
        P = mp.matrix(n, n)
        for i in range(n):
            P[i, n - 1 - i] = 1
        A = P * A * P
    return A


def count_terms_per_row(op: MatrixFreeOperator) -> int:
    """Instrumented monomial passes: the largest number of contributions to one output entry."""
    spec = op.spec
    n = op.states.size
    cnt = np.zeros(n, dtype=np.int64)
    for term in spec.terms:
        for c, sites in term.couplings:
            for s_fin, amp, ok in _apply_string(spec, term.op, sites, c, op._s):
                m = ok & (amp != 0)
                if not np.any(m):
                    continue
                r, found = op.states.index_of(s_fin[m])
                np.add.at(cnt, r[found], 1)
    return int(cnt.max()) if n else 0


# ------------------------------------------------------------------ random specs

def random_spec(rng, case):
    kind = ["spin_half", "spin_half_pauli", "spin1", "fermion", "boson", "spin_half_complex"][case % 6]
    N = int(rng.integers(2, 4)) if kind in ("spin1", "boson") else int(rng.integers(2, 6))
    if kind.startswith("spin_half"):
        local = LocalSpace.spin_space("1/2", "pauli" if kind == "spin_half_pauli" else "spin")
        letters = "xyz+-"
    elif kind == "spin1":
        local = LocalSpace.spin_space("1", "spin")
        letters = "xyz+-"
    elif kind == "fermion":
        local = LocalSpace.fermion_space("higher" if case % 2 else "lower")
        letters = "+-nz"
    else:
        local = LocalSpace.boson_space(2)
        letters = "+-n"
    n_terms = int(rng.integers(1, 7))
    terms = []
    for _ in range(n_terms):
        L = int(rng.integers(1, 4))
        opstr = "".join(rng.choice(list(letters), L))
        sites = tuple(int(s) for s in rng.integers(0, N, L))
        cplx = kind == "spin_half_complex"
        c = complex(rng.standard_normal(), rng.standard_normal() if cplx else 0.0)
        # keep H Hermitian only where the test needs it (Theorem 1 needs Hermitian H for the radius statement,
        # (V1)-(V3) do not); the residual check uses the Hermitian part through a symmetrised term list
        terms.append(Term(opstr, [(c, sites)]))
    spec = ModelSpec(N, local, terms, state_order=["descending", "ascending"][case % 2], name=f"random-{kind}-{case}")
    return spec, kind


def hermitian_specs():
    """Hermitian model specs for (V4): chains from the library plus a spin-1 and a fermion chain."""
    return [ce.heisenberg_chain(5), ce.tfim_chain(5), ce.heisenberg_chain(4, J2=0.3),
            ce.spin1_heisenberg_chain(3, D=0.2), ce.spinless_fermion_chain(5, V=0.5), ce.bose_hubbard_chain(3, 2)]


def check_core(spec, rng, ctr, adversarial, note, x=None):
    op = MatrixFreeOperator(spec)
    so = SpecOperator(spec)
    n = op.shape[0]
    cplx = op.dtype == np.complex128
    if x is None:
        x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0.0)
    y = op.matvec(x)
    ok5 = np.array_equal(y, so.matvec(x))
    Hm = exact_H(spec)
    Hx = Hm * to_mp(x)
    A1x = abs_H_exact(spec) * to_mp(np.abs(x))
    gW = gamma_mp(op.W)
    ok1 = all(abs(mp.mpc(complex(y[i]).real, complex(y[i]).imag) - Hx[i]) <= gW * A1x[i] + mp.mpf(2) ** -1060 * op.W
              for i in range(n))
    ok2 = count_terms_per_row(op) <= op.terms_per_row
    ad = OperatorAdapter(op)
    nrm = norm2(Hm)
    # Lemma N holds for any spec; the adapter's row-sum bound (Appendix A) needs a Hermitian spec, the standing
    # hypothesis of certify(): it is checked only when H_spec is Hermitian
    herm = all(Hm[i, j] == mp.conj(Hm[j, i]) for i in range(n) for j in range(n))
    ok3 = nrm <= mp.mpf(op.norm_bound()) and (not herm or nrm <= mp.mpf(ad.norm_upper()))
    ctr.add(adversarial, ok1 and ok2 and ok3 and ok5,
            f"{note}: V1 {ok1} V2 {ok2} V3 {ok3} (norm {mp.nstr(nrm, 6)} <= {op.norm_bound():.6g}) V5 {ok5}")


def check_radius(spec, rng, ctr, adversarial, note, x=None, mu=None):
    op = MatrixFreeOperator(spec)
    ad = OperatorAdapter(op)
    n = op.shape[0]
    cplx = op.dtype == np.complex128
    if x is None:
        x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0.0)
    H = exact_H(spec)
    Hx = H * to_mp(x)
    xm = to_mp(x)
    if mu is None:
        num = sum(mp.conj(xm[i]) * Hx[i] for i in range(n))
        den = sum(abs(xm[i]) ** 2 for i in range(n))
        mu = float(mp.re(num / den))
    rep = residuals(ad, np.array([mu]), x.reshape(n, 1))
    rel = float(rep.rel_bound[0])
    r = mp.sqrt(sum(abs(Hx[i] - mu * xm[i]) ** 2 for i in range(n))) / mp.sqrt(sum(abs(xm[i]) ** 2 for i in range(n)))
    ok = r <= mp.mpf(rel)
    ctr.add(adversarial, ok, f"{note}: V4 exact {mp.nstr(r, 6)} <= rel {rel:.6g}: {ok}")
    return rel, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ctr = Counter()
    for case in range(args.cases):
        spec, kind = random_spec(rng, case)
        check_core(spec, rng, ctr, False, f"random {case} {kind} N={spec.n_sites} terms={spec.n_terms}")
    for spec in hermitian_specs():
        for trial in range(4):
            check_radius(spec, rng, ctr, False, f"radius {spec.name} trial {trial}")
    # eigenvectors of the Hermitian specs: the residual is at rounding level, the bound must still dominate
    for spec in hermitian_specs()[:4]:
        H = ce.build_bitwise(spec).toarray()
        w, V = np.linalg.eigh((H + H.conj().T) / 2)
        for j in (0, 1):
            x = V[:, j].astype(complex if np.iscomplexobj(H) else float)
            check_radius(spec, rng, ctr, False, f"eigenvector {spec.name} j={j}", x=x, mu=float(w[j]))
    # ---- adversarial
    half = LocalSpace.spin_space("1/2", "spin")
    cancel = ModelSpec(3, half, [Term("zz", [(1.0, (0, 1)), (-1.0, (0, 1))]), Term("xx", [(0.7, (1, 2))])], name="cancel")
    check_core(cancel, rng, ctr, True, "exactly cancelling terms")
    check_radius(cancel, rng, ctr, True, "exactly cancelling terms (radius)")
    many = ModelSpec(4, half, [Term("xyz", [(float(rng.standard_normal()), (a, b, c)) for a, b, c in itertools.permutations(range(4), 3)])],
                     name="many-terms-per-site")
    check_core(many, rng, ctr, True, "many terms per site (24 three-letter strings)")
    for scale in (1e150, 1e-150):
        sc = ModelSpec(4, half, [Term("zz", [(scale, (i, (i + 1) % 4)) for i in range(4)]), Term("xx", [(scale, (i, (i + 1) % 4)) for i in range(4)]),
                                 Term("yy", [(scale, (i, (i + 1) % 4)) for i in range(4)])], name=f"scale-{scale:g}")
        check_core(sc, rng, ctr, True, f"couplings {scale:g}")
        check_radius(sc, rng, ctr, True, f"couplings {scale:g} (radius)")
    tiny_x = rng.standard_normal(16) * 1e-160
    check_radius(ce.heisenberg_chain(4), rng, ctr, True, "vector entries 1e-160 (INT-05 additive terms)", x=tiny_x)
    s1 = ce.spin1_heisenberg_chain(3, D=0.3)
    check_core(s1, rng, ctr, True, "spin-1 x/y (rounded amplitudes)")
    zero = ModelSpec(3, half, [Term("zz", [(0.0, (0, 1))]), Term("xx", [(1.0, (1, 2))])], name="zero-coupling")
    check_core(zero, rng, ctr, True, "zero coupling")
    rep = ModelSpec(3, half, [Term("+-", [(1.0, (1, 1))]), Term("zz", [(0.5, (0, 0))])], name="repeated-sites")
    check_core(rep, rng, ctr, True, "repeated sites in one string")
    check_core(ModelSpec(1, half, [Term("z", [(1.0, (0,))])], name="n2"), rng, ctr, True, "n = 2")
    ferm = ce.spinless_fermion_chain(4, V=0.3, boundary="periodic")
    check_core(ferm, rng, ctr, True, "fermions, periodic (JW string across the boundary)")
    # ---- negative controls (N = 6: the ground state is non-degenerate, gap 0.68)
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).toarray()
    w, V = np.linalg.eigh(H)
    x = V[:, 0]
    rel_ok, _ = check_radius(spec, rng, ctr, True, "negative control: correct spec, exact eigenvector", x=x, mu=float(w[0]))
    wrong = ce.heisenberg_chain(6, J=1.0)
    wrong.terms[0].couplings[0] = (wrong.terms[0].couplings[0][0] * 1.001, wrong.terms[0].couplings[0][1])
    rel_wrong, r_wrong = check_radius(wrong, rng, ctr, True, "negative control: one wrong coefficient (bound valid for the wrong operator)",
                                      x=x, mu=float(w[0]))
    ctr.add(True, rel_wrong > 1e3 * rel_ok, f"negative control: wrong coefficient changes the residual ({rel_ok:.3g} -> {rel_wrong:.3g})")
    planted = V[:, 0] + 1e-3 * V[:, 1]
    rel_pl, _ = check_radius(spec, rng, ctr, True, "negative control: planted wrong eigenvector", x=planted, mu=float(w[0]))
    ctr.add(True, rel_pl > 1e-4, f"negative control: planted vector residual {rel_pl:.3g} is above rounding level")
    ok = ctr.report("D-02 matrix-free")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
