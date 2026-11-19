"""Computational verification of docs/proofs/appA_model-shift.md (model shift Delta, builder shift, Weyl transfer).

N1  exact rationals: ||E||_2^2 <= ||E||_1 ||E||_inf and inf_norm_upper >= exact infinity norm
B1  planted specs (spin 1, bosons; strings of length 1-4 with x, y letters): |H_ref - H_spec| entrywise <=
    gamma_{m+2L-1} |H|_abs / (1 - gamma_{m+2L-1}) (computed |H|_abs), ||H_ref - H_spec||_2 <= _builder_shift, the same for
    hermitian_part(H_ref) (Lemma A2)
W1  sorted eigenvalues of H_ref and H_spec within _builder_shift (Theorem A5)
D1  planted producer matrix within product_tolerance: check_model_fidelity's model_shift_bound >= ||H_h - H_spec||_2;
    negative control beyond the tolerance -> FAILED
Ground truth: H_spec from checker.exact_matrix (exact rationals times sqrt of integer weights, mpmath 60 digits).
Usage: python docs/proofs/verify_appA_model-shift.py [--cases 20000] [--seed 5]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _common import Counter, TOL, bottleneck_sorted, eigh, norm2, to_mp  # noqa: E402

import certify_ed as ce  # noqa: E402
from certify_ed.certify import Options, _builder_shift, check_model_fidelity  # noqa: E402
from certify_ed.checker import exact_matrix  # noqa: E402
from certify_ed.construct import SpecOperator, build_bitwise, hermitian_part, product_tolerance  # noqa: E402
from certify_ed.fp import U, inf_norm_upper  # noqa: E402
from certify_ed.localops import LocalSpace  # noqa: E402
from certify_ed.spec import ModelSpec  # noqa: E402

u = Fraction(U)


def gam(k):
    return k * u / (1 - k * u)


def exact_H(spec):
    """H_spec as an mp matrix: D^-1 (W K) D^-1 with W K exact rational, D = diag(sqrt(w))."""
    ex = exact_matrix(spec)
    n = len(ex.weights)
    S = {}
    for r, c, a, b in zip(ex.rows, ex.cols, ex.re, ex.im):
        key = (int(r), int(c))
        S[key] = S.get(key, 0) + mp.mpc(mp.mpf(int(ex.weights[r]) * int(a)), mp.mpf(int(ex.weights[r]) * int(b))) / int(ex.den)
    d = [1 / mp.sqrt(int(w)) for w in ex.weights]
    Hm = mp.matrix(n, n)
    for (i, j), v in S.items():
        Hm[i, j] = d[i] * v * d[j]
    return Hm


def random_spec(rng, kind=None, scale=1.0, lmax=None):
    kind = kind or rng.choice(["spin1", "spin32", "boson", "spinhalf"])
    if kind == "spin1":
        local = LocalSpace.spin_space("1", "spin")
    elif kind == "spin32":
        local = LocalSpace.spin_space("3/2", "spin")
    elif kind == "boson":
        local = LocalSpace.boson_space(int(rng.integers(2, 5)))
    else:
        local = LocalSpace.spin_space("1/2", "spin")
    N = int(rng.integers(2, 4))
    static = []
    nterms = int(rng.integers(1, 4))
    letters = ["+", "-", "n"] if kind == "boson" else ["+", "-", "z", "x", "y"]
    for _ in range(nterms):
        L = int(rng.integers(1, (lmax or 4) + 1))
        if L > N:
            L = N
        sites = list(rng.choice(N, size=L, replace=False))
        c = float(rng.standard_normal()) * scale
        if rng.random() < 0.5:
            # Hermitian pair of strings with +/- letters and their conjugates
            op = "".join(rng.choice(["+", "-"], size=L))
            conj = op.translate(str.maketrans("+-", "-+"))
            static.append([op, [[c] + sites]])
            static.append([conj, [[c] + sites]])
        else:
            # Hermitian letters on distinct sites (x, y, z, n commute across sites)
            op = "".join(rng.choice([t for t in letters if t not in "+-"], size=L))
            static.append([op, [[c] + sites]])
    return ModelSpec.from_static(N, local, static, name="random")


def check_builder(spec, counters, adversarial, tag):
    Hm = exact_H(spec)
    Href = build_bitwise(spec)
    Hd = Href.toarray()
    n = Hd.shape[0]
    op = SpecOperator(spec)
    absH = op.abs_matvec(np.eye(n))                     # computed |H|_abs, column by column
    max_len = max(len(t.op) for t in spec.terms)
    m = spec.n_terms * 2 ** max_len
    g = gam(m + 2 * max_len - 1)
    E = to_mp(Hd) - Hm
    scale = float(abs(Hd).max()) + 1e-300
    tol = TOL * (1 + scale)
    ok = True
    for i in range(n):
        for j in range(n):
            bound = mp.mpf(float(absH[i, j])) * mp.mpf(g.numerator) / mp.mpf(g.denominator) / (1 - mp.mpf(g.numerator) / mp.mpf(g.denominator))
            ok &= abs(E[i, j]) <= bound + tol
    counters["B"].add(adversarial, ok, f"B1 entrywise {tag}")
    bs = _builder_shift(spec)
    nE = norm2(E)
    counters["B"].add(adversarial, nE <= mp.mpf(bs) + tol, f"B1 norm {tag}: ||E||={nE} shift={bs}")
    Hh = hermitian_part(Href).toarray()
    nE2 = norm2(to_mp(Hh) - Hm)
    counters["B"].add(adversarial, nE2 <= mp.mpf(bs) + tol, f"A2 {tag}: ||E_sym||={nE2} shift={bs}")
    if n <= 40:
        lam_ref = eigh(to_mp(Hh))[0]
        lam_spec = eigh(Hm)[0]
        counters["W"].add(adversarial, bottleneck_sorted(lam_ref, lam_spec) <= mp.mpf(bs) + tol, f"W1 {tag}")
    return Href, Hm


def check_model_shift(spec, Href, Hm, rng, counters, adversarial, tag):
    n = Href.shape[0]
    tol_p = product_tolerance(spec, float(abs(Href).max()))
    P = rng.standard_normal((n, n)) * (1j if np.iscomplexobj(Href.toarray()) else 1.0) * 0.3 * tol_p
    P = (P + P.conj().T) / 2
    Hp = sp.csr_matrix(Href.toarray() + P)
    claim = ce.Claim(np.zeros(1), np.ones((n, 1)), operator=Hp, kind="subset")
    stmt, _, _ = check_model_fidelity(claim, spec, Options())
    if stmt["status"] != "CERTIFIED":
        counters["D"].skipped += 1
        return
    Hh = (Hp + Hp.conj().T) / 2
    d = norm2(to_mp(Hh.toarray()) - Hm)
    bound = mp.mpf(stmt["evidence"]["model_shift_bound"])
    counters["D"].add(adversarial, d <= bound, f"D1 {tag}: ||Hh - Hspec||={d} Delta={bound}")
    # negative control: a perturbation far beyond the tolerance is not certified by S1
    Hp2 = sp.csr_matrix(Href.toarray() + 1e3 * tol_p * np.eye(n))
    stmt2, _, _ = check_model_fidelity(ce.Claim(np.zeros(1), np.ones((n, 1)), operator=Hp2, kind="subset"), spec, Options())
    counters["N"].add(True, stmt2["status"] == "FAILED", f"negative control {tag}: {stmt2['status']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    counters = {k: Counter() for k in ("N1", "B", "W", "D", "N")}
    # N1: norm inequality in exact rationals (real) and mp (complex); inf_norm_upper against the exact norm
    for i in range(args.cases // 4):
        n = int(rng.integers(1, 6))
        if rng.random() < 0.7:
            M = np.round(rng.standard_normal((n, n)) * rng.choice([1.0, 1e-8, 1e8]), 6)
            Mf = [[Fraction(float(x)) for x in row] for row in M]
            inf_n = max(sum(abs(x) for x in row) for row in Mf)
            one_n = max(sum(abs(Mf[r][c]) for r in range(n)) for c in range(n))
            with mp.workdps(30):
                two = norm2(to_mp(M))
            ok = two ** 2 <= mp.mpf(inf_n.numerator) / inf_n.denominator * mp.mpf(one_n.numerator) / one_n.denominator * (1 + mp.mpf("1e-20"))
            ok &= Fraction(inf_norm_upper(M)) >= inf_n and Fraction(inf_norm_upper(sp.csr_matrix(M))) >= inf_n
        else:
            M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            Mm = to_mp(M)
            inf_n = max(sum(abs(Mm[r, c]) for c in range(n)) for r in range(n))
            one_n = max(sum(abs(Mm[r, c]) for r in range(n)) for c in range(n))
            with mp.workdps(30):
                ok = norm2(Mm) ** 2 <= inf_n * one_n * (1 + mp.mpf("1e-20"))
            ok &= mp.mpf(inf_norm_upper(M)) >= inf_n
        counters["N1"].add(False, ok, f"N1 case {i}")
    # B1/W1/D1 on planted specs
    n_spec = max(60, args.cases // 200)
    for i in range(n_spec):
        spec = random_spec(rng)
        Href, Hm = check_builder(spec, counters, False, f"spec {i} {spec.local.kind}")
        check_model_shift(spec, Href, Hm, rng, counters, False, f"spec {i}")
    # adversarial specs
    for _i in range(20):
        for kind, scale, lmax in (("spin1", 1e100, 2), ("spin1", 1e-100, 2), ("boson", 1.0, 3), ("spin32", 1.0, 4),
                                  ("spin1", 1.0, 1), ("spinhalf", 1.0, 4)):
            spec = random_spec(rng, kind=kind, scale=scale, lmax=lmax)
            Href, Hm = check_builder(spec, counters, True, f"adv {kind} scale {scale} L {lmax}")
            if scale == 1.0:
                check_model_shift(spec, Href, Hm, rng, counters, True, f"adv {kind} L {lmax}")
    # a huge exact integer coupling next to a non-dyadic one
    local = LocalSpace.spin_space("1", "spin")
    spec = ModelSpec.from_static(2, local, [["zz", [[float(2 ** 40), 0, 1]]], ["+-", [[0.1, 0, 1]]], ["-+", [[0.1, 0, 1]]]])
    Href, Hm = check_builder(spec, counters, True, "huge integer coupling")
    ok = True
    for key, name in (("N1", "norm inequality / inf_norm_upper"), ("B", "builder error (Lemmas A1-A3)"),
                      ("W", "Weyl transfer builder (Theorem A5)"), ("D", "model shift Delta (Theorem A4)"),
                      ("N", "negative controls")):
        ok &= counters[key].report(name)
    print("RESULT:", "0 violations" if ok else "VIOLATIONS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
