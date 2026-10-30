"""Computational verification of Proposition 16' (C-06, docs/proofs/C06_truncated_thermodynamics.md).

Random exact spectra lambda_1 <= ... <= lambda_n (rationals); the k lowest are given as intervals (point or
widened), the rest only through L <= lambda_{k+1} (and optionally M >= lambda_n). The intervals returned by
certify_ed.thermo.thermodynamics_truncated must contain the exact Z, F, U, C of the full spectrum (mpmath
interval arithmetic at 200 digits). Adversarial cases saturate the hypotheses (L = lambda_{k+1}, M = lambda_n,
all omitted eigenvalues at L or at M, temperatures in each envelope regime, k = n - 1, huge n - k). Negative
controls: a planted L above lambda_{k+1} and an omitted count that is too small must be detected (the exact
values leave the intervals at some temperature).

Usage: python docs/proofs/verify_C06.py [--random N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import mpmath as mp  # noqa: E402

from certify_ed.thermo import thermodynamics_truncated  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_P16 import _inside  # noqa: E402


def exact_thermo(E, T):
    """(Z, F, U, C) as mpmath intervals at 200 digits for a spectrum given as values or (value, multiplicity)
    pairs (the omitted eigenvalues of the adversarial cases are up to 2^20 copies of one value)."""
    iv = mp.iv
    iv.dps = 200
    b = 1 / iv.mpf(T)
    terms = []
    for e in E:
        v, m = (e if isinstance(e, tuple) else (e, 1))
        vm = iv.mpf(v.numerator) / v.denominator if isinstance(v, Fraction) else iv.mpf(v)
        terms.append((vm, int(m)))
    w = [(vm, m, iv.exp(-b * vm)) for vm, m in terms]
    Z = sum((m * x for _, m, x in w[1:]), w[0][1] * w[0][2])
    S1 = sum((m * vm * x for vm, m, x in w[1:]), w[0][1] * w[0][0] * w[0][2])
    S2 = sum((m * vm * vm * x for vm, m, x in w[1:]), w[0][1] * w[0][0] * w[0][0] * w[0][2])
    U = S1 / Z
    C = b * b * (S2 / Z - U * U)
    F = -iv.log(Z) / b
    return Z, F, U, C


def _intervals(lam_k, rng: random.Random, widen: bool):
    out = []
    for x in lam_k:
        w = Fraction(0) if not widen or rng.random() < 0.5 else Fraction(rng.randrange(1, 100), 2 ** 40)
        out.append({"lo": float(x - w), "hi": float(x + w), "count": 1})
    return out


def _consistent_spectrum(ivs, tail, rng: random.Random, mode: str):
    """A spectrum consistent with the intervals (ends included) and with the tail description."""
    E = []
    for r in ivs:
        E.append(Fraction(r["lo"]) if mode == "lo" else Fraction(r["hi"]) if mode == "hi"
                 else Fraction(r["lo"]) + (Fraction(r["hi"]) - Fraction(r["lo"])) * Fraction(rng.randrange(0, 1001), 1000))
    E += tail
    return E


def one_case(rng: random.Random, adversarial: bool):
    n = rng.randrange(3, 14)
    k = rng.randrange(1, n)
    scale = Fraction(2) ** rng.randrange(-3, 7)
    # dyadic spectra: the floats passed as interval ends, L and M are then exactly the rationals of the hypothesis
    lam = sorted(Fraction(rng.randrange(-1000, 1000), 128) * scale for _ in range(n))
    # make the gap lambda_{k+1} - lambda_k positive (the claim must be separable)
    if lam[k] <= lam[k - 1]:
        lam[k:] = [x + (lam[k - 1] - lam[k]) + scale / 64 for x in lam[k:]]
    ivs = _intervals(lam[:k], rng, widen=not adversarial)
    hi_k = max(Fraction(r["hi"]) for r in ivs)
    gap = lam[k] - hi_k
    if adversarial:
        L = lam[k]                                               # saturated
        M = lam[-1] if rng.random() < 0.7 else None              # saturated upper bound, or none
        mode_tail = rng.choice(["atL", "atM", "spread"])
        if mode_tail == "atL":
            tail = [lam[k]] * (n - k)
        elif mode_tail == "atM" and M is not None:
            tail = [lam[-1]] * (n - k)
        else:
            tail = lam[k:]
        if rng.random() < 0.3:                                   # a huge omitted count with the same k
            extra = 2 ** rng.randrange(10, 21)
            tail = tail + [(lam[-1] if M is not None else lam[k], extra)]
            n += extra
    else:
        L = lam[k] - gap * Fraction(rng.randrange(0, 1024), 1024)
        M = (lam[-1] + Fraction(rng.randrange(0, 1024), 128) * scale) if rng.random() < 0.6 else None
        tail = lam[k:]
    ell = L - min(Fraction(r["lo"]) for r in ivs)
    temps = sorted({float(ell) * f for f in (0.05, 0.3, 0.9, 1.5, 3.0)} | {float(M - min(Fraction(r["lo"]) for r in ivs)) * 2 if M is not None else float(ell) * 10})
    res = thermodynamics_truncated(ivs, float(L), n, temps, upper_bound=None if M is None else float(M))
    ok = True
    for row in res["rows"]:
        for mode in ("lo", "hi", "mid"):
            E = _consistent_spectrum(ivs, tail, rng, mode)
            ok &= _inside(row, *exact_thermo(E, row["T"]))
    return ok


def check(rng: random.Random, n: int, adversarial: bool):
    cases = viol = 0
    for _ in range(n):
        cases += 1
        viol += not one_case(rng, adversarial)
    return cases, viol


def negative_controls(rng: random.Random):
    """Hypothesis violations that the exact values must expose (so the harness is not vacuous)."""
    lam = [Fraction(0), Fraction(1), Fraction(2), Fraction(50)]
    ivs = [{"lo": 0.0, "hi": 0.0, "count": 1}]
    temps = [0.3, 1.0, 3.0, 100.0]
    out = []
    # planted L above lambda_2: the true spectrum has weight the bound does not allow
    res = thermodynamics_truncated(ivs, 1.5, 4, temps)
    out.append(("planted L > lambda_{k+1}", not all(_inside(r, *exact_thermo(lam, r["T"])) for r in res["rows"])))
    # omitted count too small (dimension 3 declared, 4 eigenvalues exist): at T = 100, Z ~ 4 > 3
    res = thermodynamics_truncated(ivs, 1.0, 3, temps)
    out.append(("dimension too small", not all(_inside(r, *exact_thermo(lam, r["T"])) for r in res["rows"])))
    # upper bound below lambda_n: at T = 100 the true U ~ 13 exceeds the bound ~ 2.25
    res = thermodynamics_truncated(ivs, 1.0, 4, temps, upper_bound=3.0)
    out.append(("M below lambda_n", not all(_inside(r, *exact_thermo(lam, r["T"])) for r in res["rows"])))
    # the honest case passes at every temperature (control of the controls)
    res = thermodynamics_truncated(ivs, 1.0, 4, temps, upper_bound=50.0)
    out.append(("honest", all(_inside(r, *exact_thermo(lam, r["T"])) for r in res["rows"])))
    # input validation: L not above the claim, M below L, counts above n
    for args in ((ivs, 0.0, 4, temps, None), (ivs, 1.0, 4, temps, 0.5), (ivs, 1.0, 0, temps, None)):
        try:
            thermodynamics_truncated(*args[:4], upper_bound=args[4])
            out.append((f"validation {args[1:3]}", False))
        except ValueError:
            out.append((f"validation {args[1:3]}", True))
    return out


def run(n_random: int = 300, seed: int = 0) -> dict:
    mp.mp.dps = 200
    rng = random.Random(seed)
    cr, vr = check(rng, n_random, False)
    ca, va = check(rng, max(1, n_random // 2), True)
    neg = negative_controls(rng)
    return {"random_cases": cr, "random_violations": vr, "adversarial_cases": ca, "adversarial_violations": va,
            "negative_controls": neg, "negative_controls_failed": sum(1 for _, ok in neg if not ok),
            "violations": vr + va + sum(1 for _, ok in neg if not ok)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = run(a.random, a.seed)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0 if r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
