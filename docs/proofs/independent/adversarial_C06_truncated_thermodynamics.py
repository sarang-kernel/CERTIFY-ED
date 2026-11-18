"""Independent adversarial search for Proposition 16' (docs/proofs/C06_truncated_thermodynamics.md).

Ground truth: the k lowest eigenvalues placed anywhere in their intervals and the n - k omitted ones placed
anywhere in [L, M] (or [L, 1e300) without M): at L exactly, at M exactly, at the argmax of phi (T) and of psi
(2T), spread randomly; exact Z, F, U, Var, C by mpmath (120 digits); the reported binary64 intervals must
contain them. Targets: regime boundaries (T = l, T = 2l, l < T < m with m - l tiny), M = L (all omitted at one
point), n - k = 1 and 10^6, T tiny (beta l > 133: tail weight 0 on the grid), T huge (wide intervals), scales
1e+-100, M absent. Lemma C checked separately on a fine grid (mp). Negative controls: L <= max h_j, M < L, k > n
must raise.

Usage: python docs/proofs/independent/adversarial_C06_truncated_thermodynamics.py [--cases 200] [--seed 43]
"""
import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, report  # noqa: E402
from certify_ed.thermo import thermodynamics_truncated  # noqa: E402

mp.mp.dps = 120


def mpq(x):
    f = Fraction(x) if not isinstance(x, Fraction) else x
    return mp.mpf(f.numerator) / mp.mpf(f.denominator)


def exact(members, T):
    beta = 1 / mpq(T)
    E0 = min(e for e, _ in members)
    ws = [(mpq(e), m * mp.exp(-beta * (mpq(e) - mpq(E0)))) for e, m in members]
    Zt = sum(w for _, w in ws)
    mean = sum(e * w for e, w in ws) / Zt
    var = sum(e * e * w for e, w in ws) / Zt - mean ** 2
    lnZ = mp.log(Zt) - beta * mpq(E0)
    return dict(Z=mp.exp(lnZ), F=-mpq(T) * lnZ, U=mean, variance_E=var, C=beta ** 2 * var, lnZ=lnZ)


def contains(iv, val):
    return iv is None or (mp.mpf(iv[0]) <= val <= mp.mpf(iv[1]))


def lemma_c(rng, T_c):
    for _i in range(200):
        beta = mp.mpf(10) ** mp.mpf(float(rng.uniform(-3, 3)))
        l_ = mp.mpf(10) ** mp.mpf(float(rng.uniform(-3, 3))); m_ = l_ * (1 + mp.mpf(10) ** mp.mpf(float(rng.uniform(-6, 3))))
        phi = lambda e, beta=beta: e * mp.exp(-beta * e)       # noqa: E731
        psi = lambda e, beta=beta: e * e * mp.exp(-beta * e)   # noqa: E731
        clamp = lambda x, l_=l_, m_=m_: min(max(x, l_), m_)       # noqa: E731
        grid = [l_ + (m_ - l_) * mp.mpf(t) / 400 for t in range(401)]
        ok = max(phi(e) for e in grid) <= phi(clamp(1 / beta)) * (1 + mp.mpf("1e-100")) and min(phi(e) for e in grid) >= min(phi(l_), phi(m_)) * (1 - mp.mpf("1e-100"))
        ok = ok and max(psi(e) for e in grid) <= psi(clamp(2 / beta)) * (1 + mp.mpf("1e-100")) and min(psi(e) for e in grid) >= min(psi(l_), psi(m_)) * (1 - mp.mpf("1e-100"))
        ok = ok and abs(phi(1 / beta) - mp.exp(-1) / beta) <= mp.mpf("1e-100") * phi(1 / beta) and abs(psi(2 / beta) - 4 * mp.exp(-2) / beta ** 2) <= mp.mpf("1e-100") * psi(2 / beta)
        T_c.add(ok, True, f"Lemma C beta={mp.nstr(beta, 5)} l={mp.nstr(l_, 5)} m={mp.nstr(m_, 5)}")


def make(rng, i):
    kind = i % 9
    s = 1.0
    if kind == 0:
        m = int(rng.integers(1, 4)); los = np.sort(rng.standard_normal(m)) - 3; ws = np.abs(rng.standard_normal(m)) * 1e-6
        ivs = [{"lo": float(l), "hi": float(l + w), "count": int(rng.integers(1, 3))} for l, w in zip(los, ws)]
        L = float(max(iv["hi"] for iv in ivs) + abs(rng.standard_normal()) + 0.1); M = L + float(abs(rng.standard_normal())) * 5
        return ivs, L, M, sum(iv["count"] for iv in ivs) + int(rng.integers(1, 4)), [0.05, 0.5, 2.0], "random", False
    if kind == 1:   # regime boundaries: T = l and T = 2l exactly (E0 = 0)
        ivs = [{"lo": 0.0, "hi": 0.0, "count": 1}]; L = 1.0; M = 7.0
        return ivs, L, M, 5, [1.0, 2.0, 0.5, 3.5, 7.0], "regime_boundaries", True
    if kind == 2:   # M = L: all omitted at one point
        ivs = [{"lo": -1.0, "hi": -0.999, "count": 2}]; L = 0.5; M = 0.5
        return ivs, L, M, 4, [0.1, 1.0, 10.0], "M_eq_L", True
    if kind == 3:   # huge omitted count
        ivs = [{"lo": 0.0, "hi": 1e-9, "count": 1}]; L = 2.0; M = 50.0
        return ivs, L, M, 10 ** 6, [0.05, 1.0, 100.0], "n_minus_k_1e6", True
    if kind == 4:   # tiny T: beta * l > 133
        ivs = [{"lo": 0.0, "hi": 1e-12, "count": 1}, {"lo": 1e-3, "hi": 1.1e-3, "count": 1}]; L = 0.3; M = 2.0
        return ivs, L, M, 3, [1e-3, 1e-4], "tiny_T", True
    if kind == 5:   # huge T, wide intervals
        ivs = [{"lo": -1.0, "hi": -0.9, "count": 1}]; L = 0.0; M = 1.0
        return ivs, L, M, 3, [1e3, 1e6], "huge_T", True
    if kind == 6:   # scales
        s = 1e100 if i % 18 < 9 else 1e-100
        ivs = [{"lo": -2 * s, "hi": -1.99 * s, "count": 1}, {"lo": -s, "hi": -0.99 * s, "count": 2}]; L = 0.0; M = 3 * s
        return ivs, L, M, 5, [0.1 * s, s], f"scale({s:g})", True
    if kind == 7:   # M absent
        ivs = [{"lo": -1.0, "hi": -0.99, "count": 1}]; L = 0.2; M = None
        return ivs, L, M, 4, [0.05, 0.3, 2.0], "no_M", True
    # 8: m - l tiny with T between them (regime gamma with straddling enclosures)
    ivs = [{"lo": 0.0, "hi": 0.0, "count": 1}]; L = 1.0; M = 1.0 + 1e-15
    return ivs, L, M, 2, [1.0, 1.0 + 5e-16, 0.5 + 2.5e-16], "straddle", True


def members(ivs, L, M, n_tail, T, rng, mode):
    out = []
    for iv in ivs:
        lo, hi, c = Fraction(iv["lo"]), Fraction(iv["hi"]), iv["count"]
        out.append((lo if mode == "lo" else hi if mode == "hi" else (lo + hi) / 2, c))
    Lq = Fraction(L); Mq = Fraction(M) if M is not None else Lq + Fraction(10 ** 300)
    if mode == "lo":
        tail = [(Lq, n_tail)]
    elif mode == "hi":
        tail = [(Mq, n_tail)]
    elif mode == "mid":
        E0 = min(e for e, _ in out)
        cands = [Lq, Mq, E0 + Fraction(T), E0 + 2 * Fraction(T)]
        cands = [min(max(cv, Lq), Mq) for cv in cands]
        per = n_tail // len(cands)
        tail = [(cv, per + (1 if j < n_tail % len(cands) else 0)) for j, cv in enumerate(cands)]
        tail = [(cv, k_) for cv, k_ in tail if k_ > 0]
    else:
        tail = []
        rem = n_tail
        for j in range(4):
            k_ = rem if j == 3 else int(rng.integers(0, rem + 1))
            rem -= k_
            if k_ > 0:
                t = Fraction(int(rng.integers(0, 1001)), 1000)
                tail.append((Lq + t * (Mq - Lq), k_))
    return out + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_c = Tally("Lemma C envelopes (mp grid)"); lemma_c(rng, T_c)
    T_in = Tally("Proposition 16': reported intervals contain exact Z, lnZ, F, U, Var, C for members (claimed + tail anywhere in [L, M])")
    T_neg = Tally("negative controls: L <= max h, M < L, k > n raise")
    T_nov = Tally("no-value cases (ValueError/ZeroDivisionError) only below the grid or with a zero lower Z~")
    for i in range(a.cases):
        ivs, L, M, n, temps, tag, adv = make(rng, i)
        k = sum(iv["count"] for iv in ivs)
        for T in temps:
            try:
                res = thermodynamics_truncated(ivs, L, n, [T], upper_bound=M)
            except (ValueError, ZeroDivisionError) as e:
                E0 = min(iv["lo"] for iv in ivs); beta_hi = max(1.0 / T, 2.0 ** -192) + 2.0 ** -192
                legit = T < 2.0 ** -192 or all((iv["hi"] - E0) * beta_hi > 130 for iv in ivs)
                T_nov.add(legit, True, f"{tag} T={T}: {e}")
                continue
            row = res["rows"][0]
            for mode in ("lo", "hi", "mid", "spread"):
                mem = members(ivs, L, M, n - k, T, rng, mode)
                ex = exact(mem, T)
                bad = [key for key in ("Z", "lnZ", "F", "U", "variance_E", "C") if key in row and not contains(row[key], ex[key])]
                T_in.add(not bad, adv, f"{tag} T={T} mode={mode}: " + "; ".join(f"{key}: {row[key]} vs {mp.nstr(ex[key], 20)}" for key in bad))
    for _i in range(10):
        ivs = [{"lo": 0.0, "hi": 0.1, "count": 2}]
        for args, exc in (((ivs, 0.1, 4, [1.0]), ValueError), ((ivs, 0.5, 4, [1.0], 0.4), ValueError), ((ivs, 0.5, 1, [1.0]), ValueError)):
            try:
                thermodynamics_truncated(*args); T_neg.add(False, True, f"no raise for {args[1:]}")
            except exc:
                T_neg.add(True, True, "")
    sys.exit(1 if report([T_c, T_in, T_neg, T_nov]) else 0)


if __name__ == "__main__":
    main()
