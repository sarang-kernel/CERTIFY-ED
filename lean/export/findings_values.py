"""export/findings_values.py

Exact values behind the three findings of CORE.md section 8, in exact rational arithmetic (fractions.Fraction;
the square roots of the Lambda4 margins in 120-digit decimal arithmetic, used only for the printed margins, never
for a sign).  The Lean theorems state the inequalities; this script gives the numbers around them.

    python export/findings_values.py [--code <certify-ed checkout>]

Writes results/core/findings.json.  With --code it also records the lines of certify_ed/ivl.py at v3.1.0
(through `git show`, read only) and in the checkout.
"""
import argparse
import json
import os
import subprocess
from decimal import Decimal, getcontext
from fractions import Fraction as Fr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = Fr(1, 2 ** 53)
getcontext().prec = 120


def gam(k):
    return k * U / (1 - k * U)


def dec(x):
    return Decimal(x.numerator) / Decimal(x.denominator)


def lambda4():
    def q_up(n):
        return (1 - U) * (1 + gam(n + 2) * (1 - U) ** 2) * (1 + 8 * U) * (1 - U) ** 4

    def q_lo(n):
        return (1 + U) * (1 - gam(n + 2) * (1 + U) ** 2) * (1 + U) ** 3 / (1 + 8 * U)

    def sq_up(n, N):  # sign of the upper guard: (1 - g_N) q^2 - (1 + 3u)^2
        return (1 - gam(N)) * q_up(n) ** 2 - (1 + 3 * U) ** 2

    def root_up(n, N):  # (1 - g_N)^(1/2) q - (1 + 3u), in units of u
        return float((dec(1 - gam(N)).sqrt() * dec(q_up(n)) - dec(1 + 3 * U)) / dec(U))

    def root_lo(n, N):  # (1 - 3u) - (1 + g_N)^(1/2) q, in units of u
        return float((dec(1 - 3 * U) - dec(1 + gam(N)).sqrt() * dec(q_lo(n))) / dec(U))

    n26 = 2 ** 26
    lo, hi = n26, 2 ** 27
    assert sq_up(lo, 2 * lo) > 0 and sq_up(hi, 2 * hi) < 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sq_up(mid, 2 * mid) >= 0:
            lo = mid
        else:
            hi = mid
    step = n26 // 64
    samples = [root_up(n26 + k * step, 2 * (n26 + k * step)) for k in range(65)]
    return {
        "upper_guard_N_2n_margin_at_2^26_in_u": root_up(n26, 2 * n26),
        "upper_guard_N_2n_squared_margin_at_2^26_in_u": float(sq_up(n26, 2 * n26) / U),
        "upper_guard_N_n_margin_at_2^26_in_u": root_up(n26, n26),
        "lower_guard_N_2n_margin_at_2^26_in_u": root_lo(n26, 2 * n26),
        "lower_guard_N_n_margin_at_2^26_in_u": root_lo(n26, n26),
        "upper_guard_N_2n_last_n_where_it_holds": lo,
        "upper_guard_N_2n_first_n_where_it_fails": hi,
        "upper_guard_N_2n_margin_at_last_in_u": root_up(lo, 2 * lo),
        "upper_guard_N_2n_margin_at_first_failure_in_u": root_up(hi, 2 * hi),
        "upper_guard_N_2n_margin_decreasing_on_65_points_2^26_to_2^27": all(
            samples[i] > samples[i + 1] for i in range(64)),
        "upper_guard_N_2n_margin_at_2^27_in_u": samples[-1],
    }


def lambda6():
    def g6(k):  # (1 - u)^6 (1 + 4u)(1 + u / gamma_k) - 1, k = w + 2
        return (1 - U) ** 6 * (1 + 4 * U) * (1 + U / gam(k)) - 1

    k1, k2 = 3002399751580329, 3002399751580330
    return {
        "largest_w_plus_2_admitted_by_(w+2)u<=1/3": 2 ** 53 // 3,
        "w_plus_2_last_holding": k1,
        "w_plus_2_failing": k2,
        "margin_at_last_holding_in_u^2": float(g6(k1) / U ** 2),
        "margin_at_failing_in_u^2": float(g6(k2) / U ** 2),
        "failing_value_admitted_by_condition": k2 * U <= Fr(1, 3),
        "next_value_admitted_by_condition": (k2 + 1) * U <= Fr(1, 3),
        "margin_at_w_2^26": float(g6(2 ** 26 + 2)),
    }


def ln_constant():
    P, M = 192, 60
    one = 1 << P
    u_max = -(-one // 3)  # the largest upper argument of the series, ceil(2^P / 3) grid units (Lemma 17 (iv))
    u = Fr(u_max, one)
    tail = u ** (2 * M + 3) * one / ((2 * M + 3) * (1 - u * u))  # true tail bound, in grid units

    def cdiv(a, b):
        return -(-a // b)

    u2 = cdiv(u_max * u_max, one)
    pw = u_max
    for _ in range(M + 1):
        pw = cdiv(pw * u2, one)
    x = Fr(35, 100)
    return {
        "1/(1-x^2)_at_0.35": float(1 / (1 - x * x)),
        "9/8_suffices_at_0.35": 1 / (1 - x * x) <= Fr(9, 8),
        "8/7_suffices_at_0.35": 1 / (1 - x * x) <= Fr(8, 7),
        "excess_of_1/(1-u^2)_over_9/8_at_ceil(2^192/3)": float(1 / (1 - u * u) - Fr(9, 8)),
        "tail_bound_at_ceil(2^192/3)_in_grid_units": float(tail),
        "pw_after_loop_at_ceil(2^192/3)": pw,
        "remainder_added_by_3.1_code_in_grid_units": cdiv(cdiv(pw * 9, 8), 2 * M + 3),
        "remainder_added_by_3.2_code_in_grid_units": cdiv(cdiv(pw * 8, 7), 2 * M + 3),
    }


def code_lines(code):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    out = {}
    v31 = subprocess.run(["git", "-C", code, "show", "v3.1.0:certify_ed/ivl.py"], env=env, capture_output=True,
                         text=True, encoding="utf-8").stdout.splitlines()
    cur = open(os.path.join(code, "certify_ed", "ivl.py"), encoding="utf-8").read().splitlines()
    for tag, lines in (("v3.1.0", v31), ("checkout", cur)):
        out[tag] = {str(i + 1): l.strip() for i, l in enumerate(lines)
                    if "_ceil_div(pw *" in l or "_U_MAX" in l.split("#")[0] or "u_hi <= " in l}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code")
    a = ap.parse_args()
    out = {"lambda4": lambda4(), "lambda6": lambda6(), "ln_remainder_constant": ln_constant()}
    if a.code:
        out["ivl_lines"] = code_lines(a.code)
    path = os.path.join(ROOT, "results", "core", "findings.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
