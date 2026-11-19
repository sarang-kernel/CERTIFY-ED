# Independent re-derivation: Lemma 17 (dyadic interval arithmetic with outward rounding, exp and ln)

Written on 2026-10-30 before the original proof was read (not a human check). Sections 1 and 3 of `../L17_dyadic_exp_ln.md` read first, together with the
code `certify_ed/ivl.py` (the statement is about that code); proof written before section 2 was opened.
Notation: P = 192, grid G = 2⁻ᴾℤ, an interval (lo, hi) of integers stands for [lo·2⁻ᴾ, hi·2⁻ᴾ]; ⌊·⌋, ⌈·⌉ integer
floor/ceiling (Python `//` floors towards −∞; `_ceil_div(a, b) = −((−a)//b)`).

## Statement checked

Enclosure property of every operation (i)–(v) of section 1.

## My proof

Two facts used throughout. (F1) For integers a, b > 0 and a real z with a ≤ z·b... more simply: for a real y,
⌊y⌋ ≤ y ≤ ⌈y⌉; for integers a and b > 0, `a // b` = ⌊a/b⌋ and `_ceil_div(a, b)` = ⌈a/b⌉ (−⌊−a/b⌋ = ⌈a/b⌉). (F2) A
map of several variables that is monotone in each variable separately attains its extrema over a box at the
corners.

(i) `from_fraction(q)` = (⌊n2ᴾ/d⌋, ⌈n2ᴾ/d⌉): the ends are ⌊q2ᴾ⌋ ≤ q2ᴾ ≤ ⌈q2ᴾ⌉, so q is enclosed and no narrower
grid interval contains it. Addition (lo+lo, hi+hi), subtraction (lo−hi, hi−lo), negation (−hi, −lo), `scale_int`
(c·lo, c·hi for c ≥ 0, swapped for c < 0) and `sum_intervals` are the exact interval-arithmetic formulas on
integer ends (every member x + y of X + Y lies between x_lo + y_lo and x_hi + y_hi, etc.); integer arithmetic is
exact. ✓

(ii) Multiplication: xy is monotone in x for fixed y and in y for fixed x, so over the box X × Y its range is
[min, max] of the four corner products (F2); the exact value in grid units is x_i y_j/2ᴾ, and floor (lower) /
ceiling (upper) of that enclose it. Division with 0 ∉ Y: x/y is monotone in x and, y keeping one sign, in y, so
corners again; grid value x_i 2ᴾ/y_j, floors for the lower end, ceilings for the upper. Square: for x_lo ≥ 0 the
map is increasing (range [x_lo², x_hi²]); for x_hi ≤ 0 decreasing (range [x_hi², x_lo²]); otherwise 0 is attained
and the maximum is at a corner; floors/ceilings as before. `intersect_nonneg`: every member ≥ 0 lies in
[max(x_lo, 0), x_hi]. ✓

(iii) exp, x ≥ 0 a grid integer. Let t_0 = x, t_{i+1} = ⌈t_i/2⌉, and s the first index with t_s ≤ 2ᴾ⁻¹. For integers
⌈⌈a/2⌉/2⌉ = ⌈a/4⌉ (both equal the least integer ≥ a/4), so by induction t_s = ⌈x/2ˢ⌉ = t_hi, and t_lo = x >> s =
⌊x/2ˢ⌋ ≤ x/2ˢ ≤ t_hi ≤ 2ᴾ⁻¹. Put τ_lo = t_lo 2⁻ᴾ ≤ ξ := x 2⁻ᴾ⁻ˢ ≤ τ_hi = t_hi 2⁻ᴾ ≤ ½.
*Lower polynomial.* Claim: term_k^↓ ≤ 2ᴾ τ_lo^k/k! for all k, by induction: term_0^↓ = 2ᴾ; term_k^↓ =
⌊⌊term_{k−1}^↓ t_lo/2ᴾ⌋/k⌋ ≤ term_{k−1}^↓ τ_lo/k ≤ 2ᴾ τ_lo^k/k! (all quantities ≥ 0). Hence S_N^↓(t_lo) = 2ᴾ + Σ_{k≤N}
term_k^↓ ≤ 2ᴾ Σ_{k≤N} τ_lo^k/k! ≤ 2ᴾ exp(τ_lo) ≤ 2ᴾ exp(ξ).
*Upper polynomial.* term_k^↑ ≥ 2ᴾ τ_hi^k/k! by the same induction with ceilings, so S_N^↑(t_hi) ≥ 2ᴾ Σ_{k≤N} τ_hi^k/k!.
Remainder: Σ_{k>N} τ^k/k! = (τ^{N+1}/(N+1)!) Σ_{j≥0} τ^j (N+1)!/(N+1+j)! ≤ (τ^{N+1}/(N+1)!) Σ_j (τ/(N+2))^j =
(τ^{N+1}/(N+1)!)/(1 − τ/(N+2)) ≤ 2τ^{N+1}/(N+1)! for τ ≤ ½ (since τ/(N+2) ≤ ½). In grid units
2ᴾ τ_hi^{N+1}/(N+1)! = (2ᴾ τ_hi^N/N!)·τ_hi/(N+1) ≤ term_N^↑ t_hi 2⁻ᴾ/(N+1), and rem = ⌈2⌈term_N^↑ t_hi/2ᴾ⌉/(N+1)⌉ ≥
2·term_N^↑ τ_hi/(N+1) ≥ 2ᴾ·(tail). Hence S_N^↑(t_hi) + rem ≥ 2ᴾ exp(τ_hi) ≥ 2ᴾ exp(ξ).
*Squarings.* Both ends are ≥ 2ᴾ (acc starts at 2ᴾ, terms ≥ 0), so the first case of (ii) applies: if
[lo, hi] ∋ 2ᴾ e^{ξ} then [⌊lo²/2ᴾ⌋, ⌈hi²/2ᴾ⌉] ∋ 2ᴾ e^{2ξ}; after s squarings the interval encloses 2ᴾ e^{2ˢξ} =
2ᴾ exp(x2⁻ᴾ). ✓ (x = 0 returns (2ᴾ, 2ᴾ) = {1} exactly.)
*Negative arguments.* exp(−|x|2⁻ᴾ) = 1/exp(|x|2⁻ᴾ); in grid units 2ᴾ/(e/2ᴾ) = 2²ᴾ/e with e ∈ [e_lo, e_hi], e_lo ≥ 2ᴾ > 0,
so ⌊2²ᴾ/e_hi⌋ ≤ 2²ᴾ/e ≤ ⌈2²ᴾ/e_lo⌉. For |x| ≥ 1000·2ᴾ: 0 < exp(−1000) ≈ 5.1·10⁻⁴³⁵ < 2⁻¹⁹² ≈ 1.6·10⁻⁵⁸, so (0, 1) in grid
units encloses it. Arguments above 2¹⁶·2ᴾ raise (no value). `exp_interval([a, b])` = (lower end at a, upper end
at b) encloses exp over [a, b] because exp is increasing. ✓

(iv) ln, v > 0 a grid integer. With bl the bit length of v, 2^{bl−1} ≤ v < 2^{bl}, so v2⁻ᴾ = 2ᵏ m with
k = bl − 1 − P and m = v 2^{−k−P} ∈ [1, 2). The scaled m (integer 2ᴾm when k ≤ 0 — a left shift, exact — or
⌊v/2ᵏ⌋ when k > 0) satisfies m_lo ≤ 2ᴾm ≤ m_hi with m_hi = m_lo + [inexact]. u := (m − 1)/(m + 1) is increasing in
m, lies in [0, 1/3) for m ∈ [1, 2), and ln m = ln((1 + u)/(1 − u)) = 2 atanh(u) = 2Σ_{j≥0} u^{2j+1}/(2j+1) (|u| < 1).
In grid units u·2ᴾ = (2ᴾm − 2ᴾ)2ᴾ/(2ᴾm + 2ᴾ), so u_lo = ⌊(m_lo − 2ᴾ)2ᴾ/(m_lo + 2ᴾ)⌋ ≤ 2ᴾ u(m) ≤ u_hi = ⌈(m_hi − 2ᴾ)2ᴾ/
(m_hi + 2ᴾ)⌉ (monotonicity in m, then outward rounding); u_lo ≥ 0 because m_lo ≥ 2ᴾ, and u_hi ≤ ⌈2ᴾ/3⌉ because
m_hi ≤ 2ᴾ⁺¹ (m_lo ≤ 2ᴾ⁺¹ − 1), which is ≤ U_MAX = ⌊0.35·2ᴾ⌋. *Series.* For a grid integer u ≥ 0 with υ = u2⁻ᴾ:
pw_j^↓ ≤ 2ᴾ υ^{2j+1} and pw_j^↑ ≥ 2ᴾ υ^{2j+1} by induction (pw_0 = u exact; ⌊u²/2ᴾ⌋ ≤ 2ᴾυ² ≤ ⌈u²/2ᴾ⌉; products
and floors/ceilings monotone on non-negative integers), so A^↓(u_lo) = 2Σ_{j≤M} ⌊pw_j^↓/(2j+1)⌋ ≤ 2ᴾ·2Σ_{j≤M}
υ_lo^{2j+1}/(2j+1) ≤ 2ᴾ·2 atanh(υ_lo) ≤ 2ᴾ ln m. For the upper end, the tail Σ_{j>M} υ^{2j+1}/(2j+1) ≤
υ^{2M+3}/(2M+3) Σ_{i≥0} υ^{2i} = υ^{2M+3}/((2M+3)(1 − υ²)) ≤ (8/7)υ^{2M+3}/(2M+3) for υ ≤ 0.35 (1 − 0.35² = 0.8775 ≥
7/8); in grid units ≤ (8/7)pw_{M+1}^↑/(2M+3) ≤ ⌈⌈8pw_{M+1}^↑/7⌉/(2M+3)⌉, so A^↑(u_hi) ≥ 2ᴾ·2 atanh(υ_hi) ≥ 2ᴾ ln m.
*ln 2.* ln 2 = 2 atanh(1/3) ((1 + 1/3)/(1 − 1/3) = 2), and 1/3 ∈ [⌊2ᴾ/3⌋, ⌈2ᴾ/3⌉]2⁻ᴾ, so L2 = (A^↓(⌊2ᴾ/3⌋),
A^↑(⌈2ᴾ/3⌉)) ∋ 2ᴾ ln 2 by the same monotonicity; k·L2 (exact `scale_int`, ends swapped for k < 0) ∋ 2ᴾ k ln 2.
Exact addition: (A-interval) + k·L2 ∋ 2ᴾ(ln m + k ln 2) = 2ᴾ ln(v2⁻ᴾ). `ln_interval` on [a, b] with a > 0: ln is
increasing. ✓ (Non-emptiness: u_lo ≤ u_hi gives A^↓ ≤ A^↑ since every floor ≤ the corresponding ceiling.)

(v) Every finite binary64 is a dyadic rational, so `from_float` is exact. `float(Fraction(v, 2ᴾ))` is Python's
correctly rounded integer true division (CPython `long_true_divide`, round-half-even). If f = fl(q) > q, the
binary64 immediately below f is ≤ q (otherwise it would be nearer to q than f), so `nextafter(f, −∞)` is the
largest binary64 ≤ q; if f ≤ q, f itself is (the next binary64 above f is > q by the same argument, and f ≤ q).
Symmetrically for `to_float_up`. Values beyond the binary64 range make the division raise OverflowError, which
the callers treat as "no value". ✓ ∎

## Code check (current `dev/3.2`, `certify_ed/ivl.py`, 273 lines)

Lines as the section 3 table lists them (unchanged on this branch): constants 19–20, `_floor_div`/`_ceil_div`
26–31, `from_fraction` 34–37, conversions 40–65, arithmetic 79–116, `sum_intervals` 141–146, exp 150–205
(N = 40 at 150, halving 160–162, t_lo/t_hi 164–165, poly 167–174, rem 178, squarings 181–182, guards 186–187,
negative branch 203–204), ln 209–273 (M = 60 and U_MAX at 209–210, runtime range check 219–220, series 222–234,
ln2 241–246, `_ln_point` 249–266). Every constant of the proof is literal in the code (½ as `ONE >> 1`, 2 and
N + 1 in `rem`, 8/7 and 2M + 3, ⌊2ᴾ/3⌋/⌈2ᴾ/3⌉, 7/20, 1000, 2¹⁶). Directions: lower ends floor, upper ends ceiling
everywhere; the remainders are added to the upper end only. One detail worth stating in the document: the
identity ⌈⌈a/2⌉/2⌉ = ⌈a/4⌉ is what makes t_hi (line 165) equal to the loop's final t (line 161) and hence ≤ ½ —
the remainder bound uses τ_hi ≤ ½, and section 1 asserts it only implicitly.

## Comparison with section 2 of the original (read after the proof above was written)

Identical in every step, including the nested-ceiling identity for t_hi (the original cites Graham–Knuth–
Patashnik (3.11), which I had asked for), the tail bound of exp with the factor 2 for τ ≤ ½, the (8/7) constant
for u ≤ 0.35 with the remark on the earlier 9/8 (valid only up to 1/3), the ln 2 = 2 atanh(1/3) route, and the
nearest-float argument for (v). Constants coincide. Both proofs valid. The original's remark that the 3.1
constant 9/8 was insufficient for u up to 1/3 + 2⁻ᴾ is correct ((1 − u²)⁻¹ at u = 1/3 + ε exceeds 9/8); the
present code uses 8/7.

**Code mapping**: as in my code check above; the line numbers of section 3 are current (the file is unchanged
on `dev/3.2`).

**Verdict: AGREES.**

## Runs

- `verify_L17.py --random 600 --seed 7`: see VERDICTS.md for the counts printed (the script's argument is
  `--random`, not `--cases`).
- `adversarial_L17_dyadic_exp_ln.py --cases 300 --seed 29`: arithmetic 206 random + 94 adversarial, exp 325
  (specials: 0, halving boundaries, 2¹⁶, −1000 ± 1 unit, −2⁴⁰ real units), ln 319 (specials: 1, powers of two,
  the m_hi = 2·2ᴾ family with u_hi = ⌈2ᴾ/3⌉, v = 1), float conversion 300 + 300 (halfway points, just above
  powers of two), tightness 392 + 11 (relative width < 2⁻¹⁵⁰ for |x| ≤ 1); 0 violations; the OverflowError
  above 2¹⁶ is raised.
