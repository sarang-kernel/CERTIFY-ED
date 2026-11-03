# Lemma 17: dyadic interval arithmetic with outward rounding, including exp and ln

Implements: `certify_ed/ivl.py`. Used by: Proposition 16 (`thermo.py`, S7) and its truncated-spectrum
extension C-06. Notation as in `docs/theory.md`, Section 9.3.

## 1. Statement

**Grid.** P = 192. The grid is G = 2^-P · Z. An *interval* is a pair of integers (lo, hi) with lo ≤ hi,
standing for the real interval [lo·2^-P, hi·2^-P]. For a real x write ⌊x⌋_G = ⌊x·2^P⌋ and ⌈x⌉_G = ⌈x·2^P⌉ (the
integers of the grid ends immediately below and above x). All integers are Python integers (unbounded).

**Hypotheses.** None beyond the definitions: the lemma is a statement about integer arithmetic. No
floating-point operation enters except in (v), where the only facts used are that every finite binary64 is
a dyadic rational, that Python's `float(Fraction)` (integer true division) is correctly rounded to nearest,
and that `math.nextafter` returns the adjacent binary64.

**Conclusion.** For every operation below, the output interval contains the exact result of the exact
real operation applied to any members of the input intervals (the *enclosure property*).

(i) *Grid operations without rounding.* `from_fraction(q)` = (⌊q⌋_G, ⌈q⌉_G) ∋ q. Addition,
subtraction, negation, `scale_int` (multiplication by an integer c) and `sum_intervals` are exact on the
grid (integer arithmetic on the ends, the ends swapped for c < 0).

(ii) *Grid operations with rounding.* For X = [x_lo, x_hi], Y = [y_lo, y_hi] (ends as grid integers):
  * X · Y = (⌊min(p)/2^P⌋, ⌈max(p)/2^P⌉) with p the four corner products x_i y_j;
  * X / Y, for 0 ∉ Y, = (min ⌊(x_i 2^P)/y_j⌋, max ⌈(x_i 2^P)/y_j⌉) over the four corners;
  * X² = (⌊x_lo²/2^P⌋, ⌈x_hi²/2^P⌉) if x_lo ≥ 0; (⌊x_hi²/2^P⌋, ⌈x_lo²/2^P⌉) if x_hi ≤ 0;
    (0, ⌈max(x_lo², x_hi²)/2^P⌉) otherwise;
  * `intersect_nonneg`: (max(x_lo, 0), x_hi), defined when x_hi ≥ 0; it contains every member that is ≥ 0.

(iii) *exp.* For a grid point x ≥ 0 (`_exp_point_nonneg`): let s ≥ 0 be the number of halvings
t ← ⌈t/2⌉ (integer ceiling) needed from t = x until t ≤ 2^P/2, let t_lo = ⌊x/2^s⌋ and t_hi = ⌈x/2^s⌉
(integers), N = 40, and
  S_N^↓(t) = 2^P + Σ_{k=1}^{N} term_k^↓ with term_0^↓ = 2^P, term_k^↓ = ⌊⌊term_{k-1}^↓ · t / 2^P⌋ / k⌋,
  S_N^↑(t) = 2^P + Σ_{k=1}^{N} term_k^↑ with term_0^↑ = 2^P, term_k^↑ = ⌈⌈term_{k-1}^↑ · t / 2^P⌉ / k⌉,
  rem = ⌈2 · ⌈term_N^↑(t_hi) · t_hi / 2^P⌉ / (N + 1)⌉.
Then exp(x·2^-P) ∈ [lo, hi] where (lo, hi) is the result of applying the squaring of (ii) s times to the
interval (S_N^↓(t_lo), S_N^↑(t_hi) + rem). For x < 0, exp(x 2^-P) ∈ (⌊2^{2P}/e_hi⌋, ⌈2^{2P}/e_lo⌉) where
(e_lo, e_hi) is the enclosure of exp(|x| 2^-P). `exp_interval([a, b])` = (lower end of exp at a, upper end
of exp at b).

(iv) *ln.* For a grid point v > 0 (`_ln_point`), let k = bitlength(v) − 1 − P, so that v·2^-P = 2^k · m with
m ∈ [1, 2). Let m_lo = ⌊v / 2^k⌋ and m_hi = m_lo + [v/2^k not an integer] (grid integers, m_lo = m_hi = v·2^-k
exactly when k ≤ 0), u_lo = max(0, ⌊(m_lo − 2^P) 2^P / (m_lo + 2^P)⌋), u_hi = max(0, ⌈(m_hi − 2^P) 2^P / (m_hi +
2^P)⌉). With M = 60 and, for a grid integer u,
  A^↓(u) = 2 Σ_{j=0}^{M} ⌊pw_j^↓ / (2j+1)⌋, pw_0^↓ = u, pw_{j+1}^↓ = ⌊pw_j^↓ · ⌊u²/2^P⌋ / 2^P⌋,
  A^↑(u) = 2 ( Σ_{j=0}^{M} ⌈pw_j^↑ / (2j+1)⌉ + ⌈⌈8 pw_{M+1}^↑ / 7⌉ / (2M+3)⌉ ), pw_0^↑ = u,
  pw_{j+1}^↑ = ⌈pw_j^↑ · ⌈u²/2^P⌉ / 2^P⌉,
the lemma asserts ln(v·2^-P) ∈ (A^↓(u_lo), A^↑(u_hi)) + k · L2, where L2 = (A^↓(⌊2^P/3⌋), A^↑(⌈2^P/3⌉)) ∋ ln 2
and k·L2 is the exact `scale_int`. The series is only called with 0 ≤ u_lo ≤ u_hi ≤ U_MAX = ⌊0.35 · 2^P⌋
(checked at runtime; a violation raises). `ln_interval([a, b])`, a > 0, = (lower end at a, upper end at b).

(v) *Conversion to binary64.* `to_float_down(v)` is the largest binary64 ≤ v·2^-P and `to_float_up(v)` the
smallest binary64 ≥ v·2^-P (when these exist in the finite range; otherwise an OverflowError is raised and
the caller reports no value). `from_float(f)` = `from_fraction(Fraction(f))` is exact (width 0).

## 2. Proof

Throughout, "⌊·⌋, ⌈·⌉ are monotone and ⌊y⌋ ≤ y ≤ ⌈y⌉" is used without further comment; Python's `//` is the
floor and `_ceil_div(a, b) = -((-a) // b)` is the ceiling for b > 0.

**(i).** ⌊q 2^P⌋ ≤ q 2^P ≤ ⌈q 2^P⌉ gives from_fraction. For members x ∈ X, y ∈ Y (as reals), x + y ∈
[x_lo + y_lo, x_hi + y_hi] 2^-P, and the ends are integers, so no rounding occurs; likewise for −, negation,
c·x (ends swapped for c < 0, because multiplication by a negative number reverses order), and finite sums.

**(ii).** *Product.* The map (x, y) ↦ xy is bilinear, so on the rectangle X × Y its minimum and maximum
are attained at corners. Hence xy·2^P ∈ [min p, max p]·2^-P (p the corner products of the grid integers,
which carry a factor 2^{-2P}), and flooring the lower end and ceiling the upper end preserves the
inclusion. *Quotient.* For 0 ∉ Y, x/y is monotone in x and, on each side of 0, monotone in y, so the
extremes over X × Y are again at corners; (x_i 2^P)/y_j is the exact quotient in grid units and the
directed divisions bracket it. *Square.* On [x_lo, x_hi] ⊂ [0, ∞) the square is increasing, on (−∞, 0]
decreasing, and when 0 ∈ X its range is [0, max(x_lo², x_hi²)]; the ends are rounded outward as for the
product. *intersect_nonneg.* Trivial.

**(iii).** Let τ = x / 2^s as a real number (in grid units: τ 2^-P is the halved argument). *Halving count.*
For integers a and b, c > 0, ⌈⌈a/b⌉/c⌉ = ⌈a/(bc)⌉ (Graham–Knuth–Patashnik, *Concrete Mathematics*, eq.
(3.11)); hence after s iterated ceilings t = ⌈x/2^s⌉ = t_hi, and the loop stops exactly when t_hi ≤ 2^{P-1},
i.e. t_hi 2^-P ≤ 1/2. Also t_lo ≤ τ ≤ t_hi.

*Lower end.* For a real t ≥ 0 and k ≥ 1, let T_k = t^k/k! (in real units). We show by induction that
term_k^↓(t_lo) ≤ T_k(t_lo 2^-P) · 2^P: the base k = 0 is equality; the step floors the exact product and
quotient of nonnegative quantities, which can only decrease them. Therefore S_N^↓(t_lo) 2^-P ≤ Σ_{k≤N}
T_k(t_lo 2^-P) ≤ exp(t_lo 2^-P) ≤ exp(τ 2^-P), the second inequality because all Taylor terms of exp at a
nonnegative argument are nonnegative, the third by monotonicity of exp.

*Upper end.* By the same induction with ceilings, term_k^↑(t_hi) ≥ T_k(t_hi 2^-P) 2^P. The Lagrange form
is not needed; the tail of the series is bounded directly: for 0 ≤ t ≤ 1/2 (real units),
  Σ_{k>N} t^k/k! = (t^{N+1}/(N+1)!) Σ_{j≥0} t^j (N+1)!/(N+1+j)! ≤ (t^{N+1}/(N+1)!) Σ_{j≥0} (t/(N+2))^j
  = (t^{N+1}/(N+1)!) / (1 − t/(N+2)),
because (N+1+j)! ≥ (N+1)! (N+2)^j. With t ≤ 1/2 and N = 40, 1/(1 − t/(N+2)) ≤ 1/(1 − 1/84) < 2. Moreover
t^{N+1}/(N+1)! = T_N(t) · t/(N+1) ≤ term_N^↑ 2^-P · t_hi 2^-P /(N+1), so the tail is at most
2 · term_N^↑ · t_hi 2^{-2P} / (N+1) ≤ rem · 2^-P by the two ceilings in the definition of rem. Hence
(S_N^↑(t_hi) + rem) 2^-P ≥ exp(t_hi 2^-P) ≥ exp(τ 2^-P).

*Squarings.* Both ends of the interval (S_N^↓, S_N^↑ + rem) are ≥ 2^P (the lower end starts at 2^P and adds
nonnegative terms), so the interval lies in [1, ∞), where squaring is increasing; (ii) gives an enclosure
of the square at each of the s steps, and exp(x 2^-P) = exp(τ 2^-P)^{2^s}. Induction on s.

*Negative arguments.* exp(−y) = 1/exp(y) and 1/e is decreasing in e > 0; with e ∈ [e_lo, e_hi] 2^-P, e_lo ≥
2^P > 0, the reciprocal in grid units is 2^{2P}/e, bracketed by the directed divisions. *exp_interval*:
exp is increasing, so the image of [a, b] is [exp a, exp b].

**(iv).** *Decomposition.* 2^{k+P} ≤ v < 2^{k+P+1} by the definition of k, so v 2^-P = 2^k m with m = v 2^{-k-P}
∈ [1, 2). In grid units m 2^P = v 2^-k; for k ≤ 0 this is the integer v 2^{|k|} (m_lo = m_hi), for k > 0 the
integer part is ⌊v/2^k⌋ = m_lo and v/2^k < m_lo + 1 = m_hi when the division is inexact. So m_lo ≤ m 2^P ≤
m_hi, with m_hi ≤ 2·2^P (because m 2^P < 2·2^P and m_hi is the smallest integer ≥ m 2^P when inexact, equal
to it when exact).

*The argument u.* u(m) = (m − 1)/(m + 1) is increasing on [1, 2) with u(1) = 0 and u(m) < 1/3, and ln m =
ln((1+u)/(1−u)) = 2 atanh(u) (direct computation). In grid units, u(m_lo 2^-P) 2^P = (m_lo − 2^P) 2^P /
(m_lo + 2^P), floored to u_lo (the max with 0 is harmless since u ≥ 0); the same with ceiling and m_hi gives
u_hi. Thus u_lo ≤ u(m) 2^P ≤ u_hi. *Range of u_hi.* From m_hi ≤ 2·2^P, u_hi ≤ ⌈2^P · 2^P / (3 · 2^P)⌉ = ⌈2^P/3⌉
≤ 2^P/3 + 1, so u_hi 2^-P ≤ 1/3 + 2^-P < 0.35; this is why the runtime check admits arguments up to U_MAX =
⌊0.35 · 2^P⌋ ≥ ⌈2^P/3⌉ (checked in `verify_L17.check_remainder_constants`) and why the remainder constant
below is stated for u ≤ 0.35 rather than 1/3. (The 3.1 code used the constant 9/8 valid only for u ≤ 1/3;
the enclosure was in practice unaffected because the deficiency is far below one grid unit, but the proof
needs the weaker constant, so the code now uses 8/7.)

*Series.* atanh(u) = Σ_{j≥0} u^{2j+1}/(2j+1) for 0 ≤ u < 1, all terms nonnegative. By induction (as in
(iii)) pw_j^↓ ≤ u^{2j+1} 2^P and pw_j^↑ ≥ u^{2j+1} 2^P for the respective grid arguments; the floors of the
quotients give A^↓(u_lo)/2 ≤ Σ_{j≤M} u_lo^{2j+1}/(2j+1) ≤ atanh(u_lo 2^-P) ≤ atanh(u(m)), the last step by
monotonicity of atanh. For the upper end, with u = u_hi 2^-P ≤ 0.35:
  Σ_{j>M} u^{2j+1}/(2j+1) ≤ (u^{2M+3}/(2M+3)) Σ_{i≥0} u^{2i} = u^{2M+3} / ((2M+3)(1 − u²)) ≤ (8/7) u^{2M+3}/(2M+3),
since u² ≤ 0.1225 ≤ 1/8. After the loop pw_{M+1}^↑ ≥ u^{2M+3} 2^P, so the remainder is at most the term
⌈⌈8 pw^↑/7⌉/(2M+3)⌉ 2^-P. Hence A^↑(u_hi)/2 ≥ atanh(u_hi 2^-P) ≥ atanh(u(m)).

*ln 2.* ln 2 = 2 atanh(1/3) (u(2) = 1/3), and 1/3 lies in the grid interval (⌊2^P/3⌋, ⌈2^P/3⌉), so L2 ∋ ln 2 by
the series bounds just proved (the upper argument ⌈2^P/3⌉ ≤ U_MAX). *Assembly.* ln(v 2^-P) = k ln 2 + ln m
= k ln 2 + 2 atanh(u(m)); k·L2 is exact by (i) and the sum of two enclosures encloses the sum. *ln_interval*:
ln is increasing on (0, ∞).

**(v).** q = v 2^-P is a rational; f = `float(q)` is the binary64 nearest to q (CPython's integer true
division is correctly rounded; `Fraction.__float__` is `numerator / denominator`). If f > q, the binary64
immediately below f, f⁻ = nextafter(f, −∞), satisfies f⁻ ≤ q: otherwise q < f⁻ < f and f⁻ would be
closer to q than f. So `to_float_down` returns the largest binary64 ≤ q: it is ≤ q, and the next one up is
f (if f ≤ q, f itself is returned and the next one up exceeds q, else f would not be nearest). Symmetric
for `to_float_up`. `from_float`: a finite binary64 is m·2^e with integers m, e, a dyadic rational, which
`Fraction(float)` represents exactly. ∎

*Classical results cited.* Taylor series of exp and atanh with nonnegative terms (any calculus text;
e.g. Rudin, *Principles of Mathematical Analysis*, Thm 8.1 for power series, and the identity atanh u =
½ ln((1+u)/(1−u))); the nested-ceiling identity ⌈⌈a/b⌉/c⌉ = ⌈a/(bc)⌉ (Graham, Knuth, Patashnik, *Concrete
Mathematics*, 2nd ed., eq. (3.11)); correctly rounded integer division in CPython (`long_true_divide`,
Objects/longobject.c, documented as correctly rounded).

## 3. Where the code implements it

`certify_ed/ivl.py` (line numbers of this revision, 273 lines):

* grid constants `PREC`, `ONE` (19–20), `_floor_div` (26), `_ceil_div` (30).
* (i) `from_fraction` (34–37), `__add__`/`__sub__`/`__neg__` (79–86), `scale_int` (92–94),
  `sum_intervals` (141–146). Rounding direction: none (exact).
* (ii) `__mul__` (88–90): `_floor_div(min(ps), ONE)` lower, `_ceil_div(max(ps), ONE)` upper.
  `__truediv__` (96–103): floors for the lower end (`qs`), ceilings for the upper (`qs2`); raises on 0 ∈ Y.
  `square` (105–110): the three sign cases; `intersect_nonneg` (112–116).
* (iii) `_EXP_TERMS = 40` (150); `_exp_point_nonneg` (153–183): halving loop with `_ceil_div(t, 2)` (159–162),
  `t_lo = x >> s`, `t_hi = _ceil_div(x, 1 << s)` (164–165), `poly(tv, up)` termwise with floor (lower) or
  ceiling (upper) of the product and of the division by k (167–174), `rem` (178), s squarings (181–182).
  `_EXP_ARG_MAX = ONE << 16`, `_EXP_ARG_UNDER = 1000 * ONE` (186–187); `exp_interval` (190–205): OverflowError
  above the argument limit (196), the shortcut `1 if up else 0` for arguments ≤ −1000 (201–202; exp(−1000) ≈
  5e−435 < 2^−192, so (0, 1 unit) encloses it), the negative branch dividing `ONE << PREC` = 2^{2P} with
  `_ceil_div` for the upper end and `_floor_div` for the lower end (203–204).
* (iv) `_LN_TERMS = 60`, `_U_MAX = (7 * ONE) // 20` (209–210); `_atanh2_series` (213–238): runtime check of the
  argument range (219–220), `u2` floored/ceiled, `pw` floored/ceiled, the remainder `_ceil_div(_ceil_div(pw * 8,
  7), 2 * _LN_TERMS + 3)` added to the upper end only (233); `ln2_interval` (241–246) with the grid interval around
  1/3; `_ln_point` (249–266): `k`, `m`, `exact_shift`, `m_lo`, `m_hi`, `u_lo` (`_floor_div`), `u_hi`
  (`_ceil_div`), `a + kl`; `ln_interval` (269–273).
* (v) `to_float_down` / `to_float_up` (51–66), `from_float` (40–44).

Every constant of the proof appears literally: 1/2 as `half = ONE >> 1`, the factor 2 and `_EXP_TERMS + 1`
in `rem`, 8/7 and `2 * _LN_TERMS + 3` in the ln remainder, 1/3 as `_floor_div(ONE, 3)`, `_ceil_div(ONE, 3)`,
0.35 as `(7 * ONE) // 20`, 1000 and 2^16 as the exp argument guards.

## 4. Computational verification

`docs/proofs/verify_L17.py` (run: `python docs/proofs/verify_L17.py --random 2000 --seed 0`):

* grid operations: 2000 random interval pairs with random rational members (ends included with
  probability 0.2), every operation of (i)–(ii) checked against `fractions.Fraction`; the binary64
  conversions checked to be the tightest floats on the safe side;
* exp: 2000 random arguments (uniform in [−700, 700], in [−2, 2], random dyadics of 60–80 bits, arbitrary
  grid points of up to 200 bits) and 66 adversarial ones (x = 0; x = 1/2 and its grid neighbours, where the
  halving count changes; powers of two and their neighbours; odd grid points with up to seven inexact
  halvings; |x| = 700 and 745 near the binary64 overflow threshold; |x| = 1000 and its grid neighbours,
  where the underflow shortcut starts; x = 2^16 and one unit above, the argument limit), each also through `exp_interval` on a
  random interval containing it; ground truth mpmath at 200 digits;
* ln: 2000 random arguments (log-uniform over 1e−13 … 1e130, uniform in [0.5, 4], random rationals, arbitrary
  grid points of up to 600 bits) and 74 adversarial ones (z = 1, 2, 1/2, powers of two up to 2^19 and their
  grid neighbours where m → 2 and u → 1/3, z = 2^2000, 2^−150, grid points with k > 0 and low bits set so
  that the shift is inexact);
* the two remainder constants, in exact arithmetic (1/(1 − t/42) ≤ 2 for t ≤ 1/2; 1/(1 − u²) ≤ 8/7 for
  u ≤ 0.35; ⌈2^P/3⌉ ≤ U_MAX), and that both truncated series have remainders below one grid unit at their
  largest admitted arguments (tightness).

Result (seed 0, current code): 6000 random + 140 adversarial cases, 0 violations. Smallest relative slack of an exp
enclosure 6.9e−58 (about 2^−190, i.e. the enclosures are tight to a few grid units); largest ln enclosure
width 8.7e−54 (for z ≈ 2^600 the k·ln 2 term scales the width of L2).

## 5. Analytic cross-check

Closed-form special cases, all reproduced by the enclosures (part of the adversarial list):

* exp(0): s = 0, t = 0, every term after the first is 0, rem = 0, the interval is exactly [1, 1];
  `exp_interval(Interval(0, 0))` returns `UNIT` (both ends equal to 2^P).
* ln 1: v = 2^P, k = 0, m = 1, u = 0, the series is 0 on both sides (the remainder term is ⌈⌈0⌉/(2M+3)⌉ = 0),
  so the enclosure of ln 1 is exactly [0, 0].
* ln 2^j: v = 2^{j+P}, k = j, m = 1, u = 0: the enclosure is j·L2 exactly, so ln 2^j ∈ j·L2 reduces to
  ln 2 ∈ L2, i.e. to the atanh series at 1/3, which is the classical BBP-type series ln 2 = 2 Σ 3^{−(2j+1)}/(2j+1).
* The identity exp(ln z) = z: for the adversarial z the composition `exp_interval(ln_interval(Z))` contains
  Z (a consequence of the two enclosure properties, checked in `tests/test_proofs_props.py`).
* Limiting behaviour of the remainder: for t → 0 the tail is O(t^{N+1}) and rem → 0 (at t = 2^−P, rem = 1
  grid unit at most, from the outer ceiling), so the enclosure of exp near 0 has width O(N · 2^−P): the
  rounding of N terms, not the truncation.

## 6. Status

Derived by Claude on 2026-10-30; machine-checked (6000 random + 140 adversarial cases, 0 violations);
independent re-derivation: done 2026-10-30 (fresh-context session, verdict AGREES, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
