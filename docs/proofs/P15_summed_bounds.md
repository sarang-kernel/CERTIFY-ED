# Proposition 15: linear combinations of certified observables (S4 extension)

Implements: `certify_ed/correlations.py` (`combine_certified`, `modulus_upper`, `certify_correlations`,
`structure_factor`, `certify_observable_general`). Depends on Theorem 6 (`docs/theory.md`, Section 4: the
projector-level bound |tr(P O) − tr(Xᴴ O X)| ≤ b for a certified cluster). Notation as in `docs/theory.md`,
Section 9.2. The lift of the correlation statements to other operator classes is a separate track; this
document covers the mathematics that the present code implements.

## 1. Statement

**Hypotheses.**
(H1) P is the spectral projector of a certified cluster of g eigenvalues (Theorem 6), O_1 … O_m are the
matrices actually passed to `certify_observable` (Hermitian, as stored in floating point), and for each k a
value v_k ∈ C and a bound b_k ≥ 0 are given with |tr(P O_k) − v_k| ≤ b_k (Theorem 6 including its rounding
term; v_k is complex only for the Hermitian decomposition of C-05, where v_k = v_H + i v_K and b_k = b_H + b_K).
(H2) a_1 … a_m ∈ C are the coefficients *as stored in floating point* (binary64 pairs).
(H3) The standard model of floating-point arithmetic (Higham 2002, §2.2) with unit roundoff u = 2^−53: every
+, −, ×, ÷, √ returns the exact result times (1 + δ), |δ| ≤ u, with no overflow and no underflow in the
operations listed in Section 3 (the only subnormal-range operation, |x| + |y| for a tiny component, is
exact in any case).

**Proposition 15.** Σ_k a_k tr(P O_k) lies within Σ_k |a_k| b_k of Σ_k a_k v_k:
  | Σ_k a_k tr(P O_k) − Σ_k a_k v_k | ≤ Σ_k |a_k| b_k.                                                   (1)

**Floating-point corollary (what `combine_certified` returns).** Let val = fl(Σ_k a_k v_k) be the value
computed by numpy (complex products, then a summation in any order) and
  B = Σ^↑_k up(|a_k|^↑ b_k) + up(2 γ_{m+2} Σ^↑_k up(|a_k|^↑ |v_k|^↑)),
where |z|^↑ := `modulus_upper(z)` ≥ |z|, up(x) := fl(x (1 + 8u)) ≥ x for x computed with one rounding,
Σ^↑ is the sum accumulated with `add_up` (every partial sum rounded up) and γ_n = nu/(1 − nu). Then
  | Σ_k a_k tr(P O_k) − val | ≤ B.                                                                         (2)

**Lemma A (modulus bound).** For z = x + iy with x, y binary64 and (if both are nonzero) |x|, |y| ≥ 2^−500,
`modulus_upper(z)` = up(fl(√(fl(fl(x²) + fl(y²))))) ≥ |z|; if a component is below 2^−500,
`modulus_upper(z)` = up(fl(|x| + |y|)) ≥ |z|; if a component is 0 it is the other's absolute value.

**Lemma B (rounding of the complex dot product).** |fl(Σ_k a_k v_k) − Σ_k a_k v_k| ≤ γ_{m+2} Σ_k |a_k||v_k|.

**Uses in the code (each is (1)–(2) with specific coefficients).**
* Batch of observables: m independent Theorem 6 records; (1) is not even needed.
* Two-point functions C_ij = tr(P O_i O_j): a single Theorem 6 record with O = the stored matrix O_i O_j
  (i ≠ j) or O_i² (i = j); per-state value tr(P O)/g with bound up(b/g) ≥ b/g.
* Hermitian decomposition (C-05): tr(P O) = tr(P H) + i tr(P K) is (1) with a = (1, i), bound b_H + b_K
  (formed with `add_up`).
* Static structure factor: S(q) = (1/N) Σ_{i,j} e^{iq·(r_i − r_j)} C_ij is (2) with a_ij = fl(e^{iq·(r_i−r_j)})/N
  and v_ij = C_ij (C_ji = C_ij or conj(C_ij) exactly, Section 2).

**What is *not* claimed (documented hypotheses, not theorems).**
(D1) The phases a_ij are the floating-point values returned by `numpy.exp(1j θ)/N`. The statement (2) is about
these coefficients. The distance to the exact phases, |fl(e^{iθ}) − e^{iθ}|, depends on the accuracy of the
platform's libm (cos, sin), for which no bound is proved here; under the usual libm accuracy (≤ 1 ulp per
component) it is ≤ 2u·√2, i.e. the exact-phase S(q) differs from the stated one by at most
(2√2 u / N) Σ_ij |C_ij| ≈ 3e−16 · (1/N) Σ|C_ij|, far below the Theorem 6 bounds, but this is an
assumption. The record names it (`phase_hypothesis`).
(D2) The operator of a two-point function is the stored product matrix fl(O_i O_j). For local operators with
dyadic entries (spin-1/2 x, y, z, ±, n; any ±1/2, ±1, ±i/2 entries) the product entries are exact; for
entries such as 1/√2 (spin-1 x, y) the stored product is rounded and the statement refers to it, not to the
exact O_i O_j. The record names it (`operator_hypothesis`). Making the product exact or bounding its
rounding belongs to the correlation lift (other track).

## 2. Proof

**(1).** Write t_k = tr(P O_k). Then Σ a_k t_k − Σ a_k v_k = Σ a_k (t_k − v_k), and by the triangle inequality
and |a_k (t_k − v_k)| = |a_k||t_k − v_k| ≤ |a_k| b_k, the modulus is at most Σ |a_k| b_k. ∎

**Lemma A.** Let r = |z| = √(x² + y²). *Main branch* (|x|, |y| ≥ 2^−500): x², y² ≥ 2^−1000 are normal, so (H3)
applies: fl(x²) = x²(1+δ₁), fl(y²) = y²(1+δ₂), p := fl(fl(x²) + fl(y²)) = (x²(1+δ₁) + y²(1+δ₂))(1+δ₃) ≥
r²(1−u)² (all |δ| ≤ u, the terms positive). IEEE 754 requires a correctly rounded square root, so s := fl(√p)
= √p (1+δ₄) ≥ r (1−u)(1−u) = r(1−u)². Finally up(s) = fl(s(1+8u)) ≥ s(1+8u)(1−u) ≥ r(1−u)³(1+8u) ≥ r, because
(1−u)³(1+8u) = 1 + 5u − 21u² + 13u³ − 8u⁴ ≥ 1 for 0 < u ≤ 1/8 (the term 5u dominates 21u² when u ≤ 5/21). *Tiny
branch*: |x| + |y| ≥ r always (the 1-norm dominates the 2-norm), fl(|x| + |y|) = (|x| + |y|)(1+δ) with |δ| ≤ u
(addition satisfies the model in the whole range, subnormals included, since a subnormal sum is exact), and
up(·) ≥ (|x|+|y|)(1−u)²(1+8u) ≥ r. *Zero component*: exact. ∎

**Lemma B.** numpy forms each product p_k = fl(a_k v_k) with the four-multiplication formula (or with fused
multiply-adds, depending on the build); in both cases p_k = a_k v_k (1 + δ_k) with |δ_k| ≤ √2 γ_2 (Higham
2002, Lemma 3.5, for the conventional formula; Jeannerod, Kornerup, Louvet, Muller, *Math. Comp.* 86 (2017)
881–898, give ≤ 2u for the FMA variants, which is smaller). The sum of the m products is then formed by
some sequence of m − 1 complex additions (numpy uses pairwise summation; the argument does not depend on
the order). Each complex addition satisfies fl(x + y) = (x + y)(1 + δ), |δ| ≤ u (Higham, Lemma 3.5). Every
summand passes through at most m − 1 additions, so fl(Σ p_k) = Σ p_k (1 + θ_k) with |θ_k| ≤ γ_{m−1}
(Higham, Lemma 3.1: a product of at most n factors (1 + δ_i), |δ_i| ≤ u, is 1 + θ_n with |θ_n| ≤ γ_n; this is
the standard summation analysis of Higham §4.2, valid for any ordering). Hence
  |fl(Σ a_k v_k) − Σ a_k v_k| ≤ Σ |a_k v_k| |(1 + δ_k)(1 + θ_k) − 1| ≤ Σ |a_k||v_k| (√2 γ_2 + γ_{m−1} + √2 γ_2 γ_{m−1}).
Now √2 γ_2 ≤ γ_3 (equivalent to 2√2(1 − 3u) ≤ 3(1 − 2u), true for u ≤ 1/8 since 2√2 < 3), and Higham's Lemma
3.3 gives γ_j + γ_k + γ_j γ_k ≤ γ_{j+k}, so the constant is at most γ_3 + γ_{m−1} + γ_3 γ_{m−1} ≤ γ_{m+2}. (For
m = 1 there is no addition and the bound √2 γ_2 ≤ γ_3 suffices.) ∎

**(2).** By (1) and Lemma B,
  |Σ a_k t_k − val| ≤ |Σ a_k t_k − Σ a_k v_k| + |Σ a_k v_k − val| ≤ Σ |a_k| b_k + γ_{m+2} Σ |a_k||v_k|.
It remains to show that B is at least the right side. *Products:* with |a_k|^↑ ≥ |a_k| (Lemma A) and b_k ≥ 0,
fl(|a_k|^↑ b_k) ≥ |a_k|^↑ b_k (1−u), and up(·) multiplies by at least (1+8u)(1−u), so up(|a_k|^↑ b_k) ≥
|a_k|^↑ b_k (1−u)²(1+8u) ≥ |a_k| b_k. Likewise up(|a_k|^↑ |v_k|^↑) ≥ |a_k||v_k|. *Sums:* `add_up(s, t)` =
nextafter(fl(s + t), +∞) ≥ s + t, because fl(s + t) is the binary64 nearest to s + t and the next binary64
above it is therefore ≥ s + t; by induction the accumulated Σ^↑ is ≥ the exact sum of its terms. *Rounding
term:* `gamma(m+2)` computes ν = (m+2)u exactly (an integer times a power of two) and then fl(ν / fl(1 − ν))
≥ γ_{m+2} (1−u)/(1+u) ≥ γ_{m+2}(1 − 2u); the factor 2 is exact; the product with Σ^↑ loses (1−u); up(·) gives
(1+8u)(1−u). Altogether the computed term is ≥ 2 γ_{m+2} Σ|a_k||v_k| (1−2u)(1−u)³(1+8u) ≥ 2 γ_{m+2} Σ|a_k||v_k|
(the product of the factors is 1 + 3u − O(u²) ≥ 1), which is ≥ γ_{m+2} Σ|a_k||v_k| with a factor 2 to spare.
The final `add_up` of the two parts is again ≥ their sum. ∎

**Uses.** *Per-state bound:* fl(b/g) ≥ (b/g)(1−u) and up(·) ≥ (b/g)(1−u)²(1+8u) ≥ b/g. *Two-point functions:*
for site operators on different sites, O_i O_j = O_j O_i and (O_i O_j)ᴴ = O_jᴴ O_iᴴ = O_j O_i = O_i O_j, so the
operator is Hermitian and tr(P O_i O_j) is a Theorem 6 observable; C_ji = C_ij exactly. For the two-letter
case O = S⁺_i S⁻_j, Oᴴ = S⁺_j S⁻_i and tr(P Oᴴ) = tr((P O)ᴴ) = conj tr(P O) (P Hermitian), so C_ji = conj(C_ij)
exactly; the decomposition O = H + iK with H = (O + Oᴴ)/2, K = (O − Oᴴ)/(2i) gives tr(P O) = tr(P H) + i tr(P K)
by linearity, and (1) with a = (1, i) gives the bound b_H + b_K (the code checks that H and K were formed
exactly, else the record is INCONCLUSIVE). *Structure factor:* (2) with a_ij = fl(e^{iq·(r_i−r_j)})/N, so
|a_ij| ≤ |a_ij|^↑ enters as computed by Lemma A (the division by N is one more rounding, covered because
`modulus_upper` is applied to the stored a_ij, whatever its provenance). S(q) is real when C_ji = conj(C_ij),
since the terms (i, j) and (j, i) are conjugates; the code reports the real part and the imaginary part of
val separately (the latter is rounding noise bounded by B). ∎

*Classical results cited.* Triangle inequality; Higham, *Accuracy and Stability of Numerical Algorithms*,
2nd ed. (2002): Lemma 3.1 (γ_n), Lemma 3.3 (γ_j + γ_k + γ_j γ_k ≤ γ_{j+k}), Lemma 3.5 (complex arithmetic),
§4.2 (summation in any order); Jeannerod, Kornerup, Louvet, Muller (2017) for complex multiplication with
FMA; IEEE 754-2008 §5.4.1 (square root is correctly rounded).

## 3. Where the code implements it

`certify_ed/correlations.py` (line numbers of this revision, 272 lines):

* Lemma A: `modulus_upper` (129–141): the zero-component cases, the 2^−500 threshold, `up(math.sqrt(x*x +
  y*y))` (rounding: `up` is the (1 + 8u) guard of `fp.py`, applied to a quantity with three roundings).
* (2): `combine_certified` (144–163): `val = complex(np.sum(a * v))` (155); `bound` accumulated with `add_up` of
  `up(ma * bk)`; `sum_av` accumulated with `add_up` of `up(ma * modulus_upper(vk))` (158–161); `rnd = up(2 *
  gamma(m + 2) * sum_av)` (162); return `add_up(bound, rnd)` (163). Every rounding is upward. (The 3.1 code
  used `np.abs(a)` — a libm hypot — for |a_k| and multiplied the rounding term by (1 + 8u) after a
  nearest-rounded sum, which the proof could not cover for m > 8; the bound can only have grown by a few ulps.)
* per-state bound: `pr["per_state_bound"] = float(up(pr["bound"] / g))` (212) and in `structure_factor` (259).
* Hermitian decomposition: `certify_observable_general` (86–125): `bound = float(add_up(rh["bound"], rk["bound"]))`
  (102).
* Two-point functions: `certify_correlations` (166–223) with the record field `operator_hypothesis` (D2, 190).
* Structure factor: `structure_factor` (226–262): the table of C_ij with the conjugate rule from
  `conjugate_symmetry`, the phases `np.exp(1j * q.(r_i - r_j)) / N` (252), `combine_certified`; the record
  carries `phase_hypothesis` (D1, 261).

## 4. Computational verification

`docs/proofs/verify_P15.py` (run: `python docs/proofs/verify_P15.py --random 500 --seed 0`):

* Part A, (2) end to end in exact arithmetic: 500 random cases with m ∈ {1, 2, 3, 8, 16, 64, 256, 1000}
  coefficients (uniform complex, Pythagorean-modulus rationals so that |a_k| is rational and the adversary
  can saturate |d_k| = b_k exactly in the direction of conj(a_k), or all components ≈ 2^−520, the tiny branch
  of Lemma A), values v_k real or complex, bounds b_k ∈ 10^−16 … 1; exact traces t_k = v_k + d_k with |d_k| ≤
  b_k exactly rational; the check |Σ a_k t_k − val|² ≤ bound² in `fractions.Fraction`. 32 adversarial cases:
  m = 1 saturated; exact cancellations Σ a_k v_k = 0 with Σ|a_k||v_k| up to 2·10^10; m = 5000; all-tiny
  coefficients; the empty sum; the per-state division up(b/g) ≥ b/g for 20 (b, g) pairs.
* Part B, Lemma A: 511 complex numbers with components spanning 2^−520 … 2^300 (both branches, the
  threshold itself, 5e−324, components 0, (3, 4), (0.6, 0.8), (1/3, 1/7)): `modulus_upper(z)`² ≥ x² + y² exactly;
  worst overestimate in the main branch 2.2e−15 relative (two ulps: the (1 + 8u) guard).
* Part C, Lemma B: 100 numpy complex dot products with m ∈ {1, 2, 7, 100, 1000, 20000}, 30 % with exact
  cancellation (second half the negative of the first), the error measured exactly against
  γ_{m+2} Σ |a_k|^↑ |v_k|^↑; the largest observed error was 0.41 of the bound.

Result (seed 0): 1111 random + 32 adversarial cases, 0 violations.

## 5. Analytic cross-check

*Two terms, real.* a = (1, −1), v = (x, x), b = (0, 0): the exact combination is 0, val = fl(x − x) = 0 exactly,
and the bound is the rounding term 2γ_4 · 2|x| ≈ 8.9e−16 |x| ≥ 0 — nothing is certified wrongly, and the
bound shows the price of the a priori analysis for a sum that happens to be exact.

*Saturation.* With m = 1, a = (3 + 4i)/5 (|a| = 1), v = 0, b = 1 and the true trace t = conj(a) (|t| = b), the
combination a t = 1 lies exactly at distance b = 1 from val = 0: (1) is attained, and the computed bound
exceeds 1 by the rounding term only (≈ 5e−16 · |a||v| = 0 here, so by the ulp of `up`). The verification's
Pythagorean cases are this configuration at random scales.

*Structure factor at q = 0.* a_ij = 1/N exactly (fl(e^{0}) = 1), S(0) = (1/N) Σ_ij C_ij = (1/N) tr(P (Σ_i O_i)²):
Proposition 15 reduces to Theorem 6 for the operator (Σ_i O_i)²/N with the bound (1/N) Σ_ij b_ij, which is the
sum of the N² single-pair bounds; for g = 1 and O_i = S^z_i this is the total-S^z fluctuation.

*Limit of many terms.* For m → ∞ with fixed Σ|a_k||v_k|, the rounding term grows like 2(m + 2)u Σ|a_k||v_k|,
the linear-in-m behaviour of recursive summation (Higham §4.2); for the structure factor m = N² and
Σ|a_k||v_k| ≤ (1/N) Σ|C_ij| ≤ N max|C_ij|, so the term is ≈ 2N³u max|C_ij| ≈ 2e−13 for N = 10: negligible
against Theorem 6 bounds of order 10^−10 but not against exact degeneracies, which is why it is kept.

## 6. Status

Derived on 2026-10-30; machine-checked (1111 random + 32 adversarial cases, 0 violations);
independent re-derivation: done 2026-10-30 (from the statement alone, not a human check; verdict AGREES, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
