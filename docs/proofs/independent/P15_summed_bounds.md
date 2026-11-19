# Independent re-derivation: Proposition 15 (linear combinations of certified observables)

Written from the statement alone on 2026-10-30 (not a human check). Sections 1 and 3 of `../P15_summed_bounds.md` read first, with
`certify_ed/correlations.py` 130–164; proof written before section 2 was opened. u = 2⁻⁵³, γ_n = nu/(1 − nu);
Higham (2002) Lemma 3.1, Lemma 3.3, Lemma 3.5 (complex arithmetic: |fl(xy) − xy| ≤ √2 γ₂|x||y|, |fl(x + y) − (x + y)|
≤ u|x + y|), §4.2 (summation in any order: each term carries a factor 1 + θ_{m−1}).

## Statement checked

(1) |Σ a_k tr(PO_k) − Σ a_k v_k| ≤ Σ|a_k|b_k; (2) |Σ a_k tr(PO_k) − val| ≤ B with val = fl(Σ a_k v_k) and B as
defined; Lemma A (`modulus_upper` ≥ |z|); Lemma B (|fl(Σ a_k v_k) − Σ a_k v_k| ≤ γ_{m+2}Σ|a_k||v_k|).

## My proof

(1) Σ a_k tr(PO_k) − Σ a_k v_k = Σ a_k (tr(PO_k) − v_k); triangle inequality and (H1): ≤ Σ|a_k||tr(PO_k) − v_k| ≤
Σ|a_k|b_k. ✓

Lemma B. numpy forms t̂_k = fl(a_k v_k) (complex product: two real products and one addition per component,
with or without FMA) and then fl(Σ t̂_k) in some order. By Higham Lemma 3.5, |t̂_k − a_k v_k| ≤ √2γ₂|a_k||v_k| ≤
γ₃|a_k||v_k| (√2·2u(1 + u) ≤ 3u/(1 − 3u)). A complex addition rounds each component with relative error ≤ u, which
is a relative error ≤ u in modulus; unwinding the summation tree, fl(Σ t̂_k) = Σ t̂_k(1 + θ_k), |θ_k| ≤ γ_{m−1}.
Hence |fl(Σ a_k v_k) − Σ a_k v_k| ≤ Σ|t̂_k − a_k v_k| + Σ|t̂_k||θ_k| ≤ (γ₃ + (1 + γ₃)γ_{m−1})Σ|a_k||v_k| ≤
γ_{m+2}Σ|a_k||v_k| (Lemma 3.3: γ₃ + γ_{m−1} + γ₃γ_{m−1} ≤ γ_{m+2}). ✓ (For m = 1 there is no addition and the
bound γ₃ ≤ γ₃ holds; for real v_k the product is cheaper, the bound stays.)

Lemma A. z = x + iy. If y = 0 (resp. x = 0) the modulus is |x| (resp. |y|), exact. If both components are ≥ 2⁻⁵⁰⁰
in magnitude, x² and y² are ≥ 2⁻¹⁰⁰⁰, normal, and no operation underflows; with four roundings
s = fl(√fl(fl(x²) + fl(y²))) ≥ √((x² + y²)(1 − u)²)(1 − u) = |z|(1 − u)², and up(s) = fl(s(1 + 8u)) ≥ |z|(1 − u)³(1 + 8u) ≥ |z|
((1 − u)³(1 + 8u) = 1 + 5u − O(u²) ≥ 1). If an operation overflows, s = up(s) = ∞ ≥ |z| (a valid, useless bound;
(H3) excludes it). If a nonzero component is below 2⁻⁵⁰⁰: |x| + |y| ≥ √(x² + y²) = |z|, fl(|x| + |y|) ≥
(|x| + |y|)(1 − u) (an addition of two non-negative numbers has no additive underflow error: a sum below 2⁻¹⁰²² is
exact), and up(·) ≥ (|x| + |y|)(1 − u)²(1 + 8u) ≥ |z|. ✓

(2) |Σ a_k tr(PO_k) − val| ≤ |Σ a_k tr(PO_k) − Σ a_k v_k| + |Σ a_k v_k − val| ≤ Σ|a_k|b_k + γ_{m+2}Σ|a_k||v_k| by (1) and
Lemma B. It remains to show B ≥ the right side. First term: each up(|a_k|^↑ b_k) = fl(fl(|a_k|^↑ b_k)(1 + 8u)) ≥
|a_k|^↑ b_k (1 − u)²(1 + 8u) ≥ |a_k| b_k (Lemma A, b_k ≥ 0), and `add_up` rounds every partial sum towards +∞, so
Σ^↑_k up(|a_k|^↑ b_k) ≥ Σ_k |a_k| b_k (an upward-rounded sum of upper bounds is an upper bound; induction on k).
Second term: likewise Σ^↑ up(|a_k|^↑|v_k|^↑) ≥ Σ|a_k||v_k| =: S; `gamma(m + 2)` = γ_{m+2}(1 + θ₂) ≥ γ_{m+2}(1 − 2u − 4u²)
(two roundings, (m + 2)u exact); 2·γ̂ is exact; fl(2γ̂·sum_av) ≥ 2γ̂·sum_av(1 − u); up(·) adds (1 + 8u) after one
more rounding. So rnd ≥ 2γ_{m+2}S(1 − 2u − 4u²)(1 − u)²(1 + 8u) ≥ 2γ_{m+2}S(1 + 3u) ≥ γ_{m+2}S. The final
`add_up(bound, rnd)` ≥ bound + rnd. Hence B ≥ Σ|a_k|b_k + γ_{m+2}Σ|a_k||v_k| ≥ |Σ a_k tr(PO_k) − val|. ✓ ∎

The factor 2 in the rounding term is slack (the proof needs γ_{m+2}(1 + 3u) at most); (H3) is used for the
absence of overflow/underflow in the products |a_k|^↑ b_k, |a_k|^↑|v_k|^↑ and in val's products (an underflowing
product in val would add an absolute error ≤ u_min per component, not covered by (2)'s relative term — the
hypothesis excludes it, and the magnitudes of observables and coefficients make it irrelevant in practice).

### Uses

Two-point functions, the Hermitian decomposition (a = (1, i): B = add_up(b_H, b_K) ≥ b_H + b_K and val = v_H + i v_K
exactly — (2) with no rounding term needed), the per-state bound up(b/g) ≥ b/g (one rounding), and the structure
factor (coefficients fl(e^{iθ})/N as stored, D1) are instances of (1)–(2). The two documented hypotheses D1
(libm phases) and D2 (stored product matrices) are correctly labelled as hypotheses, not theorems.

### Code (current `correlations.py`; the lines of section 3 are shifted by one: `modulus_upper` 130–142,
`combine_certified` 145–164, `val` 156, loop 159–162, `rnd` 163, return 164)

Directions: every bound quantity rounded towards +∞ (`up`, `add_up`); `val` is round-to-nearest, as (2) assumes.
`modulus_upper` uses no libm call (√ is IEEE correctly rounded). ✓

## Comparison with section 2 of the original (read after the proof above was written)

Identical step by step: (1) triangle inequality; Lemma A with the same two branches and the same polynomial
(1 − u)³(1 + 8u) ≥ 1; Lemma B with √2γ₂ ≤ γ₃ and Lemma 3.3 (the original adds the FMA reference of Jeannerod et
al. 2017); (2) with the same accounting of `gamma()`'s two roundings and the factor 2 to spare; the uses
(per-state bound, Hermiticity of O_iO_j for distinct sites, C_ji = conj C_ij for S⁺S⁻, structure factor). Same
constants. Both valid.

**Code mapping**: section 3's line numbers are off by one on the current file (`modulus_upper` 130–142,
`combine_certified` 145–164); otherwise exact. `add_up(0, 0)` returns 2⁻¹⁰⁷⁴ for m = 0, a valid (non-zero) bound.

**Verdict: AGREES.**

## Runs

- `verify_P15.py --random 200 --seed 7`: A 200 + 32, B modulus 211, C rounding 50 (worst error/bound 0.23);
  461 random + 32 adversarial; 0 violations.
- `adversarial_P15_summed_bounds.py --cases 120 --seed 41`: Lemma A 60 random + 300 adversarial (2⁻⁵⁰⁰
  threshold, subnormal, 10²⁰⁰, zero component, modulus just below a rounding midpoint), Lemma B 20 + 100 (m up to
  20000, cancellation, scales 10^±100, complex v), (2) 20 + 101; 0 violations. (The first run at 300 cases
  reported one "violation" that was my test's: m = 0 gives B = 2⁻¹⁰⁷⁴ ≥ 0, correct.)
