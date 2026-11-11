# D-03 (draft): error-free residual and Gram products by integer splitting (Ozaki scheme)

Draft derivation document in the D-59 format for roadmap item D-03. **UNVERIFIED (D-12):** the module it describes,
`certify_ed/ozaki.py`, is a prototype that `certify()` does not call; it is reached only through direct calls,
`enclosure.enclose(..., residual_report=ozaki.residual_report(H, mu, X))` and `experiments/d03_ozaki_tightness.py`.
No bound or status of the package changes.

Notation as in `docs/theory.md` §1 and `thm1_fp-accounting.md`: u = 2⁻⁵³; F the set of binary64 numbers; fl(·) one
round-to-nearest operation; γ_k = ku/(1 − ku); `ULP_MIN` = 2⁻¹⁰⁷⁴; H the stored n × n matrix (dense, or sparse with
w_i stored entries in row i, w := max_i w_i); X the n × k stored claimed vectors, μ the k claimed values; R := HX −
X diag(μ) the exact residual of the stored data. For an integer t and a vector a with max|a| < 2^t, "a has exponent t".

## 1 Statement

**Why.** The default Theorem-1 bound (`enclosure.residuals`) is ‖fl(R)‖ + ‖E‖ with the componentwise rounding bound
E = γ̇_{w+2}(|H||X|/(1 − γ_{w+1})(1 + 4u) + |X||μ|)(1 + 4u). E does not shrink when the vectors get better: on the
E2 points of the 3.2.1 run it is of the same size as the residual itself (section 4). The default Gram path of a
cluster (`fp.exact_dot`: TwoProduct and a correctly rounded sum, about 110 ns per term, `cost.RATES`) is
accurate but costs g(g + 1)/2 Python-level inner products. The Ozaki scheme (Ozaki, Ogita, Oishi and Rump, Numer.
Algorithms 59, 2012) computes both products with no rounding error at all, on BLAS and SciPy kernels.

**Error model.** The standard model for every operation whose exact result is normal or zero, and fl(z) = z for
every z ∈ F (an operation whose exact result is a binary64 number is exact, in the normal and in the subnormal
range). BLAS and SciPy's CSR kernels evaluate each entry of a product as a sum of the exact products of its
operands, by multiplications, additions or FMAs in any order (no Strassen-type algorithm), the hypothesis already
recorded in `fp.float_env_record`.

**Hypotheses (checked by the code, `NotApplicable` otherwise).** (H1) all data finite; (H2) every entry of
magnitude < 2^480; (H3) n ≤ 2²⁶ and w ≤ 2²⁶ (the range of finding L-3); (H4) slices are formed only with units
s ≥ −537 (`MIN_UNIT`), so that every pair of units multiplied satisfies s + s′ ≥ −1074; an entry whose next slice
would have a smaller unit keeps its remainder, which is bounded, not summed (the pair check is kept as a second
guard).

**Construction.** β(L) := ⌊(53 − ⌈log₂ L⌉)/2⌋ (`split_beta`), the largest β with L·2^{2β} ≤ 2⁵³. A vector a with
exponent t is split into slices a_1, …, a_K and a remainder r_K: r_0 = a, s_p = t − pβ, a_p = 2^{s_p}·rint(r_{p−1}
2^{−s_p}), r_p = r_{p−1} − a_p, each step one `ldexp`, one `rint`, one `ldexp` and one subtraction in binary64.
Columns of X share one exponent per column (`split_columns`; real and imaginary parts one exponent per column,
`split_complex_columns`), rows of H one exponent per row (`split_rows`, on the stored entries), and each μ_c its
own exponent. Gram: β = β(n); residual: β = β(w).

**Theorem G (Gram).** For X ∈ F^{n×g}, Y ∈ F^{n×h} (complex: both parts in F) under (H1)–(H4), `ozaki_gram(X, Y)`
returns G and E with |G_ab − (X^H Y)_ab| ≤ E_ab for every a, b. When no remainder is left (`exhausted`), E_ab =
O(u|G_ab| + γ²_{T−1}Σ|terms|), T the number of slice products per entry.

**Theorem R (residual).** For a stored dense or sparse H, real μ and X under (H1)–(H4), `ozaki_residual(H, μ, X)`
returns R̂ and E with |R̂_ic − R_ic| ≤ E_ic for every i, c, and `upper[c]` ≥ ‖H x_c − μ_c x_c‖₂, `rel_bound[c]` ≥
`upper[c]`/‖x_c‖₂ (the Theorem-1 radius of the pair, rounded upward as in `enclosure.residuals`). E has no term in
|H||X| + |X||μ| of first order in u; its leading term is u|R̂_ic|.

**Consequence.** `ozaki.residual_report` packs R̂, E and these bounds into an `enclosure.ResidualReport`; Theorems 1,
2 and 2′ (`enclose`, `_cluster_bound`) use only `upper`, `rel_bound`, the stored residual columns and `E_norm` with
the meaning "‖R̂_c − R_c‖₂ ≤ E_norm[c]", which Theorem R provides. The cluster bounds therefore hold unchanged with
the D-03 report.

## 2 Proof

**Lemma S1 (one slice).** Let r ∈ F, s ≥ −1074 an integer, β ≥ 1, |r| < 2^{s+β}. Then m := rint(fl(r·2^{−s})) is
the integer nearest to the exact r·2^{−s}, |m| ≤ 2^β, and m·2^s ∈ F, so x := fl(m·2^s) = m·2^s and |r − x| ≤ 2^{s−1}.

*Proof.* No overflow: |r·2^{−s}| < 2^β. If |r·2^{−s}| ≥ 2⁻¹⁰²², the scaling by a power of two is exact. Otherwise
the scaled value, exact or rounded, is below 2⁻¹⁰²² < 1/2 and rint gives 0, the nearest integer of the exact value
too. |m| ≤ |r·2^{−s}| + 1/2 < 2^β + 1/2, hence |m| ≤ 2^β. Write m = m′2^j with m′ odd, |m′| < 2⁵³: m·2^s = m′2^{s+j}
with s + j ≥ −1074 and magnitude < 2^{1024}, a binary64 number (normal, or a subnormal multiple of 2⁻¹⁰⁷⁴). The last
claim is the nearest-integer property. ∎

**Lemma S2 (exact remainder).** In the setting of S1, r − x ∈ F, so fl(r − x) = r − x.

*Proof.* If m = 0, r − x = r. Otherwise |r| ≥ 2^{s−1}. Let 2^e be the spacing of F at r: for normal r with 2^E ≤
|r| < 2^{E+1}, e = E − 52 ≥ s − 53; for subnormal r, e = −1074 and |r| < 2⁻¹⁰²² gives s < −1021, so again e ≥ s −
53. If e ≥ s, r·2^{−s} is an integer and x = r. If e < s, both r and x are multiples of 2^e, so r − x = d·2^e with an
integer |d| ≤ 2^{s−1−e} ≤ 2⁵², and d·2^e ∈ F. ∎

**Lemma S3 (split).** For a vector with exponent t and slices p = 1, …, K: a = a_1 + … + a_K + r_K exactly; where
s_p = t − pβ ≥ −537, (a_p)_i = m 2^{s_p} with |m| ≤ 2^β and |(r_p)_i| ≤ 2^{s_p−1}; where s_p < −537, a_p = 0 and
r_p = r_{p−1} (the slice is not formed, and since s_p decreases with p it is not formed later either).

*Proof.* Induction on p with S1 and S2 for the formed slices: |r_0| < 2^t = 2^{s_1+β}; for p ≥ 2, |r_{p−1}| ≤
2^{s_{p−1}−1} = 2^{s_p+β−1} < 2^{s_p+β}. Each r_p = r_{p−1} − a_p is computed exactly (S2, or a_p = 0), so the
identity telescopes. ∎

**Lemma P (exact slice products).** Let a_l = m_l 2^σ, b_l = m′_l 2^τ (l = 1, …, L) with integers |m_l|, |m′_l| ≤
2^β, L·2^{2β} ≤ 2⁵³, σ + τ ≥ −1074 and σ + τ + 53 < 1024. Every evaluation of Σ_l a_l b_l by products, additions and
FMAs in any order returns the exact sum.

*Proof.* Every exact product is an integer of magnitude ≤ 2^{2β} times 2^{σ+τ}; every partial sum of any subset is
an integer of magnitude ≤ L·2^{2β} ≤ 2⁵³ times 2^{σ+τ}, a binary64 number. By induction over the evaluation tree
every operation has an exact result in F and returns it. ∎

For H X the L = w_i terms of entry (i, c) of a slice product H_q X_p share the unit σ_iq + τ_cp (row exponent of H,
column exponent of X); β = β(w) gives w·2^{2β} ≤ 2⁵³. For X_p^T Y_q the n terms of entry (a, b) share τ_ap + τ′_bq;
β = β(n). The terms μ_c x_ic are single products (L = 1). (H2) and β ≥ 13 give σ + τ + 53 ≤ 2·(480 − 13) + 53 <
1024; (H4) gives σ + τ ≥ −1074 (and `_units_ok` checks it again on the smallest units of the two operands).

**Lemma A (Sum2).** For t_1, …, t_T ∈ F with Tu < 1, Sum2 (Ogita, Rump and Oishi, SIAM J. Sci. Comput. 26, 2005,
Algorithm 4.4: TwoSum chain, the errors added in floating point, one final addition) returns v with
|v − Σt| ≤ u|Σt| + γ²_{T−1}Σ|t_l| (their Proposition 4.5, cited, not re-derived here). Only additions occur, so
gradual underflow does not change the argument (an addition with a subnormal exact result is exact, TwoSum is
error-free for all finite inputs). From |v − S| ≤ u|S| + γ²A and |S| ≤ |v| + |v − S|: |v − S| ≤ (u|v| + γ²A)/(1 − u).
The code forms A_c = fl(Σ|t_l|) by T − 1 additions, A ≤ A_c/(1 − γ_{T−1}), and the bound
(u(1 + 2u)|v| + c_T A_c)(1 + 8u) + 2·2⁻¹⁰⁷⁴ with c_T = `sum2_coefficient(T)` ≥ γ²_{T−1}/((1 − γ_{T−1})(1 − u))
(checked exactly in rational arithmetic for T = 2, …, 3000 and T = 10⁴, 10⁵, 10⁶); u(1 + 2u) ≥ u/(1 − u); the factor
1 + 8u covers the four roundings of the line and 2·2⁻¹⁰⁷⁴ the underflow of its two products (`_Sum2.result`).

**Proof of Theorem G.** By S3, X = X_s + R_X with X_s = Σ_p X_p, and Y = Y_s + R_Y, exactly. Expanding,
X^H Y = X_s^H Y_s + X_s^H R_Y + R_X^H Y. The first term is Σ_{p,q} X_p^H Y_q; for real data each gemm X_p^T Y_q is
exact by Lemma P; for complex data Re = X_{r,p}^T Y_{r,q} + X_{i,p}^T Y_{i,q} and Im = X_{r,p}^T Y_{i,q} −
X_{i,p}^T Y_{r,q}, four exact gemms (the negation is exact). Lemma A bounds the error of summing the T exact terms
of each part; |z| ≤ |Re z| + |Im z| combines the two parts (one addition, guarded). The remainder satisfies
|X_s^H R_Y + R_X^H Y| ≤ |X_s|^T|R_Y| + |R_X|^T|Y| entrywise (moduli bounded by |Re| + |Im|). The code forms the
moduli and the two gemms of nonnegative numbers; the computed value is at least the exact one times (1 − γ_{n+3})
minus the underflow of at most 2n products per entry (≤ n·2⁻¹⁰⁷⁴), and the inflation (1 + γ_{n+3})(1 + 8u) +
2n·2⁻¹⁰⁷⁴ restores an upper bound because (1 − γ²_{n+3})(1 + 8u)(1 − u)² ≥ 1 for n ≤ 2²⁶ (first order 6u − γ²_{n+3}
> 0) (`_remainder_gram`). The final addition is guarded by 1 + 8u. ∎

**Proof of Theorem R.** With H = H_s + R_H, X = X_s + R_X, μ = μ_s + r_μ (S3, exactly),

    HX − X diag(μ) = [H_s X_s − X_s diag(μ_s)] + [H R_X + R_H X_s − X_s diag(r_μ) − R_X diag(μ)],

which follows from H X = H_s X_s + H R_X + R_H X_s and X diag(μ) = X_s diag(μ_s) + X_s diag(r_μ) + R_X diag(μ). The
first bracket is Σ_{q,p} H_q X_p − Σ_{p,r} X_p diag(μ_r): every sparse (or dense) product H_q X_p is exact by Lemma P
with L = w_i terms, every product (X_p)_ic(μ_r)_c is exact (L = 1); complex H or X add the products H_{r,q}X_{r,p} −
H_{i,q}X_{i,p} and H_{r,q}X_{i,p} + H_{i,q}X_{r,p}, each exact. Lemma A bounds the summation of the T terms of each
entry. The second bracket is bounded entrywise by |H||R_X| + |R_H||X_s| + |X_s||r_μ| + |R_X||μ|; the computed value of
this nonnegative expression is at least the exact one times (1 − γ_{w+6}) (at most w products and w − 1 additions
per matvec, three additions, moduli rounded once each for complex data) minus the underflow of at most 2w + 2
products, and the code inflates by (1 + γ_{w+6})(1 + 8u)² + (2w + 4)·2⁻¹⁰⁷⁴ (as for Theorem G, valid for w ≤ 2²⁶).
Finally ‖R_c‖₂ ≤ ‖R̂_c‖₂ + ‖R̂_c − R_c‖₂ ≤ `norm2_upper`(R̂_c) + `norm2_upper`(E_c), and the sum, the division by
`norm2_lower`(x_c) and the product with 1 + 4u are covered exactly as in step 5 of `thm1_fp-accounting.md` (three
roundings, (1 + 4u) ≥ (1 − u)⁻³, + 2·2⁻¹⁰⁷⁴ for a subnormal quotient or product). ∎

**What the theorems do not need.** No hypothesis on the producer, on how BLAS orders or blocks its sums, or on
FMA; the γ_n inner-product model of `fp.py` is used only for the remainder terms, which vanish when the split is
complete (every E2 point below).

## 3 Where the code implements it

| proof object | code (`certify_ed/ozaki.py`) | rounding direction |
|---|---|---|
| β(L), (H3) | `split_beta`, line 56 | exact integer arithmetic |
| exponents t | `_exponent`, line 68 (frexp) | exact |
| slices and remainder (S1, S2, S3), the floor 2⁻⁵³⁷ of (H4) | `_slice_values`, lines 106 and 113 | exact |
| column / row splits | `split_columns` 128, `split_complex_columns` 142, `split_rows` 199 | exact |
| unit-pair check (second guard of (H4)) | `_units_ok`; raised at lines 363, 411, 489 | refusal |
| Sum2 (Lemma A) and its bound | `_Sum2.add` 267, `_Sum2.result` 280, `sum2_coefficient` 290 | upward |
| Gram slice products (Lemma P) | `ozaki_gram`, line 375 | exact |
| Gram remainder | `_remainder_gram`, line 337 | upward |
| residual slice products, μ terms | `ozaki_residual`, lines 510 and 518 | exact |
| residual remainder | `ozaki_residual`, line 542 | upward |
| Theorem-1 upper bound and radius | `ozaki_residual`, lines 549 and 551 | upward |
| hook into Theorems 1, 2, 2′ | `residual_report` 558; `enclosure.enclose(..., residual_report=...)` | as the default path |

## 4 Computational verification

`docs/proofs/verify_D-03.py` (from the repository root, `python docs/proofs/verify_D-03.py`; record
`docs/proofs/runs/verify_D-03.txt`): every quantity is compared with exact rational arithmetic (python-flint `fmpq`,
`fmpq_mat`; complex entries compared by exact squared moduli).

- **Random cases (1000):** splits (sum of slices plus remainder equal to the input, slice entries integers of at
  most β bits times their unit, every slice product equal to its exact value), Gram matrices, column dots and
  residuals of sparse and dense, real and complex Hermitian matrices with dyadic and non-dyadic entries, exact and
  perturbed eigenvectors, n up to 80, magnitudes from 2⁻³⁰⁰ to 2³⁰⁰ and columns spanning 40 orders of magnitude,
  1 to 8 slices (so that remainders are exercised); for residuals also `upper` against the exact ‖r_c‖₂.
- **Adversarial cases (14):** worst-case slice magnitudes 2^β − 1 at n = 2000; entries just below 2⁴⁸⁰; exactly
  orthogonal columns; subnormal entries; a zero column; exact rint ties; purely imaginary data; 53-bit significands;
  columns below 2⁻⁵³⁷ (remainder only); exact eigenvectors (residual exactly zero); a dense row width; the scalar
  grids of Lemma P (β rule for L up to 2²⁶) and of Lemma A (c_T). Two inputs outside (H1) or (H2) are refused, as
  required.
- **Negative controls (4), each detected:** slices four bits wider than β(n) make the BLAS inner product round; the
  plain floating-point residual violates a zero error bound; the remainder bound of a truncated split (one slice)
  shrunk 1000-fold is violated; the same for the Gram bound.
- **Result:** 1000 random + 14 adversarial cases, 2 refused (hypothesis not met), 4 negative controls detected,
  **0 violations** (35 s). `tests/test_ozaki.py` (14 tests) repeats the main checks in the suite.

**Tightness and cost on the E2 points of the 3.2.1 run** (`experiments/d03_ozaki_tightness.py`, record
`docs/proofs/runs/D-03_tightness.json`, minimum of three runs, one thread). The producer of `experiments/e2_cost.py`
is repeated in-process with the seeds of the run; the default S2 radii reproduce the run's records
(`results/run_full_20261105/e2_cost_20261105_142518`): `max_radius` to the last bits (the U(1) points differ by the
rounding of the final addition of the model shift, below 10⁻¹⁵ relative) and the sector-block radius of every k = 0
point to its printed three digits. "D-03" is the radius with the report of `ozaki.residual_report`.

| E2 point | n | w | k | Theorem 1 radius, default / D-03 | max enclosure radius, default / D-03 | S2 radius with model shift, default / D-03 | residual s, default / D-03 | Gram s, exact_gram / D-03 | slices of X, H |
|---|---|---|---|---|---|---|---|---|---|
| chain N=12 u1 J2=0.0 | 924 | 13 | 5 | 4.220e-14 / 2.617e-14 | 6.305e-14 / 4.023e-14 | 2.129e-13 / 1.901e-13 | 0.0012 / 0.0132 | 0.0140 / 0.0084 | 5, 1 |
| chain N=14 u1 J2=0.0 | 3432 | 15 | 5 | 3.450e-14 / 1.337e-14 | 7.750e-14 / 2.760e-14 | 2.803e-13 / 2.304e-13 | 0.0026 / 0.0272 | 0.0143 / 0.0109 | 5, 1 |
| chain N=16 u1 J2=0.0 | 12870 | 17 | 5 | 5.738e-14 / 2.838e-14 | 7.894e-14 / 2.838e-14 | 3.427e-13 / 2.922e-13 | 0.0071 / 0.1381 | 0.0394 / 0.0517 | 5, 1 |
| chain N=18 u1 J2=0.0 | 48620 | 19 | 5 | 6.028e-14 / 2.374e-14 | 1.134e-13 / 2.812e-14 | 4.461e-13 / 3.609e-13 | 0.1163 / 2.2379 | 0.3738 / 0.2671 | 5, 1 |
| chain N=16 k0 J2=0.0 | 810 | 13 | 4 | 3.214e-14 / 1.246e-14 | 3.214e-14 / 1.246e-14 | 3.214e-14 / 1.246e-14 | 0.0009 / 0.0102 | 0.0052 / 0.0043 | 5, 3 |
| chain N=18 k0 J2=0.0 | 2704 | 15 | 4 | 4.477e-14 / 1.934e-14 | 4.477e-14 / 1.934e-14 | 4.477e-14 / 1.934e-14 | 0.0022 / 0.0289 | 0.0072 / 0.0068 | 5, 3 |
| chain N=20 k0 J2=0.0 | 9252 | 17 | 4 | 5.462e-14 / 1.997e-14 | 5.462e-14 / 1.997e-14 | 5.462e-14 / 1.997e-14 | 0.0132 / 0.1261 | 0.0253 / 0.0260 | 5, 3 |
| chain N=22 k0 J2=0.0 | 32066 | 19 | 4 | 9.701e-14 / 5.206e-14 | 9.701e-14 / 5.206e-14 | 9.701e-14 / 5.206e-14 | 0.0193 / 0.5600 | 0.0927 / 0.0901 | 5, 3 |
| chain N=24 k0 J2=0.0 | 112720 | 21 | 4 | 9.552e-14 / 4.567e-14 | 9.552e-14 / 4.567e-14 | 9.552e-14 / 4.567e-14 | 0.0906 / 2.2064 | 0.6598 / 0.4665 | 5, 3 |
| square N=16 u1 J2=0.0 | 12870 | 33 | 7 | 1.205e-13 / 4.879e-14 | 3.312e-13 / 1.236e-13 | 1.370e-12 / 1.163e-12 | 0.0118 / 0.1931 | 0.0737 / 0.0668 | 6, 1 |
| square N=16 u1 J2=0.5 | 12870 | 49 | 4 | 1.604e-13 / 5.168e-14 | 2.517e-13 / 6.026e-14 | 2.658e-12 / 2.466e-12 | 0.0152 / 0.1554 | 0.0331 / 0.0286 | 5, 1 |

The Theorem-1 radius shrinks to 0.32 to 0.62 of the default, the largest enclosure radius (Theorems 2 and 2′ for
the degenerate clusters) to 0.24 to 0.64. The S2 radius a U(1) certificate reports adds the model shift of S1,
which the D-03 products do not touch, so there the ratio is 0.81 to 0.93; on the k = 0 sector blocks (no shift) it
is 0.37 to 0.54. Every split was complete (no remainder): 5 or 6 slices of X, 1 slice of H for the dyadic U(1) blocks
and 3 for the k = 0 blocks, whose entries carry square roots of orbit sizes. The residual costs 9.5 to 29 times the
default one, at most 2.24 s (n = 48620 and 112720); the run's records have S2 times of 0.0024 to 0.14 s and S3
times of 0.31 to 251 s at these points.
The Gram defect η is unchanged (7.77 × 10⁻¹⁵ against 7.52 × 10⁻¹⁵ at the first point; both are accurately rounded),
and on these clusters (g = 4 to 7) the D-03 Gram takes 0.76 to 1.66 times the time of `exact_gram`. The claim of the
roadmap item ("10 to 50 times faster than fsum paths") holds only for wide clusters (random unit columns):

| n | g | exact_gram s | D-03 Gram s | ratio | slices |
|---|---|---|---|---|---|
| 924 | 2 | 0.0016 | 0.0024 | 0.7 | 4 |
| 924 | 8 | 0.0169 | 0.0036 | 4.7 | 4 |
| 924 | 32 | 0.2651 | 0.0108 | 24.6 | 4 |
| 12870 | 2 | 0.0068 | 0.0136 | 0.5 | 4 |
| 12870 | 8 | 0.0837 | 0.0330 | 2.5 | 4 |
| 12870 | 32 | 1.4821 | 0.2366 | 6.3 | 4 |
| 112720 | 2 | 0.1778 | 0.1636 | 1.1 | 4 |
| 112720 | 8 | 2.2848 | 0.6122 | 3.7 | 5 |
| 112720 | 32 | 37.8785 | 3.2731 | 11.6 | 5 |

What the measurements show: on these points the residual error bound E of the default path is the larger part of
the Theorem-1 radius, and an error-free evaluation removes it at a cost that is small against S3. What they do not
show: any gain for spec or matrix-free operators (not supported), for S3 (unchanged), or on another machine.

## 5 Analytic cross-check

(i) *Exact data.* If H, X and μ have at most β significant bits per slice unit (for example H with entries in
2⁻²Z and X = e_j), the split has one slice, every product is exact, T = 1 or 2 and E = u|R̂| + O(2⁻¹⁰⁷⁴): for an
exact eigenvector R̂ = 0 and the radius is 2·2⁻¹⁰⁷⁴ (the adversarial "exact eigenvectors" case).

(ii) *Reduction to the default bound.* With K = 0 slices everything is remainder: R_X = X, R_H = H, r_μ = μ, and
the remainder bound becomes |H||X| + |H||X| + |X||μ| + |X||μ| times (1 + γ_{w+6}): the structure of the default E
(twice its leading term), so the draft contains the default accounting as its worst case.

(iii) *Gram of one vector.* For g = h = 1 and a complete split, G = fl(Σ_{p,q} x_p^T x_q) by Sum2 and E ≈ u|G|,
the same accuracy as `fp.exact_dot` (2u|s|).

## 6 Status

Derived 11 November 2026 (draft); machine-checked (1000 random + 14 adversarial cases, 4 negative controls detected,
0 violations); independent re-derivation: not done; Sarang's check (D-12): pending. UNVERIFIED: not used by
`certify()`; no default changes.
