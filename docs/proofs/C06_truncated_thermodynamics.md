# Proposition 16' (C-06): thermodynamic intervals from the k lowest eigenvalues and a bound on the rest

Implements: `certify_ed/thermo.py` (`thermodynamics_truncated`, the `tail` branch of `_rows`) and the S7 branch of
`certify_ed/certify.py` for 'lowest' claims. Depends on Proposition 16 (`docs/proofs/P16_thermo_intervals.md`),
Lemma 17 (`docs/proofs/L17_dyadic_exp_ln.md`), Corollary 5 / the S3 completeness statement for 'lowest' claims
(`docs/theory.md`, Section 3) and the operator norm bound of `fp.inf_norm_upper`. Notation as in
`docs/theory.md`, Section 9.3.

## 1. Statement

**Hypothesis (H').** A is Hermitian of dimension n with eigenvalues E_1 ≤ … ≤ E_n. Given are
(a) m intervals [l_j, h_j] with counts c_j ≥ 1, Σ_j c_j = k < n, such that E_1, …, E_k lie in the intervals with
exactly c_j of them in interval j (the S2 enclosures of a 'lowest' claim, counted by S3);
(b) a number L with h_j < L for all j and L ≤ E_{k+1} (the S3 statement `next_eigenvalue_lower_bound`:
"exactly k eigenvalues below L");
(c) the dimension n;
(d) optionally a number M ≥ E_n with M ≥ L (the operator norm bound ‖A‖ ≤ M, widened by the model shift ε).
l_j, h_j, L, M are finite binary64 numbers; k_B = 1; T > 0 is a binary64 with T ≥ 2^−192.

**Notation.** E_0 = min_j l_j, e = E − E_0, a_j = l_j − E_0, b_j = h_j − E_0, ℓ = L − E_0 > 0, μ = M − E_0 ≥ ℓ
(μ = ∞ when M is absent). φ(e) = e e^{−βe}, ψ(e) = e² e^{−βe}. Z, F, U, Var(E), C as in Proposition 16.

**Lemma C (envelopes).** For β > 0, φ is increasing on [0, 1/β] and decreasing on [1/β, ∞) with φ(1/β) = e^{−1}/β;
ψ is increasing on [0, 2/β] and decreasing on [2/β, ∞) with ψ(2/β) = 4e^{−2}/β². Both are positive on (0, ∞)
and tend to 0 at ∞. Hence for 0 < ℓ ≤ μ ≤ ∞,
  sup_{[ℓ, μ]} φ = φ(clamp(1/β, ℓ, μ)),  inf_{[ℓ, μ]} φ = min(φ(ℓ), φ(μ)) (= 0 if μ = ∞),
and the same for ψ with 2/β.

**Proposition 16' (tail bound and intervals).** Under (H'):
(i) *Omitted weight.* Σ_{j>k} e^{−βE_j} ≤ (n − k) e^{−βL}; more precisely, in shifted form,
  Σ_{j>k} e^{−βe_j} ∈ (n − k) · [e^{−βμ}, e^{−βℓ}]   (lower end 0 when μ = ∞).
(ii) *Omitted moments.* Σ_{j>k} e_j e^{−βe_j} ∈ (n − k) · [inf_{[ℓ,μ]} φ, sup_{[ℓ,μ]} φ] and
  Σ_{j>k} e_j² e^{−βe_j} ∈ (n − k) · [inf_{[ℓ,μ]} ψ, sup_{[ℓ,μ]} ψ].
(iii) *Intervals.* With W_j = c_j · exp(−β̂·[a_j, b_j]) as in Proposition 16 and the tail intervals
  W_t = (n − k) · [exp(−β̂ μ̂)^↓, exp(−β̂ ℓ̂)^↑],  T_1 = (n − k) · [φ_inf, φ_sup],  T_2 = (n − k) · [ψ_inf, ψ_sup]
(each end an outward-rounded enclosure of the quantity in (i)–(ii) by Lemma 17; the sup/inf evaluated by the
rule of Lemma C on enclosures ℓ̂ ∋ ℓ, μ̂ ∋ μ, T̂ = [T, T]):
  Z̃ = Σ_j W_j + W_t ∋ Z e^{βE_0},   F ∈ E_0 − T̂ · ln Z̃,   U ∈ E_0 + (Σ_j W_j·[a_j, b_j] + T_1) / Z̃,
  Var(E) ∈ ((Σ_j W_j·[a_j, b_j]² + T_2) / Z̃ − ((Σ_j W_j·[a_j, b_j] + T_1) / Z̃)²) ∩ [0, ∞),   C ∈ β̂² · Var-enclosure,
and ln Z, Z as in Proposition 16. All four quantities are enclosed on both sides, with or without M.

**Looseness (stated, not proved as a bound).** The widths are governed by the omitted terms. For T ≤ ℓ the tail
contributes at most (n − k) e^{−βℓ} to Z̃ relative to c_1 (the ground-level weight is exactly c_1 at e = 0), and
(n − k) φ(ℓ) = (n − k) ℓ e^{−βℓ} to the first moment; the intervals are therefore informative when
(n − k) e^{−ℓ/T} ≪ 1, i.e. T ≪ ℓ / ln(n − k). For T > ℓ the suprema are at 1/β and 2/β and the U, C intervals are
wide (of order T and T²·β² = O(1) respectively); the record says so (`TAIL_LOOSENESS`). This is the answer to the
question "can C be bounded two-sidedly from this information": yes — the lower bound of Var(E) is ≥ 0 and the
upper bound is finite for every T — but only at low temperature are the bounds narrow.

## 2. Proof

**Lemma C.** φ'(e) = (1 − βe) e^{−βe}, positive for e < 1/β and negative for e > 1/β; ψ'(e) = e(2 − βe) e^{−βe},
positive for 0 < e < 2/β and negative for e > 2/β. Positivity on (0, ∞) and the limits at ∞ are elementary
(e^{−βe} decays faster than any power grows). A function increasing then decreasing on [ℓ, μ] attains its
supremum at the turning point if that lies inside, else at the end nearer to it (the clamp), and its infimum
at one of the two ends; when μ = ∞ the infimum is the limit 0, not attained. ∎

**(i).** Each omitted eigenvalue satisfies E_j ≥ E_{k+1} ≥ L (by (b)) and E_j ≤ E_n ≤ M (by (d)), so e_j ∈ [ℓ, μ];
e ↦ e^{−βe} is decreasing, so e^{−βe_j} ∈ [e^{−βμ}, e^{−βℓ}], and the n − k terms sum into (n − k) times that
interval. Undoing the shift gives Σ_{j>k} e^{−βE_j} ≤ (n − k) e^{−βE_0} e^{−βℓ} = (n − k) e^{−βL}. ∎

**(ii).** e_j ∈ [ℓ, μ] and Lemma C: φ(e_j) ∈ [inf φ, sup φ] over [ℓ, μ], likewise ψ; sum the n − k terms. ∎

**(iii).** The proof of Proposition 16 is unchanged except that the sums over all eigenvalues are split into the
k claimed ones (enclosed termwise as there) and the n − k omitted ones (enclosed by (i)–(ii)):
  Z e^{βE_0} = Σ_{i≤k} e^{−βe_i} + Σ_{j>k} e^{−βe_j} ∈ Σ_j W_j + W_t,
  Σ_i e_i e^{−βe_i} ∈ Σ_j W_j·[a_j, b_j] + T_1,   Σ_i e_i² e^{−βe_i} ∈ Σ_j W_j·[a_j, b_j]² + T_2,
and the quotients, the difference, the intersection with [0, ∞), the products with β̂ and T̂, ln and the final
conversion are as in Proposition 16 (Lemma 17 throughout). It remains to check that W_t, T_1, T_2 as computed
enclose the sets of (i)–(ii). *Ends of W_t.* exp(−β̂ ℓ̂) is an enclosure of e^{−βℓ} (Lemma 17 (iii)), so its upper
end is ≥ e^{−βℓ}; exp(−β̂ μ̂)'s lower end is ≤ e^{−βμ}; without M the lower end is 0 ≤ e^{−βμ}. *Suprema.* The
code decides the regime by comparisons that are *conservative*: it uses φ(ℓ) as the supremum only when
ℓ̂_lo ≥ T̂_hi, which implies ℓ ≥ T = 1/β, the case where Lemma C gives sup φ = φ(ℓ); it uses φ(μ) only when
μ̂_hi ≤ T̂_lo, which implies μ ≤ 1/β and sup φ = φ(μ); otherwise it uses e^{−1}/β = T e^{−1}, which is the global
maximum of φ on [0, ∞) and hence an upper bound for the supremum over any subinterval. In each case the value
used is an upper end of an enclosure (φ(ℓ) ∈ ℓ̂ · exp(−β̂ ℓ̂), T e^{−1} ∈ T̂ · exp([−1, −1]), the products by
Lemma 17 (ii)). The same with 2T, ψ and 4T² e^{−2} = ψ(2/β). *Infima.* With M: min(φ(ℓ), φ(μ)) ∈ the lower ends of
the two enclosures, and the lower end of an interval containing the minimum is ≤ the minimum; without M, 0 is a
lower bound. The scalings by (n − k) are exact (`scale_int`). The hypotheses of Lemma 17 (iv) hold because
Z̃'s lower end is positive (the ground-level term has exp(0) = 1 exactly), and T ≥ 2^−192 keeps 0 out of the
divisor of β̂. ∎

*Classical results cited.* As in Proposition 16; Weyl's inequality for the model shift ε in L and M (the
S3 bound is already stated for the declared model; the norm bound of the stored operator is widened by ε, since
|λ_max(H)| ≤ ‖A‖ + ‖H − A‖); ‖A‖_2 ≤ ‖A‖_∞ for Hermitian A (Horn & Johnson, *Matrix Analysis*, 2nd ed., Cor
5.6.36 / the fact that the spectral radius is bounded by any induced norm) for the upper bound M.

**Reduction.** When k = n (no omitted eigenvalue) the code calls `thermodynamics` and Proposition 16' is
Proposition 16. When M is given and one instead adds the interval [L, M] with count n − k to the list and
applies Proposition 16 unchanged, the result is also correct (an eigenvalue of [L, M] contributes e^{−βe} with
e ∈ [ℓ, μ]), but weaker: the interval product [ℓ, μ]·[e^{−βμ}, e^{−βℓ}] has upper end μ e^{−βℓ}, far above sup φ
when μ ≫ ℓ. The envelopes of Lemma C are the sharp version of that product, which is why the code uses them
in both cases (observed on the N = 8 Heisenberg chain at T = 0.05: U width 1.3e−6 with the envelopes against
1.3e−5 with the extra interval; C width 5e−4 against 5e−2).

## 3. Where the code implements it

`certify_ed/thermo.py` (this revision):

* `_read_intervals` (reading and validation of the k intervals); `_rows(ivs, counts, temperatures, quantities,
  tail=None)`: the common core of Propositions 16 and 16'. With `tail = (m_tail, L, M)`: `ell` = ℓ̂ and `em` = μ̂
  from `from_fraction` of the exact differences (positivity of ℓ̂ checked), `e_m1`, `e_m2` = enclosures of e^{−1},
  e^{−2}. Per temperature: `w_l = exp_interval(-(beta * ell))`, `w_m` likewise; `phi_l = ell * w_l`, `psi_l =
  ell.square() * w_l`, `phi_m`, `psi_m`; the regime decision `ell.lo >= T_iv.hi` / `em.hi <= T_iv.lo` / else, with
  `(T_iv * e_m1).hi` for the global maximum, and the same with `two_T = T_iv.scale_int(2)` and `(T_iv.square() *
  e_m2).scale_int(4).hi`; `phi_inf`, `psi_inf` = 0 or the minimum of the lower ends; `W_tail = Interval(w_m.lo or 0,
  w_l.hi).scale_int(m_tail)`, `T1_tail`, `T2_tail`; `Zt = sum_intervals(W + [W_tail])`; `num1`, `num2` include
  `T1_tail`, `T2_tail`. The row carries `tail` = {omitted_count, weight_bounds, weight_upper,
  first_moment_bounds, second_moment_bounds, envelope_regime, weight_fraction_upper}. Rounding: every step is an
  `ivl.py` operation (outward); the regime comparisons use the conservative ends named in the proof.
* `thermodynamics_truncated(intervals, next_lower_bound, dimension, temperatures, upper_bound=None)`: validates
  k ≤ n, L > max h_j, M ≥ L; k = n delegates to `thermodynamics`; otherwise `_rows(..., tail=(n − k, L, M))`;
  the record carries `tail_bound` = {omitted_count, next_lower_bound, upper_bound, method ('envelope' or
  'envelope with upper bound'), weight_bound (the formula of (i)), statement}, `claimed_count`, the hypothesis
  `TAIL_HYPOTHESIS` and the looseness sentence.

`certify_ed/certify.py`, S7 block: for `claim.kind == "lowest"` with S3 CERTIFIED (`window[1]` finite: the
declared-model `next_eigenvalue_lower_bound`), eigenvectors and temperatures, `truncated_record` holds the
merged S2 intervals (`intervals_from_enclosure(enc, eps_s3)`), `next_lower_bound = window[1]`, `upper_bound =
add_up(adapter.norm_upper(), eps_s3)` (or None); the statement is CERTIFIED with the summary naming the tail
bound and the assumptions naming this document. A 'lowest' claim without S3 CERTIFIED keeps NOT_CHECKED /
NEEDS_WHOLE_SPECTRUM (the diagnostics sentence now mentions the C-06 route); without temperatures,
NEEDS_TEMPERATURES. Scope: `docs/SCOPE_MATRIX.md` row "thermodynamics"; `presets.py` unchanged (S7 is selected
by the temperatures option or the `whole-spectrum` / `full` presets); schema 2.1 declares `tail_bound` and
`claimed_count`.

## 4. Computational verification

`docs/proofs/verify_C06.py` (run: `python docs/proofs/verify_C06.py --random 300 --seed 0`):

* 300 random cases: dyadic spectra of 3–13 eigenvalues (so that the floats passed as interval ends, L and M are
  exactly the rationals of the hypothesis), k random, the k lowest as point or widened intervals, L anywhere in
  (h_k, E_{k+1}], M = E_n + random or absent; six temperatures spanning T/ℓ ∈ {0.05, 0.3, 0.9, 1.5, 3} and
  T = 2(M − E_0) (all three envelope regimes); for each temperature three spectra consistent with the intervals
  (lower ends, upper ends, random points) and the true omitted eigenvalues; ground truth mpmath interval
  arithmetic at 200 digits (as in verify_P16, with multiplicities for repeated values); check: the implementation's
  intervals for F, U, C, ln Z, Z meet the reference intervals.
* 150 adversarial cases: L = E_{k+1} and M = E_n exactly (saturated hypotheses), all omitted eigenvalues at L or
  all at M (saturating (i) at one end), k = n − 1, and 30 % with an additional 2^10 … 2^20 omitted eigenvalues at M
  (or at L without M) so that the omitted weight dominates.
* Negative controls (7): a planted L above E_{k+1}, a declared dimension smaller than the spectrum, an M below
  E_n — each makes the true values leave the intervals at some temperature; the honest version of the same case
  passes at every temperature; the three input validations (L not above the claim, M < L, k > n) raise.

Result (seed 0): 300 random + 150 adversarial cases (each checked at ~6 temperatures × 3 spectra), 0 violations;
7/7 negative controls behave as required.

One finding of the verification changed the harness, not the code: with a non-dyadic L the float passed to
the function rounded above E_{k+1} by one ulp and the omitted eigenvalue at L violated (b); the hypothesis is
about the float L, which S3 guarantees (its L is a rounded-down float), and the harness now uses dyadic spectra.

`tests/test_thermo_truncated.py` checks the implementation against full spectra of open Heisenberg chains
(N = 6, 8) and a free-fermion chain (N = 5) for every k ending at a gap, through `certify()` on the N = 8 chain
(S7 CERTIFIED with `tail_bound`, schema-valid body), and the negative controls (subset claim; S3 skipped; S3
not certified for a cut multiplet; planted wrong L; dimension too small; input validation).

## 5. Analytic cross-check

*Two levels, one omitted.* k = 1, E_1 = 0 (c_1 = 1), one omitted eigenvalue known to equal L = M = Δ: then
ℓ = μ = Δ, W_t = [e^{−βΔ}, e^{−βΔ}] up to rounding, T_1 = Δ e^{−βΔ}, T_2 = Δ² e^{−βΔ}, and Proposition 16'
reproduces the two-level closed form of Proposition 16 (Section 5 there) with point-interval widths (the
adversarial case "k = n − 1, omitted at L = M" is this configuration).

*Weight bound as a limit.* With M absent, W_t = (n − k)·[0, e^{−βℓ}]: Z̃ ∈ [Σ W_j, Σ W_j + (n − k) e^{−βℓ}], so
F ∈ [E_0 − T ln(Σ_j W_j + (n − k) e^{−βℓ}), E_0 − T ln Σ_j W_j]: the width of F is T ln(1 + (n − k) e^{−βℓ}/Σ_j W_j)
≤ T (n − k) e^{−ℓ/T}/c_1, which tends to 0 as T → 0 faster than any power of T — the textbook statement that the
low-temperature thermodynamics is determined by the low-lying levels up to exponentially small corrections,
here with an explicit, certified constant.

*High-temperature limit with M.* T → ∞: W_t → (n − k)·[1, 1], T_1 → (n − k)·[ℓ, μ] (both envelopes become the
end values), so U → mean of the claimed levels and of an unknown value in [ℓ, μ] per omitted eigenvalue: the
enclosure degenerates gracefully to "each omitted eigenvalue is somewhere in [L, M]".

*Comparison with the extra-interval method.* Section 2, Reduction: the envelope bounds are never weaker than
adding [L, M] as an interval to Proposition 16 (the interval product is a superset of the range of φ), and are
strictly sharper when μ > 1/β > ℓ.

## 6. Status

Derived by Claude on 2026-10-30; machine-checked (300 random + 150 adversarial cases, 0 violations);
independent re-derivation: done 2026-10-30 (fresh-context session, verdict AGREES-WITH-REMARKS, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
