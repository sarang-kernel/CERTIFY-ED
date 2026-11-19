# Independent re-derivation: C-03 (entanglement entropy and entanglement spectrum of a certified eigenspace)

Written on 2026-10-30 before the original proof was read (not a human check). Sections 1 and 3 of `../C-03_entanglement.md` read first, with
`certify_ed/entanglement.py`; proof written before section 2 was opened. Notation as in the original.

## Statements checked

Proposition 1 (T ≤ s + η/(1 − η); T ≤ s for g = 1), Proposition 2 (Fannes–Audenaert with T*), Proposition 3
(Weyl, 2T*), Lemma 3 (exact enclosure of the spectrum of ρ̂), Lemma 4 (monotonicity of g_d), Lemma 5 (interval
evaluation of −p ln p), and the certificate's conclusion.

## My proof

**Proposition 1.** The partial trace is a completely positive trace-preserving map, hence trace-norm
contractive: ‖tr_B(Y)‖₁ ≤ ‖Y‖₁ for every operator Y (Watrous, *The Theory of Quantum Information*, Thm 3.39 /
Nielsen–Chuang §9.2.1 for Hermitian Y: write Y = Y₊ − Y₋, then tr_B(Y±) ≥ 0 and ‖tr_B Y‖₁ ≤ tr(tr_B Y₊) + tr(tr_B Y₋)
= ‖Y‖₁). So 2T = ‖tr_B(P/g − XXᴴ/t)‖₁ ≤ ‖P/g − XXᴴ/t‖₁ ≤ ‖P − Q̂Q̂ᴴ‖₁/g + ‖Q̂Q̂ᴴ/g − XXᴴ/t‖₁.
*First term.* P − Q̂Q̂ᴴ is a difference of two rank-g orthogonal projectors, so its rank is ≤ 2g and
‖P − Q̂Q̂ᴴ‖₁ ≤ rank·‖P − Q̂Q̂ᴴ‖₂ ≤ 2g s (the trace norm is the sum of at most rank many singular values, each ≤ the
spectral norm). Contribution to T: (2gs)/(2g) = s.
*Second term.* X = Q̂M with M = G^{1/2}, G = XᴴX, so XXᴴ = Q̂GQ̂ᴴ and t = tr G. Q̂Q̂ᴴ/g − XXᴴ/t = Q̂(I/g − G/t)Q̂ᴴ, whose
singular values are those of I/g − G/t (Q̂ is an isometry), so its trace norm is Σ_i |1/g − g_i/t| over the
eigenvalues g_i ∈ [1 − η, 1 + η] of G (η = ‖G − I‖₂ < 1). With t = Σ_j g_j: |1/g − g_i/t| = |t − g g_i|/(gt) =
|Σ_j (g_j − g_i)|/(gt) ≤ g·2η/(gt) = 2η/t, hence Σ_i ≤ 2gη/t ≤ 2gη/(g(1 − η)) = 2η/(1 − η). Contribution to T:
η/(1 − η). For g = 1, G = t is a scalar and the second term vanishes identically. So T ≤ s + η/(1 − η), and
T ≤ s when g = 1. ∎

**Lemma 4.** g_d(T) = T ln(d − 1) + h(T) has g_d′(T) = ln(d − 1) + ln((1 − T)/T) = ln((d − 1)(1 − T)/T), which is
≥ 0 iff (d − 1)(1 − T) ≥ T iff T ≤ 1 − 1/d. So g_d increases on [0, 1 − 1/d] and decreases on [1 − 1/d, 1]. ∎

**Proposition 2.** Audenaert's sharp Fannes inequality (K. M. R. Audenaert, J. Phys. A 40 (2007) 8127, Thm 1):
for density matrices ρ, σ on ℂ^d with T = ‖ρ − σ‖₁/2, |S(ρ) − S(σ)| ≤ T ln(d − 1) + h(T) (all T ∈ [0, 1]). ρ_A and
ρ̂_A are density matrices on ℂ^{d_A} (positive semidefinite — partial traces of positive operators — with trace
1 by the normalisations tr P = g, tr XXᴴ = t). By Proposition 1, T ≤ T*; if T* ≤ 1 − 1/d_A then T ≤ T* ≤ 1 − 1/d_A
and Lemma 4 gives g_{d_A}(T) ≤ g_{d_A}(T*). ∎

**Proposition 3.** Weyl's inequality (Horn–Johnson Cor. 4.3.15) for the Hermitian ρ_A, ρ̂_A: |λ_k(ρ_A) − λ_k(ρ̂_A)|
≤ ‖ρ_A − ρ̂_A‖₂ ≤ ‖ρ_A − ρ̂_A‖₁ = 2T ≤ 2T*. ∎

**Lemma 3.** η_V = ‖VᴴV − I‖_F ≥ ‖VᴴV − I‖₂, so the eigenvalues of VᴴV lie in [1 − η_V, 1 + η_V] ⊂ (0, 2) and V is
invertible. Ostrowski's theorem (Horn–Johnson Thm 4.5.9): λ_k(Vᴴρ̂V) = θ_k λ_k(ρ̂) with θ_k ∈ [λ_min(VᴴV), λ_max(VᴴV)]
⊂ [1 − η_V, 1 + η_V], same index k (ascending order on both sides). Weyl: the ascending eigenvalues of Vᴴρ̂V and of
diag(p) (which are p₁ ≤ … ≤ p_d) differ by at most ‖Vᴴρ̂V − diag(p)‖₂ ≤ w. Hence p_k − w ≤ θ_k λ_k(ρ̂) ≤ p_k + w.
Since λ_k(ρ̂) ≥ 0 and θ_k > 0: λ_k(ρ̂) ≤ (p_k + w)/θ_k ≤ (p_k + w)/(1 − η_V) (p_k + w ≥ θ_kλ_k ≥ 0), and
λ_k(ρ̂) ≥ (p_k − w)/θ_k ≥ (p_k − w)/(1 + η_V) when p_k − w ≥ 0, trivially ≥ 0 otherwise: λ_k(ρ̂) ≥
max(0, p_k − w)/(1 + η_V). λ_k(ρ̂) ≤ 1 since ρ̂ ≥ 0 with trace 1. ∎ (Nothing is assumed about V beyond η_V < 1; the
floating-point eigendecomposition only supplies the guess.)

**Lemma 5.** f(p) = −p ln p on [0, 1] (f(0) := 0, continuous), f′(p) = −ln p − 1: f increases on [0, 1/e], decreases
on [1/e, 1], f(0) = f(1) = 0, max f = f(1/e) = 1/e. On [lo, hi] ⊂ [0, 1] the range is [min(f(lo), f(hi)),
max(f(lo), f(hi))] if 1/e ∉ [lo, hi], and [min(f(lo), f(hi)), 1/e] otherwise. The code's `_f_point(p)` =
−(p̂·ln p̂) with p̂ the grid enclosure of p encloses f(p) (Lemma 17; exact 0 at p ≤ 0 and p ≥ 1); `_f_interval`
takes min of the lower ends, max of the upper ends, and raises the upper end to the upper end of exp(−1)'s
enclosure whenever the enclosure of 1/e meets [lo, hi] (a superset test, hence safe). ∎

**Conclusion.** S(ρ̂_A) = Σ_k f(λ_k(ρ̂_A)) with λ_k(ρ̂_A) ∈ [l_k, u_k] (Lemma 3, exact rationals), so S(ρ̂_A) ∈ Ŝ :=
Σ_k f([l_k, u_k]) (Lemma 5, exact sum of enclosures). |S(ρ_A) − S(ρ̂_A)| ≤ g_{d_A}(T*) ≤ F (Proposition 2; F is the
upper end of T̂*·ln(d_A − 1)^ + ĥ(T*)^, each an enclosure by Lemma 17 — for d_A = 2 the first term is 0 exactly).
So S(ρ_A) ∈ [Ŝ_lo − F, Ŝ_hi + F], intersected with [0, ln d_A] (0 ≤ S ≤ ln d_A for any state on ℂ^{d_A}; the upper
clamp uses the upper end of the enclosure of ln d_A). Spectrum: Proposition 3 with the exact 2T* and the clamps
to [0, 1]. Conversion to binary64 outward at the end. ∎

### Code (current `entanglement.py`)

`_frac_sqrt_upper` 42–50 ((isqrt(ab) + 1)/b ≥ √(a/b): (isqrt(ab) + 1)² > ab); `reduced_density_exact` 99–142 (exact
dyadic integers, M = Σ_j tr_B(x_j x_jᴴ), t = tr M); `enclose_psd_eigenvalues` 179–221 (η_V 204 and w 215 as exact
rationals with the square root rounded up, ends 219–220); `_f_point` 226–233, `_f_interval` 236–251,
`entropy_interval` 254–256, `binary_entropy_upper` 259–261, `fannes_audenaert_bound` 264–275 (range test 268,
d_A > 2 guard 273, upper end 275); T* at 318–324; conclusion 344–349 (Ŝ ± F, clamps), spectrum 356–365;
conversion outward via `ivl.to_float_*`. Directions all outward; `value` is display only.
*Robustness remark.* `_f_point(p)` for a rational 0 < p < 2⁻¹⁹² builds the grid interval (0, 1)·2⁻¹⁹² and
`ln_interval` raises ValueError ("ln needs a positive interval"), which `entanglement_records` turns into a
NOT_CHECKED record — sound but avoidable: f is increasing on [0, 1/e], so [0, f(2⁻¹⁹²)^↑] would enclose f(p) for
such p. A lower eigenvalue end l_k in (0, 2⁻¹⁹²) is rare (p_k − w must be a tiny positive rational) but possible.

### Comparison with section 2 (read after the above)

Same route throughout: Proposition 1 via the contractivity of the partial trace (section 2's Lemma B, Jordan
decomposition; I used the same argument), the rank-2g bound for ‖P − Q̂Q̂ᴴ‖₁ (Lemma A) and the exact singular values
|1/g − m_i/t| with the mean t/g ∈ [1 − η, 1 + η] for the normalisation term; Lemma 3 via Ostrowski + Weyl with the
same choice of ends; Lemma 4 by the sign of g′; Proposition 2 by Audenaert's theorem (nats) with the monotonicity
range enforced; Proposition 3 by Weyl with ‖·‖₂ ≤ ‖·‖₁; Lemma 5 by monotonicity about 1/e. Section 2's Lemma C and
the remark on which error enters (Davis–Kahan s, not the residual radius) are consistent with my hypothesis (H1).
No gap found. Code mapping checked at the current lines (listed above; section 3's numbers agree with the current
file). Rounding directions: η_V and w rounded up (`_frac_sqrt_upper`), bracket ends outward, F upper end, clamps
exact, final conversion outward.

**Remarks.** (1) The robustness remark above (`_f_point` raising for a rational 0 < p < 2⁻¹⁹²; the record becomes
NOT_CHECKED rather than CERTIFIED with a slightly wider interval) — soundness is unaffected. (2) The document's η
in (H2) is a 2-norm; the code's η_V in Lemma 3 is a Frobenius norm (‖·‖₂ ≤ ‖·‖_F, conservative), and the caller's
η from the S3 record is used in T* — both directions are the safe ones.

**Verdict: AGREES-WITH-REMARKS.**

### Runs

* `python docs/proofs/verify_C-03.py 3` (seed 3; the script takes a positional seed and has fixed sizes): A 129
  cases, B 65, C 162 pairs + 20000 grid steps + beyond-range case reproduced, D 80 pairs, E 700 + 40 end-to-end;
  TOTAL 21176 cases, 0 violations (82 s).
* `python docs/proofs/independent/adversarial_C-03_entanglement.py --cases 120 --seed 53`: Proposition 1
  29 random + 54 adversarial cases (37 skipped with η ≥ 1), Proposition 3 83, Proposition 2 108 (incl. T = 1 − 1/d),
  Lemma 3 120 (exact rationals), Lemma 5 129, end-to-end `certify_entanglement` on Heisenberg chains N = 3, 4
  (exact-eigenspace oracle at 60 digits) 8 cases; 0 violations (81 s). Two oracle defects were fixed on the way
  (binary64 eigenvectors as "exact" eigenspace; mp roundoff of exactly-zero eigenvalues against the clamp at 0) —
  both were test artefacts, not code findings.
