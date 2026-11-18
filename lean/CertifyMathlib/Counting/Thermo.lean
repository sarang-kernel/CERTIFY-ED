import CertifyMathlib.Counting.Inertia

/-!
# Proposition 15 (summed bounds) and Proposition 16 (thermodynamic intervals)

Sources: `docs/theory.md`, sections 9.2 and 9.3, `docs/proofs/P15_summed_bounds.md` and
`docs/proofs/P16_thermo_intervals.md` in the code repository.

* `summed_bound` (Proposition 15): `|Σ a_k x_k - Σ a_k v_k| ≤ Σ |a_k| b_k` when
  `|x_k - v_k| ≤ b_k`.
* Proposition 16, hypothesis (H): every eigenvalue `λ_i` of `A` is assigned to an interval
  `κ i` with `l (κ i) ≤ λ_i ≤ h (κ i)`; the count of interval `k` is `c_k = #{i | κ i = k}`. With the
  shift `E_0` and `a_k = l_k - E_0 ≥ 0`, `b_k = h_k - E_0`:
  * `thermo_Z`: `Σ_k c_k e^{-β b_k} ≤ Z e^{β E_0} = Σ_i e^{-β (λ_i - E_0)} ≤ Σ_k c_k e^{-β a_k}`;
  * `thermo_energy`: the same for `Σ_i e_i e^{-β e_i}` with `e_i = λ_i - E_0` and the interval
    products `[a_k e^{-β b_k}, b_k e^{-β a_k}]`;
  * `thermo_second`: the same for `Σ_i e_i² e^{-β e_i}`;
  * `quotient_bounds`: the quotient of two enclosures encloses the quotient (for `⟨e⟩`, `⟨e²⟩`);
  * `free_energy_bounds`: `F = E_0 - T ln Z̃` from the enclosure of `Z̃` (monotonicity of `ln`).

The partition function is taken as the sum over Mathlib's eigenvalues, `Σ_i e^{-β λ_i}`, which is
`tr e^{-βA}`. Not formalised: the dyadic interval arithmetic of Lemma 17 and the variance and heat
capacity interval with its clipping at 0.
-/

namespace CertifyMathlib.Counting

open Matrix

/-- **Proposition 15 (summed bounds).** -/
theorem summed_bound {𝕜 : Type*} [RCLike 𝕜] {κ : Type*} (s : Finset κ) (a x v : κ → 𝕜)
    (b : κ → ℝ) (h : ∀ k ∈ s, ‖x k - v k‖ ≤ b k) :
    ‖∑ k ∈ s, a k * x k - ∑ k ∈ s, a k * v k‖ ≤ ∑ k ∈ s, ‖a k‖ * b k := by
  rw [← Finset.sum_sub_distrib]
  refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun k hk => ?_)
  rw [← mul_sub, norm_mul]
  exact mul_le_mul_of_nonneg_left (h k hk) (norm_nonneg _)

section Thermo

variable {n : Type*} [Fintype n] {K : Type*} [Fintype K] [DecidableEq K]

/-- Termwise bounds summed over the fibres of an assignment `κ : n → K`. -/
theorem sum_fiber_bounds (κ : n → K) (f : n → ℝ) (lo hi : K → ℝ)
    (hf : ∀ i, lo (κ i) ≤ f i ∧ f i ≤ hi (κ i)) :
    ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) * lo k ≤ ∑ i, f i ∧
    ∑ i, f i ≤ ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) * hi k := by
  constructor
  · rw [← Finset.sum_fiberwise Finset.univ κ f]
    refine Finset.sum_le_sum fun k _ => ?_
    rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun i hmem => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hmem
    have := (hf i).1
    rw [hmem] at this
    exact this
  · rw [← Finset.sum_fiberwise Finset.univ κ f]
    refine Finset.sum_le_sum fun k _ => ?_
    rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun i hmem => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hmem
    have := (hf i).2
    rw [hmem] at this
    exact this

variable [DecidableEq n] {𝕜 : Type*} [RCLike 𝕜]

/-- **Proposition 16, partition function.** -/
theorem thermo_Z {A : Matrix n n 𝕜} (hA : A.IsHermitian) (κ : n → K) (l h : K → ℝ)
    (hin : ∀ i, l (κ i) ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ h (κ i)) (β E0 : ℝ)
    (hβ : 0 ≤ β) :
    ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) * Real.exp (-β * (h k - E0)) ≤
      ∑ i, Real.exp (-β * (hA.eigenvalues i - E0)) ∧
    ∑ i, Real.exp (-β * (hA.eigenvalues i - E0)) ≤
      ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) * Real.exp (-β * (l k - E0)) := by
  refine sum_fiber_bounds κ _ _ _ fun i => ⟨?_, ?_⟩
  · apply Real.exp_le_exp.mpr
    have := (hin i).2
    nlinarith
  · apply Real.exp_le_exp.mpr
    have := (hin i).1
    nlinarith

/-- **Proposition 16, energy numerator** `Σ_i e_i e^{-β e_i}` with `e_i = λ_i - E_0 ≥ 0`. -/
theorem thermo_energy {A : Matrix n n 𝕜} (hA : A.IsHermitian) (κ : n → K) (l h : K → ℝ)
    (hin : ∀ i, l (κ i) ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ h (κ i)) (β E0 : ℝ)
    (hβ : 0 ≤ β) (hE0 : ∀ k, E0 ≤ l k) :
    ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) *
        ((l k - E0) * Real.exp (-β * (h k - E0))) ≤
      ∑ i, (hA.eigenvalues i - E0) * Real.exp (-β * (hA.eigenvalues i - E0)) ∧
    ∑ i, (hA.eigenvalues i - E0) * Real.exp (-β * (hA.eigenvalues i - E0)) ≤
      ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) *
        ((h k - E0) * Real.exp (-β * (l k - E0))) := by
  refine sum_fiber_bounds κ _ _ _ fun i => ⟨?_, ?_⟩
  · have h1 := (hin i).1
    have h2 := (hin i).2
    have h0 := hE0 (κ i)
    apply mul_le_mul (by linarith) (Real.exp_le_exp.mpr (by nlinarith)) (Real.exp_pos _).le
      (by linarith)
  · have h1 := (hin i).1
    have h2 := (hin i).2
    have h0 := hE0 (κ i)
    apply mul_le_mul (by linarith) (Real.exp_le_exp.mpr (by nlinarith)) (Real.exp_pos _).le
      (by linarith)

/-- **Proposition 16, second moment** `Σ_i e_i² e^{-β e_i}`. -/
theorem thermo_second {A : Matrix n n 𝕜} (hA : A.IsHermitian) (κ : n → K) (l h : K → ℝ)
    (hin : ∀ i, l (κ i) ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ h (κ i)) (β E0 : ℝ)
    (hβ : 0 ≤ β) (hE0 : ∀ k, E0 ≤ l k) :
    ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) *
        ((l k - E0) ^ 2 * Real.exp (-β * (h k - E0))) ≤
      ∑ i, (hA.eigenvalues i - E0) ^ 2 * Real.exp (-β * (hA.eigenvalues i - E0)) ∧
    ∑ i, (hA.eigenvalues i - E0) ^ 2 * Real.exp (-β * (hA.eigenvalues i - E0)) ≤
      ∑ k, ((Finset.univ.filter fun i => κ i = k).card : ℝ) *
        ((h k - E0) ^ 2 * Real.exp (-β * (l k - E0))) := by
  refine sum_fiber_bounds κ _ _ _ fun i => ⟨?_, ?_⟩
  · have h1 := (hin i).1
    have h2 := (hin i).2
    have h0 := hE0 (κ i)
    apply mul_le_mul (pow_le_pow_left₀ (by linarith) (by linarith) 2)
      (Real.exp_le_exp.mpr (by nlinarith)) (Real.exp_pos _).le (sq_nonneg _)
  · have h1 := (hin i).1
    have h2 := (hin i).2
    have h0 := hE0 (κ i)
    apply mul_le_mul (pow_le_pow_left₀ (by linarith) (by linarith) 2)
      (Real.exp_le_exp.mpr (by nlinarith)) (Real.exp_pos _).le (sq_nonneg _)

/-- **C-06, Proposition 16' (i), omitted weight.** The eigenvalues at or above `L` number
`n - cntLt hA L`; if they also lie at or below `M`, their shifted Boltzmann weight lies in
`(n - k) · [e^{-β (M - E_0)}, e^{-β (L - E_0)}]`, where `k = cntLt hA L` is the S3 count. -/
theorem tail_weight {A : Matrix n n 𝕜} (hA : A.IsHermitian) (L M β E0 : ℝ) (hβ : 0 ≤ β)
    (hM : ∀ i, hA.eigenvalues i ≤ M) :
    (Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card + cntLt hA L = Fintype.card n ∧
    ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ) * Real.exp (-β * (M - E0)) ≤
      ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L),
        Real.exp (-β * (hA.eigenvalues i - E0)) ∧
    ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L),
        Real.exp (-β * (hA.eigenvalues i - E0)) ≤
      ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ) * Real.exp (-β * (L - E0)) := by
  refine ⟨?_, ?_, ?_⟩
  · have h := Finset.card_filter_add_card_filter_not (s := Finset.univ)
      (fun i => hA.eigenvalues i < L)
    rw [Finset.card_univ] at h
    unfold cntLt
    omega
  · rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun i _ => Real.exp_le_exp.mpr ?_
    have := hM i
    nlinarith
  · rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun i hi => Real.exp_le_exp.mpr ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_lt] at hi
    nlinarith

end Thermo

/-- The quotient of two enclosures encloses the quotient (positive denominator, non-negative
numerator). -/
theorem quotient_bounds {N Z Nl Nh Zl Zh : ℝ} (hN : Nl ≤ N ∧ N ≤ Nh) (hZ : Zl ≤ Z ∧ Z ≤ Zh)
    (hZl : 0 < Zl) (hNl : 0 ≤ Nl) : Nl / Zh ≤ N / Z ∧ N / Z ≤ Nh / Zl := by
  have hZ0 : 0 < Z := lt_of_lt_of_le hZl hZ.1
  have hZh : 0 < Zh := lt_of_lt_of_le hZ0 hZ.2
  have hN0 : 0 ≤ N := le_trans hNl hN.1
  constructor
  · rw [div_le_div_iff₀ hZh hZ0]
    nlinarith [mul_le_mul_of_nonneg_right hN.1 hZ0.le, mul_le_mul_of_nonneg_left hZ.2 hN0]
  · rw [div_le_div_iff₀ hZ0 hZl]
    nlinarith [mul_le_mul_of_nonneg_left hZ.1 hN0, mul_le_mul_of_nonneg_right hN.2 hZ0.le]

/-- **Proposition 16, free energy** `F = E_0 - T ln Z̃` from an enclosure of `Z̃`. -/
theorem free_energy_bounds {Z Zl Zh T E0 : ℝ} (hZ : Zl ≤ Z ∧ Z ≤ Zh) (hZl : 0 < Zl)
    (hT : 0 ≤ T) :
    E0 - T * Real.log Zh ≤ E0 - T * Real.log Z ∧ E0 - T * Real.log Z ≤ E0 - T * Real.log Zl := by
  have hZ0 : 0 < Z := lt_of_lt_of_le hZl hZ.1
  have h1 : Real.log Z ≤ Real.log Zh := Real.log_le_log hZ0 hZ.2
  have h2 : Real.log Zl ≤ Real.log Z := Real.log_le_log hZl hZ.1
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left h1 hT]
  · nlinarith [mul_le_mul_of_nonneg_left h2 hT]

end CertifyMathlib.Counting
