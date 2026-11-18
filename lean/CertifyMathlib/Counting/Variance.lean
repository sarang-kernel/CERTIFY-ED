import CertifyMathlib.Counting.QuadForm

/-!
# Proposition 16, the variance and heat-capacity interval

Source: `docs/proofs/P16_thermo_intervals.md` in the code repository. With weights
`w_i = e^{-β e_i} ≥ 0`, `Z̃ = Σ w_i`, `⟨e⟩ = Σ w_i e_i / Z̃` and `⟨e²⟩ = Σ w_i e_i² / Z̃`, the variance
`Var(E) = Var(e) = ⟨e²⟩ - ⟨e⟩²` is non-negative (`variance_nonneg`, a weighted Cauchy-Schwarz
inequality), and enclosures of `⟨e⟩ ≥ 0` and `⟨e²⟩` give the interval
`[max(0, a₂ - b₁²), b₂ - a₁²]` for it (`variance_bounds`), the intersection with `[0, ∞)` of the
document. `C = β² Var(E)` then lies in `β²` times that interval (`heat_capacity_bounds`).
-/

namespace CertifyMathlib.Counting

/-- Weighted Cauchy-Schwarz: `(Σ w e)² ≤ (Σ w)(Σ w e²)` for non-negative weights. -/
theorem weighted_sq_le {ι : Type*} (s : Finset ι) (w e : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i) :
    (∑ i ∈ s, w i * e i) ^ 2 ≤ (∑ i ∈ s, w i) * ∑ i ∈ s, w i * e i ^ 2 := by
  obtain ⟨A, hA⟩ : ∃ A, A = ∑ i ∈ s, w i := ⟨_, rfl⟩
  obtain ⟨B, hB⟩ : ∃ B, B = ∑ i ∈ s, w i * e i := ⟨_, rfl⟩
  obtain ⟨C, hC⟩ : ∃ C, C = ∑ i ∈ s, w i * e i ^ 2 := ⟨_, rfl⟩
  rw [← hA, ← hB, ← hC]
  have hA0 : 0 ≤ A := by
    rw [hA]
    exact Finset.sum_nonneg hw
  have key : 0 ≤ ∑ i ∈ s, w i * (A * e i - B) ^ 2 :=
    Finset.sum_nonneg fun i hi => mul_nonneg (hw i hi) (sq_nonneg _)
  have hterm : ∀ i, w i * (A * e i - B) ^ 2 =
      A ^ 2 * (w i * e i ^ 2) - 2 * A * B * (w i * e i) + B ^ 2 * w i := fun i => by ring
  have expand : ∑ i ∈ s, w i * (A * e i - B) ^ 2 = A * (A * C - B ^ 2) := by
    rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ← hA, ← hB, ← hC]
    ring
  rw [expand] at key
  rcases hA0.lt_or_eq with hpos | hzero
  · nlinarith
  · have hsum : ∑ i ∈ s, w i = 0 := by
      rw [← hA]
      exact hzero.symm
    have hw0 : ∀ i ∈ s, w i = 0 := (Finset.sum_eq_zero_iff_of_nonneg hw).mp hsum
    have hB0 : B = 0 := by
      rw [hB]
      exact Finset.sum_eq_zero fun i hi => by rw [hw0 i hi, zero_mul]
    rw [hB0, ← hzero]
    simp

/-- `Var(e) = ⟨e²⟩ - ⟨e⟩² ≥ 0` for non-negative weights with positive sum. -/
theorem variance_nonneg {ι : Type*} (s : Finset ι) (w e : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hZ : 0 < ∑ i ∈ s, w i) :
    0 ≤ (∑ i ∈ s, w i * e i ^ 2) / (∑ i ∈ s, w i) - ((∑ i ∈ s, w i * e i) / (∑ i ∈ s, w i)) ^ 2 := by
  have h := weighted_sq_le s w e hw
  rw [div_pow, sub_nonneg, div_le_div_iff₀ (pow_pos hZ 2) hZ]
  nlinarith [h]

/-- **Proposition 16, variance interval.** With `0 ≤ a₁ ≤ m₁ ≤ b₁`, `a₂ ≤ m₂ ≤ b₂` and
`m₂ - m₁² ≥ 0`: `max 0 (a₂ - b₁²) ≤ m₂ - m₁² ≤ b₂ - a₁²`. -/
theorem variance_bounds {m₁ m₂ a₁ b₁ a₂ b₂ : ℝ} (h₁ : a₁ ≤ m₁ ∧ m₁ ≤ b₁) (ha₁ : 0 ≤ a₁)
    (h₂ : a₂ ≤ m₂ ∧ m₂ ≤ b₂) (hvar : 0 ≤ m₂ - m₁ ^ 2) :
    max 0 (a₂ - b₁ ^ 2) ≤ m₂ - m₁ ^ 2 ∧ m₂ - m₁ ^ 2 ≤ b₂ - a₁ ^ 2 := by
  have hm₁ : 0 ≤ m₁ := ha₁.trans h₁.1
  have hsq1 : m₁ ^ 2 ≤ b₁ ^ 2 := pow_le_pow_left₀ hm₁ h₁.2 2
  have hsq2 : a₁ ^ 2 ≤ m₁ ^ 2 := pow_le_pow_left₀ ha₁ h₁.1 2
  refine ⟨max_le hvar (by linarith), by linarith⟩

/-- **Proposition 16, heat capacity** `C = β² Var(E)`. -/
theorem heat_capacity_bounds {V lo hi β : ℝ} (hV : lo ≤ V ∧ V ≤ hi) :
    β ^ 2 * lo ≤ β ^ 2 * V ∧ β ^ 2 * V ≤ β ^ 2 * hi :=
  ⟨mul_le_mul_of_nonneg_left hV.1 (sq_nonneg β), mul_le_mul_of_nonneg_left hV.2 (sq_nonneg β)⟩

end CertifyMathlib.Counting
