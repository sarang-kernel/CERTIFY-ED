import CertifyMathlib.Enclosure.Residual

/-!
# E-03: generalized eigenproblems through the Cholesky reduction (exact part)

`docs/proofs/E-03_generalized.md`, Lemmas 1 and 2. For `A x = λ B x` with `B = L Lᴴ`
(`L` invertible) put `C = L⁻¹ A L⁻ᴴ`. Then

* `C` is symmetric when `A` is (`reducedOp_isSymmetric`), and every eigenpair `(y, λ)` of `C` gives
  the pencil eigenpair `(L⁻ᴴ y, λ)` (`pencil_eigenpair_of_reduced`);
* `‖Lᴴ x‖² = xᴴ B x = ‖x‖_B²` (`norm_sq_adjoint_eq`);
* **Lemma 2**: for `x ≠ 0` and real `μ` some eigenvalue `λ` of `C` satisfies
  `|λ - μ| ≤ ‖L⁻¹ (A x - μ B x)‖ / ‖Lᴴ x‖` (`generalized_residual_enclosure`), and
  `‖L⁻¹ s‖ ≤ ‖s‖ / √β_c` whenever `β_c ‖w‖² ≤ wᴴ B w` for all `w`, `β_c > 0`
  (`norm_inv_le_of_lower_bound`), which gives the document's second inequality
  (`generalized_residual_enclosure_beta`).

Not formalized: the converse half of Lemma 1, that the pencil eigenvalues are exactly the
eigenvalues of `C` with multiplicity (roots of `det(A - λB)`), and Lemma 3 and Propositions 1 to 3
(inertia and floating point). The theorems below are about the eigenvalues of `C`.
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
  [FiniteDimensional 𝕜 E]

section Pencil

variable (L : E ≃ₗ[𝕜] E)

/-- `L⁻ᴴ`, the adjoint of `L⁻¹`. -/
noncomputable def invAdj : E →ₗ[𝕜] E := LinearMap.adjoint (L.symm : E →ₗ[𝕜] E)

theorem adjoint_invAdj (w : E) : LinearMap.adjoint (L : E →ₗ[𝕜] E) (invAdj L w) = w := by
  apply ext_inner_right 𝕜
  intro v
  rw [LinearMap.adjoint_inner_left, invAdj, LinearMap.adjoint_inner_left]
  simp

theorem invAdj_adjoint (w : E) : invAdj L (LinearMap.adjoint (L : E →ₗ[𝕜] E) w) = w := by
  apply ext_inner_right 𝕜
  intro v
  rw [invAdj, LinearMap.adjoint_inner_left, LinearMap.adjoint_inner_left]
  simp

/-- The reduced operator `C = L⁻¹ A L⁻ᴴ`. -/
noncomputable def reducedOp (A : E →ₗ[𝕜] E) : E →ₗ[𝕜] E :=
  (L.symm : E →ₗ[𝕜] E) ∘ₗ A ∘ₗ invAdj L

theorem reducedOp_isSymmetric {A : E →ₗ[𝕜] E} (hA : A.IsSymmetric) :
    (reducedOp L A).IsSymmetric := by
  intro u v
  simp only [reducedOp, LinearMap.comp_apply]
  rw [← LinearMap.adjoint_inner_right, ← invAdj, hA, invAdj, LinearMap.adjoint_inner_left]

/-- An eigenpair of `C` gives an eigenpair of the pencil. -/
theorem pencil_eigenpair_of_reduced {A B : E →ₗ[𝕜] E}
    (hB : ∀ v, B v = L (LinearMap.adjoint (L : E →ₗ[𝕜] E) v)) {y : E} {lam : 𝕜}
    (hy : reducedOp L A y = lam • y) : A (invAdj L y) = lam • B (invAdj L y) := by
  have h := congrArg L hy
  simp only [reducedOp, LinearMap.comp_apply, LinearEquiv.coe_coe,
    LinearEquiv.apply_symm_apply] at h
  rw [h, hB, adjoint_invAdj, map_smul]

/-- `‖Lᴴ x‖² = re (xᴴ B x)`. -/
theorem norm_sq_adjoint_eq {B : E →ₗ[𝕜] E}
    (hB : ∀ v, B v = L (LinearMap.adjoint (L : E →ₗ[𝕜] E) v)) (x : E) :
    ‖LinearMap.adjoint (L : E →ₗ[𝕜] E) x‖ ^ 2 = RCLike.re ⟪x, B x⟫_𝕜 := by
  rw [hB, ← inner_self_eq_norm_sq (𝕜 := 𝕜), LinearMap.adjoint_inner_left]
  rfl

/-- The residual of `C` at `y = Lᴴ x` is `L⁻¹ (A x - μ B x)`. -/
theorem reduced_residual {A B : E →ₗ[𝕜] E}
    (hB : ∀ v, B v = L (LinearMap.adjoint (L : E →ₗ[𝕜] E) v)) (μ : ℝ) (x : E) :
    reducedOp L A (LinearMap.adjoint (L : E →ₗ[𝕜] E) x) -
        (μ : 𝕜) • LinearMap.adjoint (L : E →ₗ[𝕜] E) x =
      L.symm (A x - (μ : 𝕜) • B x) := by
  simp only [reducedOp, LinearMap.comp_apply, LinearEquiv.coe_coe]
  rw [invAdj_adjoint, map_sub, map_smul, hB, LinearEquiv.symm_apply_apply]

/-- **E-03, Lemma 2** (first inequality), for the eigenvalues of `C = L⁻¹ A L⁻ᴴ`. -/
theorem generalized_residual_enclosure {A B : E →ₗ[𝕜] E} (hA : A.IsSymmetric)
    (hB : ∀ v, B v = L (LinearMap.adjoint (L : E →ₗ[𝕜] E) v)) {ι : Type*} [Fintype ι]
    (D : EigenData (reducedOp L A) ι) (μ : ℝ) {x : E} (hx : x ≠ 0) :
    ∃ i, |D.eig i - μ| ≤
      ‖L.symm (A x - (μ : 𝕜) • B x)‖ / ‖LinearMap.adjoint (L : E →ₗ[𝕜] E) x‖ := by
  have hy : LinearMap.adjoint (L : E →ₗ[𝕜] E) x ≠ 0 := by
    intro h0
    apply hx
    rw [← invAdj_adjoint L x, h0, map_zero]
  obtain ⟨i, hi⟩ := D.exists_eig_near (reducedOp_isSymmetric L hA) μ hy
  exact ⟨i, by rwa [reduced_residual L hB μ x] at hi⟩

/-- `‖L⁻¹ s‖ ≤ ‖s‖ / √β_c` when `β_c ‖w‖² ≤ re (wᴴ B w)` for all `w` (`λ_min(B) ≥ β_c > 0`). -/
theorem norm_inv_le_of_lower_bound {B : E →ₗ[𝕜] E}
    (hB : ∀ v, B v = L (LinearMap.adjoint (L : E →ₗ[𝕜] E) v)) {βc : ℝ} (hβ : 0 < βc)
    (hlow : ∀ w, βc * ‖w‖ ^ 2 ≤ RCLike.re ⟪w, B w⟫_𝕜) (s : E) :
    ‖L.symm s‖ ≤ ‖s‖ / √βc := by
  set u := L.symm s with hu
  set w := invAdj L u with hw
  have hLw : LinearMap.adjoint (L : E →ₗ[𝕜] E) w = u := adjoint_invAdj L u
  have h1 : βc * ‖w‖ ^ 2 ≤ ‖u‖ ^ 2 := by
    have := hlow w
    rw [← norm_sq_adjoint_eq L hB w, hLw] at this
    exact this
  have h2 : ‖u‖ ^ 2 ≤ ‖s‖ * ‖w‖ := by
    have e : ⟪u, u⟫_𝕜 = ⟪s, w⟫_𝕜 := by
      rw [hw, invAdj, LinearMap.adjoint_inner_right]
      simp [hu]
    rw [← inner_self_eq_norm_sq (𝕜 := 𝕜), e]
    exact (RCLike.re_le_norm _).trans (norm_inner_le_norm _ _)
  have hs : 0 < √βc := Real.sqrt_pos.mpr hβ
  have h3 : √βc * ‖w‖ ≤ ‖u‖ := by
    have hsq : (√βc * ‖w‖) ^ 2 ≤ ‖u‖ ^ 2 := by
      rw [mul_pow, Real.sq_sqrt hβ.le]
      exact h1
    exact (pow_le_pow_iff_left₀ (mul_nonneg hs.le (norm_nonneg _)) (norm_nonneg _)
      two_ne_zero).mp hsq
  rw [le_div_iff₀ hs]
  by_cases hu0 : ‖u‖ = 0
  · rw [hu0, zero_mul]
    exact norm_nonneg _
  · have hupos : 0 < ‖u‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hu0)
    have h4 : ‖u‖ * (√βc * ‖w‖) ≤ ‖u‖ * ‖u‖ := mul_le_mul_of_nonneg_left h3 hupos.le
    have h5 : ‖u‖ * ‖u‖ ≤ ‖s‖ * ‖w‖ := by nlinarith [h2]
    have h6 : ‖u‖ * (√βc * ‖w‖) ≤ ‖s‖ * ‖w‖ := h4.trans h5
    by_cases hw0 : ‖w‖ = 0
    · rw [hw0, mul_zero] at h2
      have h7 : ‖u‖ ^ 2 = 0 := le_antisymm h2 (sq_nonneg _)
      exact absurd (pow_eq_zero_iff two_ne_zero |>.mp h7) hu0
    · have hwpos : 0 < ‖w‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hw0)
      nlinarith

/-- **E-03, Lemma 2** with `β_c ≤ λ_min(B)`:
`|λ - μ| ≤ ‖A x - μ B x‖ / (√β_c ‖x‖_B)`. -/
theorem generalized_residual_enclosure_beta {A B : E →ₗ[𝕜] E} (hA : A.IsSymmetric)
    (hB : ∀ v, B v = L (LinearMap.adjoint (L : E →ₗ[𝕜] E) v)) {ι : Type*} [Fintype ι]
    (D : EigenData (reducedOp L A) ι) {βc : ℝ} (hβ : 0 < βc)
    (hlow : ∀ w, βc * ‖w‖ ^ 2 ≤ RCLike.re ⟪w, B w⟫_𝕜) (μ : ℝ) {x : E} (hx : x ≠ 0) :
    ∃ i, |D.eig i - μ| ≤
      ‖A x - (μ : 𝕜) • B x‖ / (√βc * ‖LinearMap.adjoint (L : E →ₗ[𝕜] E) x‖) := by
  obtain ⟨i, hi⟩ := generalized_residual_enclosure L hA hB D μ hx
  refine ⟨i, hi.trans ?_⟩
  have hy : 0 < ‖LinearMap.adjoint (L : E →ₗ[𝕜] E) x‖ := by
    rw [norm_pos_iff]
    intro h0
    apply hx
    rw [← invAdj_adjoint L x, h0, map_zero]
  rw [div_le_div_iff₀ hy (mul_pos (Real.sqrt_pos.mpr hβ) hy)]
  have h := norm_inv_le_of_lower_bound L hB hβ hlow (A x - (μ : 𝕜) • B x)
  rw [le_div_iff₀ (Real.sqrt_pos.mpr hβ)] at h
  nlinarith [mul_le_mul_of_nonneg_right h hy.le]

end Pencil

end CertifyMathlib.Enclosure
