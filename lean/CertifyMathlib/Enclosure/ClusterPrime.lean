import CertifyMathlib.Enclosure.Cluster

/-!
# Theorem 2′ (the sharper cluster radius ε*)

`docs/proofs/thm2prime_eps-star.md`. In the setting of Theorem 2 (`Cluster.lean`), with
`spread(Λ) = μ_max - μ_min`, there are `g` distinct indices with `|λ_{i_j} - μ_j| ≤ ε*`,

  `ε* = ‖R‖ / √(1 - η) + ‖Xᴴ R‖ / (1 - η) + δ · spread(Λ) / √(1 - η)`.

The proof follows the document's three steps.

Step (i): `H′ = Q̂ᴴ A Q̂` is symmetric and `‖A Q̂ - Q̂ H′‖ ≤ ‖R‖ / √(1 - η)`
(`rayleigh_residual_le`): `A Q̂ z - Q̂ H′ z = (I - Q̂ Q̂ᴴ)(R M⁻¹ z)` and `I - Q̂ Q̂ᴴ` does not increase
norms (`norm_sub_proj_le`). Kahan's theorem then matches the eigenvalues of `H′` to distinct
eigenvalues of `A`.

Step (ii): `H′ - Λ = M⁻¹ Xᴴ R M⁻¹ + (M Λ M⁻¹ - Λ)` and
`M Λ y - Λ M y = (M - I)(Λ - c) y - (Λ - c)(M - I) y` for the midpoint `c` of `[lo, hi]`, so
`‖H′ - Λ‖ ≤ ‖Xᴴ R‖ / (1 - η) + δ (hi - lo) / √(1 - η)` (`rayleigh_sub_le`); Weyl's inequality
(`Weyl.lean`) pairs the sorted eigenvalues of `H′` and `Λ`.

Step (iii): composition (`cluster_eps_star`). `spread(Λ)` is replaced by `hi - lo` for any
`lo ≤ μ_j ≤ hi`, which is the document's statement for `lo = μ_min`, `hi = μ_max`.
`cluster_min` records that `min(ρ, ε*)` is a valid common radius.
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
  {F : Type*} [NormedAddCommGroup F] [InnerProductSpace 𝕜 F]
  [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F]

section Steps

variable {κ : Type*} [Fintype κ] [DecidableEq κ] {X : F →ₗ[𝕜] E} {G : F →ₗ[𝕜] F}
  (DG : EigenData G κ) {η : ℝ} (hη1 : η < 1) (hη : ∀ z, ‖G z - z‖ ≤ η * ‖z‖)
  (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜)
include hη1 hη hG

omit [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F] in
theorem polarQ_inner (u v : F) : ⟪polarQ DG X u, polarQ DG X v⟫_𝕜 = ⟪u, v⟫_𝕜 := by
  let L : F →ₗᵢ[𝕜] E := { toLinearMap := polarQ DG X, norm_map' := polarQ_norm DG hη1 hη hG }
  exact L.inner_map_map u v

theorem adjoint_polarQ_polarQ (u : F) :
    LinearMap.adjoint (polarQ DG X) (polarQ DG X u) = u := by
  apply ext_inner_right 𝕜
  intro v
  rw [LinearMap.adjoint_inner_left, polarQ_inner DG hη1 hη hG]

/-- `I - Q̂ Q̂ᴴ` does not increase norms. -/
theorem norm_sub_proj_le (w : E) :
    ‖w - polarQ DG X (LinearMap.adjoint (polarQ DG X) w)‖ ≤ ‖w‖ := by
  have horth : ⟪polarQ DG X (LinearMap.adjoint (polarQ DG X) w),
      w - polarQ DG X (LinearMap.adjoint (polarQ DG X) w)⟫_𝕜 = 0 := by
    rw [inner_sub_right, ← LinearMap.adjoint_inner_right, polarQ_inner DG hη1 hη hG, sub_self]
  have hp := norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero _ _ horth
  rw [add_sub_cancel] at hp
  have h1 : ‖w - polarQ DG X (LinearMap.adjoint (polarQ DG X) w)‖ ^ 2 ≤ ‖w‖ ^ 2 := by
    nlinarith [mul_self_nonneg ‖polarQ DG X (LinearMap.adjoint (polarQ DG X) w)‖]
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp h1

omit hη1 hη hG in
/-- `Q̂ᴴ r = M⁻¹ Xᴴ r`. -/
theorem adjoint_polarQ_apply (r : E) :
    LinearMap.adjoint (polarQ DG X) r = polarMinv DG (LinearMap.adjoint X r) := by
  apply ext_inner_right 𝕜
  intro v
  calc ⟪LinearMap.adjoint (polarQ DG X) r, v⟫_𝕜 = ⟪r, polarQ DG X v⟫_𝕜 :=
        LinearMap.adjoint_inner_left _ _ _
    _ = ⟪r, X (polarMinv DG v)⟫_𝕜 := rfl
    _ = ⟪LinearMap.adjoint X r, polarMinv DG v⟫_𝕜 := (LinearMap.adjoint_inner_left _ _ _).symm
    _ = ⟪polarMinv DG (LinearMap.adjoint X r), v⟫_𝕜 := (diagMap_isSymmetric DG.basis _ _ v).symm

variable {A : E →ₗ[𝕜] E} {Λ : F →ₗ[𝕜] F}

/-- The Rayleigh quotient block `H′ = Q̂ᴴ A Q̂`. -/
noncomputable def rayleighBlock (DG : EigenData G κ) (X : F →ₗ[𝕜] E) (A : E →ₗ[𝕜] E) :
    F →ₗ[𝕜] F :=
  LinearMap.adjoint (polarQ DG X) ∘ₗ A ∘ₗ polarQ DG X

omit hη1 hη hG in
theorem rayleighBlock_isSymmetric (hA : A.IsSymmetric) : (rayleighBlock DG X A).IsSymmetric := by
  intro u v
  simp only [rayleighBlock, LinearMap.comp_apply]
  rw [LinearMap.adjoint_inner_left, hA, LinearMap.adjoint_inner_right]

omit [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F] hG in
/-- `A Q̂ z = R (M⁻¹ z) + Q̂ (M Λ M⁻¹ z)`. -/
theorem A_polarQ_eq (z : F) :
    A (polarQ DG X z) = (A (X (polarMinv DG z)) - X (Λ (polarMinv DG z))) +
      polarQ DG X (polarM DG (Λ (polarMinv DG z))) := by
  rw [polarQ_polarM DG hη1 hη]
  show A (X (polarMinv DG z)) = _
  abel

/-- **Step (i)**: `‖A Q̂ z - Q̂ H′ z‖ ≤ nR (1 - η)^{-1/2} ‖z‖`. -/
theorem rayleigh_residual_le {nR : ℝ} (hR : ∀ z, ‖A (X z) - X (Λ z)‖ ≤ nR * ‖z‖)
    (hnR0 : 0 ≤ nR) (z : F) :
    ‖A (polarQ DG X z) - polarQ DG X (rayleighBlock DG X A z)‖ ≤
      nR * (√(1 - η))⁻¹ * ‖z‖ := by
  have e : A (polarQ DG X z) - polarQ DG X (rayleighBlock DG X A z) =
      (A (X (polarMinv DG z)) - X (Λ (polarMinv DG z))) -
        polarQ DG X (LinearMap.adjoint (polarQ DG X)
          (A (X (polarMinv DG z)) - X (Λ (polarMinv DG z)))) := by
    simp only [rayleighBlock, LinearMap.comp_apply]
    rw [A_polarQ_eq DG hη1 hη z, map_add (LinearMap.adjoint (polarQ DG X)),
      adjoint_polarQ_polarQ DG hη1 hη hG, map_add (polarQ DG X)]
    abel
  rw [e]
  refine (norm_sub_proj_le DG hη1 hη hG _).trans ?_
  refine (hR _).trans ?_
  rw [mul_assoc]
  exact mul_le_mul_of_nonneg_left (polarMinv_le DG hη1 hη z) hnR0

/-- **Step (ii)**: `‖H′ z - Λ z‖ ≤ (nXR / (1 - η) + δ (hi - lo) / √(1 - η)) ‖z‖`, written with
`κ = (1 - η)^{-1/2}`. -/
theorem rayleigh_sub_le (hη0 : 0 ≤ η) (hΛ : Λ.IsSymmetric) {g : ℕ} (DΛ : EigenData Λ (Fin g))
    {nXR lo hi : ℝ} (hXR : ∀ y, ‖LinearMap.adjoint X (A (X y) - X (Λ y))‖ ≤ nXR * ‖y‖)
    (hnXR0 : 0 ≤ nXR) (hlo : ∀ j, lo ≤ DΛ.eig j) (hhi : ∀ j, DΛ.eig j ≤ hi) (hlohi : lo ≤ hi)
    (z : F) :
    ‖rayleighBlock DG X A z - Λ z‖ ≤
      (nXR * ((√(1 - η))⁻¹ * (√(1 - η))⁻¹) + (1 - √(1 - η)) * (hi - lo) * (√(1 - η))⁻¹) *
        ‖z‖ := by
  have hκ : 0 ≤ (√(1 - η))⁻¹ := inv_nonneg.mpr (Real.sqrt_nonneg _)
  have hδ : 0 ≤ 1 - √(1 - η) := delta_nonneg hη0
  have hz : polarM DG (polarMinv DG z) = z := polarM_polarMinv DG hη1 hη z
  have hy_le : ‖polarMinv DG z‖ ≤ (√(1 - η))⁻¹ * ‖z‖ := polarMinv_le DG hη1 hη z
  have e1 : rayleighBlock DG X A z = polarMinv DG (LinearMap.adjoint X
      (A (X (polarMinv DG z)) - X (Λ (polarMinv DG z)))) +
        polarM DG (Λ (polarMinv DG z)) := by
    simp only [rayleighBlock, LinearMap.comp_apply]
    rw [A_polarQ_eq DG hη1 hη z, map_add (LinearMap.adjoint (polarQ DG X)),
      adjoint_polarQ_polarQ DG hη1 hη hG, adjoint_polarQ_apply DG]
  have e2 : polarM DG (Λ (polarMinv DG z)) - Λ (polarM DG (polarMinv DG z)) =
      (polarM DG (Λ (polarMinv DG z) - (((lo + hi) / 2 : ℝ) : 𝕜) • polarMinv DG z) -
          (Λ (polarMinv DG z) - (((lo + hi) / 2 : ℝ) : 𝕜) • polarMinv DG z)) -
        (Λ (polarM DG (polarMinv DG z) - polarMinv DG z) -
          (((lo + hi) / 2 : ℝ) : 𝕜) • (polarM DG (polarMinv DG z) - polarMinv DG z)) := by
    rw [map_sub (polarM DG), map_smul, map_sub Λ, smul_sub]
    abel
  have e3 : rayleighBlock DG X A z - Λ z = polarMinv DG (LinearMap.adjoint X
      (A (X (polarMinv DG z)) - X (Λ (polarMinv DG z)))) +
        (polarM DG (Λ (polarMinv DG z)) - Λ (polarM DG (polarMinv DG z))) := by
    rw [e1, hz]
    abel
  have hsp : ∀ w : F, ‖Λ w - (((lo + hi) / 2 : ℝ) : 𝕜) • w‖ ≤ (hi - lo) / 2 * ‖w‖ := fun w =>
    DΛ.norm_sub_smul_le hΛ (by linarith) (fun j _ => by
      rw [abs_le]
      constructor <;> linarith [hlo j, hhi j])
  have b1 : ‖polarMinv DG (LinearMap.adjoint X
      (A (X (polarMinv DG z)) - X (Λ (polarMinv DG z))))‖ ≤
      nXR * ((√(1 - η))⁻¹ * (√(1 - η))⁻¹) * ‖z‖ := by
    refine (polarMinv_le DG hη1 hη _).trans ?_
    calc (√(1 - η))⁻¹ * ‖LinearMap.adjoint X (A (X (polarMinv DG z)) - X (Λ (polarMinv DG z)))‖
        ≤ (√(1 - η))⁻¹ * (nXR * ‖polarMinv DG z‖) :=
          mul_le_mul_of_nonneg_left (hXR _) hκ
      _ ≤ (√(1 - η))⁻¹ * (nXR * ((√(1 - η))⁻¹ * ‖z‖)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hy_le hnXR0) hκ
      _ = nXR * ((√(1 - η))⁻¹ * (√(1 - η))⁻¹) * ‖z‖ := by ring
  have b2 : ‖polarM DG (Λ (polarMinv DG z) - (((lo + hi) / 2 : ℝ) : 𝕜) • polarMinv DG z) -
      (Λ (polarMinv DG z) - (((lo + hi) / 2 : ℝ) : 𝕜) • polarMinv DG z)‖ ≤
      (1 - √(1 - η)) * ((hi - lo) / 2 * ‖polarMinv DG z‖) :=
    (polarM_sub_le DG hη1 hη hη0 _).trans (mul_le_mul_of_nonneg_left (hsp _) hδ)
  have b3 : ‖Λ (polarM DG (polarMinv DG z) - polarMinv DG z) -
      (((lo + hi) / 2 : ℝ) : 𝕜) • (polarM DG (polarMinv DG z) - polarMinv DG z)‖ ≤
      (hi - lo) / 2 * ((1 - √(1 - η)) * ‖polarMinv DG z‖) :=
    (hsp _).trans (mul_le_mul_of_nonneg_left (polarM_sub_le DG hη1 hη hη0 _) (by linarith))
  have b4 : ‖polarM DG (Λ (polarMinv DG z)) - Λ (polarM DG (polarMinv DG z))‖ ≤
      (1 - √(1 - η)) * (hi - lo) * ((√(1 - η))⁻¹ * ‖z‖) := by
    rw [e2]
    refine (norm_sub_le _ _).trans ?_
    have h5 : (1 - √(1 - η)) * ((hi - lo) / 2 * ‖polarMinv DG z‖) +
        (hi - lo) / 2 * ((1 - √(1 - η)) * ‖polarMinv DG z‖) =
        (1 - √(1 - η)) * (hi - lo) * ‖polarMinv DG z‖ := by ring
    have h6 : (1 - √(1 - η)) * (hi - lo) * ‖polarMinv DG z‖ ≤
        (1 - √(1 - η)) * (hi - lo) * ((√(1 - η))⁻¹ * ‖z‖) :=
      mul_le_mul_of_nonneg_left hy_le (mul_nonneg hδ (by linarith))
    linarith
  rw [e3]
  refine (norm_add_le _ _).trans ?_
  have h7 : nXR * ((√(1 - η))⁻¹ * (√(1 - η))⁻¹) * ‖z‖ +
      (1 - √(1 - η)) * (hi - lo) * ((√(1 - η))⁻¹ * ‖z‖) =
      (nXR * ((√(1 - η))⁻¹ * (√(1 - η))⁻¹) + (1 - √(1 - η)) * (hi - lo) * (√(1 - η))⁻¹) *
        ‖z‖ := by ring
  linarith

end Steps

/-! ### Theorem 2′ -/

/-- **Theorem 2′** (abstract form, Mathlib's decreasing order): there is a strictly increasing,
hence injective, `σ` with
`|λ_{σ j} - μ_j| ≤ nR / √(1 - η) + nXR / (1 - η) + δ (hi - lo) / √(1 - η)`. -/
theorem cluster_eps_star {N g : ℕ} {A : E →ₗ[𝕜] E} (hA : A.IsSymmetric)
    (D : EigenData A (Fin N)) (hD : Antitone D.eig) {Λ : F →ₗ[𝕜] F} (hΛ : Λ.IsSymmetric)
    (DΛ : EigenData Λ (Fin g)) (hDΛ : Antitone DΛ.eig) {X : F →ₗ[𝕜] E} {G : F →ₗ[𝕜] F}
    (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) {κ : Type*} [Fintype κ] [DecidableEq κ]
    (DG : EigenData G κ) {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖G z - z‖ ≤ η * ‖z‖) {nR nXR lo hi : ℝ}
    (hR : ∀ z, ‖A (X z) - X (Λ z)‖ ≤ nR * ‖z‖) (hnR0 : 0 ≤ nR)
    (hXR : ∀ y, ‖LinearMap.adjoint X (A (X y) - X (Λ y))‖ ≤ nXR * ‖y‖) (hnXR0 : 0 ≤ nXR)
    (hlo : ∀ j, lo ≤ DΛ.eig j) (hhi : ∀ j, DΛ.eig j ≤ hi) (hlohi : lo ≤ hi) :
    ∃ σ : Fin g → Fin N, StrictMono σ ∧ ∀ j, |D.eig (σ j) - DΛ.eig j| ≤
      nR / √(1 - η) + nXR / (1 - η) + (1 - √(1 - η)) * (hi - lo) / √(1 - η) := by
  have hH := rayleighBlock_isSymmetric DG (X := X) hA
  have hn : finrank 𝕜 F = g := by rw [DΛ.finrank_eq_card, Fintype.card_fin]
  let DH : EigenData (rayleighBlock DG X A) (Fin g) :=
    ⟨hH.eigenvectorBasis hn, hH.eigenvalues hn, hH.apply_eigenvectorBasis hn⟩
  have hDH : Antitone DH.eig := hH.eigenvalues_antitone hn
  obtain ⟨σ, hσ, hb⟩ := kahan_antitone hA D hD hH DH hDH (polarQ DG X)
    (polarQ_norm DG hη1 hη hG) (rayleigh_residual_le DG hη1 hη hG hR hnR0)
  refine ⟨σ, hσ, fun j => ?_⟩
  have hw := sorted_weyl hH hΛ DH DΛ hDH hDΛ (fun v => by
    rw [norm_sub_rev]
    exact rayleigh_sub_le DG hη1 hη hG hη0 hΛ DΛ hXR hnXR0 hlo hhi hlohi v) j
  have h1 := hb j
  have hpos : 0 < √(1 - η) := Real.sqrt_pos.mpr (by linarith)
  have hsq : √(1 - η) * √(1 - η) = 1 - η := Real.mul_self_sqrt (by linarith)
  have e1 : nXR * ((√(1 - η))⁻¹ * (√(1 - η))⁻¹) = nXR / (1 - η) := by
    rw [← mul_inv, hsq, div_eq_mul_inv]
  have e2 : nR * (√(1 - η))⁻¹ = nR / √(1 - η) := (div_eq_mul_inv _ _).symm
  have e3 : (1 - √(1 - η)) * (hi - lo) * (√(1 - η))⁻¹ =
      (1 - √(1 - η)) * (hi - lo) / √(1 - η) := (div_eq_mul_inv _ _).symm
  have e : nR * (√(1 - η))⁻¹ + (nXR * ((√(1 - η))⁻¹ * (√(1 - η))⁻¹) +
      (1 - √(1 - η)) * (hi - lo) * (√(1 - η))⁻¹) =
      nR / √(1 - η) + nXR / (1 - η) + (1 - √(1 - η)) * (hi - lo) / √(1 - η) := by
    rw [e1, e2, e3]
    ring
  rw [← e]
  calc |D.eig (σ j) - DΛ.eig j| = |(D.eig (σ j) - DH.eig j) + (DH.eig j - DΛ.eig j)| := by
        ring_nf
    _ ≤ |D.eig (σ j) - DH.eig j| + |DH.eig j - DΛ.eig j| := abs_add_le _ _
    _ ≤ _ := add_le_add h1 hw

/-- `min(ρ, ε*)` is a valid common radius: whichever of Theorem 2 and Theorem 2′ gives the
smaller radius supplies the matching. -/
theorem cluster_min {N g : ℕ} {lamA : Fin N → ℝ} {μ : Fin g → ℝ} {ρ ε : ℝ}
    (h2 : ∃ σ : Fin g → Fin N, StrictMono σ ∧ ∀ j, |lamA (σ j) - μ j| ≤ ρ)
    (h2' : ∃ σ : Fin g → Fin N, StrictMono σ ∧ ∀ j, |lamA (σ j) - μ j| ≤ ε) :
    ∃ σ : Fin g → Fin N, StrictMono σ ∧ ∀ j, |lamA (σ j) - μ j| ≤ min ρ ε := by
  rcases le_total ρ ε with h | h
  · obtain ⟨σ, hσ, hb⟩ := h2
    exact ⟨σ, hσ, fun j => (hb j).trans (le_min le_rfl h)⟩
  · obtain ⟨σ, hσ, hb⟩ := h2'
    exact ⟨σ, hσ, fun j => (hb j).trans (le_min h le_rfl)⟩

/-! ### Matrix form -/

open Matrix

variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]

/-- **Theorem 2′** for matrices. `nXR ≥ ‖Xᴴ (A X - X Λ)‖`, `lo ≤ μ_j ≤ hi`. -/
theorem cluster_enclosure_eps_star {A : Matrix n n 𝕜} (hA : A.IsHermitian) {X : Matrix n m 𝕜}
    {Λ : Matrix m m 𝕜} (hΛ : Λ.IsHermitian) {η nR nXR lo hi : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
    (hR : ∀ z, ‖toEuclideanLin (A * X - X * Λ) z‖ ≤ nR * ‖z‖) (hnR0 : 0 ≤ nR)
    (hXR : ∀ z, ‖toEuclideanLin (Xᴴ * (A * X - X * Λ)) z‖ ≤ nXR * ‖z‖) (hnXR0 : 0 ≤ nXR)
    (hlo : ∀ j, lo ≤ hΛ.eigenvalues₀ j) (hhi : ∀ j, hΛ.eigenvalues₀ j ≤ hi) (hlohi : lo ≤ hi) :
    ∃ σ : Fin (Fintype.card m) → Fin (Fintype.card n), StrictMono σ ∧
      ∀ j, |hA.eigenvalues₀ (σ j) - hΛ.eigenvalues₀ j| ≤
        nR / √(1 - η) + nXR / (1 - η) + (1 - √(1 - η)) * (hi - lo) / √(1 - η) :=
  cluster_eps_star (isSymmetric_of_isHermitian hA) (sortedEigenData hA) hA.eigenvalues₀_antitone
    (isSymmetric_of_isHermitian hΛ) (sortedEigenData hΛ) hΛ.eigenvalues₀_antitone
    (inner_toEuclideanLin_gram X)
    (hermitianEigenData (Matrix.isHermitian_conjTranspose_mul_self X)) hη0 hη1 hη
    (fun z => by simpa [map_sub, Matrix.toLpLin_mul_same] using hR z) hnR0
    (fun z => by
      have h := hXR z
      rw [Matrix.toLpLin_mul_same, LinearMap.comp_apply,
        Matrix.toEuclideanLin_conjTranspose_eq_adjoint] at h
      simpa [map_sub, Matrix.toLpLin_mul_same] using h) hnXR0 hlo hhi hlohi

end CertifyMathlib.Enclosure
