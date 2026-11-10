import CertifyMathlib.Enclosure.Kahan
import CertifyMathlib.Enclosure.Weyl

/-!
# Theorem 2 (clusters with non-orthonormal vectors) and the polar-factor lemma

`docs/proofs/thm2_polar-factor.md`. `X : F → E` (the `n × g` matrix of claimed vectors),
`G = Xᴴ X` (given by `⟪X u, X v⟫ = ⟪G u, v⟫`), `η` with `‖G z - z‖ ≤ η ‖z‖` and `0 ≤ η < 1`,
`Λ` symmetric on `F` with eigenvalues `μ_j`, `R = A X - X Λ`, `δ = 1 - √(1 - η)` and
`ρ = ‖R‖ + (‖A‖ + ‖Λ‖) δ`. Then there are `g` distinct indices with `|λ_{i_j} - μ_j| ≤ ρ`.

The square root `M = G^{1/2}` and its inverse are built in the eigenbasis of `G` (`diagMap`):
`M` multiplies the `k`-th coefficient by `√g_k`, `M⁻¹` by `1/√g_k`. The polar factor is
`Q̂ = X M⁻¹`. The lemma's parts are `polarQ_norm` (a: `Q̂` is an isometry), `sqrt_dev_le` and
`polarM_sub_le` (b: `‖M - I‖ ≤ δ`), `polarMinv_le` (b: `‖M⁻¹‖ ≤ (1 - η)^{-1/2}`), `sub_polarQ_le`
(c: `‖X - Q̂‖ ≤ δ`) and `polarQ_residual_le` (d: `‖A Q̂ - Q̂ Λ‖ ≤ ρ`). Theorem 2 is then Kahan's
theorem (`Kahan.lean`) for `Q̂` and the block `Λ` (`cluster_rho`, matrix form `cluster_enclosure`).
Norms of operators are expressed by their defining bound (`‖R z‖ ≤ nR ‖z‖` for all `z`), so any
upper bound on the spectral norm may be used, as in the manuscript.
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
  {F : Type*} [NormedAddCommGroup F] [InnerProductSpace 𝕜 F]

/-! ### Diagonal maps in an orthonormal basis -/

section Diag

variable {κ : Type*} [Fintype κ] [DecidableEq κ] (b : OrthonormalBasis κ 𝕜 F)

/-- The map that multiplies the `k`-th coefficient in the basis `b` by `d k`. -/
noncomputable def diagMap (d : κ → ℝ) : F →ₗ[𝕜] F :=
  (b.repr.symm.toLinearEquiv : EuclideanSpace 𝕜 κ →ₗ[𝕜] F) ∘ₗ
    Matrix.toEuclideanLin (Matrix.diagonal fun k => (d k : 𝕜)) ∘ₗ
    (b.repr.toLinearEquiv : F →ₗ[𝕜] EuclideanSpace 𝕜 κ)

theorem repr_diagMap (d : κ → ℝ) (z : F) (k : κ) :
    b.repr (diagMap b d z) k = (d k : 𝕜) * b.repr z k := by
  simp [diagMap, Matrix.toLpLin_apply, Matrix.mulVec_diagonal]

theorem diagMap_comp (d e : κ → ℝ) (z : F) :
    diagMap b d (diagMap b e z) = diagMap b (fun k => d k * e k) z := by
  apply b.repr.injective
  refine PiLp.ext fun k => ?_
  rw [repr_diagMap, repr_diagMap, repr_diagMap]
  push_cast
  ring

theorem diagMap_one (z : F) : diagMap b (fun _ => 1) z = z := by
  apply b.repr.injective
  refine PiLp.ext fun k => ?_
  rw [repr_diagMap]
  simp

theorem diagMap_congr {d e : κ → ℝ} (h : ∀ k, d k = e k) (z : F) :
    diagMap b d z = diagMap b e z := by
  rw [funext h]

theorem diagMap_isSymmetric (d : κ → ℝ) : (diagMap b d).IsSymmetric := by
  intro u v
  rw [← b.repr.inner_map_map (diagMap b d u) v, ← b.repr.inner_map_map u (diagMap b d v),
    PiLp.inner_apply, PiLp.inner_apply]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [repr_diagMap, repr_diagMap]
  show ⟪(d k : 𝕜) • b.repr u k, b.repr v k⟫_𝕜 = ⟪b.repr u k, (d k : 𝕜) • b.repr v k⟫_𝕜
  rw [inner_smul_left, inner_smul_right, RCLike.conj_ofReal]

/-- The basis `b` diagonalises `diagMap b d`. -/
noncomputable def diagEigenData (d : κ → ℝ) : EigenData (diagMap b d) κ where
  basis := b
  eig := d
  apply_basis k := by
    apply b.repr.injective
    refine PiLp.ext fun j => ?_
    rw [repr_diagMap, map_smul, PiLp.smul_apply, b.repr_self, smul_eq_mul]
    by_cases h : j = k
    · subst h
      rfl
    · simp [h]

end Diag

/-! ### The square-root estimate -/

/-- `|√g - 1| ≤ 1 - √(1 - η)` for `|g - 1| ≤ η < 1` (lemma (b) of the document, by concavity). -/
theorem sqrt_dev_le {g η : ℝ} (hη1 : η < 1) (hg : |g - 1| ≤ η) :
    |√g - 1| ≤ 1 - √(1 - η) := by
  rw [abs_le] at hg
  have h1 : 0 ≤ 1 - η := by linarith
  rcases le_total g 1 with hle | hge
  · have hs1 : √g ≤ 1 := by simpa using Real.sqrt_le_sqrt hle
    have hs2 : √(1 - η) ≤ √g := Real.sqrt_le_sqrt (by linarith)
    rw [abs_of_nonpos (by linarith)]
    linarith
  · have hs1 : 1 ≤ √g := by simpa using Real.sqrt_le_sqrt hge
    have hs2 : √g ≤ √(1 + η) := Real.sqrt_le_sqrt (by linarith)
    have ha := Real.sq_sqrt (show 0 ≤ 1 + η by linarith)
    have hc := Real.sq_sqrt h1
    have ha0 := Real.sqrt_nonneg (1 + η)
    have hc0 := Real.sqrt_nonneg (1 - η)
    have hsum : √(1 + η) + √(1 - η) ≤ 2 := by
      have hsq : (√(1 + η) + √(1 - η)) ^ 2 ≤ 2 ^ 2 := by
        nlinarith [sq_nonneg (√(1 + η) - √(1 - η))]
      exact (pow_le_pow_iff_left₀ (by positivity) (by norm_num) two_ne_zero).mp hsq
    rw [abs_of_nonneg (by linarith)]
    linarith

theorem delta_nonneg {η : ℝ} (hη0 : 0 ≤ η) : 0 ≤ 1 - √(1 - η) := by
  have : √(1 - η) ≤ 1 := by simpa using Real.sqrt_le_sqrt (show 1 - η ≤ 1 by linarith)
  linarith

/-! ### The polar factor -/

section Polar

variable {κ : Type*} [Fintype κ] [DecidableEq κ]
variable {X : F →ₗ[𝕜] E} {G : F →ₗ[𝕜] F}

theorem gram_isSymmetric (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) : G.IsSymmetric := by
  intro u v
  rw [← hG u v, ← inner_conj_symm u (G v), ← hG v u, inner_conj_symm]

omit [DecidableEq κ] in
/-- The eigenvalues of `G` lie in `[1 - η, 1 + η]`. -/
theorem gram_eig_bound (DG : EigenData G κ) {η : ℝ} (hη : ∀ z, ‖G z - z‖ ≤ η * ‖z‖) (k : κ) :
    |DG.eig k - 1| ≤ η := by
  have h := hη (DG.basis k)
  have e : G (DG.basis k) - DG.basis k = ((DG.eig k - 1 : ℝ) : 𝕜) • DG.basis k := by
    rw [DG.apply_basis, RCLike.ofReal_sub, RCLike.ofReal_one, sub_smul, one_smul]
  rw [e, norm_smul, RCLike.norm_ofReal, DG.basis.orthonormal.1 k, mul_one, mul_one] at h
  exact h

omit [DecidableEq κ] in
theorem gram_eig_pos (DG : EigenData G κ) {η : ℝ} (hη1 : η < 1)
    (hη : ∀ z, ‖G z - z‖ ≤ η * ‖z‖) (k : κ) : 0 < DG.eig k := by
  have h := gram_eig_bound DG hη k
  rw [abs_le] at h
  linarith

variable (DG : EigenData G κ)

/-- `M = G^{1/2}`. -/
noncomputable def polarM : F →ₗ[𝕜] F := diagMap DG.basis fun k => √(DG.eig k)

/-- `M⁻¹`. -/
noncomputable def polarMinv : F →ₗ[𝕜] F := diagMap DG.basis fun k => (√(DG.eig k))⁻¹

/-- The polar factor `Q̂ = X M⁻¹`. -/
noncomputable def polarQ (X : F →ₗ[𝕜] E) : F →ₗ[𝕜] E := X ∘ₗ polarMinv DG

variable {η : ℝ} (hη1 : η < 1) (hη : ∀ z, ‖G z - z‖ ≤ η * ‖z‖)
include hη1 hη

theorem polarMinv_polarM (z : F) : polarMinv DG (polarM DG z) = z := by
  rw [polarMinv, polarM, diagMap_comp]
  rw [diagMap_congr DG.basis (e := fun _ => 1) (fun k => by
    have := Real.sqrt_pos.mpr (gram_eig_pos DG hη1 hη k)
    field_simp)]
  exact diagMap_one _ z

theorem polarM_polarMinv (z : F) : polarM DG (polarMinv DG z) = z := by
  rw [polarMinv, polarM, diagMap_comp]
  rw [diagMap_congr DG.basis (e := fun _ => 1) (fun k => by
    have := Real.sqrt_pos.mpr (gram_eig_pos DG hη1 hη k)
    field_simp)]
  exact diagMap_one _ z

omit hη1 hη in
theorem gram_eq_diagMap (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) (z : F) :
    G z = diagMap DG.basis DG.eig z := by
  apply DG.basis.repr.injective
  refine PiLp.ext fun k => ?_
  rw [repr_diagMap, DG.repr_apply (gram_isSymmetric hG)]

/-- `M⁻¹ G = M`. -/
theorem polarMinv_gram (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) (z : F) :
    polarMinv DG (G z) = polarM DG z := by
  rw [gram_eq_diagMap DG hG, polarMinv, polarM, diagMap_comp]
  refine diagMap_congr DG.basis (fun k => ?_) z
  have hpos := gram_eig_pos DG hη1 hη k
  have hs := Real.sq_sqrt hpos.le
  have hs0 := Real.sqrt_pos.mpr hpos
  field_simp
  linarith

/-- Lemma (a): `Q̂` is an isometry. -/
theorem polarQ_norm (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) (z : F) :
    ‖polarQ DG X z‖ = ‖z‖ := by
  have hGs := gram_isSymmetric hG
  have hsq : ‖polarQ DG X z‖ ^ 2 = ‖z‖ ^ 2 := by
    show ‖X (polarMinv DG z)‖ ^ 2 = ‖z‖ ^ 2
    rw [← inner_self_eq_norm_sq (𝕜 := 𝕜), hG, hGs, DG.re_inner_apply hGs, DG.norm_sq_eq]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [polarMinv, repr_diagMap, norm_mul, RCLike.norm_ofReal, mul_pow, sq_abs, inv_pow,
      Real.sq_sqrt (gram_eig_pos DG hη1 hη k).le]
    have := (gram_eig_pos DG hη1 hη k).ne'
    field_simp
  exact (pow_left_inj₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp hsq

/-- Lemma (b): `‖M z - z‖ ≤ δ ‖z‖`. -/
theorem polarM_sub_le (hη0 : 0 ≤ η) (z : F) :
    ‖polarM DG z - z‖ ≤ (1 - √(1 - η)) * ‖z‖ := by
  have h := (diagEigenData DG.basis fun k => √(DG.eig k)).norm_sub_smul_le
    (diagMap_isSymmetric DG.basis _) (c := 1) (v := z) (delta_nonneg hη0)
    (fun k _ => sqrt_dev_le hη1 (gram_eig_bound DG hη k))
  simpa [polarM] using h

/-- Lemma (b), second half: `‖M⁻¹ z‖ ≤ (1 - η)^{-1/2} ‖z‖`. -/
theorem polarMinv_le (z : F) : ‖polarMinv DG z‖ ≤ (√(1 - η))⁻¹ * ‖z‖ := by
  have h1 : 0 < 1 - η := by linarith
  refine (diagEigenData DG.basis fun k => (√(DG.eig k))⁻¹).norm_apply_le
    (diagMap_isSymmetric DG.basis _) (inv_nonneg.mpr (Real.sqrt_nonneg _)) (fun k => ?_) z
  have hpos := gram_eig_pos DG hη1 hη k
  have hb := gram_eig_bound DG hη k
  rw [abs_le] at hb
  simp only [diagEigenData]
  rw [abs_of_nonneg (inv_nonneg.mpr (Real.sqrt_nonneg _))]
  exact inv_anti₀ (Real.sqrt_pos.mpr h1) (Real.sqrt_le_sqrt (by linarith))

/-- `X = Q̂ M`. -/
theorem polarQ_polarM (z : F) : polarQ DG X (polarM DG z) = X z := by
  show X (polarMinv DG (polarM DG z)) = X z
  rw [polarMinv_polarM DG hη1 hη]

/-- Lemma (c): `‖X z - Q̂ z‖ ≤ δ ‖z‖`. -/
theorem sub_polarQ_le (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) (hη0 : 0 ≤ η) (z : F) :
    ‖X z - polarQ DG X z‖ ≤ (1 - √(1 - η)) * ‖z‖ := by
  rw [← polarQ_polarM DG hη1 hη z, ← map_sub, polarQ_norm DG hη1 hη hG]
  exact polarM_sub_le DG hη1 hη hη0 z

/-- Lemma (d): `‖A Q̂ z - Q̂ Λ z‖ ≤ ρ ‖z‖` with `ρ = nR + (nA + nΛ) δ`. -/
theorem polarQ_residual_le (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) (hη0 : 0 ≤ η)
    {A : E →ₗ[𝕜] E} {Λ : F →ₗ[𝕜] F} {nR nA nΛ : ℝ}
    (hR : ∀ z, ‖A (X z) - X (Λ z)‖ ≤ nR * ‖z‖) (hnA0 : 0 ≤ nA)
    (hnA : ∀ v, ‖A v‖ ≤ nA * ‖v‖) (hnΛ : ∀ z, ‖Λ z‖ ≤ nΛ * ‖z‖) (z : F) :
    ‖A (polarQ DG X z) - polarQ DG X (Λ z)‖ ≤ (nR + (nA + nΛ) * (1 - √(1 - η))) * ‖z‖ := by
  have hδ := delta_nonneg hη0
  have e : A (polarQ DG X z) - polarQ DG X (Λ z) =
      (A (X z) - X (Λ z)) - A (X z - polarQ DG X z) + (X (Λ z) - polarQ DG X (Λ z)) := by
    rw [map_sub]
    abel
  rw [e]
  have h1 := hR z
  have h2 : ‖A (X z - polarQ DG X z)‖ ≤ nA * ((1 - √(1 - η)) * ‖z‖) :=
    (hnA _).trans (mul_le_mul_of_nonneg_left (sub_polarQ_le DG hη1 hη hG hη0 z) hnA0)
  have h3 : ‖X (Λ z) - polarQ DG X (Λ z)‖ ≤ (1 - √(1 - η)) * (nΛ * ‖z‖) :=
    (sub_polarQ_le DG hη1 hη hG hη0 (Λ z)).trans (mul_le_mul_of_nonneg_left (hnΛ z) hδ)
  calc ‖(A (X z) - X (Λ z)) - A (X z - polarQ DG X z) + (X (Λ z) - polarQ DG X (Λ z))‖
      ≤ ‖A (X z) - X (Λ z)‖ + ‖A (X z - polarQ DG X z)‖ + ‖X (Λ z) - polarQ DG X (Λ z)‖ :=
        (norm_add_le _ _).trans (add_le_add (norm_sub_le _ _) le_rfl)
    _ ≤ nR * ‖z‖ + nA * ((1 - √(1 - η)) * ‖z‖) + (1 - √(1 - η)) * (nΛ * ‖z‖) := by
        linarith
    _ = (nR + (nA + nΛ) * (1 - √(1 - η))) * ‖z‖ := by ring

end Polar

/-! ### Theorem 2 -/

/-- **Theorem 2** (abstract form, eigenvalues in Mathlib's decreasing order): there is a strictly
increasing, hence injective, `σ` with `|λ_{σ j} - μ_j| ≤ ρ = nR + (nA + nΛ)(1 - √(1 - η))`. -/
theorem cluster_rho [FiniteDimensional 𝕜 E] {N g : ℕ} {A : E →ₗ[𝕜] E} (hA : A.IsSymmetric)
    (D : EigenData A (Fin N)) (hD : Antitone D.eig) {Λ : F →ₗ[𝕜] F} (hΛ : Λ.IsSymmetric)
    (DΛ : EigenData Λ (Fin g)) (hDΛ : Antitone DΛ.eig) {X : F →ₗ[𝕜] E} {G : F →ₗ[𝕜] F}
    (hG : ∀ u v, ⟪X u, X v⟫_𝕜 = ⟪G u, v⟫_𝕜) {κ : Type*} [Fintype κ] [DecidableEq κ]
    (DG : EigenData G κ) {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖G z - z‖ ≤ η * ‖z‖) {nR nA nΛ : ℝ}
    (hR : ∀ z, ‖A (X z) - X (Λ z)‖ ≤ nR * ‖z‖) (hnA0 : 0 ≤ nA)
    (hnA : ∀ v, ‖A v‖ ≤ nA * ‖v‖) (hnΛ : ∀ z, ‖Λ z‖ ≤ nΛ * ‖z‖) :
    ∃ σ : Fin g → Fin N, StrictMono σ ∧
      ∀ j, |D.eig (σ j) - DΛ.eig j| ≤ nR + (nA + nΛ) * (1 - √(1 - η)) :=
  kahan_antitone hA D hD hΛ DΛ hDΛ (polarQ DG X) (polarQ_norm DG hη1 hη hG)
    (polarQ_residual_le DG hη1 hη hG hη0 hR hnA0 hnA hnΛ)

/-! ### Matrix form -/

open Matrix

variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]

theorem inner_toEuclideanLin_gram (X : Matrix n m 𝕜) (u v : EuclideanSpace 𝕜 m) :
    ⟪toEuclideanLin X u, toEuclideanLin X v⟫_𝕜 = ⟪toEuclideanLin (Xᴴ * X) u, v⟫_𝕜 := by
  rw [Matrix.toLpLin_mul_same, LinearMap.comp_apply,
    Matrix.toEuclideanLin_conjTranspose_eq_adjoint, LinearMap.adjoint_inner_left]

/-- **Theorem 2** for matrices: `A` Hermitian `n × n`, `X` an `n × m` matrix of claimed vectors
with `‖Xᴴ X - I‖ ≤ η < 1`, `Λ` Hermitian `m × m`, and bounds `nR ≥ ‖A X - X Λ‖`, `nA ≥ ‖A‖`,
`nΛ ≥ ‖Λ‖`. The eigenvalues of `Λ` are matched to distinct eigenvalues of `A` within
`ρ = nR + (nA + nΛ)(1 - √(1 - η))` (both in Mathlib's sorted order `eigenvalues₀`). -/
theorem cluster_enclosure {A : Matrix n n 𝕜} (hA : A.IsHermitian) {X : Matrix n m 𝕜}
    {Λ : Matrix m m 𝕜} (hΛ : Λ.IsHermitian) {η nR nA nΛ : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
    (hR : ∀ z, ‖toEuclideanLin (A * X - X * Λ) z‖ ≤ nR * ‖z‖)
    (hnA0 : 0 ≤ nA) (hnA : ∀ v, ‖toEuclideanLin A v‖ ≤ nA * ‖v‖)
    (hnΛ : ∀ z, ‖toEuclideanLin Λ z‖ ≤ nΛ * ‖z‖) :
    ∃ σ : Fin (Fintype.card m) → Fin (Fintype.card n), StrictMono σ ∧
      ∀ j, |hA.eigenvalues₀ (σ j) - hΛ.eigenvalues₀ j| ≤ nR + (nA + nΛ) * (1 - √(1 - η)) :=
  cluster_rho (isSymmetric_of_isHermitian hA) (sortedEigenData hA) hA.eigenvalues₀_antitone
    (isSymmetric_of_isHermitian hΛ) (sortedEigenData hΛ) hΛ.eigenvalues₀_antitone
    (inner_toEuclideanLin_gram X)
    (hermitianEigenData (Matrix.isHermitian_conjTranspose_mul_self X)) hη0 hη1 hη
    (fun z => by simpa [map_sub, Matrix.toLpLin_mul_same] using hR z) hnA0 hnA hnΛ

end CertifyMathlib.Enclosure
