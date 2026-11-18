import CertifyMathlib.Counting.DavisKahan
import Mathlib.Analysis.InnerProductSpace.Trace

/-!
# Theorem 6, second and third bounds, and the polar decomposition `X = Q̂ M` as matrices

Source: `docs/theory.md` §4, `thm:dk` of `manuscript_v4/sections/s3_math.tex`, and
`docs/proofs/E-07_davis_kahan.md` (notation) in the code repository.

* `polarSqrt X = M = (Xᴴ X)^{1/2}` and `polarSqrtInv X = M⁻¹` as matrices, built from the
  enclosure part (`CertifyMathlib.Enclosure.polarM`, `polarMinv`): `M` and `M⁻¹` are Hermitian,
  `M M⁻¹ = M⁻¹ M = 1`, `M M = Xᴴ X`, `X = Q̂ M`, `Q̂ = X M⁻¹`, `‖M - I‖ ≤ δ`,
  `‖M⁻¹‖ ≤ (1 - η)^{-1/2}`.
* `mul_conjTranspose_sub_polar`: `‖X Xᴴ - Q̂ Q̂ᴴ‖ ≤ η`, since `X Xᴴ - Q̂ Q̂ᴴ = Q̂ (Xᴴ X - I) Q̂ᴴ`.
* `thm6_X`: Theorem 6, second bound, `‖X Xᴴ - P‖ ≤ s + η`.
* `norm_trace_mul_le`: `|tr(E O)| ≤ rank(E) ‖E‖ ‖O‖` for Hermitian `E`, through an orthonormal
  eigenbasis of `E`; `rank_sub_le'`: `rank(A - B) ≤ rank A + rank B`.
* `thm6_trace`: Theorem 6, third bound, `|tr(Xᴴ O X) - tr(P O)| ≤ 2 g ‖O‖ (s + η)`.
-/

namespace CertifyMathlib.Counting

open Matrix
open scoped InnerProductSpace

variable {𝕜 : Type*} [RCLike 𝕜]

section PolarParts

variable {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g] [DecidableEq g]

/-- `M = (Xᴴ X)^{1/2}` as a matrix. -/
noncomputable def polarSqrt (X : Matrix n g 𝕜) : Matrix g g 𝕜 :=
  Matrix.toEuclideanLin.symm
    (Enclosure.polarM (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self X)))

/-- `M⁻¹` as a matrix. -/
noncomputable def polarSqrtInv (X : Matrix n g 𝕜) : Matrix g g 𝕜 :=
  Matrix.toEuclideanLin.symm
    (Enclosure.polarMinv (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self X)))

omit [DecidableEq n] in
theorem toEuclideanLin_polarSqrt (X : Matrix n g 𝕜) :
    toEuclideanLin (polarSqrt X) =
      Enclosure.polarM (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self X)) :=
  LinearEquiv.apply_symm_apply _ _

omit [DecidableEq n] in
theorem toEuclideanLin_polarSqrtInv (X : Matrix n g 𝕜) :
    toEuclideanLin (polarSqrtInv X) =
      Enclosure.polarMinv (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self X)) :=
  LinearEquiv.apply_symm_apply _ _

omit [Fintype n] [DecidableEq n] in
/-- A matrix whose map is symmetric is Hermitian. -/
theorem conjTranspose_eq_of_isSymmetric (M : Matrix g g 𝕜) (h : (toEuclideanLin M).IsSymmetric) :
    Mᴴ = M := by
  apply Matrix.toEuclideanLin.injective
  refine LinearMap.ext fun u => ext_inner_right 𝕜 fun v => ?_
  rw [inner_conjTranspose_apply, h u v]

omit [DecidableEq n] in
theorem polarSqrt_conjTranspose (X : Matrix n g 𝕜) : (polarSqrt X)ᴴ = polarSqrt X :=
  conjTranspose_eq_of_isSymmetric _ (by
    rw [toEuclideanLin_polarSqrt]
    exact Enclosure.diagMap_isSymmetric _ _)

omit [DecidableEq n] in
theorem polarSqrtInv_conjTranspose (X : Matrix n g 𝕜) : (polarSqrtInv X)ᴴ = polarSqrtInv X :=
  conjTranspose_eq_of_isSymmetric _ (by
    rw [toEuclideanLin_polarSqrtInv]
    exact Enclosure.diagMap_isSymmetric _ _)

omit [DecidableEq n] in
/-- `Q̂ = X M⁻¹`. -/
theorem polarMat_eq (X : Matrix n g 𝕜) : polarMat X = X * polarSqrtInv X := by
  apply Matrix.toEuclideanLin.injective
  refine LinearMap.ext fun z => ?_
  rw [toEuclideanLin_mul_apply, toEuclideanLin_polarMat, toEuclideanLin_polarSqrtInv]
  rfl

variable {X : Matrix n g 𝕜} {η : ℝ} (hη1 : η < 1)
  (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
include hη1 hη

omit [DecidableEq n] in
theorem polarSqrt_mul_inv : polarSqrt X * polarSqrtInv X = 1 := by
  apply Matrix.toEuclideanLin.injective
  refine LinearMap.ext fun z => ?_
  rw [toEuclideanLin_mul_apply, toEuclideanLin_polarSqrt, toEuclideanLin_polarSqrtInv,
    toEuclideanLin_one_apply]
  exact Enclosure.polarM_polarMinv _ hη1 hη z

omit [DecidableEq n] in
theorem polarSqrtInv_mul : polarSqrtInv X * polarSqrt X = 1 := by
  apply Matrix.toEuclideanLin.injective
  refine LinearMap.ext fun z => ?_
  rw [toEuclideanLin_mul_apply, toEuclideanLin_polarSqrt, toEuclideanLin_polarSqrtInv,
    toEuclideanLin_one_apply]
  exact Enclosure.polarMinv_polarM _ hη1 hη z

omit [DecidableEq n] in
/-- `X = Q̂ M`. -/
theorem polarMat_mul_polarSqrt : polarMat X * polarSqrt X = X := by
  apply Matrix.toEuclideanLin.injective
  refine LinearMap.ext fun z => ?_
  rw [toEuclideanLin_mul_apply, toEuclideanLin_polarMat, toEuclideanLin_polarSqrt]
  exact Enclosure.polarQ_polarM _ hη1 hη z

/-- `M M = Xᴴ X`. -/
theorem polarSqrt_mul_self : polarSqrt X * polarSqrt X = Xᴴ * X := by
  apply Matrix.toEuclideanLin.injective
  refine LinearMap.ext fun z => ?_
  rw [toEuclideanLin_mul_apply, toEuclideanLin_polarSqrt]
  unfold Enclosure.polarM
  rw [Enclosure.diagMap_comp,
    Enclosure.gram_eq_diagMap (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self X))
      (Enclosure.inner_toEuclideanLin_gram X) z]
  exact Enclosure.diagMap_congr _
    (fun k => Real.mul_self_sqrt (Enclosure.gram_eig_pos _ hη1 hη k).le) z

omit [DecidableEq n] in
/-- `‖M⁻¹‖ ≤ (1 - η)^{-1/2}`. -/
theorem polarSqrtInv_le (z : EuclideanSpace 𝕜 g) :
    ‖toEuclideanLin (polarSqrtInv X) z‖ ≤ (√(1 - η))⁻¹ * ‖z‖ := by
  rw [toEuclideanLin_polarSqrtInv]
  exact Enclosure.polarMinv_le _ hη1 hη z

omit [DecidableEq n] in
/-- `‖M - I‖ ≤ δ`. -/
theorem polarSqrt_sub_le (hη0 : 0 ≤ η) (z : EuclideanSpace 𝕜 g) :
    ‖toEuclideanLin (polarSqrt X - 1) z‖ ≤ (1 - √(1 - η)) * ‖z‖ := by
  rw [map_sub, LinearMap.sub_apply, toEuclideanLin_one_apply, toEuclideanLin_polarSqrt]
  exact Enclosure.polarM_sub_le _ hη1 hη hη0 z

/-- `‖X Xᴴ - Q̂ Q̂ᴴ‖ ≤ η`. -/
theorem mul_conjTranspose_sub_polar (hη0 : 0 ≤ η) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (X * Xᴴ - polarMat X * (polarMat X)ᴴ) v‖ ≤ η * ‖v‖ := by
  have hX : X * Xᴴ = polarMat X * (Xᴴ * X) * (polarMat X)ᴴ := by
    calc X * Xᴴ = (polarMat X * polarSqrt X) * (polarMat X * polarSqrt X)ᴴ := by
          rw [polarMat_mul_polarSqrt hη1 hη]
      _ = polarMat X * (polarSqrt X * polarSqrt X) * (polarMat X)ᴴ := by
          rw [conjTranspose_mul, polarSqrt_conjTranspose]
          simp only [Matrix.mul_assoc]
      _ = polarMat X * (Xᴴ * X) * (polarMat X)ᴴ := by rw [polarSqrt_mul_self hη1 hη]
  have e : X * Xᴴ - polarMat X * (polarMat X)ᴴ =
      polarMat X * (Xᴴ * X - 1) * (polarMat X)ᴴ := by
    rw [hX, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one]
  have hQ := polarMat_conjTranspose_mul hη1 hη
  rw [e, toEuclideanLin_mul_apply, toEuclideanLin_mul_apply, norm_apply_of_isometry _ hQ]
  have h1 : ‖toEuclideanLin (Xᴴ * X - 1) (toEuclideanLin (polarMat X)ᴴ v)‖ ≤
      η * ‖toEuclideanLin (polarMat X)ᴴ v‖ := by
    rw [map_sub, LinearMap.sub_apply, toEuclideanLin_one_apply]
    exact hη _
  have h2 : ‖toEuclideanLin (polarMat X)ᴴ v‖ ≤ 1 * ‖v‖ :=
    norm_conjTranspose_le _ zero_le_one (fun z => by
      rw [norm_apply_of_isometry _ hQ, one_mul]) v
  rw [one_mul] at h2
  exact h1.trans (mul_le_mul_of_nonneg_left h2 hη0)

end PolarParts

section Trace

variable {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g] [DecidableEq g]

/-- The trace of a matrix in an orthonormal basis of `EuclideanSpace 𝕜 n`. -/
theorem trace_eq_sum_inner_matrix (M : Matrix n n 𝕜) {ι : Type*} [Fintype ι]
    (b : OrthonormalBasis ι 𝕜 (EuclideanSpace 𝕜 n)) :
    M.trace = ∑ i, ⟪b i, toEuclideanLin M (b i)⟫_𝕜 := by
  rw [← LinearMap.trace_eq_sum_inner]
  show _ = LinearMap.trace 𝕜 _
    (Matrix.toLin (PiLp.basisFun 2 𝕜 n) (PiLp.basisFun 2 𝕜 n) M)
  rw [Matrix.trace_toLin_eq]

/-- `|tr(E O)| ≤ #{nonzero eigenvalues of E} ‖E‖ ‖O‖`. -/
theorem norm_trace_mul_le (E O : Matrix n n 𝕜) {ι : Type*} [Fintype ι]
    (D : Enclosure.EigenData (toEuclideanLin E) ι) {K nO : ℝ}
    (hK : ∀ v, ‖toEuclideanLin E v‖ ≤ K * ‖v‖)
    (hO : ∀ v, ‖toEuclideanLin O v‖ ≤ nO * ‖v‖) :
    ‖trace (E * O)‖ ≤ (Finset.univ.filter fun i => D.eig i ≠ 0).card * K * nO := by
  rw [Matrix.trace_mul_comm, trace_eq_sum_inner_matrix _ D.basis]
  have hunit : ∀ i, ‖D.basis i‖ = 1 := fun i => D.basis.orthonormal.1 i
  have hterm : ∀ i, ⟪D.basis i, toEuclideanLin (O * E) (D.basis i)⟫_𝕜 =
      (D.eig i : 𝕜) * ⟪D.basis i, toEuclideanLin O (D.basis i)⟫_𝕜 := fun i => by
    rw [toEuclideanLin_mul_apply, D.apply_basis, map_smul, inner_smul_right]
  have heig : ∀ i, |D.eig i| ≤ K := fun i => by
    have h := hK (D.basis i)
    rw [D.apply_basis, norm_smul, RCLike.norm_ofReal, hunit, mul_one, mul_one] at h
    exact h
  have hm : ∀ i, ‖⟪D.basis i, toEuclideanLin O (D.basis i)⟫_𝕜‖ ≤ nO := fun i => by
    refine (norm_inner_le_norm _ _).trans ?_
    rw [hunit, one_mul]
    exact (hO _).trans (by rw [hunit, mul_one])
  have hK0 : ∀ i, D.eig i ≠ 0 → 0 ≤ K := fun i _ => (abs_nonneg _).trans (heig i)
  calc ‖∑ i, ⟪D.basis i, toEuclideanLin (O * E) (D.basis i)⟫_𝕜‖
      ≤ ∑ i, ‖⟪D.basis i, toEuclideanLin (O * E) (D.basis i)⟫_𝕜‖ := norm_sum_le _ _
    _ ≤ ∑ i, (if D.eig i ≠ 0 then K * nO else 0) := by
        refine Finset.sum_le_sum fun i _ => ?_
        rw [hterm, norm_mul, RCLike.norm_ofReal]
        by_cases h0 : D.eig i = 0
        · simp [h0]
        · simp only [ne_eq, h0, not_false_eq_true, ↓reduceIte]
          exact mul_le_mul (heig i) (hm i) (norm_nonneg _) (hK0 i h0)
    _ = (Finset.univ.filter fun i => D.eig i ≠ 0).card * K * nO := by
        rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul, mul_assoc]

omit [DecidableEq n] [Fintype g] [DecidableEq g] in
theorem rank_sub_le' {m : Type*} [Fintype m] (A B : Matrix m n 𝕜) :
    (A - B).rank ≤ A.rank + B.rank := by
  unfold Matrix.rank
  calc Module.finrank 𝕜 (LinearMap.range (A - B).mulVecLin) ≤
        Module.finrank 𝕜 ↥(LinearMap.range A.mulVecLin ⊔ LinearMap.range B.mulVecLin) := by
        refine Submodule.finrank_mono ?_
        rintro _ ⟨v, rfl⟩
        refine Submodule.mem_sup.mpr ⟨A *ᵥ v, ⟨v, rfl⟩, -(B *ᵥ v),
          Submodule.neg_mem _ ⟨v, rfl⟩, ?_⟩
        simp [sub_eq_add_neg]
    _ ≤ _ := Submodule.finrank_add_le_finrank_add_finrank _ _

variable {A : Matrix n n 𝕜} (hA : A.IsHermitian)

/-- **Theorem 6**, second bound: `‖X Xᴴ - P‖ ≤ s + η`. -/
theorem thm6_X (T : Finset n) {X : Matrix n g 𝕜} {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) {s : ℝ}
    (hs : ∀ v, ‖toEuclideanLin (polarMat X * (polarMat X)ᴴ - specProj hA T) v‖ ≤ s * ‖v‖)
    (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (X * Xᴴ - specProj hA T) v‖ ≤ (s + η) * ‖v‖ := by
  have e : X * Xᴴ - specProj hA T = (X * Xᴴ - polarMat X * (polarMat X)ᴴ) +
      (polarMat X * (polarMat X)ᴴ - specProj hA T) := by abel
  rw [e, map_add, LinearMap.add_apply]
  refine (norm_add_le _ _).trans ?_
  have h1 := mul_conjTranspose_sub_polar hη1 hη hη0 v
  have h2 := hs v
  linarith

omit [DecidableEq g] in
/-- **Theorem 6**, third bound: `|tr(Xᴴ O X) - tr(P O)| ≤ 2 g ‖O‖ (s + η)` (here with
`g + #T` in place of `2 g`; the two are equal when `#T = g`). -/
theorem thm6_trace (T : Finset n) {X : Matrix n g 𝕜} {K : ℝ} (hK0 : 0 ≤ K)
    (hK : ∀ v, ‖toEuclideanLin (X * Xᴴ - specProj hA T) v‖ ≤ K * ‖v‖)
    (O : Matrix n n 𝕜) {nO : ℝ} (hnO : 0 ≤ nO) (hO : ∀ v, ‖toEuclideanLin O v‖ ≤ nO * ‖v‖) :
    ‖trace (Xᴴ * O * X) - trace (specProj hA T * O)‖ ≤
      (Fintype.card g + T.card) * K * nO := by
  have htr : trace (Xᴴ * O * X) - trace (specProj hA T * O) =
      trace ((X * Xᴴ - specProj hA T) * O) := by
    rw [Matrix.sub_mul, trace_sub, Matrix.trace_mul_comm (Xᴴ * O) X, ← Matrix.mul_assoc]
  rw [htr]
  rw [← eigCols_mul_conjTranspose hA T] at hK ⊢
  have hE : (X * Xᴴ - eigCols hA T * (eigCols hA T)ᴴ).IsHermitian :=
    (isHermitian_mul_conjTranspose_self X).sub (isHermitian_mul_conjTranspose_self _)
  have h := norm_trace_mul_le _ O (Enclosure.hermitianEigenData hE) hK hO
  have hrank : (Finset.univ.filter fun i => (Enclosure.hermitianEigenData hE).eig i ≠ 0).card ≤
      Fintype.card g + T.card := by
    rw [Enclosure.hermitianEigenData_eig, ← Fintype.card_subtype,
      ← hE.rank_eq_card_non_zero_eigs]
    refine (rank_sub_le' _ _).trans (add_le_add ?_ ?_)
    · exact (rank_mul_le_left _ _).trans (rank_le_card_width _)
    · rw [← Fintype.card_coe T]
      exact (rank_mul_le_left _ _).trans (rank_le_card_width _)
  refine h.trans ?_
  rw [mul_assoc, mul_assoc]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast hrank) (mul_nonneg hK0 hnO)

end Trace

end CertifyMathlib.Counting
