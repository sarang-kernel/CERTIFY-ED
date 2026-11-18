import CertifyMathlib.Enclosure.Spectral
import Mathlib.LinearAlgebra.Matrix.Gershgorin
import Mathlib.Analysis.CStarAlgebra.Matrix

/-!
# Theorem 1 (residual enclosure) and Gershgorin

Theorem 1 of `docs/theory.md` §2: for a Hermitian `A`, a real `μ` and `x ≠ 0`, some eigenvalue `λ`
satisfies `|λ - μ| ≤ ‖A x - μ x‖ / ‖x‖`. It is proved here for a symmetric map with an orthonormal
eigenbasis (`EigenData.exists_eig_near`) and for Hermitian matrices (`residual_enclosure`), together
with

* the multi-pair form used by S2 (one Theorem 1 interval per claimed pair; the chosen indices need
  not be distinct, `residual_enclosure_family`),
* the version with the S1 model shift `σ`: the eigenvalues of an operator `H` with
  `‖H - A‖ ≤ σ` are enclosed with radius `‖A x - μ x‖ / ‖x‖ + σ`
  (`residual_enclosure_of_shift`, `residual_enclosure_of_l2_opNorm`),
* Gershgorin's theorem for the eigenvalues of a Hermitian matrix (Mathlib's `eigenvalue_mem_ball`
  specialised, `eigenvalue_mem_gershgorin`) and its use in the code, the bound
  `‖A‖₂ ≤ ‖A‖_∞` for Hermitian `A` (`norm_toEuclideanLin_le_of_row_sum`, `l2_opNorm_le_of_row_sum`).
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

namespace EigenData

variable {T : E →ₗ[𝕜] E} {ι : Type*} [Fintype ι] (D : EigenData T ι)

/-- **Theorem 1** (residual enclosure), operator form. -/
theorem exists_eig_near (hT : T.IsSymmetric) (μ : ℝ) {x : E} (hx : x ≠ 0) :
    ∃ i, |D.eig i - μ| ≤ ‖T x - (μ : 𝕜) • x‖ / ‖x‖ := by
  by_contra h
  push Not at h
  have hpos : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hr : 0 ≤ ‖T x - (μ : 𝕜) • x‖ / ‖x‖ := div_nonneg (norm_nonneg _) hpos.le
  have key := D.sq_mul_norm_sq_lt hT hx (c := μ) hr (fun i _ => h i)
  rw [← mul_pow, div_mul_cancel₀ _ hpos.ne'] at key
  exact lt_irrefl _ key

/-- **Theorem 1 with a model shift.** If `‖T v - S v‖ ≤ σ ‖v‖` for all `v`, the residual of `S`
encloses an eigenvalue of `T` with radius `‖S x - μ x‖ / ‖x‖ + σ`. `S` need not be symmetric. -/
theorem exists_eig_near_of_shift (hT : T.IsSymmetric) (S : E →ₗ[𝕜] E) {σ : ℝ}
    (hσ : ∀ v, ‖T v - S v‖ ≤ σ * ‖v‖) (μ : ℝ) {x : E} (hx : x ≠ 0) :
    ∃ i, |D.eig i - μ| ≤ ‖S x - (μ : 𝕜) • x‖ / ‖x‖ + σ := by
  obtain ⟨i, hi⟩ := D.exists_eig_near hT μ hx
  refine ⟨i, hi.trans ?_⟩
  have hpos : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have h1 : ‖T x - (μ : 𝕜) • x‖ ≤ ‖S x - (μ : 𝕜) • x‖ + σ * ‖x‖ := by
    have e : T x - (μ : 𝕜) • x = (S x - (μ : 𝕜) • x) + (T x - S x) := by abel
    rw [e]
    exact (norm_add_le _ _).trans (add_le_add le_rfl (hσ x))
  calc ‖T x - (μ : 𝕜) • x‖ / ‖x‖ ≤ (‖S x - (μ : 𝕜) • x‖ + σ * ‖x‖) / ‖x‖ :=
        div_le_div_of_nonneg_right h1 hpos.le
    _ = ‖S x - (μ : 𝕜) • x‖ / ‖x‖ + σ := by
        rw [add_div, mul_div_cancel_right₀ _ hpos.ne']

/-- The multi-pair form: every claimed pair gets its own Theorem 1 interval. The index map is not
claimed to be injective (two copies of one vector are enclosed by the same eigenvalue). -/
theorem exists_eig_near_family (hT : T.IsSymmetric) {κ : Type*} (x : κ → E) (μ : κ → ℝ)
    (hx : ∀ k, x k ≠ 0) :
    ∃ f : κ → ι, ∀ k, |D.eig (f k) - μ k| ≤ ‖T (x k) - (μ k : 𝕜) • x k‖ / ‖x k‖ :=
  ⟨fun k => (D.exists_eig_near hT (μ k) (hx k)).choose,
    fun k => (D.exists_eig_near hT (μ k) (hx k)).choose_spec⟩

end EigenData

/-! ### Matrices -/

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **Theorem 1** for a Hermitian matrix: for `x ≠ 0` and real `μ` some eigenvalue lies within
`‖A x - μ x‖ / ‖x‖` of `μ` (Euclidean norms). -/
theorem residual_enclosure {A : Matrix n n 𝕜} (hA : A.IsHermitian) (μ : ℝ)
    {x : EuclideanSpace 𝕜 n} (hx : x ≠ 0) :
    ∃ i, |hA.eigenvalues i - μ| ≤ ‖toEuclideanLin A x - (μ : 𝕜) • x‖ / ‖x‖ :=
  (hermitianEigenData hA).exists_eig_near (isSymmetric_of_isHermitian hA) μ hx

/-- The multi-pair form used by S2: one interval per claimed pair. -/
theorem residual_enclosure_family {A : Matrix n n 𝕜} (hA : A.IsHermitian) {κ : Type*}
    (x : κ → EuclideanSpace 𝕜 n) (μ : κ → ℝ) (hx : ∀ k, x k ≠ 0) :
    ∃ f : κ → n, ∀ k,
      |hA.eigenvalues (f k) - μ k| ≤ ‖toEuclideanLin A (x k) - (μ k : 𝕜) • x k‖ / ‖x k‖ :=
  (hermitianEigenData hA).exists_eig_near_family (isSymmetric_of_isHermitian hA) x μ hx

/-- Theorem 1 with the S1 model shift: the residual is computed with the stored matrix `A`, and
the eigenvalues are those of the Hermitian `H` (the declared model) with
`‖(H - A) v‖ ≤ σ ‖v‖` for every `v`. -/
theorem residual_enclosure_of_shift {H : Matrix n n 𝕜} (hH : H.IsHermitian) (A : Matrix n n 𝕜)
    {σ : ℝ} (hσ : ∀ v, ‖toEuclideanLin (H - A) v‖ ≤ σ * ‖v‖) (μ : ℝ)
    {x : EuclideanSpace 𝕜 n} (hx : x ≠ 0) :
    ∃ i, |hH.eigenvalues i - μ| ≤ ‖toEuclideanLin A x - (μ : 𝕜) • x‖ / ‖x‖ + σ :=
  (hermitianEigenData hH).exists_eig_near_of_shift (isSymmetric_of_isHermitian hH)
    (toEuclideanLin A) (fun v => by simpa [map_sub] using hσ v) μ hx

open scoped Matrix.Norms.L2Operator in
/-- The spectral norm of Mathlib (`Matrix.Norms.L2Operator`) bounds the action on Euclidean
vectors. -/
theorem norm_toEuclideanLin_le_l2_opNorm (M : Matrix n n 𝕜) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin M v‖ ≤ ‖M‖ * ‖v‖ :=
  Matrix.l2_opNorm_mulVec M v

open scoped Matrix.Norms.L2Operator in
/-- Theorem 1 with the model shift stated with the spectral norm: `‖H - A‖₂ ≤ σ`. -/
theorem residual_enclosure_of_l2_opNorm {H : Matrix n n 𝕜} (hH : H.IsHermitian)
    (A : Matrix n n 𝕜) {σ : ℝ} (hσ : ‖H - A‖ ≤ σ)
    (μ : ℝ) {x : EuclideanSpace 𝕜 n} (hx : x ≠ 0) :
    ∃ i, |hH.eigenvalues i - μ| ≤ ‖toEuclideanLin A x - (μ : 𝕜) • x‖ / ‖x‖ + σ :=
  residual_enclosure_of_shift hH A (fun v => (norm_toEuclideanLin_le_l2_opNorm (H - A) v).trans
    (mul_le_mul_of_nonneg_right hσ (norm_nonneg v))) μ hx

/-! ### Gershgorin -/

/-- **Gershgorin** for the eigenvalues of a Hermitian matrix: each `λ_i` lies in some disc
centred at a diagonal entry `A k k` with radius `∑_{j ≠ k} ‖A k j‖`. This is Mathlib's
`eigenvalue_mem_ball` applied to the eigenvector `eigenvectorBasis i`. -/
theorem eigenvalue_mem_gershgorin {A : Matrix n n 𝕜} (hA : A.IsHermitian) (i : n) :
    ∃ k, ‖(hA.eigenvalues i : 𝕜) - A k k‖ ≤ ∑ j ∈ univ.erase k, ‖A k j‖ := by
  have hμ : Module.End.HasEigenvalue (Matrix.toLin' A) (hA.eigenvalues i : 𝕜) := by
    refine Module.End.hasEigenvalue_of_hasEigenvector (x := ⇑(hA.eigenvectorBasis i)) ?_
    refine Module.End.hasEigenvector_iff.mpr ⟨?_, ?_⟩
    · rw [Module.End.mem_eigenspace_iff, Matrix.toLin'_apply, hA.mulVec_eigenvectorBasis,
        RCLike.real_smul_eq_coe_smul (K := 𝕜)]
    · exact (WithLp.ofLp_eq_zero 2).ne.2 (hA.eigenvectorBasis.orthonormal.ne_zero i)
  obtain ⟨k, hk⟩ := eigenvalue_mem_ball hμ
  exact ⟨k, by rwa [Metric.mem_closedBall, dist_eq_norm] at hk⟩

/-- Consequence of Gershgorin: `|λ_i|` is at most some row sum `∑_j ‖A k j‖`. -/
theorem abs_eigenvalue_le_row_sum {A : Matrix n n 𝕜} (hA : A.IsHermitian) (i : n) :
    ∃ k, |hA.eigenvalues i| ≤ ∑ j, ‖A k j‖ := by
  obtain ⟨k, hk⟩ := eigenvalue_mem_gershgorin hA i
  refine ⟨k, ?_⟩
  have h1 := norm_sub_norm_le (hA.eigenvalues i : 𝕜) (A k k)
  rw [RCLike.norm_ofReal] at h1
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k)]
  linarith

/-- The bound used by the code (`adapter.norm_upper`): for Hermitian `A` and any `R ≥ 0` above
every row sum, `‖A v‖ ≤ R ‖v‖` (that is, `‖A‖₂ ≤ ‖A‖_∞`). -/
theorem norm_toEuclideanLin_le_of_row_sum {A : Matrix n n 𝕜} (hA : A.IsHermitian) {Rb : ℝ}
    (hR0 : 0 ≤ Rb) (hR : ∀ k, ∑ j, ‖A k j‖ ≤ Rb) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin A v‖ ≤ Rb * ‖v‖ :=
  (hermitianEigenData hA).norm_apply_le (isSymmetric_of_isHermitian hA) hR0
    (fun i => by
      obtain ⟨k, hk⟩ := abs_eigenvalue_le_row_sum hA i
      exact hk.trans (hR k)) v

open scoped Matrix.Norms.L2Operator in
/-- The same bound for Mathlib's spectral norm. -/
theorem l2_opNorm_le_of_row_sum {A : Matrix n n 𝕜} (hA : A.IsHermitian) {Rb : ℝ}
    (hR0 : 0 ≤ Rb) (hR : ∀ k, ∑ j, ‖A k j‖ ≤ Rb) : ‖A‖ ≤ Rb := by
  rw [Matrix.l2_opNorm_def]
  exact ContinuousLinearMap.opNorm_le_bound _ hR0
    (fun v => norm_toEuclideanLin_le_of_row_sum hA hR0 hR v)

end CertifyMathlib.Enclosure
