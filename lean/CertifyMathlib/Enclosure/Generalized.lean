import CertifyMathlib.Enclosure.Residual
import CertifyMathlib.Counting.Inertia

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

For matrices, the converse half of Lemma 1: `det(A - X B) = det B · (-1)ⁿ · charpoly C` as
polynomials (`pencil_det_eq_charpoly`), so the roots of `det(A - X B)` counted with multiplicity
are the eigenvalues of the Hermitian `C` (`pencil_roots_eq_eigenvalues`), and a pencil eigenpair
`(x, λ)` gives the eigenpair `(Lᴴ x, λ)` of `C` (`reduced_eigenpair_of_pencil`).

**Lemma 3** (counting by inertia), with Sylvester's law of inertia and the counts `cntLt`, `cntLe`
of `CertifyMathlib.Counting.Inertia`: `A - σB = L (C - σI) Lᴴ` (`pencil_sub_eq`), so the numbers of
negative and non-positive eigenvalues of `A - σB` are the numbers of eigenvalues of `C` below and at
or below `σ` (`pencil_inertia`); for a Hermitian `M'` with `|xᴴ(M' - (A - σB))x| ≤ b ‖x‖²` and
`B ⪰ β_c I`, `N(σ - b/β_c) ≤ ν(M') ≤ N(σ + b/β_c) ≤ #{λ_j ≤ σ + b/β_c}` (`pencil_count_perturbed`,
through the Loewner monotonicity `cntLt_anti`).

Not formalized: Propositions 1 to 3 (floating point).
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

/-! ### Lemma 1, the converse half: the pencil eigenvalues with multiplicity (matrices) -/

section MatrixPencil

open Polynomial Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The reduced matrix `C = L⁻¹ A L⁻ᴴ`. -/
noncomputable def reducedMat (L A : Matrix n n 𝕜) : Matrix n n 𝕜 := L⁻¹ * A * (Lᴴ)⁻¹

theorem reducedMat_isHermitian (L : Matrix n n 𝕜) {A : Matrix n n 𝕜} (hA : A.IsHermitian) :
    (reducedMat L A).IsHermitian := by
  unfold reducedMat Matrix.IsHermitian
  rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_mul, Matrix.conjTranspose_nonsing_inv,
    Matrix.conjTranspose_conjTranspose, Matrix.conjTranspose_nonsing_inv, hA.eq, Matrix.mul_assoc]

theorem reducedMat_conj {L : Matrix n n 𝕜} (hL : IsUnit L.det) (A : Matrix n n 𝕜) :
    L * reducedMat L A * Lᴴ = A := by
  have hLH : IsUnit Lᴴ.det := by
    rw [Matrix.det_conjTranspose]
    exact hL.star
  unfold reducedMat
  rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hL, Matrix.one_mul,
    Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hLH, Matrix.mul_one]

/-- **E-03, Lemma 1 (pencil polynomial).** For `B = L Lᴴ` with `L` invertible,
`det(A - X B) = det B · (-1)ⁿ · charpoly(L⁻¹ A L⁻ᴴ)` as polynomials. -/
theorem pencil_det_eq_charpoly {L : Matrix n n 𝕜} (hL : IsUnit L.det) (A : Matrix n n 𝕜) :
    ((C : 𝕜 →+* 𝕜[X]).mapMatrix A - (X : 𝕜[X]) • (C : 𝕜 →+* 𝕜[X]).mapMatrix (L * Lᴴ)).det =
      C (L * Lᴴ).det * (-1) ^ Fintype.card n * (reducedMat L A).charpoly := by
  have hA : (C : 𝕜 →+* 𝕜[X]).mapMatrix A = (C : 𝕜 →+* 𝕜[X]).mapMatrix L *
      (C : 𝕜 →+* 𝕜[X]).mapMatrix (reducedMat L A) * (C : 𝕜 →+* 𝕜[X]).mapMatrix Lᴴ := by
    conv_lhs => rw [← reducedMat_conj hL A]
    rw [map_mul, map_mul]
  have hs : Matrix.scalar n (X : 𝕜[X]) = (X : 𝕜[X]) • (1 : Matrix n n 𝕜[X]) := by
    rw [Matrix.scalar_apply, Matrix.smul_one_eq_diagonal]
  have key : (C : 𝕜 →+* 𝕜[X]).mapMatrix A - (X : 𝕜[X]) • (C : 𝕜 →+* 𝕜[X]).mapMatrix (L * Lᴴ) =
      (C : 𝕜 →+* 𝕜[X]).mapMatrix L * -(charmatrix (reducedMat L A)) *
        (C : 𝕜 →+* 𝕜[X]).mapMatrix Lᴴ := by
    rw [charmatrix, neg_sub, hs, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_smul, Matrix.smul_mul,
      Matrix.mul_one, ← hA, map_mul]
  rw [key, Matrix.det_mul, Matrix.det_mul, Matrix.det_neg, ← Matrix.charpoly, ← RingHom.map_det,
    ← RingHom.map_det, Matrix.det_mul, map_mul]
  ring

/-- **E-03, Lemma 1 (eigenvalues with multiplicity).** The roots of `det(A - X B)`, counted with
multiplicity, are the eigenvalues of the Hermitian matrix `C = L⁻¹ A L⁻ᴴ`. -/
theorem pencil_roots_eq_eigenvalues {L : Matrix n n 𝕜} (hL : IsUnit L.det) {A : Matrix n n 𝕜}
    (hA : A.IsHermitian) :
    ((C : 𝕜 →+* 𝕜[X]).mapMatrix A - (X : 𝕜[X]) • (C : 𝕜 →+* 𝕜[X]).mapMatrix (L * Lᴴ)).det.roots =
      Multiset.map (RCLike.ofReal ∘ (reducedMat_isHermitian L hA).eigenvalues) Finset.univ.val := by
  have hB : (L * Lᴴ).det ≠ 0 := by
    rw [Matrix.det_mul, Matrix.det_conjTranspose]
    exact mul_ne_zero hL.ne_zero (star_ne_zero.mpr hL.ne_zero)
  have hc : (L * Lᴴ).det * (-1) ^ Fintype.card n ≠ 0 :=
    mul_ne_zero hB (pow_ne_zero _ (neg_ne_zero.mpr one_ne_zero))
  rw [pencil_det_eq_charpoly hL A, show C (L * Lᴴ).det * (-1) ^ Fintype.card n =
    C ((L * Lᴴ).det * (-1) ^ Fintype.card n) by simp, Polynomial.roots_C_mul _ hc]
  exact (reducedMat_isHermitian L hA).roots_charpoly_eq_eigenvalues

/-- **E-03, Lemma 1 (eigenvectors, converse direction).** A pencil eigenpair `(x, λ)` gives the
eigenpair `(Lᴴ x, λ)` of `C`. -/
theorem reduced_eigenpair_of_pencil {L : Matrix n n 𝕜} (hL : IsUnit L.det) {A : Matrix n n 𝕜}
    {x : n → 𝕜} {lam : 𝕜} (h : A *ᵥ x = lam • ((L * Lᴴ) *ᵥ x)) :
    reducedMat L A *ᵥ (Lᴴ *ᵥ x) = lam • (Lᴴ *ᵥ x) := by
  have hLH : IsUnit Lᴴ.det := by
    rw [Matrix.det_conjTranspose]
    exact hL.star
  unfold reducedMat
  rw [Matrix.mulVec_mulVec, Matrix.mul_assoc (L⁻¹ * A), Matrix.nonsing_inv_mul _ hLH,
    Matrix.mul_one, ← Matrix.mulVec_mulVec, h, Matrix.mulVec_smul, Matrix.mulVec_mulVec,
    ← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hL, Matrix.one_mul]

end MatrixPencil

section PencilInertia

open Matrix CertifyMathlib.Counting

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

/-- Loewner monotonicity of the counts: `X ⪯ Y` gives `#{λ(Y) < t} ≤ #{λ(X) < t}`. -/
theorem cntLt_anti {X Y : Matrix n n 𝕜} (hX : X.IsHermitian) (hY : Y.IsHermitian)
    (h : ∀ x, qf X x ≤ qf Y x) (t : ℝ) : cntLt hY t ≤ cntLt hX t := by
  obtain ⟨V, h1, h2⟩ := exists_negOn_cntLt hY t
  refine h1.trans (finrank_le_cntLt hX t V ?_)
  intro x hx hx0
  have h3 := h2 x hx hx0
  rw [qf_sub_smul_one] at h3 ⊢
  linarith [h x]

/-- The congruence `A - σB = L (C - σI) Lᴴ` for `B = L Lᴴ` and `C = L⁻¹ A L⁻ᴴ`. -/
theorem pencil_sub_eq {L : Matrix n n 𝕜} (hL : IsUnit L.det) (A : Matrix n n 𝕜) (σ : ℝ) :
    A - (σ : 𝕜) • (L * Lᴴ) =
      L * (reducedMat L A - (σ : 𝕜) • (1 : Matrix n n 𝕜)) * Lᴴ := by
  rw [Matrix.mul_sub, Matrix.sub_mul, reducedMat_conj hL, Matrix.mul_smul, Matrix.mul_one,
    Matrix.smul_mul]

omit [DecidableEq n] in
/-- `A - σ B` is Hermitian for real `σ`. -/
theorem pencil_shift_isHermitian (L : Matrix n n 𝕜) {A : Matrix n n 𝕜} (hA : A.IsHermitian)
    (σ : ℝ) : (A - (σ : 𝕜) • (L * Lᴴ)).IsHermitian := by
  unfold Matrix.IsHermitian
  rw [conjTranspose_sub, conjTranspose_smul, conjTranspose_mul, conjTranspose_conjTranspose,
    hA.eq, RCLike.star_def, RCLike.conj_ofReal]

/-- **E-03, Lemma 3 (first part).** For real `σ` and `B = L Lᴴ` with `L` invertible, the number
of negative (non-positive) eigenvalues of `A - σB` is the number of eigenvalues of
`C = L⁻¹ A L⁻ᴴ` (the pencil eigenvalues, `pencil_roots_eq_eigenvalues`) below (at or below) `σ`:
`ν(A - σB) = N(σ)`. Sylvester's law of inertia from the counting track. -/
theorem pencil_inertia {L : Matrix n n 𝕜} (hL : IsUnit L.det) {A : Matrix n n 𝕜}
    (hA : A.IsHermitian) (σ : ℝ) :
    cntLt (pencil_shift_isHermitian L hA σ) 0 = cntLt (reducedMat_isHermitian L hA) σ ∧
      cntLe (pencil_shift_isHermitian L hA σ) 0 = cntLe (reducedMat_isHermitian L hA) σ := by
  have hC := reducedMat_isHermitian L hA
  have hfac : A - (σ : 𝕜) • (L * Lᴴ) =
      (L * (hC.eigenvectorUnitary : Matrix n n 𝕜)) * dg (fun i => hC.eigenvalues i - σ) *
        (L * (hC.eigenvectorUnitary : Matrix n n 𝕜))ᴴ := by
    rw [pencil_sub_eq hL A σ, sub_smul_one_eq hC σ, conjTranspose_mul]
    simp only [Matrix.mul_assoc]
  have hunit : IsUnit (L * (hC.eigenvectorUnitary : Matrix n n 𝕜)) := by
    have hLu : IsUnit L := (Matrix.isUnit_iff_isUnit_det L).mpr hL
    have hUu : IsUnit (hC.eigenvectorUnitary : Matrix n n 𝕜) :=
      ⟨⟨_, _, unitary_mul_conjTranspose hC.eigenvectorUnitary,
        conjTranspose_mul_unitary hC.eigenvectorUnitary⟩, rfl⟩
    exact hLu.mul hUu
  obtain ⟨h1, h2⟩ := sylvester_inertia (pencil_shift_isHermitian L hA σ) _ hunit _ hfac
  refine ⟨h1.trans ?_, h2.trans ?_⟩
  · simp only [cntLt, sub_neg]
  · simp only [cntLe, sub_nonpos]

omit [DecidableEq n] in
/-- The quadratic form of `A - c B`. -/
theorem qf_pencil (L A : Matrix n n 𝕜) (c : ℝ) (x : n → 𝕜) :
    qf (A - (c : 𝕜) • (L * Lᴴ)) x = qf A x - c * qf (L * Lᴴ) x := by
  rw [qf_sub, qf_smul]

/-- **E-03, Lemma 3 (second part).** Let `B = L Lᴴ ⪰ β_c I` with `β_c > 0`, and let a Hermitian
`M'` satisfy `|xᴴ (M' - (A - σB)) x| ≤ b ‖x‖²` for all `x` (which `‖M' - (A - σB)‖₂ ≤ b` gives).
Then the number `ν` of negative eigenvalues of `M'` satisfies
`N(σ - b/β_c) ≤ ν ≤ N(σ + b/β_c) ≤ #{j : λ_j ≤ σ + b/β_c}`. -/
theorem pencil_count_perturbed {L : Matrix n n 𝕜} (hL : IsUnit L.det) {A M' : Matrix n n 𝕜}
    (hA : A.IsHermitian) (hM' : M'.IsHermitian) {σ b βc : ℝ} (hb : 0 ≤ b) (hβ : 0 < βc)
    (hB : ∀ x, βc * nsq x ≤ qf (L * Lᴴ) x)
    (hpert : ∀ x, |qf (M' - (A - (σ : 𝕜) • (L * Lᴴ))) x| ≤ b * nsq x) :
    cntLt (reducedMat_isHermitian L hA) (σ - b / βc) ≤ cntLt hM' 0 ∧
      cntLt hM' 0 ≤ cntLt (reducedMat_isHermitian L hA) (σ + b / βc) ∧
      cntLt hM' 0 ≤ cntLe (reducedMat_isHermitian L hA) (σ + b / βc) := by
  have hc : ∀ x, b * nsq x ≤ b / βc * qf (L * Lᴴ) x := by
    intro x
    have h1 := mul_le_mul_of_nonneg_left (hB x) (div_nonneg hb hβ.le)
    have e : b / βc * (βc * nsq x) = b * nsq x := by field_simp
    linarith
  have hq : ∀ x, |qf M' x - qf (A - (σ : 𝕜) • (L * Lᴴ)) x| ≤ b * nsq x := by
    intro x
    rw [← qf_sub]
    exact hpert x
  have hlo : cntLt (pencil_shift_isHermitian L hA (σ - b / βc)) 0 ≤ cntLt hM' 0 := by
    refine cntLt_anti hM' (pencil_shift_isHermitian L hA (σ - b / βc)) (fun x => ?_) 0
    have h1 := (abs_le.mp (hq x)).2
    have h2 := hc x
    rw [qf_pencil] at h1 ⊢
    nlinarith
  have hhi : cntLt hM' 0 ≤ cntLt (pencil_shift_isHermitian L hA (σ + b / βc)) 0 := by
    refine cntLt_anti (pencil_shift_isHermitian L hA (σ + b / βc)) hM' (fun x => ?_) 0
    have h1 := (abs_le.mp (hq x)).1
    have h2 := hc x
    rw [qf_pencil] at h1 ⊢
    nlinarith
  rw [(pencil_inertia hL hA _).1] at hlo hhi
  exact ⟨hlo, hhi, hhi.trans (cntLt_le_cntLe _ _)⟩

end PencilInertia

end CertifyMathlib.Enclosure
