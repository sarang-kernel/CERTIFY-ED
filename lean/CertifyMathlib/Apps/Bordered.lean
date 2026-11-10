import CertifyMathlib.Counting.Inertia

/-!
# The bordered-pencil lemma behind the E10 eta bounds

`certify_ed/apps/aklt/eta.py` bounds `eta^2`, the largest squared cosine other than 1 between two ground
spaces, by counting eigenvalues of the pencil `(C G_B⁻¹ Cᵀ, G_A)` above a threshold `t = p / q` without
forming an inverse: the number of positive eigenvalues of the bordered matrix
`M = [[-p G_A, q C], [q Cᵀ, -q G_B]]` (Haynsworth inertia additivity, `G_B > 0`) is the number of pencil
eigenvalues above `t`. A floating-point `L D Lᵀ` of `M` is checked exactly, `‖L D Lᵀ - M‖ ≤ q β`, and
with certified lower bounds `μ ≤ λ_min(G_B)`, `ν ≤ λ_min(G_A)` the code concludes that at most
`#(positive pivots)` pencil eigenvalues exceed `t' = (1 + β/μ)(t + β/ν)`.

This file proves that chain for real matrices, in terms of the counts `cntLt` of
`CertifyMathlib.Counting.Inertia`, applied to `N = -M` (positive eigenvalues of `M` are negative
eigenvalues of `N`):

* `bordered_inertia`: for `G > 0`, the bordered matrix `[[X, -q C], [-q Cᴴ, q G]]` has as many negative
  eigenvalues as its Schur complement `X - q C G⁻¹ Cᴴ` (Haynsworth);
* `bordered_pencil`: with `X = p G_A`, the count is the number of pencil eigenvalues above `p / q`,
  written as the number of negative eigenvalues of `(p/q) G_A - C G_B⁻¹ Cᴴ`;
* `qf_inv_shift`: `G_B⁻¹ ≤ (1 + β/μ) (G_B + β)⁻¹` in the Loewner order when `G_B ≥ μ`;
* `eta_count`: if `N'` is within `q β` of `N` in the quadratic form (the exact residual bound of the
  factorization), then the number of pencil eigenvalues above `t'` is at most the number of negative
  eigenvalues of `N'` (by Sylvester's law, the number of positive pivots of the factorization of `M`).
-/

namespace CertifyMathlib.Apps

open Matrix CertifyMathlib.Counting

section Basic

variable {n : Type*} [Fintype n]

theorem qf_real (M : Matrix n n ℝ) (x : n → ℝ) : qf M x = x ⬝ᵥ (M *ᵥ x) := by
  simp [qf]

theorem nsq_real (x : n → ℝ) : nsq x = x ⬝ᵥ x := by
  simp only [nsq, dotProduct, Real.norm_eq_abs, sq, abs_mul_abs_self]

theorem nsq_pos {x : n → ℝ} (hx : x ≠ 0) : 0 < nsq x := by
  obtain ⟨j, hj⟩ := exists_ne_zero_apply hx
  unfold nsq
  exact Finset.sum_pos' (fun i _ => sq_nonneg _) ⟨j, Finset.mem_univ _, pow_pos (norm_pos_iff.mpr hj) 2⟩

theorem qf_smul_real (s : ℝ) (M : Matrix n n ℝ) (x : n → ℝ) : qf (s • M) x = s * qf M x := by
  simp only [qf_real, smul_mulVec, dotProduct_smul, smul_eq_mul]

theorem qf_smul_one_real [DecidableEq n] (s : ℝ) (x : n → ℝ) : qf (s • (1 : Matrix n n ℝ)) x = s * nsq x := by
  rw [qf_smul_real, qf_real, nsq_real, one_mulVec]

theorem qf_smul_vec (M : Matrix n n ℝ) (c : ℝ) (x : n → ℝ) : qf M (c • x) = c ^ 2 * qf M x := by
  simp only [qf_real, mulVec_smul, dotProduct_smul, smul_dotProduct, smul_eq_mul]
  ring

omit [Fintype n] in
theorem transpose_eq_of_isHermitian {Y : Matrix n n ℝ} (hY : Y.IsHermitian) : Yᵀ = Y := by
  rw [← conjTranspose_eq_transpose_of_trivial]
  exact hY.eq

theorem dotProduct_mulVec_symm {Y : Matrix n n ℝ} (hY : Y.IsHermitian) (v w : n → ℝ) :
    v ⬝ᵥ (Y *ᵥ w) = w ⬝ᵥ (Y *ᵥ v) := by
  rw [dotProduct_mulVec, dotProduct_comm]
  congr 1
  rw [← transpose_eq_of_isHermitian hY, vecMul_transpose, transpose_eq_of_isHermitian hY]

/-- A matrix with `μ ‖y‖² ≤ yᵀ G y`, `μ > 0`, is invertible. -/
theorem isUnit_det_of_qf_ge [DecidableEq n] {G : Matrix n n ℝ} {μ : ℝ} (hμ : 0 < μ) (h : ∀ y, μ * nsq y ≤ qf G y) :
    IsUnit G.det := by
  rw [isUnit_iff_ne_zero]
  intro hdet
  obtain ⟨v, hv0, hv⟩ := (Matrix.exists_mulVec_eq_zero_iff).mpr hdet
  have h1 := h v
  rw [qf_real, hv, dotProduct_zero] at h1
  have h2 := nsq_pos hv0
  nlinarith

end Basic

/-! ## Counts at zero through negative subspaces -/

section Counts

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem exists_negOn_zero {A : Matrix n n ℝ} (hA : A.IsHermitian) :
    ∃ V : Submodule ℝ (n → ℝ), cntLt hA 0 ≤ Module.finrank ℝ V ∧ ∀ x ∈ V, x ≠ 0 → qf A x < 0 := by
  obtain ⟨V, h1, h2⟩ := exists_negOn_cntLt hA 0
  refine ⟨V, h1, fun x hx hx0 => ?_⟩
  have h := h2 x hx hx0
  simpa using h

theorem finrank_le_cntLt_zero {A : Matrix n n ℝ} (hA : A.IsHermitian) (V : Submodule ℝ (n → ℝ))
    (hV : ∀ x ∈ V, x ≠ 0 → qf A x < 0) : Module.finrank ℝ V ≤ cntLt hA 0 :=
  finrank_le_cntLt hA 0 V (fun x hx hx0 => by simpa using hV x hx hx0)

/-- If `xᵀ B x ≤ c xᵀ A x` for all `x` with `c > 0`, then `B` has at least as many negative
eigenvalues as `A`. -/
theorem cntLt_le_of_qf_le {A B : Matrix n n ℝ} (hA : A.IsHermitian) (hB : B.IsHermitian) (c : ℝ)
    (hc : 0 < c) (h : ∀ x, qf B x ≤ c * qf A x) : cntLt hA 0 ≤ cntLt hB 0 := by
  obtain ⟨V, h1, h2⟩ := exists_negOn_zero hA
  exact h1.trans (finrank_le_cntLt_zero hB V fun x hx hx0 =>
    lt_of_le_of_lt (h x) (mul_neg_of_pos_of_neg hc (h2 x hx hx0)))

/-- Sylvester's law of inertia for a congruence `A = K B Kᴴ` with `Kᴴ` invertible (`B` need not be
diagonal). -/
theorem cntLt_congr {A B : Matrix n n ℝ} (hA : A.IsHermitian) (hB : B.IsHermitian)
    (K J : Matrix n n ℝ) (h1 : Kᴴ * J = 1) (h2 : J * Kᴴ = 1) (hAB : A = K * B * Kᴴ) :
    cntLt hA 0 = cntLt hB 0 := by
  let e := mulVecEquiv Kᴴ J h1 h2
  have he : ∀ x, qf A x = qf B (e x) := fun x => by rw [hAB, qf_congr, mulVecEquiv_apply]
  have he' : ∀ y, qf B y = qf A (e.symm y) := fun y => by rw [he, e.apply_symm_apply]
  apply le_antisymm
  · obtain ⟨V, h1V, h2V⟩ := exists_negOn_zero hA
    have := finrank_le_cntLt_zero hB (V.map (e : (n → ℝ) →ₗ[ℝ] (n → ℝ))) (negOn_map e he h2V)
    rw [finrank_map_equiv] at this
    exact h1V.trans this
  · obtain ⟨V, h1V, h2V⟩ := exists_negOn_zero hB
    have := finrank_le_cntLt_zero hA (V.map (e.symm : (n → ℝ) →ₗ[ℝ] (n → ℝ)))
      (negOn_map e.symm he' h2V)
    rw [finrank_map_equiv] at this
    exact h1V.trans this

/-- A matrix with non-negative quadratic form has no negative eigenvalue. -/
theorem cntLt_zero_of_nonneg {T : Matrix n n ℝ} (hT : T.IsHermitian) (h : ∀ y, 0 ≤ qf T y) :
    cntLt hT 0 = 0 := by
  obtain ⟨V, h1, h2⟩ := exists_negOn_zero hT
  have hV : Module.finrank ℝ V = 0 := by
    rw [Submodule.finrank_eq_zero, Submodule.eq_bot_iff]
    intro x hx
    by_contra hx0
    exact absurd (h2 x hx hx0) (not_lt.mpr (h x))
  omega

end Counts

/-! ## Block diagonal matrices -/

section Blocks

variable {a b : Type*} [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b]

/-- The negative eigenvalues of a block diagonal matrix are those of its blocks. -/
theorem cntLt_fromBlocks {S : Matrix a a ℝ} {T : Matrix b b ℝ} (hS : S.IsHermitian)
    (hT : T.IsHermitian) (hST : (fromBlocks S 0 0 T).IsHermitian) :
    cntLt hST 0 = cntLt hS 0 + cntLt hT 0 := by
  set US := (hS.eigenvectorUnitary : Matrix a a ℝ) with hUS
  set UT := (hT.eigenvectorUnitary : Matrix b b ℝ) with hUT
  have eS : S = US * dg hS.eigenvalues * USᴴ := by
    have h := sub_smul_one_eq hS 0
    simpa using h
  have eT : T = UT * dg hT.eigenvalues * UTᴴ := by
    have h := sub_smul_one_eq hT 0
    simpa using h
  set W : Matrix (a ⊕ b) (a ⊕ b) ℝ := fromBlocks US 0 0 UT with hW
  have hS1 : USᴴ * US = 1 := conjTranspose_mul_unitary _
  have hS2 : US * USᴴ = 1 := unitary_mul_conjTranspose _
  have hT1 : UTᴴ * UT = 1 := conjTranspose_mul_unitary _
  have hT2 : UT * UTᴴ = 1 := unitary_mul_conjTranspose _
  have hW1 : Wᴴ * W = 1 := by
    rw [hW, fromBlocks_conjTranspose, fromBlocks_multiply]
    simp only [conjTranspose_zero, Matrix.mul_zero, Matrix.zero_mul, add_zero, zero_add, hS1, hT1,
      fromBlocks_one]
  have hW2 : W * Wᴴ = 1 := by
    rw [hW, fromBlocks_conjTranspose, fromBlocks_multiply]
    simp only [conjTranspose_zero, Matrix.mul_zero, Matrix.zero_mul, add_zero, zero_add, hS2, hT2,
      fromBlocks_one]
  have hdg : (dg (Sum.elim hS.eigenvalues hT.eigenvalues) : Matrix (a ⊕ b) (a ⊕ b) ℝ) =
      fromBlocks (dg hS.eigenvalues) 0 0 (dg hT.eigenvalues) := by
    simp only [dg]
    rw [fromBlocks_diagonal]
    congr 1
  have hfac : fromBlocks S 0 0 T - ((0 : ℝ) : ℝ) • (1 : Matrix (a ⊕ b) (a ⊕ b) ℝ) =
      W * dg (Sum.elim hS.eigenvalues hT.eigenvalues) * Wᴴ := by
    rw [hdg, hW, fromBlocks_conjTranspose, fromBlocks_multiply, fromBlocks_multiply]
    simp only [zero_smul, sub_zero, Matrix.mul_zero, Matrix.zero_mul, add_zero, zero_add,
      conjTranspose_zero]
    rw [← eS, ← eT]
  have h := (cntLt_eq_of_factor hST 0 W W hW1 hW2 _ hfac).1
  rw [h]
  unfold cntLt
  rw [Finset.card_filter, Finset.card_filter, Finset.card_filter, Fintype.sum_sum_type]
  rfl

end Blocks

/-! ## The bordered matrix -/

section Bordered

variable {a b : Type*} [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b]

/-- The bordered matrix `[[X, -q C], [-q Cᴴ, q G]]`; `eta.py` factors its negative
`[[-p G_A, q C], [q Cᵀ, -q G_B]]` with `X = p G_A`. -/
def bordered (X : Matrix a a ℝ) (C : Matrix a b ℝ) (G : Matrix b b ℝ) (q : ℝ) :
    Matrix (a ⊕ b) (a ⊕ b) ℝ :=
  fromBlocks X (-(q • C)) (-(q • Cᴴ)) (q • G)

omit [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b] in
theorem isHermitian_smul_real {n : Type*} {G : Matrix n n ℝ} (hG : G.IsHermitian) (q : ℝ) :
    (q • G).IsHermitian := by
  show (q • G)ᴴ = q • G
  rw [conjTranspose_smul, hG.eq, star_trivial]

omit [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b] in
theorem bordered_isHermitian {X : Matrix a a ℝ} (C : Matrix a b ℝ) {G : Matrix b b ℝ} (q : ℝ)
    (hX : X.IsHermitian) (hG : G.IsHermitian) : (bordered X C G q).IsHermitian := by
  unfold bordered
  refine IsHermitian.fromBlocks hX ?_ (isHermitian_smul_real hG q)
  rw [conjTranspose_neg, conjTranspose_smul, star_trivial]

omit [Fintype a] [DecidableEq a] in
theorem schur_isHermitian {X : Matrix a a ℝ} (C : Matrix a b ℝ) {G : Matrix b b ℝ} (q : ℝ)
    (hX : X.IsHermitian) (hG : G.IsHermitian) : (X - q • (C * G⁻¹ * Cᴴ)).IsHermitian :=
  hX.sub (isHermitian_smul_real (isHermitian_mul_mul_conjTranspose C hG.inv) q)

/-- The Schur-complement congruence of the bordered matrix. -/
theorem bordered_eq (X : Matrix a a ℝ) (C : Matrix a b ℝ) (G : Matrix b b ℝ) (q : ℝ)
    (hG : G.IsHermitian) (hGu : IsUnit G.det) :
    bordered X C G q =
      fromBlocks 1 (-(C * G⁻¹)) 0 1 * fromBlocks (X - q • (C * G⁻¹ * Cᴴ)) 0 0 (q • G) *
        (fromBlocks 1 (-(C * G⁻¹)) 0 1)ᴴ := by
  have hGi : (G⁻¹)ᴴ = G⁻¹ := hG.inv.eq
  rw [fromBlocks_conjTranspose, fromBlocks_multiply, fromBlocks_multiply]
  simp only [bordered, conjTranspose_one, conjTranspose_zero, conjTranspose_neg, conjTranspose_mul,
    hGi, Matrix.one_mul, Matrix.mul_one, Matrix.zero_mul, Matrix.mul_zero, add_zero, zero_add,
    Matrix.neg_mul, Matrix.mul_neg, Matrix.smul_mul, Matrix.mul_smul, Matrix.mul_assoc,
    Matrix.mul_nonsing_inv_cancel_left _ _ hGu, nonsing_inv_mul _ hGu]
  simp only [smul_neg, neg_neg, sub_add_cancel]

/-- **Haynsworth inertia additivity** for the bordered matrix: if `G > 0`, then
`[[X, -q C], [-q Cᴴ, q G]]` has as many negative eigenvalues as `X - q C G⁻¹ Cᴴ`. -/
theorem bordered_inertia {X : Matrix a a ℝ} {C : Matrix a b ℝ} {G : Matrix b b ℝ} {q : ℝ}
    (hG : G.IsHermitian) (hGu : IsUnit G.det) (hGpsd : ∀ y, 0 ≤ qf G y)
    (hq : 0 < q) (hN : (bordered X C G q).IsHermitian)
    (hS : (X - q • (C * G⁻¹ * Cᴴ)).IsHermitian) : cntLt hN 0 = cntLt hS 0 := by
  have hqG := isHermitian_smul_real hG q
  have hD : (fromBlocks (X - q • (C * G⁻¹ * Cᴴ)) 0 0 (q • G)).IsHermitian :=
    IsHermitian.fromBlocks hS (by simp) hqG
  set K : Matrix (a ⊕ b) (a ⊕ b) ℝ := fromBlocks 1 (-(C * G⁻¹)) 0 1 with hK
  set J : Matrix (a ⊕ b) (a ⊕ b) ℝ := fromBlocks 1 0 (G⁻¹ * Cᴴ) 1 with hJ
  have hGi : (G⁻¹)ᴴ = G⁻¹ := hG.inv.eq
  have h1 : Kᴴ * J = 1 := by
    rw [hK, hJ, fromBlocks_conjTranspose, fromBlocks_multiply]
    simp only [conjTranspose_one, conjTranspose_zero, conjTranspose_neg, conjTranspose_mul, hGi,
      Matrix.one_mul, Matrix.mul_one, Matrix.zero_mul, Matrix.mul_zero, add_zero, zero_add,
      neg_add_cancel, fromBlocks_one]
  have h2 : J * Kᴴ = 1 := by
    rw [hK, hJ, fromBlocks_conjTranspose, fromBlocks_multiply]
    simp only [conjTranspose_one, conjTranspose_zero, conjTranspose_neg, conjTranspose_mul, hGi,
      Matrix.one_mul, Matrix.mul_one, Matrix.zero_mul, Matrix.mul_zero, add_zero, zero_add,
      add_neg_cancel, fromBlocks_one]
  rw [cntLt_congr hN hD K J h1 h2 (bordered_eq X C G q hG hGu), cntLt_fromBlocks hS hqG hD,
    cntLt_zero_of_nonneg hqG (fun y => by rw [qf_smul_real]; exact mul_nonneg hq.le (hGpsd y)),
    add_zero]

/-- **The bordered-pencil lemma.** For `G_B > 0` and `p / q` with `q > 0`, the bordered matrix
`[[p G_A, -q C], [-q Cᴴ, q G_B]]` has as many negative eigenvalues as `(p/q) G_A - C G_B⁻¹ Cᴴ`,
that is, as there are eigenvalues of the pencil `(C G_B⁻¹ Cᴴ, G_A)` above `p / q`. -/
theorem bordered_pencil {GA : Matrix a a ℝ} {C : Matrix a b ℝ} {GB : Matrix b b ℝ} {p q : ℝ}
    (hGA : GA.IsHermitian) (hGB : GB.IsHermitian) (hGBu : IsUnit GB.det)
    (hGBpsd : ∀ y, 0 ≤ qf GB y) (hq : 0 < q) (hN : (bordered (p • GA) C GB q).IsHermitian)
    (hP : ((p / q) • GA - C * GB⁻¹ * Cᴴ).IsHermitian) : cntLt hN 0 = cntLt hP 0 := by
  have hX := isHermitian_smul_real hGA p
  have hS := schur_isHermitian C q hX hGB
  rw [bordered_inertia hGB hGBu hGBpsd hq hN hS]
  have key : ∀ x, qf (p • GA - q • (C * GB⁻¹ * Cᴴ)) x = q * qf ((p / q) • GA - C * GB⁻¹ * Cᴴ) x := by
    intro x
    rw [qf_sub, qf_sub, qf_smul_real, qf_smul_real, qf_smul_real]
    field_simp
  apply le_antisymm
  · exact cntLt_le_of_qf_le hS hP (1 / q) (by positivity) fun x => by
      rw [key x]; field_simp; exact le_refl _
  · exact cntLt_le_of_qf_le hP hS q hq fun x => by rw [key x]

end Bordered

/-! ## Loewner order of inverses -/

section Inverse

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- For a symmetric positive semi-definite invertible `Y`: `2 zᵀ y - zᵀ Y z ≤ yᵀ Y⁻¹ y`. -/
theorem var_le {Y : Matrix n n ℝ} (hY : Y.IsHermitian) (hYu : IsUnit Y.det)
    (hYpsd : ∀ z, 0 ≤ qf Y z) (y z : n → ℝ) : 2 * (z ⬝ᵥ y) - qf Y z ≤ qf Y⁻¹ y := by
  set u := Y⁻¹ *ᵥ y with hu_def
  have hu : Y *ᵥ u = y := by rw [hu_def, mulVec_mulVec, mul_nonsing_inv Y hYu, one_mulVec]
  have h0 := hYpsd (z - u)
  rw [qf_real] at h0
  have e1 : (z - u) ⬝ᵥ (Y *ᵥ (z - u)) = z ⬝ᵥ (Y *ᵥ z) - 2 * (z ⬝ᵥ y) + u ⬝ᵥ y := by
    simp only [mulVec_sub, sub_dotProduct, dotProduct_sub, hu]
    rw [dotProduct_mulVec_symm hY u z, hu]
    ring
  have e2 : qf Y⁻¹ y = u ⬝ᵥ y := by rw [qf_real, dotProduct_comm]
  rw [e2, qf_real]
  linarith

/-- `G⁻¹ ≤ (1 + β/μ) (G + β I)⁻¹` in the quadratic form, for `G ≥ μ > 0` and `β ≥ 0`; this is the
inequality `(G_B + β)⁻¹ ≥ G_B⁻¹ / (1 + β/μ)` of `eta.py`. -/
theorem qf_inv_shift {G : Matrix n n ℝ} (hG : G.IsHermitian) {μ β : ℝ} (hμ : 0 < μ) (hβ : 0 ≤ β)
    (hGμ : ∀ y, μ * nsq y ≤ qf G y) (y : n → ℝ) :
    qf G⁻¹ y ≤ (1 + β / μ) * qf (G + β • (1 : Matrix n n ℝ))⁻¹ y := by
  have hGu := isUnit_det_of_qf_ge hμ hGμ
  set G' := G + β • (1 : Matrix n n ℝ) with hG'
  have hqfG' : ∀ z, qf G' z = qf G z + β * nsq z := fun z => by
    rw [hG', qf_add, qf_smul_one_real]
  have hG'μ : ∀ z, μ * nsq z ≤ qf G' z := fun z => by
    rw [hqfG']
    have := hGμ z
    have := nsq_nonneg z
    nlinarith
  have hG'h : G'.IsHermitian := hG.add (isHermitian_smul_real (isHermitian_one) β)
  have hG'u := isUnit_det_of_qf_ge hμ hG'μ
  have hG'psd : ∀ z, 0 ≤ qf G' z := fun z => by
    have := hG'μ z
    have := nsq_nonneg z
    nlinarith
  set k := 1 + β / μ with hk
  have hk1 : 1 ≤ k := by
    rw [hk]
    have : 0 ≤ β / μ := div_nonneg hβ hμ.le
    linarith
  have hk0 : 0 < k := by linarith
  set u := G⁻¹ *ᵥ y with hu_def
  have hu : G *ᵥ u = y := by rw [hu_def, mulVec_mulVec, mul_nonsing_inv G hGu, one_mulVec]
  have ha : qf G u = u ⬝ᵥ y := by rw [qf_real, hu]
  have ha' : qf G⁻¹ y = u ⬝ᵥ y := by rw [qf_real, dotProduct_comm]
  have hv := var_le hG'h hG'u hG'psd y (k⁻¹ • u)
  rw [qf_smul_vec, hqfG', smul_dotProduct, smul_eq_mul] at hv
  -- β ‖u‖² ≤ (β/μ) uᵀ G u
  have hb : β * nsq u ≤ β / μ * qf G u := by
    have h1 := hGμ u
    have h2 : β * nsq u = β / μ * (μ * nsq u) := by field_simp
    rw [h2]
    exact mul_le_mul_of_nonneg_left h1 (div_nonneg hβ hμ.le)
  rw [ha'] at ⊢
  rw [ha] at hb hv
  set A := u ⬝ᵥ y
  set Q := qf G'⁻¹ y
  have hbk : β * nsq u ≤ (k - 1) * A := by rw [hk]; simpa [add_sub_cancel_left] using hb
  -- hv : 2 * (k⁻¹ * A) - (k⁻¹) ^ 2 * (A + β * nsq u) ≤ Q
  have h3 : 2 * (k⁻¹ * A) - (k⁻¹) ^ 2 * (A + (k - 1) * A) ≤ Q := by
    have : (k⁻¹) ^ 2 * (A + β * nsq u) ≤ (k⁻¹) ^ 2 * (A + (k - 1) * A) :=
      mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg _)
    linarith
  have h4 : 2 * (k⁻¹ * A) - (k⁻¹) ^ 2 * (A + (k - 1) * A) = k⁻¹ * A := by
    field_simp
    ring
  rw [h4] at h3
  have h5 : A = k * (k⁻¹ * A) := by field_simp
  rw [h5]
  exact mul_le_mul_of_nonneg_left h3 hk0.le

end Inverse

/-! ## The certified count of `eta.py` -/

section Eta

variable {a b : Type*} [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b]

theorem qf_bordered_shift (X : Matrix a a ℝ) (C : Matrix a b ℝ) (G : Matrix b b ℝ) (q β : ℝ)
    (x : a ⊕ b → ℝ) :
    qf (bordered (X + (q * β) • (1 : Matrix a a ℝ)) C (G + β • (1 : Matrix b b ℝ)) q) x =
      qf (bordered X C G q) x + q * β * nsq x := by
  have hM : bordered (X + (q * β) • (1 : Matrix a a ℝ)) C (G + β • (1 : Matrix b b ℝ)) q =
      bordered X C G q + (q * β) • (1 : Matrix (a ⊕ b) (a ⊕ b) ℝ) := by
    rw [← fromBlocks_one, fromBlocks_smul, bordered, bordered, fromBlocks_add]
    congr 1 <;> simp [smul_add, smul_smul]
  rw [hM, qf_add, qf_smul_one_real]

/-- **The count behind every E10 eta bound.** Let `G_A ≥ ν`, `G_B ≥ μ` (`μ, ν > 0`), `t = p / q` with
`q > 0`, `β ≥ 0`, and let `N'` be symmetric with `|xᵀ (N' - N) x| ≤ q β ‖x‖²` for the bordered
matrix `N = [[p G_A, -q C], [-q Cᴴ, q G_B]]`. Then, with `t' = (1 + β/μ)(t + β/ν)`, the number of
eigenvalues of the pencil `(C G_B⁻¹ Cᴴ, G_A)` above `t'` (the negative eigenvalues of
`t' G_A - C G_B⁻¹ Cᴴ`) is at most the number of negative eigenvalues of `N'`. -/
theorem eta_count {GA : Matrix a a ℝ} {GB : Matrix b b ℝ} {C : Matrix a b ℝ}
    (hGA : GA.IsHermitian) (hGB : GB.IsHermitian) {μ ν β p q : ℝ}
    (hμ : 0 < μ) (hν : 0 < ν) (hβ : 0 ≤ β) (hq : 0 < q)
    (hGBμ : ∀ y, μ * nsq y ≤ qf GB y) (hGAν : ∀ x, ν * nsq x ≤ qf GA x)
    {N' : Matrix (a ⊕ b) (a ⊕ b) ℝ} (hN' : N'.IsHermitian)
    (hres : ∀ x, |qf (N' - bordered (p • GA) C GB q) x| ≤ q * β * nsq x)
    (hP : (((1 + β / μ) * (p / q + β / ν)) • GA - C * GB⁻¹ * Cᴴ).IsHermitian) :
    cntLt hP 0 ≤ cntLt hN' 0 := by
  set GB' := GB + β • (1 : Matrix b b ℝ) with hGB'
  set X := p • GA + (q * β) • (1 : Matrix a a ℝ) with hX
  have hXh : X.IsHermitian := (isHermitian_smul_real hGA p).add (isHermitian_smul_real isHermitian_one _)
  have hGB'h : GB'.IsHermitian := hGB.add (isHermitian_smul_real isHermitian_one β)
  have hqfGB' : ∀ z, qf GB' z = qf GB z + β * nsq z := fun z => by rw [hGB', qf_add, qf_smul_one_real]
  have hGB'μ : ∀ z, μ * nsq z ≤ qf GB' z := fun z => by
    rw [hqfGB']
    have := nsq_nonneg z
    nlinarith [hGBμ z]
  have hGB'u := isUnit_det_of_qf_ge hμ hGB'μ
  have hGB'psd : ∀ z, 0 ≤ qf GB' z := fun z => by nlinarith [hGB'μ z, nsq_nonneg z]
  have hN2 := bordered_isHermitian C q hXh hGB'h
  have hS := schur_isHermitian C q hXh hGB'h
  -- the residual moves the bordered matrix by at most q β
  have hstep1 : cntLt hN2 0 ≤ cntLt hN' 0 := by
    apply cntLt_le_of_qf_le hN2 hN' 1 one_pos
    intro x
    have h := (abs_le.mp (hres x)).2
    rw [qf_sub] at h
    rw [hX, hGB', qf_bordered_shift]
    linarith
  -- the count of the bordered matrix is the count of its Schur complement
  have hstep2 : cntLt hN2 0 = cntLt hS 0 := bordered_inertia hGB'h hGB'u hGB'psd hq hN2 hS
  -- the Schur complement is below a positive multiple of t' G_A - C G_B⁻¹ Cᴴ
  set k := 1 + β / μ with hk
  have hk0 : 0 < k := by rw [hk]; have := div_nonneg hβ hμ.le; linarith
  have hstep3 : cntLt hP 0 ≤ cntLt hS 0 := by
    apply cntLt_le_of_qf_le hP hS (q / k) (div_pos hq hk0)
    intro x
    set y := Cᴴ *ᵥ x
    have hinv := qf_inv_shift hGB hμ hβ hGBμ y
    have hA := hGAν x
    rw [qf_sub, qf_sub, hX, qf_add, qf_smul_real, qf_smul_one_real, qf_smul_real, qf_smul_real,
      qf_congr, qf_congr]
    have hA' : q * β * nsq x ≤ q * β / ν * qf GA x := by
      have h2 : q * β * nsq x = q * β / ν * (ν * nsq x) := by field_simp
      rw [h2]
      exact mul_le_mul_of_nonneg_left hA (by positivity)
    have hB' : q / k * qf GB⁻¹ y ≤ q * qf GB'⁻¹ y := by
      have h2 : q * qf GB'⁻¹ y = q / k * (k * qf GB'⁻¹ y) := by field_simp
      rw [h2]
      exact mul_le_mul_of_nonneg_left hinv (div_pos hq hk0).le
    have h3 : q / k * (k * (p / q + β / ν) * qf GA x) = p * qf GA x + q * β / ν * qf GA x := by
      field_simp
    nlinarith [h3, hA', hB']
  calc cntLt hP 0 ≤ cntLt hS 0 := hstep3
    _ = cntLt hN2 0 := hstep2.symm
    _ ≤ cntLt hN' 0 := hstep1

end Eta

end CertifyMathlib.Apps
