import CertifyMathlib.Counting.QuadForm

/-!
# Eigenvalue counts of Hermitian matrices, Sylvester's law and Weyl's inequality

For a Hermitian matrix `A` with Mathlib's eigenvalues `hA.eigenvalues : n → ℝ` this file defines
the two counting functions used in every counting statement of CERTIFY-ED,

* `cntLt hA t = #{i | λ i < t}` and
* `cntLe hA t = #{i | λ i ≤ t}`,

and proves:

* `finrank_le_cntLt`, `exists_negOn_cntLt` (and the `cntLe` versions): `cntLt hA t` is the largest
  dimension of a subspace on which `A - t I` is negative. This is the min-max characterisation in
  the form needed here;
* `sylvester_inertia`, `cntLt_eq_of_factor`: Sylvester's law of inertia. If `A - t I = L D Lᴴ`
  with `D` real diagonal and `L` invertible, the number of eigenvalues below `t` is the number of
  negative entries of `D`, and the number at or below `t` the number of non-positive entries;
* `cnt_conj`: counts are unchanged by a unitary change of basis, also between two index types;
* `weyl_cntLt`, `weyl_cntLe`: Weyl's inequality in counting form. If `|xᴴ (A - B) x| ≤ ρ ‖x‖²`
  for all `x` (which holds when `‖A - B‖₂ ≤ ρ`), then `cntLt hA t ≤ cntLt hB (t + ρ)` and
  `cntLe hA t ≤ cntLe hB (t + ρ)`. For the sorted eigenvalues this is `|λ_j(A) - λ_j(B)| ≤ ρ`.

Statements about the `j`-th smallest eigenvalue are written through the counts throughout:
`λ_j ≤ x` is `j ≤ cntLe hA x`, and `λ_j ≥ y` is `cntLt hA y ≤ j - 1`. This keeps every statement
independent of the order in which Mathlib lists the eigenvalues.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

/-- The number of eigenvalues of `A` strictly below `t`, with multiplicity. -/
noncomputable def cntLt {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) : ℕ :=
  (Finset.univ.filter fun i => hA.eigenvalues i < t).card

/-- The number of eigenvalues of `A` at or below `t`, with multiplicity. -/
noncomputable def cntLe {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) : ℕ :=
  (Finset.univ.filter fun i => hA.eigenvalues i ≤ t).card

theorem unitary_mul_conjTranspose (U : unitary (Matrix n n 𝕜)) :
    (U : Matrix n n 𝕜) * (U : Matrix n n 𝕜)ᴴ = 1 := by
  have h := Unitary.coe_mul_star_self U
  rw [Unitary.coe_star, star_eq_conjTranspose] at h
  exact h

theorem conjTranspose_mul_unitary (U : unitary (Matrix n n 𝕜)) :
    (U : Matrix n n 𝕜)ᴴ * (U : Matrix n n 𝕜) = 1 := by
  have h := Unitary.coe_star_mul_self U
  simpa [star_eq_conjTranspose] using h

/-- The spectral decomposition of `A - t I`. -/
theorem sub_smul_one_eq {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) :
    A - (t : 𝕜) • (1 : Matrix n n 𝕜) =
      (hA.eigenvectorUnitary : Matrix n n 𝕜) * dg (fun i => hA.eigenvalues i - t) *
        (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ := by
  have hs : A = (hA.eigenvectorUnitary : Matrix n n 𝕜) *
      diagonal (RCLike.ofReal ∘ hA.eigenvalues) *
      star (hA.eigenvectorUnitary : Matrix n n 𝕜) := hA.spectral_theorem
  rw [star_eq_conjTranspose] at hs
  have hU := unitary_mul_conjTranspose hA.eigenvectorUnitary
  have hdg : dg (𝕜 := 𝕜) (fun i => hA.eigenvalues i - t) =
      diagonal (RCLike.ofReal ∘ hA.eigenvalues) - (t : 𝕜) • (1 : Matrix n n 𝕜) := by
    ext i j
    by_cases h : i = j
    · subst h
      simp [dg]
      first
      | done
      | rw [Algebra.smul_def, mul_one]
    · simp [dg, h]
  rw [hdg, mul_sub, sub_mul, mul_smul_comm, smul_mul_assoc, mul_one, hU, ← hs]

/-- A subspace on which `A - t I` is negative has dimension at most `cntLt hA t`. -/
theorem finrank_le_cntLt {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ)
    (V : Submodule 𝕜 (n → 𝕜)) (hV : NegOn (A - (t : 𝕜) • (1 : Matrix n n 𝕜)) V) :
    Module.finrank 𝕜 V ≤ cntLt hA t := by
  have h := finrank_le_card_neg _ _ (conjTranspose_mul_unitary hA.eigenvectorUnitary)
    (unitary_mul_conjTranspose hA.eigenvectorUnitary) _ (sub_smul_one_eq hA t) V hV
  simpa [cntLt, sub_neg] using h

/-- A subspace on which `A - t I` is non-positive has dimension at most `cntLe hA t`. -/
theorem finrank_le_cntLe {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ)
    (V : Submodule 𝕜 (n → 𝕜)) (hV : NonposOn (A - (t : 𝕜) • (1 : Matrix n n 𝕜)) V) :
    Module.finrank 𝕜 V ≤ cntLe hA t := by
  have h := finrank_le_card_nonpos _ _ (conjTranspose_mul_unitary hA.eigenvectorUnitary)
    (unitary_mul_conjTranspose hA.eigenvectorUnitary) _ (sub_smul_one_eq hA t) V hV
  simpa [cntLe, sub_nonpos] using h

/-- `A - t I` is negative on a subspace of dimension `cntLt hA t`. -/
theorem exists_negOn_cntLt {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) :
    ∃ V : Submodule 𝕜 (n → 𝕜), cntLt hA t ≤ Module.finrank 𝕜 V ∧
      NegOn (A - (t : 𝕜) • (1 : Matrix n n 𝕜)) V := by
  obtain ⟨V, h1, h2⟩ := exists_negOn _ _ (conjTranspose_mul_unitary hA.eigenvectorUnitary)
    (unitary_mul_conjTranspose hA.eigenvectorUnitary) _ (sub_smul_one_eq hA t)
  exact ⟨V, by simpa [cntLt, sub_neg] using h1, h2⟩

/-- `A - t I` is non-positive on a subspace of dimension `cntLe hA t`. -/
theorem exists_nonposOn_cntLe {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) :
    ∃ V : Submodule 𝕜 (n → 𝕜), cntLe hA t ≤ Module.finrank 𝕜 V ∧
      NonposOn (A - (t : 𝕜) • (1 : Matrix n n 𝕜)) V := by
  obtain ⟨V, h1, h2⟩ := exists_nonposOn _ _ (conjTranspose_mul_unitary hA.eigenvectorUnitary)
    (unitary_mul_conjTranspose hA.eigenvectorUnitary) _ (sub_smul_one_eq hA t)
  exact ⟨V, by simpa [cntLe, sub_nonpos] using h1, h2⟩

/-- **Sylvester's law of inertia at a shift.** If `A - t I = L diag(d) Lᴴ` and `Lᴴ` has the
two-sided inverse `K`, then `#{λ < t} = #{i | d i < 0}` and `#{λ ≤ t} = #{i | d i ≤ 0}`. -/
theorem cntLt_eq_of_factor {m : Type*} [Fintype m] [DecidableEq m] {A : Matrix n n 𝕜}
    (hA : A.IsHermitian) (t : ℝ) (L K : Matrix n m 𝕜) (hK1 : Lᴴ * K = 1) (hK2 : K * Lᴴ = 1)
    (d : m → ℝ) (h : A - (t : 𝕜) • (1 : Matrix n n 𝕜) = L * dg d * Lᴴ) :
    cntLt hA t = (Finset.univ.filter fun i => d i < 0).card ∧
    cntLe hA t = (Finset.univ.filter fun i => d i ≤ 0).card := by
  have := sylvester_card _ _ (conjTranspose_mul_unitary hA.eigenvectorUnitary)
    (unitary_mul_conjTranspose hA.eigenvectorUnitary) _ (sub_smul_one_eq hA t)
    L K hK1 hK2 d h
  simp only [sub_neg, sub_nonpos] at this
  exact this

theorem isUnit_det_conjTranspose {L : Matrix n n 𝕜} (hL : IsUnit L) : IsUnit Lᴴ.det := by
  rw [det_conjTranspose]
  exact ((Matrix.isUnit_iff_isUnit_det L).mp hL).star

/-- **Sylvester's law of inertia** for `A = L D Lᴴ` with `L` invertible and `D` real diagonal:
the number of negative eigenvalues of `A` is the number of negative entries of `D`, and the
number of non-positive eigenvalues the number of non-positive entries. -/
theorem sylvester_inertia {A : Matrix n n 𝕜} (hA : A.IsHermitian) (L : Matrix n n 𝕜)
    (hL : IsUnit L) (d : n → ℝ) (h : A = L * dg d * Lᴴ) :
    cntLt hA 0 = (Finset.univ.filter fun i => d i < 0).card ∧
    cntLe hA 0 = (Finset.univ.filter fun i => d i ≤ 0).card := by
  have hdet := isUnit_det_conjTranspose hL
  refine cntLt_eq_of_factor hA 0 L (Lᴴ)⁻¹ (mul_nonsing_inv _ hdet) (nonsing_inv_mul _ hdet) d ?_
  simpa using h

/-- Counts are invariant under a unitary change of basis `K = Qᴴ H Q`, where `Q` may go between
two index types of the same size (`Qᴴ Q = 1` and `Q Qᴴ = 1`). -/
theorem cnt_conj {p : Type*} [Fintype p] [DecidableEq p] {H : Matrix n n 𝕜}
    (hH : H.IsHermitian) (Q : Matrix n p 𝕜) (hQ1 : Qᴴ * Q = 1) (hQ2 : Q * Qᴴ = 1)
    {K : Matrix p p 𝕜} (hK : K.IsHermitian) (hKeq : K = Qᴴ * H * Q) (t : ℝ) :
    cntLt hK t = cntLt hH t ∧ cntLe hK t = cntLe hH t := by
  subst hKeq
  set U := (hH.eigenvectorUnitary : Matrix n n 𝕜) with hUdef
  have hUU : Uᴴ * U = 1 := conjTranspose_mul_unitary _
  have hUU' : U * Uᴴ = 1 := unitary_mul_conjTranspose _
  have hfac : Qᴴ * H * Q - (t : 𝕜) • (1 : Matrix p p 𝕜) =
      (Qᴴ * U) * dg (fun i => hH.eigenvalues i - t) * (Qᴴ * U)ᴴ := by
    have h1 : Qᴴ * H * Q - (t : 𝕜) • (1 : Matrix p p 𝕜) =
        Qᴴ * (H - (t : 𝕜) • (1 : Matrix n n 𝕜)) * Q := by
      rw [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one, hQ1]
    rw [h1, sub_smul_one_eq hH t, ← hUdef, conjTranspose_mul, conjTranspose_conjTranspose]
    simp only [Matrix.mul_assoc]
  have hK1 : (Qᴴ * U)ᴴ * (Qᴴ * U) = 1 := by
    rw [conjTranspose_mul, conjTranspose_conjTranspose]
    calc Uᴴ * Q * (Qᴴ * U) = Uᴴ * (Q * Qᴴ) * U := by simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hQ2, Matrix.mul_one, hUU]
  have hK2 : (Qᴴ * U) * (Qᴴ * U)ᴴ = 1 := by
    rw [conjTranspose_mul, conjTranspose_conjTranspose]
    calc Qᴴ * U * (Uᴴ * Q) = Qᴴ * (U * Uᴴ) * Q := by simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hUU', Matrix.mul_one, hQ1]
  have := cntLt_eq_of_factor hK t (Qᴴ * U) (Qᴴ * U) hK1 hK2 _ hfac
  simp only [sub_neg, sub_nonpos] at this
  exact this

omit [DecidableEq n] in
theorem isHermitian_conj {p : Type*} [Fintype p] {H : Matrix n n 𝕜} (hH : H.IsHermitian)
    (Q : Matrix n p 𝕜) : (Qᴴ * H * Q).IsHermitian := by
  show (Qᴴ * H * Q)ᴴ = Qᴴ * H * Q
  rw [conjTranspose_mul, conjTranspose_mul, conjTranspose_conjTranspose, hH.eq,
    Matrix.mul_assoc]

theorem cntLt_mono {A : Matrix n n 𝕜} (hA : A.IsHermitian) {s t : ℝ} (hst : s ≤ t) :
    cntLt hA s ≤ cntLt hA t := by
  apply Finset.card_le_card
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
  linarith

theorem cntLe_mono {A : Matrix n n 𝕜} (hA : A.IsHermitian) {s t : ℝ} (hst : s ≤ t) :
    cntLe hA s ≤ cntLe hA t := by
  apply Finset.card_le_card
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
  linarith

theorem cntLt_le_cntLe {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) :
    cntLt hA t ≤ cntLe hA t := by
  apply Finset.card_le_card
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
  exact hi.le

theorem cntLe_le_cntLt {A : Matrix n n 𝕜} (hA : A.IsHermitian) {s t : ℝ} (hst : s < t) :
    cntLe hA s ≤ cntLt hA t := by
  apply Finset.card_le_card
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
  linarith

omit [DecidableEq n] in
theorem abs_qf_sub_comm (A B : Matrix n n 𝕜) (x : n → 𝕜) :
    |qf (A - B) x| = |qf (B - A) x| := by
  rw [qf_sub, qf_sub, abs_sub_comm]

/-- **Weyl's inequality, counting form (strict).** -/
theorem weyl_cntLt {A B : Matrix n n 𝕜} (hA : A.IsHermitian) (hB : B.IsHermitian) (ρ : ℝ)
    (h : ∀ x, |qf (A - B) x| ≤ ρ * nsq x) (t : ℝ) : cntLt hA t ≤ cntLt hB (t + ρ) := by
  obtain ⟨V, hV1, hV2⟩ := exists_negOn_cntLt hA t
  have hneg : NegOn (B - ((t + ρ : ℝ) : 𝕜) • (1 : Matrix n n 𝕜)) V := by
    intro x hx hx0
    have h1 := hV2 x hx hx0
    rw [qf_sub_smul_one] at h1 ⊢
    have h2 := h x
    rw [qf_sub] at h2
    have h3 := (abs_le.mp h2).1
    have h4 : (t + ρ) * nsq x = t * nsq x + ρ * nsq x := add_mul _ _ _
    linarith
  exact hV1.trans (finrank_le_cntLt hB _ V hneg)

/-- **Weyl's inequality, counting form (non-strict).** -/
theorem weyl_cntLe {A B : Matrix n n 𝕜} (hA : A.IsHermitian) (hB : B.IsHermitian) (ρ : ℝ)
    (h : ∀ x, |qf (A - B) x| ≤ ρ * nsq x) (t : ℝ) : cntLe hA t ≤ cntLe hB (t + ρ) := by
  obtain ⟨V, hV1, hV2⟩ := exists_nonposOn_cntLe hA t
  have hnp : NonposOn (B - ((t + ρ : ℝ) : 𝕜) • (1 : Matrix n n 𝕜)) V := by
    intro x hx
    have h1 := hV2 x hx
    rw [qf_sub_smul_one] at h1 ⊢
    have h2 := h x
    rw [qf_sub] at h2
    have h3 := (abs_le.mp h2).1
    have h4 : (t + ρ) * nsq x = t * nsq x + ρ * nsq x := add_mul _ _ _
    linarith
  exact hV1.trans (finrank_le_cntLe hB _ V hnp)

omit [DecidableEq n] in
theorem qf_smul (s : ℝ) (M : Matrix n n 𝕜) (x : n → 𝕜) : qf ((s : 𝕜) • M) x = s * qf M x := by
  simp only [qf, smul_mulVec, dotProduct_smul, smul_eq_mul, RCLike.re_ofReal_mul]

/-- **E-04, Weyl along an affine path.** If `F = H₀ + s H₁` and `|xᴴ H₁ x| ≤ B ‖x‖²`, the counts
of `F` and `H₀` differ by a shift of at most `B |s|`, in both directions. -/
theorem weyl_affine {H₀ H₁ F : Matrix n n 𝕜} (h₀ : H₀.IsHermitian) (hF : F.IsHermitian)
    (s B : ℝ) (hFeq : F = H₀ + (s : 𝕜) • H₁) (hB : ∀ x, |qf H₁ x| ≤ B * nsq x) (t : ℝ) :
    cntLt hF t ≤ cntLt h₀ (t + B * |s|) ∧ cntLt h₀ t ≤ cntLt hF (t + B * |s|) ∧
    cntLe hF t ≤ cntLe h₀ (t + B * |s|) ∧ cntLe h₀ t ≤ cntLe hF (t + B * |s|) := by
  have hd : ∀ x, |qf (F - H₀) x| ≤ B * |s| * nsq x := by
    intro x
    rw [hFeq, add_sub_cancel_left, qf_smul, abs_mul]
    have := hB x
    have h0 := abs_nonneg s
    calc |s| * |qf H₁ x| ≤ |s| * (B * nsq x) := mul_le_mul_of_nonneg_left this h0
      _ = B * |s| * nsq x := by ring
  have hd' : ∀ x, |qf (H₀ - F) x| ≤ B * |s| * nsq x := fun x => by
    rw [abs_qf_sub_comm]
    exact hd x
  exact ⟨weyl_cntLt hF h₀ _ hd t, weyl_cntLt h₀ hF _ hd' t, weyl_cntLe hF h₀ _ hd t,
    weyl_cntLe h₀ hF _ hd' t⟩

end CertifyMathlib.Counting
