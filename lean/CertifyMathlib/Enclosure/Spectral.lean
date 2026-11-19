import Mathlib.Analysis.Matrix.Spectrum

/-!
# Spectral framework for the enclosure theorems

A symmetric linear map `T` on a finite-dimensional inner product space `E` over `𝕜` (`ℝ` or `ℂ`)
together with an orthonormal basis of eigenvectors with real eigenvalues is recorded as an
`EigenData`. Every enclosure theorem of this library is proved for such data; the matrix
statements follow by taking `T = Matrix.toEuclideanLin A` and Mathlib's eigenvector basis of a
Hermitian matrix (`Matrix.IsHermitian.eigenvectorBasis`).

The file contains the coordinate formulas (norm, residual norm and Rayleigh quotient in the
eigenvector basis), the counting lemma behind every min-max argument
(`EigenData.finrank_le_card_of_support`), and its uses: a subspace on which `T - c` is small
forces eigenvalues near `c` (`EigenData.finrank_le_card_near`), and a subspace on which the
Rayleigh quotient is at most `t` forces eigenvalues at most `t` (`EigenData.finrank_le_card_le`).
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

/-- An orthonormal basis `basis` of eigenvectors of `T`, with real eigenvalues `eig`. -/
structure EigenData (T : E →ₗ[𝕜] E) (ι : Type*) [Fintype ι] where
  basis : OrthonormalBasis ι 𝕜 E
  eig : ι → ℝ
  apply_basis : ∀ i, T (basis i) = (eig i : 𝕜) • basis i

namespace EigenData

variable {T : E →ₗ[𝕜] E} {ι : Type*} [Fintype ι] (D : EigenData T ι)

/-- Reindexing an eigenbasis. -/
noncomputable def reindex {ι' : Type*} [Fintype ι'] (e : ι ≃ ι') : EigenData T ι' where
  basis := D.basis.reindex e
  eig := D.eig ∘ e.symm
  apply_basis i := by
    simp only [OrthonormalBasis.reindex_apply, Function.comp_apply]
    exact D.apply_basis _

/-- The coefficient of `T v` on `basis i` is `eig i` times the coefficient of `v`. -/
theorem repr_apply (hT : T.IsSymmetric) (v : E) (i : ι) :
    D.basis.repr (T v) i = (D.eig i : 𝕜) * D.basis.repr v i := by
  rw [D.basis.repr_apply_apply, D.basis.repr_apply_apply, ← hT, D.apply_basis, inner_smul_left,
    RCLike.conj_ofReal]

/-- Parseval: the squared norm is the sum of the squared coefficients. -/
theorem norm_sq_eq (v : E) : ‖v‖ ^ 2 = ∑ i, ‖D.basis.repr v i‖ ^ 2 := by
  rw [← D.basis.repr.norm_map v, EuclideanSpace.norm_sq_eq]

theorem repr_sub_smul (hT : T.IsSymmetric) (v : E) (c : ℝ) (i : ι) :
    D.basis.repr (T v - (c : 𝕜) • v) i = ((D.eig i - c : ℝ) : 𝕜) * D.basis.repr v i := by
  rw [map_sub, map_smul, PiLp.sub_apply, PiLp.smul_apply, D.repr_apply hT, smul_eq_mul]
  push_cast
  ring

/-- The residual norm in the eigenvector basis. -/
theorem norm_sub_smul_sq (hT : T.IsSymmetric) (v : E) (c : ℝ) :
    ‖T v - (c : 𝕜) • v‖ ^ 2 = ∑ i, (D.eig i - c) ^ 2 * ‖D.basis.repr v i‖ ^ 2 := by
  rw [D.norm_sq_eq]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [D.repr_sub_smul hT, norm_mul, RCLike.norm_ofReal, mul_pow, sq_abs]

/-- The Rayleigh quotient (numerator) in the eigenvector basis. -/
theorem re_inner_apply (hT : T.IsSymmetric) (v : E) :
    RCLike.re ⟪v, T v⟫_𝕜 = ∑ i, D.eig i * ‖D.basis.repr v i‖ ^ 2 := by
  rw [← D.basis.repr.inner_map_map v (T v), PiLp.inner_apply, map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [D.repr_apply hT, ← smul_eq_mul (a := (D.eig i : 𝕜)), inner_smul_right,
    RCLike.re_ofReal_mul, inner_self_eq_norm_sq]

/-- A nonzero vector has a nonzero coefficient. -/
theorem exists_repr_ne_zero {v : E} (hv : v ≠ 0) : ∃ i, D.basis.repr v i ≠ 0 := by
  by_contra h
  push Not at h
  have h2 : ‖v‖ ^ 2 = 0 := by rw [D.norm_sq_eq]; simp [h]
  exact hv (by simpa using h2)

/-! ### Bounds from the coordinate formulas -/

/-- If every eigenvalue met by `v` is within `h` of `c`, then `‖T v - c v‖ ≤ h ‖v‖`. -/
theorem norm_sub_smul_le (hT : T.IsSymmetric) {v : E} {c h : ℝ} (hh : 0 ≤ h)
    (hv : ∀ i, D.basis.repr v i ≠ 0 → |D.eig i - c| ≤ h) :
    ‖T v - (c : 𝕜) • v‖ ≤ h * ‖v‖ := by
  have hsq : ‖T v - (c : 𝕜) • v‖ ^ 2 ≤ (h * ‖v‖) ^ 2 := by
    rw [D.norm_sub_smul_sq hT, mul_pow, D.norm_sq_eq, Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    by_cases hi : D.basis.repr v i = 0
    · simp [hi]
    · have h1 : (D.eig i - c) ^ 2 ≤ h ^ 2 := by
        rw [← sq_abs (D.eig i - c)]
        exact pow_le_pow_left₀ (abs_nonneg _) (hv i hi) 2
      exact mul_le_mul_of_nonneg_right h1 (sq_nonneg _)
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (mul_nonneg hh (norm_nonneg _)) two_ne_zero).mp hsq

/-- `‖T v‖ ≤ M ‖v‖` when every eigenvalue has modulus at most `M`, `M ≥ 0`. -/
theorem norm_apply_le (hT : T.IsSymmetric) {M : ℝ} (hM0 : 0 ≤ M) (hM : ∀ i, |D.eig i| ≤ M)
    (v : E) : ‖T v‖ ≤ M * ‖v‖ := by
  have := D.norm_sub_smul_le hT (c := 0) hM0 (v := v) (fun i _ => by simpa using hM i)
  simpa using this

/-- Strict form: if every eigenvalue met by `v ≠ 0` is farther than `r ≥ 0` from `c`, then
`r² ‖v‖² < ‖T v - c v‖²`. -/
theorem sq_mul_norm_sq_lt (hT : T.IsSymmetric) {v : E} (hv : v ≠ 0) {c r : ℝ} (hr : 0 ≤ r)
    (h : ∀ i, D.basis.repr v i ≠ 0 → r < |D.eig i - c|) :
    r ^ 2 * ‖v‖ ^ 2 < ‖T v - (c : 𝕜) • v‖ ^ 2 := by
  rw [D.norm_sub_smul_sq hT, D.norm_sq_eq, Finset.mul_sum]
  obtain ⟨i, hi⟩ := D.exists_repr_ne_zero hv
  apply Finset.sum_lt_sum
  · intro j _
    by_cases hj : D.basis.repr v j = 0
    · simp [hj]
    · have h1 : r ^ 2 ≤ (D.eig j - c) ^ 2 := by
        rw [← sq_abs (D.eig j - c)]
        exact pow_le_pow_left₀ hr (h j hj).le 2
      exact mul_le_mul_of_nonneg_right h1 (sq_nonneg _)
  · refine ⟨i, Finset.mem_univ _, ?_⟩
    have h1 : r ^ 2 < (D.eig i - c) ^ 2 := by
      rw [← sq_abs (D.eig i - c)]
      exact pow_lt_pow_left₀ (h i hi) hr two_ne_zero
    exact mul_lt_mul_of_pos_right h1 (pow_pos (norm_pos_iff.mpr hi) 2)

theorem re_inner_le (hT : T.IsSymmetric) {v : E} {t : ℝ}
    (h : ∀ i, D.basis.repr v i ≠ 0 → D.eig i ≤ t) :
    RCLike.re ⟪v, T v⟫_𝕜 ≤ t * ‖v‖ ^ 2 := by
  rw [D.re_inner_apply hT, D.norm_sq_eq, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  by_cases hi : D.basis.repr v i = 0
  · simp [hi]
  · exact mul_le_mul_of_nonneg_right (h i hi) (sq_nonneg _)

theorem le_re_inner (hT : T.IsSymmetric) {v : E} {t : ℝ}
    (h : ∀ i, D.basis.repr v i ≠ 0 → t ≤ D.eig i) :
    t * ‖v‖ ^ 2 ≤ RCLike.re ⟪v, T v⟫_𝕜 := by
  rw [D.re_inner_apply hT, D.norm_sq_eq, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  by_cases hi : D.basis.repr v i = 0
  · simp [hi]
  · exact mul_le_mul_of_nonneg_right (h i hi) (sq_nonneg _)

theorem re_inner_lt (hT : T.IsSymmetric) {v : E} (hv : v ≠ 0) {t : ℝ}
    (h : ∀ i, D.basis.repr v i ≠ 0 → D.eig i < t) :
    RCLike.re ⟪v, T v⟫_𝕜 < t * ‖v‖ ^ 2 := by
  rw [D.re_inner_apply hT, D.norm_sq_eq, Finset.mul_sum]
  obtain ⟨i, hi⟩ := D.exists_repr_ne_zero hv
  apply Finset.sum_lt_sum
  · intro j _
    by_cases hj : D.basis.repr v j = 0
    · simp [hj]
    · exact mul_le_mul_of_nonneg_right (h j hj).le (sq_nonneg _)
  · exact ⟨i, Finset.mem_univ _,
      mul_lt_mul_of_pos_right (h i hi) (pow_pos (norm_pos_iff.mpr hi) 2)⟩

theorem lt_re_inner (hT : T.IsSymmetric) {v : E} (hv : v ≠ 0) {t : ℝ}
    (h : ∀ i, D.basis.repr v i ≠ 0 → t < D.eig i) :
    t * ‖v‖ ^ 2 < RCLike.re ⟪v, T v⟫_𝕜 := by
  rw [D.re_inner_apply hT, D.norm_sq_eq, Finset.mul_sum]
  obtain ⟨i, hi⟩ := D.exists_repr_ne_zero hv
  apply Finset.sum_lt_sum
  · intro j _
    by_cases hj : D.basis.repr v j = 0
    · simp [hj]
    · exact mul_le_mul_of_nonneg_right (h j hj).le (sq_nonneg _)
  · exact ⟨i, Finset.mem_univ _,
      mul_lt_mul_of_pos_right (h i hi) (pow_pos (norm_pos_iff.mpr hi) 2)⟩

/-! ### Spans of eigenvectors and the counting lemma -/

/-- The span of the basis vectors with index in `s`. -/
noncomputable def span (s : Finset ι) : Submodule 𝕜 E :=
  Submodule.span 𝕜 (Set.range fun i : s => D.basis i)

theorem finrank_span (s : Finset ι) : finrank 𝕜 (D.span s) = s.card := by
  rw [span, finrank_span_eq_card]
  · simp
  · exact D.basis.orthonormal.linearIndependent.comp _ Subtype.val_injective

theorem repr_eq_zero_of_mem_span {s : Finset ι} {v : E} (hv : v ∈ D.span s) {i : ι}
    (hi : i ∉ s) : D.basis.repr v i = 0 := by
  rw [D.basis.repr_apply_apply]
  induction hv using Submodule.span_induction with
  | mem x hx =>
    obtain ⟨⟨j, hj⟩, rfl⟩ := hx
    have hij : i ≠ j := fun h => hi (h ▸ hj)
    exact D.basis.orthonormal.2 hij
  | zero => exact inner_zero_right _
  | add x y _ _ hx hy => rw [inner_add_right, hx, hy, add_zero]
  | smul a x _ hx => rw [inner_smul_right, hx, mul_zero]

theorem basis_mem_span {s : Finset ι} {i : ι} (hi : i ∈ s) : D.basis i ∈ D.span s :=
  Submodule.subset_span ⟨⟨i, hi⟩, rfl⟩

end EigenData

/-- Two subspaces whose dimensions add up to more than `finrank E` meet in a nonzero vector. -/
theorem exists_ne_zero_mem_inf [FiniteDimensional 𝕜 E] {S Y : Submodule 𝕜 E}
    (h : finrank 𝕜 E < finrank 𝕜 S + finrank 𝕜 Y) : ∃ v ∈ S, v ∈ Y ∧ v ≠ 0 := by
  have h1 := Submodule.finrank_sup_add_finrank_inf_eq S Y
  have h2 := Submodule.finrank_le (S ⊔ Y)
  have h3 : S ⊓ Y ≠ ⊥ := by
    intro hb
    rw [hb] at h1
    simp only [finrank_bot, add_zero] at h1
    omega
  obtain ⟨v, hv, hne⟩ := (Submodule.ne_bot_iff _).mp h3
  exact ⟨v, hv.1, hv.2, hne⟩

namespace EigenData

variable {T : E →ₗ[𝕜] E} {ι : Type*} [Fintype ι] (D : EigenData T ι)

theorem finrank_eq_card [FiniteDimensional 𝕜 E] (D' : EigenData T ι) :
    finrank 𝕜 E = Fintype.card ι :=
  finrank_eq_card_basis D'.basis.toBasis

/-- **Counting lemma.** If no nonzero vector of `Y` has all its coefficients with index in `s`
equal to zero, then `finrank Y ≤ #s`. Every min-max argument of the library is an instance. -/
theorem finrank_le_card_of_support [FiniteDimensional 𝕜 E] [DecidableEq ι]
    (Y : Submodule 𝕜 E) (s : Finset ι)
    (h : ∀ y ∈ Y, (∀ i ∈ s, D.basis.repr y i = 0) → y = 0) : finrank 𝕜 Y ≤ s.card := by
  by_contra hlt
  push Not at hlt
  have hE := D.finrank_eq_card
  have hS := D.finrank_span sᶜ
  rw [Finset.card_compl] at hS
  have hs : s.card ≤ Fintype.card ι := Finset.card_le_univ s
  obtain ⟨v, hvS, hvY, hne⟩ :=
    exists_ne_zero_mem_inf (S := D.span sᶜ) (Y := Y) (by omega)
  exact hne (h v hvY fun i hi => D.repr_eq_zero_of_mem_span hvS (by simpa using hi))

/-- **Interval count (Courant–Fischer form).** If `‖T y - c y‖ ≤ r ‖y‖` on a subspace `Y`, at least
`finrank Y` eigenvalues (with multiplicity) lie within `r` of `c`. -/
theorem finrank_le_card_near [FiniteDimensional 𝕜 E] [DecidableEq ι] (hT : T.IsSymmetric)
    (Y : Submodule 𝕜 E) {c r : ℝ} (hY : ∀ y ∈ Y, ‖T y - (c : 𝕜) • y‖ ≤ r * ‖y‖) :
    finrank 𝕜 Y ≤ (univ.filter fun i => |D.eig i - c| ≤ r).card := by
  apply D.finrank_le_card_of_support Y
  intro y hy hzero
  by_contra hne
  have hpos : 0 < ‖y‖ := norm_pos_iff.mpr hne
  rcases lt_or_ge r 0 with hr | hr
  · have h1 := hY y hy
    have h2 : r * ‖y‖ < 0 := mul_neg_of_neg_of_pos hr hpos
    linarith [norm_nonneg (T y - (c : 𝕜) • y)]
  · have key := D.sq_mul_norm_sq_lt hT hne (c := c) hr (fun i hi => by
      by_contra hle
      push Not at hle
      exact hi (hzero i (by simp [hle])))
    have h2 : ‖T y - (c : 𝕜) • y‖ ^ 2 ≤ (r * ‖y‖) ^ 2 :=
      pow_le_pow_left₀ (norm_nonneg _) (hY y hy) 2
    rw [mul_pow] at h2
    linarith

/-- **Rayleigh count.** If `re ⟪y, T y⟫ ≤ t ‖y‖²` on `Y`, at least `finrank Y` eigenvalues are `≤ t`. -/
theorem finrank_le_card_le [FiniteDimensional 𝕜 E] [DecidableEq ι] (hT : T.IsSymmetric)
    (Y : Submodule 𝕜 E) {t : ℝ} (hY : ∀ y ∈ Y, RCLike.re ⟪y, T y⟫_𝕜 ≤ t * ‖y‖ ^ 2) :
    finrank 𝕜 Y ≤ (univ.filter fun i => D.eig i ≤ t).card := by
  apply D.finrank_le_card_of_support Y
  intro y hy hzero
  by_contra hne
  have key := D.lt_re_inner hT hne (t := t) (fun i hi => by
    by_contra hle
    push Not at hle
    exact hi (hzero i (by simp [hle])))
  linarith [hY y hy]

/-- Strict Rayleigh count: `re ⟪y, T y⟫ < t ‖y‖²` for nonzero `y ∈ Y` forces `finrank Y`
eigenvalues `< t`. -/
theorem finrank_le_card_lt [FiniteDimensional 𝕜 E] [DecidableEq ι] (hT : T.IsSymmetric)
    (Y : Submodule 𝕜 E) {t : ℝ}
    (hY : ∀ y ∈ Y, y ≠ 0 → RCLike.re ⟪y, T y⟫_𝕜 < t * ‖y‖ ^ 2) :
    finrank 𝕜 Y ≤ (univ.filter fun i => D.eig i < t).card := by
  apply D.finrank_le_card_of_support Y
  intro y hy hzero
  by_contra hne
  have key := D.le_re_inner hT (v := y) (t := t) (fun i hi => by
    by_contra hle
    push Not at hle
    exact hi (hzero i (by simp [hle])))
  linarith [hY y hy hne]

end EigenData

/-! ### Matrices -/

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n] {A : Matrix n n 𝕜}

/-- Mathlib's eigenvector basis of a Hermitian matrix as `EigenData` of `toEuclideanLin A`. -/
noncomputable def hermitianEigenData (hA : A.IsHermitian) :
    EigenData (Matrix.toEuclideanLin A) n where
  basis := hA.eigenvectorBasis
  eig := hA.eigenvalues
  apply_basis j := by
    simp only [Matrix.IsHermitian.eigenvectorBasis, OrthonormalBasis.reindex_apply]
    exact (Matrix.isSymmetric_toEuclideanLin_iff.mpr hA).apply_eigenvectorBasis
      finrank_euclideanSpace _

theorem hermitianEigenData_eig (hA : A.IsHermitian) :
    (hermitianEigenData hA).eig = hA.eigenvalues := rfl

theorem isSymmetric_of_isHermitian (hA : A.IsHermitian) :
    (Matrix.toEuclideanLin A).IsSymmetric :=
  Matrix.isSymmetric_toEuclideanLin_iff.mpr hA

end CertifyMathlib.Enclosure
