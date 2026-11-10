import CertifyMathlib.Enclosure.Spectral

/-!
# Weyl's inequality and the Appendix A model shift

`docs/proofs/appA_model-shift.md`. Two Hermitian operators `T` and `S` with `‖S v - T v‖ ≤ ε ‖v‖`
for all `v` (that is, `‖S - T‖₂ ≤ ε`) have

* counting functions that differ by a shift of `ε`:
  `#{λ(T) ≤ t} ≤ #{λ(S) ≤ t + ε}` and `#{λ(T) < t} ≤ #{λ(S) < t + ε}`
  (`card_le_le_card_le_add`, `card_lt_le_card_lt_add`), proved by the min-max counting lemma of
  `Spectral.lean`;
* sorted eigenvalues within `ε` (Weyl's inequality, `sorted_weyl`, `weyl_eigenvalues₀`).

Theorem A5 of the document is the specialisation to `T = H_h` (the operator the certificate
uses) and `S = H_spec` (the declared model): `weyl_transfer_enclosure` (S2) and
`weyl_transfer_count` (S3). Lemma A3 (`‖E‖₂² ≤ ‖E‖₁ ‖E‖_∞`, here as the Schur test
`schur_bound`) and Theorem A4 (the composed shift `max(‖D‖₁, ‖D‖_∞) + Δ_builder`,
`model_shift_bound`) are the exact parts of the document; Lemmas A1 and A2 are statements about
binary64 rounding and are not formalized here.
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

/-- A perturbation of norm at most `ε` moves the quadratic form by at most `ε ‖v‖²`. -/
theorem re_inner_le_add {T S : E →ₗ[𝕜] E} {ε : ℝ} (hε : ∀ v, ‖S v - T v‖ ≤ ε * ‖v‖) (v : E) :
    RCLike.re ⟪v, S v⟫_𝕜 ≤ RCLike.re ⟪v, T v⟫_𝕜 + ε * ‖v‖ ^ 2 := by
  have h1 : RCLike.re ⟪v, S v⟫_𝕜 = RCLike.re ⟪v, T v⟫_𝕜 + RCLike.re ⟪v, S v - T v⟫_𝕜 := by
    rw [inner_sub_right, map_sub]
    ring
  have h2 : RCLike.re ⟪v, S v - T v⟫_𝕜 ≤ ‖v‖ * ‖S v - T v‖ :=
    (RCLike.re_le_norm _).trans (norm_inner_le_norm _ _)
  have h3 : ‖v‖ * ‖S v - T v‖ ≤ ‖v‖ * (ε * ‖v‖) :=
    mul_le_mul_of_nonneg_left (hε v) (norm_nonneg _)
  have h4 : ‖v‖ * (ε * ‖v‖) = ε * ‖v‖ ^ 2 := by ring
  linarith

variable [FiniteDimensional 𝕜 E] {T S : E →ₗ[𝕜] E} {ι ι' : Type*} [Fintype ι] [Fintype ι']
  [DecidableEq ι] [DecidableEq ι']

omit [DecidableEq ι] in
/-- **Counting form of Weyl's inequality** (non-strict). -/
theorem card_le_le_card_le_add (hT : T.IsSymmetric) (hS : S.IsSymmetric) (D : EigenData T ι)
    (D' : EigenData S ι') {ε : ℝ} (hε : ∀ v, ‖S v - T v‖ ≤ ε * ‖v‖) (t : ℝ) :
    (univ.filter fun i => D.eig i ≤ t).card ≤ (univ.filter fun j => D'.eig j ≤ t + ε).card := by
  rw [← D.finrank_span]
  apply D'.finrank_le_card_le hS
  intro y hy
  have h1 : RCLike.re ⟪y, T y⟫_𝕜 ≤ t * ‖y‖ ^ 2 := D.re_inner_le hT (fun i hi => by
    by_contra hlt
    exact hi (D.repr_eq_zero_of_mem_span hy (by simpa using hlt)))
  have h2 := re_inner_le_add hε y
  have h3 : (t + ε) * ‖y‖ ^ 2 = t * ‖y‖ ^ 2 + ε * ‖y‖ ^ 2 := by ring
  linarith

omit [DecidableEq ι] in
/-- **Counting form of Weyl's inequality** (strict). -/
theorem card_lt_le_card_lt_add (hT : T.IsSymmetric) (hS : S.IsSymmetric) (D : EigenData T ι)
    (D' : EigenData S ι') {ε : ℝ} (hε : ∀ v, ‖S v - T v‖ ≤ ε * ‖v‖) (t : ℝ) :
    (univ.filter fun i => D.eig i < t).card ≤ (univ.filter fun j => D'.eig j < t + ε).card := by
  rw [← D.finrank_span]
  apply D'.finrank_le_card_lt hS
  intro y hy hne
  have h1 : RCLike.re ⟪y, T y⟫_𝕜 < t * ‖y‖ ^ 2 := D.re_inner_lt hT hne (fun i hi => by
    by_contra hlt
    exact hi (D.repr_eq_zero_of_mem_span hy (by simpa using hlt)))
  have h2 := re_inner_le_add hε y
  have h3 : (t + ε) * ‖y‖ ^ 2 = t * ‖y‖ ^ 2 + ε * ‖y‖ ^ 2 := by ring
  linarith

omit [FiniteDimensional 𝕜 E] [DecidableEq ι] in
theorem card_filter_le_eq_sub (D : EigenData T ι) (t : ℝ) :
    (univ.filter fun i => t ≤ D.eig i).card =
      Fintype.card ι - (univ.filter fun i => D.eig i < t).card := by
  rw [← Finset.card_univ,
    ← Finset.card_filter_add_card_filter_not (s := univ) (fun i => D.eig i < t)]
  simp [not_lt]

omit [DecidableEq ι'] in
/-- **Counting form of Weyl's inequality** (upper tail). -/
theorem card_ge_le_card_ge_sub (hT : T.IsSymmetric) (hS : S.IsSymmetric) (D : EigenData T ι)
    (D' : EigenData S ι') {ε : ℝ} (hε : ∀ v, ‖S v - T v‖ ≤ ε * ‖v‖) (t : ℝ) :
    (univ.filter fun i => t ≤ D.eig i).card ≤ (univ.filter fun j => t - ε ≤ D'.eig j).card := by
  have hs := card_lt_le_card_lt_add hS hT D' D (ε := ε)
    (fun v => by rw [norm_sub_rev]; exact hε v) (t - ε)
  rw [sub_add_cancel] at hs
  rw [card_filter_le_eq_sub D t, card_filter_le_eq_sub D' (t - ε)]
  have hc : Fintype.card ι = Fintype.card ι' := D.finrank_eq_card.symm.trans D'.finrank_eq_card
  have h1 : (univ.filter fun i => D.eig i < t).card ≤ Fintype.card ι := by
    rw [← Finset.card_univ]; exact Finset.card_filter_le _ _
  omega

omit [DecidableEq ι] [DecidableEq ι'] in
/-- **Weyl's inequality** for sorted eigenvalues (one direction). The eigenvalues are indexed by
`Fin N` in decreasing order, Mathlib's convention (`Matrix.IsHermitian.eigenvalues₀`). -/
theorem sorted_weyl_ge {N : ℕ} (hT : T.IsSymmetric) (hS : S.IsSymmetric)
    (D : EigenData T (Fin N)) (D' : EigenData S (Fin N)) (hD : Antitone D.eig)
    (hD' : Antitone D'.eig) {ε : ℝ} (hε : ∀ v, ‖S v - T v‖ ≤ ε * ‖v‖) (k : Fin N) :
    D.eig k - ε ≤ D'.eig k := by
  by_contra h
  push Not at h
  have h1 : Finset.Iic k ⊆ univ.filter (fun i => D.eig k ≤ D.eig i) := by
    intro i hi
    simp only [Finset.mem_Iic] at hi
    simpa using hD hi
  have h2 : univ.filter (fun j => D.eig k - ε ≤ D'.eig j) ⊆ Finset.Iio k := by
    intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
    simp only [Finset.mem_Iio]
    by_contra hkj
    push Not at hkj
    have := hD' hkj
    linarith
  have h3 := card_ge_le_card_ge_sub hT hS D D' hε (D.eig k)
  have h4 := Finset.card_le_card h1
  have h5 := Finset.card_le_card h2
  rw [Fin.card_Iic] at h4
  rw [Fin.card_Iio] at h5
  omega

omit [DecidableEq ι] [DecidableEq ι'] in
/-- **Weyl's inequality**: `|λ_k(T) - λ_k(S)| ≤ ε` for the sorted eigenvalues. -/
theorem sorted_weyl {N : ℕ} (hT : T.IsSymmetric) (hS : S.IsSymmetric)
    (D : EigenData T (Fin N)) (D' : EigenData S (Fin N)) (hD : Antitone D.eig)
    (hD' : Antitone D'.eig) {ε : ℝ} (hε : ∀ v, ‖S v - T v‖ ≤ ε * ‖v‖) (k : Fin N) :
    |D.eig k - D'.eig k| ≤ ε := by
  have h1 := sorted_weyl_ge hT hS D D' hD hD' hε k
  have h2 := sorted_weyl_ge hS hT D' D hD' hD (fun v => by rw [norm_sub_rev]; exact hε v) k
  rw [abs_le]
  constructor <;> linarith

/-! ### Matrices -/

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Mathlib's sorted eigenbasis of a Hermitian matrix, with eigenvalues `eigenvalues₀`
(decreasing). -/
noncomputable def sortedEigenData {A : Matrix n n 𝕜} (hA : A.IsHermitian) :
    EigenData (Matrix.toEuclideanLin A) (Fin (Fintype.card n)) where
  basis := (isSymmetric_of_isHermitian hA).eigenvectorBasis finrank_euclideanSpace
  eig := hA.eigenvalues₀
  apply_basis i := (isSymmetric_of_isHermitian hA).apply_eigenvectorBasis finrank_euclideanSpace i

theorem sortedEigenData_eig {A : Matrix n n 𝕜} (hA : A.IsHermitian) :
    (sortedEigenData hA).eig = hA.eigenvalues₀ := rfl

/-- **Weyl's inequality for Hermitian matrices** (Horn and Johnson, Cor. 4.3.15):
`‖(B - A) v‖ ≤ ε ‖v‖` for all `v` gives `|λ_k(A) - λ_k(B)| ≤ ε` for every sorted index `k`. -/
theorem weyl_eigenvalues₀ {A B : Matrix n n 𝕜} (hA : A.IsHermitian) (hB : B.IsHermitian) {ε : ℝ}
    (hε : ∀ v, ‖toEuclideanLin (B - A) v‖ ≤ ε * ‖v‖) (k : Fin (Fintype.card n)) :
    |hA.eigenvalues₀ k - hB.eigenvalues₀ k| ≤ ε :=
  sorted_weyl (isSymmetric_of_isHermitian hA) (isSymmetric_of_isHermitian hB)
    (sortedEigenData hA) (sortedEigenData hB) hA.eigenvalues₀_antitone hB.eigenvalues₀_antitone
    (fun v => by simpa [map_sub] using hε v) k

/-- **Theorem A5 (S2).** An enclosure of the sorted eigenvalue `λ_k(H_h)` with radius `r` is an
enclosure of `λ_k(H_spec)` with radius `r + Δ`; a matching of claimed values to distinct indices
of `H_h` is therefore a matching to the same distinct indices of `H_spec`. -/
theorem weyl_transfer_enclosure {Hh Hs : Matrix n n 𝕜} (hh : Hh.IsHermitian) (hs : Hs.IsHermitian)
    {Δ : ℝ} (hΔ : ∀ v, ‖toEuclideanLin (Hs - Hh) v‖ ≤ Δ * ‖v‖) {κ : Type*}
    (f : κ → Fin (Fintype.card n)) (μ r : κ → ℝ)
    (hf : ∀ c, |hh.eigenvalues₀ (f c) - μ c| ≤ r c) :
    ∀ c, |hs.eigenvalues₀ (f c) - μ c| ≤ r c + Δ := by
  intro c
  have h1 := weyl_eigenvalues₀ hh hs hΔ (f c)
  have h2 := hf c
  calc |hs.eigenvalues₀ (f c) - μ c|
      = |(hs.eigenvalues₀ (f c) - hh.eigenvalues₀ (f c)) + (hh.eigenvalues₀ (f c) - μ c)| := by
        ring_nf
    _ ≤ |hs.eigenvalues₀ (f c) - hh.eigenvalues₀ (f c)| + |hh.eigenvalues₀ (f c) - μ c| :=
        abs_add_le _ _
    _ ≤ Δ + r c := by
        rw [abs_sub_comm] at h1
        exact add_le_add h1 h2
    _ = r c + Δ := add_comm _ _

/-- **Theorem A5 (S3).** A count bracket `#{λ(H_h) < σ - β} ≤ ν ≤ #{λ(H_h) ≤ σ + β}` becomes
`#{λ(H_spec) < σ - β - Δ} ≤ ν ≤ #{λ(H_spec) ≤ σ + β + Δ}`. -/
theorem weyl_transfer_count {Hh Hs : Matrix n n 𝕜} (hh : Hh.IsHermitian) (hs : Hs.IsHermitian)
    {Δ : ℝ} (hΔ : ∀ v, ‖toEuclideanLin (Hs - Hh) v‖ ≤ Δ * ‖v‖) {σ β : ℝ} {ν : ℕ}
    (hlow : (univ.filter fun i => hh.eigenvalues i < σ - β).card ≤ ν)
    (hup : ν ≤ (univ.filter fun i => hh.eigenvalues i ≤ σ + β).card) :
    (univ.filter fun i => hs.eigenvalues i < σ - β - Δ).card ≤ ν ∧
      ν ≤ (univ.filter fun i => hs.eigenvalues i ≤ σ + β + Δ).card := by
  have hΔ' : ∀ v, ‖toEuclideanLin Hh v - toEuclideanLin Hs v‖ ≤ Δ * ‖v‖ := fun v => by
    rw [norm_sub_rev]; simpa [map_sub] using hΔ v
  have hΔ'' : ∀ v, ‖toEuclideanLin Hs v - toEuclideanLin Hh v‖ ≤ Δ * ‖v‖ := fun v => by
    simpa [map_sub] using hΔ v
  constructor
  · have h := card_lt_le_card_lt_add (isSymmetric_of_isHermitian hs) (isSymmetric_of_isHermitian hh)
      (hermitianEigenData hs) (hermitianEigenData hh) hΔ' (σ - β - Δ)
    simp only [hermitianEigenData_eig, sub_add_cancel] at h
    exact h.trans hlow
  · have h := card_le_le_card_le_add (isSymmetric_of_isHermitian hh) (isSymmetric_of_isHermitian hs)
      (hermitianEigenData hh) (hermitianEigenData hs) hΔ'' (σ + β)
    simp only [hermitianEigenData_eig] at h
    exact hup.trans h

/-! ### Lemma A3 and Theorem A4 -/

/-- **Lemma A3** (Schur test): with row sums `≤ R` and column sums `≤ C`,
`‖M x‖² ≤ R C ‖x‖²`, i.e. `‖M‖₂² ≤ ‖M‖_∞ ‖M‖₁`. -/
theorem schur_bound {m : Type*} [Fintype m] (M : Matrix m n 𝕜) {Rb Cb : ℝ} (hR0 : 0 ≤ Rb)
    (hR : ∀ i, ∑ j, ‖M i j‖ ≤ Rb) (hC : ∀ j, ∑ i, ‖M i j‖ ≤ Cb) (x : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin M x‖ ^ 2 ≤ Rb * Cb * ‖x‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
  have hcomp : ∀ i, (toEuclideanLin M x) i = ∑ j, M i j * x j := fun i => by
    rw [Matrix.toLpLin_apply]
    rfl
  have step1 : ∀ i, ‖(toEuclideanLin M x) i‖ ^ 2 ≤ Rb * ∑ j, ‖M i j‖ * ‖x j‖ ^ 2 := by
    intro i
    rw [hcomp]
    have h1 : ‖∑ j, M i j * x j‖ ≤ ∑ j, ‖M i j‖ * ‖x j‖ :=
      (norm_sum_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j _ => norm_mul _ _))
    have h2 : (∑ j, ‖M i j‖ * ‖x j‖) ^ 2 ≤ (∑ j, ‖M i j‖) * ∑ j, ‖M i j‖ * ‖x j‖ ^ 2 :=
      Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul _ (fun j _ => norm_nonneg _)
        (fun j _ => mul_nonneg (norm_nonneg _) (sq_nonneg _)) (fun j _ => le_of_eq (by ring))
    have h3 : ‖∑ j, M i j * x j‖ ^ 2 ≤ (∑ j, ‖M i j‖ * ‖x j‖) ^ 2 :=
      pow_le_pow_left₀ (norm_nonneg _) h1 2
    have h4 : (∑ j, ‖M i j‖) * (∑ j, ‖M i j‖ * ‖x j‖ ^ 2) ≤ Rb * ∑ j, ‖M i j‖ * ‖x j‖ ^ 2 :=
      mul_le_mul_of_nonneg_right (hR i)
        (Finset.sum_nonneg fun j _ => mul_nonneg (norm_nonneg _) (sq_nonneg _))
    linarith
  calc ∑ i, ‖(toEuclideanLin M x) i‖ ^ 2 ≤ ∑ i, Rb * ∑ j, ‖M i j‖ * ‖x j‖ ^ 2 :=
        Finset.sum_le_sum fun i _ => step1 i
    _ = Rb * ∑ j, (∑ i, ‖M i j‖) * ‖x j‖ ^ 2 := by
        rw [← Finset.mul_sum, Finset.sum_comm]
        congr 1
        exact Finset.sum_congr rfl fun j _ => (Finset.sum_mul _ _ _).symm
    _ ≤ Rb * ∑ j, Cb * ‖x j‖ ^ 2 :=
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_right (hC j) (sq_nonneg _)) hR0
    _ = Rb * Cb * ∑ j, ‖x j‖ ^ 2 := by rw [← Finset.mul_sum]; ring

/-- Lemma A3, second inequality: `‖M x‖ ≤ max(R, C) ‖x‖` (since `√(R C) ≤ max(R, C)`). -/
theorem norm_le_max_of_row_col {m : Type*} [Fintype m] (M : Matrix m n 𝕜) {Rb Cb : ℝ}
    (hR0 : 0 ≤ Rb) (hC0 : 0 ≤ Cb) (hR : ∀ i, ∑ j, ‖M i j‖ ≤ Rb) (hC : ∀ j, ∑ i, ‖M i j‖ ≤ Cb)
    (x : EuclideanSpace 𝕜 n) : ‖toEuclideanLin M x‖ ≤ max Rb Cb * ‖x‖ := by
  have h := schur_bound M hR0 hR hC x
  have hmax : Rb * Cb ≤ max Rb Cb ^ 2 := by
    rw [sq]
    exact mul_le_mul (le_max_left _ _) (le_max_right _ _) hC0 (le_max_of_le_left hR0)
  have h2 : ‖toEuclideanLin M x‖ ^ 2 ≤ (max Rb Cb * ‖x‖) ^ 2 := by
    rw [mul_pow]
    exact h.trans (mul_le_mul_of_nonneg_right hmax (sq_nonneg _))
  exact (pow_le_pow_iff_left₀ (norm_nonneg _)
    (mul_nonneg (le_max_of_le_left hR0) (norm_nonneg _)) two_ne_zero).mp h2

/-- **Theorem A4** (the model shift): with `D = H_h - H_ref` having row sums `≤ R` and column sums
`≤ C`, and `‖(H_ref - H_spec) x‖ ≤ Δ_b ‖x‖`,
`‖(H_h - H_spec) x‖ ≤ (max(R, C) + Δ_b) ‖x‖`. -/
theorem model_shift_bound (Hh Href Hs : Matrix n n 𝕜) {Rb Cb Δb : ℝ} (hR0 : 0 ≤ Rb) (hC0 : 0 ≤ Cb)
    (hR : ∀ i, ∑ j, ‖(Hh - Href) i j‖ ≤ Rb) (hC : ∀ j, ∑ i, ‖(Hh - Href) i j‖ ≤ Cb)
    (hb : ∀ x, ‖toEuclideanLin (Href - Hs) x‖ ≤ Δb * ‖x‖) (x : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (Hh - Hs) x‖ ≤ (max Rb Cb + Δb) * ‖x‖ := by
  have e : toEuclideanLin (Hh - Hs) x =
      toEuclideanLin (Hh - Href) x + toEuclideanLin (Href - Hs) x := by
    rw [← LinearMap.add_apply, ← map_add, sub_add_sub_cancel]
  rw [e, add_mul]
  exact (norm_add_le _ _).trans
    (add_le_add (norm_le_max_of_row_col _ hR0 hC0 hR hC x) (hb x))

end CertifyMathlib.Enclosure
