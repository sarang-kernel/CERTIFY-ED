import Mathlib.Analysis.Matrix.Spectrum

/-!
# Quadratic forms of matrices and the dimension count

This file holds the tools on which the counting statements of CERTIFY-ED rest
(`docs/theory.md`, section 3, in the code repository). For a square matrix `M` over `𝕜`
(the real or the complex numbers) it defines the real quadratic form `qf M x = Re (xᴴ M x)`, the
subspaces on which this form is negative (`NegOn`) or non-positive (`NonposOn`), and proves the
dimension count behind Sylvester's law of inertia:

* `finrank_add_finrank_le`: two subspaces that meet only in `0` have dimensions adding to at
  most `n`;
* `finrank_le_card_neg`: if `M = L diag(d) Lᴴ` with `Lᴴ` invertible, every subspace on which `M`
  is negative has dimension at most the number of negative entries of `d`;
* `exists_negOn`: and there is such a subspace of exactly that dimension;
* `sylvester_card`: two such factorizations of one matrix have the same number of negative and
  the same number of non-positive diagonal entries (Sylvester's law of inertia).

The same statements are proved for non-positive forms and the entries `d i ≤ 0`. No eigenvalue
appears here; `CertifyMathlib.Counting.Inertia` applies the count to the spectral decomposition of a
Hermitian matrix. Mathlib v4.35.0 has no Courant-Fischer theorem for matrices and no uniqueness
part of Sylvester's law for Hermitian matrices, which is why this file builds the count itself.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

section Basic

variable {n : Type*} [Fintype n]

/-- The real quadratic form `Re (xᴴ M x)` of a square matrix. -/
noncomputable def qf (M : Matrix n n 𝕜) (x : n → 𝕜) : ℝ :=
  RCLike.re (star x ⬝ᵥ (M *ᵥ x))

/-- The squared Euclidean norm `Σ |x i|²` of a vector. -/
noncomputable def nsq (x : n → 𝕜) : ℝ :=
  ∑ i, ‖x i‖ ^ 2

theorem nsq_nonneg (x : n → 𝕜) : 0 ≤ nsq x :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem qf_add (M N : Matrix n n 𝕜) (x : n → 𝕜) : qf (M + N) x = qf M x + qf N x := by
  simp only [qf, add_mulVec, dotProduct_add, map_add]

theorem qf_sub (M N : Matrix n n 𝕜) (x : n → 𝕜) : qf (M - N) x = qf M x - qf N x := by
  simp only [qf, sub_mulVec, dotProduct_sub, map_sub]

theorem re_star_dotProduct_self (x : n → 𝕜) : RCLike.re (star x ⬝ᵥ x) = nsq x := by
  simp only [dotProduct, Pi.star_apply, RCLike.star_def, map_sum, nsq]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [RCLike.conj_mul, ← RCLike.ofReal_pow, RCLike.ofReal_re]

/-- The quadratic form of `c I` is `c ‖x‖²`. -/
theorem qf_smul_one [DecidableEq n] (c : ℝ) (x : n → 𝕜) :
    qf ((c : 𝕜) • (1 : Matrix n n 𝕜)) x = c * nsq x := by
  simp only [qf, smul_mulVec, one_mulVec, dotProduct_smul, smul_eq_mul, RCLike.re_ofReal_mul,
    re_star_dotProduct_self]

theorem qf_sub_smul_one [DecidableEq n] (M : Matrix n n 𝕜) (c : ℝ) (x : n → 𝕜) :
    qf (M - (c : 𝕜) • (1 : Matrix n n 𝕜)) x = qf M x - c * nsq x := by
  rw [qf_sub, qf_smul_one]

/-- Congruence moves the quadratic form: `xᴴ (L D Lᴴ) x = (Lᴴ x)ᴴ D (Lᴴ x)`. -/
theorem qf_congr {p : Type*} [Fintype p] (L : Matrix n p 𝕜) (D : Matrix p p 𝕜) (x : n → 𝕜) :
    qf (L * D * Lᴴ) x = qf D (Lᴴ *ᵥ x) := by
  simp only [qf, star_mulVec, conjTranspose_conjTranspose, ← mulVec_mulVec, dotProduct_mulVec]

/-- The diagonal matrix with the real entries `d`. -/
noncomputable def dg [DecidableEq n] (d : n → ℝ) : Matrix n n 𝕜 :=
  diagonal fun i => (d i : 𝕜)

theorem qf_dg [DecidableEq n] (d : n → ℝ) (y : n → 𝕜) :
    qf (dg d) y = ∑ i, d i * ‖y i‖ ^ 2 := by
  simp only [qf, dg, dotProduct, mulVec_diagonal, Pi.star_apply, RCLike.star_def, map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [mul_left_comm, RCLike.conj_mul, ← RCLike.ofReal_pow, ← RCLike.ofReal_mul, RCLike.ofReal_re]

/-- `M` is negative on every nonzero vector of `V`. -/
def NegOn (M : Matrix n n 𝕜) (V : Submodule 𝕜 (n → 𝕜)) : Prop :=
  ∀ x ∈ V, x ≠ 0 → qf M x < 0

/-- `M` is non-positive on every vector of `V`. -/
def NonposOn (M : Matrix n n 𝕜) (V : Submodule 𝕜 (n → 𝕜)) : Prop :=
  ∀ x ∈ V, qf M x ≤ 0

/-- Two subspaces of `𝕜ⁿ` that meet only in `0` have dimensions adding to at most `n`. -/
theorem finrank_add_finrank_le (V U : Submodule 𝕜 (n → 𝕜)) (h : ∀ x ∈ V, x ∈ U → x = 0) :
    Module.finrank 𝕜 V + Module.finrank 𝕜 U ≤ Fintype.card n := by
  have hbot : V ⊓ U = ⊥ := disjoint_iff.mp (Submodule.disjoint_def.mpr h)
  have h1 := Submodule.finrank_sup_add_finrank_inf_eq V U
  rw [hbot, finrank_bot, add_zero] at h1
  have h2 := Submodule.finrank_le (V ⊔ U)
  rw [Module.finrank_fintype_fun_eq_card] at h2
  omega

/-- The vectors supported in the index set `S`. -/
def coordSub (S : Finset n) : Submodule 𝕜 (n → 𝕜) where
  carrier := {y | ∀ i, i ∉ S → y i = 0}
  add_mem' := by
    intro a b ha hb
    show ∀ i, i ∉ S → (a + b) i = 0
    intro i hi
    simp [ha i hi, hb i hi]
  zero_mem' := by
    show ∀ i, i ∉ S → (0 : n → 𝕜) i = 0
    intro i _
    rfl
  smul_mem' := by
    intro c y hy
    show ∀ i, i ∉ S → (c • y) i = 0
    intro i hi
    simp [hy i hi]

omit [Fintype n] in
theorem mem_coordSub {S : Finset n} {y : n → 𝕜} :
    y ∈ coordSub (𝕜 := 𝕜) S ↔ ∀ i, i ∉ S → y i = 0 :=
  Iff.rfl

/-- The subspace of vectors supported in `S` has dimension at least `#S`. -/
theorem card_le_finrank_coordSub [DecidableEq n] (S : Finset n) :
    S.card ≤ Module.finrank 𝕜 (coordSub (𝕜 := 𝕜) S) := by
  let v : S → coordSub (𝕜 := 𝕜) S := fun i => ⟨Pi.single (i : n) 1, by
    rw [mem_coordSub]
    intro j hj
    exact Pi.single_eq_of_ne (fun h => hj (by rw [h]; exact i.2)) _⟩
  have hli : LinearIndependent 𝕜 v := by
    apply LinearIndependent.of_comp (coordSub (𝕜 := 𝕜) S).subtype
    have h0 := (Pi.basisFun 𝕜 n).linearIndependent.comp (fun i : S => (i : n))
      Subtype.val_injective
    convert h0 using 1
    funext i
    simp [v, Pi.basisFun_apply]
  have := hli.fintype_card_le_finrank
  simpa using this

omit [Fintype n] in
theorem exists_ne_zero_apply {y : n → 𝕜} (hy0 : y ≠ 0) : ∃ j, y j ≠ 0 := by
  by_contra h
  exact hy0 (funext fun i => by
    by_contra hi
    exact h ⟨i, hi⟩)

variable [DecidableEq n]

/-- `diag(d)` is negative on the vectors supported where `d < 0`. -/
theorem negOn_dg (d : n → ℝ) :
    NegOn (dg (𝕜 := 𝕜) d) (coordSub (Finset.univ.filter fun i => d i < 0)) := by
  intro y hy hy0
  rw [qf_dg]
  obtain ⟨j, hj⟩ := exists_ne_zero_apply hy0
  have hjS : d j < 0 := by
    by_contra h
    exact hj (hy j (by simp [h]))
  have key : ∑ i, d i * ‖y i‖ ^ 2 < ∑ _i : n, (0 : ℝ) := by
    apply Finset.sum_lt_sum
    · intro i _
      by_cases hi : d i < 0
      · nlinarith [sq_nonneg ‖y i‖]
      · have : y i = 0 := hy i (by simp [hi])
        simp [this]
    · refine ⟨j, Finset.mem_univ _, ?_⟩
      have h1 : 0 < ‖y j‖ := norm_pos_iff.mpr hj
      have h2 : 0 < ‖y j‖ ^ 2 := by positivity
      exact mul_neg_of_neg_of_pos hjS h2
  simpa using key

theorem nonneg_dg (d : n → ℝ) {y : n → 𝕜}
    (hy : y ∈ coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => ¬ d i < 0)) :
    0 ≤ qf (dg d) y := by
  rw [qf_dg]
  apply Finset.sum_nonneg
  intro i _
  by_cases hi : d i < 0
  · have : y i = 0 := hy i (by simp [hi])
    simp [this]
  · exact mul_nonneg (not_lt.mp hi) (sq_nonneg _)

/-- `diag(d)` is non-positive on the vectors supported where `d ≤ 0`. -/
theorem nonposOn_dg (d : n → ℝ) :
    NonposOn (dg (𝕜 := 𝕜) d) (coordSub (Finset.univ.filter fun i => d i ≤ 0)) := by
  intro y hy
  rw [qf_dg]
  apply Finset.sum_nonpos
  intro i _
  by_cases hi : d i ≤ 0
  · nlinarith [sq_nonneg ‖y i‖]
  · have : y i = 0 := hy i (by simp [hi])
    simp [this]

theorem pos_dg (d : n → ℝ) {y : n → 𝕜}
    (hy : y ∈ coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => ¬ d i ≤ 0)) (hy0 : y ≠ 0) :
    0 < qf (dg d) y := by
  rw [qf_dg]
  obtain ⟨j, hj⟩ := exists_ne_zero_apply hy0
  have hjS : 0 < d j := by
    by_contra h
    exact hj (hy j (by simp [not_lt.mp h]))
  apply Finset.sum_pos'
  · intro i _
    by_cases hi : d i ≤ 0
    · have : y i = 0 := hy i (by simp [hi])
      simp [this]
    · exact mul_nonneg (not_le.mp hi).le (sq_nonneg _)
  · refine ⟨j, Finset.mem_univ _, ?_⟩
    have h1 : 0 < ‖y j‖ := norm_pos_iff.mpr hj
    have h2 : 0 < ‖y j‖ ^ 2 := by positivity
    exact mul_pos hjS h2

/-- A subspace on which `diag(d)` is negative has dimension at most `#{i | d i < 0}`. -/
theorem finrank_le_of_negOn_dg (d : n → ℝ) (V : Submodule 𝕜 (n → 𝕜)) (hV : NegOn (dg d) V) :
    Module.finrank 𝕜 V ≤ (Finset.univ.filter fun i => d i < 0).card := by
  have h1 := finrank_add_finrank_le V
    (coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => ¬ d i < 0))
    (fun x hxV hxU => by
      by_contra hx
      have h1 := hV x hxV hx
      have h2 := nonneg_dg d hxU
      linarith)
  have h2 := card_le_finrank_coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => ¬ d i < 0)
  have h3 := Finset.card_filter_add_card_filter_not (s := Finset.univ) (fun i => d i < 0)
  rw [Finset.card_univ] at h3
  omega

/-- A subspace on which `diag(d)` is non-positive has dimension at most `#{i | d i ≤ 0}`. -/
theorem finrank_le_of_nonposOn_dg (d : n → ℝ) (V : Submodule 𝕜 (n → 𝕜))
    (hV : NonposOn (dg d) V) :
    Module.finrank 𝕜 V ≤ (Finset.univ.filter fun i => d i ≤ 0).card := by
  have h1 := finrank_add_finrank_le V
    (coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => ¬ d i ≤ 0))
    (fun x hxV hxU => by
      by_contra hx
      have h1 := hV x hxV
      have h2 := pos_dg d hxU hx
      linarith)
  have h2 := card_le_finrank_coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => ¬ d i ≤ 0)
  have h3 := Finset.card_filter_add_card_filter_not (s := Finset.univ) (fun i => d i ≤ 0)
  rw [Finset.card_univ] at h3
  omega

end Basic

section Transport

variable {n p : Type*} [Fintype n] [Fintype p]

omit [Fintype n] [Fintype p] in
theorem finrank_map_equiv (e : (p → 𝕜) ≃ₗ[𝕜] (n → 𝕜)) (V : Submodule 𝕜 (p → 𝕜)) :
    Module.finrank 𝕜 (V.map (e : (p → 𝕜) →ₗ[𝕜] (n → 𝕜))) = Module.finrank 𝕜 V :=
  (Submodule.equivMapOfInjective (e : (p → 𝕜) →ₗ[𝕜] (n → 𝕜)) e.injective V).finrank_eq.symm

/-- A linear isomorphism that carries one quadratic form into another carries negative subspaces
to negative subspaces. -/
theorem negOn_map {M : Matrix p p 𝕜} {D : Matrix n n 𝕜} (e : (p → 𝕜) ≃ₗ[𝕜] (n → 𝕜))
    (he : ∀ x, qf M x = qf D (e x)) {V : Submodule 𝕜 (p → 𝕜)} (hV : NegOn M V) :
    NegOn D (V.map (e : (p → 𝕜) →ₗ[𝕜] (n → 𝕜))) := by
  intro y hy hy0
  obtain ⟨x, hx, rfl⟩ := Submodule.mem_map.mp hy
  have hx0 : x ≠ 0 := by
    rintro rfl
    exact hy0 (by simp)
  have := hV x hx hx0
  rw [he] at this
  simpa using this

theorem nonposOn_map {M : Matrix p p 𝕜} {D : Matrix n n 𝕜} (e : (p → 𝕜) ≃ₗ[𝕜] (n → 𝕜))
    (he : ∀ x, qf M x = qf D (e x)) {V : Submodule 𝕜 (p → 𝕜)} (hV : NonposOn M V) :
    NonposOn D (V.map (e : (p → 𝕜) →ₗ[𝕜] (n → 𝕜))) := by
  intro y hy
  obtain ⟨x, hx, rfl⟩ := Submodule.mem_map.mp hy
  have := hV x hx
  rw [he] at this
  simpa using this

variable [DecidableEq n] [DecidableEq p]

/-- The linear isomorphism `x ↦ P x` of a matrix `P` with two-sided inverse `Q`. -/
noncomputable def mulVecEquiv (P : Matrix n p 𝕜) (Q : Matrix p n 𝕜) (h1 : P * Q = 1)
    (h2 : Q * P = 1) : (p → 𝕜) ≃ₗ[𝕜] (n → 𝕜) where
  toFun x := P *ᵥ x
  invFun y := Q *ᵥ y
  map_add' x y := mulVec_add P x y
  map_smul' c x := by simp [mulVec_smul]
  left_inv x := by simp only [mulVec_mulVec, h2, one_mulVec]
  right_inv y := by simp only [mulVec_mulVec, h1, one_mulVec]

theorem mulVecEquiv_apply (P : Matrix n p 𝕜) (Q : Matrix p n 𝕜) (h1 : P * Q = 1)
    (h2 : Q * P = 1) (x : p → 𝕜) : mulVecEquiv P Q h1 h2 x = P *ᵥ x :=
  rfl

/-- If `M = L diag(d) Lᴴ` and `Lᴴ` has the two-sided inverse `K`, every subspace on which `M` is
negative has dimension at most `#{i | d i < 0}`. -/
theorem finrank_le_card_neg {M : Matrix p p 𝕜} (L K : Matrix p n 𝕜) (hK1 : Lᴴ * K = 1)
    (hK2 : K * Lᴴ = 1) (d : n → ℝ) (hM : M = L * dg d * Lᴴ) (V : Submodule 𝕜 (p → 𝕜))
    (hV : NegOn M V) : Module.finrank 𝕜 V ≤ (Finset.univ.filter fun i => d i < 0).card := by
  have he : ∀ x, qf M x = qf (dg d) (mulVecEquiv Lᴴ K hK1 hK2 x) := fun x => by
    rw [hM, qf_congr, mulVecEquiv_apply]
  have h1 := finrank_le_of_negOn_dg d _ (negOn_map _ he hV)
  rwa [finrank_map_equiv] at h1

theorem finrank_le_card_nonpos {M : Matrix p p 𝕜} (L K : Matrix p n 𝕜) (hK1 : Lᴴ * K = 1)
    (hK2 : K * Lᴴ = 1) (d : n → ℝ) (hM : M = L * dg d * Lᴴ) (V : Submodule 𝕜 (p → 𝕜))
    (hV : NonposOn M V) : Module.finrank 𝕜 V ≤ (Finset.univ.filter fun i => d i ≤ 0).card := by
  have he : ∀ x, qf M x = qf (dg d) (mulVecEquiv Lᴴ K hK1 hK2 x) := fun x => by
    rw [hM, qf_congr, mulVecEquiv_apply]
  have h1 := finrank_le_of_nonposOn_dg d _ (nonposOn_map _ he hV)
  rwa [finrank_map_equiv] at h1

/-- If `M = L diag(d) Lᴴ` with `Lᴴ` invertible, `M` is negative on a subspace of dimension
`#{i | d i < 0}`. -/
theorem exists_negOn {M : Matrix p p 𝕜} (L K : Matrix p n 𝕜) (hK1 : Lᴴ * K = 1)
    (hK2 : K * Lᴴ = 1) (d : n → ℝ) (hM : M = L * dg d * Lᴴ) :
    ∃ V : Submodule 𝕜 (p → 𝕜),
      (Finset.univ.filter fun i => d i < 0).card ≤ Module.finrank 𝕜 V ∧ NegOn M V := by
  let e := mulVecEquiv Lᴴ K hK1 hK2
  have he : ∀ y, qf (dg d) y = qf M (e.symm y) := fun y => by
    rw [hM, qf_congr]
    have : Lᴴ *ᵥ e.symm y = y := e.apply_symm_apply y
    rw [this]
  refine ⟨(coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => d i < 0)).map
    (e.symm : (n → 𝕜) →ₗ[𝕜] (p → 𝕜)), ?_, negOn_map e.symm he (negOn_dg d)⟩
  rw [finrank_map_equiv]
  exact card_le_finrank_coordSub _

theorem exists_nonposOn {M : Matrix p p 𝕜} (L K : Matrix p n 𝕜) (hK1 : Lᴴ * K = 1)
    (hK2 : K * Lᴴ = 1) (d : n → ℝ) (hM : M = L * dg d * Lᴴ) :
    ∃ V : Submodule 𝕜 (p → 𝕜),
      (Finset.univ.filter fun i => d i ≤ 0).card ≤ Module.finrank 𝕜 V ∧ NonposOn M V := by
  let e := mulVecEquiv Lᴴ K hK1 hK2
  have he : ∀ y, qf (dg d) y = qf M (e.symm y) := fun y => by
    rw [hM, qf_congr]
    have : Lᴴ *ᵥ e.symm y = y := e.apply_symm_apply y
    rw [this]
  refine ⟨(coordSub (𝕜 := 𝕜) (Finset.univ.filter fun i => d i ≤ 0)).map
    (e.symm : (n → 𝕜) →ₗ[𝕜] (p → 𝕜)), ?_, nonposOn_map e.symm he (nonposOn_dg d)⟩
  rw [finrank_map_equiv]
  exact card_le_finrank_coordSub _

/-- **Sylvester's law of inertia** for congruence to real diagonal matrices: two factorizations
`M = L diag(d) Lᴴ = L' diag(d') L'ᴴ` with invertible `Lᴴ`, `L'ᴴ` have the same number of negative
entries and the same number of non-positive entries (hence also of zero and of positive ones). -/
theorem sylvester_card {q : Type*} [Fintype q] [DecidableEq q] {M : Matrix p p 𝕜}
    (L K : Matrix p n 𝕜) (hK1 : Lᴴ * K = 1) (hK2 : K * Lᴴ = 1) (d : n → ℝ)
    (hM : M = L * dg d * Lᴴ)
    (L' K' : Matrix p q 𝕜) (hK1' : L'ᴴ * K' = 1) (hK2' : K' * L'ᴴ = 1) (d' : q → ℝ)
    (hM' : M = L' * dg d' * L'ᴴ) :
    (Finset.univ.filter fun i => d i < 0).card = (Finset.univ.filter fun i => d' i < 0).card ∧
    (Finset.univ.filter fun i => d i ≤ 0).card = (Finset.univ.filter fun i => d' i ≤ 0).card := by
  obtain ⟨V, hV1, hV2⟩ := exists_negOn L K hK1 hK2 d hM
  obtain ⟨V', hV1', hV2'⟩ := exists_negOn L' K' hK1' hK2' d' hM'
  obtain ⟨W, hW1, hW2⟩ := exists_nonposOn L K hK1 hK2 d hM
  obtain ⟨W', hW1', hW2'⟩ := exists_nonposOn L' K' hK1' hK2' d' hM'
  have a1 := finrank_le_card_neg L' K' hK1' hK2' d' hM' V hV2
  have a2 := finrank_le_card_neg L K hK1 hK2 d hM V' hV2'
  have a3 := finrank_le_card_nonpos L' K' hK1' hK2' d' hM' W hW2
  have a4 := finrank_le_card_nonpos L K hK1 hK2 d hM W' hW2'
  constructor <;> omega

end Transport

end CertifyMathlib.Counting
