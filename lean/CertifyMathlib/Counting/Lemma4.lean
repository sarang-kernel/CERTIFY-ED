import CertifyMathlib.Counting.Inertia

/-!
# Lemma 4 (inertia with backward error), the D-01 row-sum bound and Proposition 14

`docs/theory.md`, section 3, Lemma 4: if `‖L D Lᴴ - P(A - σI)Pᵀ‖ ≤ β` for unit lower-triangular `L`
and block-diagonal `D` with `ν` negative eigenvalues, then `#{λ < σ - β} ≤ ν ≤ #{λ ≤ σ + β}`.

The Lean statements take the backward error `E = L D Lᴴ - P(A - σI)Pᴴ` in the form of a bound on its
quadratic form, `|xᴴ E x| ≤ β ‖x‖²` for every `x`; this follows from `‖E‖₂ ≤ β` and, for Hermitian
`E`, from the row-sum bound `‖E‖_∞ ≤ β` (`abs_qf_le_rowSum`, the exact-arithmetic step of D-01).
Only invertibility of `L` is used (unit triangularity is one way to have it), and the permutation
`P` may be any unitary matrix.

* `lemma4`: the core statement for diagonal `D`, with the strict upper count
  `ν ≤ #{λ < σ + β}`, which implies the document's `ν ≤ #{λ ≤ σ + β}` (`lemma4_doc`);
* `lemma4_hermitian`: `D` any Hermitian matrix (for example block diagonal with 1 × 1 and 2 × 2
  blocks), `ν` its number of negative eigenvalues;
* `lemma4_perm`: the permuted form `L D Lᴴ = P (A - σ I) Pᴴ + E` with `P` unitary;
* `bracket_count` (Proposition 14): two counts at `σ_lo < σ_hi` bracket the number of eigenvalues
  in the inner and outer intervals.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

/-- **Lemma 4** (core form). `L D Lᴴ = A - σ I + E` with `D = diag(d)`, `Lᴴ` invertible with inverse
`K`, and `|xᴴ E x| ≤ β ‖x‖²`. Then `#{λ < σ - β} ≤ ν ≤ #{λ < σ + β}` with `ν = #{i | d i < 0}`. -/
theorem lemma4 {m : Type*} [Fintype m] [DecidableEq m] {A : Matrix n n 𝕜}
    (hA : A.IsHermitian) (σ β : ℝ) (L K : Matrix n m 𝕜) (hK1 : Lᴴ * K = 1)
    (hK2 : K * Lᴴ = 1) (d : m → ℝ) (E : Matrix n n 𝕜)
    (hfac : L * dg d * Lᴴ = A - (σ : 𝕜) • (1 : Matrix n n 𝕜) + E)
    (hE : ∀ x, |qf E x| ≤ β * nsq x) :
    cntLt hA (σ - β) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLt hA (σ + β) := by
  constructor
  · obtain ⟨V, hV1, hV2⟩ := exists_negOn_cntLt hA (σ - β)
    have hneg : NegOn (L * dg d * Lᴴ) V := by
      intro x hx hx0
      have h1 := hV2 x hx hx0
      rw [hfac, qf_add, qf_sub_smul_one]
      rw [qf_sub_smul_one] at h1
      have h2 := (abs_le.mp (hE x)).2
      have h3 : (σ - β) * nsq x = σ * nsq x - β * nsq x := sub_mul _ _ _
      linarith
    exact hV1.trans (finrank_le_card_neg L K hK1 hK2 d rfl V hneg)
  · obtain ⟨V, hV1, hV2⟩ := exists_negOn L K hK1 hK2 d rfl
    have hneg : NegOn (A - ((σ + β : ℝ) : 𝕜) • (1 : Matrix n n 𝕜)) V := by
      intro x hx hx0
      have h1 := hV2 x hx hx0
      rw [hfac, qf_add, qf_sub_smul_one] at h1
      rw [qf_sub_smul_one]
      have h2 := (abs_le.mp (hE x)).1
      have h3 : (σ + β) * nsq x = σ * nsq x + β * nsq x := add_mul _ _ _
      linarith
    exact hV1.trans (finrank_le_cntLt hA _ V hneg)

/-- **Lemma 4** in the form of `docs/theory.md` (square invertible `L`, diagonal `D`):
`#{λ < σ - β} ≤ ν ≤ #{λ ≤ σ + β}`. -/
theorem lemma4_doc {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σ β : ℝ) (L : Matrix n n 𝕜)
    (hL : IsUnit L) (d : n → ℝ) (E : Matrix n n 𝕜)
    (hfac : L * dg d * Lᴴ = A - (σ : 𝕜) • (1 : Matrix n n 𝕜) + E)
    (hE : ∀ x, |qf E x| ≤ β * nsq x) :
    cntLt hA (σ - β) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLe hA (σ + β) := by
  have hdet := isUnit_det_conjTranspose hL
  obtain ⟨h1, h2⟩ := lemma4 hA σ β L (Lᴴ)⁻¹ (mul_nonsing_inv _ hdet) (nonsing_inv_mul _ hdet) d E
    hfac hE
  exact ⟨h1, h2.trans (cntLt_le_cntLe hA _)⟩

/-- **Lemma 4 with a Hermitian `D`** (for example block diagonal with 1 × 1 and 2 × 2 blocks):
`ν` is the number of negative eigenvalues of `D`. -/
theorem lemma4_hermitian {m : Type*} [Fintype m] [DecidableEq m] {A : Matrix n n 𝕜}
    (hA : A.IsHermitian) (σ β : ℝ) (L K : Matrix n m 𝕜) (hK1 : Lᴴ * K = 1)
    (hK2 : K * Lᴴ = 1) {D : Matrix m m 𝕜} (hD : D.IsHermitian) (E : Matrix n n 𝕜)
    (hfac : L * D * Lᴴ = A - (σ : 𝕜) • (1 : Matrix n n 𝕜) + E)
    (hE : ∀ x, |qf E x| ≤ β * nsq x) :
    cntLt hA (σ - β) ≤ cntLt hD 0 ∧ cntLt hD 0 ≤ cntLt hA (σ + β) := by
  set U := (hD.eigenvectorUnitary : Matrix m m 𝕜) with hUdef
  have hUU : Uᴴ * U = 1 := conjTranspose_mul_unitary _
  have hUU' : U * Uᴴ = 1 := unitary_mul_conjTranspose _
  have hD0 : D = U * dg (fun i => hD.eigenvalues i - 0) * Uᴴ := by
    have h := sub_smul_one_eq hD 0
    rw [← hUdef] at h
    simpa using h
  have hfac' : (L * U) * dg (fun i => hD.eigenvalues i - 0) * (L * U)ᴴ =
      A - (σ : 𝕜) • (1 : Matrix n n 𝕜) + E := by
    rw [← hfac, conjTranspose_mul]
    conv_rhs => rw [hD0]
    simp only [Matrix.mul_assoc]
  have hK1' : (L * U)ᴴ * (K * U) = 1 := by
    rw [conjTranspose_mul]
    calc Uᴴ * Lᴴ * (K * U) = Uᴴ * (Lᴴ * K) * U := by simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hK1, Matrix.mul_one, hUU]
  have hK2' : (K * U) * (L * U)ᴴ = 1 := by
    rw [conjTranspose_mul]
    calc K * U * (Uᴴ * Lᴴ) = K * (U * Uᴴ) * Lᴴ := by simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hUU', Matrix.mul_one, hK2]
  have h := lemma4 hA σ β (L * U) (K * U) hK1' hK2' _ E hfac' hE
  simpa [cntLt] using h

/-- **Lemma 4, permuted form** `L D Lᴴ = P (A - σ I) Pᴴ + E` with `P` unitary (a permutation
matrix in the code). -/
theorem lemma4_perm {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σ β : ℝ) (P : Matrix n n 𝕜)
    (hP1 : Pᴴ * P = 1) (hP2 : P * Pᴴ = 1) (L : Matrix n n 𝕜) (hL : IsUnit L) (d : n → ℝ)
    (E : Matrix n n 𝕜) (hfac : L * dg d * Lᴴ = P * (A - (σ : 𝕜) • (1 : Matrix n n 𝕜)) * Pᴴ + E)
    (hE : ∀ x, |qf E x| ≤ β * nsq x) :
    cntLt hA (σ - β) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLe hA (σ + β) := by
  have hA' : (P * A * Pᴴ).IsHermitian := by
    have := isHermitian_conj hA Pᴴ
    rwa [conjTranspose_conjTranspose] at this
  have hc : ∀ t, cntLt hA' t = cntLt hA t ∧ cntLe hA' t = cntLe hA t := fun t =>
    cnt_conj hA Pᴴ (by rw [conjTranspose_conjTranspose]; exact hP2) (by
      rw [conjTranspose_conjTranspose]; exact hP1) hA' (by rw [conjTranspose_conjTranspose]) t
  have hfac' : L * dg d * Lᴴ = P * A * Pᴴ - (σ : 𝕜) • (1 : Matrix n n 𝕜) + E := by
    rw [hfac, Matrix.mul_sub, Matrix.sub_mul, mul_smul_comm, smul_mul_assoc, Matrix.mul_one, hP2]
  have h := lemma4_doc hA' σ β L hL d E hfac' hE
  rw [(hc _).1, (hc _).2] at h
  exact h

omit [DecidableEq n] in
/-- **D-01, exact-arithmetic step (Lemma R1).** For Hermitian `E` with every absolute row sum at
most `R` (so `‖E‖_∞ ≤ R`), `|xᴴ E x| ≤ R ‖x‖²`, hence `‖E‖₂ ≤ ‖E‖_∞`. -/
theorem abs_qf_le_rowSum (E : Matrix n n 𝕜) (hE : E.IsHermitian) (R : ℝ)
    (hR : ∀ i, ∑ j, ‖E i j‖ ≤ R) (x : n → 𝕜) : |qf E x| ≤ R * nsq x := by
  have hsym : ∀ i j, ‖E i j‖ = ‖E j i‖ := by
    intro i j
    rw [← hE.apply i j]
    exact norm_star _
  have h1 : |qf E x| ≤ ∑ i, ∑ j, ‖x i‖ * (‖E i j‖ * ‖x j‖) := by
    unfold qf
    refine (RCLike.abs_re_le_norm _).trans ?_
    simp only [dotProduct, mulVec, Pi.star_apply]
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun i _ => ?_)
    rw [Finset.mul_sum]
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun j _ => ?_)
    exact le_of_eq (by rw [norm_mul, norm_mul, norm_star])
  have h3 : ∑ i, ∑ j, ‖E i j‖ * ‖x j‖ ^ 2 = ∑ i, ∑ j, ‖E i j‖ * ‖x i‖ ^ 2 := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [hsym]
  have h4 : ∑ i, ∑ j, ‖E i j‖ * ‖x i‖ ^ 2 ≤ R * nsq x := by
    unfold nsq
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    rw [← Finset.sum_mul]
    exact mul_le_mul_of_nonneg_right (hR i) (sq_nonneg _)
  calc |qf E x| ≤ ∑ i, ∑ j, ‖x i‖ * (‖E i j‖ * ‖x j‖) := h1
    _ ≤ ∑ i, ∑ j, (‖E i j‖ * ‖x i‖ ^ 2 + ‖E i j‖ * ‖x j‖ ^ 2) / 2 :=
        Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => by
          have := two_mul_le_add_sq ‖x i‖ ‖x j‖
          have hE0 := norm_nonneg (E i j)
          nlinarith
    _ = (∑ i, ∑ j, ‖E i j‖ * ‖x i‖ ^ 2 + ∑ i, ∑ j, ‖E i j‖ * ‖x j‖ ^ 2) / 2 := by
        rw [← Finset.sum_add_distrib, Finset.sum_div]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [← Finset.sum_add_distrib, Finset.sum_div]
    _ ≤ R * nsq x := by
        rw [h3]
        linarith

/-- **D-01, Step 5.** If the stored shifted matrix `S` differs from `A - σ I` by a diagonal
rounding `Δ` with `|Δ i| ≤ s`, and `L diag(d) Lᴴ - S = E` with Hermitian `E` of absolute row sums
at most `R`, then Lemma 4 holds with `β = R + s`. -/
theorem lemma4_rowSum {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σ R s : ℝ) (L : Matrix n n 𝕜)
    (hL : IsUnit L) (d : n → ℝ) (Δ : n → ℝ) (hΔ : ∀ i, |Δ i| ≤ s) (E : Matrix n n 𝕜)
    (hEh : E.IsHermitian) (hR : ∀ i, ∑ j, ‖E i j‖ ≤ R)
    (hfac : L * dg d * Lᴴ = (A - (σ : 𝕜) • (1 : Matrix n n 𝕜) + dg Δ) + E) :
    cntLt hA (σ - (R + s)) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLe hA (σ + (R + s)) := by
  refine lemma4_doc hA σ (R + s) L hL d (dg Δ + E) (by rw [hfac, add_assoc]) ?_
  intro x
  rw [qf_add, qf_dg, add_mul]
  have hE := abs_qf_le_rowSum E hEh R hR x
  have hD : |∑ i, Δ i * ‖x i‖ ^ 2| ≤ s * nsq x := by
    unfold nsq
    rw [Finset.mul_sum]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
    rw [abs_mul, abs_of_nonneg (sq_nonneg ‖x i‖)]
    exact mul_le_mul_of_nonneg_right (hΔ i) (sq_nonneg _)
  calc |∑ i, Δ i * ‖x i‖ ^ 2 + qf E x| ≤ |∑ i, Δ i * ‖x i‖ ^ 2| + |qf E x| := abs_add_le _ _
    _ ≤ R * nsq x + s * nsq x := by linarith

/-- The number of eigenvalues in the open interval `(a, b)`, `a < b`, plus `cntLe hA a`, is at most
`cntLt hA b`. -/
theorem card_Ioo_add_cntLe {A : Matrix n n 𝕜} (hA : A.IsHermitian) {a b : ℝ} (hab : a < b) :
    (Finset.univ.filter fun i => a < hA.eigenvalues i ∧ hA.eigenvalues i < b).card +
      cntLe hA a ≤ cntLt hA b := by
  unfold cntLe cntLt
  have h := Finset.card_filter_add_card_filter_not
    (s := Finset.univ.filter fun i => hA.eigenvalues i < b) (fun i => a < hA.eigenvalues i)
  have h1 : (Finset.univ.filter fun i => a < hA.eigenvalues i ∧ hA.eigenvalues i < b).card ≤
      ((Finset.univ.filter fun i => hA.eigenvalues i < b).filter
        fun i => a < hA.eigenvalues i).card := by
    apply Finset.card_le_card
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
    exact ⟨hi.2, hi.1⟩
  have h2 : (Finset.univ.filter fun i => hA.eigenvalues i ≤ a).card ≤
      ((Finset.univ.filter fun i => hA.eigenvalues i < b).filter
        fun i => ¬ a < hA.eigenvalues i).card := by
    apply Finset.card_le_card
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
    exact ⟨by linarith, not_lt.mpr hi⟩
  omega

/-- `cntLe hA b ≤ cntLt hA a + #{a ≤ λ ≤ b}`. -/
theorem cntLe_le_cntLt_add {A : Matrix n n 𝕜} (hA : A.IsHermitian) (a b : ℝ) :
    cntLe hA b ≤ cntLt hA a +
      (Finset.univ.filter fun i => a ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ b).card := by
  unfold cntLe cntLt
  refine (Finset.card_le_card ?_).trans (Finset.card_union_le _ _)
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at hi ⊢
  by_cases h : hA.eigenvalues i < a
  · exact Or.inl h
  · exact Or.inr ⟨not_lt.mp h, hi⟩

/-- **Proposition 14 (bracketing count).** With the two Lemma 4 sandwiches at `σ_lo` and `σ_hi`,
`I_in = (σ_lo + β_lo, σ_hi - β_hi)` (assumed non-empty) and
`I_out = [σ_lo - β_lo, σ_hi + β_hi]`: `#{λ ∈ I_in} ≤ ν_hi - ν_lo ≤ #{λ ∈ I_out}`, written without
subtraction. -/
theorem bracket_count {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σlo βlo σhi βhi : ℝ)
    (νlo νhi : ℕ) (hin : σlo + βlo < σhi - βhi)
    (hlo : cntLt hA (σlo - βlo) ≤ νlo ∧ νlo ≤ cntLe hA (σlo + βlo))
    (hhi : cntLt hA (σhi - βhi) ≤ νhi ∧ νhi ≤ cntLe hA (σhi + βhi)) :
    (Finset.univ.filter fun i =>
        σlo + βlo < hA.eigenvalues i ∧ hA.eigenvalues i < σhi - βhi).card + νlo ≤ νhi ∧
    νhi ≤ νlo + (Finset.univ.filter fun i =>
        σlo - βlo ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ σhi + βhi).card := by
  have h1 := card_Ioo_add_cntLe hA hin
  have h2 := cntLe_le_cntLt_add hA (σlo - βlo) (σhi + βhi)
  omega

/-- **Proposition 14, exact count.** If `g` distinct eigen-indices have eigenvalues in `I_in` and
`ν_hi - ν_lo = g`, then `I_in` contains exactly `g` eigenvalues. -/
theorem bracket_exact {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σlo βlo σhi βhi : ℝ)
    (νlo νhi : ℕ) (hin : σlo + βlo < σhi - βhi)
    (hlo : cntLt hA (σlo - βlo) ≤ νlo ∧ νlo ≤ cntLe hA (σlo + βlo))
    (hhi : cntLt hA (σhi - βhi) ≤ νhi ∧ νhi ≤ cntLe hA (σhi + βhi))
    {g : ℕ} (ψ : Fin g → n) (hψ : Function.Injective ψ)
    (hψin : ∀ k, σlo + βlo < hA.eigenvalues (ψ k) ∧ hA.eigenvalues (ψ k) < σhi - βhi)
    (hν : νhi = νlo + g) :
    (Finset.univ.filter fun i =>
        σlo + βlo < hA.eigenvalues i ∧ hA.eigenvalues i < σhi - βhi).card = g := by
  have hge : g ≤ (Finset.univ.filter fun i =>
      σlo + βlo < hA.eigenvalues i ∧ hA.eigenvalues i < σhi - βhi).card := by
    have : (Finset.univ.image ψ).card = g := by
      rw [Finset.card_image_of_injective _ hψ, Finset.card_univ, Fintype.card_fin]
    rw [← this]
    apply Finset.card_le_card
    intro i hi
    rw [Finset.mem_image] at hi
    obtain ⟨k, -, rfl⟩ := hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact hψin k
  have := (bracket_count hA σlo βlo σhi βhi νlo νhi hin hlo hhi).1
  omega

/-- **Proposition 14 (e).** If `g` distinct eigen-indices have eigenvalues in `I_in`, then
`ν_hi - ν_lo ≥ g`; a smaller difference shows that a hypothesis is false. -/
theorem bracket_ge {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σlo βlo σhi βhi : ℝ)
    (νlo νhi : ℕ) (hin : σlo + βlo < σhi - βhi)
    (hlo : cntLt hA (σlo - βlo) ≤ νlo ∧ νlo ≤ cntLe hA (σlo + βlo))
    (hhi : cntLt hA (σhi - βhi) ≤ νhi ∧ νhi ≤ cntLe hA (σhi + βhi))
    {g : ℕ} (ψ : Fin g → n) (hψ : Function.Injective ψ)
    (hψin : ∀ k, σlo + βlo < hA.eigenvalues (ψ k) ∧ hA.eigenvalues (ψ k) < σhi - βhi) :
    νlo + g ≤ νhi := by
  have hge : g ≤ (Finset.univ.filter fun i =>
      σlo + βlo < hA.eigenvalues i ∧ hA.eigenvalues i < σhi - βhi).card := by
    have : (Finset.univ.image ψ).card = g := by
      rw [Finset.card_image_of_injective _ hψ, Finset.card_univ, Fintype.card_fin]
    rw [← this]
    apply Finset.card_le_card
    intro i hi
    rw [Finset.mem_image] at hi
    obtain ⟨k, -, rfl⟩ := hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact hψin k
  have := (bracket_count hA σlo βlo σhi βhi νlo νhi hin hlo hhi).1
  omega

/-- **Exact count of a 2 × 2 pivot block** (Lemma 4, `ν` of a block-diagonal `D`): a Hermitian
2 × 2 block with negative determinant has exactly one negative eigenvalue; with positive
determinant it has two or none, according to the sign of the trace. With
`cnt_blockDiagonal'` (`CertifyMathlib.Counting.Assembly`) the count of a block-diagonal `D` is the
sum of these block counts. -/
theorem cntLt_fin_two {D : Matrix (Fin 2) (Fin 2) 𝕜} (hD : D.IsHermitian) :
    (RCLike.re D.det < 0 → cntLt hD 0 = 1) ∧
    (0 < RCLike.re D.det → RCLike.re D.trace < 0 → cntLt hD 0 = 2) ∧
    (0 < RCLike.re D.det → 0 < RCLike.re D.trace → cntLt hD 0 = 0) := by
  have hdet : RCLike.re D.det = hD.eigenvalues 0 * hD.eigenvalues 1 := by
    rw [hD.det_eq_prod_eigenvalues, Fin.prod_univ_two, ← RCLike.ofReal_mul, RCLike.ofReal_re]
  have htr : RCLike.re D.trace = hD.eigenvalues 0 + hD.eigenvalues 1 := by
    rw [hD.trace_eq_sum_eigenvalues, Fin.sum_univ_two, ← RCLike.ofReal_add, RCLike.ofReal_re]
  have hcnt : cntLt hD 0 = (if hD.eigenvalues 0 < 0 then 1 else 0) +
      (if hD.eigenvalues 1 < 0 then 1 else 0) := by
    rw [cntLt, Finset.card_filter, Fin.sum_univ_two]
  rw [hcnt, hdet, htr]
  refine ⟨fun h => ?_, fun h1 h2 => ?_, fun h1 h2 => ?_⟩
  · rcases mul_neg_iff.mp h with ⟨ha, hb⟩ | ⟨ha, hb⟩
    · simp [not_lt.mpr ha.le, hb]
    · simp [ha, not_lt.mpr hb.le]
  · have h3 : hD.eigenvalues 0 < 0 ∧ hD.eigenvalues 1 < 0 := by
      rcases mul_pos_iff.mp h1 with ⟨ha, hb⟩ | ⟨ha, hb⟩
      · exact absurd h2 (not_lt.mpr (by linarith))
      · exact ⟨ha, hb⟩
    simp [h3.1, h3.2]
  · have h3 : 0 < hD.eigenvalues 0 ∧ 0 < hD.eigenvalues 1 := by
      rcases mul_pos_iff.mp h1 with ⟨ha, hb⟩ | ⟨ha, hb⟩
      · exact ⟨ha, hb⟩
      · exact absurd h2 (not_lt.mpr (by linarith))
    simp [not_lt.mpr h3.1.le, not_lt.mpr h3.2.le]

end CertifyMathlib.Counting
