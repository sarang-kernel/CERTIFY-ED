import CertifyMathlib.Counting.Inertia

/-!
# Corollary 5 (completeness of a 'lowest' claim and the sorted pairing)

Source: `docs/proofs/cor5_pairing.md` in the code repository. Notation: `A_u` is the Hermitian
operator the enclosures and the count were computed with, `A_m` the declared model with
`|xᴴ (A_u - A_m) x| ≤ Δ ‖x‖²`.

The claimed values are `μ : ι → ℝ` with radii `r : ι → ℝ` (`ι` a finite type with `k` elements).
Hypothesis (H1) of the document is formalised as: a function `grp : ι → G` (the enclosure groups),
radii constant on each group, group hulls `[lo g, hi g]` that contain every member's interval and
are pairwise disjoint, and an injection `φ : ι → n` into the eigen-indices with
`|λ (φ j) - μ j| ≤ r j`. `injective_of_groups` builds the global injection from injections on the
groups (Step 1 of the document). (H2) is the lower half of Lemma 4 with `ν = k`:
`cntLt hA c ≤ k` at the cut `c = σ - β`. (H3) is `μ j + r j < c`.

Conclusions, in counting form (see `CertifyMathlib.Counting.Inertia` for the translation):

* `image_eq_of_match`: exactly `k` eigenvalues lie below `c`, and they are the matched ones;
* `cor5_le`: `#{l | μ l ≤ μ j} ≤ cntLe hA (μ j + r j)`;
* `cor5_lt`: `cntLt hA (μ j - r j) ≤ #{l | μ l < μ j}`;
* `cor5_sorted`: for claimed values sorted as `μ : Fin k → ℝ` (monotone), the `(j+1)`-th smallest
  eigenvalue lies in `[μ j - r j, μ j + r j]`, written as `j + 1 ≤ cntLe hA (μ j + r j)` and
  `cntLt hA (μ j - r j) ≤ j`;
* `cor5_model`: the same statements for the declared model, widened by `Δ`, and exactly `k`
  eigenvalues of the model below `σ - β - Δ` (Step 4).

The count-form proof does not need the exchange argument of the sorted-pairing lemma: it compares
the counts at each threshold directly. The document's remark that the pairing fails for
non-uniform radii shows up in the proofs as the place where `r l = r j` within a group is used.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]
variable {ι : Type*} [Fintype ι]

/-- An injective matching within a uniform radius `r` puts `#{l | μ l ≤ x}` eigenvalues at or
below `x + r`, for every `x`. -/
theorem count_le_cntLe_of_match {A : Matrix n n 𝕜} (hA : A.IsHermitian) (μ : ι → ℝ) (r : ℝ)
    (φ : ι → n) (hφ : Function.Injective φ) (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r)
    (x : ℝ) : (Finset.univ.filter fun l => μ l ≤ x).card ≤ cntLe hA (x + r) := by
  unfold cntLe
  rw [← Finset.card_image_of_injective _ hφ]
  apply Finset.card_le_card
  intro i hi
  rw [Finset.mem_image] at hi
  obtain ⟨l, hl, rfl⟩ := hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl ⊢
  have := (abs_le.mp (hrad l)).2
  linarith

/-- The matched eigenvalues lie below any cut above every enclosure: `k ≤ cntLt hA c`. -/
theorem card_le_cntLt_of_match {A : Matrix n n 𝕜} (hA : A.IsHermitian) (μ r : ι → ℝ)
    (φ : ι → n) (hφ : Function.Injective φ) (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r l)
    (c : ℝ) (hcut : ∀ l, μ l + r l < c) : Fintype.card ι ≤ cntLt hA c := by
  unfold cntLt
  rw [← Finset.card_univ, ← Finset.card_image_of_injective _ hφ]
  apply Finset.card_le_card
  intro i hi
  rw [Finset.mem_image] at hi
  obtain ⟨l, -, rfl⟩ := hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  have := (abs_le.mp (hrad l)).2
  have := hcut l
  linarith

/-- **Corollary 5, Steps 1 to 2.** With (H2) `cntLt hA c ≤ k` and (H3), exactly `k` eigenvalues
lie below `c` (so `λ_{k+1} ≥ c`), and they are exactly the matched ones. -/
theorem image_eq_of_match {A : Matrix n n 𝕜} (hA : A.IsHermitian) (μ r : ι → ℝ) (φ : ι → n)
    (hφ : Function.Injective φ) (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r l) (c : ℝ)
    (hcut : ∀ l, μ l + r l < c) (hcount : cntLt hA c ≤ Fintype.card ι) :
    Finset.univ.image φ = (Finset.univ.filter fun i => hA.eigenvalues i < c) ∧
    cntLt hA c = Fintype.card ι := by
  have hsub : Finset.univ.image φ ⊆ Finset.univ.filter fun i => hA.eigenvalues i < c := by
    intro i hi
    rw [Finset.mem_image] at hi
    obtain ⟨l, -, rfl⟩ := hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have := (abs_le.mp (hrad l)).2
    have := hcut l
    linarith
  have hcard : (Finset.univ.image φ).card = Fintype.card ι := by
    rw [Finset.card_image_of_injective _ hφ, Finset.card_univ]
  have heq := Finset.eq_of_subset_of_card_le hsub (by unfold cntLt at hcount; omega)
  refine ⟨heq, ?_⟩
  unfold cntLt
  rw [← heq, hcard]

/-- Uniform-radius lower count: every eigenvalue below `x - r ≤ c` is a matched one, so
`cntLt hA (x - r) ≤ #{l | μ l < x}`. -/
theorem cntLt_le_count_of_match {A : Matrix n n 𝕜} (hA : A.IsHermitian) (μ : ι → ℝ) (r : ℝ)
    (φ : ι → n) (hφ : Function.Injective φ) (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r)
    (c : ℝ) (hcut : ∀ l, μ l + r < c) (hcount : cntLt hA c ≤ Fintype.card ι) (x : ℝ)
    (hx : x - r ≤ c) : cntLt hA (x - r) ≤ (Finset.univ.filter fun l => μ l < x).card := by
  obtain ⟨heq, -⟩ := image_eq_of_match hA μ (fun _ => r) φ hφ hrad c hcut hcount
  unfold cntLt
  calc (Finset.univ.filter fun i => hA.eigenvalues i < x - r).card
      ≤ ((Finset.univ.filter fun l => μ l < x).image φ).card := by
        apply Finset.card_le_card
        intro i hi
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi
        have hic : i ∈ Finset.univ.filter fun i => hA.eigenvalues i < c := by
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          linarith
        rw [← heq, Finset.mem_image] at hic
        obtain ⟨l, -, rfl⟩ := hic
        rw [Finset.mem_image]
        refine ⟨l, ?_, rfl⟩
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        have := (abs_le.mp (hrad l)).1
        linarith
    _ ≤ (Finset.univ.filter fun l => μ l < x).card := Finset.card_image_le

section Groups

variable {G : Type*} {A : Matrix n n 𝕜} (hA : A.IsHermitian) (μ r : ι → ℝ) (φ : ι → n)
  (grp : ι → G) (lo hi : G → ℝ)

omit [Fintype ι] in
/-- **Corollary 5, Step 1.** Injections on the groups combine into one injection when the group
hulls are disjoint. -/
theorem injective_of_groups (hgrp : ∀ j l, grp j = grp l → φ j = φ l → j = l)
    (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r l)
    (hlo : ∀ j, lo (grp j) ≤ μ j - r j) (hhi : ∀ j, μ j + r j ≤ hi (grp j))
    (hdisj : ∀ g g', g ≠ g' → hi g < lo g' ∨ hi g' < lo g) : Function.Injective φ := by
  intro j l h
  by_cases hg : grp j = grp l
  · exact hgrp j l hg h
  · exfalso
    have hj := abs_le.mp (hrad j)
    have hl := abs_le.mp (hrad l)
    rw [h] at hj
    have := hlo j
    have := hhi j
    have := hlo l
    have := hhi l
    rcases hdisj (grp j) (grp l) hg with h' | h' <;> linarith [hj.1, hj.2, hl.1, hl.2]

variable (hφ : Function.Injective φ) (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r l)
  (hr : ∀ j l, grp j = grp l → r j = r l)
  (hlo : ∀ j, lo (grp j) ≤ μ j - r j) (hhi : ∀ j, μ j + r j ≤ hi (grp j))
  (hdisj : ∀ g g', g ≠ g' → hi g < lo g' ∨ hi g' < lo g)
include hφ hrad hr hlo hhi hdisj

/-- **Corollary 5, upper half of the pairing**: `#{l | μ l ≤ μ j} ≤ cntLe hA (μ j + r j)`. -/
theorem cor5_le (j : ι) :
    (Finset.univ.filter fun l => μ l ≤ μ j).card ≤ cntLe hA (μ j + r j) := by
  have hr0 : ∀ l, 0 ≤ r l := fun l => (abs_nonneg _).trans (hrad l)
  unfold cntLe
  rw [← Finset.card_image_of_injective _ hφ]
  apply Finset.card_le_card
  intro i hi
  rw [Finset.mem_image] at hi
  obtain ⟨l, hl, rfl⟩ := hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl ⊢
  have h1 := (abs_le.mp (hrad l)).2
  by_cases hg : grp l = grp j
  · have := hr l j hg
    linarith
  · rcases hdisj (grp l) (grp j) hg with h | h
    · have := hhi l
      have := hlo j
      have := hr0 j
      linarith
    · have := hhi j
      have := hlo l
      have := hr0 j
      have := hr0 l
      linarith

/-- **Corollary 5, lower half of the pairing**: `cntLt hA (μ j - r j) ≤ #{l | μ l < μ j}`. -/
theorem cor5_lt (c : ℝ) (hcut : ∀ l, μ l + r l < c) (hcount : cntLt hA c ≤ Fintype.card ι)
    (j : ι) : cntLt hA (μ j - r j) ≤ (Finset.univ.filter fun l => μ l < μ j).card := by
  have hr0 : ∀ l, 0 ≤ r l := fun l => (abs_nonneg _).trans (hrad l)
  obtain ⟨heq, -⟩ := image_eq_of_match hA μ r φ hφ hrad c hcut hcount
  unfold cntLt
  calc (Finset.univ.filter fun i => hA.eigenvalues i < μ j - r j).card
      ≤ ((Finset.univ.filter fun l => μ l < μ j).image φ).card := by
        apply Finset.card_le_card
        intro i hi
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi
        have hic : i ∈ Finset.univ.filter fun i => hA.eigenvalues i < c := by
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          have := hcut j
          have := hr0 j
          linarith
        rw [← heq, Finset.mem_image] at hic
        obtain ⟨l, -, rfl⟩ := hic
        rw [Finset.mem_image]
        refine ⟨l, ?_, rfl⟩
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        have h1 := (abs_le.mp (hrad l)).1
        by_cases hg : grp l = grp j
        · have := hr l j hg
          linarith
        · rcases hdisj (grp l) (grp j) hg with h | h
          · have := hhi l
            have := hlo j
            have := hr0 j
            have := hr0 l
            linarith
          · have := hhi j
            have := hlo l
            have := hr0 j
            linarith
    _ ≤ (Finset.univ.filter fun l => μ l < μ j).card := Finset.card_image_le

end Groups

/-- **Corollary 5, sorted form.** For sorted claimed values `μ 0 ≤ μ 1 ≤ …` the `(j+1)`-th
smallest eigenvalue lies in `[μ j - r j, μ j + r j]`: `j + 1 ≤ cntLe hA (μ j + r j)` and
`cntLt hA (μ j - r j) ≤ j`. -/
theorem cor5_sorted {k : ℕ} {G : Type*} {A : Matrix n n 𝕜} (hA : A.IsHermitian)
    (μ r : Fin k → ℝ) (hμ : Monotone μ) (φ : Fin k → n) (grp : Fin k → G) (lo hi : G → ℝ)
    (hφ : Function.Injective φ) (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r l)
    (hr : ∀ j l, grp j = grp l → r j = r l)
    (hlo : ∀ j, lo (grp j) ≤ μ j - r j) (hhi : ∀ j, μ j + r j ≤ hi (grp j))
    (hdisj : ∀ g g', g ≠ g' → hi g < lo g' ∨ hi g' < lo g)
    (c : ℝ) (hcut : ∀ l, μ l + r l < c) (hcount : cntLt hA c ≤ k) (j : Fin k) :
    (j : ℕ) + 1 ≤ cntLe hA (μ j + r j) ∧ cntLt hA (μ j - r j) ≤ (j : ℕ) := by
  constructor
  · have hsub : Finset.Iic j ⊆ Finset.univ.filter fun l => μ l ≤ μ j := by
      intro l hl
      simp only [Finset.mem_Iic] at hl
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      exact hμ hl
    have h1 := Finset.card_le_card hsub
    rw [Fin.card_Iic] at h1
    exact h1.trans (cor5_le hA μ r φ grp lo hi hφ hrad hr hlo hhi hdisj j)
  · have hsub : (Finset.univ.filter fun l => μ l < μ j) ⊆ Finset.Iio j := by
      intro l hl
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl
      simp only [Finset.mem_Iio]
      by_contra h
      exact absurd (hμ (not_lt.mp h)) (not_le.mpr hl)
    have h1 := Finset.card_le_card hsub
    rw [Fin.card_Iio] at h1
    have hcount' : cntLt hA c ≤ Fintype.card (Fin k) := by simpa using hcount
    exact (cor5_lt hA μ r φ grp lo hi hφ hrad hr hlo hhi hdisj c hcut hcount' j).trans h1

/-- **Corollary 5, Step 4 (transfer to the declared model).** With `|xᴴ (A_u - A_m) x| ≤ Δ ‖x‖²`
and (H3) in the document's form `μ j + r j + Δ < σ - β - Δ`: exactly `k` eigenvalues of `A_m` lie
below `σ - β - Δ`, and the pairing holds for `A_m` with radii `r j + Δ`. -/
theorem cor5_model {G : Type*} {Au Am : Matrix n n 𝕜} (hAu : Au.IsHermitian)
    (hAm : Am.IsHermitian) (Δ : ℝ) (hΔ0 : 0 ≤ Δ) (hΔ : ∀ x, |qf (Au - Am) x| ≤ Δ * nsq x)
    (μ r : ι → ℝ) (φ : ι → n) (grp : ι → G) (lo hi : G → ℝ)
    (hφ : Function.Injective φ) (hrad : ∀ l, |hAu.eigenvalues (φ l) - μ l| ≤ r l)
    (hr : ∀ j l, grp j = grp l → r j = r l)
    (hlo : ∀ j, lo (grp j) ≤ μ j - r j) (hhi : ∀ j, μ j + r j ≤ hi (grp j))
    (hdisj : ∀ g g', g ≠ g' → hi g < lo g' ∨ hi g' < lo g)
    (σ β : ℝ) (hcut : ∀ l, μ l + r l + Δ < σ - β - Δ)
    (hcount : cntLt hAu (σ - β) ≤ Fintype.card ι) :
    cntLt hAm (σ - β - Δ) = Fintype.card ι ∧
    (∀ j, (Finset.univ.filter fun l => μ l ≤ μ j).card ≤ cntLe hAm (μ j + r j + Δ)) ∧
    (∀ j, cntLt hAm (μ j - r j - Δ) ≤ (Finset.univ.filter fun l => μ l < μ j).card) := by
  have hΔ' : ∀ x, |qf (Am - Au) x| ≤ Δ * nsq x := fun x => by
    rw [abs_qf_sub_comm]
    exact hΔ x
  have hcut1 : ∀ l, μ l + r l < σ - β := fun l => by
    have := hcut l
    linarith
  have hcut2 : ∀ l, μ l + r l < σ - β - 2 * Δ := fun l => by
    have := hcut l
    linarith
  refine ⟨le_antisymm ?_ ?_, fun j => ?_, fun j => ?_⟩
  · have h1 := weyl_cntLt hAm hAu Δ hΔ' (σ - β - Δ)
    rw [show σ - β - Δ + Δ = σ - β by ring] at h1
    exact h1.trans hcount
  · have h1 := card_le_cntLt_of_match hAu μ r φ hφ hrad _ hcut2
    have h2 := weyl_cntLt hAu hAm Δ hΔ (σ - β - 2 * Δ)
    rw [show σ - β - 2 * Δ + Δ = σ - β - Δ by ring] at h2
    exact h1.trans h2
  · exact (cor5_le hAu μ r φ grp lo hi hφ hrad hr hlo hhi hdisj j).trans
      (weyl_cntLe hAu hAm Δ hΔ _)
  · have h1 := weyl_cntLt hAm hAu Δ hΔ' (μ j - r j - Δ)
    rw [show μ j - r j - Δ + Δ = μ j - r j by ring] at h1
    exact h1.trans (cor5_lt hAu μ r φ grp lo hi hφ hrad hr hlo hhi hdisj (σ - β) hcut1 hcount j)

end CertifyMathlib.Counting
