import CertifyMathlib.Counting.Compose

/-!
# Sorted-index forms of the eigenvalue statements

`eigAsc hA j` is the `(j+1)`-th smallest eigenvalue of a Hermitian matrix `A`, with multiplicity
(`eigenvalues₀` of Mathlib lists the eigenvalues in decreasing order; `eigAsc` reverses it).
The two equivalences `eigAsc_le_iff` and `le_eigAsc_iff` translate the counting statements of the
other modules into statements about the sorted eigenvalues:
`λ_(j+1) ≤ t ↔ j + 1 ≤ #{i | λ_i ≤ t}` and `t ≤ λ_(j+1) ↔ #{i | λ_i < t} ≤ j`.
Sorted forms follow for Weyl's inequality (quadratic-form and spectral-norm hypotheses, and E-04
along an affine path), Corollary 5 and its transfer to the declared model, and Theorem 8
(with a unitary `Q`, with spectral-norm hypotheses, and with the polar-factor lemma).
-/

namespace CertifyMathlib.Counting

open Matrix

section Combinatorics

variable {K : ℕ} {μ : Fin K → ℝ}

/-- If `μ` is monotone and `μ j ≤ t`, at least `j + 1` values are `≤ t`. -/
theorem succ_le_card_le (hμ : Monotone μ) (j : Fin K) {t : ℝ} (ht : μ j ≤ t) :
    (j : ℕ) + 1 ≤ (Finset.univ.filter fun l => μ l ≤ t).card := by
  have hsub : Finset.Iic j ⊆ Finset.univ.filter fun l => μ l ≤ t := by
    intro l hl
    rw [Finset.mem_Iic] at hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact (hμ hl).trans ht
  calc (j : ℕ) + 1 = (Finset.Iic j).card := (Fin.card_Iic (b := j)).symm
    _ ≤ _ := Finset.card_le_card hsub

/-- If `μ` is monotone and `t ≤ μ j`, at most `j` values are `< t`. -/
theorem card_lt_le (hμ : Monotone μ) (j : Fin K) {t : ℝ} (ht : t ≤ μ j) :
    (Finset.univ.filter fun l => μ l < t).card ≤ j := by
  have hsub : (Finset.univ.filter fun l => μ l < t) ⊆ Finset.Iio j := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl
    exact Finset.mem_Iio.mpr (not_le.mp fun h => by linarith [hμ h])
  calc _ ≤ (Finset.Iio j).card := Finset.card_le_card hsub
    _ = j := Fin.card_Iio (b := j)

theorem le_of_succ_le_card (hμ : Monotone μ) (j : Fin K) {t : ℝ}
    (h : (j : ℕ) + 1 ≤ (Finset.univ.filter fun l => μ l ≤ t).card) : μ j ≤ t := by
  by_contra hjt
  rw [not_le] at hjt
  have hsub : (Finset.univ.filter fun l => μ l ≤ t) ⊆ Finset.Iio j := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl
    exact Finset.mem_Iio.mpr (not_le.mp fun h' => by linarith [hμ h'])
  have h2 : (Finset.univ.filter fun l => μ l ≤ t).card ≤ j :=
    (Finset.card_le_card hsub).trans (Fin.card_Iio (b := j)).le
  omega

theorem le_of_card_lt_le (hμ : Monotone μ) (j : Fin K) {t : ℝ}
    (h : (Finset.univ.filter fun l => μ l < t).card ≤ j) : t ≤ μ j := by
  by_contra hjt
  rw [not_le] at hjt
  have h2 := succ_le_card_le hμ j (le_refl (μ j))
  have hsub : (Finset.univ.filter fun l => μ l ≤ μ j) ⊆ Finset.univ.filter fun l => μ l < t := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl ⊢
    exact lt_of_le_of_lt hl hjt
  have h3 := Finset.card_le_card hsub
  omega

end Combinatorics

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

/-- The eigenvalues of a Hermitian matrix in ascending order: `eigAsc hA j` is the `(j+1)`-th
smallest eigenvalue, with multiplicity. -/
noncomputable def eigAsc {A : Matrix n n 𝕜} (hA : A.IsHermitian) (j : Fin (Fintype.card n)) :
    ℝ :=
  hA.eigenvalues₀ (Fin.rev j)

theorem eigAsc_monotone {A : Matrix n n 𝕜} (hA : A.IsHermitian) : Monotone (eigAsc hA) :=
  fun _ _ hij => hA.eigenvalues₀_antitone (Fin.rev_le_rev.mpr hij)

/-- The reindexing `Fin (card n) ≃ n` along which `eigAsc` lists `hA.eigenvalues`. -/
noncomputable def ascEquiv : Fin (Fintype.card n) ≃ n :=
  Fin.revPerm.trans (Fintype.equivOfCardEq (Fintype.card_fin _))

theorem eigAsc_eq {A : Matrix n n 𝕜} (hA : A.IsHermitian) (j : Fin (Fintype.card n)) :
    eigAsc hA j = hA.eigenvalues (ascEquiv j) := by
  simp [eigAsc, ascEquiv, Matrix.IsHermitian.eigenvalues]

theorem cntLe_eq {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) :
    cntLe hA t = (Finset.univ.filter fun j => eigAsc hA j ≤ t).card := by
  unfold cntLe
  symm
  exact Finset.card_equiv ascEquiv fun j => by simp [eigAsc_eq]

theorem cntLt_eq {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) :
    cntLt hA t = (Finset.univ.filter fun j => eigAsc hA j < t).card := by
  unfold cntLt
  symm
  exact Finset.card_equiv ascEquiv fun j => by simp [eigAsc_eq]

theorem cntLe_le_card {A : Matrix n n 𝕜} (hA : A.IsHermitian) (t : ℝ) :
    cntLe hA t ≤ Fintype.card n := by
  unfold cntLe
  exact Finset.card_le_univ _

/-- `λ_(j+1) ≤ t ↔ j + 1 ≤ #{i | λ_i ≤ t}`. -/
theorem eigAsc_le_iff {A : Matrix n n 𝕜} (hA : A.IsHermitian) (j : Fin (Fintype.card n))
    (t : ℝ) : eigAsc hA j ≤ t ↔ (j : ℕ) + 1 ≤ cntLe hA t := by
  rw [cntLe_eq]
  exact ⟨succ_le_card_le (eigAsc_monotone hA) j, le_of_succ_le_card (eigAsc_monotone hA) j⟩

/-- `t ≤ λ_(j+1) ↔ #{i | λ_i < t} ≤ j`. -/
theorem le_eigAsc_iff {A : Matrix n n 𝕜} (hA : A.IsHermitian) (j : Fin (Fintype.card n))
    (t : ℝ) : t ≤ eigAsc hA j ↔ cntLt hA t ≤ j := by
  rw [cntLt_eq]
  exact ⟨card_lt_le (eigAsc_monotone hA) j, le_of_card_lt_le (eigAsc_monotone hA) j⟩

/-- The two halves of a pairing in counting form give the sorted form. -/
theorem eigAsc_mem_of_counts {A : Matrix n n 𝕜} (hA : A.IsHermitian) {j : ℕ} {lo hi : ℝ}
    (h1 : j + 1 ≤ cntLe hA hi) (h2 : cntLt hA lo ≤ j) :
    ∃ hj : j < Fintype.card n, lo ≤ eigAsc hA ⟨j, hj⟩ ∧ eigAsc hA ⟨j, hj⟩ ≤ hi := by
  have hj : j < Fintype.card n := by
    have := cntLe_le_card hA hi
    omega
  exact ⟨hj, (le_eigAsc_iff hA ⟨j, hj⟩ lo).mpr h2, (eigAsc_le_iff hA ⟨j, hj⟩ hi).mpr h1⟩

/-- **Weyl's inequality, sorted form**: `|λ_j(A) - λ_j(B)| ≤ ρ` when
`|xᴴ (A - B) x| ≤ ρ ‖x‖²`. -/
theorem weyl_sorted {A B : Matrix n n 𝕜} (hA : A.IsHermitian) (hB : B.IsHermitian) (ρ : ℝ)
    (h : ∀ x, |qf (A - B) x| ≤ ρ * nsq x) (j : Fin (Fintype.card n)) :
    |eigAsc hA j - eigAsc hB j| ≤ ρ := by
  have h' : ∀ x, |qf (B - A) x| ≤ ρ * nsq x := fun x => by
    rw [abs_qf_sub_comm]
    exact h x
  have e1 : eigAsc hB j ≤ eigAsc hA j + ρ := (eigAsc_le_iff hB j _).mpr
    (((eigAsc_le_iff hA j _).mp le_rfl).trans (weyl_cntLe hA hB ρ h _))
  have e2 : eigAsc hA j ≤ eigAsc hB j + ρ := (eigAsc_le_iff hA j _).mpr
    (((eigAsc_le_iff hB j _).mp le_rfl).trans (weyl_cntLe hB hA ρ h' _))
  rw [abs_sub_le_iff]
  constructor <;> linarith

/-- **Weyl's inequality, sorted form**, with the hypothesis `‖A - B‖₂ ≤ ρ`. -/
theorem weyl_sorted_opNorm {A B : Matrix n n 𝕜} (hA : A.IsHermitian) (hB : B.IsHermitian)
    (ρ : ℝ) (h : ∀ z : EuclideanSpace 𝕜 n, ‖Matrix.toEuclideanLin (A - B) z‖ ≤ ρ * ‖z‖)
    (j : Fin (Fintype.card n)) : |eigAsc hA j - eigAsc hB j| ≤ ρ :=
  weyl_sorted hA hB ρ (abs_qf_le_of_toEuclideanLin _ ρ h) j

/-- **E-04, sorted form**: along `F = H₀ + s H₁` with `|xᴴ H₁ x| ≤ B ‖x‖²`,
`|λ_j(F) - λ_j(H₀)| ≤ B |s|`. -/
theorem weyl_affine_sorted {H₀ H₁ F : Matrix n n 𝕜} (h₀ : H₀.IsHermitian) (hF : F.IsHermitian)
    (s B : ℝ) (hFeq : F = H₀ + (s : 𝕜) • H₁) (hB : ∀ x, |qf H₁ x| ≤ B * nsq x)
    (j : Fin (Fintype.card n)) : |eigAsc hF j - eigAsc h₀ j| ≤ B * |s| := by
  have e1 : eigAsc h₀ j ≤ eigAsc hF j + B * |s| := (eigAsc_le_iff h₀ j _).mpr
    (((eigAsc_le_iff hF j _).mp le_rfl).trans (weyl_affine h₀ hF s B hFeq hB _).2.2.1)
  have e2 : eigAsc hF j ≤ eigAsc h₀ j + B * |s| := (eigAsc_le_iff hF j _).mpr
    (((eigAsc_le_iff h₀ j _).mp le_rfl).trans (weyl_affine h₀ hF s B hFeq hB _).2.2.2)
  rw [abs_sub_le_iff]
  constructor <;> linarith

/-- **Corollary 5, sorted form.** For sorted claimed values `μ 0 ≤ μ 1 ≤ ⋯ ≤ μ (k-1)`,
`|λ_(j+1)(A) - μ j| ≤ r j` for every `j < k`, and `λ_(k+1)(A) ≥ c`. -/
theorem cor5_eigAsc {k : ℕ} {G : Type*} {A : Matrix n n 𝕜} (hA : A.IsHermitian)
    (μ r : Fin k → ℝ) (hμ : Monotone μ) (φ : Fin k → n) (grp : Fin k → G) (lo hi : G → ℝ)
    (hφ : Function.Injective φ) (hrad : ∀ l, |hA.eigenvalues (φ l) - μ l| ≤ r l)
    (hr : ∀ j l, grp j = grp l → r j = r l)
    (hlo : ∀ j, lo (grp j) ≤ μ j - r j) (hhi : ∀ j, μ j + r j ≤ hi (grp j))
    (hdisj : ∀ g g', g ≠ g' → hi g < lo g' ∨ hi g' < lo g)
    (c : ℝ) (hcut : ∀ l, μ l + r l < c) (hcount : cntLt hA c ≤ k) :
    (∀ j : Fin k, ∃ hj : (j : ℕ) < Fintype.card n, |eigAsc hA ⟨j, hj⟩ - μ j| ≤ r j) ∧
    (∀ hk : k < Fintype.card n, c ≤ eigAsc hA ⟨k, hk⟩) := by
  refine ⟨fun j => ?_, fun hk => (le_eigAsc_iff hA ⟨k, hk⟩ c).mpr hcount⟩
  obtain ⟨h1, h2⟩ := cor5_sorted hA μ r hμ φ grp lo hi hφ hrad hr hlo hhi hdisj c hcut hcount j
  obtain ⟨hj, e1, e2⟩ := eigAsc_mem_of_counts hA h1 h2
  exact ⟨hj, abs_sub_le_iff.mpr ⟨by linarith, by linarith⟩⟩

/-- **Corollary 5, Step 4, sorted form**: for the declared model `A_m`,
`|λ_(j+1)(A_m) - μ j| ≤ r j + Δ` for `j < k`, and `λ_(k+1)(A_m) ≥ σ - β - Δ`. -/
theorem cor5_model_eigAsc {k : ℕ} {G : Type*} {Au Am : Matrix n n 𝕜} (hAu : Au.IsHermitian)
    (hAm : Am.IsHermitian) (Δ : ℝ) (hΔ0 : 0 ≤ Δ) (hΔ : ∀ x, |qf (Au - Am) x| ≤ Δ * nsq x)
    (μ r : Fin k → ℝ) (hμ : Monotone μ) (φ : Fin k → n) (grp : Fin k → G) (lo hi : G → ℝ)
    (hφ : Function.Injective φ) (hrad : ∀ l, |hAu.eigenvalues (φ l) - μ l| ≤ r l)
    (hr : ∀ j l, grp j = grp l → r j = r l)
    (hlo : ∀ j, lo (grp j) ≤ μ j - r j) (hhi : ∀ j, μ j + r j ≤ hi (grp j))
    (hdisj : ∀ g g', g ≠ g' → hi g < lo g' ∨ hi g' < lo g)
    (σ β : ℝ) (hcut : ∀ l, μ l + r l + Δ < σ - β - Δ) (hcount : cntLt hAu (σ - β) ≤ k) :
    (∀ j : Fin k, ∃ hj : (j : ℕ) < Fintype.card n, |eigAsc hAm ⟨j, hj⟩ - μ j| ≤ r j + Δ) ∧
    (∀ hk : k < Fintype.card n, σ - β - Δ ≤ eigAsc hAm ⟨k, hk⟩) := by
  have hcount' : cntLt hAu (σ - β) ≤ Fintype.card (Fin k) := by
    rw [Fintype.card_fin]
    exact hcount
  obtain ⟨h1, h2, h3⟩ := cor5_model hAu hAm Δ hΔ0 hΔ μ r φ grp lo hi hφ hrad hr hlo hhi hdisj
    σ β hcut hcount'
  refine ⟨fun j => ?_, fun hk => (le_eigAsc_iff hAm ⟨k, hk⟩ _).mpr ?_⟩
  · have a2 : (j : ℕ) + 1 ≤ cntLe hAm (μ j + r j + Δ) :=
      (succ_le_card_le hμ j le_rfl).trans (h2 j)
    have a3 : cntLt hAm (μ j - r j - Δ) ≤ j := (h3 j).trans (card_lt_le hμ j le_rfl)
    obtain ⟨hj, e1, e2⟩ := eigAsc_mem_of_counts hAm a2 a3
    exact ⟨hj, abs_sub_le_iff.mpr ⟨by linarith, by linarith⟩⟩
  · rw [h1, Fintype.card_fin]

/-- The counting conclusions of Theorem 8 give the sorted form: if `e : Fin K ≃ α` lists the
claimed values `c` in increasing order, `|λ_(j+1) - c (e j)| ≤ r + ε` for `j < K`, and
`λ_(K+1) ≥ L - ε`. -/
theorem sorted_of_thm8_counts {α : Type*} [Fintype α] {A : Matrix n n 𝕜} (hA : A.IsHermitian)
    (c : α → ℝ) {K : ℕ} (e : Fin K ≃ α) (hmono : Monotone fun j => c (e j)) (r ε L : ℝ)
    (hx : ∀ q, c q - r ≤ L)
    (h1 : cntLt hA (L - ε) ≤ Fintype.card α)
    (h2 : ∀ x, (Finset.univ.filter fun q => c q ≤ x).card ≤ cntLe hA (x + r + ε))
    (h3 : ∀ x, x - r ≤ L → cntLt hA (x - r - ε) ≤ (Finset.univ.filter fun q => c q < x).card) :
    (∀ j : Fin K, ∃ hj : (j : ℕ) < Fintype.card n, |eigAsc hA ⟨j, hj⟩ - c (e j)| ≤ r + ε) ∧
    (∀ hK : K < Fintype.card n, L - ε ≤ eigAsc hA ⟨K, hK⟩) := by
  have hcardα : Fintype.card α = K := (Fintype.card_congr e).symm.trans (Fintype.card_fin K)
  have hfle : ∀ t, (Finset.univ.filter fun q => c q ≤ t).card =
      (Finset.univ.filter fun l => c (e l) ≤ t).card := fun t =>
    Finset.card_equiv e.symm fun q => by simp
  have hflt : ∀ t, (Finset.univ.filter fun q => c q < t).card =
      (Finset.univ.filter fun l => c (e l) < t).card := fun t =>
    Finset.card_equiv e.symm fun q => by simp
  refine ⟨fun j => ?_, fun hK => (le_eigAsc_iff hA ⟨K, hK⟩ _).mpr (h1.trans hcardα.le)⟩
  have a2 : (j : ℕ) + 1 ≤ cntLe hA (c (e j) + r + ε) :=
    (succ_le_card_le hmono j le_rfl).trans ((hfle _).symm.le.trans (h2 _))
  have a3 : cntLt hA (c (e j) - r - ε) ≤ j :=
    (h3 _ (hx (e j))).trans ((hflt _).le.trans (card_lt_le hmono j le_rfl))
  obtain ⟨hj, e1, e2⟩ := eigAsc_mem_of_counts hA a2 a3
  exact ⟨hj, abs_sub_le_iff.mpr ⟨by linarith, by linarith⟩⟩

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {m : ι → Type*} [∀ s, Fintype (m s)]
  [∀ s, DecidableEq (m s)] {κ : ι → Type*} [∀ s, Fintype (κ s)]

omit [DecidableEq n] [Fintype ι] [DecidableEq ι] [∀ s, Fintype (κ s)] in
theorem radius_nonneg (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian)
    (c : ∀ s, κ s → ℝ) (r : ℝ) (φ : ∀ s, κ s → m s)
    (hrad : ∀ s l, |(hH s).eigenvalues (φ s l) - c s l| ≤ r) (Ecut βmax : ℝ)
    (hcut : ∀ s l, c s l + r < Ecut - βmax) (q : Σ s, κ s) : c q.1 q.2 - r ≤ Ecut - βmax := by
  have h0 := (abs_nonneg _).trans (hrad q.1 q.2)
  have := hcut q.1 q.2
  linarith

/-- **Theorem 8, sorted form** (unitary `Q`, quadratic-form hypotheses). With `e` listing the
claimed values `c_(1) ≤ ⋯ ≤ c_(K)` in increasing order and `ε = ρ + m_b`:
`|λ_j(H_spec) - c_(j)| ≤ r + ε` for `j ≤ K`, and `λ_(K+1)(H_spec) ≥ Ecut - βmax - ε`. -/
theorem thm8_sorted {Hspec Hb : Matrix n n 𝕜}
    (hHs : Hspec.IsHermitian) (hHb : Hb.IsHermitian) (mb : ℝ)
    (hm : ∀ x, |qf (Hb - Hspec) x| ≤ mb * nsq x)
    (Q : Matrix n (Σ s, m s) 𝕜) (hQ1 : Qᴴ * Q = 1) (hQ2 : Q * Qᴴ = 1)
    (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian) (ρ : ℝ)
    (hρ : ∀ y, |qf (Qᴴ * Hb * Q - blockDiagonal' H) y| ≤ ρ * nsq y)
    (c : ∀ s, κ s → ℝ) (r : ℝ) (φ : ∀ s, κ s → m s) (hφ : ∀ s, Function.Injective (φ s))
    (hrad : ∀ s l, |(hH s).eigenvalues (φ s l) - c s l| ≤ r) (Ecut : ℝ) (β : ι → ℝ)
    (βmax : ℝ) (hβ : ∀ s, β s ≤ βmax)
    (hν : ∀ s, cntLt (hH s) (Ecut - β s) ≤ Fintype.card (κ s))
    (hcut : ∀ s l, c s l + r < Ecut - βmax) {K : ℕ} (e : Fin K ≃ Σ s, κ s)
    (hmono : Monotone fun j => c (e j).1 (e j).2) :
    (∀ j : Fin K, ∃ hj : (j : ℕ) < Fintype.card n,
      |eigAsc hHs ⟨j, hj⟩ - c (e j).1 (e j).2| ≤ r + (ρ + mb)) ∧
    (∀ hK : K < Fintype.card n, Ecut - βmax - (ρ + mb) ≤ eigAsc hHs ⟨K, hK⟩) := by
  obtain ⟨h1, h2, h3⟩ :=
    thm8 hHs hHb mb hm Q hQ1 hQ2 H hH ρ hρ c r φ hφ hrad Ecut β βmax hβ hν hcut
  exact sorted_of_thm8_counts hHs (fun q => c q.1 q.2) e hmono r (ρ + mb) (Ecut - βmax)
    (radius_nonneg H hH c r φ hrad Ecut βmax hcut) h1 h2 h3

/-- **Theorem 8, sorted form**, with the spectral-norm hypotheses `‖H - H_spec‖₂ ≤ m_b` and
`‖Qᴴ H Q - B‖₂ ≤ ρ`. -/
theorem thm8_opNorm_sorted {Hspec Hb : Matrix n n 𝕜} (hHs : Hspec.IsHermitian)
    (hHb : Hb.IsHermitian)
    (mb : ℝ) (hm : ∀ z : EuclideanSpace 𝕜 n, ‖Matrix.toEuclideanLin (Hb - Hspec) z‖ ≤ mb * ‖z‖)
    (Q : Matrix n (Σ s, m s) 𝕜) (hQ1 : Qᴴ * Q = 1) (hQ2 : Q * Qᴴ = 1)
    (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian) (ρ : ℝ)
    (hρ : ∀ z : EuclideanSpace 𝕜 (Σ s, m s),
      ‖Matrix.toEuclideanLin (Qᴴ * Hb * Q - blockDiagonal' H) z‖ ≤ ρ * ‖z‖)
    (c : ∀ s, κ s → ℝ) (r : ℝ) (φ : ∀ s, κ s → m s) (hφ : ∀ s, Function.Injective (φ s))
    (hrad : ∀ s l, |(hH s).eigenvalues (φ s l) - c s l| ≤ r) (Ecut : ℝ) (β : ι → ℝ)
    (βmax : ℝ) (hβ : ∀ s, β s ≤ βmax)
    (hν : ∀ s, cntLt (hH s) (Ecut - β s) ≤ Fintype.card (κ s))
    (hcut : ∀ s l, c s l + r < Ecut - βmax) {K : ℕ} (e : Fin K ≃ Σ s, κ s)
    (hmono : Monotone fun j => c (e j).1 (e j).2) :
    (∀ j : Fin K, ∃ hj : (j : ℕ) < Fintype.card n,
      |eigAsc hHs ⟨j, hj⟩ - c (e j).1 (e j).2| ≤ r + (ρ + mb)) ∧
    (∀ hK : K < Fintype.card n, Ecut - βmax - (ρ + mb) ≤ eigAsc hHs ⟨K, hK⟩) := by
  obtain ⟨h1, h2, h3⟩ :=
    thm8_opNorm hHs hHb mb hm Q hQ1 hQ2 H hH ρ hρ c r φ hφ hrad Ecut β βmax hβ hν hcut
  exact sorted_of_thm8_counts hHs (fun q => c q.1 q.2) e hmono r (ρ + mb) (Ecut - βmax)
    (radius_nonneg H hH c r φ hrad Ecut βmax hcut) h1 h2 h3

/-- **Theorem 8, sorted form**, with the polar-factor lemma and the hypotheses of
`docs/proofs/thm8_rho.md`: `ε = D + (‖H‖ + ‖B‖) δ + m_b` with `δ = 1 - √(1 - η)`. -/
theorem thm8_polar_sorted {Hspec Hb : Matrix n n 𝕜} (hHs : Hspec.IsHermitian)
    (hHb : Hb.IsHermitian)
    (mb : ℝ) (hm : ∀ z : EuclideanSpace 𝕜 n, ‖toEuclideanLin (Hb - Hspec) z‖ ≤ mb * ‖z‖)
    (W : Matrix n (Σ s, m s) 𝕜) (hcard : Fintype.card (Σ s, m s) = Fintype.card n) {η : ℝ}
    (hη0 : 0 ≤ η) (hη1 : η < 1) (hη : ∀ z, ‖toEuclideanLin (Wᴴ * W) z - z‖ ≤ η * ‖z‖)
    (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian) {D nH nB : ℝ}
    (hR : ∀ z, ‖toEuclideanLin (Hb * W - W * blockDiagonal' H) z‖ ≤ D * ‖z‖) (hnH0 : 0 ≤ nH)
    (hnH : ∀ v, ‖toEuclideanLin Hb v‖ ≤ nH * ‖v‖)
    (hnB : ∀ z, ‖toEuclideanLin (blockDiagonal' H) z‖ ≤ nB * ‖z‖)
    (c : ∀ s, κ s → ℝ) (r : ℝ) (φ : ∀ s, κ s → m s) (hφ : ∀ s, Function.Injective (φ s))
    (hrad : ∀ s l, |(hH s).eigenvalues (φ s l) - c s l| ≤ r) (Ecut : ℝ) (β : ι → ℝ)
    (βmax : ℝ) (hβ : ∀ s, β s ≤ βmax)
    (hν : ∀ s, cntLt (hH s) (Ecut - β s) ≤ Fintype.card (κ s))
    (hcut : ∀ s l, c s l + r < Ecut - βmax) {K : ℕ} (e : Fin K ≃ Σ s, κ s)
    (hmono : Monotone fun j => c (e j).1 (e j).2) :
    (∀ j : Fin K, ∃ hj : (j : ℕ) < Fintype.card n,
      |eigAsc hHs ⟨j, hj⟩ - c (e j).1 (e j).2| ≤
        r + (D + (nH + nB) * (1 - √(1 - η)) + mb)) ∧
    (∀ hK : K < Fintype.card n,
      Ecut - βmax - (D + (nH + nB) * (1 - √(1 - η)) + mb) ≤ eigAsc hHs ⟨K, hK⟩) := by
  obtain ⟨h1, h2, h3⟩ := thm8_polar hHs hHb mb hm W hcard hη0 hη1 hη H hH hR hnH0 hnH hnB
    c r φ hφ hrad Ecut β βmax hβ hν hcut
  exact sorted_of_thm8_counts hHs (fun q => c q.1 q.2) e hmono r _ (Ecut - βmax)
    (radius_nonneg H hH c r φ hrad Ecut βmax hcut) h1 h2 h3

end CertifyMathlib.Counting
