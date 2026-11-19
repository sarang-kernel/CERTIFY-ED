import CertifyMathlib.Enclosure.Spectral

/-!
# Kahan's inclusion theorem (general Hermitian block)

`docs/proofs/thm2_polar-factor.md`, section 2. Let `Q : F → E` be an isometry (orthonormal
columns), `H` a symmetric map on `F` with eigenvalues `θ_1 ≤ … ≤ θ_g`, `A` symmetric on `E` with
eigenvalues `λ_1 ≤ … ≤ λ_N`, and `‖A Q z - Q H z‖ ≤ r ‖z‖` for all `z`. Then there are distinct
indices `i_1, …, i_g` with `|λ_{i_j} - θ_j| ≤ r`. `H` need not be the Rayleigh quotient `Qᴴ A Q`.

The proof has the two steps of the document. Step 1 (interval lemma, `kahan_interval_count`) is the
Courant–Fischer count of `Spectral.lean` applied to the image under `Q` of the span of the
eigenvectors of `H` with eigenvalues in `[a, b]`. Step 2 replaces Hall's marriage theorem by an
explicit order-preserving matching on the real line (`exists_strictMono_matching`): the index of
`θ_j` is `p_j = max_{l ≤ j} (#{i : λ_i < θ_l - r} + j - l)`. This gives the stronger conclusion that
the matching can be chosen strictly increasing.
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

/-! ### Step 2: an order-preserving matching on the line -/

section Matching

variable {N g : ℕ} (lam : Fin N → ℝ)

/-- The number of entries of `lam` below `x`. -/
noncomputable def nBelow (x : ℝ) : ℕ := (univ.filter fun i => lam i < x).card

variable {lam}

theorem lt_nBelow_of_lt (hlam : Monotone lam) {i : Fin N} {x : ℝ} (h : lam i < x) :
    (i : ℕ) < nBelow lam x := by
  have hsub : Finset.Iic i ⊆ univ.filter (fun i' => lam i' < x) := by
    intro i' hi'
    simp only [Finset.mem_Iic] at hi'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact lt_of_le_of_lt (hlam hi') h
  have := Finset.card_le_card hsub
  rw [Fin.card_Iic] at this
  unfold nBelow
  omega

theorem le_of_nBelow_le (hlam : Monotone lam) {i : Fin N} {x : ℝ} (h : nBelow lam x ≤ i) :
    x ≤ lam i := by
  by_contra h'
  push Not at h'
  have := lt_nBelow_of_lt hlam h'
  omega

theorem nBelow_le_of_le (hlam : Monotone lam) {i : Fin N} {x : ℝ} (h : x ≤ lam i) :
    nBelow lam x ≤ i := by
  have hsub : univ.filter (fun i' => lam i' < x) ⊆ Finset.Iio i := by
    intro i' hi'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi'
    simp only [Finset.mem_Iio]
    by_contra hc
    push Not at hc
    have := hlam hc
    linarith
  have := Finset.card_le_card hsub
  rw [Fin.card_Iio] at this
  exact this

theorem card_filter_fin_Ico_le (a b : ℕ) :
    (univ.filter fun i : Fin N => a ≤ (i : ℕ) ∧ (i : ℕ) < b).card ≤ b - a := by
  have h : (univ.filter fun i : Fin N => a ≤ (i : ℕ) ∧ (i : ℕ) < b).map Fin.valEmbedding ⊆
      Finset.Ico a b := by
    intro x hx
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and,
      Fin.valEmbedding_apply] at hx
    obtain ⟨i, ⟨h1, h2⟩, rfl⟩ := hx
    exact Finset.mem_Ico.mpr ⟨h1, h2⟩
  have := Finset.card_le_card h
  rwa [Finset.card_map, Nat.card_Ico] at this

variable (lam) (θ : Fin g → ℝ) (r : ℝ)

/-- The greedy index `p_j = max_{l ≤ j} (nBelow (θ_l - r) + (j - l))`. -/
noncomputable def greedyIndex (j : Fin g) : ℕ :=
  (univ.filter fun l : Fin g => l ≤ j).sup fun l => nBelow lam (θ l - r) + ((j : ℕ) - l)

theorem le_greedyIndex {j l : Fin g} (hl : l ≤ j) :
    nBelow lam (θ l - r) + ((j : ℕ) - l) ≤ greedyIndex lam θ r j :=
  Finset.le_sup (f := fun l : Fin g => nBelow lam (θ l - r) + ((j : ℕ) - l)) (by simp [hl])

theorem exists_greedyIndex_eq (j : Fin g) :
    ∃ l, l ≤ j ∧ greedyIndex lam θ r j = nBelow lam (θ l - r) + ((j : ℕ) - l) := by
  obtain ⟨l, hl, heq⟩ := Finset.exists_mem_eq_sup (univ.filter fun l : Fin g => l ≤ j)
    ⟨j, by simp⟩ (fun l => nBelow lam (θ l - r) + ((j : ℕ) - l))
  exact ⟨l, by simpa using hl, heq⟩

theorem greedyIndex_strictMono : StrictMono (greedyIndex lam θ r) := by
  intro j j' hjj'
  obtain ⟨l, hl, heq⟩ := exists_greedyIndex_eq lam θ r j
  have h2 := le_greedyIndex lam θ r (hl.trans hjj'.le)
  have h3 : (j : ℕ) < j' := Fin.lt_def.mp hjj'
  have h4 : (l : ℕ) ≤ j := Fin.le_iff_val_le_val.mp hl
  omega

variable {lam θ r}

theorem greedyIndex_upper (hlam : Monotone lam)
    (hcount : ∀ j k : Fin g, j ≤ k →
      (k : ℕ) - j + 1 ≤ (univ.filter fun i => θ j - r ≤ lam i ∧ lam i ≤ θ k + r).card)
    (j : Fin g) : ∃ h : greedyIndex lam θ r j < N, lam ⟨_, h⟩ ≤ θ j + r := by
  by_contra hne
  push Not at hne
  obtain ⟨l, hl, heq⟩ := exists_greedyIndex_eq lam θ r j
  have hc := hcount l j hl
  have hsub : univ.filter (fun i => θ l - r ≤ lam i ∧ lam i ≤ θ j + r) ⊆
      univ.filter (fun i : Fin N =>
        nBelow lam (θ l - r) ≤ (i : ℕ) ∧ (i : ℕ) < greedyIndex lam θ r j) := by
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
    refine ⟨nBelow_le_of_le hlam hi.1, ?_⟩
    by_contra hge
    push Not at hge
    have hlt : greedyIndex lam θ r j < N := lt_of_le_of_lt hge i.isLt
    have h1 := hne hlt
    have h2 : lam ⟨_, hlt⟩ ≤ lam i := hlam (show (⟨_, hlt⟩ : Fin N) ≤ i from hge)
    linarith [hi.2]
  have h3 := (Finset.card_le_card hsub).trans (card_filter_fin_Ico_le _ _)
  have h4 : (l : ℕ) ≤ j := Fin.le_iff_val_le_val.mp hl
  omega

/-- **Order-preserving matching on the line.** If every window `[θ_j - r, θ_k + r]` (`j ≤ k`)
contains at least `k - j + 1` of the `λ_i`, there is a strictly increasing `σ` with
`|λ_{σ j} - θ_j| ≤ r`. -/
theorem exists_strictMono_matching (hlam : Monotone lam)
    (hcount : ∀ j k : Fin g, j ≤ k →
      (k : ℕ) - j + 1 ≤ (univ.filter fun i => θ j - r ≤ lam i ∧ lam i ≤ θ k + r).card) :
    ∃ σ : Fin g → Fin N, StrictMono σ ∧ ∀ j, |lam (σ j) - θ j| ≤ r := by
  choose hlt hup using greedyIndex_upper hlam hcount
  refine ⟨fun j => ⟨greedyIndex lam θ r j, hlt j⟩, ?_, ?_⟩
  · intro j j' h
    show greedyIndex lam θ r j < greedyIndex lam θ r j'
    exact greedyIndex_strictMono lam θ r h
  · intro j
    have hlow : θ j - r ≤ lam ⟨_, hlt j⟩ := le_of_nBelow_le hlam (by
      show nBelow lam (θ j - r) ≤ greedyIndex lam θ r j
      have := le_greedyIndex lam θ r (le_refl j)
      omega)
    have hu := hup j
    rw [abs_le]
    constructor <;> linarith

end Matching

/-! ### Step 1 and the theorem -/

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
  {F : Type*} [NormedAddCommGroup F] [InnerProductSpace 𝕜 F] [FiniteDimensional 𝕜 E]

/-- **Interval lemma** (Step 1): `#{θ_l ∈ [a, b]} ≤ #{λ_i ∈ [a - r, b + r]}`. -/
theorem kahan_interval_count {A : E →ₗ[𝕜] E} (hA : A.IsSymmetric) {ι : Type*} [Fintype ι]
    [DecidableEq ι] (D : EigenData A ι) {H : F →ₗ[𝕜] F} (hH : H.IsSymmetric) {κ : Type*}
    [Fintype κ] (DH : EigenData H κ) (Q : F →ₗ[𝕜] E) (hQ : ∀ z, ‖Q z‖ = ‖z‖) {r : ℝ}
    (hr : ∀ z, ‖A (Q z) - Q (H z)‖ ≤ r * ‖z‖) {a b : ℝ} (hab : a ≤ b) :
    (univ.filter fun l => a ≤ DH.eig l ∧ DH.eig l ≤ b).card ≤
      (univ.filter fun i => a - r ≤ D.eig i ∧ D.eig i ≤ b + r).card := by
  have hQinj : Function.Injective Q := by
    intro z w hzw
    have h0 : ‖z - w‖ = 0 := by rw [← hQ, map_sub, hzw, sub_self, norm_zero]
    exact sub_eq_zero.mp (norm_eq_zero.mp h0)
  have hY : finrank 𝕜 ((DH.span (univ.filter fun l => a ≤ DH.eig l ∧ DH.eig l ≤ b)).map Q) =
      (univ.filter fun l => a ≤ DH.eig l ∧ DH.eig l ≤ b).card := by
    rw [← DH.finrank_span]
    exact (LinearEquiv.finrank_eq (Submodule.equivMapOfInjective Q hQinj _)).symm
  have hcount := D.finrank_le_card_near hA
    ((DH.span (univ.filter fun l => a ≤ DH.eig l ∧ DH.eig l ≤ b)).map Q)
    (c := (a + b) / 2) (r := r + (b - a) / 2) (by
      intro y hy
      obtain ⟨z, hz, rfl⟩ := Submodule.mem_map.mp hy
      have hzb : ‖H z - (((a + b) / 2 : ℝ) : 𝕜) • z‖ ≤ (b - a) / 2 * ‖z‖ :=
        DH.norm_sub_smul_le hH (by linarith) (fun l hl => by
          have hls : a ≤ DH.eig l ∧ DH.eig l ≤ b := by
            by_contra hns
            exact hl (DH.repr_eq_zero_of_mem_span hz (by simpa using hns))
          rw [abs_le]
          constructor <;> linarith [hls.1, hls.2])
      have e : A (Q z) - (((a + b) / 2 : ℝ) : 𝕜) • Q z =
          (A (Q z) - Q (H z)) + Q (H z - (((a + b) / 2 : ℝ) : 𝕜) • z) := by
        rw [map_sub, map_smul]
        abel
      rw [e, hQ z]
      calc ‖(A (Q z) - Q (H z)) + Q (H z - (((a + b) / 2 : ℝ) : 𝕜) • z)‖
          ≤ ‖A (Q z) - Q (H z)‖ + ‖Q (H z - (((a + b) / 2 : ℝ) : 𝕜) • z)‖ := norm_add_le _ _
        _ ≤ r * ‖z‖ + (b - a) / 2 * ‖z‖ := by rw [hQ]; exact add_le_add (hr z) hzb
        _ = (r + (b - a) / 2) * ‖z‖ := by ring)
  rw [hY] at hcount
  refine hcount.trans (Finset.card_le_card ?_)
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, abs_le] at hi ⊢
  constructor <;> linarith [hi.1, hi.2]

/-- **Kahan's inclusion theorem**, eigenvalues in increasing order: there is a strictly increasing
(in particular injective) `σ` with `|λ_{σ j} - θ_j| ≤ r`. -/
theorem kahan {N g : ℕ} {A : E →ₗ[𝕜] E} (hA : A.IsSymmetric) (D : EigenData A (Fin N))
    (hD : Monotone D.eig) {H : F →ₗ[𝕜] F} (hH : H.IsSymmetric) (DH : EigenData H (Fin g))
    (hDH : Monotone DH.eig) (Q : F →ₗ[𝕜] E) (hQ : ∀ z, ‖Q z‖ = ‖z‖) {r : ℝ}
    (hr : ∀ z, ‖A (Q z) - Q (H z)‖ ≤ r * ‖z‖) :
    ∃ σ : Fin g → Fin N, StrictMono σ ∧ ∀ j, |D.eig (σ j) - DH.eig j| ≤ r := by
  apply exists_strictMono_matching hD
  intro j k hjk
  have h1 := kahan_interval_count hA D hH DH Q hQ hr (hDH hjk)
  refine le_trans ?_ h1
  have hsub : Finset.Icc j k ⊆
      univ.filter fun l => DH.eig j ≤ DH.eig l ∧ DH.eig l ≤ DH.eig k := by
    intro l hl
    rw [Finset.mem_Icc] at hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨hDH hl.1, hDH hl.2⟩
  have h2 := Finset.card_le_card hsub
  rw [Fin.card_Icc] at h2
  have h3 : (j : ℕ) ≤ k := Fin.le_iff_val_le_val.mp hjk
  omega

/-- **Kahan's inclusion theorem** in Mathlib's order (eigenvalues decreasing, as
`Matrix.IsHermitian.eigenvalues₀`). -/
theorem kahan_antitone {N g : ℕ} {A : E →ₗ[𝕜] E} (hA : A.IsSymmetric) (D : EigenData A (Fin N))
    (hD : Antitone D.eig) {H : F →ₗ[𝕜] F} (hH : H.IsSymmetric) (DH : EigenData H (Fin g))
    (hDH : Antitone DH.eig) (Q : F →ₗ[𝕜] E) (hQ : ∀ z, ‖Q z‖ = ‖z‖) {r : ℝ}
    (hr : ∀ z, ‖A (Q z) - Q (H z)‖ ≤ r * ‖z‖) :
    ∃ σ : Fin g → Fin N, StrictMono σ ∧ ∀ j, |D.eig (σ j) - DH.eig j| ≤ r := by
  have hm : Monotone (D.reindex Fin.revPerm).eig := by
    intro i j h
    simp only [EigenData.reindex, Function.comp_apply, Fin.revPerm_symm, Fin.revPerm_apply]
    exact hD (Fin.rev_le_rev.mpr h)
  have hmH : Monotone (DH.reindex Fin.revPerm).eig := by
    intro i j h
    simp only [EigenData.reindex, Function.comp_apply, Fin.revPerm_symm, Fin.revPerm_apply]
    exact hDH (Fin.rev_le_rev.mpr h)
  obtain ⟨σ, hσ, hb⟩ := kahan hA (D.reindex Fin.revPerm) hm hH (DH.reindex Fin.revPerm) hmH Q hQ hr
  refine ⟨fun j => Fin.rev (σ (Fin.rev j)), ?_, ?_⟩
  · intro j j' h
    exact Fin.rev_lt_rev.mpr (hσ (Fin.rev_lt_rev.mpr h))
  · intro j
    have := hb (Fin.rev j)
    simpa [EigenData.reindex, Fin.revPerm_symm, Fin.revPerm_apply] using this

end CertifyMathlib.Enclosure
