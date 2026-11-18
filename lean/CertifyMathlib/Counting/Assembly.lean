import CertifyMathlib.Counting.Completeness

/-!
# Theorem 8 (sector assembly) and the spectrum of a block-diagonal matrix

Source: `docs/theory.md`, section 5, and `docs/proofs/thm8_rho.md` in the code repository.

* `cnt_blockDiagonal'`: the spectrum of `B = ⊕ H_s` is the multiset union of the sector spectra,
  in counting form: `#{λ(B) < t} = Σ_s #{λ(H_s) < t}` and the same for `≤`, for every `t`.
  Equal counts at every threshold are the same as equal multisets of eigenvalues.
* `thm8_blocks`: Steps 1 and 2 of the derivation. Sector-wise matchings within the uniform radius
  `r` and sector counts `cntLt (H_s) (E_cut - β_s) ≤ k_s` give, for the pooled claimed values,
  exactly `K = Σ k_s` eigenvalues of `B` below `E_cut - max β`, and the sorted pairing of `B`'s
  lowest `K` eigenvalues with the pooled claimed values within `r`.
* `thm8`: Steps 3 to 5 under the hypothesis that a unitary `Q` (in the derivation the polar factor
  of `W = [V_1 ⋯ V_m]`) satisfies `|yᴴ (Qᴴ H Q - B) y| ≤ ρ ‖y‖²`, and that the built matrix `H` is
  within `m_b` of the declared model `H_spec`: the pairing holds for `H_spec` with radius
  `r + ρ + m_b` and `λ_{K+1}(H_spec) ≥ E_cut - max β - (ρ + m_b)`.

Not formalised: the polar-factor lemma of `docs/proofs/thm2_polar-factor.md` (that `Q̂ = W(WᴴW)^{-1/2}`
is unitary with `‖W - Q̂‖ ≤ δ = 1 - √(1 - η)`), from which the derivation obtains the hypothesis
on `Qᴴ H Q - B` with `ρ = D + (‖H‖ + ‖B‖) δ`. It is the one missing lemma of Theorem 8.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜]
variable {ι : Type*} [Fintype ι] [DecidableEq ι] {m : ι → Type*} [∀ s, Fintype (m s)]
  [∀ s, DecidableEq (m s)]

omit [Fintype ι] [∀ s, Fintype (m s)] [∀ s, DecidableEq (m s)] in
theorem isHermitian_blockDiagonal' (H : ∀ s, Matrix (m s) (m s) 𝕜)
    (hH : ∀ s, (H s).IsHermitian) : (blockDiagonal' H).IsHermitian := by
  show (blockDiagonal' H)ᴴ = blockDiagonal' H
  rw [blockDiagonal'_conjTranspose]
  congr 1
  funext s
  exact hH s

omit [Fintype ι] [∀ s, Fintype (m s)] in
theorem blockDiagonal'_sub_smul_one (H : ∀ s, Matrix (m s) (m s) 𝕜) (t : ℝ) :
    blockDiagonal' H - (t : 𝕜) • (1 : Matrix (Σ s, m s) (Σ s, m s) 𝕜) =
      blockDiagonal' (fun s => H s - (t : 𝕜) • (1 : Matrix (m s) (m s) 𝕜)) := by
  show _ = blockDiagonal' (H - (t : 𝕜) • (1 : ∀ s, Matrix (m s) (m s) 𝕜))
  rw [blockDiagonal'_sub, blockDiagonal'_smul, blockDiagonal'_one]

omit [DecidableEq ι] in
/-- The number of elements of a filter on a sigma type is the sum over the fibres. -/
theorem card_filter_sigma {κ : ι → Type*} [∀ s, Fintype (κ s)] (P : ∀ s, κ s → Prop)
    [∀ s l, Decidable (P s l)] :
    (Finset.univ.filter fun q : (Σ s, κ s) => P q.1 q.2).card =
      ∑ s, (Finset.univ.filter fun l => P s l).card := by
  rw [Finset.card_filter, Fintype.sum_sigma]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [Finset.card_filter]

/-- **The spectrum of a block-diagonal matrix is the union of the block spectra**, in counting
form. -/
theorem cnt_blockDiagonal' (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian)
    (t : ℝ) :
    cntLt (isHermitian_blockDiagonal' H hH) t = ∑ s, cntLt (hH s) t ∧
    cntLe (isHermitian_blockDiagonal' H hH) t = ∑ s, cntLe (hH s) t := by
  let U : ∀ s, Matrix (m s) (m s) 𝕜 := fun s => ((hH s).eigenvectorUnitary : Matrix (m s) (m s) 𝕜)
  have hfac : blockDiagonal' H - (t : 𝕜) • (1 : Matrix (Σ s, m s) (Σ s, m s) 𝕜) =
      blockDiagonal' U * dg (fun q : (Σ s, m s) => (hH q.1).eigenvalues q.2 - t) *
        (blockDiagonal' U)ᴴ := by
    rw [blockDiagonal'_sub_smul_one, blockDiagonal'_conjTranspose]
    have hdg : (dg (fun q : (Σ s, m s) => (hH q.1).eigenvalues q.2 - t) :
        Matrix (Σ s, m s) (Σ s, m s) 𝕜) =
        blockDiagonal' (fun s => dg (fun i => (hH s).eigenvalues i - t)) := by
      simp only [dg]
      rw [blockDiagonal'_diagonal]
    rw [hdg, ← blockDiagonal'_mul, ← blockDiagonal'_mul]
    congr 1
    funext s
    exact sub_smul_one_eq (hH s) t
  have hK1 : (blockDiagonal' U)ᴴ * blockDiagonal' U = 1 := by
    rw [blockDiagonal'_conjTranspose, ← blockDiagonal'_mul, ← blockDiagonal'_one]
    congr 1
    funext s
    exact conjTranspose_mul_unitary _
  have hK2 : blockDiagonal' U * (blockDiagonal' U)ᴴ = 1 := by
    rw [blockDiagonal'_conjTranspose, ← blockDiagonal'_mul, ← blockDiagonal'_one]
    congr 1
    funext s
    exact unitary_mul_conjTranspose _
  obtain ⟨h1, h2⟩ := cntLt_eq_of_factor (isHermitian_blockDiagonal' H hH) t _ _ hK1 hK2 _ hfac
  rw [h1, h2]
  constructor
  · refine (card_filter_sigma (fun s i => (hH s).eigenvalues i - t < 0)).trans ?_
    refine Finset.sum_congr rfl fun s _ => ?_
    simp only [cntLt, sub_neg]
  · refine (card_filter_sigma (fun s i => (hH s).eigenvalues i - t ≤ 0)).trans ?_
    refine Finset.sum_congr rfl fun s _ => ?_
    simp only [cntLe, sub_nonpos]

section Sectors

variable {κ : ι → Type*} [∀ s, Fintype (κ s)]

/-- **Theorem 8, Steps 1 and 2** (the assembled block matrix `B = ⊕ H_s`). -/
theorem thm8_blocks (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian)
    (c : ∀ s, κ s → ℝ) (r : ℝ) (φ : ∀ s, κ s → m s) (hφ : ∀ s, Function.Injective (φ s))
    (hrad : ∀ s l, |(hH s).eigenvalues (φ s l) - c s l| ≤ r) (Ecut : ℝ) (β : ι → ℝ)
    (βmax : ℝ) (hβ : ∀ s, β s ≤ βmax)
    (hν : ∀ s, cntLt (hH s) (Ecut - β s) ≤ Fintype.card (κ s))
    (hcut : ∀ s l, c s l + r < Ecut - βmax) :
    cntLt (isHermitian_blockDiagonal' H hH) (Ecut - βmax) = Fintype.card (Σ s, κ s) ∧
    (∀ x, (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 ≤ x).card ≤
      cntLe (isHermitian_blockDiagonal' H hH) (x + r)) ∧
    (∀ x, x - r ≤ Ecut - βmax →
      cntLt (isHermitian_blockDiagonal' H hH) (x - r) ≤
        (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 < x).card) := by
  have hν' : ∀ s, cntLt (hH s) (Ecut - βmax) ≤ Fintype.card (κ s) := fun s =>
    (cntLt_mono (hH s) (by linarith [hβ s])).trans (hν s)
  refine ⟨?_, fun x => ?_, fun x hx => ?_⟩
  · rw [(cnt_blockDiagonal' H hH _).1, Fintype.card_sigma]
    refine Finset.sum_congr rfl fun s _ => le_antisymm (hν' s) ?_
    exact card_le_cntLt_of_match (hH s) (c s) (fun _ => r) (φ s) (hφ s) (hrad s) _ (hcut s)
  · rw [(cnt_blockDiagonal' H hH _).2]
    refine (card_filter_sigma (κ := κ) (fun s l => c s l ≤ x)).trans_le ?_
    exact Finset.sum_le_sum fun s _ =>
      count_le_cntLe_of_match (hH s) (c s) r (φ s) (hφ s) (hrad s) x
  · rw [(cnt_blockDiagonal' H hH _).1]
    refine le_of_le_of_eq ?_ (card_filter_sigma (κ := κ) (fun s l => c s l < x)).symm
    exact Finset.sum_le_sum fun s _ =>
      cntLt_le_count_of_match (hH s) (c s) r (φ s) (hφ s) (hrad s) _ (hcut s) (hν' s) x hx

/-- **Theorem 8** (Steps 3 to 5, given the unitary `Q` with `|yᴴ (Qᴴ H Q - B) y| ≤ ρ ‖y‖²`). -/
theorem thm8 {n : Type*} [Fintype n] [DecidableEq n] {Hspec Hb : Matrix n n 𝕜}
    (hHs : Hspec.IsHermitian) (hHb : Hb.IsHermitian) (mb : ℝ)
    (hm : ∀ x, |qf (Hb - Hspec) x| ≤ mb * nsq x)
    (Q : Matrix n (Σ s, m s) 𝕜) (hQ1 : Qᴴ * Q = 1) (hQ2 : Q * Qᴴ = 1)
    (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian) (ρ : ℝ)
    (hρ : ∀ y, |qf (Qᴴ * Hb * Q - blockDiagonal' H) y| ≤ ρ * nsq y)
    (c : ∀ s, κ s → ℝ) (r : ℝ) (φ : ∀ s, κ s → m s) (hφ : ∀ s, Function.Injective (φ s))
    (hrad : ∀ s l, |(hH s).eigenvalues (φ s l) - c s l| ≤ r) (Ecut : ℝ) (β : ι → ℝ)
    (βmax : ℝ) (hβ : ∀ s, β s ≤ βmax)
    (hν : ∀ s, cntLt (hH s) (Ecut - β s) ≤ Fintype.card (κ s))
    (hcut : ∀ s l, c s l + r < Ecut - βmax) :
    cntLt hHs (Ecut - βmax - (ρ + mb)) ≤ Fintype.card (Σ s, κ s) ∧
    (∀ x, (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 ≤ x).card ≤
      cntLe hHs (x + r + (ρ + mb))) ∧
    (∀ x, x - r ≤ Ecut - βmax →
      cntLt hHs (x - r - (ρ + mb)) ≤
        (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 < x).card) := by
  obtain ⟨hB1, hB2, hB3⟩ := thm8_blocks H hH c r φ hφ hrad Ecut β βmax hβ hν hcut
  have hK := isHermitian_conj hHb Q
  have hc : ∀ t, cntLt hK t = cntLt hHb t ∧ cntLe hK t = cntLe hHb t := fun t =>
    cnt_conj hHb Q hQ1 hQ2 hK rfl t
  have hρ' : ∀ y, |qf (blockDiagonal' H - Qᴴ * Hb * Q) y| ≤ ρ * nsq y := fun y => by
    rw [abs_qf_sub_comm]
    exact hρ y
  have hm' : ∀ x, |qf (Hspec - Hb) x| ≤ mb * nsq x := fun x => by
    rw [abs_qf_sub_comm]
    exact hm x
  refine ⟨?_, fun x => ?_, fun x hx => ?_⟩
  · have h1 := weyl_cntLt hHs hHb mb hm' (Ecut - βmax - (ρ + mb))
    rw [← (hc _).1] at h1
    have h2 := weyl_cntLt hK (isHermitian_blockDiagonal' H hH) ρ hρ
      (Ecut - βmax - (ρ + mb) + mb)
    rw [show Ecut - βmax - (ρ + mb) + mb + ρ = Ecut - βmax by ring, hB1] at h2
    exact h1.trans h2
  · have h1 := weyl_cntLe (isHermitian_blockDiagonal' H hH) hK ρ hρ' (x + r)
    rw [(hc _).2] at h1
    have h2 := weyl_cntLe hHb hHs mb hm (x + r + ρ)
    rw [show x + r + ρ + mb = x + r + (ρ + mb) by ring] at h2
    exact (hB2 x).trans (h1.trans h2)
  · have h1 := weyl_cntLt hHs hHb mb hm' (x - r - (ρ + mb))
    rw [← (hc _).1] at h1
    have h2 := weyl_cntLt hK (isHermitian_blockDiagonal' H hH) ρ hρ
      (x - r - (ρ + mb) + mb)
    rw [show x - r - (ρ + mb) + mb + ρ = x - r by ring] at h2
    exact h1.trans (h2.trans (hB3 x hx))

end Sectors

end CertifyMathlib.Counting
