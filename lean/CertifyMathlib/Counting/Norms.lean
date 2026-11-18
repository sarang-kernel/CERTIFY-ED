import CertifyMathlib.Counting.Lemma4
import CertifyMathlib.Counting.Assembly

/-!
# From spectral-norm hypotheses to quadratic-form hypotheses

The documents state their perturbation hypotheses with the spectral norm, `‖E‖₂ ≤ β`. The other
files of this library take them in the weaker form `|xᴴ E x| ≤ β ‖x‖²`. This file proves that the
first implies the second (`abs_qf_le_of_toEuclideanLin`, Cauchy-Schwarz) for the operator form
`‖E z‖ ≤ β ‖z‖` on `EuclideanSpace 𝕜 n` that Mathlib and the enclosure part of this repository use,
and restates Lemma 4, Weyl's inequality and Theorem 8 with the norm hypotheses of the documents.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

/-- `‖M z‖ ≤ ρ ‖z‖` for all `z` (that is, `‖M‖₂ ≤ ρ`) gives `|xᴴ M x| ≤ ρ ‖x‖²`. -/
theorem abs_qf_le_of_toEuclideanLin (M : Matrix n n 𝕜) (ρ : ℝ)
    (h : ∀ z : EuclideanSpace 𝕜 n, ‖Matrix.toEuclideanLin M z‖ ≤ ρ * ‖z‖) (x : n → 𝕜) :
    |qf M x| ≤ ρ * nsq x := by
  have hq : qf M x = RCLike.re (inner 𝕜 (WithLp.toLp 2 x : EuclideanSpace 𝕜 n)
      (Matrix.toEuclideanLin M (WithLp.toLp 2 x))) := by
    rw [EuclideanSpace.inner_eq_star_dotProduct]
    simp [qf, dotProduct_comm]
  have hn : nsq x = ‖(WithLp.toLp 2 x : EuclideanSpace 𝕜 n)‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq]
    simp [nsq]
  rw [hq, hn]
  calc |RCLike.re (inner 𝕜 (WithLp.toLp 2 x : EuclideanSpace 𝕜 n)
        (Matrix.toEuclideanLin M (WithLp.toLp 2 x)))|
      ≤ ‖inner 𝕜 (WithLp.toLp 2 x : EuclideanSpace 𝕜 n)
        (Matrix.toEuclideanLin M (WithLp.toLp 2 x))‖ := RCLike.abs_re_le_norm _
    _ ≤ ‖(WithLp.toLp 2 x : EuclideanSpace 𝕜 n)‖ *
        ‖Matrix.toEuclideanLin M (WithLp.toLp 2 x)‖ := norm_inner_le_norm _ _
    _ ≤ ‖(WithLp.toLp 2 x : EuclideanSpace 𝕜 n)‖ *
        (ρ * ‖(WithLp.toLp 2 x : EuclideanSpace 𝕜 n)‖) :=
        mul_le_mul_of_nonneg_left (h _) (norm_nonneg _)
    _ = ρ * ‖(WithLp.toLp 2 x : EuclideanSpace 𝕜 n)‖ ^ 2 := by ring

/-- **Lemma 4** with the spectral-norm hypothesis `‖L D Lᴴ - (A - σ I)‖₂ ≤ β` of
`manuscript_v4` (strict upper count). -/
theorem lemma4_opNorm {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σ β : ℝ) (L : Matrix n n 𝕜)
    (hL : IsUnit L) (d : n → ℝ)
    (hβ : ∀ z : EuclideanSpace 𝕜 n, ‖Matrix.toEuclideanLin
      (L * dg d * Lᴴ - (A - (σ : 𝕜) • (1 : Matrix n n 𝕜))) z‖ ≤ β * ‖z‖) :
    cntLt hA (σ - β) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLt hA (σ + β) := by
  have hdet := isUnit_det_conjTranspose hL
  exact lemma4 hA σ β L (Lᴴ)⁻¹ (mul_nonsing_inv _ hdet) (nonsing_inv_mul _ hdet) d
    (L * dg d * Lᴴ - (A - (σ : 𝕜) • (1 : Matrix n n 𝕜))) (by abel)
    (abs_qf_le_of_toEuclideanLin _ β hβ)

/-- **Weyl's inequality** with the hypothesis `‖A - B‖₂ ≤ ρ`, in counting form, both
directions. -/
theorem weyl_opNorm {A B : Matrix n n 𝕜} (hA : A.IsHermitian) (hB : B.IsHermitian) (ρ : ℝ)
    (h : ∀ z : EuclideanSpace 𝕜 n, ‖Matrix.toEuclideanLin (A - B) z‖ ≤ ρ * ‖z‖) (t : ℝ) :
    cntLt hA t ≤ cntLt hB (t + ρ) ∧ cntLt hB t ≤ cntLt hA (t + ρ) ∧
    cntLe hA t ≤ cntLe hB (t + ρ) ∧ cntLe hB t ≤ cntLe hA (t + ρ) := by
  have h1 := abs_qf_le_of_toEuclideanLin _ ρ h
  have h2 : ∀ x, |qf (B - A) x| ≤ ρ * nsq x := fun x => by
    rw [abs_qf_sub_comm]
    exact h1 x
  exact ⟨weyl_cntLt hA hB ρ h1 t, weyl_cntLt hB hA ρ h2 t, weyl_cntLe hA hB ρ h1 t,
    weyl_cntLe hB hA ρ h2 t⟩

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {m : ι → Type*} [∀ s, Fintype (m s)]
  [∀ s, DecidableEq (m s)] {κ : ι → Type*} [∀ s, Fintype (κ s)]

/-- **Theorem 8** with the spectral-norm hypotheses `‖H - H_spec‖₂ ≤ m_b` and
`‖Qᴴ H Q - B‖₂ ≤ ρ`. -/
theorem thm8_opNorm {Hspec Hb : Matrix n n 𝕜} (hHs : Hspec.IsHermitian) (hHb : Hb.IsHermitian)
    (mb : ℝ) (hm : ∀ z : EuclideanSpace 𝕜 n, ‖Matrix.toEuclideanLin (Hb - Hspec) z‖ ≤ mb * ‖z‖)
    (Q : Matrix n (Σ s, m s) 𝕜) (hQ1 : Qᴴ * Q = 1) (hQ2 : Q * Qᴴ = 1)
    (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian) (ρ : ℝ)
    (hρ : ∀ z : EuclideanSpace 𝕜 (Σ s, m s),
      ‖Matrix.toEuclideanLin (Qᴴ * Hb * Q - blockDiagonal' H) z‖ ≤ ρ * ‖z‖)
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
        (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 < x).card) :=
  thm8 hHs hHb mb (abs_qf_le_of_toEuclideanLin _ mb hm) Q hQ1 hQ2 H hH ρ
    (abs_qf_le_of_toEuclideanLin _ ρ hρ) c r φ hφ hrad Ecut β βmax hβ hν hcut

end CertifyMathlib.Counting
