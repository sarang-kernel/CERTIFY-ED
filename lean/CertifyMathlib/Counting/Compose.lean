import CertifyMathlib.Counting.Norms
import CertifyMathlib.Counting.Eigenspace
import CertifyMathlib.Enclosure.Cluster

/-!
# Theorem 8 with the polar factor of `W = [V_1 ⋯ V_m]`

Source: `docs/proofs/thm8_rho.md`, Step 3, and `docs/proofs/thm2_polar-factor.md` in the code
repository. `Assembly.lean` proves Theorem 8 for any unitary `Q` with
`|yᴴ (Qᴴ H Q - B) y| ≤ ρ ‖y‖²`. This file supplies that `Q`: the polar factor
`Q̂ = W (Wᴴ W)^{-1/2}` of the enclosure part (`CertifyMathlib.Enclosure.polarQ`), written as a
matrix (`polarMat`).

* `polarMat_conjTranspose_mul`, `polarMat_mul_conjTranspose`: for `η < 1`, `Q̂ᴴ Q̂ = 1`, and for a
  square `W` (hypothesis (T1), `Σ_s d_s = n`) also `Q̂ Q̂ᴴ = 1`.
* `polarMat_residual`: `‖H Q̂ - Q̂ B‖ ≤ D + (‖H‖ + ‖B‖) δ` with `δ = 1 - √(1 - η)` (polar-factor
  lemma (d), `CertifyMathlib.Enclosure.polarQ_residual_le`).
* `polarMat_qf`: `|yᴴ (Q̂ᴴ H Q̂ - B) y| ≤ ρ ‖y‖²`, since `Q̂ᴴ H Q̂ - B = Q̂ᴴ (H Q̂ - Q̂ B)` and `Q̂ᴴ`
  is an isometry.
* `norm_toEuclideanLin_le_frob`: `‖M‖₂ ≤ ‖M‖_F`; `frob_blockColumns`: the squared Frobenius norm
  of `H W - W B` is the sum over the sectors of `‖H V_s - V_s H_s‖_F²`.
* `thm8_polar`: Theorem 8 with the hypotheses of the document: `η = ‖Wᴴ W - I‖ < 1`,
  `‖H W - W B‖ ≤ D`, norm bounds for `H` and `B`, `‖H - H_spec‖ ≤ m_b`, and the sector
  hypotheses (T2), (T3). `thm8_sectors` takes `W` as the block columns `V_s` and `D` as a bound
  on `(Σ_s ‖H V_s - V_s H_s‖_F²)^{1/2}`, as the code does.

Norms of operators are given by their defining bound (`‖M z‖ ≤ c ‖z‖` for all `z`), so any upper
bound on the spectral norm may be used.
-/

namespace CertifyMathlib.Counting

open Matrix
open scoped InnerProductSpace

variable {𝕜 : Type*} [RCLike 𝕜]

section Iso

variable {n p : Type*} [Fintype n] [DecidableEq n] [Fintype p] [DecidableEq p]

theorem toEuclideanLin_one_apply (z : EuclideanSpace 𝕜 p) :
    toEuclideanLin (1 : Matrix p p 𝕜) z = z := by
  show Matrix.toLpLin 2 2 (1 : Matrix p p 𝕜) z = z
  rw [Matrix.toLpLin_one]
  rfl

omit [Fintype n] [DecidableEq n] in
theorem toEuclideanLin_mul_apply {o : Type*} [Fintype o] [DecidableEq o] (A : Matrix n p 𝕜)
    (B : Matrix p o 𝕜) (z : EuclideanSpace 𝕜 o) :
    toEuclideanLin (A * B) z = toEuclideanLin A (toEuclideanLin B z) := by
  show Matrix.toLpLin 2 2 (A * B) z = Matrix.toLpLin 2 2 A (Matrix.toLpLin 2 2 B z)
  rw [Matrix.toLpLin_mul_same]
  rfl

/-- A matrix whose map is an isometry satisfies `Qᴴ Q = 1`. -/
theorem conjTranspose_mul_self_of_isometry (Q : Matrix n p 𝕜)
    (h : ∀ z : EuclideanSpace 𝕜 p, ‖toEuclideanLin Q z‖ = ‖z‖) : Qᴴ * Q = 1 := by
  have hi := (LinearMap.norm_map_iff_inner_map_map (toEuclideanLin Q)).mp h
  apply Matrix.toEuclideanLin.injective
  refine LinearMap.ext fun u => ext_inner_right 𝕜 fun v => ?_
  rw [← Enclosure.inner_toEuclideanLin_gram, hi, toEuclideanLin_one_apply]

/-- A square isometry also satisfies `Q Qᴴ = 1`. -/
theorem mul_conjTranspose_self_of_isometry (Q : Matrix n p 𝕜)
    (hcard : Fintype.card p = Fintype.card n)
    (h : ∀ z : EuclideanSpace 𝕜 p, ‖toEuclideanLin Q z‖ = ‖z‖) : Q * Qᴴ = 1 :=
  (Matrix.mul_eq_one_comm_of_equiv (Fintype.card_eq.mp hcard).some).mp
    (conjTranspose_mul_self_of_isometry Q h)

/-- If `Q Qᴴ = 1` then `Qᴴ` is an isometry. -/
theorem norm_conjTranspose_apply (Q : Matrix n p 𝕜) (hQ : Q * Qᴴ = 1)
    (w : EuclideanSpace 𝕜 n) : ‖toEuclideanLin Qᴴ w‖ = ‖w‖ :=
  (LinearMap.norm_map_iff_inner_map_map (toEuclideanLin Qᴴ)).mpr (fun u v => by
    rw [Enclosure.inner_toEuclideanLin_gram, conjTranspose_conjTranspose, hQ,
      toEuclideanLin_one_apply]) w

omit [DecidableEq n] in
/-- `‖M z‖ ≤ ‖M‖_F ‖z‖`. -/
theorem norm_toEuclideanLin_le_frob (M : Matrix n p 𝕜) (z : EuclideanSpace 𝕜 p) :
    ‖toEuclideanLin M z‖ ≤ √(frob M) * ‖z‖ := by
  have key : ‖toEuclideanLin M z‖ ^ 2 ≤ frob M * ‖z‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, frob, Finset.sum_mul]
    refine Finset.sum_le_sum fun i _ => ?_
    have hi : (toEuclideanLin M z) i = ∑ j, M i j * z j := by
      simp [Matrix.toLpLin_apply, Matrix.mulVec, dotProduct]
    have hrow : ‖(toEuclideanLin M z) i‖ ≤ ∑ j, ‖M i j‖ * ‖z j‖ := by
      rw [hi]
      refine (norm_sum_le _ _).trans (le_of_eq ?_)
      simp only [norm_mul]
    calc ‖(toEuclideanLin M z) i‖ ^ 2 ≤ (∑ j, ‖M i j‖ * ‖z j‖) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hrow 2
      _ ≤ (∑ j, ‖M i j‖ ^ 2) * ∑ j, ‖z j‖ ^ 2 := Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  have hf : 0 ≤ frob M := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
  calc ‖toEuclideanLin M z‖ = √(‖toEuclideanLin M z‖ ^ 2) := (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ √(frob M * ‖z‖ ^ 2) := Real.sqrt_le_sqrt key
    _ = √(frob M) * ‖z‖ := by rw [Real.sqrt_mul hf, Real.sqrt_sq (norm_nonneg _)]

end Iso

section Polar

variable {n p : Type*} [Fintype n] [DecidableEq n] [Fintype p] [DecidableEq p]

/-- The polar factor `Q̂ = W (Wᴴ W)^{-1/2}` of the enclosure part, as a matrix. -/
noncomputable def polarMat (W : Matrix n p 𝕜) : Matrix n p 𝕜 :=
  Matrix.toEuclideanLin.symm
    (Enclosure.polarQ (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self W))
      (toEuclideanLin W))

omit [DecidableEq n] in
theorem toEuclideanLin_polarMat (W : Matrix n p 𝕜) :
    toEuclideanLin (polarMat W) = Enclosure.polarQ
      (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self W)) (toEuclideanLin W) :=
  LinearEquiv.apply_symm_apply _ _

variable {W : Matrix n p 𝕜} {η : ℝ} (hη1 : η < 1)
  (hη : ∀ z, ‖toEuclideanLin (Wᴴ * W) z - z‖ ≤ η * ‖z‖)
include hη1 hη

/-- Polar-factor lemma (a): `Q̂` is an isometry. -/
theorem polarMat_norm (z : EuclideanSpace 𝕜 p) : ‖toEuclideanLin (polarMat W) z‖ = ‖z‖ := by
  rw [toEuclideanLin_polarMat]
  exact Enclosure.polarQ_norm
    (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self W)) hη1 hη
    (Enclosure.inner_toEuclideanLin_gram W) z

theorem polarMat_conjTranspose_mul : (polarMat W)ᴴ * polarMat W = 1 :=
  conjTranspose_mul_self_of_isometry _ (polarMat_norm hη1 hη)

theorem polarMat_mul_conjTranspose (hcard : Fintype.card p = Fintype.card n) :
    polarMat W * (polarMat W)ᴴ = 1 :=
  mul_conjTranspose_self_of_isometry _ hcard (polarMat_norm hη1 hη)

/-- Polar-factor lemma (d) for matrices: `‖A Q̂ - Q̂ Λ‖ ≤ ‖A W - W Λ‖ + (‖A‖ + ‖Λ‖) δ`. -/
theorem polarMat_residual (hη0 : 0 ≤ η) {A : Matrix n n 𝕜} {Λ : Matrix p p 𝕜} {nR nA nΛ : ℝ}
    (hR : ∀ z, ‖toEuclideanLin (A * W - W * Λ) z‖ ≤ nR * ‖z‖) (hnA0 : 0 ≤ nA)
    (hnA : ∀ v, ‖toEuclideanLin A v‖ ≤ nA * ‖v‖)
    (hnΛ : ∀ z, ‖toEuclideanLin Λ z‖ ≤ nΛ * ‖z‖) (z : EuclideanSpace 𝕜 p) :
    ‖toEuclideanLin (A * polarMat W - polarMat W * Λ) z‖ ≤
      (nR + (nA + nΛ) * (1 - √(1 - η))) * ‖z‖ := by
  have hR' : ∀ z, ‖toEuclideanLin A (toEuclideanLin W z) - toEuclideanLin W (toEuclideanLin Λ z)‖
      ≤ nR * ‖z‖ := fun z => by
    have := hR z
    rwa [map_sub, LinearMap.sub_apply, toEuclideanLin_mul_apply, toEuclideanLin_mul_apply]
      at this
  have h := Enclosure.polarQ_residual_le
    (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self W)) hη1 hη
    (Enclosure.inner_toEuclideanLin_gram W) hη0 hR' hnA0 hnA hnΛ z
  rw [← toEuclideanLin_polarMat] at h
  rwa [map_sub, LinearMap.sub_apply, toEuclideanLin_mul_apply, toEuclideanLin_mul_apply]

/-- The hypothesis of `thm8`: `|yᴴ (Q̂ᴴ A Q̂ - Λ) y| ≤ ρ ‖y‖²` from `‖A Q̂ - Q̂ Λ‖ ≤ ρ`. -/
theorem polarMat_qf (hcard : Fintype.card p = Fintype.card n) {A : Matrix n n 𝕜}
    {Λ : Matrix p p 𝕜} {ρ : ℝ}
    (hres : ∀ z, ‖toEuclideanLin (A * polarMat W - polarMat W * Λ) z‖ ≤ ρ * ‖z‖) (y : p → 𝕜) :
    |qf ((polarMat W)ᴴ * A * polarMat W - Λ) y| ≤ ρ * nsq y := by
  have h1 : (polarMat W)ᴴ * polarMat W = 1 := polarMat_conjTranspose_mul hη1 hη
  have h2 : polarMat W * (polarMat W)ᴴ = 1 := polarMat_mul_conjTranspose hη1 hη hcard
  have he : (polarMat W)ᴴ * A * polarMat W - Λ =
      (polarMat W)ᴴ * (A * polarMat W - polarMat W * Λ) := by
    rw [Matrix.mul_sub, ← Matrix.mul_assoc, ← Matrix.mul_assoc, h1, Matrix.one_mul]
  rw [he]
  refine abs_qf_le_of_toEuclideanLin _ ρ (fun z => ?_) y
  rw [toEuclideanLin_mul_apply, norm_conjTranspose_apply _ h2]
  exact hres z

end Polar

section Sectors

variable {n : Type*} [Fintype n] [DecidableEq n]
variable {ι : Type*} [Fintype ι] [DecidableEq ι] {m : ι → Type*} [∀ s, Fintype (m s)]
  [∀ s, DecidableEq (m s)] {κ : ι → Type*} [∀ s, Fintype (κ s)]

/-- **Theorem 8** with the polar-factor lemma: the hypotheses are those of
`docs/proofs/thm8_rho.md`, with `ρ = D + (‖H‖ + ‖B‖) δ` and `δ = 1 - √(1 - η)`. -/
theorem thm8_polar {Hspec Hb : Matrix n n 𝕜} (hHs : Hspec.IsHermitian) (hHb : Hb.IsHermitian)
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
    (hcut : ∀ s l, c s l + r < Ecut - βmax) :
    cntLt hHs (Ecut - βmax - (D + (nH + nB) * (1 - √(1 - η)) + mb)) ≤
      Fintype.card (Σ s, κ s) ∧
    (∀ x, (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 ≤ x).card ≤
      cntLe hHs (x + r + (D + (nH + nB) * (1 - √(1 - η)) + mb))) ∧
    (∀ x, x - r ≤ Ecut - βmax →
      cntLt hHs (x - r - (D + (nH + nB) * (1 - √(1 - η)) + mb)) ≤
        (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 < x).card) :=
  thm8 hHs hHb mb (abs_qf_le_of_toEuclideanLin _ mb hm) (polarMat W)
    (polarMat_conjTranspose_mul hη1 hη) (polarMat_mul_conjTranspose hη1 hη hcard) H hH _
    (polarMat_qf hη1 hη hcard (polarMat_residual hη1 hη hη0 hR hnH0 hnH hnB))
    c r φ hφ hrad Ecut β βmax hβ hν hcut

/-- `W = [V_1 ⋯ V_m]`: the matrix with the block columns `V_s`. -/
def blockColumns (V : ∀ s, Matrix n (m s) 𝕜) : Matrix n (Σ s, m s) 𝕜 :=
  Matrix.of fun i q => V q.1 i q.2

omit [DecidableEq n] [∀ s, DecidableEq (m s)] in
/-- The block columns of `H W - W B` are `H V_s - V_s H_s`, so the squared Frobenius norms add
up over the sectors. -/
theorem frob_blockColumns (Hb : Matrix n n 𝕜) (V : ∀ s, Matrix n (m s) 𝕜)
    (H : ∀ s, Matrix (m s) (m s) 𝕜) :
    frob (Hb * blockColumns V - blockColumns V * blockDiagonal' H) =
      ∑ s, frob (Hb * V s - V s * H s) := by
  have hentry : ∀ i (q : Σ s, m s),
      (Hb * blockColumns V - blockColumns V * blockDiagonal' H) i q =
        (Hb * V q.1 - V q.1 * H q.1) i q.2 := by
    intro i q
    simp only [Matrix.sub_apply, Matrix.mul_apply, blockColumns, Matrix.of_apply]
    congr 1
    rw [Fintype.sum_sigma]
    rw [Finset.sum_eq_single q.1]
    · refine Finset.sum_congr rfl fun k _ => ?_
      rw [blockDiagonal'_apply_eq]
    · intro t _ ht
      refine Finset.sum_eq_zero fun k _ => ?_
      rw [blockDiagonal'_apply_ne _ _ _ ht, mul_zero]
    · intro h
      exact absurd (Finset.mem_univ _) h
  unfold frob
  rw [Finset.sum_comm]
  rw [Fintype.sum_sigma]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ => ?_
  rw [hentry i ⟨s, k⟩]

/-- **Theorem 8** with `W` given by the sector bases `V_s` and `D ≥ (Σ_s ‖H V_s - V_s H_s‖_F²)^{1/2}`,
as `compose.certify_low_energy` computes it. -/
theorem thm8_sectors {Hspec Hb : Matrix n n 𝕜} (hHs : Hspec.IsHermitian) (hHb : Hb.IsHermitian)
    (mb : ℝ) (hm : ∀ z : EuclideanSpace 𝕜 n, ‖toEuclideanLin (Hb - Hspec) z‖ ≤ mb * ‖z‖)
    (V : ∀ s, Matrix n (m s) 𝕜) (hcard : Fintype.card (Σ s, m s) = Fintype.card n) {η : ℝ}
    (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin ((blockColumns V)ᴴ * blockColumns V) z - z‖ ≤ η * ‖z‖)
    (H : ∀ s, Matrix (m s) (m s) 𝕜) (hH : ∀ s, (H s).IsHermitian) {D nH nB : ℝ}
    (hD : √(∑ s, frob (Hb * V s - V s * H s)) ≤ D) (hnH0 : 0 ≤ nH)
    (hnH : ∀ v, ‖toEuclideanLin Hb v‖ ≤ nH * ‖v‖)
    (hnB : ∀ z, ‖toEuclideanLin (blockDiagonal' H) z‖ ≤ nB * ‖z‖)
    (c : ∀ s, κ s → ℝ) (r : ℝ) (φ : ∀ s, κ s → m s) (hφ : ∀ s, Function.Injective (φ s))
    (hrad : ∀ s l, |(hH s).eigenvalues (φ s l) - c s l| ≤ r) (Ecut : ℝ) (β : ι → ℝ)
    (βmax : ℝ) (hβ : ∀ s, β s ≤ βmax)
    (hν : ∀ s, cntLt (hH s) (Ecut - β s) ≤ Fintype.card (κ s))
    (hcut : ∀ s l, c s l + r < Ecut - βmax) :
    cntLt hHs (Ecut - βmax - (D + (nH + nB) * (1 - √(1 - η)) + mb)) ≤
      Fintype.card (Σ s, κ s) ∧
    (∀ x, (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 ≤ x).card ≤
      cntLe hHs (x + r + (D + (nH + nB) * (1 - √(1 - η)) + mb))) ∧
    (∀ x, x - r ≤ Ecut - βmax →
      cntLt hHs (x - r - (D + (nH + nB) * (1 - √(1 - η)) + mb)) ≤
        (Finset.univ.filter fun q : (Σ s, κ s) => c q.1 q.2 < x).card) := by
  have hR : ∀ z, ‖toEuclideanLin (Hb * blockColumns V - blockColumns V * blockDiagonal' H) z‖ ≤
      D * ‖z‖ := fun z => by
    refine (norm_toEuclideanLin_le_frob _ z).trans ?_
    rw [frob_blockColumns]
    exact mul_le_mul_of_nonneg_right hD (norm_nonneg _)
  exact thm8_polar hHs hHb mb hm (blockColumns V) hcard hη0 hη1 hη H hH hR hnH0 hnH hnB
    c r φ hφ hrad Ecut β βmax hβ hν hcut

end Sectors

end CertifyMathlib.Counting
