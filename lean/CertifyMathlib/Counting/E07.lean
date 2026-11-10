import CertifyMathlib.Counting.Theorem6

/-!
# E-07: Lemma 1 (complete), Lemma 2, Proposition 1 (`s_DK`) and Lemma 3 (alignment)

Source: `docs/proofs/E-07_davis_kahan.md` in the code repository. Notation: `X` (`n × g`) the
claimed vectors, `η` with `‖Xᴴ X - I‖ ≤ η < 1`, `Q̂ = X M⁻¹` the polar factor, `M = (Xᴴ X)^{1/2}`,
`δ = 1 - √(1 - η)`, `H' = Q̂ᴴ A Q̂`, `R = A X - X Λ`, `P` the spectral projector of `A` on the
eigen-indices `T`.

* `frob_eq_sum_cols`, `frob_mul_le`, `frob_conjTranspose`, `frob_mul_right_le`: the squared
  Frobenius norm as a sum over columns, and its behaviour under multiplication by a bounded matrix.
* `e07_lemma1_sharp`, `e07_lemma1`: Lemma 1, `δ'² ‖(I - P) Q̂‖_F² ≤ ‖A Q̂ - Q̂ Λ‖_F²` for any Hermitian `Λ` whose
  eigenvalues are at distance at least `δ'` from the eigenvalues of `A` outside `T` (the projector
  identity `(I - P)(A Q̂ - Q̂ Λ) = A Y - Y Λ` with `Y = (I - P) Q̂`, then `frob_sylvester`).
* `polar_identity_i`, `e07_lemma2_i`: Lemma 2 (i), `A Q̂ - Q̂ H' = (I - Q̂ Q̂ᴴ) R M⁻¹` and
  `‖A Q̂ - Q̂ H'‖_F² ≤ ‖R‖_F² / (1 - η)`.
* `polar_identity_ii`, `e07_lemma2_ii`: Lemma 2 (ii),
  `H' - Λ = M⁻¹ Xᴴ R M⁻¹ + (M - I)(Λ - c I) M⁻¹ - (Λ - c I)(M - I) M⁻¹` and
  `‖H' - Λ‖ ≤ ‖Xᴴ R‖ / (1 - η) + 2 δ w / √(1 - η)` when `‖Λ - c I‖ ≤ w` (so `2 w` is the spread).
* `e07_prop1`, `e07_prop1_proj`: Proposition 1, `‖(I - P) Q̂‖ ≤ ‖R‖_F / (√(1 - η) (dist - w'))`,
  and with `proj_sub_le` the bound `‖Q̂ Q̂ᴴ - P‖ ≤ s_DK` for `#T = g` and `s_DK < 1`.
* `thm6_ritz`: Theorem 6 with the separation measured from the Ritz values (the wording of
  `thm:dk` in the manuscript), from `sinTheta` with `Λ = H'` and `‖A Q̂ - Q̂ H'‖ ≤ ‖A Q̂ - Q̂ Λ‖`.
* `e07_lemma3`: Lemma 3, for isometries `Q₁`, `Q₂` of equal width with `‖(I - Q₁Q₁ᴴ) Q₂‖ ≤ s < 1`,
  the polar factor `U` of `Q₁ᴴ Q₂` is unitary and `‖Q₂ - Q₁ U‖ ≤ s + s²`.
-/

namespace CertifyMathlib.Counting

open Matrix
open scoped InnerProductSpace

variable {𝕜 : Type*} [RCLike 𝕜]

section Frob

variable {n p : Type*} [Fintype n] [DecidableEq n] [Fintype p] [DecidableEq p]

omit [Fintype n] [DecidableEq n] in
theorem toEuclideanLin_single_apply (M : Matrix n p 𝕜) (j : p) (i : n) :
    (toEuclideanLin M (EuclideanSpace.single j 1)) i = M i j := by
  simp [Matrix.toLpLin_apply]

omit [DecidableEq n] in
theorem frob_eq_sum_cols (M : Matrix n p 𝕜) :
    frob M = ∑ j, ‖toEuclideanLin M (EuclideanSpace.single j 1)‖ ^ 2 := by
  simp only [EuclideanSpace.norm_sq_eq, toEuclideanLin_single_apply, frob]
  exact Finset.sum_comm

omit [DecidableEq n] [DecidableEq p] in
theorem frob_conjTranspose (M : Matrix n p 𝕜) : frob Mᴴ = frob M := by
  simp only [frob, conjTranspose_apply, norm_star]
  exact Finset.sum_comm

theorem frob_mul_le {m : Type*} [Fintype m] [DecidableEq m] (K : Matrix m n 𝕜) {c : ℝ}
    (_hc : 0 ≤ c) (hK : ∀ v, ‖toEuclideanLin K v‖ ≤ c * ‖v‖) (M : Matrix n p 𝕜) :
    frob (K * M) ≤ c ^ 2 * frob M := by
  rw [frob_eq_sum_cols, frob_eq_sum_cols, Finset.mul_sum]
  refine Finset.sum_le_sum fun j _ => ?_
  rw [toEuclideanLin_mul_apply, ← mul_pow]
  exact pow_le_pow_left₀ (norm_nonneg _) (hK _) 2

theorem frob_mul_right_le {m : Type*} [Fintype m] [DecidableEq m] (M : Matrix m n 𝕜)
    (K : Matrix n n 𝕜) {c : ℝ} (hc : 0 ≤ c) (hK : ∀ v, ‖toEuclideanLin Kᴴ v‖ ≤ c * ‖v‖) :
    frob (M * K) ≤ c ^ 2 * frob M := by
  rw [← frob_conjTranspose, conjTranspose_mul, ← frob_conjTranspose M]
  exact frob_mul_le _ hc hK _

/-- `a ≤ b` from `a² ≤ b²` for non-negative reals. -/
theorem le_of_sq_le_sq' {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (h : a ^ 2 ≤ b ^ 2) : a ≤ b := by
  calc a = √(a ^ 2) := (Real.sqrt_sq ha).symm
    _ ≤ √(b ^ 2) := Real.sqrt_le_sqrt h
    _ = b := Real.sqrt_sq hb

/-- `‖I - Q Qᴴ‖ ≤ 1` for an isometry `Q`. -/
theorem norm_one_sub_proj_le (Q : Matrix n p 𝕜) (hQ : Qᴴ * Q = 1) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (1 - Q * Qᴴ) v‖ ≤ 1 * ‖v‖ := by
  have h := norm_sq_proj Q hQ v
  have hsq : ‖toEuclideanLin (1 - Q * Qᴴ) v‖ ^ 2 ≤ ‖v‖ ^ 2 := by
    nlinarith [sq_nonneg ‖toEuclideanLin Qᴴ v‖]
  rw [one_mul]
  exact le_of_sq_le_sq' (norm_nonneg _) (norm_nonneg _) hsq

end Frob

section Lemma1

variable {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g] [DecidableEq g]
variable {A : Matrix n n 𝕜} (hA : A.IsHermitian)
include hA

omit [Fintype g] [DecidableEq g] in
theorem mul_one_sub_specProj_comm (T : Finset n) :
    A * (1 - specProj hA T) = (1 - specProj hA T) * A := by
  have h := specProj_comm hA T 0
  simpa using h

omit [DecidableEq g] in
/-- The coordinates of `(I - P) Y` on the eigenvectors with index in `T` vanish. -/
theorem coord_one_sub_specProj (T : Finset n) (Y : Matrix n g 𝕜) (V : Matrix g g 𝕜) (i : n)
    (j : g) (hi : i ∈ T) :
    ((hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ * ((1 - specProj hA T) * Y) * V : Matrix n g 𝕜) i j =
      0 := by
  have hU2 := conjTranspose_mul_unitary hA.eigenvectorUnitary
  rw [one_sub_specProj, spec]
  have e : (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ * ((hA.eigenvectorUnitary : Matrix n n 𝕜) *
      dg (fun i => 1 - ind T i) * (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ * Y) * V =
      dg (fun i => 1 - ind T i) * ((hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ * Y * V) := by
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ
      (hA.eigenvectorUnitary : Matrix n n 𝕜), hU2, Matrix.one_mul]
  rw [e]
  simp [dg, Matrix.diagonal_mul, ind, hi]

/-- **E-07, Lemma 1**, first inequality: `δ'² ‖(I - P) Q̂‖_F² ≤ ‖(I - P)(A Q̂ - Q̂ Λ)‖_F²` (with
`Λ = H'` this is `‖(I - P) Q̂‖_F ≤ ‖(I - P) R'‖_F / δ'`), through the projector identity. -/
theorem e07_lemma1_sharp (T : Finset n) (Qh : Matrix n g 𝕜) {Λ : Matrix g g 𝕜}
    (hΛ : Λ.IsHermitian) {δ' : ℝ} (hδ : 0 ≤ δ')
    (hsep : ∀ i, i ∉ T → ∀ j, δ' ≤ |hA.eigenvalues i - hΛ.eigenvalues j|) :
    δ' ^ 2 * frob ((1 - specProj hA T) * Qh) ≤
      frob ((1 - specProj hA T) * (A * Qh - Qh * Λ)) := by
  have hid : A * ((1 - specProj hA T) * Qh) - ((1 - specProj hA T) * Qh) * Λ =
      (1 - specProj hA T) * (A * Qh - Qh * Λ) := by
    rw [← Matrix.mul_assoc A, mul_one_sub_specProj_comm hA T, Matrix.mul_sub,
      Matrix.mul_assoc, Matrix.mul_assoc]
  have h1 := frob_sylvester hA hΛ ((1 - specProj hA T) * Qh) δ' hδ (fun i j hne => by
    by_cases hi : i ∈ T
    · exact absurd (coord_one_sub_specProj hA T Qh (hΛ.eigenvectorUnitary : Matrix g g 𝕜) i j hi)
        hne
    · exact hsep i hi j)
  rw [hid] at h1
  exact h1

/-- **E-07, Lemma 1** (Frobenius sin Θ), with the projector identity:
`δ'² ‖(I - P) Q̂‖_F² ≤ ‖A Q̂ - Q̂ Λ‖_F²`. -/
theorem e07_lemma1 (T : Finset n) (Qh : Matrix n g 𝕜) {Λ : Matrix g g 𝕜} (hΛ : Λ.IsHermitian)
    {δ' : ℝ} (hδ : 0 ≤ δ')
    (hsep : ∀ i, i ∉ T → ∀ j, δ' ≤ |hA.eigenvalues i - hΛ.eigenvalues j|) :
    δ' ^ 2 * frob ((1 - specProj hA T) * Qh) ≤ frob (A * Qh - Qh * Λ) := by
  refine (e07_lemma1_sharp hA T Qh hΛ hδ hsep).trans ?_
  have h2 := frob_mul_le (1 - specProj hA T) zero_le_one (fun v => by
    rw [one_mul]
    exact norm_one_sub_specProj_le hA T v) (A * Qh - Qh * Λ)
  simpa using h2

end Lemma1

section Lemma2

variable {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g] [DecidableEq g]

/-- `A Q - Q (Qᴴ A Q) = (I - Q Qᴴ)(A (Q M) - (Q M) Λ) M⁻¹` for an isometry `Q` and
`M M⁻¹ = 1`. -/
theorem polar_identity_i (A : Matrix n n 𝕜) (Q : Matrix n g 𝕜) (M Mi Λ : Matrix g g 𝕜)
    (hQ : Qᴴ * Q = 1) (hMMi : M * Mi = 1) :
    A * Q - Q * (Qᴴ * A * Q) = (1 - Q * Qᴴ) * (A * (Q * M) - (Q * M) * Λ) * Mi := by
  have h0 : (1 - Q * Qᴴ) * Q = 0 := by
    rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, hQ, Matrix.mul_one, sub_self]
  have e1 : (1 - Q * Qᴴ) * (A * (Q * M)) * Mi = (1 - Q * Qᴴ) * (A * Q) := by
    simp only [Matrix.mul_assoc]
    rw [hMMi, Matrix.mul_one]
  have e2 : (1 - Q * Qᴴ) * ((Q * M) * Λ) * Mi = 0 := by
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc (1 - Q * Qᴴ) Q, h0, Matrix.zero_mul]
  rw [Matrix.mul_sub (1 - Q * Qᴴ) (A * (Q * M)) ((Q * M) * Λ),
    Matrix.sub_mul ((1 - Q * Qᴴ) * (A * (Q * M))) ((1 - Q * Qᴴ) * ((Q * M) * Λ)) Mi, e1, e2,
    sub_zero, Matrix.sub_mul (1 : Matrix n n 𝕜) (Q * Qᴴ) (A * Q), Matrix.one_mul]
  simp only [Matrix.mul_assoc]

/-- **E-07, Lemma 2 (i)**: `‖A Q̂ - Q̂ H'‖_F² ≤ ‖R‖_F² / (1 - η)`. -/
theorem e07_lemma2_i {X : Matrix n g 𝕜} {η : ℝ} (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) (A : Matrix n n 𝕜)
    (Λ : Matrix g g 𝕜) :
    frob (A * polarMat X - polarMat X * ((polarMat X)ᴴ * A * polarMat X)) ≤
      ((√(1 - η))⁻¹) ^ 2 * frob (A * X - X * Λ) := by
  have hQ := polarMat_conjTranspose_mul hη1 hη
  rw [polar_identity_i A (polarMat X) (polarSqrt X) (polarSqrtInv X) Λ hQ
    (polarSqrt_mul_inv hη1 hη), polarMat_mul_polarSqrt hη1 hη]
  have hk : 0 ≤ (√(1 - η))⁻¹ := inv_nonneg.mpr (Real.sqrt_nonneg _)
  calc frob ((1 - polarMat X * (polarMat X)ᴴ) * (A * X - X * Λ) * polarSqrtInv X)
      ≤ ((√(1 - η))⁻¹) ^ 2 * frob ((1 - polarMat X * (polarMat X)ᴴ) * (A * X - X * Λ)) :=
        frob_mul_right_le _ _ hk (fun v => by
          rw [polarSqrtInv_conjTranspose]
          exact polarSqrtInv_le hη1 hη v)
    _ ≤ ((√(1 - η))⁻¹) ^ 2 * (1 ^ 2 * frob (A * X - X * Λ)) :=
        mul_le_mul_of_nonneg_left (frob_mul_le _ zero_le_one (norm_one_sub_proj_le _ hQ) _)
          (sq_nonneg _)
    _ = ((√(1 - η))⁻¹) ^ 2 * frob (A * X - X * Λ) := by ring

omit [DecidableEq n] in
/-- `H' - Λ = M⁻¹ Xᴴ R M⁻¹ + ((M - I)(Λ - c I) M⁻¹ - (Λ - c I)(M - I) M⁻¹)` for
`Q̂ = X M⁻¹`. -/
theorem polar_identity_ii (A : Matrix n n 𝕜) (X : Matrix n g 𝕜) (M Mi Λ : Matrix g g 𝕜) (c : 𝕜)
    (hMi : Miᴴ = Mi) (hMiM : Mi * M = 1) (hMMi : M * Mi = 1) (hXX : Xᴴ * X = M * M) :
    (X * Mi)ᴴ * A * (X * Mi) - Λ = Mi * (Xᴴ * (A * X - X * Λ)) * Mi +
      ((M - 1) * (Λ - c • (1 : Matrix g g 𝕜)) * Mi - (Λ - c • (1 : Matrix g g 𝕜)) * (M - 1) * Mi) := by
  have e2 : (M - 1) * (Λ - c • (1 : Matrix g g 𝕜)) * Mi -
      (Λ - c • (1 : Matrix g g 𝕜)) * (M - 1) * Mi = M * Λ * Mi - Λ := by
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one, Matrix.smul_mul,
      Matrix.mul_smul, Matrix.mul_assoc, hMMi, smul_sub]
    abel
  have h3 : Mi * (Xᴴ * (X * Λ)) * Mi = M * Λ * Mi := by
    rw [← Matrix.mul_assoc Xᴴ X Λ, hXX, ← Matrix.mul_assoc Mi (M * M) Λ,
      ← Matrix.mul_assoc Mi M M, hMiM, Matrix.one_mul]
  have e1 : Mi * (Xᴴ * (A * X - X * Λ)) * Mi = Mi * Xᴴ * A * (X * Mi) - M * Λ * Mi := by
    rw [Matrix.mul_sub Xᴴ, Matrix.mul_sub Mi, Matrix.sub_mul, h3]
    simp only [Matrix.mul_assoc]
  rw [conjTranspose_mul, hMi, e1, e2]
  abel

/-- `‖P₁ P₂ P₃‖ ≤ a b c`. -/
theorem norm_mul3_le (P1 P2 P3 : Matrix g g 𝕜) {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (h1 : ∀ v, ‖toEuclideanLin P1 v‖ ≤ a * ‖v‖) (h2 : ∀ v, ‖toEuclideanLin P2 v‖ ≤ b * ‖v‖)
    (h3 : ∀ v, ‖toEuclideanLin P3 v‖ ≤ c * ‖v‖) (z : EuclideanSpace 𝕜 g) :
    ‖toEuclideanLin (P1 * P2 * P3) z‖ ≤ a * (b * (c * ‖z‖)) := by
  rw [toEuclideanLin_mul_apply, toEuclideanLin_mul_apply]
  exact (h1 _).trans (mul_le_mul_of_nonneg_left ((h2 _).trans
    (mul_le_mul_of_nonneg_left (h3 _) hb)) ha)

theorem norm_add_sub_le (P Q1 Q2 : Matrix g g 𝕜) (z : EuclideanSpace 𝕜 g) :
    ‖toEuclideanLin (P + (Q1 - Q2)) z‖ ≤
      ‖toEuclideanLin P z‖ + ‖toEuclideanLin Q1 z‖ + ‖toEuclideanLin Q2 z‖ := by
  rw [map_add, map_sub, LinearMap.add_apply, LinearMap.sub_apply]
  calc ‖toEuclideanLin P z + (toEuclideanLin Q1 z - toEuclideanLin Q2 z)‖
      ≤ ‖toEuclideanLin P z‖ + ‖toEuclideanLin Q1 z - toEuclideanLin Q2 z‖ := norm_add_le _ _
    _ ≤ ‖toEuclideanLin P z‖ + (‖toEuclideanLin Q1 z‖ + ‖toEuclideanLin Q2 z‖) :=
        add_le_add le_rfl (norm_sub_le _ _)
    _ = _ := by ring

/-- **E-07, Lemma 2 (ii)**: `‖H' - Λ‖ ≤ ‖Xᴴ R‖ / (1 - η) + 2 δ w / √(1 - η)` when
`‖Λ - c I‖ ≤ w`. -/
theorem e07_lemma2_ii {X : Matrix n g 𝕜} {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) (A : Matrix n n 𝕜)
    (Λ : Matrix g g 𝕜) {c w nXR : ℝ} (hw : 0 ≤ w) (hnXR : 0 ≤ nXR)
    (hΛ : ∀ z, ‖toEuclideanLin (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) z‖ ≤ w * ‖z‖)
    (hXR : ∀ z, ‖toEuclideanLin (Xᴴ * (A * X - X * Λ)) z‖ ≤ nXR * ‖z‖)
    (z : EuclideanSpace 𝕜 g) :
    ‖toEuclideanLin ((polarMat X)ᴴ * A * polarMat X - Λ) z‖ ≤
      (nXR * ((√(1 - η))⁻¹) ^ 2 + 2 * (1 - √(1 - η)) * w * (√(1 - η))⁻¹) * ‖z‖ := by
  rw [polarMat_eq, polar_identity_ii A X (polarSqrt X) (polarSqrtInv X) Λ (c : 𝕜)
    (polarSqrtInv_conjTranspose X) (polarSqrtInv_mul hη1 hη) (polarSqrt_mul_inv hη1 hη)
    (polarSqrt_mul_self hη1 hη).symm]
  have hk : 0 ≤ (√(1 - η))⁻¹ := inv_nonneg.mpr (Real.sqrt_nonneg _)
  have hδ : 0 ≤ 1 - √(1 - η) := Enclosure.delta_nonneg hη0
  have t1 := norm_mul3_le (polarSqrtInv X) (Xᴴ * (A * X - X * Λ)) (polarSqrtInv X) hk hnXR
    (polarSqrtInv_le hη1 hη) hXR (polarSqrtInv_le hη1 hη) z
  have t2 := norm_mul3_le (polarSqrt X - 1) (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) (polarSqrtInv X)
    hδ hw (polarSqrt_sub_le hη1 hη hη0) hΛ (polarSqrtInv_le hη1 hη) z
  have t3 := norm_mul3_le (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) (polarSqrt X - 1) (polarSqrtInv X)
    hw hδ hΛ (polarSqrt_sub_le hη1 hη hη0) (polarSqrtInv_le hη1 hη) z
  refine (norm_add_sub_le _ _ _ z).trans ?_
  calc _ ≤ (√(1 - η))⁻¹ * (nXR * ((√(1 - η))⁻¹ * ‖z‖)) +
        (1 - √(1 - η)) * (w * ((√(1 - η))⁻¹ * ‖z‖)) +
        w * ((1 - √(1 - η)) * ((√(1 - η))⁻¹ * ‖z‖)) := add_le_add (add_le_add t1 t2) t3
    _ = _ := by ring

/-- **E-07, Proposition 1**: `‖(I - P) Q̂‖ ≤ ‖R‖_F / (√(1 - η) (dist - w'))` when the eigenvalues
of `A` outside `T` lie at distance at least `w + dist` from `c`, `‖Λ - c I‖ ≤ w` and
`‖H' - Λ‖ ≤ w' < dist` (Lemma 2 (ii) gives such a `w'`). -/
theorem e07_prop1 {A : Matrix n n 𝕜} (hA : A.IsHermitian) (T : Finset n) {X : Matrix n g 𝕜}
    {η : ℝ} (hη1 : η < 1) (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
    (Λ : Matrix g g 𝕜) {c w w' dist : ℝ} (hw : 0 ≤ w) (hw' : 0 ≤ w')
    (hΛ : ∀ z, ‖toEuclideanLin (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) z‖ ≤ w * ‖z‖)
    (hH' : ∀ z, ‖toEuclideanLin ((polarMat X)ᴴ * A * polarMat X - Λ) z‖ ≤ w' * ‖z‖)
    (hdist : w' < dist) (hgap : ∀ i, i ∉ T → w + dist ≤ |hA.eigenvalues i - c|)
    (z : EuclideanSpace 𝕜 g) :
    ‖toEuclideanLin ((1 - specProj hA T) * polarMat X) z‖ ≤
      (√(1 - η))⁻¹ * √(frob (A * X - X * Λ)) / (dist - w') * ‖z‖ := by
  have hk : 0 ≤ (√(1 - η))⁻¹ := inv_nonneg.mpr (Real.sqrt_nonneg _)
  have hΛ' : ∀ z, ‖toEuclideanLin ((polarMat X)ᴴ * A * polarMat X -
      (c : 𝕜) • (1 : Matrix g g 𝕜)) z‖ ≤ (w + w') * ‖z‖ := fun z => by
    have e : (polarMat X)ᴴ * A * polarMat X - (c : 𝕜) • (1 : Matrix g g 𝕜) =
        ((polarMat X)ᴴ * A * polarMat X - Λ) + (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) := by abel
    rw [e, map_add, LinearMap.add_apply]
    refine (norm_add_le _ _).trans ?_
    have h1 := hH' z
    have h2 := hΛ z
    nlinarith
  have hres : ∀ z, ‖toEuclideanLin (A * polarMat X - polarMat X * ((polarMat X)ᴴ * A * polarMat X))
      z‖ ≤ (√(1 - η))⁻¹ * √(frob (A * X - X * Λ)) * ‖z‖ := fun z => by
    refine (norm_toEuclideanLin_le_frob _ z).trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg _))
    calc √(frob (A * polarMat X - polarMat X * ((polarMat X)ᴴ * A * polarMat X)))
        ≤ √(((√(1 - η))⁻¹) ^ 2 * frob (A * X - X * Λ)) :=
          Real.sqrt_le_sqrt (e07_lemma2_i hη1 hη A Λ)
      _ = (√(1 - η))⁻¹ * √(frob (A * X - X * Λ)) := by
          rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hk]
  exact sinTheta hA T (polarMat X) (Λ := (polarMat X)ᴴ * A * polarMat X) (c := c) (w := w + w')
    (sep := dist - w') (by linarith) (by linarith) hΛ'
    (fun i hi => by linarith [hgap i hi]) hres z

end Lemma2

section Lemma3

variable {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g] [DecidableEq g]

/-- **E-07, Proposition 1**, projector form: `‖Q̂ Q̂ᴴ - P‖ ≤ s_DK` with
`s_DK = ‖R‖_F / (√(1 - η) (dist - w'))`, when `P` has rank `g` (`#T = g`) and `s_DK < 1`. -/
theorem e07_prop1_proj {A : Matrix n n 𝕜} (hA : A.IsHermitian) (T : Finset n)
    {X : Matrix n g 𝕜} {η : ℝ} (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
    (Λ : Matrix g g 𝕜) {c w w' dist : ℝ} (hw : 0 ≤ w) (hw' : 0 ≤ w')
    (hΛ : ∀ z, ‖toEuclideanLin (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) z‖ ≤ w * ‖z‖)
    (hH' : ∀ z, ‖toEuclideanLin ((polarMat X)ᴴ * A * polarMat X - Λ) z‖ ≤ w' * ‖z‖)
    (hdist : w' < dist) (hgap : ∀ i, i ∉ T → w + dist ≤ |hA.eigenvalues i - c|)
    (hcard : T.card = Fintype.card g)
    (hs1 : (√(1 - η))⁻¹ * √(frob (A * X - X * Λ)) / (dist - w') < 1)
    (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (polarMat X * (polarMat X)ᴴ - specProj hA T) v‖ ≤
      (√(1 - η))⁻¹ * √(frob (A * X - X * Λ)) / (dist - w') * ‖v‖ := by
  have hY := e07_prop1 hA T hη1 hη Λ hw hw' hΛ hH' hdist hgap
  have hs0 : 0 ≤ (√(1 - η))⁻¹ * √(frob (A * X - X * Λ)) / (dist - w') :=
    div_nonneg (mul_nonneg (inv_nonneg.mpr (Real.sqrt_nonneg _)) (Real.sqrt_nonneg _))
      (by linarith)
  rw [← eigCols_mul_conjTranspose hA T] at hY ⊢
  exact proj_sub_le (eigCols hA T) (polarMat X) (eigCols_conjTranspose_mul hA T)
    (polarMat_conjTranspose_mul hη1 hη) (by rw [Fintype.card_coe, hcard]) hs0 hs1 hY v

/-- **E-07, Lemma 3** (alignment): the polar factor `U` of `Q₁ᴴ Q₂` is unitary and
`‖Q₂ - Q₁ U‖ ≤ s + s²`. -/
theorem e07_lemma3 {k : Type*} [Fintype k] [DecidableEq k] (Q1 : Matrix n k 𝕜)
    (Q2 : Matrix n g 𝕜) (h1 : Q1ᴴ * Q1 = 1) (h2 : Q2ᴴ * Q2 = 1)
    (hcard : Fintype.card k = Fintype.card g) {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s < 1)
    (hY : ∀ z, ‖toEuclideanLin ((1 - Q1 * Q1ᴴ) * Q2) z‖ ≤ s * ‖z‖) :
    (polarMat (Q1ᴴ * Q2))ᴴ * polarMat (Q1ᴴ * Q2) = 1 ∧
      polarMat (Q1ᴴ * Q2) * (polarMat (Q1ᴴ * Q2))ᴴ = 1 ∧
      ∀ z, ‖toEuclideanLin (Q2 - Q1 * polarMat (Q1ᴴ * Q2)) z‖ ≤ (s + s ^ 2) * ‖z‖ := by
  have hPP : Q1 * Q1ᴴ * (Q1 * Q1ᴴ) = Q1 * Q1ᴴ := by
    rw [Matrix.mul_assoc, ← Matrix.mul_assoc Q1ᴴ, h1, Matrix.one_mul]
  have hP2 : (1 - Q1 * Q1ᴴ) * (1 - Q1 * Q1ᴴ) = 1 - Q1 * Q1ᴴ := by
    rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_sub, Matrix.mul_one, hPP, sub_self, sub_zero]
  have hPh : (1 - Q1 * Q1ᴴ)ᴴ = 1 - Q1 * Q1ᴴ := by
    rw [conjTranspose_sub, conjTranspose_one, conjTranspose_mul, conjTranspose_conjTranspose]
  have hPQ : (1 - Q1 * Q1ᴴ) * ((1 - Q1 * Q1ᴴ) * Q2) = (1 - Q1 * Q1ᴴ) * Q2 := by
    rw [← Matrix.mul_assoc, hP2]
  have hgram : (Q1ᴴ * Q2)ᴴ * (Q1ᴴ * Q2) =
      1 - ((1 - Q1 * Q1ᴴ) * Q2)ᴴ * ((1 - Q1 * Q1ᴴ) * Q2) := by
    calc (Q1ᴴ * Q2)ᴴ * (Q1ᴴ * Q2) = Q2ᴴ * (Q1 * Q1ᴴ) * Q2 := by
          rw [conjTranspose_mul, conjTranspose_conjTranspose]
          simp only [Matrix.mul_assoc]
      _ = Q2ᴴ * Q2 - Q2ᴴ * (1 - Q1 * Q1ᴴ) * Q2 := by
          simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one]
          abel
      _ = 1 - ((1 - Q1 * Q1ᴴ) * Q2)ᴴ * ((1 - Q1 * Q1ᴴ) * Q2) := by
          rw [h2, conjTranspose_mul, hPh]
          congr 1
          simp only [Matrix.mul_assoc, hPQ]
  have hη' : ∀ z, ‖toEuclideanLin ((Q1ᴴ * Q2)ᴴ * (Q1ᴴ * Q2)) z - z‖ ≤ s ^ 2 * ‖z‖ := fun z => by
    rw [hgram, map_sub, LinearMap.sub_apply, toEuclideanLin_one_apply, sub_sub_cancel_left,
      norm_neg, toEuclideanLin_mul_apply]
    refine (norm_conjTranspose_le _ hs0 hY _).trans ?_
    rw [sq, mul_assoc]
    exact mul_le_mul_of_nonneg_left (hY z) hs0
  have hs2 : s ^ 2 < 1 := by nlinarith
  have hd : ∀ z, ‖toEuclideanLin (Q1ᴴ * Q2 - polarMat (Q1ᴴ * Q2)) z‖ ≤
      (1 - √(1 - s ^ 2)) * ‖z‖ := fun z => by
    rw [map_sub, LinearMap.sub_apply, toEuclideanLin_polarMat]
    exact Enclosure.sub_polarQ_le
      (Enclosure.hermitianEigenData (isHermitian_conjTranspose_mul_self (Q1ᴴ * Q2))) hs2 hη'
      (Enclosure.inner_toEuclideanLin_gram _) (sq_nonneg s) z
  have hδs : 1 - √(1 - s ^ 2) ≤ s ^ 2 := by
    have h4 : 0 ≤ 1 - s ^ 2 := by nlinarith
    have h3 : 1 - s ^ 2 ≤ √(1 - s ^ 2) :=
      calc 1 - s ^ 2 = √((1 - s ^ 2) ^ 2) := (Real.sqrt_sq h4).symm
        _ ≤ √(1 - s ^ 2) := Real.sqrt_le_sqrt (by nlinarith [sq_nonneg s])
    linarith
  refine ⟨polarMat_conjTranspose_mul hs2 hη', polarMat_mul_conjTranspose hs2 hη' hcard.symm,
    fun z => ?_⟩
  have e : Q2 - Q1 * polarMat (Q1ᴴ * Q2) =
      (1 - Q1 * Q1ᴴ) * Q2 + Q1 * (Q1ᴴ * Q2 - polarMat (Q1ᴴ * Q2)) := by
    rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_sub, ← Matrix.mul_assoc]
    abel
  rw [e, map_add, LinearMap.add_apply]
  refine (norm_add_le _ _).trans ?_
  rw [toEuclideanLin_mul_apply Q1, norm_apply_of_isometry Q1 h1]
  have h5 := add_le_add (hY z) (hd z)
  have h6 : (1 - √(1 - s ^ 2)) * ‖z‖ ≤ s ^ 2 * ‖z‖ := mul_le_mul_of_nonneg_right hδs (norm_nonneg _)
  nlinarith

/-- **Theorem 6**, first bound, with the separation measured from the Ritz values as in `thm:dk` of
the manuscript. If the Ritz values (the eigenvalues of `H' = Q̂ᴴ A Q̂`) lie in `[c - w, c + w]`
(`‖H' - c I‖ ≤ w`), every eigenvalue of `A` outside `T` lies at distance at least `sep` from that
interval, and `‖A Q̂ - Q̂ Λ‖ ≤ ρ` for some `Λ` (for `Λ = diag(μ)`, `ρ = ‖R‖ + (‖A‖ + ‖Λ‖) δ` by
`polarMat_residual`), then `‖Q̂ Q̂ᴴ - P‖ ≤ ρ / sep` when `#T = g` and `ρ / sep < 1`. -/
theorem thm6_ritz {A : Matrix n n 𝕜} (hA : A.IsHermitian) (T : Finset n) {X : Matrix n g 𝕜}
    {η : ℝ} (hη1 : η < 1) (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
    (Λ : Matrix g g 𝕜) {c w sep ρ : ℝ} (hsep : 0 < sep) (hw : 0 ≤ w)
    (hH' : ∀ z, ‖toEuclideanLin ((polarMat X)ᴴ * A * polarMat X -
      (c : 𝕜) • (1 : Matrix g g 𝕜)) z‖ ≤ w * ‖z‖)
    (hgap : ∀ i, i ∉ T → w + sep ≤ |hA.eigenvalues i - c|)
    (hρ : ∀ z, ‖toEuclideanLin (A * polarMat X - polarMat X * Λ) z‖ ≤ ρ * ‖z‖)
    (hcard : T.card = Fintype.card g) (hs0 : 0 ≤ ρ / sep) (hs1 : ρ / sep < 1)
    (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (polarMat X * (polarMat X)ᴴ - specProj hA T) v‖ ≤ ρ / sep * ‖v‖ := by
  have hQh := polarMat_conjTranspose_mul hη1 hη
  have e : A * polarMat X - polarMat X * ((polarMat X)ᴴ * A * polarMat X) =
      (1 - polarMat X * (polarMat X)ᴴ) * (A * polarMat X - polarMat X * Λ) := by
    rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_sub]
    have h1 : polarMat X * (polarMat X)ᴴ * (polarMat X * Λ) = polarMat X * Λ := by
      rw [Matrix.mul_assoc, ← Matrix.mul_assoc (polarMat X)ᴴ, hQh, Matrix.one_mul]
    have h2 : polarMat X * (polarMat X)ᴴ * (A * polarMat X) =
        polarMat X * ((polarMat X)ᴴ * A * polarMat X) := by
      simp only [Matrix.mul_assoc]
    rw [h1, h2]
    abel
  have hres : ∀ z, ‖toEuclideanLin (A * polarMat X -
      polarMat X * ((polarMat X)ᴴ * A * polarMat X)) z‖ ≤ ρ * ‖z‖ := fun z => by
    rw [e, toEuclideanLin_mul_apply]
    refine (norm_one_sub_proj_le _ hQh _).trans ?_
    rw [one_mul]
    exact hρ z
  have hY := sinTheta hA T (polarMat X) hsep hw hH' hgap hres
  rw [← eigCols_mul_conjTranspose hA T] at hY ⊢
  exact proj_sub_le (eigCols hA T) (polarMat X) (eigCols_conjTranspose_mul hA T) hQh
    (by rw [Fintype.card_coe, hcard]) hs0 hs1 hY v

end Lemma3

end CertifyMathlib.Counting
