import CertifyMathlib.Counting.Compose

/-!
# Theorem 6 (projector bound): the Davis-Kahan sin Θ theorem in the spectral norm

Source: `docs/theory.md` §4 and `thm:dk` of `manuscript_v4/sections/s3_math.tex`, with the
notation of `docs/proofs/E-07_davis_kahan.md` in the code repository.

* `sinTheta_core`: the separation argument for a Sylvester equation. If `B Y - Y C = Z` with
  `‖C‖ ≤ w`, `‖Z‖ ≤ ρ` and `‖B y‖ ≥ (w + sep) ‖y‖` on the range of `Y`, then `‖Y‖ ≤ ρ / sep`.
* `specProj hA T`: the spectral projector of the Hermitian `A` on the eigen-indices `T`
  (`U diag(1_T) Uᴴ` with Mathlib's eigenvector unitary `U`), and `eigCols hA T`, the matrix of the
  eigenvectors with index in `T`, with `specProj hA T = eigCols (eigCols)ᴴ`.
* `sinTheta`: for `Q̂` with `Q̂ᴴ Q̂ = 1` and any `Λ` with `‖Λ - c I‖ ≤ w`, if every eigenvalue of
  `A` outside `T` lies at distance at least `w + sep` from `c` (that is, outside the interval
  `(c - w - sep, c + w + sep)`), then `‖(I - P) Q̂‖ ≤ ‖A Q̂ - Q̂ Λ‖ / sep`.
* `proj_sub_le`: for two isometries `Q₁` (`n × k`) and `Q₂` (`n × g`) with `k = g` and
  `‖(I - Q₁Q₁ᴴ) Q₂‖ ≤ s < 1`, `‖Q₂Q₂ᴴ - Q₁Q₁ᴴ‖ ≤ s` (projectors of equal rank).
* `thm6_proj`: Theorem 6, first bound, `‖Q̂ Q̂ᴴ - P‖ ≤ s` with `Q̂` the polar factor of `X` and
  `s = ρ / sep`, `ρ = ‖R‖ + (‖A‖ + ‖Λ‖) δ`.

Norms of operators are given by their defining bound (`‖M z‖ ≤ c ‖z‖` for all `z`).
-/

namespace CertifyMathlib.Counting

open Matrix
open scoped InnerProductSpace

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ### The separation argument -/

/-- If `B Y - Y C = Z`, `‖Z‖ ≤ ρ`, `‖C‖ ≤ w` and `B` stretches the range of `Y` by at least
`w + sep`, then `‖Y‖ ≤ ρ / sep`. -/
theorem sinTheta_core {n g : Type*} [Fintype n] [Fintype g]
    (Y : EuclideanSpace 𝕜 g →ₗ[𝕜] EuclideanSpace 𝕜 n)
    (B : EuclideanSpace 𝕜 n →ₗ[𝕜] EuclideanSpace 𝕜 n)
    (C : EuclideanSpace 𝕜 g →ₗ[𝕜] EuclideanSpace 𝕜 g) {ρ w sep : ℝ} (hsep : 0 < sep)
    (hw : 0 ≤ w) (hZ : ∀ z, ‖B (Y z) - Y (C z)‖ ≤ ρ * ‖z‖) (hC : ∀ z, ‖C z‖ ≤ w * ‖z‖)
    (hgap : ∀ z, (w + sep) * ‖Y z‖ ≤ ‖B (Y z)‖) (z : EuclideanSpace 𝕜 g) :
    ‖Y z‖ ≤ ρ / sep * ‖z‖ := by
  set N := ‖LinearMap.toContinuousLinearMap Y‖ with hNdef
  have hN0 : 0 ≤ N := norm_nonneg _
  have hN : ∀ z, ‖Y z‖ ≤ N * ‖z‖ := fun z => (LinearMap.toContinuousLinearMap Y).le_opNorm z
  have key : ∀ z, (w + sep) * ‖Y z‖ ≤ (ρ + N * w) * ‖z‖ := fun z => by
    calc (w + sep) * ‖Y z‖ ≤ ‖B (Y z)‖ := hgap z
      _ = ‖(B (Y z) - Y (C z)) + Y (C z)‖ := by rw [sub_add_cancel]
      _ ≤ ‖B (Y z) - Y (C z)‖ + ‖Y (C z)‖ := norm_add_le _ _
      _ ≤ ρ * ‖z‖ + N * (w * ‖z‖) :=
          add_le_add (hZ z) ((hN _).trans (mul_le_mul_of_nonneg_left (hC z) hN0))
      _ = (ρ + N * w) * ‖z‖ := by ring
  by_cases hF : ∀ z : EuclideanSpace 𝕜 g, z = 0
  · simp [hF z]
  obtain ⟨z0, hz0⟩ := not_forall.mp hF
  have hpos : 0 < ‖z0‖ := norm_pos_iff.mpr hz0
  have hws : 0 < w + sep := by linarith
  have hρN : 0 ≤ ρ + N * w := by
    have h1 := key z0
    have h2 : 0 ≤ (w + sep) * ‖Y z0‖ := mul_nonneg hws.le (norm_nonneg _)
    by_contra hneg
    have : (ρ + N * w) * ‖z0‖ < 0 := mul_neg_of_neg_of_pos (not_le.mp hneg) hpos
    linarith
  have hNle : N ≤ (ρ + N * w) / (w + sep) := by
    refine ContinuousLinearMap.opNorm_le_bound _ (div_nonneg hρN hws.le) fun z => ?_
    show ‖Y z‖ ≤ (ρ + N * w) / (w + sep) * ‖z‖
    rw [div_mul_eq_mul_div, le_div_iff₀ hws]
    nlinarith [key z]
  rw [le_div_iff₀ hws] at hNle
  have hNs : N ≤ ρ / sep := by
    rw [le_div_iff₀ hsep]
    nlinarith
  exact (hN z).trans (mul_le_mul_of_nonneg_right hNs (norm_nonneg _))

/-! ### Isometries and orthogonal projectors -/

section Iso

variable {n p : Type*} [Fintype n] [DecidableEq n] [Fintype p] [DecidableEq p]

theorem inner_conjTranspose_apply (M : Matrix n p 𝕜) (w : EuclideanSpace 𝕜 n)
    (u : EuclideanSpace 𝕜 p) :
    ⟪toEuclideanLin Mᴴ w, u⟫_𝕜 = ⟪w, toEuclideanLin M u⟫_𝕜 := by
  rw [Matrix.toEuclideanLin_conjTranspose_eq_adjoint, LinearMap.adjoint_inner_left]

theorem inner_apply_conjTranspose (M : Matrix n p 𝕜) (u : EuclideanSpace 𝕜 p)
    (w : EuclideanSpace 𝕜 n) :
    ⟪toEuclideanLin M u, w⟫_𝕜 = ⟪u, toEuclideanLin Mᴴ w⟫_𝕜 := by
  rw [← inner_conjTranspose_apply, conjTranspose_conjTranspose]

/-- `Qᴴ Q = 1` makes `Q` an isometry. -/
theorem norm_apply_of_isometry (Q : Matrix n p 𝕜) (hQ : Qᴴ * Q = 1) (z : EuclideanSpace 𝕜 p) :
    ‖toEuclideanLin Q z‖ = ‖z‖ :=
  (LinearMap.norm_map_iff_inner_map_map (toEuclideanLin Q)).mpr (fun u v => by
    rw [Enclosure.inner_toEuclideanLin_gram, hQ, toEuclideanLin_one_apply]) z

/-- `‖Mᴴ‖ ≤ ‖M‖` in the defining-bound form. -/
theorem norm_conjTranspose_le (M : Matrix n p 𝕜) {s : ℝ} (hs : 0 ≤ s)
    (h : ∀ z, ‖toEuclideanLin M z‖ ≤ s * ‖z‖) (w : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin Mᴴ w‖ ≤ s * ‖w‖ := by
  have h1 : ‖toEuclideanLin Mᴴ w‖ ^ 2 =
      RCLike.re ⟪w, toEuclideanLin M (toEuclideanLin Mᴴ w)⟫_𝕜 := by
    rw [← inner_self_eq_norm_sq (𝕜 := 𝕜), inner_conjTranspose_apply]
  have h2 : RCLike.re ⟪w, toEuclideanLin M (toEuclideanLin Mᴴ w)⟫_𝕜 ≤
      ‖w‖ * (s * ‖toEuclideanLin Mᴴ w‖) :=
    (RCLike.re_le_norm _).trans ((norm_inner_le_norm _ _).trans
      (mul_le_mul_of_nonneg_left (h _) (norm_nonneg _)))
  have hy := norm_nonneg (toEuclideanLin Mᴴ w)
  by_cases h0 : ‖toEuclideanLin Mᴴ w‖ = 0
  · rw [h0]
    exact mul_nonneg hs (norm_nonneg _)
  · have hpos : 0 < ‖toEuclideanLin Mᴴ w‖ := lt_of_le_of_ne hy (Ne.symm h0)
    nlinarith

/-- The ranges of `Q Qᴴ` and `1 - Q Qᴴ` are orthogonal. -/
theorem inner_isometry_orth (Q : Matrix n p 𝕜) (hQ : Qᴴ * Q = 1) (u : EuclideanSpace 𝕜 p)
    (y : EuclideanSpace 𝕜 n) : ⟪toEuclideanLin Q u, toEuclideanLin (1 - Q * Qᴴ) y⟫_𝕜 = 0 := by
  have e : Qᴴ * (1 - Q * Qᴴ) = 0 := by
    rw [Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, hQ, Matrix.one_mul, sub_self]
  rw [inner_apply_conjTranspose, ← toEuclideanLin_mul_apply, e, map_zero, LinearMap.zero_apply,
    inner_zero_right]

/-- Pythagoras for the orthogonal projector `Q Qᴴ`. -/
theorem norm_sq_proj (Q : Matrix n p 𝕜) (hQ : Qᴴ * Q = 1) (v : EuclideanSpace 𝕜 n) :
    ‖v‖ ^ 2 = ‖toEuclideanLin Qᴴ v‖ ^ 2 + ‖toEuclideanLin (1 - Q * Qᴴ) v‖ ^ 2 := by
  have hv : v = toEuclideanLin Q (toEuclideanLin Qᴴ v) + toEuclideanLin (1 - Q * Qᴴ) v := by
    rw [map_sub, LinearMap.sub_apply, toEuclideanLin_one_apply, ← toEuclideanLin_mul_apply]
    abel
  conv_lhs => rw [hv]
  rw [norm_add_sq (𝕜 := 𝕜), inner_isometry_orth Q hQ, map_zero, mul_zero, add_zero,
    norm_apply_of_isometry Q hQ]

/-- A square matrix bounded below has its conjugate transpose bounded below by the same
constant. -/
theorem conjTranspose_lower {k : Type*} [Fintype k] [DecidableEq k] (Cm : Matrix k p 𝕜)
    (hcard : Fintype.card k = Fintype.card p) {a : ℝ} (ha : 0 < a)
    (h : ∀ z, a * ‖z‖ ^ 2 ≤ ‖toEuclideanLin Cm z‖ ^ 2) (y : EuclideanSpace 𝕜 k) :
    a * ‖y‖ ^ 2 ≤ ‖toEuclideanLin Cmᴴ y‖ ^ 2 := by
  have hinj : Function.Injective (toEuclideanLin Cm) := by
    rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
    intro z hz
    have h1 := h z
    rw [hz, norm_zero] at h1
    have h1' : a * ‖z‖ ^ 2 ≤ a * 0 := by simpa using h1
    have h3 : ‖z‖ ^ 2 ≤ 0 := le_of_mul_le_mul_left h1' ha
    have h2 : ‖z‖ ≤ 0 := by nlinarith [norm_nonneg z]
    exact norm_le_zero_iff.mp h2
  have hsurj : Function.Surjective (toEuclideanLin Cm) :=
    (LinearMap.injective_iff_surjective_of_finrank_eq_finrank
      (by rw [finrank_euclideanSpace, finrank_euclideanSpace, hcard])).mp hinj
  obtain ⟨z, hz⟩ := hsurj y
  have e : ⟪y, y⟫_𝕜 = ⟪toEuclideanLin Cmᴴ y, z⟫_𝕜 := by
    rw [inner_conjTranspose_apply, hz]
  have h1 : ‖y‖ ^ 2 ≤ ‖toEuclideanLin Cmᴴ y‖ * ‖z‖ := by
    rw [← inner_self_eq_norm_sq (𝕜 := 𝕜), e]
    exact (RCLike.re_le_norm _).trans (norm_inner_le_norm _ _)
  have h2 : a * ‖z‖ ^ 2 ≤ ‖y‖ ^ 2 := by
    rw [← hz]
    exact h z
  by_cases hY : ‖y‖ = 0
  · rw [hY]
    nlinarith [sq_nonneg ‖toEuclideanLin Cmᴴ y‖]
  have hY2 : 0 < ‖y‖ ^ 2 := pow_pos (lt_of_le_of_ne (norm_nonneg _) (Ne.symm hY)) 2
  have h3 : (‖y‖ ^ 2) ^ 2 ≤ (‖toEuclideanLin Cmᴴ y‖ * ‖z‖) ^ 2 :=
    pow_le_pow_left₀ (sq_nonneg _) h1 2
  have h4 : a * (‖y‖ ^ 2) ^ 2 ≤ ‖toEuclideanLin Cmᴴ y‖ ^ 2 * ‖y‖ ^ 2 := by
    calc a * (‖y‖ ^ 2) ^ 2 ≤ a * (‖toEuclideanLin Cmᴴ y‖ * ‖z‖) ^ 2 :=
          mul_le_mul_of_nonneg_left h3 ha.le
      _ = ‖toEuclideanLin Cmᴴ y‖ ^ 2 * (a * ‖z‖ ^ 2) := by ring
      _ ≤ ‖toEuclideanLin Cmᴴ y‖ ^ 2 * ‖y‖ ^ 2 := mul_le_mul_of_nonneg_left h2 (sq_nonneg _)
  exact le_of_mul_le_mul_right (by nlinarith [h4]) hY2

/-- **Projectors of equal rank.** For isometries `Q₁` (`n × k`) and `Q₂` (`n × g`) with `k = g`,
`‖(I - Q₁Q₁ᴴ) Q₂‖ ≤ s < 1` gives `‖Q₂Q₂ᴴ - Q₁Q₁ᴴ‖ ≤ s`. -/
theorem proj_sub_le {k : Type*} [Fintype k] [DecidableEq k] (Q1 : Matrix n k 𝕜)
    (Q2 : Matrix n p 𝕜) (h1 : Q1ᴴ * Q1 = 1) (h2 : Q2ᴴ * Q2 = 1)
    (hcard : Fintype.card k = Fintype.card p) {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s < 1)
    (hY : ∀ z, ‖toEuclideanLin ((1 - Q1 * Q1ᴴ) * Q2) z‖ ≤ s * ‖z‖) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (Q2 * Q2ᴴ - Q1 * Q1ᴴ) v‖ ≤ s * ‖v‖ := by
  -- the cosine matrix `Q₁ᴴ Q₂` is bounded below by `1 - s²`
  have hA : ∀ z, (1 - s ^ 2) * ‖z‖ ^ 2 ≤ ‖toEuclideanLin (Q1ᴴ * Q2) z‖ ^ 2 := fun z => by
    have e1 := norm_sq_proj Q1 h1 (toEuclideanLin Q2 z)
    rw [norm_apply_of_isometry Q2 h2, ← toEuclideanLin_mul_apply,
      ← toEuclideanLin_mul_apply] at e1
    have h3 : ‖toEuclideanLin ((1 - Q1 * Q1ᴴ) * Q2) z‖ ^ 2 ≤ (s * ‖z‖) ^ 2 :=
      pow_le_pow_left₀ (norm_nonneg _) (hY z) 2
    nlinarith [h3, e1]
  have hB : ∀ y, (1 - s ^ 2) * ‖y‖ ^ 2 ≤ ‖toEuclideanLin (Q1ᴴ * Q2)ᴴ y‖ ^ 2 :=
    conjTranspose_lower (Q1ᴴ * Q2) hcard (by nlinarith) hA
  have hp : ‖toEuclideanLin (Q1 * Q1ᴴ) v‖ = ‖toEuclideanLin Q1ᴴ v‖ := by
    rw [toEuclideanLin_mul_apply, norm_apply_of_isometry Q1 h1]
  -- the part in the range of `P`
  have hC : ‖toEuclideanLin (1 - Q2 * Q2ᴴ) (toEuclideanLin (Q1 * Q1ᴴ) v)‖ ^ 2 ≤
      s ^ 2 * ‖toEuclideanLin Q1ᴴ v‖ ^ 2 := by
    have e2 := norm_sq_proj Q2 h2 (toEuclideanLin (Q1 * Q1ᴴ) v)
    have hq : toEuclideanLin Q2ᴴ (toEuclideanLin (Q1 * Q1ᴴ) v) =
        toEuclideanLin (Q1ᴴ * Q2)ᴴ (toEuclideanLin Q1ᴴ v) := by
      rw [conjTranspose_mul, conjTranspose_conjTranspose, toEuclideanLin_mul_apply,
        toEuclideanLin_mul_apply]
    have h3 := hB (toEuclideanLin Q1ᴴ v)
    rw [hq] at e2
    rw [hp] at e2
    nlinarith [e2, h3]
  -- the part in the range of `I - P`
  have hPP : Q1 * Q1ᴴ * (Q1 * Q1ᴴ) = Q1 * Q1ᴴ := by
    rw [Matrix.mul_assoc, ← Matrix.mul_assoc Q1ᴴ, h1, Matrix.one_mul]
  have hP2 : (1 - Q1 * Q1ᴴ) * (1 - Q1 * Q1ᴴ) = 1 - Q1 * Q1ᴴ := by
    rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_sub, Matrix.mul_one, hPP, sub_self, sub_zero]
  have hPh : (1 - Q1 * Q1ᴴ)ᴴ = 1 - Q1 * Q1ᴴ := by
    rw [conjTranspose_sub, conjTranspose_one, conjTranspose_mul, conjTranspose_conjTranspose]
  have hD : ‖toEuclideanLin Q2ᴴ (toEuclideanLin (1 - Q1 * Q1ᴴ) v)‖ ≤
      s * ‖toEuclideanLin (1 - Q1 * Q1ᴴ) v‖ := by
    have e : toEuclideanLin Q2ᴴ (toEuclideanLin (1 - Q1 * Q1ᴴ) v) =
        toEuclideanLin ((1 - Q1 * Q1ᴴ) * Q2)ᴴ (toEuclideanLin (1 - Q1 * Q1ᴴ) v) := by
      rw [conjTranspose_mul, hPh, toEuclideanLin_mul_apply,
        ← toEuclideanLin_mul_apply (1 - Q1 * Q1ᴴ), hP2]
    rw [e]
    exact norm_conjTranspose_le _ hs0 hY _
  -- `(Q₂Q₂ᴴ - P) v = Q₂Q₂ᴴ (I - P) v - (I - Q₂Q₂ᴴ) P v`, two orthogonal pieces
  have hsplit : toEuclideanLin (Q2 * Q2ᴴ - Q1 * Q1ᴴ) v =
      toEuclideanLin Q2 (toEuclideanLin Q2ᴴ (toEuclideanLin (1 - Q1 * Q1ᴴ) v)) -
        toEuclideanLin (1 - Q2 * Q2ᴴ) (toEuclideanLin (Q1 * Q1ᴴ) v) := by
    simp only [map_sub, LinearMap.sub_apply, toEuclideanLin_one_apply, toEuclideanLin_mul_apply]
    abel
  have hsq : ‖toEuclideanLin (Q2 * Q2ᴴ - Q1 * Q1ᴴ) v‖ ^ 2 ≤ (s * ‖v‖) ^ 2 := by
    rw [hsplit, norm_sub_sq (𝕜 := 𝕜), inner_isometry_orth Q2 h2, map_zero, mul_zero, sub_zero,
      norm_apply_of_isometry Q2 h2]
    have e3 := norm_sq_proj Q1 h1 v
    have hD2 : ‖toEuclideanLin Q2ᴴ (toEuclideanLin (1 - Q1 * Q1ᴴ) v)‖ ^ 2 ≤
        (s * ‖toEuclideanLin (1 - Q1 * Q1ᴴ) v‖) ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hD 2
    have e4 : (s * ‖v‖) ^ 2 = s ^ 2 * ‖toEuclideanLin Q1ᴴ v‖ ^ 2 +
        s ^ 2 * ‖toEuclideanLin (1 - Q1 * Q1ᴴ) v‖ ^ 2 := by
      rw [mul_pow, e3]
      ring
    rw [e4]
    nlinarith [hC, hD2]
  have hs' : 0 ≤ s * ‖v‖ := mul_nonneg hs0 (norm_nonneg _)
  calc ‖toEuclideanLin (Q2 * Q2ᴴ - Q1 * Q1ᴴ) v‖ =
        √(‖toEuclideanLin (Q2 * Q2ᴴ - Q1 * Q1ᴴ) v‖ ^ 2) := (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ √((s * ‖v‖) ^ 2) := Real.sqrt_le_sqrt hsq
    _ = s * ‖v‖ := Real.sqrt_sq hs'

end Iso

/-! ### Diagonal matrices and the spectral projector -/

section Spectral

variable {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g] [DecidableEq g]

/-- `1` on `T` and `0` off `T`. -/
def ind (T : Finset n) (i : n) : ℝ := if i ∈ T then 1 else 0

theorem dg_mul_dg (d e : n → ℝ) :
    (dg d : Matrix n n 𝕜) * dg e = dg (fun i => d i * e i) := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [dg]
  · simp [dg, h]

omit [Fintype n] in
theorem dg_sub (d e : n → ℝ) : (dg (fun i => d i - e i) : Matrix n n 𝕜) = dg d - dg e := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [dg]
  · simp [dg, h]

omit [Fintype n] in
theorem dg_const (c : ℝ) : (dg (fun _ => c) : Matrix n n 𝕜) = (c : 𝕜) • (1 : Matrix n n 𝕜) := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [dg]
  · simp [dg, h]

theorem toEuclideanLin_dg_apply (d : n → ℝ) (x : EuclideanSpace 𝕜 n) (i : n) :
    (toEuclideanLin (dg d : Matrix n n 𝕜) x) i = (d i : 𝕜) * x i := by
  simp [Matrix.toLpLin_apply, dg, Matrix.mulVec_diagonal]

theorem norm_dg_sq (d : n → ℝ) (x : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (dg d : Matrix n n 𝕜) x‖ ^ 2 = ∑ i, d i ^ 2 * ‖x i‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [toEuclideanLin_dg_apply, norm_mul, RCLike.norm_ofReal, mul_pow, sq_abs]

/-- `‖diag(d)‖ ≤ K` when every `|d i| ≤ K`. -/
theorem norm_dg_le (d : n → ℝ) {K : ℝ} (hK : 0 ≤ K) (hd : ∀ i, |d i| ≤ K)
    (x : EuclideanSpace 𝕜 n) : ‖toEuclideanLin (dg d : Matrix n n 𝕜) x‖ ≤ K * ‖x‖ := by
  have hsq : ‖toEuclideanLin (dg d : Matrix n n 𝕜) x‖ ^ 2 ≤ (K * ‖x‖) ^ 2 := by
    rw [norm_dg_sq, mul_pow, EuclideanSpace.norm_sq_eq, Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right ?_ (sq_nonneg _)
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) (hd i) 2
  calc ‖toEuclideanLin (dg d : Matrix n n 𝕜) x‖ =
        √(‖toEuclideanLin (dg d : Matrix n n 𝕜) x‖ ^ 2) := (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ √((K * ‖x‖) ^ 2) := Real.sqrt_le_sqrt hsq
    _ = K * ‖x‖ := Real.sqrt_sq (mul_nonneg hK (norm_nonneg _))

/-- `U diag(d) Uᴴ` with the eigenvector unitary `U` of `A`. -/
noncomputable def spec {A : Matrix n n 𝕜} (hA : A.IsHermitian) (d : n → ℝ) : Matrix n n 𝕜 :=
  (hA.eigenvectorUnitary : Matrix n n 𝕜) * dg d * (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ

/-- The spectral projector of `A` on the eigen-indices `T`. -/
noncomputable def specProj {A : Matrix n n 𝕜} (hA : A.IsHermitian) (T : Finset n) :
    Matrix n n 𝕜 :=
  spec hA (ind T)

variable {A : Matrix n n 𝕜} (hA : A.IsHermitian)

theorem spec_mul (d e : n → ℝ) : spec hA d * spec hA e = spec hA (fun i => d i * e i) := by
  have hU2 := conjTranspose_mul_unitary hA.eigenvectorUnitary
  simp only [spec, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ
    (hA.eigenvectorUnitary : Matrix n n 𝕜), hU2, Matrix.one_mul, ← Matrix.mul_assoc (dg d),
    dg_mul_dg]

theorem spec_sub (d e : n → ℝ) : spec hA d - spec hA e = spec hA (fun i => d i - e i) := by
  simp only [spec, dg_sub, Matrix.mul_sub, Matrix.sub_mul]

theorem spec_one : spec hA (fun _ => 1) = 1 := by
  rw [spec, dg_const, RCLike.ofReal_one, one_smul, Matrix.mul_one,
    unitary_mul_conjTranspose hA.eigenvectorUnitary]

theorem one_sub_specProj (T : Finset n) :
    1 - specProj hA T = spec hA (fun i => 1 - ind T i) := by
  rw [specProj, ← spec_one hA, spec_sub]

theorem sub_smul_eq_spec (c : ℝ) :
    A - (c : 𝕜) • (1 : Matrix n n 𝕜) = spec hA (fun i => hA.eigenvalues i - c) :=
  sub_smul_one_eq hA c

theorem norm_spec_apply (d : n → ℝ) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (spec hA d) v‖ =
      ‖toEuclideanLin (dg d : Matrix n n 𝕜)
        (toEuclideanLin (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ v)‖ := by
  rw [spec, toEuclideanLin_mul_apply, toEuclideanLin_mul_apply,
    norm_apply_of_isometry _ (conjTranspose_mul_unitary hA.eigenvectorUnitary)]

theorem norm_conjTranspose_unitary (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ v‖ = ‖v‖ :=
  norm_conjTranspose_apply _ (unitary_mul_conjTranspose hA.eigenvectorUnitary) v

/-- `‖I - P‖ ≤ 1`. -/
theorem norm_one_sub_specProj_le (T : Finset n) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (1 - specProj hA T) v‖ ≤ ‖v‖ := by
  rw [one_sub_specProj, norm_spec_apply]
  refine (norm_dg_le _ zero_le_one (fun i => ?_) _).trans ?_
  · by_cases h : i ∈ T <;> simp [ind, h]
  · rw [one_mul, norm_conjTranspose_unitary]

/-- On the range of `I - P` the matrix `A - c I` stretches by at least `a` when every eigenvalue
outside `T` is at distance at least `a` from `c`. -/
theorem gap_one_sub_specProj (T : Finset n) {c a : ℝ} (ha : 0 ≤ a)
    (hgap : ∀ i, i ∉ T → a ≤ |hA.eigenvalues i - c|) (v : EuclideanSpace 𝕜 n) :
    a * ‖toEuclideanLin (1 - specProj hA T) v‖ ≤
      ‖toEuclideanLin (A - (c : 𝕜) • (1 : Matrix n n 𝕜))
        (toEuclideanLin (1 - specProj hA T) v)‖ := by
  rw [← toEuclideanLin_mul_apply, sub_smul_eq_spec hA, one_sub_specProj, spec_mul,
    norm_spec_apply, norm_spec_apply]
  set x := toEuclideanLin (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ v
  have hsq : (a * ‖toEuclideanLin (dg (fun i => 1 - ind T i) : Matrix n n 𝕜) x‖) ^ 2 ≤
      ‖toEuclideanLin (dg (fun i => (hA.eigenvalues i - c) * (1 - ind T i)) :
        Matrix n n 𝕜) x‖ ^ 2 := by
    rw [mul_pow, norm_dg_sq, norm_dg_sq, Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    by_cases h : i ∈ T
    · simp [ind, h]
    · have hi := hgap i h
      have hi2 : a ^ 2 ≤ (hA.eigenvalues i - c) ^ 2 := by
        rw [← sq_abs (hA.eigenvalues i - c)]
        exact pow_le_pow_left₀ ha hi 2
      simp only [ind, h, ↓reduceIte, sub_zero, mul_one, one_pow, one_mul]
      exact mul_le_mul_of_nonneg_right hi2 (sq_nonneg _)
  calc a * ‖toEuclideanLin (dg (fun i => 1 - ind T i) : Matrix n n 𝕜) x‖ =
        √((a * ‖toEuclideanLin (dg (fun i => 1 - ind T i) : Matrix n n 𝕜) x‖) ^ 2) :=
          (Real.sqrt_sq (mul_nonneg ha (norm_nonneg _))).symm
    _ ≤ √(‖toEuclideanLin (dg (fun i => (hA.eigenvalues i - c) * (1 - ind T i)) :
          Matrix n n 𝕜) x‖ ^ 2) := Real.sqrt_le_sqrt hsq
    _ = _ := Real.sqrt_sq (norm_nonneg _)

theorem specProj_comm (T : Finset n) (c : ℝ) :
    (A - (c : 𝕜) • (1 : Matrix n n 𝕜)) * (1 - specProj hA T) =
      (1 - specProj hA T) * (A - (c : 𝕜) • (1 : Matrix n n 𝕜)) := by
  rw [sub_smul_eq_spec hA, one_sub_specProj, spec_mul, spec_mul]
  congr 1
  funext i
  ring

/-- **sin Θ theorem, spectral norm** (Davis-Kahan, interval form). -/
theorem sinTheta (T : Finset n) (Qh : Matrix n g 𝕜) {Λ : Matrix g g 𝕜} {c w sep ρ : ℝ}
    (hsep : 0 < sep) (hw : 0 ≤ w)
    (hΛ : ∀ z, ‖toEuclideanLin (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) z‖ ≤ w * ‖z‖)
    (hgap : ∀ i, i ∉ T → w + sep ≤ |hA.eigenvalues i - c|)
    (hres : ∀ z, ‖toEuclideanLin (A * Qh - Qh * Λ) z‖ ≤ ρ * ‖z‖) (z : EuclideanSpace 𝕜 g) :
    ‖toEuclideanLin ((1 - specProj hA T) * Qh) z‖ ≤ ρ / sep * ‖z‖ := by
  have hid : (A - (c : 𝕜) • (1 : Matrix n n 𝕜)) * ((1 - specProj hA T) * Qh) -
      ((1 - specProj hA T) * Qh) * (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) =
      (1 - specProj hA T) * (A * Qh - Qh * Λ) := by
    rw [← Matrix.mul_assoc, specProj_comm, Matrix.mul_assoc, Matrix.mul_assoc, ← Matrix.mul_sub]
    congr 1
    rw [Matrix.sub_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul,
      Matrix.mul_one]
    abel
  refine sinTheta_core (toEuclideanLin ((1 - specProj hA T) * Qh))
    (toEuclideanLin (A - (c : 𝕜) • (1 : Matrix n n 𝕜)))
    (toEuclideanLin (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜))) hsep hw (fun z => ?_) hΛ (fun z => ?_) z
  · have e : toEuclideanLin (A - (c : 𝕜) • (1 : Matrix n n 𝕜))
        (toEuclideanLin ((1 - specProj hA T) * Qh) z) -
        toEuclideanLin ((1 - specProj hA T) * Qh)
          (toEuclideanLin (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) z) =
        toEuclideanLin (1 - specProj hA T) (toEuclideanLin (A * Qh - Qh * Λ) z) := by
      rw [← toEuclideanLin_mul_apply, ← toEuclideanLin_mul_apply, ← LinearMap.sub_apply,
        ← map_sub, hid, toEuclideanLin_mul_apply]
    rw [e]
    exact (norm_one_sub_specProj_le hA T _).trans (hres z)
  · rw [toEuclideanLin_mul_apply]
    exact gap_one_sub_specProj hA T (by linarith) hgap _

/-- The eigenvectors with index in `T`, as the columns of an `n × T` matrix. -/
noncomputable def eigCols (T : Finset n) : Matrix n T 𝕜 :=
  Matrix.of fun i k => (hA.eigenvectorUnitary : Matrix n n 𝕜) i k.1

theorem eigCols_conjTranspose_mul (T : Finset n) : (eigCols hA T)ᴴ * eigCols hA T = 1 := by
  have hU := conjTranspose_mul_unitary hA.eigenvectorUnitary
  ext k l
  have h := congrFun (congrFun hU k.1) l.1
  rw [Matrix.mul_apply] at h
  rw [Matrix.mul_apply]
  simp only [conjTranspose_apply, eigCols, Matrix.of_apply] at h ⊢
  rw [h, Matrix.one_apply, Matrix.one_apply]
  by_cases hkl : k = l
  · simp [hkl]
  · have : k.1 ≠ l.1 := fun h' => hkl (Subtype.ext h')
    simp [hkl, this]

theorem sum_coe_sort_ind (T : Finset n) (f : n → 𝕜) :
    ∑ k : T, f k.1 = ∑ k, (ind T k : 𝕜) * f k := by
  have hT : (Finset.univ.filter fun k => k ∈ T) = T := by
    ext k
    simp
  calc ∑ k : T, f k.1 = ∑ k ∈ T, f k := Finset.sum_coe_sort T f
    _ = ∑ k ∈ Finset.univ.filter (fun k => k ∈ T), f k := by rw [hT]
    _ = ∑ k, if k ∈ T then f k else 0 := Finset.sum_filter _ _
    _ = ∑ k, (ind T k : 𝕜) * f k :=
        Finset.sum_congr rfl fun k _ => by by_cases h : k ∈ T <;> simp [ind, h]

theorem eigCols_mul_conjTranspose (T : Finset n) :
    eigCols hA T * (eigCols hA T)ᴴ = specProj hA T := by
  ext i j
  have h := sum_coe_sort_ind T (fun k => (hA.eigenvectorUnitary : Matrix n n 𝕜) i k *
    star ((hA.eigenvectorUnitary : Matrix n n 𝕜) j k))
  rw [Matrix.mul_apply, specProj, spec, Matrix.mul_apply]
  simp only [conjTranspose_apply, eigCols, Matrix.of_apply, dg, Matrix.mul_diagonal]
  rw [h]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- **Theorem 6**, first bound: `‖Q̂ Q̂ᴴ - P‖ ≤ s` for the polar factor `Q̂` of `X`, when the
cluster has as many vectors as `T` has indices and `s = ρ / sep < 1`. -/
theorem thm6_proj (T : Finset n) {X : Matrix n g 𝕜} {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) {Λ : Matrix g g 𝕜}
    {c w sep nR nA nΛ : ℝ} (hsep : 0 < sep) (hw : 0 ≤ w)
    (hΛ : ∀ z, ‖toEuclideanLin (Λ - (c : 𝕜) • (1 : Matrix g g 𝕜)) z‖ ≤ w * ‖z‖)
    (hgap : ∀ i, i ∉ T → w + sep ≤ |hA.eigenvalues i - c|)
    (hR : ∀ z, ‖toEuclideanLin (A * X - X * Λ) z‖ ≤ nR * ‖z‖) (hnA0 : 0 ≤ nA)
    (hnA : ∀ v, ‖toEuclideanLin A v‖ ≤ nA * ‖v‖) (hnΛ : ∀ z, ‖toEuclideanLin Λ z‖ ≤ nΛ * ‖z‖)
    (hcard : T.card = Fintype.card g)
    (hs0 : 0 ≤ (nR + (nA + nΛ) * (1 - √(1 - η))) / sep)
    (hs1 : (nR + (nA + nΛ) * (1 - √(1 - η))) / sep < 1) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (polarMat X * (polarMat X)ᴴ - specProj hA T) v‖ ≤
      (nR + (nA + nΛ) * (1 - √(1 - η))) / sep * ‖v‖ := by
  have hY := sinTheta hA T (polarMat X) hsep hw hΛ hgap
    (polarMat_residual hη1 hη hη0 hR hnA0 hnA hnΛ)
  rw [← eigCols_mul_conjTranspose hA T] at hY ⊢
  exact proj_sub_le (eigCols hA T) (polarMat X) (eigCols_conjTranspose_mul hA T)
    (polarMat_conjTranspose_mul hη1 hη) (by rw [Fintype.card_coe, hcard]) hs0 hs1 hY v

end Spectral

end CertifyMathlib.Counting
