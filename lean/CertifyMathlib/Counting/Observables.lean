import CertifyMathlib.Counting.E07

/-!
# C-07 (moments, Lemmas 1 and 2, Proposition 2) and C-04 (correlation lift, Lemma 0 and Proposition 1)

Sources: `docs/proofs/C-07_moments.md` and `docs/proofs/C-04_correlation_lift.md` in the code
repository. Mathlib v4.35.0 has no trace norm and no Hölder inequality for traces. The documents use
`|tr(E O)| ≤ ‖E‖₁ ‖O‖₂ ≤ rank(E) ‖E‖₂ ‖O‖₂`; the finite-dimensional fact that is needed,
`|tr(E O)| ≤ rank(E) ‖E‖ ‖O‖` for Hermitian `E`, is `norm_trace_herm_le` here (proved through an
orthonormal eigenbasis of `E` in `Theorem6.lean`).

* `c07_lemma1`: `|tr(P M) - tr(X Xᴴ M)| ≤ ((#T + g) s + g η) ‖M‖` (the document's `(2 g s + g η) ‖M‖`
  when `#T = g`), from `P - Q̂ Q̂ᴴ` (rank at most `#T + g`, norm at most `s`) and
  `Q̂ Q̂ᴴ - X Xᴴ = Q̂ (I - Xᴴ X) Q̂ᴴ` (rank at most `g`, norm at most `η`).
* `trace_mul_conjTranspose_eq`, `c07_lemma2`: `tr(X Xᴴ O Hᵏ O) = Σ_i ⟨O x_i, Hᵏ O x_i⟩` for
  Hermitian `O` (`x_i` the columns of `X`), `‖O Hᵏ O‖ ≤ ‖O‖² ‖H‖ᵏ`, and the moment bound.
* `c07_prop2_identity`, `c07_prop2_bounds`: Proposition 2 (f-sum form), the identity
  `tr(P O H O) - tr(P H P O²) = tr(P [O, [H, O]])/2` and `a tr(P O²) ≤ Re tr(P H P O²) ≤ b tr(P O²)`.
* `c04_lemma0`: `tr(P_s Vᴴ O V) = tr(V P_s Vᴴ O)`.
* `c04_prop1`: `|tr(Π_V O) - tr(Π_Q O)| ≤ 2 g ‖O‖ δ (1 + √(1 + ι))` with `V = Q M` the polar
  decomposition of the sector basis, `Π_V = V P_s Vᴴ`, `Π_Q = Q P_s Qᴴ`, `P_s = Q₁ Q₁ᴴ` an orthogonal
  projector of rank at most `g`.
* `c04_conclusion`: the exact part of the conclusion, `|v - tr(Π_V O)| ≤ b₆` and
  `|v - tr(Π_Q O)| ≤ b₆ + 2 g ‖O‖ δ (1 + √(1 + ι))`, from Theorem 6 in the block, Lemma 0 and
  Proposition 1.
-/

namespace CertifyMathlib.Counting

open Matrix
open scoped InnerProductSpace

variable {𝕜 : Type*} [RCLike 𝕜] {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g]
  [DecidableEq g]

/-- `|tr(E O)| ≤ r ‖E‖ ‖O‖` for Hermitian `E` of rank at most `r`. -/
theorem norm_trace_herm_le (E O : Matrix n n 𝕜) (hE : E.IsHermitian) {K nO : ℝ} {r : ℕ}
    (hK0 : 0 ≤ K) (hnO : 0 ≤ nO) (hK : ∀ v, ‖toEuclideanLin E v‖ ≤ K * ‖v‖)
    (hO : ∀ v, ‖toEuclideanLin O v‖ ≤ nO * ‖v‖) (hr : E.rank ≤ r) :
    ‖trace (E * O)‖ ≤ r * K * nO := by
  have h := norm_trace_mul_le E O (Enclosure.hermitianEigenData hE) hK hO
  have hrank : (Finset.univ.filter fun i => (Enclosure.hermitianEigenData hE).eig i ≠ 0).card ≤
      r := by
    rw [Enclosure.hermitianEigenData_eig, ← Fintype.card_subtype,
      ← hE.rank_eq_card_non_zero_eigs]
    exact hr
  refine h.trans ?_
  rw [mul_assoc, mul_assoc]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast hrank) (mul_nonneg hK0 hnO)

omit [DecidableEq n] in
theorem rank_conj_le {m : Type*} [Fintype m] [DecidableEq m] (Q : Matrix n m 𝕜)
    (B : Matrix m m 𝕜) : (Q * B * Qᴴ).rank ≤ Fintype.card m :=
  ((rank_mul_le_left _ _).trans (rank_mul_le_left _ _)).trans (rank_le_card_width _)

omit [DecidableEq n] in
theorem rank_conj_le' {m : Type*} [Fintype m] [DecidableEq m] (V : Matrix n m 𝕜)
    (P : Matrix m m 𝕜) : (V * P * Vᴴ).rank ≤ P.rank :=
  (rank_mul_le_left _ _).trans (rank_mul_le_right _ _)

/-- `X Xᴴ = Q̂ (Xᴴ X) Q̂ᴴ`. -/
theorem mul_conjTranspose_eq_polar {X : Matrix n g 𝕜} {η : ℝ} (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) :
    X * Xᴴ = polarMat X * (Xᴴ * X) * (polarMat X)ᴴ := by
  calc X * Xᴴ = (polarMat X * polarSqrt X) * (polarMat X * polarSqrt X)ᴴ := by
        rw [polarMat_mul_polarSqrt hη1 hη]
    _ = polarMat X * (polarSqrt X * polarSqrt X) * (polarMat X)ᴴ := by
        rw [conjTranspose_mul, polarSqrt_conjTranspose]
        simp only [Matrix.mul_assoc]
    _ = polarMat X * (Xᴴ * X) * (polarMat X)ᴴ := by rw [polarSqrt_mul_self hη1 hη]

/-- **C-07, Lemma 1** (projector term). -/
theorem c07_lemma1 {A : Matrix n n 𝕜} (hA : A.IsHermitian) (T : Finset n) {X : Matrix n g 𝕜}
    {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1) (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
    {s : ℝ} (hs0 : 0 ≤ s)
    (hs : ∀ v, ‖toEuclideanLin (polarMat X * (polarMat X)ᴴ - specProj hA T) v‖ ≤ s * ‖v‖)
    (Mo : Matrix n n 𝕜) {nM : ℝ} (hnM0 : 0 ≤ nM)
    (hM : ∀ v, ‖toEuclideanLin Mo v‖ ≤ nM * ‖v‖) :
    ‖trace (specProj hA T * Mo) - trace (X * Xᴴ * Mo)‖ ≤
      ((T.card + Fintype.card g) * s + Fintype.card g * η) * nM := by
  rw [← eigCols_mul_conjTranspose hA T] at hs ⊢
  set Q1 := eigCols hA T
  have hsplit : trace (Q1 * Q1ᴴ * Mo) - trace (X * Xᴴ * Mo) =
      trace ((Q1 * Q1ᴴ - polarMat X * (polarMat X)ᴴ) * Mo) +
        trace ((polarMat X * (polarMat X)ᴴ - X * Xᴴ) * Mo) := by
    rw [Matrix.sub_mul, Matrix.sub_mul, trace_sub, trace_sub]
    ring
  rw [hsplit]
  refine (norm_add_le _ _).trans ?_
  have hE1 : (Q1 * Q1ᴴ - polarMat X * (polarMat X)ᴴ).IsHermitian :=
    (isHermitian_mul_conjTranspose_self _).sub (isHermitian_mul_conjTranspose_self _)
  have hE2 : (polarMat X * (polarMat X)ᴴ - X * Xᴴ).IsHermitian :=
    (isHermitian_mul_conjTranspose_self _).sub (isHermitian_mul_conjTranspose_self _)
  have h1 := norm_trace_herm_le _ Mo hE1 (r := T.card + Fintype.card g) hs0 hnM0
    (fun v => by
      rw [← neg_sub, map_neg, LinearMap.neg_apply, norm_neg]
      exact hs v) hM
    (by
      refine (rank_sub_le' _ _).trans (add_le_add ?_ ?_)
      · rw [← Fintype.card_coe T]
        exact (rank_mul_le_left _ _).trans (rank_le_card_width _)
      · exact (rank_mul_le_left _ _).trans (rank_le_card_width _))
  have h2 := norm_trace_herm_le _ Mo hE2 (r := Fintype.card g) hη0 hnM0
    (fun v => by
      rw [← neg_sub, map_neg, LinearMap.neg_apply, norm_neg]
      exact mul_conjTranspose_sub_polar hη1 hη hη0 v) hM
    (by
      have e : polarMat X * (polarMat X)ᴴ - X * Xᴴ =
          polarMat X * (1 - Xᴴ * X) * (polarMat X)ᴴ := by
        rw [mul_conjTranspose_eq_polar hη1 hη, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one]
      rw [e]
      exact rank_conj_le _ _)
  push_cast at h1 h2
  nlinarith [h1, h2]

/-- `tr(X Xᴴ N) = Σ_i ⟨x_i, N x_i⟩` with `x_i` the columns of `X`. -/
theorem trace_mul_conjTranspose_eq (X : Matrix n g 𝕜) (N : Matrix n n 𝕜) :
    trace (X * Xᴴ * N) = ∑ j, ⟪toEuclideanLin X (EuclideanSpace.basisFun g 𝕜 j),
      toEuclideanLin N (toEuclideanLin X (EuclideanSpace.basisFun g 𝕜 j))⟫_𝕜 := by
  rw [Matrix.mul_assoc, Matrix.trace_mul_comm, trace_eq_sum_inner_matrix _
    (EuclideanSpace.basisFun g 𝕜)]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [toEuclideanLin_mul_apply, toEuclideanLin_mul_apply, ← inner_apply_conjTranspose]

/-- `‖Hᵏ v‖ ≤ ‖H‖ᵏ ‖v‖`. -/
theorem norm_pow_apply_le (H : Matrix n n 𝕜) {nH : ℝ} (hnH0 : 0 ≤ nH)
    (hH : ∀ v, ‖toEuclideanLin H v‖ ≤ nH * ‖v‖) (k : ℕ) (v : EuclideanSpace 𝕜 n) :
    ‖toEuclideanLin (H ^ k) v‖ ≤ nH ^ k * ‖v‖ := by
  induction k generalizing v with
  | zero => simp
  | succ k ih =>
      rw [pow_succ, toEuclideanLin_mul_apply, pow_succ, mul_assoc]
      refine (ih _).trans (mul_le_mul_of_nonneg_left ?_ (pow_nonneg hnH0 k))
      exact hH v

/-- **C-07, Lemma 2**: `tr(X Xᴴ O Hᵏ O) = Σ_i ⟨O x_i, Hᵏ O x_i⟩` for Hermitian `O`, and
`|tr(P O Hᵏ O) - Σ_i ⟨O x_i, Hᵏ O x_i⟩| ≤ ((#T + g) s + g η) ‖O‖² ‖H‖ᵏ`. -/
theorem c07_lemma2 {A : Matrix n n 𝕜} (hA : A.IsHermitian) (T : Finset n) {X : Matrix n g 𝕜}
    {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1) (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖)
    {s : ℝ} (hs0 : 0 ≤ s)
    (hs : ∀ v, ‖toEuclideanLin (polarMat X * (polarMat X)ᴴ - specProj hA T) v‖ ≤ s * ‖v‖)
    (O H : Matrix n n 𝕜) (hO : Oᴴ = O) {nO nH : ℝ} (hnO0 : 0 ≤ nO) (hnH0 : 0 ≤ nH)
    (hnO : ∀ v, ‖toEuclideanLin O v‖ ≤ nO * ‖v‖) (hnH : ∀ v, ‖toEuclideanLin H v‖ ≤ nH * ‖v‖)
    (k : ℕ) :
    trace (X * Xᴴ * (O * H ^ k * O)) = ∑ j,
      ⟪toEuclideanLin O (toEuclideanLin X (EuclideanSpace.basisFun g 𝕜 j)),
        toEuclideanLin (H ^ k) (toEuclideanLin O (toEuclideanLin X (EuclideanSpace.basisFun g 𝕜 j)))⟫_𝕜 ∧
    ‖trace (specProj hA T * (O * H ^ k * O)) - trace (X * Xᴴ * (O * H ^ k * O))‖ ≤
      ((T.card + Fintype.card g) * s + Fintype.card g * η) * (nO * (nH ^ k * nO)) := by
  constructor
  · rw [trace_mul_conjTranspose_eq]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [toEuclideanLin_mul_apply, toEuclideanLin_mul_apply]
    conv_lhs => rw [← inner_conjTranspose_apply]
    rw [hO]
  · refine c07_lemma1 hA T hη0 hη1 hη hs0 hs _ (mul_nonneg hnO0 (mul_nonneg
      (pow_nonneg hnH0 k) hnO0)) fun v => ?_
    rw [toEuclideanLin_mul_apply, toEuclideanLin_mul_apply]
    refine (hnO _).trans ?_
    rw [mul_assoc]
    refine mul_le_mul_of_nonneg_left ?_ hnO0
    refine (norm_pow_apply_le H hnH0 hnH k _).trans ?_
    rw [mul_assoc]
    exact mul_le_mul_of_nonneg_left (hnO v) (pow_nonneg hnH0 k)

omit [DecidableEq n] in
/-- **C-04, Lemma 0**: `tr(P_s Vᴴ O V) = tr(V P_s Vᴴ O)`. -/
theorem c04_lemma0 {d : Type*} [Fintype d] (V : Matrix n d 𝕜) (P : Matrix d d 𝕜)
    (O : Matrix n n 𝕜) : trace (P * (Vᴴ * O * V)) = trace (V * P * Vᴴ * O) := by
  calc trace (P * (Vᴴ * O * V)) = trace (Vᴴ * O * V * P) := Matrix.trace_mul_comm _ _
    _ = trace (Vᴴ * (O * V * P)) := by simp only [Matrix.mul_assoc]
    _ = trace (O * V * P * Vᴴ) := Matrix.trace_mul_comm _ _
    _ = trace (O * (V * P * Vᴴ)) := by simp only [Matrix.mul_assoc]
    _ = trace (V * P * Vᴴ * O) := Matrix.trace_mul_comm _ _

/-- **C-04, Proposition 1** (isometry term). -/
theorem c04_prop1 {d k : Type*} [Fintype d] [DecidableEq d] [Fintype k] [DecidableEq k]
    {V : Matrix n d 𝕜} {ι : ℝ} (hι0 : 0 ≤ ι) (hι1 : ι < 1)
    (hι : ∀ z, ‖toEuclideanLin (Vᴴ * V) z - z‖ ≤ ι * ‖z‖) (Q1 : Matrix d k 𝕜)
    (hQ1 : Q1ᴴ * Q1 = 1) (O : Matrix n n 𝕜) {nO : ℝ} (hnO0 : 0 ≤ nO)
    (hnO : ∀ v, ‖toEuclideanLin O v‖ ≤ nO * ‖v‖) :
    ‖trace (V * (Q1 * Q1ᴴ) * Vᴴ * O) - trace (polarMat V * (Q1 * Q1ᴴ) * (polarMat V)ᴴ * O)‖ ≤
      (2 * Fintype.card k) * ((1 - √(1 - ι)) * (1 + √(1 + ι))) * nO := by
  set P := Q1 * Q1ᴴ with hPdef
  have hPh : Pᴴ = P := by
    rw [hPdef, conjTranspose_mul, conjTranspose_conjTranspose]
  have hδ : 0 ≤ 1 - √(1 - ι) := Enclosure.delta_nonneg hι0
  have hQ := polarMat_conjTranspose_mul hι1 hι
  -- `‖P u‖ ≤ ‖u‖`
  have hP : ∀ u, ‖toEuclideanLin P u‖ ≤ 1 * ‖u‖ := fun u => by
    rw [hPdef, toEuclideanLin_mul_apply, norm_apply_of_isometry Q1 hQ1]
    exact norm_conjTranspose_le Q1 zero_le_one (fun z => by
      rw [norm_apply_of_isometry Q1 hQ1, one_mul]) u
  -- `‖M u‖ ≤ √(1 + ι) ‖u‖`
  have hMu : ∀ u, ‖toEuclideanLin (polarSqrt V) u‖ ≤ √(1 + ι) * ‖u‖ := fun u => by
    have e1 : ‖toEuclideanLin (polarSqrt V) u‖ = ‖toEuclideanLin V u‖ := by
      calc ‖toEuclideanLin (polarSqrt V) u‖ =
            ‖toEuclideanLin (polarMat V * polarSqrt V) u‖ := by
            rw [toEuclideanLin_mul_apply, norm_apply_of_isometry _ hQ]
        _ = ‖toEuclideanLin V u‖ := by rw [polarMat_mul_polarSqrt hι1 hι]
    rw [e1]
    have e2 : ‖toEuclideanLin V u‖ ^ 2 ≤ (1 + ι) * ‖u‖ ^ 2 := by
      have h1 := Enclosure.inner_toEuclideanLin_gram V u u
      rw [← inner_self_eq_norm_sq (𝕜 := 𝕜), h1]
      have h2 : RCLike.re ⟪toEuclideanLin (Vᴴ * V) u, u⟫_𝕜 =
          RCLike.re ⟪u, u⟫_𝕜 + RCLike.re ⟪toEuclideanLin (Vᴴ * V) u - u, u⟫_𝕜 := by
        rw [inner_sub_left, map_sub]
        ring
      rw [h2, inner_self_eq_norm_sq (𝕜 := 𝕜)]
      have h3 : RCLike.re ⟪toEuclideanLin (Vᴴ * V) u - u, u⟫_𝕜 ≤ ι * ‖u‖ * ‖u‖ :=
        (RCLike.re_le_norm _).trans ((norm_inner_le_norm _ _).trans
          (mul_le_mul_of_nonneg_right (hι u) (norm_nonneg _)))
      nlinarith
    calc ‖toEuclideanLin V u‖ = √(‖toEuclideanLin V u‖ ^ 2) :=
          (Real.sqrt_sq (norm_nonneg _)).symm
      _ ≤ √((1 + ι) * ‖u‖ ^ 2) := Real.sqrt_le_sqrt e2
      _ = √(1 + ι) * ‖u‖ := by rw [Real.sqrt_mul (by linarith), Real.sqrt_sq (norm_nonneg _)]
  -- the difference `Π_V - Π_Q = Q (M P M - P) Qᴴ`
  have hdiff : V * P * Vᴴ - polarMat V * P * (polarMat V)ᴴ =
      polarMat V * ((polarSqrt V - 1) * P * polarSqrt V + P * (polarSqrt V - 1)) *
        (polarMat V)ᴴ := by
    have hV : V * P * Vᴴ = polarMat V * (polarSqrt V * P * polarSqrt V) * (polarMat V)ᴴ := by
      calc V * P * Vᴴ = (polarMat V * polarSqrt V) * P * (polarMat V * polarSqrt V)ᴴ := by
            rw [polarMat_mul_polarSqrt hι1 hι]
        _ = polarMat V * (polarSqrt V * P * polarSqrt V) * (polarMat V)ᴴ := by
            rw [conjTranspose_mul, polarSqrt_conjTranspose]
            simp only [Matrix.mul_assoc]
    rw [hV]
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.add_mul, Matrix.mul_add, Matrix.one_mul,
      Matrix.mul_one, Matrix.mul_assoc]
    abel
  have hE : (V * P * Vᴴ - polarMat V * P * (polarMat V)ᴴ).IsHermitian := by
    have hPH : P.IsHermitian := hPh
    exact (isHermitian_mul_mul_conjTranspose V hPH).sub
      (isHermitian_mul_mul_conjTranspose (polarMat V) hPH)
  have hnorm : ∀ v, ‖toEuclideanLin (V * P * Vᴴ - polarMat V * P * (polarMat V)ᴴ) v‖ ≤
      ((1 - √(1 - ι)) * (1 + √(1 + ι))) * ‖v‖ := fun v => by
    rw [hdiff, toEuclideanLin_mul_apply, toEuclideanLin_mul_apply, norm_apply_of_isometry _ hQ]
    have hQh : ‖toEuclideanLin (polarMat V)ᴴ v‖ ≤ 1 * ‖v‖ :=
      norm_conjTranspose_le _ zero_le_one (fun z => by
        rw [norm_apply_of_isometry _ hQ, one_mul]) v
    rw [one_mul] at hQh
    set w := toEuclideanLin (polarMat V)ᴴ v
    rw [map_add, LinearMap.add_apply]
    refine (norm_add_le _ _).trans ?_
    have t1 := norm_mul3_le (polarSqrt V - 1) P (polarSqrt V) hδ zero_le_one
      (polarSqrt_sub_le hι1 hι hι0) hP hMu w
    have t2 : ‖toEuclideanLin (P * (polarSqrt V - 1)) w‖ ≤ 1 * ((1 - √(1 - ι)) * ‖w‖) := by
      rw [toEuclideanLin_mul_apply]
      exact (hP _).trans (mul_le_mul_of_nonneg_left (polarSqrt_sub_le hι1 hι hι0 w) zero_le_one)
    have hw := hQh
    have hs : 0 ≤ √(1 + ι) := Real.sqrt_nonneg _
    calc _ ≤ (1 - √(1 - ι)) * (1 * (√(1 + ι) * ‖w‖)) + 1 * ((1 - √(1 - ι)) * ‖w‖) :=
          add_le_add t1 t2
      _ = ((1 - √(1 - ι)) * (1 + √(1 + ι))) * ‖w‖ := by ring
      _ ≤ ((1 - √(1 - ι)) * (1 + √(1 + ι))) * ‖v‖ :=
          mul_le_mul_of_nonneg_left hw (mul_nonneg hδ (by linarith))
  have hrank : (V * P * Vᴴ - polarMat V * P * (polarMat V)ᴴ).rank ≤ 2 * Fintype.card k := by
    have hPr : P.rank ≤ Fintype.card k := by
      rw [hPdef]
      exact (rank_mul_le_left _ _).trans (rank_le_card_width _)
    refine (rank_sub_le' _ _).trans ?_
    have := (rank_conj_le' V P).trans hPr
    have := (rank_conj_le' (polarMat V) P).trans hPr
    omega
  have hsub : trace (V * P * Vᴴ * O) - trace (polarMat V * P * (polarMat V)ᴴ * O) =
      trace ((V * P * Vᴴ - polarMat V * P * (polarMat V)ᴴ) * O) := by
    rw [Matrix.sub_mul, trace_sub]
  rw [hsub]
  have h := norm_trace_herm_le _ O hE (mul_nonneg hδ (by
    have := Real.sqrt_nonneg (1 + ι)
    linarith)) hnO0 hnorm hnO hrank
  push_cast at h
  exact h

/-- **C-04, conclusion** (exact part, no rounding): if `|v - tr(P_s Vᴴ O V)| ≤ b₆` (Theorem 6 in the
block), then `|v - tr(Π_V O)| ≤ b₆` and `|v - tr(Π_Q O)| ≤ b₆ + 2 g ‖O‖ δ (1 + √(1 + ι))`. -/
theorem c04_conclusion {d k : Type*} [Fintype d] [DecidableEq d] [Fintype k] [DecidableEq k]
    {V : Matrix n d 𝕜} {ι : ℝ} (hι0 : 0 ≤ ι) (hι1 : ι < 1)
    (hι : ∀ z, ‖toEuclideanLin (Vᴴ * V) z - z‖ ≤ ι * ‖z‖) (Q1 : Matrix d k 𝕜)
    (hQ1 : Q1ᴴ * Q1 = 1) (O : Matrix n n 𝕜) {nO : ℝ} (hnO0 : 0 ≤ nO)
    (hnO : ∀ v, ‖toEuclideanLin O v‖ ≤ nO * ‖v‖) (v : 𝕜) {b6 : ℝ}
    (hb6 : ‖v - trace (Q1 * Q1ᴴ * (Vᴴ * O * V))‖ ≤ b6) :
    ‖v - trace (V * (Q1 * Q1ᴴ) * Vᴴ * O)‖ ≤ b6 ∧
    ‖v - trace (polarMat V * (Q1 * Q1ᴴ) * (polarMat V)ᴴ * O)‖ ≤
      b6 + (2 * Fintype.card k) * ((1 - √(1 - ι)) * (1 + √(1 + ι))) * nO := by
  have h0 := c04_lemma0 V (Q1 * Q1ᴴ) O
  have h1 : ‖v - trace (V * (Q1 * Q1ᴴ) * Vᴴ * O)‖ ≤ b6 := by
    rw [← h0]
    exact hb6
  refine ⟨h1, ?_⟩
  have h2 := c04_prop1 hι0 hι1 hι Q1 hQ1 O hnO0 hnO
  calc ‖v - trace (polarMat V * (Q1 * Q1ᴴ) * (polarMat V)ᴴ * O)‖ =
        ‖(v - trace (V * (Q1 * Q1ᴴ) * Vᴴ * O)) + (trace (V * (Q1 * Q1ᴴ) * Vᴴ * O) -
          trace (polarMat V * (Q1 * Q1ᴴ) * (polarMat V)ᴴ * O))‖ := by
        rw [sub_add_sub_cancel]
    _ ≤ ‖v - trace (V * (Q1 * Q1ᴴ) * Vᴴ * O)‖ + ‖trace (V * (Q1 * Q1ᴴ) * Vᴴ * O) -
          trace (polarMat V * (Q1 * Q1ᴴ) * (polarMat V)ᴴ * O)‖ := norm_add_le _ _
    _ ≤ b6 + (2 * Fintype.card k) * ((1 - √(1 - ι)) * (1 + √(1 + ι))) * nO := add_le_add h1 h2

omit [DecidableEq n] in
/-- **C-07, Proposition 2** (f-sum identity). For an idempotent `P` commuting with `H`,
`tr(P O H O) - tr(P H P O²) = tr(P [O, [H, O]]) / 2` with `[O, [H, O]] = 2 O H O - O² H - H O²`;
dividing by `g` gives `m₁ - tr(P H P O²)/g = tr(P [O, [H, O]])/(2 g)`. -/
theorem c07_prop2_identity (P H O : Matrix n n 𝕜) (hP2 : P * P = P) (hPH : P * H = H * P) :
    trace (P * (O * H * O)) - trace (P * H * P * (O * O)) =
      trace (P * ((2 : 𝕜) • (O * H * O) - O * O * H - H * (O * O))) / 2 := by
  have e1 : trace (P * (O * O * H)) = trace (P * H * (O * O)) := by
    calc trace (P * (O * O * H)) = trace ((P * (O * O)) * H) := by simp only [Matrix.mul_assoc]
      _ = trace (H * (P * (O * O))) := Matrix.trace_mul_comm _ _
      _ = trace (P * H * (O * O)) := by rw [← Matrix.mul_assoc, ← hPH]
  have e2 : P * H * P = P * H := by
    rw [Matrix.mul_assoc, ← hPH, ← Matrix.mul_assoc, hP2]
  have e3 : trace (P * (H * (O * O))) = trace (P * H * (O * O)) := by rw [Matrix.mul_assoc]
  rw [e2, Matrix.mul_sub, Matrix.mul_sub, Matrix.trace_sub, Matrix.trace_sub, Matrix.mul_smul,
    Matrix.trace_smul, e1, e3, smul_eq_mul]
  ring

/-- **C-07, Proposition 2** (enclosure of `tr(P H P O²)`). If `P` is an orthogonal projector, `O` is
Hermitian and `a ‖P x‖² ≤ Re ⟨P x, H P x⟩ ≤ b ‖P x‖²` for all `x` (the eigenvalues of `H` on the
range of `P` lie in `[a, b]`), then `Re tr(P O²) = Σ_i ‖P O e_i‖² ≥ 0` and
`a Re tr(P O²) ≤ Re tr(P H P O²) ≤ b Re tr(P O²)`. -/
theorem c07_prop2_bounds (P H O : Matrix n n 𝕜) (hPh : Pᴴ = P) (hP2 : P * P = P) (hO : Oᴴ = O)
    {a b : ℝ}
    (hab : ∀ x : EuclideanSpace 𝕜 n,
      a * ‖toEuclideanLin P x‖ ^ 2 ≤
        RCLike.re ⟪toEuclideanLin P x, toEuclideanLin H (toEuclideanLin P x)⟫_𝕜 ∧
      RCLike.re ⟪toEuclideanLin P x, toEuclideanLin H (toEuclideanLin P x)⟫_𝕜 ≤
        b * ‖toEuclideanLin P x‖ ^ 2) :
    RCLike.re (trace (P * (O * O))) =
      ∑ j, ‖toEuclideanLin P (toEuclideanLin O (EuclideanSpace.basisFun n 𝕜 j))‖ ^ 2 ∧
    a * RCLike.re (trace (P * (O * O))) ≤ RCLike.re (trace (P * H * P * (O * O))) ∧
    RCLike.re (trace (P * H * P * (O * O))) ≤ b * RCLike.re (trace (P * (O * O))) := by
  have hz : ∀ j, toEuclideanLin (P * O) (EuclideanSpace.basisFun n 𝕜 j) =
      toEuclideanLin P (toEuclideanLin O (EuclideanSpace.basisFun n 𝕜 j)) := fun j =>
    toEuclideanLin_mul_apply _ _ _
  have t1 : trace (P * (O * O)) = trace (P * O * (P * O)ᴴ * 1) := by
    rw [Matrix.mul_one, conjTranspose_mul, hO, hPh]
    calc trace (P * (O * O)) = trace (P * (P * (O * O))) := by rw [← Matrix.mul_assoc P P, hP2]
      _ = trace (P * (O * O) * P) := Matrix.trace_mul_comm _ _
      _ = trace (P * O * (O * P)) := by simp only [Matrix.mul_assoc]
  have t2 : trace (P * H * P * (O * O)) = trace (P * O * (P * O)ᴴ * H) := by
    rw [conjTranspose_mul, hO, hPh]
    calc trace (P * H * P * (O * O)) = trace ((P * H) * (P * (O * O))) := by
          simp only [Matrix.mul_assoc]
      _ = trace ((P * (O * O)) * (P * H)) := Matrix.trace_mul_comm _ _
      _ = trace (P * O * (O * P) * H) := by simp only [Matrix.mul_assoc]
  rw [t1, t2, trace_mul_conjTranspose_eq, trace_mul_conjTranspose_eq]
  simp only [hz, toEuclideanLin_one_apply, map_sum, inner_self_eq_norm_sq]
  refine ⟨trivial, ?_, ?_⟩
  · rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => (hab _).1
  · rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => (hab _).2

end CertifyMathlib.Counting
