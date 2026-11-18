import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Lp.Matrix

/-!
# Bauer-Fike and the residual forms of E-02

The exact statements behind the non-Hermitian claim kind of CERTIFY-ED (derivation document
`docs/proofs/E-02_non_hermitian.md` of the code repository; roadmap items E-02 and N-13).

* `bauerFike_core`, `bauerFike`, `bauerFike_matrix`: Theorem 2 of the document. If `A = V D V⁻¹`
  with `D` diagonal, every eigenvalue `λ` of `A + E` satisfies `min_i |λ - d_i| ≤ ‖V⁻¹‖ ‖E‖ ‖V‖`;
  `bauerFike_matrix_sharp` is the bound `‖V⁻¹ E V‖`.
* `bauerFike_residual`, `bauerFike_residual_matrix`: the residual form of Theorem 2,
  `min_i |d_i - μ| ≤ ‖V‖ ‖V⁻¹‖ ‖A x - μ x‖ / ‖x‖` for every `x ≠ 0`.
* `backward_error`: Lemma 1 of the document. `μ` is an exact eigenvalue of `A + K` for a `K` of
  norm at most `‖A x - μ x‖ / ‖x‖`.
* `neumann_bound`, `neumann_inverse`: Lemma 3 of the document. From `‖I - W V‖ ≤ θ < 1` the matrix
  `V` is invertible with `‖V⁻¹‖ ≤ ‖W‖ / (1 - θ)`.
* `similarity_inclusion`, `disc_inclusion_matrix`, `disc_inclusion_certified`: Theorem 4 (b) of
  the document, the form the code uses. If `A V = V diag(μ) + R` and `V` is invertible, every
  eigenvalue of `A` lies within `‖V⁻¹ R‖` of some `μ_i`; with the bound of Lemma 3, within
  `‖W‖ / (1 - θ) * ‖R‖`. `A` need not be diagonalizable.

Norms of operators are written by their defining bound, as in `CertifyMathlib.Enclosure`:
"`‖R‖ ≤ r`" is the hypothesis `∀ v, ‖R v‖ ≤ r * ‖v‖`. Vectors live in `EuclideanSpace 𝕜 n`, so
the norm is the Euclidean norm of the code, and matrices act by `Matrix.toEuclideanLin`. The count
of eigenvalues per group of discs (Theorem 4 (c)) and the floating-point accounting
(Propositions 5 to 7) are not formalized here.
-/

namespace CertifyMathlib.NonHermitian

section Abstract

variable {𝕜 : Type*} [RCLike 𝕜] {E F : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E]
  [NormedAddCommGroup F] [NormedSpace 𝕜 F]

/-- **Bauer-Fike**, vector form. `T P = P D` with `P` invertible (inverse `Q`); if every vector
`w` satisfies `m ‖w‖ ≤ ‖D w - λ w‖` (for a diagonal `D`, `m = min_i |d_i - λ|`), then an
eigenvalue `λ` of `T + K` forces `m ≤ ‖Q‖ ‖K‖ ‖P‖`. -/
theorem bauerFike_core (T K : E →ₗ[𝕜] E) (P : F →ₗ[𝕜] E) (Q : E →ₗ[𝕜] F) (D : F →ₗ[𝕜] F)
    (hQP : ∀ w, Q (P w) = w) (hPQ : ∀ x, P (Q x) = x) (hT : ∀ w, T (P w) = P (D w))
    {cP cQ e m : ℝ} (hcQ : 0 ≤ cQ) (he : 0 ≤ e)
    (hP : ∀ w, ‖P w‖ ≤ cP * ‖w‖) (hQ : ∀ x, ‖Q x‖ ≤ cQ * ‖x‖) (hK : ∀ x, ‖K x‖ ≤ e * ‖x‖)
    {lam : 𝕜} (hD : ∀ w, m * ‖w‖ ≤ ‖D w - lam • w‖) {x : E} (hx : x ≠ 0)
    (hlam : T x + K x = lam • x) : m ≤ cQ * e * cP := by
  obtain ⟨w, rfl⟩ : ∃ w, x = P w := ⟨Q x, (hPQ x).symm⟩
  have hw0 : w ≠ 0 := by
    rintro rfl
    exact hx (map_zero P)
  have hwpos : 0 < ‖w‖ := norm_pos_iff.mpr hw0
  have h1 : D w - lam • w = -Q (K (P w)) := by
    have h := congrArg Q hlam
    rw [map_add, hT, hQP, map_smul, hQP] at h
    rw [← h]
    abel
  have h2 : m * ‖w‖ ≤ cQ * e * cP * ‖w‖ := by
    calc m * ‖w‖ ≤ ‖D w - lam • w‖ := hD w
      _ = ‖Q (K (P w))‖ := by rw [h1, norm_neg]
      _ ≤ cQ * ‖K (P w)‖ := hQ _
      _ ≤ cQ * (e * ‖P w‖) := mul_le_mul_of_nonneg_left (hK _) hcQ
      _ ≤ cQ * (e * (cP * ‖w‖)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (hP _) he) hcQ
      _ = cQ * e * cP * ‖w‖ := by ring
  exact le_of_mul_le_mul_right h2 hwpos

/-- The **residual form** of Bauer-Fike: for `T P = P D` and any `x ≠ 0`,
`m ≤ ‖P‖ ‖Q‖ ‖T x - μ x‖ / ‖x‖`, where `m ‖w‖ ≤ ‖D w - μ w‖` for all `w`. -/
theorem bauerFike_residual (T : E →ₗ[𝕜] E) (P : F →ₗ[𝕜] E) (Q : E →ₗ[𝕜] F) (D : F →ₗ[𝕜] F)
    (hQP : ∀ w, Q (P w) = w) (hPQ : ∀ x, P (Q x) = x) (hT : ∀ w, T (P w) = P (D w))
    {cP cQ m : ℝ} (hcP : 0 ≤ cP) (hm : 0 ≤ m)
    (hP : ∀ w, ‖P w‖ ≤ cP * ‖w‖) (hQ : ∀ x, ‖Q x‖ ≤ cQ * ‖x‖)
    {mu : 𝕜} (hD : ∀ w, m * ‖w‖ ≤ ‖D w - mu • w‖) {x : E} (hx : x ≠ 0) :
    m ≤ cP * cQ * ‖T x - mu • x‖ / ‖x‖ := by
  have hxpos : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hQT : Q (T x - mu • x) = D (Q x) - mu • Q x := by
    have hx' : T x - mu • x = T (P (Q x)) - mu • P (Q x) := by rw [hPQ]
    rw [hx', map_sub, hT, hQP, map_smul, hQP]
  have h1 : m * ‖x‖ ≤ cP * cQ * ‖T x - mu • x‖ := by
    calc m * ‖x‖ = m * ‖P (Q x)‖ := by rw [hPQ]
      _ ≤ m * (cP * ‖Q x‖) := mul_le_mul_of_nonneg_left (hP _) hm
      _ = cP * (m * ‖Q x‖) := by ring
      _ ≤ cP * ‖D (Q x) - mu • Q x‖ := mul_le_mul_of_nonneg_left (hD _) hcP
      _ = cP * ‖Q (T x - mu • x)‖ := by rw [hQT]
      _ ≤ cP * (cQ * ‖T x - mu • x‖) := mul_le_mul_of_nonneg_left (hQ _) hcP
      _ = cP * cQ * ‖T x - mu • x‖ := by ring
  rw [le_div_iff₀ hxpos]
  exact h1

/-- **Theorem 4 (b)**, operator form: if `T P = P D + R` with `P` invertible (inverse `Q`), every
eigenvalue `λ` of `T` satisfies `m ≤ f` whenever `‖Q R‖ ≤ f` and `m ‖w‖ ≤ ‖D w - λ w‖`. -/
theorem similarity_inclusion (T : E →ₗ[𝕜] E) (P : F →ₗ[𝕜] E) (Q : E →ₗ[𝕜] F) (D : F →ₗ[𝕜] F)
    (R : F →ₗ[𝕜] E) (hQP : ∀ w, Q (P w) = w) (hPQ : ∀ x, P (Q x) = x)
    (hT : ∀ w, T (P w) = P (D w) + R w) {f m : ℝ} (hQR : ∀ w, ‖Q (R w)‖ ≤ f * ‖w‖)
    {lam : 𝕜} (hD : ∀ w, m * ‖w‖ ≤ ‖D w - lam • w‖) {x : E} (hx : x ≠ 0)
    (hlam : T x = lam • x) : m ≤ f := by
  obtain ⟨w, rfl⟩ : ∃ w, x = P w := ⟨Q x, (hPQ x).symm⟩
  have hw0 : w ≠ 0 := by
    rintro rfl
    exact hx (map_zero P)
  have hwpos : 0 < ‖w‖ := norm_pos_iff.mpr hw0
  have h1 : D w - lam • w = -Q (R w) := by
    have h := congrArg Q hlam
    rw [hT, map_add, hQP, map_smul, hQP] at h
    rw [← h]
    abel
  have h2 : m * ‖w‖ ≤ f * ‖w‖ := by
    calc m * ‖w‖ ≤ ‖D w - lam • w‖ := hD w
      _ = ‖Q (R w)‖ := by rw [h1, norm_neg]
      _ ≤ f * ‖w‖ := hQR w
  exact le_of_mul_le_mul_right h2 hwpos

/-- **Lemma 3**, the inequality: `‖I - W V‖ ≤ θ` and `‖W‖ ≤ c_W` give
`(1 - θ) ‖z‖ ≤ c_W ‖V z‖`. -/
theorem neumann_bound (V W : E →ₗ[𝕜] E) {θ cW : ℝ} (hG : ∀ z, ‖z - W (V z)‖ ≤ θ * ‖z‖)
    (hW : ∀ y, ‖W y‖ ≤ cW * ‖y‖) (z : E) : (1 - θ) * ‖z‖ ≤ cW * ‖V z‖ := by
  have h1 : ‖z‖ ≤ ‖W (V z)‖ + ‖z - W (V z)‖ := by
    calc ‖z‖ = ‖W (V z) + (z - W (V z))‖ := by
          congr 1
          abel
      _ ≤ ‖W (V z)‖ + ‖z - W (V z)‖ := norm_add_le _ _
  have h2 := hW (V z)
  have h3 := hG z
  linarith

/-- **Lemma 3**: in finite dimension, `‖I - W V‖ ≤ θ < 1` makes `V` invertible, with
`‖V⁻¹ y‖ ≤ c_W / (1 - θ) ‖y‖`. -/
theorem neumann_inverse [FiniteDimensional 𝕜 E] (V W : E →ₗ[𝕜] E) {θ cW : ℝ} (hθ : θ < 1)
    (hG : ∀ z, ‖z - W (V z)‖ ≤ θ * ‖z‖) (hW : ∀ y, ‖W y‖ ≤ cW * ‖y‖) :
    ∃ Vi : E →ₗ[𝕜] E, (∀ y, V (Vi y) = y) ∧ (∀ z, Vi (V z) = z) ∧
      ∀ y, ‖Vi y‖ ≤ cW / (1 - θ) * ‖y‖ := by
  have hpos : 0 < 1 - θ := by linarith
  have hinj : Function.Injective V := by
    intro a b hab
    have h := neumann_bound V W hG hW (a - b)
    rw [map_sub, hab, sub_self, norm_zero, mul_zero] at h
    have h0 : ‖a - b‖ ≤ 0 := by
      by_contra hc
      have hc' : 0 < ‖a - b‖ := not_le.mp hc
      have := mul_pos hpos hc'
      linarith
    exact sub_eq_zero.mp (norm_le_zero_iff.mp h0)
  have hsurj : Function.Surjective V := LinearMap.injective_iff_surjective.mp hinj
  let e : E ≃ₗ[𝕜] E := LinearEquiv.ofBijective V ⟨hinj, hsurj⟩
  have he : ∀ z, e z = V z := fun z => rfl
  refine ⟨e.symm.toLinearMap, fun y => ?_, fun z => ?_, fun y => ?_⟩
  · show V (e.symm y) = y
    rw [← he, LinearEquiv.apply_symm_apply]
  · show e.symm (V z) = z
    rw [← he, LinearEquiv.symm_apply_apply]
  · show ‖e.symm y‖ ≤ cW / (1 - θ) * ‖y‖
    have h := neumann_bound V W hG hW (e.symm y)
    rw [← he, LinearEquiv.apply_symm_apply] at h
    rw [div_mul_eq_mul_div, le_div_iff₀ hpos]
    linarith

end Abstract

section Backward

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

open scoped InnerProductSpace

/-- **Lemma 1** (backward error of a residual): for `x ≠ 0` and any `μ`, the rank-one map
`K v = -(⟪x, v⟫ / ‖x‖²) (T x - μ x)` satisfies `(T + K) x = μ x` and
`‖K v‖ ≤ ‖T x - μ x‖ / ‖x‖ * ‖v‖`. -/
theorem backward_error (T : E →ₗ[𝕜] E) (mu : 𝕜) {x : E} (hx : x ≠ 0) :
    ∃ K : E →ₗ[𝕜] E, T x + K x = mu • x ∧ ∀ v, ‖K v‖ ≤ ‖T x - mu • x‖ / ‖x‖ * ‖v‖ := by
  have hxpos : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hxne : ‖x‖ ≠ 0 := hxpos.ne'
  set r := T x - mu • x with hr
  let c : 𝕜 := -((‖x‖ ^ 2 : ℝ) : 𝕜)⁻¹
  have hin : ∀ v, innerₛₗ 𝕜 x v = ⟪x, v⟫_𝕜 := fun v => rfl
  refine ⟨c • LinearMap.smulRight (innerₛₗ 𝕜 x) r, ?_, fun v => ?_⟩
  · have hxx : ⟪x, x⟫_𝕜 = ((‖x‖ ^ 2 : ℝ) : 𝕜) := by
      rw [inner_self_eq_norm_sq_to_K]
      norm_cast
    have hne : ((‖x‖ ^ 2 : ℝ) : 𝕜) ≠ 0 := by
      have : (‖x‖ ^ 2 : ℝ) ≠ 0 := by positivity
      exact_mod_cast this
    simp only [LinearMap.smul_apply, LinearMap.smulRight_apply, hin, hxx, c]
    rw [smul_smul, neg_mul, inv_mul_cancel₀ hne, neg_smul, one_smul, hr]
    abel
  · simp only [LinearMap.smul_apply, LinearMap.smulRight_apply, hin, c]
    rw [norm_smul, norm_smul, norm_neg, norm_inv, RCLike.norm_ofReal,
      abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖x‖ ^ 2)]
    have hcs : ‖⟪x, v⟫_𝕜‖ ≤ ‖x‖ * ‖v‖ := norm_inner_le_norm x v
    calc (‖x‖ ^ 2)⁻¹ * (‖⟪x, v⟫_𝕜‖ * ‖r‖) ≤ (‖x‖ ^ 2)⁻¹ * (‖x‖ * ‖v‖ * ‖r‖) := by
          gcongr
      _ = ‖r‖ / ‖x‖ * ‖v‖ := by
          field_simp

end Backward

section Euclid

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n]

/-- A diagonal map `(D w)_i = d_i w_i` on `EuclideanSpace 𝕜 n` satisfies
`m ‖w‖ ≤ ‖D w - λ w‖` for every `0 ≤ m ≤ min_i |d_i - λ|`. -/
theorem diag_lower (D : EuclideanSpace 𝕜 n →ₗ[𝕜] EuclideanSpace 𝕜 n) (d : n → 𝕜)
    (hDd : ∀ w i, D w i = d i * w i) (lam : 𝕜) {m : ℝ} (hm : 0 ≤ m)
    (hmd : ∀ i, m ≤ ‖d i - lam‖) (w : EuclideanSpace 𝕜 n) :
    m * ‖w‖ ≤ ‖D w - lam • w‖ := by
  have hsq : (m * ‖w‖) ^ 2 ≤ ‖D w - lam • w‖ ^ 2 := by
    rw [mul_pow, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have hi : (D w - lam • w) i = (d i - lam) * w i := by
      rw [PiLp.sub_apply, PiLp.smul_apply, hDd, smul_eq_mul, sub_mul]
    rw [hi, norm_mul, mul_pow]
    exact mul_le_mul_of_nonneg_right (pow_le_pow_left₀ hm (hmd i) 2) (by positivity)
  exact (sq_le_sq₀ (mul_nonneg hm (norm_nonneg _)) (norm_nonneg _)).mp hsq

/-- **Bauer-Fike** on `EuclideanSpace 𝕜 n` with a diagonal `D`: every eigenvalue `λ` of `T + K`
lies within `‖Q‖ ‖K‖ ‖P‖` of some `d_i`. -/
theorem bauerFike [Nonempty n] (T K P Q D : EuclideanSpace 𝕜 n →ₗ[𝕜] EuclideanSpace 𝕜 n)
    (d : n → 𝕜) (hDd : ∀ w i, D w i = d i * w i)
    (hQP : ∀ w, Q (P w) = w) (hPQ : ∀ x, P (Q x) = x) (hT : ∀ w, T (P w) = P (D w))
    {cP cQ e : ℝ} (hcQ : 0 ≤ cQ) (he : 0 ≤ e)
    (hP : ∀ w, ‖P w‖ ≤ cP * ‖w‖) (hQ : ∀ x, ‖Q x‖ ≤ cQ * ‖x‖) (hK : ∀ x, ‖K x‖ ≤ e * ‖x‖)
    {lam : 𝕜} {x : EuclideanSpace 𝕜 n} (hx : x ≠ 0) (hlam : T x + K x = lam • x) :
    ∃ i, ‖lam - d i‖ ≤ cQ * e * cP := by
  obtain ⟨i, -, hi⟩ :=
    Finset.exists_min_image Finset.univ (fun i => ‖d i - lam‖) Finset.univ_nonempty
  refine ⟨i, ?_⟩
  rw [norm_sub_rev]
  exact bauerFike_core T K P Q D hQP hPQ hT hcQ he hP hQ hK
    (diag_lower D d hDd lam (norm_nonneg _) (fun j => hi j (Finset.mem_univ j))) hx hlam

/-! ### Matrices -/

open Matrix

variable [DecidableEq n]

theorem toEuclideanLin_diagonal_apply (d : n → 𝕜) (w : EuclideanSpace 𝕜 n) (i : n) :
    toEuclideanLin (diagonal d) w i = d i * w i := by
  simp [Matrix.toLpLin_apply, Matrix.mulVec_diagonal]

theorem toEuclideanLin_mul_apply (X Y : Matrix n n 𝕜) (w : EuclideanSpace 𝕜 n) :
    toEuclideanLin (X * Y) w = toEuclideanLin X (toEuclideanLin Y w) := by
  simp [Matrix.toLpLin_apply, Matrix.mulVec_mulVec]

theorem toEuclideanLin_one_apply (w : EuclideanSpace 𝕜 n) :
    toEuclideanLin (1 : Matrix n n 𝕜) w = w := by
  simp

/-- **Bauer-Fike** for matrices (Theorem 2 of the document): if `A = V diag(d) W` with
`W = V⁻¹`, every eigenvalue `λ` of `A + E` satisfies `|λ - d_i| ≤ ‖W‖ ‖E‖ ‖V‖` for some `i`. -/
theorem bauerFike_matrix [Nonempty n] (A V W Em : Matrix n n 𝕜) (d : n → 𝕜)
    (hWV : W * V = 1) (hVW : V * W = 1) (hA : A = V * diagonal d * W)
    {cV cW e : ℝ} (hcW : 0 ≤ cW) (he : 0 ≤ e)
    (hV : ∀ v, ‖toEuclideanLin V v‖ ≤ cV * ‖v‖) (hW : ∀ v, ‖toEuclideanLin W v‖ ≤ cW * ‖v‖)
    (hE : ∀ v, ‖toEuclideanLin Em v‖ ≤ e * ‖v‖) {lam : 𝕜} {x : EuclideanSpace 𝕜 n}
    (hx : x ≠ 0) (hlam : toEuclideanLin (A + Em) x = lam • x) :
    ∃ i, ‖lam - d i‖ ≤ cW * e * cV := by
  have hAV : A * V = V * diagonal d := by
    rw [hA, Matrix.mul_assoc, hWV, Matrix.mul_one]
  refine bauerFike (toEuclideanLin A) (toEuclideanLin Em) (toEuclideanLin V) (toEuclideanLin W)
    (toEuclideanLin (diagonal d)) d (toEuclideanLin_diagonal_apply d) (fun w => ?_) (fun w => ?_)
    (fun w => ?_) hcW he hV hW hE hx ?_
  · rw [← toEuclideanLin_mul_apply W V, hWV, toEuclideanLin_one_apply]
  · rw [← toEuclideanLin_mul_apply V W, hVW, toEuclideanLin_one_apply]
  · rw [← toEuclideanLin_mul_apply A V, hAV, toEuclideanLin_mul_apply]
  · rw [map_add, LinearMap.add_apply] at hlam
    exact hlam

/-- The residual form for matrices: for `A = V diag(d) W`, `W = V⁻¹`, and any `x ≠ 0`, some `d_i`
lies within `‖V‖ ‖W‖ ‖A x - μ x‖ / ‖x‖` of `μ`. -/
theorem bauerFike_residual_matrix [Nonempty n] (A V W : Matrix n n 𝕜) (d : n → 𝕜)
    (hWV : W * V = 1) (hVW : V * W = 1) (hA : A = V * diagonal d * W)
    {cV cW : ℝ} (hcV : 0 ≤ cV)
    (hV : ∀ v, ‖toEuclideanLin V v‖ ≤ cV * ‖v‖) (hW : ∀ v, ‖toEuclideanLin W v‖ ≤ cW * ‖v‖)
    (mu : 𝕜) {x : EuclideanSpace 𝕜 n} (hx : x ≠ 0) :
    ∃ i, ‖d i - mu‖ ≤ cV * cW * ‖toEuclideanLin A x - mu • x‖ / ‖x‖ := by
  have hAV : A * V = V * diagonal d := by
    rw [hA, Matrix.mul_assoc, hWV, Matrix.mul_one]
  obtain ⟨i, -, hi⟩ :=
    Finset.exists_min_image Finset.univ (fun i => ‖d i - mu‖) Finset.univ_nonempty
  refine ⟨i, ?_⟩
  refine bauerFike_residual (toEuclideanLin A) (toEuclideanLin V) (toEuclideanLin W)
    (toEuclideanLin (diagonal d)) (fun w => ?_) (fun w => ?_) (fun w => ?_) hcV (norm_nonneg _)
    hV hW (diag_lower _ d (toEuclideanLin_diagonal_apply d) mu (norm_nonneg _)
      (fun j => hi j (Finset.mem_univ j))) hx
  · rw [← toEuclideanLin_mul_apply W V, hWV, toEuclideanLin_one_apply]
  · rw [← toEuclideanLin_mul_apply V W, hVW, toEuclideanLin_one_apply]
  · rw [← toEuclideanLin_mul_apply A V, hAV, toEuclideanLin_mul_apply]

/-- **Theorem 4 (b)** for matrices: if `A V = V diag(μ) + R` and `W V = V W = 1`, every eigenvalue
`λ` of `A` lies within `‖W R‖` of some `μ_i`. `A` need not be diagonalizable. -/
theorem disc_inclusion_matrix [Nonempty n] (A V W R : Matrix n n 𝕜) (mu : n → 𝕜)
    (hWV : W * V = 1) (hVW : V * W = 1) (hR : A * V = V * diagonal mu + R) {f : ℝ}
    (hf : ∀ v, ‖toEuclideanLin (W * R) v‖ ≤ f * ‖v‖) {lam : 𝕜} {x : EuclideanSpace 𝕜 n}
    (hx : x ≠ 0) (hlam : toEuclideanLin A x = lam • x) : ∃ i, ‖lam - mu i‖ ≤ f := by
  obtain ⟨i, -, hi⟩ :=
    Finset.exists_min_image Finset.univ (fun i => ‖mu i - lam‖) Finset.univ_nonempty
  refine ⟨i, ?_⟩
  rw [norm_sub_rev]
  refine similarity_inclusion (toEuclideanLin A) (toEuclideanLin V) (toEuclideanLin W)
    (toEuclideanLin (diagonal mu)) (toEuclideanLin R) (fun w => ?_) (fun w => ?_) (fun w => ?_)
    (fun w => ?_) (diag_lower _ mu (toEuclideanLin_diagonal_apply mu) lam (norm_nonneg _)
      (fun j => hi j (Finset.mem_univ j))) hx hlam
  · rw [← toEuclideanLin_mul_apply W V, hWV, toEuclideanLin_one_apply]
  · rw [← toEuclideanLin_mul_apply V W, hVW, toEuclideanLin_one_apply]
  · rw [← toEuclideanLin_mul_apply A V, hR, map_add, LinearMap.add_apply,
      toEuclideanLin_mul_apply]
  · rw [← toEuclideanLin_mul_apply W R]
    exact hf w

/-- **Bauer-Fike**, the sharper bound of Theorem 2: if `A V = V diag(d)` and `W V = V W = 1`,
every eigenvalue `λ` of `A + E` lies within `‖W E V‖` of some `d_i`. -/
theorem bauerFike_matrix_sharp [Nonempty n] (A V W Em : Matrix n n 𝕜) (d : n → 𝕜)
    (hWV : W * V = 1) (hVW : V * W = 1) (hAV : A * V = V * diagonal d) {f : ℝ}
    (hf : ∀ v, ‖toEuclideanLin (W * (Em * V)) v‖ ≤ f * ‖v‖) {lam : 𝕜}
    {x : EuclideanSpace 𝕜 n} (hx : x ≠ 0) (hlam : toEuclideanLin (A + Em) x = lam • x) :
    ∃ i, ‖lam - d i‖ ≤ f :=
  disc_inclusion_matrix (A + Em) V W (Em * V) d hWV hVW
    (by rw [Matrix.add_mul, hAV]) hf hx hlam

/-- **Theorem 4 (b) with the bound of Lemma 3**, the statements N2 and N3 of the code together:
for any matrix `W` (a computed inverse) with `‖I - W V‖ ≤ θ < 1` and `‖W‖ ≤ c_W`, and any `R`
with `A V = V diag(μ) + R` and `‖R‖ ≤ r`, every eigenvalue `λ` of `A` lies within
`c_W / (1 - θ) * r` of some `μ_i`. -/
theorem disc_inclusion_certified [Nonempty n] (A V W R : Matrix n n 𝕜) (mu : n → 𝕜)
    (hR : A * V = V * diagonal mu + R) {θ cW r : ℝ} (hθ : θ < 1)
    (hG : ∀ z, ‖z - toEuclideanLin W (toEuclideanLin V z)‖ ≤ θ * ‖z‖)
    (hW : ∀ y, ‖toEuclideanLin W y‖ ≤ cW * ‖y‖) (hRr : ∀ v, ‖toEuclideanLin R v‖ ≤ r * ‖v‖)
    {lam : 𝕜} {x : EuclideanSpace 𝕜 n} (hx : x ≠ 0) (hlam : toEuclideanLin A x = lam • x) :
    ∃ i, ‖lam - mu i‖ ≤ cW / (1 - θ) * r := by
  obtain ⟨Vi, hVVi, hViV, hVi⟩ :=
    neumann_inverse (toEuclideanLin V) (toEuclideanLin W) hθ hG hW
  have hβ : 0 ≤ cW / (1 - θ) := by
    by_contra hc
    have hc' : cW / (1 - θ) < 0 := not_le.mp hc
    have hxn : 0 < ‖x‖ := norm_pos_iff.mpr hx
    have h1 := hVi x
    have h2 : cW / (1 - θ) * ‖x‖ < 0 := mul_neg_of_neg_of_pos hc' hxn
    linarith [norm_nonneg (Vi x)]
  obtain ⟨i, -, hi⟩ :=
    Finset.exists_min_image Finset.univ (fun i => ‖mu i - lam‖) Finset.univ_nonempty
  refine ⟨i, ?_⟩
  rw [norm_sub_rev]
  refine similarity_inclusion (toEuclideanLin A) (toEuclideanLin V) Vi
    (toEuclideanLin (diagonal mu)) (toEuclideanLin R) hViV hVVi (fun w => ?_) (fun w => ?_)
    (diag_lower _ mu (toEuclideanLin_diagonal_apply mu) lam (norm_nonneg _)
      (fun j => hi j (Finset.mem_univ j))) hx hlam
  · rw [← toEuclideanLin_mul_apply A V, hR, map_add, LinearMap.add_apply,
      toEuclideanLin_mul_apply]
  · calc ‖Vi (toEuclideanLin R w)‖ ≤ cW / (1 - θ) * ‖toEuclideanLin R w‖ := hVi _
      _ ≤ cW / (1 - θ) * (r * ‖w‖) := mul_le_mul_of_nonneg_left (hRr w) hβ
      _ = cW / (1 - θ) * r * ‖w‖ := by ring

end Euclid

end CertifyMathlib.NonHermitian
