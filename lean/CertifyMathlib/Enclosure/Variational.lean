import CertifyMathlib.Enclosure.Residual

/-!
# E-05: variational certificates (exact part)

`docs/proofs/E-05_variational.md`, section 1, Lemmas 1 and 2 and Proposition 1. For a trial vector
`ψ ≠ 0` with Rayleigh quotient `θ = ⟨ψ, Hψ⟩ / ⟨ψ, ψ⟩` and residual `r = ‖Hψ - θψ‖ / ‖ψ‖`:

* Lemma 1 (Rayleigh–Ritz): `λ_min ≤ θ ≤ λ_max` (`rayleigh_ritz`);
* Lemma 2: an eigenvalue lies within `r` of `θ` (Theorem 1 at `μ = θ`), and
  `r² = ‖Hψ‖² / ‖ψ‖² - θ²`, where `‖Hψ‖² = ⟨ψ, H²ψ⟩` for Hermitian `H` (`residual_sq_eq`);
* Proposition 1 (Temple): if `θ < ρ` and at most one eigenvalue (with multiplicity) is `< ρ`
  (that is, `ρ ≤ λ₂`), then `λ₁ ≥ θ - r² / (ρ - θ)` (`temple`).

* Lemma 5 (Rayleigh quotient of a perturbed vector): for `‖ψ - ψ̂‖ ≤ e < ‖ψ̂‖`,
  `|θ(ψ) - θ(ψ̂)| ≤ (2 r(ψ̂) ‖ψ̂‖ e + (‖H‖ + |θ(ψ̂)|) e²) / (‖ψ̂‖ - e)²` (`rayleigh_perturb`);
* the quotient step of Lemma 3: a quotient with numerator and positive denominator in intervals lies
  between the extreme corner quotients (`quotient_mem_corners`).

The floating-point parts of Lemma 3 are in `FloatAccounting.lean` (`rayleigh_numerator_error`);
Lemma 4 (MPS contraction) is not formalized.
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

namespace EigenData

variable {T : E →ₗ[𝕜] E} {ι : Type*} [Fintype ι] (D : EigenData T ι)

/-- **Lemma 1 (Rayleigh–Ritz).** For `ψ ≠ 0`, some eigenvalue is `≤ θ` and some is `≥ θ`. -/
theorem rayleigh_ritz (hT : T.IsSymmetric) {ψ : E} (hψ : ψ ≠ 0) {θ : ℝ}
    (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, T ψ⟫_𝕜) :
    (∃ i, D.eig i ≤ θ) ∧ (∃ i, θ ≤ D.eig i) := by
  constructor
  · by_contra h
    push Not at h
    have := D.lt_re_inner hT hψ (t := θ) (fun i _ => h i)
    linarith
  · by_contra h
    push Not at h
    have := D.re_inner_lt hT hψ (t := θ) (fun i _ => h i)
    linarith

/-- **Lemma 2, identity**: `‖Tψ - θψ‖² = ‖Tψ‖² - θ² ‖ψ‖²` at the Rayleigh quotient. -/
theorem residual_sq_eq (D' : EigenData T ι) (hT : T.IsSymmetric) {ψ : E} {θ : ℝ}
    (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, T ψ⟫_𝕜) :
    ‖T ψ - (θ : 𝕜) • ψ‖ ^ 2 = ‖T ψ‖ ^ 2 - θ ^ 2 * ‖ψ‖ ^ 2 := by
  have h0 := D'.norm_sub_smul_sq hT ψ 0
  simp only [RCLike.ofReal_zero, zero_smul, sub_zero] at h0
  rw [D'.norm_sub_smul_sq hT ψ θ, h0, D'.norm_sq_eq]
  rw [D'.re_inner_apply hT, D'.norm_sq_eq] at hθ
  have e : ∀ i, (D'.eig i - θ) ^ 2 * ‖D'.basis.repr ψ i‖ ^ 2 =
      D'.eig i ^ 2 * ‖D'.basis.repr ψ i‖ ^ 2 - 2 * θ * (D'.eig i * ‖D'.basis.repr ψ i‖ ^ 2) +
        θ ^ 2 * ‖D'.basis.repr ψ i‖ ^ 2 := fun i => by ring
  simp only [e, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [← hθ]
  ring

/-- `‖Tψ‖² = re ⟨ψ, T²ψ⟩` for symmetric `T`. -/
theorem norm_apply_sq_eq (hT : T.IsSymmetric) (ψ : E) :
    ‖T ψ‖ ^ 2 = RCLike.re ⟪ψ, T (T ψ)⟫_𝕜 := by
  rw [← hT ψ (T ψ), inner_self_eq_norm_sq]

/-- The weighted Kato–Temple identity for a vector of any norm. -/
theorem temple_identity (hT : T.IsSymmetric) {ψ : E} {θ : ℝ}
    (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, T ψ⟫_𝕜) (lam a : ℝ) :
    ∑ i, ‖D.basis.repr ψ i‖ ^ 2 * ((D.eig i - lam) * (D.eig i - a)) =
      ‖T ψ - (θ : 𝕜) • ψ‖ ^ 2 + (θ - lam) * (θ - a) * ‖ψ‖ ^ 2 := by
  rw [D.norm_sub_smul_sq hT ψ θ]
  rw [D.re_inner_apply hT] at hθ
  have h0 := D.norm_sq_eq ψ
  have e : ∀ i, ‖D.basis.repr ψ i‖ ^ 2 * ((D.eig i - lam) * (D.eig i - a)) =
      (D.eig i - θ) ^ 2 * ‖D.basis.repr ψ i‖ ^ 2
        + (2 * θ - lam - a) * (D.eig i * ‖D.basis.repr ψ i‖ ^ 2)
        - ((2 * θ - lam - a) * θ - (θ - lam) * (θ - a)) * ‖D.basis.repr ψ i‖ ^ 2 := by
    intro i
    ring
  simp only [e, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [← hθ, ← h0]
  ring

/-- **Proposition 1 (Temple).** Let `lam1` be the smallest eigenvalue and `θ < ρ` with every
eigenvalue equal to `lam1` or `≥ ρ`. Then `θ - r² / (ρ - θ) ≤ lam1`, `r = ‖Tψ - θψ‖ / ‖ψ‖`. -/
theorem temple_of_sep (hT : T.IsSymmetric) {ψ : E} (hψ : ψ ≠ 0) {θ ρ lam1 : ℝ}
    (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, T ψ⟫_𝕜) (hθρ : θ < ρ) (hl : lam1 ≤ ρ)
    (hsep : ∀ i, D.eig i = lam1 ∨ ρ ≤ D.eig i) :
    θ - (‖T ψ - (θ : 𝕜) • ψ‖ / ‖ψ‖) ^ 2 / (ρ - θ) ≤ lam1 := by
  have hpos : 0 < ‖ψ‖ := norm_pos_iff.mpr hψ
  have hsum : 0 ≤ ∑ i, ‖D.basis.repr ψ i‖ ^ 2 * ((D.eig i - lam1) * (D.eig i - ρ)) := by
    refine Finset.sum_nonneg fun i _ => mul_nonneg (sq_nonneg _) ?_
    rcases hsep i with h | h
    · rw [h, sub_self, zero_mul]
    · exact mul_nonneg (by linarith) (by linarith)
  rw [D.temple_identity hT hθ lam1 ρ] at hsum
  have hr : (‖T ψ - (θ : 𝕜) • ψ‖ / ‖ψ‖) ^ 2 * ‖ψ‖ ^ 2 = ‖T ψ - (θ : 𝕜) • ψ‖ ^ 2 := by
    have hne : ‖ψ‖ ≠ 0 := hpos.ne'
    rw [div_pow]
    field_simp
  have hk : (θ - lam1) * (ρ - θ) ≤ (‖T ψ - (θ : 𝕜) • ψ‖ / ‖ψ‖) ^ 2 := by
    have h2 : (θ - lam1) * (ρ - θ) * ‖ψ‖ ^ 2 ≤ (‖T ψ - (θ : 𝕜) • ψ‖ / ‖ψ‖) ^ 2 * ‖ψ‖ ^ 2 := by
      rw [hr]
      nlinarith
    exact le_of_mul_le_mul_right h2 (pow_pos hpos 2)
  have hd : 0 < ρ - θ := sub_pos.mpr hθρ
  have : θ - lam1 ≤ (‖T ψ - (θ : 𝕜) • ψ‖ / ‖ψ‖) ^ 2 / (ρ - θ) := by
    rw [le_div_iff₀ hd]
    exact hk
  linarith

/-- **Proposition 1 (Temple)**, with `ρ ≤ λ₂` written as "at most one eigenvalue, counted with
multiplicity, lies below `ρ`", and `λ₁ = D.eig i` the smallest eigenvalue. -/
theorem temple [DecidableEq ι] (hT : T.IsSymmetric) {ψ : E} (hψ : ψ ≠ 0) {θ ρ : ℝ}
    (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, T ψ⟫_𝕜) (hθρ : θ < ρ)
    (hρ : (univ.filter fun j => D.eig j < ρ).card ≤ 1) {i : ι} (hmin : ∀ j, D.eig i ≤ D.eig j) :
    θ - (‖T ψ - (θ : 𝕜) • ψ‖ / ‖ψ‖) ^ 2 / (ρ - θ) ≤ D.eig i := by
  have hpos : 0 < ‖ψ‖ := norm_pos_iff.mpr hψ
  have hiθ : D.eig i ≤ θ := by
    have h := D.le_re_inner hT (v := ψ) (t := D.eig i) (fun j _ => hmin j)
    rw [← hθ] at h
    exact le_of_mul_le_mul_right h (pow_pos hpos 2)
  have hiρ : D.eig i < ρ := lt_of_le_of_lt hiθ hθρ
  refine D.temple_of_sep hT hψ hθ hθρ hiρ.le (fun j => ?_)
  by_cases hj : ρ ≤ D.eig j
  · exact Or.inr hj
  · left
    push Not at hj
    have hij : i = j := Finset.card_le_one.mp hρ i (by simp [hiρ]) j (by simp [hj])
    rw [hij]

end EigenData

/-! ### Matrices -/

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **E-05, Lemma 1** for a Hermitian matrix. -/
theorem rayleigh_ritz_matrix {H : Matrix n n 𝕜} (hH : H.IsHermitian) {ψ : EuclideanSpace 𝕜 n}
    (hψ : ψ ≠ 0) {θ : ℝ} (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, toEuclideanLin H ψ⟫_𝕜) :
    (∃ i, hH.eigenvalues i ≤ θ) ∧ (∃ i, θ ≤ hH.eigenvalues i) :=
  (hermitianEigenData hH).rayleigh_ritz (isSymmetric_of_isHermitian hH) hψ hθ

/-- **E-05, Proposition 1 (Temple)** for a Hermitian matrix. -/
theorem temple_matrix {H : Matrix n n 𝕜} (hH : H.IsHermitian) {ψ : EuclideanSpace 𝕜 n}
    (hψ : ψ ≠ 0) {θ ρ : ℝ} (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, toEuclideanLin H ψ⟫_𝕜)
    (hθρ : θ < ρ) (hρ : (univ.filter fun j => hH.eigenvalues j < ρ).card ≤ 1) {i : n}
    (hmin : ∀ j, hH.eigenvalues i ≤ hH.eigenvalues j) :
    θ - (‖toEuclideanLin H ψ - (θ : 𝕜) • ψ‖ / ‖ψ‖) ^ 2 / (ρ - θ) ≤ hH.eigenvalues i :=
  (hermitianEigenData hH).temple (isSymmetric_of_isHermitian hH) hψ hθ hθρ hρ hmin

/-! ### Lemma 5: the Rayleigh quotient of a perturbed vector -/

section RayleighPerturb

variable {T : E →ₗ[𝕜] E}

/-- **E-05, Lemma 5.** If `‖ψ - ψ̂‖ ≤ e < ‖ψ̂‖` and `‖T z‖ ≤ n_H ‖z‖`, the Rayleigh quotients satisfy
`|θ(ψ) - θ(ψ̂)| ≤ (2 ‖T ψ̂ - θ(ψ̂) ψ̂‖ e + (n_H + |θ(ψ̂)|) e²) / (‖ψ̂‖ - e)²`; the first term is
`2 r(ψ̂) ‖ψ̂‖ e` of the document. -/
theorem rayleigh_perturb (hT : T.IsSymmetric) {nH : ℝ} (hnH0 : 0 ≤ nH)
    (hnH : ∀ z, ‖T z‖ ≤ nH * ‖z‖) {ψ ψh : E} {θ θh e : ℝ}
    (hθ : θ * ‖ψ‖ ^ 2 = RCLike.re ⟪ψ, T ψ⟫_𝕜) (hθh : θh * ‖ψh‖ ^ 2 = RCLike.re ⟪ψh, T ψh⟫_𝕜)
    (he : ‖ψ - ψh‖ ≤ e) (hlt : e < ‖ψh‖) :
    |θ - θh| ≤ (2 * ‖T ψh - (θh : 𝕜) • ψh‖ * e + (nH + |θh|) * e ^ 2) / (‖ψh‖ - e) ^ 2 := by
  obtain ⟨d, rfl⟩ : ∃ d, ψ = ψh + d := ⟨ψ - ψh, by abel⟩
  have hd : ‖d‖ ≤ e := by simpa using he
  have he0 : 0 ≤ e := (norm_nonneg _).trans hd
  have hpos : 0 < ‖ψh‖ - e := by linarith
  have hψn : ‖ψh‖ - e ≤ ‖ψh + d‖ := by
    have := norm_sub_le (ψh + d) d
    rw [add_sub_cancel_right] at this
    linarith
  -- the identity `(θ - θ̂) ‖ψ‖² = 2 re⟨d, (T - θ̂) ψ̂⟩ + re⟨d, (T - θ̂) d⟩`
  have hsym : RCLike.re ⟪ψh, T d⟫_𝕜 = RCLike.re ⟪d, T ψh⟫_𝕜 := by
    rw [← hT ψh d, inner_re_symm]
  have e1 : RCLike.re ⟪ψh + d, T (ψh + d)⟫_𝕜 = RCLike.re ⟪ψh, T ψh⟫_𝕜 +
      2 * RCLike.re ⟪d, T ψh⟫_𝕜 + RCLike.re ⟪d, T d⟫_𝕜 := by
    simp only [map_add, inner_add_left, inner_add_right]
    rw [hsym]
    ring
  have e2 : ‖ψh + d‖ ^ 2 = ‖ψh‖ ^ 2 + 2 * RCLike.re ⟪d, ψh⟫_𝕜 + ‖d‖ ^ 2 := by
    rw [@norm_add_sq 𝕜, inner_re_symm (𝕜 := 𝕜)]
  have e3 : RCLike.re ⟪d, T ψh - (θh : 𝕜) • ψh⟫_𝕜 = RCLike.re ⟪d, T ψh⟫_𝕜 - θh * RCLike.re ⟪d, ψh⟫_𝕜 := by
    rw [inner_sub_right, inner_smul_right, map_sub, RCLike.re_ofReal_mul]
  have e4 : RCLike.re ⟪d, T d - (θh : 𝕜) • d⟫_𝕜 = RCLike.re ⟪d, T d⟫_𝕜 - θh * ‖d‖ ^ 2 := by
    rw [inner_sub_right, inner_smul_right, map_sub, RCLike.re_ofReal_mul, inner_self_eq_norm_sq]
  have key : (θ - θh) * ‖ψh + d‖ ^ 2 = 2 * RCLike.re ⟪d, T ψh - (θh : 𝕜) • ψh⟫_𝕜 +
      RCLike.re ⟪d, T d - (θh : 𝕜) • d⟫_𝕜 := by
    rw [e3, e4]
    have : (θ - θh) * ‖ψh + d‖ ^ 2 = θ * ‖ψh + d‖ ^ 2 - θh * ‖ψh + d‖ ^ 2 := by ring
    rw [this, hθ, e1, e2]
    linarith
  -- the two terms
  have b1 : |2 * RCLike.re ⟪d, T ψh - (θh : 𝕜) • ψh⟫_𝕜| ≤ 2 * ‖T ψh - (θh : 𝕜) • ψh‖ * e := by
    rw [abs_mul, abs_two]
    have := (RCLike.abs_re_le_norm ⟪d, T ψh - (θh : 𝕜) • ψh⟫_𝕜).trans
      (norm_inner_le_norm d (T ψh - (θh : 𝕜) • ψh))
    nlinarith [norm_nonneg (T ψh - (θh : 𝕜) • ψh)]
  have hTd : ‖T d - (θh : 𝕜) • d‖ ≤ (nH + |θh|) * ‖d‖ := by
    refine (norm_sub_le _ _).trans ?_
    rw [norm_smul, RCLike.norm_ofReal]
    linarith [hnH d]
  have b2 : |RCLike.re ⟪d, T d - (θh : 𝕜) • d⟫_𝕜| ≤ (nH + |θh|) * e ^ 2 := by
    have h1 := (RCLike.abs_re_le_norm ⟪d, T d - (θh : 𝕜) • d⟫_𝕜).trans
      (norm_inner_le_norm d (T d - (θh : 𝕜) • d))
    have h2 : ‖d‖ * ‖T d - (θh : 𝕜) • d‖ ≤ ‖d‖ * ((nH + |θh|) * ‖d‖) :=
      mul_le_mul_of_nonneg_left hTd (norm_nonneg _)
    have hc : 0 ≤ nH + |θh| := by positivity
    have h3 : ‖d‖ ^ 2 ≤ e ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hd 2
    nlinarith [mul_le_mul_of_nonneg_left h3 hc]
  have hX : |θ - θh| * ‖ψh + d‖ ^ 2 ≤ 2 * ‖T ψh - (θh : 𝕜) • ψh‖ * e + (nH + |θh|) * e ^ 2 := by
    have : |θ - θh| * ‖ψh + d‖ ^ 2 = |(θ - θh) * ‖ψh + d‖ ^ 2| := by
      rw [abs_mul, abs_of_nonneg (sq_nonneg ‖ψh + d‖)]
    rw [this, key]
    exact (abs_add_le _ _).trans (add_le_add b1 b2)
  rw [le_div_iff₀ (by positivity)]
  have hsq : (‖ψh‖ - e) ^ 2 ≤ ‖ψh + d‖ ^ 2 := pow_le_pow_left₀ hpos.le hψn 2
  exact (mul_le_mul_of_nonneg_left hsq (abs_nonneg _)).trans hX

end RayleighPerturb

/-- **E-05, Lemma 3 (iii)** (the quotient step): with the numerator in `[N_lo, N_hi]` and the
denominator in `[D_lo, D_hi] ⊂ (0, ∞)`, the quotient lies between the smallest and the largest of the
four corner quotients. -/
theorem quotient_mem_corners {N D Nl Nh Dl Dh : ℝ} (hNl : Nl ≤ N) (hNh : N ≤ Nh) (hDl : Dl ≤ D)
    (hDh : D ≤ Dh) (hD0 : 0 < Dl) :
    min (min (Nl / Dl) (Nl / Dh)) (min (Nh / Dl) (Nh / Dh)) ≤ N / D ∧
      N / D ≤ max (max (Nl / Dl) (Nl / Dh)) (max (Nh / Dl) (Nh / Dh)) := by
  have hD : 0 < D := lt_of_lt_of_le hD0 hDl
  have hDh0 : 0 < Dh := lt_of_lt_of_le hD hDh
  have h1 : Nl / D ≤ N / D := div_le_div_of_nonneg_right hNl hD.le
  have h2 : N / D ≤ Nh / D := div_le_div_of_nonneg_right hNh hD.le
  constructor
  · refine le_trans ?_ h1
    refine min_le_of_left_le ?_
    rcases le_total 0 Nl with h | h
    · exact min_le_of_right_le (div_le_div_of_nonneg_left h hD hDh)
    · refine min_le_of_left_le ?_
      rw [div_le_div_iff₀ hD0 hD]
      nlinarith
  · refine le_trans h2 ?_
    refine le_max_of_le_right ?_
    rcases le_total 0 Nh with h | h
    · exact le_max_of_le_left (div_le_div_of_nonneg_left h hD0 hDl)
    · refine le_max_of_le_right ?_
      rw [div_le_div_iff₀ hD hDh0]
      nlinarith

end CertifyMathlib.Enclosure
