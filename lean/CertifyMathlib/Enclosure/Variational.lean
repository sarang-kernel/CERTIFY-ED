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

Lemmas 3 to 5 of the document (floating-point enclosure of `θ` and `r`, MPS contraction, Rayleigh
quotient of the source state) are statements about binary64 arithmetic and are not formalized here.
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

end CertifyMathlib.Enclosure
