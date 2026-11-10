import CertifyMathlib.Enclosure.Spectral

/-!
# Theorem 3 (Kato–Temple)

`docs/theory.md` §2: with `‖x‖ = 1`, `θ = xᴴ A x`, `r = ‖A x - θ x‖` and an interval `(α, β) ∋ θ`
containing exactly one eigenvalue `λ`,
`θ - r² / (β - θ) ≤ λ ≤ θ + r² / (θ - α)`.

The proof is the classical one: with weights `w_i = |⟨b_i, x⟩|²` (they sum to `1`),
`∑ w_i (λ_i - λ)(λ_i - a) = r² + (θ - λ)(θ - a)` for every real `a`, and for `a = α` and `a = β`
every term on the left is `≥ 0` because no eigenvalue other than `λ` lies in `(α, β)`.

The Lean statements are slightly more general than the document: `λ` may be any real number in
`[α, β]` such that every eigenvalue equals `λ` or lies outside `(α, β)` (so `λ` may have any
multiplicity), and each bound uses only the hypothesis it needs (`α < θ` for the upper bound,
`θ < β` for the lower bound). The division-free inequalities
`(λ - θ)(θ - α) ≤ r²` and `(θ - λ)(β - θ) ≤ r²` hold without any hypothesis on `θ`.
-/

open scoped InnerProductSpace ComplexConjugate
open Module Finset

namespace CertifyMathlib.Enclosure

variable {𝕜 : Type*} [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]

namespace EigenData

variable {T : E →ₗ[𝕜] E} {ι : Type*} [Fintype ι] (D : EigenData T ι)

/-- The Kato–Temple identity. -/
theorem kato_temple_identity (hT : T.IsSymmetric) {x : E} (hx : ‖x‖ = 1) {θ : ℝ}
    (hθ : θ = RCLike.re ⟪x, T x⟫_𝕜) (lam a : ℝ) :
    ∑ i, ‖D.basis.repr x i‖ ^ 2 * ((D.eig i - lam) * (D.eig i - a)) =
      ‖T x - (θ : 𝕜) • x‖ ^ 2 + (θ - lam) * (θ - a) := by
  have h0 : ∑ i, ‖D.basis.repr x i‖ ^ 2 = 1 := by rw [← D.norm_sq_eq, hx, one_pow]
  have h1 : ∑ i, D.eig i * ‖D.basis.repr x i‖ ^ 2 = θ := by rw [hθ, D.re_inner_apply hT]
  rw [D.norm_sub_smul_sq hT x θ]
  have e : ∀ i, ‖D.basis.repr x i‖ ^ 2 * ((D.eig i - lam) * (D.eig i - a)) =
      (D.eig i - θ) ^ 2 * ‖D.basis.repr x i‖ ^ 2
        + (2 * θ - lam - a) * (D.eig i * ‖D.basis.repr x i‖ ^ 2)
        - ((2 * θ - lam - a) * θ - (θ - lam) * (θ - a)) * ‖D.basis.repr x i‖ ^ 2 := by
    intro i
    ring
  simp only [e, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, h0, h1]
  ring

/-- Nonnegativity of the left side when no eigenvalue other than `lam` lies in `(α, β)`. -/
theorem kato_temple_sum_nonneg {x : E} {α lam β a : ℝ} (hαl : α ≤ lam) (hlβ : lam ≤ β)
    (ha : a = α ∨ a = β) (hsep : ∀ i, D.eig i = lam ∨ D.eig i ≤ α ∨ β ≤ D.eig i) :
    0 ≤ ∑ i, ‖D.basis.repr x i‖ ^ 2 * ((D.eig i - lam) * (D.eig i - a)) := by
  refine Finset.sum_nonneg fun i _ => mul_nonneg (sq_nonneg _) ?_
  rcases hsep i with h | h | h
  · rw [h, sub_self, zero_mul]
  · rcases ha with rfl | rfl
    · nlinarith
    · nlinarith
  · rcases ha with rfl | rfl
    · nlinarith
    · nlinarith

/-- **Kato–Temple, division-free upper form**: `(λ - θ)(θ - α) ≤ r²`. -/
theorem kato_temple_upper_mul (hT : T.IsSymmetric) {x : E} (hx : ‖x‖ = 1) {θ α lam β : ℝ}
    (hθ : θ = RCLike.re ⟪x, T x⟫_𝕜) (hαl : α ≤ lam) (hlβ : lam ≤ β)
    (hsep : ∀ i, D.eig i = lam ∨ D.eig i ≤ α ∨ β ≤ D.eig i) :
    (lam - θ) * (θ - α) ≤ ‖T x - (θ : 𝕜) • x‖ ^ 2 := by
  have h := D.kato_temple_sum_nonneg (x := x) hαl hlβ (a := α) (Or.inl rfl) hsep
  rw [D.kato_temple_identity hT hx hθ lam α] at h
  nlinarith

/-- **Kato–Temple, division-free lower form**: `(θ - λ)(β - θ) ≤ r²`. -/
theorem kato_temple_lower_mul (hT : T.IsSymmetric) {x : E} (hx : ‖x‖ = 1) {θ α lam β : ℝ}
    (hθ : θ = RCLike.re ⟪x, T x⟫_𝕜) (hαl : α ≤ lam) (hlβ : lam ≤ β)
    (hsep : ∀ i, D.eig i = lam ∨ D.eig i ≤ α ∨ β ≤ D.eig i) :
    (θ - lam) * (β - θ) ≤ ‖T x - (θ : 𝕜) • x‖ ^ 2 := by
  have h := D.kato_temple_sum_nonneg (x := x) hαl hlβ (a := β) (Or.inr rfl) hsep
  rw [D.kato_temple_identity hT hx hθ lam β] at h
  nlinarith

/-- **Theorem 3, upper bound**: `λ ≤ θ + r² / (θ - α)` when `α < θ`. -/
theorem kato_temple_upper (hT : T.IsSymmetric) {x : E} (hx : ‖x‖ = 1) {θ α lam β : ℝ}
    (hθ : θ = RCLike.re ⟪x, T x⟫_𝕜) (hαl : α ≤ lam) (hlβ : lam ≤ β)
    (hsep : ∀ i, D.eig i = lam ∨ D.eig i ≤ α ∨ β ≤ D.eig i) (hαθ : α < θ) :
    lam ≤ θ + ‖T x - (θ : 𝕜) • x‖ ^ 2 / (θ - α) := by
  have h := D.kato_temple_upper_mul hT hx hθ hαl hlβ hsep
  have hpos : 0 < θ - α := sub_pos.mpr hαθ
  have : lam - θ ≤ ‖T x - (θ : 𝕜) • x‖ ^ 2 / (θ - α) := by
    rw [le_div_iff₀ hpos]
    exact h
  linarith

/-- **Theorem 3, lower bound**: `θ - r² / (β - θ) ≤ λ` when `θ < β`. -/
theorem kato_temple_lower (hT : T.IsSymmetric) {x : E} (hx : ‖x‖ = 1) {θ α lam β : ℝ}
    (hθ : θ = RCLike.re ⟪x, T x⟫_𝕜) (hαl : α ≤ lam) (hlβ : lam ≤ β)
    (hsep : ∀ i, D.eig i = lam ∨ D.eig i ≤ α ∨ β ≤ D.eig i) (hθβ : θ < β) :
    θ - ‖T x - (θ : 𝕜) • x‖ ^ 2 / (β - θ) ≤ lam := by
  have h := D.kato_temple_lower_mul hT hx hθ hαl hlβ hsep
  have hpos : 0 < β - θ := sub_pos.mpr hθβ
  have : θ - lam ≤ ‖T x - (θ : 𝕜) • x‖ ^ 2 / (β - θ) := by
    rw [le_div_iff₀ hpos]
    exact h
  linarith

end EigenData

/-! ### Matrices -/

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- For a Hermitian matrix the quadratic form is real: `xᴴ A x = re (xᴴ A x)`. -/
theorem inner_apply_self_eq_re {A : Matrix n n 𝕜} (hA : A.IsHermitian) (x : EuclideanSpace 𝕜 n) :
    ((RCLike.re ⟪x, toEuclideanLin A x⟫_𝕜 : ℝ) : 𝕜) = ⟪x, toEuclideanLin A x⟫_𝕜 := by
  have h := (isSymmetric_of_isHermitian hA).coe_re_inner_apply_self x
  rw [(isSymmetric_of_isHermitian hA) x x] at h
  exact h

/-- **Theorem 3 (Kato–Temple)** for a Hermitian matrix, in the form of the document:
`‖x‖ = 1`, `θ = xᴴ A x`, `r = ‖A x - θ x‖`, `θ ∈ (α, β)`, and `λ` the only eigenvalue in `(α, β)`
(any multiplicity). Then `θ - r² / (β - θ) ≤ λ ≤ θ + r² / (θ - α)`. -/
theorem kato_temple {A : Matrix n n 𝕜} (hA : A.IsHermitian) {x : EuclideanSpace 𝕜 n}
    (hx : ‖x‖ = 1) {θ α lam β : ℝ} (hθ : θ = RCLike.re ⟪x, toEuclideanLin A x⟫_𝕜)
    (hθI : θ ∈ Set.Ioo α β) (hlam : lam ∈ Set.Ioo α β)
    (hone : ∀ i, hA.eigenvalues i ∈ Set.Ioo α β → hA.eigenvalues i = lam) :
    θ - ‖toEuclideanLin A x - (θ : 𝕜) • x‖ ^ 2 / (β - θ) ≤ lam ∧
      lam ≤ θ + ‖toEuclideanLin A x - (θ : 𝕜) • x‖ ^ 2 / (θ - α) := by
  have hsep : ∀ i, (hermitianEigenData hA).eig i = lam ∨ (hermitianEigenData hA).eig i ≤ α ∨
      β ≤ (hermitianEigenData hA).eig i := by
    intro i
    rw [hermitianEigenData_eig]
    by_cases hi : hA.eigenvalues i ∈ Set.Ioo α β
    · exact Or.inl (hone i hi)
    · right
      simp only [Set.mem_Ioo, not_and_or, not_lt] at hi
      exact hi
  exact ⟨(hermitianEigenData hA).kato_temple_lower (isSymmetric_of_isHermitian hA) hx hθ
      hlam.1.le hlam.2.le hsep hθI.2,
    (hermitianEigenData hA).kato_temple_upper (isSymmetric_of_isHermitian hA) hx hθ
      hlam.1.le hlam.2.le hsep hθI.1⟩

end CertifyMathlib.Enclosure
