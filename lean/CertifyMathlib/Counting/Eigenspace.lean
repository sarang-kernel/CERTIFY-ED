import CertifyMathlib.Counting.Inertia

/-!
# Davis-Kahan sin Θ, Frobenius form: the separation inequality (E-07, Lemma 1, core)

Source: `docs/proofs/E-07_davis_kahan.md`, Lemma 1, in the code repository. Its proof writes
`Y = (I - P) Q̂` and `A_⊥ Y - Y H' = (I - P) R'` in the eigenbases of `A` and `H'`, where every entry
is multiplied by an eigenvalue difference of modulus at least `δ'`, and sums the squares.

`frob_sylvester` is that step for general Hermitian `A` (`n × n`), `H` (`g × g`) and any `Y`
(`n × g`): if every nonzero coordinate `(Uᴴ Y V)_{ij}` of `Y` in the two eigenbases (`U`, `V` the
eigenvector unitaries) pairs eigenvalues at distance at least `δ ≥ 0`, then
`δ² ‖Y‖_F² ≤ ‖A Y - Y H‖_F²`. For `Y = (I - P) Q̂` the coordinates on the cluster eigenvectors are
zero, so only the separated pairs enter, which is the hypothesis of Lemma 1.

Not formalised here: the projector identities that turn `(I - P) R'` into `A Y - Y H'`, Lemma 2
(polar identities), Proposition 1 (`s_DK`), Propositions 2 and 3, and Theorem 6.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜]

/-- The squared Frobenius norm. -/
noncomputable def frob {m p : Type*} [Fintype m] [Fintype p] (X : Matrix m p 𝕜) : ℝ :=
  ∑ i, ∑ j, ‖X i j‖ ^ 2

theorem frob_eq_re_trace {m p : Type*} [Fintype m] [Fintype p] (X : Matrix m p 𝕜) :
    frob X = RCLike.re (trace (Xᴴ * X)) := by
  simp only [frob, trace, diag, mul_apply, conjTranspose_apply, map_sum]
  conv_lhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
  simp only [RCLike.star_def]
  rw [RCLike.conj_mul, ← RCLike.ofReal_pow, RCLike.ofReal_re]

theorem frob_conjTranspose_mul {m p q : Type*} [Fintype m] [Fintype p] [Fintype q]
    [DecidableEq m] (U : Matrix m q 𝕜) (hU : U * Uᴴ = 1) (X : Matrix m p 𝕜) :
    frob (Uᴴ * X) = frob X := by
  rw [frob_eq_re_trace, frob_eq_re_trace, conjTranspose_mul, conjTranspose_conjTranspose,
    Matrix.mul_assoc, ← Matrix.mul_assoc U, hU, Matrix.one_mul]

theorem frob_mul_conjTranspose {m p q : Type*} [Fintype m] [Fintype p] [Fintype q]
    [DecidableEq p] (V : Matrix p q 𝕜) (hV : V * Vᴴ = 1) (X : Matrix m p 𝕜) :
    frob (X * V) = frob X := by
  rw [frob_eq_re_trace, frob_eq_re_trace]
  congr 1
  conv_lhs => rw [conjTranspose_mul, Matrix.mul_assoc, trace_mul_comm, Matrix.mul_assoc,
    Matrix.mul_assoc, hV, Matrix.mul_one]

/-- **E-07, Lemma 1 (core).** The separation inequality of the Frobenius sin Θ theorem. -/
theorem frob_sylvester {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g] [DecidableEq g]
    {A : Matrix n n 𝕜} (hA : A.IsHermitian) {H : Matrix g g 𝕜} (hH : H.IsHermitian)
    (Y : Matrix n g 𝕜) (δ : ℝ) (hδ : 0 ≤ δ)
    (hsep : ∀ i j, ((hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ * Y *
      (hH.eigenvectorUnitary : Matrix g g 𝕜)) i j ≠ 0 →
        δ ≤ |hA.eigenvalues i - hH.eigenvalues j|) :
    δ ^ 2 * frob Y ≤ frob (A * Y - Y * H) := by
  have hAs : A = (hA.eigenvectorUnitary : Matrix n n 𝕜) * dg hA.eigenvalues *
      (hA.eigenvectorUnitary : Matrix n n 𝕜)ᴴ := by
    simpa using sub_smul_one_eq hA 0
  have hHs : H = (hH.eigenvectorUnitary : Matrix g g 𝕜) * dg hH.eigenvalues *
      (hH.eigenvectorUnitary : Matrix g g 𝕜)ᴴ := by
    simpa using sub_smul_one_eq hH 0
  have hU1 := unitary_mul_conjTranspose hA.eigenvectorUnitary
  have hU2 := conjTranspose_mul_unitary hA.eigenvectorUnitary
  have hV1 := unitary_mul_conjTranspose hH.eigenvectorUnitary
  have hV2 := conjTranspose_mul_unitary hH.eigenvectorUnitary
  generalize (hA.eigenvectorUnitary : Matrix n n 𝕜) = U at hAs hU1 hU2 hsep
  generalize (hH.eigenvectorUnitary : Matrix g g 𝕜) = V at hHs hV1 hV2 hsep
  have hUA : Uᴴ * A = dg hA.eigenvalues * Uᴴ := by
    conv_lhs => rw [hAs]
    simp only [← Matrix.mul_assoc]
    rw [hU2, Matrix.one_mul]
  have hHV : H * V = V * dg hH.eigenvalues := by
    conv_lhs => rw [hHs]
    simp only [Matrix.mul_assoc]
    rw [hV2, Matrix.mul_one]
  have hkey : Uᴴ * (A * Y - Y * H) * V =
      dg hA.eigenvalues * (Uᴴ * Y * V) - (Uᴴ * Y * V) * dg hH.eigenvalues := by
    calc Uᴴ * (A * Y - Y * H) * V = Uᴴ * A * Y * V - Uᴴ * Y * (H * V) := by
          rw [Matrix.mul_sub, Matrix.sub_mul]
          simp only [Matrix.mul_assoc]
      _ = dg hA.eigenvalues * Uᴴ * Y * V - Uᴴ * Y * (V * dg hH.eigenvalues) := by
          rw [hUA, hHV]
      _ = dg hA.eigenvalues * (Uᴴ * Y * V) - (Uᴴ * Y * V) * dg hH.eigenvalues := by
          simp only [Matrix.mul_assoc]
  have hinvZ : frob (A * Y - Y * H) = frob (Uᴴ * (A * Y - Y * H) * V) := by
    rw [frob_mul_conjTranspose V hV1, frob_conjTranspose_mul U hU1]
  have hinvY : frob Y = frob (Uᴴ * Y * V) := by
    rw [frob_mul_conjTranspose V hV1, frob_conjTranspose_mul U hU1]
  rw [hinvZ, hinvY, hkey]
  set X := Uᴴ * Y * V with hX
  unfold frob
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun j _ => ?_
  have hentry : (dg hA.eigenvalues * X - X * dg hH.eigenvalues : Matrix n g 𝕜) i j =
      ((hA.eigenvalues i - hH.eigenvalues j : ℝ) : 𝕜) * X i j := by
    simp only [Matrix.sub_apply, dg, diagonal_mul, mul_diagonal, RCLike.ofReal_sub]
    ring
  rw [hentry, norm_mul, mul_pow, RCLike.norm_ofReal, sq_abs]
  by_cases h0 : X i j = 0
  · simp [h0]
  · have h1 := hsep i j h0
    have h2 : δ ^ 2 ≤ |hA.eigenvalues i - hH.eigenvalues j| ^ 2 := pow_le_pow_left₀ hδ h1 2
    rw [sq_abs] at h2
    exact mul_le_mul_of_nonneg_right h2 (sq_nonneg _)

end CertifyMathlib.Counting
