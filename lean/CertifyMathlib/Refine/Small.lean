import CertifyMathlib.Refine.Residual
import CertifyCore.Kernel.Small

/-!
# Theorem 1 for the three kernel-checked certificates (N-05)

`CertifyCore/Kernel/Small.lean` evaluates `checkLowest` by the kernel on the literal data of the conformance
fixtures `matrix-free`, `dense-arrays` and `npz` (dimension 64). With the refinement theorems of
`CertifyMathlib.Refine.Residual` and two further kernel evaluations, exact symmetry of the decoded matrix
(`symOK`) and vectors of the right length that are nonzero (`vecOK`), the conclusion of Theorem 1 holds for
those data: every claimed pair has an eigenvalue of the decoded matrix within its recorded existence radius, and,
for any symmetric `H` within the recorded model shift of the decoded matrix, an eigenvalue of `H` within its
exported radius, which meets the accuracy target.

What is not covered: that the literal data are the stored numbers of the certificate (they were written by
`cecheck kdata` after its hash comparisons, a level N step), and S1, the hypothesis `‖H - A‖₂ ≤ shift`.
-/

namespace CertifyMathlib.Refine

open CertifyCore.K

set_option maxRecDepth 100000 in
theorem matrixFree_symOK : symOK Data.matrixFree.A = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem matrixFree_vecOK : vecOK Data.matrixFree = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem denseArrays_symOK : symOK Data.denseArrays.A = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem denseArrays_vecOK : vecOK Data.denseArrays = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem npz_symOK : symOK Data.npz.A = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem npz_vecOK : vecOK Data.npz = true := by decide +kernel

/-- Theorem 1 for `matrix-free`: each claimed pair has an eigenvalue of the decoded matrix within its existence
radius, and the existence radius is at most the radius. -/
theorem matrixFree_theorem1 : ∀ p ∈ Data.matrixFree.pairs,
    ∃ i, |(isHermitian_of_symOK matrixFree_symOK).eigenvalues i - (p.mu : ℝ)| ≤ (p.ex : ℝ) ∧ (p.ex : ℝ) ≤ p.rad :=
  checkLowest_theorem1 _ matrixFree_symOK matrixFree_vecOK matrixFree_S2_S3

/-- Theorem 1 for `dense-arrays`. -/
theorem denseArrays_theorem1 : ∀ p ∈ Data.denseArrays.pairs,
    ∃ i, |(isHermitian_of_symOK denseArrays_symOK).eigenvalues i - (p.mu : ℝ)| ≤ (p.ex : ℝ) ∧
      (p.ex : ℝ) ≤ p.rad :=
  checkLowest_theorem1 _ denseArrays_symOK denseArrays_vecOK denseArrays_S2_S3

/-- Theorem 1 for `npz`. -/
theorem npz_theorem1 : ∀ p ∈ Data.npz.pairs,
    ∃ i, |(isHermitian_of_symOK npz_symOK).eigenvalues i - (p.mu : ℝ)| ≤ (p.ex : ℝ) ∧ (p.ex : ℝ) ≤ p.rad :=
  checkLowest_theorem1 _ npz_symOK npz_vecOK npz_S2_S3

open scoped Matrix.Norms.L2Operator in
/-- The S2 conclusion for `matrix-free`, for every symmetric `H` within the recorded model shift. -/
theorem matrixFree_S2 {H : Matrix (Fin Data.matrixFree.A.length) (Fin Data.matrixFree.A.length) ℝ}
    (hH : H.IsHermitian) (hσ : ‖H - toMatrix Data.matrixFree.A‖ ≤ (Data.matrixFree.shift : ℝ)) :
    ∀ p ∈ Data.matrixFree.pairs, ∃ i, |hH.eigenvalues i - (p.mu : ℝ)| ≤ (p.radArr : ℝ) ∧
      (p.radArr : ℝ) ≤ Data.matrixFree.target :=
  checkLowest_S2 _ matrixFree_vecOK matrixFree_S2_S3 hH hσ

open scoped Matrix.Norms.L2Operator in
/-- The S2 conclusion for `dense-arrays`. -/
theorem denseArrays_S2 {H : Matrix (Fin Data.denseArrays.A.length) (Fin Data.denseArrays.A.length) ℝ}
    (hH : H.IsHermitian) (hσ : ‖H - toMatrix Data.denseArrays.A‖ ≤ (Data.denseArrays.shift : ℝ)) :
    ∀ p ∈ Data.denseArrays.pairs, ∃ i, |hH.eigenvalues i - (p.mu : ℝ)| ≤ (p.radArr : ℝ) ∧
      (p.radArr : ℝ) ≤ Data.denseArrays.target :=
  checkLowest_S2 _ denseArrays_vecOK denseArrays_S2_S3 hH hσ

open scoped Matrix.Norms.L2Operator in
/-- The S2 conclusion for `npz`. -/
theorem npz_S2 {H : Matrix (Fin Data.npz.A.length) (Fin Data.npz.A.length) ℝ}
    (hH : H.IsHermitian) (hσ : ‖H - toMatrix Data.npz.A‖ ≤ (Data.npz.shift : ℝ)) :
    ∀ p ∈ Data.npz.pairs, ∃ i, |hH.eigenvalues i - (p.mu : ℝ)| ≤ (p.radArr : ℝ) ∧
      (p.radArr : ℝ) ≤ Data.npz.target :=
  checkLowest_S2 _ npz_vecOK npz_S2_S3 hH hσ

/-- Negative controls for the two extra checks: an asymmetric matrix and a zero vector are rejected. -/
theorem symOK_rejects_asymmetric : symOK [[(1, (1 : Rat))], []] = false := by decide +kernel

/-- A pair whose vector is zero, for the negative control of `vecOK`. -/
def zeroPair (n : Nat) : Pair := ⟨0, List.replicate n 0, 0, 0, 0, 0, 0, 0⟩

theorem vecOK_rejects_zero_vector :
    vecOK { Data.npz with pairs := [zeroPair Data.npz.A.length] } = false := by decide +kernel

end CertifyMathlib.Refine
