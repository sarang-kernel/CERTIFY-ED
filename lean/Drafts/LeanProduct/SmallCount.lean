import CertifyMathlib.Refine.Inertia
import CertifyMathlib.Refine.Small

/-!
# The exact S3 count for the three kernel-checked certificates (N-05)

For the literal data of `matrix-free`, `dense-arrays` and `npz` the kernel evaluates the list elimination
`gaussQ` of the decoded `A - σ I` at the recorded top shift `σ` and counts its negative pivots;
`count_of_gaussQ` turns that evaluation into a statement about the eigenvalues of the decoded matrix: exactly
`k` of them (the number of claimed pairs) lie below `σ`. This is the count that `checkLowest` reads off the
fraction-free elimination (`negAtSigma`), now with Sylvester's law proved for the decoded data.

Status (11 November 2026): draft, not built. `lake build` of this file as `CertifyMathlib.Refine.SmallCount` had
not finished after 67 minutes, inside the kernel evaluations of the `gaussQ` theorems (rational elimination of a
dense 64 by 64 matrix), and was abandoned; nanoda re-checks the integer evaluations of `checkLowest` on the same
data in 127 to 149 seconds each (CORE.md section 6).
The theorems below it are kernel-checked only once the evaluations finish. A faster route is to refine
`CertifyCore.K.bareiss` itself (see LEAN_PRODUCT.md section 4).
-/

namespace CertifyMathlib.Refine

open CertifyCore.K CertifyMathlib.Counting

set_option maxRecDepth 100000 in
theorem matrixFree_gauss : (gaussQ Data.matrixFree.A.length (denseRows Data.matrixFree.A Data.matrixFree.sigma)).map
    (fun ps => ps.countP (fun x => decide (x < 0))) = some Data.matrixFree.pairs.length := by decide +kernel

set_option maxRecDepth 100000 in
theorem denseArrays_gauss :
    (gaussQ Data.denseArrays.A.length (denseRows Data.denseArrays.A Data.denseArrays.sigma)).map
      (fun ps => ps.countP (fun x => decide (x < 0))) = some Data.denseArrays.pairs.length := by decide +kernel

set_option maxRecDepth 100000 in
theorem npz_gauss : (gaussQ Data.npz.A.length (denseRows Data.npz.A Data.npz.sigma)).map
    (fun ps => ps.countP (fun x => decide (x < 0))) = some Data.npz.pairs.length := by decide +kernel

theorem count_of_gauss_map (A : List (List (Nat × Rat))) (hsym : symOK A = true) (σ : Rat) (k : Nat)
    (h : (gaussQ A.length (denseRows A σ)).map (fun ps => ps.countP (fun x => decide (x < 0))) = some k) :
    cntLt (isHermitian_of_symOK hsym) (σ : ℝ) = k := by
  cases hg : gaussQ A.length (denseRows A σ) with
  | none => rw [hg] at h; simp at h
  | some ps =>
    rw [hg] at h
    simp only [Option.map_some, Option.some.injEq] at h
    rw [count_of_gaussQ A hsym σ hg, h]

/-- `matrix-free`: exactly `k` eigenvalues of the decoded matrix lie below the recorded top shift. -/
theorem matrixFree_count :
    cntLt (isHermitian_of_symOK matrixFree_symOK) (Data.matrixFree.sigma : ℝ) = Data.matrixFree.pairs.length :=
  count_of_gauss_map _ _ _ _ matrixFree_gauss

/-- `dense-arrays`: exactly `k` eigenvalues below the recorded top shift. -/
theorem denseArrays_count :
    cntLt (isHermitian_of_symOK denseArrays_symOK) (Data.denseArrays.sigma : ℝ) = Data.denseArrays.pairs.length :=
  count_of_gauss_map _ _ _ _ denseArrays_gauss

/-- `npz`: exactly `k` eigenvalues below the recorded top shift. -/
theorem npz_count :
    cntLt (isHermitian_of_symOK npz_symOK) (Data.npz.sigma : ℝ) = Data.npz.pairs.length :=
  count_of_gauss_map _ _ _ _ npz_gauss

end CertifyMathlib.Refine
