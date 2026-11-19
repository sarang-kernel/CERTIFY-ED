import CertifyMathlib.Refine.Small
import CertifyMathlib.Refine.Inertia

/-! Axiom audit of the main theorems of `CertifyMathlib.Refine` (LEAN_PRODUCT.md section 6). Run with
`lake env lean Drafts/LeanProduct/AxiomsLeanProduct.lean`; every theorem must depend at most on `propext`,
`Classical.choice` and `Quot.sound`. -/

#print axioms CertifyMathlib.Refine.pairOK_theorem1
#print axioms CertifyMathlib.Refine.pairOK_theorem1_shift
#print axioms CertifyMathlib.Refine.pairOK_theorem1_shift_rad
#print axioms CertifyMathlib.Refine.checkLowest_theorem1
#print axioms CertifyMathlib.Refine.checkLowest_S2
#print axioms CertifyMathlib.Refine.matrixFree_theorem1
#print axioms CertifyMathlib.Refine.denseArrays_theorem1
#print axioms CertifyMathlib.Refine.npz_theorem1
#print axioms CertifyMathlib.Refine.matrixFree_S2
#print axioms CertifyMathlib.Refine.denseArrays_S2
#print axioms CertifyMathlib.Refine.npz_S2
#print axioms CertifyMathlib.Refine.symOK_rejects_asymmetric
#print axioms CertifyMathlib.Refine.vecOK_rejects_zero_vector
#print axioms CertifyMathlib.Refine.ldl_of_pivots
#print axioms CertifyMathlib.Refine.cntLt_of_pivots
#print axioms CertifyMathlib.Refine.pivots_of_gaussQ
#print axioms CertifyMathlib.Refine.count_of_gaussQ
