import CertifyMathlib.NonHermitian

/-! Axioms of every theorem of `CertifyMathlib.NonHermitian` (roadmap items E-02 and N-13).
Run from `reverification_lean/`: `lake env lean Drafts/NonHermitian/PrintAxioms.lean`. -/

open CertifyMathlib.NonHermitian

#print axioms bauerFike_core
#print axioms bauerFike_residual
#print axioms similarity_inclusion
#print axioms neumann_bound
#print axioms neumann_inverse
#print axioms backward_error
#print axioms diag_lower
#print axioms bauerFike
#print axioms toEuclideanLin_diagonal_apply
#print axioms toEuclideanLin_mul_apply
#print axioms toEuclideanLin_one_apply
#print axioms bauerFike_matrix
#print axioms bauerFike_residual_matrix
#print axioms disc_inclusion_matrix
#print axioms bauerFike_matrix_sharp
#print axioms disc_inclusion_certified
