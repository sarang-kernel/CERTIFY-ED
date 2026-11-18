import CertifyCore.Kernel.Data.MatrixFree
import CertifyCore.Kernel.Data.DenseArrays
import CertifyCore.Kernel.Data.Npz
/-
Kernel-checked S2/S3 arithmetic of the three smallest validation certificates (the conformance fixtures
`matrix-free`, `dense-arrays` and `npz` of `code/certify-ed/conformance/reference`, dimension 64, claims of the
4, 4 and 1 lowest eigenpairs; certify-ed reports S2 and S3 CERTIFIED for all three). `checkLowest` is defined in
`CertifyCore/Kernel/Defs.lean`; its docstring lists what `checkLowest c = true` asserts. The proofs are
evaluations of `checkLowest` by the Lean kernel (`decide +kernel`); no compiled code is involved.
-/
namespace CertifyCore.K

set_option maxRecDepth 100000 in
theorem matrixFree_S2_S3 : checkLowest Data.matrixFree = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem denseArrays_S2_S3 : checkLowest Data.denseArrays = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem npz_S2_S3 : checkLowest Data.npz = true := by decide +kernel

/-- The radius forgery of the C verifier's validation, applied to the literal data: the residual bound, the
existence radius and the radius of the pair with index 0 divided by 10^4 (records only). -/
def forgeRadius (c : Cert) : Cert :=
  { c with pairs := c.pairs.mapIdx (fun i p => if i = 0 then { p with rb := p.rb / 10000, ex := p.ex / 10000, rad := p.rad / 10000 } else p) }

set_option maxRecDepth 100000 in
/-- Negative control: the forged data are rejected by the same kernel evaluation. -/
theorem npz_forged_radius_rejected : checkLowest (forgeRadius Data.npz) = false := by decide +kernel

end CertifyCore.K
