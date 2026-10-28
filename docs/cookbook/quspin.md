# QuSpin

`claim_from_quspin` rebuilds the spec from QuSpin's `static` list and the basis, and hands over QuSpin's
matrix as the producer operator. For a symmetry block (`kblock`, `pblock`, ...) the basis projector becomes
the claim's `sector_isometry`; S1 then audits that the block is an invariant, orthonormal restriction of the
spec operator, and the certificate carries the certified shift from the block to the full operator.

```python
"""QuSpin: a J1-J2 chain in the S^z = 0, k = 0 block; the claim and the spec come from claim_from_quspin."""
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

import certify_ed as ce

L = 10
spec = ce.heisenberg_chain(L, J2=0.3)                  # the intended model, written independently of QuSpin
static = spec.to_quspin_static()
basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pauli=0)
H = hamiltonian(static, [], basis=basis, dtype=np.float64,
                check_symm=False, check_herm=False, check_pcon=False)
E, V = H.eigsh(k=4, which="SA", v0=np.random.default_rng(1).standard_normal(basis.Ns))
order = np.argsort(E)
claim, spec_q = ce.claim_from_quspin(H, basis, static, E[order], V[:, order], kind="lowest")
print("sector basis size", basis.Ns, "; full dimension", spec.dim)
cert = ce.certify(claim, spec)
print(cert.summary())
print("sector dimension", cert.body["subject"]["sector"]["dimension"],
      " lift shift", f"{cert.body['subject']['sector']['lift_shift']:.2e}")
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
sector basis size 26 ; full dimension 1024
CERTIFY-ED certificate (certify-ed/certificate/2.1)
  subject : heisenberg_chain
  claim   : lowest claim, 4 eigenvalues
  [   CERTIFIED] S1.model_fidelity: producer sector block is an invariant, orthonormal restriction of the spec operator (isometry defect 8.1e-14, invariance defect 1.3e-13)
  [   CERTIFIED] S2.eigenpairs: 4 claimed values matched to distinct eigenvalues; max radius 1.92e-14 (sector block); 1.34e-12 for the full operator
  [   CERTIFIED] S3.completeness: exactly 4 eigenvalues below -1.748322187; certified gap to the next >= 5.302e-01
  [ NOT_CHECKED] S4.eigenspaces: no clusters among the claimed pairs
  [ NOT_CHECKED] S8.degeneracy: no clusters among the claimed pairs
  scope   : preset -; in scope S1, S2, S3
  narrowed: S5 (default: not part of the default selection); S6 (default: not part of the default selection); S7 (default: not part of the default selection)
  not checked: S4 [NO_CLUSTERS]; S8 [NO_CLUSTERS]
  out of scope: 6 quantities with reason codes (see scope.out_of_scope, docs/SCOPE_MATRIX.md)
  not verified:
    - that the model specification is the Hamiltonian the authors intend (only producer operator == spec is checked)
    - correctness of the producer's code in general (only this result is checked)
    - interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')
    - anything outside the certified window [-inf, -1.7483221869846926]
    - eigenvector subspaces of isolated (non-cluster) pairs beyond their residuals
sector dimension 26  lift shift 1.32e-12
```

- The radii come in two forms: `1.92e-14 (sector block)` is the Theorem 1 radius in the 26-dimensional
  block; `1.34e-12 for the full operator` adds the lift shift from the isometry and invariance defects.
  The exported `arrays["radius"]` includes the lift.
- S3 counts in the sector block: "exactly 4 eigenvalues below −1.748" is a statement about the k = 0,
  Sᶻ = 0 block, not about the full 1024-dimensional operator.
- Write the spec yourself (`heisenberg_chain(L, J2=0.3)`) and pass that to `certify`, as here, rather
  than the one rebuilt from `static`: the point of S1 is to compare the producer with a model declared
  independently of it.
