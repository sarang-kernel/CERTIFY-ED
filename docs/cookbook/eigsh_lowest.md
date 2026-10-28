# `scipy.sparse.linalg.eigsh`, the k lowest

A sparse matrix and an iterative solver. Two things to get right: sort what `eigsh` returns (it does not),
and record the producer's settings in `Claim.producer`, which goes into the certificate unchanged.

```python
"""Sparse matrix, scipy.sparse.linalg.eigsh for the k lowest eigenpairs."""
import numpy as np
import scipy.sparse.linalg as spla

import certify_ed as ce

spec = ce.heisenberg_chain(12, 1.0, J2=0.4)            # J1-J2 chain, L = 12 (dimension 4096)
H = ce.build_bitwise(spec)                             # scipy.sparse CSR
k = 5
rng = np.random.default_rng(0)
w, V = spla.eigsh(H, k=k, which="SA", tol=1e-12, v0=rng.standard_normal(H.shape[0]))
order = np.argsort(w)                                  # eigsh does not sort
w, V = w[order], V[:, order]

claim = ce.Claim(eigenvalues=w, eigenvectors=V, operator=H, kind="lowest",
                 producer={"name": "scipy.sparse.linalg.eigsh", "tol": 1e-12})
cert = ce.certify(claim, spec)
print(cert.summary())
print("radii", np.array2string(cert.arrays["radius"], precision=2))
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
CERTIFY-ED certificate (certify-ed/certificate/2.1)
  subject : heisenberg_chain
  claim   : lowest claim, 5 eigenvalues
  [   CERTIFIED] S1.model_fidelity: producer matrix equals the spec matrix entrywise (max |diff| 0.0e+00 <= 2.7e-12)
  [   CERTIFIED] S2.eigenpairs: 5 claimed values matched to distinct eigenvalues; max radius 1.09e-13 (+3.3e-13 model shift)
  [   CERTIFIED] S3.completeness: exactly 5 eigenvalues below -4.156272135; certified gap to the next >= 1.545e-01
  [   CERTIFIED] S4.eigenspaces: 1/1 clusters certified at projector level
  [   CERTIFIED] S8.degeneracy: 1/1 clusters: eigenvalue count in the level enclosure certified by bracketing inertia counts; degeneracies [3]
  scope   : preset -; in scope S1, S2, S3, S4, S8
  narrowed: S5 (default: not part of the default selection); S6 (default: not part of the default selection); S7 (default: not part of the default selection)
  out of scope: 6 quantities with reason codes (see scope.out_of_scope, docs/SCOPE_MATRIX.md)
  not verified:
    - that the model specification is the Hamiltonian the authors intend (only producer operator == spec is checked)
    - correctness of the producer's code in general (only this result is checked)
    - interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')
    - anything outside the certified window [-inf, -4.156272134986229]
    - eigenvector subspaces of isolated (non-cluster) pairs beyond their residuals
radii [3.60e-13 3.77e-13 4.41e-13 4.41e-13 4.41e-13]
```

- The radii (≈ 4 × 10⁻¹³, including the model shift) are what the residuals support, not the solver's
  `tol`. The accuracy presumption is `1e-8 × max(1, ‖H‖)` unless `Claim(accuracy_target=...)` says
  otherwise; a radius above it makes S2 FAILED.
- The three highest claimed values are a triplet (S8: `degeneracies [3]`); S3 certifies that exactly five
  eigenvalues lie below −4.156 and the sixth is at least 0.1545 above the fifth, so `k = 5` is well posed.
- Dimension 4096 is factorized densely for the inertia count (`Options.inertia_max_dim = 60000` is the
  default limit; above it S3 is NOT_CHECKED with `SIZE_LIMIT`).
