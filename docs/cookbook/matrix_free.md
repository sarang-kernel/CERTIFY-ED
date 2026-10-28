# Matrix-free producers

When only a `matvec` is available, CERTIFY-ED cannot bound the rounding inside it. With a declared spec, S1
compares the producer's `matvec` with the spec operator on random probes and S2–S4 are then computed for
the spec operator (which CERTIFY-ED can apply with bounded rounding). Without a spec, the residuals are
estimates and S2 is INCONCLUSIVE.

```python
"""Matrix-free producers: a LinearOperator with a declared spec (S1 by probes), and without one."""
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

import certify_ed as ce

spec = ce.heisenberg_chain(8, 1.0)
H = ce.build_bitwise(spec)                             # stands in for the producer's internal representation
w, V = la.eigh(H.toarray())

def matvec(x):                                         # the producer exposes only this
    return H @ x

L = spla.LinearOperator(H.shape, matvec=matvec, dtype=np.float64)
claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=L, kind="lowest",
                 producer={"name": "my_matvec"})

# with a spec: S1 compares the matvec with the spec operator on random probes; S2-S4 then bound the spec operator
cert = ce.certify(claim, spec)
print(cert.summary())
print("operator note:", cert.body["operator_note"])

# without a spec: nothing ties the matvec to a model; the rounding inside it cannot be bounded
cert2 = ce.certify(claim, None)
for s in cert2.statements:
    print(f"no spec  {s['id']:20s} {s['status']:12s} {s['evidence'].get('reason_code')}")
print("no spec  S2 summary:", cert2.statements[1]["summary"])
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
CERTIFY-ED certificate (certify-ed/certificate/2.1)
  subject : heisenberg_chain
  claim   : lowest claim, 4 eigenvalues
  [   CERTIFIED] S1.model_fidelity: matrix-free producer agrees with the spec on 3 random vectors (max rel. diff 0.0e+00)
  [   CERTIFIED] S2.eigenpairs: 4 claimed values matched to distinct eigenvalues; max radius 4.11e-14 (+6.8e-14 model shift)
  [   CERTIFIED] S3.completeness: exactly 4 eigenvalues below -2.74250724; certified gap to the next >= 3.859e-01
  [   CERTIFIED] S4.eigenspaces: 1/1 clusters certified at projector level
  [   CERTIFIED] S8.degeneracy: 1/1 clusters: eigenvalue count in the level enclosure certified by bracketing inertia counts; degeneracies [3]
  scope   : preset -; in scope S1, S2, S3, S4, S8
  narrowed: S5 (default: not part of the default selection); S6 (default: not part of the default selection); S7 (default: not part of the default selection)
  out of scope: 6 quantities with reason codes (see scope.out_of_scope, docs/SCOPE_MATRIX.md)
  not verified:
    - that the model specification is the Hamiltonian the authors intend (only producer operator == spec is checked)
    - correctness of the producer's code in general (only this result is checked)
    - interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')
    - anything outside the certified window [-inf, -2.7425072398402923]
    - eigenvector subspaces of isolated (non-cluster) pairs beyond their residuals
operator note: producer operator compared with the spec on random probes (S1); S2-S4 bounds are computed for the spec operator
no spec  S1.model_fidelity    NOT_CHECKED  NEEDS_SPEC
no spec  S2.eigenpairs        INCONCLUSIVE None
no spec  S3.completeness      NOT_CHECKED  NEEDS_MATRIX
no spec  S4.eigenspaces       NOT_CHECKED  NEEDS_WHOLE_SPECTRUM
no spec  S8.degeneracy        NOT_CHECKED  NEEDS_MATRIX
no spec  S2 summary: residuals computed but rounding not boundable for this operator (estimates only)
```

- `operator_note` records the substitution: the radii refer to the spec operator, tied to the producer by
  the probe comparison. The probes are a check of agreement on random vectors, not an entrywise proof;
  `Options.probe_vectors` (default 3) sets how many.
- Without a spec nothing can be certified for a black box: S2 says why (`rounding not boundable`), S3 and
  S8 need a stored matrix (`NEEDS_MATRIX`). Reason codes are in the [scope matrix](../SCOPE_MATRIX.md).
- From the command line, a matrix-free producer is `operator.py` in a bundle with `matvec(x)` and `dim`;
  it runs only with `--allow-code`.
