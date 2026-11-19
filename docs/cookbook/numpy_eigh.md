# numpy arrays from `scipy.linalg.eigh`

The producer is LAPACK. `claim_from_arrays` wraps eigenvalues, eigenvectors and the matrix; `kind="all"`
claims the whole spectrum (S3 then checks that the count equals the dimension), `kind="lowest"` claims the
k lowest (S3 counts eigenvalues below a shift by an LDLᵀ factorization and certifies the gap).

```python
"""Dense arrays from scipy.linalg.eigh: the whole spectrum (kind 'all') and the k lowest (kind 'lowest')."""
import numpy as np
import scipy.linalg as la

import certify_ed as ce

spec = ce.tfim_chain(8, 1.0, 0.8)                      # transverse-field Ising chain, L = 8
H = ce.build_bitwise(spec).toarray()
w, V = la.eigh(H)
print("lowest ten:", np.array2string(w[:10], precision=6))

# the whole spectrum: every eigenvalue claimed, count must equal the dimension
cert_all = ce.certify(ce.claim_from_arrays(w, V, operator=H, kind="all"), spec)
for s in cert_all.statements:
    print(f"all      {s['id']:20s} {s['status']}")
print("all      max radius", f"{cert_all.arrays['radius'].max():.2e}")

# 'the six lowest': the sixth level is a quadruplet, so 'the k lowest' is ill-posed for k = 6
for k in (6, 5):
    cert = ce.certify(ce.claim_from_arrays(w[:k], V[:, :k], operator=H, kind="lowest"), spec)
    s3 = cert.statements[2]
    print(f"k={k}      S3.completeness      {s3['status']}")
    if s3["status"] == "CERTIFIED":
        print(f"k={k}      gap >= {s3['evidence']['gap_lower_bound']:.6f}; next eigenvalue >= {s3['evidence']['next_eigenvalue_lower_bound']:.6f}")
    else:
        print(f"k={k}      reason: {s3['evidence']['reason']}")
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
lowest ten: [-9.365204 -9.317101 -7.756266 -7.490737 -7.490737 -6.533218 -6.533218
 -6.533218 -6.533218 -6.464372]
all      S1.model_fidelity    CERTIFIED
all      S2.eigenpairs        CERTIFIED
all      S3.completeness      CERTIFIED
all      S4.eigenspaces       CERTIFIED
all      S8.degeneracy        CERTIFIED
all      max radius 1.55e-12
k=6      S3.completeness      INCONCLUSIVE
k=6      reason: the claim is complete below -6.533217956, but the highest claimed level is a (quasi-)degenerate multiplet of at least 4 eigenvalues within [-6.533217956, -6.533217768], of which only 1 are claimed: 'the k lowest' is ill-posed for this k
k=5      S3.completeness      CERTIFIED
k=5      gap >= 0.861767; next eigenvalue >= -6.628970
```

Reading it:

- For `kind="all"` the radius is the only number that matters; 1.55 × 10⁻¹² here, for a dense 256 × 256
  matrix. With a complete spectrum every level is a cluster of its own, so S4 and S8 run on all of them.
- `k = 6` is **INCONCLUSIVE, not FAILED**: the claim is complete below the sixth level, but that level is a
  quadruplet of which only one member is claimed. "The six lowest eigenvalues" is not a well-posed claim
  for this spectrum; the certificate says so and names the interval. Claim the whole multiplet or stop
  before it (`k = 5`), which is CERTIFIED with a gap of 0.86.
- Model shift: `build_bitwise` and the producer's matrix agree entrywise here (max |diff| 0), so the shift
  is the builder's own rounding bound, a few 10⁻¹⁴, added to every radius.
