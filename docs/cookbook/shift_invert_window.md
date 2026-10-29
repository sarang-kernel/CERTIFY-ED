# Interior windows: shift-invert `eigsh` and `kind="window"`

Eigenvalues in the middle of the spectrum come from a shift-invert solve: `eigsh(H, k, sigma=sigma,
which="LM")` returns the `k` eigenvalues closest to `sigma`. The matching claim is a **window** claim: "the
list holds every eigenvalue in the open interval (a, b)". S3 then counts the eigenvalues below `a` and below
`b` by two inertia factorizations and compares the difference with the number of certified enclosures
inside (docs/theory.md, Lemma 4 at two shifts). Two things to get right:

- `eigsh` does not sort; sort what it returns.
- The window edges must lie in gaps. Compute two more values than you claim: the outermost ones show where
  the neighbours are, the edges go halfway into those gaps, the claim is the inner list. An edge that
  touches a claimed value or an unclaimed neighbour makes S3 INCONCLUSIVE (the counts cannot be attributed),
  and a window that reaches past the list is FAILED.

```python
"""Interior window by shift-invert: scipy.sparse.linalg.eigsh(sigma=..., which="LM"), then kind="window"."""
import numpy as np
import scipy.sparse.linalg as spla

import certify_ed as ce

spec = ce.heisenberg_chain(12, 1.0, J2=0.4, boundary="open")     # J1-J2 chain, L = 12, open ends
states = ce.u1_state_set(spec, 6, label="Sz=0")                 # the Sz = 0 block (dimension 924)
H = ce.build_bitwise(spec, states)                               # scipy.sparse CSR on that block
sigma = -1.0                                                     # the energy the eigenvalues are wanted around
rng = np.random.default_rng(0)
# shift-invert: the eigenvalues closest to sigma are the largest in magnitude of (H - sigma)^-1
w, V = spla.eigsh(H, k=10, sigma=sigma, which="LM", tol=1e-12, v0=rng.standard_normal(H.shape[0]))
order = np.argsort(w)
w, V = w[order], V[:, order]
# A window claim says: the list holds EVERY eigenvalue in (a, b). The two outermost values computed are
# only there to show where the gaps are; the window edges go into those gaps and the claim is the inner 8.
a, b = 0.5 * (w[0] + w[1]), 0.5 * (w[-2] + w[-1])
w_in, V_in = w[1:-1], V[:, 1:-1]
print("eigsh around sigma =", sigma, ":", np.array2string(w, precision=6))
print("window (%.6f, %.6f), %d claimed inside" % (a, b, w_in.size))

cert = ce.quick(H, w_in, V_in, spec, kind="window", window=(a, b), states=states,
                producer={"name": "scipy.sparse.linalg.eigsh", "sigma": sigma, "which": "LM", "tol": 1e-12})
print(cert.summary())
for s in cert.statements[:3]:
    print(f"  {s['id']:18s} {s['status']:12s} {s['summary']}")

# the same list with a window that reaches past it: FAILED (eigenvalues exist there), not an error
cert2 = ce.quick(H, w_in, V_in, spec, kind="window", window=(a, b + 0.2), states=states, only="S2,S3")
s3 = [s for s in cert2.statements if s["id"] == "S3.completeness"][0]
print("wider window:", s3["status"], "-", s3["summary"])
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
eigsh around sigma = -1.0 : [-1.026149 -1.021446 -1.021392 -1.020198 -1.015014 -1.007654 -1.005275
 -0.999545 -0.991217 -0.985182]
window (-1.023798, -0.988199), 8 claimed inside
3 of 5 statements CERTIFIED (S1, S2, S3), 2 NOT_CHECKED (S4, S8); max certified radius 2.91e-13 (rounded up); completeness holds.
  S1.model_fidelity  CERTIFIED    producer matrix equals the spec matrix entrywise (max |diff| 0.0e+00 <= 1.4e-12)
  S2.eigenpairs      CERTIFIED    8 claimed values matched to distinct eigenvalues; max radius 2.55e-14 (+2.6e-13 model shift)
  S3.completeness    CERTIFIED    claim contains every eigenvalue in (-1.023797586, -0.9881994701)
wider window: FAILED - at least 41 eigenvalue(s) in the window are not claimed
```

- The certified window in S3 is the inward-rounded `(a + beta_a, b - beta_b)`, where `beta_a`, `beta_b` are
  the backward-error bounds of the two factorizations (`evidence.counts`); the certificate's `not_verified`
  list says that nothing outside it is claimed.
- The same claim through every other entry point: `ce.Claim(w_in, V_in, operator=H, kind="window",
  window=(a, b))`; `claim_from_arrays / claim_from_npz / claim_from_hdf5 / claim_from_file(..., kind="window",
  window=(a, b))`; a bundle with `"kind": "window", "window": [a, b]` in `meta.json`; and on the command
  line `certify-ed certify claim.npz --spec spec.json --kind window --window A B` (also `--window=A,B`, the
  spelling to use when A is negative; the flag overrides the bundle's `meta.json`). An endpoint that is not finite, a reversed pair, or a window
  claim without a window is refused when the `Claim` is built, with the reason in the message.
- For a sector block pass `states=` (U(1) block in the spec's encoding, as here) or `sector_isometry=`;
  S1 then audits the block against the spec and the radii carry the lift to the full operator.
