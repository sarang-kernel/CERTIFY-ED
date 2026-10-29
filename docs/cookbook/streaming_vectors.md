# Vectors larger than RAM, and the plan

## Streamed eigenvectors (memmap, HDF5)

`certify()` accepts the eigenvector matrix as a numpy memmap (`np.load(path, mmap_mode="r")` of an `.npy`
file) or as an `h5py` dataset, and reads it in column blocks: the residuals `H X - X diag(mu)`, their
rounding bounds, the per-column norms, and later the few columns of each cluster (Gram matrix, projector
bounds, observables) and of each isolated pair (Kato-Temple). The block width follows `Options(block_mb=...)`
(default `budget_mb`, else 512 MB of temporaries). Nothing in the certificate depends on the block width or
on where the vectors live: with a sparse operator the product `H X[:, a:b]` is, column for column, the same
floating-point result as `H X` (one ordered sum per entry), and the test suite checks the whole certificate
(radii, enclosures, Kato-Temple intervals, thermodynamic intervals, hashes) for memmap and HDF5 input in both
memory layouts against the in-memory run.

```python
"""Eigenvectors larger than RAM: a numpy memmap (.npy on disk) or an HDF5 dataset, read in column blocks."""
import os
import tempfile

import numpy as np
import scipy.linalg as la

import certify_ed as ce

spec = ce.heisenberg_chain(12)
states = ce.u1_state_set(spec, 6, label="Sz=0")             # dimension 924
H = ce.build_bitwise(spec, states)                           # sparse: the streamed path applies (see the page)
w, V = la.eigh(H.toarray())                                  # the producer's full diagonalization

d = tempfile.mkdtemp()
path = os.path.join(d, "V.npy")
mm = np.lib.format.open_memmap(path, mode="w+", dtype=V.dtype, shape=V.shape)   # what a producer would write
mm[:] = V
mm.flush()
del mm

V_disk = np.load(path, mmap_mode="r")                        # never loaded whole by certify()
cert = ce.quick(H, w, V_disk, spec, kind="all", states=states, block_mb=64, plan=True)
print(cert.summary())
cert_mem = ce.quick(H, w, V, spec, kind="all", states=states)
print("same radii as in memory:", np.array_equal(cert.arrays["radius"], cert_mem.arrays["radius"]),
      "| same enclosures:", cert.body["enclosures"] == cert_mem.body["enclosures"])
print("plan recorded:", cert.body["plan"]["peak_mb"], "MB predicted peak,", round(cert.body["plan"]["total_seconds"], 1), "s predicted")
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
3 of 5 statements CERTIFIED (S1, S2, S3), 2 NOT_CHECKED (S4, S7); max certified radius 1.98e-13 (rounded up); completeness holds; preset whole-spectrum.
same radii as in memory: True | same enclosures: True
plan recorded: 26.1 MB predicted peak, 0.3 s predicted
plan: n = 924, nnz = 6572, k = 924, kind = all, sparse operator, streamed vectors; baseline 0 MB
  statement            runs  method                         seconds   peak MB  basis
  S1.model_fidelity    yes   spec reference, entrywise          0.00         0  3 nnz (b+8) bytes; ~0.1 us per entry
  S2.eigenpairs        yes   residual enclosure (Thm 1-2)      0.27        26  4 blocks of 924 columns; 462 clusters x 3 exact dots
  S3.completeness      yes   count = dimension (kind 'all')      0.00         0  no factorization
  S4.eigenspaces       no    projector bounds per cluster      0.18         0  cluster columns only
  S5.corroboration     no    second solver                     0.00         0  not modelled
  S6.exact_recheck     no    integer arithmetic                0.00        39  n^2 x 48 bytes (the 3.1 estimate); time not modelled
  S7.thermodynamics    no    interval sums                     0.37         0  per interval and temperature
  S8.degeneracy        no    dense inertia x 924             394.84        55  two factorizations per cluster
  total 0.3 s predicted, peak 26 MB; inertia: dense (dimension 924 <= 2500: Bunch-Kaufman (about 55 MB))
  advisory: measured rates of one machine (certify_ed.cost.RATES); the plan never changes a status
```

Limits, stated plainly:

- **A dense operator reads the vectors whole.** BLAS `gemm` is not invariant under column blocking (the
  result of one column depends on how many columns are multiplied at once, verified on OpenBLAS), so a dense
  `H` is applied to all claimed vectors in one call. This is no restriction in practice: a dense `n x n`
  operator is at least as large as `n x k` vectors, so when the vectors do not fit, neither does the dense
  operator. Pass the sparse matrix (or a spec, for the matrix-free operator) for the streamed path.
- **Whole-spectrum S3 by dense inertia still needs H in memory.** For `kind="all"` S3 is the complete count
  (no factorization); for `lowest` and `window` claims the inertia counts factorize the stored matrix, dense
  (Bunch-Kaufman, about 8.5 n^2 entries of workspace) or sparse (SuperLU). The vectors are not involved.
- The eigenvalue list, the per-column scalars (norms, radii) and the residual columns of one cluster are in
  memory; the residual matrix `R` is kept only for in-memory vectors.
- S6 (exact re-check, dimension <= `exact_max_dim`) loads the vectors; S5 does not use them.

## The plan: predicted time and memory per statement

`Options(plan=True)`, `--plan`, or `certify-ed plan CLAIM.npz [--spec SPEC.json] [--kind ...] [--preset ...]`
print the predicted wall seconds and peak memory of every statement before anything runs
(`certify_ed.cost.plan_for_claim`): memory counted array by array from `n`, `nnz` and `k`; time from the
rates of one machine (`cost.RATES`, measured on Heisenberg chain blocks of dimension 924, 3432 and 12870;
`experiments/planner_calibration.py` re-measures them). The plan is recorded under `body["plan"]`. It is
advisory: it never changes a status.

What it does change is the path tried for the inertia counts when `Options.inertia_method == "auto"`: the
3.1 rule (dense for a dense matrix or a sparse one of dimension <= 2500, else SuperLU) stands, except that a
sparse matrix whose dense `LDL^T` workspace would exceed `budget_mb` (or, without a budget, the available
physical memory) goes to SuperLU instead of failing with `MemoryError` (which would have made S3
NOT_CHECKED). On the fixtures the statuses with and without the guard are identical (`tests/test_planner.py`).
