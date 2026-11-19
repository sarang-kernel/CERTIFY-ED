# Single-precision inputs: bounds the computation did not support (3.0.0 to 3.2.3)

Status: fixed in the 3.3 development line (not yet released); the CHANGELOG entry 'Single-precision inputs (soundness
fix)' under 3.3.0.dev0 describes the fix.

Certificates computed from inputs in a floating format narrower than binary64 (float16, float32, complex64) could state
bounds that the computation did not support: numpy and scipy formed the products, and the sums of repeated sparse
entries, in the inputs' own precision (unit roundoff 2^-24 for float32), while every bound counts binary64 rounding
(unit roundoff 2^-53).

## Affected versions

3.0.0, 3.1.0, 3.2.0, 3.2.1, 3.2.2 and 3.2.3, checked at their release commits (2bd9a39, e860a83, 55d9f61, f3c478b,
ddf83e1, 145cc00), and dev/3.3 up to cd0ca97. The 1.x and 2.x lines were not checked.

## Affected inputs

- Any input in a narrower floating format, in any position: the operator, the claimed eigenvectors, a sector isometry or
  block, an observable or symmetry matrix, the inputs of the standalone records. Example: A = [[1, 2^-30], [2^-30, 1]]
  in float32 (exact entries), x = (1, 1), claimed eigenvalue 1, kind 'subset': S2 CERTIFIED with radius 8.9e-16 (3.6e-15
  for complex64), although the eigenvalues of A are 1 -+ 2^-30, at distance 9.3e-10.
- A binary64 operator with single-precision vectors (the norm of the vectors was formed in single precision). Example:
  the float64 CSR matrix [[1, 2^-20], [2^-20, 1]], x = (7, 7) in float32, accuracy_target 1e-5: S2 CERTIFIED with radius
  9.536742991635583e-07 (complex64: 9.536743018280948e-07), below the distance 2^-20 = 9.5367431640625e-07 to the
  eigenvalues, at all seven commits above.
- A single-precision sparse matrix with repeated entries (COO, or CSR with duplicate indices), in memory or read from a
  file by `claim_from_file`, `claim_from_npz`, `claim_from_bundle` or the command line (3.1.0 onwards): the format
  conversions summed the repeated entries in single precision, so the statements concerned a rounded matrix. Example: a
  float32 COO matrix with (0, 0) stored as 1 and 2^-25 and (1, 1) = 3, claimed eigenvalues (1, 3), X = I: S2 CERTIFIED
  with radius 6.661338147750968e-16 (complex64: 3.108624468950454e-15), although the first eigenvalue is 1 + 2^-25, at
  distance 3.0e-08; at all seven commits in memory and from 3.1.0 on through an npz file and `claim_from_file`.

Not affected: certificates whose inputs were all float64 or complex128, and integer inputs (an integer beyond 2^53 was
rounded to binary64 without notice; it is refused now). One integer case differs, to the safe side: for an integer
matrix whose absolute row sums exceed 2^63, 3.x gave a negative norm bound (norm_upper_bound -9.2e18 for the int64
matrix [[2^62, 2^62], [2^62, 2^62]]) and a spurious S2 FAILED; the 3.3 development line gives the certificate of the
float64 copy.

## Affected statements and records

- S2, and the statements that rest on its enclosures or on an inertia count of a narrower operator: S3, S4, S7 and S8.
  In the 3.2.3 kind 'lowest' variant of the first example S3 is INCONCLUSIVE, so no false S3 was demonstrated; S3 rests
  on S2 all the same.
- S1: its entrywise comparison and its probes promote a single-precision matrix to binary64 (checked on the Heisenberg
  chain of 6 sites: a float32 copy of the matrix gives the S1 status, difference and model-shift bound of the float64
  matrix at 3.2.3 and cd0ca97, by entries and by probes). Two parts were affected: a single-precision sparse matrix was
  compared after the single-precision sum of its repeated entries (the same matrix as a float32 COO with an extra entry
  2^-28 on the diagonal: S1 CERTIFIED with max |diff| 0 at 3.2.3 and cd0ca97, FAILED with 3.7e-09 for its float64 copy),
  and the sector audit formed the products of a single-precision isometry or block in that precision.
- S6 is not affected: it works in exact arithmetic on the spec operator and on the exact rational values of the claimed
  eigenvalues and vectors (on the same chain with float32 and complex64 vectors, S6 gives the same radii at cd0ca97 as
  on this branch; its bound on the next eigenvalue differs only through the floating-point hint it verifies).
- The standalone records and helpers that formed products or sums of the given arrays: `certify_variational`,
  `certify_generalized` and its inertia count, `certified_moments`, `enclose_spectrum`, `certified_sweep` and
  `sweep.matrix_affinity_defect`, the sector audits and the decomposition audit of `compose.certify_low_energy`,
  `compat.symveig` with `symmetrize=True`, `eigenspace.commutes_signed_permutation`, `sectors.trace_moments_check`,
  `correlations.hermitian_parts`, `davis_kahan.labels_hook`, the Rayleigh interval of Kato-Temple and the norm bounds of
  `fp`. The non-Hermitian record converted its inputs to binary64 already.

No evidence-run input, example or test of the repository passes single-precision data (git grep: float32 appears only as
the storage of a fixture's vectors, converted to float64 before use, in tests/test_whole_spectrum.py and
RESULT_whole_spectrum.md), so no published number changes.

## How to recognise an affected 3.x certificate

3.x certificates do not record the input dtypes. When the eigenvectors were stored (`store_eigenvectors`), the arrays
sidecar records their dtype: `arrays.dtypes.eigenvectors` is '<f4' (float32) or '<c8' (complex64) for affected vectors.
Otherwise `claim.eigenvectors_sha256` of a 3.x certificate equals `certificate_v2.array_sha256` of the producer's
vectors in their own dtype, so a holder of those vectors can check whether they were single precision. The operator's
dtype cannot be read off a certificate (`matrix_sha256` hashes the matrix after its conversion): the holder of the
certificate has to check the producer's code or files for narrower operators, isometries and observables and for
repeated sparse entries.

## Workaround for 3.x

Convert every array to float64 or complex128 before the call: the operator, the vectors, the sector isometry and block,
observables, symmetry matrices and the inputs of the standalone records. For a COO, CSR or CSC matrix convert the stored
value array and keep the index arrays, `B = A.copy(); B.data = A.data.astype(np.float64)` (or `np.complex128`): no entry
is summed before the conversion, whatever the scipy version. `A.astype(np.float64)` converts before it sums repeated
entries only from scipy 1.11; up to scipy 1.10 it sums them first, in single precision and in place on `A` (scipy's
source: 1.8.1 and 1.10.1 sum, then convert; 1.11.0 converts, then sums). A format conversion first
(`A.tocsr().astype(...)`) sums them in single precision on every version. For files, load the arrays, convert them the
same way and pass the arrays to `Claim`; 3.x's readers convert single-precision sparse files to CSR in single precision.
Checked at 3.0.0 and 3.2.3 (scipy 1.17.1): with the stored values converted, the first example gives radius 9.3e-10 and
the COO example 2.98e-08 = 2^-25, for COO, CSR and CSC, float32 and complex64, and `A` is unchanged;
`A.tocsr().astype(np.float64)` still gives 6.7e-16.
