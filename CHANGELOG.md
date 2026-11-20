# Changelog

## [3.3.0.dev0] — unreleased (dev/3.3)

`dev/3.3` starts at `v3.2.2`, merges `fix/3.2.3` (tag `v3.2.3`) and carries everything of [3.2.3] and [3.2.2] below. The integration sets the version to
3.3.0.dev0, lets `parallel.process_memory_cap_mb` read the cap through `cost.process_memory_cap` (the reader of the
3.2.2 memory gate), so that the worker budgets and the gate see the same limit, and widens the version pattern of
`tests/test_review_fixes.py`.

### Non-finite follow-ups

No constant changes; statuses change only on inputs whose accounting overflows. When the radii of the claimed pairs are
finite but the backward error of a factorization overflows (entries of about 1e200: its squares exceed the largest
float), S3 now carries `NONFINITE_BOUND` (it was INCONCLUSIVE without a code). The non-Hermitian record (N1) and the
pencil record (G2) follow the rules of 'Non-finite bounds give no statement': a residual bound or radius that is not
finite is stored as ∞ (it was nan for an entry within 12u of the largest float, where ∞ · 0 = nan), and the reason
code is `NONFINITE_BOUND` for an overflow and `UNRESOLVED_VECTOR_NORM` for a vector whose norm cannot be bounded away
from zero (both were `NOT_APPLICABLE`, and G2 named the vector norm as the cause in both cases); N3 uses
`NONFINITE_BOUND` for an overflowing disc radius. `fp.exact_sum` returns the IEEE sum of infinite terms (∞ or -∞ for
one sign, nan for both signs or a nan term) where `math.fsum` raised a ValueError, and nan for finite terms whose
exact sum exceeds the largest float, where `math.fsum` raised an OverflowError. `fp.exact_dot` no longer applies the
error-free split to a product that is not finite, so ∞ · 0 gives nan (the pair was dropped as tiny, and the dot
product of (∞, 1) and (0, 1) came out as 1). The callers of these sums state nothing on a nan: the Gram entries and
squared norms of complex vectors scaled by 1e155 (S2 and S3 INCONCLUSIVE with `UNRESOLVED_VECTOR_NORM`; this raised
before), the Rayleigh interval used by `certify_variational` and by the Kato-Temple bound (no finite end: V1 to V3
INCONCLUSIVE with `NONFINITE_BOUND`, and the theta record stores null and ∞ instead of nan), and x^H B x in
`certify_generalized` (G2 `NONFINITE_BOUND`; its evidence lists the unresolved and the overflowed pairs separately).
`certified_moments` no longer raises for a complex observable entry whose products overflow (1e308 + 1e308i, for
example) or for a Hamiltonian whose norm to the power k overflows (entries of 1e200), and gives M1 to M3
INCONCLUSIVE with `NONFINITE_BOUND`; its record stores such a bound as ∞ with no interval, and a nan value as null.
S3 takes its code from a flag set where the count is decided, not from the text of the evidence. The sum rules M2 and
M3 are INCONCLUSIVE with `NONFINITE_BOUND` when an end of either interval is not finite (an interval (-∞, ∞) overlaps
every interval, and M3 was CERTIFIED on one), and so is M1 when an end of a moment interval overflows. Two record fields
that could hold nan store ∞: `theta_source_vs_contracted` of a contracted trial state and the computed residual norm
`residual_computed` of S2, and so does the S1 comparison of the specification invariants, where couplings of 1e160
give inf - inf. A factorization with an infinite or nan pivot gives no count (the conversion of the pivot to a
fraction raised, for a B with entries near the largest float in `certify_generalized`), and neither does a count at a
shift that is not finite (LAPACK raised a ValueError). The S3 count records
carry `nonfinite_bound` only when it is true, so the record of a finite count is unchanged. The INCONCLUSIVE summary of
a window S3 now gives the reason recorded in its evidence (for example, inertia inconclusive at a window edge) instead
of a fixed text. `docs/SCOPE_MATRIX.md` states which families give `UNRESOLVED_VECTOR_NORM` and which give
`NONFINITE_BOUND` for an overflowing squared norm. Tests: `tests/test_nonfinite_followups.py` (each case fails at the
base). The computational source hash changes.

### Result documents moved under docs/

No code, bound or status changes, and the computational source hash stays the same. The result documents that were at
the repository root move into the documentation tree: the `RESULT_*.md` files of the evidence runs and development
tracks and `E6_inspection_worksheet_3.2.0.md` to `docs/results/` (its `README.md` lists them), `ROADMAP.md` to
`docs/roadmap/ROADMAP.md`, and `HASH_MAP.csv` to `docs/project/HASH_MAP.csv`. The file names are unchanged, so entries
below and earlier documents that name a file without a folder still identify it. `experiments/figures/fig_mutation.py`
reads `docs/results/RESULT_integration_3.2.md`, `tests/test_packaging.py` checks `docs/roadmap/ROADMAP.md`, a test of
`tests/test_figures.py` checks that path without run records (in a separate process, so that the script's helper module
`_common` does not shadow the module of that name of `docs/proofs/` in later tests), `.gitignore` keeps `docs/results/`
tracked, and `mkdocs.yml` and `MANIFEST.in` leave both folders out of the documentation site and the source archive, as
before. The copies of the 3.2.2 and 3.2.3 result documents in `docs/project/sources/runs/` are removed (the 3.2.2 copy
lacked the note on single-precision inputs), and the project changelog cites `docs/results/` instead.

### Wording of the records

No bound, status rule or record field changes, and the computational source hash stays the same. The status lines of the
derivation documents and the headers of the independent re-derivations in `docs/proofs/` name no tool; an independent
re-derivation is described as written before the original proof was read (the statement and the code mapping were read
first), not a human check (`tests/test_proofs_props.py` pins the new wording). Entries of earlier versions below that
quote these status lines quote the new wording. `docs/theory.md` section 10 now says that the independent re-derivations
are done (not a human check), as their status lines say. The practice survey calls its re-coding of 12 papers a check
reading, a second automatic reading and not a human check (`PREREG_practice_survey.md` v0.3, wording only); the blog, a
decision record and the roadmap had called it a hand check or a hand coding. Suggested readings of mutation survivors
are called automatic suggestions, and one free-text note of a Lean result file (`lean/results/apps/mathlib_apps.json`)
is dated. Sentences on how the texts were prepared and on the statement on assistance made at each submission are worded
without naming a tool. Working notes of the development environment are removed from the track reports, the project
changelog and the derivation documents, except where they give the reason for a failed run. `tools/lint_docs.py`
compares product and tool names by their SHA-256 digests, so that it does not spell them, and
`tests/test_process_words.py` applies that check to every tracked text file except the third-party survey data.

### Single-precision inputs (soundness fix)

Certificates computed from inputs in a floating format narrower than binary64 (float16, float32, complex64) could state
bounds that the computation did not support: numpy and scipy formed the products, and the sums of repeated sparse
entries, in the inputs' own precision (unit roundoff 2^-24 for float32), while every bound counts binary64 rounding
(unit roundoff 2^-53).

Affected versions: 3.0.0, 3.1.0, 3.2.0, 3.2.1, 3.2.2 and 3.2.3, checked at their release commits (2bd9a39, e860a83,
55d9f61, f3c478b, ddf83e1, 145cc00), and dev/3.3 up to cd0ca97. The 1.x and 2.x lines were not checked.

Affected inputs:

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

Affected statements and records:

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

How to recognise an affected 3.x certificate: 3.x certificates do not record the input dtypes. When the eigenvectors
were stored (`store_eigenvectors`), the arrays sidecar records their dtype: `arrays.dtypes.eigenvectors` is '<f4'
(float32) or '<c8' (complex64) for affected vectors. Otherwise `claim.eigenvectors_sha256` of a 3.x certificate equals
`certificate_v2.array_sha256` of the producer's vectors in their own dtype, so a holder of those vectors can check
whether they were single precision. The operator's dtype cannot be read off a certificate (`matrix_sha256` hashes the
matrix after its conversion): the holder of the certificate has to check the producer's code or files for narrower
operators, isometries and observables and for repeated sparse entries.

Workaround for 3.x: Convert every array to float64 or complex128 before the call: the operator, the vectors, the sector
isometry and block, observables, symmetry matrices and the inputs of the standalone records. For a COO, CSR or CSC
matrix convert the stored value array and keep the index arrays, `B = A.copy(); B.data = A.data.astype(np.float64)` (or
`np.complex128`): no entry is summed before the conversion, whatever the scipy version. `A.astype(np.float64)` converts
before it sums repeated entries only from scipy 1.11; up to scipy 1.10 it sums them first, in single precision and in
place on `A` (scipy's source: 1.8.1 and 1.10.1 sum, then convert; 1.11.0 converts, then sums). A format conversion first
(`A.tocsr().astype(...)`) sums them in single precision on every version. For files, load the arrays, convert them the
same way and pass the arrays to `Claim`; 3.x's readers convert single-precision sparse files to CSR in single precision.
Checked at 3.0.0 and 3.2.3 (scipy 1.17.1): with the stored values converted, the first example gives radius 9.3e-10 and
the COO example 2.98e-08 = 2^-25, for COO, CSR and CSC, float32 and complex64, and `A` is unchanged;
`A.tocsr().astype(np.float64)` still gives 6.7e-16.

Advisory: `docs/failures/single_precision_inputs.md` (the text above).

The fix. The public functions of the 30 modules listed in `tests/test_precision_registry.py` that take a stored matrix
or vectors pass them through `precision.binary64` before any product or format conversion: float16, float32 and
complex64 data are converted exactly to float64 or complex128 (a sparse matrix keeps its stored entries, repeated ones
included, which are then summed in binary64; LIL and DOK through their own astype; DIA without reading its padding). The
registry test runs each of them on single-precision data with a repeated entry against the binary64 copy; it fails when
a public function of those modules is neither listed nor exempted with a reason, and when a module of the package is
neither listed nor exempted with a reason (wrappers of certify(), the S5 corroboration, readers of producer formats that
produce float64 arrays, modules without a stored-matrix input). The low-level kernels (`OperatorAdapter` products,
`enclosure.residuals`, the inertia factorizations, `ldl_residual`, `sectors.audit_sector`, the Ozaki splits) and the
caller-supplied products of `rayleigh_interval`, `kato_temple` and `residuals(products=...)` refuse floating data other
than binary64 with `PrecisionError`, so an entry point that misses its conversion fails instead of certifying. The S1
comparator `construct.matrix_difference`, `construct.hermitian_part`, `basis.sector_matrix` and
`invariants.matrix_invariants` (which carries a rounding bound) convert as well; inside certify() they receive converted
matrices already.

File readers (`io.formats.read_operator` and `read_basis`, `io.bundle`, `claim_from_file`, `claim_from_npz`,
`claim_from_bundle`, `claim_from_quspin`) return a single-precision sparse matrix as read (a COO keeps its repeated
entries), and certify() converts it; `source_sha256` in the record is that of the arrays as read. Writers
(`write_operator`, `write_bundle`, `export_npz`) store the exact binary64 copy of a single-precision sparse matrix,
summed in binary64, so the files of a float32 matrix and of its float64 copy certify alike; such a matrix therefore
reads back as float64 (a round trip changes its dtype, and an .npz file is larger). The Matrix Market writers write 17
significant digits (`mmwrite(..., precision=17)`), so that binary64 values round-trip on every scipy version (below 1.12
scipy writes 16 by default); a narrower dense array (vectors, eigenvalues, a dense operator or isometry) is written
there as its exact binary64 copy and reads back as that copy (scipy 1.17's writer stops the process on float32 data and
writes an unreadable file for complex64 data at this precision); the other formats keep a dense array's dtype. The text
of such files changes, their values do not. Claimed vectors on disk (numpy memmap, HDF5 dataset) are converted block by
block; any other matrix on disk is read into memory, an HDF5 dataset in binary64 as well (it is then a stored matrix,
like a float32 dataset). An operator given as a list of lists or as an array-like (an object with `__array__` and no
`matvec`) is a stored matrix whatever its dtype; in 3.x it failed as a black box without a `matvec`.
`compose.certify_low_energy` converts each sector's isometry, block and vectors once, before the U(1) restriction and
the decomposition audit, and lists the conversions in its record. A binary64 sparse matrix in the other byte order is
copied to native order.

Record and hashes. `arithmetic.input_widening` of the certificate (`input_widening` of the standalone records and of the
composition) lists, only when there are any, the inputs that were in a narrower floating format, with the caller's dtype
and `source_sha256` (array_sha256 of the input as given; per stored array for a sparse matrix). Hashes are of the
binary64 data: `matrix_sha256` widens before it sums repeated entries (it differs from the 3.x hash only for a
single-precision matrix whose repeated entries sum differently in single precision), and `eigenvectors_sha256` of
single-precision vectors is that of the converted array, which the arrays sidecar stores.

Integers and other types. Integer inputs were not affected, and their certificates and records are those of 3.x, bodies
and hashes (15 integer cases against cd0ca97: operators dense, sparse and with the plan, vectors, claimed eigenvalues,
observables, symmetry diagonals, isometries, stored eigenvectors, variational, moments, non-Hermitian, spectrum and
inertia records; the generalized record differs only in its hash over the run time, as between any two runs), except the
integer matrices with absolute row sums beyond 2^63 named above: integers are checked, used as given, converted only
where they are multiplied, and not recorded. Integer vectors, which 3.x refused with a numpy casting error, now certify;
boolean inputs (a TypeError in 3.x) are converted exactly. Values that binary64 cannot hold are refused with
`PrecisionError` (a `ValueError`, exported by the package) under one rule wherever they enter, the entry points of the
exact layer (`checker.exact_eigenpair_radii`, `exact_recheck_lowest`, `exact_sectors.exact_recheck_sector_lowest`)
included: a floating format wider than binary64 (an x87 or IEEE 128-bit long double; the decision is
`precision.classify_format`, tested for both on every platform), an integer beyond 2^53 that is not a binary64 number
(also in a list next to floats, where numpy would round it: Python integers, numpy integer scalars and integer arrays
alike; a Python integer beyond 2^64 or of 2^1024 or more), an object value v with float(v) != v (a Fraction 1/3, a
Decimal 0.1; exact object values are accepted, a change for `variational.state_from_dense`, which rounded them in 3.x;
NaN goes on to the finite check, as in a float array), non-numeric data, and an array-like whose conversion raises. A
complex array of the claimed eigenvalues of a Hermitian claim is converted as in 3.x (the real part, with numpy's
ComplexWarning). `claim_from_file` checks the eigenvalue file before it converts it. Every `certify-ed` command prints
'<command>: refused: ...' and exits with 2.

Memory. A narrower dense input now costs the caller's array plus its binary64 copy (1.5 times a float64 input for
float32); `cost.plan_for_claim` adds the caller's narrower arrays to its peaks. The copies are made before the memory
gates, so a claim that does not fit raises MemoryError instead of giving NOT_CHECKED (BUDGET_MEMORY). Array-likes are
materialized. Integer operators are used as given and converted where they are multiplied, as in 3.x; the exactness
check of an integer CSR, CSC or BSR matrix reads its nnz stored values only, so a value left past them is not refused (a
test for each of the three formats, with the first and the last stored value as negative controls). Upper moduli that a
caller passes to `OperatorAdapter.abs_matvec` must be binary64, like its vectors.

float64 and complex128 inputs give the same certificates and records as before, bit for bit (bodies without environment
and times, and arrays, on 14 models and records against cd0ca97, and claims with a complex eigenvalue array; hashes over
the run time differ between any two runs), also for vectors, observables and symmetry diagonals given as lists or
array-likes and for the sector composition; array-like observables, which 3.x refused, now certify. The computational
source hash changes (new module `precision.py` and the modules changed here). Tests: `tests/test_single_precision.py`,
`tests/test_precision_registry.py`. Not exercised here: the GPU products of D-05 (no device), which receive the
converted matrix, and a Linux run.

### The caller's matrices stay as given

`certify()` and `certify_variational()` now work on private copies of the CSR matrices they are given whose indices are
unsorted or repeated: for `certify()` the operator, the sector isometry, and the matrices in `Options.observables` and
`Options.approximate_symmetries`, on every device and again for the CPU recompute of a failed GPU run. A matrix given in
several places (the operator also as an observable, say) is copied once, so that a step working through one place sees
what the others did, as on the caller's matrix. Two matrix objects that share their arrays (the operator `D` and
`sp.csr_matrix(D)` as an observable) count as two matrices and get two copies; their first certificate can differ from
the one before this change in values and rounding bounds (the bounds were slightly larger in the cases tried; statuses
equal) and equals the certificate for two independent matrices. An entry of `Options.approximate_symmetries` that cannot
be read by index as (S, lattice) is passed on as given. The copies keep the index dtypes of the caller's matrices.
Before, the first run summed their duplicates and sorted their indices in place, through arrays shared with the caller's
matrices: the absolute value of the adapter, the S1 comparison, the sector audit, the S4 norm bounds and the label
records all did so. Without a model, or with a sector isometry, observables or approximate symmetries, a second call
with the same input then saw different matrices and could give a different certificate (or record); otherwise the
statuses and bounds were equal (S1's comparison record and the plan could differ), but the caller's operator was changed
and could be left with more stored values than `nnz`. Apart from the shared arrays above, the first certificate is
unchanged, since the copies go through the same steps as the caller's matrices did before, with two exceptions: the
no-underflow scan below, and, with `device="gpu"`, the certificate after a failed device run, which now equals the CPU
certificate (the recompute saw matrices of `Options` that the failed run had summed, and a matrix given both as the
operator and elsewhere had two copies). The second certificate is now equal to the first
(`tests/test_input_unchanged.py`, `tests/test_gpu.py`). Other entry points that take a matrix, for example
`OperatorAdapter`, `enclose_spectrum` and `certified_moments`, still sum it in place.

The copies cost one more matrix in memory each, for such inputs only, also for matrices of statements that are
skipped. They are made before the budget and memory gates, and the plan does not count them, so an operator that fits
in memory once but not twice now raises `MemoryError` from `certify()`. They replace the copy that
`certify(device="gpu")` made for the operator alone.

The no-underflow scan of a stored operator (`certify` and `certify_variational`) reads only its `nnz` stored values.
Before, values left past `nnz` could be scanned (left by a summation through another matrix object that shares the
arrays, inside `certify()` or before the call), and an operator whose stored values are normal numbers could get a
spurious UNDERFLOW_RISK; inside `certify()` this depended on whether the caller had read the matrix's
`has_canonical_format` before the call.

The computational source hash changes (`certify.py`, `variational.py`).

### Upper moduli in `fp`

`fp.modulus_up` returns, entrywise, an upper bound m ≥ |z| on the modulus of a real or complex array, formed from
correctly rounded products, sums and square roots with exact scaling by powers of two, at most a relative 12u above |z|
for normal values (`docs/proofs/fp_modulus-up.md`, Lemma M); `fp.abs_up` does the same for a dense array or a scipy
sparse matrix. No bound uses them in this change, and no bound, status rule or record changes. `modulus_up` returns real
floating input as `np.abs` gives it, integer and boolean input as float64 (one step up from 2⁵³), a float for every
scalar, and raises TypeError for a scipy sparse matrix, an object array and a complex type wider than complex128; it
works in passes of at most 2¹⁸ entries without copying its input. Known issues: the moduli of float16, float32 and
complex64 data are upper bounds, but a bound whose products are formed in single precision is not covered (the corollary
counts binary64 roundings; the entry points convert such input to binary64, see 'Single-precision inputs' above);
`abs_up`, like `abs`, sums the duplicate entries of a sparse matrix in place, so a product formed with the matrix before
the call is not covered by its moduli (direct calls of `commutator_bound`, `audit_sector`, `compress_operator` and
`certify_observable` on a matrix with duplicates); the floating-point items that the lemma assumes beyond those
`probe_float_environment` checks are tested with the test suite, not probed at run time. The derivation is new and
unverified: Sarang's check (D-12) is pending (independent re-derivation: verdict AGREES on the lemma, GAP on its
corollary, resolved; `docs/proofs/verify_modulus-up.py`, `tests/test_modulus_up.py`). The computational source hash
changes (`fp.py`).

### Non-finite bounds give no statement

A bound of the floating-point accounting that is not finite now gives no statement. Such a bound arises when a
product, a sum or a modulus overflows, for operator entries near the largest float (∞, and nan where an ∞ meets a zero
or another ∞), for real and complex data alike. S2 and S3 are INCONCLUSIVE with the new reason code `NONFINITE_BOUND`
(listed in `scope.REASON_CODES`, docs/SCOPE_MATRIX.md and the published schemas) and say that the radius is not finite
(the S2 summary used to say that an eigenvalue lies within it); S2 makes no accuracy statement, with the same code,
when the sector lift shift is not finite; the residual bounds and radii of the certificate are stored as ∞ instead of
nan. A nonzero claimed vector whose norm binary64 does not resolve (its squared norm underflows or overflows, as for
entries of about 10⁻¹⁷⁰ or 10¹⁷⁰) gets the new reason code `UNRESOLVED_VECTOR_NORM` instead, with the advice to
normalise the vectors (S2 and S3 were INCONCLUSIVE without a code). `inertia_dense` and `inertia_sparse` give no count
when their backward error is not finite (β = ∞, `conclusive` false; before, β could be ∞ or nan with a count, and
`certify(kind="lowest")` could then raise ValueError in the factorization of S3); the guard of `inertia_sparse` is
defensive (no input of the test suite reaches it; a test reaches it through a stand-in residual bound). A sector audit
with a non-finite defect does not pass (inf ≤ tol · inf passed before): S1 on a sector isometry is INCONCLUSIVE, and
the sector composition is INCONCLUSIVE with a summary that says the audit bounds are not finite. The variational
records (V1 to V3) and the moment records (M1 to M3) are INCONCLUSIVE with `NONFINITE_BOUND` when a bound of their
accounting is not finite (before, V1 and V2 could be CERTIFIED with a nan bound and M1 was CERTIFIED unconditionally),
and a sum-rule verdict on a nan endpoint is INCONCLUSIVE instead of CERTIFIED or FAILED. A non-finite observable bound
is reported as ∞ instead of nan. Records of real data change here: the label record of a cyclic symmetry gains the keys
`separation` (the floating-point root separation it divides by), `separation_note` (it is not a certified lower bound)
and `separation_rigorous` (false). Known issue: the label record's `certified` flag still rests on the floating-point
separation (an open D-12 item). `tests/test_nonfinite_bounds.py` covers these cases; the correlation records, the
sweeps, the MPS contraction, the Ozaki prototype and the GPU path are not covered by a test. The computational source
hash changes.

### Upper bounds for complex moduli

The bounds now take every modulus of complex data from `fp.modulus_up` (`fp.abs_up` for matrices; "Upper moduli in `fp`"
above). Before, they used NumPy's complex absolute value and counted it as one to three roundings; on Windows (NumPy
2.4.4) its error reaches 2.47u, and no bound on it is established. This covers the residual bounds of S2 (on the CPU and
on the GPU, whose moduli are formed on the host by the same functions, so that they equal the CPU moduli by
construction; the device path was not run after the change), the Rayleigh intervals of the Kato-Temple step and of the
variational records, the norm bounds, the backward errors of S3, and the bounds of observables, moments, commutators,
sector audits, pencils, non-Hermitian operators, sweeps, MPS contractions and the Ozaki prototype. Two two-sided values,
the C-13 fidelity and the Frobenius invariant of the S1 evidence, are formed from the squares of the components instead.
No constant and no status rule changes. For complex128 data the bound values change in the last bits (each modulus lies
at or above the exact one and, for normal values, at most a relative 12u above it, where NumPy's could lie below it).
For complex64 data the moduli were float32 values before and are binary64 upper bounds now, so those bounds can move
either way by more than that (a bound formed in single precision is not covered; the entry points convert such data, see
'Single-precision inputs'). An operator with an entry within a relative 12u of the largest float now has an infinite
modulus, and the statements that rest on it state nothing, by the rules of the previous entry: S2, V1, V2, N1 and G2 are
INCONCLUSIVE where they were CERTIFIED, S3 and V3 are INCONCLUSIVE with `NONFINITE_BOUND`, and `commutator_bound` is ∞
where it was finite (`tests/test_nonfinite_moduli.py`); for such an entry the moment, observable and sweep-affinity
bounds and the backward error of `inertia_dense` overflow with NumPy's moduli as well. Real data, and complex data whose
entries all have a zero real or imaginary part, give the same bounds and records as before, except in the non-Hermitian
record: its arithmetic text names `fp.modulus_up` instead of the hypothesis H-mod, which the bounds no longer need (its
four assumption strings and its record hash change, for real data too), and the bounds of a real matrix with complex
eigenpairs, whose eigenvalues and vectors are complex data, rise in the last bits. Cost: a modulus pass takes about four
times as long as `np.abs` on large arrays and about 0.1 ms per call on small ones; the S1 invariants take each modulus
once and cost what they cost before (Heisenberg chain, N = 20: 7.3 s against 8.9 s at commit cdf6ebb; 8.9 s and 9.0 s
against 11.7 s and 10.4 s at cd0ca97, the base of this stack, on a busier host); matrix-free operators pay the moduli in
every |H||x| application. The derivation is new and unverified: Sarang's check (D-12) is pending
(`docs/proofs/fp_modulus-up.md`, sections 3 and 4, and the correction notes of 13 derivation documents;
`tests/test_abs_audit.py`, which also fails on stale lines of its list, and the dynamic pin of
`tests/test_modulus_up.py`; `docs/proofs/verify_thm1_fp-accounting.py`). Known issue: the label audit of a cyclic
symmetry divides by a floating-point root separation and its record's `certified` flag rests on it (an open D-12 item;
the record says so, previous entry). The computational source hash changes.

### Housekeeping after joining the histories

No bound, status rule or record field changes, and the computational source hash stays the same. CI installs ruff
0.17.1 and runs pytest with one BLAS thread (the recorded radii of `tests/test_whole_spectrum.py` are compared
exactly); the forged-record test names `SchemaValidationError`, which ruff's rule B017 asks for; the nightly mutation
job passes its inputs through the environment. `CITATION.cff` describes the Zenodo DOI without the open question of
which DOI is the concept record. An author's e-mail address is removed from one arXiv comment of
`experiments/survey/survey_sample.json` (noted in `RESULT_practice_survey.md`). `docs/symveig_audit.md` keeps its
technical findings, `docs/applications.md` states the current reading of the S = 1 discrepancy, and
`legacy/README.md` says how `legacy/v1` relates to the tag `v1.0.0`.

### GPU products (roadmap D-05; branch feature/gpu-d05, first named track/gpu-d05)

With `device` left at `"cpu"` no bound, status rule or record field of `certify()` changes. With `device="gpu"`, S2's
evidence gains a `device` record and every other number of the certificate is the same bit for bit. The computational
source hash changes (new module `gpu.py`, the device hook of `OperatorAdapter`, the products of `enclosure.residuals` and
of the Kato-Temple step taken through `OperatorAdapter.both_products`).

- **GPU products for S2 (`certify_ed/gpu.py`, `Options(device="gpu")`, default `"cpu"`).** For a stored sparse operator,
  H X and |H| |X| are formed on a CUDA device by kernels that perform scipy's CSR operations in scipy's order: each row
  sum starts from +0.0 and adds the products of the stored row in index order, every product and every sum rounded
  once to nearest (`__dmul_rn`, `__dadd_rn`, NVRTC option `--fmad=false`), complex products as
  (ar xr - ai xi) + i (ar xi + ai xr). One pass over the matrix forms both products for up to eight vectors, in numpy's
  C layout. Under binary64 round to nearest with gradual underflow these are the operations of the CPU path, so the
  products, and every bound built from them, are bit for bit those of the CPU path; no new rounding analysis enters.
  Two guards: `gpu.probe()` (once per device: +, - and * against numpy on three magnitude ranges, gradual underflow on
  fixed cases, the kernels against scipy on random real and complex CSR matrices with empty rows, unsorted indices,
  subnormal and large entries) and a spot check of 64 rows of every product against scipy. Without CuPy or a device,
  after a failed probe or spot check or any other CuPy or CUDA error, for a dense or matrix-free operator, an
  unsupported dtype or too little device memory, the CPU forms the products and S2's evidence `device` says why
  (`requested`, `used`, `reason` or `fallback`; for a device in use its name, compute capability, CUDA and CuPy
  versions, the number of products, the spot-checked rows and the seconds; the certificate schema documents the
  field). `used` is true only when products of the certificate came from the device. When the device fails after
  products were taken from it (`gpu_products_at_fallback` > 0), those products are not trusted and `certify()` computes
  the whole certificate again with the CPU products (`recomputed_on_cpu`). The device copy of H is freed when
  `certify()` returns. The module needs NVRTC but not the CUDA headers (it launches no CuPy elementwise kernel).
  `Options.device` must be `"cpu"` or `"gpu"`; any other value, `None` included, makes `certify()` raise
  `ValueError`. A recorded device error is put on one line and cut to 300 characters. A CSR operator with duplicate
  entries that is certified with its model is served by the CPU (`|A| does not share the pattern of A`), because the
  S1 comparison compacts it in place before the products are formed.
  Device memory is bounded by `CERTIFY_ED_GPU_MEM_GB` or `GPU_MEM_GB` (default 8 GiB) and by 90 % of the free device
  memory; column blocks change no bit. `CERTIFY_ED_GPU_DUTY` or `GPU_DUTY` in (0, 1) sets a duty cycle for shared
  machines.
- `enclosure.OperatorAdapter(op, norm_bound=None, device=None)` and `OperatorAdapter.both_products(X)` (the pair
  `(matvec(X), abs_matvec(X))`, one pass on the GPU). On the CPU `both_products` calls `matvec` and `abs_matvec` as
  before, so the CPU path is unchanged.
- `tests/test_gpu.py` (the tests that need a CUDA device skip without one; the fallback tests, and the tests of the
  residual and Kato-Temple wiring and of the device record with a stand-in device that forms the scipy products, run
  everywhere),
  `experiments/d05_fp_probe_gpu.py` (the binary64 arithmetic of a device: round to nearest of +, -, *, /, sqrt and fma,
  directed rounding, subnormals, contraction of a*b + c under NVRTC's default flags) and `experiments/d05_gpu_bench.py`
  (products and residual bounds against the CPU on Heisenberg chains, with a breakdown of the CPU time of
  `residuals`, and `certify()` with S2 on both devices). Measurements on an NVIDIA H100 80GB HBM3 in `RESULT_D-05.md`.

### Scale track (roadmap D-03, D-04, H-04, groundwork of G-10; track/scale)

No bound, status rule or record field of `certify()` changes; the computational source hash changes (new modules
`ozaki.py` and `parallel.py`, a keyword of `enclosure.enclose`, the sector loop of `compose.certify_low_energy`
moved into a function).

- **D-03 error-free residual and Gram products (prototype, UNVERIFIED, not the default).** `certify_ed/ozaki.py`
  splits operands into slices of beta bits that share one power of two per row (H) or column (X, mu), so that every
  slice product on BLAS or SciPy's sparse kernels is exact (Lemma P: L 2^(2 beta) <= 2^53 and units >= -1074), sums
  the slice products with Sum2 (Ogita, Rump and Oishi 2005) and bounds the remainders: `ozaki_gram` /
  `ozaki_column_dots` (X^H Y with an entrywise bound), `ozaki_residual` (H X - X diag(mu) with an entrywise bound
  without the gamma_w (|H||X| + |X||mu|) term), `residual_report` for the new keyword
  `enclosure.enclose(..., residual_report=None)` (None, the value of every caller, keeps the 3.2 path bit for bit).
  Refused with `NotApplicable`: non-finite data, entries >= 2^480, more than 2^26 terms, spec and black-box
  operators; slices are formed only with units >= 2^-537 (every unit pair then sums to >= -1074), smaller parts are
  bounded as remainder. Draft derivation `docs/proofs/D-03_ozaki_draft.md` (Theorems G and R, status UNVERIFIED,
  independent re-derivation not done, Sarang's check pending); `docs/proofs/verify_D-03.py`: 1000 random + 14
  adversarial cases against exact rationals (python-flint), 4 negative controls detected, 0 violations
  (`docs/proofs/runs/verify_D-03.txt`); `tests/test_ozaki.py` (14). Measurement on the eleven E2 points of the 3.2.1
  run (`experiments/d03_ozaki_tightness.py`, record `docs/proofs/runs/D-03_tightness.json`; the default radii
  reproduce the run's records): Theorem-1 radii 0.32 to 0.62 of the default, largest enclosure radius 0.24 to 0.64,
  the reported S2 radius 0.81 to 0.93 on U(1) blocks (the S1 model shift dominates) and 0.37 to 0.54 on k = 0 blocks;
  the residual costs 9.5 to 29 times the default (at most 2.24 s at dimension 112720); the Gram defect eta is
  unchanged and the D-03 Gram is 0.76 to 1.66 times as fast as `fp.exact_gram` on the E2 clusters (g = 4 to 7),
  up to 24.6 times on 32 random columns.
- **D-04 parallel certification with a memory budget per worker.** `certify_ed/parallel.py`: `run_tasks` runs
  independent tasks on W worker processes (default W = 1: the plain loop in this process, no worker). Budget per
  worker = (process cap - memory held) x 0.9 / W, the cap read from the job object (Windows) or RLIMIT_AS (POSIX),
  else the available physical memory; each worker caps itself at it, so the workers together stay inside the cap
  (the 3.2.0 E4 crash). A task predicted above the budget, failing in a worker (exception, dead worker) or showing a
  resource limit (NOT_CHECKED for memory or time, an inertia `resource_failure`) runs afterwards in this process, so
  its result is the serial one. Workers exchange pickles through files (no pipes; `multiprocessing` pools need named
  pipes, which the restricted environment of the development machine refuses). `certify_many(jobs, workers)` for (claim, spec,
  options) triples; `compose.certify_low_energy(..., workers=1, cap_mb=None)` for sectors (the parallel path follows
  the serial loop's choice between a sector's U(1) restriction and the full matrix; the scheduling record is under
  `parallel`). `tests/test_parallel.py`: every status and every recorded number (timings excluded) identical for 1,
  2 and 4 workers; deferral, retry after MemoryError, a dying worker, the worker cap.
- **H-04 performance regression tests and G-10 groundwork.** `experiments/perf_points.py` measures E2 rows exactly as
  `experiments/e2_cost.py` does (producer and certifier processes, 4.5 GiB cap, one thread, seeds of the runs) or
  copies them from a run record (`--from-run`); `tests/fixtures/perf_baseline.json` holds the four small rows of
  the 3.2.1 run (dimensions 810 to 3432). `tests/test_perf_regression.py` is skipped unless pytest runs with `--perf`
  (new option in `tests/conftest.py`) or `CERTIFY_ED_PERF=1`: statuses equal to the baseline's, time <= 2 x baseline
  + 2 s, peak memory <= 1.5 x baseline + 64 MB (factors settable by environment variables).
  `experiments/bench_table.py` tabulates certify() time, S3 and S8 time and peak memory of any measurement against
  a run's E2 record (Markdown, CSV, JSON).

### New statements track (roadmap E-02, N-13; track/non-hermitian)

A new claim kind for non-Hermitian eigenproblems. Everything below is UNVERIFIED (D-12, Sarang's check pending) and
additive: no existing bound, status rule or record field changes. The source hash changes (new module, extended
enumerations).

- `certify_ed/nonhermitian.py` (`certify_nonhermitian`, `format_nonhermitian`; record schema
  `certify-ed/nonhermitian/1.0`) and the command `certify-ed nonhermitian`. Four statements for claimed eigenpairs
  `(mu_i, v_i)` of a dense real or complex `A` of dimension at most 2000: N1 backward errors (each `mu_i` is an exact
  eigenvalue of some `A + E_i` with `||E_i||_2 <= eps_i`); N2 a bound on `||V^-1||_2` from a computed inverse `W`
  (Neumann); N3 Bauer-Fike inclusion (every eigenvalue of `A` lies in a disc of radius `rho = beta ||R||_F` about some
  `mu_i`; `A` need not be diagonalizable); N4 counting (each connected group of discs holds exactly as many eigenvalues
  as centres, with algebraic multiplicity). N2 to N4 need the whole spectrum; the record says NOT_CHECKED with
  NEEDS_WHOLE_SPECTRUM for a subset and NEEDS_DEPENDENCY when N2 fails.
- New reason code `ILL_CONDITIONED_EIGENVECTORS` (N2 inconclusive: `||I - W V||` not provably below 1); its next action
  in `diagnostics.py`; the record schema and the certificate schema list the new kind and reason code
  (`schema/*.json` regenerated). `docs/SCOPE_MATRIX.md` has the new rows.
- `docs/proofs/E-02_non_hermitian.md` (derivation, floating-point accounting with an opt-out for subnormal inputs),
  `docs/proofs/verify_E-02.py` with the recorded run `docs/proofs/runs/verify_E-02.txt` (2696 random and 547
  adversarial cases in exact arithmetic, 0 violations; negative controls detected in 36/36, 36/36, 34/36 and 200/200
  cases), an independent second derivation and adversarial script under `docs/proofs/independent/` (verdict
  AGREES-WITH-REMARKS, `docs/proofs/VERDICTS.md`), the cases in `corpus/nonhermitian/` (11 cases, 0 unsound, 0 alarms)
  and `tests/test_nonhermitian.py` (14 tests).
- The exact parts that N1 to N3 use (the bound of Lemma 1, Theorem 2 with its residual form, the bound of Lemma 3, and
  Theorem 4 (b) combined with Lemma 3) are formalized in Lean 4 with Mathlib in the separate repository
  certify-ed-project (`reverification_lean/NONHERMITIAN.md`, now `docs/lean/NONHERMITIAN.md` of this repository). The counting statement of N4 and the floating-point
  accounting are not.

### Lean product track: external verifiers (roadmap item N-04; track/external-verifier)

- `certify-ed verify CERT.json` checks the integrity of a saved certificate (body hash, array hashes, scope block, as
  `Certificate.load` does) and prints its summary. With `--external CMD`, or the environment variable
  `CERTIFY_ED_EXTERNAL_VERIFIER`, it then runs a configured external verifier on the certificate, for example the exact
  checker `cecheck` of the Lean re-verification through its driver `Verify/cecheck_bundle.py` in certify-ed-project (now `lean/Verify/`), or
  the C verifier of `verifier/`. The command is a template with the placeholders `{certificate}`, `{sidecar}`,
  `{bundle}`, `{operator}` and `{report}`; it runs without a shell. Exit code 0 of the program is recorded as PASS, 1
  as FAIL, any other as ERROR, and a run past `--timeout` as TIMEOUT.
- The verdict goes to a separate external-verification record (schema `certify-ed/external-verification/1`, default
  `CERT.external.json`): the hashes and statuses of the certificate as read, the command and the SHA-256 of the
  program, exit code, run time, the tails of its output, and the verifier's JSON report (hashed; its `verdict`, `level`
  and `statements` copied as the verifier's words). The certificate file is not modified and no status, bound or count
  depends on the verdict; the record states this in `status_effect`. The command exits with 0 when the integrity
  check passed and the verifier (if any) reported PASS, 1 when the integrity check failed (no verifier is then run),
  and 2 when the verifier reported FAIL, ERROR or TIMEOUT.
- API: `certify_ed.external.verify_external(certificate, command, bundle=, operator=, sidecar=, report=, record=,
  timeout=)`, also importable from `certify_ed.api`. `certify_ed/external.py` is listed in
  `provenance.NON_COMPUTATIONAL`, and no module that computes a status imports it (test
  `test_no_status_code_imports_the_external_module`), so the computational source hash is unchanged.
- Tests: `tests/test_external_verifier.py` (stub verifiers for PASS, FAIL, ERROR, TIMEOUT and a missing program; the
  certificate bytes and statuses unchanged after every outcome; an unset placeholder stops the run before the program
  starts; CLI exit codes and the environment variable; no verifier runs on a certificate that fails its integrity
  check). `docs/api.md` regenerated (`docs/make_api.py` gains the section on external verifiers).

### Applications track (roadmap N-11, N-12; track/applications)

Applications N-11 and N-12 of the development roadmap. New code only; no existing bound, status rule or record field
changes. The source hash changes because files under `certify_ed/apps/` are added or extended.

- `certify_ed/apps/pyrochlore/nine_coupling.py`: the nine-coupling spin-1/2 crown of Kránitz and Penc (Appendix D of
  arXiv:2601.07800): Table V orbits, printed couplings with both readings of the digit in parentheses, exact matching
  conditions, D3d automorphisms, the abelian subgroup {E, C2', i, σd} with the spin flip, a block-by-block exact count
  (`count_below_by_block`, the construction of `exact_sectors.exact_count_below` with one block in memory at a time), a
  floating-point-layer count on the exactly representable block congruence (`count_below_by_block_float`, via
  `inertia.inertia_dense`), an integer Rayleigh-quotient upper bound, `certify_nine(layer=...)`, and the exact count of
  selected blocks at a given shift (`certify_blocks`) with `combine_blocks`, which joins disjoint block records into
  the exact lower bound.
- `experiments/n11_crown_nine.py`: record of the certificate at the printed couplings (`--float-layer` for the
  floating-point layer, `--compare` for the other readings, `--blocks` and `--combine` for an exact re-check split
  over processes or machines).
- `certify_ed/apps/aklt/estimate.py` (`plan_eta`): the block sizes, memory and time of `eta.certify_eta` without forming
  any matrix; `lattices.honeycomb_method3()`: the union of two Method III terms of Pomata and Wei for the honeycomb.
- `experiments/n12_honeycomb_estimate.py`: the honeycomb estimate (largest bordered block 1,293,292 rows), calibrated on
  the E10 blocks of the 3.2.1 run, and the exact product of the printed gap constants.
- `tests/test_apps_applications_next.py`: 8 tests (orbit table against the reduced model, embedding, D3d, block counts
  against `exact_count_below` with a non-symmetry as negative control, single-block counts joined by `combine_blocks`
  with a missing block, a repeated block, a changed shift and a nonzero count as negative controls, Rayleigh upper
  bound, plan against the E10 block sizes, honeycomb geometry).

## [3.2.3] — 2026-11-12 (branch `fix/3.2.3` on `v3.2.2`; found by the 3.2.2 evidence run `run_full_20261112`)

Two corrections found by the 3.2.2 evidence run; no bound changes. The comparison of that run with the 3.2.1 run
(`run_full_20261105`) has no MISMATCH leaf in E0, E1, E3, E5, E7, E8, E9 and E10; its E2 and E4 leaves and the abort of E6
are explained in `RESULT_evidence_run_3.2.2.md`.

- **S1 and the memory gate of 3.2.2.** Without `budget_mb`, 3.2.2 let the memory left under the process memory cap gate
  every statement, S1 included. The S1 estimate (dimension x terms x 2 entries of 32 bytes, made for a budget) is a coarse
  upper bound: under the experiments' job-object cap of 4608 MB the 3.2.2 run refused S1 (NOT_CHECKED, `BUDGET_MEMORY`) for
  the E2 block chain N=22, k = 0, at an estimate of 16896 MB, where the 3.2.1 run had certified S1 at a peak of 2989 MB for
  the whole certifier process, and for the second case of E4 W4 (J1-J2 chain, J2 = 0.5, N=20, k = 0, estimate
  7680 MB; the first case, J2 = 0, estimates 3840 MB and was certified in both runs). A reference build that does not
  fit ends in a caught `MemoryError` and an S1 NOT_CHECKED, as in 3.2.1 (E2 chain N=24, k = 0). `scope.Budget.memory_ok`
  gains `use_cap` and `certify._gate` passes it; S1 is gated by an explicit budget only, and the factorizations and the
  other statements keep the gate of 3.2.2. Test: `tests/test_memory_gate.py::test_cap_does_not_gate_s1_but_still_gates_factorizations`.
- **E6 copies next to the repository.** `experiments/e6_mutation.py` made its copies of the repository in the system temp
  directory. In the restricted environment of the 3.2.2 run git could not create a repository there, so
  `tests/test_review_fixes.py::test_imp05_source_fingerprint_ignores_lost_executable_bit` failed in the copy, the
  unmutated baseline counted as killed and E6 stopped before its first mutant. The copies are now made next to the
  repository (or in `CERTIFY_ED_E6_COPY_DIR`), and the test skips when git can work neither in its temp directory nor next to
  the package, as it already did without git.

## [3.2.2] — 2026-11-11 (branch `track/hardening-3.2.2` on `dev/3.2` fa86456; the tag `v3.2.2` and its evidence run follow the merge)

Hardening release (Wave 5 of the development roadmap, items N-01, N-02, N-03, H-02). No bound changes inside the range of the
floating-point guards: on ten fixture certificates (dense and sparse operators, real and complex data, spec and matrix-free
operators, kinds lowest/window/all, a vectorless window, S6, S7, an UNDERFLOW_RISK input) every statement record of this
version equals that of fa86456 except for the timings. What changes is that the code now refuses to bound where the proof does
not reach, and that without a memory budget the process memory cap decides whether a factorization is started. No new
mathematics: the range is the hypothesis already stated in `docs/proofs/thm1_fp-accounting.md` and `D-01_sparse-residual.md`.

- **Range of the floating-point guards (N-01, finding L-3 of the Lean formalization).** The guards of Theorem 1 (Lambda4 for the
  norms, Lambda5 for |A||x|, Lambda6 for E) and the residual of D-01 (hypothesis H3) are established for vectors of n <= 2^26
  entries and row widths w <= 2^26 only; Lambda4 with N = 2n fails from n = 84886744 on, and the sentence that extended Lambda6
  beyond 2^26 fails at w + 2 = floor(2^53 / 3) (erratum of 10 Nov 2026). New: `fp.GUARD_RANGE_MAX = 2**26`,
  `fp.GuardRangeError` (a `ValueError`), `fp.check_guard_range`, `fp.guard_range_findings`. `fp.norm2_upper` and `fp.norm2_lower`
  (and through them `fro_upper`), `enclosure._rounding_bound`, `enclosure.rayleigh_interval` and
  `ldl_residual.sparse_ldl_residual_bound` raise it above the range before computing anything. `certify()` checks the dimension
  and `adapter.width` before S2: above the range S2 and the statements resting on its enclosures or on the factorization bounds
  (S3 for kinds lowest/window/all, S4, S7, S8) are INCONCLUSIVE with the new reason code `FP_GUARD_RANGE` and a `guard_range`
  evidence record. A `GuardRangeError` from a norm of more than 2^26 entries inside that range (for example ||X_g||_F of a
  cluster with n g > 2^26) is reported the same way by S2 and by S1; an S4 cluster record then carries FP_GUARD_RANGE, and
  `inertia.inertia` reports no count (for example ||R||_F of a dense LDL^T with n > 8192), which makes S3 INCONCLUSIVE with
  FP_GUARD_RANGE and an S8 record INCONCLUSIVE. `scope.REASON_CODES`, `diagnostics.NEXT_ACTION_BY_CODE`, the schemas
  (re-rendered), `docs/SCOPE_MATRIX.md`, `docs/faq_inconclusive.md` and `docs/presets.md` name the code. Status changes against
  3.2.1: only above the range, where 3.2.1 returned bounds outside the range in which they are established. The largest
  dimension in the 3.2.1 run record is 1116229 (E9); whether a normed array of that run exceeded 2^26 entries is shown by the
  3.2.2 run (a FP_GUARD_RANGE status in the comparison). Tests: `tests/test_guard_range.py` (2^26 allowed, 2^26 + 1 refused with
  a zero-stride view and a fake adapter, no large allocation; certify() with the limit lowered; a spec whose row width bound is
  2^25 + 52 against one with 2^26 + 54).
- **Pinning tests for the E6 survivors (N-02).** `tests/test_e6_pins.py` pins the formula at each of the seven soundness-relevant
  survivors of the 3.2 mutation sample (`E6_inspection_worksheet_3.2.0.md`) and at the `checker.py:535` gap: the complex flag of
  `sectors.audit_sector`, the rounding term, kappa and the observable main term of `eigenspace.py`, the count of a complex block
  from its real embedding in `exact_sectors.py`, the fermion letter amplitudes in `localops.py`, the counted flag of
  `spectrum.merge_intervals` and the separation test of the witness count in `checker.certify_by_counts`. Each of the eight
  mutants of `experiments/e6_mutation.py` (same site indices as in the 3.2.0 and 3.2.1 runs) fails these tests;
  `experiments/e6_pin_check.py` repeats that check. The `localops.py:189` mutant changes no matrix: every builder (bitwise,
  `SpecOperator`, `MatrixFreeOperator`, `checker.exact_matrix`) masks amplitudes whose output digit is out of range, so the
  worksheet's reading 'the built matrix changes' does not hold; the test pins the contract of `shift_components` instead.
- **Memory gate from the process memory cap (N-03).** `cost.process_memory_cap()` reads the cap the evidence runs set (a Windows
  job object with a per-process or job memory limit, `RLIMIT_AS` on POSIX) and the memory in use; `cost.memory_limit_mb`,
  `cost.memory_cap_note`. Without `Options.budget_mb` the memory left under the cap takes the place of the budget in
  `scope.Budget.memory_ok` (statement gate), in `inertia.inertia` (dense and sparse factorizations, `cost.dense_inertia_mb` /
  `cost.sparse_inertia_mb`) and in `cost.recommend_inertia_method`. A factorization predicted above it is NOT_CHECKED with
  `BUDGET_MEMORY` before it starts, instead of a process that ends at the cap (3.2.0 run, E4 W4). With a budget, or without a cap,
  nothing changes. `certify()` now sets `BUDGET_MEMORY` explicitly on an S3 that is NOT_CHECKED because a factorization was not
  attempted or not completed (`scope.classify_not_checked` read the words 'memory budget' of such a summary as `BUDGET_TIME`;
  with a budget the statement gate refuses first, so this mattered only for a refusal inside the factorization). Status changes
  against 3.2.1: under a cap and without a budget, an S3 or S8 whose fill-model prediction exceeds the headroom is NOT_CHECKED
  although it may have fitted (the fill model is fitted to the Heisenberg chain); the 3.2.2 run's comparison shows any such
  leaf. Test: `tests/test_memory_gate.py` caps a child process (job object, 326 MB on the development machine) above its own
  usage and certifies a diagonal operator of dimension 10000 whose predicted sparse factorization (856 MB) exceeds the headroom.
- **Property-based tests (H-02).** `tests/test_properties_fp.py` compares fp.py and enclosure.py with exact rational arithmetic
  (`fractions.Fraction`) on inputs drawn by hypothesis (derandomized; magnitudes 2^-600 to 2^200, real and complex, dense and
  sparse): the gamma constants, the norm guards for every magnitude and with the 3u slack in the normal range, `exact_dot`,
  `fro_upper`, `inf_norm_upper`, the entrywise bound E, `upper`, `xnorm_lower`, the Theorem 1 radius against the exact residual
  norm, the Rayleigh interval and the cluster bound's eta. hypothesis is added to the `test` and `dev` extras; the module is
  skipped without it.

The source hash changes (computational files: `fp.py`, `enclosure.py`, `ldl_residual.py`, `inertia.py`, `certify.py`, `cost.py`,
`scope.py`, `diagnostics.py`, `__init__.py`, and `experiments/e6_pin_check.py`). The number of mutation sites of `fp.py`,
`enclosure.py`, `inertia.py` and `certify.py` grows, so the E6 sample of the 3.2.2 run differs from that of 3.2.1 although the
seed is the same.

Full test suite at ddf83e1 (Windows, Python 3.11.17, numpy 2.4.4, scipy 1.17.1, hypothesis 6.170.15; a machine shared with other work): 1210 passed, 16 skipped (no C compiler for
the verifier, the Linux .so probe, figure scripts without run records, the 12870 fixture, a check that needs a primary git
checkout), 0 failed, in 3 h 04 min. The eight mutants of N-02 were re-run against that head with `experiments/e6_pin_check.py`:
all killed.

## [3.2.1] — 2026-11-04 (tag `v3.2.1`; the version of the 3.2 evidence run)

One robustness fix found by the 3.2.0 evidence run; no bound, status rule or record field changes.

- `cost.residual_threads` (used by `ldl_residual.sparse_ldl_residual_bound`, the D-01 a-posteriori S3 residual) now
  returns 1 unless `CERTIFY_ED_RESIDUAL_THREADS=<n>` asks for more. In the 3.2.0 run, E4 W4 (J1-J2 square 4x4, J2 = 0.5,
  dimension 12870) under the experiment's 5 GiB job-object cap ended with an access violation inside SciPy's sparse
  product while eight row blocks of the residual were formed concurrently: an allocation failure in a worker thread
  under a committed-memory cap is not turned into `MemoryError`. With one thread the same case certifies S3 under the
  cap (gap >= 0.287); without the cap both settings certify. The bound is independent of the thread count (test
  `test_residual_bound_does_not_depend_on_the_default_thread_count`), so no certified quantity changes. The 3.2.0 run's
  other steps are unaffected; the 3.2.1 run repeats every step from this tag (D-44).

Post-tag, non-computational (the run's source hash `cb73dec3e4433a3b` is unchanged): `experiments/make_numbers.py` counts
an E2 row whose certifier hit the experiment's S3 limit and left its certificate without S3 as a certificate, not a crash
(macro `CnumSthreeTimedOut`; commit 44d318c5); the figure scripts read run labels from the manifests and
`docs/figures_run_3.2.1/` holds the record-driven figures of the 3.2.1 run; `RESULT_evidence_run_3.2.1.md` records the
run and its comparison with 3.1.0 (10 MISMATCH leaves, all documented changes) and with 3.2.0 (none).

Documentation corrections of 10 Nov 2026 (non-computational; source hash `cb73dec3e4433a3b` unchanged): two explanations in
`RESULT_evidence_run_3.2.1.md` (the E6 `copy.files` leaf reflects the number of tracked files, 215 at `v3.1.0` and 915 at
`v3.2.1`; the 64 leaves missing against the 3.2.0 run are 48 in E2 and 16 in E4); the non-empty-interval hypothesis of
Proposition 14 in `docs/theory.md` 9.1; an erratum to the closing condition of the guard Lambda6 in
`docs/proofs/thm1_fp-accounting.md`. The last two were found by the Lean formalization in the separate repository
certify-ed-project (`reverification_lean/COUNTING.md`, `CORE.md`; now in `docs/lean/` of this repository).

Further errata of 10 Nov 2026 from the same formalization (`reverification_lean/ENCLOSURE.md`, now `docs/lean/ENCLOSURE.md`; non-computational, source
hash unchanged): one inequality in the proof of Theorem U5 (`docs/proofs/int05_underflow.md`; conclusion unaffected), the
argument for part (b) of D-02 (a second-order margin, not a spare index; `docs/proofs/D-02_matrix-free.md`), the width of
`from_float` below the grid resolution (`docs/proofs/L17_dyadic_exp_ln.md`), and the index of the computed |H|_abs sums for
complex amplitudes in Lemma A1 (`docs/proofs/appA_model-shift.md`; covered by the code's width). No bound or status changes.

## [3.2.0] — 2026-11-04 (tag `v3.2.0` on `dev/3.2`; evidence run pending, see RESULT_integration_3.2.md)

`__version__` = 3.2.0 (3.2.0.dev0 during development: every entry below was written under that name). Summary of the release: the
D-12 backlog derived, machine-checked and independently re-derived (22 documents in `docs/proofs/`, verdicts in
`docs/proofs/VERDICTS.md`; Sarang's check pending for every document; two errors found by the independent check were corrected
by looser code); INT-05 closed; Theorem 2 with Kahan's constant rho; new statements C-03, C-04, C-06, C-07 (first part), C-12,
C-13 (API), C-16, E-03, E-04, E-05, E-07; the a-posteriori sparse S3 residual (D-01) and matrix-free S2 (D-02); the independent C
verifier (`verifier/`); thirteen figures (`docs/figures/`); the Waves 0-1 engineering of the roadmap. Statuses of 3.1.0 records
can change only by the documented bound changes (listed per track below) and the downgrades to INCONCLUSIVE under UNDERFLOW_RISK.

### Release integration (2026-11-03/04)

- Lead corrections after the independent check: `enclosure._rounding_bound` uses gamma_{w+3} for complex data (thm1 (S2) and
  D-02 Lemma M3 gaps; looser), `fp.norm2_lower` returns 0 when the squared norm overflows (int05 ERROR; looser),
  `sweep.certified_crossing_interval` adds the affinity defects at interior grid points (E-04 ERROR; looser),
  `exact_degeneracy._record` names the exact bracket (C-16 wording). Negative controls in `tests/test_independent_check.py`.
- Appendix A verification script run (386 random + 681 adversarial, 0 violations).
- Figures regenerated from the final tree: `fig_scope_vs_survey` (3.2 scope), `fig_rounding_accounting` (INT-05 terms),
  `fig_roadmap_progress` (status 2026-11-03: 43 of 78 items, 270 of 470 points).
- Release suite 1168 passed / 8 skipped; 16-mutant sample 10 killed, the two reachable survivors (exact_sectors weights branch,
  eigenspace guard) turned into tests, the other four equivalent or heuristic (RESULT_integration_3.2.md).
- Known limits carried into the release notes: C-07 continued-fraction bounds, C-13 CLI, C-16 sector path, `exact_bracket`
  dimension cap 600, D-01 out-of-core, verifier scope (no sector/complex/vectorless paths), E6 survivors of the 3.1.0 run
  awaiting verdicts.

### Development history (3.2.0.dev0)

The computational source changes (certify.py, fp.py, scope.py), so records made
with it are not comparable by `source_hash` with the 3.1.0 evidence run; no statement that 3.1.0 certified
becomes *more* certified here (soundness rule), and the only status changes are downgrades to INCONCLUSIVE
under the new reason code below.

Integration (2026-10-29): seven tracks of roadmap Waves 0 and 1 merged in the order known-issues, core-api,
format-packaging, producers, statements, whole-spectrum (its commits re-written to LF line endings, content
unchanged), viewer-corpus. Conflicts were additive except two: `certify.py`'s observable loop now applies the
Hermitian decomposition (C-05) to the parsed operators of C-15, and an inverted window is refused when the
Claim is built (C-14) rather than reported by S3 (C-02 test adjusted). `Certificate.load` validates the schema
before the hashes. Full suite on the merged branch: see `RESULT_integration_3.2.md`. Seventeen CLI commands:
env, certify, import, plan, statement, crossing, sweep, levelstats, extrapolate, view, presets, conform, spectrum,
list, test, reproduce, numbers.

### Variational and dynamics track (roadmap E-05, E-03, C-07 first part; track/variational)

Derivations in `docs/proofs/<id>_<name>.md` with verification scripts (D-59; independent re-derivation pending). The
three modules are additive (no existing bound changes); their results are standalone JSON records validated by the new
`schema/statement-record-1.0.schema.json` (`schema.validate_record`), whose status vocabulary adds
CERTIFIED_UNDER_HYPOTHESIS (a bound that rests on a named, declared hypothesis; the schema requires the hypothesis
list to be non-empty) and DECLARED (a producer value nothing rests on) to the certificate statuses, so that neither can
be read as CERTIFIED. New reason code DECLARED_ESTIMATE. `diagnostics.record_next_action` / `annotate_record` give every
non-certified record statement a `next_action`.

- **E-05 variational certificates for non-ED states.** `certify_ed.variational.certify_variational(operator_or_None, psi,
  spec=..., gap_hypothesis=..., gap_source=..., gap_certificate=..., declared=...)` and `certify-ed variational`: V1 the
  Rayleigh upper bound λ₁ ≤ θ (unconditional; the rounded quotient is enclosed through `enclosure.rayleigh_interval`,
  the model shift σ of a spec is added), V2 the Theorem 1 existence radius r at θ (+σ by Weyl), V3 Temple's lower bound
  λ₁ ≥ θ − r²/(ρ − θ) for θ < ρ ≤ λ₂ — CERTIFIED when ρ is read from a certificate of a 'lowest' claim on the same
  operator (S2 and S3 CERTIFIED: second smallest lower endpoint of the enclosures for k ≥ 2, S3's
  `next_eigenvalue_lower_bound` for k = 1), CERTIFIED_UNDER_HYPOTHESIS with the named hypothesis `declared_gap` (ρ and
  the producer's source text, repeated in the statement text) when ρ is declared, INCONCLUSIVE when θ_hi ≥ ρ — and V4
  producer-declared stochastic estimates (energy, variance from samples) with status DECLARED (DECLARED_ESTIMATE).
  Adapters: a dense vector; an MPS (`state_from_mps`: open-boundary tensors contracted left to right for dimension
  ≤ 2²², else SIZE_LIMIT) with the entrywise rounding bound γ_M|A⁽¹⁾|⋯|A⁽ᴸ⁾| of the contraction recorded as
  `contraction_error` and the Rayleigh-quotient difference between the source MPS and the contracted vector (Lemma 5)
  — the certified statements are about the contracted vector; a sparse CI vector (`state_from_sparse_ci`, exact scatter,
  duplicate indices refused). The no-underflow scan of the state and operator entries makes V1–V3 INCONCLUSIVE
  (UNDERFLOW_RISK); a norm too small to resolve gives NOT_CHECKED; matrix-free and non-Hermitian operators are refused.
  Mathematics: `docs/proofs/E-05_variational.md` (Rayleigh–Ritz, Theorem 1, Temple 1928 / Kato 1949 with the exact
  hypothesis, the floating-point accounting of θ and r, the MPS contraction bound). Verification: 666 random + 294
  adversarial cases, 0 violations; negative control (ρ = λ₃ > λ₂ declared on two-level mixtures): the bound is broken
  in 66/66 cases while the record stays CERTIFIED_UNDER_HYPOTHESIS and names the hypothesis. Tests
  `tests/test_variational.py` (12): 8-site ring, certificate-sourced ρ (k = 1 and k = 4), false gap hypothesis,
  orthogonal trial state, mismatched certificate, tiny norms (2⁻⁴⁵⁰ valid, 2⁻⁹⁵⁰ UNDERFLOW_RISK), complex data,
  non-Hermitian operator refused, MPS (Neel product state, random D = 3, size limit), sparse CI, declared estimates,
  record schema negative controls, CLI.
- **E-03 generalized eigenproblems A x = λ B x, B positive definite.** `certify_ed.generalized.certify_generalized(A, B,
  eigenvalues, eigenvectors, kind='lowest' | 'subset', lambda_min_B=..., lambda_min_B_source=...)`, dense path (n ≤ 4000):
  G1 a certified lower bound β_c on λ_min(B) from the inertia of B − βI (Bunch–Kaufman LDLᴴ with its backward error; a B
  whose inertia at shift 0 is negative is refused with UNSUPPORTED_MODEL), G2 per pair |λ − μ| ≤ ‖Ax − μBx‖₂/(√β_c ‖x‖_B)
  (Theorem 1 for the similar Hermitian matrix L⁻¹AL⁻ᴴ, ‖L⁻¹‖₂ = λ_min(B)^{-1/2}; residual and xᴴBx with the standard-model
  accounting), G3 completeness of a 'lowest' claim by the inertia of A − σB (Sylvester's law for the congruence
  L(C − σI)Lᴴ; collar b/β_c for the backward error b of the factorization plus the rounding of forming A − σB) with
  λ_{k+1} ≥ σ − b/β_c; overlapping enclosures and ν ≠ k are INCONCLUSIVE with the count recorded (`missing_at_least`). A
  declared λ_min(B) (used only when the inertia is inconclusive) is the named hypothesis `declared_lambda_min_B` and makes
  every statement CERTIFIED_UNDER_HYPOTHESIS. Mathematics: `docs/proofs/E-03_generalized.md`. Verification against
  mpmath ground truth (Cholesky, L⁻¹AL⁻ᴴ, exact B⁻¹-norm residuals): 303 random + 100 adversarial cases, 0 violations;
  negative control (declared λ_min(B) ten times the true one, inertia disabled): the radii undercut the exact residual in
  40/40 pairs while the record names the hypothesis. Tests `tests/test_generalized.py` (8): random pencils against
  `scipy.linalg.eigh(A, B)` (real and complex), perturbed vectors, indefinite B refused, wrong eigenvalue, swapped pairs, a
  missing eigenvalue detected, 'subset', the declared path with a false declaration, size limit, schema.
- **C-07 first part: certified moments and sum rules.** `certify_ed.moments.certified_moments(H, O, X, s, eta, K, level)` and
  `moments_from_certificate(cert, H, O, K, members)`: M1 certified intervals for m_k = tr(P O Hᵏ O)/g, k = 0..K, on a
  CERTIFIED S4 cluster — O applied once and H k times to the stored vectors (Hᵏ is never formed), the rounding of every
  application accounted in the Theorem 1 style (entrywise γ̇_w|H||z| bounds, propagated with ‖H‖; `fp.exact_dot` for the
  quadratic form), plus the projector term (2gs + gη)‖O‖²‖H‖ᵏ from the trace-norm bound on P − XXᴴ; M2 the zeroth-moment
  sum rule m₀ = tr(PO²)/g and M3 the f-sum rule m₁ − tr(PHPO²)/g = tr(P[O,[H,O]])/(2g), each certified by a second route
  (Theorem 6 on the explicitly formed O² and [O,[H,O]] with their product rounding bounds) and reported as CERTIFIED when
  the two intervals overlap and FAILED when they are disjoint (consistency statements, not new theorems; FAILED is
  added to the record status vocabulary for them); M4 the continued-fraction (Lanczos) bounds, NOT_IMPLEMENTED — the
  open part, described in `docs/theory.md` §9.4 and the scope matrix. Mathematics: `docs/proofs/C-07_moments.md`.
  Verification: 418 random + 192 adversarial cases, 0 violations (exact rationals for the rounding term, mpmath for the
  exact projector moments); negative controls: s understated 10⁴× breaks the enclosure in 132/138 moments, a random
  orthonormal X with s = η = 0 claimed in 138/138, a wrong level interval makes M3 FAILED in 40/40. Tests
  `tests/test_moments.py` (6): 8-site ring (ground state, triplet cluster, rotated basis), the negative controls, a
  complex operator with the exact-rational rounding check, underflow and input checks, schema.
### Independent verifier (roadmap B-01; track/verifier)

- **`verifier/`: a second-language re-check of certificates.** `verifier/src/cev.c` (C11, standard library
  only, no code shared with `certify_ed`) reads a certificate and its `.arrays.npz` sidecar with its own JSON,
  SHA-256, ZIP/`.npy` (stored and deflate) and exact decimal conversion, and re-checks: the body hash under
  `certify-ed-canonical-json-1` (shortest round-trip floats reproduced by exact integer arithmetic), every
  array hash, the specification hash, the operator hash (canonical CSR, or the Hermitian part), the per-pair
  enclosure records and exported arrays (model shift included), the recorded Theorem 2 / 2' cluster arithmetic,
  the disjointness of distinct groups, the S3 count-and-ordering logic for `lowest`, `window` and `all` claims
  from the stored enclosures and the recorded inertia counts, the no-underflow hypothesis, and whether a
  CERTIFIED S2/S3 is permitted by those data. With the producer's `operator.npz` and stored vectors
  (`Options(store_eigenvectors=True)`) it recomputes every Theorem-1 radius, the Gram defect and the cross term
  as directed-rounding intervals (`fesetround`; error-free transformations for the Gram matrix) and requires the
  recorded bounds to be at least the upper ends. Exit code 0/1/2, text report, JSON summary. `verifier/build.py`
  compiles it with gcc/clang/cc (`CEV_CC`). What it cannot check (S1, S4-S8, the factorizations behind the
  counts, complex data) is listed in `verifier/README.md`.
- Validation (`verifier/validation/run_validation.py`, standard library only; `RESULTS.md`): 63 bundles (corpus
  + conformance), 8 refused by certify-ed, 55 certificates verified with exit 0; the 18 S2/S3 statuses
  certify-ed set to FAILED are all reproduced from the stored data; 80 forged certificates (20 each: unsound
  radius with resealed hashes, altered array hash, altered body hash, status forged to CERTIFIED) all caught.
- `tests/test_verifier.py` (7 tests; skipped without a C compiler): agreement on the conformance fixture, the
  three negative controls, canonical JSON vs `json.dumps`, deflate and Fortran-ordered sidecars, the `window`
  and `all` S3 paths.
  `.github/workflows/verifier.yml` builds and runs it (syntax only, not exercised here). Documentation:
  `verifier/README.md`, `docs/SCOPE_MATRIX.md` (how to read a verifier report).
### Scale track (roadmap D-01, D-02; track/scale)

Derivations in `docs/proofs/D-01_sparse-residual.md` and `docs/proofs/D-02_matrix-free.md` with verification scripts
`verify_D-01.py` (400 random + 17 adversarial cases, exact rational arithmetic, 0 violations) and `verify_D-02.py`
(182 random specs + 18 adversarial cases at 60 digits, 0 violations); decision D-59, independent re-derivation
pending (D-12). No existing bound became tighter without a proof; the 3.1 a-priori sparse bound is kept, labelled
conditional, and is no longer the default.

- **D-01 a-posteriori backward error for the sparse S3 path (known issue MS-M11).** `inertia.inertia_sparse` now
  measures β: the residual L D Lᴴ − P(A − σI)Pᵀ of the stored SuperLU factors is formed in floating point, row block by
  row block (new module `certify_ed/ldl_residual.py`, `sparse_ldl_residual_bound`; deterministic block partition,
  `threads` for the blocks), and bounded entrywise by |E_c|/(1 − u) + γ_{t+1}(|L||D||L|ᵀ); ‖E‖₂ ≤ ‖E‖_∞. Lemma 4 then
  needs no hypothesis on SuperLU's arithmetic. Method name `sparse-superlu-residual` (was `sparse-superlu-symmetric`;
  `tests/test_planner.py` adjusted); the inertia record carries `residual` (measured residual, rounding and shift
  terms, blocks, threads, seconds). The 3.1 bound (γ_{w+1}‖L‖_F‖U‖_F + …, Higham Thm 9.3) is `inertia_method=
  'sparse-apriori'`, method name `sparse-superlu-apriori`. On the Heisenberg S^z = 0 blocks the measured β is 4–50×
  smaller than the 3.1 bound (`experiments/e2_s3_reach.py`, `results/e2_s3_reach.json`: d = 924 / 3432 / 12870 S3
  CERTIFIED in 3.0 / 15.5 / 95.4 s, β = 9.0e-11 / 2.4e-9 / 1.2e-6 against 3.7e-10 / 7.2e-9 / 5.8e-6; peak 1.55 GB at
  12870 with 8 threads). Memory: `inertia()` refuses a sparse factorization whose predicted peak
  (`cost.sparse_inertia_mb`, fill model plus the new `cost.sparse_residual_mb`) exceeds `budget_mb`
  (`resource_failure`, S3 NOT_CHECKED); d = 48620 is refused at 19 GB predicted; out-of-core factorization is not
  implemented (the bound would carry over unchanged: it needs only the stored factors). Tests
  `tests/test_sparse_residual.py` (46: sandwich on random and planted-inertia sparse matrices, exact rational
  entrywise check, determinism across threads and block sizes, memory gate, adversarial zero/ill-conditioned pivots,
  eigenvalue at σ, overflowing scale; negative controls: a corrupted factor entry, a flipped pivot sign and a wrong
  permutation raise β, an omitted lowest eigenvalue is not certified).
- **D-02 matrix-free residual bounds from the spec.** New module `certify_ed/matrixfree.py`:
  `MatrixFreeOperator(spec, states)` applies the spec term by term (bit-identical to `construct.SpecOperator`; the
  full state set needs no index search) and derives from the spec the rounding count W = w_row + L + ρ + 1 (w_row =
  Σ_t Π_l c(o_l) monomial contributions per output entry — n_terms for spin-1/2 and fermions, where the 3.1 count was
  n_terms·2^L + 2L + 2 —, ρ rounded amplitudes per string), the norm bound Σ_t |c_t| Π_l ‖o_l‖₂ and the named
  amplitude hypothesis. `OperatorAdapter` takes `width = min(3.1 count, W − 2)` and `norm = min(row-sum bound,
  spec bound)`. `Options(matrix_free=True)`: a claim without an operator is certified at any dimension without a
  reference matrix; S1 is CERTIFIED with `model_shift_bound = 0` and `amplitude_hypothesis` in its evidence and
  statement text; S2 radii concern the exact spec operator; S3/S8 NOT_CHECKED (NEEDS_MATRIX). Scale
  (`experiments/d02_matrixfree_scale.py`, Heisenberg chain, one-magnon exact eigenvector as a memmap, S1–S2):
  S2 CERTIFIED at every point with the radius at rounding level and the memory under 3.2 GB: d = 2^16: S2 8 s (one matvec 3 s), radius 3.96e-14, peak 67 MB, W = 51; d = 2^20: S2 169 s (one matvec 40 s), radius 6.34e-14, peak 257 MB, W = 63; d = 2^22: S2 653 s (one matvec 210 s), radius 7.71e-14, peak 869 MB, W = 69; d = 2^24: S2 1896 s (one matvec 734 s), radius 9.21e-14, peak 3160 MB, W = 75. The time is the Python-level term-by-term application (about 60 vectorised passes per product at N = 20–24); the bound itself costs one extra pass (|H|_abs|x|) and the norm pass. Tests `tests/test_matrixfree.py` (14: bit identity, counts, norm bound, radii not wider than the 3.1
  spec path and containing the eigenvalues, end to end on a memmap vector, spin-1 hypothesis, cancellation and
  couplings 1e±150; negative controls: a wrong coefficient and a planted wrong eigenvector fail S2).

### New statements track (roadmap C-03, C-04, E-04, C-12, C-13, E-07; track/new-statements)

Every statement of this track has a derivation in `docs/proofs/<id>_<name>.md` with its verification script
`docs/proofs/verify_<id>.py` (decision D-59: derived here, machine-checked in exact rational arithmetic where the
quantity is rational and at 40-50 mpmath digits where it is not; independent re-derivation pending, D-12). No
existing bound became tighter; new bounds are combined with old ones as min(old, new) only where both are proven.

- **C-03 entanglement entropy and spectrum.** `Options.entanglement = {name: {"sites": [...], "log_base": "e" | 2,
  "max_dim_a": 64}}` adds, to every S4 cluster record, `entanglement[name]`: a certified interval for S(rho_A)
  with rho_A = tr_B(P)/g (the reduced state of the certified eigenvector for g = 1; for a cluster the normalised
  projector, the only basis-independent object), the entanglement spectrum as intervals, the computed value, the
  trace-distance bound T* = s (+ eta/(1 - eta) for g > 1) and the Fannes-Audenaert term. Mathematics
  (`docs/proofs/C-03_entanglement.md`): ||rho_A - rho_hat_A||_1 <= 2 T* from the rank-2g trace-norm bound on
  P - Qhat Qhat^H and the contractivity of the partial trace; |S(rho_A) - S(rho_hat_A)| <= T* ln(d_A - 1) + h(T*)
  (Audenaert 2007; the upper bound T* may replace T only for T* <= 1 - 1/d_A, where the function is increasing:
  outside that range the record is NOT_CHECKED with the reason, never an interval); |lambda_k(rho_A) -
  lambda_k(rho_hat_A)| <= 2 T* (Weyl). The reduced matrix of the stored vectors is formed in exact rational
  arithmetic, its spectrum enclosed by Ostrowski's and Weyl's theorems around a floating-point
  eigendecomposition with exactly computed defects, and -p ln p, h and ln evaluated on dyadic intervals
  (`ivl.py`). New module `certify_ed/entanglement.py`; one hook in the S4 loop; `scope.OUT_OF_SCOPE_QUANTITIES`
  lists the quantity as DEFAULT_SELECTION (not NOT_IMPLEMENTED) and omits it when requested. Symmetry-adapted
  claims are not lifted (UNSUPPORTED_CLAIM). Verification: 1 150 random + 20 026 adversarial cases, 0
  violations; tests `tests/test_entanglement.py` (11) against 40-digit references, planted rotations and the
  negative controls.

- **C-04 correlations for symmetry-adapted claims.** A claim with `sector_isometry` V used to leave
  `Options.correlations` NOT_CHECKED (UNSUPPORTED_CLAIM). Now `correlations.certify_correlations_lifted` builds
  the site operators in the full space and certifies tr(Pi_Q O) through the isometry: Theorem 6 in the block on
  the compressed Hermitian operator (V^H O V)_h, plus g times the compression rounding bound (standard model,
  `compress_operator`), plus the isometry term 2 g ||O|| delta (1 + sqrt(1 + iota)), delta = 1 - sqrt(1 - iota),
  iota = S1's isometry defect, where Pi_Q = Q P_s Q^H is the orthogonal projector onto the lift of the certified
  block eigenspace by the polar factor Q of V (`docs/proofs/C-04_correlation_lift.md`). Each pair record carries
  `bound` (for Pi_Q) and `bound_V` (for Pi_V = V P_s V^H, the exact identity tr(P_s V^H O V) = tr(Pi_V O)), the
  defects and `refers_to`; two-letter requests go through the Hermitian decomposition; structure factors apply
  unchanged. A full-space matrix in `Options.observables` is lifted the same way for sector claims. Without
  S1's audit the records are NOT_CHECKED (DEPENDENCY_NARROWED); when the audit failed they are INCONCLUSIVE.
  The lift does not claim that range(Pi_Q) is an H-eigenspace (that rests on the invariance defect, recorded)
  nor that other sectors are absent from the level. Verification: 240 random + 14 adversarial cases, 0
  violations; tests `tests/test_correlation_lift.py` (7) against full-space values on the 8-site ring (k = 0 and
  parity sectors, S^z S^z and S+ S-, S(q)), with a planted 0.1 % stretch (INCONCLUSIVE), S1 skipped
  (NOT_CHECKED) and a planted iota = 0.05 where only the isometry term makes the interval contain tr(Pi_Q O).

- **E-04 certified parameter intervals.** `certify_ed.sweep.certified_sweep(spec_fn, couplings, k, ...)` certifies
  the k lowest eigenpairs at every grid coupling and assembles eigenvalue tubes between them: with H1 the
  derivative spec (declared, or inferred from the coefficients at the first two couplings) and B >= ||H1||_2 (the
  smaller of the row-sum bound of the built H1 plus the builder shift and the triangle bound over the terms, both
  rounded up), Weyl's inequality per sorted index gives |lambda_j(F_k(c)) - lambda_j(H(c_k))| <= B |c - c_k| for the
  affine interpolant F_k(c) = H(c_k) + (c - c_k) H1 of each segment, so every S2 enclosure widens to a tube and the
  rank certified by S3 is preserved (equivalently: the count below a shifting threshold); the neighbouring
  certificate's tube, shifted by the segment's affinity defect (exactly zero when the coefficients are affine in
  binary64, computed in rationals), is intersected with it (`docs/proofs/E-04_parameter_intervals.md`). `tube_at`,
  `next_level_lower_bound_at`, U(1) blocks (`states_fn`, zero-leak check for H1) and symmetry sectors (`sector_fn`:
  tubes for the stored block family with ranks in the block; the lift to the full operator is certified at the grid
  points only, recorded) are supported. `certified_crossing_interval(scan_a, level_a, scan_b, level_b)` returns the
  coupling runs where two tubes are certified apart ('<' / '>') and the crossing intervals between them, using the
  2B-Lipschitz bound on the margin between sample points so that no coupling is missed. CLI `certify-ed sweep
  --model ... --param ... --values ...`. Verification: 1 049 random + 107 adversarial cases, 0 violations; the tubes
  contain dense-diagonalisation eigenvalues at 100 % of random intermediate couplings; negative controls: a wrong
  B (B/20) is caught at every sampled coupling (violation or empty intersection), a wrong declared H1 is flagged by
  the affinity defect. Tests `tests/test_sweep.py` (7), including the k = 0 / k = pi ground-state crossing of the
  8-site J1-J2 chain bracketed at the Majumdar-Ghosh point J2 = 1/2.

- **E-07 Davis-Kahan alternative for S4.** `davis_kahan.sin_theta_davis_kahan`: with the Ritz matrix H' = Qhat^H A Qhat,
  ||A Qhat - Qhat H'||_F <= ||R||_F / sqrt(1 - eta) and ||H' - Lambda|| <= w' = ||X^H R||/(1 - eta) + delta spread /
  sqrt(1 - eta), the Frobenius form of the sin(Theta) theorem (proved in `docs/proofs/E-07_davis_kahan.md`) gives
  ||Qhat Qhat^H - P||_2 <= s_DK = ||R||_F / (sqrt(1 - eta)(dist - w')), dist the certified distance of the cluster's
  claimed values to every other eigenvalue (S3's window). `certify_eigenspace` now takes both proven bounds and
  uses s = min(s_6, s_DK); the cluster record carries `sin_theta_bounds` = {theorem6, davis_kahan, used, ...}.
  The (||A|| + ||Lambda||) delta term of Theorem 6's rho and its 2.5 sigma model-shift allowance do not enter s_DK, so
  s_DK is smaller in every tested case (2.4x on the 8-site ring; 100/100 random cases in the verification). A cluster
  with s_6 >= 1 but s_DK < 1 is now CERTIFIED at projector level (status change by a proven bound; D-12 pending).
  `enclosure._cluster_bound` exports its ingredients (`R_fro`, `XR`, `delta`, `spread`) in its result dict (additive).
  Negative control: an overstated dist (an eigenvalue inside the claimed gap) undercuts the exact sin(theta) in 20/20
  planted cases, which is why dist comes from S3's certified window.
- **C-12 approximate-symmetry labels.** `Options.approximate_symmetries = {name: (S, lattice[, allowed_values])}`
  (S a matrix or a diagonal in the claim space) adds `quantum_numbers[name]` with `approximate_symmetry: true` to
  every S4 cluster: the commutator bound c >= ||[H, S]||_2 is computed from the stored reference matrix
  (`davis_kahan.commutator_bound`, infinity norm of the anti-Hermitian commutator plus the standard-model rounding)
  and recorded as the hypothesis; every sorted eigenvalue of X^H S X then has an eigenvalue of S within
  D = (2 s + 2 s^2 + eta) ||S|| + (c + ||S|| w/2) / gamma_0 + rounding (w the width of the cluster's level enclosure,
  gamma_0 its midpoint's distance to the rest), and the labels snap to the lattice / allowed values when D is below
  half their spacing (`docs/proofs/E-07_davis_kahan.md`, Proposition 2; the tight case A = diag(0, gamma),
  S = c' sigma_x attains the bound). Without a stored reference matrix the record is NOT-CHECKED (NEEDS_MATRIX).
- **C-13 fidelities.** `davis_kahan.certified_fidelity(X1, s1, eta1, X2, s2, eta2)` and
  `fidelity_from_certificates(cert_a, cert_b, members_a, members_b)` (stored eigenvectors and CERTIFIED S4 clusters
  needed): tr(P_1 P_2) = |<psi_1|psi_2>|^2 for g = 1 (the mean subspace overlap tr(P_1 P_2)/g for g > 1) with the bound
  g (s_1 + eta_1) + g (1 + eta_1)(s_2 + eta_2) + rounding of ||X_1^H X_2||_F^2 (Proposition 3). Typical use: the overlap of
  ground states at two couplings (fidelity susceptibility inputs), two sizes, or two sectors in one basis.
  Verification of the three statements: 529 random + 24 adversarial cases, 0 violations; tests
  `tests/test_davis_kahan.py` (7).
### Figures track (track/figures)

- **Figure set for the manuscript, the report and the docs site.** `experiments/figures/` holds one script per
  figure (listed in `provenance.NON_COMPUTATIONAL`: they read records and documents and draw, none computes a
  bound) and writes PDF + PNG + `<name>.numbers.json` into `docs/figures/`. Concept figures (no data):
  `fig_certificate_anatomy` (blocks and statement dependencies), `fig_enclosure_geometry` (Theorems 1–3,
  Lemma 4, Corollary 5 on the real line), `fig_inertia_completeness` (counting-function sandwich, vectorless
  window count, refusals), `fig_rounding_accounting` (the Theorem 1 accounting of theory.md §2 as a flow).
  Evidence figures from the records of `run_full_20261022` (and `run_full_20261021` for the second stack):
  `fig_failure_matrix` (E1), `fig_cost_memory` (E2, two stacks), `fig_two_stack` (compare CSV, 0 MISMATCH),
  `fig_mutation` (E6 with `classification.json`; the dev/3.2 16-mutant sample from RESULT_integration_3.2.md),
  `fig_corpus_detection` (corpus report), `fig_scope_vs_survey` (survey percentages with Wilson intervals
  against the scope-matrix row status), `fig_roadmap_progress` (78 items by wave and theme; the roadmap CSV
  snapshot with a `wave` column is `experiments/figures/data/`). Applications: `fig_crown_sectors` (E9),
  `fig_aklt_eta` (E10). `docs/figures/INDEX.md` is rendered by `experiments/figures/make_index.py` from the
  numbers files (every number in a caption comes from the data file the entry names);
  `docs/figures/MANUSCRIPT_PLACEMENT.md` proposes the placement in manuscript v4 and the report with
  `\caption` scaffolds naming the existing `make_numbers` macros and the macros still needed. mkdocs nav gains
  a Figures section. Tests: `tests/test_figures.py` runs every script (skipped, not passed, when the record
  files are absent; a negative control checks the data-absent exit code) and checks INDEX.md is in sync.
  Deviations recorded in the captions: the E1 record stores per-statement statuses for the correct result only
  (the statement column is derived from the diagnosis text); the E9 landscape is recorded at the re-optimised
  couplings, not the printed ones, and the named sectors carry no certified interval.
### Backlog proofs (D-59; track/backlog-proofs)

Six items of the UNVERIFIED backlog derived in `docs/proofs/` (one document per statement in the D-59 format:
statement, proof, where the code implements it, computational verification, analytic cross-check, status line) with
verification scripts `docs/proofs/verify_<id>.py` and their recorded outputs in `docs/proofs/runs/`. Every status line
reads "Derived on 2026-10-30; machine-checked (...); independent re-derivation: pending (Phase 2); Sarang's
check (D-12): pending" — nothing is called verified.

- **Theorem 2 with Kahan's constant** (`thm2_polar-factor.md`). The polar-factor lemma (Q̂ = X(XᴴX)^{−1/2} is an
  isometry, ‖X − Q̂‖ ≤ δ, ‖AQ̂ − Q̂Λ‖ ≤ ρ = ‖R‖ + (‖A‖ + ‖Λ‖)δ) and Kahan's inclusion theorem for an *arbitrary*
  Hermitian block (proof: Courant–Fischer interval count plus Hall's theorem) give the cluster radius ρ. The code used
  ε = (w + √(w² + 4ρ²))/2 ≥ ρ (the Weyl-perturbation proof); `enclosure._cluster_bound` now returns min(ρ, ε*), both
  proven, and `bound` carries `rho` (and `eps` = ρ for the schema). `certify._subspace_shift` and
  `sectors.decomposition_shift` return ρ instead of (1 + √5)/2 ρ. Bound change: cluster radii shrink by up to the
  factor 1.618 where ρ was the smaller bound (tests/test_whole_spectrum.py compares them within that bracket; the
  Theorem 1 pipeline is bit-identical to the frozen reference).
- **Theorem 2′** (`thm2prime_eps-star.md`): derived as stated (Kahan for the Rayleigh block of Q̂, Weyl for
  Q̂ᴴAQ̂ − Λ with the commutator identity [M, Λ] = [M − I, Λ − cI]); `_cluster_bound` evaluates every term with
  explicit rounding counts (≤ 6 per `up`) and the INT-05 additive terms.
- **Corollary 5** (`cor5_pairing.md`): the pairing of the sorted claimed values with λ₁ … λ_k is proved through the
  disjoint union intervals of the enclosure groups and a sorted-pairing lemma for a uniform radius (it fails for
  non-uniform radii, hence the group structure); the model shift enters by Weyl. **Theorem 8** (`thm8_rho.md`): the
  sector assembly with radius ρ (Corollary 5 per sector, the K lowest of B, Weyl for the unitary polar factor of W,
  Weyl again for the builder shift). `compose.py` and the sector path of `certify()` symmetrise the built reference
  matrix (`construct.hermitian_part`; a build with irrational local amplitudes can differ from its adjoint in the
  last bit), and `D` is formed with `norm2_upper` instead of `up(sqrt(sum of squares))`.
- **Appendix A model shift** (`appA_model-shift.md`): the builder error (which operations round, Lemma A1), the
  symmetrisation (Lemma A2), ‖E‖₂ ≤ √(‖E‖₁‖E‖_∞) ≤ max(…) (Lemma A3), Δ = max(‖D‖₁, ‖D‖_∞) + Δ_builder (Theorem A4) and
  the Weyl transfer of S2/S3 (Theorem A5). `_builder_shift` divides by (1 − γ_W) where it multiplied by (1 + γ_W)
  (the latter is sufficient only for Wu ≲ 10⁻⁸). `OperatorAdapter.width` for a spec operator is m + 2L_max + 2 (was
  m + L_max + 2, short by L_max − 4 roundings for strings of five or more letters). Its verification script was
  written but not executed at the time (it ran on 2026-11-03); the document records the later run.
- **Theorem 1 accounting** (`thm1_fp-accounting.md`, `appendixA_scaffold.tex`): the INT-04 list as a proof with the
  scalar lemmas behind every factor ((1 − γ₃)(1 + 4u) ≥ 1; the norm guards give a 3u slack; the |A||x| and E guards),
  hypotheses n, w ≤ 2²⁶, and a statement-level LaTeX scaffold for the manuscript's Appendix A.
- **INT-05 closed** (`int05_underflow.md`). Model fl(z) = z(1 + δ) + η, |η| ≤ 2⁻¹⁰⁷⁵, η = 0 for ±, derived from the
  binary64 format. `fp.norm2_upper`/`norm2_lower` carry `norm_underflow_term(N)` ≥ √(N u_min)√((1 + γ_N)/(1 − γ_N));
  `enclosure.residuals` adds `residual_additive_term(n, w, cplx)` ≥ √n c_w u_min to the residual bound and 2 ULP_MIN
  to the radius; `fp.exact_dot` leaves products below 2⁻⁹⁶⁷ out of Dekker's TwoProduct (exact above that threshold
  for normal factors; a subnormal factor is rescaled by 2^±60) and adds n_tiny·2⁻⁹⁶⁶ to its error bound;
  `gram_additive_term` and `+ k ULP_MIN` guard the products of `_cluster_bound`, `compose`, `decomposition_shift`,
  `_subspace_shift` and `kato_temple`. Radii never shrink (Theorem U6; difference ≤ D) and are bit-identical for
  every radius above ≈ 10⁻¹⁴⁵. The UNDERFLOW_RISK scan covers the stored inputs and the model shift only; the
  post-enclosure scan of radii and residual bounds is removed (`test_radius_forced_tiny_by_exact_eigenpair_of_scaled_operator`
  is now CERTIFIED with radii of order 10⁻²⁷⁴ that enclose the exact eigenvalues).
- **Mutation survivors** (RESULT_integration_3.2.md): `sectors.py:240` — a planted label defect two ulps below the
  decision threshold (leak within the rounding allowance rnd) is not certified (`test_label_audit_defect_within_rounding_allowance_of_the_threshold`);
  `certify.py` Kato–Temple `pos > 0` — the second-lowest pair's isolation interval starts at the lowest
  enclosure's upper end; with −∞ the Rayleigh quotient of a ground-state-contaminated vector would be an unsound
  upper end (`test_kato_temple_second_lowest_pair_uses_its_lower_neighbour_as_isolation_end`); `checker.py:261` —
  `_float_vector_to_int` reconstructs the vector exactly (the mutant 2⁵⁴ doubles it; `radius_exact` is scale
  invariant, so only a reconstruction test sees it). The 15 E6 survivors of the 3.1.0 run were not examined (the
  worksheet is not in the clone).
- Tests: `tests/test_backlog_proofs.py` (25); `tests/test_underflow_check.py` and `tests/test_whole_spectrum.py` adjusted
  to the INT-05 and Theorem 2 changes. Computational source changes (enclosure.py, fp.py, certify.py, compose.py,
  sectors.py, construct.py): records are not comparable by `source_hash` with 3.2.0.dev0 before this track.
### Props-proofs track (D-59 backlog: Props 14-16, Lemma 17; roadmap C-06, C-16; track/props-proofs)

- **Derivations (D-59).** `docs/proofs/` holds one derivation document per statement with a verification script
  in exact rational arithmetic or mpmath at 200 digits: `L17_dyadic_exp_ln.md` (dyadic interval arithmetic,
  exp and ln by truncated series with proved remainders), `P16_thermo_intervals.md` (S7), `P14_degeneracy_bracketing.md`
  (S8), `P15_summed_bounds.md` (S4 combinations), `C06_truncated_thermodynamics.md` (Proposition 16'),
  `C16_exact_bracket.md`. Each document has the six sections of D-59 and ends with the status line "Derived on
  2026-10-30; machine-checked (...); independent re-derivation: pending (Phase 2); Sarang's check (D-12):
  pending". `tests/test_proofs_props.py` runs every script at reduced size and pins the code facts the
  proofs use. The statement strings keep the word UNVERIFIED in the D-12 sense (independent re-derivation
  pending) and name the derivation document.
- **Corrections found by the derivations (both make a bound looser, never tighter).** `ivl.py`: the atanh
  remainder constant is 8/7 for arguments up to 0.35 (the rounded upper argument can exceed 1/3 by one grid
  unit; the 3.1 constant 9/8 was stated for u <= 1/3), with a runtime range check; `exp_interval` encloses
  arguments below -1000 by [0, 2^-192] and refuses arguments above 2^16 (a beta*e of 2^190 made the halving loop
  attempt 2^190 squarings). `correlations.combine_certified`: the moduli |a_k|, |v_k| are bounded with an IEEE
  sqrt (`modulus_upper`) instead of a libm hypot and the rounding term is formed with directed rounding; the
  correlation record names the stored-operator hypothesis (D2) and the structure-factor record the
  floating-point-phase hypothesis (D1). `degeneracy.py`: the S8 statement names the open interval.
- **C-06 truncated-spectrum thermodynamics.** `thermo.thermodynamics_truncated`: for a 'lowest' claim with S3
  CERTIFIED (lower bound L on lambda_{k+1}) and the dimension n, the omitted weight is bounded by
  (n - k) exp(-beta L) and the omitted first and second moments by the envelopes of e exp(-beta e) and
  e^2 exp(-beta e) over [L, M] (M the operator norm bound) or [L, inf); two-sided intervals for Z, F, U, C,
  informative when T << L - E_0. S7 is CERTIFIED for such claims with evidence `tail_bound` (schema 2.1 declares
  `tail_bound` and `claimed_count`); a 'lowest' claim with S3 CERTIFIED and no temperatures gets
  NEEDS_TEMPERATURES. `tests/test_thermo_truncated.py` checks against full spectra of chains with negative
  controls (no S3, planted wrong L, dimension too small).
- **C-16 exact bracket.** `certify_ed/exact_degeneracy.py`: the characteristic polynomial of the exact declared
  model (or of a symmetry-sector block), its squarefree factorization and Sturm sequences give, in exact
  arithmetic, the number of eigenvalues in a rational bracket with their multiplicities: "exactly g in [a, b],
  all equal" or "not all equal". With `Options.exact_recheck` every S8 cluster record carries `exact_bracket`;
  a contradicted count makes S8 FAILED. Dimension cap 600 (charpoly cost). `tests/test_exact_degeneracy.py`:
  Heisenberg singlet and triplet exactly degenerate; a field of 2^-30 gives a cluster of three that is certified
  as "not all equal" (negative control); sector counts add up to the k = 0 block.

### Statements track (roadmap C-01, C-11, C-02, C-05, C-09, C-10; track/statements)

- **C-01 gap record.** `body.gaps` (`certify_ed/gaps.py`): for a claim whose S2 and S3 are both CERTIFIED, the
  S2 enclosures (radii including the model shift) are merged into levels (maximal chains of overlapping
  intervals), each holding exactly its count of eigenvalues (Corollary 5 for 'lowest', the window counts for
  'window', count = dimension for 'all', then the pigeonhole step of theory §6), and every difference of
  consecutive levels gets an outward-rounded interval. For 'lowest' claims the gap above the highest claimed
  level has S3's `gap_lower_bound` as lower end and no upper end (the next eigenvalue is not claimed); it is
  repeated as `S3.evidence.gap_interval`. To bound λ_{k+1} − λ_k on both sides, claim k + 1 pairs: the upper
  end is then μ_{k+1} + r_{k+1} − (μ_k − r_k) (Theorem 1). No new statement id (the scope matrix calls this a
  presentation task); the schema has `$defs/gaps_record`. Negative controls: a planted wrong (k+1)-th
  eigenvalue gives S2 FAILED and no record; without S3 there is no record (`tests/test_gaps_crossings.py`).
- **C-11 certified crossings.** `certify_ed.crossings.certified_crossing(cert_a, cert_b, level=0)` and
  `certify-ed crossing A.json B.json [--level L] [--json out]`: the rank-`level` eigenvalue intervals of two
  certificates are compared; relation `<`, `>` or `not separated`, the margin (rounded down), both intervals,
  both certificate hashes and spec hashes, and the caveat when a certificate has no S3 CERTIFIED (then the
  interval is the S2 enclosure of the claimed value at that sorted position and the statement says so). A
  certificate without S2 CERTIFIED is refused (`ValueError`). Pure interval comparison of certified data.
- **C-02 vectorless window count.** An eigenvalue-only claim with `kind='window'` used to leave S3 NOT_CHECKED
  (NEEDS_EIGENVECTORS). Now `_certify_window_vectorless` certifies the number of eigenvalues in (a, b) from
  inertia counts alone: Lemma 4 at the edge, plus two counts at e ± δ whose agreement makes the edge collar
  eigenvalue-free through the count inequality of Proposition 14 (docs/theory.md §3, "Vectorless window
  count"). S3 is CERTIFIED when the count equals the number of claimed values, FAILED otherwise
  (`missing_at_least` / `spurious_at_least`), INCONCLUSIVE when an edge is within the backward error of an
  eigenvalue (the summary says "window edge ... of an eigenvalue"; `next_action` says to move the edge).
  Evidence: `vectorless: true`, `count_in_window`, `edges`, `count_statement`, `pairing`. No claimed value is
  paired with an eigenvalue: S2, S4, S8 stay NOT_CHECKED and the summary says so. Six factorizations per
  window. Tests: `tests/test_vectorless_window.py` (count matches the dense spectrum at four windows, planted
  wrong counts FAILED both ways, edge on an eigenvalue INCONCLUSIVE, random-window negative control, gates).
- **C-05 Hermitian decomposition.** `Options.observables` and `Options.correlations` accept non-Hermitian
  operators: `correlations.certify_observable_general` splits O = H + iK (H = (O + O^H)/2, K = (O - O^H)/(2i)),
  certifies tr(P H) and tr(P K) by Theorem 6 and records the recombination tr(P O) = tr(P H) + i tr(P K) with
  bound b_H + b_K (Proposition 15); the record names both parts (`parts`, `definition`, `recombine`,
  `value` = [re, im]). The parts are checked to be formed exactly in floating point (TwoSum); otherwise the
  record is INCONCLUSIVE. Correlation requests take a two-letter string (`{"letter": "+-", ...}` for
  S+_i S-_j, built as one two-site term from the spec; C_ji = conj(C_ij) in the structure factor). A
  non-Hermitian observable used to reach `certify_observable` unchecked; `certify_observables_batch` used to
  raise. Tests: `tests/test_hermitian_decomposition.py` against the exact N = 4 ring values (-1/3, 1/6, 1/2;
  S(pi) = 4/3), with a rounded-parts negative control.
- **C-09 level statistics.** `certify_ed.levelstats.level_statistics(cert, unfolding='none'|degree, bins, bin_range)`
  and `certify-ed levelstats CERT.json [--unfold poly:D]`: from a kind-'all' certificate with S2 and S3
  CERTIFIED, the levels of `gaps.merge_enclosures` carry ranks, and interval arithmetic (`ivl.py`) gives the
  spacings s_n, the r-ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) (undefined when the larger spacing
  may vanish, as inside a multiplet), the mean r as an interval, and a spacing histogram in units of the mean
  spacing with per-bin at-least / at-most counts. The unfolding is a declared parameter: none, or a polynomial
  fitted to the staircase in floating point, evaluated on the intervals by interval Horner, recorded with its
  coefficients and marked "declared, not certified"; the unfolded spacings are clipped to >= 0 only when the
  derivative of the polynomial is positive on the hull of the spectrum (interval check). A certificate
  without a counted spectrum is refused (`LevelStatisticsError`, CLI exit 2 with a message). The scope block's
  "level statistics" row keeps NOT_IMPLEMENTED (no certificate statement) with the detail naming the tool.
- **C-10 extrapolation report.** `certify_ed.extrapolation.extrapolation_report(certs, model="1/N"|"1/N^2"|callable,
  sizes=None, per_site=False, level=0)` and `certify-ed extrapolate C1.json C2.json ... [--model] [--per-site]`:
  Markdown and JSON listing the per-size certified intervals by certificate hash (through
  `crossings.level_interval`, with the caveat when S3 is not CERTIFIED), a numpy least-squares fit of the
  declared model to the midpoints, and the fitted limit with the sentence that it is NOT certified and
  depends on the declared model. The extrapolated record has no status field and its report section contains
  no status word (test). `certify_ed/extrapolation.py` is in `provenance.NON_COMPUTATIONAL`; `gaps.py`,
  `crossings.py` and `levelstats.py` are not (their intervals are claimed-certified numbers).
- New public names: `certified_crossing`, `level_statistics`, `extrapolation_report` (docs/api.md regenerated).
  Tests: `tests/test_levelstats_extrapolation.py`.
### Whole spectrum, interior windows, streaming vectors, planner, operator text (roadmap C-17, C-14, A-06, D-08, C-15; track/whole-spectrum)
- C-17 whole-spectrum path. `quick(H, w, V, spec, kind="all")` and `certify-ed certify --kind all` use the
  `whole-spectrum` preset unless a preset, `--only` or `--skip` is given (recorded in the scope block as any
  preset); `temperatures=[...]` adds S7. The evaluation of the S2 bounds was reorganised without changing a
  bound: `enclosure.residuals` works in column blocks (bit-identical for sparse and spec operators; a dense
  operator keeps one `gemm` call because OpenBLAS `gemm` is not invariant under column blocking), the rounding
  bound `E` is reduced to its column norms and not stored, the dense `|A|` is formed transiently, the exact
  Gram inner products of a cluster are cached across merges, the Kato-Temple products `H x`, `|H||x|` are
  formed for a block of isolated pairs at once (for a sparse operator the intervals are bit-identical to the
  per-pair ones; for a dense operator they differ in the last bits because `gemm` columns and `gemv` are
  different floating-point evaluations of the same bound), and the S4 per-cluster Python loops over all other
  pairs became array operations. `fp.exact_sum` (cascaded error-free summation, Ogita-Rump-Oishi SumK with
  Knuth's TwoSum) returns `math.fsum`'s value and falls back to it whenever a rounding boundary is within the
  slack; `exact_dot` uses it for long vectors. Timings and peak memory, whole-spectrum preset with two
  temperatures, Heisenberg chain Sz = 0 blocks, dense operator unless noted (8 cores, OpenBLAS; before -> after):
  the 924 block takes 3-4 s either way (the git provenance call dominates);
  dimension 3432 (dense operator): total 34 s -> 22 s, S2 18 s -> 14 s, peak RSS 0.9 GB -> 0.7 GB (radii sha 7b132274 = 7b132274)
  dimension 12870 (sparse operator): total 568 s -> 398 s, S2 536 s -> 366 s, peak RSS 8.7 GB -> 2.9 GB (radii sha def5ba16 = def5ba16)
  dimension 12870 (dense operator): total 798 s -> 635 s, S2 657 s -> 585 s, peak RSS 9.8 GB -> 8.5 GB (radii sha 01b8df55 = 01b8df55)
  (the 12870 sparse runs overlapped with the test suite; the dense ones ran alone). The dense 12870 run of dev/3.2
  needed 10.1 GB on a 16 GB machine and paged; the remaining S2 time is two n^2 k gemms (about 4 TFlop each at
  40 GFlop/s) plus the exact Gram inner products of the 6400 clusters.
  The radii are bit-identical before and after on every fixture (`tests/test_whole_spectrum.py` against the
  frozen `tests/_enclosure_reference.py` and against the recorded radii in `tests/fixtures/`).
- C-14 interior windows everywhere. `Claim` validates `kind` and `window` (`certify.parse_window`: a pair
  a < b of finite numbers, from a tuple, a list or 'a,b' / 'a b'); the same rule applies in
  `claim_from_arrays/npz/hdf5/file/bundle`, in the bundle's `meta.json` and on the command line, where
  `--window A B` (or `--window=A,B`) overrides the bundle. The certificate's `claim.window` is a list.
  Cookbook: `docs/cookbook/shift_invert_window.md` (`eigsh(sigma=..., which="LM")`, executed output).
- A-06 streaming vectors. `claim.eigenvectors` may be a numpy memmap or an h5py dataset: the residuals, the
  Gram matrices and the cluster/Kato-Temple columns are read in blocks of `Options(block_mb=...)` (default
  `budget_mb`, else 512 MB); the certificate is identical to the in-memory one (memmap and HDF5, C and F
  layouts, tested). Limits: a dense operator reads the vectors whole (no restriction: it is larger than the
  vectors); S3 by inertia for `lowest`/`window` claims still factorizes the stored matrix in memory
  (`docs/cookbook/streaming_vectors.md`). `array_sha256` and the underflow scan work in row chunks.
- D-08 planner. `cost.plan_certification` / `plan_for_claim` predict seconds and peak memory per statement from
  n, nnz and k (dense LDL^T workspace 8.5 n^2 entries; sparse LU from a fitted fill model; residual arrays
  counted one by one; rates measured on this machine, `cost.RATES`, `experiments/planner_calibration.py`);
  `Options(plan=True)` / `--plan` print and record it (`body["plan"]`, schema re-rendered); `certify-ed plan`
  prints it without running. `inertia_method="auto"` keeps the 3.1 rule and additionally sends a sparse matrix
  whose dense workspace would exceed `budget_mb` (or the available memory) to SuperLU
  (`cost.recommend_inertia_method`); the S3/S8 memory gate estimates the path that will be taken (3.1 estimated
  the dense path only, so a budget below the dense workspace made S3 NOT_CHECKED even when SuperLU fitted).
  `tests/test_planner.py`: predicted memory within a factor 2 of the measured peak (dense LDL^T at 3432, SuperLU
  at 3432 and 12870, the recorded whole-spectrum runs); statuses of `auto` equal those of the explicit method
  on the fixtures; `plan=True` changes nothing in the certificate. Advisory throughout: nothing here changes a
  status.
- C-15 operator text. `certify_ed.operators.parse_operator`: Pauli strings ("XIZY", "X0 Z2"), letter_site
  shorthand ("Sz_0 Sz_3", "S+_0 S-_3", "n_0 n_1"), linear combinations ("0.5 S+_0 S-_1 + 0.5 S-_0 S+_1"),
  dictionaries and QuSpin-style operator lists become the sparse operators `Options.observables` certifies;
  `Options.correlations` accepts "Sz_0 Sz_3" for `{"letter": "z", "sites": [0, 3]}`. Non-Hermitian results
  ("S+_0 S-_3" alone) are refused with the Theorem-6 message (the C-05 Hermitian split is not in this branch).
  Tests against Kronecker-built matrices on 4 sites.

### Usability (roadmap A-01, A-02, A-09, A-04, A-05; track/core-api)
- `certify_ed.quick(H, w, V=None, spec=None, kind="lowest", **options)`: builds the Claim and Options and calls
  `certify()` (no check of its own). `Certificate.summary()` is now one sentence; the former multi-line text is
  `Certificate.report()`; `_repr_html_` / `_repr_markdown_` render a statement table in Jupyter.
- `Certificate.methods_paragraph(style="plain"|"latex")` and `certify-ed statement CERT.json` (`certify_ed/report.py`,
  outside the fingerprint): a methods-section paragraph whose every number is a certificate field, rounded in the safe
  direction (radii up, gap bounds down), with hypotheses, run-stack caveat, body hash and a BibTeX entry (concept DOI;
  version DOI placeholder).
- `evidence["next_action"]` on every INCONCLUSIVE and NOT_CHECKED statement (`certify_ed/diagnostics.py`).
- New reason code `MISSING_DEPENDENCY`: without python-flint, S6 is NOT_CHECKED instead of an import error; the QuSpin
  importer names the extra to install. `certify_ed.apps` and the exact checker itself still need python-flint.
- `Options(budget_seconds="auto")` / `--budget auto`: a default budget from a cost model fitted to seven E2 numbers of the
  3.1.0 evidence run (`certify_ed/cost.py`; advisory, never changes a status); `Options(progress=True)` / `--progress`.

### Producers (roadmap F-01, A-07, F-02, F-05; track/producers)
- F-01 `certify_ed.integrations.quspin.certify_hamiltonian(H, w, V, basis=None, static=None, spec=None, kind="lowest",
  **options)`: one call from a QuSpin `hamiltonian` and the output of `H.eigsh` / `H.eigh` to a Certificate. The
  basis and the operator-string list are taken from the object; eigenvalues are sorted for `lowest` / `window`
  claims; the Claim and ModelSpec come from `claim_from_quspin`, so a symmetry block (kblock, pblock, ...) keeps
  the S1 sector audit through `basis.get_proj`. Time-dependent Hamiltonians are refused. `spec=` certifies
  against an independently written model. Docs: `docs/integrations/quspin.md`; a draft upstream pull-request
  text proposing `quspin.tools.certify` is `integrations/quspin_PR.md` (text only). Tests:
  `tests/test_integrations_quspin.py` (full, U(1), kblock and kblock+pblock bases give the statuses of the direct
  route; unsorted eigsh output; spec override with a wrong coupling -> S1 FAILED; negative controls: a planted
  eigenvalue and a swapped eigenvector are not certified).
- A-07 producer record hook `certify_ed.record(spec, path, kind=..., window=..., fmt=...)`: `with record(...) as
  rec: ...; rec.capture(w, V, H)` (or the attributes `rec.eigenvalues`, `rec.eigenvectors`, `rec.operator`,
  `rec.basis`) writes a producer bundle (schema 1.0) on exit with `meta.environment` (Python, numpy, scipy,
  certify-ed versions, platform) and `meta.timing` (UTC start/end, wall and process seconds); the decorator
  `certify_ed.recorded(spec, path)` wraps a function returning `(w, V)` or `(w, V, H)`. Nothing is written when
  the block raises; exiting without eigenvalues raises. The bundle schema (`certify_ed/schema.py`,
  `schema/producer-bundle-1.0.schema.json`) documents the two new optional fields. Module `certify_ed/recorder.py`
  (NON_COMPUTATIONAL). Tests: `tests/test_recorder.py` (bundle passes `check_bundle` and the CLI `conform` and
  `certify --from`; statuses equal the direct Claim; decorator; U(1) basis; window kind and csv format; meta
  validates with jsonschema; negative controls: a planted eigenvalue and a planted operator entry written through
  the hook are not certified).
- F-02 file importers for other ED codes (`certify_ed/io/hphi.py`, `diagham.py`, `spinpack.py`, `alps.py`;
  `claim_from_hphi`, `claim_from_diagham`, `claim_from_spinpack`, `claim_from_alps`; docs/integrations/ed_codes.md).
  Layout sources: HPhi `energy.dat` / `Eigenvalue.dat` / binary `eigenvec` (manual 3.5: int iterations, unsigned long
  local_size, complex double x (local_size + 1), first component unused; the 4-byte-long variant is detected from the
  file size); DiagHam binary real vectors (wiki: int dimension + doubles) with complex vectors ASSUMED Re/Im
  interleaved, spectrum text files (quantum-number columns, energy last: observed convention); SPINPACK: ASSUMED
  text layouts (`table`, `keyvalue`), recorded as `layout_status` in the claim's producer record, no vectors; ALPS
  `/spectrum/sectors/<n>/energies`, `quantumnumbers`, and the `SAVE_EIGENVECTORS` eigenvector and basis tables
  (reordered into the spec encoding when a spec is given). Every reader checks header against file size and column
  counts and raises `FormatError`. The basis order of HPhi/DiagHam/SPINPACK relative to a ModelSpec is not
  established: the caller passes `operator=` and `states=`. Fixture generators: `tests/fixtures/ed_codes.py`.
  Tests: `tests/test_io_ed_codes.py` (22: layouts, truncation and header checks, round trips through certify()
  on a 6-site chain with S1-S3 CERTIFIED, MPI rank concatenation, negative controls: planted energy and swapped
  vector in the producer's files are not certified).
- F-05 `ModelSpec.from_pauli_strings({"XXI": 0.5, ...})`, `ModelSpec.from_openfermion(QubitOperator)`,
  `ModelSpec.from_qiskit(SparsePauliOp)` (`certify_ed/integrations/pauli.py`, docs/integrations/pauli.md): spin-1/2
  Pauli-convention specs with character j on site j, basis index = big-endian bit string (the ordering of
  `get_sparse_operator` and `SparsePauliOp.to_matrix()`; Qiskit's little-endian labels give site = n-1-qubit,
  recorded in `parameters`), real coefficients required, identity term kept. The packages are read through their
  object interfaces (`.terms`; `.to_list()`, `.num_qubits`) and never imported. Tests: `tests/test_pauli_specs.py`
  (15: TFIM from strings equals `tfim_chain` entry by entry, a 2-qubit H2-like operator certifies with kind='all'
  against numpy, complex odd-Y operator, OpenFermion/Qiskit-like objects, JSON round trip; negative controls:
  a producer matrix with one coefficient off by 1e-3 -> S1 FAILED, planted eigenvalue not certified, wrong
  endianness -> S1 FAILED).
- `schema/certificate-2.1.schema.json` re-rendered: the file on dev/3.2 lacked the reason codes
  `MISSING_DEPENDENCY` and `UNDERFLOW_RISK` that `scope.REASON_CODES` already carried (the Wave 0 merge did not
  re-render).

### INT-01: the no-underflow hypothesis is enforced (E-06; design decision D-57, conservative option)
- The standard model behind every floating-point bound assumes no underflow: nonzero operands, radii and
  residual bounds of magnitude at least 2^-900 (`fp.UNDERFLOW_THRESHOLD`). 3.1.0 stated this and probed
  flush-to-zero, but never looked at the input. `certify()` now scans, before S2, the claimed eigenvalues,
  the eigenvector entries, the operator's stored entries (sparse data array; for a spec or matrix-free
  operator the norm bound and one matvec on a seeded random probe) and the model shift, and after the
  enclosure the radii, existence radii, residual bounds and residual norms. Exact zeros are fine.
- Any finding makes S2 INCONCLUSIVE with the new reason code `UNDERFLOW_RISK` (evidence: where, first
  indices, minimal magnitude, threshold; the summary keeps what the status would otherwise have been), and
  S3, S4, S7, S8 INCONCLUSIVE with the same code, since each rests on the S2 enclosure. S1 and S6 (exact
  arithmetic) are not affected. Statuses of inputs without a finding are identical to a run without the
  check (test `test_standard_fixtures_unchanged_by_underflow_check`); the existing fixtures have none.
- This is the conservative choice and it is reversible: `Options(underflow_check=False)` (CLI
  `--assume-no-underflow`) skips the scan and records `underflow_check: false` in the certificate's
  `scope.options` and `arithmetic.underflow_check.performed`; the certificate then carries the hypothesis
  as the user's. Rescaling the input instead of rejecting it was not implemented (it would need a
  statement about the rescaled operator). Alternatives remain open to Sarang (D-57).
- `docs/theory.md` section 1 names the hypothesis and the enforcement; `docs/SCOPE_MATRIX.md` and
  `scope.REASON_CODES` carry the code. Tests: `tests/test_underflow_check.py` (19: planted subnormal
  eigenvector entry, planted 2^-950 dense and sparse matrix entries, a radius of 1e-274 from an exact
  eigenpair of a 1e-258-scaled operator with eigenvectors of norm 1e125, tiny claimed eigenvalue,
  matrix-free probe, spec operator probe, complex components, opt-out recorded, CLI flag, fixtures
  unchanged, LAPACK numerical zeros not flagged, negative control: a finding never leaves S2 or a
  dependent statement CERTIFIED).
- **INT-05 (new known issue, found while implementing INT-01; needs a design decision).** The scan covers
  the stored inputs and the reported outputs. The *intermediate* quantities of the Theorem 1 bound are not
  covered by the hypothesis as stated, and routine inputs violate it there: `scipy.linalg.eigh` returns
  eigenvector entries of order 1e-265 (numerical zeros of a symmetry sector; above 2^-900), for which the
  residual vector R and the entrywise rounding bound E = gamma (|H||x| + |x||mu|) have entries of order
  1e-278 (below 2^-900), and their squares in `np.linalg.norm` underflow to zero. The lost mass is tiny
  compared with the guards (1 + 8u) of `fp.norm2_upper`, but saying so is a bound the theory does not
  contain (D-12): either the hypothesis is strengthened to cover the intermediates (which would make
  almost every LAPACK input INCONCLUSIVE), or the standard model is replaced by the underflow-tolerant
  model fl(a op b) = (a op b)(1 + delta) + eta, |eta| <= u lambda (Higham 2002, eq. 2.6; Demmel 1984)
  with an absolute term carried through `norm2_upper`, `residuals` and the Gram bounds. An early version
  of this branch scanned R and E and flagged `test_wrong_bond_sign_is_localized`; it was withdrawn. Test
  `test_lapack_eigenvectors_with_numerically_zero_entries_are_not_flagged` pins the observation.

### INT-04: code-facing accounting of the Theorem 1 radius (documentation)
- `docs/theory.md` section 2 now lists, line by line, how `enclosure.residuals` evaluates the Theorem 1
  radius in floating point: three roundings each covered by a factor (1 + 4u), the norm bounds of
  `fp.norm2_upper` / `fp.norm2_lower` with their (1 + gamma_{n+2}) and (1 + 8u) guards, and the model shift
  added afterwards with `add_up`. The code's bound is unchanged. The manuscript's Appendix A (k <= 6
  roundings, 1 + 8u, shift multiplied in) is to be rewritten from this list by Sarang (D-16).
  Test `tests/test_theorem1_accounting.py` recomputes the radius of a fixed 3 x 3 case with `Fraction`
  arithmetic and pins the code's value between the exact radius and the documented-factor evaluation.

### Merged from `fix/3.1.1` (f441227)

Two of the known issues of 3.1.0, fixed without touching what any statement asserts (merged with a merge commit;
under D-44 a version the paper cites needs its own full run).

- **INT-02.** `Certificate.verify_scope()` checks the scope block: every statement id exactly once (in scope, narrowed,
  or out of scope), known reason codes, every statement in scope present, a narrowed statement that is present is
  `NOT_CHECKED`, nothing reported outside the scope lists; `None` for certificates written before the scope block.
  `verify_integrity()` reports `scope_ok`/`scope_problems` and fails on `False`, so `Certificate.load()` refuses such a
  certificate even when its body hash was recomputed.
- **INT-03.** The exported `radius` and `existence_radius` arrays add the model shift with upward rounding (`add_up`), as
  the statements do; without a shift the radii are exported unchanged. The builder's rounding bound makes the shift
  nonzero for almost every certificate, so this changes almost every exported radius by at most one ulp.
- Tests: `tests/test_fix_3_1_1.py` (12; including a writer that drops a statement and recomputes the hash).
  Full suite 556 passed, 1 skipped; ruff clean.
### Format, packaging and governance (roadmap B-02, F-09, A-03, H-01, H-05, B-09, G-05, G-12; track/format-packaging)

No statement, bound or number changes in this track; the version is bumped in the known-issues track.
`certify_ed/schema.py` and `certify_ed/conformance.py` are listed in `provenance.NON_COMPUTATIONAL`.

- **B-02.** Published JSON Schema (draft 2020-12) of the certificate body, `schema/certificate-2.1.schema.json`,
  and of the producer bundle, `schema/producer-bundle-1.0.schema.json`, rendered from `certify_ed/schema.py`
  (the enumerations are the code's; a test keeps the files in sync). `Certificate.load(validate_schema=None|True|False)`
  validates when `jsonschema` is installed (extra `schema`) and otherwise runs the structural checks (every
  statement id exactly once in the scope block, reason codes declared and known, in_scope statuses equal to the
  statements); a non-conforming body raises `SchemaValidationError`. This also closes INT-02 independently of
  `fix/3.1.1`. `docs/schema.md`: field by field, stability promise, how a third-party writer conforms.
- **F-09.** `certify-ed conform DIR...` (`certify_ed/conformance.py`): bundle schema, import, certify, certificate
  schema, expected statuses. Three reference bundles under `conformance/reference/` (dense text, npz, matrix-free)
  with `expected.json`; `conformance/README.md` for third-party writers. Negative controls: unknown status,
  unknown reason code, CORROBORATED outside S5, duplicated/missing scope entries, tampered files with a recomputed
  hash, a bundle with a planted wrong eigenvalue.
- **A-03.** Extras `exact`, `quspin`, `hdf5`, `plots`, `schema`, `all` (the union), `dev`; `MANIFEST.in` ships the
  schemas, the reference bundles, docs and tests in the sdist. `python -m build` gives an sdist and a wheel; the
  wheel installs into a fresh venv and `certify-ed env` runs. `packaging/conda-forge/meta.yaml` (noarch draft),
  `packaging/RELEASE_CHECKLIST.md` (version bump, annotated tag, full run per D-44, checksums, sigstore, Zenodo
  new version under 10.5281/zenodo.20066565, GitHub release; never a force-push).
- **H-01.** `.github/workflows/ci.yml` replaces `tests.yml`: lint, build, schema sync; ubuntu/macos/windows x
  Python 3.11/3.12 with `.[all]` and QuSpin 1.0.1 (`--no-deps` after its real dependencies; the wheel declares the
  obsolete `importlib` backport; documented in `docs/reproducibility.md`), test log uploaded; `nightly-mutation.yml`
  runs a 32-mutant E6 sample without a failure gate.
- **H-05.** `containers/Dockerfile` and `containers/certify-ed.def` pin the 3.1.0 evidence-run stack (Python 3.11,
  numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0, QuSpin 1.0.1, h5py, matplotlib, mpmath).
- **B-09.** `packaging/SIGNING.md`: keyless sigstore signing and verification of release assets and `SHA256SUMS`.
  The certificate schema reserves `environment.distribution_sha256`; 3.1 does not write it (the hash of an installed
  distribution is not well defined; `source_hash` identifies the code from a checkout).
- **G-05 / G-12.** `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `GOVERNANCE.md` (maintainers,
  decisions, release cadence, deprecation policy: schema fields never removed within a major version),
  `docs/SCHEMA_RFC_PROCESS.md`, issue templates (bug, feature, *I could not certify X* feeding `docs/SCOPE_MATRIX.md`),
  `ROADMAP.md` summarising the scored expansion roadmap.
- Tests: `tests/test_schema_conformance.py` (17), `tests/test_packaging.py` (7).

### Viewer and corpus (roadmap A-08, B-07; track/viewer-corpus)
- `certify-ed view CERT.json [-o CERT.html]` and `certify_ed.viewer.view_file` (A-08): one self-contained HTML
  page per certificate (inline CSS, no scripts, no external resources): header (model, claim kind and k,
  version, hashes, environment, floating-point probe, no-underflow scan), the statements in plain words
  (status, summary, next action, key evidence), the scope block (in scope / narrowed with the user's reason /
  not checked and out of scope with the reason code and its scope-matrix explanation), the enclosure table
  (per pair: claimed value, radius and existence radius for the producer operator and, from the sidecar, with
  the model shift; enclosure interval; cluster), the Kato-Temple intervals, the S3 gap and inertia data, the
  S4/S8 cluster tables, S7 intervals, S6 exact data, the methods paragraph of `report.py`, and an integrity
  footer (body hash stored and recomputed, array hashes, `verify_scope`, schema validation). Every number is
  the certificate's value rounded by `report.round_decimal` in the safe direction (radii and error bounds up,
  gap and separation bounds down, certified windows inward, enclosures outward; claimed values and counts
  exact) and carries the exact value in `data-exact`. A tampered body, a modified or missing sidecar, or a
  scope block that drops a statement produces a red INTEGRITY CHECK FAILED banner and exit code 1 (the page is
  still written). User strings are HTML-escaped. `certify_ed/viewer.py` is in `provenance.NON_COMPUTATIONAL`.
  Tests: `tests/test_viewer.py` (20: every schema fixture certificate renders with every statement id and
  status; every displayed number checked against its exact value and direction; enclosure rows against the
  body and the sidecar; tampered body / sidecar / scope block banners; `<script>` in the narrowing reason and
  the claim description escaped; pre-2.1 certificate without scope block; CLI exit codes). `docs/viewer.md`.

- Planted-error corpus `corpus/` (B-07): `corpus/make_corpus.py` writes 60 producer bundles (1.7 MB; Heisenberg chain
  N = 8 and 10, J1-J2 chain N = 8 at J2 = 0.5, TFIM N = 8; dense `eigh` references, deterministic) with one planted
  error each in fourteen classes (wrong eigenvalue by 1e-5 and past the next level, unnormalised isolated vector and
  cluster member, swapped pairs, missing multiplet member, duplicated vector, wrong coupling, wrong matrix entry,
  non-Hermitian producer matrix, NaN eigenvector entry and eigenvalue, mislabeled sector, subnormal entry, wrong
  window count) plus one control per model, each with an `expected.json` naming the statements a correct checker
  must not certify (derived from the statement definitions, not from a checker). `corpus/run_corpus.py` runs any
  checker command on every case, reads a certificate or a flat `{statement: status}` result, and scores it
  (detected / unsound / silent / refused / conditional / error; false and weak alarms on the controls) into
  `results/<name>/report.md` and `report.json`. `corpus/README.md` describes every class, what a correct checker
  must say, the scoring rules, the revision history of the expectations and how to submit a case.
  Score of CERTIFY-ED 3.2.0.dev0 on the corpus (`corpus/results/certify-ed/`): detected 53 / missed 0 (unsound 0,
  silent 0, error 0) / false alarms 0 (weak 0) of 53 planted cases, 7 controls clean; the 8 NaN cases are detected
  by refusal at import. Disclosure: the first run, with `unnormalised_cluster_member` expecting only S4, scored
  detected 49 / missed 4 (silent 4: S4 NOT_CHECKED behind an INCONCLUSIVE S3, S2 INCONCLUSIVE on distinctness);
  the expectation was then revised to S2 and S4 (Theorem 2 and Theorem 3 share the near-orthonormality
  hypothesis, as the case note already said) and both reports are committed
  (`results/certify-ed_expectations-before-revision/`). No change was made to the certifier for the corpus.
  Tests: `tests/test_corpus.py` (71: every committed case is a readable bundle with a consistent expectation,
  index hashes, in-process determinism, the planted changes are present in the arrays, scorer verdicts, summary
  counts, result reader, runner end to end with a checker that certifies everything (unsound) and one that
  refuses, certify-ed on one planted case and its control, the committed report has no unsound case).
  `docs/corpus.md`. `corpus/` is outside the computational source fingerprint (it is neither `certify_ed/` nor
  `experiments/`); `experiments/e1_failure_matrix.py` is unchanged.

## [3.1.0] - 2026-10-22

`__version__` = 3.1.0 (tag `v3.1.0`; evidence run `results/run_full_20261022`, made from the tagged commit on the
Windows workstation with the reference library versions, D-54). Two soundness fixes found
while writing the proofs for the manuscript scaffold (`ms_v3`), sharper application bounds, and the
fixes of the independent review of 2026-10-21 (ids IMP-xx: code audit; SS-xx: second stack; MS-Nxx:
manuscript check). The computational source hash changes with this release, so the records of the run
of 2026-10-21 (source 7fb63586baf7ea6f) are superseded by the next full run.

### Known issues of 3.1.0 (found while filling the manuscript's evidence pointers, 2026-10-22)
- **INT-01.** The no-underflow hypothesis of the floating-point statements is assumed, not enforced: `fp.py` probes
  gradual underflow and denormals-are-zero, but no input check rejects operands or radii below 2^-900 (the
  manuscript said it did). Needs a design decision (reject, rescale, or state the hypothesis as the user's).
- **INT-02.** No validator rejects a certificate whose scope block omits a statement: `verify_integrity` checks the
  hashes only, and the completeness check of the scope block exists only as a test helper. Fixed on `fix/3.1.1`.
- **INT-03.** The exported `radius` and `existence_radius` arrays are `enc.radius + eps_shift` in round-to-nearest
  (`certify.py` 1161-1162) and can understate the sum by half an ulp; the statements themselves use `add_up`.
  Fixed on `fix/3.1.1`.
- **INT-04.** Appendix A's accounting of the Theorem 1 radius (k <= 6 roundings, factor 1 + 8u, model shift
  multiplied in) does not describe the code (three roundings, factor 1 + 4u, model shift added with upward
  rounding). The code's bound is sound; the text is to be rewritten (D-16).

### Evidence run of 3.1.0 (2026-10-22)
- `results/run_full_20261022`: tag `v3.1.0` (`e860a83`), computational source `acbef11af991c860` for every step, no step
  dirty; Windows workstation, Python 3.11.17, numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0
  (FLINT 3.6.0), QuSpin 1.0.1 (the library versions of the Linux 3.0.0 run). Tests: 541 passed, 1 skipped, 1 warning.
  E0–E10 ok. Numbers: 61/61 and 45/45 checks, after one change to `make_numbers.py` (`55c2ef9`): the W4 check
  asserted the Linux outcome for the 4×4 J2 = 0.5 completeness (NOT_CHECKED); here S3 is CERTIFIED (gap ≥ 0.2870)
  because a job object counts committed memory where RLIMIT_AS counts address space. E6: 38 of 64 mutants killed;
  15 survivors await the human verdict of MS-N03 (`results/run_full_20261022/E6_inspection_worksheet.md`).
- Two stacks: against `run_full_20261021` (Linux, 3.0.0) `compare_runs.py` finds no MISMATCH leaf
  (`RESULT_evidence_run_3.1.0.md`). With the library versions pinned, the E9 exact lower bound and every E10
  factorization residual are bit-identical across the two operating systems (the up-to-6 % differences of the
  candidate run came from library versions); E7 exact radii still differ (ratios 0.25–1.47).
- `compare_runs.py`: named classes for prose with run-stack digits (floating-point numbers only; counts stay strict),
  records extended by a newer version, records drawn from the stack (E5 ARPACK rows, E6 mutant samples; compared by
  their invariants in the RESULT document), explained differences (`EXPLAINED`, with reasons), and timing/memory
  leaves; negative controls in `tests/test_compare_runs.py`.

### Windows portability (2026-10-22, found by the first attempt of the evidence run)
The first full run on the Windows workstation (`results/run_full_20261022_attempt1`, kept with an ATTEMPT_NOTE)
failed in four places that the Linux runs never exercised. None touched a certified statement.
- **Matrix-free importer.** scipy 1.17's `LinearOperator.matmat` calls the producer's `matvec` with (n, 1)
  columns, which a function written for vectors broadcasts to (n, n). `certify_ed/io/formats.py` now always calls
  the producer's `matvec`/`rmatvec` with a vector of shape (n,), checks that n numbers come back (FormatError
  naming the file otherwise) and builds `matmat` column by column. Test: `test_matrix_free_matvec_is_called_with_vectors_and_checked`.
- **E2, E4: memory cap and peak memory.** `resource` and `preexec_fn` do not exist on Windows. New
  `experiments/portable.py`: RLIMIT_AS on POSIX (unchanged), a job object with a per-process memory limit on
  Windows; peak working set on Windows; CPU model and RAM from the registry and GlobalMemoryStatusEx (the harness
  recorded neither on Windows, and `make_numbers.py` read /proc unguarded). E2 rows and E4 settings record the cap
  mechanism (`memory_cap`). Tests: `tests/test_portable.py` (a capped child must raise MemoryError).
- **E6: the mutants face the shipped suite.** The mutation harness copied only `certify_ed/`, `tests/`,
  `examples/`; six 3.1.0 tests read `pyproject.toml` or `experiments/` (so the baseline failed on every platform) and
  25 tests skipped in the copy against 1 in the repository. The copy is now every tracked file (`git ls-files`).
  `tests/test_driver.py`'s `needs_repo` also requires a git checkout, as `test_imp06` already did.

### Candidate run and integration (2026-10-22)
- `release/3.1.0` at `v3.1.0-candidate`: fix track + scope framework + survey/scope generators + v4 macro emission +
  `compare_runs.py` + `make_result_candidate.py` + E9 named-sector step. Full suite on the Windows integration stack
  (Python 3.12.15, numpy 2.5.4, scipy 1.18.1, python-flint 0.8.0): 502 passed, 31 skipped (30 need QuSpin, 1 needs gcc);
  ruff clean. Computational source `1d4e33529dfce716`.
- `results/run_candidate_20261021` (E7, E9, E10, E8 with the pre-merge source `e707838f6852d474`) reproduces every statement
  of `run_full_20261021`; see `RESULT_candidate_run.md`. The 3.1.0 evidence run (all experiments, tests enabled, one
  commit, no edits in flight) is still to be made on the reference stack.

### Fixed (review of 2026-10-21)
- **IMP-18 (critical).** `checker.radius_exact` returned radius 0 for a zero test vector, so a wrong
  eigenvalue claimed with a zero vector (or, in `exact_recheck_sector_lowest`, with a vector from another
  sector whose exact projection vanishes) was CERTIFIED "with no floating-point assumption". The function
  now returns None for ‖x′‖ = 0 and `exact_recheck_lowest`, `exact_recheck_sector_lowest` and S6 report
  INCONCLUSIVE; the hypothesis x ≠ 0 is stated in the docstrings.
- **IMP-21, IMP-22 (critical).** A NaN claimed eigenvalue passed every comparison: `certify()` reported S2
  and S3 CERTIFIED for a full-spectrum claim and `enclose_spectrum` returned `complete=True` with an
  eigenvalue in no interval. `enclose()` and `enclose_spectrum()` refuse non-finite claims (ValueError);
  `certify()` and the exact re-checks report S2/S3/S6 INCONCLUSIVE; the accuracy comparison is written
  fail-safe (`not (radius <= target)`); a group with a non-finite radius is never 'distinct'. A zero
  claimed vector no longer crashes S3 (IMP-23): INCONCLUSIVE with the pair named.
- **IMP-13 (major).** A correct claim with valid but non-orthogonal eigenvectors of a degenerate level was
  S2 FAILED (cluster radius 4.9 against a target 4.5e-8). Theorem 2′ (`docs/theory.md`, UNVERIFIED, D-12)
  gives the cluster radius ε* = ‖R‖/√(1−η) + ‖XᴴR‖/(1−η) + δ·spread(Λ)/√(1−η); `_cluster_bound` uses
  min(ε, ε*) and records both. Every cluster member keeps its Theorem-1 radius as `existence_radius`
  (records, arrays); the cluster radius remains the radius of the distinct matching and of every
  indexed statement. When only the cluster radius misses the accuracy target, S2 is INCONCLUSIVE with
  the diagnosis, not FAILED. The review's case is now CERTIFIED (radius 8e-14).
- **IMP-12 / IMP-20.** `compose.certify_low_energy` uses ε = ρ (Weyl for the square W) instead of
  (1+√5)/2·ρ, and records symmetrised producer blocks (`hermitian_part` per sector, `symmetrised_blocks`).
- **IMP-08.** The certified window of S3 'window' claims is rounded inward (`add_up(a, β_a)`,
  `sub_down(b, β_b)`); the same window feeds Kato–Temple and the S4 separations.
- **IMP-14.** `inertia_dense` checks that the permuted LDLᵀ factor is lower triangular, as
  `inertia_exact` does.
- **IMP-24.** `fp.probe_float_environment` detects round-toward-zero (1 + 3·2⁻⁵⁴ must give 1 + 2⁻⁵²,
  and the mirror). The rounding-mode tests run on Windows through `ucrtbase` as well as on glibc.
- **IMP-05 / SS-02.** Provenance calls git with `core.filemode=false`: a lost executable bit (Windows
  checkouts) no longer marks the computational source as dirty.
- **IMP-06.** The numbers step records its provenance (`provenance.numbers_fingerprint`: commit, blob
  hashes of `make_numbers*.py`, source and tests hashes) in `manifest.steps.numbers.provenance`.
- **IMP-07 / MS-N02.** `\NtestsTotal`, `\NtestsVtwo` (and new `\NtestsPassed`, `\NtestsSkipped`) come
  from the run's tests step (`manifest.steps.tests.summary`), not from a live collection.
- **IMP-26.** `__version__` = 3.1.0.dev0 (pyproject, `.zenodo.json`); every certificate's `environment`
  carries `source_hash`, `source_dirty`, `git_commit`, `python_flint` and `flint_c_library`.
- **SS-04.** `cli.NEEDS["E3"]` includes QuSpin (part d imports it); `experiments/harness.Run` logs
  "ABORTED" and records `_aborted` when the experiment raises, instead of "finished".
- **MS-N12.** The FLINT C-library version (`flint.__FLINT_VERSION__`) is recorded in
  `environment.json`, `certify-ed env` and certificates; macros `\FlintCVersion`, `\PythonFlintVersion`,
  `\CertifyEdVersion`.
- **MS-N03.** `e6_recheck.py` (`_provenance`) and `e6_classify.py` (`provenance`) stamp their outputs with
  the commit, `source_hash` and `tests_hash`; `make_numbers_apps.py` requires the stamps on
  `classification.json`, `recheck.json` and `inspection.json` (CHECK FAILED otherwise), checks that they
  refer to the E6 run's source, and that every "test added, mutant now killed" verdict has a recheck
  entry (`\MutRecheckKilled`, `\MutRecheckTestsHash`).
- **IMP-28 / MS-N01.** `\AKLTCritMin` is rounded down (0.44, was 0.45); a check verifies that the
  printed range contains every criterion value.
- **IMP-27 / SS-15.** The AKLT agreement check tests exactly the printed `\AKLTMaxExcess` (the fixed
  5e-9 threshold is gone).
- **MS-N04 / MS-N05.** E2 text macros (`\CSoneUoneMax`, `\CStwoMax`, `\CkTwentyTwoSone`,
  `\CkTwentyTwoMem`) come from the same run as the table cells; `\CSoneUoneMax` is the maximum over
  all S^z rows (new `\CSoneUoneAtMaxDim` for the largest block).
- **MS-N06.** S3 feasibility is stated through nnz/fill: `\CnotFitAtOrBelowMaxDim`, `\CnotFitDim`,
  `\CnotFitNnzMin` and a check that every block of dimension ≤ `\CmaxFeasibleDim` whose S3 did not fit
  has more nonzeros than the largest certified block.
- **MS-N07.** `\TcShiftFraction` names the 0.9 of E3(c) as the shift-placement fraction; the check says so;
  `\TcCasesRecorded` keeps the count of recorded rows (one duplicate) next to `\TcCases`.
- **MS-N13.** `\VaMatricesGenerated` (matrices with and without an Arb reference) next to `\VaMatrices`
  and `\VaNoRef`; `\TbN` gives n for the E3(b) ratios, whose range is now rounded outward.
- Statement check "same computational source" falls back to the records' `source_hash` when the
  recorded commits are not in the repository (rewritten history).

### Added (review of 2026-10-21)
- `tests/test_adversarial.py`: the 14 attack cases of the review (7 failed on the reviewed HEAD).
- `tests/test_review_fixes.py`: one or more regression tests per finding above.

### Added (E9 named sectors, branch feat/e9-named-sectors; changes the computational source hash)
- `e9_pyrochlore.py` step S1N: the lowest eigenvalue of the two pair-spin sectors Kranitz & Penc name,
  S~ = (0,0,0,2,0,2) and (0,2,2,0,2,2), at the printed couplings (S^z = 0 block, ARPACK, floating point),
  recorded under `spin1_printed.named_sectors`; `make_numbers_apps.py` emits `\CrownOneNamedSectorA/B`
  from it (review B3-2) and checks that both reproduce the published value to 1e-5 and lie above the
  certified all-singlet eigenvalue. `tests/test_e9_named_sectors.py`. To be merged after the candidate
  run at source e707838f6852d474 has finished (the step adds about half a minute to E9).

### Added (manuscript v4 macros, branch feat/v4-macros; post-processing only, source hash unchanged)
- `make_numbers.py` / `make_numbers_apps.py` emit every macro the v4 scaffold used from its hand-written
  `generated/numbers_todo.tex`, each with a comment naming the record and field it comes from:
  `\AKLTCritMinDown`, `\AKLTMaxExcessObserved`, `\AKLTExcessCheck` (the check's own threshold, RA-23),
  `\TbRatioMinDown`, `\TbRatioMaxUp`, `\TbPairs`, `\TcCasesUnique` (MS-N13, MS-N07), `\CSoneSzMaxTable`,
  `\CkTwentyTwoSoneTable`, `\CnnzSquareJtwoNotChecked` (MS-N04/05/06), `\NtestsRun`,
  `\NtestsAddedAfterRun` (MS-N02), `\FnPoneOrPtwo`, `\FnBothMissed`, `\FnBothMissedList` and the
  "P1 or P2" column of Table e1 (MS-H02), `\CrownOneRingEzero`, `\CrownOneRingRadiusExact`,
  `\CrownOneRingRadiusFP` (RA-09), `\CrownOneMinClassDim`, `\CrownOneBigClassDim`,
  `\CrownOneBigClassAbove` (MS-N09), `\SourceHashNew` (IMP-26), `\FlintLibVersion` (MS-N12),
  `\ExactCheckerLines` (+ per-file, MS-N11), `\DecSquareNoSymDim` (from the E10 run.log),
  `\SymveigPtwoOutside` (from docs/symveig_audit.md, RA-32), the literature values
  `\CrownHalfReducedPaper`, `\CrownHalfHeadlinePaper` (marked as such, B2-4/RA-21) and, when the record
  carries them, `\MutGapsClosedStamped` (MS-N03) and `\CrownOneNamedSectorA/B` (B3-2).
- `generated/numbers_todo.tex` is now written by `make_numbers_apps.py`: the macros whose value no record of
  the run contains, as `\todo{reason}` (nothing is substituted); empty when every macro has a record.
- `make_numbers.py` checks that no macro is defined twice across the generated files; the environment
  macros that both scripts know are written by `make_numbers_apps.py` with `\providecommand` (before this,
  `\FlintCVersion`, `\PythonFlintVersion` and `\CertifyEdVersion` were defined twice, which would have stopped LaTeX with "command already defined"; found by the duplicate-macro check, no LaTeX build was run).
- `tests/test_make_numbers_helpers.py`: the helpers behind these macros (code-line counter, pytest collection
  parser, duplicate detection, P1-or-P2 verdicts, E10 log and symveig-audit parsers,
  pending-file writer).

### Added (scope framework, branch feat/scope-framework)
- **Universal producer import** (`certify_ed.io`, `docs/producer_bundle.md`): eigenvalues, eigenvectors,
  operators and sector bases from .npy, .npz (dense, `save_npz`, CSR/COO triplets), HDF5, CSV/TSV/text,
  JSON and Matrix Market; matrix-free operators from a Python file (opt-in `--allow-code`); the
  *producer bundle* directory convention (`eigenvalues.*`, `eigenvectors.*`, `operator.*`, `basis.*`,
  `spec.json`, `meta.json`) so that any code is certified by writing files; `claim_from_file`,
  `claim_from_bundle`, `write_bundle`; non-finite input is rejected at the door (review IMP-21).
  `certify-ed certify --from DIR` / `certify-ed import DIR`.
- **Statement selection and presets** (`presets.py`, `docs/presets.md`): `Options.preset / only / skip /
  narrowing_reason / budget_seconds / budget_mb`; presets `lab-minimal`, `benchmark`, `proof-lemma`,
  `whole-spectrum`, `labels`, `full` as data; `certify-ed presets`. Budgets turn statements into
  NOT_CHECKED (reason `BUDGET_TIME` / `BUDGET_MEMORY`), never into exceptions.
- **Scope block** (`scope.py`, `docs/SCOPE_MATRIX.md`): every certificate (schema 2.1) carries `scope`
  with the preset, the statements in scope, the narrowed statements with the user's reason, and the
  out-of-scope quantities with reason codes; `Certificate.summary()` prints it.
- **New statements, all UNVERIFIED (D-12) with proofs written before the code (D-46, `docs/theory.md` §9):**
  `S7.thermodynamics` (Prop 16: interval bounds on Z, F, U, C from a complete whole-spectrum enclosure,
  dyadic interval arithmetic with enclosed exp/ln, `ivl.py`, `thermo.py`); S4 extensions (Prop 15:
  batches of observables, two-point functions, static structure factors, `correlations.py`,
  `Options.correlations`); `S8.degeneracy` (Prop 14: number of eigenvalues in each cluster by bracketing
  inertia counts, `degeneracy.py`). `spectrum.intervals_from_enclosure` / `merge_intervals` expose the
  Prop 9 interval list, recorded as `spectrum_enclosure` for whole-spectrum claims.
- Tests: 58 (import round trips), 18 (presets/scope), 8 (S7), 8 (S4 extensions), 8 (S8).

### Fixed
- **Exact 'k lowest' re-check (S6, `exact_recheck_lowest`, `exact_recheck_sector_lowest`).** A component
  of overlapping residual intervals with m ≥ 2 claimed values was taken to hold m eigenvalues
  whenever one count above it agreed. Residual intervals give *an* eigenvalue each, not distinct
  ones: two copies of the first-excited vector, claimed as "the two lowest", were certified although
  the ground state is missing. Such components are now bracketed by counts on both sides, and their
  eigenvalues are certified inside the bracket ("level enclosure"). Single-value components keep
  their residual interval: the counts on the two sides pin the index. Results carry `enclosures`
  and `next_lower_bound`. FAILED now means an unclaimed eigenvalue lies under a claimed one; a cut
  multiplet is INCONCLUSIVE (it was FAILED). `checker.certify_by_counts` is shared by both routines.
- **Outward rounding of reported endpoints.** μ − r, μ + r, σ ∓ β, separations and cut comparisons
  are formed with `fp.sub_down` / `fp.add_up` (one ulp outward). Round-to-nearest could place an
  endpoint half an ulp inside the exact interval. This affects enclosures and their records,
  cluster merging, S3 margins and gaps, Kato–Temple, S4 separations, sector assembly and
  whole-spectrum intervals.
- `inertia_exact` checks that the permuted LDLᵀ factor is exactly unit lower triangular.

### Changed
- AKLT: thresholds are placed just above the floating-point η² (margins 1e-9, 1e-7, 1e-5). The
  certified η now exceeds the published values by 2e-11 to 2e-9 (it was about 2e-6). On the cross
  and the decorated square, the remaining excess comes from the factorization term t′ − t (β/μ, β/ν).
  Reported η and the square-octagon sum are floats rounded upward from the exact rationals.
- Pyrochlore S = 1/2 crown: a ladder of shifts (relative margins 1e-10, 1e-9, 2e-8). The certified bound
  is λ_min ≥ −0.550158299 (it was −0.550158314), within 1e-9 of the Lanczos value.
- Provenance: the post-processing scripts left the computational fingerprint; the test suite has its
  own `tests_hash`, and `--resume` re-runs E6 when the tests changed. E6 is classified
  automatically after it runs.
- `make_numbers_apps.py`: E6 table, E8 width-scaling figure (`fig_widths.pdf`), criterion values for
  AKLT, run-manifest table, tighter statement checks.
- Tests: 370 before the review fixes; 7 of them close the 8 test gaps found by the mutation analysis of
  the run of 2026-10-21: 2×2 pivots with det > 0, the Dekker splitter, exact residuals with huge entries,
  the round-up probe, the SU(2) check in mixed Pauli letters, the sector compression diagnostic and the
  TFIM bonds. The review fixes add `test_adversarial.py` (14) and `test_review_fixes.py` (see the
  README for the current count).

## [3.0.0] - 2026-10-18

One package for the combined CERTIFY-ED / symveig paper.

### Added
- **Exact re-check layer** (no floating-point assumption), for every local space:
  - `checker.exact_matrix`: the declared model in the scaled basis D⁻¹HD with integer weights;
    spin-S binom(2S, k), bosons k!.
  - `exact_eigenpair_radii`, `exact_recheck_lowest`.
  - `exact_sectors.exact_recheck_sector_lowest` and `exact_count_below`: symmetry sectors with
    Gaussian characters, non-abelian groups allowed when the irreps met are one-dimensional.
  - `checker.inertia_exact`: lean exact LDLᵀ check.
- **S6.exact_recheck** statement (`Options(exact_recheck=True)`).
- **`certify_ed.spectrum.enclose_spectrum`**: whole-spectrum enclosure by sectors (formerly
  symveig). Counts are certified by the cluster theorem, conservation and Hermiticity are checked
  exactly, and rounding bounds use |A||x|.
- **`certify_ed.compat.symveig`**: the symveig 1.0.0 API on the new engine, with symveig's tests ported.
- **`certify_ed.apps.aklt`**: certified Pomata–Wei criteria for five lattices, one pipeline;
  kernel-lemma checks.
- **`certify_ed.apps.pyrochlore`**: exact S = 1/2 crown bound (now through `exact_count_below`);
  S = 1 printed-coupling check; pair-spin landscape.
- **CLI** `certify-ed` (`env`, `certify`, `spectrum`, `list`, `test`, `reproduce`, `numbers`) and
  `run.py`.
- Experiments E7 (exact benchmarks), E8 (symveig comparison), E9 (pyrochlore), E10 (AKLT).
  `make_numbers_apps.py` reads their records.
- `fp.probe_float_environment` is recorded in every experiment's environment.
- Run-time float probe, total-spin and momentum label certification, manufactured and metamorphic
  tests, mutation analysis E6 with recheck and inspection tooling (2026-10-17).
- **Provenance** (`certify_ed.provenance`): every experiment record and every reproduction step
  carries a fingerprint of the computational source (`certify_ed/` and `experiments/` without the
  command line, test helpers and this module). Both numbers scripts check that all records they read
  share one fingerprint, with no uncommitted changes.
- `certify-ed reproduce` / `run.py`:
  - `--jobs N` runs experiments in parallel, longest first; E2, E6 and E8 always run alone;
  - `--resume` keeps the steps of an interrupted run that finished with the same source;
  - unknown experiment names are rejected.
- `apps.pyrochlore.spin_one.printed_coupling_check` also re-derives the six-ring energy in exact
  arithmetic (scaled spin-1 basis). The S = 1 contradiction now needs no floating-point assumption.
- Tests (356) for every AKLT lattice geometry, the spin-1 pair-spin tools, the sector-decomposition
  audits (coverage, orthogonality, trace moments, certified spectral shift) and the driver
  (parallel jobs, resume, numbers on partial runs), and for six gaps found by mutation analysis of the
  final tree (internal builder-disagreement path, 1-D eigenvectors, dense non-Hermitian producers,
  fermionic translation sign, float-model caveat, square-lattice default).
- Tooling: `ruff check .` is clean (configuration in `pyproject.toml`). CI runs lint, builds the
  wheel, runs the suite on Python 3.10–3.12 and does a quick reproduction. SPDX license metadata.

### Changed
- Package metadata in `pyproject.toml` (`setup.py` removed). Version 3.0.0. Authors: Sarang Vehale,
  Ritu Goel.
- Tests moved from `tests/v2/` to `tests/`. Helpers moved to `certify_ed.testing`.
- `Certificate` is the v2 certificate class (`CertificateV2` is kept as an alias).
- Experiment results can be directed with `CERTIFY_RESULTS`.

### Removed (moved to `legacy/`)
- v1 modules (`hamiltonian`, `oracles`, `certificates`, `observables`, `symmetries`), `validators/`,
  v1 tests and examples. Tag `v1.0.0-baseline`.
- symveig 1.0.0 (kept unchanged in `legacy/symveig_1_0_0` for E8).
- The script versions of the applications and their certificate files (2026-10-17).

### Fixed
- The numbers step compared whole commits, so a run that crossed a commit touching only the command
  line failed its provenance check. It now compares computational sources.
- The E2 check "the N=22 momentum block has the largest S1 time" compared runs instead of blocks. It
  failed when the same block's run with S3 took longer than its run without.
- Mutation workspaces now include `examples/`, which the test suite runs.

### Fixed (found while merging symveig; see docs/symveig_audit.md)
- Merged-interval multiplicities are reported as certified counts only when proved.
- Sector decompositions are rejected when the quantity is not conserved.

## [2.0.0.dev0] - 2026-10-04
v2 verification core S1–S5, sector assembly, importers, certificate v2, experiments E0–E5.

## [1.0.0] - 2026-05
CERTIFY-ED v1 (arXiv:2605.11787), now in `legacy/v1`.
