# verifier: an independent re-check of CERTIFY-ED certificates

`verifier/src/cev.c` is a second-language verifier (C11, about 1 500 lines, no dependencies beyond the C
standard library) that re-checks the arithmetic claims a CERTIFY-ED certificate (`certify-ed/certificate/2.1`)
records, from the certificate's own stored data. It shares no code with `certify_ed`: it has its own JSON parser
and canonical serialiser, its own SHA-256, its own ZIP/`.npy` reader (stored and deflate members), its own exact
decimal conversion, and its own directed-rounding evaluation of every bound (`fesetround`, IEEE-754 binary64).
Roadmap item B-01.

```
python verifier/build.py                       # -> verifier/build/cev[.exe]   (gcc, clang or cc; CEV_CC overrides)
verifier/build/cev CERT.json --operator operator.npz [--report out.txt] [--json out.json] [--quiet]
```

Exit code 0: every check passed (SKIP lines allowed). 1: at least one FAIL. 2: the inputs could not be read.
`--report` writes the plain-text report, `--json` a summary `{verdict, exit_code, counts, checks: [{id, status,
detail}]}`. Statuses per check: PASS, FAIL, SKIP (the check needs an input that is not available), REPRODUCED /
NOT_REPRODUCED (for a status certify-ed set to FAILED: whether the verifier reaches the same conclusion from the
stored data), INFO.

## What the certificate must carry

| input | where | used for |
|---|---|---|
| the JSON body | `CERT.json` | every check |
| the array sidecar (`eigenvalues`, `radius`, `existence_radius`, `residual_bound`) | `CERT.arrays.npz`, named in `arrays.sidecar` | hashes, consistency, S2/S3 logic |
| the claimed eigenvectors (`eigenvectors` in the sidecar) | written by `Options(store_eigenvectors=True)`; not stored by default | recomputation of the Theorem-1 radii (R2, R3), of the Gram defect (R4) and of the cross term (R5); the underflow scan (U1) |
| the operator | not stored in the certificate; passed with `--operator` (the producer bundle's `operator.npz`, scipy CSR); its canonical-CSR hash, or the hash of its Hermitian part when `hermitian_part` is set, must equal `subject.operator.matrix_sha256` (H5), else the recomputation is not performed | R2-R5 |

Without the vectors or the operator, the verifier checks the hash chain, the per-pair records, the recorded
cluster arithmetic and the S2/S3 logic, and says so (SKIP lines on R2 and U1). It never rebuilds the operator from
the model specification (that would re-implement `certify_ed.construct`, and the point is to share nothing).

## Checks

| id | what is recomputed | FAIL means |
|---|---|---|
| H1.body_sha256 | SHA-256 of the canonical JSON of the body without `integrity` (`certify-ed-canonical-json-1`: sorted keys, `,` `:` separators, non-ASCII verbatim, floats as Python's shortest round-trip repr, produced here by exact integer arithmetic, not by `printf`) | the body was altered after it was sealed |
| H2.`<array>` | SHA-256 of dtype string + `json.dumps(list(shape))` + C-order bytes of each sidecar member; Fortran-ordered members are transposed; every member must be listed and every listed member present | an array was altered, or the sidecar does not belong to the certificate |
| H3.eigenvectors_sha256 | hash of the stored vectors vs `claim.eigenvectors_sha256` | the vectors are not the claimed ones |
| H4.spec_sha256 | hash of the canonical JSON of `subject.spec` vs `subject.spec_sha256` | the specification record was altered |
| H5.operator_sha256 | canonical CSR (sorted indices, summed duplicates, no explicit zeros) of `--operator`, or of its Hermitian part `(A + A^T)/2` when `hermitian_part` is set, vs `subject.operator.matrix_sha256` | the operator file is not the certified one (R2-R5 are then not run) |
| C1, C2 | `claim.eigenvalues` == `arrays.eigenvalues` bitwise; `n_eigenvalues`; shapes and dtypes of the members vs the `arrays` block; vector length vs operator dimension | internal inconsistency |
| C3.enclosure_records | one record per pair; `claimed` == eigenvalue; `lower` <= RD(mu - r) and `upper` >= RU(mu + r); `existence_radius` <= `radius` | an enclosure does not contain mu +- r, or the radii are out of order |
| C4.arrays_vs_records | `arrays.radius[i]` >= RU(record radius + model shift), same for the existence radius; `arrays.residual_bound` == record | the exported arrays do not carry the recorded radii plus the model shift |
| C5.G`<n>`.members | every member record of a group has the group's radius, method, distinctness and size | group/record mismatch |
| G`<n>`.R_fro, .eta, .delta, .rho, .spread, .eps_star, .radius | the recorded cluster arithmetic (Theorem 2 / 2'): R_fro >= RU ||(residual bounds)||_2; eta < 1; delta >= RU(1 - sqrt(1 - eta)); rho >= RU(R_fro + (||H|| + ||Lambda||) delta) with ||H|| = `subject.operator.norm_upper_bound`; spread >= RU(max mu - min mu); eps_star >= RU(R_fro/sqrt(1-eta) + XR/(1-eta) + delta spread/sqrt(1-eta)); group radius >= min(rho, eps_star) | a recorded cluster bound is smaller than the formula it claims to evaluate |
| C6.disjoint | the union intervals of groups marked distinct are pairwise disjoint and finite | distinctness claimed for overlapping enclosures |
| R2.residual_bound | for every pair, the interval of ||A x - mu x||_2 from the operator and the stored vector, all operations in directed rounding (upper and lower bound); `residual_bound` must be >= the upper bound | a stored residual bound is below the verifier's upper bound: below the lower bound it is unsound; between the two it is not confirmed |
| R3.existence_radius | same for the Theorem-1 radius RU(||r|| / ||x||), ||x|| rounded down | as above |
| R4.G`<n>`.eta | ||X_g^T X_g - I||_F as an interval from exact dot products (TwoSum/TwoProduct error-free transformations, error terms summed with directed rounding); recorded eta must be >= the upper bound | the recorded orthonormality defect is too small |
| R5.G`<n>`.XR | ||X_g^T R_g||_F as an interval from the residual intervals; recorded XR must be >= the upper bound | the recorded cross term is too small |
| U1.underflow_scan | nonzero magnitudes below `arithmetic.underflow_check.threshold` (2^-900) in the stored eigenvalues and vectors (INT-01 hypothesis) | informational; a finding forbids CERTIFIED S2/S3 |
| S2.status | CERTIFIED is permitted only if every enclosure is distinct and finite, max `arrays.radius` <= the accuracy target, no underflow finding (recorded or found), and `runtime_probe.standard_model_hypotheses_hold` | S2 CERTIFIED where the stored data do not permit it |
| S2.failed | for a FAILED S2: max `arrays.radius` > target (REPRODUCED/NOT_REPRODUCED); with the operator, `S2.failed.independent` reports the largest directed-rounding lower bound of a Theorem-1 radius, which refutes the claim independently of certify-ed's rounding terms when it exceeds the target | - |
| S3.inertia, .beta, .bound, .ordering, .gap, .window_record (kind `lowest`) | a conclusive count at `top_shift` with `negative` == k; `beta_effective` >= RU(count beta + model shift); `next_eigenvalue_lower_bound` <= RD(sigma - beta); every RU(mu_i + radius_i) < that bound and every enclosure distinct (Lemma 4 / Corollary 5); `gap_lower_bound` <= RD(bound - max upper end); the certified window record | S3 CERTIFIED where the counts and the enclosures do not permit it |
| S3.inertia, .beta, .ordering, .window_record (kind `window`) | counts at both edges, n = nu_b - nu_a == `count_from_inertia`; betas; the number of enclosures strictly inside (RU(a + beta_a), RD(b - beta_b)) equals n with every enclosure distinct; the certified window lies inside | as above |
| S3.status (kind `all`) | every enclosure distinct and k == dimension | as above |
| S3.hypotheses | a CERTIFIED S3 inherits the floating-point hypotheses of S2 | as for S2 |
| S3.failed | `lowest`: the count at `witness_shift` minus the enclosures reaching below it gives >= 1 unclaimed eigenvalue below the top claim, with the collar below the top enclosure; `window`: nu_b - nu_a minus the enclosures touching the window >= 1 | - |
| C7.scope_statuses | `scope.in_scope` statuses equal the statements' | inconsistent record |

Sector certificates (`subject.sector` set): S3 is re-checked with the pre-lift radii of the sector block, as
certify-ed does; the `sector_lift_shift` is used in C4.

## What it does not check

- S1 model fidelity: whether the operator is the declared model needs `certify_ed.construct`; the verifier only
  checks the hashes of the specification and of the operator file. The physics of the model is out of reach.
- S4 eigenspaces, S5-S8: not re-derived (the S4 Davis-Kahan arithmetic and the S8 bracketing are candidates for a
  later extension; their inputs are in the record).
- Theorem 1 and the inertia counts themselves: the verifier re-evaluates ||A x - mu x|| / ||x|| and the recorded
  formulas, not the LDL^T factorizations behind `counts[].negative` and `counts[].beta` (those need the matrix
  factorization; a factorization-free re-check is not possible from the record).
- A vectorless window count (C-02, `count_in_window` without enclosures) and the `subset` kind are reported as
  SKIP / INFO.
- Complex operators and vectors (`<c16`): the hashes are checked, the recomputation (R2-R5) is skipped.
- A radius lowered by a forger but still above the exact residual is not an error and is not flagged: the verifier
  tests soundness, not equality with certify-ed's number.

## Validation (`verifier/validation/run_validation.py`)

The harness (standard library only) runs certify-ed of the checkout on every corpus case and conformance fixture
with `Options(store_eigenvectors=True)` (`verifier/make_validation_set.py`, the only script here that imports
`certify_ed`), runs the verifier on every certificate with the bundle's operator, and writes
`verifier/validation/RESULTS.md` and `results.json`. Mutation tests forge 20 certificates four ways each:

- `radius`: one pair's `existence_radius` and `residual_bound` divided by 1e4 in the record and the arrays, every
  hash resealed (only R2/R3 can catch it);
- `array_hash`: one array hash altered, body resealed (H2);
- `body_hash`: the body hash altered (H1);
- `status`: a non-CERTIFIED S2 or S3 set to CERTIFIED in the statement and the scope block, body resealed
  (S2.status / S3.*).

Results on dev/3.2 (2026-10-31, MinGW gcc 14.2 on Windows, `-O1 -frounding-math -ffp-contract=off`):

| quantity | count |
|---|---|
| bundles | 63 (60 corpus + 3 conformance) |
| refused by certify-ed (NaN inputs, no certificate) | 8 |
| certificates verified, exit 0 | 55 of 55 |
| S2/S3 statuses certify-ed set to FAILED | 18, all REPRODUCED |
| mutants caught (radius / array hash / body hash / status) | 20/20, 20/20, 20/20, 20/20 |

The canonical-JSON serialiser was compared byte for byte with `json.dumps` on 3 011 random doubles (uniform,
random bit patterns, subnormals, 1e16 / 1e15 boundaries, decimal ties) and on strings with escapes, non-ASCII and
non-BMP characters (`tests/test_verifier.py::test_canonical_json_matches_python`).

## Toolchain note

A Rust toolchain (conda-forge `rust` 1.99.1, target `x86_64-pc-windows-msvc`) was installed for this item but has
no linker on the development machine (no MSVC `link.exe`; `-C link-self-contained` is not supported on the MSVC
target), and the `windows-gnu` standard library was not available; the verifier is therefore C, compiled with the
MinGW gcc 14.2 shipped with GNU Octave. `build.py` finds gcc, clang or cc on PATH (or `CEV_CC`); the test module
`tests/test_verifier.py` skips when none is found. Directed rounding requires `-frounding-math` and
`-ffp-contract=off`; the binary refuses to run when `1 + 1e-30` does not round up under FE_UPWARD.
