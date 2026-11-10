# The independent C verifier

`verifier/src/cev.c` of the code repository is a second program, written in C, that re-checks the arithmetic claims
recorded in a CERTIFY-ED certificate (schema `certify-ed/certificate/2.1`) from the certificate's own stored data. It
was built for roadmap item B-01 on the branch `dev/3.2`. Sources: `code/certify-ed/verifier/README.md` (what it checks
and does not check), `code/certify-ed/verifier/validation/RESULTS.md` and `results.json` (the validation of 31 October
2026), and `report/15_completion_plan.md`. It should not be confused with experiment E0 of the evidence runs, which
validates certify-ed's own checks against Arb, QuSpin and a Kronecker-product construction.

## Why a second program

A certificate is produced by the same code whose correctness it is meant to support. A defect in that code (a wrong
rounding direction, a missing term, a status set without the evidence that permits it) or a deliberate change to a
certificate after it was written would not be caught by re-running the same code. A second implementation that shares no
code with `certify_ed` and re-derives the recorded arithmetic from the stored numbers addresses both. The language was
chosen for practical reasons: a Rust toolchain was installed for the item but has no linker on the development machine
(no MSVC `link.exe`, and the GNU target's standard library was not available), so the verifier is C11, compiled with the
MinGW gcc 14.2 that ships with GNU Octave 10.3 (`verifier/README.md`, "Toolchain note").

## What it is

The program has about 1 500 lines and no dependency beyond the C standard library. It has its own JSON parser and
canonical serialiser, its own SHA-256, its own reader for ZIP archives and `.npy` arrays (stored and deflated members),
its own exact decimal conversion, and its own evaluation of every bound in directed rounding: the upper end of a
quantity is computed with the processor's rounding mode set toward plus infinity and the lower end toward minus infinity
(`fesetround`). Directed rounding needs the compiler flags `-frounding-math` and `-ffp-contract=off`, and the binary
refuses to run when $1 + 10^{-30}$ does not round up under the upward mode. It never rebuilds the operator from the
model specification, because that would re-implement `certify_ed.construct`; the operator is passed as a file and
identified by its hash.

```
python verifier/build.py
verifier/build/cev CERT.json --operator operator.npz --report out.txt --json out.json
```

The exit code is 0 when every check passed (SKIP lines allowed), 1 when at least one check failed, and 2 when the inputs
could not be read. Each check reports PASS, FAIL, SKIP (an input it needs is missing), INFO, or, for a status that
certify-ed set to FAILED, REPRODUCED or NOT_REPRODUCED (whether the verifier reaches the same conclusion from the stored
data).

## What it checks

The checks fall into six groups (the full table is in `verifier/README.md`). The hash chain, H1 to H5: the SHA-256 of
the canonical JSON of the certificate body, of every array in the sidecar file, of the stored eigenvectors, of the model
specification, and of the operator file in a canonical sparse form (H5 must pass before any recomputation uses the
operator). Internal consistency, C1 to C7: the claimed eigenvalues equal the stored arrays bit for bit, every enclosure
record contains $\mu \pm r$ with outward rounding, the exported radius arrays carry the recorded radii plus the model
shift, every member of a cluster group has the group's radius, the union intervals of groups marked distinct are
disjoint and finite, and the statuses of the scope block equal those of the statements. The recorded cluster arithmetic
of Theorems 2 and 2', the G rows: $\lVert R \rVert_F$, $\eta < 1$, $\delta$, $\rho$, the spread and $\varepsilon^*$ are
re-evaluated from the recorded inputs in directed rounding, and a recorded value smaller than the formula it claims to
evaluate fails. Recomputation from the operator and the stored vectors, R2 to R5: an interval for
$\lVert Ax - \mu x \rVert$ and the Theorem 1 radius in directed rounding, the orthonormality defect
$\lVert X_g^T X_g - I \rVert_F$ and the cross term $\lVert X_g^T R_g \rVert_F$ from error-free transformations; a stored
bound below the verifier's lower bound is unsound, and one between its lower and upper bounds is not confirmed. The
underflow scan U1 (informational; a finding forbids CERTIFIED S2 and S3). And the status logic of S2 and S3 for the
claim kinds `lowest`, `window` and `all`: CERTIFIED is permitted only if the stored data permit it (Lemma 4 and
Corollary 5 for the counts), and a FAILED status is reproduced from the stored data when possible. For sector
certificates S3 is re-checked with the pre-lift radii of the sector block, as certify-ed does.

## What it does not check

The verifier does not address S1: whether the operator is the declared model needs `certify_ed.construct`, so only the
hashes of the specification and of the operator file are checked. It does not re-derive S4 to S8. It does not repeat the
$LDL^T$ factorizations behind the inertia counts; it re-evaluates the recorded counts and backward errors through the
formulas of Lemma 4 and Corollary 5, which no factorization-free re-check can replace. Vectorless window counts and the
`subset` kind are reported as SKIP or INFO; complex operators and vectors get the hash checks only. A radius lowered by
a forger but still above the exact residual is not flagged, because the verifier tests soundness, not equality with
certify-ed's number. A PASS is therefore a statement about the internal arithmetic of the certificate, not an additional
certification of the physics: a verifier PASS on a certificate whose S1 is FAILED still concerns the producer's operator
only (`code/certify-ed/docs/SCOPE_MATRIX.md`).

## Validation

**What was done.** The script `verifier/validation/run_validation.py` (standard library only) runs certify-ed on every
case of the test corpus and on the conformance fixtures with the eigenvectors stored
(`Options(store_eigenvectors=True)`, through `verifier/make_validation_set.py`, the only script of the folder that
imports `certify_ed`), then runs the verifier on every certificate with the bundle's operator. Then it forges
certificates: 20 certificates are each altered in four ways, and the verifier must fail each forgery. The four forgeries
are designed so that exactly one group of checks can catch each. In `radius`, one pair's existence radius and residual
bound are divided by $10^4$ in the record and in the arrays and every hash is resealed, so only the recomputation R2 and
R3 can catch it. In `array_hash`, one array hash is changed and the body resealed (H2). In `body_hash`, the body hash is
changed (H1). In `status`, a statement S2 or S3 that was not CERTIFIED is set to CERTIFIED in the statement and the
scope block and the body resealed (S2.status or the S3 checks).

**Instrument checks.** The canonical JSON serialiser was compared byte for byte with Python's `json.dumps` on 3 011
random doubles (uniform values, random bit patterns, subnormal numbers, the boundaries $10^{16}$ and $10^{15}$, decimal
ties) and on strings with escapes and non-ASCII characters
(`tests/test_verifier.py::test_canonical_json_matches_python`; the module has 7 tests and skips when no C compiler is
found). The FAILED statuses of certify-ed serve as positive controls for the S2 and S3 logic: the verifier must reach
the same conclusion.

**Results** (31 October 2026, on `dev/3.2`, MinGW gcc 14.2 on Windows, `-O1 -frounding-math -ffp-contract=off`):

| quantity | count |
|---|---|
| bundles | 63 (60 corpus cases and 3 conformance fixtures) |
| refused by certify-ed (NaN inputs, no certificate written) | 8 |
| certificates passed by the verifier (exit 0) | 55 of 55 |
| verifier exit 1 or unreadable input | 0 |
| S2 and S3 statuses that certify-ed set to FAILED | 18, all REPRODUCED |
| forgeries caught: radius, array hash, body hash, status | 20 of 20, 20 of 20, 20 of 20, 20 of 20 (80 of 80) |

`RESULTS.md` lists, for each forgery, the checks that caught it; for example, the radius forgeries were caught by
R2.residual_bound and R3.existence_radius, and a status changed from FAILED to CERTIFIED on the mislabelled-sector case
was caught by five S3 checks.

**What it shows and does not show.** On the 55 certificates of the corpus and the conformance fixtures, an independent
implementation in another language, with its own parsing, hashing and directed-rounding arithmetic, agrees that every
recorded radius, cluster bound and S2 or S3 status is permitted by the stored data, and it caught every one of the 80
forgeries. The corpus is the project's own test corpus, so the result shows agreement on those cases, not on every
certificate that certify-ed can produce, and the parts of a certificate listed in the previous section remain unchecked
by any second program.

## Reproducing

From the root of the code clone, with gcc, clang or cc on the path (or `CEV_CC` naming a compiler):
`python verifier/build.py`, then `python verifier/validation/run_validation.py`, which rewrites
`verifier/validation/RESULTS.md` and `results.json`; `python -m pytest tests/test_verifier.py` runs the unit tests.
