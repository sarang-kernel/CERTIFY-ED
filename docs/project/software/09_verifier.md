# The independent C verifier

A certificate written by certify-ed can be re-checked by a second program that shares no code with the package:
`verifier/src/cev.c`, a C11 program of about 1 500 lines that needs only the C standard library (roadmap item B-01).
The reason for a second program in a second language is that a reader who doubts the Python package can check what a
certificate records without trusting that package: an error in the hashing, the JSON handling, the rounding or the
logic of S2 and S3 would have to be made twice, independently, to pass unnoticed. This document describes what the
verifier checks, what it does not check, how it is built and how it was validated. Facts are from
`code/certify-ed/verifier/README.md` and `code/certify-ed/verifier/validation/RESULTS.md`.

## What it recomputes

The verifier has its own JSON parser and canonical serialiser, its own SHA-256, its own reader of ZIP and `.npy` files
(stored and deflated members), its own exact decimal conversion, and its own evaluation of every bound in directed
rounding (`fesetround` on IEEE 754 binary64). From the certificate and, optionally, the operator file it runs these
groups of checks. The hash chain (H1 to H5): the body hash over the canonical JSON, the hash of every side-file array,
the hash of the stored eigenvectors against the claim, the hash of the spec record, and the canonical CSR hash of the
operator file against the certificate (only when the operator hash matches does the verifier recompute anything from
the operator). Internal consistency (C1 to C7): eigenvalues equal bit for bit between body and side file, shapes and
types, every enclosure containing μ ± r in directed rounding, the exported arrays carrying the recorded radii plus the
model shift, group records consistent, enclosures claimed distinct pairwise disjoint, and scope statuses equal to the
statements. The recorded cluster arithmetic (G checks): every term of Theorem 2 and of the sharper bound ε\*
recomputed from the recorded inputs, and the group radius at least min(ρ, ε\*). Recomputation from the operator and
the stored vectors (R2 to R5): the residual norm and the Theorem 1 radius of every pair as an interval in directed
rounding, the orthonormality defect η from error-free dot products, and the cross term. The underflow scan (U1) of the
INT-01 hypothesis. The logic of S2 and S3 (S2 and S3 checks): a CERTIFIED S2 is permitted only if every enclosure is
distinct and finite, the largest radius is within the accuracy target, there is no underflow finding and the
floating-point probe held; a CERTIFIED S3 must follow from the recorded inertia counts, the backward error, the
ordering of the enclosures and the gap; for a FAILED status the verifier reports whether it reaches the same
conclusion from the stored data (REPRODUCED or NOT_REPRODUCED).

## What it does not check

The verifier does not check S1, because deciding whether the operator is the declared model would mean re-implementing
the reference builders of `certify_ed.construct`, which is the code it is meant not to share; it checks only the
hashes of the spec and of the operator file. It does not re-derive S4 to S8. It does not redo the inertia counts,
because the factorizations behind them are not stored in the certificate; it checks the recorded counts against the
recorded enclosures. Complex operators and vectors get the hash checks but not the recomputation. A radius lowered by
a forger but still above the true residual is not flagged, because the verifier tests soundness, not equality with
certify-ed's number. Without the stored eigenvectors (they are saved only with `Options(store_eigenvectors=True)`) or
without the operator file, the recomputation checks are reported as SKIP.

## Building and running

```
python verifier/build.py                                  # writes verifier/build/cev or cev.exe
verifier/build/cev CERT.json --operator operator.npz --report out.txt --json out.json
```

`build.py` takes the compiler from `--cc`, then the environment variable `CEV_CC`, then `gcc`, `clang` or `cc` on the
path, and compiles with `-frounding-math -ffp-contract=off`, which directed rounding requires (no contraction into
fused multiply-add). The binary refuses to run when 1 + 1e-30 does not round up under upward rounding, so a build
without working directed rounding cannot produce a passing report. The exit code is 0 when every check passed (SKIP
lines allowed), 1 when at least one check failed, and 2 when the inputs could not be read. The test module
`tests/test_verifier.py` builds and runs the verifier and is skipped when no compiler is found.

A toolchain note explains why the verifier is in C. A Rust toolchain was installed first (conda-forge `rust` 1.99.1,
target `x86_64-pc-windows-msvc`), but the workstation has no MSVC linker, self-contained linking is not supported on
that target, and the GNU-target standard library was not available. The verifier was therefore written in C and
compiled with the MinGW gcc 14.2 that ships with GNU Octave.

## Validation

`verifier/validation/run_validation.py` runs certify-ed of the checkout on every corpus case and conformance fixture
with the eigenvectors stored (`verifier/make_validation_set.py`, the only script of the folder that imports
`certify_ed`), runs the verifier on every certificate with the bundle's operator, and forges 20 certificates in four
ways each: one pair's radii divided by 1e4 with every hash resealed (only the recomputation can catch it), one array
hash altered with the body resealed, the body hash altered, and a non-CERTIFIED S2 or S3 set to CERTIFIED with the
body resealed. On `dev/3.2` on 31 October 2026, compiled with MinGW gcc 14.2 and `-O1 -frounding-math
-ffp-contract=off`, the results were as follows.

| Quantity | Count |
|---|---|
| bundles | 63 (60 corpus cases and 3 conformance fixtures) |
| refused by certify-ed (NaN inputs, no certificate written) | 8 |
| certificates passing the verifier with exit code 0 | 55 of 55 |
| S2 or S3 statuses set to FAILED by certify-ed and reproduced by the verifier | 18 of 18 |
| forged certificates caught (radius, array hash, body hash, status) | 20/20, 20/20, 20/20, 20/20 |

The canonical-JSON serialiser was compared byte for byte with Python's `json.dumps` on 3 011 random doubles (uniform
values, random bit patterns, subnormal numbers, the boundaries at 1e16 and 1e15, decimal ties) and on strings with
escapes and non-ASCII characters (`tests/test_verifier.py::test_canonical_json_matches_python`). The validation was
run on the 3.2 development branch on corpus cases and conformance fixtures. None of the experiment scripts E0 to E10
calls the verifier, so the evidence runs contain no verifier step.
