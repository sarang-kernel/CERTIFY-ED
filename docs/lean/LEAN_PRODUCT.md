# LEAN_PRODUCT.md: the Lean checks as part of the product (Wave 6, items N-04 to N-06)

This part of the re-verification joins the two programs and the theorems described in `CORE.md`, `ENCLOSURE.md` and
`COUNTING.md` where the roadmap `docs/project/roadmap/DEVELOPMENT_ROADMAP.md` (Wave 6) says the link pays most. It has three
products. The first is a one-command driver that runs the exact checker `cecheck` on any certify-ed certificate and
reports a verdict per statement (item N-04), together with an optional path in the code that records such a verdict
without basing any status on it. The second is a set of refinement theorems on Mathlib (item N-05): they show that
when the kernel-reducible checks of `CertifyCore` accept the decoded rational data of a certificate, the conclusions
of Theorem 1 and of Sylvester's law of inertia hold for those data. The third, the mechanical gaps of the
floating-point model (item N-06), was not reached; section 5 says what each needs. The trust levels N, K and K2 are
those of `docs/project/README.md`: N is a compiled Lean program, K a theorem accepted by the Lean kernel with no `sorry`, no
`native_decide` and no new axiom, and K2 the same theorem accepted again by the independent kernel nanoda. Numbers
produced here are in `lean/results/lean_product.json`; paths without a folder of this repository are
relative to `archive_v5/`.

A word on terms. A refinement theorem says that a program, or a function written to be evaluated, computes what a
mathematical statement needs: if the function returns true on some data, the statement holds for the same data. The
decoded data of a certificate are the stored binary64 numbers read as exact rationals; `CertifyCore/Kernel/Defs.lean`
keeps them as lists, a matrix as one list of (column, value) pairs per row.

## 1 How to run

The commands assume the environment of `lean/logs/lean_env.json` and the Python environment
`certifyed-311` with the certify-ed source on `PYTHONPATH`, as in `CORE.md` section 1.

```
cd lean
lake build cecheck CertifyMathlib.Refine.Residual CertifyMathlib.Refine.Small CertifyMathlib.Refine.Inertia
set CE=..\..\archive\archive_v5\code\certify-ed
set PYTHONPATH=%CE%
python Verify\cecheck_bundle.py CERT.json --bundle BUNDLE_DIR --json VERDICT.json
python -m pytest -q Verify\test_cecheck_bundle.py --basetemp <scratch dir>
lake env lean Drafts\LeanProduct\AxiomsLeanProduct.lean
lake env leanchecker CertifyMathlib.Refine
python Drafts\Counting\nanoda_counting.py --lean-bin <toolchain bin> --work <scratch dir> --theorems Drafts\LeanProduct\nanoda_theorems.txt --out <results.jsonl>
```

With the code branch `track/external-verifier` installed, the same check runs from certify-ed:

```
certify-ed verify CERT.json --bundle BUNDLE_DIR --external "python <path>\lean\Verify\cecheck_bundle.py {certificate} --bundle {bundle} --json {report}"
```

## 2 The verifier driver (N-04)

Why this step. `cecheck` re-checks a certificate in exact rational arithmetic, and on the validation set of `CORE.md`
section 5 it agreed with the C verifier on every row, but it could only be run through the scripts of the validation:
export a whole index, then run the checker on every exported directory. A user with one certificate had no command
to type, and certify-ed had no place to keep what an external checker said. The roadmap asks for both, under the
condition that no status of certify-ed may depend on the external verdict, because `cecheck` is a level N program
whose own correctness is not a theorem.

What was done. `Verify/cecheck_bundle.py` takes a certificate, its array sidecar (by default the one the certificate
names) and the operator, as a producer bundle (`--bundle`) or a SciPy `.npz` file (`--operator`). It writes the
directory `cecheck` reads with the functions of `export/export_certs.py` (the certificate byte for byte, the arrays
as raw little-endian bytes, the operator in canonical CSR form and replaced by its Hermitian part when the
certificate records `hermitian_part`), runs `.lake/build/bin/cecheck`, and turns its list of checks into one verdict
per statement. The exporter is trusted for nothing about the numbers, for the reason given in `CORE.md` section 2:
`cecheck` recomputes every SHA-256 the certificate records over the exported bytes. The verdicts are these.

| verdict | meaning |
|---|---|
| `CONFIRMED` | a CERTIFIED status, and every check `cecheck` makes for that statement passed, the integrity checks included |
| `NOT_CONFIRMED` | a CERTIFIED status for which at least one of those checks failed |
| `REPRODUCED` | a FAILED status whose recorded reason `cecheck` finds on the stored numbers |
| `REFUTED` | REPRODUCED, and an exact inertia count contradicts the claim itself (`CORE.md` section 5) |
| `NOT_REPRODUCED` | a FAILED status whose recorded reason `cecheck` does not find |
| `NO_CLAIM` | INCONCLUSIVE or NOT_CHECKED: there is nothing to confirm; the exact checks are listed all the same |
| `NOT_RECHECKED` | a statement `cecheck` does not check: S1, S4, S5, S6 and S7 (`CORE.md` section 3) |

The groups of checks behind each verdict are fixed in the script: the integrity group is H1 to H5, C1 to C4 and C7,
S2 uses E1 to E5, the G records and the S2 status checks, S3 its status and count checks, S8 the exact bracket counts,
and the Kato-Temple records form their own group. Every verdict has level N. The exit code is that of `cecheck` (0
no check failed, 1 a check failed, 2 the input could not be read), and 2 also when the export or the run fails.

How the instrument was checked. First, the export was compared byte for byte with `export/export_certs.py` on nine
certificates, and for the seven whose bundle holds an `operator.npz` the export with `--operator` was identical as
well, except for the manifest, which names the source of the operator. Second, the
mapping from checks to verdicts was run on the archived `cecheck` reports of the core part
(`lean/results/core/genuine/`) through a stand-in program that returns the archived report, and gave
the categories of `CORE.md` section 5. Third, the built `cecheck` was run through the driver on the same nine
certificates and on three forgeries, each of which changes one thing a checker must notice. The certificates were
produced by `verifier/make_validation_set.py` of the code at `fa86456` (branch `dev/3.2`, two documentation commits
after `a814d28`, computational source hash `cb73dec3e4433a3b`) for six corpus cases and the three conformance
fixtures; their statuses are those of `code/certify-ed/verifier/validation/results.json`. The test file
`Verify/test_cecheck_bundle.py` repeats the byte comparison, tests the mapping on synthetic reports (each failing
check withdraws the confirmation of its statement), checks that a missing `cecheck` gives ERROR, and runs the
genuine certificate of the fixture `npz` and two of its forgeries end to end; its five tests pass.

What happened (`lean/results/lean_product.json`, key `driver`).

| certificate | exit code | integrity | S2 recorded, verdict | S3 recorded, verdict | S8 recorded, verdict | failed checks |
|---|---|---|---|---|---|---|
| `heis8_correct` | 0 | PASS | CERTIFIED, CONFIRMED | CERTIFIED, CONFIRMED | CERTIFIED, CONFIRMED | none |
| `heis8_wrong_eigenvalue_large` | 0 | PASS | FAILED, REFUTED | INCONCLUSIVE, NO_CLAIM | CERTIFIED, CONFIRMED | none |
| `heis8_swapped_pairs` | 0 | PASS | FAILED, REPRODUCED | INCONCLUSIVE, NO_CLAIM | INCONCLUSIVE, NO_CLAIM | none |
| `heis8_mislabeled_sector` | 0 | PASS | CERTIFIED, CONFIRMED | FAILED, REFUTED | NOT_CHECKED, NO_CLAIM | none |
| `heis8_wrong_window_count` | 0 | PASS | CERTIFIED, CONFIRMED | FAILED, REFUTED | CERTIFIED, CONFIRMED | none |
| `heis8_subnormal_entries` | 0 | PASS | INCONCLUSIVE, NO_CLAIM | INCONCLUSIVE, NO_CLAIM | INCONCLUSIVE, NO_CLAIM | none |
| `npz` | 0 | PASS | CERTIFIED, CONFIRMED | CERTIFIED, CONFIRMED | CERTIFIED, CONFIRMED | none |
| `dense-arrays` | 0 | PASS | CERTIFIED, CONFIRMED | CERTIFIED, CONFIRMED | CERTIFIED, CONFIRMED | none |
| `matrix-free` | 0 | PASS | CERTIFIED, CONFIRMED | CERTIFIED, CONFIRMED | NOT_CHECKED, NO_CLAIM | none |
| forgery of `heis8_correct`: body changed, not resealed | 1 | FAIL | CERTIFIED, NOT_CONFIRMED | CERTIFIED, NOT_CONFIRMED | CERTIFIED, NOT_CONFIRMED | `H1.body_sha256` |
| forgery of `heis8_subnormal_entries`: S2 set to CERTIFIED, resealed | 1 | PASS | CERTIFIED, NOT_CONFIRMED | INCONCLUSIVE, NO_CLAIM | INCONCLUSIVE, NO_CLAIM | `S2.status.rule` |
| forgery of `heis8_correct`: existence radius of pair 0 divided by 10^6, resealed | 1 | PASS | CERTIFIED, NOT_CONFIRMED | CERTIFIED, NOT_CONFIRMED | CERTIFIED, NOT_CONFIRMED | `E2.existence_radius`, `S2.status.exact`, `S3.status.exact`, `S8.C0.exact` |

S1 and S4 are NOT_RECHECKED in every row, and no certificate of this set carries S5, S6 or S7. Every genuine
certificate exits with 0 and every forgery with 1. The verdicts repeat the categories of `CORE.md`
section 5: the wrong eigenvalue is REFUTED, the swapped pairs are REPRODUCED but not refuted, the mislabelled sector
and the wrong window count are refuted at S3, and the subnormal case, which certify-ed leaves INCONCLUSIVE, has no
claim to confirm although its exact radius checks hold. The body change is caught by H1; the resealed status forgery
by the rule that CERTIFIED needs no recorded underflow finding; the resealed existence radius that is too small by
E2 and, through it, by the S2, S3 and S8 status checks. The driver is as strong as `cecheck` and no stronger: it adds
no check, and a verdict CONFIRMED is a level N statement about the stored numbers.

The optional path in the code. The branch `track/external-verifier` of the code repository (from `dev/3.2`, two
commits, `d704471`) adds `certify-ed verify CERT.json`. It runs the integrity check of `Certificate.load` and prints
the summary; with `--external CMD`, or the environment variable `CERTIFY_ED_EXTERNAL_VERIFIER`, it then runs the
configured program and writes its verdict to a separate external-verification record (schema
`certify-ed/external-verification/1`). The command is a template with the placeholders `{certificate}`, `{sidecar}`,
`{bundle}`, `{operator}` and `{report}`; it runs without a shell. The record holds the hashes and statuses of the
certificate as read, the command and the SHA-256 of the program, its exit code, its run time and the end of its
output, and the program's JSON report, hashed, with its fields `verdict`, `level` and `statements`. Its field
`status_effect` says that no status, bound or count depends on the verdict, and the certificate file is not changed.
The module `certify_ed/external.py` is listed among the files outside the computational source hash, and a test
checks that no module that computes a status imports it, so the source hash stays `cb73dec3e4433a3b`. Ten tests with
stub programs cover PASS, FAIL, ERROR, TIMEOUT and a missing program, unchanged certificate bytes and statuses after
every outcome, a placeholder without a value, the exit codes of the command and the environment variable. Run with
the driver above on the certificate of `heis8_correct` the record says PASS, and on the radius forgery it says FAIL
while certify-ed still reports all five statements CERTIFIED: the forgery is resealed, so certify-ed's own hash check
cannot see it, and the record shows the disagreement without changing a status.

## 3 Refinement of the residual check (N-05)

Why this step. `CertifyCore/Kernel/Small.lean` evaluates `checkLowest` by the Lean kernel on the literal data of the
three smallest validation certificates, and `CertifyMathlib.Enclosure.residual_enclosure` is Theorem 1 at level K2.
Nothing connected the two: the evaluation says that certain rational inequalities hold, and the theorem says what
follows from inequalities of that shape for a symmetric matrix. A refinement theorem closes that step, so that for
these data the S2 re-check reaches level K instead of N.

What was done. `CertifyMathlib/Refine/Residual.lean` imports `CertifyCore.Kernel.Defs` and works with its definitions
as they are. `toMatrix` reads the row lists as a real matrix whose entry (i, j) is the sum of the values a row stores at
column j (the expression `denseRow` uses), and `toVec` reads a vector list as a Euclidean vector. The lemmas
`rowDot_eq_sum`, `dotL_eq_sum` and `residual_vec` show that the list functions `rowDot`, `dotL` and `residual` compute
the matrix-vector product, the inner product and the residual of these objects, so that `dotL r r` is the squared
Euclidean norm of A x minus mu x. The main theorems are these.

| theorem | statement |
|---|---|
| `pairOK_theorem1` | if `pairOK c p` holds, the vector has the matrix's length and is nonzero, and the decoded matrix is symmetric, some eigenvalue of it lies within the recorded existence radius of the claimed value |
| `pairOK_theorem1_shift`, `pairOK_theorem1_shift_rad` | for every symmetric H with spectral norm of H minus A at most the recorded model shift, some eigenvalue of H lies within the exported existence radius, and within the exported radius |
| `checkLowest_theorem1` | if `checkLowest c`, `symOK c.A` and `vecOK c` hold, every claimed pair has an eigenvalue of the decoded matrix within its existence radius, and the existence radius is at most the radius |
| `checkLowest_S2` | if `checkLowest c` and `vecOK c` hold, then for every symmetric H within the model shift every claimed pair has an eigenvalue of H within its exported radius, and every exported radius meets the accuracy target |
| `matrixFree_theorem1`, `denseArrays_theorem1`, `npz_theorem1`, `matrixFree_S2`, `denseArrays_S2`, `npz_S2` | the last two for the three literal certificates (`CertifyMathlib/Refine/Small.lean`) |

Two checks had to be added, because `checkLowest` does not make them. `symOK` tests exact symmetry of the decoded
matrix; `cecheck` tests it as `H5.symmetric`, but `checkLowest` does not, and Theorem 1 needs it. `vecOK` tests that
every vector has the matrix's length and is not zero; `pairOK` and the single-group check accept a zero vector,
because then the residual is zero as well and the inequality r^2 <= ex^2 x^2 holds with nothing on either side,
while `cecheck` rejects a zero vector in E2. Both are Boolean functions that the kernel evaluates, and both were
evaluated by `decide +kernel` on the three literal certificates. Their negative controls are
`symOK_rejects_asymmetric` and `vecOK_rejects_zero_vector`; the existing control `npz_forged_radius_rejected` shows
that a radius forgery does not reach the refinement theorems at all.

How the result was checked. The files contain no `sorry`, no `native_decide` and no axiom declaration, and their
theorems depend on the axioms listed in section 6. The S2 form needs no symmetry of the stored matrix, only of the
declared model H, which is how certify-ed states S2. The theorems use the conclusion of `residual_enclosure` and
`residual_enclosure_of_l2_opNorm` unchanged; their eigenvalues are Mathlib's `IsHermitian.eigenvalues`.

What this shows and does not show. For the three literal certificates the conclusion of Theorem 1 holds at level K
for the decoded data, and with the hypothesis of S1 (the spectral norm of the difference between the declared model
and the stored matrix at most the recorded shift) the S2 statement holds for the model. Four things stay outside.
First, that the literals are the stored numbers of the certificates: they were written by `cecheck kdata` after its
hash comparisons, a level N step. Second, S1 itself, which is a hypothesis here. Third, the cluster radii of Theorems 2
and 2', which `checkLowest` also checks; the refinement covers the per-pair radius only. Fourth, the code of
`cecheck` itself (`CertifyCore/Check.lean`, with dyadic numbers and arrays): the theorems are about the
kernel-reducible restatement of `Kernel/Defs.lean`, which the kernel evaluated, not about the compiled program, so
`cecheck` remains at level N for every other certificate.

## 4 Refinement of the exact inertia count (N-05, second part)

Why this step. The S3 re-check counts eigenvalues below a shift sigma by elimination and Sylvester's law, and
`COUNTING.md` proves Sylvester's law (`sylvester_inertia`, `cntLt_eq_of_factor`) on Mathlib. The missing link is the
same as in section 3: that the elimination computes what the law needs.

What was done. `CertifyMathlib/Refine/Inertia.lean` proves the mathematical core and a kernel-evaluable rational
elimination with its refinement. `Pivots n M ps` says that Gaussian elimination without pivoting runs through the
n by n real matrix M with the nonzero pivots ps; `schur` is the Schur complement of the first entry.
`ldl_of_pivots` builds, from one elimination step per pivot, a factorization M = L diag(ps) L^T with det L = 1, and
counts its negative diagonal entries. `cntLt_of_pivots` then gives, by `cntLt_eq_of_factor`, that the number of
eigenvalues of a symmetric A below sigma is the number of negative pivots of A minus sigma I. `gaussQ` is the same
elimination on lists of rationals, written so that the kernel can evaluate it; `pivots_of_gaussQ` proves that its
pivots are those of `Pivots` for the decoded matrix, and `count_of_gaussQ` puts the pieces together for a kernel
certificate: if `gaussQ` returns pivots for the decoded A minus sigma I, the number of eigenvalues of the decoded
symmetric matrix below sigma is the number of negative ones. All of this is kernel-checked in Lean with no `sorry`.

What did not work, and what is open. The count theorems for the three literal certificates
(`Drafts/LeanProduct/SmallCount.lean`) need the kernel to evaluate `gaussQ` on the dense 64 by 64 rational matrix,
and that evaluation did not finish: the build had not ended after 67 minutes and was abandoned, while nanoda
re-checks the integer evaluations of `checkLowest` on the same data in 127 to 149 seconds each (`CORE.md` section 6). The integer fraction-free form that `checkLowest`
and `cecheck` use is faster for the kernel, but its refinement needs one more theorem: that every division of the
Bareiss elimination is exact and that its pivots are the leading principal minors (Sylvester's determinant identity).
With it, `CertifyCore.K.bareiss` would refine to `Pivots` through the relation d_k = pi_1 ... pi_k between the
fraction-free and the rational pivots, which holds over the rationals by a short computation. Neither the identity
nor the link to `CertifyCore.K.bareiss` is formalized, so the S3 count stays at level N for every certificate, and
the theorem behind it is at level K for any data on which `gaussQ` can be evaluated.

## 5 The mechanical gaps of the floating-point model (N-06)

Item N-06 names four gaps: Theorem U5 of `code/certify-ed/docs/proofs/int05_underflow.md` for complex data (the real
case is `FloatUnderflow.lean`), part (c) of D-01 for complex data (the real case is in `D01Float.lean`), part (a) of
Proposition 2 of `code/certify-ed/docs/proofs/E-07_davis_kahan.md` (C-12), and Lemma 5 of the same document. None was
attempted in this phase; the time went to sections 2 to 4. The first two repeat existing files with complex
products and rounded moduli, the size of `FloatUnderflow.lean` and of part (c) of `D01Float.lean`. Part (a) of C-12 is
exact mathematics: for a unit vector phi in the range of the cluster projector P with P S phi = m phi, the residual
S phi minus m phi lies in the range of I minus P, its norm is at most (c + w/2 times the norm of S) divided by gamma_0 by
the spectral gap of A minus the midpoint of the window, and Theorem 1 for S finishes the proof; it uses only theorems
that exist in `CertifyMathlib.Enclosure` and `CertifyMathlib.Counting` and is the cheapest of the four. Lemma 5 is
a floating-point count of the commutator, the labels and the fidelity in the model of `FloatModel.lean`.

## 6 Levels and second checks

Every theorem below is kernel-checked in Lean (level K): its file builds with no `sorry`, no `native_decide` and no
axiom declaration, and `#print axioms` (`Drafts/LeanProduct/AxiomsLeanProduct.lean`, 17 theorems) lists for each of
them only `propext`, `Classical.choice` and `Quot.sound`. The three modules `CertifyMathlib.Refine.Residual`, `.Small`
and `.Inertia` were replayed by `lake env leanchecker CertifyMathlib.Refine`, which ended with exit code 0 after
1844 s and printed nothing (level K+). nanoda (`Drafts/Counting/nanoda_counting.py`, memory limit 4096 MiB, the
axioms propext, Classical.choice and Quot.sound allowed) was run on five theorems
(`Drafts/LeanProduct/nanoda_theorems.txt`); an export holds the theorem with every declaration it depends on, so an
accepted theorem also re-checks the lemmas below it.

| theorem | file | level | nanoda |
|---|---|---|---|
| `pairOK_theorem1` | `Refine/Residual.lean` | K2 | accepted, 34729 declarations |
| `pairOK_theorem1_shift_rad` | `Refine/Residual.lean` | K2 | accepted, 35119 declarations |
| `pairOK_theorem1_shift`, `checkLowest_theorem1`, `checkLowest_S2` | `Refine/Residual.lean` | K+ | not run |
| `npz_S2`, `npz_theorem1` | `Refine/Small.lean` | K+ | stopped at the memory limit while checking (peak 4223 and 4224 MiB) |
| the other four theorems of `Refine/Small.lean` and the two negative controls | `Refine/Small.lean` | K+ | not run |
| `count_of_gaussQ` (with `ldl_of_pivots`, `cntLt_of_pivots`, `pivots_of_gaussQ`) | `Refine/Inertia.lean` | K2 | accepted, 35878 declarations |

The two theorems of `Small.lean` that nanoda could not finish contain the kernel evaluations of `checkLowest`,
`symOK` and `vecOK` on the 64-dimensional data; nanoda needed more than 4096 MiB for them, and with the machine
shared by other work the limit was not raised. Each export took 619 to 791 s and 3983 to 3991 MiB.

## 7 Findings for the documents

Two properties of `CertifyCore/Kernel/Defs.lean` were found while stating the refinement theorems, and neither
affects a stored certificate. First, `checkLowest` does not test that the decoded matrix is symmetric; Theorem 1 and
Sylvester's law need it, `cecheck` tests it (`H5.symmetric`), and the three literal certificates pass the kernel
evaluation of `symOK`. Second, `pairOK` and the single-group check of `groupOK` accept a zero vector, for which the
radius inequality holds trivially; `cecheck` rejects a zero vector in E2, and the three literals pass `vecOK`. The
docstring of `Kernel/Defs.lean` lists what `checkLowest c = true` asserts and is correct; the two properties are
hypotheses that a statement about eigenvalues needs in addition, and the refinement theorems carry them.

## 8 Files

| path | contents |
|---|---|
| `Verify/cecheck_bundle.py` | the driver of section 2 |
| `Verify/test_cecheck_bundle.py` | its tests |
| `CertifyMathlib/Refine/Residual.lean` | the refinement of the residual check (section 3) |
| `CertifyMathlib/Refine/Small.lean` | the residual theorems for the three literal certificates |
| `CertifyMathlib/Refine/Inertia.lean` | elimination, Sylvester's law for pivots, and the list elimination (section 4) |
| `Drafts/LeanProduct/SmallCount.lean` | the count theorems for the three literals, whose kernel evaluation did not finish |
| `Drafts/LeanProduct/RefineResidual.lean`, `RefineInertia.lean` | the working copies of the two refinement files |
| `Drafts/LeanProduct/AxiomsLeanProduct.lean`, `nanoda_theorems.txt` | the axiom audit and the theorem list of the nanoda run |
| `Drafts/LeanProduct/probe_names.lean` | `#check` of the Mathlib and core names the two refinement files use |
| `results/lean_product.json` | the numbers of this document |
