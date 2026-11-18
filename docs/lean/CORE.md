# CORE.md: the core-Lean exact checker and kernel theorems

This part of the re-verification uses Lean 4.35.0 without Mathlib. It has two products. The first is `cecheck`,
a compiled Lean program that re-checks certify-ed certificates from their raw stored numbers in exact rational
arithmetic (trust level N in the scale of `README.md`). The second is a set of Lean theorems about the finite
arithmetic behind the bounds, accepted by the Lean kernel with no `sorry` and no `native_decide` (trust level
K). Every number below that comes from the project record is cited by its path relative to `sources/archive_v5/`;
every number produced here is in `reverification_lean/results/core/`.

A word on terms. An eigenpair claim is a value mu and a vector x that a producer says are an eigenvalue and
eigenvector of a symmetric matrix A. The residual is the vector A x minus mu x. Theorem 1 of certify-ed says that
some eigenvalue lies within the residual norm divided by the norm of x of mu; certify-ed evaluates that bound in
floating point and adds the rounding terms of its accounting, and the result is the recorded existence radius.
An inertia count at a shift sigma is the number of eigenvalues below sigma; certify-ed obtains it from a
floating-point LDL^T factorisation with a recorded backward-error bound beta. The statements S1 to S8 of a
certificate are those of `code/certify-ed/docs/SCOPE_MATRIX.md`.

## 1 How to run

The commands assume the environment of `reverification_lean/logs/lean_env.json` (the Lean binaries on `PATH`)
and the Python environment `certifyed-311` with the certify-ed source on `PYTHONPATH`. They write only below
`reverification_lean/work/core/` (git-ignored) and `reverification_lean/results/core/`.

```
cd reverification_lean
lake build cecheck CertifyCore
set CE=..\sources\archive_v5\code\certify-ed
set PYTHONPATH=%CE%
python %CE%\verifier\make_validation_set.py --out work\core\certs --cases %CE%\corpus\cases --conformance %CE%\conformance\reference
python export\export_certs.py --index work\core\certs\index.json --certs work\core\certs --out work\core\export
python export\run_cecheck.py --exported work\core\export --out results\core\genuine --workers 2
python export\make_mutants.py --code %CE% --certs work\core\certs --out work\core\mutants
python export\export_certs.py --index work\core\mutants\index.json --certs work\core\mutants --out work\core\export_mutants
python export\run_cecheck.py --exported work\core\export_mutants --out results\core\mutants --workers 2
python %CE%\verifier\build.py --out work\core\bin\cev.exe
python export\run_cev.py --cev work\core\bin\cev.exe --index work\core\certs\index.json --certs work\core\certs --out work\core\cev\genuine --summary results\core\cev_genuine.json
python export\run_cev.py --cev work\core\bin\cev.exe --index work\core\mutants\index.json --certs work\core\mutants --out work\core\cev\mutants --summary results\core\cev_mutants.json
python export\python_integrity.py --index work\core\certs\index.json --certs work\core\certs --out results\core\python_integrity_genuine.json
python export\python_integrity.py --index work\core\mutants\index.json --certs work\core\mutants --out results\core\python_integrity_mutants.json
python export\agreement.py --work work\core --results results\core
python export\second_kernel.py --lean-bin <toolchain bin> lean4lean leanchecker nanoda
python export\findings_values.py --code %CE%
```

The C verifier is built with the MinGW gcc of GNU
Octave (`CEV_CC` set to its `gcc.exe`, its directory on `PATH`). The certify-ed run needs
`PYTHONDONTWRITEBYTECODE=1` and `GIT_OPTIONAL_LOCKS=0` so that it writes nothing into the read-only source tree.
One certificate is checked with `.lake\build\bin\cecheck.exe work\core\export\<case>`, which prints every check
and exits with 0 (no check failed), 1 (a check failed) or 2 (the input could not be read).

## 2 What is trusted on the way from a certificate to `cecheck`

The exporter `export/export_certs.py` copies the certificate JSON byte for byte, writes each member of the
array sidecar as raw little-endian bytes in C order, and writes the operator in canonical CSR form (sorted
column indices, duplicates summed, explicit zeros removed). `cecheck` recomputes every SHA-256 the certificate
records over exactly these bytes, with its own SHA-256 (`CertifyCore/Sha256.lean`): the body hash over its own
canonical serialisation of the parsed JSON (sorted keys, the separators `,` and `:`, numbers as their original
text, strings escaped as Python's `json.dumps` with `ensure_ascii=False` escapes them), each array hash over the
dtype string, `json.dumps(list(shape))` and the bytes, and the operator hash over the CSR triple as
`certify_ed/certificate_v2.py` defines it. A byte the exporter got wrong therefore makes a hash comparison fail;
it cannot make a check pass. The exporter selects only which operator to hand over: the bundle's `operator.npz`,
or for the two conformance bundles without one the operator certify-ed itself builds from the bundle, in every
case replaced by its Hermitian part (A + A^T)/2 when the certificate records `hermitian_part`. `cecheck` accepts
the operator only if its hash equals `subject.operator.matrix_sha256` and it is exactly symmetric.

The JSON reader (`CertifyCore/Json.lean`) keeps every number as the text it was written with, because the
canonical form needs the exact text; a certificate file is itself canonical JSON written by Python, so the text
is Python's shortest representation of the float. The value of a number is the binary64 number nearest to its
decimal text, rounded half to even, computed by `CertifyCore/Dyadic.lean` in integer arithmetic; on 3015 test
values (random doubles, random bit patterns, subnormal numbers and boundary cases) it agreed with Python's
`float()` in every case. For all 55 certificates the canonical serialisation reproduced the file byte for byte.

Two things are taken from the record without recomputation. The first is the model shift of S1, the recorded
bound on the distance between the stored operator and the declared model; rebuilding the declared model from its
specification is the work of `certify_ed.construct` and is not repeated here, so S1 itself is not re-checked. The
second is the choice of the recorded shifts sigma: they are numbers in the certificate, and `cecheck` counts
eigenvalues exactly at them.

## 3 What `cecheck` checks

All comparisons are exact. Stored numbers are decoded from their bit patterns into dyadic rationals m 2^e; sums,
products and squares are exact; an inequality r >= sqrt(q) is checked as r >= 0 and r^2 >= q. Where a radius of
certify-ed contains square roots of computed quantities (Theorems 2 and 2'), `cecheck` computes its own rational
upper bounds of those square roots with 160 binary digits and checks the recorded radius against the formula
evaluated with them.

| check | what is recomputed | a failure means |
|---|---|---|
| `H1.body_sha256` | SHA-256 of the canonical body without `integrity` | the body was changed after it was sealed |
| `H2.<array>`, `H3`, `H4`, `H5` | array hashes, the claimed-eigenvector hash, the specification hash, the operator hash; canonical CSR form and exact symmetry of the operator | an array, the specification or the operator is not the certified one |
| `C1` to `C4`, `C7` | claimed values equal to the stored array; enclosure records consistent (lower end at most mu minus r, upper end at least mu plus r, existence radius at most radius); exported radius at least record radius plus model shift; scope statuses equal statement statuses | an internal inconsistency |
| `E1.residual_bound` | for every pair, recorded residual bound at least the exact norm of A x minus mu x | an unsound record |
| `E2.existence_radius` | recorded existence radius at least the exact residual norm divided by the exact norm of x (Theorem 1) | an unsound record |
| `E3.G<n>.cluster` | for every cluster, the Theorem 2 radius and the Theorem 2' radius from the exact residuals, the exact Gram defect, the exact cross term and the exact infinity norm of A, against the recorded group radius | the cluster radius is below what the theorems give |
| `G<n>.record` | the recorded intermediate quantities of the cluster bound (eta, R_fro, XR, spread, delta, rho, eps_star) against their exact values and formulas | an unsound recorded intermediate |
| `E4`, `E5` | Theorem 1 for single groups; the enclosures of groups marked distinct are pairwise disjoint | distinctness or a radius is not supported |
| `S2.status.*` | for CERTIFIED: all of the above, every pair distinct, largest radius at most the accuracy target; the rule that CERTIFIED needs the floating-point runtime probe and no recorded underflow finding | a CERTIFIED status the data do not permit |
| `S2.failed` | for FAILED: the exact residual radius plus the model shift exceeds the target (REPRODUCED); exact counts show no eigenvalue of the operator within the target plus the model shift of the claimed value (REFUTED) | not reproduced |
| `S3.count<i>` | the exact number of eigenvalues below every recorded shift, against the recorded `negative` | informational |
| `S3.status.*` | for CERTIFIED (kind `lowest`): every exported enclosure below the recorded bound b, b plus model shift at most the top shift sigma, exactly k eigenvalues below sigma by exact count, recorded gap at most b minus the top enclosure; kind `window` and `all` likewise; for FAILED: the exact count at the witness shift exceeds the number of claimed enclosures reaching below it (REFUTED) | a CERTIFIED status the data do not permit |
| `S8.C<n>.exact` | for every CERTIFIED degeneracy bracket: the exact number of eigenvalues between the two recorded shifts equals the cluster size, and the cluster's enclosures lie inside the inner bracket | the degeneracy count is not supported |
| `K<n>.kato_temple` | for every Kato-Temple record: exactly one eigenvalue in the isolation interval by exact counts, the exact Rayleigh quotient inside it, and the recorded interval containing the exact Kato-Temple interval | an unsound sharpened enclosure |

What `cecheck` does not check: S1 (see section 2), the eigenspace bounds of S4, and S5 to S7. Theorems 1, 2 and
2', Sylvester's law of inertia and Weyl's inequality are used as stated; they are the subject of the Mathlib part
of the re-verification. Complex operators and sector certificates do not occur among the 55 certificates and
are not handled (the checker reports SKIP or FAIL for them).

## 4 Exact eigenvalue counts

`CertifyCore/Inertia.lean` computes the inertia of A minus sigma I exactly. The matrix is scaled by a power of two
to integers (A and sigma are binary64 numbers, so this is exact and keeps the inertia). Fraction-free (Bareiss)
elimination with symmetric diagonal pivoting gives the leading principal minors d_1, d_2, ... of a symmetric
permutation of the matrix; the number of negative eigenvalues is the number of sign changes in 1, d_1, d_2, ...
(Sylvester's law with Jacobi's rule). When every remaining diagonal entry is zero but an off-diagonal entry is not,
a congruence adds one row and column to another; when the remaining block is zero its size is the number of zero
eigenvalues. Every division is exact, and the program checks this at each step.

The first measurement showed that a dense elimination is too slow for the larger certificates: one count for
`tfim8_correct` (dimension 256, one connected block) took 247 s, and the Heisenberg chain of 10 sites, of dimension
1024, has blocks of up to 252 states. Two exact reductions were therefore added. The matrix is split into the
connected components of its sparsity graph. In addition, the bit complement and the bit reversal of the basis
index (a spin flip and a reflection of the chain) are tested for exact invariance of A; those that leave A
invariant, commute and are independent generate a group G of order 2^m, and for each character chi of G the
vectors sum over g in G of chi(g) e_{g r} (one per orbit representative r on whose stabiliser chi is trivial)
form a basis in which A minus sigma I is block diagonal. A change of basis by an invertible matrix keeps the
inertia (Sylvester), so the inertias of the character blocks add up; the program checks that the block sizes add
up to the dimension and that every block is exactly symmetric. With both reductions the same count for
`tfim8_correct` took 2.9 s, with largest block 72, and a count for `heis10_correct` 16.5 s, with largest block 110.

Over the 55 certificates `cecheck` computed 211 exact counts. The largest dense block was 128 states, in
`tfim8_wrong_coupling`, where the changed coupling leaves only one of the two symmetries; its five counts took 30
to 37 s each. The 70 counts that certify-ed recorded in its S3 evidence were recomputed at the recorded shifts and
all 70 equal the recorded number of negative eigenvalues. The cost of an elimination grows with the cube of the
block size and with the bit length of the entries; the limit of this method is set by the largest block after
the reductions, not by the dimension, and no certificate here came near it.

## 5 Results on the validation certificates

Why this test. The C verifier of certify-ed was validated on 63 producer bundles (60 corpus cases and 3
conformance fixtures), of which certify-ed refuses 8 and certifies 55; 18 of the S2 and S3 statuses it sets are
FAILED, and 80 forged certificates, 20 each of four kinds, must be rejected
(`code/certify-ed/verifier/README.md`). The same set is a ready test for a third, independent checker: the corpus
cases each plant one known error (`code/certify-ed/corpus/README.md`), so a checker that accepts a correct case and
reproduces every refutation has been tested on both sides. A Lean checker that disagreed with the C verifier on
any case would point to an error in one of the two or in certify-ed.

What was done. The 55 certificates were produced with `verifier/make_validation_set.py` of the archived code
(branch `dev/3.2` at `a814d28`, certify-ed 3.2.1) with stored eigenvectors; their statuses agree with the archived
validation record (`code/certify-ed/verifier/validation/results.json`) in every case. The 80 forgeries were made
by the functions of `verifier/validation/run_validation.py` itself, on the same 20 certificates in the same order,
and their descriptions are identical to the archived ones. The C verifier was rebuilt from `verifier/src/cev.c`
and run on both sets with the bundle operators; certify-ed's own integrity check (`Certificate.load` with
verification) was run on both sets.

How the instrument was checked. The SHA-256 code reproduces the standard test vectors; the canonical JSON writer
reproduces every certificate file byte for byte; the decimal reader agrees with Python on 3015 values; the exact
counts at the 70 recorded shifts agree with certify-ed. The forgeries are the negative controls: each one changes
exactly one thing that a correct checker must notice.

What happened (`results/core/agreement.csv`, `results/core/agreement_stats.json`).

| quantity | count |
|---|---|
| certificates checked by `cecheck`, no check failed | 55 of 55 |
| S2 and S3 statuses certify-ed set to FAILED (one per certificate) | 18 (12 S2, 6 S3) |
| of these, REPRODUCED by `cecheck`, by the C verifier, by both | 18, 18, 18 of 18 |
| of these, also contradicted by an exact count (category REFUTED, `cecheck` only) | 15 (9 S2, 6 S3) |
| forgeries rejected by `cecheck` (radius / array hash / body hash / status) | 20, 20, 20, 20 of 20 each |
| forgeries rejected by the C verifier | 80 of 80 |
| forgeries rejected by certify-ed's integrity check | 40 of 80 (the hash forgeries) |
| rows where `cecheck` and the C verifier give the same answer to the same question | 135 of 135 (55 genuine, 80 forgeries) |
| CERTIFIED S2 statuses confirmed exactly | 31 |
| CERTIFIED S3 statuses confirmed exactly | 24 |
| cluster radii (Theorem 2 or 2') confirmed exactly | 55 |
| CERTIFIED degeneracy brackets (S8) confirmed by exact counts | 42 |
| Kato-Temple records confirmed exactly | 15 |

The two categories of a FAILED status have fixed meanings. REPRODUCED means that the checker finds, on the
stored numbers, the reason that certify-ed recorded: for S2 a pair whose residual radius plus the model shift
exceeds the accuracy target, for S3 more eigenvalues below the witness shift (or inside the claimed window) than
the claim covers. REFUTED means REPRODUCED and, in addition, an exact inertia count of the stored operator
contradicts the claim itself. Only `cecheck` computes exact counts, so the C verifier answers REPRODUCED or
NOT_REPRODUCED and never REFUTED. The table therefore compares the two checkers on the question both answer:
the columns `c_failed_reproduced` and `lean_failed_reproduced` say per statement whether each checker reproduces
the FAILED status, and `failed_reproduced_both` says whether both do. The exact refutation is reported apart, in
`lean_exact_refutation`, and does not enter the column `agree`; `agree` is yes for a genuine certificate when both
checkers exit with 0 and both reproduce every FAILED status, and for a forgery when both exit with 1. The columns
`lean_failed_category` and `c_failed_category` give the category each checker assigns; they differ in the 15 rows
where `cecheck` adds a refutation. The definitions are repeated in `agreement_stats.json` under `definitions`.

A refutation is stronger than a reproduction. For the nine FAILED S2 statuses of the cases with a wrong
eigenvalue or a non-Hermitian matrix, exact counts at mu minus and plus the target show that no eigenvalue of
the operator lies within the accuracy target (plus the model shift) of the claimed value, so the claim is false,
not only uncertified. For S3 the reason certify-ed records is itself a count, and `cecheck` reproduces it with an
exact count, so each of the six reproduced S3 failures (mislabelled sector, wrong window count) is also a
refutation: the exact count at the witness shift exceeds the number of claimed enclosures that reach below it,
so lowest eigenvalues are missing from the claim. For the three cases with swapped pairs the claimed values are
eigenvalues and only their pairing with the vectors is wrong; there the exact residual radius exceeds the target
(REPRODUCED) and an eigenvalue does lie within the target of each value, which `cecheck` reports.

What the results show and do not show. Three of the 20 status forgeries are rejected only by the rule that
CERTIFIED needs the floating-point hypotheses: in `heis8_subnormal_entries`, `heis10_subnormal_entries` and
`heis10_wrong_coupling` certify-ed reports S2 INCONCLUSIVE because the stored vectors have entries below 2^-900
(reason code `UNDERFLOW_RISK` of `code/certify-ed/certify_ed/certify.py`, threshold `UNDERFLOW_THRESHOLD` in
`code/certify-ed/certify_ed/fp.py`), and the exact recomputation shows that every radius
and every distinctness claim of S2 holds exactly for the stored operator. The same holds for
`j1j2_8_subnormal_entries`. For these four certificates the exact check establishes what the floating-point
check could not, because exact arithmetic needs no hypothesis about underflow. The agreement shows that the three
checkers implement the same certificate semantics on this set; it does not show that the set exercises every
code path of certify-ed, and it says nothing about S1, S4 to S7, or certificates of other shapes. The run times
are in `results/core/genuine/index.json`: 1785 s of checker time for the 55 certificates (two at a time), most of
it in the exact counts of the dimension-1024 cases.

## 6 Kernel theorems (level K)

The theorems are in `CertifyCore/Theorems/` and `CertifyCore/Kernel/`. They are over the rationals of core Lean
(`Rat`), with u = 2^-53 exactly and gamma_k = k u/(1 - k u), the definitions of `certify_ed/fp.py`. Their axioms
are printed in `results/core/axioms.txt`; every theorem depends at most on `propext`, `Classical.choice` and
`Quot.sound`.

| file | theorems | what they state | source |
|---|---|---|---|
| `Gamma.lean` | `gamma_nonneg`, `gamma_mono`, `gamma_add`, `gamma_succ`, `gamma_ge`, `gamma_le_two` | gamma_k >= 0; m <= n implies gamma_m <= gamma_n; gamma_m + gamma_n + gamma_m gamma_n <= gamma_{m+n}; gamma_a + u (1 + gamma_a) <= gamma_{a+1}; k u <= gamma_k <= 2 k u when k u <= 1/2; all for every k with k u < 1 | `docs/proofs/thm1_fp-accounting.md`, Lambda1 to Lambda3 |
| `Thm1.lean` | `lambda4_upper_2n`, `lambda4_upper_n`, `lambda4_lower_2n`, `lambda4_lower_n` | the norm guards: (1 - gamma_N)[(1 - u)(1 + gamma_{n+2}(1 - u)^2)(1 + 8u)(1 - u)^4]^2 >= (1 + 3u)^2 and the matching lower guard, for every n <= 2^26 and N = n or 2n | Lambda4 |
| `Thm1.lean` | `lambda5`, `lambda5_complex` | the guard of the absolute product: (1 - gamma_w)(1 - u)^3(1 + 4u)/(1 - gamma_{w+1}(1 - u)^2) >= 1 + u for every w <= 2^50, and the same with gamma_{w+2}, gamma_{w+3}, the correction for complex data | Lambda5 and the correction of 3 November 2026 |
| `Thm1.lean` | `lambda6`, `lambda6_fails_at_boundary` | the guard of the entrywise bound: (1 - u)^6 (1 + 4u)(1 + u/gamma_{w+2}) >= 1 for w + 2 <= 3002399751580329, and it fails at w + 2 = 3002399751580330 | Lambda6 |
| `Thm1.lean` | `one_add_4u_covers_three`, `one_sub_u_cube_mul`, `guard_covers_six`, `step5_chain` | (1 - gamma_3)(1 + 4u) >= 1; (1 - u)^3 (1 + 4u) >= 1; (1 - u)^6 (1 + 8u) >= 1; the three inequalities of step 5 | steps 2, 3, 5 |
| `Int05.lean` | `binary64_on_grid`, `small_multiple_binary64`, `add_exact_below` | every binary64 number is an integer multiple of 2^-1074, and the sum or difference of two binary64 numbers below 2^-1022 in magnitude is a binary64 number, so it is computed exactly | `docs/proofs/int05_underflow.md`, model (M-U) |
| `Int05.lean` | `additive_sum_bound`, `norm_underflow_factor`, `ulp_min_covers` | Lemma U1's additive term is at most k u_min (1 + g), u_min = 2^-1075; (1 + 2 gamma_{N+2})^2 >= (1 + gamma_N)/(1 - gamma_N) when 3N + 6 <= 2^53; adding 2^-1074 covers a rounding to the subnormal grid | Lemmas U1, U3, the term A_{n,w} |
| `L17.lean` | `ln_remainder_const`, `ln_remainder_old_const_too_small`, `ln_tail_bound`, `exp_tail_bound` | 1/(1 - x^2) <= 8/7 for 0 <= x <= 0.35, while 9/8 fails at 0.35; the finite tails of the atanh series are at most (8/7) x^{2M+3}/(2M+3) and of the exponential series at most 2 t^{N+1}/(N+1)! for t <= 1/2 | `docs/proofs/L17_dyadic_exp_ln.md`, (iii) and (iv) |
| `L17.lean` | `ln_argument_range`, `umax_admits`, `ceil_ceil` | the series argument is at most 1/3; the ceiling of 2^192/3 is at most U_MAX; nested ceilings | (iii), (iv) |
| `E04.lean` | `crossing_slack`, `crossing_separation` | if the tube ends are B-Lipschitz, the margin on a sample interval of width D is at most the larger end value plus B D, so the criterion separates the tubes on the whole interval | `docs/proofs/E-04_parameter_intervals.md`, Lemma 3 |
| `Counting.lean` | `pigeon`, `sorted_pairing`, `below_index_lt`, `cor5_pairing` | the sorted-pairing lemma; if at most k of the sorted eigenvalues lie below s and k claimed eigenvalues are matched injectively below s within radius r, then the j-th claim is within r of the j-th eigenvalue | `docs/proofs/cor5_pairing.md` |
| `Counting.lean` | `sum_mul_dCount`, `sum_dCount`, `c16_single_root` | for the distinct roots in a bracket with their multiplicities, the sum of the d_j is the number of distinct roots, the sum of j d_j the number with multiplicity, and if these are 1 and g the bracket holds one root of multiplicity g | `docs/proofs/C16_exact_bracket.md` |
| `Counting.lean` | `pooled_count`, `pooled_length` | counts below a cut and dimensions add over sectors | `docs/proofs/thm8_rho.md` |
| `Kernel/Small.lean` | `matrixFree_S2_S3`, `denseArrays_S2_S3`, `npz_S2_S3`, `npz_forged_radius_rejected` | the S2 and S3 arithmetic of the three smallest certificates, evaluated by the kernel; a radius forgery of `npz` is rejected by the same evaluation | conformance fixtures |

Three of these theorems bear on statements of the derivation documents: `lambda6_fails_at_boundary` shows that
the closing condition of the Lambda6 derivation admits one value where the inequality fails,
`ln_remainder_old_const_too_small` confirms that the ln remainder constant of the 3.1 code was too small for the
stated argument range, and the norm guards hold on the stated range n <= 2^26 with little room. Section 8 gives
the exact values, the lines of the documents and of the code concerned, and whether a stored certificate is
affected. The proofs of the norm guards reduce each guard to a polynomial inequality in a = n u on the interval
[0, 2^-27] and close it with linear arithmetic over the monomials of a; the margins were computed beforehand in
exact rational arithmetic.

The theorems about gamma_k and the scalar guards are statements about numbers. That the code's floating-point
operations satisfy the error model in which these numbers appear, and the theorems of numerical linear algebra
that use them, are not formalised here.

After the build the library was checked again in three ways by `export/second_kernel.py`, with the results in
`results/core/second_kernel.json`. The README defines level K2 by a second checker of the toolchain; here K2
means accepted by `leanchecker` and by `nanoda`, for the reasons that follow.

The first check is `leanchecker`, the kernel replay program of the toolchain. The call `leanchecker CertifyCore`
takes every module whose name begins with `CertifyCore` (all 21) and replays its declarations into the environment
of its imports, so that the kernel checks every theorem of this section and every evaluation of section 7 again
from the stored proof terms. It ended with exit code 0 after 473 s and printed nothing (an earlier run through
`lake env` took 562 s, with the same result). This replay uses the same kernel as the build, so it guards against
a compiled file that differs from what the kernel accepted, not against an error of the kernel itself.

The second is `lean4lean`, a reimplementation of the Lean kernel written in Lean; the toolchain directory contains
the program `bin/lean4lean.exe`. It was run once for each of the 21 modules and once on the toolchain's own module
`Init.Prelude` as a control. All 22 calls ended with exit code 1 after 5.6 to 19.0 s, each with the message that
the `.olean` file has an incompatible header. The program carries the version string 4.35.0-nightly-2026-09-06,
while the toolchain and every `.olean` file carry 4.35.0 (commit `eb6cafb7290e8bd5f2937522d8d804f58f02bafc`), and
the control fails in the same way, so this build of lean4lean cannot read any module of this toolchain. It
checked nothing, and its exit codes say nothing about the library.

The third is `nanoda` (`bin/nanoda_bin.exe` of the same toolchain), an implementation of the Lean type theory
written in Rust and independent of the Lean kernel, used in the way of the nanoda check of the earlier mm3x3 project. Each of the 76 theorems was exported by
`leanexport` together with every declaration it depends on (78 to 3876 declarations per theorem), and nanoda
checked each export from scratch with only the axioms propext, Classical.choice and Quot.sound allowed. It accepted
all 76 with exit code 0. The three kernel evaluations of section 7 took 127 to 149 s each and every other theorem
at most 42 s; export and checking took 708 s and 773 s in total.

## 7 Kernel-checked certificates

`CertifyCore/Kernel/Defs.lean` restates the S2 and S3 checks for a claim of kind `lowest` with structural
recursion only, so that the kernel can evaluate them. Its function `checkLowest` returns true exactly when, in
exact rational arithmetic on the stored numbers, every pair satisfies the residual-bound and existence-radius
inequalities, every single group satisfies Theorem 1, every cluster radius is at least the Theorem 2 or 2' radius
computed with rational upper bounds of the square roots (whose defining inequalities are checked), the groups are
disjoint, the largest radius meets the target, the enclosures lie below the recorded bound, the bound plus the
model shift is at most the top shift, the recorded gap is consistent, and the fraction-free elimination of
2^s (A - sigma I) has nonzero leading minors with exactly k sign changes. The three smallest certificates, the
conformance fixtures `matrix-free`, `dense-arrays` and `npz` (dimension 64; 1, 4 and 4 claimed pairs; S2 and S3
CERTIFIED), were written as Lean literals by `cecheck kdata` after their exact check passed, from the same bytes
whose hashes it compared, and `checkLowest` was evaluated on them by `decide +kernel`. The negative control
`npz_forged_radius_rejected` evaluates the same function on the radius forgery of the C validation and obtains
false. The step from the sign count to the number of eigenvalues below sigma (Sylvester's law and Jacobi's rule)
and the theorems behind the radii are not part of these statements.

## 8 Findings for the derivation documents

Three theorems of section 6 bear on statements of the derivation documents. None of the three affects a stored
certificate. The exact values below are in `results/core/findings.json`, written by `export/findings_values.py`
in exact rational arithmetic; the inequalities themselves are the Lean theorems named. Document lines are those
of the archived files. Code lines are those of `code/certify-ed` on branch `dev/3.2` at `a814d28` unless a tag is
named; the line numbers inside `thm1_fp-accounting.md` refer to the branch `track/backlog-proofs` and differ from
these. For the question whether a certificate is affected, the largest value of a dimension field (`dim`,
`dimension` or `n`) in the result files under `code/certify-ed/results/` is 1116229
(`code/certify-ed/results/run_full_20261022/e9_pyrochlore_20261022_185936/data.json`), below 2^21.

| finding | Lean theorems | document | code | stored certificate affected |
|---|---|---|---|---|
| the closing condition of Lambda6 admits w + 2 = 3002399751580330, where the guard fails | `lambda6`, `lambda6_fails_at_boundary` | `code/certify-ed/docs/proofs/thm1_fp-accounting.md` lines 54 to 56 | `certify_ed/enclosure.py` lines 295 to 315 | none |
| the ln remainder constant 9/8 of the 3.1 code is too small at 0.35; 8/7 suffices | `ln_remainder_const`, `ln_remainder_old_const_too_small` | `code/certify-ed/docs/proofs/L17_dyadic_exp_ln.md` lines 112 to 125 | `certify_ed/ivl.py` line 212 at `v3.1.0`; lines 210, 219, 220 and 233 now | none |
| the upper norm guard with N = 2n has margin 0.75u at n = 2^26 and fails from n = 84886744 | `lambda4_upper_2n` and the three other guards | `code/certify-ed/docs/proofs/thm1_fp-accounting.md` lines 44 to 50 | `certify_ed/fp.py` lines 30 and 149 to 167 | none |

Lambda6, the guard of the entrywise rounding bound. The lemma (line 54) states
(1 - u)^6 (1 + 4u)(1 + u/gamma_{w+2}) >= 1 for 1 <= w <= 2^26, and its derivation ends (lines 55 and 56) with the
sufficient condition (w + 2)u <= 1/3. The theorem `lambda6` proves the inequality for every
w + 2 <= 3002399751580329, so the lemma as stated holds, with the left side exceeding 1 by 1.49e-8 at w = 2^26.
The theorem `lambda6_fails_at_boundary` proves that it fails at w + 2 = 3002399751580330, which is floor(2^53/3):
the left side minus 1 is 2.0 u^2 at the last value where it holds and -7.0 u^2 at the failing one. The failing
value satisfies (w + 2)u <= 1/3, since three times it is 2^53 - 2, and it is the only such value, since the next
one violates the condition. A correct closing condition is (w + 2)u <= (1 - 5u)/3, which is exactly
w + 2 <= 3002399751580329; alternatively the derivation can stop at the stated range w <= 2^26. The code applies
the guard in `_rounding_bound` (`certify_ed/enclosure.py` lines 295 to 315, `g = gamma_dot(adapter.width + 2,
cplx)` at line 300), where w is the row width (the largest number of nonzero entries in a row, or an upper bound on it for a
matrix-free operator), and it does not check w <= 2^26.
The failing value would need a row with 3002399751580328 nonzero entries, far beyond any stored operator.

The ln remainder constant of Lemma 17. The atanh series is called with an upper argument that can exceed 1/3 by
one grid unit of 2^-192; the largest is ceil(2^192/3) 2^-192 (lines 112 to 117). The tail bound uses
1/(1 - u^2), which equals 9/8 at u = 1/3 and exceeds it by 8.96e-59 at the largest argument, so the derivation
states the bound for u <= 0.35 with the constant 8/7 (lines 122 to 125). The theorem `ln_remainder_const` proves
1/(1 - x^2) <= 8/7 for 0 <= x <= 0.35, and `ln_remainder_old_const_too_small` proves that 9/8 fails at x = 0.35,
where 1/(1 - x^2) = 1.1396. The 3.1 code (tag `v3.1.0`, commit `e860a83`, `certify_ed/ivl.py` line 212) adds the
remainder `_ceil_div(_ceil_div(pw * 9, 8), 2 * _LN_TERMS + 3)` and documents the range u <= 1/3 (line 201)
without checking it; the archived code adds `_ceil_div(_ceil_div(pw * 8, 7), 2 * _LN_TERMS + 3)` (line 233) and
rejects arguments above `_U_MAX`, which is 0.35 (lines 210, 219 and 220). No enclosure computed by the 3.1 code was
wrong, and the reason can be stated exactly. At the largest argument the true tail is at most 0.00118 grid units,
while the remainder that either version adds is 1 grid unit (after the loop `pw` is 1 there, and both ceilings
give 1). For every positive argument the added remainder is at least 1 grid unit, because `pw` stays at least 1,
and the tail is smaller for smaller arguments. The constant 9/8 therefore fails in the proof, not in the computed
enclosures. The ln intervals are used by Proposition 16 (`thermo.py`, statement S7).

Lambda4, the norm guards. The lemma (lines 44 to 50) states the upper guard
(1 - gamma_N)^{1/2}(1 - u)(1 + gamma_{n+2}(1 - u)^2)(1 + 8u)(1 - u)^4 >= 1 + 3u and the matching lower guard for
1 <= n <= 2^26 and N equal to n or 2n; N = 2n is the case of complex data, whose n entries have 2n real
components. The theorems `lambda4_upper_2n`, `lambda4_upper_n`, `lambda4_lower_2n` and `lambda4_lower_n` prove all
four on the stated range, so the lemma holds. The derivation's own argument (lines 47 to 50) gives only 1 + 2u in
the worst case and leaves the constant 3u to an exact check on a grid (every n <= 2 10^4 and the powers 2^j with
15 <= j <= 26, line 159); the theorems close it for every n <= 2^26. The binding case is the upper guard with
N = 2n: at n = 2^26 its left side is 1 + 3.75u, a margin of 0.75u above 1 + 3u (1.50u in the squared form in which
the Lean theorems state it), while the lower guard with N = 2n has margin 3.25u and the two guards with N = n have
margins near 3.4e7 u. Beyond the stated range the binding margin decreases: the upper guard with N = 2n holds for
n <= 84886743 and fails from n = 84886744, about 1.265 times 2^26, where the margin is -4.1e-8 u; at n = 2^27 it
is -3.0u. These two values come from an exact bisection on the squared form in `export/findings_values.py`, and
the decrease was sampled at 65 points between 2^26 and 2^27; they are not Lean theorems. The code applies the
guards in `norm2_upper` and `norm2_lower` (`certify_ed/fp.py` lines 149 to 167, with `_GUARD`, which is 1 + 8u, at
line 30) and checks no bound on the length, so a complex vector with more than 84886743 entries would leave the
range in which the slack of step 5 is available. The largest stored dimension, 1116229, is far below this.

## 9 Files

| path | contents |
|---|---|
| `CertifyCore.lean`, `CertifyCore/` | the library: `Sha256`, `Json`, `Dyadic`, `Inertia`, `Load`, `Check`, `KData`, `Main` (the program), `Theorems/`, `Kernel/` |
| `export/` | the exporter, the forgery generator, the drivers of `cecheck`, of the C verifier and of certify-ed's integrity check, the agreement table, the second-kernel runs (`second_kernel.py`) and the values of section 8 (`findings_values.py`) |
| `results/core/genuine/`, `results/core/mutants/` | one `cecheck` report per certificate, and `index.json` |
| `results/core/agreement.csv`, `agreement_stats.json` | the comparison of `cecheck`, certify-ed and the C verifier |
| `results/core/cev_*.json`, `python_integrity_*.json` | the C verifier's and certify-ed's results on the same files |
| `results/core/axioms.txt` | the output of `#print axioms` for every theorem |
| `results/core/second_kernel.json` | the leanchecker, lean4lean and nanoda runs of section 6 |
| `results/core/findings.json` | the exact values of section 8 (`export/findings_values.py`) |
| `results/core/summary.json` | the numbers of this document |
