# DEVELOPMENT_ROADMAP.md: how CERTIFY-ED develops after 3.2.1

This roadmap continues the expansion roadmap of 23 October 2026 (`project_docs/roadmap/ROADMAP_expansion.md`, Waves 0
to 4) with four further waves. It was written on 11 November 2026, after the 3.2.1 evidence run, archive v5 and the
Lean re-verification of this repository. Its inputs are the item list of the expansion roadmap
(`project_docs/roadmap/ROADMAP_expansion_items.csv`, read from its copy
`experiments/figures/data/ROADMAP_expansion_items_2026-11-03.csv`), the open items of
[../failures/05_open_items.md](../failures/05_open_items.md), the findings L-1 to L-11 and the list of what is not
formalized in `docs/lean/VERIFICATION_REPORT.md`, and the applications that the archive names as not
attempted. Paths without a folder of this repository are relative to `archive_v5/`.

## Where the project stands

On 3 November 2026 the expansion roadmap counted 78 items: 43 done (270 of 470 points), 6 partial, 1 blocked and 28 not
started (`project_docs/roadmap/ROADMAP_expansion.md`, section 8). Its scores combine adoption and rigor on two axes of 0
to 5, and its efforts are S (days), M (weeks), L (months) and XL (a year or more); this roadmap uses the same scale. The
Lean re-verification has since moved one item a long way: roadmap item B-05 asked for machine-checked proofs of Theorem
1 and Theorem 3 including the floating-point accounting, and those proofs now exist at level K2 (kernel-checked in Lean
and accepted by the independent kernel nanoda), with the hypotheses about binary64 arithmetic stated, so what remains of
B-05 is the link between those theorems and the code. The re-verification also produced new candidates, numbered N-01
to N-13 below, and one defect of the code: the floating-point guards hold for row widths and dimensions up to 2^26,
and the code does not check that range (finding L-3).

## Rules that constrain every wave

Four rules of the decision log shape the order. New mathematics counts as established only after Sarang Vehale has
re-derived it (D-12); every new statement therefore enters the code as UNVERIFIED with a derivation document, a verify
script and an independent second derivation, and is cited only after his check. One version gets one full evidence run
(D-44), so a release is a version with its run, not a tag alone. Manuscript prose is Sarang's (D-16), so this roadmap
plans code, proofs and records, and no text of the paper. The cited version stays 3.1.0 until Sarang decides (D-61);
3.2.2 adds a candidate to that decision and does not make it. Releases follow the version numbers: 3.2.2 is a patch that
changes no statement except that it refuses to certify where a proof does not reach, and new capabilities go to 3.3 and
later.

## Wave 5: release 3.2.2, hardening

Wave 5 closes the known gaps that need no new mathematics. The first item is the range check of finding L-3: above
w = 2^26 or n = 2^26 the statements S2 and S3 become INCONCLUSIVE with a reason code, because there the guards of
Theorem 1 are not established (and Lambda4 fails from n = 84886744 on). The second is a set of pinning tests for the
seven soundness-relevant survivors of the 3.2 mutation sample and for the test gap at `checker.py:535`, each test
shown to kill its mutant. The third is the memory gate: when no budget is given, the process memory cap decides whether
a statement is attempted, so that a factorization that cannot fit is NOT_CHECKED before it starts instead of ending the
process, as in the 3.2.0 run. The fourth is property-based testing of `fp.py` and `enclosure.py` against exact rational
arithmetic (H-02). The wave ends with the 3.2.2 evidence run and its comparison with the 3.2.1 run. Whether a 3.1.1 run
is still wanted (B-03) is Sarang's decision.

## Wave 6: the Lean checks as part of the product

The re-verification leaves two programs at level N and a set of theorems at level K2 with no formal link between them.
Wave 6 makes the link where it pays most. `cecheck` becomes a supported external verifier (N-04): a one-command driver
exports any certificate bundle and reports, statement by statement, what the exact checker confirms, and certify-ed
gains an optional path that calls a configured external verifier and records its verdict without basing any status on
it. Refinement proofs (N-05) then show on Mathlib that when the exact radius check of `CertifyCore` accepts the decoded
rational data, the conclusion of Theorem 1 holds for those data, and the same for the exact inertia count and Sylvester's
law; with them the S2 and S3 re-checks reach level K for the decoded data. The remaining gaps of the formalization
follow in order of cost: the mechanical ones in the existing floating-point model (N-06), then Sturm's theorem for C-16
(N-07), the trace norm for C-04 and C-07 (N-08), the Bunch-Kaufman propositions of E-03 (N-09) and kernel-checked
certificates for blocks larger than dimension 64 (N-10). The trusted computing base (B-04) and a uniform rounding-error
module (B-06) belong here, because the Lean floating-point model is the natural specification for both; B-06 changes
bounds and needs a derivation document. A third kernel would strengthen level K2; `lean4lean` cannot read files of Lean
4.35.0 today.

## Wave 7: new statements and applications

The statements of Wave 7 extend what a certificate can say, and each needs Sarang's check before it is cited.
Non-Hermitian problems (E-02) come first: left and right residuals and the Bauer-Fike theorem for diagonalizable
matrices with a certified bound on the condition number of the eigenvector matrix, with a Lean statement of Bauer-Fike
next to the derivation document (N-13). Then the continued-fraction bounds that complete dynamical response (C-07), time
evolution (C-08), the claim path of eigenvector fidelities (C-13) and the exact layer at larger n through multimodular
arithmetic (D-06). Two applications are open in the archive. The spin-1/2 crown certificate covers the reduced model and
not the nine-coupling headline -0.549832(8), which was reproduced in floating point only
(`project_docs/notes/claims_evidence_map.md`, row A1); it can be certified if the printed couplings determine the
problem (N-11). The AKLT honeycomb lattice was not attempted; its size decides whether this machine suffices (N-12).

## Wave 8: scale and ecosystem

Wave 8 makes the tool cheaper and easier to reach. Error-free dot products by the Ozaki scheme (D-03) would tighten the
residual and Gram bounds; parallelism across pairs and sectors (D-04) must carry a memory budget per worker and a test
that bounds do not depend on the number of workers, the lesson of the 3.2.0 crash; performance regression tests from E2
points (H-04) and a benchmark page (G-10) make cost visible per release. GPU residuals (D-05) need a GPU and macOS support
(D-07) needs a Mac, neither of which this machine has. The ecosystem items are importers and adapters: completing the
HPhi, DiagHam, SPINPACK and ALPS readers against real output files (F-02), a Julia verifier (F-03), PETSc and SLEPc
(F-04), TeNPy and ITensor (F-06), NetKet (F-07) and writers for C, C++ and Fortran codes (F-08), with the reproducibility
tools as their own package (G-08), strict typing (H-03), environment capture (H-06), mutation testing in continuous
integration (B-08) and signed releases (B-09).

## Acts of the authors

Several items can only be done by Sarang and Ritu Goel: the public push and the Zenodo version (G-01, blocked on a
credential), the software paper (G-02), journal engagement (G-06), coding the practice survey by hand (G-09), pilot
users (G-11) and an external soundness audit (B-10). Tutorials (G-04) and a certificate deposit (G-07) follow a
release that is cited. Besides these, the checks of the derivation documents under D-12, the decision on the proposal
D-62, the cited version and the E6 verdicts remain his.

## The first phase

The first phase runs now, in parallel, and touches all four directions. Hardening prepares 3.2.2 (N-01, N-02, N-03,
H-02) on a branch `fix/3.2.2`. The Lean work builds the verifier driver and hook (N-04), attempts the refinement proof of
the radius check and, if time allows, of the inertia count (N-05), and closes the mechanical gaps (N-06). The new
statement E-02 gets its derivation document, verify script, independent second derivation, an implementation behind an
UNVERIFIED marker and the Bauer-Fike theorem in Lean (N-13). The applications work attempts N-11 and N-12. The scale work
prototypes D-03 and D-04 and adds H-04. Features go to a branch `dev/3.3` (3.3.0.dev0). The phase ends with the merge, the
tag `v3.2.2`, the 3.2.2 evidence run, a rebuild of the Lean libraries and a refresh of the documentation of this
repository. Items marked "now" or "part now" in the table are in it.

## What the first phase did

The five tracks of the first phase ended on 11 November 2026; the column "Outcome" of the table gives the result of each
item, and `../failures/05_open_items.md` (section "After the first development phase") what they left open. Version
3.2.2 (branch `fix/3.2.2`, tag `v3.2.2`) carries N-01, N-02, N-03 and H-02. The branch `dev/3.3` (3.3.0.dev0) carries
3.2.3 together with E-02, the external verifier path of N-04, the prototypes of D-03 and D-04, H-04 and the code of N-11
and N-12. The Lean parts (N-04 driver, N-05, N-13) are in `lean/` of this repository, with their reports in
`docs/lean/` (`LEAN_PRODUCT.md`, `NONHERMITIAN.md`, `VERIFICATION_REPORT.md`). N-06 was not reached. The evidence run of 3.2.2
(`run_full_20261112`) did not complete; version 3.2.3 (branch `fix/3.2.3`, tag `v3.2.3`) removes its two causes, and the
evidence run of 3.2.3 (`run_full_20261112b`) completed on 13 November 2026 with no change of status against the 3.2.1
run except one E2 point that had reached the time limit.

## Every item

The table lists every open item of the expansion roadmap with its score and effort from the CSV, and the new candidates
N-01 to N-13 with an effort estimated here (they have no score). "Needs D-12" is the CSV's column `needs_D12` for the
old items. Two words of the CSV titles are replaced by plainer ones to keep
to the writing rules of `docs/project/STYLE.md`.

| Id | Item | Wave | Release | Effort | Score | Needs D-12 | First phase | Outcome |
|---|---|---|---|---|---|---|---|---|
| B-03 | Release 3.1.1 with INT-02 (scope-block validator) and INT-03 (upward-rounded exported radii) merged, its own full run (D-44) and the INT-01 decision implemented (partial) | 5 | decision | M | 5 | no |  |  |
| H-02 | Property-based tests (hypothesis) for fp.py and enclosure.py against exact rational arithmetic | 5 | 3.2.2 | S | 5 | no | now | done in 3.2.2 |
| N-01 | Refuse to certify outside the range of the floating-point guards: check w <= 2^26 and n <= 2^26 in `enclosure._rounding_bound` and the norm guards of `fp.py` (finding L-3) | 5 | 3.2.2 | S |  | no | now | done in 3.2.2 (reason code `FP_GUARD_RANGE`) |
| N-02 | Pinning tests for the seven soundness-relevant E6 survivors and the `checker.py:535` gap | 5 | 3.2.2 | S |  | no | now | done in 3.2.2; each of the 8 mutants fails the tests |
| N-03 | Memory gate that uses the process memory cap when no budget is given | 5 | 3.2.2 | S |  | no | now | done in 3.2.2; in 3.2.3 the cap no longer gates S1 |
| B-04 | Identify and measure the trusted computing base: split a `certify-ed-core` package (fp, enclosure, inertia, checker) with a counted line budget and no optional imports | 6 | 3.3 | M | 7 | no |  |  |
| B-05 | Machine-checked proofs of Theorem 1 (residual enclosure) and Theorem 3 (Kato-Temple) including the rounding model (Lean/Isabelle with Flocq-style FP) | 6 | 3.3 | XL | 8 | yes | part now | part: see N-05 |
| B-06 | Uniform rounding-error calculus module replacing the ad hoc (1+4u) factors, with property-based tests against exact rationals | 6 | 3.3 | M | 6 | yes |  |  |
| N-04 | `cecheck` as a supported external verifier: a one-command driver in this repository and an optional `verify --external` path in certify-ed that records the verdict without trusting it | 6 | 3.3 | M |  | no | now | done: driver in `lean/Verify/`, `verify --external` on `dev/3.3` |
| N-05 | Refinement proofs: the exact radius check and the exact inertia count of `CertifyCore` imply the conclusions of the Mathlib theorems, which lifts those re-checks from level N to level K | 6 | 3.3 | L |  | no | now | partial: the radius check is refined (K2 and K+); for the count only the general theorem |
| N-06 | Mechanical Lean gaps: Theorem U5 and D-01 part (c) for complex data, C-12 part (a), E-07 Lemma 5 | 6 | 3.3 | M |  | no | now | not reached |
| N-07 | Sturm's theorem and squarefree factorization on Mathlib, then C-16 | 6 | 3.4 | L |  | no |  |  |
| N-08 | Trace norm and Hölder's inequality for traces on Mathlib, then the floating-point lemmas of C-04 and C-07 | 6 | 3.4 | L |  | no |  |  |
| N-09 | E-03 Propositions 1 to 3 (Bunch-Kaufman counts in floating point) in the Lean model | 6 | 3.4 | L |  | no |  |  |
| N-10 | Kernel-checked certificates beyond dimension 64 (proof by reflection on exact data) | 6 | 3.4 | M |  | no |  |  |
| C-07 | Dynamical response: certified Lanczos continued-fraction bounds (Kaniel-Paige-Saad with a posteriori residuals), certified moments tr(P O H^k O) and sum-rule checks (partial) | 7 | 3.4 | L | 9 | yes |  |  |
| C-08 | Time evolution: a posteriori error bounds for Krylov/Chebyshev propagators; interval exponentiation on a certified full spectrum for small d | 7 | 3.4 | L | 6 | yes |  |  |
| C-13 | Fidelity and overlaps between certified eigenvectors at two couplings (Davis-Kahan sin-theta bounds) (partial) | 7 | 3.3 | M | 5 | yes |  |  |
| D-06 | Exact layer at larger n: multimodular (nmod + CRT) rational arithmetic through FLINT for S6, lifting the ~4000 dense limit | 7 | 3.4 | L | 7 | no |  |  |
| E-02 | Non-Hermitian problems: left/right residuals, Bauer-Fike for diagonalizable matrices, pseudospectral radii; Lindbladians and non-Hermitian spin chains | 7 | 3.3 | L | 7 | yes | now | on `dev/3.3`, UNVERIFIED, Sarang's check pending |
| N-11 | Certify the nine-coupling headline of the spin-1/2 crown if the printed couplings determine the problem | 7 | application | M |  | no | now | floating-point layer certified; exact re-check of all 8 blocks, count 0 in each (22 Nov 2026) |
| N-12 | AKLT honeycomb lattice: size and memory estimate, and a run if it fits this machine | 7 | application | M |  | no | now | estimate only (largest block 1293292 rows), not run |
| N-13 | Bauer-Fike and the residual form of E-02 on Mathlib | 7 | 3.3 | M |  | no | now | done for Theorem 2, its residual form and Theorem 4 (b) (K2); the count is not formalized |
| B-08 | Mutation testing in CI on a rotating sample with a survivor ledger and a verdict workflow (e6_classify/e6_recheck stamped) (partial) | 8 | 3.3 | M | 5 | no |  |  |
| B-09 | Signed, reproducible releases (sigstore) and an SBOM; certificate records the package hash (partial) | 8 | 3.3 | S | 4 | no |  |  |
| D-03 | Error-free dot products via the Ozaki scheme on BLAS (integer-split products) for compensated residual norms, 10-50x faster than fsum paths | 8 | 3.3 | M | 6 | no | now | prototype on `dev/3.3`, not the default, draft without independent re-derivation |
| D-04 | Parallelism across pairs and sectors (multiprocessing; optional MPI via mpi4py) with memory caps per worker (portable.py job objects / RLIMIT_AS) | 8 | 3.3 | M | 4 | no | now | on `dev/3.3`, one worker by default; speed-up 1.77 (2 workers) to 2.75 (8) on 11 sector claims (22 Nov 2026; second run 2.80 with 8, 27 Nov) |
| D-05 | GPU residuals (CuPy) with a GPU floating-point model probe (FMA, FTZ, rounding) and the same strict bounds | 8 | needs a GPU | L | 6 | yes |  |  |
| D-07 | macOS support and testing of the memory-cap mechanism; documented cap semantics per OS (address space vs committed memory, as seen on E4 W4) | 8 | needs macOS | S | 3 | no |  |  |
| F-02 | Importers for HPhi, DiagHam, SPINPACK and ALPS output files (eigenvalues, vectors, sector labels) (partial) | 8 | 3.3 | M | 6 | no |  |  |
| F-03 | Julia package (CertifyED.jl): the independent verifier (B-01) plus importers for Sunny.jl, KrylovKit, Expokit.jl results | 8 | 3.4 | L | 7 | no |  |  |
| F-04 | PETSc/SLEPc importer (binary vectors, parallel layouts) and a shift-invert window example | 8 | 3.4 | M | 5 | no |  |  |
| F-06 | TeNPy / ITensor adapters producing variational claims for E-05 (energy, variance, bond dimension in the record) | 8 | 3.4 | M | 7 | no |  |  |
| F-07 | NetKet / NQS adapter: energy and variance estimates with sampling error declared; certificate states what is deterministic and what is not | 8 | 3.4 | M | 5 | no |  |  |
| F-08 | Header-only C/C++ and a Fortran module writing the producer bundle from in-house codes (arrays, spec, environment) | 8 | 3.4 | M | 5 | no |  |  |
| G-08 | Reproducibility tooling as its own package (run manifests, provenance hashes, compare_runs classes, two-stack reports) | 8 | 3.4 | M | 6 | no |  |  |
| G-10 | Benchmark page: tightness and cost against Arb, VERSOFT verifyeig and INTLAB from E3, updated per release | 8 | 3.3 | S | 6 | no | part now | script on `dev/3.3`, table of 4 small points |
| H-03 | Type hints with mypy strict, a stable public API list and semantic versioning | 8 | 3.3 | M | 3 | no |  |  |
| H-04 | Performance regression tests from E2 points, run per release | 8 | 3.3 | S | 4 | no | now | on `dev/3.3`, behind `pytest --perf` |
| H-06 | Deterministic environment capture in every certificate (library versions, FP probe, CPU model, cap mechanism) and a `certify-ed env --check` command | 8 | 3.3 | S | 5 | no |  |  |
| B-10 | Invite an external soundness audit (two groups, written report) and publish it with the responses | community | Sarang | S | 6 | no |  |  |
| G-01 | Public release of 3.1.0: GitHub push (D-56), Zenodo version under the concept DOI, release notes (blocked) | community | Sarang | S | 6 | no |  |  |
| G-02 | Software paper (JOSS or SciPost Codebases) for the package, separate from the physics paper | community | Sarang | M | 7 | no |  |  |
| G-04 | Tutorial notebooks and a 20-minute recorded walkthrough; a module for a summer school | community | after 3.3 | M | 5 | no |  |  |
| G-06 | Journal engagement: a short proposal to SciPost/PRB editors for a 'certificate attached' statement in methods, backed by the practice survey | community | Sarang | S | 6 | no |  |  |
| G-07 | Opt-in certificate deposit: a Zenodo community for certificates and bundles, with the hash in the paper | community | after 3.3 | M | 5 | no |  |  |
| G-09 | Code the practice survey by hand (agreement below 80 % on 9 of 13 fields today) and publish it as a short preprint | community | Sarang | M | 6 | no |  |  |
| G-11 | Pilot users: three external groups certify one published result each, with a write-up of what blocked them | community | Sarang | M | 7 | no |  |  |
