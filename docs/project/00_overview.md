# Overview of CERTIFY-ED and of these documents

## What CERTIFY-ED is

Exact diagonalization (ED) computes eigenvalues and eigenvectors of a quantum Hamiltonian written as a matrix, and its
numbers are used as benchmarks for other methods and, increasingly, as lemmas inside published proofs. An ED code is
called the producer here. CERTIFY-ED is software that takes a producer's output and a separately declared model and
issues a certificate: a machine-readable list of statements about the result, each marked CERTIFIED, FAILED,
INCONCLUSIVE or NOT_CHECKED, with its evidence and with a list of what it does not say. The statements say that the
producer's matrix is the matrix of the declared model (S1), that each claimed eigenvalue lies within a stated radius
of a distinct eigenvalue (S2), that no eigenvalue below or inside the claimed range is missing (S3), what can be said
about degenerate eigenspaces and their observables (S4), whether independent solvers agree (S5, evidence only), the S2
and S3 results again in integer arithmetic with no floating-point assumption (S6), thermodynamic intervals from a
complete spectrum (S7), and the number of eigenvalues in each cluster (S8). The software never runs the producer's
code. Two applications use it to certify numbers that enter published proofs: the cluster eigenvalue of the pyrochlore
bound of Kránitz and Penc, and the η criteria of the AKLT gap proofs of Pomata and Wei.
`docs/project/software/01_architecture.md` describes the design.

The project is the work of Sarang Vehale, with Ritu Goel. Three of its rules govern how every document of this
repository speaks about results. Under decision D-12, new mathematics counts as established only after Sarang Vehale
has re-derived it; the 22 derivation documents of the code repository are therefore derived, machine-checked and
independently re-derived (a second derivation written from the statement alone, not a human check), with Sarang's
check pending. Under decision D-16, the prose of the manuscripts is Sarang's. Under decision D-61, the version the
manuscript cites is still 3.1.0 until Sarang decides whether 3.2.1 replaces it.

## The versions in brief

| Version | Date | What it was |
|---|---|---|
| 1.0.0 | May 2026 | the first CERTIFY-ED: a validator for small dense LAPACK results, preprint arXiv:2605.11787, submitted to SciPost Physics Codebases and rejected by three referee reports (decisions D-04, D-05, D-15) |
| symveig 1.0.0 | June 2026 | a separate package and preprint (arXiv:2606.16217) for sector-resolved whole-spectrum enclosures, later found to have correctness problems and merged (decisions D-33, D-42, D-53) |
| 2.0.0.dev0 | 4 October 2026 | the version 2 core: certificates with the statements S1 to S5, sector assembly, importers, the experiments E0 to E5 (decisions D-17 to D-31) |
| 3.0.0 | 18 October 2026 | one codebase: the version 2 core, the whole-spectrum enclosure that replaced symveig's engine, the exact layer (S6), the two applications, one reproduction driver, and provenance by computational source (decisions D-42 to D-44); evidence run `run_full_20261021` on Linux after the soundness fixes of 21 October (decision D-46) |
| 3.1.0 | 22 October 2026 | the fixes of the independent review of 21 October, the scope framework, Windows portability; evidence run `run_full_20261022`; the cited version (decisions D-48 to D-57) |
| 3.2.0 | 4 November 2026 | the expansion roadmap (Waves 0 and 1) and the completion plan: new statements and records, the 22 derivation documents, the independent C verifier, scale work; its evidence run failed in E4 (decisions D-58 to D-60) |
| 3.2.1 | 4 November 2026 | 3.2.0 with the S3 residual on one thread by default; evidence run `run_full_20261105` of 5 November 2026, the 3.2 evidence run (decision D-61) |

The dates are those of the release sections of `code/certify-ed/CHANGELOG.md` and of the decision log.
`docs/project/CHANGELOG_PROJECT.md` at the root of this repository gives every recorded change of the project from its first work
to 6 November 2026, and `docs/blog/BLOG.md` tells the same history as a narrative.

## How the documents are organised

The folder `docs/` holds the documentation of the project. `docs/project/math/` gives the mathematics behind the statements
and the status of each result. `docs/project/applications/` describes the two applications to published proofs.
`docs/project/software/` describes the software: its architecture, every module, the command line and the presets, the Python
interface, provenance and the source hash, the file formats and the JSON schemas, the importers, the test suite and
mutation analysis, and the C verifier. `docs/project/experiments/` describes the experiments E0 to E10, every run directory of
the archive, the three evidence runs and the comparisons between runs. `docs/project/decisions/` explains the 61 entries of
the decision log. `docs/project/failures/` records every defect, failed run and open gap. `docs/project/reproduction/` gives the
commands that recreate every result from the sources. `docs/project/CODE_INDEX.md` lists every program file of the code
repository with what it does, what it defines, the arguments of a script, the files it reads and writes where the code
names them, the package modules it imports, its place in the source and test fingerprints, and the statements or
experiments its docstring names. Outside `docs/`, `lean/` holds the Lean 4 re-verification: an exact
checker of the certificates, kernel theorems about the arithmetic, formal proofs of the statements on Mathlib and the
recomputed application numbers, with its own report `docs/lean/VERIFICATION_REPORT.md`; what the
formalization found in the derivation documents is in [failures/07_found_while_formalizing.md](failures/07_found_while_formalizing.md).

Every number in these documents is copied from a file of the project archive v5 of 6 November 2026, unpacked in
`archive_v5/` (not tracked by git; see `docs/project/sources/SOURCES.md`), and the file is named, by its path relative to
that folder. The code repository is `code/certify-ed/` in the archive: branch `dev/3.2` at `a814d28`, with the tags
`v3.1.0` = `e860a83`, `v3.2.0` = `55d9f61` and `v3.2.1` = `f3c478b`, and the run records in
`code/certify-ed/results/`. Nothing in this repository changes the code repository; in particular the computational
source hash of the 3.2.1 run, `cb73dec3e4433a3b`, is still the hash of its head (`docs/project/software/05_provenance.md`).

## Reading order

A reader who wants to know what the software certifies and how far that can be trusted can read, in order, this
overview, `docs/project/software/01_architecture.md`, `docs/project/experiments/README.md`, `docs/project/experiments/02_evidence_runs.md` and
`docs/project/failures/README.md`. A reader who wants to check a particular claim can start from the experiment document that
produced it and follow its source files. A reader who wants to run the software or recreate a result can read
`docs/project/software/03_cli_and_presets.md`, `docs/project/software/04_api.md` and then `docs/project/reproduction/README.md`, with
`docs/project/CODE_INDEX.md` as the map of the code. A reader who wants to know why the project is shaped as it is can read
`docs/project/decisions/README.md` and the decisions it points to. The mathematics and the applications are in `docs/project/math/`
and `docs/project/applications/`.
