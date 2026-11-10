# certify-ed-project

This repository holds the documentation, the long-form blog, the project changelog and the Lean re-verification of
CERTIFY-ED, the software that issues certificates for exact-diagonalization results (Sarang Vehale, with Ritu Goel).
The software itself lives in its own repository; a full copy of it, with every branch, tag and run record, is in
sources/ (see below). Nothing in this repository changes the software or its run records.

## Layout

| Folder or file | Contents |
|---|---|
| `docs/` | the documentation of the whole project: mathematics, software, experiments, applications, decisions, failures, reproduction, and `CODE_INDEX.md`, which documents every program of the code repository |
| `blog/` | `BLOG.md`, a long-form account of the project from its first version to 6 November 2026, with figures |
| `CHANGELOG_PROJECT.md` | every recorded change of the project, dated, from the first work on it to 6 November 2026 |
| `reverification_lean/` | the Lean 4 re-verification: a core-Lean exact checker and kernel theorems (`CertifyCore`), formal proofs on Mathlib (`CertifyMathlib`), the recomputation of the application numbers, and `VERIFICATION_REPORT.md` |
| `tools/` | `lint_docs.py` (the style checker of STYLE.md) and the scripts that generate tables |
| `sources/` | `archive_v5/` (the project archive v5 of 6 November 2026, unpacked; not tracked by git, see `sources/SOURCES.md`) and `certify-ed_dev-3.2.bundle` (the code repository as a git bundle) |
| `STYLE.md` | the writing rules every document follows |

## Trust levels

Every result in `reverification_lean/` carries one of these levels, the same scale as the re-verification of the
mm3x3 project.

| Level | Meaning |
|---|---|
| K | a Lean theorem, accepted by the Lean kernel, with no `sorry` and no `native_decide`; its axioms are printed |
| K2 | a level K theorem that the independent kernel `nanoda` (a separate implementation of Lean's type theory, run on the exported declarations) also accepts |
| K+ | a level K theorem whose module `leanchecker` replays; this re-runs the same Lean kernel on the compiled files and guards against elaborator and file corruption, not against a kernel defect |
| N | a compiled Lean program recomputed the number or the status from the raw data and agrees with the record |
| P | the number is in a Python run record of certify-ed and has not been recomputed in Lean |
| H | checked by a named person |

A level K theorem about a mathematical statement is a formal proof of that statement as written in Lean. Whether
such a proof meets the project's rule D-12 (new mathematics counts as established only after Sarang Vehale has
re-derived it) is his decision; until he takes it, the derivation documents keep the status "Sarang's check pending".

## Provenance of these documents

The software, the derivation documents, the Lean files and these documents were prepared with the help of an
automated system, under the direction of Sarang Vehale. The record of that process (plans, reports of the
individual work tracks) is kept apart from the research material, in the agent-material package. A statement in this
repository is a human-checked result only where it names the person who checked it.

## Building

The Lean toolchain is Lean 4.35.0 (`lean-toolchain`); Mathlib is the release `v4.35.0`. See
`reverification_lean/README.md`. The style check is `python tools/lint_docs.py docs blog reverification_lean CHANGELOG_PROJECT.md`.
