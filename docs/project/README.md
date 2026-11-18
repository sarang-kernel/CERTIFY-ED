# The project record of CERTIFY-ED

This part of the repository holds the documentation of the whole project, the long-form blog, the project changelog
and the Lean re-verification of CERTIFY-ED, the software that issues certificates for exact-diagonalization results
(Sarang Vehale, with Ritu Goel). Until 18 November 2026 it was a separate repository, certify-ed-project; its history
was imported with every file moved to the place given below, so `git log --follow` shows each file's earlier commits.
Nothing in it changes the software or its run records.

## Layout

| Folder or file | Contents |
|---|---|
| `docs/project/` | the documentation of the whole project: mathematics, software, experiments, applications, decisions, failures, reproduction, and `CODE_INDEX.md`, which documents every program of the code at archive v5 |
| `docs/blog/` | `BLOG.md`, a long-form account of the project from its first version to 6 November 2026, with figures |
| `docs/project/CHANGELOG_PROJECT.md` | every recorded change of the project, dated, from the first work on it |
| `lean/` | the Lean 4 re-verification: a core-Lean exact checker and kernel theorems (`CertifyCore`), formal proofs on Mathlib (`CertifyMathlib`), the recomputation of the application numbers, and their result files |
| `docs/lean/` | the reports of the Lean work, among them `VERIFICATION_REPORT.md`; paths in them without a folder are relative to `lean/` |
| `tools/` | `lint_docs.py` (the style checker of `docs/project/STYLE.md`) and the scripts that generate tables and reports |
| `docs/project/sources/` | `SOURCES.md` (where the research archive and the run records are, and how they were checked) and `runs/` (the result documents of the 3.2.2 and 3.2.3 evidence runs) |
| `docs/project/STYLE.md` | the writing rules every document of the project record follows |

The research archive v5 (`archive_v5/`) is not part of the repository; `docs/project/sources/SOURCES.md` says why and
where the generators look for it.

The documents were written while the code and the record were two repositories, so they speak of "the code
repository". Since the import that means the code of this repository: a commit id cited with it is a commit of this
repository, and a path of the form `code/certify-ed/...` is a path of the archived clone of the code at `a814d28`
(`archive_v5/code/certify-ed/`). The folder `lean/` was named `reverification_lean/` before the import; comments in the
Lean files keep the old name.

## Trust levels

Every result in `lean/` carries one of these levels, the same scale as the re-verification of the
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
individual work tracks) is kept apart from the research material, in a separate private repository. A statement in
this record is a human-checked result only where it names the person who checked it.

## Building

The Lean toolchain is Lean 4.35.0 (`lean/lean-toolchain`); Mathlib is the release `v4.35.0`. See
`docs/lean/README.md`. The style check is
`python tools/lint_docs.py docs/project docs/blog docs/lean`.
