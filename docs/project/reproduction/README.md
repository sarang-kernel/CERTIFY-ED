# Reproducing the results

This section says how to obtain again every result of the archive: the environment, the test suite and the
experiments, the numbers, figures and documents made from the records, and the checks of runs, certificates and
derivations. It also records which of these commands were run again for this documentation, on 10 November 2026, and
what they gave.

## What reproduction means here

The project's rule is that two runs of the same experiments must certify the same statements, while digits that depend
on the floating-point stack may differ (decision D-54, finding SS-19 of the review). A reproduction therefore succeeds
when the comparison of the new run with the archived run (`04_checks.md`) has no `MISMATCH` leaf; equal digits are not
required where the comparison classes them as stack-dependent, timing or sample-dependent. A version that the paper
cites needs one full run at a single computational source (decision D-44), and the source fingerprint of
`certify_ed/provenance.py` is how a run shows which source it used (`docs/software/05_provenance.md`).

## Reading order

| File | Content |
|---|---|
| `01_environment.md` | the reference versions, installation, the checkout from the bundle, shell settings, the checks of the environment |
| `02_tests_and_experiments.md` | the test suite, the full evidence run, single experiments, the scripts outside the evidence runs |
| `03_numbers_figures_documents.md` | macros and tables from a run, the figures, the manuscript, the consolidated report |
| `04_checks.md` | the comparison of runs, the verifier, the derivation scripts, the corpus, conformance, schemas, examples |

## The runs to reproduce

The archived run directories are in `code/certify-ed/results/` of the archive (the directory is not in the git bundle,
because `results/` is ignored by git). Three of them are the evidence runs of the versions that matter for the paper;
their step times are in `docs/experiments/02_evidence_runs.md` and the other runs in
`docs/experiments/01_run_history.md`.

| Version | Tag and commit | Source and test fingerprints | Run | Role |
|---|---|---|---|---|
| 3.1.0 | `v3.1.0` = `e860a83` | `acbef11af991c860`, `37eb92860974bd3d` | `run_full_20261022` | the run of the cited version |
| 3.2.0 | `v3.2.0` = `55d9f61` | `47446976d4dbfdb1`, `e21564ee2f7be8d3` | `run_full_20261104` | kept as a record; E4 failed |
| 3.2.1 | `v3.2.1` = `f3c478b` | `cb73dec3e4433a3b`, `9da2ba19ca572962` | `run_full_20261105` | the 3.2 evidence run (decision D-61) |

The cited version is still 3.1.0; whether the paper cites the 3.2.1 run is Sarang's call (decision D-61).

## Checking that a checkout matches a run

`python -m certify_ed.cli env` prints the floating-point probe, the versions and the fingerprints of the checkout. The
fingerprints can also be computed from the commits alone, without a checkout, by the script of this documentation:

```
python tools\codeindex\source_hash.py [REPO] [REV ...]
```

With no argument it reads `sources/archive_v5/code/certify-ed` and the revisions `v3.1.0`, `v3.2.0`, `v3.2.1` and
`HEAD`. It reads `NON_COMPUTATIONAL` from `certify_ed/provenance.py` as committed at each revision, lists the blobs
with `git ls-tree`, writes them in the form of `git ls-files -s`, hashes them in index order and prints one line per
revision; it never writes to the repository. On 10 November 2026 it printed the fingerprints of the table above for
the three tags, and for `HEAD` (`a814d28`, branch `dev/3.2`) the same pair as `v3.2.1`, with 12 entries of
`NON_COMPUTATIONAL` at `v3.1.0` and 27 at the later revisions. Each pair agrees with `source_hash` and `tests_hash` in
the `manifest.json` of the run in the same row.

## What was run again for this documentation

Every check below was made on 10 November 2026 on the Windows workstation, in a clone of
`sources/certify-ed_dev-3.2.bundle` at `a814d28` (the same computational source as `v3.2.1`), with the archived
records copied out where a command writes into a run directory. No full run was made.

| Check | Where described | Result |
|---|---|---|
| `python -m certify_ed.cli env` | `01_environment.md` | clean checkout, source fingerprint `cb73dec3e4433a3b` |
| `tools/codeindex/source_hash.py` | this file | the fingerprints of the table above |
| `tools/codeindex/make_code_index.py --check` | `docs/CODE_INDEX.md` | up to date with the commit `a814d28` |
| E1 alone, compared with the 3.2.1 run | `02_tests_and_experiments.md` | no `MISMATCH` |
| compare 3.1.0 with 3.2.1 from the records | `04_checks.md` | CSV equal row for row to the archived one; 10 `MISMATCH`, exit code 1 |
| `numbers` on a copy of the 3.2.1 run | `03_numbers_figures_documents.md` | 44/44 and 61/61 checks; macros and tables byte-identical, figure PDFs differ only in `/CreationDate` |
| the six record-driven figures | `03_numbers_figures_documents.md` | every `numbers.json` equal to `docs/figures_run_3.2.1/` |
| report assembly (`check_render.py`) | `03_numbers_figures_documents.md` | 0 unresolved macros, 0 placeholders; PDF not built |
| verifier build and validation | `04_checks.md` | 55 of 55 pass, 18 of 18 FAILED statuses reproduced, every forgery caught (20 of 20 in each of the four kinds); `RESULTS.md` unchanged |
| conformance and schemas | `04_checks.md` | three bundles `CONFORMANT`; three schema files in sync |
| `verify_thm8_rho.py --cases 10 --seed 1` | `04_checks.md` | 0 violations |
| planted-error corpus | `04_checks.md` | 53 of 53 planted errors detected, 0 false alarms, 7 clean controls; all 60 rows equal to the committed report |

Not run again: the test suite, the full evidence run and the single experiments other than E1, the long derivation
runs of `docs/proofs/runs/`, the examples, and the PDF builds of the manuscript and the report (no TeX engine and no
pandoc on the machine of the checks).
