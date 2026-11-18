# Every run directory in the archive

The folder `code/certify-ed/results/` of the archive holds nine run directories. This document lists them in time
order, says why each run was made, what it recorded and why it was kept or superseded. The facts in the table are from
the `manifest.json` and `SUMMARY.md` of each directory; the reasons are from the decision log
(`project_docs/DECISIONS_LOG.md`), the changelog (`code/certify-ed/CHANGELOG.md`) and the result files named in the
text. Earlier records of version 2 (October 2026, before the merge into 3.0.0) lived in `experiments/results/` of the
code at the time and are not part of the archive's run directories.

| Directory | Started, finished | Version, commit | Platform | Source hash | Tests step | Number scripts |
|---|---|---|---|---|---|---|
| `run_full_20261018` | 18 October 2026 15:01 to 16:21 | 3.0.0, `764a9e8` | Linux | not recorded | 334 passed | `make_numbers.py` failed, `make_numbers_apps.py` 5/5 |
| `run_full_final` | 18 October 2026 18:01 to 19:47 | 3.0.0, `98608a4` | Linux | `2025a7ca1f3aba1e` | 349 passed | 37/37 and 37/37 |
| `run_quick_final` | 18 October 2026 19:53 to 20:01 | 3.0.0, `4c6510d` | Linux | `2025a7ca1f3aba1e` | 356 passed | quick mode: `make_numbers_apps.py` 26/26 |
| `run_full_20261021` | 21 October 2026 06:09 to 07:56 | 3.0.0, `fe20f86` | Linux | `7fb63586baf7ea6f` | 363 passed | 41/41; `make_numbers_apps.py` 51/54 when regenerated with the 3.1.0 scripts |
| `run_candidate_20261021` | 22 October 2026 01:05 to 05:24 | 3.1.0.dev0, `04d95e9` | Windows | three hashes, one step dirty | not run | `make_numbers_apps.py` failed, 51/54 |
| `run_full_20261022_attempt1` | 22 October 2026 10:28 to 13:56 | 3.1.0 first cut, `ada1362` | Windows | `31ed73fd9519e64b` | 1 failed, 536 passed, 1 skipped | `make_numbers_apps.py` failed, `make_numbers.py` skipped |
| `run_full_20261022` | 22 October 2026 16:58 to 23:17 | 3.1.0, `e860a83` | Windows | `acbef11af991c860` | 541 passed, 1 skipped | 45/45 and 61/61 |
| `run_full_20261104` | 4 November 2026 08:42 to 5 November 2026 06:49 | 3.2.0, `55d9f61` | Windows | `47446976d4dbfdb1` | 1170 passed, 8 skipped | `make_numbers.py` failed, `make_numbers_apps.py` 60/61 |
| `run_full_20261105` | 5 November 2026 06:51 to 22:46 | 3.2.1, `f3c478b` | Windows | `cb73dec3e4433a3b` | 1172 passed, 8 skipped | 44/44 and 61/61 |

The Linux runs used Python 3.11.15, the Windows runs of 3.1.0 and 3.2 Python 3.11.17; all except the candidate run
used numpy 2.4.4, SciPy 1.17.1, python-flint 0.9.0 and QuSpin 1.0.1. The candidate run used Python 3.12.15, numpy
2.5.4, SciPy 1.18.1 and python-flint 0.8.0, without QuSpin.

## The runs of 18 October 2026

`run_full_20261018` was the first full run of the merged codebase certify-ed 3.0.0 (decision D-43). Its number script
`make_numbers.py` failed two checks. The first, "all experiments ran with the same certify_ed source tree", failed
because the run crossed a commit (`764a9e8` to `bcc6f12`) that changed only the command line; this spurious failure is
why decision D-44 replaced the check of equal commits by the computational source hash. The second, "E2: the N=22
momentum block has the largest S1 time", compared runs instead of blocks and was a defect of the check, fixed in the
same step. The run is superseded.

`run_full_final` was the full run at the first source hash, `2025a7ca1f3aba1e`, extended at 19:04 from the same commit
to add E6; both number scripts passed every check (37 of 37 each). `run_quick_final` is the quick mode of the same
source (E2, E4 and E5 skipped). These runs carried the evidence of 3.0.0 until 21 October 2026. They were superseded
when writing the appendix proofs against the code found two soundness gaps (decision D-46): their records come from
code paths that were then corrected, even though the numbers they report were not affected.

## The 3.0.0 run of 21 October 2026

`run_full_20261021` reran every experiment at the corrected source `7fb63586baf7ea6f` (decision D-46), on Linux, and
is the record from which reports v1.0 and v1.1 were filled. It was extended at 07:07 from commit `e999cc2` to add E6.
Its own number scripts passed (41 of 41 for `make_numbers.py`). When the 3.1.0 scripts regenerate its numbers,
`make_numbers_apps.py` reports three failed checks, because the post-run E6 files of a 3.0.0 run carry no provenance
stamp; this is intended (`code/certify-ed/docs/reproducibility.md`). It serves as the Linux side of the two-stack
comparison for 3.1.0 (`03_comparisons.md`).

## The candidate run of 22 October 2026

`run_candidate_20261021` ran E7 to E10 on the Windows workstation, with newer library versions and without QuSpin, as
a candidate reproduction of the applications before the 3.1.0 run. Its manifest records three different source hashes
and one step with uncommitted changes, although the computational files did not change between the steps.
`code/certify-ed/RESULT_candidate_run.md` gives the cause: the exclusion list `NON_COMPUTATIONAL` was extended while
the run was in flight, the running process computed each step's hash with the list it had imported at its start, and a
post-processing script created in `experiments/` during the last step made that step's tree dirty. Both facts were
reproduced after the run. The lesson, recorded in `docs/project/software/05_provenance.md`, is that an evidence run must be
made from one commit with no edits in flight. The record-by-record comparison of its E7 to E10 records with the 3.0.0
run found no MISMATCH leaf: every statement, count, pivot sign, status and string agreed, and only timings,
stack-dependent bounds within their tolerance and the residual statistics of E8 differed
(`code/certify-ed/RESULT_candidate_run.md`).

## The 3.1.0 runs of 22 October 2026

The first attempt, `run_full_20261022_attempt1` at commit `ada1362`, failed four steps (the tests, E2, E4 and E6)
because of portability defects of 3.1.0 on Windows; `docs/project/failures/04_failed_runs_and_attempts.md` describes each. Its
successful steps are valid records of that commit but are superseded, because an evidence run must be one commit for
every step. The second run, `run_full_20261022` at tag `v3.1.0` = `e860a83`, completed every step and is the 3.1.0
evidence run, still the cited run of the manuscript. `02_evidence_runs.md` gives its steps.

## The 3.2 runs of 4 and 5 November 2026

`run_full_20261104` is the evidence run of tag `v3.2.0` = `55d9f61` (decision D-60). It completed every step except
E4, which ended with an access violation in the W4 case "J1-J2 square 4x4, J2 = 0.5"; the number scripts therefore
failed. `run_full_20261105` is the evidence run of tag `v3.2.1` = `f3c478b`, which differs from 3.2.0 only in the
default thread count of the a-posteriori S3 residual, and completed every step (decision D-61). Both are described in
`02_evidence_runs.md`.
