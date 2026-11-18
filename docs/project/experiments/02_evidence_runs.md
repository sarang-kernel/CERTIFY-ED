# The evidence runs of 3.1.0, 3.2.0 and 3.2.1

An evidence run is a full run of the test suite and of every experiment, with mutation analysis, at one computational
source and with no edits in flight, so that the number scripts can produce the manuscript's numbers from it (decisions
D-24 and D-44). Three versions received one: 3.1.0 (the cited version), 3.2.0 and 3.2.1. This document gives each run
step by step, with the failed first attempt of 3.1.0 and the E4 crash of 3.2.0, and explains why each run was made.
The step times are the `seconds` fields of each run's `manifest.json`; the rest is from
`code/certify-ed/RESULT_evidence_run_3.1.0.md`, `code/certify-ed/RESULT_evidence_run_3.2.1.md`, the `SUMMARY.md` and
log files of each run directory, and decisions D-54, D-57, D-60 and D-61.

All three runs were made on the Windows workstation that decision D-54 made the primary machine: an 11th Gen Intel
Core i5-1135G7 at 2.40 GHz with 15.8 GB of memory (the macros `MachineCPU` and `MachineRAM` of `paper/numbers.tex`), 8
logical processors, with Python 3.11.17, NumPy 2.4.4, SciPy 1.17.1, python-flint 0.9.0 with FLINT 3.6.0, QuSpin 1.0.1
and h5py 3.16.0. The run-time probe found the floating-point standard model to hold in each run. The command was the
same each time, from a clean checkout of the tag:

```
certify-ed reproduce --full --with-mutation --jobs 2 --out results/run_full_<date>
```

## Steps

| Step | 3.1.0 `run_full_20261022` | 3.2.0 `run_full_20261104` | 3.2.1 `run_full_20261105` |
|---|---|---|---|
| tests | 541 passed, 1 skipped; 466.4 s | 1170 passed, 8 skipped; 1085.0 s | 1172 passed, 8 skipped; 1170.3 s |
| E0 | ok, 656.9 s | ok, 2528.7 s | ok, 885.0 s |
| E1 | ok, 40.9 s | ok, 321.6 s | ok, 76.4 s |
| E2 | ok, 2785.8 s | ok, 5443.8 s | ok, 4853.6 s |
| E3 | ok, 80.0 s | ok, 634.8 s | ok, 176.5 s |
| E4 | ok, 747.2 s | failed, 3655.8 s | ok, 1972.6 s |
| E5 | ok, 8661.6 s | ok, 15636.1 s | ok, 26049.5 s |
| E6 | ok, 8573.2 s | ok, 50758.8 s | ok, 24917.5 s |
| E7 | ok, 1692.8 s | ok, 5410.9 s | ok, 1821.1 s |
| E8 | ok, 259.9 s | ok, 290.2 s | ok, 284.2 s |
| E9 | ok, 2434.5 s | ok, 7129.9 s | ok, 2101.1 s |
| E10 | ok, 6805.7 s | ok, 8609.1 s | ok, 5847.9 s |
| `make_numbers.py` | 45/45 checks | failed (0/0) | 44/44 checks |
| `make_numbers_apps.py` | 61/61 checks | 60/61 checks | 61/61 checks |

| Field | 3.1.0 | 3.2.0 | 3.2.1 |
|---|---|---|---|
| tag and commit | `v3.1.0` = `e860a83` | `v3.2.0` = `55d9f61` | `v3.2.1` = `f3c478b` |
| `source_hash` at every step | `acbef11af991c860` | `47446976d4dbfdb1` | `cb73dec3e4433a3b` |
| `tests_hash` | `37eb92860974bd3d` | `e21564ee2f7be8d3` | `9da2ba19ca572962` |
| started, finished | 22 October 2026 16:58, 23:17 | 4 November 2026 08:42, 5 November 2026 06:49 | 5 November 2026 06:51, 22:46 |
| E6 | 38 of 64 killed | 36 of 64 killed | 36 of 64 killed |

The step times are wall times of a workstation running two experiments at a time, and they are not measurements of the
code: E3 took 634.8 s in the 3.2.0 run and 176.5 s in the 3.2.1 run, although its records are identical leaf by leaf
apart from timing fields (`03_comparisons.md`). The timings that the experiments measure on purpose are in the E2 and
E8 records, which run alone.

## The 3.1.0 run and its first attempt

3.1.0 is the version built after the independent review of 21 October 2026: it fixes the review's findings
(`docs/project/failures/03_review_and_derivation_findings.md`), adds the scope framework of decision D-49 and the named
sectors of E9, and was the first version whose evidence run was made on the Windows workstation. The first attempt, at
commit `ada1362`, ran from 10:28 to 13:56 on 22 October 2026. The tests step reported 1 failed, 536 passed and 1
skipped; E2, E4 and E6 failed after 5.4 s, 5.0 s and 155.9 s; the number scripts then failed or were skipped for lack
of records. All four failures were portability defects of the 3.1.0 code on Windows and none was a certified
statement: a SciPy 1.17.1 change in how `LinearOperator.matmat` calls a producer's matrix-vector function, the
POSIX-only module `resource` used for the memory caps of E2 and E4, and a mutation copy that left out files six tests
read (`docs/project/failures/04_failed_runs_and_attempts.md` and `results/run_full_20261022_attempt1/ATTEMPT_NOTE.md`). The
steps that ran are valid records of that commit, but under decision D-44 an evidence run must be one commit for every
step, so the attempt was kept as a record and superseded. The tag `v3.1.0` was moved to the fixed commit `e860a83`
before it was published.

The second run completed every step at the single source `acbef11af991c860`, with 541 passed and 1 skipped tests, and
both number scripts passed every check. After the run, filling the manuscript's evidence pointers found four places
where the text and the code disagree (INT-01 to INT-04); decision D-57 kept the run as it stands, because none of them
changes a statement or status, and recorded them as known issues of 3.1.0
(`docs/project/failures/02_integration_findings.md`). This run is the cited evidence of the manuscript. Its comparison with
the Linux 3.0.0 run is the two-stack evidence for 3.1.0 (`03_comparisons.md`).

## The 3.2.0 run and the E4 crash

3.2.0 was tagged from `dev/3.2` after Phases 1 and 2 of the completion plan (decision D-60), and its run was to decide
whether the 3.2 work changes any statement of 3.1.0. The run took 22 h and completed every step except E4. In E4, the
W4 case "J1-J2 square 4x4, J2 = 0.5" (dimension 12870) ended the experiment process with return code 3221225477
(0xC0000005, a Windows access violation) under the experiment's 5 GiB job-object memory cap. The log
`results/run_full_20261104/E4.log` ends after the third W4 case, with W1, W2, W3 and the first three W4 cases
recorded; the record lacks the fourth case. Because E4 failed, `make_numbers.py` produced no numbers and
`make_numbers_apps.py` failed its check "every experiment step ok".

The reason the run was diagnosed rather than simply repeated is that a native crash is not one of the outcomes the
certifier is designed to produce: a factorization that does not fit in memory must give NOT_CHECKED, never a dead
process (decision D-20 and the `resource_failure` path of `inertia.inertia_count`). The diagnosis was made in a
separate clone so that the run's source would not change. The new a-posteriori S3 residual of 3.2
(`ldl_residual.sparse_ldl_residual_bound`, roadmap item D-01) formed its row blocks on eight threads. Under a cap on
committed memory, an allocation failure inside SciPy's sparse matrix product in a worker thread ends the whole process
instead of raising `MemoryError` in the calling thread, so the certifier's handling of memory failures never runs. The
control experiments of the diagnosis were the two remaining combinations: with one thread the case certifies S3 under
the same cap (gap at least 0.287), and without the cap both one and eight threads certify. The fix, on branch
`fix/3.2.1` and tag `v3.2.1` = `f3c478b`, makes one thread the default (`cost.residual_threads`), lets
`CERTIFY_ED_RESIDUAL_THREADS` opt in to more, and adds a test that pins the bound's independence of the thread count.
3.2.1 differs from 3.2.0 only in this default.

The E2 record of the same 3.2.0 run shows one more trace of the problem. For the point "square N=16 u1 J2=0.5"
(dimension 12870, the same matrix as the crashed W4 case), the certifier child reported a failure with an empty error
text (`certifier failed: ` in `results/run_full_20261104/E2.log`), and the record holds only the certificate without
S3. E2 runs each certifier in a child process, so the failure did not end the experiment; the record does not say why
the child ended, and that it was the same crash is an inference from the identical matrix, cap mechanism and code
path, not a recorded fact.

Decision D-61 keeps the 3.2.0 run as a record and rejects citing it with E4 patched in from a later run, because one
commit and one run is the rule.

## The 3.2.1 run

The 3.2.1 run took 15 h 55 min (06:51 to 22:46 on 5 November 2026) from a clean tree, at the source `cb73dec3e4433a3b`
at every step. Every step was ok: 1172 tests passed and 8 were skipped; all four W4 benchmarks of E4 certified S3
under the 5 GiB cap, including the case that crashed in 3.2.0 (gap at least 0.2870); E6 killed 36 of 64 mutants with
no timeouts. The number scripts passed 44 of 44 and 61 of 61 checks after one non-computational change to
`make_numbers.py` (commit `44d318c`): two E2 points hit the experiment's 900 s certifier limit on the now
single-threaded residual and left their certificates without S3, and the E2 check had treated a certifier run without
S3 as a crash. The two points are the chain N = 18 (dimension 48620) and the J2 = 0.5 square (dimension 12870); in
3.1.0 both had S3 NOT_CHECKED under the 4.5 GiB memory cap, so no S3 statement was lost there (`E2_cost.md`). Because
the change is to a non-computational file, the run's source hash is unchanged and the numbers were regenerated from
the same records; the manifest lists three regenerations, the last at 23:31 on 5 November 2026.

The cost of the single-threaded residual shows in E2: the S3 step of the points at dimension 12870 took 250 to 800 s
instead of 60 to 180 s in 3.1.0 (`RESULT_evidence_run_3.2.1.md`). The measured residual is the price of roadmap item
D-01, which replaced the a-priori bound of 3.1 by a measured backward error.

The run is the 3.2 evidence run (decision D-61). Its comparison with the 3.1.0 run and with the 3.2.0 run is in
`03_comparisons.md`; every MISMATCH leaf against 3.1.0 is a documented 3.2 change, and against 3.2.0 there is none.
Whether 3.2.1 becomes the cited version is Sarang Vehale's decision; until then the manuscript cites 3.1.0.
