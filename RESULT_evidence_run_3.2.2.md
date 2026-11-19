# RESULT: the 3.2.2 evidence run (results/run_full_20261112) and its comparison with 3.2.1

Date: 2026-11-12. Machine: the Windows workstation (D-54), 8 logical processors, 16 GiB; env certifyed-311 (Python 3.11.17,
NumPy 2.4.4, SciPy 1.17.1, python-flint 0.9.0 / FLINT 3.6.0, QuSpin 1.0.1). Command: `certify-ed reproduce --full
--with-mutation --jobs 2 --out results/run_full_20261112` from tag `v3.2.2` = a9b9838, clean clone; computational source
`746958b63a65b43e` at every step (`source_dirty = False`), tests fingerprint `13f62e6cfde21503`. Wall time 15 h 34 min
(01:32 to 17:06). Floating-point standard-model probe: holds.

The machine was shared with other work for most of the run (a full test suite of `dev/3.3`, a Lean build and replay, and
load from outside this run; the platform reported between 4.2 and 9.1 busy cores of 8 at the checks made during the
run, above 7.7 at most of them). Step
times are therefore 1.6 to 4 times those of the 3.2.1 run and are not a measurement of 3.2.2 (tests 4712 s against 1170 s,
E10 10712 s against 5848 s, E5 43334 s against 26050 s, E2 producer build times about twice as long).

## Steps

| step | status | seconds | record |
|---|---|---|---|
| tests | ok | 4712 | 1211 passed, 15 skipped |
| E0 verifier validation | ok | 1441 | e0_verifier_validation_20261112_084437 |
| E1 failure matrix | ok | 112 | e1_failure_matrix_20261112_090848 |
| E2 cost and memory | ok | 3831 | e2_cost_20261112_145511 (see below) |
| E3 tightness | ok | 309 | e3_tightness_20261112_091032 |
| E4 workflows | ok | 3160 | e4_workflows_W1W2W3W4_20261112_075210 (see below) |
| E5 ARPACK multiplet | ok | 43334 | e5_arpack_multiplet_20261112_025143 |
| E6 mutation (64) | failed | 3422 | e6_mutation_20261112_155733: the unmutated baseline did not pass, no mutant was run (see below) |
| E7 exact benchmarks | ok | 3243 | e7_exact_benchmarks_20261112_065753 |
| E8 symveig comparison | ok | 652 | e8_symveig_comparison_20261112_165435 |
| E9 pyrochlore crown | ok | 4077 | e9_pyrochlore_20261112_054955 |
| E10 AKLT | ok | 10712 | e10_aklt_20261112_025125 |
| numbers | failed | | make_numbers.py (StopIteration: no completed E2 row for the square lattice at J2 = 0), make_numbers_apps.py (KeyError: the E6 record has no summary); both follow from the E2 and E6 items below |

## Comparison with the 3.2.1 run (results/run_full_20261105, same machine and stack)

`python experiments/compare_runs.py results/run_full_20261105 results/run_full_20261112 <csv>` compares 3896 leaves. E0, E1,
E3, E5, E7, E8, E9 and E10 have no MISMATCH leaf: every status, count, integer and string is identical, and the floats agree
within the tolerances of the script (the 36 sample-dependent leaves of E5 and one text-number leaf of E1 are classes of the
script, not differences of a statement). The guard range of 3.2.2 did not act anywhere: no record of the run contains
`FP_GUARD_RANGE`. The MISMATCH leaves are 11 in E2, 1 in E4 and 2 in E6.

### E2 and E4: the memory gate of 3.2.2 also gated S1

3.2.2 lets the memory left under the process memory cap gate a statement when no `budget_mb` is given (N-03). The E2
certifier runs under a job-object cap of 4608 MB, and the E4 benchmarks under the experiment's cap.

| E2 row | 3.2.1 | 3.2.2 | leaves |
|---|---|---|---|
| chain N=18, U(1) block, dimension 48620 | certifier stopped at the 900 s limit | S1 and S2 CERTIFIED; S3, S4, S8 NOT_CHECKED before any factorization (fill-model estimate 18714 MB), 9.0 s | 23 missing on the 3.2.1 side; 1 MISMATCH (S8 summary of the run without S3: "factorization skipped" against the memory text, status NOT_CHECKED in both) |
| chain N=22, k = 0, dimension 32066 | S1 CERTIFIED (sector restriction, isometry defect 7.8e-11), S2 CERTIFIED, S3 not attempted to completion, peak 2989 MB, 729 s | S1 NOT_CHECKED: "estimated 16896 MB for reference matrix from the spec exceeds the memory left under the process memory cap"; S2 CERTIFIED; S3, S8 NOT_CHECKED at 8167 MB, 3.2 s | 6 MISMATCH (S1 status and summary, twice; S8 summary, twice); the S2 radius falls from 1.26e-9 to 9.7e-14 because no model shift is added without S1 (class "explained") |
| chain N=24, k = 0, dimension 112720 | S1 NOT_CHECKED (reference build ended in MemoryError), 831 s | S1 NOT_CHECKED at an estimate of 73728 MB, S3 and S8 at 100733 MB, 4.8 s | 4 MISMATCH, all summaries of statuses that are NOT_CHECKED in both runs |
| square N=16, J2 = 0, dimension 12870 | all five statements CERTIFIED, 759 s | certifier stopped at the 900 s limit | 46 missing on the 3.2.2 side |
| square N=16, J2 = 0.5 | 900 s limit | 900 s limit | none |

In E4, the second W4 case (J1-J2 chain, J2 = 0.5, N = 20, k = 0, dimension 9252) has S1 NOT_CHECKED in 3.2.2 where 3.2.1
certified it (1 MISMATCH). The E4 record keeps no summaries; the S1 estimate of that spec is 1048576 x 120 x 2 x 32 bytes =
7680 MB, above the cap, while the first case (J2 = 0, 60 terms) estimates 3840 MB and has S1 CERTIFIED in both runs.

Reading. The S1 estimate (dimension x terms x 2 entries of 32 bytes) was made as a coarse upper bound for an explicit
budget. Under a cap it now refuses S1 for blocks that fit: the N=22 certifier of 3.2.1 certified S1 with a peak of 2989 MB
for the whole process against the estimate of 16896 MB. Where the reference build does not fit, 3.2.1 already ended it with a
caught MemoryError and an S1 NOT_CHECKED (N=24), so the gate buys nothing for S1. For S3 and S8 the gate acts as intended:
no factorization started above the headroom, and the N=18 U(1) point, which reached the 900 s limit in 3.2.1, now ends
in 9 s with S3 NOT_CHECKED. No status went from NOT_CHECKED to CERTIFIED or from CERTIFIED to FAILED; the three S1 cases
went from CERTIFIED to NOT_CHECKED. The square J2 = 0 point is a timing effect of the load (its 3.2.1 certifier took 759 s
of the 900 s allowed, the producer build times of this run are about twice those of 3.2.1), not a change of 3.2.2.

### E6: the baseline failed in the copy

E6 copies the tracked files into a temporary directory and requires the unmutated suite to pass there first. The baseline
ended after 3287 s with a failed test (record: `baseline.status = killed`; the copy has 941 files against 915 in 3.2.1, the
new files of 3.2.2). A replica of the copy in the same temporary directory gives the cause: `tests/test_review_fixes.py::
test_imp05_source_fingerprint_ignores_lost_executable_bit` fails because `git init` returns 128 both in pytest's temporary
directory and in the fallback next to the copied package, which is also under the temporary directory (1 failed, 1004
passed, 10 skipped, stopping at the first failure). In the restricted environment of this machine git cannot create a repository under the
temporary directory. The copies of the 3.2.1 run were made the same way and passed; what changed between the two runs is
not established (the temporary directory of the restricted environment now lies under an AppContainer folder). The four test files new in 3.2.2 pass in the replica (46 passed). E6 therefore did not measure 3.2.2, and the
survivor verdicts (RA-25) and the pin tests of N-02 remain the account of the 3.2 sample.

## Consequences: 3.2.3

`fix/3.2.3` (from `v3.2.2`, CHANGELOG [3.2.3]) makes two changes. The process memory cap no longer gates S1; an explicit
budget still does, and the factorizations keep the gate (test
`tests/test_memory_gate.py::test_cap_does_not_gate_s1_but_still_gates_factorizations`). E6 makes its copies next to the
repository (or in `CERTIFY_ED_E6_COPY_DIR`), and test_imp05 skips when git can work neither in its temporary directory nor
next to the package. In a copy next to the repository the test runs and passes; in the replica under the temporary directory
it now skips.

Under D-44 the 3.2.2 run is not a complete evidence run (E6 and the numbers step failed). A 3.2.3 run is needed before 3.2.3
can be cited, and it should be made on a machine without other load, because the E2 points near the 900 s limit depend on it.
The cited version stays 3.1.0 until Sarang decides (D-61).

## Status

What the 3.2.2 run shows: inside the range of the floating-point guards no statement of E0 to E10 changed, except the three S1
statements that the memory gate refused (CERTIFIED in 3.2.1, NOT_CHECKED in 3.2.2), and no record reached the guard range.
What it does not show: the mutation score of 3.2.2 and the cost of 3.2.2 on an unloaded machine.

## Note: single-precision inputs

Note of 2026-11-19 (single-precision advisory, `docs/failures/single_precision_inputs.md`). The inputs of this run are
binary64, so the advisory does not affect any number above. Checked at the run's commit a9b9838: `git grep` finds no
float16, float32, complex64, '<f4', '<c8' or np.single in experiments/, certify_ed/apps/, examples/, corpus/,
conformance/ and run.py; of the 186 data files of that commit (.npz, .npy, .h5, .mtx), every floating array is float64
except the float32 storage of the vectors in tests/fixtures/whole_spectrum_252_claim.npz, which the test converts to
float64 before use; and the run record (run_full_20261112, 58 files) contains none of these dtype names.
