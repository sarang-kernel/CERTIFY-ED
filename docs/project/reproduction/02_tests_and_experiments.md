# The test suite and the experiments

All commands are run from the root of a clean checkout of the tag to be reproduced, with the settings of
`01_environment.md`. Every command runs without interaction. Each section says what the command writes and how to tell
that the result reproduces the record of the archive.

## The test suite

```
python -m pytest -q --basetemp ..\pytest-tmp
```

The suite of `v3.2.1` gave 1172 passed and 8 skipped in the 3.2.1 run, and that of `v3.1.0` gave 541 passed and 1
skipped (`tests.summary` in `manifest.json` of `run_full_20261105` and `run_full_20261022`). The skipped tests are
those whose optional dependency or platform is absent (QuSpin, a C compiler, gcc on Linux for the `-ffast-math` test).
On the Windows workstation the 3.2 suite takes about 20 minutes; the tests step of the 3.2.1 run took 1170.3 s. `ruff
check certify_ed tests experiments conformance corpus` is the lint step and was clean at the 3.2 integration.

## The full evidence run

```
python -m certify_ed.cli reproduce --full --with-mutation --jobs 2 --out results\run_full_<date>
```

This is the command of the three evidence runs (`run.py --full --with-mutation --jobs 2` is the same). It runs the
test suite, the experiments E0 to E10 with E6 included, and the two number scripts into one run directory. Two
experiments run at a time, longest first, and E2, E6 and E8 run alone afterwards, because timing or memory is part of
what they measure (the comment on `EXCLUSIVE` in `certify_ed/cli.py`). The rule of decision D-44 applies: start from a
clean checkout of the tag and make no change in the checkout until the run has finished, not even a new untracked file
under `experiments/`, because the fingerprint of every step is computed over the working tree at that step
(`docs/project/software/05_provenance.md`). On the workstation the 3.2.1 run took from 06:51 to 22:46 on 5 November 2026, and
needs about 2.5 GB of memory per parallel job (`docs/reproducibility.md` of the repository); E2 caps each child at 4.5
GiB and E4 at 5 GiB.

A run directory holds `manifest.json` (commit, fingerprint, environment, floating-point probe, the status, time,
records and fingerprint of every step), `SUMMARY.md` (the same as a table), `run.log`, `tests.log`, one `E<k>.log` per
experiment, one record folder per experiment (`data.json`, `environment.json`, `run.log`), the compare files if made,
and `paper/` with the macros, tables and figures. An interrupted run is finished with `--resume`, which keeps every
step that succeeded at the same computational source and reruns E6 when the test fingerprint changed:

```
python -m certify_ed.cli reproduce --full --with-mutation --jobs 2 --out results\run_full_<date> --resume
```

## Single experiments

```
python -m certify_ed.cli list                                            # the experiments and their expected minutes
python -m certify_ed.cli reproduce --only E1 --no-tests --no-numbers --out ..\run_E1
python -m certify_ed.cli reproduce --full --only E7,E9,E10 --no-tests --out ..\run_apps
python -m certify_ed.cli reproduce --full --only E6 --with-mutation --no-tests --out ..\run_E6
```

`--only` takes a comma-separated list of E0 to E10, `--skip` leaves experiments out, and without `--full` the driver
runs the quick mode (E0, E1, E3, E7, E8, E9 and E10 at reduced size, E2, E4 and E5 skipped). A run of E1 alone from a
fresh clone of the bundle at `a814d28`, made on 10 November 2026, reproduced the E1 record of the 3.2.1 run with no
MISMATCH leaf (the comparison command is in `04_checks.md`). The settings that each experiment uses are in its record
(`settings`) and in `docs/project/experiments/`. The expected full-mode times printed by `list` are those of an earlier
two-core machine; the step times of the evidence runs are in `docs/project/experiments/02_evidence_runs.md`.

The experiment scripts can also be run directly, for example `python experiments\e1_failure_matrix.py`; they then
write their record to `experiments/results/<name>_<timestamp>/`, or to the folder named by the environment variable
`CERTIFY_RESULTS`. The driver is the supported path, because only the driver writes the manifest that the number
scripts check.

| Experiment | Script and arguments in full mode | Leaves | `same` | `MISMATCH` | Other classes |
|---|---|---|---|---|---|
| E0 | `e0_verifier_validation.py` | 84 | 77 | 0 | stack-dependent 2, fp-statistic 1, timing 4 |
| E1 | `e1_failure_matrix.py` | 164 | 130 | 0 | float-differs 1, text-numbers 3, timing 30 |
| E2 | `e2_cost.py` | 461 | 268 | 7 | stack-dependent 18, text-numbers 11, explained 6, timing 103, missing 48 |
| E3 | `e3_tightness.py` | 252 | 201 | 2 | float-differs 11, stack-dependent 25, timing 13 |
| E4 | `e4_workflows.py W1 W2 W3 W4 N=20` | 211 | 162 | 0 | float-differs 7, stack-dependent 9, fp-statistic 1, text-numbers 5, explained 1, timing 17, missing 9 |
| E5 | `e5_arpack_multiplet.py 12` | 388 | 327 | 0 | sample-dependent 59, timing 2 |
| E6 | `e6_mutation.py 64 2` | 473 | 70 | 1 | sample-dependent 389, explained 11, timing 2 |
| E7 | `e7_exact_benchmarks.py` | 345 | 336 | 0 | timing 9 |
| E8 | `e8_symveig_comparison.py` | 279 | 212 | 0 | stack-dependent 23, fp-statistic 11, timing 33 |
| E9 | `e9_pyrochlore.py full` | 895 | 893 | 0 | timing 2 |
| E10 | `e10_aklt.py full` | 344 | 333 | 0 | timing 11 |

The table shows, per experiment, the arguments of the 3.2.1 run (the `args` fields of `manifest.json`; E6: 64 mutants
on two workers) and how the leaves of its records compare with those of the 3.1.0 run
(`run_full_20261105/compare_vs_run_full_20261022.csv`, classes as in `04_checks.md`). It is a guide to what a new run
should give against an archived one: statuses, counts and exact bounds must fall in `same` (the script requires them
to be identical), while times, memory, sampled draws and digits that depend on the floating-point stack fall in the
other classes. The 10 `MISMATCH` leaves of this pair are the E2 `fill_nnz` counts, two E3 `c_gap` values that moved by
about 1e-9, and the E6 count of copied files; `docs/project/experiments/03_comparisons.md` explains each.

## Scripts outside the evidence runs

Three scripts of `experiments/` reproduce single results of the 3.2 work and are not run by the driver.

```
python experiments\e2_s3_reach.py --out ..\e2_s3_reach.json --budget-mb 7000 --max-N 18
python experiments\d02_matrixfree_scale.py --max-N 24 --out ..\d02_matrixfree_scale.json
python experiments\planner_calibration.py 12 14
```

`e2_s3_reach.py` measures the reach of the sparse S3 path with the a-posteriori residual for the chain S^z = 0 blocks
of dimension 924, 3432, 12870 and 48620 (roadmap item D-01, `code/certify-ed/RESULT_scale.md`).
`d02_matrixfree_scale.py` certifies S2 radii for the one-magnon state of Heisenberg chains of 16, 20, 22 and 24 sites
(limited by `--min-N` and `--max-N`), whose matrices, of dimension up to 2^24, are never stored (roadmap item D-02).
`planner_calibration.py` measures the machine rates and memory constants of the cost model (`certify_ed/cost.py`) on
the given chain sizes. Their outputs are not part of any evidence run and are not compared by `compare_runs.py`.
