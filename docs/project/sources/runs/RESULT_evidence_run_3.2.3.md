# RESULT: the 3.2.3 evidence run (results/run_full_20261112b) and its comparison with 3.2.1

Date: 2026-11-12/13. Machine: the Windows workstation (D-54), 8 logical processors, 16 GiB; env certifyed-311 (Python 3.11.17,
NumPy 2.4.4, SciPy 1.17.1, python-flint 0.9.0 / FLINT 3.6.0, QuSpin 1.0.1). Command: `certify-ed reproduce --full
--with-mutation --jobs 2 --out results/run_full_20261112b` from tag `v3.2.3` = 9f42adc, clean clone; computational source
`bfc115e824386443` at every step (`source_dirty = False`), tests fingerprint `2568eaf5bf36745b`. The experiments ran from
19:06 on 12 November to 17:25 on 13 November. Floating-point standard-model probe: holds.

The numbers step first failed one check (43/44): its E2 check expected every S3 NOT_CHECKED to be a factorization that was
started and failed, while the memory gate of 3.2.2 refuses such a factorization before it starts. Commit 5dedf9d changes
only that check of `experiments/make_numbers.py` (a non-computational number script, `provenance.NON_COMPUTATIONAL`; the
source hash stays `bfc115e824386443`): it accepts a refusal by the gate, and CfailSeconds and CfailDim are taken only from
a factorization that was started and failed ("---" in this run, which has none). The run was then completed with
`--resume` at 5dedf9d (17:30 to 17:49): the test step ran again (1212 passed, 15 skipped), every experiment was kept as
the same computational source, and the numbers step passed (make_numbers.py 44/44, make_numbers_apps.py 61/61), as for
the 3.2.1 run after commit 44d318c5.

The machine was shared with other work during most of the run, so step times are not a measurement of 3.2.3 (E10 11426 s
against 5848 s in 3.2.1, E5 42493 s against 26050 s; the test step took 3233 s in the run and 1138 s in the resumed step).

## Steps

| step | status | seconds | record |
|---|---|---|---|
| tests | ok | 1138 | 1212 passed, 15 skipped (resumed step; 3233 s and the same counts in the first pass) |
| E0 verifier validation | ok | 1518 | e0_verifier_validation_20261113_020257 |
| E1 failure matrix | ok | 99 | e1_failure_matrix_20261113_022821 |
| E2 cost and memory | ok | 2832 | e2_cost_20261113_074841 |
| E3 tightness | ok | 294 | e3_tightness_20261113_022954 |
| E4 workflows | ok | 3106 | e4_workflows_W1W2W3W4_20261113_011115 |
| E5 ARPACK multiplet | ok | 42493 | e5_arpack_multiplet_20261112_200030 |
| E6 mutation (64) | ok | 31549 | e6_mutation_20261113_083548: 37 killed, 27 survived, 0 timeouts (3.2.1: 36 of 64) |
| E7 exact benchmarks | ok | 3130 | e7_exact_benchmarks_20261113_001856 |
| E8 symveig comparison | ok | 250 | e8_symveig_comparison_20261113_172137 |
| E9 pyrochlore crown | ok | 4087 | e9_pyrochlore_20261112_231051 |
| E10 AKLT | ok | 11426 | e10_aklt_20261112_200021 |
| numbers | ok | | make_numbers.py 44/44, make_numbers_apps.py 61/61 (at 5dedf9d, see above) |

## Comparison with the 3.2.1 run (results/run_full_20261105, same machine and stack)

`python experiments/compare_runs.py results/run_full_20261105 results/run_full_20261112b <csv>`: E0, E1, E3, E4, E5, E7, E8,
E9 and E10 have no MISMATCH leaf. Against the 3.2.2 run, the S1 statements of E2 (chain N=22, k = 0) and E4 (second W4
case) are CERTIFIED again, and the E6 baseline passed (2312 s).

E2 has 5 MISMATCH leaves, all summaries of S8 statements that are NOT_CHECKED in both runs: where 3.2.1 wrote "factorization
skipped" or "no clusters among the claimed pairs", 3.2.3 names the memory gate ("estimated ... MB for sparse LU
factorizations ... exceeds the memory left under the process memory cap"). Five S3 summaries of the same rows differ in the
same way and are classed as explained. 25 leaves are missing on one side. 23 are missing on the 3.2.1 side, all of the
chain N=18 U(1) point, whose 3.2.1 certifier reached the 900 s limit and whose 3.2.3 certifier ends with S1 and S2 CERTIFIED
and S3, S4, S8 NOT_CHECKED by the gate. 2 are missing on the 3.2.3 side: the entry `fill_nnz[0]` of the chain N=22 and N=24
rows at k = 0, null in 3.2.1 and absent in 3.2.3, because no factorization started. The square N=16, J2 = 0 point is certified in all five statements as in 3.2.1 (876 s of the 900 s allowed, 759 s in
3.2.1); the J2 = 0.5 point reaches the limit in both runs. No status changed between 3.2.1 and 3.2.3 except the N=18 U(1)
point, from a time-out to NOT_CHECKED.

E6 has 1 MISMATCH leaf: the copy holds 942 files against 915 (the files added in 3.2.2 and 3.2.3). The sample is not the one
of 3.2.1, because the number of mutation sites grew in `certify.py`, `fp.py`, `inertia.py` and `enclosure.py` (298 leaves of
the class sample-dependent); 37 of 64 mutants were killed (3.2.1: 36 of 64). `make_numbers_apps.py` reports 19 survivors
that await a human verdict (finding RA-25).

## Status

`run_full_20261112b` is a complete evidence run of 3.2.3 in the sense of D-44: every step ok at one computational source.
Inside the range of the floating-point guards it certifies what the 3.2.1 run certified, and no record reached the guard
range. Which version the manuscript cites (3.1.0, 3.2.1 or 3.2.3) is Sarang's decision (D-61), as are the verdicts on the
survivors.
