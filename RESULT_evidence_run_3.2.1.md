# RESULT: the 3.2.1 evidence run (results/run_full_20261105) and its comparison with 3.1.0

Date: 2026-11-05/06. Machine: the Windows workstation (D-54), 8 logical processors, 16 GiB; env certifyed-311 (Python 3.11.17, NumPy 2.4.4,
SciPy 1.17.1, python-flint 0.9.0 / FLINT 3.6.0, QuSpin 1.0.1, h5py 3.16.0). Command: `certify-ed reproduce --full --with-mutation
--jobs 2 --out results/run_full_20261105` from tag `v3.2.1` = f3c478b, clean tree; computational source `cb73dec3e4433a3b` at every
step (`source_dirty = False`), tests fingerprint `9da2ba19ca572962`. Wall time 15 h 55 min (06:51 to 22:46). Floating-point
standard-model probe: holds.

## Steps

| step | status | seconds | record |
|---|---|---|---|
| tests | ok | 1170 | 1172 passed, 8 skipped |
| E0 verifier validation | ok | 885 | e0_verifier_validation_20261105_102652 |
| E1 failure matrix | ok | 76 | e1_failure_matrix_20261105_104157 |
| E2 cost and memory | ok | 4854 | e2_cost_20261105_142518 (two points hit the experiment's 900 s S3 limit, see below) |
| E3 tightness | ok | 177 | e3_tightness_20261105_104254 |
| E4 workflows | ok | 1973 | e4_workflows_W1W2W3W4_20261105_095402 (all four W4 benchmarks S3 CERTIFIED under the 5 GiB cap) |
| E5 ARPACK multiplet | ok | 26050 | e5_arpack_multiplet_20261105_071109 |
| E6 mutation (64) | ok | 24918 | e6_mutation_20261105_154612: 36 killed, 28 survived, 0 timeouts (3.1.0: 38 of 64) |
| E7 exact benchmarks | ok | 1821 | e7_exact_benchmarks_20261105_092337 |
| E8 symveig comparison | ok | 284 | e8_symveig_comparison_20261105_224131 |
| E9 pyrochlore crown | ok | 2101 | e9_pyrochlore_20261105_084835 |
| E10 AKLT | ok | 5848 | e10_aklt_20261105_071106 |
| numbers | ok | | make_numbers.py 44/44, make_numbers_apps.py 61/61 (re-run after commit 44d318c5 changed the non-computational E2 check: a certifier run that hit the experiment's S3 limit and left its certificate without S3 is not a crash; the run's source hash is unchanged) |

## Why 3.2.1 and not 3.2.0

The 3.2.0 run (`results/run_full_20261104`, tag v3.2.0 = 55d9f61, 22 h) completed every step except E4: its W4 case "J1-J2 square
4x4, J2 = 0.5" (dimension 12870) ended with an access violation (rc 3221225477) under the experiment's 5 GiB job-object cap. Diagnosis
in a separate clone: `ldl_residual.sparse_ldl_residual_bound` (the D-01 a-posteriori S3 residual, new in 3.2) formed its row blocks
on eight threads; under a committed-memory cap an allocation failure inside SciPy's sparse product in a worker thread ends the
process instead of raising MemoryError. With one thread the case certifies S3 under the cap (gap >= 0.287), uncapped both settings
certify. 3.2.1 (CHANGELOG) makes one thread the default (`CERTIFY_ED_RESIDUAL_THREADS` opts in). The bound does not depend on the
thread count: `compare_runs` of the 3.2.1 run against the 3.2.0 run gives 3472 identical leaves, 0 float differences, 0 MISMATCH
(the remaining classes are timings, the sample-dependent E5/E6 records and 64 leaves missing on one side: 16 in E4, which the
3.2.0 record lacks after the crash, and 48 in E2, the certifier fields of the momentum blocks N = 22 and N = 24, where the 3.2.0
certifier reached the 900 s limit and the 3.2.1 certifier finished with S3 NOT_CHECKED; corrected 10 Nov 2026, an earlier text
attributed all 64 to E4).

Cost of the single-threaded residual: the S3 step of the E2 points at dimension 12870 takes 250-800 s instead of 60-180 s in 3.1.0
(the measured residual is the price of D-01; 3.1.0 used the a-priori bound), and two E2 points - chain N = 18 (d = 48620) and the
J2 = 0.5 square (d = 12870, the denser fill) - hit the experiment's 900 s certifier limit; in 3.1.0 both were S3 NOT_CHECKED under the
4.5 GiB memory cap, so no S3 statement is lost at those points. Their certificates without S3 (S1, S2 CERTIFIED) are in the record.

## Comparison with the 3.1.0 run (results/run_full_20261022, Windows, same stack)

`experiments/compare_runs.py <3.1.0 run> results/run_full_20261105` (`compare_vs_run_full_20261022.csv`, 3896 leaves): same 3009,
float-differs 19, stack-dependent 77, fp-statistic 13, text-numbers 19, sample-dependent 448, explained 18, timing 226, missing 57,
**MISMATCH 10**. Every MISMATCH leaf is one of the documented 3.2 changes:

| leaves | what | documented where |
|---|---|---|
| E2 `rows[*].certifier.fill_nnz` (7) | the sparse inertia record counts the fill as 2 nnz(L) - n (one factor of the symmetric factorization) instead of nnz(L) + nnz(U) | CHANGELOG, Scale track (D-01); inertia.py |
| E3 `c_gap[2].certified_lb`, `c_gap[7].certified_lb` (2) | the certified lower bound of the next eigenvalue moves up by 1.0e-9 and 1.3e-9 relative: the measured backward error beta of the sparse factorization is smaller than the 3.1 a-priori bound (4-50x on the chain blocks) | docs/proofs/D-01_sparse-residual.md (tighter bound with its proof document; independent verdict AGREES-WITH-REMARKS, Sarang's check pending) |
| E6 `copy.files` (1) | the mutation harness copies every file git tracks (`copy.source` = `git ls-files` in both runs); the repository tracks 215 files at `v3.1.0` and 915 at `v3.2.1` (corrected 10 Nov 2026: an earlier text attributed the difference to a change of the copy rule) | `git ls-tree -r --name-only v3.1.0` and `v3.2.1`; E6 `data.json` of both runs |

The certified application results are unchanged: **E9 (Kranitz-Penc crown) 893 of 893 compared leaves identical; E10 (Pomata-Wei
AKLT) 333 of 333 identical** - the exact lower bound -0.5501582988212038 for the S = 1/2 reduced model, the all-singlet upper bound
-1.6380246425131773 for S = 1 at the printed couplings, and every certified eta. E7 (exact benchmarks) 336 identical; E1 (failure
matrix) identical statuses. The float-differs leaves (19: E3 11, E4 7, E1 1, every one a radius or bound that shrank by less than 1e-9
relative, the largest 9.3e-10) are 3.2 radii that moved with the Theorem 2 constant rho (docs/proofs/thm2_polar-factor.md; cluster radii can only shrink) and the INT-05 additive terms (they can only
grow radii by multiples of 2^-1074); none changes a status.

## E6 survivors

28 of 64 mutants survived (the automatic classes of `e6_classify.py` cover 10; 18 await a human verdict). The lead's readings of all
28, by class, are in `E6_inspection_worksheet_3.2.0.md` (the 3.2.0 run's sample; the 3.2.1 sample has the same sites). Seven are
soundness-relevant test gaps of the pinning-test kind (sectors.py:57, eigenspace.py:151/147/129, exact_sectors.py:202, localops.py:189,
spectrum.py:103); none changes a statement of this run (the records were produced by the unmutated code). The tests belong to 3.2.2
together with the verdicts (RA-25).

## Status

3.2.1 has its own full run (D-44) and is comparable with 3.1.0 record by record; whether it becomes the cited version is Sarang's
decision (D-60). Until then the manuscript cites 3.1.0; `manuscript_v4/generated/` of this archive is produced from this run.
