# Evidence run of 3.1.0 (Windows) against the 3.0.0 run (Linux): two stacks

Reference run `run_full_20261021_copy`: code 7fb63586baf7ea6f (commit fe20f86), platform linux, python 3.11.15, flint C library not recorded by 3.0.0.

Candidate run `run_full_20261022`: code acbef11af991c860 (commit e860a83), platform win32, python 3.11.17, flint C library 3.6.0; started 2026-10-22T16:58:14, finished 2026-10-22T23:17:28.

## Steps

| step | candidate status | seconds | source hash at the step |
|---|---|---:|---|
| tests | ok: 541 passed, 1 skipped, 1 warning in 459.66s (0:07:39) | 466.4 |  |
| E10 | ok | 6805.7 | acbef11af991c860 |
| E5 | ok | 8661.6 | acbef11af991c860 |
| E9 | ok | 2434.5 | acbef11af991c860 |
| E4 | ok | 747.2 | acbef11af991c860 |
| E7 | ok | 1692.8 | acbef11af991c860 |
| E1 | ok | 40.9 | acbef11af991c860 |
| E3 | ok | 80 | acbef11af991c860 |
| E0 | ok | 656.9 | acbef11af991c860 |
| E2 | ok | 2785.8 | acbef11af991c860 |
| E6 | ok | 8573.2 | acbef11af991c860 |
| E8 | ok | 259.9 | acbef11af991c860 |

## E7 — benchmark ground energies, exact re-check

| system | U(1) dim / block dim | status ref / cand | E0 ref | E0 cand | exact radius ref | exact radius cand | ratio |
|---|---|---|---|---|---|---|---:|
| Heisenberg chain, N=20, PBC (Sz=0,k=0 block) | 184756 / 9252 | CERTIFIED / CERTIFIED | -8.90438653 | -8.90438653 | 3.635e-14 | 3.719e-14 | 1.02 |
| J1-J2 chain, J2=0.5 (Majumdar-Ghosh), N=20, PBC (Sz=0,k=0) | 184756 / 9252 | CERTIFIED / CERTIFIED | -7.5 | -7.5 | 4.205e-14 | 3.393e-14 | 0.81 |
| Heisenberg square 4x4, PBC (Sz=0) | 12870 / 12870 | CERTIFIED / CERTIFIED | -11.22848321 | -11.22848321 | 1.203e-14 | 1.769e-14 | 1.47 |
| J1-J2 square 4x4, J2=0.5, PBC (Sz=0) | 12870 / 12870 | CERTIFIED / CERTIFIED | -8.457923351 | -8.457923351 | 4.347e-14 | 1.100e-14 | 0.25 |

## E9 — pyrochlore crown

S = 1/2 reduced model (n1 = 381414, n3 = 817957; 128 sectors, largest 3890): lower bound rounded down -0.550158299 (ref) / -0.550158299 (cand); exact lower bound -0.5501582988212038 / -0.5501582988212038; delta 8.5824e-10 / 8.5824e-10; no_floating_point_assumption True / True; 749 s / 1959 s.

S = 1 at the printed couplings (all-singlet sector): eigenvalue upper bound -1.6380246425131773 (ref) / -1.6380246425131773 (cand) — bit-identical; published -1.632985; contradiction True / True; margin 5.039643e-03 / 5.039643e-03; exact re-check CERTIFIED / CERTIFIED; E0 radius (exact) 4.661e-15 / 5.093e-15.

S = 1 landscape (floating point) at c1 = 0.05397514068, c3 = 0.08288267138: 92 / 92 classes; lowest class [0, 4, 4, 0, 4, 4] at -1.6343596 (ref) / [0, 4, 4, 0, 4, 4] at -1.6343596 (cand).

S = 1 at the printed couplings, the two pair-spin sectors the paper names (floating point; new in 3.1.0): [0, 0, 0, 2, 0, 2] (dim 2495, multiplicity 6): -1.6329867; [0, 2, 2, 0, 2, 2] (dim 51315, multiplicity 3): -1.6329849. Both lie above the all-singlet bound -1.6380246425.

## E10 — Pomata–Wei AKLT gap criteria in exact arithmetic

| lattice | pencil | largest block | positive pivots ref / cand | eta upper ref / cand | beta ref / cand | criterion holds |
|---|---:|---:|---|---|---|---|
| star | 128 | 70 | 32 / 32 (need <= 32) | 0.111043022 / 0.111043022 | 4.146296237e-05 / 4.146296237e-05 | True / True |
| square_octagon_par | 1024 | 420 | 256 / 256 (need <= 256) | 0.1061446859 / 0.1061446859 | 0.6491835829 / 0.6491835829 | None / None |
| square_octagon_perp | 1024 | 420 | 256 / 256 (need <= 256) | 0.1589663312 / 0.1589663312 | 0.506310935 / 0.506310935 | None / None |
| cross | 4096 | 1584 | 256 / 256 (need <= 256) | 0.1997384507 / 0.1997384507 | 6121684912 / 6121684912 | True / True |
| decorated_honeycomb | 2592 | 673 | 216 / 216 (need <= 216) | 0.1530329086 / 0.1530329086 | 2.050491414 / 2.050491414 | True / True |
| decorated_square | 27648 | 4034 | 1728 / 1728 (need <= 1728) | 0.2203543193 / 0.2203543193 | 5.451273301e+13 / 5.451273301e+13 | True / True |

Square-octagon combined criterion 4 eta_perp + 2 eta_par: upper 0.8481546964 (ref) / 0.8481546964 (cand); holds True / True.

Kernel checks (H U = 0, rank U, nullity of H mod p, kernel = VBS range):

| lattice | sites | dim | VBS rank ref / cand | nullity mod p ref / cand | kernel = VBS range ref / cand |
|---|---:|---:|---|---|---|
| star | 6 | 4096 | 16 / 16 | 16 / 16 | True / True |
| square_octagon | 8 | 65536 | 64 / 64 | 64 / 64 | True / True |
| decorated_honeycomb | 9 | 62208 | 36 / 36 | 36 / 36 | True / True |
| decorated_honeycomb | 9 | 62208 | 36 / 36 | 36 / 36 | True / True |

## E8 — comparison with symveig

17 benchmark models; elapsed 161.5062459 s (ref) / 258.3449428 s (cand).

| model | L | dim | complete ref / cand | counted claims ref / cand | reference outside ref / cand | max half-width symveig / certify_ed (ref) | (cand) | ratio ref / cand |
|---|---:|---:|---|---|---|---|---|---|
| 1D Heisenberg OBC | 4 | 16 | True / True | 14 / 14 | 0, 0 / 0, 0 | 9.33e-15 / 1.24e-14 | 9.33e-15 / 1.24e-14 | 0.75 / 0.75 |
| 1D Heisenberg OBC | 6 | 64 | True / True | 59 / 59 | 0, 0 / 0, 0 | 6.26e-14 / 3.38e-14 | 6.22e-14 / 3.38e-14 | 1.855263158 / 1.842105263 |
| 1D Heisenberg OBC | 8 | 256 | True / True | 242 / 242 | 0, 0 / 0, 0 | 4.56e-13 / 1.19e-13 | 4.54e-13 / 1.19e-13 | 3.828358209 / 3.817164179 |
| 1D Heisenberg OBC | 10 | 1024 | True / True | 982 / 982 | 0, 0 / 0, 0 | 3.44e-12 / 4.99e-13 | 3.44e-12 / 4.98e-13 | 6.897686833 / 6.91087344 |
| 1D Heisenberg OBC | 12 | 4096 | True / True | 3964 / 3964 | 0, 0 / 0, 0 | 2.64e-11 / 2.13e-12 | 2.64e-11 / 2.13e-12 | 12.41142143 / 12.41142143 |
| 1D Heisenberg PBC | 4 | 16 | True / True | 15 / 15 | 0, 0 / 0, 0 | 1.15e-14 / 1.72e-14 | 1.15e-14 / 1.51e-14 | 0.6710663011 / 0.7647058824 |
| 1D Heisenberg PBC | 6 | 64 | True / True | 61 / 61 | 0, 0 / 0, 0 | 6.88e-14 / 7.05e-14 | 6.93e-14 / 5.97e-14 | 0.9763779528 / 1.159851301 |
| 1D Heisenberg PBC | 8 | 256 | True / True | 250 / 250 | 0, 0 / 0, 0 | 4.88e-13 / 2.22e-13 | 4.88e-13 / 2.23e-13 | 2.195608782 / 2.19123506 |
| 1D Heisenberg PBC | 10 | 1024 | True / True | 1014 / 1014 | 0, 0 / 0, 0 | 3.64e-12 / 3.07e-12 | 3.64e-12 / 1.01e-12 | 1.18452112 / 3.616692427 |
| 1D Heisenberg PBC | 12 | 4096 | True / True | 4076 / 4076 | 0, 0 / 0, 0 | 2.77e-11 / 4.41e-12 | 2.77e-11 / 4.42e-12 | 6.277620968 / 6.258492462 |
| 1D J1-J2 PBC | 4 | 16 | True / True | 16 / 16 | 0, 0 / 0, 0 | 1.20e-14 / 3.55e-14 | 1.20e-14 / 3.38e-14 | 0.3375 / 0.3552631579 |
| 1D J1-J2 PBC | 6 | 64 | True / True | 64 / 64 | 0, 0 / 0, 0 | 7.42e-14 / 8.10e-14 | 7.28e-14 / 8.44e-14 | 0.9150684932 / 0.8631578947 |
| 1D J1-J2 PBC | 8 | 256 | True / True | 254 / 254 | 0, 0 / 0, 0 | 5.30e-13 / 3.70e-13 | 5.32e-13 / 3.75e-13 | 1.432083958 / 1.419087137 |
| 1D J1-J2 PBC | 10 | 1024 | True / True | 1016 / 1016 | 0, 0 / 0, 0 | 3.99e-12 / 1.32e-12 | 3.99e-12 / 1.06e-12 | 3.018195571 / 3.763670647 |
| 1D J1-J2 PBC | 12 | 4096 | True / True | 4070 / 4070 | 0, 0 / 0, 0 | 3.05e-11 / 2.26e-11 | 3.05e-11 / 5.82e-12 | 1.344836714 / 5.232805646 |
| 2D Heisenberg 3x3 PBC | 9 | 512 | True / True | 512 / 512 | 0, 0 / 0, 0 | 1.67e-12 / 1.69e-12 | 1.67e-12 / 1.20e-12 | 0.9855605146 / 1.391707634 |
| 2D Heisenberg 4x3 PBC | 12 | 4096 | True / True | 4072 / 4072 | 4.440892099e-14, 0 / 0, 0 | 3.84e-11 / 1.13e-11 | 3.84e-11 / 9.28e-12 | 3.398631645 / 4.137779161 |

The half-widths are LAPACK residual norms and the ratio is their quotient: run-stack statistics (SS-19), reported, not compared by tolerance.

Models where symveig's intervals exclude the reference eigenvalue: 1 (ref: 2D Heisenberg 4x3 PBC L=12) / 0 (cand: none); where certify_ed's do: 0 (ref) / 0 (cand). Whether a residual-only interval misses the reference is itself stack-dependent; the record reports the distance, and the statements compared strictly are completeness and the counted claims.

## E2 — cost and memory of certification (each child capped at the same memory)

| point | dim | S3 ref / cand | certify s ref / cand | peak MiB ref / cand | cap mechanism (cand) |
|---|---:|---|---|---|---|
| chain N=12 u1 J2=0.0 | 924 | CERTIFIED / CERTIFIED | 0.1 / 0.7 | 114.6 / 66.0 | job object |
| chain N=14 u1 J2=0.0 | 3432 | CERTIFIED / CERTIFIED | 0.4 / 2.9 | 163.6 / 141.8 | job object |
| chain N=16 u1 J2=0.0 | 12870 | CERTIFIED / CERTIFIED | 9.6 / 65.5 | 1101.1 / 1057.1 | job object |
| chain N=18 u1 J2=0.0 | 48620 | NOT_CHECKED / NOT_CHECKED | 494.8 / 793.0 | 4072.2 / 4050.7 | job object |
| chain N=16 k0 J2=0.0 | 810 | CERTIFIED / CERTIFIED | 0.3 / 0.9 | 114.7 / 80.0 | job object |
| chain N=18 k0 J2=0.0 | 2704 | CERTIFIED / CERTIFIED | 1.2 / 2.7 | 182.0 / 169.7 | job object |
| chain N=20 k0 J2=0.0 | 9252 | CERTIFIED / CERTIFIED | 10.4 / 17.8 | 1008.6 / 910.9 | job object |
| chain N=22 k0 J2=0.0 | 32066 | NOT_CHECKED / NOT_CHECKED | 325.5 / 471.1 | 3323.8 / 2988.4 | job object |
| chain N=24 k0 J2=0.0 | 112720 | NOT_CHECKED / NOT_CHECKED | 292.7 / 490.2 | 3818.6 / 4213.2 | job object |
| square N=16 u1 J2=0.0 | 12870 | CERTIFIED / CERTIFIED | 24.2 / 175.4 | 1855.4 / 1830.7 | job object |
| square N=16 u1 J2=0.5 | 12870 | NOT_CHECKED / NOT_CHECKED | 75.9 / 557.5 | 4163.0 / 4122.0 | job object |

S3 outcome identical on 11 of 11 points. NOT_CHECKED means the inertia count did not fit under the cap; the certificate says so and S1+S2 are measured alone.

## E5 — ARPACK misses a multiplet member (rows depend on the stack; invariants)

Which seeds make ARPACK miss a member of the fourfold level depends on the floating-point stack, so rows are not compared one by one. What must hold on every stack is that no incorrect result is certified.

| | ref | cand |
|---|---|---|
| ARPACK runs | 12 | 12 |
| incorrect results (a member missed) | 3 | 3 |
| incorrect results certified (must be 0) | 0 | 0 |
| S3 of the full claim | [('FAILED', 3), ('INCONCLUSIVE', 9)] | [('FAILED', 3), ('INCONCLUSIVE', 9)] |
| S3 of the k lowest | [('CERTIFIED', 9), ('INCONCLUSIVE', 3)] | [('CERTIFIED', 9), ('INCONCLUSIVE', 3)] |

## E6 — mutation analysis (a sample of the code that ran)

| | ref | cand |
|---|---|---|
| mutation sites | 2834 | 3051 |
| copy given to the mutants | certify_ed/, tests/, examples/ | git ls-files |
| baseline | survived in 54 s | survived in 511 s |
| mutants | 64 | 64 |
| killed | 30 | 38 |
| timeouts | 0 | 0 |
| survived | 34 | 26 |
| score | 0.469 | 0.594 |

The two samples are drawn from different code (the mutation sites differ), so the scores describe two test suites against two codes; survivors await the human verdict of MS-N03 (e6_classify / e6_recheck).

## Record-by-record comparison (experiments/compare_runs.py)

| experiment | leaves | same | float within 1e-9 | stack-dependent (within STACK_RTOL) | max rel. diff of stack-dependent | fp statistics | max rel. diff of fp statistics | prose with run-stack digits | extended | sample-dependent | explained | timing | missing | MISMATCH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e0 | 84 | 75 | 0 | 2 | 2.054e-03 | 3 | 8.155e-02 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| e1 | 164 | 123 | 5 | 1 | 6.439e-15 | 1 | 1.498e-02 | 3 | 1 | 0 | 0 | 30 | 0 | 0 |
| e10 | 344 | 319 | 14 | 0 | 0.000e+00 | 0 | 0.000e+00 | 0 | 0 | 0 | 0 | 11 | 0 | 0 |
| e2 | 459 | 220 | 15 | 22 | 8.033e-07 | 0 | 0.000e+00 | 15 | 0 | 0 | 4 | 112 | 71 | 0 |
| e3 | 252 | 97 | 89 | 46 | 9.084e-09 | 6 | 1.421e-01 | 1 | 0 | 0 | 0 | 13 | 0 | 0 |
| e4 | 202 | 96 | 50 | 10 | 1.983e-09 | 7 | 5.541e-02 | 5 | 0 | 0 | 7 | 17 | 10 | 0 |
| e5 | 388 | 157 | 9 | 0 | 0.000e+00 | 0 | 0.000e+00 | 0 | 0 | 208 | 0 | 2 | 12 | 0 |
| e6 | 473 | 77 | 0 | 0 | 0.000e+00 | 0 | 0.000e+00 | 0 | 0 | 381 | 11 | 2 | 2 | 0 |
| e7 | 345 | 320 | 12 | 4 | 3.247e-14 | 0 | 0.000e+00 | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
| e8 | 279 | 174 | 1 | 54 | 1.683e-11 | 15 | 7.430e-01 | 0 | 0 | 0 | 0 | 35 | 0 | 0 |
| e9 | 895 | 766 | 93 | 4 | 5.266e-16 | 0 | 0.000e+00 | 0 | 0 | 0 | 0 | 2 | 30 | 0 |

Explained differences (experiments/compare_runs.py EXPLAINED):

* e2 (4 leaves): the reason text of a NOT_CHECKED S3 names the allocation failure as the platform reports it (MemoryError vs SuperLU's gstrf error); the status is compared strictly
* e4 (4 leaves): quantum numbers of a degenerate level at the cut: which member of the multiplet is listed is a tie
* e4 (3 leaves): 4x4 J2=0.5: the S3 factorization fits under a committed-memory cap (job object) and not under an address-space cap (RLIMIT_AS) of the same size
* e6 (11 leaves): the mutation sites and the sampled mutants belong to the code that ran; between two versions of the code they differ, and so do the counts and the score (a version difference, reported, not a reproduction failure)

Fields present in one record only (fields added or removed between the versions): e2 71 (candidate only); e4 10 (candidate only); e5 12 (candidate only); e6 2 (candidate only); e9 30 (candidate only).

No MISMATCH leaf: every status, count, pivot sign and statement string agrees; what differs is in one of the named classes above (timings, run-stack digits and bounds, sampled records compared by their invariants, fields added in the newer version, and the explained differences listed).

## Numbers step of the candidate run

Numbers regenerated after the run (2026-10-22T23:19:45, 2026-10-22T23:20:10, 2026-10-22T23:20:31); the list below is the latest regeneration, with the provenance of the number scripts at commit 55c2ef9.

* `make_numbers_apps.py`: returncode 0; 61/61 checks pass; skipped E6 inspection (15 survivors await a human verdict)
* `make_numbers.py`: returncode 0; 45/45 checks pass

## Coverage

Every experiment of the reference run (E0–E10) was reproduced.
