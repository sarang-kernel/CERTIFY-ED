# Candidate reproduction run: what was re-derived

Reference run `run_full_20261021_copy`: code 7fb63586baf7ea6f (commit fe20f86), platform linux, python 3.11.15, flint C library not recorded by 3.0.0.

Candidate run `run_candidate_20261021`: code e707838f6852d474 (commit 04d95e9), platform win32, python 3.12.15, flint C library 3.3.0; started 2026-10-22T01:05:48, finished 2026-10-22T05:24:59.

## Steps

| step | candidate status | seconds | source hash at the step |
|---|---|---:|---|
| E3 | skipped (needs quspin) |  |  |
| E7 | ok | 2500.2 | e707838f6852d474 |
| E9 | ok | 3592.5 | 819671ccacd816f4 |
| E10 | ok | 9144.4 | 819671ccacd816f4 |
| E8 | ok | 287.2 | 209234087abaea2a |

The per-step source hashes differ although the computational files did not change between the steps. The fingerprint's exclusion list (certify_ed/provenance.py NON_COMPUTATIONAL) was extended while the run was in flight, and the running process computed each step's hash with the list it had imported at start, over the working tree of the moment; a post-processing script created in experiments/ during the last step made its tree 'dirty'. Both facts were reproduced after the run: the start-of-run exclusion list applied to the step commits gives exactly the recorded hashes, the current list applied to every step commit gives one hash, and `git diff` over the computational files between any two step commits is empty. The 3.1.0 evidence run must be made from one commit with no edits in flight (the manifest check 'every step ran at one computational source' then passes).

## E7 — benchmark ground energies, exact re-check

| system | U(1) dim / block dim | status ref / cand | E0 ref | E0 cand | exact radius ref | exact radius cand | ratio |
|---|---|---|---|---|---|---|---:|
| Heisenberg chain, N=20, PBC (Sz=0,k=0 block) | 184756 / 9252 | CERTIFIED / CERTIFIED | -8.90438653 | -8.90438653 | 3.635e-14 | 3.849e-14 | 1.06 |
| J1-J2 chain, J2=0.5 (Majumdar-Ghosh), N=20, PBC (Sz=0,k=0) | 184756 / 9252 | CERTIFIED / CERTIFIED | -7.5 | -7.5 | 4.205e-14 | 3.962e-14 | 0.94 |
| Heisenberg square 4x4, PBC (Sz=0) | 12870 / 12870 | CERTIFIED / CERTIFIED | -11.22848321 | -11.22848321 | 1.203e-14 | 1.872e-14 | 1.56 |
| J1-J2 square 4x4, J2=0.5, PBC (Sz=0) | 12870 / 12870 | CERTIFIED / CERTIFIED | -8.457923351 | -8.457923351 | 4.347e-14 | 3.273e-14 | 0.75 |

## E9 — pyrochlore crown

S = 1/2 reduced model (n1 = 381414, n3 = 817957; 128 sectors, largest 3890): lower bound rounded down -0.550158299 (ref) / -0.550158299 (cand); exact lower bound -0.5501582988212038 / -0.5501582988210859; delta 8.5824e-10 / 8.5812e-10; no_floating_point_assumption True / True; 749 s / 2934 s.

S = 1 at the printed couplings (all-singlet sector): eigenvalue upper bound -1.6380246425131773 (ref) / -1.6380246425131773 (cand) — bit-identical; published -1.632985; contradiction True / True; margin 5.039643e-03 / 5.039643e-03; exact re-check CERTIFIED / CERTIFIED; E0 radius (exact) 4.661e-15 / 4.403e-15.

S = 1 landscape (floating point) at c1 = 0.05397514068, c3 = 0.08288267138: 92 / 92 classes; lowest class [0, 4, 4, 0, 4, 4] at -1.6343596 (ref) / [0, 4, 4, 0, 4, 4] at -1.6343596 (cand).

## E10 — Pomata–Wei AKLT gap criteria in exact arithmetic

| lattice | pencil | largest block | positive pivots ref / cand | eta upper ref / cand | beta ref / cand | criterion holds |
|---|---:|---:|---|---|---|---|
| star | 128 | 70 | 32 / 32 (need <= 32) | 0.111043022 / 0.111043022 | 4.146296237e-05 / 4.138521439e-05 | True / True |
| square_octagon_par | 1024 | 420 | 256 / 256 (need <= 256) | 0.1061446859 / 0.1061446859 | 0.6491835829 / 0.6134172065 | None / None |
| square_octagon_perp | 1024 | 420 | 256 / 256 (need <= 256) | 0.1589663312 / 0.1589663312 | 0.506310935 / 0.5164057052 | None / None |
| cross | 4096 | 1584 | 256 / 256 (need <= 256) | 0.1997384507 / 0.1997384507 | 6121684912 / 5763978915 | True / True |
| decorated_honeycomb | 2592 | 673 | 216 / 216 (need <= 216) | 0.1530329086 / 0.1530329086 | 2.050491414 / 1.95943228 | True / True |
| decorated_square | 27648 | 4034 | 1728 / 1728 (need <= 1728) | 0.2203543193 / 0.2203543193 | 5.451273301e+13 / 5.577236809e+13 | True / True |

Square-octagon combined criterion 4 eta_perp + 2 eta_par: upper 0.8481546964 (ref) / 0.8481546964 (cand); holds True / True.

Kernel checks (H U = 0, rank U, nullity of H mod p, kernel = VBS range):

| lattice | sites | dim | VBS rank ref / cand | nullity mod p ref / cand | kernel = VBS range ref / cand |
|---|---:|---:|---|---|---|
| star | 6 | 4096 | 16 / 16 | 16 / 16 | True / True |
| square_octagon | 8 | 65536 | 64 / 64 | 64 / 64 | True / True |
| decorated_honeycomb | 9 | 62208 | 36 / 36 | 36 / 36 | True / True |
| decorated_honeycomb | 9 | 62208 | 36 / 36 | 36 / 36 | True / True |

## E8 — comparison with symveig

17 benchmark models; elapsed 161.5062459 s (ref) / 282.6976125 s (cand).

| model | L | dim | complete ref / cand | counted claims ref / cand | reference outside ref / cand | max half-width symveig / certify_ed (ref) | (cand) | ratio ref / cand |
|---|---:|---:|---|---|---|---|---|---|
| 1D Heisenberg OBC | 4 | 16 | True / True | 14 / 14 | 0, 0 / 0, 0 | 9.33e-15 / 1.24e-14 | 9.33e-15 / 1.27e-14 | 0.75 / 0.7368421053 |
| 1D Heisenberg OBC | 6 | 64 | True / True | 59 / 59 | 0, 0 / 0, 0 | 6.26e-14 / 3.38e-14 | 6.17e-14 / 3.46e-14 | 1.855263158 / 1.782051282 |
| 1D Heisenberg OBC | 8 | 256 | True / True | 242 / 242 | 0, 0 / 0, 0 | 4.56e-13 / 1.19e-13 | 4.55e-13 / 1.21e-13 | 3.828358209 / 3.750915751 |
| 1D Heisenberg OBC | 10 | 1024 | True / True | 982 / 982 | 0, 0 / 0, 0 | 3.44e-12 / 4.99e-13 | 3.44e-12 / 4.97e-13 | 6.897686833 / 6.921428571 |
| 1D Heisenberg OBC | 12 | 4096 | True / True | 3964 / 3964 | 0, 0 / 0, 0 | 2.64e-11 / 2.13e-12 | 2.64e-11 / 2.13e-12 | 12.41142143 / 12.40954564 |
| 1D Heisenberg PBC | 4 | 16 | True / True | 15 / 15 | 0, 0 / 0, 0 | 1.15e-14 / 1.72e-14 | 1.11e-14 / 1.47e-14 | 0.6710663011 / 0.7575757576 |
| 1D Heisenberg PBC | 6 | 64 | True / True | 61 / 61 | 0, 0 / 6.661338148e-16, 0 | 6.88e-14 / 7.05e-14 | 6.88e-14 / 6.16e-14 | 0.9763779528 / 1.117117117 |
| 1D Heisenberg PBC | 8 | 256 | True / True | 250 / 250 | 0, 0 / 0, 0 | 4.88e-13 / 2.22e-13 | 4.89e-13 / 2.32e-13 | 2.195608782 / 2.111217641 |
| 1D Heisenberg PBC | 10 | 1024 | True / True | 1014 / 1014 | 0, 0 / 0, 0 | 3.64e-12 / 3.07e-12 | 3.64e-12 / 9.99e-13 | 1.18452112 / 3.639857809 |
| 1D Heisenberg PBC | 12 | 4096 | True / True | 4076 / 4076 | 0, 0 / 0, 0 | 2.77e-11 / 4.41e-12 | 2.77e-11 / 4.43e-12 | 6.277620968 / 6.249046759 |
| 1D J1-J2 PBC | 4 | 16 | True / True | 16 / 16 | 0, 0 / 0, 0 | 1.20e-14 / 3.55e-14 | 1.20e-14 / 3.39e-14 | 0.3375 / 0.3540983607 |
| 1D J1-J2 PBC | 6 | 64 | True / True | 64 / 64 | 0, 0 / 0, 0 | 7.42e-14 / 8.10e-14 | 7.37e-14 / 8.50e-14 | 0.9150684932 / 0.8668407311 |
| 1D J1-J2 PBC | 8 | 256 | True / True | 254 / 254 | 0, 0 / 0, 0 | 5.30e-13 / 3.70e-13 | 5.31e-13 / 3.57e-13 | 1.432083958 / 1.488024883 |
| 1D J1-J2 PBC | 10 | 1024 | True / True | 1016 / 1016 | 0, 0 / 0, 0 | 3.99e-12 / 1.32e-12 | 3.99e-12 / 1.07e-12 | 3.018195571 / 3.739745992 |
| 1D J1-J2 PBC | 12 | 4096 | True / True | 4070 / 4070 | 0, 0 / 4.085620731e-14, 0 | 3.05e-11 / 2.26e-11 | 3.05e-11 / 5.83e-12 | 1.344836714 / 5.221361404 |
| 2D Heisenberg 3x3 PBC | 9 | 512 | True / True | 512 / 512 | 0, 0 / 0, 0 | 1.67e-12 / 1.69e-12 | 1.67e-12 / 1.12e-12 | 0.9855605146 / 1.496316942 |
| 2D Heisenberg 4x3 PBC | 12 | 4096 | True / True | 4072 / 4072 | 4.440892099e-14, 0 / 0, 0 | 3.84e-11 / 1.13e-11 | 3.84e-11 / 1.08e-11 | 3.398631645 / 3.544307951 |

The half-widths are LAPACK residual norms and the ratio is their quotient: run-stack statistics (SS-19), reported, not compared by tolerance.

Models where symveig's intervals exclude the reference eigenvalue: 1 (ref: 2D Heisenberg 4x3 PBC L=12) / 2 (cand: 1D Heisenberg PBC L=6, 1D J1-J2 PBC L=12); where certify_ed's do: 0 (ref) / 0 (cand). Whether a residual-only interval misses the reference is itself stack-dependent; the record reports the distance, and the statements compared strictly are completeness and the counted claims.

## Record-by-record comparison (experiments/compare_runs.py)

| experiment | leaves | same | float within 1e-9 | stack-dependent (within STACK_RTOL) | max rel. diff of stack-dependent | fp statistics (E8 ratios) | max rel. diff of fp statistics | timing | missing | MISMATCH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e10 | 344 | 280 | 17 | 36 | 5.843e-02 | 0 | 0.000e+00 | 11 | 0 | 0 |
| e7 | 345 | 320 | 12 | 4 | 1.074e-14 | 0 | 0.000e+00 | 9 | 0 | 0 |
| e8 | 279 | 161 | 3 | 63 | 1.682e-11 | 17 | 7.424e-01 | 35 | 0 | 0 |
| e9 | 865 | 764 | 92 | 7 | 1.180e-13 | 0 | 0.000e+00 | 2 | 0 | 0 |

No MISMATCH leaf: every statement, count, pivot sign, status and string agrees; only timings, the stack-dependent bounds (within their tolerance) and the E8 residual statistics differ.

## Numbers step of the candidate run

* `make_numbers_apps.py`: returncode 1; 51/54 checks pass; CHECK FAILED run manifest: every experiment step ok; CHECK FAILED run manifest: every step ran at one computational source ['209234087abaea2a', '819671ccacd816f4', 'None', 'e707838f6852d474']; CHECK FAILED E6-E10 ran without uncommitted changes to certify_ed/ or experiments/; skipped E6 (mutation analysis; opt-in: --with-mutation)
* `make_numbers.py`: skipped — needs records of e0, e1, e2, e3, e4, e5 (full run)

The failed manifest checks are the ones expected of a reproduction made while the branch was being integrated (a skipped step, several per-step hashes, one dirty step; see Steps above) and would fail the 3.1.0 evidence run, which is their purpose.

## Not run here

E0–E6 (and E3) need QuSpin, which does not install on this stack (quspin 1.0.1 requires the PyPI package `importlib`, which cannot be installed on Python 3.12); they are left to the reference stack. The candidate run is a two-stack reproduction of the statements, not the 3.1.0 evidence run: the evidence run must be made from the tagged 3.1.0 commit with the test step enabled (`python run.py --full --out results/run_<id>`).
