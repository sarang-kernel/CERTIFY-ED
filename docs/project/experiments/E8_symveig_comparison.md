# E8: symveig 1.0.0 against certify_ed.spectrum

## Why this experiment

The symveig package (arXiv:2606.16217, Vehale and Goel) enclosed whole spectra sector by sector. When its source
arrived on 18 October 2026, an audit reproduced four problems (decision D-42): its claim of exact multiplicities and
of an enclosure of the whole spectrum is not proved and is false for supplied eigenpairs (P1), its sector path never
checks that the operator conserves the sector quantum number (P2; under a staggered field 58 of 64 eigenvalues lie
outside the enclosures it reports, `docs/symveig_audit.md` of the code repository), its containment check of the
global path is circular (P3), and its claim to be the first such tool is wrong (P4). The function was rebuilt as
`certify_ed.spectrum.enclose_spectrum`, which certifies counts by the cluster theorem, checks conservation and
Hermiticity exactly and uses rounding bounds of the form |A||x|. E8 compares the two on symveig's own 17 benchmark
configurations, so that the paper can say what the replacement costs and gains on the problems symveig was designed
for.

## Design

The script is `experiments/e8_symveig_comparison.py`. It builds the same matrices with symveig's model builders (one-
and two-dimensional Heisenberg models and a J1-J2 chain, open and periodic, sizes L = 4 to 12, dimensions 16 to 4096)
and runs both packages' sector paths. For each configuration it records the largest and the median half-width of the
enclosures, whether certify-ed's enclosure is complete (every eigenvalue counted), how many claims symveig counted,
the time of each, and how far each package's intervals lie from a reference eigenvalue when they exclude it. E8 runs
alone, because it measures time.

## How the instrument is checked

Both packages see the same matrices, built by symveig's own code, so the comparison is not affected by a difference of
model construction. The reference eigenvalues come from a dense solver. The half-widths are LAPACK residual norms of
the respective methods, which depend on the floating-point stack; the comparison classes them as floating-point
statistics and compares completeness and the counted claims strictly (`03_comparisons.md`).

## Results

| Quantity | 3.0.0 `run_full_20261021` | 3.1.0 `run_full_20261022` | 3.2.1 `run_full_20261105` |
|---|---|---|---|
| smallest ratio of median half-widths, symveig to certify-ed, at d = 4096 | 11 | 11 | 17 |
| largest ratio | 29 | 29 | 29 |
| fitted growth exponent of the median width, symveig | 1.45 | 1.48 | 1.48 |
| fitted growth exponent of the median width, certify-ed | 0.97 | 0.98 | 0.95 |
| fitted growth exponent of the largest width, symveig | 1.47 | 1.47 | 1.47 |
| fitted growth exponent of the largest width, certify-ed | 1.17 | 1.06 | 1.05 |
| smallest time ratio, certify-ed to symveig, at d = 4096 | 0.9 | 0.9 | 1.0 |
| largest time ratio | 2.3 | 3.0 | 2.0 |

The values are the macros of `code/certify-ed/results/<run>/paper/numbers_apps.tex` of each run. Every certify-ed
enclosure is complete in every run. At d = 4096 certify-ed's median half-widths are 17 to 29 times smaller than
symveig's in 3.2.1 (11 to 29 in 3.0.0 and 3.1.0), and its widths grow like d^0.95 against d^1.48 for symveig's
medians. certify-ed takes 1.0 to 2.0 times symveig's time at d = 4096 in 3.2.1. The per-configuration values of the
3.2.1 run, from `E8` of its `data.json` (half-widths to three digits):

| Model | L | Dimension | certify-ed complete | symveig counted claims | median half-width symveig | median half-width certify-ed | max half-width symveig | max half-width certify-ed |
|---|---|---|---|---|---|---|---|---|
| 1D Heisenberg OBC | 4 | 16 | True | 14 | 7.73e-15 | 5.58e-15 | 9.33e-15 | 1.24e-14 |
| 1D Heisenberg OBC | 6 | 64 | True | 59 | 6.08e-14 | 1.54e-14 | 6.22e-14 | 3.38e-14 |
| 1D Heisenberg OBC | 8 | 256 | True | 242 | 3.79e-13 | 5.18e-14 | 4.54e-13 | 1.19e-13 |
| 1D Heisenberg OBC | 10 | 1024 | True | 982 | 2.98e-12 | 1.98e-13 | 3.44e-12 | 4.98e-13 |
| 1D Heisenberg OBC | 12 | 4096 | True | 3964 | 2.34e-11 | 8.08e-13 | 2.64e-11 | 2.13e-12 |
| 1D Heisenberg PBC | 4 | 16 | True | 15 | 1e-14 | 9.77e-15 | 1.15e-14 | 1.51e-14 |
| 1D Heisenberg PBC | 6 | 64 | True | 61 | 5.46e-14 | 2.7e-14 | 6.93e-14 | 4.02e-14 |
| 1D Heisenberg PBC | 8 | 256 | True | 250 | 4.07e-13 | 8.2e-14 | 4.88e-13 | 1.4e-13 |
| 1D Heisenberg PBC | 10 | 1024 | True | 1014 | 3.14e-12 | 3.13e-13 | 3.64e-12 | 6.24e-13 |
| 1D Heisenberg PBC | 12 | 4096 | True | 4076 | 2.45e-11 | 1.18e-12 | 2.77e-11 | 2.73e-12 |
| 1D J1-J2 PBC | 4 | 16 | True | 16 | 1.18e-14 | 1.73e-14 | 1.2e-14 | 2.26e-14 |
| 1D J1-J2 PBC | 6 | 64 | True | 64 | 5.73e-14 | 3.64e-14 | 7.28e-14 | 5.33e-14 |
| 1D J1-J2 PBC | 8 | 256 | True | 254 | 4.42e-13 | 1.02e-13 | 5.32e-13 | 2.33e-13 |
| 1D J1-J2 PBC | 10 | 1024 | True | 1016 | 3.43e-12 | 3.9e-13 | 3.99e-12 | 6.67e-13 |
| 1D J1-J2 PBC | 12 | 4096 | True | 4070 | 2.69e-11 | 1.51e-12 | 3.05e-11 | 3.6e-12 |
| 2D Heisenberg 3x3 PBC | 9 | 512 | True | 512 | 1.66e-12 | 3.66e-13 | 1.67e-12 | 7.44e-13 |
| 2D Heisenberg 4x3 PBC | 12 | 4096 | True | 4072 | 3.39e-11 | 1.95e-12 | 3.84e-11 | 5.86e-12 |

In the Linux 3.0.0 run, symveig's intervals excluded the reference eigenvalue in one configuration (2D Heisenberg 4x3,
L = 12, by 4.440892099e-14), and in the Windows 3.1.0 run in none; certify-ed's intervals excluded it in none on
either stack (`code/certify-ed/RESULT_evidence_run_3.1.0.md`). Whether a residual-only interval misses the reference
is itself stack-dependent, which is why the record reports the distance and the comparison treats completeness and
counts as the strict statements. In 3.2.1 no configuration has a reference outside either package's intervals.

## What the result shows and does not show

E8 shows that on symveig's own benchmarks the replacement is complete everywhere, narrower by an order of magnitude at
the largest size, and between 1.0 and 2.0 times as slow in 3.2.1 (0.9 to 3.0 times in 3.1.0). It does not test P2,
because the benchmark models conserve the sector quantum number; the 58 of 64 eigenvalues outside symveig's enclosures
under a staggered field come from the audit, not from E8, and the record says so in the comment of the macro
`SymveigPtwoOutside`.
