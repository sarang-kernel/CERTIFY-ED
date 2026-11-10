# E3: tightness of the certified statements

## Why this experiment

A bound can be true and useless. If certified radii were orders of magnitude larger than the actual errors, a
certificate would say little more than "the result is roughly right". E3 measures how far each kind of bound lies from
the truth, so that the paper can say which bounds are tight and which are not. A finding that the radii were far
looser than the actual errors would have called for a sharper theorem, and E3 did locate the one place where that is
the case (the assembly of sectors).

## Design

The script is `experiments/e3_tightness.py`. Part (a) takes `eigsh` results of decreasing tolerance and compares the
certified radius of Theorem 1, the Kato-Temple interval of Theorem 3 and the actual error of the claimed eigenvalue,
the actual error being the distance of the Rayleigh quotient to the reference eigenvalue of a dense LAPACK solution of
the same matrix (`w_ref` in the script). Part (b) compares the float64 residual bounds of the standard floating-point
model with residuals computed in Arb without any model assumption. Part (c) compares the certified lower bound on the
next eigenvalue with the true next eigenvalue, and records how many factorizations the shift placement needed.
Part (d) compares the radius of the sector assembly with the actual deviation of the pooled sector values from the full
spectrum.

## How the instrument is checked

Each comparison has a known truth: the reference eigenvalue in (a), Arb's residual in (b), the true next eigenvalue in
(c), the full spectrum in (d). The ratios therefore measure the bounds, not the solver. Part (c) is explicitly a
measurement of the shift heuristic and not of the tightness of the bound, because the shift is placed at a fixed
fraction of the floating-point gap estimate (review finding MS-N07).

## Results

| Quantity | 3.0.0 `run_full_20261021` | 3.1.0 `run_full_20261022` | 3.2.1 `run_full_20261105` |
|---|---|---|---|
| (a) decades of ARPACK tolerance over which Kato-Temple tracks the error | 10 | 10 | 10 |
| (a) largest ratio of the Kato-Temple half-width to the actual error | 6 | 8 | 8 |
| (b) smallest ratio of our radii to Arb residuals | 2.6 | 2.5 | 2.5 |
| (b) largest ratio | 5.2 | 5.8 | 5.8 |
| (c) shift placed at this fraction of the gap estimate | not recorded | 0.9 | 0.9 |
| (c) cases with one factorization sufficing (distinct cases) | 11 | 11 | 11 |
| (c) cases recorded (one row duplicated) | not recorded | 12 | 12 |
| (d) sectors assembled | 134 | 134 | 134 |
| (d) assembly radius | 2.0e-10 | 1.3e-10 | 1.3e-10 |
| (d) actual error | 4.6e-14 | 5.2e-14 | 5.2e-14 |
| (d) looseness, as a power of ten | 4 | 3 | 3 |

The values are the macros of `code/certify-ed/results/<run>/paper/numbers.tex` of each run ("not recorded" means that
the number scripts of that run did not write the macro). Kato-Temple intervals track the actual error over 10 decades
of tolerance, at most 8 times above it in 3.1.0 and 3.2.1 (6 in 3.0.0). The float64 residual radii are 2.5 to 5.8
times the Arb residuals, which is the cost of the worst-case rounding terms of the standard model. In part (c) the
first shift sufficed in all 11 distinct cases. Part (d) is the weakest point on tightness: the assembly over 134
sectors gives a radius of 1.3e-10 against an actual error of 5.2e-14, about three orders of magnitude loose (2.0e-10
against 4.6e-14 in 3.0.0; the smaller radius of 3.1.0 coincides with decision D-52, under which Theorem 8 uses ρ in
place of 1.618ρ for a square W).

Between 3.1.0 and 3.2.1 the E3 record has 201 identical leaves, 11 float-differs leaves and 2 MISMATCH leaves
(`03_comparisons.md`). The two MISMATCH leaves are the certified lower bounds `c_gap[2].certified_lb` and
`c_gap[7].certified_lb`, which rise by 1.271e-09 and 1.029e-09 relative because the backward error of the sparse
factorization is measured in 3.2 instead of bounded a priori; the 11 float-differs leaves are the ten other `c_gap`
lower bounds and the invariance total of part (d), which moved by less than 1e-9. The 3.2.0 and 3.2.1 records are
identical apart from timings.

## What the result shows and does not show

E3 shows that the single-pair radii and the Kato-Temple intervals are within a factor of ten of the truth on these
problems, and that the sector assembly is the loose bound, by about three orders of magnitude. It does not show
tightness for clusters of nearly degenerate eigenvalues beyond the cases drawn, and part (c) measures how often the
shift heuristic needs a second factorization, not how tight the gap bound is.
