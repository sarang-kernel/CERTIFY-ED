# E0: the bounds against independent references

## Why this experiment

Every certified statement rests on bounds computed in floating point under the standard model. Before any other
experiment can be read, the bounds themselves have to be shown to hold, and the check must use references that share
none of certify-ed's code: otherwise an error in a shared component would cancel out. E0 is that check. A violation in
E0 would have stopped the use of the affected statement until the bound was corrected; E0 also found the one real
defect of the reference builder before any paper result depended on it.

## Design

The script is `experiments/e0_verifier_validation.py`. Part (a) compares S2 enclosures, and the float64 residual
bounds they use, with eigenvalues and residuals computed in ball arithmetic (Arb at 128 bits, through python-flint) on
random and lattice matrices up to dimension 128. Part (b) compares the S3 inertia counts with their backward-error
bounds against the exact counts read from a dense eigensolver, for the dense and the sparse factorizations. Part (c)
compares the two reference builders of S1 with each other and with QuSpin on random operator-string Hamiltonians over
every supported local space. Part (d) is a reproducible record of the Jordan-Wigner episode of decision D-21: random
fermion models built under the old direction of the string, compared with QuSpin.

## How the instrument is checked

The references are independent by construction: Arb computes with its own interval arithmetic and does not use the
floating-point model; the exact counts come from a different algorithm (a dense eigensolver) than the counts they
check; QuSpin is an independent code. Part (d) is itself a negative control for part (c): it shows that the comparison
with QuSpin does detect a convention error when one is present.

## Results

| Quantity | 3.0.0 `run_full_20261021` | 3.1.0 `run_full_20261022` | 3.2.1 `run_full_20261105` |
|---|---|---|---|
| (a) eigenpairs checked | 6,391 | 6,391 | 6,391 |
| (a) matrices | 273 | 273 | 273 |
| (a) largest Arb ball width | 1.4e-14 | 1.4e-14 | 1.4e-14 |
| (b) inertia counts checked | 2,400 | 2,400 | 2,400 |
| (b) matrices | 300 | 300 | 300 |
| (b) median backward error β | 6.5e-12 | 6.5e-12 | 3.4e-12 |
| (b) largest backward error β | 7.5e-3 | 5.4e-3 | 4.4e-7 |
| (c) models compared | 280 | 280 | 280 |
| (c) largest difference between builders and QuSpin | 3.2e-16 | 3.2e-16 | 3.2e-16 |
| (d) models under the old string direction | 80 | 80 | 80 |
| (d) of which disagree with QuSpin | 40 | 40 | 40 |
| (d) number-conserving models, all agree | 40 | 40 | 40 |

The values are the macros of `code/certify-ed/results/<run>/paper/numbers.tex` of each run. In every run there was no
violation (the record fields `a_enclosures.violations_interval`, `a_enclosures.violations_residual_bound`,
`a_enclosures.violations_group_count` and `b_inertia.violations` are 0 in the `data.json` of each E0 record): every
enclosure of part (a) contains its Arb eigenvalue and every residual bound is above its Arb residual, and every count
of part (b) equals the exact count. In part (d), exactly the 40 models whose strings change the particle number
disagree with QuSpin under the old string direction, and all 40 number-conserving models agree, which is why the
deterministic tests had passed before E0 (decision D-21).

The backward errors of part (b) are the quantities that differ between versions. The largest β fell from 7.5e-3
(3.0.0) and 5.4e-3 (3.1.0) to 4.4e-7 in 3.2.1, and the median from 6.5e-12 to 3.4e-12. In 3.2 the sparse path measures
β by the a-posteriori residual of roadmap item D-01 instead of the a-priori SuperLU bound; a smaller β is a tighter
but still certified count, admitted on the strength of `docs/proofs/D-01_sparse-residual.md` (verdict
AGREES-WITH-REMARKS, Sarang's check pending). The 3.2.0 run's E0 record is identical to the 3.2.1 record in every leaf
except four timings (`03_comparisons.md`).

## What the result shows and does not show

E0 shows that, on the matrices it draws, the floating-point bounds contain the values computed without the
floating-point model, and that the builders agree with QuSpin to 3.2e-16. It does not show that the bounds hold for
every matrix; that is the role of the derivations (`docs/project/math/`), and the worst-case rounding terms are far from what
random tests can reach (decision D-35). The sizes are small (dimension at most 128 for part (a)), because Arb must
compute every eigenvalue in ball arithmetic.
