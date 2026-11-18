# E4: four research workflows

## Why this experiment

E0 to E3 test the statements one by one on prepared problems. A user meets them inside a workflow: many symmetry
blocks computed separately and assembled, a degenerate multiplet whose labels matter, a matrix too large to store, or
a benchmark value quoted to many digits. E4 runs four such workflows end to end, so that the paper can show what a
certificate says in each situation and what it cannot say. A workflow that could not be certified at all would have
marked a gap in the scope of the statements.

## Design

The script is `experiments/e4_workflows.py`, and every child process runs under a 5 GiB memory cap (`WcapGiB`). W1
certifies the lowest states of every (S^z, k) block of the J1-J2 ring with N = 16 and J2 = 0.3, computed with QuSpin,
and assembles a certified low spectrum of the whole Hamiltonian (Theorem 8). W2 takes the Haldane triplet of the
spin-1 ring with N = 8 and certifies its quantum numbers and basis-independent observables. W3 is a matrix-free
producer at dimension 1,048,576 (a transverse-field Ising chain) and shows what can and cannot be certified without a
stored matrix. W4 certifies the ground energies of four commonly quoted benchmark systems; their exact re-check is E7.

## How the instrument is checked

W1's assembly is checked by the completeness of the block decomposition (every block present, mutually orthogonal,
dimensions adding up). W2 contrasts the certified trace tr(PO) with the spread of single-vector expectation values
over random bases of the triplet, a quantity that must not be reported as a property of the level. In W3 the
model-fidelity check is a random probe and is marked probabilistic in the certificate. W4's values are re-derived
without any floating-point assumption in E7, which is the independent check of the same numbers.

## Results

| Quantity | 3.0.0 `run_full_20261021` | 3.1.0 `run_full_20261022` | 3.2.1 `run_full_20261105` |
|---|---|---|---|
| W1 symmetry blocks | 242 | 242 | 242 |
| W1 producer time (s) | 6 | 22 | 21 |
| W1 certification time (s) | 18 | 48 | 53 |
| W1 certified lowest eigenvalues of H | 13 | 13 | 13 |
| W1 common radius | 5.2e-9 | 3.3e-9 | 3.3e-9 |
| W1 certified lower bound on the next eigenvalue | -5.708271755 | -5.708271753 | -5.708271753 |
| W2 certified S^z labels of the triplet | -1,0,1 | -1,-0,1 | -1,-0,1 |
| W2 bound on the labels | 1.8e-11 | 1.8e-11 | 1.8e-11 |
| W2 certified tr(P S^z_0 S^z_1) | -1.342925102940 | -1.342925102940 | -1.342925102940 |
| W2 bound on it | 8.4e-12 | 8.7e-12 | 4.7e-12 |
| W2 single-vector values, from | -0.68 | -0.66 | -0.66 |
| W2 single-vector values, to | -0.32 | -0.32 | -0.32 |
| W3 S2 radius through the reference operator | 1.9e-12 | 8.6e-12 | 8.6e-12 |
| W3 S1 probe difference | 1.1e-16 | 1.1e-16 | 1.1e-16 |
| W3 producer time (s) | 84 | 245 | 275 |
| W3 certification time (s) | 49 | 159 | 211 |

The values are the macros of `code/certify-ed/results/<run>/paper/numbers.tex` of each run. W1 certifies the 13 lowest
eigenvalues of the whole Hamiltonian from 242 blocks with a common radius of 3.3e-9 in 3.1.0 and 3.2.1 (5.2e-9 in
3.0.0), and bounds the next eigenvalue below. W2 certifies the S^z labels of the triplet as exactly -1, 0 and 1 within
1.8e-11 (the record prints the middle label as -0 in 3.1.0 and 3.2.1, a signed zero), and the observable tr(P S^z_0
S^z_1) = -1.342925102940 within a bound of 4.7e-12 in 3.2.1, while a single-vector expectation value in the triplet
ranges from -0.66 to -0.32 depending on the basis. W3 certifies S2 at dimension 1,048,576 through the reference
operator, and S1 by probes with a difference of 1.1e-16; S3 and S8 are NOT_CHECKED with the summary "no stored matrix
to factorize", and S4 is NOT_CHECKED because it requires a certified complete window (the `summaries` of `W3` in the
3.2.1 record).

The four W4 benchmarks, from the E4 records (`W4` of `data.json`):

| System | Block dimension | E0 (3.2.1 record) | 3.0.0 S3 | 3.1.0 S3, gap bound | 3.2.1 S3, gap bound | 3.2.1 radius |
|---|---|---|---|---|---|---|
| Heisenberg chain, N=20, PBC (Sz=0,k=0 block) | 9252 | -8.904386529876408 | CERTIFIED | CERTIFIED, 0.7486883846512437 | CERTIFIED, 0.74868842096234 | 3.0335081829017624e-10 |
| J1-J2 chain, J2=0.5 (Majumdar-Ghosh), N=20, PBC (Sz=0,k=0) | 9252 | -7.499999999999985 | CERTIFIED | CERTIFIED, 0.26822047931220533 | CERTIFIED, 0.26822104708744726 | 2.742543589387109e-10 |
| Heisenberg square 4x4, PBC (Sz=0) | 12870 | -11.228483208428848 | CERTIFIED | CERTIFIED, 0.5207372639661435 | CERTIFIED, 0.5207381530311305 | 1.1451257016742383e-12 |
| J1-J2 square 4x4, J2=0.5, PBC (Sz=0) | 12870 | -8.457923351394804 | NOT_CHECKED | CERTIFIED, 0.2870406984424196 | CERTIFIED, 0.28704072776169015 | 2.5616225434434264e-12 |

In the 3.0.0 run on Linux, the S3 of the J1-J2 square at J2 = 0.5 was NOT_CHECKED, because its factorization did not
fit under the 5 GiB address-space cap; on Windows, under a job-object cap of the same size, it fits and S3 is
CERTIFIED with a gap of at least 0.2870 (`03_comparisons.md`). In the 3.2.0 run this case ended the E4 process with an
access violation, the crash of the multi-threaded residual described in `02_evidence_runs.md`; the 3.2.0 record
therefore holds W1, W2, W3 and the first three W4 cases. In the 3.2.1 run all four cases certify S3 under the cap. The
W4 radii of 3.2.1 for the chains (3.0e-10 and 2.7e-10) are smaller than those of 3.1.0 (4.9e-10 and 4.4e-10). In 3.2
the cluster radius of Theorem 2 uses Kahan's constant ρ in place of (w + √(w² + 4ρ²))/2 (decision D-59,
`docs/proofs/thm2_polar-factor.md`), and the two radii shrink by close to the factor 1.618 that decision D-52 records
as the largest ratio of the old radius to ρ. The comparison script classes these leaves as stack-dependent
(`03_comparisons.md` explains why a difference of this size passes that class).

## What the result shows and does not show

E4 shows that the statements compose in realistic workflows: assembly from hundreds of blocks, labels of a degenerate
level, a matrix-free claim at dimension about 10^6, and benchmark values with certified gaps. It also shows the
limits: a matrix-free claim gets no completeness statement, and a single-vector observable of a degenerate level is
not a property of the level. The benchmark values themselves are re-checked without the floating-point model in E7.
