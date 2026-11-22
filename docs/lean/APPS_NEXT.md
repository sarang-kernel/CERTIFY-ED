# APPS_NEXT.md: the nine-coupling crown and the honeycomb AKLT criterion

This part takes up the two applications that the archive names as open, roadmap items N-11 and N-12 of
`docs/project/roadmap/DEVELOPMENT_ROADMAP.md` in this repository. N-11 asks whether the nine couplings that Kránitz and Penc
print for their headline spin-1/2 bound determine the crown Hamiltonian, and if so for a certified value of its lowest
eigenvalue. N-12 asks which paper states a finite-size criterion for the AKLT model on the honeycomb lattice, which
cluster that criterion needs, and whether the eta machinery of certify-ed can decide it on this machine. The code is
on the code branch `track/applications` (based on `dev/3.2` at `fa86456`):
`certify_ed/apps/pyrochlore/nine_coupling.py`, `certify_ed/apps/aklt/estimate.py`, the builder `honeycomb_method3` in
`certify_ed/apps/aklt/lattices.py`, the scripts `experiments/n11_crown_nine.py` and
`experiments/n12_honeycomb_estimate.py`, and the tests `tests/test_apps_applications_next.py`. The records are in
`results/apps_next/`; paths of archive files are relative to `archive_v5/`, other paths to
`lean/`.

The outcome in two sentences. The printed couplings determine the problem once the digit that the paper writes in
parentheses is read as one more decimal, and at those couplings the lowest eigenvalue of the crown lies in
[-0.5498329145683885, -0.5498329144473251], certified by the floating-point layer of certify-ed under the binary64
model; the exact re-check of all 8 blocks, which needs no such model, raises the lower end to -0.5498329145062226
(section 3; it rests on Lemmas 11 and 12 and Proposition 13 of the exact layer, which are marked UNVERIFIED). The
honeycomb criterion of Pomata and Wei does not fit this machine, or any single machine, with the present eta machinery:
its largest bordered block has 1293292 rows, which needs 12462 GiB for one binary64 copy alone.

## 1 How to run

From the root of a clone of the code branch, with the environment `certifyed-311`:

```
set CERTIFY_RESULTS=<output folder>
python experiments/n11_crown_nine.py appended --float-layer
python experiments/n11_crown_nine.py appended
python experiments/n11_crown_nine.py appended --blocks 0,1,2,3
python experiments/n11_crown_nine.py appended --blocks 4,5,6,7
python experiments/n11_crown_nine.py --combine <run folder> <run folder>
python experiments/n12_honeycomb_estimate.py
python -m pytest -q tests/test_apps_applications_next.py --basetemp <scratch folder>
```

The crown runs cap their own memory at 5 GiB (a job object on Windows); the first counts with the floating-point layer
and is the record `results/apps_next/n11_crown_nine_float.json`, the second is the exact re-check in one process, and
the next three are the same re-check split into two block sets and joined (section 3). On 22 November 2026 the
exact re-check was run as 8 processes, `python experiments/n11_crown_nine.py appended --blocks i --cap-gib 16` for
i = 0 to 7, joined by `--combine` (record `results/apps_next/n11_crown_nine_exact_8_blocks.json`). The last writes
`results/apps_next/n12_honeycomb_estimate.json`. The values read from the arXiv sources of the three papers, with the
SHA-256 of each source file as downloaded on 11 November 2026, are in `results/apps_next/sources_extract.json`, and
the floating-point values of the other readings of the printed digits in `results/apps_next/float_readings.json`.

## 2 N-11: what the paper prints, and whether it determines the problem

Why. The certified crown bound of the archive, -0.5501582988212038, is the value of the reduced model in which the two
outer sites of each tetrahedron are equivalent (`c3 = c12`) and the second-neighbour coupling `c5` is zero; it
confirms the reduced-model value of the paper and not its headline -0.549832(8)
(`project_docs/notes/claims_evidence_map.md`, row A1). The headline was reproduced only in floating point, as
-0.54983289, by the independent review of 21 October 2026 (`review/REVIEW_B_applications.md`, finding B2-1). A
certified value at the printed couplings would turn the headline into a statement without an unchecked floating-point
eigenvalue, provided the printed numbers fix the Hamiltonian.

What was done. Appendix D of the paper (arXiv:2601.07800, file `Pyrochlore_lower_bound_final.tex`) defines the crown
Hamiltonian as a sum over 19 orbits of site pairs under the point group of the cluster, each orbit with one coupling
`c_m` times the Heisenberg exchange `S_i . S_j`; its Table V lists the 153 pairs of the 18 sites orbit by orbit. The
appendix states the matching conditions of the embedding into the lattice with nearest-neighbour exchange only (`J =
1`, `J_2 = J_3a = J_3b = 0`), which fix `c13 = -c6/2`, `c15 = -c7`, `c17 = J/2 - c1 - 2 c3 - 2 c12`, `c18 = -2 (c2 +
c5 + c11/2)` and `c19 = -2 c4` and set `c8`, `c9`, `c10`, `c14` and `c16` to zero, and it prints the nine free
couplings `c1`, `c2`, `c3`, `c4`, `c5`, `c6`, `c7`, `c11` and `c12` next to `e_LB = -0.549832(8)`. The table was
parsed from the source and compared with the table typed into `nine_coupling.py`; the explicit Hamiltonian of Eq. (D3)
was parsed as well, and each of its 14 coefficient groups was compared with one orbit of the table. The
nearest-neighbour orbits `o1`, `o3`, `o12` and `o17` were compared with the 36 bonds of the reduced model in
`certify_ed/apps/pyrochlore/model.py`, and the site permutations that map every orbit onto itself were enumerated.

How the instrument was checked. The comparisons are equalities of integer pairs, so one misplaced site would fail
them. The test `test_nine_coupling_orbit_table_matches_reduced_model` builds the nine-coupling model at the reduced
couplings (`c1 = 381414/10^7`, `c3 = c12 = 817957/10^7`, every other free coupling zero) and checks that it has
exactly the 36 bonds and couplings of `model.py`; `test_nine_coupling_printed_values_and_embedding` checks the parsing
of the printed values and that a coupling changed by `10^-9` violates the matching conditions.

What happened. The table of the source equals the table of the code and partitions the 153 pairs, and the explicit
form agrees with it orbit by orbit: every coefficient group of Eq. (D3) is exactly one orbit, 123 bonds in all, with
the derived coefficients of the matching conditions. The nearest-neighbour orbits are the bonds of `model.py` with the
same coupling names. The orbit-preserving permutations form a group of order 12 with one identity, seven involutions,
two elements of order 3 and two of order 6, the point group D3d of the cluster; the swaps of the outer pairs are not
in it. The paper does not say what the digit in parentheses means. Read as one more decimal, `c1 = 0.037416(7)` is
0.0374167; read as a digit to drop, it is 0.037416. The first reading is the one the project already uses for the
reduced model, whose printed couplings `c1 = 0.038141(4)` and `c3 = 0.081795(7)` are the ratio 381414/817957 of
experiment E9, and it is the default of `nine_coupling.py`; all certified numbers below are at it. In floating point
the lowest eigenvalue on S^z = 0 is -0.5498329144473226 with the digits appended, -0.5498335636261985 with the digits
dropped, and -0.5498328933843968 with every digit appended except that of `c4` (`c4 = 0.0000415`), which is how the
script of the review typed the couplings and reproduces its -0.54983289.

What it shows and does not show. Nothing besides the reading of the parenthesized digit is missing: with the table,
the matching conditions and the nine values the Hamiltonian is fixed, with exact rational couplings (all couplings are
listed in the record). Whichever reading is chosen, couplings that satisfy the matching conditions exactly give a
valid Anderson-type bound `e_GS >= lambda_min(H_0)` by the argument of the paper, so a certificate at the printed
couplings is a certificate of a lower bound on the ground-state energy per site. The argument and the matching
conditions are taken from the paper and are not re-derived; whether the embedding conditions hold on the real
pyrochlore lattice was not checked here, as it was not checked by the review (finding B2-5).

## 3 N-11: the certified enclosure at the printed couplings

Why. With the problem fixed, the remaining question is the value. The method is that of the reduced-model certificate
(`certify_ed/apps/pyrochlore/spin_half.py`) with the symmetry that remains. Every multiplet of 18 spins 1/2 has
integer total spin and so a member with S^z = 0, and a count of the eigenvalues of every symmetry block below a shift
bounds the lowest eigenvalue from below.

What was done. `certify_nine` takes the nine printed values as exact rationals, derives the other couplings from the
matching conditions, and multiplies the Hamiltonian by `K = 80000000` so that every coupling of the model
specification is an integer. On the 48620 states of S^z = 0 it uses three commuting symmetries, a C2' rotation and the
inversion of D3d and the global spin flip. The exact layer of certify-ed handles one-dimensional characters with
values 1, i, -1 and -i only, so the elements of order 3 cannot be used, and the subgroup of order 4 of D3d times the
spin flip, of order 8, is the largest usable group; it gives 8 blocks of dimensions 6194, 6066, 6031, 6159, 6089,
5961, 5996, 6124. Every generator is checked exactly to commute with the integer matrix. A Lanczos estimate
-0.5498329144473226 places the dyadic shift `g = -23061663926287/41943040000000` at a relative margin of 1e-10 below
it. Each block is counted at `g` in two ways. The floating-point layer (`count_below_by_block_float`) forms the block
congruence `V^T A V - g V^T V` of the integer matrix `A` with the orbit sums `V`, which is exact in binary64 (checked
entry by entry), and counts its negative eigenvalues with `inertia.inertia_dense`, the dense LDL^T count of certify-ed
with a bound `beta` on its backward error under the binary64 model; a count of 0 gives no eigenvalue below `g - beta /
d_min`, with `d_min` the smallest squared norm of the orbit sums. The exact re-check (`count_below_by_block`) uses the
exact sector matrices of `exact_sectors` and `checker.inertia_exact`, block by block. For the upper side the Lanczos
vector is rounded to integers and its Rayleigh quotient is evaluated in integer arithmetic, an upper bound on
`lambda_min` by the variational principle.

How the instrument was checked. `test_count_by_block_agrees_with_exact_count_below` compares both block counts with
`exact_sectors.exact_count_below` and with dense floating-point eigenvalues on the 153 states with two down spins, at
a point below the lowest eigenvalue and at two points between eigenvalues; the counts agree and the block dimensions
are the same. A negative control gives the swap of one outer pair as a generator; it is not a symmetry of the
nine-coupling model, and both counts stop with an error. `test_rayleigh_upper_is_an_upper_bound` checks that the
integer Rayleigh quotient is at least the lowest eigenvalue for the eigenvector and for random vectors, and the code
requires the lower bound to lie below the Rayleigh quotient.

What happened. The floating-point layer finds 0 eigenvalues below `g` in each of the 8 blocks, with largest slack
`beta / d_min` of 6.607763286486234e-11 in units of `H_0`, so `lambda_min(H_0) >= -0.5498329145683885` under the
binary64 model; the integer Rayleigh quotient gives `lambda_min(H_0) <= -0.5498329144473251`. The enclosure has width
1.211e-10. The run took 3018 s with a peak of 2637 MiB under the cap (record `n11_crown_nine_float.json`). The exact
re-check (`checker.inertia_exact`, the arithmetic of the reduced-model certificate) was run on the same blocks and the
same shift and was stopped after 3 of the 8 blocks. The counted blocks are dimension 6194 with count 0 and slack
0.000313 in units of `K H_0` (5568 s), dimension 6066 with count 0 and slack 0.000259 in units of `K H_0` (7424 s),
dimension 6031 with count 0 and slack 0.000188 in units of `K H_0` (12240 s). On the shared machine, where the process
received about a quarter of one core, the blocks took 1.5 to 3.4 hours each and the remaining five would have needed
10 to 15 hours more, so the run was stopped; the counted blocks are in
`results/apps_next/n11_crown_nine_exact_partial.json`. The process used 9680 s of CPU time in all, less than one hour
of one core per block on average. The script now also counts selected blocks (`--blocks`), writing its record after
every block, and joins block records into the exact lower bound (`--combine`, which requires the blocks 0 to 7 once
each, one shift and count 0 everywhere), so that the re-check can be split over processes or machines with free cores.
The test `test_block_runs_combine_to_the_full_count` checks on 153 states that single-block counts joined this way
give the count and bound of the full count, and that a missing block, a repeated block, a changed shift and a nonzero
count are refused.

The complete exact re-check (22 November 2026). The run was pre-registered
(`results/apps_next/n11_crown_nine_exact_8_blocks_preregistration.md`: the fixed shift `g`, a clock of 6 hours, no
change of the shift within the run). The 8 blocks were counted in 8 processes, one per block, on a Linux host with free
cores (code at main 71f155df, source hash 47dc665a8a7ea32e; Python 3.12.3, numpy 2.5.4, scipy 1.18.1, python-flint
0.9.0; 16 cores, BLAS threads 1, a cap of 16 GiB per process). Every count is 0. The counts took 1038 to 1158 s per
block (8800 s in all; the longest process 1188 s), with peaks of 2981 to 3198 MiB. `--combine` checked that the blocks
are 0 to 7 once each, with dimensions summing to 48620, and gave the exact lower bound `g - max(delta)/K`, that is
lambda_min(H_0) >= -0.5498329145062226 (rounded down -0.549832915), 3.91e-12 below the shift and 6.217e-11 above the
lower end of the floating-point layer; with the integer Rayleigh quotient the enclosure has width 5.890e-11. The three
blocks counted in the first run agree in count and in the six logged decimals of their slack. The record is
`results/apps_next/n11_crown_nine_exact_8_blocks.json`.

| quantity | value |
|---|---|
| couplings `c1` to `c19` (exact) | c1 = 374167/10000000, c2 = 11513/10000000, c3 = 837299/10000000, c4 = 831/20000000, c5 = -5469/10000000, c6 = -2109/10000000, c7 = -5393/10000000, c11 = 8821/5000000, c12 = 401217/5000000, c13 = 2109/20000000, c15 = 5393/10000000, c17 = 1346367/10000000, c18 = -2973/1000000, c19 = -831/10000000; the others 0 |
| `K`, S^z = 0 dimension, group order | 80000000, 48620, 8 |
| blocks | 6194, 6066, 6031, 6159, 6089, 5961, 5996, 6124 |
| Lanczos estimate | -0.5498329144473226 |
| shift `g` | `-23061663926287/41943040000000` |
| floating-point layer: largest slack | 6.607763286486234e-11 |
| floating-point layer: lower bound | -0.5498329145683885 (rounded down -0.549832915) |
| integer Rayleigh quotient: upper bound | -0.5498329144473251 (rounded up -0.549832914) |
| exact re-check | 8 of 8 blocks counted, count 0 in each (22 November 2026; the first run was stopped after 3) |
| exact lower bound (no floating-point assumption; Lemmas 11, 12 and Proposition 13 marked UNVERIFIED) | -0.5498329145062226 (rounded down -0.549832915; the exact rational is in the record) |
| published `e_LB` | -0.549832(8) |
| floating-point value of the review | -0.54983289 (at `c4 = 0.0000415`) |
| lower bound minus the reduced-model certificate -0.5501582988212038 | 0.000325384 |

What it shows and does not show. At the printed couplings the nine-coupling crown has no eigenvalue below
-0.5498329145062226, by the exact re-check of all 8 blocks, which needs no floating-point assumption (the floating-point
layer gives -0.5498329145683885 under the binary64 model); by the argument of the paper the ground-state energy per site
of the spin-1/2 pyrochlore Heisenberg antiferromagnet is then at least -0.549832915, which is 0.0003254 above the
reduced-model certificate. The printed headline lies above the enclosure: read with its digit, -0.5498328 exceeds the
lower bound by 1.145e-07, and the six decimals -0.549832 of the abstract (`e_GS >= -0.549832`) exceed it by 9.145e-07.
The printed values are the first digits of the eigenvalue, cut toward zero rather than rounded down, which for a lower
bound overstates it by less than 1e-6; the same holds for the reduced-model value -0.550158(2) against the certified
-0.5501582988212038. A safe six-decimal statement at the printed couplings is `e_GS >= -0.549833`. The value of the
review, -0.54983289, belongs to a Hamiltonian with `c4` cut to 0.0000415 and lies outside this enclosure, as it should.
The floating-point layer rests on the binary64 model of certify-ed and on Sylvester's law and Weyl's inequality; the
exact re-check rests on Lemmas 11 and 12 and Proposition 13 of the exact layer, which are marked UNVERIFIED, as for the
reduced-model certificate (finding RA-20 of `report/07_applications.md`). The number is not recomputed in Lean (section
7).

## 4 N-12: which paper states a criterion for the honeycomb lattice, and which cluster

Why. The manuscript (`manuscript_v4/sections/s6_applications.tex`) says that the honeycomb case is the companion paper
of Lemm, Sandvik and Wang, and the archive records the honeycomb as not attempted, with a space of dimension 2^26
(`project_docs/DECISIONS_LOG.md`, D-36). Before a cost is estimated, the criterion and its cluster have to be
identified in the papers themselves.

What was done. The arXiv sources of Pomata and Wei (arXiv:1911.01410, file `hexgap.tex`) and of Lemm, Sandvik and Wang
(arXiv:1910.11810, file `AKLT_conjecture.tex`) were read.

What happened. Both papers state a finite-size criterion for the honeycomb lattice. Pomata and Wei treat it first,
with Method III: for a plaquette p of one third A of the plaquettes, the term `Gamma_p` is p together with its three
neighbouring plaquettes of the second third B (18 sites); two terms whose plaquettes are neighbours in the triangular
lattice formed by A overlap in one B hexagon C, each term overlaps with 6 others, and the criterion is `eta < 1/6`; L
and R are two hexagons joined by one edge. Their Table I gives a rank of 2^12 for the projector of one term and 2^18
for the union, `dim H_L = dim H_R = 2^10`, `dim H_C = 2^6` and `D_tot = 2^26`, and they find `eta = 0.1445124916`.
Lemm, Sandvik and Wang use a different criterion: if the gap `gamma_F(a)` of a weighted AKLT Hamiltonian on a 36-site
patch F of spins 3/2 with open boundary conditions exceeds `(a^2 - 2a + 3)/(10 + 4a)`, the lattice model is gapped;
with `a = 1.4` they find `gamma_F(1.4) > 0.145` by DMRG against a threshold below 0.1385, and a gap of at least
0.00646. The patch has `4^36` = 4722366482869645213696 states.

What it shows and does not show. The sentence of the manuscript names only one of the two papers: Pomata and Wei state
and use a criterion for the honeycomb lattice themselves (it is their headline result), and the companion paper
reaches the same conclusion by another criterion. The cluster that the eta machinery would need is the 30-site union
of two terms of Method III. The criterion of Lemm, Sandvik and Wang rests on a gap in a space of dimension `4^36`
computed by DMRG and is out of reach of exact diagonalization of any kind; it is not considered further. Manuscript
prose is Sarang Vehale's (D-16); the sentence is reported here and not changed.

## 5 N-12: the size of the Pomata-Wei criterion with the eta machinery

Why. certify-ed decides each Pomata-Wei criterion by the exact inertia of the bordered matrix `M = [[-p G_A, q C], [q
C^T, -q G_B]]` in S^z blocks and symmetry characters (`certify_ed/apps/aklt/eta.py`), with dense blocks of up to 4034
rows so far. Whether the honeycomb fits under a 5 GiB cap depends on the size of its blocks, which can be counted
without forming any matrix.

What was done. The builder `honeycomb_method3` constructs the union of two terms from the plaquette picture of the
paper (30 sites of spin 3/2 and the 36 edges of the two terms; the one lattice edge between L and R belongs to neither
term and is left out). `estimate.plan_eta` repeats the bookkeeping of `certify_eta`: the reduced boundary dimensions,
the labels of the S^z blocks of M, the pairing of blocks by the spin flip (a pair is factored once, a block mapped to
itself splits into its two flip characters) and the site permutations that preserve L, C and R. Memory uses about 100
bytes per entry for the exact re-check of a dense block, a count of the arrays that `inertia_exact` holds at its peak,
and time uses seconds per `n^3` calibrated on the E10 log of the 3.2.1 run
(`code/certify-ed/results/run_full_20261105/E10.log`).

How the instrument was checked. For the four lattices whose blocks the E10 log prints, `plan_eta` gives the logged
largest blocks: star 70, cross 1584 and decorated honeycomb 673 exactly, and for the decorated square, where the
mirror splits blocks unevenly, 3997 against the logged 4034 and 3959 (their sum, 7993, is the unsplit block of the
plan). The honeycomb dimensions of the plan equal Table I of the paper in all six entries (yes). The calibration gives
1.558e-08 s per `n^3` for the cross and 1.009e-08 s for the decorated square; the larger is used.

What happened.

| quantity | value |
|---|---|
| reduced dimensions L, C, R, Gamma_i, Gamma_j, union | 1024, 64, 1024, 4096, 4096, 262144 |
| dim V, dim A, dim B | 67108864, 4194304, 4194304 |
| rows of the bordered matrix | 8388608 |
| site permutations preserving L, C, R | 1 (the identity) |
| largest block factored (keys 44 and 46, paired by the flip) | 1293292 rows |
| one binary64 copy of that block | 12462 GiB |
| exact re-check of that block, about 100 bytes per entry | 155773 GiB |
| sum of `n^3` over all factorizations | 4.155e+18 |
| time of all counts at the calibrated rate | 6.474e+10 s, about 2051 years of one core |
| object arrays Z and Y built before the blocks | 536870912 entries, 4.0 GiB |
| largest block after a reduction by total spin (not implemented) | 355300 rows (spin 2) |

What it shows and does not show. The criterion does not fit this machine: one binary64 copy of the largest block needs
12462 GiB, the exact re-check about 155773 GiB, and the counts of all blocks about 2051 years of one core. No single
machine meets these numbers, so the requirement for a larger machine is not a matter of size alone, and a different
method is needed. The union has no site symmetry that preserves L, C and R, so the present machinery has no further
reduction. Two directions would change the picture, and neither exists in the code: a reduction by total spin, under
which the largest bordered block would have 355300 rows, still beyond dense exact inertia, and a matrix-free method
that bounds the second largest eigenvalue of `E' + F'` on the space of dimension 2^26 with certified residuals, as
Pomata and Wei did in floating point. Before the eta problem, `certify_eta` also builds the Gram matrix of L from its
VBS map, with 16777216 physical states. The estimate is a count and a calibrated rate, not a run.

## 6 N-12: the constants of the final gap bound, and the two numbers of the paper

Why. The archive notes that the final equation of the honeycomb proof of Pomata and Wei gives 0.002660333395 while the
conclusion states 0.002550333395 (`report/07_applications.md`). Which number follows from the printed constants
decides which one is a misprint.

What was done. The final bound of Method III is `gamma^[2p] gamma^[4p] (1 - 6 eta)`, with `gamma^[2p] = 0.02571076873`
(the minimum of a computed two-plaquette constant and `gamma^[p]/6`), `gamma^[4p] = 0.7784203312` and the plaquette
constant `gamma_p = 0.3130520508`, all printed in `hexgap.tex`. The product of the printed values was formed in exact
rational arithmetic (`experiments/n12_honeycomb_estimate.py`).

What happened. The product is 0.0026603333944698595 (exact value in the record). It agrees with the final equation,
0.002660333395, to 5.3e-13, the rounding of the printed factors, and lies above the conclusion's 0.002550333395 by
0.00011.

What it shows and does not show. The conclusion differs from the product of the printed constants in its second
significant digit, so the conclusion is the misprint; being smaller, it remains a valid but weaker lower bound if the
final equation holds. Only the arithmetic of the last step is checked; the constants are floating-point results of the
paper. Each would need its own certificate: `gamma_p` from the six-site plaquette (4096 states), `gamma^[2p]` and
`gamma^[4p]` from compressed graphs built with a proposition of the paper that certify-ed does not implement. The
paper also notes that one diagonalization of Method III used a tolerance of `10^-5`, with an error of at most `10^-4`.

## 7 A Lean recomputation of the new crown value

`ceapps` selects its targets in `CertifyApps/Main.lean`; a new target needs a new subcommand there, that is, an edit
of an existing file, and a separate executable would need an edit of `lakefile.toml`. Both are outside this part, so
no Lean file was written and the new value is not recomputed in Lean. The engine of `ceapps` counted the 128 sectors
of the reduced crown (largest 3890) in 1360.1 s and the decorated-square blocks of 4034 rows in 2468.4 s (`APPS.md`,
sections 5 and 7), so the 8 blocks of about 6000 rows here are within its reach once a subcommand exists.

## 8 Files

| path | contents |
|---|---|
| `APPS_NEXT.md` | this document |
| `results/apps_next/n11_crown_nine_float.json` | the record of `experiments/n11_crown_nine.py appended --float-layer` |
| `results/apps_next/n11_crown_nine_exact_partial.json` | the blocks of the exact re-check counted before the run was stopped |
| `results/apps_next/n11_crown_nine_exact_8_blocks.json` | the exact re-check of all 8 blocks, one process per block, and the joined lower bound |
| `results/apps_next/n11_crown_nine_exact_8_blocks_preregistration.md` | the pre-registration of that run (question, method, stop rules, outcomes) |
| `results/apps_next/float_readings.json` | floating-point lowest eigenvalues for the other readings of the printed digits |
| `results/apps_next/n12_honeycomb_estimate.json` | the record of `experiments/n12_honeycomb_estimate.py` |
| `results/apps_next/sources_extract.json` | the values read from the arXiv sources, with their SHA-256 |
