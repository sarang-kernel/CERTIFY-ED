# Experiment E7: benchmark ground energies with no floating-point assumption

Experiment E7 re-derives four benchmark ground energies in exact arithmetic, through the exact symmetry-sector path of
statement S6 (Lemmas 10 to 12 and Proposition 13 in `docs/math/theorems.md`). The records are
`code/certify-ed/results/run_full_20261022/e7_exact_benchmarks_20261022_193038/data.json` (the 3.1.0 evidence run) and
`code/certify-ed/results/run_full_20261105/e7_exact_benchmarks_20261105_092337/data.json` (the 3.2.1 evidence run); the
comparison of the two runs finds all 336 compared leaves identical except timings
(`code/certify-ed/RESULT_evidence_run_3.2.1.md`). The script is `experiments/e7_exact_benchmarks.py` of the code
repository.

## Why this experiment

Every floating-point statement rests on the model of [arithmetic_model.md](../math/arithmetic_model.md): IEEE 754
binary64, round-to-nearest, BLAS kernels within the $\gamma_n$ bounds, no flush-to-zero. The exact layer makes no such
assumption, and E7 asks whether it reaches the sizes of the benchmarks that the floating-point layer certifies in the
workflow W4 of experiment E4 (four standard spin-1/2 models of dimension 9 252 to 12 870 after symmetry reduction).
Without symmetry sectors the exact path is limited to `Options.exact_max_dim` = 4 000
(`code/certify-ed/docs/limitations.md`). There was also a specific gap to close: in the 3.1.0 evidence run the
floating-point S3 statement of the J1-J2 square at $J_2 = 0.5$ was NOT_CHECKED, because its sparse factorization
exceeded the memory cap of the experiment, and the exact sector route certifies it (`report/07_applications.md`). If the
exact path had failed at these sizes, the claim that the exact layer covers realistic benchmarks would have had to be
withdrawn.

## What was done

For each system a floating-point producer computes the ground-state vector in the block named below. The exact path then
writes the declared model in the scaled basis (Lemma 10), splits the block into the character sectors of a symmetry
group whose elements (translations, reflection, spin flip) are checked exactly to commute with the integer matrix and to
preserve the weights (Lemma 12), evaluates the residual of the claimed vector in exact arithmetic (Theorem 1 with an
integer square root), and makes one exact count (Lemma 11) at a rational point between the claimed ground energy and the
next level. The count at that point equals 1, the number of claimed values below it, so the claimed value encloses the
lowest eigenvalue of the block, and the next eigenvalue lies above the point minus $\delta$ (Proposition 13).

| system | block | dimension of the $S^z = 0$ space | block dimension | sectors | group order | $E_0$ | exact radius | count point | $\delta$ |
|---|---|---|---|---|---|---|---|---|---|
| Heisenberg chain, N = 20, periodic | $S^z = 0$, momentum 0 | 184756 | 9252 | 4 (2518, 2234, 2136, 2364) | 80 | -8.904386529876417 | 3.719300427741141e-14 | -8.488448517628111 | 2.4676572307452423e-10 |
| J1-J2 chain, N = 20, $J_2 = 0.5$ (Majumdar-Ghosh point), periodic | $S^z = 0$, momentum 0 | 184756 | 9252 | 4 | 80 | -7.499999999999966 | 3.392663354321801e-14 | -7.350988293958421 | 4.202423307156557e-10 |
| Heisenberg square 4 x 4, periodic | $S^z = 0$ | 12870 | 12870 | 32 (381 to 441) | 32 | -11.228483208428859 | 1.768521922198284e-14 | -10.93918404054615 | 6.823729400534755e-11 |
| J1-J2 square 4 x 4, $J_2 = 0.5$, periodic | $S^z = 0$ | 12870 | 12870 | 32 | 32 | -8.45792335139484 | 1.099862660492978e-14 | -8.298456206119047 | 8.376229085041161e-11 |

Every status is CERTIFIED in both runs, and every number in the table is the same in both records. The sector dimensions
are from `code/certify-ed/results/run_full_20261105/E7.log`. The four certifications took 584 s, 684 s, 238 s and 289 s
in the 3.2.1 run and 598 s, 615 s, 201 s and 236 s in the 3.1.0 run (`E7.log` of each run; the records split each time
into `producer_seconds` and `exact_check_seconds`, and almost all of it is the exact check).

## How the instrument was checked

The exact layer has its own tests (two copies of the first excited vector claimed as "the two lowest" must not be
certified, decision D-46; a zero vector must not give radius 0). In E7 itself there are two consistency checks with
known physics. The J1-J2 chain at $J_2 = 0.5$ is the Majumdar-Ghosh point, where the ground energy is known exactly: the
stored value is -7.499999999999966, and the exact enclosure of radius 3.392663354321801e-14 around it contains -7.5. The
floating-point layer certifies the same four energies independently in experiment E4, from its own producer vectors and
with S1, S2 and S3 CERTIFIED for all four in the 3.2.1 run (for example $E_0$ = -8.904386529876408 for the Heisenberg
chain and -7.499999999999985 for the Majumdar-Ghosh chain,
`code/certify-ed/results/run_full_20261105/e4_workflows_W1W2W3W4_20261105_095402/data.json`); the two experiments agree
to within their radii. A second machine stack, used by the review of 21 October 2026, reproduced every statement of E7,
while the radii differed by up to 56 % (finding SS-19 in `report/07_applications.md`): the radii depend on the claimed
floating-point vector, which differs between stacks, and each is a valid bound for its own vector.

## What it shows and does not show

E7 shows that the exact layer certifies ground energies of blocks of dimension 9 252 and 12 870 with 201 s to 684 s of
exact computation on the development machine (8 logical processors, 16 GiB), with no floating-point hypothesis, given
the symmetry reduction. For the two squares the block is the whole $S^z = 0$ space; because these Hamiltonians are
invariant under total spin rotations and 16 spins 1/2 have integer total spin, every multiplet has a member with
$S^z = 0$, so the certified value is the ground energy of the full model. For the two chains the block is the momentum-0
part of $S^z = 0$, and the statement is about that block only: the other momentum sectors were not counted, so E7 does
not by itself certify that the global ground state has momentum 0. The exact layer rests on Lemmas 11 and 12 and
Proposition 13, which are marked UNVERIFIED and have no derivation document; the results are therefore "certified by the
implementation, subject to them" (finding RA-20). In the 3.2.1 run the floating-point S3 statements of all four W4
benchmarks were certified under the 5 GiB cap of experiment E4 (`code/certify-ed/RESULT_evidence_run_3.2.1.md`), so the
specific gap of the 3.1.0 run is now also closed in floating point.

## Earlier runs

The archive holds nine E7 records in `code/certify-ed/results/`. The exact radius depends on the claimed floating-point
vector and therefore on the machine stack, while the status and the integer evidence do not. The table gives the exact
radius and the time of each certification as printed in `E7.log` of the runs on the two platforms recorded in
`manifest.json`:

| system | `run_full_20261018` and `run_full_20261021` (linux) | `run_candidate_20261021` (win32, started 22 October 2026) | `run_full_20261022` and `run_full_20261105` (win32) |
|---|---|---|---|
| Heisenberg chain, N = 20 | 3.6345230555176985e-14 (198 s, 214 s) | 3.8493462260434025e-14 (787 s) | 3.719300427741141e-14 (598 s, 584 s) |
| J1-J2 chain, N = 20, $J_2 = 0.5$ | 4.2052889238651513e-14 (245 s, 264 s) | 3.961693965992654e-14 (938 s) | 3.392663354321801e-14 (615 s, 684 s) |
| Heisenberg square 4 x 4 | 1.203101297864281e-14 (77 s, 92 s) | 1.8719702767382392e-14 (329 s) | 1.768521922198284e-14 (201 s, 238 s) |
| J1-J2 square 4 x 4, $J_2 = 0.5$ | 4.346872439596962e-14 (94 s, 116 s) | 3.273146905871673e-14 (414 s) | 1.099862660492978e-14 (236 s, 289 s) |

Every certification in the nine records is CERTIFIED. The two linux runs give identical radii, and so do the two win32
evidence runs; the largest difference between the linux run of 21 October and the candidate run, for the Heisenberg
square, matches the figure of up to 56 % quoted by finding SS-19. The note `project_docs/notes/exact_checker.md` reports
an earlier run, 20261017_105514, with radii 1.2e-14, 4.3e-14, 3.6e-14 and 3.4e-14 and times of 103 s, 127 s, 212 s and
271 s for the square, the J1-J2 square, the chain and the J1-J2 chain; its record is not in the archive. The claims map
gives the values of the 21 October run rounded up, 3.7e-14, 4.3e-14, 1.3e-14 and 4.4e-14 with 92 s to 264 s each (row C3
of `project_docs/notes/claims_evidence_map.md`). What each step of the exact path computes is described in
[exact_layer.md](../math/exact_layer.md).

## Reproducing

From the root of the code clone: `python run.py --only E7` runs the experiment; the exact sector path is
`certify_ed.exact_sectors.exact_recheck_sector_lowest`. E7 took 1821 s in the 3.2.1 run
(`code/certify-ed/RESULT_evidence_run_3.2.1.md`).
