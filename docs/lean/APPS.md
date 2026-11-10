# APPS.md: the application numbers recomputed in compiled Lean

This part of the re-verification recomputes the certified numbers of the three application experiments of
certify-ed, E7 (exact benchmarks), E9 (the pyrochlore crown) and E10 (AKLT models), with a compiled Lean program
that does not call the Python code. The program is `ceapps` (root `CertifyApps/Main.lean`) and its library is
`CertifyApps`; it uses core Lean only, together with the exact binary64 decoding and the JSON reader of
`CertifyCore` (`CORE.md`). Every result here is at trust level N in the scale of `README.md`: a compiled Lean
program recomputed the number from the model specification and compared it with the run record, and the
comparison itself is made in exact rational arithmetic inside the program. The record values are those of the
3.2.1 evidence run, `code/certify-ed/results/run_full_20261105`; for every compared field the 3.1.0 evidence run `code/certify-ed/results/run_full_20261022` holds the same
values (JSON equality with the timing fields removed, checked by `apps_export/run_all.py`: true for E7, E9 and E10).
Paths of record files are relative to `sources/archive_v5/`; every other path is relative to `reverification_lean/`.

The outcome in one sentence: 13 of 13 targets agree with the records, 0 disagree and 0 were skipped (`results/apps/summary.json`).

## 1 How to run

The commands assume the environment of `logs/lean_env.json` (the Lean binaries on `PATH`) and, for the
comparison with the Python builder only, the Python environment `certifyed-311` with the certify-ed source on
`PYTHONPATH`. They write below `results/apps/` and `work/apps/` (git-ignored).

```
cd reverification_lean
lake build CertifyApps ceapps
.lake\build\bin\ceapps.exe selftest > results\apps\selftest.txt
set A=..\sources\archive_v5
set PYTHONPATH=%A%\code\certify-ed
set PYTHONDONTWRITEBYTECODE=1
python apps_export\compare_models.py --exe .lake\build\bin\ceapps.exe --code %A%\code\certify-ed --work work\apps\export --out results\apps\compare_models.json
python apps_export\compare_aklt.py --exe .lake\build\bin\ceapps.exe --code %A%\code\certify-ed --work work\apps\export --out results\apps\compare_aklt.json
python apps_export\run_all.py --exe .lake\build\bin\ceapps.exe --archive %A% --out results\apps --work work\apps\final
python apps_export\make_apps_md.py --lean . --archive %A%
```

One target is run by itself with, for example,
`ceapps e7 chain_J2=0 %A%\code\certify-ed\results\run_full_20261105\e7_exact_benchmarks_20261105_092337\data.json out.json 1`, `ceapps crown-half <E9 record> out.json 1`,
`ceapps crown-one <E9 record> out.json` or `ceapps aklt cross <E10 record> out.json 1` (the last argument is the
number of threads for the residual; every run here used 1). The exit code is 0 when the target agrees with the
record, 1 when it does not, 2 when an input could not be read. Setting `CEAPPS_DEBUG=1` prints the time of each
factorization step.

## 2 How eigenvalues are counted, and why this way

Every certified number of the three experiments rests on counting eigenvalues of a symmetric matrix below a
point: the number of negative eigenvalues of `M = b A - a den N` (the inertia of `M`, its numbers of negative,
zero and positive eigenvalues) for a sector matrix `A` and a rational point `a / b`, or the number of positive
eigenvalues of the bordered matrix of the AKLT pencil. certify-ed counts them with
`certify_ed.checker.inertia_exact`: a floating-point LDL^T factorization proposes factors, the factors are
moved to dyadic grids, the residual `L D L^T - M` is computed exactly with FLINT integers, and Sylvester's law
of inertia (a congruence keeps the inertia) and Weyl's inequality (a perturbation of norm `beta` moves every
eigenvalue by at most `beta`) turn the signs of the pivots into counts that are exact up to the window
`[-beta, beta]` (`code/certify-ed/certify_ed/checker.py`).

Why not an exact elimination. The core track measured the fraction-free (Bareiss) elimination of
`CertifyCore/Inertia.lean` at 247 s for one dense block of dimension 256 (`CORE.md`, section 4). The blocks here
have dimensions up to 4034, and the cost of an exact elimination grows with the cube of the dimension and with
the bit length of the entries, which grows linearly along the elimination; the dimension alone would multiply
the cost by some thousands, so exact elimination was not a feasible route. The route of certify-ed needs exact
arithmetic only for the residual, whose entries are products of numbers of fixed size.

What `CertifyApps/Engine.lean` does. It factors `D M D` (`D = diag(w)^(-1/2)` for positive integer weights `w`,
the squared norms of the basis vectors of a sector; `w = 1` otherwise) in binary64 floating point with
symmetric diagonal pivoting, forms `G = D^-1 L |D_d|^(1/2)`, rounds `G` to an integer matrix `Gt` with entries
below `2^62` and keeps `S = sign(D_d)`. `Gt` is lower triangular with nonzero diagonal (checked), so `Gt S Gt^T`
has the inertia of `S`. The residual `E = Gt S Gt^T 2^(sigma - 2F) - P^T M P` is computed in exact integer
arithmetic: every entry of `Gt` is split into three 21-bit limbs, the limb products are accumulated in 64-bit
words whose true values provably stay below `2^63`, and each residual entry is formed in a fixed-width signed
number of eight 21-bit limbs. The bound `beta` is the Frobenius norm of `W^(-1/2) E W^(-1/2)`, bounded from
above by summing, for every entry, the square of an upper bound made of its leading 24 bits; it bounds the
spectral norm of the residual of the orthonormalised problem. Floating point only proposes `Gt` and `S`: a bad
proposal gives a large `beta` or a failed check, never a wrong count. The two differences from certify-ed are
the form of the factor (one factor `Gt` with 62 bits instead of `L` on a `2^-52` grid and `D` with 52 bits) and
the norm (a weighted Frobenius norm instead of the infinity norm of the unweighted residual); both are valid
bounds of the same quantity, and the recomputed residual bound came out smaller than the recorded one in every case (E7, E9 spin 1/2 and E10).

How the instrument was checked. `ceapps selftest` (`results/apps/selftest.txt`) factors 8 symmetric integer
matrices of dimension 8 to 90 with entries of 10 to 130 bits, definite and indefinite, with and without weights,
and compares the count with the exact Bareiss elimination of `CertifyCore` and the limb residual with the same
residual computed with Lean's arbitrary-precision integers; the counts agree in every case and the bucket bound
is never below the exact sum. Section 3 checks the matrices themselves.

Two further theorems enter: Theorem 1 of certify-ed (for a nonzero vector `v`, some eigenvalue lies within
`|| H v - mu v || / || v ||` of `mu`), used for the E7 and spin-1 enclosures, and the bordered-pencil lemma
for E10. They are used as stated (`docs/math/exact_layer.md` and `docs/applications/aklt_pomata_wei.md` of this
repository describe them); the vectors are computed here by inverse iteration, first in floating point and
then with integer arithmetic and iterative refinement, so that the radii are limited by the vector only far
below the record's radii.

## 3 The models rebuilt in Lean and the check against the Python builder

Why. A recomputation is only as good as the model it is applied to. The models are rebuilt in Lean from the
specification builders of certify-ed (`certify_ed/spec.py`: `heisenberg_chain`, `heisenberg_square`,
`spin1_heisenberg_chain`), from `certify_ed/apps/pyrochlore/model.py` (the 36 bonds and the couplings of the
reduced crown, the six pair swaps and the spin flip) and from `certify_ed/apps/aklt/lattices.py` and `graph.py`
(the six unions of subgraphs, the reduced VBS maps and their Gram matrices). Symmetry sectors are built in Lean
from orbit sums with one-dimensional characters of values 1, i, -1, -i; that every generator commutes with the
integer Hamiltonian and that every sector matrix is Hermitian is checked exactly in every run.

What was done. `apps_export/compare_models.py` and `apps_export/compare_aklt.py` export the Lean matrices
(`ceapps export`, `export-sector`, `export-spin1-ring`, `export-aklt`, `export-aklt-ham`) and compare them, as
exact rationals and entry by entry, with `certify_ed.checker.exact_matrix`, with
`certify_ed.exact_sectors.sector_integer_matrix` (the two sector bases differ by one factor per column, which is
read off and applied), with the region maps and Gram matrices of `apps/aklt/graph.py`, and with the scaled AKLT
Hamiltonian of `apps/aklt/kernel.py`, a sum over edges of projectors onto the largest pair spin that for two
spins 1 is the bilinear-biquadratic form `1/3 + X/2 + X^2/6` in `X = S_u . S_v`.

| model | states | nonzero entries | equal |
|---|---|---|---|
| `chain8`, `S^z = 0` | 70 | 354 | yes |
| `chain8_J2`, `S^z = 0` | 70 | 694 | yes |
| `chain12_J2`, `S^z = 0` | 924 | 12876 | yes |
| `square42_J2`, `S^z = 0` | 70 | 852 | yes |
| `square4`, `S^z = 0` | 12870 | 228162 | yes |
| `square4_J2`, `S^z = 0` | 12870 | 449462 | yes |
| `crown`, `S^z = 0` | 48620 | 975260 | yes |
| spin-1 ring of 6 sites, all states | 729 | 4384 | yes |

| sector matrix | character exponents | dimension | group order | equal |
|---|---|---|---|---|
| `chain8` | 0 0 2 | 1 | 32 | yes |
| `chain8_J2` | 0 2 2 | 2 | 32 | yes |
| `chain12_J2` | 0 0 0 | 35 | 48 | yes |
| `chain12_J2` | 0 2 2 | 21 | 48 | yes |
| `square42_J2` | 1 0 2 | 5 | 16 | yes |
| `square42_J2` | 2 2 0 | 3 | 16 | yes |
| `square4_J2` | 1 2 0 | 392 | 32 | yes |
| `square4` | 0 0 0 | 441 | 32 | yes |
| `crown` | 0 2 0 2 2 0 0 | 179 | 128 | yes |
| `crown` | 2 2 2 2 2 2 2 | 10 | 128 | yes |

| AKLT lattice | dimensions L, C, R, Gamma_i, Gamma_j | unit cosines at least | Gram matrices, counts and rank equal |
|---|---|---|---|
| `star` | 8, 8, 8, 16, 16 | 32 | yes |
| `square_octagon_par` | 16, 16, 16, 64, 64 | 256 | yes |
| `square_octagon_perp` | 16, 16, 16, 64, 64 | 256 | yes |
| `cross` | 64, 16, 64, 64, 64 | 256 | yes |
| `decorated_honeycomb` | 24, 6, 72, 36, 36 | 216 | yes |
| `decorated_square` | 144, 12, 192, 144, 144 | 1728 | yes |
| `star`, Hamiltonian of L u C | 4096 states, 54272 entries | | yes; H U = 0 on the 16 VBS columns: yes |
| `square_octagon_par`, Hamiltonian of L u C | 65536 states, 1097728 entries | | yes; H U = 0 on the 64 VBS columns: yes |

How the instrument was checked, and what it shows. The comparison is a set equality of exact entries, so a
single changed coupling, sign or state would make it fail. Every case is equal (`results/apps/compare_models.json`,
`results/apps/compare_aklt.json`). The full E7 chains of 20 sites were not exported (184756 states); they are
built by the same Lean code as the chains of 8 and 12 sites, and the E7 runs check their sector dimensions
against the record (section 4).

## 4 E7: the four benchmark ground energies

Why. E7 is the case in which the exact layer of certify-ed reaches realistic sizes, blocks of dimension 9252 and
12870 after symmetry reduction (`docs/applications/exact_benchmarks_e7.md`). Each record claims that the
recorded `E0` is within the recorded exact radius of the lowest eigenvalue of the block, and that exactly one
eigenvalue of the block lies below the recorded count point (`experiments/e7_exact_benchmarks.py`,
`certify_ed.exact_sectors.exact_recheck_sector_lowest`). An error in the symmetry reduction, in a count or in a
radius would show here as a different sector dimension, a different count, or a radius that does not cover the
eigenvalue.

What was done. For each system the Lean program builds the model on the `S^z = 0` states, the group (translation,
reflection and spin flip of the chain, momentum fixed to 0; the two translations and the spin flip of the
square), checks every generator against the Hamiltonian, builds the character sectors and compares their
dimensions with the record, and counts the eigenvalues of every sector below the recorded count point (its
binary64 value). In the sector that holds the counted eigenvalue it computes a vector by inverse iteration and
from it two exact radii: around its own Rayleigh quotient, and around the recorded `E0`. The second radius
tests the record directly: if it is at most the recorded radius, the recorded enclosure contains an eigenvalue
whatever vector certify-ed used. Together with the count, that eigenvalue is the lowest of the block.

| system | record `E0` | record exact radius | radius around the record `E0` (Lean) | Lean Rayleigh quotient | Lean radius | count (record / Lean) | `delta` (record / Lean) | seconds | peak MiB | agrees |
|---|---|---|---|---|---|---|---|---|---|---|
| Heisenberg chain, N = 20 | -8.904386529876417 | 3.719300427741141e-14 | 2.5249289783497136e-14 | -8.9043865298764423 | 2.1599193648251390e-23 | 1 / 1 | 2.4676572307452423e-10 / 4.9432867231835551e-13 | 578.5 | 126 | yes |
| J1-J2 chain, N = 20, J2 = 0.5 | -7.499999999999966 | 3.392663354321801e-14 | 3.3750779948604759e-14 | -7.5000000000000000 | 6.2805712975047203e-23 | 1 / 1 | 4.202423307156557e-10 / 3.9205600890254744e-13 | 673.0 | 129 | yes |
| Heisenberg square 4 x 4 | -11.228483208428859 | 1.768521922198284e-14 | 8.6548064142404770e-16 | -11.228483208428860 | 6.6295588784963786e-31 | 1 / 1 | 6.823729400534755e-11 / 2.2583362660115805e-13 | 130.3 | 35 | yes |
| J1-J2 square 4 x 4, J2 = 0.5 | -8.45792335139484 | 1.099862660492978e-14 | 7.1742627698341029e-15 | -8.4579233513948467 | 3.8644449557691711e-23 | 1 / 1 | 8.376229085041161e-11 / 1.7130522014567057e-13 | 157.7 | 44 | yes |

The recorded enclosure lies below the count point minus the Lean `delta` in every case (checked in rationals by
`apps_export/make_apps_md.py`: true), so the eigenvalue in it is the lowest of the block.
The sector dimensions, the dimension of the block and of the `S^z = 0` space equal the record in every case;
the group orders are 80 and 32 (`results/apps/e7_*.json`). The radius around the Lean Rayleigh quotient is the
radius of a vector accurate far beyond binary64, so the Lean enclosure is much narrower than the record's and
lies inside it. For the J1-J2 chain at the Majumdar-Ghosh point, where the ground energy is -7.5, the Lean
enclosure is -7.5000000000000000 with radius 6.2805712975047203e-23.

What it shows and does not show. The four E7 statements of the record are recomputed in Lean from the
specification: the same sectors, the same count at the same point, and an exact radius around the recorded
`E0` that is below the recorded radius. As in the record, the statement for the chains is about the momentum-0
block only. The counts rest on Sylvester's law and Weyl's inequality and the enclosures on Theorem 1, used as
stated.

## 5 E9: the lower bound for the spin-1/2 crown

Why. The certified bound `lambda_min(H_crown) >= -0.5501582988212038` turns the reduced-model value of
Kránitz and Penc into a statement with no floating-point assumption (`docs/applications/crown_kranitz_penc.md`).
The record claims that at the recorded shift no eigenvalue lies below the shift minus `delta` in any of the 128
sectors of the `S^z = 0` space (`certify_ed/apps/pyrochlore/spin_half.py`). A count of 1 in any sector, or a
residual bound larger than the recorded one, would mean that the recorded bound is not reproduced.

What was done. The crown is rebuilt with `c1 = 381414/10^7`, `c3 = c12 = 817957/10^7` and `c17` from the
embedding constraint, on the 48620 states of `S^z = 0`; the seven generators are checked against the
Hamiltonian; the 128 sectors are built and every one is counted at the recorded shift; the bound is the shift
minus the largest Weyl slack `delta` of the 128 counts.

| quantity | record | Lean |
|---|---|---|
| `S^z = 0` dimension, sectors, largest sector | 48620, 128, 3890 | 48620, 128, 3890 |
| shift | `-9230124599117/16777216000000` | the same |
| eigenvalues below the shift minus `delta`, all sectors | 0 | 0 |
| `delta` | 8.582401785032515e-10 | 6.5807061378185269e-14 |
| lower bound | -0.5501582988212038 | -0.55015829796302939 |
| seconds, peak MiB | | 1360.1, 276 |

The Lean bound is at least the record's exact bound `lower_bound_exact` (checked in rationals: yes), so the recorded bound is recomputed; agreement: yes. The smaller `delta` comes from the smaller residual bound of section 2; the shift,
and therefore the margin below the Lanczos estimate of the record, is the record's.

What it shows and does not show. The bound of the reduced model is recomputed. As in the record, it is not the
paper's headline bound of the nine-coupling model, the Anderson decomposition and the embedding constraint are
taken from the paper, and the counts rest on Sylvester's law and Weyl's inequality.

## 6 E9: the spin-1 crown at the printed couplings

Why. The record claims that at the printed couplings the crown has an eigenvalue at most
`-12 c1 + c17 (E0 + r)`, with `E0` and `r` the enclosure of the lowest eigenvalue of the spin-1 Heisenberg
ring of 6 sites, and that this number lies below the published bound -1.632985
(`certify_ed/apps/pyrochlore/spin_one.py`, `printed_coupling_check`). It rests on the identity that the
all-singlet class of the crown is `-12 c1 + c17 H_ring`.

What was done. The ring is rebuilt on all 729 states; its lowest eigenvalue is enclosed by a vector from
inverse iteration, counted below a point found by the program (one eigenvalue below it, the enclosure under
its window), and the radius around the recorded `E0` is computed. The two identities behind the
all-singlet class, `(S_a + S_b) . S_h u = 0` and `S_a . S_b u = -2 u` for the pair singlet `u`, are checked
exactly on three spins 1. The bound is evaluated in rationals, once from the record's `E0` and radius and
once from the Lean enclosure.

| quantity | record | Lean |
|---|---|---|
| `E0` of the ring | -8.617423181814216 | -8.6174231818142135 |
| radius | 5.092986520018812e-15 | 4.9852152691217544e-28 (around the Lean value); 2.2831585090295931e-15 (around the record `E0`) |
| count point, count | | -8.1174231818140470, 1 |
| eigenvalue upper bound | -1.6380246425131773 | -1.6380246425131775; from the record's numbers -1.6380246425131772 |
| below the published -1.632985 | true | true |
| seconds, peak MiB | | 6.6, 23 |

The recorded enclosure of the ring lies below the count point minus `delta` (checked in rationals: true), so it holds the lowest eigenvalue of the ring.
The record's float `eigenvalue_upper` is the binary64 number nearest the exact value from its own `E0` and
radius (yes), the record's enclosure is recomputed (yes),
and the pair-singlet identities hold (yes); agreement: yes.

The other pair-spin classes are built by `heisenberg_mixed` in floating point, with square roots in the
matrix elements, and enter no certified number; they were not rebuilt, since no exact comparison with them
is possible.

What it shows and does not show. The certified part of the spin-1 statement is recomputed. The two sectors the
paper names and the floating-point values of the 92 classes (`spin1_landscape`) are not certified by the record
and are not targets.

## 7 E10: the AKLT eta bounds

Why. For each lattice the spectral-gap criterion of Pomata and Wei needs `eta < 1/z`, or
`4 eta_perp + 2 eta_par < 1` for the square-octagon lattice, and certify-ed replaced the floating-point `eta` by
an upper bound `sqrt(t')` in exact arithmetic (`docs/applications/aklt_pomata_wei.md`,
`certify_ed/apps/aklt/eta.py`). The record states, at its threshold `t`, the number of positive pivots, the lower
bound on the unit cosines, `t'` and the criterion. A recomputed count above the unit-cosine bound, or a `t'`
that does not decide the criterion, would mean the record is not reproduced.

What was done. For each lattice the Lean program builds the reduced regions, the union coordinates, the Gram
matrices and the rank modulo `2^31 - 1` of the union VBS states, certified lower bounds of the smallest
eigenvalues of the five Gram matrices, the symmetry group (spin flip, and the mirror for the decorated square)
with exact checks of every equivariance, and the bordered matrices of every block and character at the
record's threshold; it then forms `t' = (1 + beta/mu)(t + beta/nu)` exactly (`CertifyApps/Eta.lean`).

| lattice | threshold `t` | positive pivots (record / Lean) | unit cosines at least | record `t'` | Lean `t'` | record `eta` bound | Lean `eta` bound | criterion (record / Lean) | seconds | peak MiB | agrees |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `star` | `616527636979/50000000000000` | 32 / 32 | 32 | 0.012330552740086294 | 0.012330552739588056 | 0.11104302202338649 | 0.11104302202114303 | holds / holds | 0.2 | 13 | yes |
| `square_octagon_perp` | `1263514721581/50000000000000` | 256 / 256 | 256 | 0.025270294439955796 | 0.025270294431682869 | 0.15896633115208958 | 0.15896633112606854 | n/a / n/a | 2.9 | 21 | yes |
| `square_octagon_par` | `563334716459/50000000000000` | 256 / 256 | 256 | 0.011266694339722041 | 0.011266694329240721 | 0.10614468587603454 | 0.10614468582666172 | n/a / n/a | 2.9 | 21 | yes |
| `cross` | `3989544843597/100000000000000` | 256 / 256 | 256 | 0.03989544869676715 | 0.039895448436108296 | 0.19973845072185567 | 0.19973845006935519 | holds / holds | 84.6 | 136 | yes |
| `decorated_honeycomb` | `2341907109863/100000000000000` | 216 / 216 | 216 | 0.023419071107230096 | 0.023419071098786437 | 0.15303290857599913 | 0.15303290854841137 | holds / holds | 6.0 | 27 | yes |
| `decorated_square` | `1213900630589/25000000000000` | 1728 / 1728 | 1728 | 0.04855602603738212 | 0.048556025224217495 | 0.22035431930729685 | 0.22035431746216704 | holds / holds | 2468.4 | 814 | yes |

For the square-octagon lattice `4 eta_perp + 2 eta_par` is at most 0.84815469615759761 in Lean and at most
0.8481546963604274 in the record; the criterion holds in both (agreement: yes).
In every lattice the Lean `t'` is at most the record's, so the recorded `eta` bounds are recomputed, and the unit
cosine bounds and the numbers of positive pivots are the record's (`results/apps/e10_*.json`). The excess of
`t'` over `t` is much smaller in Lean, for the reason given in section 2; the bounds themselves are dominated
by the placement of `t`, which is the record's.

What it shows and does not show. The five E10 criteria of the record are recomputed from the geometry. The
criterion itself, the kernel lemma (ground space equals VBS range) and the bordered-pencil lemma are used as
stated; the bordered-pencil lemma is new mathematics of the project and awaits Sarang Vehale's check under
D-12. The exact kernel check of the record (`kernel` in the E10 record) was not repeated; the Lean check
`H U = 0` of section 3 covers one of its two directions for the star and the square-octagon.

## 8 What was not recomputed

Every target named for this part was run to the end; none was skipped for time or memory.
Floating-point values that the records do not certify are not targets: the two crown sectors named by the
paper, the values of the 92 pair-spin classes at the re-optimised couplings, and the floating-point `eta` of each lattice. The honeycomb
lattice (Method III of Pomata and Wei) was not attempted by certify-ed and is not attempted here.

## 9 Files

| path | contents |
|---|---|
| `CertifyApps.lean`, `CertifyApps/` | the library: `Engine` (counting), `SpinHalf` (spin-1/2 models and sectors), `Models` (the E7 and E9 models and the spin-1 ring), `Vector` (inverse iteration and exact radii), `Report` (records and JSON), `Apps` (E7 and E9), `Vbs` and `Eta` (E10), `AkltHam` (AKLT projector Hamiltonians), `Export`, `ExportAklt`, `Main` (the program) |
| `apps_export/` | the comparison with the Python builder, the driver of all runs, this document's generator |
| `results/apps/summary.json` | per target: record value, Lean value, agreement, seconds, peak memory, skipped reason |
| `results/apps/<target>.json`, `.log` | the full output of each run and its log |
| `results/apps/measurements.json` | wall time, exit code and peak working set of each run |
| `results/apps/compare_models.json`, `compare_aklt.json`, `selftest.txt` | the checks of sections 2 and 3 |

