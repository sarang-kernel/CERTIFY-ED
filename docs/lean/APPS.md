# APPS.md: the application numbers recomputed in compiled Lean

This part of the re-verification recomputes the certified numbers of the three application experiments of
certify-ed, E7 (exact benchmarks), E9 (the pyrochlore crown) and E10 (AKLT models), with a compiled Lean program
that does not call the Python code. The program is `ceapps` (root `CertifyApps/Main.lean`) and its library is
`CertifyApps`; it uses core Lean only, together with the exact binary64 decoding and the JSON reader of
`CertifyCore` (`CORE.md`). Every result here is at trust level N in the scale of `docs/project/README.md`: a compiled Lean
program recomputed the number from the model specification and compared it with the run record, and the
comparison itself is made in exact rational arithmetic inside the program. The record values are those of the
3.2.1 evidence run, `code/certify-ed/results/run_full_20261105`; for every compared field the 3.1.0 evidence run `code/certify-ed/results/run_full_20261022` holds the same
values (JSON equality with the timing fields removed, checked by `apps_export/run_all.py`: true for E7, E9 and E10).
Paths of record files are relative to `archive_v5/`; every other path is relative to `lean/`.

The lemmas that turn the E10 counts into the certified statement are, in addition, written as theorems on
Mathlib in the module `CertifyMathlib.Apps` (section 9), and the kernel equality on which every E10 number rests
is checked exactly for all twelve subgraphs (section 8). Section 9 states, for every certified number, which
Lean theorems compose it and which steps remain informal.

The outcome in one sentence: 14 of 14 targets agree with the records, 0 disagree and 0 were skipped (`results/apps/summary.json`).

## 1 How to run

The commands assume the environment of `logs/lean_env.json` (the Lean binaries on `PATH`) and, for the
comparison with the Python builder only, the Python environment `certifyed-311` with the certify-ed source on
`PYTHONPATH`. They write below `results/apps/` and `work/apps/` (git-ignored).

```
cd lean
lake build CertifyApps ceapps
.lake\build\bin\ceapps.exe selftest > results\apps\selftest.txt
set A=..\..\archive\archive_v5
set PYTHONPATH=%A%\code\certify-ed
set PYTHONDONTWRITEBYTECODE=1
python apps_export\compare_models.py --exe .lake\build\bin\ceapps.exe --code %A%\code\certify-ed --work work\apps\export --out results\apps\compare_models.json
python apps_export\compare_aklt.py --exe .lake\build\bin\ceapps.exe --code %A%\code\certify-ed --work work\apps\export --out results\apps\compare_aklt.json
python apps_export\run_all.py --exe .lake\build\bin\ceapps.exe --archive %A% --out results\apps --work work\apps\final
python apps_export\compare_big.py --exe .lake\build\bin\ceapps.exe --code %A%\code\certify-ed --work work\apps\big --out work\apps\big\measure_square4_J2.json square4_J2
python apps_export\compare_big.py --exe .lake\build\bin\ceapps.exe --code %A%\code\certify-ed --work work\apps\big --out work\apps\big\full_chain20_J2.json chain20_J2
python apps_export\compare_big.py --merge results\apps\compare_chain20.json work\apps\big\measure_chain12_J2.json work\apps\big\measure_square4_J2.json work\apps\big\full_chain20.json work\apps\big\full_chain20_J2.json
lake build CertifyMathlib.Apps
lake env lean apps_export\PrintAxiomsApps.lean > results\apps\mathlib_axioms.txt
python apps_export\nanoda_apps.py --lean-bin <toolchain bin> --work work\apps\nanoda --theorems apps_export\nanoda_apps_theorems.txt --out results\apps\nanoda_apps.jsonl
python apps_export\run_leanchecker_apps.py
python apps_export\make_mathlib_record.py
python apps_export\make_apps_md.py --lean . --archive %A%
```

`compare_big.py` is run once per model (`chain12_J2` and `square4_J2` to measure, then `chain20` and
`chain20_J2`), so that each peak memory belongs to one model. `run_all.py` includes the kernel check of section 8
(`ceapps aklt-kernels <E10 record> out.json`).

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
for E10 (`docs/project/math/exact_layer.md` and `docs/project/applications/aklt_pomata_wei.md` describe them). Both are Lean
theorems on Mathlib: Theorem 1 is `CertifyMathlib.Enclosure.residual_enclosure` (`ENCLOSURE.md`), and the
bordered-pencil lemma is `CertifyMathlib.Apps.bordered_pencil`, with its perturbed form `eta_count` (section 9).
The vectors are computed here by inverse iteration, first in floating point and then with integer arithmetic
and iterative refinement, so that the radii are limited by the vector only far below the record's radii.

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
`results/apps/compare_aklt.json`).

The full E7 chains. The two chains of 20 sites (184756 states of `S^z = 0`) are compared by
`apps_export/compare_big.py`, which holds the entries in sorted integer arrays instead of dictionaries of
fractions and tests equality of the rational entries by exact cross products. The cost was measured first:
the model `square4_J2` (449462 entries) took 19.2 s with a peak of 191 MiB,
which scaled by the number of entries gives about 1649 MiB and 166 s for the chain with
next-nearest bonds, well inside the limits of 6 GiB and 2 h, so both chains were run.

| model | states | nonzero entries (Lean / Python) | seconds (Lean export, Python build) | peak MiB (Lean, Python) | equal |
|---|---|---|---|---|---|
| `chain20`, `S^z = 0` | 184756 | 2066052 / 2066052 | 11.9, 30.4 | 15, 678 | yes |
| `chain20_J2`, `S^z = 0` | 184756 | 4048556 / 4048556 | 19.9, 41.1 | 15, 1285 | yes |

Both full chains are equal entry by entry (`results/apps/compare_chain20.json`): yes.

## 4 E7: the four benchmark ground energies

Why. E7 is the case in which the exact layer of certify-ed reaches realistic sizes, blocks of dimension 9252 and
12870 after symmetry reduction (`docs/project/applications/exact_benchmarks_e7.md`). Each record claims that the
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
Kránitz and Penc into a statement with no floating-point assumption (`docs/project/applications/crown_kranitz_penc.md`).
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
an upper bound `sqrt(t')` in exact arithmetic (`docs/project/applications/aklt_pomata_wei.md`,
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
kernel equality (ground space equals VBS range) is checked exactly for both subgraphs of every lattice in
section 8, and the bordered-pencil lemma and the steps from the count to the criterion are Lean theorems on
Mathlib (section 9). The bordered-pencil lemma is new mathematics of the project; its formalization does not
replace Sarang Vehale's reading under D-12, which is still recorded as pending. The theorem of Pomata and Wei,
that the criterion implies a spectral gap of the infinite lattice, is taken from the paper.

## 8 E10: the kernel equality for every subgraph

Why. Every E10 number assumes that the ground space of the subgraph Hamiltonian `H_X`, a sum of projectors onto
the largest spin of each edge, is the range of the VBS map `U_X`, for `X = Gamma_i = L u C` and `X = Gamma_j =
C u R` (`project_docs/notes/aklt_kernel_lemma.md`). The record confirms this by exact computation for four
subgraphs (`kernel` in the E10 record); the cross and the decorated square, with 16777216 and 4100625 states,
relied on the general argument alone, which is marked for Sarang Vehale's check under D-12. A second ground
state outside the VBS range would make the computed `eta` describe the wrong subspaces.

What was done. `ceapps aklt-kernels` (`CertifyApps/Kernel.lean`) shows both directions for all twelve
subgraphs. The inclusion: every edge term, as the integer matrix of `kernel.py` in the scaled basis, annihilates
every reduced VBS column (checked in integers, edge by edge), and the rank of `U_X` is at least the rank of its
Gram matrix modulo `p = 2^31 - 1`. The bound on the kernel: the common null space of the edge terms modulo `p`
is computed site by site, keeping at each step only the frontier sites (those with an edge to a site not yet
added) explicitly and the interior through a set of coordinates on which the subspace is determined; this
replaces the elimination of the full `S^z` blocks of `kernel.py`, which is out of reach at 16777216 states, and
needs equations with at most a few thousand rows and unknowns per step. Over the rationals the kernel of `H_X`
is the common kernel of the edge terms, whose dimension is at most its dimension modulo `p`; when that bound
equals the rank of `U_X`, the kernel is the VBS range (`CertifyMathlib.Apps.aklt_kernel_certificate`).

| lattice | subgraph | sites | states | VBS rank at least | kernel dimension at most (mod p) | edge terms annihilate U | kernel = VBS range | record (rank / bound) | seconds |
|---|---|---|---|---|---|---|---|---|---|
| `star` | LC | 6 | 4096 | 16 | 16 | yes | yes | 16 / 16, blocks equal: yes | 1.042 |
| `star` | CR | 6 | 4096 | 16 | 16 | yes | yes | not in the record | 0.973 |
| `square_octagon_par` | LC | 8 | 65536 | 64 | 64 | yes | yes | 64 / 64, blocks equal: yes | 2.33 |
| `square_octagon_par` | CR | 8 | 65536 | 64 | 64 | yes | yes | not in the record | 2.482 |
| `square_octagon_perp` | LC | 8 | 65536 | 64 | 64 | yes | yes | not in the record | 2.487 |
| `square_octagon_perp` | CR | 8 | 65536 | 64 | 64 | yes | yes | not in the record | 2.491 |
| `cross` | LC | 12 | 16777216 | 64 | 64 | yes | yes | not in the record | 103.136 |
| `cross` | CR | 12 | 16777216 | 64 | 64 | yes | yes | not in the record | 152.54 |
| `decorated_honeycomb` | LC | 9 | 62208 | 36 | 36 | yes | yes | 36 / 36, blocks equal: yes | 0.674 |
| `decorated_honeycomb` | CR | 9 | 62208 | 36 | 36 | yes | yes | 36 / 36, blocks equal: yes | 0.624 |
| `decorated_square` | LC | 12 | 4100625 | 144 | 144 | yes | yes | not in the record | 33.317 |
| `decorated_square` | CR | 12 | 4100625 | 144 | 144 | yes | yes | not in the record | 30.632 |

All twelve kernels equal the VBS range: yes; the four subgraphs of the record give
the record's rank, bound and per-block dimensions: yes (`results/apps/e10_kernels.json`;
333.9 s and 94 MiB for all twelve).
The rank equals the number of reduced VBS columns, the product over the sites of `z_v - deg(v) + 1`, in every
case (yes), as the lemma states.

What it shows and does not show. The ground spaces used by every E10 bound are now established by exact
computation for all six lattices, so the E10 numbers no longer depend on the general kernel lemma, which remains
a re-derived argument for arbitrary graphs. The step from the computed facts to the equality is a Lean theorem;
that the program computes the null space modulo `p` correctly, and that the integer edge terms are the scaled
projectors (they equal the matrices of `kernel.py` entry by entry for the star and the square-octagon, section
3), are not part of it.

## 9 How each certified number is composed

Why. A recomputation at level N shows that a compiled program finds the same numbers; it does not by itself
say which mathematics turns those numbers into the certified statement. This section names, for every
certified number, the computed facts, the Lean theorems that turn them into the statement, and the steps that
are still informal. `Counting` and `Enclosure` are the modules `CertifyMathlib.Counting` (`COUNTING.md`) and
`CertifyMathlib.Enclosure` (`ENCLOSURE.md`); `Apps` is `CertifyMathlib.Apps`, written for this part.

| number | computed by `ceapps` (level N) | Lean theorems that turn it into the statement | steps that remain informal |
|---|---|---|---|
| E7, the four ground energies | the sector matrices; per sector `Gt`, `S` and the exact residual of the weighted factorization with its Frobenius bound; the counts at the count point; the exact radius around the recorded `E0` | `Counting.sylvester_inertia` (the count of `Gt S Gt^T` is the number of negative entries of `S`), `Apps.cntLt_congr` (the diagonal weights are a congruence), `Counting.weyl_cntLt` with `Counting.abs_qf_le_of_toEuclideanLin` (the residual moves the count by at most `beta`) and the spectral norm bounded by the Frobenius norm (`Counting.norm_toEuclideanLin_le_frob` of `Counting/Compose.lean`, a file of the counting part that is not yet committed), `Counting.cnt_blockDiagonal'` with `Counting.cnt_conj` (counts add over the sectors of an orthonormal symmetry basis), `Enclosure.residual_enclosure` (Theorem 1) | that the program computes these facts; that the orbit sums form an orthogonal basis in which the Hamiltonian is block diagonal (the program checks commutation and Hermiticity); the one-line step from count and enclosure to the lowest eigenvalue |
| E9, spin-1/2 lower bound | the 128 sector matrices; their counts at the shift with the Weyl slack `delta` | as for E7, without Theorem 1 | as for E7; the Anderson decomposition and the reduced model are taken from the paper of Kránitz and Penc |
| E9, spin-1 upper bound | the spin-1 ring, its count and the exact radius; the two pair-singlet identities | `Enclosure.residual_enclosure`, `Counting.sylvester_inertia`, `Counting.weyl_cntLt` | that the all-singlet class of the crown is `-12 c1 + c17 H_ring` (derivation of `spin_one.py`, of which only the two identities are recomputed); that an eigenvalue of an invariant block is one of the crown |
| E10, eta bound of each lattice | Gram matrices and their certified lower bounds `mu`, `nu`; the bordered matrices of every block and character, their counts and residual bound `beta`; the rank modulo `p` of the union VBS states; `t'` | `Apps.e10_eta_bound`, composed of `Apps.eta_count`, `Apps.bordered_inertia`, `Apps.qf_inv_shift`, `Apps.pencil_count_eq` and `Apps.eig_le_of_count`; `Apps.rank_zmod_le_rank_rat` (unit cosines); `Counting.sylvester_inertia` (pivots and the lower bounds `mu`, `nu`) | that the program computes these facts; the reduction to the coordinates of `V = range U_L (x) range U_C (x) range U_R` (docstring of `eta.py`); the block decomposition by symmetry characters (`W = Q N`, `N >= I`, docstring of `eta.py`); that the union VBS states lie in `A n B`; that the eigenvalues of `K` are the squared cosines (the definition used) |
| E10, the criterion | `t' < 1/z^2` in rationals; for the square-octagon lattice `4 s_perp + 2 s_par < 1` with rational upper bounds of the square roots | `Apps.e10_criterion`, `Apps.criterion_of_eta_sq`, `Apps.square_octagon_criterion`, `Apps.sqrt_le_isqrt_div` | none for the arithmetic (`ceapps` checks `t' <= s^2` for both square-root bounds `s` in rationals, field `square_root_bounds_checked`); the theorem of Pomata and Wei that the criterion gives a gap |
| E10, the kernel equality | for all twelve subgraphs: edge terms times VBS columns in integers, rank of the VBS Gram matrix and dimension of the common null space modulo `p` | `Apps.aklt_kernel_certificate`, with `Apps.rank_zmod_le_rank_rat` and `Apps.psd_sum_mulVec_eq_zero` | that the site-by-site elimination computes the null space modulo `p` of the stacked edge terms (`CertifyApps/Kernel.lean`, docstring); that the integer edge terms are `D^-1` times a positive multiple of the projector times `D` |

The module `CertifyMathlib.Apps`. It has three files, `Bordered.lean`, `Kernel.lean` and `Criterion.lean`, and the
root `CertifyMathlib/Apps.lean`; it imports Mathlib v4.35.0 and `CertifyMathlib.Counting.Inertia`. Its 37
theorems use no `sorry`, `admit`, `native_decide` or new axiom (word search: sorry 0, admit 0, native_decide 0, axiom 0),
and `#print axioms` shows only `propext`, `Classical.choice` and `Quot.sound` for every one of them (`results/apps/mathlib_axioms.txt`).
leanchecker replayed the three files and the root (exit code 0, 1087.6 s, `results/apps/leanchecker_apps.json`).
nanoda accepted 15 of the 15 main theorems, each exported with all its dependencies
(rejected 0, not run 0; `results/apps/nanoda_apps.jsonl`). Trust levels as in `docs/project/README.md`: K kernel-checked by
the build, K+ also replayed by leanchecker, K2 also accepted by nanoda (`results/apps/mathlib_apps.json`).

| theorem | statement | covers | declarations checked by nanoda | level |
|---|---|---|---|---|
| `Apps.bordered_inertia` | Haynsworth inertia additivity for the bordered matrix [[X, -q C], [-q C^H, q G]] with G > 0: its negative eigenvalues are those of the Schur complement X - q C G^-1 C^H | E10, the bordered matrix of eta.py | 35445 | K2 |
| `Apps.bordered_pencil` | the bordered matrix with X = p G_A has as many negative eigenvalues as (p/q) G_A - C G_B^-1 C^H, the number of pencil eigenvalues above p/q | E10, the bordered-pencil lemma | 35460 | K2 |
| `Apps.qf_inv_shift` | G^-1 <= (1 + beta/mu) (G + beta I)^-1 in the quadratic form for G >= mu > 0, beta >= 0 | E10, the inequality (G_B + beta)^-1 >= G_B^-1 / (1 + beta/mu) | 20918 | K2 |
| `Apps.eta_count` | a symmetric N' within q beta of the bordered matrix has at least as many negative eigenvalues as there are pencil eigenvalues above t' = (1 + beta/mu)(p/q + beta/nu) | E10, the certified count of eta.py | 35959 | K2 |
| `Apps.linearIndependent_int_of_zmod` | integer vectors independent modulo a prime are independent over Z | E10, ranks modulo p | 7924 | K2 |
| `Apps.rank_zmod_le_rank_rat` | the rank of an integer matrix modulo a prime is at most its rank over Q | E10, ranks modulo p (unit cosines, kernel check) | 17253 | K2 |
| `Apps.psd_sum_mulVec_eq_zero` | a vector killed by a sum of positive semidefinite matrices is killed by every term | E10, kernel of a projector Hamiltonian | 14380 | K2 |
| `Apps.aklt_kernel_certificate` | T_e = D^-1 P_e D with P_e >= 0, T_e U = 0, rank_p U >= r and nullity_p of the stacked T_e <= r give ker (sum T_e) = range U of dimension r over Q | E10, the kernel equality | 24434 | K2 |
| `Apps.pencil_count_eq` | for G_A = F F^H and K = F^-1 C G_B^-1 C^H F^-H, the negative eigenvalues of t G_A - C G_B^-1 C^H are the eigenvalues of K above t | E10, pencil count as count of squared cosines | 35400 | K2 |
| `Apps.eig_le_of_count` | if at most u values exceed t' < 1 and at least u equal 1, every value below 1 is at most t' | E10, unit cosines | 7356 | K2 |
| `Apps.e10_eta_bound` | composed: residual bound, pivot count u and at least u unit cosines give eta^2 <= t' | E10, eta bound of each lattice | 35968 | K2 |
| `Apps.e10_criterion` | composed: with t' < 1/z^2 and z >= 1 every cosine other than 1 is below 1/z | E10, criterion of star, cross and the decorated lattices | 35978 | K2 |
| `Apps.criterion_of_eta_sq` | eta^2 <= t' and t' < 1/z^2 give eta < 1/z | E10, rational criterion test | 7503 | K2 |
| `Apps.sqrt_le_isqrt_div` | sqrt(num/den) <= (isqrt(num den) + 1)/den | E10, square-root bound of __init__.py | 11712 | K2 |
| `Apps.square_octagon_criterion` | eta_perp^2 <= a, eta_par^2 <= b, a <= s_perp^2, b <= s_par^2, 4 s_perp + 2 s_par < 1 give 4 eta_perp + 2 eta_par < 1 | E10, square-octagon criterion | 7507 | K2 |

What it shows and does not show. Every E10 number now rests on Lean theorems from the computed facts to the
criterion on `eta`, and on an exact computation for the kernel equality; the general kernel lemma is no longer
needed for the six lattices. Three kinds of step remain outside Lean: the correctness of the compiled programs
(level N, not K: no theorem says that `ceapps` computes what the hypotheses state), the geometric reductions of
`eta.py` listed in the table, and the theorem of Pomata and Wei itself.

## 10 What was not recomputed

Every target named for this part was run to the end; none was skipped for time or memory.
Floating-point values that the records do not certify are not targets: the two crown sectors named by the
paper, the values of the 92 pair-spin classes at the re-optimised couplings, and the floating-point `eta` of each lattice. The honeycomb
lattice (Method III of Pomata and Wei) was not attempted by certify-ed and is not attempted here. The general
kernel lemma (any graph, any spins) and the theorem of Pomata and Wei are not formalized.

## 11 Files

| path | contents |
|---|---|
| `CertifyApps.lean`, `CertifyApps/` | the library: `Engine` (counting), `SpinHalf` (spin-1/2 models and sectors), `Models` (the E7 and E9 models and the spin-1 ring), `Vector` (inverse iteration and exact radii), `Report` (records and JSON), `Apps` (E7 and E9), `Vbs` and `Eta` (E10), `AkltHam` (AKLT projector Hamiltonians), `Kernel` (the kernel equality), `Export`, `ExportAklt`, `Main` (the program) |
| `CertifyMathlib/Apps.lean`, `CertifyMathlib/Apps/` | the Mathlib module of section 9: `Bordered`, `Kernel`, `Criterion` |
| `apps_export/` | the comparisons with the Python builder (`compare_models.py`, `compare_aklt.py`, `compare_big.py`), the driver of all runs (`run_all.py`), the axiom listing (`PrintAxiomsApps.lean`), the second checkers (`nanoda_apps.py`, `nanoda_apps_theorems.txt`, `run_leanchecker_apps.py`), the module record (`make_mathlib_record.py`) and this document's generator |
| `results/apps/summary.json` | per target: record value, Lean value, agreement, seconds, peak memory, skipped reason; the model checks and the module record |
| `results/apps/<target>.json`, `.log` | the full output of each run and its log, including `e10_kernels` |
| `results/apps/measurements.json` | wall time, exit code and peak working set of each run |
| `results/apps/compare_models.json`, `compare_aklt.json`, `compare_chain20.json`, `selftest.txt` | the checks of sections 2 and 3 |
| `results/apps/mathlib_apps.json`, `mathlib_axioms.txt`, `nanoda_apps.jsonl`, `leanchecker_apps.json` | the record of the Mathlib module |

