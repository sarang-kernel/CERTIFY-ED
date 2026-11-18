# E7: the benchmark values with no floating-point assumption

## Why this experiment

The W4 benchmarks of E4 are ground energies that are quoted in the literature and used as references for other
methods. E4 certifies them under the standard floating-point model, and in the 3.0.0 run the S3 of the J1-J2 square at
J2 = 0.5 was NOT_CHECKED because its factorization did not fit under the memory cap. A number meant to be quoted as a
reference should not depend on the hypotheses of the floating-point model, nor on whether a factorization fits a cap.
E7 re-derives the claim "E0 is the lowest eigenvalue of the stated block" for all four systems with the exact sector
re-check, in integer arithmetic (decision D-41). A failure of an exact re-check would have meant that a quoted
benchmark value was not what the floating-point certificate said.

## Design

The script is `experiments/e7_exact_benchmarks.py`. For each system the producer is SciPy's `eigsh` in floating point.
The claim is then re-checked by `certify_ed.exact_sectors`: the sector is built from orbit sums with exact characters,
floating point proposes a factorization, and FLINT checks it exactly, so that the radius of E0 and the count below it
hold with no floating-point assumption (Lemmas 10 to 12 and Proposition 13 of `code/certify-ed/docs/theory.md`). The
chains of N = 20 are reduced to the (S^z = 0, k = 0) block of dimension 9252 out of 184756; the 4x4 squares are used
in the S^z = 0 block of dimension 12870.

## How the instrument is checked

E7 checks the same numbers as W4 by a different method with different hypotheses, so agreement between the two is a
check of both. The exact layer itself is tested in the suite (the duplicated-vector case of decision D-46 included)
and kept small, 713 lines of code in 3.2.1 (`docs/project/software/01_architecture.md`). One value is known in closed form:
the Majumdar-Ghosh chain at J2 = 0.5 has the exact ground energy -7.5 for N = 20 in the units of the spec, a classical
result independent of any computation here. Its certified interval must contain -7.5, and it does in every run: the
distance of the claimed value from -7.5 is just inside the exact radius (checked for this document in exact rational
arithmetic from the record values).

## Results

| System | U(1) dimension / block dimension | Status 3.0.0 / 3.1.0 / 3.2.1 | E0 (3.2.1 record) | exact radius 3.0.0 | exact radius 3.1.0 | exact radius 3.2.1 |
|---|---|---|---|---|---|---|
| Heisenberg chain, N=20, PBC (Sz=0,k=0 block) | 184756 / 9252 | CERTIFIED / CERTIFIED / CERTIFIED | -8.904386529876417 | 3.6345230555176985e-14 | 3.719300427741141e-14 | 3.719300427741141e-14 |
| J1-J2 chain, J2=0.5 (Majumdar-Ghosh), N=20, PBC (Sz=0,k=0) | 184756 / 9252 | CERTIFIED / CERTIFIED / CERTIFIED | -7.499999999999966 | 4.2052889238651513e-14 | 3.392663354321801e-14 | 3.392663354321801e-14 |
| Heisenberg square 4x4, PBC (Sz=0) | 12870 / 12870 | CERTIFIED / CERTIFIED / CERTIFIED | -11.228483208428859 | 1.203101297864281e-14 | 1.768521922198284e-14 | 1.768521922198284e-14 |
| J1-J2 square 4x4, J2=0.5, PBC (Sz=0) | 12870 / 12870 | CERTIFIED / CERTIFIED / CERTIFIED | -8.45792335139484 | 4.346872439596962e-14 | 1.099862660492978e-14 | 1.099862660492978e-14 |

The values are from the E7 records (`E7` of `data.json`). All four systems are CERTIFIED with exact radii between
1.1e-14 and 4.3e-14 in every run, including the J1-J2 square at J2 = 0.5. The certified ground energies are
-8.904386529876417 (Heisenberg chain), -7.499999999999966 (Majumdar-Ghosh chain), -11.228483208428859 (Heisenberg
square) and -8.45792335139484 (J1-J2 square), each within its radius in the 3.1.0 and 3.2.1 records. The radii differ
between 3.0.0 and 3.1.0 because the floating-point proposal differs between the Linux and Windows stacks; an exact
check certifies whatever proposal it receives, so the radius is a property of the proposal and the status is what must
agree (`code/certify-ed/RESULT_evidence_run_3.1.0.md` gives the ratios of the radii, 0.25 to 1.47). The 3.1.0 and
3.2.1 records agree in 336 of 345 leaves, the rest being timings. The exact check of a system took between 201 and 615
s in 3.1.0 and between 238 and 684 s in 3.2.1 (macros `EsevenMinSeconds` and `EsevenMaxSeconds`).

## What the result shows and does not show

E7 shows that four widely quoted benchmark energies are lowest eigenvalues of their blocks to within about 1e-14, with
no dependence on the floating-point model, the BLAS or the memory cap. It certifies the lowest eigenvalue of the
stated block, not of the whole Hamiltonian: that the ground state lies in the stated block is the usual physical
assumption and is not part of the claim. The exact sector method needs characters with values in {±1, ±i}, so it does
not reach every momentum sector of every lattice.
