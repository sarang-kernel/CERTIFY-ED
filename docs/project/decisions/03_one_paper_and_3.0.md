# Decisions D-32 to D-47: one paper, the applications, and certify-ed 3.0.0

This file explains entries D-32 to D-47 of `project_docs/DECISIONS_LOG.md` (version 1.19), taken between 15 and 21
October 2026. In this period the project merged its two preprints into one plan, added two applications to published
proofs, merged the symveig package, released certify-ed 3.0.0 and wrote the appendix proofs against the code. Every
number below is from that file unless another file is named.

## D-32 (15 October 2026): no target venue; a survey of neighbouring practice

SciPost Physics Codebases was dropped as the target, so no venue constrains the scope any longer. Sarang asked for a
wider view of laboratory practice and of the caveats the project had missed. The answer was a sourced survey in five
strands (ED practice in laboratories, neighbouring methods, certified computation, software testing, and
reproducibility and reporting), summarised in a report with 64 unranked directions (D1 to D64, unranked because of
D-07). Three findings changed earlier positions. First, certified sparse inertia counts are established prior art
(Yamamoto 2001; Rump 2006; Terao, Watanabe and Ozaki 2025 at sizes near 10^6), so neither the device used for S3 nor
its reach is the project's own. Second, published proofs of Anderson-type and Knabe-type bounds use uncertified
Lanczos or DMRG numbers, one of them at dimension 48,620 (Kránitz and Penc 2026). Third, a June 2026 preprint called
symveig overlaps with sector-resolved enclosures. The second and third findings and the numbers of Terao and coauthors
had not been checked against the sources when the entry was written. The ms_v2 manuscript stayed a scaffold of
arguments and evidence, not a submission.

## D-33 (17 October 2026): symveig belongs to the project and joins one paper

symveig (arXiv:2606.16217, Vehale and Goel) turned out to be the project's own preprint, and it was folded into one
paper with CERTIFY-ED version 2, following Sarang's view that one solid paper is better than two weaker ones. The
caution about its authorship in D-32 was withdrawn. The thesis of the combined paper is that ED numbers used as
anchors and as lemmas of proofs are made certified. The plan is `PAPER_PLAN.md` v0.1.

## D-34 (17 October 2026): the first application is the pyrochlore bound of Kránitz and Penc

The flagship application is to certify the numerical lemmas of published mathematical proofs of bounds, beginning with
the crown cluster of Kránitz and Penc (arXiv:2601.07800) for the pyrochlore Heisenberg antiferromagnet. An
Anderson-type bound gives a lower bound on the ground-state energy per site from the lowest eigenvalue of a finite
cluster, so the cluster eigenvalue has to be a certified lower bound. For S = 1/2 in the reduced model (coupling c5
= 0) the entry records a certified bound e_GS ≥ −0.550158313, obtained in the floating-point standard model over 128
exact integer sectors of the group Z2^7 and confirmed by an exact integer re-check of every factorization residual.
For S = 1, at the printed couplings with c5 = 0, the configuration in which every outer pair forms a singlet gives an
exact eigenvalue −1.6380246, below the printed bound −1.632985. The entry then took an unprinted c5 as the likeliest
cause; D-51 later superseded that reading. The rule recorded here still holds: the authors are contacted before
anything is stated publicly. The Lemm, Sandvik and Yang hexagonal chain was rejected as a target because it needs
completeness at about 10^7.

## D-35 (17 October 2026): mutation analysis is evidence, with classified survivors

Mutation analysis (E6) inserts small deliberate faults into the code and counts how many the test suite detects. It
became part of the evidence, with the rule that surviving mutants are classified (equivalent, sound but looser,
unsound below the resolution of the tests, or a real gap) rather than reported as one score. The first run killed 66
of 160 mutants. The real gaps it found were closed by tests: the exact checker was untested, as were SU(2)
off-diagonal terms, edge sectors of the U(1) symmetry, window claims, the QuSpin importer for `pauli=-1` and bosons,
and a guard of the lift. Mutants that shrink a worst-case rounding term cannot be detected by tests, because real
rounding errors are far below the worst case; those parts are protected by the exact checker and, as a later step, by
mechanised proofs of the checker's lemmas.

## D-36 (17 October 2026): the second application is the AKLT gap criteria

The spectral gap criteria of Pomata and Wei for two-dimensional AKLT models (arXiv:1911.01410) became the second
application. Their criterion asks that a quantity η, computed from a finite subgraph, stays below a threshold (for
example 1/4); the project bounds η from above in exact arithmetic. The entry records certified upper bounds of
0.1110450359 (star, below 1/4), 0.84817 for the square-octagon combination 4η⊥ + 2η∥ (below 1), 0.1997398308 (cross,
below 1/3) and 0.1530359435 (decorated honeycomb, below 1/4); the decorated square was still running. The geometry of
every subgraph had to be inferred from the paper's text, so each was confirmed by reproducing the paper's η to 7 to 10
digits before it was certified. A negative control, a threshold placed below η², is correctly not certified. The
honeycomb lattice was not attempted: its space has dimension 2^26 and its final bound needs three further numerical
constants, each its own eigenvalue lemma.

## D-37 (17 October 2026): the AKLT kernel equality is a lemma, also checked by computation

The equality between the kernel of the subgraph Hamiltonian and the range of the valence-bond-solid (VBS) map is
treated as a standard lemma, re-derived in `notes/aklt_kernel_lemma.md` (a Schwinger-boson divisibility argument) and
listed for Sarang's check under D-12. Where the size allows it is also confirmed exactly: the nullity over the
rationals is at most the nullity modulo a prime in every S^z block, and the VBS ranks are computed exactly. This
confirms the lemma for the star, the square-octagon and both decorated-honeycomb subgraphs; the cross (16.7 million
states) and the decorated square (4.1 million) rely on the lemma alone. A floating-point inertia approach to the same
question was rejected because it took more than 10 minutes per block, against seconds for the modular nullity, and
needed a gap assumption.

## D-38 (17 October 2026): the bordered pencil for the decorated lattices

For the decorated lattices the matrix pencil that defines η is certified through a bordered matrix whose inertia is
additive (Haynsworth), written in fully reduced boundary coordinates and split into exact symmetry blocks by the
lattice mirror and the spin flip. The adjugate scaling used for star, square-octagon and cross was rejected because
its determinants reach hundreds of digits and a Schur complement would be dense at 27k by 27k; the bordered matrix
keeps plain Gram entries. A perturbation argument transfers the bound from the factorized matrix to the exact one
using certified constants. Symmetry is used only as an exact integer congruence, after the equivariance of every
ingredient under the signed permutation has been checked exactly. The largest exactly checked block fell from 8842
to 4034. A residual check by row blocks replaced the full-matrix one after a run was killed for lack of memory at 5.9 GB.

## D-39 (17 October 2026): all inspected survivors of the first E6 run resolved

All 37 surviving mutants that needed inspection received a verdict. Twelve were real test gaps (13 mutants) and were
closed by tests. Three of the gaps could have produced wrong output: a Kato-Temple isolation distance taken from a
non-adjacent level gives a wrong bound, and the reported Kato-Temple intervals had never been compared with exact
eigenvalues; a partial failure of S4 would have been reported as CERTIFIED; and the Hermitian part of a slightly
non-Hermitian producer matrix was untested. The other verdicts were equivalent (9), no effect on any statement (6),
sound but looser (4), stricter (1), rounding allowance below the resolution of the tests (3), heuristic (1) and an
edge-case verdict (1). Under D-35 the score is reported only with this classification.

## D-40 (17 October 2026): the S = 1 contradiction is certified, the corrected optimum is an estimate

The contradiction between the printed couplings and the published S = 1 bound was made exact: an exact rational
evaluation, with the energy of the spin-1 six-site ring enclosed by CERTIFY-ED (S2 and S3 certified, radius 1.4e-13),
gives an eigenvalue at most −1.6380246 against the published −1.632985, a margin of 5.0e-3. The corrected max-min
value −1.6343596 is shared by several classes of pair spins, including the class in which every pair has spin 2
(1,116,229 states at S^z = 0). Certifying it would need exact inertia at that size, which is out of reach here, so it
is reported as an estimate.

## D-41 (17 October 2026): the decorated square is certified

The decorated square completed with η at most 0.2203633381, below 1/4 (paper: 0.2203543174). The run used the mirror
times spin-flip blocking, a largest exact block of 4034 and β = 5.2e13 on entries of about 1e45, and took 46 min. The
four W4 benchmark ground energies of experiment E4 were also certified with no floating-point assumption by the exact
sector re-check (experiment E7), including the J1-J2 4x4 lattice at J2 = 0.5 whose S3 had been NOT_CHECKED in W4. From
this entry on, every criterion is quoted from the sharp threshold.

## D-42 (18 October 2026): symveig is merged as a capability

When the symveig source arrived, an audit (`notes/symveig_audit.md`) reproduced four problems. P1: the claim that the
enclosures carry exact multiplicities and that their union contains the whole spectrum is not proved, and it is false
for supplied eigenpairs (a counterexample is in the tests). P2: the sector path never checks that the operator
conserves the sector quantum number; under a staggered field, 58 of 64 eigenvalues lie outside the enclosures the
package reports. P3: the containment check of the global path is circular. P4: the claim to be the first such tool in
scientific Python is wrong, because Arb (through python-flint) exists, although Arb fails on degenerate lattice
spectra. The decision was to rebuild the function inside CERTIFY-ED as `certify_ed.spectrum.enclose_spectrum`, which
certifies counts by the cluster theorem, checks conservation and Hermiticity exactly and uses rounding bounds of the
form |A||x|. E8 compares the two on symveig's 17 configurations. Shipping symveig's engine unchanged was rejected
because of P1 and P2, and the CPC manuscript of symveig was not to be submitted as it stood.

## D-43 (18 October 2026): one codebase, certify-ed 3.0.0

Sarang asked for one homologated codebase. certify-ed 3.0.0 holds the v2 core, the whole-spectrum function, the exact
layer and both applications, with one driver (`python run.py`, the same as `certify-ed reproduce`); v1 and symveig
1.0.0 moved to `legacy/` and are not installed. Several improvements were made while unifying. The exact layer now
covers every local space through a scaled basis with integer weights, which removed its restriction to spin 1/2. The
exact re-check became statement S6 of the certificate. There is one exact factorization (`checker.inertia_exact`, by
row blocks). One AKLT pipeline serves all lattices. The crown S = 1/2 bound comes from the generic
`exact_count_below`. The command line gained `env`, `certify`, `spectrum`, `list`, `test`, `reproduce` and `numbers`,
and the packaging gained `pyproject.toml`, citation metadata and a CI workflow. The quick run finished in 11.5 min
with every step ok and 23 of 23 application checks passing. Keeping the v1 modules importable from `certify_ed` was
rejected because they mixed two products; the v1 code is preserved by the tag `v1.0.0-baseline` and in `legacy/v1`.

## D-44 (18 October 2026): provenance by computational source, and one run per cited version

Every record and every reproduction step carries a fingerprint of `certify_ed/` and `experiments/` without the files
that cannot change a number (then `cli.py`, `testing.py` and `provenance.py`). The number scripts refuse records with
mixed fingerprints or with uncommitted changes. This replaced a check of equal commits, which had failed spuriously
when the full run of 18 October crossed a commit that changed only the command line. The driver gained `--jobs N` (E2,
E6 and E8 always run alone because they measure time or memory) and `--resume`. The rule that follows, and that later
decisions cite, is that a version the paper cites needs one full run at a single computational source. A single
monolithic script was rejected in favour of a package with one driver, which stays importable, tested and runnable
experiment by experiment. `docs/project/software/05_provenance.md` describes the mechanism.

## D-45 (21 October 2026): the one-paper manuscript as a scaffold

The combined manuscript was built as a scaffold (`ms_v3`) and not as a draft. Each section opens with a box that says
what its prose has to do; the claims follow, each with a pointer to its evidence; every number, table and figure is
generated from the run; red boxes mark decisions only Sarang can take; and the project's own mathematics is marked
`UNVERIFIED`. The plain `article` class is used because no venue is chosen. The same step sharpened the application
bounds (the crown S = 1/2 bound became −0.550158299), gave the test suite its own fingerprint (`tests_hash`, on which
E6 depends), classified E6 survivors automatically after the run, and made the number scripts print bounds rounded in
the safe direction. Rejected were drafting full prose (contrary to D-16), the SciPost class, and waiting for Sarang's
prose before assembling the evidence.

## D-46 (21 October 2026): writing the proofs against the code found two soundness gaps

Writing the appendix proofs while reading the code found two gaps, both fixed with tests before the evidence run.
First, interval endpoints and margins were formed in round-to-nearest, which can land half a unit in the last place
inside the exact interval; they are now rounded one unit outward (`fp.sub_down`, `fp.add_up`). Second, the exact "k
lowest" re-check (S6, the routine of E7) certified a group of two or more claimed values when one count above it
agreed. A residual interval guarantees an eigenvalue, not a distinct one: two copies of the first excited vector
claimed as the two lowest were certified while the ground state was missing. Such groups are now bracketed by counts
on both sides; a cut multiplet is INCONCLUSIVE, and FAILED requires an unclaimed eigenvalue below a claimed one. No
reported result changed in substance, but all records were rerun at the fixed source (`results/run_full_20261021`,
source `7fb63586baf7ea6f`). The lesson recorded is that mutation analysis could not have found either gap, because the
tests cannot resolve half a unit in the last place and none used duplicated vectors. The policy that follows is that
every new certified statement gets its proof written against the code before the run the paper cites. Shipping the run
of 18 October with an erratum was rejected because its records came from the unsound code paths.
`docs/project/failures/README.md` lists both gaps.

## D-47 (21 October 2026): Sarang's decisions on four open items

Sarang decided that Kránitz and Penc may be contacted by e-mail, that he himself checks the proofs marked
`UNVERIFIED`, that a statement on the use of automated assistance can be attached to the paper, and that symveig is
dropped as a separate paper. The consequences recorded are a note to the authors updated to v0.2 (later withdrawn,
D-51), a list of every unchecked item in the report in the order to check it, a factual draft of the assistance
statement in the manuscript, and no CPC submission of symveig. Because arXiv:2606.16217 v1 is public and contains the
problems P1 and P2, dropping it is a public act; the options are set out and the recommendation is to act soon (D-53
took the decision). The compatibility layer `certify_ed.compat.symveig` stays, with the corrected behaviour.
