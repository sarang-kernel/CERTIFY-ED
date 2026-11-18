# Decisions D-17 to D-31: the design of version 2 and the evidence policy

This file explains entries D-17 to D-31 of `project_docs/DECISIONS_LOG.md` (version 1.19). They were taken between
September and 11 October 2026, while the version 2 core of CERTIFY-ED was written and its first experiments were run.
Every date and number below is from that file unless another file is named. The statements S1 to S6 are defined in
`docs/software/01_architecture.md`; the experiments E0 to E10 in `docs/experiments/README.md`.

## D-17 (25 September to 4 October 2026): a revised manuscript and a v2 core, as an exception to D-10

Sarang asked for a substantially revised manuscript rather than a patch, and the v2 core was implemented together with
a complete draft. This bypassed D-10, which wanted the engineering plan derived from the signed-off enumeration: the
v2 core is a subset of the enumeration (`SCOPE_ENUMERATION` v0.1) chosen to answer the referee reports with evidence,
namely the statements S1 to S5, the assembly of sectors and the importers. It was also in tension with D-16, because
the draft contained load-bearing prose. The resolution recorded, pending Sarang, was that the draft is a scaffold of
arguments and evidence and that he re-drafts or edits the sections that carry the argument before any submission.

## D-18 (September 2026): five statements, four statuses and a not-checked list

A certificate is a list of statements: S1 model fidelity, S2 eigenvalue enclosures, S3 completeness, S4 eigenspaces
and S5 corroboration. Each statement has one of four statuses, CERTIFIED, FAILED, INCONCLUSIVE or NOT_CHECKED, and
every certificate carries a list of what was not checked. Two alternatives were rejected. Pass or fail flags conflate
"wrong" with "not shown". The v1 practice of taking agreement among several solvers as evidence of correctness was
rejected after Report 1 pointed out that the solvers share components (the tridiagonal reduction DSYTRD). S5 stays,
but only as corroboration, labelled as such, with a table of the components the solvers share.

## D-19 (September 2026): a default accuracy target for S2

S2 presumes a target accuracy of 1e-8 times max(1, the norm of H) unless the producer states its own. Without it, a
true but useless enclosure was CERTIFIED: in the planted error F9 (swapped eigenvectors) the radius was 0.81. The
entry would be revisited if users reported false FAILED statuses on results that are legitimately of low precision.

## D-20 (September 2026): completeness by Sylvester's law of inertia

S3 counts eigenvalues below a shift σ by Sylvester's law of inertia: the number of negative pivots of an L D L^T
factorization of H minus σ equals the number of eigenvalues below σ. The count is made certain by an a-posteriori
backward error β of the factorization. The sparse path uses SuperLU in symmetric mode without pivoting. The shift is
placed by a ladder of probes at 0.9, 0.5, 0.2 and 0.05 of the estimated gap, then at a minimal margin. Beyond the size
limits the statement is NOT_CHECKED, never an uncertified count. Rejected were heuristics based on Lanczos that no
state is missed (they are not certificates) and the QDLDL factorization, which had similar fill and was slower in a
one-point test. The known limit is that fill-in restricts S3 to dimensions of about 1e4 under 4.5 GiB of memory, which
E2 measures.

## D-21 (4 October 2026): the direction of the Jordan-Wigner string is an option

For fermions the Jordan-Wigner transformation attaches a string of signs to each operator, and the string can run over
lower or higher site indices. CERTIFY-ED makes the direction an option of the local space, with QuSpin's convention
(string on the higher-index sites) as default. E0 found the problem: random fermion strings that change the particle
number disagreed with QuSpin, because pairing terms had the opposite sign, while number-conserving terms agreed, which
is why the deterministic tests had passed. The manuscript reports the episode as an instance of the class of errors S1
is meant to catch.

## D-22 (4 October 2026): radii refer to the declared model

Before this entry, the statements were about the producer's floating-point matrix. From here on, the S1 difference
between the producer's matrix and the reference matrix, plus the rounding of the reference builder (together called
the model shift), is added to the S2 radii, the S3 margins, the Kato-Temple intervals and the assembly bound. In
practice the shift is about 1e-15; what changes is what the theorem asserts.

## D-23 (4 October 2026): an ordering condition for the assembly of sectors

When eigenvalues from several symmetry sectors are assembled into the low spectrum of the whole Hamiltonian, every
claimed value must lie below the cut minus the largest backward error of the sectors; otherwise the assembly is
INCONCLUSIVE. The gap was found while writing the proof of the assembly theorem (Theorem 6 at the time).

## D-24 (4 October 2026): every number of the paper is generated from records

Every number, table and figure of the paper is produced from experiment records by `experiments/make_numbers.py`,
which also checks the qualitative sentences the text makes about them. All experiments run at one git commit, and E2
and E4 never run at the same time because running them together exhausted the memory. A second-stack replication
remained a standing requirement before submission.

## D-25 (4 October 2026): no claim of mathematical novelty

The contribution is stated as the scope of the statements (model, completeness, assembly), the independence from the
producer and the engineering for the domain, not as new mathematics. Promotional words such as `first`, `robust` and
`state-of-the-art` are not used, and "complete" is used only in a defined, technical sense with criteria R1 to R8. The
reasons are Reports 1, 2 and 4 and the project's own audit. Under D-12 the proofs of Appendix A were marked
`UNVERIFIED` until re-derived.

## D-26 (4 October 2026): every iterative solver call is seeded

Two E2 runs at consecutive commits gave different numbers of eigenvalues for the 4x4 block. SciPy from 1.15 on seeds
ARPACK from operating-system entropy when no start vector is given, so the records were not reproducible. Every
iterative call (the certifier's gap hint, the ARPACK reference in the experiments and the experiment producers) now
receives a seeded start vector (commit `6143b5d`). Correctness was not affected, because the hint never enters a
certified quantity; the reproducibility of shifts, gaps and run times was.

## D-27 (4 October 2026): the ARPACK multiplet truncation becomes experiment E5

In E2, ARPACK returned an incomplete fourfold level of the 4x4 block for some start vectors. The episode became
experiment E5 and a subsection of the paper, reported at whatever frequency the seeded runs give. At the same time S3
was changed to check the count below the top claimed value at the first probe that over-counts, so that it fails after
two factorizations instead of up to six. Two alternatives were rejected: re-seeding E2 until the 4x4 run came out
clean (a selection of favourable runs) and dropping the 4x4 point. The reason recorded is that E5 is the only error in
the paper that nobody designed, which partly answers the objection that the planted errors of E1 were chosen to suit
the method, and it shows that disagreement between two solvers cannot say which of them is wrong.

## D-28 (4 October 2026): the below-top probe accounts for the backward error

E5 showed on the 4x4 block that β (about 5e-7) can exceed the margin (1.1e-7) near a fourfold level, so the probe
below the top claimed value could not decide. The probe now retries at 4 times (β plus the model shift) below the top
enclosure when β exceeds the margin, the same remedy the probe above the claim already had (commit `e432742`).
Enlarging the global margin was rejected because it would weaken certified gaps everywhere. The first E5 run, which
reported truncated lists as INCONCLUSIVE instead of FAILED, was superseded, and all experiments were rerun at
`e432742`.

## D-29 (11 October 2026): records of interrupted runs are never read

A record from an interrupted run is moved to `experiments/results/superseded/incomplete/` and is never read by
`make_numbers.py`; an interrupted experiment is rerun, not patched. The trigger was an E5 rerun cut off at the end of
a working period on 4 October: the partial record made `make_numbers.py` fail, and the PDF of that date carried the E5
table of the superseded first run, which contradicted the text. The rerun of 11 October at `536b814` (same
`certify_ed` tree) reproduced the outcomes of `e432742` seed by seed. Reusing the `e432742` record was rejected
although it would have given the same numbers, because the record set would no longer come from one tree and one
latest run. `make_numbers.py` now reports the run date of each experiment when records come from different days.

## D-30 (11 October 2026): a table of the contribution by kind

The paper carries a table that separates the contribution into algorithmic, architectural, integration, usability,
performance and workflow parts, with what is and is not claimed for each. The 4 October draft had made the separation
only in prose. No novelty in performance is claimed.

## D-31 (11 October 2026): a visual check of every rendered figure

The pre-submission checklist gained a visual check of every rendered figure and table. The fill-in panel of Figure 3
was empty in the PDF of 4 October because of a collapsed logarithmic axis, and it had passed both the independent
review and the 36 sentence checks, which read data and not pixels. The fix is commit `5c3f8b8`.
