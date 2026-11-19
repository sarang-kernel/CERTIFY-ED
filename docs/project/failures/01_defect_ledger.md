# Ledger of defects, from the first draft to 21 October 2026

This document lists every defect the project found in its own work, and in work it relied on, up to the independent
review of 21 October 2026, with how it was found, why it happened where the record says so, and what was done. The
source is the defect chapter of the consolidated report, `report/08_defects.md`, read together with the decision log;
the defects after 21 October are in `02_integration_findings.md`, `03_review_and_derivation_findings.md` and
`04_failed_runs_and_attempts.md`. Identifiers such as Tech-1, CC3 or IMP-18 are those of the audit or review that
found the defect.

## Before May 2026: the first draft

A first draft presented textbook results as new theorems with incomplete proofs, used illustrative figures and
benchmark numbers instead of measured ones, and claimed bitwise reproducibility across platforms without a public code
repository. An internal mathematical review (`certify_ed_verification.md`) found all three. The draft was rebuilt: v1
makes no theorem claims of that kind, its figures and numbers come from real runs (v1.0.0 had 53 tests and 81
validator tests, agreement with QuSpin to 1.6e-14 and with mpmath to 1.6e-15), it has a public repository and a Zenodo
record, and the claim of bitwise reproducibility was dropped. The reason these defects existed is the one decision
D-12 later states as a rule: drafted text and mathematics can be fluent and wrong, and only a check
against measured data or a re-derivation finds that.

## June to July 2026: the audits of v1

| Id | Defect | Status in version 2 and later |
|---|---|---|
| Tech-1 | the agreement with QuSpin was written as 1.6e-10 in one section and 1.6e-14 everywhere else | no longer in the paper |
| Tech-2, CC3 | "three independent LAPACK paths" share the tridiagonal reduction DSYTRD, and two use the same driver DSYEVD | agreement among solvers removed as evidence of correctness; S5 declares the shared components, and a cyclic Jacobi solver shares no tridiagonalisation (decision D-18) |
| Tech-3 | the construction of the Hamiltonian was outside what was checked | now S1, the first statement, with attribution of errors to terms |
| Gap-1 | the swapped-eigenvector counterexample to "residuals suffice" was buried | E1 case F9 and decision D-19 |
| Gap-2 | no certificate schema | certificate format v2 with canonical JSON, schema identifier, hashes, HDF5 export |
| Gap-3 | no path for symmetry sectors | sector audits, assembly (Theorem 8), the QuSpin importer |
| Gap-4 | sparse against dense compared only at N = 6 | E2 up to dimension 112,720 |
| CC1 | author names swapped in `CITATION.cff` | fixed |
| CC2 | test counts inconsistent across documents | one source, the test run of the evidence run |
| CC4 | `is_certified` promised more than "residuals below 1e-10" | replaced by statement statuses |
| CC5 | a Jordan-Wigner test of the Kitaev chain hard-coded `passed=True` | the v1 code moved to `legacy/` |
| CC6 | hashing and saving used different JSON options | canonical JSON in format v2 |
| CM1 to CM13 | no model tag, a builder for spin 1/2 only, inconsistent tolerances, an unshifted partition function, a wrong README badge, a Zenodo DOI off by one (20066565 against 20066566), a single author in `setup.py`, no precision regime, no threat model, hard-coded thresholds, no introspection interface | addressed by the redesign (ModelSpec, two builders, documented tolerances, the floating-point probe, both authors); still open: the threat model and the question of which Zenodo DOI is the concept DOI |
| CL1 to CL10 | minor items in comments, docstrings and boundary semantics, and silent symmetrisation in the builders | v1 code retired; silent symmetrisation is never done |

These defects were found by reading: two SciPost referees and the project's own audits of the manuscript and the code.
Their common cause is the one decision D-05 diagnosed, a paper that claimed more than its code covered.

## May 2026: the C1 pilot

The Chebyshev recurrence with the shifted argument (H minus the estimate)/α overflowed to 10^34 to 10^71 at moderate
sizes, and the assumption that smoothing handles degenerate clusters automatically was false. Both were found by
running the pilot, and both were corrected (argument H/α; the theorem must track the cluster residual; decision D-02).

## 25 September to 4 October 2026: the version 2 core

| Defect | Found by | Why it happened | Fix |
|---|---|---|---|
| a true but useless enclosure (F9, radius 0.81) was CERTIFIED | E1 | S2 had no accuracy target, so any true radius certified | default presumption 1e-8 max(1, the norm of H) (D-19) |
| the Jordan-Wigner string direction disagreed with QuSpin for terms that change the particle number | E0, random strings | the deterministic tests used only number-conserving terms, which are unaffected | a `LocalSpace` option with QuSpin's default (D-21) |
| radii referred to the producer's matrix, not to the declared model | reading what the theorem says | the statements had been written about the matrix the certifier received | the model shift added to radii, margins, Kato-Temple intervals and the assembly bound (D-22) |
| the sector assembly needed an ordering hypothesis across blocks | writing the proof of the assembly theorem | the hypothesis was implicit in the code | claims must lie below the cut minus the largest β, otherwise INCONCLUSIVE (D-23) |
| ARPACK records were not reproducible | two E2 runs disagreed on the number of eigenvalues | SciPy from 1.15 seeds ARPACK from operating-system entropy when no start vector is given | a seeded start vector everywhere (D-26, `6143b5d`) |
| the S3 probe below the top value failed near a fourfold level | E5 | β (about 5e-7) exceeded the margin (1.1e-7) | retry at 4 times (β plus the shift) below (D-28); the first E5 run superseded and disclosed |
| the bound on the builder's rounding was loose and in one case not valid | review | | γ_width times the largest row of the absolute matrix, through the matrix-free operator (`adea9c9`) |
| the HDF5 export was untested | own audit | | tests added; the export carries the integrity record |
| running out of memory crashed the certifier | an independent review | the factorization's memory failure was not caught | the certificate reports NOT_CHECKED itself (`4ff9398`) |
| S1 was skipped and S2 INCONCLUSIVE above `reference_max_dim` | independent review | the reference matrix could not be built at that size | probes against the matrix-free reference (W3 at dimension 2^20) |
| the bound on the model shift assumed a symmetric absolute matrix | independent review | | the square root of the product of the 1-norm and the infinity-norm, with column sums |
| S4 measured separations from the claimed values, not from Ritz values, and lacked the model shift | independent review | | fixed, so that Theorem 6 and Proposition 7 cover the code |
| S5 returned CERTIFIED for agreement | independent review | agreement is evidence, not a certificate | status CORROBORATED |
| E0 excused violations of the residual bound up to 1e-12 relative | independent review | a tolerance had been added to the comparison | compared with the lower end of the Arb ball, with no slack |

## 11 October 2026: the delivery check

The PDF of 4 October carried the superseded E5 table, because a rerun of E5 had been interrupted and its partial
record broke `make_numbers.py`; decision D-29 moved partial records to a folder the scripts never read, and the rerun
reproduced the outcomes seed by seed. The fill-in panel of Figure 3 was empty because the logarithmic axis was set
before its limits; the fix is `5c3f8b8`, and decision D-31 added a visual check of every rendered figure, because the
36 sentence checks read data, not pixels, and had passed.

## 17 October 2026: mutation analysis and the applications

Three mutation runs found test gaps. The first (66 of 160 mutants killed) found the exact checker untested and gaps in
SU(2) off-diagonal terms, U(1) edge sectors, window claims, the QuSpin importer for `pauli=-1` and bosons, and a guard
of the lift (decision D-35). The second found 12 real gaps (13 mutants) among 37 inspected survivors; three could have
produced wrong output, namely a Kato-Temple isolation taken from a non-adjacent level, a partial S4 failure reported
as CERTIFIED, and the untested Hermitian part of a slightly non-Hermitian producer matrix (decision D-39). The third
found an untested branch of `check_semidefinite_lower_bound` and a wrong Hermiticity evidence field on the
sector-producer path. All were closed by tests. In the AKLT application, the exact check of the decorated square was
killed for lack of memory at 5.9 GB, and the adjugate scaling gave determinants of hundreds of digits; both led to the
bordered pencil with a residual check by row blocks (decision D-38).

## 18 October 2026: symveig and the single codebase

The audit of symveig reproduced its problems P1 to P4 (decision D-42; the consolidated report lists P1 to P8), and the
function was rebuilt as `certify_ed.spectrum`. The provenance check compared commits and failed when the first full
run crossed a commit that changed only the command line, which led to the computational source fingerprint (decision
D-44); the E2 statement check compared runs instead of blocks. The mutation workspaces lacked `examples/`, which the
suite runs, so the baseline would have failed. The driver logged "started" at submission time and interleaved parallel
log lines; it now logs inside the worker with a lock. `make_numbers.py` printed no summary line. The documentation
quoted -0.550158313 while the pipeline certified -0.55015831338. The floating-point containment check of E8 failed for
tight enclosures, because the reference error (about 9e-15) exceeded the widths; a helper with a reference tolerance
replaced it. Mutation of the final tree found six more gaps (the internal builder-disagreement path, one-dimensional
eigenvector claims, dense non-Hermitian producers, the fermionic translation sign, the list of floating-point caveats,
the default of the square lattice), closed in `4c6510d`.

## 21 October 2026: proofs written against the code

Writing the appendix proofs while reading the code (decision D-46) found two soundness gaps. First, interval endpoints
μ ∓ r, σ ∓ β, separations and cut comparisons were formed in round-to-nearest, which can land half a unit in the last
place inside the exact value; `fp.sub_down` and `fp.add_up` are now used everywhere, with four tests (`6a95cb8`).
Second, the exact "k lowest" check certified a group of several claimed values from one count above it, so two copies
of an excited vector could hide a missing ground state; bracketing counts on both sides fix it (`fe20f86`). The same
work found that a cut multiplet was FAILED in S6 although the claim is then ill-posed rather than false (now
INCONCLUSIVE), that `inertia_exact` did not check that the permuted factor is exactly unit lower triangular
(Sylvester's law needs invertibility; check added), that floats reported from exact rationals and the bounds printed
by the number scripts were rounded to nearest (now rounded in the safe direction, upper bounds up and lower bounds
down), that the AKLT agreement check of at most 2e-10 was stricter than the method delivers on the decorated square
(excess 1.9e-9; set to 5e-9 with the reason documented), that the documentation said the |A||x| widths do not grow
with the dimension (the fitted slope was d^0.97), that the exact checker was quoted as about 600 lines (it is about
700), and that one citation (Yamamoto 2001) could not be confirmed online and was dropped. Mutation of the final run
found eight more test gaps, two of them soundness-relevant: no test exercised a 2x2 pivot with positive determinant,
and no test checked the Dekker splitting constant on which the error-free product and the Gram bound rest; seven tests
were added in `tests/test_gaps_from_mutation_4.py`, and all eight mutants were re-checked as killed. The
classification of those 16 survivors was made automatically and awaits Sarang's check (finding RA-25).

The lesson recorded with decision D-46 is that mutation analysis could not have found the two soundness gaps: tests
cannot resolve half a unit in the last place, and no test used duplicated vectors. Writing each new proof against the
code before the run the paper cites became a standing step.
