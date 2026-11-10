# The integration findings INT-01 to INT-05

After the 3.1.0 evidence run (`run_full_20261022`, tag `v3.1.0` = `e860a83`), the manuscript's evidence pointers were
filled in: for every claim of the text, the file and line of the code that implements it. Doing that found four places
where the text and the code disagree, INT-01 to INT-04; work on the 3.2 branch then found a fifth, INT-05. None of
them changes a statement or a status of the 3.1.0 run. Decision D-57 recorded the first four as known issues of 3.1.0
and kept the run as it stands, because the run is the record the paper cites and no fix changes a statement. The
sources are `review/INTEGRATION_FINDINGS_2026-10-22.md` (line numbers there refer to `v3.1.0`), decisions D-57 to
D-59, the CHANGELOG of 3.1.0 and 3.2.0, and `report/08_defects.md`.

## INT-01: the no-underflow hypothesis was assumed, not enforced

The manuscript said that inputs and radii below 2^-900 are rejected, so that no underflow occurs. The standard
floating-point model behind every bound needs that hypothesis: the error of an operation is relative only while no
result falls into the subnormal range. In the code, `fp.py` documented "no underflow or overflow" as a hypothesis and
the run-time probe checked that gradual underflow works (no flush-to-zero, no denormals-are-zero), but no input or
radius was compared with 2^-900 anywhere. The cause was that the hypothesis had been stated in the text and in a
docstring without a corresponding check. The decision was Sarang's to take, because the options (reject such inputs,
rescale them, or state the hypothesis as the user's responsibility) change what a statement asserts. Version
3.2.0.dev0 enforces it conservatively (decision D-58). Before S2, `certify()` scans the claimed eigenvalues, the
eigenvector entries, the stored entries of the operator (or, for a spec or matrix-free operator, its norm bound and
one product with a seeded random probe) and the model shift, and after the enclosure the radii, existence radii,
residual bounds and residual norms; exact zeros are allowed. Any nonzero magnitude below 2^-900
(`fp.UNDERFLOW_THRESHOLD`) makes S2 INCONCLUSIVE with the reason code `UNDERFLOW_RISK`, and S3, S4, S7 and S8 with it,
because each rests on the S2 enclosure; S1 and S6 are not affected. `Options(underflow_check=False)` or
`--assume-no-underflow` skips the scan and records the choice in the certificate, which then carries the hypothesis as
the user's. Rescaling the input was not implemented, because it would need a statement about the rescaled operator.
The CHANGELOG section "INT-01: the no-underflow hypothesis is enforced" lists the 19 tests of
`tests/test_underflow_check.py`, among them a negative control: a finding never leaves S2 or a dependent statement
CERTIFIED.

## INT-02: no validator rejected a certificate whose scope block omits a statement

The manuscript said that a certificate whose scope block omits a statement is rejected. In 3.1.0,
`Certificate.verify_integrity()` checked the body and array hashes only, and the completeness of the scope block was
checked only by a helper in the tests. A certificate edited to drop a statement from its scope block, and resealed,
would therefore have loaded without complaint. The fix, on the branch `fix/3.1.1` and merged into `dev/3.2`, is
`Certificate.verify_scope()`: every statement identifier appears exactly once in the scope block, every reason code is
known, every statement in scope is present, every narrowed statement is NOT_CHECKED, and nothing lies outside the
scope lists (certificates older than the scope block return `None`). It is called by `verify_integrity()` and
therefore by `Certificate.load()`. The tests are in `tests/test_fix_3_1_1.py`.

## INT-03: exported radii added the model shift in round-to-nearest

The manuscript said that the exported radii bound the error. In `certify.py` the exported arrays were formed as
`arrays["radius"] = enc.radius + eps_shift` (and the same for `existence_radius`) in round-to-nearest, while the
statements themselves use upward rounding (`add_up`). Because the rounding bound of the reference builder makes the
model shift nonzero for almost every certificate, almost every exported radius could understate the exact sum by up to
half a unit in the last place. The cause is that the export was written separately from the statement code and did not
reuse its directed rounding. The statements and statuses were never affected, only the exported arrays. The fix on
`fix/3.1.1` uses `add_up` whenever the shift is nonzero; the independent C verifier checks exactly this property
(check C4, `docs/software/09_verifier.md`).

## INT-04: Appendix A described the accounting of the Theorem 1 radius differently from the code

Appendix A of the manuscript said that the radius of Theorem 1 takes at most six roundings, covered by a factor 1 +
8u/k, with the model shift multiplied in. The code (`enclosure.py`) has three roundings, each covered by a factor (1 +
4u): the bound on the product of the absolute matrix and the absolute vector, the error matrix, and the relative
radius; the model shift is added afterwards with upward rounding. The code's bound is sound as far as was checked; the
text described a different, earlier accounting. The text is Sarang's to rewrite (decision D-16). For 3.2,
`docs/theory.md` describes the accounting as the code implements it, with a test that pins it, and the derivation
document `docs/proofs/thm1_fp-accounting.md` derives it; its independent verdict is GAP for the step that covers
complex data, and the code uses the looser constant γ_{w+3} for complex data as a result
(`03_review_and_derivation_findings.md`).

## INT-05: subnormal intermediate entries in routine LAPACK output

While INT-01 was being enforced on `dev/3.2`, it turned out that ordinary eigenvectors from LAPACK produce
intermediate quantities below the threshold: `scipy.linalg.eigh` returns eigenvector entries of order 1e-265
(numerical zeros of a symmetry sector, above 2^-900), for which the residual vector and the entrywise rounding bound
of Theorem 1 have entries of order 1e-278, and their squares underflow to zero in the norm. An early version of the
branch scanned those intermediates and flagged an ordinary test case, and was withdrawn; a test pins the observation
(CHANGELOG, section INT-01). The no-underflow hypothesis is then violated inside the computation of the residual, not
in the scanned inputs, and a check of every intermediate value would have marked ordinary results INCONCLUSIVE.
Decision D-58 recorded this as a new known issue for a design decision, because the remedy changes a bound. It was
closed in 3.2.0 by the derivation document `docs/proofs/int05_underflow.md`, which uses the floating-point model with
an additive underflow term, fl(z) = z(1 + δ) + η, and adds the corresponding terms to norms, residuals, exact dot
products and the Gram and cluster products. Radii never shrink under this change (Theorem U6 of the document) and are
bit-identical above about 1e-145; the `UNDERFLOW_RISK` scan covers the stored inputs and the model shift only. The
independent re-derivation of this document found an error (Theorem U5 was false as stated for overflowing squared
norms), which was corrected by looser code (`03_review_and_derivation_findings.md`). Sarang's check is pending.

## What the five findings have in common

Each finding is a place where a sentence of a document described a property the code did not implement, or implemented
differently. They were found by tying every claim of the text to a line of code, which is a reading of the code
against the text, not a test. That is the same route as decision D-46 (writing the proofs against the code), applied
to the prose of the manuscript. The findings came after the evidence run had been made; the project's response was to
state them as known issues of the cited version rather than rerun it, since none changed a statement (decision D-57).
