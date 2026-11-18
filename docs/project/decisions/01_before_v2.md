# Decisions D-01 to D-16: the first version, the referee reports and the change of direction

This file explains the first sixteen entries of the project's decision log, `project_docs/DECISIONS_LOG.md` (version
1.19 of 6 November 2026). The log is append-only: an entry is never edited, and a reversal is a new entry that names
the old one. Every date and number below is taken from that file unless another file is named. The entries are
paraphrased; the log keeps the original wording. For each decision the text says what was decided, why, which
alternatives were rejected, and what followed from it.

Two research directions that predate CERTIFY-ED appear in the first entries. Both use the Krawczyk method, an interval
version of Newton's method that encloses an eigenpair by checking that an interval map sends a box into itself. C2 was
a variant that worked block by block on the symmetry sectors of a Hamiltonian; C1 was a variant for Hamiltonians
written as sums of Pauli strings, preconditioned by a Chebyshev expansion.

## D-01 (May 2026): direction C2 is stopped

The symmetry-block-diagonal Krawczyk method (C2) was stopped before a manuscript was written. A literature audit found
that Miyajima (2014) and Rump and Lange (2023) already cover the same ground. The pilot study had a threshold fixed in
advance: C2 would continue only if it made the enclosures 100 times tighter. The pilot measured about 5 times, and
that factor came from the ratio of the full dimension to the largest block, a structural property of the problem
rather than a gain of the algorithm. The rejected alternative was to continue to a manuscript anyway. The findings
stay in the C2 pilot report as material for a possible note on a negative result, and the log says the direction is
not to be revisited in its old form.

## D-02 (May to June 2026): the target of direction C1

C1 was fixed as a Chebyshev-preconditioned Krawczyk method for sums of Pauli strings, with the preconditioner taken as
a Chebyshev expansion of a smoothed inverse evaluated at H/α. Faber, Padé, Zolotarev and conformal-map preconditioners
were compared and not chosen. A shifted argument (H minus the estimate, divided by α) was rejected because it produced
overflows of 10^34 to 10^71, a real defect that was then fixed. The pilot agreed between two software stacks to a
maximum difference of 7.9e-13. The entry records one caveat for any later theorem: smoothing suppresses degenerate
subspaces but does not remove them, so the polynomial approximation residual has to be carried through the proof.

## D-03 (13 July 2026): C1 is put aside

Sarang Vehale chose to finish CERTIFY-ED before returning to C1, although C1 may be the more original contribution.
The cost of the delay was named and accepted. The decision stands until he sets a different priority.

## D-04 (June to July 2026): no rebuttal to the SciPost referees

The first CERTIFY-ED manuscript (v1, see `docs/project/00_overview.md`) received two rejecting reports at SciPost. The project
did not send a point-by-point defence or write to the editor. Two independent experts had converged on the same
criticism, and Sarang judged it to be correct. The entry would be reopened only if the editor invited a response.

## D-05 (13 July 2026): the failure of v1 was a mismatch of scope and claim

The diagnosis recorded here is that v1 claimed to serve exact-diagonalization (ED) practitioners in general while it
covered a narrow case, small dense problems solved by LAPACK. Both referees and the project's own audits arrived at
the same point. The consequence written into the log is that a change of framing alone could not repair the paper.

## D-06 (13 July 2026): broaden to a general checking layer

The project chose to broaden CERTIFY-ED into a general layer that checks ED results (version 2), rather than sharpen
the narrow v1 or offer v1 to another venue. Three alternatives were rejected. A sharpened v1.1, resubmitted within one
or two months, was rejected by Sarang as still written for one set of referees. A quick software publication of v1
(JOSS, SoftwareX or CPC) was set aside because he wanted to finish the work properly. Reframing v1 as a
reproducibility framework had been rejected earlier as a downgrade. The stated priority is quality first and
publication second; the decision would be reviewed only if the broader scope proved infeasible with the available
resources.

## D-07 (13 July 2026): enumerate everything first

The first phase of version 2 was a complete enumeration of what checking an ED result can mean, with no filtering by
perceived importance. Sarang asked explicitly for all combinations to be listed; narrowing would come later, as a
separate and recorded step, when an editor or referee asks for it. The rejected alternative was to enumerate only the
combinations that seemed to matter.

## D-08 (13 July 2026): no constraints in the scope document

Platform, programming language and interoperability were left open in the scope enumeration. Where a choice would
eventually be forced, the document lists the options instead of choosing one.

## D-09 (13 July 2026): no preliminary list from Sarang

Sarang did not provide his own list of checks before the enumeration was written, so that the enumeration would not be
anchored to it. His experience enters at review time instead.

## D-10 (13 July 2026): the engineering plan follows the enumeration

The engineering plan is derived from the enumeration after it is signed off, and is not drafted in parallel with it.
The reason given is that planning first had silently filtered the scope of v1. D-17 later records a deliberate
exception to this rule.

## D-11 (13 July 2026): talk to producer authors after the scope exists

Contact with the authors of ED codes (QuSpin, XDiag, HΦ and others) was scheduled for after the scope document
existed, with concrete questions about what each code can export. A concrete question gets a more useful answer than a
general one, and what a producer can export is a design input for the importers rather than a late discovery.

## D-12 (13 July 2026): new mathematics needs a person who re-derives it

This is the rule that governs the status of every derivation in the project. Colleagues of Sarang act as reviewers,
and anyone willing to re-derive mathematics is reserved for the sections that contain new mathematics. The reason
recorded is that automatically drafted mathematics can be fluent and subtly wrong, that Ritu Goel reads without
re-deriving, and that one person who re-derives is worth more than several who check plausibility. In the documents of
this repository the rule has a fixed consequence: a new result counts as established only after Sarang Vehale has
re-derived it. The 22 derivation documents in `code/certify-ed/docs/proofs/` are therefore described as derived,
machine-checked and independently re-derived (a second derivation written from the statement alone, not a human
check), with Sarang's check pending (see D-59).

## D-13 (13 July 2026): documentation is mandatory

Every project document is versioned and has a changelog, `PROJECT_STATE.md` is the entry point, the decision log is
append-only, and design decisions are written down when they are taken, including the rejected approaches. The reason
recorded is that the project has no memory across working periods except what is written down, and that a stale
handover leads to the confident re-proposal of ideas that were already rejected. Sarang asked for documents that would
hold up in the future.

## D-14 (13 July 2026): one idea taken from an unsolicited e-mail

An unsolicited e-mail of unknown origin received no reply, but one technical idea in it was adopted on its merits:
invariants resolved by eigenspace, computed through the projector onto a cluster of eigenvalues. Such quantities do
not depend on the basis chosen inside a degenerate eigenspace. The idea became part of statement S4 and also exposed a
gap in the project's own audit.

## D-15 (21 July 2026): no response to the third rejecting report

A further SciPost report called the contribution insubstantial, read the paper as claiming that checking BLAS and
LAPACK was its main contribution (the third misreading of that kind), and said the text read as mostly
machine-generated. No response was sent. Three rejections settled the outcome; the repeated misreading confirmed the
diagnosis of D-05; the remark on provenance was substantially correct (the drafting had relied heavily on automated
text generation, disclosed at submission), and the criticism of style matched flaws that the project's own audit of
the manuscript had already listed. A rebuttal and any defence of the drafting method were rejected.

## D-16 (21 July 2026): Sarang drafts the prose of every manuscript

From version 2 on, Sarang drafts the prose of the manuscripts. Other contributors and tools may audit, check facts,
tighten and review adversarially, but they do not write the load-bearing sections. Two alternatives were rejected:
continuing with machine-drafted prose, which had produced the style the referees criticised and does not build
Sarang's own record as a writer, and concealing the assistance, which was never an option. The style targets recorded
with the decision are that every paragraph carries a specific claim tied to a number or a mechanism, that sentences
without content are cut, that central claims are stated once, and that nothing is asserted that is not shown.
Assistance continues to be disclosed at every submission. Machine-drafted text remains acceptable for internal
documents, audits and code documentation, which is the category the documents of this repository belong to.
