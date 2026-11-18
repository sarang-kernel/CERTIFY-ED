# A proposed decision on the Lean results (D-62, not adopted)

This file is a proposal. It is not an entry of `project_docs/DECISIONS_LOG.md`, and nothing in this repository treats it
as decided. The decision belongs to Sarang Vehale, because it changes how rule D-12 is applied, and D-12 is his rule.
Until he decides, every document of this repository keeps the wording of D-12: a derivation document that has a Lean
theorem behind it is still "derived, machine-checked, independently re-derived (not a human check), Sarang's check
pending", and the Lean level is reported next to that wording, never instead of it.

## The question

Rule D-12 (13 July 2026) says that new mathematics counts as established only after a person, Sarang Vehale, has
re-derived it. The rule was written when the only alternatives to a human re-derivation were a second written
derivation and random testing, and both can share an error with the first derivation. The re-verification in
`lean/` adds a third kind of evidence. A statement of a derivation document is written as a Lean
theorem, the proof is checked by the Lean kernel (level K in `README.md`), and the exported proof is accepted again by
a second checker that was written independently of Lean (nanoda, level K2) or replayed by Lean's own `leanchecker`
(level K+). The proof is then no longer the weak point. What can still be wrong is the statement: the Lean theorem may
say something other than the document, it may rest on a Mathlib definition that does not mean what the document means,
or the document's statement may not be what the code needs. These three gaps are exactly the ones that the column
`correspondence` of `results/mathlib_enclosure.json` and `results/mathlib_counting.json` describes in words, and that a
reader has to check by reading.

The findings of the formalization show that the gaps are real. The Lean work found sentences of the documents that
are false as written (the closing condition of guard 6 of Theorem 1, which admits one failing value beyond the stated
range; the display of Proposition 14 without its hypothesis; one inequality in the proof of Theorem U5), an argument
that holds for a different reason than the one given (D-02 part b, by a second-order margin), and two statements that
are slightly too strong (the exactness of `from_float`, the index of Lemma A1 for complex data). It also confirmed a
remark the documents already make (the constant 9/8 of the 3.1 logarithm bound) and showed that guard 4, true in its
stated range, fails just beyond it, a range that the code does not check. `docs/project/failures/07_found_while_formalizing.md`
lists all eleven. None of them changes a stored certificate. Each one would have been found by a careful human re-derivation
as well, and each was found here because the Lean kernel does not accept an argument that only nearly holds.

## The options

Option A keeps D-12 unchanged. Lean results are reported as an additional level of evidence, and every derivation
document still waits for Sarang's re-derivation. This costs the most of Sarang's time and gains nothing from the
formalization except confidence.

Option B replaces the re-derivation by a statement review for every statement that has a Lean theorem at level K2 or
K+ with no `sorry`, no added axiom and no `native_decide`. Sarang reads the Lean statement, the Mathlib definitions it
uses and the `correspondence` text, and records that the Lean statement says what the document and the code need. The
proof is then taken as established by the kernel and the second checker. Statements without a Lean theorem, and the
parts of statements marked `partial` in the report, keep the full D-12 requirement. Reading a statement and its
definitions is a much smaller task than re-deriving a proof, and it is the task that the formalization cannot do.

Option C counts a K2 theorem as established without any human reading. This is rejected in this proposal, because the
statement gap described above is precisely what a kernel cannot close, and the findings listed above show that
document statements and Lean statements do diverge.

## The proposal

The proposal is option B, with three conditions. First, the statement review is recorded per statement, with the Lean
name, the commit of this repository and the date, in the same way that a D-12 check is recorded now. Second, the
review covers the hypotheses as well as the conclusion, because every finding of the formalization was a hypothesis
or a range, not a conclusion. Third, a statement whose Lean form is weaker than the document (for example only for
real symmetric matrices, or only in exact arithmetic) is established only in that weaker form, and the document says
so.

The decision is reopened if a Lean theorem at level K2 is later found to be wrong for a reason other than its
statement, for example an unsound Mathlib definition or a defect in both kernels, or if Sarang finds that the statement
reviews take as long as re-derivations.

## Where the evidence is

`docs/lean/VERIFICATION_REPORT.md` lists every statement with its Lean names, its level, what is missing
and the source paragraph. `docs/lean/README.md` defines the levels K, K2, K+, N, P and H, and
`docs/project/STYLE.md` rule 4 fixes the words used for each.
