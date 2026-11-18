# The mathematics of CERTIFY-ED: map of the documents

This folder explains the mathematics behind a CERTIFY-ED certificate: what each statement of a certificate asserts, the
model of floating-point arithmetic on which the floating-point statements rest, every theorem and lemma with its
hypotheses, proof outline, implementation and check status, and the 22 derivation documents of version 3.2 with their
machine checks and the verdicts of their independent second derivations. The applications of the theory (the pyrochlore
crown, the AKLT lattices, the exact benchmarks, symveig and the C verifier) are in `docs/project/applications/`.

Every path below is relative to `archive_v5/` (see `docs/project/sources/SOURCES.md`). The code is the clone
`code/certify-ed/` on branch `dev/3.2` at commit `a814d28`, with the tags `v3.1.0` (`e860a83`), `v3.2.0` (`55d9f61`) and
`v3.2.1` (`f3c478b`). Function names refer to that commit. The version that the manuscript cites is still 3.1.0, until
Sarang Vehale decides otherwise (decision D-61 in `project_docs/DECISIONS_LOG.md`); the statements added in 3.2 are
described here because they are in the code and in its 3.2.1 evidence run, not because a paper cites them.

## The documents

| file | contents |
|---|---|
| [certificate_statements.md](certificate_statements.md) | what a certificate states: the statuses, the eight statements S1 to S8, and the statements added in 3.2 (C-03, C-04, C-06, C-07, C-12, C-13, C-16, E-03, E-04, E-05, E-07, D-01, D-02), each with the theorem it rests on and the module that implements it; the inputs, which operator each statement is about, the assembly from sectors, and the `not_verified` list |
| [arithmetic_model.md](arithmetic_model.md) | the floating-point model: binary64, the unit roundoff u, the constants gamma_n, directed rounding of endpoints, the guard factors, the run-time probe, the code-facing accounting of the Theorem 1 radius (INT-04), the no-underflow hypothesis (INT-01) and the underflow-tolerant model (INT-05) |
| [theorems.md](theorems.md) | every theorem and lemma: Theorem 1, the polar-factor lemma, Theorems 2 and 2', Theorem 3 (Kato-Temple), Lemma 4 (inertia with backward error), Corollary 5, Theorem 6, Proposition 7, Theorem 8, Proposition 9, the exact layer (Lemmas 10 to 12, Proposition 13), Propositions 14 to 16 and 16', Lemma 17, and the model shift of Appendix A; the size of the bounds in the 3.2.1 evidence run; one table at the end gives the check status of each |
| [exact_layer.md](exact_layer.md) | the exact layer behind statement S6: what is recomputed in integer and rational arithmetic (exact matrix, exact radii, exact counts, symmetry sectors, the "k lowest" decision), by which function of `checker.py` and `exact_sectors.py`, at which sizes, how it enters a certificate, its tests, the correction of 21 October 2026, and what remains trusted |
| [derivation_documents.md](derivation_documents.md) | the 22 derivation documents of `code/certify-ed/docs/proofs/` one by one: what each derives, its verification script and case counts, the independent verdict and its remarks, and the code corrections that followed |

## Names and numbering

The theorem numbers are those of `code/certify-ed/docs/theory.md`, which follows the numbering of the third manuscript
scaffold (`manuscript_v3`): Theorem 1 (residual bound), Theorems 2 and 2' (clusters), Theorem 3 (Kato-Temple), Lemma 4
(inertia), Corollary 5 (completeness), Theorem 6 (projector), Proposition 7 (quantum numbers), Theorem 8 (sector
assembly), Proposition 9 (whole spectrum), Lemmas 10 to 12 and Proposition 13 (exact layer), Propositions 14 to 16 and
Lemma 17 (statements added in 3.1). The fourth manuscript scaffold (`manuscript_v4/sections/s3_math.tex`) refers to the
same results by LaTeX labels (`thm:residual`, `lem:polar`, `thm:cluster`, `lem:epsstar`, `thm:kato`, `lem:inertia`,
`cor:complete`, `thm:dk`, `prop:qn`, `thm:compose`, `prop:spectrum`, `lem:scaled`, `lem:exactcount`, `lem:sectors`,
`prop:exactlowest`); [theorems.md](theorems.md) gives both names. The two lemmas of the AKLT application (the kernel
lemma and the bordered-pencil lemma) are numbered 14 and 15 in the manuscript scaffolds, which collides with
Propositions 14 and 15 of `theory.md`; these documents therefore call them by name only, and they are explained in
`docs/project/applications/aklt_pomata_wei.md`. The statements added in 3.2 are named by their roadmap identifiers (C-03, E-04,
D-01 and so on), and their mathematics is in the derivation documents named after them.

## How strongly each result has been checked

The project rule D-12 says that new mathematics counts as established only after Sarang Vehale has re-derived it
himself. Under decision D-59 (30 October 2026) the results that were waiting for that check received a derivation
document each. Each document was then machine-checked by a script in exact rational or high-precision arithmetic on
random and adversarial cases, and a second derivation was written independently from the statement and the code mapping
alone, without reading the original proof first, and compared with it. That second derivation was produced
automatically; it is not a human check. The status of every derivation document is therefore written in these documents
as: derived, machine-checked, independently re-derived (a second derivation written from the statement alone, not a
human check), Sarang's check pending. None of these results is called established, and no document of this folder says
that a person checked a result unless that person is named.

Classical results that the project takes from the literature (Theorem 1 from Parlett 1998, Theorem 3 from Kato 1949 and
Temple 1928, Lemma 4 from Sylvester's law of inertia and Weyl's inequality, Kahan's inclusion theorem, the Davis-Kahan
theorem) carry their literature reference instead. Where the manuscript scaffold and the code documentation disagree on
whether a result is classical, [theorems.md](theorems.md) reports both labels.

The Lean files of `lean/` are a separate, later check. Where a result of this folder has a Lean
counterpart, `docs/lean/VERIFICATION_REPORT.md` gives its trust level; nothing in this folder depends on it.

## Recreating the machine checks

Each derivation document has a script `code/certify-ed/docs/proofs/verify_<id>.py`, and each independent derivation has
a script `code/certify-ed/docs/proofs/independent/adversarial_<id>.py`. They run from the root of the code clone with
the package on the path. In a Windows command prompt, from the root of this repository:

```
git clone docs/project/sources/certify-ed_dev-3.2.bundle certify-ed
cd certify-ed
git checkout a814d28
python -m pip install numpy scipy mpmath python-flint
set PYTHONPATH=%CD%
python docs/proofs/verify_P14.py --random 3000 --seed 0
python docs/proofs/independent/adversarial_P14_degeneracy_bracketing.py --cases 300 --seed 37
```

The arguments that produced the recorded counts are given per document in
[derivation_documents.md](derivation_documents.md). The outputs of the long runs are kept in
`code/certify-ed/docs/proofs/runs/`.
