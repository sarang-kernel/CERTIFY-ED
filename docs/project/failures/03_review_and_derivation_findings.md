# Findings of the independent review and of the independent re-derivation

Two outside readings found defects that none of the project's own routes had reached: the independent review of 21
October 2026 and the independent re-derivation of the derivation documents on 30 October 2026 (decision D-59). This
document gives both, with the reason each finding arose and what was done. Sources:
`review/REVIEW_CERTIFY-ED_2026-10-21.md`, `review/REVIEW_findings_all.csv`, `review/REVIEW_impl.md`,
`review/REVIEW_B_applications.md`, `report/08_defects.md`, `report/14_review.md`, and
`code/certify-ed/docs/proofs/VERDICTS.md`.

## The independent review of 21 October 2026

The review read the archive of 21 October 2026 in six tracks: an audit of the code with attack tests, an independent
reimplementation of the applications, a replication of the runs, a check of the report against the archive, a check of
the manuscript, and a check of the mathematics by the coordinating reader. `review/REVIEW_findings_all.csv` lists 179
findings.

| Track of the review | Findings | critical | major | minor | info |
|---|---|---|---|---|---|
| Applications | 28 | 3 | 3 | 6 | 16 |
| Code audit | 31 | 3 | 1 | 15 | 12 |
| Manuscript check | 50 | 0 | 12 | 28 | 10 |
| Mathematics (lead) | 14 | 0 | 1 | 7 | 6 |
| Replication | 20 | 0 | 1 | 9 | 10 |
| Report check | 36 | 0 | 5 | 15 | 16 |
| all | 179 | 6 | 23 | 80 | 70 |

### The critical findings

Three input paths gave a false CERTIFIED. IMP-18: `checker.radius_exact` returned 0 for a zero test vector, so the
exact layer certified a wrong eigenvalue with no floating-point assumption; the cause was that Theorem 1 needs x ≠ 0
and the code never checked it. IMP-21 and IMP-22: a NaN claimed eigenvalue gave S2 and S3 CERTIFIED and made
`enclose_spectrum` report `complete = True`; the cause was that comparisons with NaN are false, so tests written as
"if bad: fail" pass silently. For 3.1.0 a zero vector gives INCONCLUSIVE, the hypothesis x ≠ 0 is stated where
Proposition 13 uses Theorem 1, non-finite inputs are rejected when they are read, the comparisons are written to fail
safe, and the review's attack tests were added to the suite.

Three findings concerned the S = 1 crown of the pyrochlore application. B3-2: the published bound -1.632985 is
reproduced at the printed couplings in the two pair-spin sectors the paper names. B3-3: the project's diagnosis that
an unprinted coupling c5 was the likeliest cause was wrong, because the paper states that c5 is neglected. B4-1: the
sentence of the note v0.2 to the authors, "we could not reproduce the bound", was false. The cause of all three was
that the project had read its own computation and not the paper's text closely enough; the review's independent crown
code, read against the paper's source, found the omitted all-singlet sector as the actual discrepancy. The note v0.2
was withdrawn, v0.3 waits for Sarang's check, and the wording of the report, the scaffold and the decisions was
corrected (decision D-51).

### Major findings and their resolution

IMP-13 and M-02: a correct claim given by a valid but non-orthogonal basis of a degenerate level got S2 FAILED, with a
radius of 4.91 against a target of 4.5e-8, because the term (‖A‖ + ‖Λ‖)δ of the cluster radius grows with the norm and
not with the residual; the radius became min(ρ, ε\*) with the sharper bound of the review, and a claim that misses the
target only through its orthonormality defect is INCONCLUSIVE (decision D-52). M-01 and RA-22: the radius ε = (w +
√(w² + 4ρ²))/2 of Theorem 2 is never smaller than Kahan's ρ, and the remark describing Kahan's theorem as a special
case was wrong; Theorem 2 was restated as Kahan's theorem applied to the polar factor. RA-20 and RA-21: the report
said "certified, exact" for results whose lemmas were unchecked, and "the S = 1/2 bound is certified" for what is the
reduced-model value, not the paper's nine-coupling headline (B2-4, B4-2); the wording was narrowed everywhere. RA-26
and RA-27: the private account address was the author of all 52 commits and appeared in the configuration, three
nested tarballs and the report, and every commit carried a private link; the history was rewritten (decision D-48).
RA-33: the options for the public symveig preprint did not cover its code release; decision D-53 chose a corrected
replacement and a deprecation of the code. RA-34: the report's list of open items omitted items the report itself
implied. MS-N03: the E6 post-run files had no provenance stamp, and the E6 classification was presented as a verdict
although it had been made automatically (RA-25); stamps became mandatory and the classification is now labelled as
suggestions. SS-03: QuSpin was not available on the second stack, which left E0, E1, E3 (part d), E4 and E5 without a
two-stack check until the Windows environment of decision D-54. The remaining major findings (MS-H02, MS-H06, MS-H07,
MS-A01 to MS-A05, MS-M01, MS-M11, MS-R07, B3-5) concern how the manuscript answers the referees, its novelty claims,
its related work and its appendix hypotheses, and were addressed in the ms_v4 scaffold, whose prose remains Sarang's
(decision D-16).

### Minor findings that changed the code or the records

The S3 window endpoints were formed in round-to-nearest (IMP-08), the floating-point probe did not detect
round-toward-zero (IMP-24), `inertia_dense` did not check that L is triangular (IMP-14), `compose.py` symmetrised
non-Hermitian producer blocks silently (IMP-20), the provenance flagged a Windows checkout as dirty because the
executable bit is lost (IMP-05), and three code states reported the same version 3.0.0 (IMP-26); all were fixed in
3.1.0. The manuscript macros rounded the AKLT criterion minimum up to 0.45 against a table value of 0.44418 (IMP-28,
MS-N01), took test counts from the working tree rather than the run (MS-N02), mixed E2 text and table from different
runs (MS-N04, MS-N05) and printed 1.3e-13 where the record gives 1.4e-13 (RA-09); the number scripts now take every
number from the run and round bounds in the safe direction.

## The independent re-derivation of 30 October 2026

Under decision D-59 each of the 22 derivation documents in `code/certify-ed/docs/proofs/` was derived a second time
from its statement and its code section alone, without its proof, then compared step by step, and attacked with an
independent adversarial script in exact or 60-digit arithmetic. The second derivation is a second machine-assisted
reading, not a human check; Sarang's check under decision D-12 is pending for every document. The verdicts, from
`VERDICTS.md`:

| Derivation document | Verdict |
|---|---|
| `appA_model-shift.md` | AGREES-WITH-REMARKS |
| `C-03_entanglement.md` | AGREES-WITH-REMARKS |
| `C-04_correlation_lift.md` | GAP |
| `C06_truncated_thermodynamics.md` | AGREES-WITH-REMARKS |
| `C16_exact_bracket.md` | AGREES-WITH-REMARKS |
| `cor5_pairing.md` | AGREES |
| `E-04_parameter_intervals.md` | ERROR |
| `E-07_davis_kahan.md` | GAP |
| `int05_underflow.md` | ERROR |
| `L17_dyadic_exp_ln.md` | AGREES |
| `P14_degeneracy_bracketing.md` | AGREES |
| `P15_summed_bounds.md` | AGREES |
| `P16_thermo_intervals.md` | AGREES-WITH-REMARKS |
| `thm1_fp-accounting.md` | GAP |
| `thm2_polar-factor.md` | AGREES-WITH-REMARKS |
| `thm2prime_eps-star.md` | AGREES-WITH-REMARKS |
| `thm8_rho.md` | AGREES |
| `E-05_variational.md` | GAP |
| `E-03_generalized.md` | AGREES-WITH-REMARKS |
| `C-07_moments.md` | AGREES |
| `D-01_sparse-residual.md` | AGREES-WITH-REMARKS |
| `D-02_matrix-free.md` | GAP |

In all there are 6 AGREES, 9 AGREES-WITH-REMARKS, 5 GAP and 2 ERROR verdicts, as decision D-59 records.

### The two ERROR rows

The first ERROR is in `int05_underflow.md`. Theorem U5 claimed a valid Theorem 1 radius for every finite stored matrix
and vector, with no hypothesis on their magnitude. It is false: for A = diag(1, 2), μ = 1 + 1e-8 and x = (1e160, 0),
the squared norm of x overflows, `fp.norm2_lower(x)` returned infinity, and the radius came out as 2^-1073 while the
true distance to the nearest eigenvalue is 1e-8; `enclose` even marked the pair distinct. The INT-01 scan looks for
small magnitudes only, so it did not catch the large one. The adversarial script found 11 violations of U5 before the
correction. The fix is looser: `norm2_lower` returns 0 when the sum of squares overflows, so that the radius becomes
infinite and no enclosure is claimed; two negative-control tests in `tests/test_independent_check.py` fail on the old
code and pass on the new. The statement of the theorem needs a hypothesis that the sum of squares does not overflow,
or a sentence in Lemma U3, which is a change of proof text left to Sarang.

The second ERROR is in `E-04_parameter_intervals.md`. The certified crossing criterion of
`sweep.certified_crossing_interval` applied Lemma 3 across the jump of the eigenvalue tube at interior grid points of
a scan and ignored the affinity defect there, so a crossing could be hidden: before the fix, the run "<" on [0, 1.045]
hid a crossing at 0.99755, and the adversarial script found 60 targeted crossing cases that violate. The fix adds the
scan's affinity defects at interior grid points to the slack (looser), with a negative control in
`tests/test_independent_check.py`; the adversarial count is 0 after it.

### The five GAP rows and the remarks

A GAP verdict means a step of the proof is not justified as written, while no certified bound is affected. The five
are: `thm1_fp-accounting.md`, where for complex data the moduli are rounded once more than the proof accounts for, so
that the stated inequality does not follow; `C-04_correlation_lift.md`, where a composition inequality of the rounding
constants is false for the complex constant, though the statement holds by another route; `E-07_davis_kahan.md`, where
a 2-norm increment is used in a Frobenius-norm argument; `E-05_variational.md`, where a step of Lemma 3 presupposes
what it uses; and `D-02_matrix-free.md`, where Lemma M3 omits the rounding of the moduli for complex data. For `thm1`
and `D-02` the code took the looser constant γ_{w+3} for complex data at once (allowed without a proof document,
decision D-59); the other three need only text changes, left to Sarang. Among the remarks of the AGREES-WITH-REMARKS
rows, one changed the wording of a certified sentence: the S8 exact bracket printed "exactly g eigenvalues in [lo,
hi]" with the outward-rounded float ends, which an eigenvalue within one unit in the last place outside the rational
bracket would make false; the sentence now names the exact rational bracket (`C16_exact_bracket.md`).

The change to γ_{w+3} had one side effect: a test that compared complex-data radii bit for bit with a frozen reference
failed, so the first attempt at the 3.2.0 release mutation sample failed its baseline at `b832d8e`; the comparison was
changed to "at least, and within 1 + 1e-12" for complex data in `e1ebaed` (`04_failed_runs_and_attempts.md`).

## What the two readings have in common

Both read the work from outside, with adversarial intent, and both found what testing had not: inputs no test used (a
zero vector, a NaN, a huge vector), a sentence of a document that the code did not implement, and a step of a proof
that does not follow. `report/08_defects.md` calls this the fifth route by which defects were found, after proofs
written against the code, independent references, runs at scale and mutation analysis. Decision D-59 makes the
independent re-derivation part of the regime for every new derivation, and the review's attack tests are part of the
suite since 3.1.0.
