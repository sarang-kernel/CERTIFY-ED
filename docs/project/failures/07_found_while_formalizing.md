# Findings made while formalizing the mathematics in Lean

Writing a statement as a Lean theorem forces every hypothesis and every range to be stated, and the Lean kernel does
not accept an argument that only nearly holds. The re-verification in `lean/` therefore read the
derivation documents more strictly than the earlier checks, which were random and adversarial testing by the verify
scripts and an independent second derivation written before the original proof was read (not a human check). It found eleven
places where a document and the Lean statement differ. None of them changes a stored certificate, a status or a
published number. Seven were corrected in the code repository by documentation commits that change no computational
file (`40d59f4` and `fa86456`, see `docs/project/sources/SOURCES.md`; the source hash stays `cb73dec3e4433a3b`), one is a missing
check in the code that is a candidate for release 3.2.2, and three confirm a remark the documents already make or
record a difference of form. The table of `docs/lean/VERIFICATION_REPORT.md` lists them with the identifiers
L-1 to L-11 used here, and `lean/results/findings_documents.json` holds the same data. Paths are
relative to `archive_v5/`.

## The range of the floating-point guards (L-1, L-2, L-3)

Theorem 1 in floating point rests on six elementary inequalities, the guards Lambda1 to Lambda6 of
`code/certify-ed/docs/proofs/thm1_fp-accounting.md`, each stated for row widths w and dimensions n up to 2^26. The
guards are statements about finitely many rational numbers, so the core-Lean library `CertifyCore` could state each one
exactly and prove it by evaluation in the kernel. That was a reasonable first target because the whole floating-point
accounting stands on these guards, and because the verify script had tested them only on a grid of values (every w and
n up to 20 000 and the powers of two from 2^15 to 2^26). The theorems `lambda6` and `lambda6_fails_at_boundary` in
`CertifyCore/Theorems/Thm1.lean` show that Lambda6 holds for every w up to 2^26, as stated, but that the closing
sentence of its proof (lines 54 to 56), which extends it to every w with (w + 2)u at most 1/3, is false at exactly one
admitted value, w + 2 = 3002399751580330 = floor(2^53/3), where the margin is minus 7 u^2. A correct closing condition
is (w + 2)u at most (1 - 5u)/3. The upper norm guard Lambda4 with N = 2n (lines 44 to 50) holds for n up to 2^26 with a
margin of 0.75u and fails from n = 84886744 on; this second value comes from an exact bisection in
`export/findings_values.py`, not from a Lean theorem. Both statements are therefore correct for the range they state,
and the corrections only concern sentences that reach beyond it. The code is where the range matters: neither
`_rounding_bound` in `certify_ed/enclosure.py` nor the norm guards of `certify_ed/fp.py` (lines 149 to 167) check that
w and n stay below 2^26. No stored operator comes near (the largest dimension in the run records is 1116229), so no
certificate is affected, but a certificate for a larger operator would rest on a guard outside the range in which it holds. The
erratum of commit `40d59f4` records L-1 and L-2. The missing check (L-3) was added in 3.2.2 (commit `1f47d4c` on the
branch `fix/3.2.2`, tag `v3.2.2`, section [3.2.2] of the code's `CHANGELOG.md`): outside w and n at most 2^26 the
floating-point guards refuse to form a bound, and S2 with the statements that rest on it is INCONCLUSIVE with the
reason code `FP_GUARD_RANGE`; see [05_open_items.md](05_open_items.md).

## Constants that the documents already discuss (L-4, L-10)

Lemma 17 encloses the logarithm by a truncated atanh series and adds a bound for the remainder. Its document
(`code/certify-ed/docs/proofs/L17_dyadic_exp_ln.md`, lines 112 to 125) remarks that the 3.1 code used the constant 9/8,
valid only for arguments up to 1/3, while the runtime check admits arguments up to 0.35, and that the 3.2 code uses
8/7. `CertifyCore/Theorems/L17.lean` confirms the remark with exact numbers: 1/(1 - x^2) is 1.1396 at x = 0.35, above
9/8, and 8/7 suffices. It also settles the question the remark leaves open: at the largest argument that occurs,
ceil(2^192/3) in grid units, the neglected tail is at most 0.00118 grid units, while both versions of the code add one
whole grid unit, so no enclosure computed by 3.1 was wrong. For D-01 part (a) with complex data the Lean model of
floating point gives the constant gamma_{t+3}, smaller than the document's 2 gamma_{t+5}; the document's constant is
the more conservative one and needs no change.

## A missing hypothesis in Proposition 14 (L-5)

The bracketing count of statement S8 says that the number of eigenvalues in the inner interval
(sigma_lo + beta_lo, sigma_hi - beta_hi) is at most nu_hi - nu_lo (`code/certify-ed/docs/theory.md`, section 9.1,
lines 362 to 375). The Lean theorem `bracket_count` of `CertifyMathlib.Counting` needed the hypothesis that the inner
interval is not empty, and the reason is a counterexample: for a 1 x 1 matrix with eigenvalue 0, shifts minus s and s
and backward errors 2s, the counts nu_lo = 1 and nu_hi = 0 satisfy both inertia sandwiches, the inner interval is empty,
and nu_hi - nu_lo = -1 is smaller than 0. The derivation document `P14_degeneracy_bracketing.md` carries the hypothesis
and the code uses the inequality only for a window inside the inner interval, which implies it, so no S8 statement was
affected. Commit `40d59f4` adds the hypothesis to `theory.md`.

## Arguments that hold for a different reason (L-6, L-7)

Part (b) of D-02 says that the Theorem 1 radius computed with `adapter.width = W - 2` bounds the residual of the exact
specification operator, and the document (`code/certify-ed/docs/proofs/D-02_matrix-free.md`, lines 47 to 49 and
106 to 107) explains this by the slack of one gamma index that the real case keeps. Formalizing the step showed that there is
no spare index once the second rounded pass and the rounding factors of the code are counted: the statement holds
through a second-order inequality, `FP.d02b_key`, with a margin of about W u^2, which an exact rational evaluation at
W = 3, 10, 1000, 2^20 and 2^26 also shows. The result stands; its stated reason did not. In the
proof of Theorem U5 (`code/certify-ed/docs/proofs/int05_underflow.md`, lines 160 to 166) the inequality
(1 - u)^k (1 + 8u)^2 at least 1 + 10u for k up to 12 is false, the left side being about 1 + 4u; the proof needs only
that the additive term is at least sqrt(n) c_w (1 + u) u_min, which `addTerm_ge` proves. Both are corrected by errata in
commit `fa86456`.

## Statements that are slightly too strong (L-8, L-9)

The conversion step (v) of Lemma 17 says that `from_float` is exact. It is exact only when f 2^P is an integer; for the
smallest subnormal binary64 number 2^-1074 and P = 192 the result is the interval from 0 to one grid unit
(`from_fraction_tiny`, `two_pow_neg_1074_grid`), which still contains f, so the enclosure property that the lemma needs
is unaffected. The second sentence of Lemma A1 (`code/certify-ed/docs/proofs/appA_model-shift.md`, lines 25 and
67 to 69) gives the factor (1 plus or minus gamma_{m+2L_max-1}) between computed and exact row sums of |H|_abs; for complex
amplitudes `np.abs` of a computed amplitude adds one rounding, and the factor is gamma_{m+2L_max}. The code divides by
(1 - gamma(width)) with width = m + 2L_max + 2 (`certify_ed/certify.py`, lines 389 to 395), which covers the extra index.
Commit `fa86456` adds both errata.

## A difference of form in Theorem 6 (L-11)

The manuscript's Theorem 6 (`thm:dk` in `manuscript_v4/sections/s3_math.tex`) measures the separation from the Ritz
values, while the usual sin Theta theorem measures it from an interval that contains them. Under the hypotheses of the
manuscript the two agree, and the Lean form `thm6_ritz` takes the location of the Ritz values as a hypothesis instead
of formalizing the two-line argument of the manuscript. Nothing needs correcting; the record says which form was
kernel-checked.

## What these findings show

All eleven concern hypotheses, ranges or the reason given for a step, and none concerns a conclusion that the code
relies on. This matches the pattern of the earlier checks: the independent re-derivation of October 2026 also found
gaps in arguments that their verify scripts had passed ([03_review_and_derivation_findings.md](03_review_and_derivation_findings.md)).
Random testing cannot find a guard that fails at one value near 2^53/3, and a second derivation written from the
statement tends to reproduce the statement's range. The findings do not show that the remaining documents are free of
such problems: the statements that were formalized only in part, or not at all, are listed in the report, and
Sarang's check under rule D-12 is still pending for every derivation document.
