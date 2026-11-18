# Decisions D-48 to D-61: the review, release 3.1.0, the roadmap and the 3.2 evidence run

This file explains entries D-48 to D-61 of `project_docs/DECISIONS_LOG.md` (version 1.19), taken between the evening
of 21 October and 6 November 2026. They follow the independent review of 21 October
(`review/REVIEW_CERTIFY-ED_2026-10-21.md`) and lead to the evidence runs of 3.1.0 and 3.2.1. Every number below is
from the decision log unless another file is named.

## D-48 (21 October 2026, evening): the history of the code repository is rewritten before the 3.1.0 run

The review found the private account address of the author as author and committer of all 52 commits, in the
repository configuration, in three nested repository tarballs and in the report, and the same private link to a
working record in a trailer of every commit (findings RA-26 and RA-27). The project's own rule is that this address is
never sent anywhere, and a push, a Zenodo deposit or the sharing of the archive would have published it. Every commit
was therefore rewritten with "Sarang Vehale" and the address used for the papers as author and committer, the trailers
were removed, and commit messages are plain from then on. Every commit hash quoted in older records changed, so
`HASH_MAP.csv` (old hash to new hash) stays in the repository and every quoted hash still resolves. The rewrite was
made before the 3.1.0 run, so that the run's `git_commit` fields point into the new history. Three alternatives were
rejected: publishing without history (it loses the record that the commit appendix and D-44 rest on), keeping the
trailers as provenance (one private link on 52 commits is not provenance), and rewriting after the run (it would
orphan the run's commit fields).

## D-49 (21 October 2026, evening): certify everything a laboratory run produces, and declare the scope

certify-ed accepts by default everything an ED run of a laboratory produces, lets the user narrow the work with
presets, `--only`, `--skip` and time budgets, and declares in every certificate what was in scope, what was narrowed
and what is out of scope, with a reason. The log compares this to tools such as ffmpeg and dd, which accept any input
and report what they did. The trigger was the referee history (the community already tests; the gap was not
demonstrated) and the finding that the paper's claims about practice had no sourced basis. The content of the decision
is a scope matrix (`docs/SCOPE_MATRIX.md`) with one row per quantity laboratories report, a mandatory scope block in
every certificate (a narrowed statement is NOT_CHECKED with the reason, never silently absent, and the schema check
fails on a silently absent statement), a universal producer import (the producer bundle, in which every field except
the eigenvalues and the specification is optional), and new statements: S7 (thermodynamics from a complete
whole-spectrum enclosure), extensions of S4 (correlations and structure factors) and degeneracy counts. Each new
statement gets its proof written against the code before its tests and run (D-46) and is marked `UNVERIFIED` (D-12).
Rejected were keeping the narrower 3.0.0 scope (a referee reads a narrow default as the tool's reach), narrowing
silently by preset (a certificate must never be quieter than the run that produced it), and one importer per producer
(one bundle layout covers every code).

## D-50 (21 October 2026, evening): a pre-registered survey of published ED practice

A survey of published ED practice was pre-registered (`PREREG_practice_survey.md` v0.1) before any paper was fetched.
The design draws arXiv records from five categories between 1 January 2024 and 21 October 2026, up to 600 records,
codes a subsample of 80 papers (seed 20261021) on fixed fields, has the first pass made by an automatic classifier
with a fixed schema, hand-checks 15 % of the papers (seed 20261022) with agreement per field, re-codes fields below 80
% agreement, reports Wilson 95 % intervals, and treats fewer than 40 coded papers as a pilot whose percentages do not
enter the manuscript. The survey decides the default scope and the presets, not which statements certify-ed certifies.
Its population is arXiv preprints of those years and categories, not the whole community, and the manuscript says so.
Rejected were citing the earlier horizons report for claims about practice (unsourced), a survey coded only by hand
(too slow) and an automatic coding without a hand check (the classifier can mislabel, and the check measures how
often).

## D-51 (21 October 2026, evening): the note to Kránitz and Penc is corrected before sending

Version 0.2 of the note to the authors was withdrawn, and version 0.3, prepared during the review, was adopted as the
draft, to be checked by Sarang against the paper before it is sent. Version 0.2 was wrong in two places. The paper
states that c5 is neglected, so an unprinted c5 could not be the cause; and the published −1.632985 is reproduced at
the printed couplings in the two pair-spin sectors the paper names ([0,0,0,2,0,2]: −1.6329867; [0,2,2,0,2,2]:
−1.6329849, in floating point), so "we could not reproduce the bound" was false. The finding is that the all-singlet
sector, at −1.6380246 (certified), lies 5.0e-3 below the published value and was evidently not included in the
published search; a cutting-plane bound shows that no value of c5 raises the max-min above about −1.6342. The
commitment of D-34 is unchanged: nothing is stated publicly before the authors reply, and documents describe the
matter as pending their reply. Sending v0.2, or v0.3 without Sarang's check, was rejected (the note goes out under the
authors' names, D-16).

## D-52 (21 October 2026, evening): Theorem 2 is restated through Kahan's theorem

The radius of the cluster enclosure (Theorem 2) is restated as Kahan's inclusion theorem applied to the polar factor
of the claimed eigenvectors, with radius ρ and a three-line lemma on the polar factor. The radius used up to then, ε =
(w + √(w² + 4ρ²))/2, is never smaller than ρ; Kahan's theorem holds for any Hermitian Λ, and the old remark describing
Kahan's theorem as a special case was wrong. On the run the ratio ε/ρ was between 1.000148 and 1.618 (finding IMP-12).
A second problem was that the term (‖A‖ + ‖Λ‖)δ of ρ made a correct claim fail when its degenerate level was given by
a valid but non-orthogonal basis (IMP-13: radius 4.91 against 4.5e-8). The review proposed a sharper bound ε\*, which
became a separate lemma, marked `UNVERIFIED`, and the code uses min(ρ, ε\*). The review's check of ε\* used 15,876
cases with no violation. Rejected were keeping ε (looser), adopting ε\* alone (ρ is sometimes smaller, with a 5 %
quantile of the ratio of 0.20), and returning FAILED when only the orthonormality defect misses the target (the claim
is not contradicted, so the status is INCONCLUSIVE). If the check of ε\* under D-12 fails, the code falls back to ρ.

## D-53 (21 October 2026, evening): a corrected replacement of the symveig preprint and deprecation of its code

The open choice of D-47 was settled as option B: a corrected replacement v2 of arXiv:2606.16217, together with a
deprecation of the public symveig 1.0.0 code on GitHub and Zenodo in the same week. The review (RA-33) had noted that
the earlier options did not cover the code release, which has the same P1 and P2 behaviour. A withdrawal would
overstate the errors (the residual intervals are valid) and leave no corrected record. The replacement, in Sarang's
prose, adds an erratum, drops the guarantee of exact counts and of the whole spectrum for supplied eigenpairs, states
that the commutation precondition of the sector path is not checked, drops the claim to be the first such tool, and
points to `certify_ed.spectrum`. Withdrawing at once, waiting, and leaving the code release untouched were rejected.

## D-54 (22 October 2026): the Windows workstation is the primary machine

Sarang decided that nothing waits for the Ryzen machine and that the Windows workstation is his primary machine. The
steps the earlier handover had left for the Ryzen (QuSpin installation, full test suite, the `v3.1.0` tag, the 3.1.0
evidence run with mutation analysis, the number scripts, the two-stack comparison and the PDF builds) were done on the
workstation. The library versions of the 3.0.0 run `run_full_20261021` were reproduced in a pinned environment (Python
3.11, numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0, QuSpin 1.0.1, h5py 3.16.0, matplotlib 3.10.9, mpmath 1.4.1).
QuSpin 1.0.1 is installed with `--no-deps`, because its metadata requires the PyPI package `importlib`, a Python 2
backport that does not install on Python 3, and its real dependencies are installed explicitly
(`docs/reproduction/01_environment.md`). The two-stack rule for 3.1.0 is met by the Linux 3.0.0 run against the
Windows 3.1.0 run: statements must agree, while digits that depend on the floating-point stack need not.

## D-55 (22 October 2026): the four v1 figures are not carried into ms_v4

The four figures of v1 (sum-rule errors, finite-size scaling, validator summary, error injection) illustrate the v1
validator design, which certificates replaced. In ms_v4 sum rules are one of the uncertified evidence practices the
survey counts, finite-size extrapolation is declared out of scope, test counts come from run-record macros and the
error-injection bars are replaced by the E1 failure matrix and the E6 mutation tables, which also show what is not
detected. ms_v4 keeps its three generated figures (`fig_tightness`, `fig_widths`, `fig_cost`). Re-plotting the scaling
figure from certified energies was rejected because it would still be out of scope.

## D-56 (22 October 2026): the public repository is extended, not rewritten

The public GitHub repository holds 18 commits of v1, authored by Sarang, with `main` at `630bf60` and tag `v1.0.0` at
`7580950` (the Zenodo v1.0.0 release). The project's rewritten history shares no commit with it, so a force-push to
`main` would orphan the public history and the commit behind the Zenodo release. Instead, `release/3.1.0` and the tags
`v3.1.0-candidate` and `v3.1.0` are pushed as they are, together with a branch `main-3.1.0` whose merge commit has the
public `main` and `v3.1.0` as parents and the tree of `v3.1.0`; Sarang can then fast-forward `main`. No published
commit or tag changes. The push needs a credential with write access.

## D-57 (22 October 2026, night): the 3.1.0 run stands; four integration findings are known issues

After the 3.1.0 evidence run (`run_full_20261022`, tag `v3.1.0` = `e860a83`), filling the manuscript's evidence
pointers showed four places where text and code disagree (INT-01 to INT-04, described in
`docs/failures/02_integration_findings.md`). None changes a statement or status of the run. The paper cites 3.1.0 with
these known issues; INT-02 (no check that the scope block lists every statement) and INT-03 (exported radii rounded to
nearest) are fixed on the branch `fix/3.1.1`, which is neither merged into `release/3.1.0` nor tagged. INT-01 (the
no-underflow hypothesis) and INT-04 (the text of Appendix A) were left to Sarang. Folding the fixes into `v3.1.0` and
rerunning was rejected because the run is the record the paper cites and no fix changes a statement; leaving INT-02
and INT-03 open was rejected because both are small and testable and do not touch the mathematics.

## D-58 (29 October 2026): the expansion roadmap, executed on dev/3.2

Sarang asked on 23 October for at least 50 scored improvements and approved Waves 0 and 1 of the resulting roadmap (78
scored items in five waves). Seven tracks of work delivered 30 items done and 3 partial (173 of 470 points). The
decision has four parts. Nothing in `dev/3.2` adds mathematics: every new record rests on results already stated or is
explicitly marked as not certified, and the items of Waves 2 to 4 that need new mathematics wait. The paper keeps
citing 3.1.0, because 3.2.0.dev0 changes the computational source and would need its own full run. INT-01 is enforced
conservatively by a new reason code `UNDERFLOW_RISK`, and the subnormal intermediate residual entries that routine
LAPACK output produces become a new known issue, INT-05. The public release stays blocked on a GitHub credential.
Merging `dev/3.2` into `release/3.1.0` was rejected (it would change the cited code without a run), and so was running
the 64-mutant E6 sample on the merged branch at that point, because each mutant runs the full 23-minute suite; a
16-mutant sample was run instead.

## D-59 (30 October 2026, recorded 3 November 2026): clearing the D-12 backlog with derivation documents

Sarang asked that the mathematics he had reserved for himself be derived now and checked computationally and
analytically. D-12 stands: a derivation produced within the project is not a second person's check. The regime adopted
has four parts. Each derivation is a document `docs/proofs/<id>_<name>.md` with six fixed sections: statement, proof,
where the code implements it with its rounding directions, a computational check by a script `verify_<id>.py` in exact
rational or high-precision arithmetic with random and adversarial cases, an analytic cross-check, and a status line of
fixed form. A second derivation is then written independently from the statement and the code section alone, without
the proof, and compared; the verdict (AGREES, AGREES-WITH-REMARKS, GAP or ERROR) goes into `docs/proofs/VERDICTS.md`
with its own adversarial script. This second derivation is not a human check. A bound may become tighter only on the
strength of a proof document, while a looser correction may go in at once with a negative-control test. Prose remains
Sarang's. The outcome at 3.2.0 was 22 documents, 0 violations in every executed check, and verdicts of 6 AGREES, 9
AGREES-WITH-REMARKS, 5 GAP (proof text only; no certified bound affected) and 2 ERROR, both corrected by looser code
(`docs/failures/03_review_and_derivation_findings.md`). Rejected were calling any of the results established before
Sarang's check, publishing derivations without the independent second derivation, and keeping the old Theorem 2
constant.

## D-60 (3 November 2026): certify-ed 3.2.0 gets its own full run

Version 3.2.0 was tagged from `dev/3.2` after Phases 1 and 2 of the completion plan and was to get its own full
evidence run with the 64-mutant sample (D-44); 3.1.0 remains the cited version until that run is compared with
`run_full_20261022` and Sarang decides. At the tag the full suite gave 1161 passed and 8 skipped at `e69bca5`, and the
roadmap stood at 43 of 78 items done and 6 partial (270 of 470 points). Citing 3.2.0 without its run and a separate
release 3.1.1 were rejected. The entry would be reopened if the run showed a MISMATCH against 3.1.0 that no documented
bound change explains, or if the E6 sample exposed a soundness-relevant test gap.

## D-61 (6 November 2026): the 3.2 evidence run is the 3.2.1 run

The 3.2.0 run (`run_full_20261104`, tag `v3.2.0` = `55d9f61`, 22 h) completed every step except E4, whose W4 case
"J1-J2 square 4x4, J2 = 0.5" (dimension 12 870) ended with a native access violation under the experiment's 5 GiB
memory cap. The cause was the new a-posteriori S3 residual (D-01 of the roadmap), which formed its row blocks on eight
threads; under a cap on committed memory, an allocation failure inside SciPy's sparse product in a worker thread ends
the process instead of raising `MemoryError`. Version 3.2.1 (tag `v3.2.1` = `f3c478b`) makes one thread the default,
and the 3.2.1 run `run_full_20261105` completed every step (`docs/experiments/02_evidence_runs.md` gives the numbers).
The decision is that the 3.2 evidence run is `run_full_20261105`; the 3.2.0 run is kept as a record; the manuscript
macros of archive v5 are generated from the 3.2.1 run, with the 3.1.0 macros kept beside them so that the cited run is
switched by swapping two directories; and which version the paper cites is Sarang's decision. The tighter E3 bounds
that come from the measured residual are admitted under the soundness rule because they rest on a proof document with
an independent verdict (AGREES-WITH-REMARKS); Sarang's check is pending. Rejected were citing the 3.2.0 run with E4
patched in from a later run (one commit, one run), raising the experiment's 900 s certifier limit for the two E2
points that hit it (it would change the E2 settings and hence the comparison), and keeping eight threads by default
(the failure depends on the memory cap and is silent).
