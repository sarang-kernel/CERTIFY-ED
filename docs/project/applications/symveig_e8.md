# symveig: the audit, the whole-spectrum question and experiment E8

symveig 1.0.0 is an earlier package by Sarang Vehale and Ritu Goel: a pure NumPy and SciPy implementation of eigenvalue
enclosures for all eigenvalues of a Hermitian matrix, with an optional decomposition into the sectors of an abelian
conserved quantity. It is described in the preprint arXiv:2606.16217 v1 (posted on 15 June 2026); a manuscript for
Computer Physics Communications was prepared and never submitted. This document describes the audit of symveig, the
question of what a whole-spectrum enclosure may claim, the replacement of symveig by `certify_ed.spectrum`, and
experiment E8, which compares the two on symveig's own benchmarks. Sources: `code/certify-ed/docs/symveig_audit.md` (the
audit of 18 October 2026), `report/06_symveig.md`, and the E8 records
`code/certify-ed/results/run_full_20261022/e8_symveig_comparison_20261022_231258/data.json` (3.1.0 evidence run) and
`code/certify-ed/results/run_full_20261105/e8_symveig_comparison_20261105_224131/data.json` (3.2.1 evidence run).

## How symveig came into the project

The survey of 15 October 2026 found the preprint without knowing that it was the authors' own; on 17 October Sarang
Vehale confirmed that it was, and decided that one paper should contain both works (decision D-33, "one solid paper >
two mediocre papers"). On 18 October he provided the package, the manuscript sources and the PDF. The package's own 11
tests pass, and the numbers of the paper's Table 1 reproduce (for example the sector half-width 2.77e-11 at L = 12 with
periodic boundaries).

## What symveig does

Each computed eigenpair $(\mu, x)$ gets the residual interval
$\lvert \lambda - \mu \rvert \le \lVert Ax - \mu x \rVert / \lVert x \rVert$ (Theorem 1), with the floating-point error
of the matrix-vector product bounded by $\gamma_{4n} \lVert A \rVert_F \lVert x \rVert$. Overlapping intervals are
merged and reported with a multiplicity. The optional sector path splits the matrix by the diagonal of a conserved
quantity such as $S^z$ and encloses each block.

## The audit

Every finding of `docs/symveig_audit.md` was reproduced, and two of them are tests in `tests/test_spectrum.py` of
CERTIFY-ED.

The first finding (P1) concerns the multiplicities. The paper and the code say that each interval contains exactly as
many true eigenvalues as its multiplicity and that the union contains the full spectrum. That is not proved: a residual
interval contains at least one eigenvalue, not a distinct one, and counting needs a certified bound on the
orthonormality of the vectors, which symveig never computes. With eigenpairs supplied by the user, the advertised use
for validating approximate eigensolvers, the claim is false: for $A = \mathrm{diag}(0, 1)$ with two copies of $e_1$,
symveig returns one interval of multiplicity 2 around 0, and the eigenvalue 1 lies outside every interval
(`tests/test_spectrum.py::test_duplicated_eigenvector_is_not_counted`). With LAPACK's own, numerically orthonormal
vectors no failure was observed; there the guarantee is unproved, not refuted (finding RA-31).

The second finding (P2) concerns the sector path, which is said to verify each sector but never checks that the matrix
conserves the quantity. On the Heisenberg chain of 6 sites with a small staggered transverse field of strength 0.05,
which breaks the conservation of $S^z$, 58 of the 64 true eigenvalues lie outside every interval that symveig labels
`verified`, and the reported ground-state interval [-2.80277564, -2.80277564] excludes the true value -2.86389024
(`tests/test_spectrum.py::test_non_conserved_quantity_rejected`). The abstract of the preprint makes the sector path
conditional on a commuting conserved quantity, so this is an unchecked precondition rather than a false theorem (finding
RA-32); a user who violates it nevertheless receives wrong enclosures with that label.

The remaining findings are smaller. For the global path, the validation of the paper compares with the same LAPACK
result whose eigenvalues are the interval midpoints, so containment is automatic (P3). The claim to be the first such
tool in the Python stack is too strong, because Arb, available through python-flint, computes eigenvalue enclosures in
ball arithmetic, although it fails to isolate the degenerate spectra of Heisenberg chains of 8 and 10 sites even at 256
bits (P4). The reported gain of the sector path, a factor 3 to 9 in width, is real for symveig's bound but comes from
the form $\gamma_{4n} \lVert A \rVert_F \lVert x \rVert$, which grows with the dimension while the true error does not,
and the reported speed-up of 133 is the ordinary gain of block diagonalization, measured against a complex-arithmetic
solver applied to a real matrix (P5). An opt-in second tier based on a lemma of Rump and Lange is not the lemma as
stated and assumes orthonormality instead of checking it (P6). There is no check of the floating-point environment (P7),
and the matrix is made Hermitian silently (P8).

## The whole-spectrum question

The audit raises a question that any whole-spectrum tool must answer: under which conditions may one say that a list of
intervals contains every eigenvalue, and that each interval contains exactly a stated number of them? The answer used by
CERTIFY-ED is Proposition 9 (`docs/project/math/theorems.md`). It needs four things. First, if sectors are used, the block
structure must be checked exactly: no stored entry may couple two different values of the conserved quantity (the check
that P2 shows to be necessary). Second, each interval must carry a number of distinct eigenvalue indices, which for a
single pair is Theorem 1 and for a cluster needs Theorem 2, and therefore a computed orthonormality defect $\eta < 1$
(the step missing in P1). Third, the intervals of one block must be pairwise disjoint and rounded outward. Fourth, the
counts must add up to the dimension. Then, by a pigeonhole argument, every eigenvalue lies in some interval and each
interval holds exactly its count. Without the cluster theorem a merged interval holds at least one eigenvalue and no
more can be said. The thermodynamic statement S7 needs exactly this output: it is computed only when the enclosure is
complete.

The replacement is `certify_ed.spectrum.enclose_spectrum(H, S_diag=None, eigh_result=None)` (decision D-42, 18 October
2026). Its counts are certified by the cluster theorem when $\eta < 1$ and are otherwise reported as existence only; the
enclosure is marked complete only if every count is certified and the counts sum to n; the sector structure is checked
exactly, with an error otherwise; exact Hermiticity is required and nothing is symmetrised silently; the rounding bound
is the $\lvert A \rvert \lvert x \rvert$ bound with row-width constants of
[arithmetic_model.md](../math/arithmetic_model.md), and the interval ends are rounded outward since 21 October 2026.
`certify_ed.compat.symveig` keeps the names of the 1.0.0 interface on the new engine, and symveig's own tests are
ported. Proposition 9 is marked UNVERIFIED and has no derivation document.

## Experiment E8: the same matrices, two bounds

**Why.** P5 says that the gain of the sector path in symveig is mostly an artefact of the form of its bound. If that is
right, a bound of the form $\lvert A \rvert \lvert x \rvert$ with row-width constants should be much tighter on the same
sectors, and its widths should grow more slowly with the dimension. E8 also checks whether CERTIFY-ED's enclosures are
complete in the sense of Proposition 9 on the matrices for which symveig reports multiplicities, which is the use case
where the counts need the cluster theorem.

**What was done.** symveig 1.0.0, unchanged in `legacy/symveig_1_0_0/` of the code repository, and `certify_ed.spectrum`
enclose the whole spectrum of the same matrices: symveig's 17 benchmark configurations, the Heisenberg chain with open
and periodic boundaries and the J1-J2 chain with $J_2 = 0.5$ for L = 4 to 12, and the Heisenberg model on periodic 3 x 3
and 4 x 3 lattices, all through the $S^z$ sectors (`experiments/e8_symveig_comparison.py`).

**How the instrument was checked.** Both tools' intervals are compared with reference eigenvalues; in every row of both
runs the reference lies inside the intervals ("reference outside by 0" in the logs), and every CERTIFY-ED enclosure is
complete. The comparison uses the same matrices and the same eigensolver output, so the difference in width is the
difference between the two bounds and the cost of the count guarantee.

**Results.** Largest and median half-widths of the intervals, from `code/certify-ed/results/run_full_20261105/E8.log`
(3.2.1) and `code/certify-ed/results/run_full_20261022/E8.log` (3.1.0); symveig's values are identical in both runs.

| model | L | dimension | symveig largest | certify-ed largest (3.2.1) | certify-ed largest (3.1.0) | symveig median | certify-ed median (3.2.1) | certify-ed median (3.1.0) | complete (3.2.1 / 3.1.0) |
|---|---|---|---|---|---|---|---|---|---|
| 1D Heisenberg OBC | 4 | 16 | 9.33e-15 | 1.24e-14 | 1.24e-14 | 7.7e-15 | 5.6e-15 | 5.6e-15 | True / True |
| 1D Heisenberg OBC | 6 | 64 | 6.22e-14 | 3.38e-14 | 3.38e-14 | 6.1e-14 | 1.5e-14 | 1.5e-14 | True / True |
| 1D Heisenberg OBC | 8 | 256 | 4.54e-13 | 1.19e-13 | 1.19e-13 | 3.8e-13 | 5.2e-14 | 5.2e-14 | True / True |
| 1D Heisenberg OBC | 10 | 1024 | 3.44e-12 | 4.98e-13 | 4.98e-13 | 3.0e-12 | 2.0e-13 | 2.0e-13 | True / True |
| 1D Heisenberg OBC | 12 | 4096 | 2.64e-11 | 2.13e-12 | 2.13e-12 | 2.3e-11 | 8.1e-13 | 8.1e-13 | True / True |
| 1D Heisenberg PBC | 4 | 16 | 1.15e-14 | 1.51e-14 | 1.51e-14 | 1.0e-14 | 9.8e-15 | 1.0e-14 | True / True |
| 1D Heisenberg PBC | 6 | 64 | 6.93e-14 | 4.02e-14 | 5.97e-14 | 5.5e-14 | 2.7e-14 | 3.5e-14 | True / True |
| 1D Heisenberg PBC | 8 | 256 | 4.88e-13 | 1.40e-13 | 2.23e-13 | 4.1e-13 | 8.2e-14 | 1.1e-13 | True / True |
| 1D Heisenberg PBC | 10 | 1024 | 3.64e-12 | 6.24e-13 | 1.01e-12 | 3.1e-12 | 3.1e-13 | 4.8e-13 | True / True |
| 1D Heisenberg PBC | 12 | 4096 | 2.77e-11 | 2.73e-12 | 4.42e-12 | 2.4e-11 | 1.2e-12 | 1.9e-12 | True / True |
| 1D J1-J2 PBC | 4 | 16 | 1.20e-14 | 2.26e-14 | 3.38e-14 | 1.2e-14 | 1.7e-14 | 2.5e-14 | True / True |
| 1D J1-J2 PBC | 6 | 64 | 7.28e-14 | 5.33e-14 | 8.44e-14 | 5.7e-14 | 3.6e-14 | 5.4e-14 | True / True |
| 1D J1-J2 PBC | 8 | 256 | 5.32e-13 | 2.33e-13 | 3.75e-13 | 4.4e-13 | 1.0e-13 | 1.5e-13 | True / True |
| 1D J1-J2 PBC | 10 | 1024 | 3.99e-12 | 6.67e-13 | 1.06e-12 | 3.4e-12 | 3.9e-13 | 5.9e-13 | True / True |
| 1D J1-J2 PBC | 12 | 4096 | 3.05e-11 | 3.60e-12 | 5.82e-12 | 2.7e-11 | 1.5e-12 | 2.4e-12 | True / True |
| 2D Heisenberg 3x3 PBC | 9 | 512 | 1.67e-12 | 7.44e-13 | 1.20e-12 | 1.7e-12 | 3.7e-13 | 5.9e-13 | True / True |
| 2D Heisenberg 4x3 PBC | 12 | 4096 | 3.84e-11 | 5.86e-12 | 9.28e-12 | 3.4e-11 | 1.9e-12 | 3.0e-12 | True / True |

At dimension 4096 (L = 12) the median half-widths of CERTIFY-ED are 17 to 29 times smaller than symveig's in the 3.2.1
run and 11 to 29 times smaller in the 3.1.0 run, and the largest half-widths are smaller as well. At the smallest
dimensions the largest half-width of CERTIFY-ED can be larger than symveig's (for L = 4 the ratio is 0.8, 0.8 and 0.5 in
the 3.2.1 log), because a merged cluster interval carries the cost of the count guarantee. Fitted on the dimensions from
64 upward, the median half-widths grow about as $d^{1.48}$ for symveig and $d^{0.95}$ for CERTIFY-ED in the 3.2.1 run
($d^{0.98}$ in the 3.1.0 run), and the largest half-widths as $d^{1.47}$ and $d^{1.05}$ ($d^{1.06}$). At dimension 4096
CERTIFY-ED took 1.0 to 2.0 times symveig's time in the 3.2.1 run and 0.9 to 3.0 times in the 3.1.0 run, for the
$\lvert A \rvert \lvert x \rvert$ products and the Gram matrices of the clusters. The ratios, exponents and time ratios
are the macros `EeightMedianRatioMin`, `EeightMedianRatioMax`, `EeightSlopeSymMedian`, `EeightSlopeOursMedian`,
`EeightSlopeSymMax`, `EeightSlopeOursMax`, `EeightSlowdownMin` and `EeightSlowdownMax` of
`manuscript_v4/generated/numbers_apps.tex` (3.2.1) and `manuscript_v4/generated_3.1.0/numbers_apps.tex` (3.1.0).

**Why the two runs differ.** The two evidence runs used the same machine and library stack, yet the CERTIFY-ED widths of
the models with degenerate levels (periodic chains and the two-dimensional lattices) are smaller in the 3.2.1 run, while
the open chain, whose levels in these sectors are not degenerate, is unchanged. The widths of degenerate levels are
cluster radii, and between the two versions the Theorem 2 term of the cluster radius changed from
$(w + \sqrt{w^2 + 4\rho^2})/2$ to $\rho$, which is never larger (`docs/project/math/derivation_documents.md`, entry
`thm2_polar-factor.md`). The comparison tool `experiments/compare_runs.py` files these leaves under "stack-dependent"
(`code/certify-ed/results/run_full_20261105/compare_vs_run_full_20261022.csv`); no status changed.

**What it shows and does not show.** E8 shows that sector decomposition shrinks rounding bounds that depend on the
dimension, and that the form of the bound decides how the widths scale: with the $\lvert A \rvert \lvert x \rvert$ bound
what remains is mostly the eigensolver's own residual, which still grows with the dimension, so the widths should not be
described as flat. It does not show that symveig's intervals are wrong on these matrices; with LAPACK's own vectors they
contained the reference eigenvalues. The finding about symveig is that its counts are not proved and its sector path
does not check its precondition, not that its intervals fail on conserving models.

The figure below is from symveig's own package (`symveig/code/symveig-1.0.0/results/figures/fig_width_vs_dim_norm.png`);
it shows the half-widths of symveig following the line $6 u\, d \lVert H_s \rVert_F$, the dimension-dependent form that
P5 refers to.

![symveig half-widths against the product of block dimension and Frobenius
norm](figures/fig_symveig_width_vs_dim_norm.png)

## What was decided about the preprint

The preprint is public and states P1, and it contains the unchecked sector path. On 21 October 2026 Sarang Vehale said
that symveig could be dropped because it had not been sent to the journal (decision D-47). The question was then what to
do with the public version 1, which arXiv cannot delete. Of the three options considered (withdraw now with a reason;
replace now with a corrected version 2; wait and withdraw when the combined paper is posted), decision D-53 chose the
replacement: a short version 2 with an erratum on page 1 and a corrected abstract that drops the exact-count and
full-spectrum guarantee for supplied eigenpairs, states that the commutation precondition of the sector path is not
checked, drops the claim to be the first such tool, and points to `certify_ed.spectrum`, together with a deprecation
notice on the public symveig code and a new Zenodo version carrying the same notice (`report/06_symveig.md`). A
withdrawal was judged to overstate the errors, because the residual intervals themselves are valid. The text of the
erratum is Sarang Vehale's to write (decision D-16), and on 6 November 2026 the replacement was still among his open
items (`project_docs/SESSION_HANDOVER_2026-11-06.md`).

## Reproducing

From the root of the code clone: `python run.py --only E8`. E8 took 284 s in the 3.2.1 run
(`code/certify-ed/RESULT_evidence_run_3.2.1.md`). The audit's two counterexamples are the tests named above.
