# The 22 derivation documents

Version 3.2 of CERTIFY-ED contains 22 derivation documents in `code/certify-ed/docs/proofs/`. Each derives one result on
which a certified statement rests, maps it to the code, and reports a machine check; each was then derived a second
time, independently, and the two derivations were compared. This document describes the regime, then each document in
turn, then the code changes that followed. The verdicts and their remarks are copied from
`code/certify-ed/docs/proofs/VERDICTS.md`, the case counts from the status line of each document (its section 6) and
from the run outputs in `code/certify-ed/docs/proofs/runs/`, and the account of the corrections from
`report/15_completion_plan.md`.

## Why the documents exist and how they were checked

Rule D-12 (13 July 2026) reserves new mathematics for a re-derivation by Sarang Vehale: a result counts as established
only after he has derived it himself. By the end of October 2026 that list had grown with the code: the cluster radius
of Theorem 2 had been restated (decision D-52), the floating-point accounting of Theorem 1 had been found to differ from
the manuscript (INT-04), the underflow question INT-05 was open, Propositions 14 to 16 and Lemma 17 had been written in
3.1 before their code, and the roadmap for 3.2 added new statements. On 30 October 2026 Sarang Vehale asked that these
derivations be written now and checked computationally and analytically. Decision D-59 (recorded on 3 November 2026 in
`project_docs/DECISIONS_LOG.md`) set the regime, and it keeps D-12 in force: a derivation written in the project is not
a second person's check, so every document carries "Sarang's check (D-12): pending".

Each document has six fixed sections: the statement; the proof; where the code implements it, with the rounding
direction of every computed quantity; a computational verification; an analytic cross-check (special cases, limits, or
comparison with a textbook result); and a status line. The test `tests/test_proofs_props.py` pins the wording of the
status lines. The computational verification is a script `docs/proofs/verify_<id>.py` that evaluates every inequality of
the statement on random cases and on adversarial cases (inputs built to make the inequality as tight as possible:
degenerate levels, tiny and huge magnitudes, subnormal numbers, near-singular matrices), against a ground truth in exact
rational arithmetic (`fractions.Fraction`) or in mpmath at 30 to 200 decimal digits. Such a script can find a false
constant or a missing term, which would make the plan change; it cannot prove a statement, because it tests finitely
many cases. That is why each document was also derived a second time.

The second derivation followed a fixed protocol, recorded in the protocol column of `VERDICTS.md` for every row: read
only section 1 (the statement) and section 3 (the code mapping), write an own proof, then compare section 2 of the
original step by step, run the original verification script at a reduced size, and run an independent adversarial search
`docs/proofs/independent/adversarial_<id>.py` with its own ground truth (exact arithmetic or mpmath at 60 digits). The
own proofs are in `docs/proofs/independent/<id>.md`. The second derivation was produced automatically, without the first
proof in view; it is not a human check. Its verdict is one of four classes: AGREES (the proof is valid as written),
AGREES-WITH-REMARKS (valid, with remarks that do not affect the statement), GAP (a step of the proof is not justified as
written, while the statement or the implemented bound stands) and ERROR (the statement or the code is wrong in some
case).

One rule governed what followed from a verdict (D-59, and `CONTRIBUTING.md` of the code): a bound may become tighter
only on the strength of a proof document, and a looser correction may be implemented at once, together with a
negative-control test that fails on the old code. The totals are 6 AGREES, 9 AGREES-WITH-REMARKS, 5 GAP and 2 ERROR,
with 0 violations in every executed verification run. The second derivations were made in two parts: 17 documents before
the integration of 3 November 2026 (5 AGREES, 7 AGREES-WITH-REMARKS, 3 GAP, 2 ERROR;
`code/certify-ed/RESULT_integration_3.2.md`) and the five later documents E-05, E-03, C-07, D-01 and D-02 afterwards.
`VERDICTS.md` prints the date 30 October 2026 on every row.

Each entry below has four parts: what the document derives, its machine check (the script, the arguments of the recorded
run where the document gives them, and the counts of its status line), the independent verdict with its remarks, and the
outcome for the code. The status of every document is the same: derived, machine-checked, independently re-derived (a
second derivation written from the statement alone, not a human check), Sarang's check pending.

## Theorem 1 and the arithmetic

### `thm1_fp-accounting.md`: Theorem 1 in floating point

**Derives.** Theorem 1-fl: the value `rel` returned by `enclosure.residuals` is at least the exact Theorem 1 radius
$\lVert Ax - \mu x \rVert / \lVert x \rVert$ of the stored numbers, through the stages (S2) `absHX_up`
$\ge \lvert A \rvert \lvert x \rvert$, (S3) $E_i \ge \lvert R_i - (Ax - \mu x)_i \rvert$, (S4) the norm guards, and (S5)
the radius. The six-step chain is summarised in [arithmetic_model.md](arithmetic_model.md). This document is the proof
that the INT-04 known issue asked for.

**Machine check.** `verify_thm1_fp-accounting.py --cases 20000 --seed 6`, every stored number as an exact rational and
the norms bracketed by decimal square roots with directed rounding: 20 000 random and 246 adversarial cases, with 40 000
and 480 stage checks and 20 000 and 12 grid points of the scalar lemmas; 0 violations
(`docs/proofs/runs/thm1_fp-accounting.txt`).

**Verdict: GAP.** The main statement and stages (S3) to (S5) agree. Stage (S2) is not justified for complex data: the
moduli $\lvert a_{ij} \rvert$ and $\lvert x_j \rvert$ are rounded once more, so the argument gives only
$\text{absHX\_up}_i \ge a_i(1 - 8wu^2)$. The certified radius is not affected, because the factor 2 in the complex
constant of stage (S3) absorbs the shortfall. The independent adversarial search (600 cases, seed 19) included 150
complex cases tuned to stage (S2) and found no violation, which is consistent with a gap in the argument rather than a
failing bound.

**Outcome.** A looser code change: `enclosure._rounding_bound` uses $\gamma_{w+3}$ instead of $\gamma_{w+1}$ for complex
data, with the test `test_rounding_bound_uses_width_plus_three_for_complex_data` (3 November 2026). The proof text is to
be restated by Sarang Vehale.

### `int05_underflow.md`: the Theorem 1 radius for inputs of any magnitude

**Derives.** The underflow-tolerant model (M-U) from the binary64 format, the dot-product bound U1, the error-free inner
product U2, the norm bounds U3 with the term $T_N$, the residual entry U4, Theorem U5 (the radius bounds the Theorem 1
radius for any magnitude), Theorem U6 (radii never shrink relative to 3.2.0.dev0 and are bit-identical above about
$10^{-145}$), and Corollary U7 (the additive terms of the Gram and cluster bounds). Details are in
[arithmetic_model.md](arithmetic_model.md).

**Machine check.** `verify_int05_underflow.py --cases 20000 --seed 7`, exact rationals for every binary64 number
including subnormals: 2 500 residual, 2 500 monotonicity, 5 000 dot-product, 5 000 `exact_dot`, 5 000 norm cases and 18
992 model operations at random, and 373 adversarial cases; 0 violations (`docs/proofs/runs/int05_underflow.txt`).

**Verdict: ERROR.** Theorem U5 is false as stated when the squared norm of x overflows: `fp.norm2_lower` returned
infinity and the radius became 2 ULP_MIN. The exact counterexample $A = \mathrm{diag}(1, 2)$, $\mu = 1 + 10^{-8}$,
$x = (10^{160}, 0)$ has true radius $10^{-8}$, while the code returned about $10^{-323}$ and marked the pair distinct.
The fault predated the INT-05 work. The verdict also records a GAP in Lemma U4 for complex data, covered by the factor 2
of the complex constant. The independent search (seed 23) found 11 violations of U5 before the correction, all in the
overflow family, and 0 after.

**Outcome.** A looser code change: `fp.norm2_lower` returns 0 when the computed norm is not finite, so that the radius
is infinite; two negative-control tests in `tests/test_independent_check.py` fail on the old code and pass on the new
one. The statement of U5 is to be given the hypothesis that the sum of squares does not overflow, or Lemma U3 the clause
for that case.

### `appA_model-shift.md`: the model shift

**Derives.** Lemmas A1 to A3 (the entrywise builder error, the symmetrisation, the norm bound
$\Delta_{\rm builder} = \gamma_W \sqrt{rc}$), Theorem A4
($\Delta = \max(\lVert D \rVert_1, \lVert D \rVert_\infty) + \Delta_{\rm builder}$) and Theorem A5 (the Weyl transfer to
the declared model), stated in [theorems.md](theorems.md).

**Machine check.** `verify_appA_model-shift.py --cases 400 --seed 5`, run on 3 November 2026 with Sarang Vehale's
approval: 386 random and 681 adversarial cases (spin-1 and boson chains with non-dyadic couplings, strings of length 1
to 4, products of $S^x$ and $S^y$), among them 139 negative controls that must fail; 0 violations
(`docs/proofs/runs/appA_model-shift.txt`).

**Verdict: AGREES-WITH-REMARKS.** The proof is valid with the same constants. Lemma A2 is stated in the form with
$\max((\lvert H \rvert_{\rm abs})_{ij}, (\lvert H \rvert_{\rm abs})_{ji})$, which does not imply the 2-norm bound for an
asymmetric $\lvert H \rvert_{\rm abs}$ (counterexample with rows (1, 1) and (0, 0)); the proof of Lemma A3 correctly
uses the symmetric average, and Lemma A2 should state it. Two classical facts (the 2-norm is monotone in the entrywise
absolute values, and the bound on $\lvert \hat H \rvert$) are used without being named, and the rounding budget has one
spare unit of W, not two. The independent search (16 cases with seed 17 and 24 with seed 18, about 77 s per case for the
mpmath eigendecompositions of $81 \times 81$ spin-1 matrices) found 0 violations.

**Outcome.** The derivation itself made three bounds of the code looser: the factor $1/(1 - \gamma_W)$ in
`certify._builder_shift`, the adapter width $m + 2L_{\max} + 2$ for spec operators, and the explicit symmetrisation
`construct.hermitian_part` of the built reference. No change after the verdict.

### `L17_dyadic_exp_ln.md`: Lemma 17

**Derives.** The enclosure property of the dyadic interval arithmetic on the grid $2^{-192}$, the enclosures of exp
(halving, Taylor polynomial of degree 40, squarings) and ln (atanh series of 60 terms with $u \le 0.35$, and $\ln 2$),
and the outward conversions to binary64.

**Machine check.** `verify_L17.py --random 2000 --seed 0`: every interval operation against `fractions.Fraction`, exp
and ln at random and adversarial arguments, and the conversions checked to be the tightest floats on the safe side; 6000
random and 140 adversarial cases, 0 violations.

**Verdict: AGREES.** The proofs agree step by step (the nested-ceiling identity, the tail bounds with the constants 2
and 8/7, $\ln 2 = 2\,\mathrm{atanh}(1/3)$, the nearest-float argument). The independent search (300 cases, seed 29)
found 0 violations.

**Outcome.** The derivation itself found that the remainder constant written in 3.1 (9/8 for $u \le 1/3$) does not cover
the rounded upper argument, which can exceed 1/3 by one grid unit; the code uses 8/7 for $u \le 0.35$, a looser bound.
No change after the verdict.

## Clusters, completeness and sectors

### `thm2_polar-factor.md`: Theorem 2 and the polar-factor lemma

**Derives.** The polar-factor lemma (a) to (d), Kahan's inclusion theorem for any Hermitian block, and the Theorem 2
radius $\rho = \lVert R \rVert + (\lVert A \rVert + \lVert \Lambda \rVert)\delta$.

**Machine check.** `verify_thm2_polar-factor.py --cases 20000 --seed 1`, mpmath at 30 digits on the binary64 inputs
taken exactly, with a slack of $10^{-24}$: 17 224 random and 600 adversarial cases, with 34 448 and 900 lemma checks and
5 000 and 600 checks of Kahan's theorem on random Hermitian blocks; 0 violations
(`docs/proofs/runs/thm2_polar-factor.txt`).

**Verdict: AGREES-WITH-REMARKS.** The proof is valid with the same constant, and the code mapping holds. The remark
concerns a neighbouring function: `sectors.audit_orthogonality` has no rounding term for the computed product
$V_a^H V_b$, which feeds the $\eta$ of `sectors.decomposition_shift` (filed under `thm8_rho.md`). The independent search
(500 cases, seed 3, with 129 negative controls) found 0 violations.

**Outcome.** This document is the justification of the one tightening of the cluster radius: the code uses
$\min(\rho, \varepsilon^*)$ in place of $(w + \sqrt{w^2 + 4\rho^2})/2$. In the 3.2.1 evidence run the radii of the
experiments E3 and E4 that moved with this change shrank by less than $10^{-9}$ relative
(`code/certify-ed/RESULT_evidence_run_3.2.1.md`), while the whole-spectrum half-widths of degenerate levels in
experiment E8 became visibly smaller (for example 2.73e-12 against 4.42e-12 for the periodic Heisenberg chain at L = 12,
`docs/applications/symveig_e8.md`); no status changed.

### `thm2prime_eps-star.md`: Theorem 2'

**Derives.** The distinct matching within
$\varepsilon^* = \lVert R \rVert/\sqrt{1-\eta} + \lVert X^H R \rVert/(1-\eta) + \delta\,\mathrm{spread}(\Lambda)/\sqrt{1-\eta}$.

**Machine check.** `verify_thm2prime_eps-star.py --cases 20000 --seed 2`, the case generator of the Theorem 2 script
plus families with a large spread and with $\Lambda$ the diagonal of $X^H A X$: 17 094 random and 621 adversarial cases
(51 282 and 1 863 inequality checks), 0 violations (`docs/proofs/runs/thm2prime_eps-star.txt`).

**Verdict: AGREES-WITH-REMARKS.** The proof has the same structure and constant. The remark: the code-mapping row
"$E_{\rm norm}[i] \ge \lVert Rc_i - r_i \rVert$" is justified under the standard model only; under INT-05 the missing
term is covered by `norm_underflow_term(g)` inside `norm2_upper` (a ratio of at most 1.4e-153 for $g \le 64$,
$n \le 10^9$, $w \le 10^4$), and the table should say so. The independent search (400 cases, seed 5, with subnormal
operator and vector cases) found 0 violations.

**Outcome.** No code change.

### `cor5_pairing.md`: Corollary 5 and the sorted-pairing lemma

**Derives.** Completeness of a `lowest` claim from the inertia count, the sorted-pairing lemma within groups of uniform
radius, and the effect of the model shift $\Delta$.

**Machine check.** `verify_cor5_pairing.py --cases 20000 --seed 3`: the sorted-pairing lemma in exact rationals on 20
000 random and 500 adversarial cases, Corollary 5 against the mpmath eigenvalues of $A + E$ on 347 random cases (53
skipped because the hypothesis was not met), and 400 adversarial negative controls that must not be certified; together
20 347 random and 900 adversarial cases, 0 violations (`docs/proofs/runs/cor5_pairing.txt`). The run also checks that
the instrument can fail: with non-uniform radii the sorted pairing breaks in 238 of 2000 cases, as the lemma predicts,
and when the perturbation $\lVert E \rVert = 10^{-3}$ exceeds a declared $\Delta = 10^{-12}$ the conclusion fails in 100
of 100 certified cases, which shows that the model shift is needed.

**Verdict: AGREES.** The lemma was proved by a different, counting argument, and one step of the corollary was made
explicit. The independent search (300 cases, seed 11) also tested whether each hypothesis is necessary: removing
hypothesis H1a, H1c or H3 produced failures in 41, 277 and 42 of 300 cases.

**Outcome.** No code change.

### `thm8_rho.md`: Theorem 8 with the radius $\rho$

**Derives.** The sector assembly: pooled sector claims match $\lambda_1, \dots, \lambda_K$ of the declared model within
$r + \varepsilon$, $\varepsilon = \rho + m$, and $\lambda_{K+1} \ge E_{\rm cut} - \max \beta - \varepsilon$.

**Machine check.** `verify_thm8_rho.py --cases 240 --seed 4`: Heisenberg chains of 4 and 6 spins 1/2 decomposed into
magnetization and momentum blocks, `compose.certify_low_energy` at several cuts, each certified conclusion compared with
the eigenvalues of the exact spec matrix at 60 digits; 240 random and 49 adversarial cases (31 skipped), with 160 and 32
inequality checks of the proof steps and 5 negative controls that must not be certified; 0 violations
(`docs/proofs/runs/thm8_rho.txt`).

**Verdict: AGREES.** Same proof (Corollary 5 per sector, the initial segment, sorted pairing, the unitary polar factor
and Weyl's inequality twice). Remark: `sectors.audit_orthogonality` omits the rounding term of the computed product
$V_a^H V_b$; it feeds only `sectors.decomposition_shift`, which no certified path calls (tests only). The independent
search (150 cases, seed 13, 18 negative controls) found 0 violations.

**Outcome.** No code change; the remark on `audit_orthogonality` is open.

### `P14_degeneracy_bracketing.md`: Proposition 14

**Derives.** The bracketing inequalities
$\#\{\lambda \in I_{\rm in}\} \le \nu_{\rm hi} - \nu_{\rm lo} \le \#\{\lambda \in I_{\rm out}\}$ and the conclusion
"exactly g eigenvalues in $I_{\rm in}$" of statement S8, items (a) to (e).

**Machine check.** `verify_P14.py --random 3000 --seed 0`: 3000 random rational spectra with clusters of up to 4
eigenvalues and counts produced by an adversary that moves every eigenvalue within the backward error, in exact
arithmetic, plus the code on certified cases; 3060 random and 1006 adversarial cases, 0 violations.

**Verdict: AGREES.** Identical proof. Remark: when one cluster contains several Theorem 2 groups, distinctness across
the groups rests on the disjoint union intervals of `enclosure.enclose`, which the document uses implicitly. The
independent search (300 cases, seed 37, 77 negative controls) found 0 violations.

**Outcome.** No code change.

## Observables and thermodynamics

### `P15_summed_bounds.md`: Proposition 15

**Derives.** The bound $B$ on $\lvert \sum_k a_k \mathrm{tr}(P O_k) - \mathrm{fl}(\sum_k a_k v_k) \rvert$ with directed
rounding: Lemma A (`modulus_upper` is at least $\lvert z \rvert$) and Lemma B (the rounding of the complex inner
product).

**Machine check.** `verify_P15.py --random 500 --seed 0`: the bound end to end in exact arithmetic with up to 1000
coefficients, moduli chosen so that the adversary can saturate every term exactly, and components near $2^{-520}$ for
the small-number branch of Lemma A; 1111 random and 32 adversarial cases, 0 violations.

**Verdict: AGREES.** Identical proof; the line numbers of section 3 are off by one on the current file; the two
hypotheses D1 and D2 are correctly labelled as hypotheses. The independent search (120 cases, seed 41) found 0
violations.

**Outcome.** The derivation itself replaced the moduli from the C library's `hypot` by an upper bound from the IEEE
square root and added directed rounding and the constant $\gamma_{m+2}$ in `correlations.combine_certified` (looser). No
change after the verdict.

### `P16_thermo_intervals.md`: Proposition 16

**Derives.** That F, U, Var(E), C, ln Z and Z of a complete whole-spectrum enclosure lie in the dyadic interval
expressions computed by `thermo.py`.

**Machine check.** `verify_P16.py --random 400 --seed 0`: 400 random enclosures of 1 to 6 intervals at two temperatures
each, with several spectra consistent with each enclosure (at the lower ends, at the upper ends, at random), compared
with exact values; 4000 random and 95 adversarial cases, 0 violations.

**Verdict: AGREES-WITH-REMARKS.** Same proof. Section 1 omits the case in which the lower end of $\tilde Z$ is 0 on the
grid (the code raises an error and S7 is INCONCLUSIVE), which section 2 treats; for $T > 2^{192}$, $\beta$ is below the
grid and wide energy ranges give no value; the line numbers of section 3 belong to an earlier, 112-line revision of
`thermo.py` (the current file has 221 lines). The independent search (200 cases, seed 31, 80 negative controls, 100
cases without a value) found 0 violations.

**Outcome.** No code change.

### `C06_truncated_thermodynamics.md`: Proposition 16' (C-06)

**Derives.** The envelopes of Lemma C for the omitted weight and moments, and two-sided intervals for Z, F, U, Var(E)
and C from the k lowest eigenvalues, the bound $L \le \lambda_{k+1}$ and optionally $M \ge \lambda_n$ (stated in
[theorems.md](theorems.md)).

**Machine check.** `verify_C06.py --random 300 --seed 0`: dyadic spectra of 3 to 13 eigenvalues, so that the floats
passed as interval ends, L and M are exactly the rationals of the hypothesis, at six temperatures across the range of
$T/\ell$; 300 random and 150 adversarial cases, 0 violations.

**Verdict: AGREES-WITH-REMARKS.** Same proof. One side remark in part (iii) is wrong: the lower end of $\tilde Z$ is not
always positive, because the weight of the ground interval can be 0 on the grid; the code then raises an error and S7 is
INCONCLUSIVE, so no bound is affected. The independent search (200 cases, seed 43, with 30 negative controls) found 0
violations.

**Outcome.** No code change.

### `C16_exact_bracket.md`: exact count and multiplicities in a bracket (C-16)

**Derives.** Under the hypotheses that the spec's couplings are the binary64 numbers it contains, taken as exact
rationals, that python-flint's integer and rational arithmetic is exact, and that the bracket [a, b] has rational ends:
with p the characteristic polynomial of $\mathrm{den} \cdot K$ (for complex K, of its real embedding, where every root
appears twice), its squarefree factorization $p = c \prod_j s_j^{\,j}$ (Yun's algorithm), and the Sturm counts
$d_j = V_j(a') - V_j(b')$ at the scaled ends (moved outward by $2^{-200}$ while an end is a root, at most 64 times), the
number of distinct eigenvalues in [a, b] is $\sum_j d_j$, the number counted with multiplicity is
$\sum_j (j/\mathrm{factor})\, d_j$, and exactly $d_j$ of the distinct eigenvalues have multiplicity $j/\mathrm{factor}$.
The decision rules follow: one eigenvalue of multiplicity g means "exactly g eigenvalues in [a, b], all equal"; g
eigenvalues with more than one distinct value means "not all equal"; any other count contradicts Proposition 14 and the
statement is FAILED.

**Machine check.** `verify_C16.py --random 200 --seed 0`: integer matrices with planted rational eigenvalues of
multiplicity 1 to 3 (built with random unimodular similarity transforms, dimension 2 to 12), known polynomials, the
sector identity and model Hamiltonians; 1501 random and 42 adversarial cases, 0 violations.

**Verdict: AGREES-WITH-REMARKS.** The proof is identical and valid. The remark is about the code: the certified sentence
printed the outward-rounded binary64 ends of the counted rational bracket, so an eigenvalue within one unit in the last
place outside the counted bracket lay inside the printed interval and made the sentence false. The exact example has
eigenvalues $1 + 2^{-60}$ and 3/2 and the bracket $[1 + 2^{-52}, 2]$, and the sentence read "exactly 1 in [1, 2]". The
independent search (200 cases, seed 47, with roots $2^{-60}$ apart and bracket ends on roots) found 0 violations of the
counts.

**Outcome.** A wording correction: the sentence now names the exact rational bracket (`bracket_exact`) and gives the
binary64 bracket only as an approximation, with a negative-control test in `tests/test_independent_check.py`. No count,
bracket or bound changed.

### `C-03_entanglement.md`: entanglement entropy and spectrum (C-03)

**Derives.** For a certified cluster and a declared cut A|B, with $\rho_A = \mathrm{tr}_B(P)/g$ and the stored-vector
estimate $\hat\rho_A$: the trace distance $T = \lVert \rho_A - \hat\rho_A \rVert_1/2 \le T^* = s + \eta/(1 - \eta)$
(equal to s for g = 1), from the rank-2g trace-norm bound on $P - \hat Q \hat Q^H$ and the contractivity of the partial
trace; $\lvert S(\rho_A) - S(\hat\rho_A) \rvert \le T^* \ln(d_A - 1) + h(T^*)$ (the Fannes-Audenaert inequality,
Audenaert 2007) provided $T^* \le 1 - 1/d_A$; and $\lvert \lambda_k(\rho_A) - \lambda_k(\hat\rho_A) \rvert \le 2T^*$
(Weyl). The reduced matrix of the stored vectors is formed exactly in rationals, and $x \ln x$, the binary entropy h and
ln are evaluated on the dyadic intervals of Lemma 17.

**Machine check.** `verify_C-03.py` (seed 20261030; exact rationals for $\eta$, T, the reduced matrix and the
enclosures, mpmath at 50 digits for the rest): 1 150 random and 20 026 adversarial cases, 0 violations.

**Verdict: AGREES-WITH-REMARKS.** The function `_f_point` raises an error for a rational $0 < p < 2^{-192}$, so the
record becomes NOT_CHECKED, which is sound but avoidable; $\eta_V$ is computed in the Frobenius norm where the 2-norm
suffices, which is conservative. The independent search (120 cases, seed 53) found 0 violations.

**Outcome.** No code change.

### `C-04_correlation_lift.md`: observables through the sector isometry (C-04)

**Derives.** For a symmetry-adapted claim $(V, H_s)$ audited by S1 ($\iota \ge \lVert V^H V - I \rVert$) and a cluster
certified in the block: $\mathrm{tr}(P_s V^H O V) = \mathrm{tr}(\Pi_V O)$ exactly (Lemma 0), the rounding bound of the
compressed operator $V^H O V$ (Lemma 1), and
$\lvert \mathrm{tr}(\Pi_V O) - \mathrm{tr}(\Pi_Q O) \rvert \le 2g \lVert O \rVert_2 \delta (1 + \sqrt{1+\iota})$ with Q
the polar factor of V (Proposition 1).

**Machine check.** `verify_C-04.py` (seed 20261030; exact rationals for the identities, mpmath at 50 digits for polar
factors, norms and traces): 240 random and 14 adversarial cases, 0 violations.

**Verdict: GAP.** In Lemma 1 (ii) the step $(1 + \dot\gamma_a)(1 + \dot\gamma_b) \le 1 + \dot\gamma_{a+b}$ is false for
the complex constant $\dot\gamma_m = 2\gamma_{m+4}$ (20u against 12u at a = b = 1). The statement stays true through
Higham's constants, $\le \gamma_{a+b+4} \le \dot\gamma_{a+b}/2$, so no bound is affected; the remark adds that
Proposition 1 holds with g in place of 2g. The independent search (150 cases, seed 61) found 0 violations.

**Outcome.** A repair of the proof text, to be made by Sarang Vehale; no code change.

### `C-07_moments.md`: moments and sum rules (C-07, first part)

**Derives.** For a certified cluster:
$\lvert \mathrm{tr}(PM) - \mathrm{tr}(XX^H M) \rvert \le (2gs + g\eta) \lVert M \rVert$ for Hermitian M, applied to
$M = O H^k O$; the evaluation of $\mathrm{tr}(XX^H O H^k O)$ by one application of O and k applications of H to the
stored vectors, with the rounding propagated as $D_k = \lVert H \rVert D_{k-1} + \varepsilon_k$ (Lemmas 1 to 4); and the
zeroth- and first-moment sum-rule statements (Propositions 1 and 2).

**Machine check.** `verify_C-07.py --cases 40` (seed 20261030; exact fractions for the rounding part, mpmath at 60
digits for P and the moments): 418 random and 192 adversarial cases, 0 violations; the negative controls (a projector
distance understated $10^4$ times, a random orthonormal X claimed with $s = \eta = 0$, a wrong level interval) are
detected in 132 of 138, 138 of 138 and 40 of 40 cases (`docs/proofs/runs/verify_C-07.txt`).

**Verdict: AGREES.** The independent search (60 cases, seed 101) found 0 violations.

**Outcome.** No code change. The second part of C-07 (bounds on the spectral function from the moments through the
continued fraction) is open and recorded as NOT_IMPLEMENTED (`theory.md`, section 9.4).

## Problems beyond one Hermitian matrix

### `E-03_generalized.md`: generalized eigenproblems (E-03)

**Derives.** For $Ax = \lambda Bx$ with B positive definite: the reduction to $C = L^{-1} A L^{-H}$ with $B = LL^H$; the
enclosure $\lvert \lambda - \mu \rvert \le \lVert Ax - \mu Bx \rVert_2 / (\sqrt{\beta_c} \lVert x \rVert_B)$ for any
certified $\beta_c \le \lambda_{\min}(B)$; and the count of a `lowest` claim through the inertia of $A - \sigma B$
(Sylvester's law for the congruence) with the collar $b/\beta_c$ (statements G1 to G3).

**Machine check.** `verify_E-03.py --cases 80` (seed 20261030; mpmath at 60 digits for the Cholesky factor, C and its
eigenvalues): 303 random and 100 adversarial cases (indefinite B, wrong eigenvalues, swapped pairs, near-singular B, a
false declared $\lambda_{\min}(B)$), 0 violations; the negative control (a declared $\lambda_{\min}(B)$ ten times the
true value with the inertia check disabled) puts the radius below the exact residual in 40 of 40 pairs
(`docs/proofs/runs/verify_E-03.txt`).

**Verdict: AGREES-WITH-REMARKS.** The forming error of $\hat M = \mathrm{fl}(A - \sigma B)$ is stated as
$u \lvert \hat M \rvert$ and is $u(1 + 2u) \lvert \hat M \rvert$, which the upward guard covers; the paragraph on
$\nu < k$ describes the overlapping case; the refusal of an indefinite B is conservative; the contract of
`inertia_dense` is taken as a hypothesis. The independent search (80 cases, seed 97) found 0 violations.

**Outcome.** No code change.

### `E-04_parameter_intervals.md`: eigenvalue tubes over a coupling interval (E-04)

**Derives.** For the affine interpolant $F_k(c) = H(c_k) + (c - c_k) H_1$ of certified grid Hamiltonians and
$B \ge \lVert H_1 \rVert_2$: $\lvert \lambda_j(F_k(c)) - \lambda_j(H(c_k)) \rvert \le B \lvert c - c_k \rvert$ for every
sorted index j (Weyl), so the S2 enclosures widen to tubes with the S3 ranks preserved (Propositions 1 and 2); a
sampling criterion for when two tubes are certified apart on a sample interval (Lemma 3); bounds for B from the terms of
the spec (Lemma 4); and the conclusion that a run of samples where two tubes are not separated, flanked by opposite
certified orderings, contains a crossing.

**Machine check.** `verify_E-04.py` (seed 20261030; mpmath at 50 digits, exact rationals for the row-sum bound): 1 049
random and 107 adversarial cases, 0 violations, with the negative controls detected.

**Verdict: ERROR.** `sweep.certified_crossing_interval` applied Lemma 3 across the jump of the tube at interior grid
points, where the relation between two tubes can be wrong by up to the sum of the two affinity defects $d_k^a + d_k^b$;
a scalar counterexample shows it. Before the correction a run reported as ordered on [0, 1.045] hid a crossing at
0.99755, and the independent search (120 cases, seed 71) found 60 targeted crossing cases that violated the statement
before the correction and 0 after it. The tubes themselves were not affected. The verdict also notes that Lemma 4 is
stated more generally than it holds (it holds for the letters of the three kinds of local space).

**Outcome.** A looser code change: `certified_crossing_interval` adds the affinity defects at interior grid points to
the slack, with the negative-control test
`test_crossing_criterion_accounts_for_the_affinity_defect_at_interior_grid_points` in `tests/test_independent_check.py`.
Recommended and not implemented: symmetrising the derivative and the stored blocks of the sector branch before the norm
bounds.

### `E-05_variational.md`: variational certificates (E-05)

**Derives.** For a trial vector $\psi$ and Hermitian H:
$\lambda_1 \le \theta = \langle \psi, H\psi \rangle / \langle \psi, \psi \rangle$ (Rayleigh-Ritz); an eigenvalue within
$r = \lVert H\psi - \theta\psi \rVert / \lVert \psi \rVert$ of $\theta$ (Theorem 1); Temple's inequality
$\lambda_1 \ge \theta - r^2/(\rho - \theta)$ for $\theta < \rho \le \lambda_2$; and the contraction bound of an MPS to a
dense vector (Lemmas 1 to 5, Proposition 1, statements V1 to V3).

**Machine check.** `verify_E-05.py --cases 200` (seed 20261030; exact rationals for $\theta$ and $r^2$, mpmath at 60
digits for the eigenvalues): 666 random and 294 adversarial cases, 0 violations; the negative control is detected in 66
of 66 cases (`docs/proofs/runs/verify_E-05.txt`).

**Verdict: GAP.** Lemma 3 (v) proves the model-shift statements by applying Weyl's inequality, Rayleigh-Ritz and
Temple's inequality to the floating-point build $H_{\rm ref}$, which presupposes that it is Hermitian; at the level of
rounding it need not be (a boson chain with complex hopping has $\max \lvert H_{\rm ref} - H_{\rm ref}^H \rvert$ =
1.1e-16). The statements hold by the direct route through the Hermitian $H_{\rm spec}$ with
$r \ge r_{\rm stored} + \sigma$, which is what the code uses; the text "any $r \ge r_{\rm stored}$" is not established.
No certified bound is affected. The independent search (80 cases, seed 91) found 0 violations.

**Outcome.** A repair of the proof text and of section 1; no code change.

### `E-07_davis_kahan.md`: Davis-Kahan statements (E-07, C-12, C-13)

**Derives.** With $H' = \hat Q^H A \hat Q$: $\lVert A\hat Q - \hat Q H' \rVert_F \le \lVert R \rVert_F/\sqrt{1-\eta}$
and $\lVert H' - \Lambda \rVert_2 \le w'$; the Frobenius form of the $\sin\Theta$ theorem,
$\lVert \hat Q \hat Q^H - P \rVert_2 \le s_{DK} = \lVert R \rVert_F / (\sqrt{1-\eta}\,(\mathrm{dist} - w'))$ (E-07);
labels for approximately commuting operators (C-12); fidelities between certified eigenspaces (C-13); and a rounding
lemma.

**Machine check.** `verify_E-07.py` (seed 20261030; mpmath at 50 digits): 529 random and 24 adversarial cases, 0
violations; both negative controls are detected in 20 of 20 cases.

**Verdict: GAP.** With a model shift $\sigma$ the increment of $\lVert R \rVert_F$ is stated as $\sigma\sqrt{1+\eta}$,
which is a 2-norm bound; the Frobenius route needs $\sigma\sqrt{g(1+\eta)}$. The implemented bound is valid through the
2-norm $\sin\Theta$ theorem, because the interval separation holds. Two remarks concern the squared moduli in the
fidelity and the window used for a constant in the label bound. The independent search (100 cases, seed 83) found 0
violations.

**Outcome.** A repair of the proof text (recommended: justify the increment through the 2-norm theorem); no code change.

## Scale

### `D-01_sparse-residual.md`: the measured backward error of sparse factors (D-01)

**Derives.** For the stored SuperLU factors of $A - \sigma I$, the residual $E = L D L^H - P(A - \sigma I)P^T$ formed in
floating point in row blocks has a computable entrywise bound (a), a row-sum bound (b), and
$\beta \ge \lVert E \rVert_2$ plus $2u \max \lvert S_{ii} \rvert$ for the forming error of the shifted matrix (c); with
Lemma 4 this gives the inertia sandwich with no hypothesis on how SuperLU computed its factors. The a-priori bound of
3.1 assumed that SuperLU forms every entry by a standard inner-product update, which is a property of the library's
supernodal kernels that was not established (known issue MS-M11).

**Machine check.** `verify_D-01.py` (default arguments, 202 s): 400 random sparse Hermitian matrices of dimension 2 to
13 (real and complex, three densities, entry scales $10^{-3}$, 1, $10^3$, shifts that vary the inertia) with E computed
exactly in rationals, and 17 adversarial cases (15 matrices, among them an exact zero pivot, the zero matrix and entry
scales up to $10^{300}$, plus the two scalar inequalities checked exactly), 4 of them skipped as expected-inconclusive;
0 violations.

**Verdict: AGREES-WITH-REMARKS.** The first-order margin of one scalar lemma is 6u, not 5u, which is harmless; the
modulus of a complex number through the C function `hypot` is a library function outside the one-rounding model (an
error of at most 3u is assumed); the counted factorization is $L \cdot \mathrm{Re}(D) \cdot L^H$, so the imaginary parts
of complex pivots land in E; the computed $E_c$ is not Hermitian, but mirroring its lower triangle is valid because the
exact E is; Hermiticity of the input is supplied by the callers through `hermitian_part`; the underflow hypothesis for L
and D is not scanned. The independent search (40 cases with seed 103 and 16 with seed 104, residuals in exact rationals
and corrupted factors) found 0 violations.

**Outcome.** This is the second tightening admitted under the soundness rule: the sparse S3 path uses the measured
$\beta$ instead of the a-priori bound. In the 3.2.1 evidence run two lower bounds on the next eigenvalue in experiment
E3 moved up by 1.0e-9 and 1.3e-9 relative, because the measured $\beta$ was 4 to 50 times smaller than the a-priori
bound on the chain blocks (`code/certify-ed/RESULT_evidence_run_3.2.1.md`). Decision D-61 admits the change; its revisit
clause reverts the two bounds if Sarang Vehale's check finds a gap.

### `D-02_matrix-free.md`: matrix-free residual bounds (D-02)

**Derives.** For a claim with no stored matrix, the spec is applied term by term with the rounding width
$W = w_{\rm row} + L + \rho + 1$ for real data and $w_{\rm row} + 3L + \rho + 3$ for complex data, where $w_{\rm row}$
bounds the number of monomial contributions to one output entry, L is the longest operator string and $\rho$ the number
of letters per string whose amplitudes are rounded: (a) the entrywise bound, (b) the Theorem 1 radius for the exact spec
operator (the amplitude roundings are inside W, so no model shift is added), and Lemma N, the norm bound
$\lVert H \rVert_2 \le \sum_t \lvert c_t \rvert \prod_l \lVert o_{t,l} \rVert_2$, combined with the row-sum bound of
Appendix A by a minimum.

**Machine check.** `verify_D-02.py` (default arguments): specs from six families (spin 1/2 in two conventions, spin 1,
fermions with both Jordan-Wigner conventions, bosons with cutoff 2, complex couplings) on 2 to 5 sites, against an
independent mpmath construction; 182 random and 18 adversarial cases, 0 violations.

**Verdict: GAP.** Lemma M3 counts the second pass $\mathrm{fl}(\lvert H \rvert_{\rm abs} \lvert x \rvert)$ without the
rounding of $\lvert x_j \rvert$ for a complex vector. For a real spec applied to complex-typed vectors, a case the code
reaches, the second pass is one or two roundings short of the budget, so the inequality for `absHX_up` is not
established; statement (b) is not affected, because the complex branch uses $2\gamma_{W+4}$. Section 1 does not state
the model for the modulus of a complex number, and on the development machine NumPy's complex absolute value was
measured at up to 1.88u relative error over 140 000 arguments. Lemma N evaluates the local norms on the stored
amplitudes, which the guard covers. The independent search (40 cases, seed 107) found 0 violations.

**Outcome.** The looser code change of `enclosure._rounding_bound` ($\gamma_{w+3}$ for complex data, commit `2e1dc59`)
restores the second-pass inequality for up to two roundings of the modulus; the proof text is to be restated with the
modulus hypothesis $h \ge 2$.

## Summary table

| document | statement | script and recorded arguments | counts (random + adversarial) | verdict | code change after the derivation or the verdict |
|---|---|---|---|---|---|
| `thm1_fp-accounting.md` | Theorem 1-fl | `verify_thm1_fp-accounting.py --cases 20000 --seed 6` | 20 000 + 246 | GAP | $\gamma_{w+3}$ for complex data (looser) |
| `int05_underflow.md` | INT-05, U1 to U7 | `verify_int05_underflow.py --cases 20000 --seed 7` | see the entry above, + 373 | ERROR | INT-05 additive terms; `norm2_lower` returns 0 on overflow (looser) |
| `appA_model-shift.md` | Appendix A | `verify_appA_model-shift.py --cases 400 --seed 5` | 386 + 681 | AGREES-WITH-REMARKS | $1/(1 - \gamma_W)$, width $m + 2L_{\max} + 2$, `hermitian_part` (looser) |
| `L17_dyadic_exp_ln.md` | Lemma 17 | `verify_L17.py --random 2000 --seed 0` | 6000 + 140 | AGREES | remainder constant 8/7 (looser) |
| `thm2_polar-factor.md` | Theorem 2, polar-factor lemma | `verify_thm2_polar-factor.py --cases 20000 --seed 1` | 17 224 + 600 | AGREES-WITH-REMARKS | cluster radius $\min(\rho, \varepsilon^*)$ (tighter, justified by the document) |
| `thm2prime_eps-star.md` | Theorem 2' | `verify_thm2prime_eps-star.py --cases 20000 --seed 2` | 17 094 + 621 | AGREES-WITH-REMARKS | none |
| `cor5_pairing.md` | Corollary 5 | `verify_cor5_pairing.py --cases 20000 --seed 3` | 20 347 + 900 | AGREES | none |
| `thm8_rho.md` | Theorem 8 | `verify_thm8_rho.py --cases 240 --seed 4` | 240 + 49 | AGREES | none |
| `P14_degeneracy_bracketing.md` | Proposition 14 | `verify_P14.py --random 3000 --seed 0` | 3060 + 1006 | AGREES | none |
| `P15_summed_bounds.md` | Proposition 15 | `verify_P15.py --random 500 --seed 0` | 1111 + 32 | AGREES | directed rounding, $\gamma_{m+2}$, square-root moduli (looser) |
| `P16_thermo_intervals.md` | Proposition 16 | `verify_P16.py --random 400 --seed 0` | 4000 + 95 | AGREES-WITH-REMARKS | none |
| `C06_truncated_thermodynamics.md` | Proposition 16' (C-06) | `verify_C06.py --random 300 --seed 0` | 300 + 150 | AGREES-WITH-REMARKS | none |
| `C16_exact_bracket.md` | C-16 | `verify_C16.py --random 200 --seed 0` | 1501 + 42 | AGREES-WITH-REMARKS | wording of the certified sentence (no bound changed) |
| `C-03_entanglement.md` | C-03 | `verify_C-03.py`, seed 20261030 | 1 150 + 20 026 | AGREES-WITH-REMARKS | none |
| `C-04_correlation_lift.md` | C-04 | `verify_C-04.py`, seed 20261030 | 240 + 14 | GAP | none (proof text) |
| `C-07_moments.md` | C-07 | `verify_C-07.py --cases 40`, seed 20261030 | 418 + 192 | AGREES | none |
| `E-03_generalized.md` | E-03 | `verify_E-03.py --cases 80`, seed 20261030 | 303 + 100 | AGREES-WITH-REMARKS | none |
| `E-04_parameter_intervals.md` | E-04 | `verify_E-04.py`, seed 20261030 | 1 049 + 107 | ERROR | affinity defects at interior grid points (looser) |
| `E-05_variational.md` | E-05 | `verify_E-05.py --cases 200`, seed 20261030 | 666 + 294 | GAP | none (proof text) |
| `E-07_davis_kahan.md` | E-07, C-12, C-13 | `verify_E-07.py`, seed 20261030 | 529 + 24 | GAP | none (proof text) |
| `D-01_sparse-residual.md` | D-01 | `verify_D-01.py` | 400 + 17 | AGREES-WITH-REMARKS | measured sparse $\beta$ (tighter, justified by the document) |
| `D-02_matrix-free.md` | D-02 | `verify_D-02.py` | 182 + 18 | GAP | $\gamma_{w+3}$ for complex data (looser) |

Every document has the status: derived, machine-checked, independently re-derived (a second derivation written from the
statement alone, not a human check), Sarang's check pending. The reduced runs of the original scripts and the counts of
the independent adversarial scripts are listed per row in `code/certify-ed/docs/proofs/VERDICTS.md`; all report 0
violations, except the two pre-correction counts quoted above for `int05_underflow.md` and
`E-04_parameter_intervals.md`.

## What the checks show and do not show

The machine checks show that no counterexample was found among the tested inputs, including inputs built to make each
inequality tight. The second derivations show that a proof written without sight of the first reached the same
statement, or, in seven cases, found a step that does not hold as written. Two of those seven were faults in the code
(an overflow case of the Theorem 1 radius and a crossing criterion at grid points), and both were corrected with looser
bounds and negative-control tests before version 3.2.0 was tagged. The other five are faults in proof texts whose
statements, or the bounds the code implements, stand. One remark of an AGREES-WITH-REMARKS verdict also found a fault in
the code, the wording of the C-16 sentence, which was corrected in the same way. Neither kind of check is the human
re-derivation that rule D-12 requires. The order of reading proposed to Sarang Vehale (`report/15_completion_plan.md`)
puts first the documents that changed a bound (`thm2_polar-factor.md`, `thm1_fp-accounting.md`, `int05_underflow.md`,
`appA_model-shift.md`), then the ERROR and GAP documents, then the rest; each status line is to change from "pending" to
"done" with a date when he has checked it.

The exact layer of statement S6 (Lemmas 10 to 12 and Proposition 13), Theorem 6, Proposition 7 and Proposition 9 have no
derivation document; they remain marked UNVERIFIED (see the table at the end of [theorems.md](theorems.md)).
