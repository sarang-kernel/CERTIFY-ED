# The exact layer: what is re-checked without floating-point arithmetic

Statement S6 and the two modules behind it, `checker.py` and `exact_sectors.py`, re-derive the central statements of a
certificate in integer and rational arithmetic. This document says what each step computes, which function computes it,
at which sizes it has run, what it is tested against, and what remains trusted. The statements of the underlying results
(Lemmas 10 to 12 and Proposition 13) are in [theorems.md](theorems.md); here they are followed through the code. The
sources are `project_docs/notes/exact_checker.md` (version 0.3 of 21 October 2026), the proofs in
`manuscript_v4/sections/appB_exact.tex`, section 7 of `code/certify-ed/docs/theory.md`, and the code of `dev/3.2` at
commit `a814d28`.

## Why a second layer

The floating-point statements are bounds under the standard model of floating-point arithmetic
([arithmetic_model.md](arithmetic_model.md)). A run-time probe (`fp.probe_float_environment`) checks the rounding mode,
the absence of flush-to-zero and denormals-are-zero, and full-precision BLAS, and if it finds a non-standard environment
every certified floating-point statement is downgraded to INCONCLUSIVE. The exact layer removes the dependence on that
model. Floating point appears in it only as a *proposal*: a claimed eigenvector, or an $LDL^T$ factorization. Each
proposal is then checked in exact arithmetic, so a poor proposal can make the result INCONCLUSIVE, while the certified
conclusion rests only on the exact checks and the lemmas behind them. The note `exact_checker.md` calls the layer a
small checker "in the de Bruijn sense": about 700 lines of code (`checker.py` 422, `exact_sectors.py` 274, counted
without docstrings and comments on 21 October 2026), separate from the producer and from the main certifier.

## What is re-checked, by which code, at which sizes

| object | exact re-check | functions | scope and size |
|---|---|---|---|
| declared model | exact rational matrix $K = D^{-1} H D$ in the scaled basis (integer weights $\binom{2S}{k}$ for spin S, $k!$ for bosons, 1 for fermions and spin 1/2); binary64 couplings taken exactly; Hermiticity of $WK$ checked exactly | `checker.exact_matrix`, `checker.is_hermitian_exact` | every local space since 18 October 2026; before that date only local dimension q = 2 |
| S2 radii | $\lVert Hx' - \lambda x' \rVert / \lVert x' \rVert$ in integers and rationals, bounded above with an integer square root; Theorem 1 | `checker.exact_eigenpair_radii`, `checker.radius_exact`, `checker.scaled_test_vector` | any size that fits in memory |
| S3 count of a `lowest` claim | exact radii merged into components; exact inertia at separation points; Weyl margin $\delta = \beta/(b \cdot \mathrm{den})$; components holding two or more claimed values bracketed by counts on both sides (since 21 October 2026) | `checker.exact_recheck_lowest`, `checker.certify_by_counts`, `checker._plan_points`, `checker._decide`, `checker.inertia_exact` | dense, $n \le$ `Options.exact_max_dim` = 4 000, real matrices |
| S3 counts in symmetry sectors | orbit-sum bases for one-dimensional irreducible representations with Gaussian character values; exact congruence; real embedding of complex sectors; completeness of the decomposition checked by dimension; symmetry checked exactly against the model | `exact_sectors.exact_recheck_sector_lowest`, `exact_sectors.exact_count_below` | translations at momentum 0 and $\pi$ (with reflection and spin flip), all momenta on lattices of side 1, 2 or 4, any site permutation; dense per sector |
| inertia of large integer matrices | floating-point $LDL^T$; residual $\tilde L \tilde D \tilde L^T - A$ in FLINT integers, block of rows by block of rows | `checker.inertia_exact_int`, `checker.residual_inf_norm_exact`; the AKLT scripts of 17 October 2026 used `pw_symm.ldl_inertia_lean` | dense blocks up to about 9000 (memory) |

The rows follow the table of `exact_checker.md`, with the function names of `a814d28` added; the limit of 4 000 is also
listed in `code/certify-ed/docs/limitations.md`, and above it S6 is NOT_CHECKED with the reason code `SIZE_LIMIT`. The
sizes reached in practice are larger than 4 000 because the sector path factors one block at a time: experiment E7
certifies blocks of dimension 9 252 (four sectors of 2136 to 2518) and 12 870 (32 sectors of 381 to 441), see
[exact_benchmarks_e7.md](../applications/exact_benchmarks_e7.md); the crown cluster of dimension 48 620 is counted in
128 blocks, the largest of dimension 3890 ([crown_kranitz_penc.md](../applications/crown_kranitz_penc.md)); and the AKLT
bounds factor bordered integer blocks of dimension up to 4034 with `checker.inertia_exact`
([aklt_pomata_wei.md](../applications/aklt_pomata_wei.md)).

## The exact matrix (Lemma 10)

For one spin S the basis is indexed by $k = 0, \dots, 2S$, the number of quanta lowered from $S^z = S$, so that
$S^z = S - k$, $S^- e_k = \sqrt{(k+1)(2S-k)}\, e_{k+1}$ and $S^+ e_k = \sqrt{k(2S-k+1)}\, e_{k-1}$. With
$w_k = \binom{2S}{k}$ one has $w_k/w_{k+1} = (k+1)/(2S-k)$, and the scaled matrix elements are integers:
$$(D^{-1} S^- D)_{k+1,k} = k + 1, \qquad (D^{-1} S^+ D)_{k-1,k} = 2S - k + 1,$$
while $S^z$ is unchanged. For bosons with $w_k = k!$, $(D^{-1} b^\dagger D)_{k+1,k} = 1$ and $(D^{-1} b D)_{k-1,k} = k$.
For fermions and spin 1/2, $w = 1$, the entries are 0 and $\pm 1$, and Jordan-Wigner strings are diagonal with entries
$\pm 1$. On many sites $D = \bigotimes_i D_i$ and an operator string transforms factor by factor, so its matrix has
dyadic entries: $S^\pm$ give integers, $S^z$ gives half-integers for half-integer S, and $S^x = (S^+ + S^-)/2$ and
$S^y = (S^+ - S^-)/2i$ give dyadic and Gaussian-dyadic entries (finding MS-M04: version 3 of the manuscript said
"integer"). A binary64 coupling is a dyadic rational, so $K = (K_{\rm re} + i K_{\rm im})/\mathrm{den}$ with integer
matrices $K_{\rm re}$, $K_{\rm im}$ and a positive integer den. Since D is real and diagonal, $DHD = D^2 K = WK$ is
Hermitian whenever H is; the code checks the Hermiticity of $WK$ exactly, as a check of the declared model, and a
non-Hermitian specification gives FAILED ("declared model is not Hermitian", `checker.exact_recheck_lowest`). The
symmetry $\binom{2S}{k} = \binom{2S}{2S-k}$ keeps the spin flip a symmetry of K. Status: "classical, re-derived" in the
manuscript (the unnormalised monomial basis of $\mathrm{Sym}^{2S}(\mathbb{C}^2)$, Arovas et al. 1988, kept as a check).

The `lowest` re-check without sectors handles real matrices only: for a model with complex entries
`checker.exact_recheck_lowest` returns NOT_CHECKED ("complex model: exact inertia implemented for real matrices"), while
the exact radii and the sector path accept complex entries.

## Exact radii

The claimed vector x is divided by the square roots of the weights in floating point, $y = \mathrm{fl}(x/\sqrt{w})$
(`checker.scaled_test_vector`), and y is converted exactly to integer vectors times a common power of two
(`checker._float_vector_to_int`). The test vector is $x' = Dy$, which equals the claimed vector up to one rounding per
entry; any nonzero y gives a valid interval, and the statement does not depend on how close $x'$ and x are. For
$x' = Dy$ one has $Hx' = DKy$, so
$$\lVert Hx' - \lambda x' \rVert^2 = \sum_r w_r \lvert (Ky - \lambda y)_r \rvert^2, \qquad \lVert x' \rVert^2 = \sum_r w_r \lvert y_r \rvert^2$$
are rational numbers, and an upper bound on the square root of their quotient is found with integer square roots
(`checker.radius_exact`); Theorem 1 then gives an eigenvalue within that radius of $\lambda$. The hypothesis $x' \ne 0$
is checked exactly ($\lVert x' \rVert^2 > 0$). For a zero test vector, a zero claimed vector or a vector whose exact
projection onto the claimed sector vanishes, `radius_exact` returns no interval and every caller reports INCONCLUSIVE.
Version 3.0.0 returned radius 0 for such a vector, which certified any eigenvalue "with no floating-point assumption";
this was attack A1 of the review of 21 October 2026 (finding IMP-18), and it is now covered by the five `test_imp18_*`
tests of `tests/test_review_fixes.py` and by
`tests/test_adversarial.py::test_A1_exact_recheck_zero_vector_wrong_value_not_certified`. In the sector path the claimed
vector is first projected exactly onto the claimed sector, in integer arithmetic, so the radius is a statement about H
restricted to that sector.

## Exact counts (Lemma 11)

At a rational point $g = a/b$ with $b > 0$ the matrix
$$M = \mathrm{den} \cdot D(bH - aI)D = b \cdot W(\mathrm{den}\, K) - a \cdot \mathrm{den} \cdot W$$
is an integer symmetric matrix (`checker._weighted_shift_matrix`), congruent to $b \cdot \mathrm{den} \cdot (H - g)$;
$b > 0$ and $\mathrm{den} > 0$ keep the sign of the congruence (finding MS-M08). A floating-point $LDL^T$ factorization
proposes factors, and `checker.inertia_exact` moves them to dyadic grids: $\tilde L = \mathrm{rint}(L\, 2^{52})/2^{52}$,
and $\tilde D$ with 52 bits relative to its largest entry. After the row permutation P the rounded $\tilde L$ is exactly
unit lower triangular (checked), hence invertible, and $\tilde D$ is block diagonal with $1 \times 1$ and $2 \times 2$
blocks of dyadic numbers whose inertia is counted exactly. The residual $E = \tilde L \tilde D \tilde L^T - PMP^T$ is
computed exactly with FLINT integers, 256 rows at a time by default, and $\beta$ is its largest absolute row sum, so
$\beta \ge \lVert E \rVert_\infty \ge \lVert E \rVert_2$ because E is symmetric. By Sylvester's law the number $\nu$ of
negative eigenvalues of $\tilde D$ is the number of negative eigenvalues of $M + E$, and since
$$M + E = D\bigl(b\, \mathrm{den}\, (H - g) + D^{-1} E D^{-1}\bigr) D,$$
it is also the number of negative eigenvalues of $H - g + F$ with $F = D^{-1} E D^{-1}/(b \cdot \mathrm{den})$. Every
weight is at least 1, so $\lVert D^{-1} \rVert \le 1$ and $\lVert F \rVert \le \beta/(b \cdot \mathrm{den}) = \delta$,
and Weyl's inequality gives
$$\#\{\lambda < g - \delta\} \le \nu \le \#\{\lambda < g + \delta\}.$$
When a pivot sign is ambiguous the function returns no count, and the caller reports INCONCLUSIVE. Matrices with entries
above about $2^{900}$ are scaled down by a power of two before the floating-point proposal, and the scaling is undone
exactly in the residual. Status: UNVERIFIED in the manuscript, no derivation document.

## Symmetry sectors (Lemma 12)

The group acts on basis states by permutations with phases $\pm 1$ (site permutations and the spin flip). That it
preserves the weights and commutes with K is checked exactly (`exact_sectors.check_commutes`); a generator that is not a
symmetry of the declared model gives FAILED. For a one-dimensional character $\chi$ with values in $\{\pm 1, \pm i\}$
the orbit sums $v = \sum_g \overline{\chi(g)}\, \phi_g(s)\, e_{gs}$ (`exact_sectors.orbit_columns`) are Gaussian-integer
vectors; orbit sums of different orbits have disjoint supports, so they are orthogonal in every diagonal inner product,
in particular in $\langle u, v \rangle_W = u^H W v$, and zero orbit sums are dropped. Writing $q = Dv/\lVert Dv \rVert$
and $n_v = \lVert Dv \rVert \ge 1$ (because $\lVert Dv \rVert^2 = \sum_r w_r \lvert v_r \rvert^2$ is a positive
integer), the sector block is $B_\chi = N(Q^H H Q) N$ with $N \ge I$, and the argument of Lemma 11 goes through with N
in place of D and the same $\delta$. The identity term of the shifted block is $a \cdot \mathrm{den} \cdot v^H W v$, not
$a \cdot \mathrm{den} \cdot \lVert v \rVert^2$, as the congruence requires (finding MS-M01). A complex block is counted
through its real embedding $\begin{pmatrix} B_{\rm re} & -B_{\rm im} \\ B_{\rm im} & B_{\rm re} \end{pmatrix}$, in which
every eigenvalue appears twice; the count is halved when it is even, and an odd count, which can arise only when the
perturbation E splits a pair across $g \pm \delta$, gives no count and makes the result INCONCLUSIVE (findings M-09 and
IMP-17; `exact_sectors._count_sector_below`). The group hypotheses (closure, $\chi$ a homomorphism, zero orbit sums
dropped) make distinct orbit sums disjoint and the sector dimensions add up, and all three are enforced by the code
(finding MS-M07). If the sector dimensions add up to the dimension of the fixed subspace (checked; otherwise the
function stops with the message that the one-dimensional Gaussian irreducible representations cover only part of the
dimensions), the vectors q of all sectors form an orthonormal basis, H is block diagonal in it, and the counts add up
with $\delta = \max_\chi \delta_\chi$.

Non-abelian groups are allowed as long as every irreducible representation met is one-dimensional with Gaussian
character values, for example translations at momentum 0 or $\pi$ together with reflection and spin flip. A translation
of order L has the character values $e^{2\pi i j/L}$, which are all in $\{\pm 1, \pm i\}$ exactly when L is 1, 2 or 4;
this is the origin of the scope "all momenta on lattices of side 1, 2 or 4" in the table above. Status: UNVERIFIED in
the manuscript, no derivation document.

## The "k lowest" decision (Proposition 13)

The exact radii are merged into connected components $C_1 < \dots < C_m$, with $m_j$ claimed values in $C_j$ and
$K_j = m_1 + \dots + m_j$ (`checker._components`). `checker._plan_points` places the count points: one in every gap
between components (the midpoint when both neighbours hold a single value), a point close to each side of every
component with $m_j \ge 2$, including below $C_1$ (at a distance
$\min(\max(64 \cdot \text{width}, \text{scale}/10^{11}), \text{gap}/4)$), and a point halfway between the top component
and a floating-point estimate of the next eigenvalue. The estimate is only a hint: in a certificate it is the
`next_eigenvalue_estimate` of statement S3, and without one the code uses the top of the last component plus
$\max(10^{-6}, 10^{-6} \lvert \text{top} \rvert)$. Points are rational numbers with denominators of at most $2^{40}$.

`checker._decide` returns CERTIFIED when no component meets any window $[g - \delta_g, g + \delta_g]$ and every count
equals the number of claimed values below its point. Then a component with one claimed value contains $\lambda_{K_j}$
(its residual interval): at least $K_j - 1$ eigenvalues lie below the window under it and at most $K_j$ below the window
above it, so the index is pinned without any distinctness argument. A component with $m_j \ge 2$ claimed values gets a
*level enclosure*: the eigenvalues $\lambda_{K_{j-1}+1}, \dots, \lambda_{K_j}$ lie in
$[a_j - \delta_{a_j}, b_j + \delta_{b_j}]$ between its bracketing points. And
$\lambda_{k+1} \ge g_{\rm top} - \delta_{\rm top}$ is reported as `next_lower_bound`.

The other statuses follow `checker._status` and `checker.certify_by_counts`. FAILED is reported when a count exceeds the
number of claimed values below a separated point that has a claimed component above it: an unclaimed eigenvalue lies
under a claimed one. An excess only above the top component may be the rest of a degenerate level that the claim cuts;
when the top component holds one claimed value, one more count just below that component decides between a level missing
under the top claim (FAILED) and an ill-posed claim (INCONCLUSIVE). Every other way of not certifying (an unseparated
window, an ambiguous pivot, an odd count in a complex block, a count below the number of claimed values) gives
INCONCLUSIVE. Here the manuscript and the code differ in one place: `appB_exact.tex` also treats an under-count as a
refutation of the claim (finding MS-M12), while the code at `a814d28` and section 7 of `theory.md` report INCONCLUSIVE
for it. The difference concerns only the choice between FAILED and INCONCLUSIVE, never a CERTIFIED status, and the
code's choice is the more cautious one.

**The correction of 21 October 2026 (decision D-46).** Before that date a component of overlapping residual intervals
holding $m \ge 2$ claimed values was certified when one count above it equalled the number of claimed values below.
Residual intervals give *an* eigenvalue each, not distinct ones, and two copies of the first excited vector claimed as
"the two lowest" were therefore certified although the ground state is missing. Such components are now bracketed by
counts on both sides, also below the first component, and their eigenvalues are certified inside the bracket;
single-value components keep their residual interval, because the counts on the two sides pin the index, and a cut
multiplet is INCONCLUSIVE rather than FAILED (`exact_checker.md`). The tests
`test_exact_recheck_duplicated_vector_does_not_hide_a_missing_ground_state` and
`test_exact_recheck_multiplet_gets_a_level_enclosure` in `tests/test_exact_s2.py` cover both halves. The E7 benchmark
claims, the S = 1 printed-coupling check and the crown and AKLT counts use single values or counts only, so their
results did not change in substance; they were re-run anyway (`code/certify-ed/results/run_full_20261021`). Status of
Proposition 13: UNVERIFIED in the manuscript, no derivation document.

## How S6 enters a certificate

S6 runs when `Options.exact_recheck` is set (the default is `False`), in `certify._exact_statement`. It is NOT_CHECKED
with a reason code when the specification is missing (`NEEDS_SPEC`), when the claim has no eigenvectors
(`NEEDS_EIGENVECTORS`), when the claim is not of kind `lowest` or is symmetry-adapted (`UNSUPPORTED_CLAIM`;
symmetry-adapted claims are re-checked by calling `exact_sectors.exact_recheck_sector_lowest` directly), when the
dimension exceeds `Options.exact_max_dim` (`SIZE_LIMIT`), and when python-flint is not installed (`MISSING_DEPENDENCY`,
the extra `certify-ed[exact]`). Non-finite claimed values or vectors give INCONCLUSIVE. Two links to the rest of the
certificate follow from `certify.certify`. When S6 is CERTIFIED, the entry "interval-arithmetic proof: bounds rely on
the standard floating-point model" is left out of the `not_verified` list. And when the run-time probe finds a
non-standard floating-point environment, every certified statement except S6 is downgraded to INCONCLUSIVE
(`tests/test_fp_environment.py::test_certificate_downgrades_under_ftz`).

The exact layer is also used outside S6. The pyrochlore crown application (`apps/pyrochlore/spin_half.py`) calls
`exact_sectors.exact_count_below` for a single count at a dyadic shift, which bounds $\lambda_{\min}$ from below with no
floating-point hypothesis; the S = 1 ring of the same application is re-checked by S6, with an exact radius of
5.092986520018812e-15 against a floating-point radius of 1.3545154070606862e-13
([crown_kranitz_penc.md](../applications/crown_kranitz_penc.md)); the AKLT bounds count the eigenvalues of bordered
integer matrices with `checker.inertia_exact` ([aklt_pomata_wei.md](../applications/aklt_pomata_wei.md)); and the 3.2
item C-16 counts the eigenvalues of the declared model in a rational bracket exactly, with their multiplicities
(`exact_degeneracy.exact_bracket`).

## Tests

At commit `a814d28` the exact layer is tested by the following files of `code/certify-ed/tests/`. The note of 21 October
2026 lists `tests/v2/test_exact_s2.py` (13 tests) and `tests/v2/test_exact_sectors.py` (5 tests); in the code of
`a814d28` the files are directly in `tests/`, with 13 and 6 tests.

| file | tests | what they check (from the test names and the note) |
|---|---|---|
| `test_exact_s2.py` | 13 | the exact matrix agrees with the floating-point builder to 1 ulp; the radius is exactly 0 for an exact eigenvector; the radius bounds the true distance; exact radii are at most the floating-point radii; a complex Hermitian (Dzyaloshinskii-Moriya) model; a non-Hermitian spec is detected; higher spins and bosons; a complete claim is certified; a missing level gives FAILED; a truncated multiplet is not certified; S6 inside `certify`; the duplicated-vector case of D-46; the level enclosure of a multiplet |
| `test_exact_sectors.py` | 6 | the momentum-0 sector of a chain is certified and the sector dimensions add up; a missing level gives FAILED; a vector from the wrong sector is not certified; complex momenta on the square lattice with the full block; a non-symmetry is rejected; the momentum-0 sector of a spin-1 chain with flip and reflection |
| `test_review_fixes.py` | 5 for IMP-18 | zero test vectors in `radius_exact`, in `exact_eigenpair_radii`, in `exact_recheck_lowest`, in S6 and in the sector path |
| `test_adversarial.py` | case A1 | a zero claimed vector with a wrong value is not certified |
| `test_exact_degeneracy.py` | 5 | the exact count and multiplicities in a rational bracket of the 3.2 item C-16 (`exact_degeneracy.exact_bracket`, S8 evidence `exact_bracket`; [certificate_statements.md](certificate_statements.md)) |

## What stays trusted

Python's integers, FLINT's integer matrix product, and the code of these functions, which is reviewed and tested but not
mechanically proved (`exact_checker.md`; mechanised proofs of the checker lemmas, item D9 of the horizons list, would
remove the last item). On the mathematical side, Lemma 10 is classical and was re-derived as a check, while Lemmas 11
and 12 and Proposition 13 are marked UNVERIFIED in `manuscript_v4/sections/appB_exact.tex` until a second person
re-derives them (rule D-12), and none of them has a derivation document among the 22
([derivation_documents.md](derivation_documents.md)). Results obtained with the exact layer are therefore stated as
certified by the implementation, subject to the UNVERIFIED Lemmas 11 and 12 and Proposition 13, as in rows A1 and C3 of
`project_docs/notes/claims_evidence_map.md` (finding RA-20).
