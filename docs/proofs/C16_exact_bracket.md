# C-16: exact eigenvalue count and multiplicities in a rational bracket (S8 evidence `exact_bracket`)

Implements: `certify_ed/exact_degeneracy.py`; wired into the S8 statement of `certify_ed/certify.py` when
`Options.exact_recheck` is on. Depends on `checker.exact_matrix` (the declared model as an exact rational matrix)
and `exact_sectors.sector_integer_matrix` (exact symmetry-sector blocks); trusts python-flint for integer and
rational arithmetic, exactly as the S6 layer does. Notation as in `docs/theory.md`, Sections 6 (exact layer) and 9.1.

## 1. Statement

**Setting.** H is the Hermitian matrix of the declared model on the claim's state set (n states, the U(1) block
or the full space). `exact_matrix` gives K = D⁻¹ H D with D = diag(√w_s), w_s positive integers, such that
den·K has Gaussian-integer entries (den a positive integer); K is similar to H. For a symmetry sector,
`sector_integer_matrix` gives N = Vᴴ (W K den) V and the diagonal Gram matrix G = Vᴴ W V (positive integers),
V the orbit-sum columns, with N v = μ den G v for the eigenvalues μ of H restricted to the sector.

**Hypotheses.**
(H1) The spec's couplings are the binary64 numbers it contains (taken as exact rationals) and the local
operators are exact rationals (`LocalSpace.exact_scaled_components`): H is a well-defined Hermitian matrix with
rational entries (checked exactly by `is_hermitian_exact`).
(H2) python-flint's integer and rational arithmetic, `fmpz_mat.charpoly`, `fmpq_mat.charpoly`, `fmpz_poly.
factor_squarefree`, `gcd`, `divmod`, and evaluation are exact (the same trust as S6; no floating-point
hypothesis anywhere in this statement).
(H3) The bracket [a, b] is given by two rationals a < b (in the code: the binary64 ends of Proposition 14's inner
bracket I_in, taken exactly).

**Definitions.** p ∈ Z[x] is the characteristic polynomial of den·K (real K), or of the real embedding
[[Re, −Im], [Im, Re]] of den·K (complex K; then every root has even multiplicity and `factor` = 2); for a sector,
p is a positive integer multiple of the characteristic polynomial of G⁻¹N (rational entries), with the same
embedding rule. p = c ∏_j s_j^j is the squarefree factorization (Yun): each s_j ∈ Z[x] is squarefree, the s_j
are pairwise coprime, and the roots of s_j are exactly the roots of p of multiplicity j.

**Proposition (C-16).** Under (H1)–(H3), with a' = a·den, b' = b·den moved outward by 2⁻²⁰⁰ as long as one of
them is a root of some s_j (at most 64 times; the final bracket contains the original one), and
  d_j = V_j(a') − V_j(b'),  where V_j(x) is the number of sign changes of the Sturm sequence of s_j at x,
the following hold, with multiplicities divided by `factor`:
(i) the number of distinct eigenvalues of H (resp. of H restricted to the sector) in [a, b] is Σ_j d_j;
(ii) the number of eigenvalues in [a, b] counted with multiplicity is Σ_j (j/factor)·d_j;
(iii) exactly d_j of the distinct eigenvalues in [a, b] have multiplicity j/factor.
Consequently, for a cluster of g claimed pairs whose Proposition 14 bracket is [a, b]:
* if Σ_j (j/factor) d_j = g and Σ_j d_j = 1: "exactly g eigenvalues in [a, b], all equal" (one eigenvalue of
  multiplicity g: the level is exactly g-fold degenerate), status CERTIFIED, `equal = True`;
* if Σ_j (j/factor) d_j = g and Σ_j d_j > 1: "exactly g eigenvalues in [a, b], not all equal" with the list of
  multiplicities, status CERTIFIED, `equal = False` (the level is *not* exactly degenerate);
* if Σ_j (j/factor) d_j ≠ g: the floating-point statement of Proposition 14 is contradicted, status FAILED.

The wording of every certified statement is a count in a closed rational bracket plus the multiplicity
structure; "degenerate" means "one root of multiplicity g in the bracket", nothing more.

## 2. Proof

*Similarity.* K = D⁻¹HD has the same characteristic polynomial as H, hence the same eigenvalues with the same
algebraic multiplicities (Horn & Johnson, *Matrix Analysis*, 2nd ed., Cor 1.3.4). H is Hermitian, hence
diagonalizable, so the algebraic multiplicity of each eigenvalue equals its geometric multiplicity, i.e. the
number of equal eigenvalues in the spectral decomposition (Horn & Johnson, Thm 4.1.5 / Cor 4.4.4). Scaling:
den·K has eigenvalues den·λ, so λ is an eigenvalue of H of multiplicity m iff den·λ is a root of p of
multiplicity m. *Embedding.* For complex K, the real matrix E = [[Re K, −Im K], [Im K, Re K]] is similar over C
to diag(K, K̄) (conjugate by [[I, iI],[I, −iI]]/√2), so its characteristic polynomial is χ_K · χ_{K̄} =
χ_K · conj(χ_K); χ_K has real roots (those of Hermitian H), hence real coefficients, so χ_E = χ_K²: every root
of p has exactly twice its multiplicity in K, and dividing by `factor` = 2 restores it. If a multiplicity were
odd the code raises — it cannot happen under (H1), so an odd multiplicity would expose a violated hypothesis.
*Sector.* N v = μ den G v with G diagonal positive is equivalent to (G⁻¹N) v = (den μ) v, so den·μ runs over
the eigenvalues of G⁻¹N with the same multiplicities; G⁻¹N is similar to G^{-1/2} N G^{-1/2}, which is Hermitian
(N is Hermitian as a congruence of the Hermitian W K den), so again algebraic = geometric multiplicity. A
positive integer multiple of a polynomial (clearing denominators) has the same roots and multiplicities.

*Squarefree factorization.* For p ∈ Z[x] nonzero, p = c ∏_j s_j^j with s_j squarefree and pairwise coprime,
unique up to units, and a root of p has multiplicity j iff it is a root of s_j (Yun 1976; von zur Gathen &
Gerhard, *Modern Computer Algebra*, 3rd ed., §14.6, Thm 14.23 and Alg. 14.21). `fmpz_poly.factor_squarefree`
returns this decomposition.

*Sturm's theorem.* For a squarefree s ∈ R[x] and reals α < β with s(α)s(β) ≠ 0, the number of distinct real
roots of s in (α, β) equals V(α) − V(β), where V(x) counts sign changes (zeros dropped) in the canonical Sturm
sequence s_0 = s, s_1 = s', s_{i+1} = −rem(s_{i−1}, s_i) (Basu, Pollack, Roy, *Algorithms in Real Algebraic
Geometry*, Thm 2.50; Sturm 1829). Multiplying any s_i by a positive constant changes no sign, so the primitive
integer sequence the code builds (each remainder's numerator divided by its content, both positive) has the
same V. The code applies Sturm to each s_j with the nudged ends, which are not roots of any s_j (checked
exactly), so the hypothesis s_j(α)s_j(β) ≠ 0 holds; by squarefreeness and coprimality, the distinct roots of p in
(α, β) are partitioned by multiplicity into the roots of the s_j there, which gives (i)–(iii) for the open
nudged bracket (α, β) ⊇ [a', b']: the nudging only ever moves the ends outward, and it stops as soon as no end
is a root, so (α, β) contains [a', b'] and every root in (α, β) \ [a', b'] would have to lie within 64·2⁻²⁰⁰ of an
end — the record reports the bracket actually used (`bracket_used`, `bracket_nudged`), and the statement is made
for that bracket. (In the typical case no end is a root and (α, β) = (a', b') ⊃ [a', b'] up to the ends, which
are not roots.)

*Decision rules.* Immediate from (i)–(iii): the count with multiplicity equals g and one distinct root means
one eigenvalue of multiplicity g; the count equals g with several distinct roots means the g eigenvalues are
not all equal; a count ≠ g contradicts "exactly g eigenvalues in I_in" of Proposition 14, whose computed closed
inner bracket is the [a, b] used here (its ends are inside the exact open I_in, so the exact count in [a, b] is at
most the exact count in I_in = g and at least the g eigenvalues of Theorem 2 in J ⊂ [a, b]: a mismatch means a
hypothesis of Proposition 14 or of this statement is false). ∎

*What is not claimed.* No statement is made about which claimed pair corresponds to which eigenvalue, nor about
eigenvectors; nothing is said for a cluster whose S8 record has no inner bracket (the exact record is absent).
The hypothesis (H2) is the trust in python-flint shared with S6 and listed in the certificate's list of things it does not establish (`not_verified` field).

## 3. Where the code implements it

`certify_ed/exact_degeneracy.py` (this revision):

* `exact_charpoly(ex)`: assembles den·K (or its real embedding, `factor = 2`) as an object array and calls
  `flint.fmpz_mat(...).charpoly()`; dimension cap `MAX_DIM = 600` (the embedding counts double); returns
  (p, den, factor).
* `sector_charpoly(Are, Aim, D, den)`: `fmpq_mat` of G⁻¹N (entries N_ij / D_i, embedding when Aim ≠ 0),
  `.charpoly()`, `.numer()` (a positive integer multiple).
* `_primitive`, `sturm_sequence(s)`: the canonical sequence with `divmod` in `fmpq_poly`, `-r.numer()` divided by
  its content; `_sign_changes`, `sturm_count(s, a, b)` (raises when an end is a root).
* `roots_in_bracket(p, a, b, factor)`: `factor_squarefree`, the outward nudging by `NUDGE = 2^-200` while an end
  is a root of some s_j (at most 64 times), the per-part Sturm counts, the division of multiplicities by
  `factor` (raises on an odd one); returns count, distinct, multiplicities, bracket_used, nudged.
* `_record(...)`: the record and the three decision rules; `bracket` is the used bracket rounded outward to
  binary64 (`np.nextafter`) and `bracket_exact` the rationals as strings; `exact_bracket(source, a, b, g, den)`
  dispatches on ExactMatrix vs sector tuple; `exact_brackets_for_clusters(ex, recs)` shares one characteristic
  polynomial across the S8 clusters and uses each record's `bracket_inner`.

`certify_ed/certify.py`, S8 block: with `opts.exact_recheck`, a declared spec, no sector isometry, python-flint
present and dimension ≤ min(exact_max_dim, MAX_DIM), each cluster record gets `exact_bracket`; the S8 summary
appends "exact brackets (C-16): e exactly degenerate, d not degenerate, f contradicted"; a contradicted record
makes S8 FAILED; the assumptions name this document. Missing flint → evidence `exact_bracket` = {NOT_CHECKED,
MISSING_DEPENDENCY}; above the cap → {NOT_CHECKED, SIZE_LIMIT}; a non-Hermitian exact model → {FAILED}.
Schema 2.1 declares the S8 evidence key `exact_bracket`.

## 4. Computational verification

`docs/proofs/verify_C16.py` (run: `python docs/proofs/verify_C16.py --random 200 --seed 0`):

* Part A (exact ground truth), 600 cases: integer matrices U diag(den·d) U⁻¹ with U a random unimodular matrix
  (elementary row operations, the inverse tracked exactly) and planted rational eigenvalues d with
  multiplicities 1–3, dimension 2–12; `exact_charpoly` + `roots_in_bracket` against the planted count, number of
  distinct values and multiplicity histogram for a random bracket, a bracket whose ends are planted eigenvalues
  (the nudging path), and the whole range.
* Part B (Sturm on known polynomials), 801 cases: products ∏ (q_i x − p_i)^{m_i} with random rational roots and
  multiplicities 1–4, three random brackets each, against the known roots; a root chosen as a bracket end makes
  `sturm_count` raise; one polynomial of degree 141 with 141 distinct rational roots (coefficient growth in the
  primitive remainder sequence), 61 of them in [−10, 10].
* Part C (sector polynomial identity), 100 cases: random symmetric integer N and positive diagonal D (m ≤ 6):
  p(x) = (−1)^m·lead/∏D · det(N − xD) at m + 1 random rational points, determinants in exact `fmpq_mat`; and a
  random Hermitian Gaussian-integer block, whose embedding has `factor` 2 (1 when the imaginary part vanishes),
  even multiplicities and exactly m eigenvalues in a huge bracket.
* Part D (models, cross-check), 42 levels: Heisenberg N = 6 (periodic, open) and the spin-1 chain N = 4; for
  every level (mpmath eigenvalues at 60 digits clustered at 10⁻³⁰) the exact bracket of half-width 10⁻²⁰ returns
  the level's multiplicity as one root of that multiplicity.

Result (seed 0): 1501 random + 42 model cases, 0 violations. Two harness defects were found and fixed during
the run (a reduced-fraction numerator in the degree-141 construction and a mis-scaled diagonal entry in the
determinant check); the implementation needed no change.

`tests/test_exact_degeneracy.py`: through `certify()`, the Heisenberg N = 8 singlet and triplet are exactly
degenerate (`equal = True`, multiplicities {1: 1} and {3: 1}); a field of 2⁻³⁰ splits the triplet into three
levels 2⁻³⁰ apart, S8 brackets them as one cluster of 3 (Proposition 14, a true count) and the exact bracket
says `equal = False` with multiplicities {1: 3} (negative control for "degenerate"); a bracket holding the
triplet with g = 2 is FAILED (negative control for the count); the planted-multiplicity integer matrix; and the
k = 0 sectors of the periodic N = 8 chain, whose counts add up to the k = 0 block.

## 5. Analytic cross-check

*Diagonal matrices.* For K = diag(d_i) the characteristic polynomial is ∏ (x − den d_i), the squarefree parts
are ∏ over the values of multiplicity j, and Sturm's count on each is the number of distinct planted values
in the bracket: Part A is this case conjugated by a unimodular matrix, which leaves p unchanged.

*SU(2) multiplets.* For the isotropic Heisenberg chain [H, S²] = 0 = [H, S^z], so each level carries a total
spin S and is (2S + 1)-fold degenerate within the full space: the lowest triplet of the N = 8 chain has
multiplicity exactly 3, as the exact bracket certifies; a uniform field h S^z splits it into S^z = −1, 0, 1 with
energies shifted by −h, 0, h exactly, three simple eigenvalues 2⁻³⁰ apart for h = 2⁻³⁰, as the negative control
finds (`multiplicities` {1: 3}).

*Sector sums.* The k = 0 subspace of the periodic chain is the direct sum of the (reflection, flip) sectors, so
the eigenvalue count of the k = 0 block in any bracket is the sum of the sector counts; `test_sector_blocks_
add_up_to_the_k0_block_count` checks this for the ground state and for the lowest levels up to a gap.

*Degree and cost.* p has degree n (2n for the embedding); `fmpz_mat.charpoly` took 0.8 s at n = 256 (Heisenberg
N = 8) and about 16 s at n = 400 with small entries, which sets `MAX_DIM = 600` (N = 10 at S^z = 0 is n = 252,
N = 12 is 924 and is refused with SIZE_LIMIT).

## 6. Status

Derived on 2026-10-30; machine-checked (1501 random + 42 adversarial cases, 0 violations);
independent re-derivation: done 2026-10-30 (written before the original proof was read, not a human check; verdict AGREES-WITH-REMARKS, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
