# Independent re-derivation: C-16 (exact eigenvalue count and multiplicities in a rational bracket)

Written on 2026-10-30 before the original proof was read (not a human check). Sections 1 and 3 of `../C16_exact_bracket.md` read first, with
`certify_ed/exact_degeneracy.py`; proof written before section 2 was opened. (H2) — exactness of python-flint's
integer/rational arithmetic, `charpoly`, `factor_squarefree`, polynomial division and evaluation — is taken as
the trusted base, as the statement says; the construction of `exact_sectors.sector_integer_matrix` (that N, G
describe H restricted to the sector) is not re-derived here and is used as stated in the Setting.

## Statement checked

(i)–(iii) and the three decision rules, for the bracket actually used after nudging.

## My proof

*Eigenvalues and multiplicities of H from p.* K = D⁻¹HD is similar to H, so χ_K = χ_H and the eigenvalues of K
are those of H with the same algebraic multiplicities (Horn–Johnson, *Matrix Analysis*, Cor. 1.3.4). H is
Hermitian, therefore unitarily diagonalizable, so the algebraic multiplicity of an eigenvalue equals the number
of times it appears in the sorted list λ₁ ≤ … ≤ λₙ (Horn–Johnson Thm 4.1.5). den·K has eigenvalues den·λ_i with
the same multiplicities; its entries are Gaussian integers by the Setting, so p = χ_{den K} ∈ ℤ[x] when K is real.
*Complex K.* Let M = den·K = A + iB with A, B real n × n and E = [[A, −B], [B, A]]. With
S = [[I, I], [−iI, iI]] one checks S⁻¹ES = diag(M, M̄) (S is invertible, S⁻¹ = ½[[I, iI], [I, −iI]]; the identity is
the standard realification), so χ_E = χ_M·χ_{M̄}. The roots of χ_M are the real numbers den·λ_i, hence χ_M has
real coefficients; as its coefficients are also Gaussian integers (symmetric functions of the entries of M),
χ_M ∈ ℤ[x] and χ_{M̄} = conj(χ_M) = χ_M. So p = χ_E = χ_M²: every root of p has exactly twice its multiplicity as an
eigenvalue, and dividing the multiplicities by `factor` = 2 restores them; an odd multiplicity in p is
impossible under (H1) and the code's ValueError is a hypothesis check.
*Sector.* G positive diagonal: det(N − x G) = det(G)·det(G⁻¹N − xI), so the solutions μ den of N v = (μ den) G v
are the eigenvalues of G⁻¹N with the same algebraic multiplicities; G⁻¹N = G^{−1/2}(G^{−1/2} N G^{−1/2})G^{1/2} is
similar to a Hermitian matrix (N = Vᴴ(WK den)V is Hermitian since WK = DHD is), so algebraic = geometric
multiplicity again. A positive integer multiple of χ_{G⁻¹N} (clearing denominators, `numer()` of a polynomial
with positive denominator) has the same roots and multiplicities.

*Squarefree factorization.* For nonzero p ∈ ℤ[x], Yun's algorithm gives p = c∏_j s_j^j with s_j squarefree and
pairwise coprime; a root of p has multiplicity exactly j iff it is a root of s_j (von zur Gathen–Gerhard, *Modern
Computer Algebra*, Thm 14.23; Yun 1976). Coprimality means no root is counted in two factors.

*Sturm's theorem.* For squarefree s ∈ ℝ[x] (so gcd(s, s′) = 1 and the canonical sequence s₀ = s, s₁ = s′,
s_{i+1} = −rem(s_{i−1}, s_i) ends in a nonzero constant) and α < β with s(α)s(β) ≠ 0, V(α) − V(β) is the number of
distinct real roots of s in (α, β) (Basu–Pollack–Roy, *Algorithms in Real Algebraic Geometry*, Thm 2.50).
Multiplying any member of the sequence by a positive constant (the code divides each remainder's integer
numerator by its content; `fmpq_poly.numer()` has a positive denominator removed) changes no sign at any point,
so the primitive integer sequence has the same V. (The code's loop stops when the remainder is zero or the last
member has degree 0; for squarefree s the last nonzero remainder is a constant, so the sequence is complete.)

*Nudging.* The ends α = a′, β = b′ are moved outward by 2⁻²⁰⁰ while either is a root of some s_j; after the
loop s_j(α)s_j(β) ≠ 0 for all j (checked exactly) and (α, β) ⊇ [a′, b′]. By Sturm on each s_j, d_j = V_j(α) − V_j(β)
is the number of distinct roots of multiplicity exactly j in (α, β) (coprimality: the sets are disjoint; Yun:
they exhaust the distinct roots of p in (α, β)). Hence for the bracket (α, β): (i) Σ_j d_j distinct eigenvalues,
(ii) Σ_j (j/factor)d_j eigenvalues with multiplicity, (iii) d_j of them of multiplicity j/factor. Since no end of
(α, β) is a root, the counts in (α, β) and in [α, β] coincide. The decision rules follow: count = g with one
distinct root ⇒ one eigenvalue of multiplicity g; count = g with several ⇒ not all equal; count ≠ g contradicts
Proposition 14's "exactly g in I_in" when [a, b] is the computed inner bracket (its ends lie inside the exact
open I_in, so [a, b] holds at most g eigenvalues, and at least g because J ⊂ [a, b] holds the g of Theorem 2;
with nudging, (α, β) ⊇ [a, b] can only hold more, so a count below g is a contradiction and a count above g
either a contradiction or a root within 64·2⁻²⁰⁰·den of an end — the record's `bracket_nudged` flag tells which
bracket was used). ∎

*Remark on the wording of the record (code, not the proposition).* `_record` prints the certified statement as
"exactly g eigenvalues … in [lo, hi]" with lo = nextafter(float(α/den), −∞), hi = nextafter(float(β/den), +∞)
formatted to 15 significant digits. That closed interval is *wider* than the bracket actually counted ([α, β]/den
is strictly inside it, by up to one binary64 ulp plus the display rounding at each end). An eigenvalue in
[lo, α/den) — within an ulp below the bracket's end — is not counted but is inside the printed interval; nothing
in the hypotheses excludes it (Proposition 14's I_in = (σ_lo + β_lo, σ_hi − β_hi) says nothing about the collar
below its left end). So the printed sentence can be false by one eigenvalue while every count in the record is
right; `bracket_exact` carries the correct ends. The proposition itself (about the rational bracket) is
unaffected; the fix is to word the statement for `bracket_exact` and label `bracket` as an outward-rounded display
(or to count in the closed outward-rounded binary64 bracket [lo, hi] itself, which is then the statement's
interval). Recorded in VERDICTS.md.

### Code (current `exact_degeneracy.py`, 259 lines)

`exact_charpoly` 49–72 (embedding 56–63, `fmpz_mat.charpoly` 71), `sector_charpoly` 75–101 (`fmpq_mat.charpoly`,
`.numer()` 99–100), `_primitive` 105–110, `sturm_sequence` 113–123, `_sign_changes` 126–134, `sturm_count` 137–145
(raises when an end is a root), `roots_in_bracket` 148–178 (`factor_squarefree` 155, nudge loop 158–165, Sturm
per part and the `factor` division 169–177), `_record` 188–218 (decision rules 202–217; the display issue above
at 196 and 206–216), `exact_bracket` 221–241, `exact_brackets_for_clusters` 244–259. No floating-point operation
enters any count; the only floats are the display ends and the input bracket (taken exactly).

## Comparison with section 2 of the original (read after the proof above was written)

Same chain: similarity and diagonalizability for the multiplicities, the realification S⁻¹ES = diag(M, M̄) with
χ_M ∈ ℤ[x] having real roots (hence χ_E = χ_M²), the sector pencil via G⁻¹N ~ G^{−1/2}NG^{−1/2}, Yun, Sturm with the
positive-scaling remark, and the nudging argument "(α, β) ⊇ [a′, b′], the statement is made for that bracket".
Same constants (none). Both valid. The original explicitly limits the statement to the *used* bracket, which is
right — but the code's record did not: see the remark above, demonstrated by an exact example (eigenvalues
1 + 2⁻⁶⁰ and 3/2, bracket [1 + 2⁻⁵², 2] → count 1, statement "exactly 1 eigenvalue in [1, 2]" while two lie in
[1, 2]). Corrected on this branch (wording only: the statement now names `bracket_exact` and labels the binary64
interval as "about"), with a negative-control test (`tests/test_independent_check.py::
test_exact_bracket_statement_names_the_exact_bracket`, fails on the previous code). No count or bound changed.

**Verdict: AGREES-WITH-REMARKS** (proposition and proof valid; a code-side wording defect in the certified
sentence, corrected).

## Runs

- `verify_C16.py --random 100 --seed 7`: see VERDICTS.md for the printed counts; 0 violations.
- `adversarial_C16_exact_bracket.py --cases 200 --seed 47`: `roots_in_bracket` 34 random + 166 adversarial
  (planted multiplicities up to n = 6 all equal, roots 2⁻⁶⁰ apart with the bracket between them, ends on roots,
  empty and full brackets, complex Hermitian with factor 2), `sturm_count` 270, nudging 200; 0 violations.
