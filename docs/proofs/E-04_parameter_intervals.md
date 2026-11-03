# E-04 Certified parameter intervals (eigenvalue tubes over a coupling interval)

Notation as in `docs/theory.md`. A family of Hermitian n × n operators depends on one real coupling c. Grid
couplings c₀ < c₁ < … < c_m; at each c_k a certificate (S1 with model shift, S2, S3) of the k lowest eigenpairs of
H(c_k), the spec's operator at that coupling. λ_j(A) denotes the j-th smallest eigenvalue of a Hermitian A
(j = 1 … n, counted with multiplicity). A ModelSpec stores its Hamiltonian as a list of terms (operator string,
list of (coefficient, sites)); "the coefficients are affine in c" means coef_t(c) = a_t + c b_t for every
coupling entry t.

## 1 Statement

**Definitions.** H1 is the *derivative spec*: the ModelSpec with the same term structure and coefficients b_t
(declared by the user, or inferred as b_t = fl((coef_t(c₁) − coef_t(c₀))/(c₁ − c₀)) from the first two grid
couplings). On the segment [c_k, c_{k+1}] the certified object is the affine interpolant

    F_k(c) := H(c_k) + (c − c_k) H1,

where H(c_k) is the exact spec operator at c_k (the operator the certificate at c_k refers to) and H1 the exact
spec operator of the derivative spec. The *affinity defect* d_k is an upper bound on ‖H(c_{k+1}) − F_k(c_{k+1})‖₂
= ‖H(c_{k+1}) − H(c_k) − (c_{k+1} − c_k)H1‖₂; d_k = 0 exactly when the coefficients are affine in binary64. B is an
upper bound on ‖H1‖₂.

**Proposition 1 (Weyl tube).** For every c and every index j,

    |λ_j(F_k(c)) − λ_j(H(c_k))| ≤ B |c − c_k|,      |λ_j(F_k(c)) − λ_j(H(c_{k+1}))| ≤ B |c − c_{k+1}| + d_k.

Hence if the certificate at c_k gives λ_j(H(c_k)) ∈ [l_j, u_j] (S2 enclosure with the model shift, rank j
certified by S3 for a 'lowest' claim, Corollary 5), then for c ∈ [c_k, c_{k+1}]

    λ_j(F_k(c)) ∈ [l_j − B|c − c_k|, u_j + B|c − c_k|] ∩ [l′_j − B|c − c_{k+1}| − d_k, u′_j + B|c − c_{k+1}| + d_k],

with [l′_j, u′_j] the enclosure at c_{k+1}. Without S3 the enclosure [l_j, u_j] contains an eigenvalue of
H(c_k) of some index i (Theorem 1), and the tube then contains λ_i(F_k(c)) for the same i.

**Proposition 2 (index preservation, count form).** For every threshold σ and c,

    #{λ(F_k(c)) < σ − B|c − c_k|} ≤ #{λ(H(c_k)) < σ} ≤ #{λ(F_k(c)) ≤ σ + B|c − c_k|}.

In particular S3's bound λ_{k+1}(H(c_k)) ≥ L gives λ_{k+1}(F_k(c)) ≥ L − B|c − c_k|, and the k claimed tubes
stay the k lowest eigenvalues of F_k(c) as long as they lie below that line.

**Lemma 3 (sampling criterion for separation).** Let T_a(c) = [α⁻(c), α⁺(c)] and T_b(c) = [β⁻(c), β⁺(c)] be tubes
whose ends are B-Lipschitz in c (every tube above is: an intersection of functions of the form const ± B|c − c_k|).
Put m(c) = α⁺(c) − β⁻(c). If on a sample interval [c_i, c_{i+1}] of width D, max(m(c_i), m(c_{i+1})) + B D < 0, then
m(c) < 0, i.e. T_a(c) lies strictly below T_b(c), for every c ∈ [c_i, c_{i+1}].

**Bounds used for B.** (i) Row-sum bound of the built H1: ‖H1‖₂ ≤ ‖H1_built‖_∞ + ‖H1 − H1_built‖₂, the second term
being the builder shift of `certify._builder_shift`. (ii) Triangle bound Σ_t |b_t| ‖O_t‖₂ with
‖O_t‖₂ ≤ Π_letters ‖o‖_∞ (Lemma 4). Both are upper bounds; the code uses the smaller.

**Lemma 4 (term norm).** For an operator string O = o₁ ⊗ … ⊗ o_m on distinct sites (with Jordan–Wigner signs where
the local space is fermionic), ‖O‖₂ ≤ Π_i ‖o_i‖₂ ≤ Π_i ‖o_i‖_∞.

## 2 Proof

**Proposition 1.** F_k(c) − H(c_k) = (c − c_k)H1 is Hermitian with ‖(c − c_k)H1‖₂ ≤ |c − c_k| B. Weyl's inequality
(Horn & Johnson, *Matrix Analysis*, Theorem 4.3.1; for Hermitian A, E: |λ_j(A + E) − λ_j(A)| ≤ ‖E‖₂ for every j) gives
the first inequality. For the second, F_k(c) − H(c_{k+1}) = (F_k(c) − F_k(c_{k+1})) + (F_k(c_{k+1}) − H(c_{k+1})), of norm
≤ |c − c_{k+1}| B + d_k; Weyl again. The tube statement follows by substituting the enclosures; the intersection of
two sets both containing λ_j(F_k(c)) contains it. Without S3, Theorem 1 at c_k gives an index i with
λ_i(H(c_k)) ∈ [l_j, u_j], and the first inequality with j := i gives the tube for λ_i(F_k(c)). ∎

**Proposition 2.** By Proposition 1 with E := F_k(c) − H(c_k), λ_j(F_k(c)) ∈ [λ_j(H(c_k)) − B|c − c_k|,
λ_j(H(c_k)) + B|c − c_k|] for every j. If λ_j(F_k(c)) < σ − B|c − c_k| then λ_j(H(c_k)) ≤ λ_j(F_k(c)) + B|c − c_k| < σ, so
every index counted on the left is counted in the middle; if λ_j(H(c_k)) < σ then λ_j(F_k(c)) ≤ σ + B|c − c_k|, so
every index counted in the middle is counted on the right. For the gap statement: λ_{k+1}(H(c_k)) ≥ L and Weyl
give λ_{k+1}(F_k(c)) ≥ L − B|c − c_k|. ∎

**Lemma 3.** For c ∈ [c_i, c_{i+1}]: α⁺(c) ≤ α⁺(c_i) + B(c − c_i) and β⁻(c) ≥ β⁻(c_i) − B(c − c_i) (Lipschitz), so
m(c) ≤ m(c_i) + 2B(c − c_i); symmetrically m(c) ≤ m(c_{i+1}) + 2B(c_{i+1} − c). Hence
m(c) ≤ min(m(c_i) + 2B(c − c_i), m(c_{i+1}) + 2B(c_{i+1} − c)). The minimum of two affine functions with slopes +2B and
−2B on an interval of width D is at most the average of their values at the two ends plus B D (the two lines cross
where the min is largest; at the crossing both equal (m(c_i) + m(c_{i+1}))/2 + B D, and elsewhere the min is smaller),
so m(c) ≤ (m(c_i) + m(c_{i+1}))/2 + B D ≤ max(m(c_i), m(c_{i+1})) + B D < 0. That the tube ends are B-Lipschitz: each is
a maximum (lower ends) or minimum (upper ends) of finitely many functions l − B|c − c_k| or u + B|c − c_k| (+ constants),
which are B-Lipschitz, and max/min of B-Lipschitz functions are B-Lipschitz. ∎

*Continuity and crossings.* The sorted eigenvalues of F_k(c) are Lipschitz in c (Proposition 1 with two values of
c), hence continuous. If T_a(c₁) < T_b(c₁) and T_a(c₂) > T_b(c₂) for certified tubes of two branches, the
difference of the two eigenvalue branches changes sign between c₁ and c₂, so there is a c* ∈ (c₁, c₂) where they
are equal (intermediate value theorem): a crossing lies in every not-separated run flanked by opposite certified
relations. For two levels of one scan with certified ranks, λ_a(c) ≤ λ_b(c) for a < b at every c, and the
procedure returns no crossing.

**Bound (i) for B.** For Hermitian A, ‖A‖₂ ≤ (‖A‖₁‖A‖_∞)^{1/2} = ‖A‖_∞ (Horn & Johnson, Corollary 5.6.35 / the
Schur test); `fp.inf_norm_upper` bounds the row sums with their rounding. ‖H1‖₂ ≤ ‖H1_built‖₂ + ‖H1 − H1_built‖₂ is
the triangle inequality; the second term is the bound `_builder_shift` already used by S1 for the model shift.
**Bound (ii) and Lemma 4.** ‖Σ_t b_t O_t‖₂ ≤ Σ_t |b_t|‖O_t‖₂ (triangle inequality). ‖A ⊗ B‖₂ = ‖A‖₂‖B‖₂ (singular
values of a Kronecker product are the products); Jordan–Wigner strings multiply by diagonal ±1 matrices, which
are unitary and do not change the spectral norm; ‖o‖₂ ≤ ‖o‖_∞ for a Hermitian letter (as above) and for a
non-Hermitian letter ‖o‖₂ ≤ (‖o‖₁‖o‖_∞)^{1/2} ≤ max(‖o‖₁, ‖o‖_∞) — the code bounds both by the ∞-norm of |o| and
of |o|ᵀ through `inf_norm_upper(np.abs(M))`, which is the larger row sum; for the letters of the three local
spaces (S^±, S^z, c^±, n, b^±) the matrices have at most one nonzero per row and per column, so ‖o‖₁ = ‖o‖_∞ and
the bound is exact. The float letter matrices differ from the exact ones by a relative u per entry (one rounding
of a square root), covered by the factor (1 + 4u). ∎

**Affinity defect.** For the spec family, H(c_{k+1}) − H(c_k) − Δc H1 = Σ_t e_t O_t with e_t = coef_t(c_{k+1}) −
coef_t(c_k) − Δc b_t, so by the triangle inequality and Lemma 4 the norm is ≤ Σ_t |e_t| ‖O_t‖₂; e_t is computed
*exactly* in rationals (binary64 coefficients and Δc are dyadic), so d_k = 0 whenever the coefficients are affine.
For a sector block family (stored matrices) the defect is ‖H_s(c_{k+1}) − H_s(c_k) − Δc H1_s‖₂ ≤ ‖·‖_∞ of the
computed difference plus the rounding of its three operations per entry (≤ γ₃(|H_b| + |H_a| + |Δc||H1|)
entrywise, standard model), summed by rows. ∎

## 3 Where the code implements it

`certify_ed/sweep.py` (branch `track/new-statements`):

* `derivative_spec` (56–72): H1 from the coefficients at c₀, c₁ (declaration; the defect checks it).
* Lemma 4: `term_norm_bound` (75–83), product of `inf_norm_upper(|o|)` over the letters times (1 + 4u).
* B: `h1_norm_bound` (86–98): triangle bound (line 90 ff., `add_up`/`up`), row-sum bound plus builder shift (97),
  the smaller returned (98); in `certified_sweep` line 219 (full or U(1)-block family, with
  `certify._builder_shift(h1, states)`), line 217 for a sector block family (row-sum bound of the stored H1_s).
* Affinity defect: `affinity_defect` (112–135; exact rational e_t, |e_t| rounded up, Σ with `add_up`), used at
  line 221; `matrix_affinity_defect` (101–109) for sector blocks, used at line 266.
* Proposition 1: `tube_at` (308–336): line 322 widen = up(B·|c − c_k|); line 324 adds d_k for the right grid point;
  lines 325–326 `sub_down`/`add_up` of the certificate's outward-rounded ends; the intersection is max/min of the
  two sides. `rank_certified` is the conjunction of the grid points' flags (S3 CERTIFIED and kind 'lowest'/'all',
  for the operator scope recorded in `rank_scope`). Proposition 2's gap line: `next_level_lower_bound_at`
  (339–354), same widening.
* Lemma 3: `certified_crossing_interval` (357–418): m_< and m_> at the sample points rounded up (nextafter), the
  sample width D rounded up (387), slack = up(B·D) (388), the criterion at 389 / 391; runs and crossing intervals
  (405: a not-separated run flanked by '<' and '>').
* `certified_sweep` (172–289) builds H(c_k) and H1 from the specs (zero-leak check for a U(1) block), runs
  `certify()` at every grid point, reads the S2 enclosures (radii including the model shift, `crossings._radii`;
  for sector claims the block radii of `body["enclosures"]`), S3's status and `next_eigenvalue_lower_bound`, and
  records the family, B with its source, the defects, `exactly_affine`, the per-point hashes and `rank_scope`.
* CLI: `certify-ed sweep --model … --param … --values …` (`cli.cmd_sweep`).

Rounding directions: every widening is rounded up, every tube end outward, every margin up; the sample width
up. The certificate ends l_j, u_j are already outward (`fp.sub_down`/`add_up` in `certify.py`).

*Sector scans.* With `sector_fn` the scan concerns the stored block family F_k(c) = H_s(c_k) + (c − c_k) H1_s
(H_s = V₀ᴴHV₀, H1_s = V₀ᴴH1V₀ in floating point): the ranks are ranks within the block, S1 audits V₀ at every grid
point, and the lift to the full operator (sector lift shift) is certified at the grid points only and recorded per
point; between grid points the tube is a statement about the block family. The caveat is in the record.

## 4 Computational verification

`docs/proofs/verify_E-04.py` (seed 20261030; mpmath at 50 digits for eigenvalues and spectral norms; exact
rationals for the row-sum bound; the end-to-end parts compare with binary64 LAPACK eigenvalues within 10⁻¹²).
Run of 2026-10-30:

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | Weyl per sorted index on random Hermitian H0, H1 (n ≤ 8, real and complex, ‖H1‖ over 10⁻³…10³, c ∈ [−5, 5]), with B the exact ‖H1‖₂ and the exact-rational row-sum bound | 80 random + 7 adversarial (H1 = I: equality; rank-one H1; degenerate H0; c = 0.4, 0.5, 0.6 across a crossing of H0 + cH1; c = 50) | 0 |
| B | Proposition 2, three random thresholds per case of A | 261 | 0 |
| C | Lemma 3 on exact two-grid-point tubes (intersections with a kink inside the segment): when the criterion holds (318 of the 400 cases), m(c) < 0 at 1001 points of the sample interval | 300 random + 100 adversarial (kink at the midpoint; the criterion holds by a relative 10⁻⁹ in 50 cases and fails by the same margin in the other 50, as expected) | 0 |
| D | Lemma 4 against the exact spectral norm of the Kronecker product, all six local spaces, strings of 1–3 letters | 108 | 0 |
| E | end to end: TFIM N = 4 (k = 2, grid 0.3/0.8/1.3) and XXZ N = 6 in the S^z = 0 block (k = 1, Δ grid 0.5/1/2): tube ∋ dense eigenvalue at 100 random couplings each | 300 tube evaluations | 0 |
| F | negative controls: B/20 → violation or inconsistent (empty) intersection at 50/50 random couplings; declared H1 = 0 → `exactly_affine` False, defect 2 = ‖H(0.8) − H(0.3)‖₂ | — | detected |

Total: 1 156 cases, 0 violations. Tests `tests/test_sweep.py` (7): TFIM N = 6 (k = 2) and XXZ (S^z = 0) tubes
contain the dense eigenvalues at 60 + 30 random couplings (100 %) with the gap line below λ₃; grid-point tubes
equal the certificate intervals (one ulp); linear growth of the width; wrong B (negative control: every one of 40
random points violated or detected as inconsistent); wrong declared H1 flagged by the defect while the tubes of
the declared (constant) family stay correct; the leak refusal for a non-conserving H1; two sector scans of the
8-site J1–J2 chain (k = 0 and k = π, S^z = 0) whose crossing interval [0.424, 0.585] brackets the Majumdar–Ghosh
point J2 = 1/2 (where the two ground states are exactly degenerate), with every certified run checked against
dense block diagonalisation; the CLI.

## 5 Analytic cross-check

1. *TFIM field sweep.* H(h) = −J Σ σ^z_iσ^z_{i+1} − h Σ σ^x_i: H1 = −Σ σ^x_i with ‖H1‖₂ = N exactly (the eigenvalues
   of Σσ^x are −N … N), and the triangle bound Σ_i |−1|·‖σ^x‖_∞ = N is tight: the record shows B = N (= 6 for the test,
   4 for the N = 4 runs). The coefficients −h are affine in h with b = −1 exactly, so d_k = 0 for grid values whose
   differences are exact in binary64 (e.g. 0.5, 1.0, 1.5) and of order 10⁻¹⁶·N otherwise (0.3, 0.8, 1.3), as the
   runs show.
2. *Equality in Weyl.* H1 = I shifts every eigenvalue by exactly c: |λ_j(H0 + cI) − λ_j(H0)| = |c| = |c|‖I‖₂ (part A,
   adversarial). The tube widths B|c − c_k| cannot be reduced in general.
3. *Crossing at the Majumdar–Ghosh point.* For the periodic J1–J2 chain at J2 = J1/2 the two dimer coverings are
   exact degenerate ground states with momenta 0 and π (Majumdar & Ghosh 1969), energy −3NJ1/8 = −3 for N = 8; the
   k = 0 and k = π block ground states computed at J2 = 0.5 are both −3.000000000000 and the certified crossing
   interval contains 0.5.
4. *Limiting case B → 0.* With H1 = 0 (declared) the family is constant on each segment, B = 0 (up to one ulp),
   the left tube is the certificate's interval for all c, and the right tube is that of the next grid point widened
   by d_k = ‖H(c_{k+1}) − H(c_k)‖₂: their intersection contains λ_j(H(c_k)) (test), as Proposition 1 states for
   F_k ≡ H(c_k).

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (1 049 random + 107 adversarial cases, 0 violations; negative
controls detected); independent re-derivation: done 2026-10-30 (fresh-context session, verdict ERROR, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
