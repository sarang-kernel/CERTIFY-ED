# Proposition 14: the number of eigenvalues in a bracket around a cluster (S8)

Implements: `certify_ed/degeneracy.py` (`degeneracy_counts`). Depends on Lemma 4 (`docs/theory.md`, Section 3:
an inertia count with backward error) and Theorem 2 (Section 2: distinct eigenvalues for a cluster of
certified pairs). Notation as in `docs/theory.md`, Section 9.1.

## 1. Statement

**Setting.** A is Hermitian with eigenvalues λ_1 ≤ … ≤ λ_n (with multiplicity); #S denotes the number of
eigenvalues, counted with multiplicity, in a set S ⊂ R. A cluster C of g claimed pairs is given with:

(H1) *Theorem 2 for C.* The union J = [J_L, J_U] of the g certified enclosures (radii widened by the model
shift ε when the operator used differs from the declared model by at most ε in norm) contains at least g
eigenvalues of A counted with multiplicity (the g pairs are "certified distinct": Theorem 2 attaches g
different eigenvalues to them).

(H2) *Lemma 4 at two shifts.* Shifts σ_lo < J_L and σ_hi > J_U with conclusive inertia counts ν_lo, ν_hi and
backward-error bounds β_lo, β_hi ≥ 0 (each already including ε) such that
  #{λ < σ − β} ≤ ν ≤ #{λ ≤ σ + β}   for (σ, ν, β) = (σ_lo, ν_lo, β_lo) and (σ_hi, ν_hi, β_hi).

Define I_in = (σ_lo + β_lo, σ_hi − β_hi) (open) and I_out = [σ_lo − β_lo, σ_hi + β_hi] (closed).

**Proposition 14.**
(a) ν_hi − ν_lo ≤ #I_out.
(b) If I_in ≠ ∅, i.e. σ_lo + β_lo < σ_hi − β_hi, then #I_in ≤ ν_hi − ν_lo.
(c) If J ⊂ I_in (which implies I_in ≠ ∅) and ν_hi − ν_lo = g, then #I_in = g: I_in contains exactly g
eigenvalues, and they are the eigenvalues of (H1).
(d) If J ⊂ I_in and ν_hi − ν_lo > g, then I_out contains at least ν_hi − ν_lo > g eigenvalues, so at least
ν_hi − ν_lo − g eigenvalues beyond those of (H1).
(e) If J ⊂ I_in, then ν_hi − ν_lo < g is impossible under (H1)–(H2): it certifies that one of the
hypotheses is false (the code reports FAILED).

The certified statement of (c) is a *count in an interval*: "exactly g eigenvalues in I_in". Nothing is
said about the equality of those eigenvalues; see C-16 (`docs/proofs/C16_exact_bracket.md`) for what is
decidable about equality.

## 2. Proof

For reals x ≤ y, #{λ < y} − #{λ ≤ x} = #(x, y) (eigenvalues strictly between x and y) and
#{λ ≤ y} − #{λ < x} = #[x, y]: both are immediate from counting the sorted eigenvalues (the first drops
those ≤ x from those < y; the second keeps both ends).

(a) By (H2), ν_hi ≤ #{λ ≤ σ_hi + β_hi} and ν_lo ≥ #{λ < σ_lo − β_lo}. Subtracting,
ν_hi − ν_lo ≤ #{λ ≤ σ_hi + β_hi} − #{λ < σ_lo − β_lo}. Since σ_lo < σ_hi and β_lo, β_hi ≥ 0 we have
σ_lo − β_lo < σ_hi + β_hi, so the right side equals #[σ_lo − β_lo, σ_hi + β_hi] = #I_out.

(b) By (H2), ν_hi ≥ #{λ < σ_hi − β_hi} and ν_lo ≤ #{λ ≤ σ_lo + β_lo}. Subtracting,
ν_hi − ν_lo ≥ #{λ < σ_hi − β_hi} − #{λ ≤ σ_lo + β_lo}, and with x = σ_lo + β_lo < y = σ_hi − β_hi the
right side is #(x, y) = #I_in. (Without x < y the right side could be negative and the inequality
ν_hi − ν_lo ≥ #I_in = 0 would not follow from (H2); the hypothesis I_in ≠ ∅ is therefore part of the
statement. In the code it is implied by the test J ⊂ I_in, which precedes every use of (b).)

(c) J ⊂ I_in and (H1) give #I_in ≥ #J ≥ g. By (b), #I_in ≤ ν_hi − ν_lo = g. Hence #I_in = g. The g
eigenvalues of (H1) lie in J ⊂ I_in and are g eigenvalues counted with multiplicity; since I_in holds
exactly g, they are all of them.

(d) By (a), #I_out ≥ ν_hi − ν_lo > g; the g eigenvalues of (H1) lie in J ⊂ I_in ⊂ I_out, so at least
ν_hi − ν_lo − g eigenvalues of I_out are not among them.

(e) As in (c), #I_in ≥ g; by (b), #I_in ≤ ν_hi − ν_lo < g. Contradiction. ∎

*Classical results cited.* Lemma 4 is Sylvester's law of inertia plus Weyl's inequality (Horn & Johnson,
*Matrix Analysis*, 2nd ed., Thm 4.5.8 (Sylvester) and Cor 4.3.15 (Weyl)); the ε-widening of β is Weyl's
inequality applied to the model shift (eigenvalues of A and of the declared H differ by at most ‖A − H‖ ≤ ε,
so #{λ(H) < σ − β − ε} ≤ #{λ(A) < σ − β} and #{λ(A) ≤ σ + β} ≤ #{λ(H) ≤ σ + β + ε}). Theorem 2 is
`docs/theory.md` Section 2 (Kahan's theorem for clusters).

## 3. Where the code implements it

`certify_ed/degeneracy.py` (line numbers of this revision, 105 lines):

* (H1): `rad = add_up(enc.radius[C], eps_shift)`; J_L = `min(sub_down(mu, rad))`, J_U = `max(add_up(mu, rad))`
  (lines 57–59): J is rounded outward, so the exact J is inside the computed one. `distinct` (61–65)
  requires every pair of C to belong to a Theorem 2 group with `distinct = True`; otherwise INCONCLUSIVE
  ((H1) unavailable).
* (H2): `_probe` (30–45) obtains a conclusive count at a shift beyond J (direction −1 below J_L, +1 above
  J_U), retrying up to three times with the shift moved outward when β + ε is not smaller than the distance
  from the shift to J; `b_lo = add_up(r_lo.beta, eps_shift)`, `b_hi = add_up(r_hi.beta, eps_shift)` (82–83)
  are β_lo, β_hi (upper bounds, rounded up). The inequality form of Lemma 4 is the one `inertia.py` states in
  its module docstring and `InertiaResult.negative` is ν.
* I_in = `(add_up(s_lo, b_lo), sub_down(s_hi, b_hi))` (84): the computed I_in is a *subset* of the exact
  one (left end rounded up, right end rounded down), so #(computed I_in) ≤ #(exact I_in) and (b) holds a
  fortiori for the computed interval; J ⊂ computed I_in implies J ⊂ exact I_in. I_out = `(sub_down(s_lo,
  b_lo), add_up(s_hi, b_hi))` (85) is a *superset* of the exact one, so (a) holds for it a fortiori.
* `diff = r_hi.negative - r_lo.negative` (86) is ν_hi − ν_lo.
* Decision (89–104): `if not (I_in[0] < JL and JU < I_in[1])` → INCONCLUSIVE (J ⊄ I_in: neither (c) nor (e)
  applies); `elif diff == g` → CERTIFIED with the statement "exactly g eigenvalues in the open interval
  (I_in[0], I_in[1])" (c); `elif diff > g` → INCONCLUSIVE with `degeneracy_at_least = g` and the reason naming
  I_out (d); `else` → FAILED (e).

The one wording issue found: the 3.1 statement printed I_in with square brackets although I_in is open; the
statement now says "in the open interval" to match the proof (the computed ends are rounded inward, so the
closed computed interval is in any case inside the exact open I_in).

## 4. Computational verification

`docs/proofs/verify_P14.py` (run: `python docs/proofs/verify_P14.py --random 3000 --seed 0`):

* Part A, the statement in exact rational arithmetic: 3000 random spectra (2–11 rationals), a cluster of
  g ≤ 4 consecutive eigenvalues with J their hull widened at random, random shifts and backward errors; the
  counts ν are produced by an adversary that moves every eigenvalue by ±β, by a random amount within β, or
  exactly onto the shift (where the strict inequality of Lemma 4 matters). Checked: Lemma 4 holds for the
  adversary's counts (sanity of the model), (a), (b) when I_in ≠ ∅, and the decision rules (c)–(e) when
  J ⊂ I_in, each against the exact counts. 1000 adversarial cases place the shifts exactly β away from J
  (J ⊂ I_in fails by equality on one side), or put the neighbouring eigenvalue exactly at σ_lo + β_lo or
  σ_hi − β_hi (the ends of I_in).
* Part B, the implementation on diagonal matrices (exact spectrum = the diagonal): 60 random matrices with
  planted levels of multiplicity 1–3 and, in half of them, a near-degenerate pair split by 10^−12 … 10^−6,
  clustered with a random tolerance 10^−9 … 10^−3; every CERTIFIED record's "exactly g in I_in" is checked
  against the diagonal (246 CERTIFIED, 4 INCONCLUSIVE records, all consistent). Six fixed cases: a
  truncated triplet (INCONCLUSIVE, at least 2, count 3); a pair split by 10^−9 claimed as one cluster
  (CERTIFIED "2 in the bracket", true, width ≥ 10^−9 reported) and as two singlets (INCONCLUSIVE, since the
  minimal shift margin 10^−8·scale exceeds the splitting: both eigenvalues are in each bracket); the same
  split by 10^−6 as two singlets (two CERTIFIED singlets, each bracket holding exactly one); a model shift
  ε = 10^−10 (CERTIFIED, both β ≥ ε); an omitted member inside J (never CERTIFIED with g = 2).

Result (seed 0): 3060 random + 1006 adversarial cases, 0 violations.

## 5. Analytic cross-check

*Exact counts (β = 0).* With β_lo = β_hi = 0 Lemma 4 is an equality, ν(σ) = #{λ < σ}, and (a)–(b) collapse
to ν_hi − ν_lo = #[σ_lo, σ_hi) = #(σ_lo, σ_hi) when no eigenvalue sits at σ_lo: the classical
spectrum-slicing count (Parlett, *The Symmetric Eigenvalue Problem*, §3.3). The proposition is this count
with the two shifts replaced by the windows [σ ± β] that the backward error allows.

*Reduction to Corollary 5.* For σ_lo → −∞ (ν_lo = 0, β_lo = 0) the proposition is the completeness
statement of S3 for a 'lowest' claim: #(−∞, σ_hi − β_hi) ≤ ν_hi ≤ #(−∞, σ_hi + β_hi], which is Corollary 5 of
`docs/theory.md`.

*Diagonal matrices.* For a diagonal A the inertia at σ is the exact number of diagonal entries below σ up
to the factorization's β (of order 10^−15 for the dense path); the brackets of Part B are 10^−8 wide and the
counts reproduce the planted multiplicities exactly, as Section 4 records.

## 6. Status

Derived by Claude on 2026-10-30; machine-checked (3060 random + 1006 adversarial cases, 0 violations);
independent re-derivation: done 2026-10-30 (fresh-context session, verdict AGREES, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
