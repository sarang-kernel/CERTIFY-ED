# Corollary 5 (completeness of a 'lowest' claim and the pairing of claimed values with λ₁ … λ_k)

Derivation document in the D-59 format. Notation as in `docs/theory.md`: A_u is the Hermitian operator the
enclosures and the factorization were computed with (the producer's Hermitian part, or the sector block);
A_m is the declared model operator (the exact spec operator on the same basis), with ‖A_u − A_m‖ ≤ Δ
(Δ = `eps_shift`, 0 when A_u = A_m); eigenvalues λ₁(·) ≤ … ≤ λₙ(·).

## 1 Statement

**Hypotheses.** k claimed values μ₁ ≤ … ≤ μ_k (sorted), each with a radius r_j ≥ 0, and

- (H1, S2 "all distinct") a partition of {1, …, k} into groups G₁, …, G_p such that (a) within a group all
  radii are equal (r_j = r_G for j ∈ G); (b) for every group there is an injection j ↦ i_j of G into
  {1, …, n} with |λ_{i_j}(A_u) − μ_j| ≤ r_G; (c) the union intervals I_G := [min_{j∈G}(μ_j − r_G), max_{j∈G}(μ_j + r_G)]
  of different groups are pairwise disjoint;
- (H2, Lemma 4 at the shift σ) an integer ν and β ≥ 0 with #{i : λ_i(A_u) < σ − β} ≤ ν ≤ #{i : λ_i(A_u) ≤ σ + β},
  and ν = k;
- (H3, the cut) max_j (μ_j + r_j + Δ) < σ − β − Δ.

**Conclusion.** For j = 1, …, k, |λ_j(A_m) − μ_j| ≤ r_j + Δ (the j-th smallest eigenvalue of the declared model
lies in the j-th sorted enclosure, widened by Δ); exactly k eigenvalues of A_m lie below σ − β − Δ;
λ_{k+1}(A_m) ≥ σ − β − Δ; and λ_{k+1}(A_m) − λ_k(A_m) ≥ (σ − β − Δ) − (μ_k + r_k + Δ).

**Lemma (sorted pairing).** Let a₁ ≤ … ≤ a_m and b₁ ≤ … ≤ b_m be real numbers and r ≥ 0. If some bijection
π of {1, …, m} satisfies |b_{π(j)} − a_j| ≤ r for all j, then |b_j − a_j| ≤ r for all j.

**Remark (why the groups).** The lemma is false for non-uniform radii: a₁ = 0 with r₁ = 10, a₂ = 1 with
r₂ = 0.1, b = {1.05, 9}: the bijection a₁ ↦ 9, a₂ ↦ 1.05 respects the radii, the sorted pairing a₂ ↦ 9 does
not. Hypothesis (H1a) is what the enclosure provides: all members of a cluster group carry the cluster radius
(`enclosure.enclose`, `final_radius`), a singleton group has one radius, and an 'existence-only' group makes
`all_distinct` False.

## 2 Proof

**Sorted pairing lemma.** If π is not the identity there are indices a < b (so a_a ≤ a_b) with
b_{π(a)} > b_{π(b)} (an inversion). Then both |b_{π(b)} − a_a| ≤ r and |b_{π(a)} − a_b| ≤ r: indeed
b_{π(b)} − a_a < b_{π(a)} − a_a ≤ r and a_a − b_{π(b)} ≤ a_b − b_{π(b)} ≤ r; likewise b_{π(a)} − a_b ≤ b_{π(a)} − a_a ≤ r
and a_b − b_{π(a)} < a_b − b_{π(b)} ≤ r. So the bijection π′ = π ∘ (a b) also respects r and has one inversion
fewer. After finitely many exchanges the bijection is the identity on sorted indices, which is the sorted
pairing. ∎ (A standard exchange argument; e.g. the bottleneck matching of sorted sequences.)

**Step 1 (one injection for all claimed values).** By (H1b) each group has its injection; by (H1c) the matched
eigenvalue of j ∈ G lies in I_G (|λ_{i_j} − μ_j| ≤ r_G puts it inside [μ_j − r_G, μ_j + r_G] ⊂ I_G), and the I_G are
disjoint, so two different groups never use the same eigenvalue index. The union of the group injections is
an injection φ of {1, …, k} into {1, …, n} with |λ_{φ(j)}(A_u) − μ_j| ≤ r_j.

**Step 2 (the matched eigenvalues are λ₁ … λ_k of A_u).** For every j, λ_{φ(j)}(A_u) ≤ μ_j + r_j < σ − β − 2Δ ≤ σ − β
by (H3). Hence #{i : λ_i(A_u) < σ − β} ≥ k. By (H2) it is ≤ ν = k. So exactly k eigenvalues of A_u lie below
σ − β; they are λ₁(A_u), …, λ_k(A_u) (the k smallest), the image of φ is exactly this index set, and
λ_{k+1}(A_u) ≥ σ − β.

**Step 3 (sorted pairing).** Order the groups by their union intervals, I_{G(1)} < I_{G(2)} < … (disjoint closed
intervals are totally ordered). Every claimed value of G(q) and every eigenvalue matched to G(q) lies in
I_{G(q)}, so in the sorted order of the k claimed values and in the sorted order λ₁(A_u) ≤ … ≤ λ_k(A_u) the
members of G(1) come first, then those of G(2), and so on, with the same counts. Within G(q), φ restricted to
G(q) is a bijection onto the eigenvalues at the corresponding sorted positions, respecting the uniform radius
r_{G(q)} (H1a); the sorted pairing lemma gives |λ_j(A_u) − μ_j| ≤ r_j for every sorted position j in G(q).
Hence |λ_j(A_u) − μ_j| ≤ r_j for j = 1, …, k.

**Step 4 (transfer to the declared model).** A_u and A_m are Hermitian with ‖A_u − A_m‖ ≤ Δ, so Weyl's
inequality (Horn–Johnson, *Matrix Analysis*, 2nd ed., Cor. 4.3.15) gives |λ_j(A_m) − λ_j(A_u)| ≤ Δ for every j.
Therefore |λ_j(A_m) − μ_j| ≤ r_j + Δ for j ≤ k; λ_{k+1}(A_m) ≥ λ_{k+1}(A_u) − Δ ≥ σ − β − Δ; for j ≤ k,
λ_j(A_m) ≤ μ_j + r_j + Δ < σ − β − Δ by (H3), so exactly k eigenvalues of A_m lie below σ − β − Δ; and
λ_{k+1}(A_m) − λ_k(A_m) ≥ (σ − β − Δ) − (μ_k + r_k + Δ). ∎

**Remarks on the hypotheses.** (1) Only the *lower* inequality of Lemma 4 is used (ν bounds the count below
σ − β from above). (2) (H3) needs the strict inequality only to make "exactly k below σ − β − Δ" a statement
about an open half-line; the code checks it on outward-rounded numbers. (3) Nothing is assumed about
eigenvalues above σ + β. (4) Δ enters twice in (H3) because the enclosures and the count both move by Δ.

## 3 Where the code implements it

`certify_ed/certify.py`, `_certify_lowest(A, mu_sorted, enc, opts, eps_shift)` (branch `track/backlog-proofs`).
`A` = A_u (the Hermitian part of the producer's matrix, `certify()`; for a sector claim the sector block, with
Δ = 0 since S3 then refers to the block); `eps_shift` = Δ (`eps_s3`, the S1 model shift,
`docs/proofs/appA_model-shift.md`).

| hypothesis / quantity | code | direction |
|---|---|---|
| r_j + Δ | `rad = add_up(enc.radius[argsort(enc.mu)], eps_shift)` | ≥ exact sum (one ulp outward) |
| μ_j + r_j + Δ | `hi = add_up(mu_sorted, rad)`; `hi_k = hi.max()` | ≥ exact |
| (H1) | `distinct = enc.all_distinct`: every group `distinct` (method 'single' or 'cluster', finite radii) and the safety net of `enclosure.enclose` (union intervals, formed with `sub_down`/`add_up`, pairwise disjoint); cluster members share `final_radius`; the injection is Theorem 2 / Theorem 2′ (`docs/proofs/thm2_polar-factor.md`, `thm2prime_eps-star.md`), a singleton's is Theorem 1 | — |
| (H2) | `r = inertia_count(A, sigma, ...)`, `r.negative` = ν, `r.beta` = β (Lemma 4, `inertia.py`; `r.conclusive` required) | β is an upper bound on the backward error |
| β + Δ, σ − β − Δ | `beta = add_up(r.beta, eps_shift)`; `s_minus_b = sub_down(sigma, beta)` | ≤ exact σ − β − Δ |
| (H3) | `hi_k < s_minus_b` | since `hi_k` ≥ max_j(μ_j + r_j + Δ) and `s_minus_b` ≤ σ − β − Δ, the exact (H3) follows |
| CERTIFIED | `if r.negative == k and distinct and hi_k < s_minus_b` | — |
| λ_{k+1}(A_m) ≥ | `next_eigenvalue_lower_bound = s_minus_b` | ≤ exact bound |
| gap | `gap_lower_bound = sub_down(s_minus_b, hi_k)` | ≤ (σ − β − Δ) − (μ_k + r_k + Δ) |
| window | `certified_window = ["-inf", s_minus_b]` | — |

The exported S3 statement ("exactly k eigenvalues below `next_eigenvalue_lower_bound`") is the count part of
the conclusion; the pairing part (λ_j in the j-th enclosure) is what `gaps.py` (C-01) and the S4/S8 statements
use through `enc.radius` with the model shift.

Other branches of `_certify_lowest` (FAILED when `_missing_below` finds ν_low > #enclosures reaching below
s_low + β_low; INCONCLUSIVE otherwise) make no completeness statement and are outside this document;
their soundness is a count inequality of Lemma 4 alone (an eigenvalue counted below s_low + β that no
enclosure can account for is unclaimed), and the task's negative controls exercise them.

## 4 Computational verification

`docs/proofs/verify_cor5_pairing.py`:

- P (exact rationals, `fractions.Fraction`): random sorted lists a, b with a planted bijection respecting a
  uniform r → the sorted pairing respects r (the lemma); adversarial: non-uniform radii (the lemma's
  conclusion is *expected to fail* in some cases; the script reports how often it fails and treats that as
  the documented counterexample, not as a violation); ties, m = 1, r = 0.
- C (the code): random Hermitian A (n ≤ 9), a 'lowest-k' claim with planted clusters of non-orthonormal
  vectors (so that some groups have g ≥ 2 with a uniform cluster radius) and optional perturbation of the
  claimed values; `enclose` + `_certify_lowest(A, mu_sorted, enc, Options(), eps_shift)`. When the result is
  CERTIFIED, the conclusion is checked against the eigenvalues of A + E computed with mpmath (30 digits) for
  a planted Hermitian E with ‖E‖₂ ≤ Δ (E = 0 when Δ = 0): λ_j(A + E) ∈ [μ_j − r_j − Δ, μ_j + r_j + Δ] for j ≤ k,
  exactly k eigenvalues below `next_eigenvalue_lower_bound`, λ_{k+1} ≥ that bound, the gap bound. Negative
  controls (adversarial): a missing eigenvalue (the claim skips λ_j, j ≤ k), a duplicated vector, a claimed
  value moved by more than its radius, a shift Δ smaller than the planted ‖E‖ (then the check is against the
  model the code was told about, and a CERTIFIED result for the wrong Δ is *not* a violation of the corollary
  but is counted separately) — the first three must not be CERTIFIED.

Counts (run of 2026-10-30, `--cases 20000 --seed 3`, output in `docs/proofs/runs/cor5_pairing.txt`): P 20 000
random + 500 adversarial (ties); C 400 code cases of which 347 were CERTIFIED and checked (53 skipped: the code
returned INCONCLUSIVE or FAILED, which is not a statement); N 400 negative controls (200 missing eigenvalue, 200
duplicated vector), none CERTIFIED; 0 violations. Documented counterexample: non-uniform radii broke the
sorted pairing in 238 of 2 000 cases. Informational: with ‖E‖ = 10⁻³ against a declared Δ = 10⁻¹² the conclusion
failed for A + E in 100 of 100 CERTIFIED cases (Δ is necessary).

## 5 Analytic cross-check

(i) *k = 1.* One claimed value μ with radius r: Theorem 1 gives an eigenvalue in [μ − r, μ + r]; ν = 1 at σ with
μ + r < σ − β says at most one eigenvalue lies below σ − β; so λ₁ ∈ [μ − r, μ + r] and λ₂ ≥ σ − β. The corollary
reduces to the two-sided statement a single Lemma 4 count gives for a single enclosure.

(ii) *All claimed values in one cluster.* p = 1, G = {1, …, k} with uniform r: Theorem 2 gives an injection into
k distinct eigenvalues within r; Step 2 identifies them with λ₁ … λ_k; the lemma sorts them. The conclusion
λ_j ∈ [μ_j − r, μ_j + r] is then exactly the "level enclosure" of the exact layer (Proposition 13, m_j ≥ 2 case)
with the cluster radius in place of the bracket.

(iii) *Δ → 0.* Step 4 is void and the conclusion is the statement for A_u itself; Δ enters linearly, which is
what Weyl's inequality gives (no factor, no square).

(iv) *The theory.md form.* "If k claimed values have distinct enclosures inside (−∞, σ − β) and ν = k, they
enclose λ₁ … λ_k, and λ_{k+1} ≥ σ − β" is the case Δ = 0 of the conclusion; "distinct enclosures" is (H1), and
the proof shows that the pairing is the sorted one only because the groups' intervals are disjoint and
cluster radii are uniform (Remark in section 1).

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (20 347 random + 900 adversarial cases, 0 violations); independent re-derivation: done 2026-10-30 (fresh-context session, verdict AGREES, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
