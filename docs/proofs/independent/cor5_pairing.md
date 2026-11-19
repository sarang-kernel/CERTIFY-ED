# Independent re-derivation: Corollary 5 (completeness and sorted pairing)

Written from the statement alone on 2026-10-30 (not a human check). Sections 1 and 3 of `../cor5_pairing.md` read first; proof written before
section 2 was opened. Notation: λ_i := λ_i(A_u) ascending; A_m the model operator with ‖A_u − A_m‖ ≤ Δ.

## Statement checked

Hypotheses (H1a–c), (H2), (H3) as in section 1; conclusions: |λ_j(A_m) − μ_j| ≤ r_j + Δ for j ≤ k, exactly k
eigenvalues of A_m below σ − β − Δ, λ_{k+1}(A_m) ≥ σ − β − Δ, gap ≥ (σ − β − Δ) − (μ_k + r_k + Δ). Lemma (sorted
pairing) and the remark that it fails for non-uniform radii.

## My proof

### Sorted pairing lemma

a₁ ≤ … ≤ a_m, b₁ ≤ … ≤ b_m, bijection π with |b_{π(j)} − a_j| ≤ r. Fix j. If b_j > a_j + r, then for every
i ≤ j, b_{π(i)} ≤ a_i + r ≤ a_j + r < b_j, so π({1, …, j}) ⊂ {l : b_l < b_j} ⊂ {1, …, j − 1}: j distinct images in
j − 1 slots — impossible. If b_j < a_j − r, then for every i ≥ j, b_{π(i)} ≥ a_i − r ≥ a_j − r > b_j, so
π({j, …, m}) ⊂ {l : b_l > b_j} ⊂ {j + 1, …, m}: m − j + 1 images in m − j slots — impossible. Hence |b_j − a_j| ≤ r. ∎
(Counterexample of the remark re-checked: a = (0, 1), r = (10, 0.1), b = (1.05, 9); π: 0 ↦ 9, 1 ↦ 1.05 respects
the radii, the sorted pairing 1 ↦ 9 violates 0.1.)

### Corollary 5

*Step 1 (one injection for all claims).* By (H1b) each group G has an injection j ↦ i_j into eigenvalue indices
with |λ_{i_j} − μ_j| ≤ r_G, so λ_{i_j} ∈ [μ_j − r_G, μ_j + r_G] ⊂ I_G. By (H1c) the I_G are pairwise disjoint, so
the images of different groups are disjoint, and the union φ of the injections is an injection {1, …, k} → {1, …, n}
with |λ_{φ(j)} − μ_j| ≤ r_j (using H1a: r_j = r_G).

*Step 2 (the image is {1, …, k}).* By (H3) and Δ ≥ 0, λ_{φ(j)} ≤ μ_j + r_j < σ − β − 2Δ ≤ σ − β for every j, so the
k distinct indices φ(j) all lie in L := {i : λ_i < σ − β}. By (H2), |L| ≤ ν = k. Hence |L| = k and φ({1, …, k}) =
L. Since the λ_i are ascending, L is an initial segment: L = {1, …, k}, and λ_{k+1} ≥ σ − β (if λ_{k+1} < σ − β
then |L| ≥ k + 1). So φ is a permutation of {1, …, k} and λ_{k+1}(A_u) ≥ σ − β.

*Step 3 (φ preserves the groups).* Each group is a block of consecutive indices: for j₁ < l < j₂ with j₁, j₂ ∈ G,
μ_{j₁} ≤ μ_l ≤ μ_{j₂} ∈ I_G (an interval), so μ_l ∈ I_G; as μ_l also lies in I_{G(l)} and the I's are disjoint,
G(l) = G. Order the groups by the position of their (disjoint) intervals on the line; the index blocks G appear in
that order (μ_j < μ_{j′} when I_G lies left of I_{G′}). Set J_G := {i ≤ k : λ_i ∈ I_G}. Then φ(G) ⊂ J_G (Step 1),
and the J_G are disjoint, cover {1, …, k} (every i ≤ k equals φ(j) for some j, and λ_i ∈ I_{G(j)}) and are
blocks of consecutive indices in the same order (the λ_i are ascending and the intervals disjoint). From
|φ(G)| = |G| and φ(G) ⊂ J_G with Σ|J_G| = k = Σ|G| follows φ(G) = J_G, |J_G| = |G|. Two partitions of {1, …, k} into
consecutive blocks with equal sizes in the same order coincide: J_G = G. So φ restricted to G is a bijection of G
onto G.

*Step 4 (sorted pairing within a group).* On G, a = (μ_j)_{j∈G}, b = (λ_i)_{i∈G} (both ascending in the index),
uniform radius r_G, bijection φ|_G with |λ_{φ(j)} − μ_j| ≤ r_G. The lemma gives |λ_j(A_u) − μ_j| ≤ r_G = r_j for
all j ∈ G, hence for all j = 1, …, k.

*Step 5 (model shift).* Weyl's inequality (Horn–Johnson, *Matrix Analysis*, Cor. 4.3.15): |λ_j(A_m) − λ_j(A_u)| ≤
‖A_m − A_u‖ ≤ Δ for all j. Hence |λ_j(A_m) − μ_j| ≤ r_j + Δ (j ≤ k); λ_{k+1}(A_m) ≥ λ_{k+1}(A_u) − Δ ≥ σ − β − Δ;
for j ≤ k, λ_j(A_m) ≤ μ_j + r_j + Δ < σ − β − Δ by (H3), so exactly k eigenvalues of A_m lie below σ − β − Δ;
and λ_{k+1}(A_m) − λ_k(A_m) ≥ (σ − β − Δ) − (μ_k + r_k + Δ). ∎

Where the hypotheses enter: (H1a) only in Step 4 (the lemma needs a uniform radius; the remark's counterexample
shows it cannot be dropped); (H1c) in Steps 1 and 3; (H2) only through its first inequality #{λ < σ − β} ≤ ν
(the second inequality of Lemma 4 is not needed for the conclusion); (H3) in Steps 2 and 5, where the strictness
"<" is what makes the count below σ − β − Δ exact and the gap bound meaningful.

## Comparison with section 2 of the original (read after the proof above was written)

Lemma: the original uses an exchange argument on inversions (each exchange keeps the radius constraint and reduces
the number of inversions); I used a counting argument. Both valid. Steps 1, 2 and 4 of the original coincide with
my Steps 1, 2 and 5. The original's Step 3 states that the members of each group occupy the same sorted
positions among the claimed values and among λ₁ … λ_k "with the same counts"; my Step 3 spells out why
(φ(G) ⊂ J_G, Σ|J_G| = k, consecutive blocks in the same order). Valid, same constants (r_j + Δ, σ − β − Δ, gap).
The original's remark (1) (only the lower inequality of Lemma 4 is used) matches my observation.

**Code mapping (current `dev/3.2` lines, `certify._certify_lowest` 521–640).** rad 533 (`add_up`, ≥ r_j + Δ),
hi 534, hi_k 536, `distinct = enc.all_distinct` 540, inertia 583, beta 598 (`add_up(r.beta, eps_shift)`),
s_minus_b 599 (`sub_down`), CERTIFIED condition 600 (`r.negative == k and distinct and hi_k < s_minus_b`), exported
bound and gap 602–603 (`gap_lower_bound = sub_down(s_minus_b, hi_k)`, which is ≤ the document's bound since
hi_k ≥ μ_k + r_k + Δ). (H1a) and (H1c) in `enclosure.enclose` 597–626: cluster and single members share
`final_radius`, existence-only groups set `distinct = False`, and the safety net 614–625 marks a group non-distinct
when its outward-rounded union interval touches another (`Ls <= prefmax` or `Us >= nextL`; touching counts as a
clash, so exact disjointness follows). Operator consistency: `inertia_op = op if is_matrix(op) else Href`
(certify.py 1092) and `eps_s3 = model_shift if audit is None else 0.0` (1095) — the inertia is taken on the
operator the enclosures used (or on the spec matrix when the operator is applied matrix-free from the same spec).
Directions all outward. The section 3 table agrees with the code.

**Verdict: AGREES.**

## Runs

- `verify_cor5_pairing.py --cases 2000 --seed 7`: lemma 2000 random + 500 adversarial, Corollary 5 vs mpmath
  174 random (26 skipped), negative controls 200; 0 violations.
- `adversarial_cor5_pairing.py --cases 300 --seed 11`: lemma (exact rationals) 1786 random + 941 adversarial;
  Corollary 5 conclusion (exact rationals, model eigenvalues moved by ≤ Δ) 202 + 998; necessity 3/3 (breaking
  H1a: conclusion failed in 41 of 300 cases; H1c (touching): 277 of 300; H3 at equality: 42 of 300);
  `_certify_lowest` CERTIFIED outcomes vs mpmath eigenvalues of A + E 13 + 82 (85 skipped: not CERTIFIED);
  negative controls 107 (missing eigenvalue, duplicated vector) never CERTIFIED; 0 violations.
