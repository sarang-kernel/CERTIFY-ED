# Independent re-derivation: Proposition 14 (number of eigenvalues in a bracket around a cluster)

Fresh-context session, 2026-10-30. Sections 1 and 3 of `../P14_degeneracy_bracketing.md` read first, with
`certify_ed/degeneracy.py`; proof written before section 2 was opened. Notation: N(x) := #{i : λ_i < x},
N̄(x) := #{i : λ_i ≤ x} (counts with multiplicity); for a < b, #(a, b) = N(b) − N̄(a) and #[a, b] = N̄(b) − N(a).

## Statement checked

(a)–(e) of section 1 under (H1) (J holds ≥ g eigenvalues, attached to the g pairs) and (H2) (Lemma 4 at σ_lo < J_L
and σ_hi > J_U: N(σ − β) ≤ ν ≤ N̄(σ + β)).

## My proof

(a) ν_hi − ν_lo ≤ N̄(σ_hi + β_hi) − N(σ_lo − β_lo) = #[σ_lo − β_lo, σ_hi + β_hi] = #I_out (the upper inequality of
(H2) at σ_hi, the lower one at σ_lo; I_out is non-empty since σ_lo < σ_hi). ✓

(b) ν_hi − ν_lo ≥ N(σ_hi − β_hi) − N̄(σ_lo + β_lo) (the lower inequality at σ_hi, the upper one at σ_lo). When
σ_lo + β_lo < σ_hi − β_hi this difference is #(σ_lo + β_lo, σ_hi − β_hi) = #I_in. ✓ (Without I_in ≠ ∅ the
difference N(b) − N̄(a) with a ≥ b can be negative and says nothing.)

(c) J ⊂ I_in gives I_in ≠ ∅ and, with (H1), #I_in ≥ #J ≥ g. By (b), #I_in ≤ ν_hi − ν_lo = g. Hence #I_in = g. The
g eigenvalues of (H1) lie in J ⊂ I_in and are g distinct indices, so they exhaust the count: the eigenvalues in
I_in are exactly those attached to the pairs. ✓

(d) By (a), #I_out ≥ ν_hi − ν_lo > g. The g eigenvalues of (H1) are among the eigenvalues of I_out (J ⊂ I_in ⊂
I_out), so I_out holds at least ν_hi − ν_lo − g eigenvalues beyond them. ✓

(e) J ⊂ I_in and (H1) give #I_in ≥ g, and (b) gives ν_hi − ν_lo ≥ #I_in ≥ g. So ν_hi − ν_lo < g is inconsistent
with (H1)–(H2) together: at least one of them is false (a wrong count, a wrong backward error, or pairs that are
not attached to g distinct eigenvalues). ✓ ∎

### Code (current `degeneracy.py`, 105 lines; lines as in section 3, unchanged)

J: `rad = add_up(radius, eps)`, JL = min `sub_down`, JU = max `add_up` (57–59) — the computed J contains the exact
one; `distinct` (61) requires every pair's Theorem 2 group to be distinct. For a cluster assembled from *several*
enclosure groups (`cluster_groups` merges consecutive values closer than `cluster_tol`), (H1) needs the g matched
eigenvalues to be distinct across groups: that holds because distinct groups have pairwise disjoint union
intervals (`enclose`'s safety net, enclosure.py 614–625, marks a group non-distinct otherwise) and each group's
matched eigenvalues lie in its own union interval — a step the proof should state (it is implicit in "certified
distinct"). Shifts: `_probe` (30–45) returns σ with the computed β + ε smaller than the outward-rounded distance
to J (so the inner bracket can contain J), s_lo < JL, s_hi > JU. β_lo, β_hi = `add_up(beta, eps)` (82–83) are upper
bounds on the backward error plus the model shift: Lemma 4 with a larger β is weaker, hence still true, and
Theorem A5 of the model-shift document transfers the count to the declared model with β + ε. Computed I_in
(84) = (add_up(s_lo, b_lo), sub_down(s_hi, b_hi)) ⊆ exact I_in; computed I_out (85) ⊇ exact I_out; `diff` (86) =
ν_hi − ν_lo. Decision (89–103): J ⊄ computed I_in → INCONCLUSIVE (then J ⊂ exact I_in is not established);
diff == g → CERTIFIED: by (c) the exact I_in holds exactly g eigenvalues; the computed I_in ⊆ exact I_in still
contains J, so it holds ≥ g and ≤ g, i.e. exactly g — the printed statement about the *computed* open interval
is correct; diff > g → INCONCLUSIVE with (d) on the computed I_out ⊇ exact I_out (a fortiori); diff < g →
FAILED by (e). Rounding directions all as section 3 states. ✓

## Comparison with section 2 of the original (read after the proof above was written)

Identical in every step (the two counting identities, (a)–(e) in the same order with the same role of I_in ≠ ∅,
Sylvester + Weyl for Lemma 4 and the ε-widening). Same constants. Both valid.

**Remark.** The multi-group case of (H1) (a cluster assembled by `cluster_groups` from several Theorem 2 groups:
the g matched eigenvalues are distinct across groups because the groups' union intervals are pairwise disjoint,
enforced by `enclose`'s safety net) is implicit in "certified distinct" in both the statement and the proof;
one sentence would make it explicit. Code mapping as in section 3 (current file unchanged, 105 lines).

**Verdict: AGREES.**

## Runs

- `verify_P14.py --random 600 --seed 7`: model 600 random + 200 adversarial, code 20 random (75 CERTIFIED
  statements) + 6 adversarial (truncated multiplet, non-degenerate pair, pair as singlets in/outside the margin,
  model shift, omitted member); 0 violations.
- `adversarial_P14_degeneracy_bracketing.py --cases 300 --seed 37`: (a)–(e) exact 300 random + 900 adversarial
  (ν anywhere in the Lemma 4 range, eigenvalues exactly at σ ± β, I_in empty, J touching I_in, g = multiplicity
  or less); code 39 random + 123 adversarial (47 not CERTIFIED: close levels / shift cases), every CERTIFIED
  count checked against the exact eigenvalues of the declared model A + E; negative controls 77 (duplicated
  vector, omitted level member) never certified with the claimed g; 0 violations.
