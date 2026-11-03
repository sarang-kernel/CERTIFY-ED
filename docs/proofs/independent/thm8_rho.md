# Independent re-derivation: Theorem 8 (sector assembly with radius ρ)

Fresh-context session, 2026-10-30. Sections 1 and 3 of `../thm8_rho.md` read first; proof written before section 2
was opened. Notation as in the preamble of the original: W = [V₁ ⋯ V_m] (n × n when Σd_s = n), B = ⊕_s H_s,
η = ‖WᴴW − I‖ < 1, D ≥ ‖HW − WB‖, δ = 1 − √(1 − η), ρ = D + (‖H‖ + ‖B‖)δ, ‖H − H_spec‖ ≤ m, ε = ρ + m.

## Statement checked

Under (T1)–(T3): |λ_j(H_spec) − c₍ⱼ₎| ≤ r + ε for j ≤ K and λ_{K+1}(H_spec) ≥ E_cut − max_s β_s − ε.

## My proof

*Step 1 (spectrum of H against spectrum of B).* By (T1) W is square and ‖WᴴW − I‖ < 1, so WᴴW is positive
definite and the polar factor Q̂ = W(WᴴW)^{−1/2} is a square matrix with Q̂ᴴQ̂ = I, i.e. unitary. The polar-factor
lemma (`independent/thm2_polar-factor.md`, parts (b)–(d) with X = W, Λ = B, A = H) gives
‖HQ̂ − Q̂B‖ ≤ ‖HW − WB‖ + (‖H‖ + ‖B‖)δ ≤ D + (‖H‖ + ‖B‖)δ = ρ. Since Q̂ is unitary, Q̂ᴴHQ̂ − B = Q̂ᴴ(HQ̂ − Q̂B) has
norm ≤ ρ, and Q̂ᴴHQ̂ is Hermitian with the same eigenvalues as H. Weyl's inequality (Horn–Johnson, *Matrix
Analysis*, Cor. 4.3.15) for the Hermitian pair (Q̂ᴴHQ̂, B): |λ_j(H) − λ_j(B)| ≤ ρ for all j = 1, …, n.
With ‖H − H_spec‖ ≤ m and Weyl once more: |λ_j(H_spec) − λ_j(B)| ≤ ρ + m = ε for all j.

*Step 2 (the K lowest eigenvalues of B).* The spectrum of B is the multiset union of the spectra of the H_s. By (T2)
and Corollary 5 (`independent/cor5_pairing.md`, Δ = 0, σ = E_cut, ν_s = k_s): for each s, |λ_i(H_s) − c_{s,i}| ≤
r_{s,i} ≤ r for i ≤ k_s, and λ_{k_s+1}(H_s) ≥ E_cut − β_s. Put E* := E_cut − max_s β_s. By (T3), λ_i(H_s) ≤
c_{s,i} + r < E* for i ≤ k_s, and λ_i(H_s) ≥ E_cut − β_s ≥ E* for i > k_s. Hence the multiset
S := {λ_i(H_s) : s, i ≤ k_s} (K elements) consists exactly of the eigenvalues of B below E*, and all other
eigenvalues of B are ≥ E*: λ₁(B) ≤ … ≤ λ_K(B) is S sorted and λ_{K+1}(B) ≥ E*.

*Step 3 (sorted pairing of the pooled claims with S).* The map c_{s,i} ↦ λ_i(H_s) is a bijection between the K
pooled claimed values and S with every distance ≤ r (a uniform radius). The sorted-pairing lemma
(`independent/cor5_pairing.md`) gives |λ_j(B) − c₍ⱼ₎| ≤ r for j = 1, …, K.

*Step 4.* |λ_j(H_spec) − c₍ⱼ₎| ≤ |λ_j(H_spec) − λ_j(B)| + |λ_j(B) − c₍ⱼ₎| ≤ ε + r for j ≤ K, and
λ_{K+1}(H_spec) ≥ λ_{K+1}(B) − ε ≥ E* − ε = E_cut − max_s β_s − ε. ∎

Where the hypotheses enter: (T1) for the unitary polar factor (a non-square W gives only Kahan's distinct
matching, not the sorted pairing of all n eigenvalues); (T2) for the per-sector pairing and the count; (T3) for
S being an initial segment of the spectrum of B — without it an unclaimed eigenvalue of one block in
[E*, E_cut − β_s) could lie below a claimed value of another block. The uniform radius r in Step 3 is essential
(the lemma fails for non-uniform radii), which is why the code pools with r = max of all sector radii.

## Comparison with section 2 of the original (read after the proof above was written)

The original orders the steps differently (sectors first, then B against H, then H against H_spec) but the
content is the same: Corollary 5 per sector, the initial-segment argument for the spectrum of B using (T3), the
sorted-pairing lemma with the uniform radius r, the unitary polar factor of the square W with the polar-factor
lemma (d), Weyl twice. Same constants ρ = D + (‖H‖ + ‖B‖)δ and ε = ρ + m. Both proofs valid. The original's
remark (2) (‖H‖ for the direct sum of U(1) block builds is the maximum of the block norms) is correct and is
what the code does.

**Code mapping (current `dev/3.2`, `compose.certify_low_energy` 51–178).** Hermitian parts 54, 73, 78; ‖B‖ 74,
‖H‖ 57/83 (`inf_norm_upper`, max over blocks); per-sector (T2) 97–118 (`inside` with `add_up`/`sub_down`,
`distinct`, `r.negative == mu_below.size`, status COMPLETE; rmax 117; beta_max 118); η 129–134 (sparse Gram with
the γ_dot(wmax)‖W‖_F² rounding term; `covered != n` and `not eta < 1` → FAILED, 147–152); D 137
(`norm2_upper` of the invariance defects, ≥ ‖HW − WB‖_F by the block-column structure); (T3) 157 (outward);
δ, ρ 167–168; ε 169 (one rounding inside `up`); radius and next bound 173–174 (outward). All directions as the
table states. `audit_sector` (sectors.py 51–71) adds the rounding term of the sparse products to `iso` and `inv`
(lines 60, 65–67), so D and η are bounds for the *stored* V_s.

**Remark (outside the certified path).** `sectors.decomposition_shift` (125–148) evaluates the same ρ from a
`DecompositionAudit`, with η = `norm2_upper([iso_s, orth × m(m − 1)])`; its input `orth =
audit_orthogonality(Vs)` (74–82) is `up(fro_upper(V_aᴴV_b))` of the *computed* sparse product with no term for the
rounding of that product (compare `iso` in `audit_sector`, which adds γ_dot(w_v)‖V‖_F²). For sector bases that
overlap in support (momentum sectors) the exact ‖V_aᴴV_b‖_F for the stored V's can exceed `orth` by the rounding
error of the product. `decomposition_shift` is not called by `certify_low_energy` (which forms η from the full
W) nor by `certify.py`; it is reached only from tests. No certified bound is affected; the fix, if the function is
kept public, is `orth = up(fro_upper(M) + gamma_dot(w, cplx) * fro_upper(Va) * fro_upper(Vb))` per pair.

**Verdict: AGREES** (remark on an uncalled helper recorded in VERDICTS.md).

## Runs

- `verify_thm8_rho.py --cases 60 --seed 7`: statement vs exact spectrum 60 random + 13 adversarial (7 skipped),
  steps 2–3 in mp 40 + 10, negative controls 5; 0 violations.
- `adversarial_thm8_rho.py --cases 150 --seed 13`: spectral core (mp) 50 random + 240 adversarial (10 skipped,
  η ≥ 1; includes η → 1, one-dimensional sectors, repeated eigenvalues, complex, scales 10^±150, B far from the
  compression); pooling argument (exact rationals) 31 + 419; necessity of (T3) 1 deterministic control (fails
  as expected); `certify_low_energy` on Heisenberg chains N = 4–6, CERTIFIED outcomes vs mpmath eigenvalues of the
  exact spec matrix 6 + 6 (cut in a gap; cut exactly on an eigenvalue — CERTIFIED there is legitimate since
  λ_{K+1} = E_cut ≥ E_cut − β_max − ε, and the conclusion held); negative controls 18 (perturbed non-orthonormal
  bases, missing sector, missing lowest claimed value) never CERTIFIED; 0 violations.
