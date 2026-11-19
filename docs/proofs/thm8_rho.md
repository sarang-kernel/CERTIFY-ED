# Theorem 8 (sector assembly with radius ρ)

Derivation document in the D-59 format. Notation as in `docs/theory.md` §5 and `compose.py`: H_spec is the
declared model operator on the full basis (Hermitian, n × n); H is the Hermitian matrix CERTIFY-ED built for
it (`construct.hermitian_part(build_bitwise(...))`, or the direct sum of the U(1)-block builds that the
sector audits used), with ‖H − H_spec‖ ≤ m (the builder shift, `docs/proofs/appA_model-shift.md`); V₁ … V_m are
the producer's sector bases (n × d_s), H_s the Hermitian parts of its sector blocks (d_s × d_s);
W = [V₁ ⋯ V_m], B = ⊕_s H_s, η = ‖WᴴW − I‖, D ≥ ‖HW − WB‖, δ = 1 − √(1 − η), ρ = D + (‖H‖ + ‖B‖)δ, ε = ρ + m.
Eigenvalues λ₁(·) ≤ λ₂(·) ≤ ….

## 1 Statement

**Hypotheses** (the conditions `compose.certify_low_energy` checks before it reports CERTIFIED).

- (T1) Σ_s d_s = n and η < 1 (W is square and invertible).
- (T2) For every sector s the claimed pairs below E_cut (k_s of them, claimed values c_{s,1} ≤ … ≤ c_{s,k_s}
  with radii ≤ r) satisfy the hypotheses of Corollary 5 for A_u = A_m = H_s with σ = E_cut and the inertia
  count ν_s = k_s, β_s (`docs/proofs/cor5_pairing.md`: all distinct, enclosures below E_cut − β_s, count k_s).
- (T3) max over all pooled claimed values of (c + r) < E_cut − max_s β_s.

**Conclusion.** Let c₍₁₎ ≤ … ≤ c₍_K₎ (K = Σ_s k_s) be the pooled claimed values. Then
|λ_j(H_spec) − c₍ⱼ₎| ≤ r + ε for j = 1, …, K, and λ_{K+1}(H_spec) ≥ E_cut − max_s β_s − ε.

## 2 Proof

**Step 1 (each sector, Corollary 5).** By (T2) and Corollary 5 (Δ = 0), the k_s claimed values of sector s are
paired in sorted order with λ₁(H_s) ≤ … ≤ λ_{k_s}(H_s) within r, exactly k_s eigenvalues of H_s lie below
E_cut − β_s, and λ_{k_s+1}(H_s) ≥ E_cut − β_s.

**Step 2 (the spectrum of B).** B = ⊕_s H_s is Hermitian with spectrum the multiset union of the sector
spectra. The K eigenvalues λ_i(H_s), i ≤ k_s, s = 1 … m, are each ≤ c_{s,i} + r ≤ max(c + r) < E_cut − max_s β_s by
(T3); every other eigenvalue of B is some λ_i(H_s) with i > k_s, hence ≥ E_cut − β_s ≥ E_cut − max_s β_s. So the
K lowest eigenvalues of B are exactly the paired ones, λ_{K+1}(B) ≥ E_cut − max_s β_s, and the pooled claimed
values are matched to λ₁(B), …, λ_K(B) by a bijection respecting the uniform radius r. The sorted-pairing lemma
(`docs/proofs/cor5_pairing.md`, section 2) gives |λ_j(B) − c₍ⱼ₎| ≤ r for j ≤ K.

**Step 3 (B against H).** By (T1) W is square with η < 1, so its polar factor Q̂ = W(WᴴW)^{−1/2} is unitary
(polar-factor lemma (a), `docs/proofs/thm2_polar-factor.md`) and ‖W − Q̂‖ ≤ δ (lemma (c)). With R̂ := HQ̂ − Q̂B,

    R̂ = (HW − WB) + H(Q̂ − W) − (Q̂ − W)B,   so   ‖R̂‖ ≤ ‖HW − WB‖ + (‖H‖ + ‖B‖)δ ≤ D + (‖H‖ + ‖B‖)δ = ρ

(lemma (d) with X = W, Λ = B; ‖HW − WB‖ ≤ ‖HW − WB‖_F = (Σ_s ‖HV_s − V_sH_s‖_F²)^{1/2} ≤ D by the block-column
structure, each block bounded by the audit's `invariance_defect`). Q̂ᴴHQ̂ is Hermitian with the spectrum of H
(unitary similarity), and Q̂ᴴHQ̂ − B = Q̂ᴴR̂ is Hermitian with ‖Q̂ᴴR̂‖ ≤ ‖R̂‖ ≤ ρ. Weyl's inequality (Horn–Johnson,
*Matrix Analysis*, 2nd ed., Cor. 4.3.15) gives |λ_j(H) − λ_j(B)| ≤ ρ for every j = 1, …, n. (Theorem 2 with X = W
gives the same constant by Kahan's theorem; for a square W Weyl's inequality is the shorter route.)

**Step 4 (H against H_spec).** H and H_spec are Hermitian with ‖H − H_spec‖ ≤ m, so |λ_j(H_spec) − λ_j(H)| ≤ m
for every j (Weyl again).

**Step 5 (assembly).** For j ≤ K: |λ_j(H_spec) − c₍ⱼ₎| ≤ |λ_j(H_spec) − λ_j(H)| + |λ_j(H) − λ_j(B)| + |λ_j(B) − c₍ⱼ₎|
≤ m + ρ + r = r + ε. And λ_{K+1}(H_spec) ≥ λ_{K+1}(H) − m ≥ λ_{K+1}(B) − ρ − m ≥ E_cut − max_s β_s − ε. ∎

**Remarks on the hypotheses.** (1) Hermiticity of H is used in Step 3 (spectrum of Q̂ᴴHQ̂ = spectrum of H,
Weyl) and in Step 4; the built matrix can differ from its adjoint in the last bit when local amplitudes are
irrational, which is why the code symmetrises it (`construct.hermitian_part`) and why the builder shift
covers the symmetrisation (Appendix A document, Lemma A2). (2) ‖H‖ in ρ is the norm of the symmetrised built
matrix, bounded by its ∞-norm; when the sectors were audited inside exactly invariant U(1) blocks, H is the
direct sum of the block builds, and its 2-norm is the largest block 2-norm, bounded by the largest block
∞-norm (the blocks are Hermitian; all blocks are touched because W is square and every V_s lies in one
block). (3) The direction of every inequality is from the stored, outward-rounded quantities to the exact ones:
the code's D, η, δ, ρ, m are upper bounds on the quantities of this proof (section 3). (4) The pooled check
(T3) is what makes "the K lowest" well defined; without it an unclaimed eigenvalue of one block could lie below
a claimed value of another, and the statement would only be a matching, not a sorted pairing.

## 3 Where the code implements it

`certify_ed/compose.py`, `certify_low_energy` (branch `track/backlog-proofs`).

| hypothesis / quantity | code | direction |
|---|---|---|
| H Hermitian | `H = hermitian_part(...)` for the full build and `Ha = hermitian_part(Ha)` for a U(1) block | exact symmetrisation up to one rounding, covered by `_builder_shift` |
| H_s Hermitian | `Hs = ((Hs + Hs.conj().T) / 2)` when the producer's block is not exactly Hermitian (`hermitian_part` note) | the statement concerns the Hermitian part |
| ‖H‖ | `Hn = max(Hn, inf_norm_upper(Ha))` | `inf_norm_upper` ≥ ‖·‖_∞ ≥ ‖·‖₂ for a Hermitian matrix; the max over blocks is the norm of the direct sum (Remark 2) |
| ‖B‖ | `Bn = max_s inf_norm_upper(Hs)` | same, block diagonal |
| (T2) per sector | `enc = enclose(OperatorAdapter(Hs), ...)`; `inside = all(add_up(enc.mu, enc.radius) < sub_down(E_cut, r.beta))`; `distinct = enc.all_distinct`; `r.negative == mu_below.size`; status `COMPLETE` | Corollary 5's (H1)–(H3) with Δ = 0, outward-rounded |
| r | `rmax = max over sectors of enc.radius.max()` | ≥ every radius |
| (T1) | `covered != n` → FAILED; `eta = up(fro_upper(G) + gamma_dot(wmax, cplx) * fro_upper(W) ** 2)` with `G = WᴴW − I` (sparse product; rounding bound as for the Gram matrix of Theorem 2, `docs/proofs/thm2prime_eps-star.md` section 3); `not eta < 1` → FAILED | η is an upper bound |
| D | `D = norm2_upper([invariance_defect_s])` | ≥ (Σ_s inv_s²)^{1/2} ≥ ‖HW − WB‖_F; each inv_s ≥ ‖H_aV_s − V_sH_s‖_F by `sectors.audit_sector` (Frobenius bound plus the rounding of the products) |
| (T3) | `add_up(max(pooled), rmax) >= sub_down(E_cut, beta_max)` → INCONCLUSIVE | outward-rounded |
| δ, ρ | `delta = up(1 - sqrt(1 - eta)) + 4 U`; `rho = up(D + (Hn + Bn) * delta) + ULP_MIN` | as in `docs/proofs/thm2_polar-factor.md` section 3 |
| ε | `eps = up(rho + m_build)` | m_build = max over the audit spaces of `_builder_shift` (Remark 2: the block builds form H) |
| conclusion | `radius = add_up(rmax, eps)`; `next_lower_bound = sub_down(sub_down(E_cut, beta_max), eps)`; `lowest = sorted(pooled)` | outward |

Before this branch `eps_decomposition` was formed with `up(math.sqrt(sum of squares))` for D (its squares could
underflow, `docs/proofs/int05_underflow.md`) and the built H was not symmetrised; the constant ρ itself dates from
3.1.0 (review IMP-12/IMP-20). `sectors.decomposition_shift` (not used by `certify_low_energy`) evaluates the same
ρ from a `DecompositionAudit` with η bounded by the block Frobenius sum, and now returns ρ instead of (1 + √5)ρ/2.

## 4 Computational verification

`docs/proofs/verify_thm8_rho.py`:

- A: Heisenberg chains (spin-1/2, N = 4 and 6, exact dyadic matrices) decomposed into U(1) blocks and into
  momentum blocks (complex isometries), eigenpairs from `scipy.linalg.eigh` of each block,
  `certify_low_energy` at several cuts; when CERTIFIED, the conclusion is checked against the eigenvalues
  of the exact spec matrix (mpmath, 60 digits): |λ_j − c₍ⱼ₎| ≤ `radius` for j ≤ K and λ_{K+1} ≥ `next_lower_bound`,
  and the intermediate inequalities of Steps 2–3 (‖HW − WB‖ ≤ D, ‖HQ̂ − Q̂B‖ ≤ ρ, sorted pairing of eig(B) with
  eig(H) within ρ, the K lowest of B equal the pooled matched values) are checked in mp for dimensions ≤ 30;
- B: a spin-1 chain (N = 3, irrational amplitudes, m > 0): the eigenvalues of the declared model from
  `checker.exact_matrix` (the rational matrix K = D⁻¹HD, similar to H_spec) — this checks Step 4, i.e. that
  the builder shift is large enough;
- adversarial: a sector basis perturbed by 10⁻¹⁴ … 3·10⁻¹¹ (η > 0), a sector block perturbed by 10⁻¹³ (D > 0; both
  within the audit tolerance 10⁻¹⁰): the
  statement must still hold whenever the result is CERTIFIED; negative controls (must not be CERTIFIED): a
  dropped sector, a corrupted block entry (10⁻³), a sector that omits an eigenvalue below the cut, a
  duplicated eigenvector in one sector, a cut inside a claimed enclosure.

Counts (run of 2026-10-30, `--cases 240 --seed 4`, output in `docs/proofs/runs/thm8_rho.txt`): T 240 random + 49
adversarial CERTIFIED cases checked against the exact spectrum (31 perturbed cases skipped: not CERTIFIED), S 160
random + 32 adversarial cases of the Step 2–3 inequalities (dimension ≤ 30), N 5 negative controls (none
CERTIFIED); 0 violations.

## 5 Analytic cross-check

(i) *One sector, V = I.* m = 1, W = I, η = 0, D = ‖H − H₁‖ (= 0 when the producer's block is the built matrix),
ρ = 0 (D = 0), and the theorem is Corollary 5 for H itself plus the builder shift m: |λ_j(H_spec) − c₍ⱼ₎| ≤ r + m.

(ii) *Exact decomposition.* W unitary and HV_s = V_sH_s exactly: η = D = 0 ⇒ ρ = 0, and the pooled sorted
spectrum of B *is* the spectrum of H; the theorem reduces to "the union of the sector spectra is the spectrum",
with only the sector radii and the builder shift remaining.

(iii) *Comparison with Theorem 2.* Theorem 2 with X = W, Λ = B gives a matching of the eigenvalues of B to
distinct eigenvalues of H within ρ (Kahan); for the square W this matching is the sorted one by Weyl. Both
routes give the same constant, which is why ρ replaces the (1 + √5)/2 ρ of the pre-3.1 Weyl-perturbation proof.

(iv) *Scaling.* Multiplying H_spec and all H_s by t > 0 scales D, ‖H‖, ‖B‖, r, m, and hence ε and the radius,
by t, while η and δ are unchanged — the conclusion is homogeneous of degree one, as a spectral statement must be.

## 6 Status

Derived on 2026-10-30; machine-checked (240 (160 inequality checks) random + 49 (32 inequality checks, 5 negative controls) adversarial cases, 0 violations); independent re-derivation: done 2026-10-30 (from the statement alone, not a human check; verdict AGREES, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
