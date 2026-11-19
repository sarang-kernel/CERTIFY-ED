# Theorem 2′ (sharper cluster radius ε*)

Derivation document in the D-59 format. Notation as in `docs/theory.md` and `docs/proofs/thm2_polar-factor.md`:
A Hermitian n × n with eigenvalues λ₁ ≤ … ≤ λₙ; X ∈ ℂⁿˣᵍ, G = XᴴX, η = ‖G − I‖ < 1, M = G^{1/2}, Q̂ = XM⁻¹,
δ = 1 − √(1 − η); Λ Hermitian g × g with eigenvalues μ₁ ≤ … ≤ μ_g; R = AX − XΛ.

## 1 Statement

**Hypotheses.** As in Theorem 2: η < 1, Λ Hermitian (in the implementation Λ = diag(μ) real), A finite.
spread(Λ) := μ_g − μ₁.

**Theorem 2′.** There are g distinct indices i₁, …, i_g with

    |λ_{i_j} − μ_j| ≤ ε* := ‖R‖/√(1 − η) + ‖XᴴR‖/(1 − η) + δ · spread(Λ)/√(1 − η).

**Remark.** ε* is not comparable with ρ of Theorem 2 in general: for exact eigenvectors of one degenerate level
(R = 0, spread = 0) ε* = 0 while ρ = (‖A‖ + ‖Λ‖)δ; for an orthonormal X with a large ‖XᴴR‖ (Λ far from the
Rayleigh quotient) ρ = ‖R‖ < ε*. The code uses min(ρ, ε*), both proven.

## 2 Proof

Three steps: a Rayleigh–Ritz residual for Q̂ (Kahan), a perturbation of the Ritz block (Weyl), and the
composition of the two matchings. M, M⁻¹, Q̂ are as in the polar-factor lemma, whose parts (a)–(c) are used.

**Step (i): the pair (Q̂, H′) with H′ := Q̂ᴴAQ̂.** H′ is Hermitian. From AX = R + XΛ and X = Q̂M,

    AQ̂ = AXM⁻¹ = RM⁻¹ + XΛM⁻¹ = RM⁻¹ + Q̂ MΛM⁻¹.                                   (1)

Multiplying (1) by the orthogonal projector P⊥ := I − Q̂Q̂ᴴ (P⊥Q̂ = 0 by lemma (a)) gives P⊥AQ̂ = P⊥RM⁻¹, and since
AQ̂ − Q̂H′ = AQ̂ − Q̂Q̂ᴴAQ̂ = P⊥AQ̂,

    ‖AQ̂ − Q̂H′‖ = ‖P⊥RM⁻¹‖ ≤ ‖P⊥‖ ‖R‖ ‖M⁻¹‖ ≤ ‖R‖/√(1 − η)                           (2)

(‖P⊥‖ ≤ 1 for an orthogonal projector; ‖M⁻¹‖ ≤ (1 − η)^{−1/2} by lemma (b)). Kahan's inclusion theorem
(`docs/proofs/thm2_polar-factor.md`, section 2; here H′ is the Rayleigh quotient, the classical case) matches the
eigenvalues θ₁ ≤ … ≤ θ_g of H′ to distinct eigenvalues of A: there is an injection k ↦ i_k with
|λ_{i_k} − θ_k| ≤ ‖R‖/√(1 − η).

**Step (ii): H′ against Λ.** Multiply (1) from the left by Q̂ᴴ = M⁻¹Xᴴ (lemma (a) gives Q̂ᴴQ̂ = I):

    H′ − Λ = M⁻¹XᴴRM⁻¹ + (MΛM⁻¹ − Λ).                                                   (3)

The first term has norm ≤ ‖M⁻¹‖² ‖XᴴR‖ ≤ ‖XᴴR‖/(1 − η). For the second, MΛM⁻¹ − Λ = (MΛ − ΛM)M⁻¹ = [M, Λ]M⁻¹,
and for any scalar c, [M, Λ] = [M − I, Λ − cI] (the identity and scalar matrices commute with everything).
Take c = (μ₁ + μ_g)/2, so that ‖Λ − cI‖ = max_j |μ_j − c| = spread(Λ)/2 (Λ Hermitian: its norm is its largest
eigenvalue modulus). For any B, C, ‖[B, C]‖ = ‖BC − CB‖ ≤ 2‖B‖‖C‖, hence

    ‖MΛM⁻¹ − Λ‖ ≤ 2 ‖M − I‖ ‖Λ − cI‖ ‖M⁻¹‖ ≤ 2 · δ · (spread/2) · (1 − η)^{−1/2} = δ · spread(Λ)/√(1 − η)

by lemma (b). Together: ‖H′ − Λ‖ ≤ ‖XᴴR‖/(1 − η) + δ · spread(Λ)/√(1 − η) =: τ.

H′ and Λ are Hermitian, so Weyl's inequality (H. Weyl, Math. Ann. 71, 1912; Horn and Johnson, *Matrix
Analysis*, 2nd ed., Theorem 4.3.1 and Corollary 4.3.15: |λ_k(B + E) − λ_k(B)| ≤ ‖E‖ for Hermitian B, E) gives
|θ_k − μ_k| ≤ ‖H′ − Λ‖ ≤ τ for the sorted eigenvalues, k = 1, …, g.

**Step (iii): composition.** For each k, |λ_{i_k} − μ_k| ≤ |λ_{i_k} − θ_k| + |θ_k − μ_k| ≤ ‖R‖/√(1 − η) + τ = ε*.
The map k ↦ i_k is the injection of step (i). ∎

**Remarks on the hypotheses.** (1) η < 1 is needed for M⁻¹ to exist with the stated bound; at η → 1 all three
terms of ε* diverge. (2) Λ Hermitian is needed for Weyl in step (ii) and for Kahan's theorem; a real diagonal
Λ is the implemented case. (3) Nothing requires the μ_j to be distinct or the λ_i to be simple. (4) The proof
does not use that the μ_j are close to eigenvalues: for Λ far from the spectrum ε* is simply large.

## 3 Where the code implements it

`certify_ed/enclosure.py`, `_cluster_bound` (branch `track/backlog-proofs`; `up(x) = x(1 + 8u)` covers up to six
roundings on exact upper-bound inputs, see `docs/proofs/thm2_polar-factor.md` section 3 for η, δ, ‖R‖ and the
convention). Rc denotes the *computed* residual columns `res.residual_columns(idx)` and r_i = (AX − XΛ)_{:,i}
the exact ones; `res.E_norm[i]` ≥ ‖Rc_i − r_i‖ (Theorem 1 accounting, step 3–4).

| quantity | code | bound it implements | roundings inside `up` |
|---|---|---|---|
| ‖X‖₂ | `xnorm = up(sqrt(1 + eta))` | ‖X‖₂ = √λ_max(G) ≤ √(1 + η) | 2 (addition, sqrt) |
| ‖Rc − R‖_F | `Rerr = norm2_upper(res.E_norm[idx])` | ‖Rc − R‖_F = ‖(‖Rc_i − r_i‖)_i‖₂ ≤ ‖(E_norm_i)_i‖₂ | — |
| ‖XᴴR‖₂ | `XR_bound = up(fro_upper(XR) + fro_upper(Xg) * fro_upper(Rc) * gd + xnorm * Rerr) + gram_additive_term(g, n, cplx) + 3 ULP_MIN`, `XR = fl(XgᴴRc)` | ‖XᴴR‖₂ ≤ ‖XᴴR‖_F ≤ ‖fl(XᴴRc)‖_F + ‖fl(XᴴRc) − XᴴRc‖_F + ‖Xᴴ(Rc − R)‖_F; second term: entrywise ≤ γ_dot(n)(|X|ᴴ|Rc|)_ab (Higham 2002 §3.1, §3.6 for complex) plus the additive underflow part (`docs/proofs/int05_underflow.md`), and ‖|X|ᴴ|Rc|‖_F ≤ ‖X‖_F‖Rc‖_F; third term ≤ ‖X‖₂‖Rc − R‖_F | 5 (two products, `gd` product, product, two additions = 6 operations; the product `fro_upper(Xg) * fro_upper(Rc)` and `gd`'s two internal roundings are covered by the (1 + 5u)² slack of the two `fro_upper`); the `3 ULP_MIN` cover the absolute errors of the three products if subnormal |
| (1 − η)^{−1/2} | `inv_sqrt = up(1 / sqrt(1 - eta))` | ≥ exact (1 − η)^{−1/2} for the stored η (η ≥ exact η) | 3 |
| (1 − η)^{−1} | `inv = up(1 / (1 - eta))` | ≥ exact | 2 |
| spread | `spread = up(sub_up(max(mu_g), min(mu_g)))` | `sub_up` is already an upper bound on μ_g − μ₁ | 0 |
| ε* | `eps_star = up(Rg_fro * inv_sqrt + XR_bound * inv + delta * spread * inv_sqrt) + 4 ULP_MIN` | the three terms of ε*, every factor an upper bound and non-negative | 6 (four products, two additions); `4 ULP_MIN` for the four products' absolute errors if subnormal |
| radius | `min(rho, eps_star)`; `bound = {'eps': rho, 'rho': rho, 'eps_star': eps_star, 'used': ...}` | both proven; `used` records which is smaller | 0 |

The claimed vectors are the stored binary64 numbers; `mu_g` are the stored claimed values (Λ = diag(mu_g)
exactly). Rounding direction: every factor towards +∞; the exact ε* for the stored numbers is ≤ `eps_star`.

## 4 Computational verification

`docs/proofs/verify_thm2prime_eps-star.py` (mpmath, 30 digits, slack 10⁻²⁴; same case generator as the
Theorem 2 script with its ten adversarial families, plus the family "spread" with μ spread 10 to 100 and
"ritz" with Λ = the diagonal of XᴴAX). Per case:

- E1: ‖AQ̂ − Q̂H′‖ ≤ ‖R‖/√(1 − η) (inequality (2));
- E2: ‖H′ − Λ‖ ≤ τ, and the identity (3) holds to 10⁻²⁴ (both terms formed separately);
- E3: sorted pairing |θ_k − μ_k| ≤ τ (Weyl);
- E4: the bottleneck matching distance d* between μ and the eigenvalues of A is ≤ ε* (Theorem 2′);
- C: the code's `eps_star` ≥ exact ε*, its `XR_bound` ≥ ‖XᴴR‖₂ (recomputed from the returned `bound` and the
  exact norms), and its radius min(ρ, ε*) ≥ d*.

Counts (run of 2026-10-30, `--cases 20000 --seed 2`, output in `docs/proofs/runs/thm2prime_eps-star.txt`): 20 000
generated random cases (17 094 met η < 1 − 10⁻²⁰) and 800 adversarial cases (600 of the ten shared families, 100
"spread", 100 "ritz"); per counter: E1–E3 51 282 random + 1 863 adversarial checks (3 085 skipped), E4 17 094 + 621,
C 15 337 + 547 (1 831 skipped where the code's guarded η reached 1); 0 violations.

## 5 Analytic cross-check

(i) *Exact eigenvectors of a degenerate level.* Let AV = λ₀V with VᴴV = I, X = VC for an invertible C,
Λ = λ₀I. Then R = AVC − VCλ₀ = 0, spread = 0, so ε* = 0: the g claimed values λ₀ are matched to g distinct
eigenvalues equal to λ₀, which is correct (the level has multiplicity ≥ g because V spans a g-dimensional
eigenspace). ρ = (‖A‖ + |λ₀|)δ is positive unless C is unitary. This is the IMP-13 case and the reason for
Theorem 2′ (`tests/test_review_fixes.py::test_imp13_valid_nonorthogonal_degenerate_vectors_certified`).

(ii) *Orthonormal X, Λ = XᴴAX.* η = δ = 0, XᴴR = XᴴAX − Λ = 0, so ε* = ‖R‖ = ρ: both radii reduce to the
Rayleigh–Ritz form of Kahan's theorem.

(iii) *g = 1.* X = x, Λ = μ, H′ = xᴴAx/‖x‖² = θ (the Rayleigh quotient), M = ‖x‖. Step (ii) reads
|θ − μ| ≤ |xᴴ(Ax − μx)|/‖x‖², which is the identity θ − μ = xᴴ(Ax − μx)/‖x‖² with the triangle inequality;
step (i) is Theorem 1 for the normalized vector: some eigenvalue is within ‖Ax − θx‖/‖x‖ ≤ ‖Ax − μx‖/‖x‖ of θ.
The composed bound |λ − μ| ≤ ‖Ax − μx‖/‖x‖ + |xᴴ(Ax − μx)|/‖x‖² is weaker than Theorem 1 by the second term,
as expected from a bound designed for g > 1 (the code takes min(ρ, ε*), and for g = 1 Theorem 1 itself).

(iv) *Scaling.* Replacing A by sA and Λ by sΛ (s > 0) scales R, ‖XᴴR‖, spread and ε* by s; replacing X by tX
changes η, so the bound is not scale-free in X (nor should it be: η is the orthonormality defect).

## 6 Status

Derived on 2026-10-30; machine-checked (17 094 (51 282 inequality checks) random + 621 (1 863 inequality checks) adversarial cases, 0 violations); independent re-derivation: done 2026-10-30 (from the statement alone, not a human check; verdict AGREES-WITH-REMARKS, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
