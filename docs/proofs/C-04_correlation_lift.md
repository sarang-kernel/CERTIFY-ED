# C-04 Correlation functions and observables for symmetry-adapted claims (lift through the sector isometry)

Notation as in `docs/theory.md` §5 (Theorem 8) and §4 (Theorem 6). H is the n × n Hermitian operator of the
declared model on the full space (or on the exactly invariant U(1) block that S1's audit uses; the argument is
the same). The producer supplies a sector basis V (n × d_s) and a block H_s (d_s × d_s, Hermitian after the
symmetrisation of S1) with the claim that V is an isometry onto an H-invariant subspace and H_s = VᴴHV. S1
(`sectors.audit_sector`) returns upper bounds ι ≥ ‖VᴴV − I‖₂ (`isometry_defect`, a Frobenius bound) and
ν ≥ ‖HV − VH_s‖₂ (`invariance_defect`). S2–S4 run in the block: X (d_s × g) are the claimed vectors of a
cluster, η = ‖XᴴX − I‖₂, P_s the spectral projector of H_s onto the cluster's g eigenvalues, and Theorem 6
gives ‖XXᴴ − P_s‖₂ ≤ s + η (`projector_error`). O is an operator on the full space built from the spec
(`correlations.site_operator`), or supplied as a full-space matrix.

Polar decomposition V = QM with M = (VᴴV)^{1/2} and Q an exact isometry (QᴴQ = I) with range Q = range V;
the eigenvalues of M lie in [√(1−ι), √(1+ι)] and ‖M − I‖₂ ≤ δ := 1 − √(1−ι) (the larger of 1 − √(1−ι) and
√(1+ι) − 1). Define Π_V := V P_s Vᴴ and Π_Q := Q P_s Qᴴ.

## 1 Statement

**Hypotheses.** (H1) S1's audit passed with ι < 1 and ν recorded. (H2) S4 certified the cluster in the block:
‖XXᴴ − P_s‖₂ ≤ s + η (Theorem 6), g = rank P_s. (H3) O Hermitian (a non-Hermitian O is split exactly as
O = H + iK, C-05, and each part is treated separately; the recombination is Proposition 15).

**Lemma 0 (what is lifted).** tr(P_s VᴴOV) = tr(Π_V O) exactly (cyclicity of the trace). Π_Q is an orthogonal
projector of rank g onto a g-dimensional subspace of range V. When ι = ν = 0, Π_V = Π_Q is the spectral
projector of H onto the eigenvalues of the cluster *within the invariant subspace range V*.

**Lemma 1 (compression rounding).** Let Õ := VᴴOV (exact, Hermitian) and Õ_h the Hermitian part of
fl(Vᴴ fl(OV)) formed in binary64, with w_O the largest number of nonzeros in a row of O and w_V the largest
number of nonzeros in a column of V. Under the standard model (`docs/theory.md` §1, no underflow),

    ‖Õ_h − Õ‖₂ ≤ ‖Õ_h − Õ‖_F ≤ r_c := γ̇_{w_O + w_V} ‖ |V|ᵀ|O||V| ‖_F + u ‖Õ_fl‖_F (1 + u),

where γ̇ is `fp.gamma_dot` (γ_m for real data, 2γ_{m+4} for complex data).

**Proposition 1 (isometry term).** |tr(Π_V O) − tr(Π_Q O)| ≤ 2g ‖O‖₂ δ (1 + √(1+ι)).

**Conclusion (the certified statement).** With v := tr(XᴴÕ_hX) (computed) and b₆ the Theorem 6 bound for the
stored Hermitian Õ_h in the block (|v − tr(P_s Õ_h)| ≤ b₆, `eigenspace.certify_observable`, which includes
its own rounding term),

    |v − tr(Π_V O)| ≤ b_V := b₆ + g r_c,
    |v − tr(Π_Q O)| ≤ b   := b_V + 2g ‖O‖₂ δ (1 + √(1+ι)).

The record reports both intervals; the headline statement is the one about Π_Q (an orthogonal projector).
The identification of range Π_Q with an eigenspace of H is *not* part of the statement: it rests on the
invariance defect ν (recorded), and a cluster of the sector may be part of a level of H that other sectors
share. The lift certifies the sector's contribution tr(Π_Q O) to such a level.

## 2 Proof

**Lemma 0.** tr(P_s VᴴOV) = tr(V P_s Vᴴ O) is tr(AB) = tr(BA) with A = P_s Vᴴ, B = OV. Π_Q² = Q P_s QᴴQ P_s Qᴴ =
Q P_s² Qᴴ = Π_Q and Π_Qᴴ = Π_Q, rank Π_Q = rank P_s = g because Q is injective. If ι = ν = 0 then M = I, V = Q,
and HV = VH_s gives H(Vy) = V(H_s y): V maps eigenvectors of H_s to eigenvectors of H with the same
eigenvalues, so V P_s Vᴴ projects onto the span of the lifted eigenvectors of the cluster, i.e. the H-eigenspace
of those eigenvalues intersected with range V. ∎

**Lemma 1.** Standard model fl(a ∘ b) = (a ∘ b)(1 + δ), |δ| ≤ u. (i) W := fl(OV): entry (i, j) is an inner product
of at most w_O nonzero terms, so |W − OV| ≤ γ̇_{w_O} |O||V| entrywise (Higham 2002, eq. (3.5) and §3.6 for the
complex case, which `gamma_dot` dominates). (ii) Õ_fl := fl(VᴴW): entry (i, j) is an inner product of at most
w_V terms (the nonzeros of column i of V), so |Õ_fl − VᴴW| ≤ γ̇_{w_V} |V|ᵀ|W| ≤ γ̇_{w_V}(1 + γ̇_{w_O}) |V|ᵀ|O||V|.
Adding (i) multiplied by |V|ᵀ: |Õ_fl − Õ| ≤ (γ̇_{w_O} + γ̇_{w_V} + γ̇_{w_O}γ̇_{w_V}) |V|ᵀ|O||V| ≤ γ̇_{w_O + w_V} |V|ᵀ|O||V|
(Higham 2002, Lemma 3.3: (1 + γ_a)(1 + γ_b) ≤ 1 + γ_{a+b}; the same for γ̇ since it is γ of a shifted index times
a constant ≥ 1 applied consistently). (iii) The Hermitian part Õ_h^exact := (Õ_fl + Õ_flᴴ)/2 satisfies
Õ_h^exact − Õ = ((Õ_fl − Õ) + (Õ_fl − Õ)ᴴ)/2 because Õ is Hermitian, so ‖Õ_h^exact − Õ‖_F ≤ ‖Õ_fl − Õ‖_F
(triangle inequality; ‖Aᴴ‖_F = ‖A‖_F). (iv) Forming the Hermitian part in binary64: the addition rounds once
(error ≤ u |Õ_fl + Õ_flᴴ|_{ij} ≤ u(|Õ_fl| + |Õ_flᴴ|)_{ij}) and the halving is exact, so
‖fl(Õ_h) − Õ_h^exact‖_F ≤ u ‖(|Õ_fl| + |Õ_flᴴ|)/2‖_F ≤ u ‖Õ_fl‖_F. Together ‖Õ_h − Õ‖_F ≤ γ̇_{w_O+w_V}
‖ |V|ᵀ|O||V| ‖_F + u‖Õ_fl‖_F, and ‖·‖₂ ≤ ‖·‖_F. The code evaluates ‖ |V|ᵀ|O||V| ‖_F from the floating-point
product of the absolute values, bounded above by `fp.fro_upper`; the computed product underestimates the exact
one by at most a relative γ̇_{w_O+w_V} entrywise (inner products of non-negative terms), so the exact norm is at
most fro_upper(computed)/(1 − γ̇_{w_O+w_V}), and the factor 1/(1 − γ̇) is applied (rounded up). u‖Õ_fl‖_F is bounded
by u·fro_upper(Õ_fl)(1 + 4u). ∎

**Proposition 1.** Π_V − Π_Q = Q(M P_s M − P_s)Qᴴ since V = QM. Because Q is an isometry, the singular values of
Q Z Qᴴ are those of Z (for any d_s × d_s matrix Z), hence ‖Π_V − Π_Q‖₁ = ‖M P_s M − P_s‖₁. Write
M P_s M − P_s = (M − I) P_s M + P_s (M − I): both terms have rank ≤ g, so the difference has rank ≤ 2g, and
‖M P_s M − P_s‖₂ ≤ ‖M − I‖₂‖P_s‖₂‖M‖₂ + ‖P_s‖₂‖M − I‖₂ ≤ δ√(1+ι) + δ = δ(1 + √(1+ι)), using ‖P_s‖₂ = 1,
‖M‖₂ ≤ √(1+ι) and ‖M − I‖₂ ≤ δ (the eigenvalues m of M lie in [√(1−ι), √(1+ι)], and |m − 1| ≤ 1 − √(1−ι)
there because 1 − √(1−ι) ≥ √(1+ι) − 1 for ι ∈ [0, 1), as (1 − √(1−ι)) − (√(1+ι) − 1) = 2 − √(1−ι) − √(1+ι) ≥ 0 by
concavity of √). The trace norm of a rank-r matrix is ≤ r times its spectral norm (sum of ≤ r singular values),
so ‖Π_V − Π_Q‖₁ ≤ 2g δ(1 + √(1+ι)). Finally |tr((Π_V − Π_Q)O)| ≤ ‖Π_V − Π_Q‖₁ ‖O‖₂ (Hölder's inequality for the
trace, Horn & Johnson, *Topics in Matrix Analysis*, 1991, Theorem 3.3.14 / the duality of ‖·‖₁ and ‖·‖₂). ∎

**Conclusion.** |v − tr(P_s Õ_h)| ≤ b₆ is Theorem 6 (`docs/theory.md` §4; the record of 3.1.0) applied in the
block to the stored Hermitian matrix Õ_h, with the rounding of tr(XᴴÕ_hX) included by `certify_observable`.
|tr(P_s Õ_h) − tr(P_s Õ)| = |tr(P_s(Õ_h − Õ))| ≤ ‖P_s‖₁‖Õ_h − Õ‖₂ ≤ g r_c (Hölder; ‖P_s‖₁ = g for a rank-g
orthogonal projector; Lemma 1). tr(P_s Õ) = tr(Π_V O) (Lemma 0). Adding Proposition 1 gives the bound for Π_Q.
The non-Hermitian case: O = H + iK exactly (C-05 checks the TwoSum error term is zero), tr(Π O) = tr(Π H) +
i tr(Π K) by linearity, and |tr(Π O) − (v_H + i v_K)| ≤ b_H + b_K is Proposition 15 with coefficients 1 and i. ∎

**Which bound on ‖O‖₂.** For Hermitian O, ‖O‖₂ ≤ ‖O‖_∞ (= ‖O‖₁ by symmetry; ‖O‖₂² ≤ ‖O‖₁‖O‖_∞), computed by
`fp.inf_norm_upper` with its rounding guard. For the compressed Õ_h the same bound is used inside Theorem 6.

## 3 Where the code implements it

`certify_ed/correlations.py` (branch `track/new-statements`):

* Lemma 1: `compress_operator` (lines 256–276). W = O @ V and Õ_fl = Vᴴ @ W in sparse binary64; `absprod` =
  |V|ᵀ|O||V|; r_c = γ̇_{w_O + w_V} · fro_upper(absprod) · up(1/(1 − γ̇_{w_O+w_V})) (the last factor covers the rounding
  of the absolute product); then the Hermitian part (exactly Hermitian by construction: the two roundings of
  conjugate entries are conjugate); the term u·fro_upper(Õ_fl)(1 + 4u) is added and the sum guarded with `fp.up`.
  Returns Õ_h, r_c and `inf_norm_upper(Õ_h)`.
* Proposition 1: `_isometry_term` (279–284): δ = up(1 − √(1−ι)) + 4u (the same expression as
  `certify._subspace_shift` and `compose.certify_low_energy`), result 2g‖O‖δ(1 + √(1+ι))(1 + 4u), guarded;
  ∞ when ι ≥ 1.
* Conclusion: `certify_observable_lifted` (287–338). Hermitian O: Theorem 6 in the block on Õ_h (line 301,
  `certify_observable` with `O_norm` = the compressed norm bound), b_V = b₆ + g r_c (line 304, `add_up`),
  b = b_V + isometry term (305–306). Non-Hermitian O: `hermitian_parts` (line 314), the two parts lifted
  recursively, bounds added with `add_up`; `parts_formed_exactly` False → INCONCLUSIVE. The record carries
  `value`, `bound`, `bound_V`, `isometry_term`, `compression_rounding`, `theorem6_bound_in_block`,
  `isometry_defect`, `invariance_defect` and `refers_to` (what Π_Q and Π_V are).
* Correlations: `certify_correlations_lifted` (341–384): site operators built in the full space
  (`site_operator(..., states=None)`), O_iO_j symmetrised exactly for i ≠ j (commuting Hermitian factors), the
  two-letter case lifted through the Hermitian decomposition; per-state values tr(Π·)/g with bound b/g;
  `structure_factor` (Proposition 15) applies unchanged to the pair records.
* Hooks in `certify.py`: `_correlations(..., audit=audit)` dispatches sector claims to the lifted path, returns
  NOT_CHECKED (DEPENDENCY_NARROWED) when S1 produced no audit and INCONCLUSIVE when the audit failed (the
  lift's hypotheses H1 are not established); in the S4 observable loop a full-space matrix observable for a
  sector claim is lifted by `certify_observable_lifted` when the audit passed. ι and ν are S1's
  `isometry_defect` and `invariance_defect` (Frobenius upper bounds, ≥ the spectral norms).

Rounding directions: every bound is an upper bound rounded up (`up`, `add_up`, index shifts); the value v is
the computed floating-point trace and carries no rounding claim of its own beyond b₆'s rounding term.

## 4 Computational verification

`docs/proofs/verify_C-04.py` (seed 20261030; exact rationals for parts A and C; mpmath at 50 digits for the
polar factors, spectral norms and traces of parts B and D; comparison margin 10⁻⁴⁰). Run of 2026-10-30:

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | Lemma 0's identity tr(P_s VᴴOV) = tr(VP_sVᴴO) on random rational V, rational rank-g projectors P_s = Y(YᵀY)⁻¹Yᵀ and rational symmetric O: exact equality | 60 | 0 |
| B | Proposition 1: exact \|tr(Π_V O) − tr(Π_Q O)\| (Q the 50-digit polar factor) against 2g‖O‖₂δ(1 + √(1+ι)) with ι the exact ‖VᴴV − I‖₂; random complex cases ι ∈ [10⁻⁸, 0.8], g ≤ d_s ≤ 4, n ≤ 8 | 100 random + 10 adversarial (ι = 0.9, 0.999, 0.999999 with g = 1, g = d_s and O the projector onto the stretched direction; ι = 0) | 0 |
| C | Lemma 1: exact ‖Õ_h − VᴴOV‖_F (rational) against r_c of `compress_operator`; random sparse real and complex V, O with scales 10⁻³…10³ | 40 random + 4 adversarial (entries spanning 10⁻⁸…10⁸, dense O, one nonzero per column of V, ±1 entries with cancellation) | 0 |
| D | end to end: planted block decomposition V = Q_f[:, :d] M (ι ∈ [10⁻¹⁰, 10⁻²]), H_s random Hermitian, X its binary64 eigenvector, s and η the exact sin θ and normalisation defect (50 digits); the record's `bound` contains the exact tr(Π_Q O) | 40 | 0 |

Total: 254 cases, 0 violations. Tests `tests/test_correlation_lift.py` (7): on the 8-site Heisenberg ring the
lifted ⟨S^z_iS^z_j⟩ (36 pairs), S(0), S(π) and ⟨S⁺_iS⁻_j⟩ in the k = 0 and even-parity sectors agree with the
full-space ground-state values within the bounds (bounds < 10⁻¹¹); a full-space matrix observable is lifted; a
planted 0.1 % stretch of one basis vector fails S1 and makes the records INCONCLUSIVE (S2–S4 in the block stay
CERTIFIED: the lift is what fails); with S1 skipped the records are NOT_CHECKED; a planted ι = 0.05 at function
level shows |v − tr(Π_Q O)| > b_V but ≤ b (the isometry term is needed and sufficient); the compression
rounding bound is checked in exact rationals.

## 5 Analytic cross-check

1. *Exact isometry.* For ι = 0, M = I, δ = 0 and Π_V = Π_Q; Proposition 1 reduces to 0 ≤ 0 and the certified
   statement is Theorem 6 applied to VᴴOV, which for an exactly invariant V is the sector's spectral projector
   of H (Lemma 0). The 8-site tests have ι ≈ 10⁻¹⁶ (rounding of the orbit sums only): the isometry term is of
   order 10⁻¹⁵‖O‖ and the bounds are dominated by Theorem 6's rounding term.
2. *Rank-one stretch (the planted case).* V = Q diag(√(1+ι), 1, …) and P_s = e₀e₀ᵀ: Π_V = (1+ι) q₀q₀ᴴ and
   Π_Q = q₀q₀ᴴ, so |tr(Π_V O) − tr(Π_Q O)| = ι |⟨q₀, Oq₀⟩| ≤ ι‖O‖. Proposition 1 gives 2‖O‖δ(1 + √(1+ι)) with
   δ = 1 − √(1−ι) ≥ ι/2, i.e. ≥ ι‖O‖(1 + √(1+ι)) ≥ ι‖O‖: consistent, and tight up to the factor 1 + √(1+ι) ≈ 2
   for small ι (one of the two terms of the rank-2g split is empty here).
3. *SU(2) check of the two-letter lift.* In a singlet ⟨S⁺_iS⁻_j⟩ = 2⟨S^z_iS^z_j⟩ for i ≠ j and ⟨S⁺_iS⁻_i⟩ = 1/2 +
   ⟨S^z_i⟩ = 1/2; the lifted records reproduce both within their bounds (test), which checks the Hermitian
   decomposition path through the isometry against an independent identity.
4. *Relation to Theorem 8.* `compose.certify_low_energy` lifts eigenvalues through the square W = [V₁ ⋯ V_m]
   with the same δ = 1 − √(1−η) and ‖M − I‖ ≤ δ device (polar factor of a near-isometry); Proposition 1 is the
   same estimate applied to a projector of rank g instead of to the eigenvalues, with the rank-2g factor
   replacing Weyl's inequality.

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (240 random + 14 adversarial cases, 0 violations);
independent re-derivation: done 2026-10-30 (fresh-context session, verdict GAP, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
