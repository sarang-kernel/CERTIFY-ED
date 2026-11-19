# Independent re-derivation: C-04 (lift of observables through the sector isometry)

Not a human check. Written from sections 1 and 3 of `docs/proofs/C-04_correlation_lift.md` only (section 2 read
afterwards; the comparison is at the end). Notation: V (n × d_s) the producer's sector basis, ι ≥ ‖VᴴV − I‖₂ with ι < 1,
H_s the supplied block, P_s the spectral projector of H_s on the cluster (rank g), O Hermitian on the full space, Õ :=
VᴴOV, Õ_h the stored Hermitian part of fl(Vᴴ fl(OV)), Π_V := V P_s Vᴴ, Π_Q := Q P_s Qᴴ with V = QM the polar
decomposition, δ := 1 − √(1 − ι). Classical facts used: Higham (2002) Lemma 3.1 (any summation order), Lemma 3.3 (γ_j +
γ_k + γ_jγ_k ≤ γ_{j+k}), Lemma 3.5 (complex product: |fl(xy) − xy| ≤ √2 γ₂ |x||y|), §3.6; the polar decomposition; ‖AB‖₁
≤ ‖A‖₁‖B‖₂ and |tr(AB)| ≤ ‖A‖₁‖B‖₂ (Hölder for Schatten norms); ‖Z‖₁ ≤ rank(Z)·‖Z‖₂.

## Proofs

**Polar factor.** The eigenvalues of VᴴV lie in [1 − ι, 1 + ι] ⊂ (0, 2), so M := (VᴴV)^{1/2} is Hermitian positive
definite with spectrum in [√(1−ι), √(1+ι)] and Q := VM⁻¹ satisfies QᴴQ = M⁻¹VᴴVM⁻¹ = I, range Q = range V.
‖M − I‖₂ = max(1 − √(1−ι), √(1+ι) − 1) = 1 − √(1−ι) = δ because t ↦ √t is concave (the chord from 1−ι to 1+ι lies
below the graph, so 1 − √(1−ι) ≥ √(1+ι) − 1). ‖M‖₂ ≤ √(1+ι).

**Lemma 0.** tr(P_s VᴴOV) = tr(V P_s Vᴴ O) = tr(Π_V O) (cyclicity). Π_Q² = Q P_s (QᴴQ) P_s Qᴴ = Π_Q, Π_Qᴴ = Π_Q,
rank Π_Q = rank P_s = g (Q is injective, Qᴴ surjective onto ℂ^{d_s}), range Π_Q ⊂ range Q = range V. If ι = 0 then
M = I, V = Q; if moreover ν = 0 then HV = VH_s, so range V is H-invariant, H restricted to range V is V H_s Vᴴ, and
its spectral projector on the cluster is V P_s Vᴴ = Π_V = Π_Q. ∎

**Lemma 1 (compression rounding).** Write W := fl(OV) and Õ_fl := fl(VᴴW).
(a) Entry (OV)_{ik} = Σ_j O_{ij}V_{jk} is an inner product with at most w_O nonzero terms (the nonzeros of row i
of O); zero terms are added exactly. Real data: |W − OV| ≤ γ_{w_O} |O||V| entrywise (Higham eq. (3.5), valid for
every summation order). Complex data: each product carries a relative error ≤ √2 γ₂ ≤ γ₃ (Lemma 3.5) and the
recursive summation of m complex terms carries factors (1 + θ), |θ| ≤ γ_{m−1} (complex addition is rounded
componentwise, so |fl(z) − z| ≤ u|z| and Lemma 3.1 applies verbatim), hence |W − OV| ≤ c_{w_O} |O||V| with
c_m := (1 + γ₃)(1 + γ_{m−1}) − 1 ≤ γ_{m+2} (Lemma 3.3).
(b) Entry (VᴴW)_{lk} = Σ_i conj(V_{il}) W_{ik} has at most w_V nonzero terms (nonzeros of column l of V), so
|Õ_fl − VᴴW| ≤ c_{w_V} |V|ᵀ|W| with c_m = γ_m (real) or ≤ γ_{m+2} (complex).
(c) Õ_fl − Õ = Vᴴ(W − OV) + (Õ_fl − VᴴW), so entrywise
|Õ_fl − Õ| ≤ (c_{w_O} + c_{w_V}(1 + c_{w_O})) |V|ᵀ|O||V| = ((1 + c_{w_O})(1 + c_{w_V}) − 1) |V|ᵀ|O||V|.
Real: (1+γ_a)(1+γ_b) − 1 ≤ γ_{a+b} = γ̇_{a+b}. Complex: ≤ (1+γ_{a+2})(1+γ_{b+2}) − 1 ≤ γ_{a+b+4} ≤ 2γ_{a+b+4} = γ̇_{a+b}.
Hence ‖Õ_fl − Õ‖_F ≤ γ̇_{w_O+w_V} ‖ |V|ᵀ|O||V| ‖_F (the Frobenius norm is monotone in the absolute entries).
*Caution for the complex case:* the shortcut "γ̇_a + γ̇_b + γ̇_aγ̇_b ≤ γ̇_{a+b}" is false for the project's
γ̇_m = 2γ_{m+4} (a = b = 1: 2γ₅ + 2γ₅ ≈ 20u > 2γ₆ ≈ 12u); the composition has to be done with Higham's constants,
as above, and then the stated constant holds with room to spare (γ_{a+b+4} ≤ γ̇_{a+b}/2).
(d) Hermitian part. Õ is exactly Hermitian (O Hermitian as stored). Õ_h = fl((Õ_fl + Õ_flᴴ)/2): the halving is
exact (no underflow), the addition is rounded componentwise, so Õ_h = 𝓗(Õ_fl) + E₃ with |E₃| ≤ u |𝓗(Õ_fl)| entrywise,
where 𝓗(Z) = (Z + Zᴴ)/2. Since 𝓗 is linear, 𝓗(Õ) = Õ and ‖𝓗(Z)‖_F ≤ ‖Z‖_F,
‖Õ_h − Õ‖_F ≤ ‖𝓗(Õ_fl − Õ)‖_F + ‖E₃‖_F ≤ ‖Õ_fl − Õ‖_F + u ‖Õ_fl‖_F ≤ γ̇_{w_O+w_V} ‖|V|ᵀ|O||V|‖_F + u ‖Õ_fl‖_F,
which is ≤ r_c as stated (the factor (1 + u) is slack); ‖·‖₂ ≤ ‖·‖_F. Exact Hermiticity of Õ_h: rounding commutes
with conjugation and with the swap of the two summands, so fl((a + b̄)/2) and fl((b + ā)/2) are conjugate; the
diagonal fl((a + ā)/2) = Re a is exact. ∎

**Proposition 1 (isometry term).** Π_V − Π_Q = Q(M P_s M − P_s)Qᴴ and M P_s M − P_s = (M − I)P_s M + P_s(M − I).
For any Z, |tr(Q Z Qᴴ O)| = |tr(Z QᴴOQ)| ≤ ‖Z‖₁ ‖QᴴOQ‖₂ ≤ ‖Z‖₁ ‖O‖₂ (QᴴOQ is a compression by an isometry).
Both summands have rank ≤ g, so ‖(M − I)P_s M‖₁ ≤ g δ √(1+ι) and ‖P_s(M − I)‖₁ ≤ g δ, giving

    |tr(Π_V O) − tr(Π_Q O)| ≤ g ‖O‖₂ δ (1 + √(1+ι)) ≤ 2g ‖O‖₂ δ (1 + √(1+ι)).

The stated bound therefore holds with a factor 2 to spare (it is what one gets from rank(Π_V − Π_Q) ≤ 2g and
‖Π_V − Π_Q‖₂ ≤ ‖M P_s M − P_s‖₂ ≤ δ(1 + √(1+ι))). ∎

**Conclusion.** b₆ is a hypothesis (Theorem 6 in the block, for the stored Hermitian Õ_h with the norm bound
‖Õ_h‖₂ ≤ `inf_norm_upper(Õ_h)`): |v − tr(P_s Õ_h)| ≤ b₆. Then |tr(P_s Õ_h) − tr(P_s Õ)| ≤ ‖P_s‖₁ ‖Õ_h − Õ‖₂ = g ‖Õ_h − Õ‖₂
≤ g r_c (Lemma 1), and tr(P_s Õ) = tr(Π_V O) (Lemma 0): |v − tr(Π_V O)| ≤ b₆ + g r_c = b_V. Adding Proposition 1:
|v − tr(Π_Q O)| ≤ b_V + 2g‖O‖₂δ(1 + √(1+ι)) = b. The identification of range Π_Q with an eigenspace of H is not
claimed (it would need ν = 0, Lemma 0); the record states this. Non-Hermitian O: O = H + iK exactly (C-05 check),
tr(Π O) = tr(Π H) + i tr(Π K), the two real bounds add (|a + ib − (a' + ib')| ≤ |a − a'| + |b − b'|). ∎

## Code (current `certify_ed/correlations.py`)

Section 3 cites the line numbers of `track/new-statements`; in the current file `compress_operator` is 287–309,
`_isometry_term` 312–317, `certify_observable_lifted` 320–371, `certify_correlations_lifted` 374–419.
* Lemma 1: `OV = O @ V`, `Ot = V.conj().T @ OV` (sparse binary64, any accumulation order — covered); w_O = max row
  nonzeros of O (300), w_V = max column nonzeros of V (301, via csc); `absprod` = |V|ᵀ(|O||V|) computed in floating
  point with nonnegative data, so its entries are ≥ exact·(1 − γ_{w_O+w_V}) ≥ exact·(1 − γ̇_{w_O+w_V}): the factor
  `up(1/(1 − g_abs))` (305) restores an upper bound; `g_abs = gamma_dot(w_O + w_V, cplx)` is γ̇ (for the absolute
  product γ_{w_O+w_V} would do — conservative). `fro_upper` is an upper bound on the Frobenius norm of the stored
  data. Hermitian part 306 (`(Ot + Ot.conj().T) * 0.5`, exactly Hermitian; `eliminate_zeros` only drops stored
  zeros); rounding term `U * fro_upper(Ot) * (1 + 4U)` (308) = u‖Õ_fl‖_F with slack; the sum guarded by `up`
  (1 + 8u, covering the ≤ 5 unguarded roundings of 304–308: two in γ, two products, one addition).
  Returns `inf_norm_upper(Oh)` ≥ ‖Õ_h‖₂ (Hermitian).
* Proposition 1: `_isometry_term` (312–317): ∞ when ι ≥ 1; δ = `up(1 − sqrt(1 − ι)) + 4U` — the three roundings of
  1 − √(1 − ι) have absolute error ≤ 2u (1 − ι is exact for ι ≥ 1/2, else relatively accurate to u/2; √ correctly
  rounded; the final subtraction is exact by Sterbenz), so the +4U absolute term and the guard cover them; result
  `up(2 g O_norm δ (1 + sqrt(1 + ι)) (1 + 4U))` — 4 roundings covered by (1 + 4u)(1 + 8u).
* Conclusion: `certify_observable_lifted`: Hermiticity test `abs(Oc − Oc.conj().T).max() == 0` (331), i.e. O exactly
  Hermitian as stored; `certify_observable(ec, X, Oh, …, O_norm=Oh_norm)` (334) is b₆ with v = tr(XᴴÕ_hX) (its own
  rounding term inside); `O_norm = inf_norm_upper(Oc)` ≥ ‖O‖₂ (336); `bound_V = add_up(b₆, up(g·rnd))` (337),
  `bound = add_up(bound_V, iso_term)` (339). Non-Hermitian: `hermitian_parts` (347), recursion on the two parts,
  `add_up` of the bounds (357), `INCONCLUSIVE` when the parts were not formed exactly (362–365). All directions
  outward; `value` and the `statement` intervals are display only.
* `certify_correlations_lifted`: O_iO_j symmetrised as (O + Oᴴ)/2 for i ≠ j — exact only when O_iO_j is already
  Hermitian (commuting Hermitian factors on different sites: the product is Hermitian, so (O + Oᴴ)/2 = O with the
  halving of a doubled value exact; for a two-letter operator the Hermitian decomposition path is used).

### Comparison with section 2 (read after the above)

Lemma 0, Proposition 1 (via rank 2g), the Conclusion and the ‖O‖₂ ≤ ‖O‖_∞ remark agree with my derivation.
Lemma 1 (i), (iii), (iv) and the treatment of the floating-point absolute product agree.

**GAP (proof of Lemma 1 (ii), complex data).** Section 2 composes the two stages with "(1 + γ̇_a)(1 + γ̇_b) ≤ 1 + γ̇_{a+b},
the same for γ̇ since it is γ of a shifted index times a constant ≥ 1 applied consistently". For the project's
complex constant γ̇_m = 2γ_{m+4} this is false: with a = b = 1, (1 + γ̇₁)² − 1 = 4γ₅ + 4γ₅² ≈ 20u while
γ̇₂ = 2γ₆ ≈ 12u (checked numerically with `fp.gamma_dot`; the inequality fails for every pair of widths I tried,
e.g. (100, 100): 416u vs 408u). The doubling and the index shift do not commute with the composition. The
*statement* of Lemma 1 is nevertheless true: composing Higham's own constants (product error √2γ₂ ≤ γ₃, summation
γ_{m−1}, so each stage is ≤ γ_{m+2} relative to |·|-products) gives (1 + γ_{a+2})(1 + γ_{b+2}) − 1 ≤ γ_{a+b+4} ≤
γ̇_{a+b}/2 — see my proof of Lemma 1 (c). For real data the step is Higham's Lemma 3.3 and is correct. No
certified bound is affected (the constant in the code, `gamma_dot(w_O + w_V, cplx)`, is the one the corrected
proof justifies, with a factor 2 to spare); the correction is to the proof text only.

**Remarks.** (1) Proposition 1 holds with g in place of 2g (the two summands of M P_s M − P_s each have trace norm
≤ g‖·‖₂; my adversarial run confirms the sharper bound on 150 cases) — the code's 2g is conservative. (2) Section 3
line numbers are those of `track/new-statements`; current positions are listed above. (3) In Lemma 1 the code
applies γ̇ (complex) also to the absolute product's underestimation, where γ (real) suffices — conservative.

**Verdict: GAP** (invalid step in the proof of Lemma 1 for complex data; statement true by a different argument;
certified bounds unaffected).

### Runs

* `python docs/proofs/verify_C-04.py 5` (positional seed; fixed sizes): A 60, B 110, C 44, D 40; TOTAL 254 cases,
  0 violations (62 s).
* `python docs/proofs/independent/adversarial_C-04_correlation_lift.py --cases 150 --seed 61`: Lemma 1 150 cases
  (exact rationals; real/complex; ±1 cancellation rows, 1 ± 2⁻⁵², 0/1 sector bases, scales 10^±40, 2⁻⁴⁰), 0
  violations and Õ_h exactly Hermitian in all; Proposition 1 150 cases with the true ι up to 0.999 and g = d_s, 0
  violations for both the stated 2g bound and the sharper g bound; end to end `certify_observable_lifted` 47 cases
  (3 skipped: η ≥ 1 or ι ≥ 1), Hermitian and exactly decomposable non-Hermitian O, exact 0/1 and perturbed V, scales
  10^±30 and 2^±90, 0 violations (94 s).
