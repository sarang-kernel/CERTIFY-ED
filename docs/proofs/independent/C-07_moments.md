# Independent re-derivation: C-07 (certified moments tr(P O Hᵏ O)/g and two sum-rule consistency statements)

Written from sections 1 and 3 of `docs/proofs/C-07_moments.md` only (section 2 read afterwards; comparison at the
end). Notation as there. Classical facts: Hölder |tr(AB)| ≤ ‖A‖₁‖B‖₂, ‖Z‖₁ ≤ rank(Z)‖Z‖₂; Higham (2002) eq. (3.5),
Lemma 3.3; the contracts of `fp.exact_dot`, `fp.norm2_upper`, `fp.inf_norm_upper` (Hermitian: ≥ ‖·‖₂) and of
Theorem 6 / `eigenspace.certify_observable` (|v − tr(PM)| ≤ β for a stored Hermitian M with a norm bound).

## Proofs

**Lemma 1.** P − XXᴴ = (P − Q̂Q̂ᴴ) + Q̂(I − M²)Q̂ᴴ (X = Q̂M). The first term has rank ≤ 2g and 2-norm ≤ s, so trace norm
≤ 2gs; the second has singular values |1 − m_i²| ≤ η (eigenvalues of M² = XᴴX in [1 − η, 1 + η]), so trace norm ≤ gη.
Hölder: |tr(PM) − tr(XXᴴM)| ≤ ‖P − XXᴴ‖₁‖M‖₂ ≤ (2gs + gη)‖M‖₂. ∎

**Lemma 2.** tr(XXᴴOHᵏO) = tr(XᴴOHᵏOX) = Σ_i x_iᴴOHᵏOx_i = Σ_i ⟨Ox_i, HᵏOx_i⟩ (O = Oᴴ), real since OHᵏO is Hermitian.
‖OHᵏO‖₂ ≤ ‖O‖₂²‖H‖₂ᵏ (submultiplicativity). Lemma 1 with M = OHᵏO, divided by g, gives the bound. ∎

**Lemma 3.** Fix i; y := Ox_i, ŷ = fl(Ox_i): |ŷ − y| ≤ γ̇_{w_O}|O||x_i| entrywise (row inner products with ≤ w_O terms), and
fl(|O||x_i|) ≥ (|O||x_i|)(1 − γ_{w_O}) (non-negative sums; complex moduli roundings absorbed by the factor-2 slack of γ̇),
so ‖ŷ − y‖ ≤ e_y. Likewise ‖ẑ^{(k)} − Hẑ^{(k−1)}‖ ≤ ε_k. Induction: ẑ^{(k)} − Hᵏŷ = (ẑ^{(k)} − Hẑ^{(k−1)}) + H(ẑ^{(k−1)} −
H^{k−1}ŷ), so ‖ẑ^{(k)} − Hᵏŷ‖ ≤ ε_k + ‖H‖D_{k−1} ≤ D_k. Now with δ := y − ŷ,
⟨y, Hᵏy⟩ − ⟨ŷ, ẑ^{(k)}⟩ = [⟨δ, Hᵏŷ⟩ + ⟨ŷ, Hᵏδ⟩ + ⟨δ, Hᵏδ⟩] + ⟨ŷ, Hᵏŷ − ẑ^{(k)}⟩, of modulus ≤ ‖H‖ᵏ(2‖ŷ‖ + e_y)e_y + ‖ŷ‖D_k;
|v_i − ⟨ŷ, ẑ^{(k)}⟩| ≤ d_i, and since ⟨y, Hᵏy⟩ is real, |Re v_i − ⟨y, Hᵏy⟩| ≤ |v_i − ⟨y, Hᵏy⟩|. Summing over i and
dividing by g; the floating-point sum of the g real parts carries ≤ γ_{g−1}Σ|Re v_i| ≤ γ_{g+1}Σ|Re v_i| (the code uses
`math.fsum`, which is correctly rounded — the γ term is pure slack), the division by g one more u|m̂_k|. ∎

**Lemma 4.** Induction over the left-to-right chain as in E-05 Lemma 4: the j-th product's entries are inner products
of length w₁ (j = 2; rows of F₁) or n (j ≥ 3; rows of the dense intermediate), so with g_j the corresponding γ̇ the
entrywise error is ≤ (Π(1 + g_j) − 1)|F₁|⋯|F_m| ≤ γ_M|F₁|⋯|F_m|, M = w₁ + (m − 2)n real, via γ_{2(a+4)} ≥ 2γ_{a+4} for complex
data (Higham Lemma 3.3). The computed non-negative chain underestimates by at most a relative γ_M; hence E =
up(γ_M)P̂/(1 − γ_M) ≥ |T̂ − F₁⋯F_m|. If F := F₁⋯F_m is Hermitian, 𝓗(T̂) − F = 𝓗(T̂ − F) has Frobenius norm ≤ ‖T̂ − F‖_F ≤ ‖E‖_F,
and the floating-point average adds ≤ u|𝓗(T̂)| entrywise, i.e. ≤ u‖𝓗(T̂)‖_F ≤ u‖T̂_h‖_F(1 + u); the u² shortfall of
u‖T̂_h‖_F is covered by the `up` guard on dev. ∎

**Proposition 1.** m₀ = tr(PO²)/g is enclosed by Lemma 3 with k = 0 (I_A). Theorem 6 for the stored Hermitian Ô₂ with
‖Ô₂‖₂ ≤ ‖O‖² + dev₂: |v − tr(PÔ₂)| ≤ β; |tr(PÔ₂) − tr(PO²)| ≤ ‖P‖₁‖Ô₂ − O²‖₂ ≤ g·dev₂ (Lemma 4); so m₀ ∈ I_B. Two
enclosures of one number intersect; disjointness contradicts a hypothesis (FAILED is the right verdict, CERTIFIED
means "consistent", not a new fact). ∎

**Proposition 2.** HP = PH and P² = P give PHP = HP = PH. [O, [H, O]] = O(HO − OH) − (HO − OH)O = 2OHO − O²H − HO² =: C,
Hermitian. tr(PC) = 2tr(POHO) − tr(PO²H) − tr(PHO²) and tr(PO²H) = tr(HPO²) = tr(PHPO²), tr(PHO²) = tr(PHPO²), so
tr(PC)/(2g) = m₁ − tr(PHPO²)/g. Next, B := PO²P ⪰ 0 is supported on range P with tr B = tr(PO²) = g m₀ ≥ 0, and
A′ := PHP satisfies aP ⪯ A′ ⪯ bP (every eigenvalue of H on range P lies in [a, b]); for B ⪰ 0 supported on range P,
tr(A′B) ∈ [a tr B, b tr B], and tr(PHPO²) = tr(PHP·PO²P) = tr(A′B). Hence tr(PHPO²)/g ∈ [a, b]·m₀ (an interval
product with m₀ ≥ 0; the code clips the m₀ interval at 0). The right-hand side: Theorem 6 on Ĉ_h with ‖Ĉ_h − C‖₂ ≤
dev_C (Lemma 4 applied to the three chains, the two subtractions adding ≤ u(|S₁| + |C|)(1 + 4u) entrywise) and
‖C‖₂ ≤ 2‖O‖²‖H‖ + ‖O‖²‖H‖ + ‖H‖‖O‖² = 4‖O‖²‖H‖, then /(2g). Two enclosures of the same number must intersect. ∎

## Code (current `certify_ed/moments.py`)

* H and O exactly Hermitian (`_as_matrix`, 60–71); norms by `inf_norm_upper` (139); underflow scan (148–162).
* Lemma 1: `proj_norm1 = up(2gs + gη)` (176); Lemma 2/3: Y = O @ X (170), e_y (171: γ̇_{w_O} fl(|O||x_i|)/(1 − γ_{w_O}),
  `norm2_upper`), ‖ŷ_i‖ upper (172), ε_k (179), Z = H @ Z (180), D_k = up(‖H‖D + ε) (181), `exact_dot` (184), the three
  rounding terms each with `up` (190), `math.fsum` of the real parts (191), sum_rnd = γ_{g+1}Σ|Re v| + u|total| (192),
  m̂ = total/g (193), proj_term = up(proj_norm1‖O‖²‖H‖ᵏ) (194), bound = up((…)/g) + 2u|m̂| (195), interval by
  sub_down/add_up (197). ‖H‖ᵏ as `Hn ** k` (a correctly-rounded libm pow, one rounding, inside `up`).
* Lemma 4: `hermitian_matrix_with_bound` 86–108 (widths: given for the first factor, n after; M with 2(w + 4) for complex;
  E = up(γ_M)/(1 − γ_M)·P̂).
* Proposition 1: 216–231 (O2h, dev₂ = up(‖E₂‖_F + u‖O2h‖_F), `certify_observable` with ‖Ô₂‖ ≤ up(‖O‖² + dev₂),
  endpoints (v ∓ up(β + g dev₂))/g with sub_down/add_up and a 4u step).
* Proposition 2: 240–267 (three chains with widths [w_O, n] / [w_H, n], S₁ = 2·OHO − OOH, C = S₁ − HOO, E_C with the
  subtraction roundings u(|S₁| + |C|)(1 + 4u), Ch, dev_C, ‖C‖ ≤ 4‖O‖²‖H‖, Theorem 6, /(2g) with 4u steps; m₀ clipped at
  0, `_interval_mul` with 4u outward steps, lhs by sub_down/sub_up, overlap test).
* `moments_from_certificate` 289–312: X, (s, η) from a CERTIFIED S4 cluster; level = union of the members' outward
  enclosures (300). Section 3's line numbers agree with the current file.

### Comparison with section 2 (read after the above)

Lemmas 1–4 and Propositions 1–2 agree with my derivations step by step (section 2 uses the polarisation identity
⟨y, My⟩ − ⟨ŷ, Mŷ⟩ = ⟨y − ŷ, M(y + ŷ)⟩ where I expanded the three cross terms — the same bound; for the interval
[a, b]·m₀ it uses the eigen-expansion of PHP where I used the Loewner inequalities aP ⪯ PHP ⪯ bP — equivalent). The
code mapping, the rounding directions and the `fsum` remark agree. No gap found.

**Verdict: AGREES.**

### Runs

* `python docs/proofs/verify_C-07.py --cases 40 --seed 9`: A 200 (exact rational tr(XᴴOHᵏOX)/g inside the rounding
  term), B 215 + 149 (exact moments inside the intervals, sum rules consistent), C 40 (wrong level refuted); negative
  controls 126/135 and 135/135; 415 random + 189 adversarial cases, 0 violations (328 s).
* `python docs/proofs/independent/adversarial_C-07_moments.py --cases 60 --seed 101`: Lemma 1 42 cases with the true s,
  η (18 skipped: η ≥ 1; M aligned with the projector error), Lemma 3 vector bounds 20 and Lemma 4 chain bounds 60
  (exact rationals; ±1 and 1 + 2⁻⁵² entries, scales 10^±3, complex), M1 60 end-to-end records (K ≤ 4, g ≤ 3, exact
  eigenspace at 60 digits), M2/M3 60 (CERTIFIED in all; Theorem-6 and lhs intervals contain the exact quantities);
  informational: with random orthonormal vectors and the claimed s the sum rules reported FAILED in 47 of 60 records;
  0 violations (147 s).
