# C-07 (first part) Certified moments tr(P O Hᵏ O)/g on a certified eigenspace, and two sum-rule consistency statements

Notation as in `docs/theory.md`: H Hermitian n × n, O Hermitian n × n (both stored), X (n × g) the stored vectors of an
S4 cluster with claimed values in the level interval [a, b], P the spectral projector of H onto the cluster's g
eigenvalues (rank g), η = ‖XᴴX − I‖₂ < 1, Q̂ the polar factor of X, s ≥ ‖Q̂Q̂ᴴ − P‖₂ the certified sin θ bound (Theorem 6 or
E-07). ‖·‖ is the Euclidean / spectral norm, ‖·‖₁ the trace norm, ‖·‖_F Frobenius; u = 2⁻⁵³, γ_m, γ̇_m as in `fp.py`; w_H,
w_O the largest numbers of nonzeros per row of H and O. The moments are m_k := tr(P O Hᵏ O)/g, k = 0, …, K.

## 1 Statement

**Lemma 1 (projector term).** For every Hermitian M, |tr(PM) − tr(XXᴴM)| ≤ (2gs + gη)‖M‖₂.

**Lemma 2 (moments through the stored vectors).** tr(XXᴴ O Hᵏ O) = Σ_{i=1}^{g} ⟨Ox_i, Hᵏ(Ox_i)⟩ (real), and
‖O Hᵏ O‖₂ ≤ ‖O‖₂²‖H‖₂ᵏ. Hence, with Lemma 1,

    | m_k − (1/g) Σ_i ⟨Ox_i, Hᵏ Ox_i⟩ | ≤ (2s + η) ‖O‖₂² ‖H‖₂ᵏ.

**Lemma 3 (floating-point accounting of the repeated application; `moments.certified_moments`).** Let ŷ_i = fl(Ox_i),
ẑ_i^{(0)} = ŷ_i and ẑ_i^{(k)} = fl(H ẑ_i^{(k−1)}). Put e_y,i := `norm2_upper`(γ̇_{w_O} fl(|O||x_i|)/(1 − γ_{w_O})) ≥ ‖ŷ_i − Ox_i‖,
ε_k,i := `norm2_upper`(γ̇_{w_H} fl(|H||ẑ_i^{(k−1)}|)/(1 − γ_{w_H})) ≥ ‖ẑ_i^{(k)} − Hẑ_i^{(k−1)}‖, D_0,i = 0 and
D_k,i := up(‖H‖·D_{k−1,i} + ε_k,i) ≥ ‖ẑ_i^{(k)} − Hᵏŷ_i‖. Let (v_i, d_i) = `exact_dot`(ŷ_i, ẑ_i^{(k)}) (|v_i − ⟨ŷ_i, ẑ_i^{(k)}⟩| ≤ d_i)
and m̂_k = fl(Σ_i Re v_i)/g. Then

    | m̂_k − (1/g) Σ_i ⟨Ox_i, HᵏOx_i⟩ | ≤ (1/g) Σ_i [ ‖H‖ᵏ (2‖ŷ_i‖ + e_y,i) e_y,i + ‖ŷ_i‖ D_k,i + d_i ] + γ_{g+1} Σ_i |Re v_i|/g + u|m̂_k|,

where ‖ŷ_i‖ is replaced by its upper bound `norm2_upper`(ŷ_i) and ‖H‖, ‖O‖ by the row-sum bounds of `inf_norm_upper`.
The certified interval of M1 is m̂_k ± b_k with b_k = up(projector term + rounding terms)/g + 2u|m̂_k| (one more rounding
than the division needs). Hᵏ is never formed.

**Lemma 4 (explicitly formed products; `moments.hermitian_matrix_with_bound`).** For stored F₁, …, F_m (n × n) and the
left-to-right floating-point product T̂ = fl(⋯fl(fl(F₁F₂)F₃)⋯F_m), with w₁ the largest number of nonzeros per row of F₁,

    |T̂ − F₁⋯F_m| ≤ γ_M |F₁|⋯|F_m|   entrywise,   M = w₁ + (m − 2) n   (real data; 2(w₁ + 4) + (m − 2)·2(n + 4) complex).

The computed |F₁|⋯|F_m| is divided by (1 − γ_M) and multiplied by up(γ_M). The Hermitian average T̂_h = (T̂ + T̂ᴴ)/2
of a product whose exact value is Hermitian satisfies ‖T̂_h − F₁⋯F_m‖₂ ≤ ‖E‖_F + u‖T̂_h‖_F.

**Proposition 1 (M2, zeroth-moment sum rule).** m₀ = tr(P O²)/g. The moment route gives m₀ ∈ I_A (Lemma 3 with k = 0);
Theorem 6 applied to the formed Hermitian matrix Ô₂ = fl(O·O)_h with ‖Ô₂ − O²‖₂ ≤ dev₂ gives tr(P O²)/g ∈ I_B :=
[(v − β − g·dev₂)/g, (v + β + g·dev₂)/g] where (v, β) is `eigenspace.certify_observable`'s (value, bound) for Ô₂ with
‖Ô₂‖₂ ≤ ‖O‖² + dev₂. Both intervals contain m₀; the statement is CERTIFIED when they overlap and FAILED when they are
disjoint (a contradiction between two proven enclosures: inconsistent inputs or a bug).

**Proposition 2 (M3, first-moment sum rule, f-sum form).** For the cluster projector P with HP = PH and every eigenvalue
of H on range P in [a, b]:

    m₁ − tr(P H P O²)/g = tr(P [O, [H, O]]) / (2g),   and   tr(P H P O²)/g ∈ [a, b]·m₀   (interval product, m₀ ≥ 0).

The left-hand side is enclosed by I_{m₁} − [a, b]·I_{m₀}; the right-hand side by Theorem 6 on the formed Hermitian
matrix Ĉ_h with C = [O, [H, O]] = 2OHO − O²H − HO², ‖Ĉ_h − C‖₂ ≤ dev_C, ‖C‖₂ ≤ 4‖O‖²‖H‖, divided by 2g. CERTIFIED when
the two intervals overlap, FAILED when disjoint. Not a new theorem: a consistency check between two certified routes.

**M4 (not implemented).** Continued-fraction (Lanczos) bounds on the spectral function from the certified moments:
NOT_CHECKED with NOT_IMPLEMENTED (`docs/theory.md` §9, `docs/SCOPE_MATRIX.md`).

## 2 Proof

**Lemma 1.** P − Q̂Q̂ᴴ is Hermitian of rank ≤ 2g (the difference of two rank-g projectors) with ‖P − Q̂Q̂ᴴ‖₂ ≤ s, so
‖P − Q̂Q̂ᴴ‖₁ ≤ 2gs (trace norm ≤ rank × spectral norm). XXᴴ − Q̂Q̂ᴴ = Q̂(M² − I)Q̂ᴴ with X = Q̂M, M² = XᴴX: it is Hermitian of
rank ≤ g with ‖·‖₂ ≤ ‖M² − I‖₂ = η, so ‖XXᴴ − Q̂Q̂ᴴ‖₁ ≤ gη. Hölder: |tr((P − XXᴴ)M)| ≤ ‖P − XXᴴ‖₁‖M‖₂ ≤ (2gs + gη)‖M‖₂. ∎

**Lemma 2.** tr(XXᴴ O Hᵏ O) = tr(Xᴴ O Hᵏ O X) = Σ_i x_iᴴ O Hᵏ O x_i = Σ_i ⟨Ox_i, Hᵏ Ox_i⟩ (O Hermitian), real because O Hᵏ O
is Hermitian. ‖O Hᵏ O‖₂ ≤ ‖O‖₂‖H‖₂ᵏ‖O‖₂ (submultiplicativity). Combine with Lemma 1 for M = O Hᵏ O and divide by g. ∎

**Lemma 3.** (i) *ŷ_i.* Each entry of fl(Ox_i) is an inner product of ≤ w_O terms: |ŷ_i − Ox_i| ≤ γ̇_{w_O}|O||x_i| entrywise
(Higham 2002 eq. (3.5); `gamma_dot` for complex data); the computed fl(|O||x_i|) of non-negative terms is ≥ (1 − γ_{w_O})
times the exact one; `norm2_upper` bounds the 2-norm of the entrywise bound: ‖ŷ_i − Ox_i‖ ≤ e_y,i. (ii) *ẑ_i^{(k)}.*
Likewise ‖ẑ_i^{(k)} − Hẑ_i^{(k−1)}‖ ≤ ε_k,i, and by induction ‖ẑ_i^{(k)} − Hᵏŷ_i‖ ≤ ‖H‖‖ẑ_i^{(k−1)} − Hᵏ⁻¹ŷ_i‖ + ε_k,i ≤
‖H‖D_{k−1,i} + ε_k,i, which `up` rounds up: D_k,i. (iii) *Exact quantity versus computed.* With M = Hᵏ (Hermitian),
⟨Ox_i, M Ox_i⟩ − ⟨ŷ_i, Mŷ_i⟩ = ⟨Ox_i − ŷ_i, M(Ox_i + ŷ_i)⟩ (polarisation for Hermitian M, the cross terms being conjugate),
of modulus ≤ e_y,i ‖M‖ (‖ŷ_i‖ + ‖Ox_i‖) ≤ ‖H‖ᵏ(2‖ŷ_i‖ + e_y,i)e_y,i. Then |⟨ŷ_i, Mŷ_i⟩ − ⟨ŷ_i, ẑ_i^{(k)}⟩| ≤ ‖ŷ_i‖D_k,i, and
|⟨ŷ_i, ẑ_i^{(k)}⟩ − v_i| ≤ d_i. (iv) *Sum.* `math.fsum` is correctly rounded, so fl(Σ Re v_i) = (Σ Re v_i)(1 + δ), |δ| ≤ u —
the term γ_{g+1}Σ|Re v_i| is the bound for an arbitrary summation order (Higham Lemma 8.4), kept for the case where fsum
is replaced; the division by g adds u|m̂_k|. The code adds 2u|m̂_k| after `up` of the sum. ∎

**Lemma 4.** Induction over the products, exactly as in Lemma 4 of `docs/proofs/E-05_variational.md` with inner products
of length w₁ (first product, entries of F₁F₂ are sums over the nonzeros of a row of F₁) and n afterwards (the running
product is dense): |T̂_j − T_j| ≤ ((1 + c_{j−1})(1 + γ̇_{ℓ_j}) − 1)|F₁|⋯|F_j| with ℓ_j the inner-product length, and
∏(1 + γ_{ℓ_j}) − 1 ≤ γ_{Σℓ_j} = γ_M (Higham Lemma 3.3; complex constants as in E-05). The Hermitian average: T̂_h − T = ((T̂ − T) + (T̂ − T)ᴴ)/2 plus the rounding of the average (≤ u|T̂_h| entrywise, since the exact average is within u of the computed one). ‖((T̂ − T) + (T̂ − T)ᴴ)/2‖₂ ≤ ‖T̂ − T‖₂ ≤ ‖T̂ − T‖_F ≤ ‖E‖_F (entrywise |T̂ − T| ≤ E), and the rounding part has 2-norm ≤ u‖T̂_h‖_F. ∎

**Proposition 1.** m₀ = tr(P O I O)/g = tr(P O²)/g (k = 0). Theorem 6 (`eigenspace.certify_observable`, `docs/theory.md`
§4): |tr(P Ô₂) − tr(Xᴴ Ô₂ X)| ≤ 2g‖Ô₂‖₂(s + η) + rounding = β, and tr(P O²) = tr(P Ô₂) + tr(P(O² − Ô₂)) with
|tr(P(O² − Ô₂))| ≤ ‖P‖₁‖O² − Ô₂‖₂ ≤ g·dev₂ (Hölder, ‖P‖₁ = g). Hence tr(P O²)/g ∈ I_B. Two intervals that both contain
m₀ intersect; disjointness contradicts one of the two proofs' hypotheses (the inputs s, η, the stored data) and is
reported as FAILED. ∎

**Proposition 2.** Since HP = PH and P² = P, tr(P O H O) − tr(P H P O²) = tr(P O H O) − tr(H P O²). Expand
tr(P[O,[H,O]]) = tr(P(OHO − OOH − HOO + OHO)) = 2tr(POHO) − tr(POOH) − tr(PHOO) = 2tr(POHO) − tr(HPO²) − tr(PHO²) =
2tr(POHO) − 2tr(PHO²) (cyclicity and HP = PH). Dividing by 2g: tr(P[O,[H,O]])/(2g) = m₁ − tr(PHPO²)/g. For the second
claim: PHP = Σ_j λ_j|φ_j⟩⟨φ_j| over the cluster eigenpairs with λ_j ∈ [a, b], so tr(PHPO²) = Σ_j λ_j ⟨φ_j, O²φ_j⟩ with
⟨φ_j, O²φ_j⟩ = ‖Oφ_j‖² ≥ 0 and Σ_j‖Oφ_j‖² = tr(PO²) = g·m₀; a convex combination of the λ_j weighted by non-negative
numbers summing to g·m₀ lies in [a·g·m₀, b·g·m₀]. The interval product [a, b]·[m₀_lo, m₀_hi] (with m₀_lo clipped at 0,
since m₀ ≥ 0) contains it; the code rounds its endpoints outward by 4u. The left-hand side interval is I_{m₁} minus the
product interval (`sub_down`/`sub_up`); the right-hand side is Theorem 6 on Ĉ_h plus g·dev_C as in Proposition 1, with
‖C‖₂ ≤ 2‖O‖²‖H‖ + 2‖O‖²‖H‖ = 4‖O‖²‖H‖ and dev_C from Lemma 4 applied to the three products (the bounds add; the two
subtractions add u(|S₁| + |C|) entrywise, with (1 + 4u) for their own roundings). ∎

## 3 Where the code implements it

`certify_ed/moments.py` (branch `track/variational`):

* Lemma 1: `proj_norm1` at line 176 (`up(2gs + gη)`), used in `proj_term` at 194 with ‖O‖²‖H‖ᵏ.
* Lemma 2 / Lemma 3: ŷ = fl(OX) at 170, e_y at 171, ‖ŷ_i‖ upper at 172, ε_k at 179, ẑ^{(k)} = fl(Hẑ^{(k−1)}) at 180, D_k at
  181 (`up`), `exact_dot` at 184, the three rounding terms at 190, the sum term at 192, the interval half-width at 195
  (`up`, + 2u|m̂_k|). Hᵏ is never formed (only H @ Z).
* Lemma 4: `hermitian_matrix_with_bound` (86–110): M accumulated at 99, E at 107 (`up(γ_M)/(1 − γ_M)`).
* Proposition 1: 216–233 (Ô₂ at 216, dev₂ at 218, Theorem 6 at 219 with ‖Ô₂‖ ≤ ‖O‖² + dev₂, the interval at 220–223,
  overlap at 225; FAILED when disjoint).
* Proposition 2: 240–270 (the three products 241–243, C and E_C at 244–246, dev_C at 248, ‖C‖ bound at 249, Theorem 6 at
  250, the right-hand interval 251–254, the product interval at 257, the left-hand interval at 258, overlap at 259).
* `moments_from_certificate` (289–312): reads X, (s, η) of the CERTIFIED S4 cluster and the level interval (union of the
  members' outward-rounded enclosures, line 300) from a certificate; warns when H's hash differs from the certificate's.
* The no-underflow scan of X, H, O at 149–151 (INCONCLUSIVE / UNDERFLOW_RISK before any bound); M4 at `_m4` (277–281).

Rounding directions: every error term up (`norm2_upper`, `up`, `add_up`), interval endpoints outward (`sub_down`,
`add_up`, `sub_up`, the 4u steps of the interval product).

**Correction (2026-11-18), complex moduli; UNVERIFIED (D-12).** Lemma 3 (i)–(ii) and Lemma 4 take the computed
moduli |O|, |H|, |x_i|, |ẑ_i| and |F_j| as exact ("the computed fl(|O||x_i|) of non-negative terms is ≥ (1 − γ_{w_O})
times the exact one"). For complex data the code took them from `numpy.abs`, whose error is not bounded (`docs/proofs/fp_modulus-up.md`).
Since 3.3.0.dev0 `moments` takes them from `fp.abs_up` and `fp.modulus_up` (≥ the exact moduli, Lemma M there),
so these steps hold as written; the rounding term u(|S1| + |C|) of the formed commutator in M3 uses upper moduli too.
No constant changes.

## 4 Computational verification

`docs/proofs/verify_C-07.py` (seed 20261030, `--cases 40`; exact Fractions for the rounding part, mpmath at 60 digits
for the exact P and moments). Run of 2026-10-31 (`docs/proofs/runs/verify_C-07.txt`):

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | Lemma 3 alone (s = η = 0 fed): \|m̂_k − exact rational tr(XᴴOHᵏOX)/g\| ≤ the recorded rounding term, random H, O (n = 2…5, g = 1…2, real and complex, scales 10⁻²…10², k = 0…4) | 200 | 0 |
| B | full statement: random H with a planted cluster (n = 4…7, g = 1…3, real and complex), exact P and exact tr(POHᵏO)/g at 60 digits, X = rotated exact eigenvectors + perturbation 10⁻⁹…10⁻² with the exact s, η (rounded up) as inputs: every moment inside its interval and both sum rules CERTIFIED; adversarial: vectors tilted by 10⁻³ out of the eigenspace with the exact s; Heisenberg chains N = 4, 6 with O = S^z₀S^z₁ and the exact ground-state projector, k = 0…4 | 218 random + 152 adversarial | 0 |
| C | **negative controls**: a wrong level interval (shifted beyond the combined widths of the two certified intervals divided by m₀) makes M3 FAILED in 40/40 cases; s understated by 10⁴ on the tilted vectors breaks the enclosure in 132/138 moments; a random orthonormal X with s = η = 0 claimed breaks it in 138/138 moments | 40 adversarial | 0 |

Total: 418 random + 192 adversarial cases, 0 violations. Tests `tests/test_moments.py` (6): the 8-site Heisenberg ring from a certificate (ground state k = 0…3
and the triplet cluster against exact traces; a rotated triplet basis), the tilted-vector negative control (exact s
holds, s/10⁴ breaks; a random unit vector with s = η = 0 breaks every moment), the wrong level interval (FAILED), K = 0
and no level (NOT_CHECKED), a complex operator with the exact-rational rounding check, underflow and input checks, the
record schema.

## 5 Analytic cross-check

1. *Exact eigenvector, k = 0, 1.* X = φ (unit, exact), s = η = 0: m₀ = ‖Oφ‖², m₁ = ⟨Oφ, HOφ⟩, and the intervals shrink to
   the rounding terms (≈ 10⁻¹³ on the 8-site ring: tests). The f-sum rule reads ⟨Oφ, HOφ⟩ − λ‖Oφ‖² = ⟨φ, [O,[H,O]]φ⟩/2, the
   textbook form ⟨φ|[O,[H,O]]|φ⟩ = 2Σ_n (E_n − E_φ)|⟨n|O|φ⟩|² (Thomas–Reiche–Kuhn structure), which the moment route
   reproduces: m₁ − λm₀ = Σ_n (E_n − λ)|⟨n|O|φ⟩|².
2. *O commuting with H.* [H, O] = 0: C = 0, so the right-hand side of M3 is 0 ± rounding, and m₁ = tr(PHO²)/g with
   PHO² = PHPO² exactly, so the left-hand side is 0 up to the width of [a, b]·m₀ — both intervals contain 0. For O = S^z_total
   on the Heisenberg ring (a conserved quantity) this is the case.
3. *Scaling.* O → tO multiplies every m_k by t² and every bound by t² (‖O‖² enters each term quadratically, e_y linearly in
   the product with ‖ŷ‖ ∝ t); H → tH multiplies m_k by tᵏ and the bound's ‖H‖ᵏ, D_k by tᵏ: the enclosures are scale-covariant.
4. *Projector term versus first-order perturbation.* For X = φ + εχ (χ ⊥ φ, unit), the exact error of ⟨Ox, HᵏOx⟩ against
   tr(POHᵏO) is 2ε Re⟨Oφ, HᵏOχ⟩ + O(ε²), of modulus ≤ 2ε‖OHᵏO‖, while s = ε/√(1 + ε²) + O(ε²) and Lemma 1 gives
   (2s + η)‖OHᵏO‖ ≈ 2ε‖OHᵏO‖: the projector term is first order with the same constant, tight up to the ratio
   ‖OHᵏO‖/|⟨Oφ, HᵏOχ⟩| (which is why the negative control needs an understatement of 10⁴ and a tilt direction with a
   nonzero matrix element).

## 6 Status

Derived on 2026-10-31; machine-checked (418 random + 192 adversarial cases, 0 violations; negative controls 132/138, 138/138 and 40/40); independent re-derivation: done 2026-10-30 (from the statement alone, not a human check; verdict AGREES, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
Correction for complex moduli on 2026-11-18 (end of section 3; UNVERIFIED, D-12).
