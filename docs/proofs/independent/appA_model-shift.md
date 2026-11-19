# Independent re-derivation: Appendix A (the model shift Δ)

Written from the statement alone on 2026-10-30 (not a human check). Sections 1 and 3 of `../appA_model-shift.md` read first; proof written before
section 2 was opened. `verify_appA_model-shift.py` was not run for this re-derivation (its run of 2026-11-03
is `docs/proofs/runs/appA_model-shift.txt`). Notation as in the original's preamble; a := |H|_abs (entrywise, exact amplitudes),
γ_k = ku/(1 − ku), Higham, *Accuracy and Stability of Numerical Algorithms* (2002): Lemma 3.1 (products of
1 + δ_i), Lemma 3.3 (γ_j + γ_k + γ_jγ_k ≤ γ_{j+k}), §4.2 eq. (4.4) (summation of m terms in any order:
|fl(Σx) − Σx| ≤ γ_{m−1}Σ|x|).

## Statements checked

Lemma A1 (entrywise builder error), Lemma A2 (symmetrisation, entrywise), Lemma A3 (norm bound; Δ_builder =
γ_W√(rc) bounds both ‖H_ref − H_spec‖₂ and ‖fl(Ĥ) − H_spec‖₂), Theorem A4 (Δ = max(‖D‖₁, ‖D‖_∞) + Δ_builder),
Theorem A5 (Weyl transfer to S2, S3, S4).

## My proof

### Lemma A1

Fix an entry (i, j) and let t₁, …, t_p (p ≤ m) be the exact amplitudes of the monomials mapping j to i, so
H_spec,ij = Σ_q t_q and a_ij = Σ_q |t_q|. The computed amplitude of monomial q: each local factor f̂_ℓ = f_ℓ(1 + θ_ℓ)
with |θ_ℓ| ≤ u (one rounding of a square root, or exact), and each of the L ≤ L_max multiplications of a complex
number by a real or purely imaginary factor rounds each component once — for (a + bi)·r the components ar, br
each carry one relative error ≤ u, and |(arδ₁, brδ₂)| ≤ u|(ar, br)|, so the complex product has relative error ≤ u
in modulus; the same for (a + bi)·(ir) = −br + ari; the multiplication by the exact sign is exact. Hence
t̂_q = t_q Π(1 + θ)(1 + δ) with 2L factors, and by Higham Lemma 3.1, |t̂_q − t_q| ≤ γ_{2L}|t_q|. The entry is
fl(Σ_q t̂_q) in some order; componentwise summation gives for the real and imaginary parts
|fl(Σ) − Σ|_{re} ≤ γ_{p−1}Σ|Re t̂_q| and likewise for Im, and the modulus of the error is ≤ γ_{p−1}Σ_q|t̂_q| ≤
γ_{m−1}(1 + γ_{2L})a_ij. Therefore
|H_ref,ij − H_spec,ij| ≤ γ_{m−1}(1 + γ_{2L})a_ij + γ_{2L}a_ij = (γ_{m−1} + γ_{2L} + γ_{m−1}γ_{2L})a_ij ≤ γ_{m+2L−1}a_ij
(Higham Lemma 3.3), and γ_{m+2L−1} ≤ γ_{m+2L_max−1}. For the computed sums of |t̂_q| (`abs_matvec(ones)` at
row i: all monomials landing on i; each monomial type is an injective map of states, so at most m of them land on
one state): |t̂_q| is exact when t̂_q is real or purely imaginary (one nonzero component), and the sum of p
non-negative terms has relative error ≤ γ_{p−1}; so the computed row sum lies in (row sum of a) ×
[(1 − γ_{m−1})(1 − γ_{2L}), (1 + γ_{m−1})(1 + γ_{2L})] ⊂ [1 − γ_{m+2L−1}, 1 + γ_{m+2L−1}]. (If a coupling is
genuinely complex, |t̂_q| = hypot(·, ·) adds one rounding; the budget below absorbs it.) ∎

### Lemma A2

H_spec = H_specᴴ, so Ĥ − H_spec = (E + Eᴴ)/2 with E = H_ref − H_spec. For a unitarily invariant norm ‖Eᴴ‖ = ‖E‖
and the triangle inequality gives ‖Ĥ − H_spec‖ ≤ ‖E‖; entrywise |Ĥ − H_spec| ≤ (|E| + |Eᴴ|)/2. With the rounding
of `hermitian_part`: fl(Ĥ)_ij = fl((H_ref,ij + conj(H_ref,ji))/2) = Ĥ_ij(1 + δ) (the halving is exact, the sum rounds
once per component, modulus error ≤ u|Ĥ_ij| as above). Let γ := γ_{m+2L_max−1} and M_ij := max(a_ij, a_ji).
Then |Ĥ − H_spec|_ij ≤ γ(a_ij + a_ji)/2 ≤ γM_ij, |H_spec,ij| = |H_spec,ji| ≤ min(a_ij, a_ji), so |Ĥ_ij| ≤ (1 + γ)M_ij
and |fl(Ĥ) − H_spec|_ij ≤ γM_ij + u(1 + γ)M_ij ≤ (γ + γ₁ + γγ₁)M_ij ≤ γ_{m+2L_max}M_ij ≤ γ_W M_ij (γ₁ = u/(1 − u) ≥ u;
Lemma 3.3). ∎

### Lemma A3

‖E‖₂² = ρ(EᴴE) ≤ ‖EᴴE‖_∞ ≤ ‖Eᴴ‖_∞‖E‖_∞ = ‖E‖₁‖E‖_∞ (the spectral radius is at most any induced norm; ‖·‖_∞ is
submultiplicative), and √(xy) ≤ max(x, y). For E = H_ref − H_spec, Lemma A1 gives |E| ≤ γ_W a entrywise, hence
‖E‖_∞ ≤ γ_W r and ‖E‖₁ ≤ γ_W c, so ‖E‖₂ ≤ γ_W√(rc) = Δ_builder. ✓

*The symmetrised matrix.* Here the entrywise bound of Lemma A2 in the form γ_W·max(a_ij, a_ji) does **not** by
itself imply ‖fl(Ĥ) − H_spec‖₂ ≤ γ_W√(rc) when a is not symmetric: for a = [[1, 1], [0, 0]] (r = 2, c = 1,
√(rc) = √2) the matrix M = max(a, aᵀ) = [[1, 1], [1, 0]] has ‖M‖₂ = (1 + √5)/2 ≈ 1.618 > √2, so a matrix F with
|F| = γ_W M entrywise (e.g. F = γ_W M, Hermitian) satisfies the entrywise conclusion of Lemma A2 and has
‖F‖₂ ≈ 1.618 γ_W > Δ_builder. The *statement* nevertheless holds, by a route that keeps the average instead of
the maximum: from the proof of Lemma A2, |fl(Ĥ) − H_spec|_ij ≤ γ(a_ij + a_ji)/2 + u|Ĥ_ij| and
|Ĥ_ij| ≤ |H_spec,ij| + |Ĥ − H_spec|_ij ≤ (a_ij + a_ji)/2 + γ(a_ij + a_ji)/2, so
|fl(Ĥ) − H_spec| ≤ (γ(1 + u) + u)(a + aᵀ)/2 =: γ′(a + aᵀ)/2 entrywise, with γ′ ≤ γ + γ₁ + γγ₁ ≤ γ_{m+2L_max}
(Lemma 3.3). Since the spectral norm is monotone on entrywise moduli (‖F‖₂ ≤ ‖|F|‖₂, Horn–Johnson, *Matrix
Analysis*, 2nd ed., Thm 8.1.18 / 5.6.P) and ‖(a + aᵀ)/2‖₂ ≤ (‖a‖₂ + ‖aᵀ‖₂)/2 = ‖a‖₂ ≤ √(‖a‖₁‖a‖_∞) = √(rc), we get
‖fl(Ĥ) − H_spec‖₂ ≤ γ_{m+2L_max}√(rc) ≤ γ_W√(rc). ∎ (When a is symmetric — every spec whose monomial set is closed
under the adjoint — M = a and the max-route also works.)

### Theorem A4

‖H_h − H_spec‖₂ ≤ ‖H_h − H_ref‖₂ + ‖H_ref − H_spec‖₂ ≤ √(‖D‖₁‖D‖_∞) + Δ_builder ≤ max(‖D‖₁, ‖D‖_∞) + Δ_builder,
by Lemma A3 for D and Lemma A1 + A3 (unsymmetrised route) for H_ref. When there is no producer matrix,
H_h = fl(Ĥ) and Lemma A3 (symmetrised route) gives Δ = Δ_builder directly; for the matrix-free spec operator the
operator applied is H_spec up to the per-matvec roundings of Theorem 1's accounting, and Δ_builder is added
although not needed. ∎

### Theorem A5

H_h, H_spec Hermitian, ‖H_h − H_spec‖₂ ≤ Δ: Weyl (Horn–Johnson Cor. 4.3.15) gives |λ_j(H_spec) − λ_j(H_h)| ≤ Δ for
all j. (S2) If |λ_i(H_h) − μ| ≤ r then |λ_i(H_spec) − μ| ≤ r + Δ; an injection j ↦ i_j with |λ_{i_j}(H_h) − μ_j| ≤ r
is an injection with |λ_{i_j}(H_spec) − μ_j| ≤ r + Δ. (S3) λ_j(H_spec) < σ − β − Δ implies λ_j(H_h) < σ − β, so
#{λ(H_spec) < σ − β − Δ} ≤ #{λ(H_h) < σ − β} ≤ ν, and ν ≤ #{λ(H_h) ≤ σ + β} ≤ #{λ(H_spec) ≤ σ + β + Δ}.
(S4) If an interval I is at distance ≥ sep from the remaining eigenvalues of H_h, each eigenvalue moves by ≤ Δ,
so the eigenvalues of H_spec inside I + [−Δ, Δ] are the images of those in I and the distance to the rest is ≥
sep − 2Δ. ∎

## Code check (current `dev/3.2`)

`certify._builder_shift` (certify.py 363–390): `width = n_terms·2^max_len + 2·max_len + 2` = W (n_terms counts
couplings, spec.py 136–137, as m requires); `row`, `col` = max of `abs_matvec(ones)`, `abs_rmatvec(ones)`
(construct.py 244–281: `np.abs(a) * np.abs(x[src])`, `np.add.at` accumulation — the sum of the nonnegative |t̂_q| in
some order, as Lemma A1 assumes); return `up(g·sqrt(row·col)/(1 − gamma(width))·(1 + 4u))`, g = `gamma(width)`.
Budget: row ≥ r(1 − γ_{m+2L−1}) (one more unit if |t̂| needs a hypot), so √(row·col) ≥ √(rc)(1 − γ_{m+2L});
`gamma(width)` = γ_W(1 + θ), |θ| ≤ 2u + O(u²) (fl(1 − Wu) and the division; Wu is exact), so (1 − g)⁻¹ ≥
(1 − γ_{m+2L})⁻¹ and g ≥ γ_W(1 − 2u) ≥ γ_{W−1} = γ_{m+2L+1} ≥ γ_{m+2L_max} as my route needs; six roundings
covered by (1 + 4u)(1 + 8u) ≥ (1 − u)⁻⁶. Direction: towards +∞. ✓

`check_model_fidelity` (certify.py 459, 474–478): `Hh = (Hp + Hpᴴ)/2` when not exactly Hermitian (the same
expression `certify()` line 973 forms for the operator S2–S4 use), `Dm = Hh − Href` (one rounding per entry,
relative u), `up(max(inf_norm_upper(Dm), inf_norm_upper(Dmᴴ)) + builder)`; `fp.inf_norm_upper` (fp.py 173–188)
= max row sum × (1 + γ_{width+1})(1 + 8u), which covers the γ_{width−1} of the row sum, the u of the subtraction
and its three roundings. ✓ Theorem A5's uses: `eps_s3 = model_shift` (certify.py 1095), `_certify_lowest`
(533, 598), `compose` `eps = up(rho + m_build)` (compose.py 169). ✓

## Comparison with section 2 of the original (read after the proof above was written)

Lemma A1: same decomposition (2L factors per monomial, γ_{p−1} for the sum, Lemma 3.3), same constant. Lemma A2:
same, the original writes the rounding term as u(1 + γ)·max(a_ij, a_ji) — fine entrywise. Lemma A3: for the
unsymmetrised matrix identical. For the symmetrised matrix the original does **not** go through the max-form of
Lemma A2 (which would be insufficient, see my counterexample a = [[1, 1], [0, 0]]); it bounds ‖fl(Ĥ) − H_spec‖₂ ≤
‖Ĥ − H_spec‖₂ + ‖fl(Ĥ) − Ĥ‖₂ ≤ ‖E‖₂ + u(1 + γ)‖(a + aᵀ)/2‖₂ — the same average-route as mine, with the same
conclusion γ_W√(rc). One imprecision: ‖fl(Ĥ) − Ĥ‖₂ ≤ u‖|Ĥ|‖₂ needs |Ĥ| ≤ (1 + γ)(a + aᵀ)/2 entrywise and the
monotonicity of ‖·‖₂ on nonnegative matrices (Horn–Johnson Thm 8.1.18); the original uses both without naming
them. The statement of Lemma A2 (max-form) is thus correct but is not what Lemma A3 needs; the proof of A3 is
self-contained, so the statement chain is valid. Theorems A4, A5: identical. Same constants throughout.

**Code mapping** (current lines in my code check above; the table of section 3 agrees). Budget remark: the
table says `gamma(width)` ≥ γ_W(1 − u)² "absorbed by the two spare units of W"; by my count g ≥ γ_{W−1} =
γ_{m+2L_max+1} ≥ γ_{m+2L_max}, i.e. one spare unit after the two internal roundings of `gamma()`, and the hypot
rounding of a genuinely complex coupling in `abs_matvec` is covered by the (1 − g)⁻¹ factor's one spare unit.
Sufficient, not two spare.

**Verdict: AGREES-WITH-REMARKS** (statement and proof valid, same constant; Lemma A2's max-form is not the form
Lemma A3 uses — the average form (a + aᵀ)/2 should be stated in A2; two unnamed classical facts in A3).

## Runs

- `verify_appA_model-shift.py`: not run for this re-derivation (run of 2026-11-03: `docs/proofs/runs/appA_model-shift.txt`).
- `adversarial_appA_model-shift.py --cases 16 --seed 17`: N1 4 random + 12 adversarial, W1 4 + 12, D1 4 + 12,
  A5 transfer 4 + 12, negative controls 16; 0 violations (1235 s: the mp eigendecompositions of the 81 × 81
  spin-1 cases dominate; a second batch `--cases 24 --seed 18` is reported in VERDICTS.md if it finished).
