# Appendix A: the model shift Δ (producer matrix → declared model)

Derivation document in the D-59 format. Notation: spec = the declared model (a `ModelSpec`: local spaces,
operator strings with their couplings as the binary64 numbers given); H_spec = the exact operator of the spec
on the basis `states` (n × n, Hermitian by hypothesis on the spec); H_ref = CERTIFY-ED's floating-point build
of it (`construct.build_bitwise`); H_p = the producer's stored matrix, H_h = (H_p + H_pᴴ)/2 its Hermitian part
(the operator S2–S4 certify); u = 2⁻⁵³, γ_k = ku/(1 − ku). For a matrix M, |M| is the entrywise modulus,
‖M‖_∞ = max_i Σ_j |M_ij|, ‖M‖₁ = ‖Mᴴ‖_∞.

## 1 Statement

**The builder.** Each entry of H_ref is formed as follows (`construct._apply_string`, `build_bitwise`,
`_finalize`): for every coupling (c, sites) of every term with operator string o₁ … o_L, and every choice of
one component per letter (`LocalSpace.shift_components`: S^x, S^y have two components, all other letters one),
a *monomial* maps each basis state s to a state s′ with amplitude t = c · f₁ ⋯ f_L, where f_ℓ is the local
amplitude of letter ℓ at the digit of s (real, or purely imaginary for S^y; exact for spin-½ and fermions,
one rounding — a square root — for S ≥ 1 and bosons), times an exact sign (Jordan–Wigner). The computed
amplitude is t̂ = fl(⋯fl(fl(c·f̂₁)·f̂₂)⋯f̂_L) with f̂_ℓ = fl(f_ℓ). The entry (s′, s) is the floating-point sum of the
monomials that map s to s′ (`coo → csr`, `sum_duplicates`, any order), at most m := n_terms · 2^{L_max} of them,
where L_max is the longest string. A coupling whose monomial leaves `states` contributes to `leak`; the
statements below assume leak = 0 (otherwise S1 is FAILED).

**Lemma A1 (entrywise builder error).** |H_ref − H_spec|_{ij} ≤ γ_{m+2L_max−1} (|H|_abs)_{ij}, where
(|H|_abs)_{ij} := Σ over the monomials mapping j to i of |t| (exact amplitudes). The same bound holds for
the *computed* `SpecOperator.abs_matvec` / `abs_rmatvec` sums up to a factor (1 ± γ_{m+2L_max−1}).

**Lemma A2 (symmetrisation).** For E := H_ref − H_spec and Ĥ := (H_ref + H_refᴴ)/2 (exact), ‖Ĥ − H_spec‖ ≤ ‖E‖ in
every unitarily invariant norm and entrywise |Ĥ − H_spec| ≤ (|E| + |Eᴴ|)/2. With one rounding per entry
(`construct.hermitian_part`), |fl(Ĥ) − H_spec|_{ij} ≤ γ_{m+2L_max} · max((|H|_abs)_{ij}, (|H|_abs)_{ji})
≤ γ_W · max((|H|_abs)_{ij}, (|H|_abs)_{ji}) with W := m + 2L_max + 2.

**Lemma A3 (norm bound).** For any matrix E, ‖E‖₂ ≤ √(‖E‖₁ ‖E‖_∞) ≤ max(‖E‖₁, ‖E‖_∞). Consequently
Δ_builder := γ_W √(r·c) ≥ ‖H_ref − H_spec‖₂ and ≥ ‖fl(Ĥ) − H_spec‖₂, where r := ‖|H|_abs‖_∞ (largest row sum) and
c := ‖|H|_abs‖₁ (largest column sum) — no symmetry of |H|_abs is assumed.

**Theorem A4 (the model shift).** Let S1 have certified H_p against H_ref (`matrix_difference` within
`product_tolerance`) and put D := H_h − H_ref. Then

    ‖H_h − H_spec‖₂ ≤ max(‖D‖₁, ‖D‖_∞) + Δ_builder =: Δ.

When the producer supplies no matrix (H_h := fl(Ĥ) or the matrix-free spec operator), Δ = Δ_builder; for a
sector claim Δ_builder of the audit space is used.

**Theorem A5 (Weyl transfer).** H_h and H_spec are Hermitian with ‖H_h − H_spec‖₂ ≤ Δ, so |λ_j(H_spec) − λ_j(H_h)| ≤ Δ
for every j. Hence: (S2) an enclosure "some eigenvalue of H_h within r of μ" becomes "some eigenvalue of H_spec
within r + Δ of μ", and a matching of claimed values to distinct eigenvalues of H_h within r becomes one to
distinct eigenvalues of H_spec within r + Δ (compose with the sorted bijection j ↦ j); (S3) a count
#{λ(H_h) < σ − β} ≤ ν ≤ #{λ(H_h) ≤ σ + β} becomes #{λ(H_spec) < σ − β − Δ} ≤ ν ≤ #{λ(H_spec) ≤ σ + β + Δ}, i.e.
β → β + Δ; (S4) separations shrink by 2Δ. (The full S3 pairing is `docs/proofs/cor5_pairing.md`, Step 4.)

## 2 Proof

**Lemma A1.** Fix a monomial with exact amplitude t = c f₁ ⋯ f_L (an exact sign is absorbed in one f). Each
f̂_ℓ = f_ℓ(1 + δ_ℓ), |δ_ℓ| ≤ u (one rounding when f_ℓ is irrational, δ_ℓ = 0 otherwise). Each multiplication
is complex × real or complex × purely-imaginary (the only amplitude types produced by `shift_components`): for
z = a + ib and real f, fl(z f) = (fl(af), fl(bf)) = (af(1 + δ′), bf(1 + δ″)), and |fl(zf) − zf|² = (δ′af)² + (δ″bf)²
≤ u²|zf|², so the product carries a relative error ≤ u in modulus; the same holds for a purely imaginary
factor (the components are swapped and one is negated, exactly). After L multiplications
t̂ = t Π_{ℓ}(1 + δ_ℓ)(1 + δ′_ℓ) with 2L factors, so |t̂ − t| ≤ γ_{2L}|t| (Higham 2002, Lemma 3.1: a product of k factors
(1 + δ_i), |δ_i| ≤ u, equals 1 + θ_k with |θ_k| ≤ γ_k). A floating-point sum of p complex terms in any order
satisfies fl(Σ t̂_k) = Σ t̂_k(1 + θ_k^{(p−1)}) with |θ_k^{(p−1)}| ≤ γ_{p−1} (Higham 2002, §4.2, eq. (4.4), applied
componentwise: complex addition rounds each component with relative error ≤ u, which is a relative error ≤ u
in modulus). With p ≤ m, the computed entry is Σ_k t_k(1 + θ_k^{(2L)})(1 + θ_k^{(p−1)}) = Σ_k t_k(1 + θ_k) with
|θ_k| ≤ γ_{2L+m−1} (Lemma 3.3 of Higham: (1 + θ_a)(1 + θ_b) = 1 + θ_{a+b}). The exact entry is Σ_k t_k, so the
error is ≤ γ_{m+2L−1} Σ_k |t_k| = γ_{m+2L−1}(|H|_abs)_{ij}. For L ≤ L_max, γ_{m+2L−1} ≤ γ_{m+2L_max−1}. The
`_finalize` steps (`sum_duplicates`, `eliminate_zeros`, taking the real part of a real spec whose imaginary
parts are exactly zero, sorting indices) add no rounding. The computed |H|_abs sums (`abs_matvec(ones)`,
`abs_rmatvec(ones)`) are sums of the |t̂_k| (|t̂_k| ≤ |t_k|(1 + γ_{2L}) and ≥ |t_k|(1 − γ_{2L})) with p − 1 roundings,
hence within a factor (1 ± γ_{m+2L_max−1}) of the exact sums (all terms non-negative, no cancellation). ∎

**Lemma A2.** H_spec is Hermitian, so Ĥ − H_spec = (E + Eᴴ)/2 and ‖(E + Eᴴ)/2‖ ≤ (‖E‖ + ‖Eᴴ‖)/2 = ‖E‖ for a
unitarily invariant norm; entrywise |(E + Eᴴ)/2|_{ij} ≤ (|E_{ij}| + |E_{ji}|)/2. The rounding of
fl((H_ref,ij + conj(H_ref,ji))/2) is one operation (the division by 2 is exact): relative error ≤ u on an entry of
modulus ≤ ((|H|_abs)_{ij} + (|H|_abs)_{ji})/2 · (1 + γ_{m+2L_max−1}). Altogether
|fl(Ĥ) − H_spec|_{ij} ≤ (γ_{m+2L_max−1} + u(1 + γ_{m+2L_max−1})) · max((|H|_abs)_{ij}, (|H|_abs)_{ji})
≤ γ_{m+2L_max} · max(…) ≤ γ_W · max(…), using γ_a + u(1 + γ_a) ≤ γ_{a+1} (Higham Lemma 3.3) and W ≥ m + 2L_max. ∎

**Lemma A3.** ‖E‖₂² = λ_max(EᴴE) ≤ ‖EᴴE‖_∞ ≤ ‖Eᴴ‖_∞ ‖E‖_∞ = ‖E‖₁ ‖E‖_∞ (the spectral radius of a matrix is at
most any induced norm; ‖·‖_∞ is submultiplicative; ‖Eᴴ‖_∞ = ‖E‖₁); √(ab) ≤ max(a, b). With |E| ≤ γ_W |H|_abs
entrywise (Lemma A1), ‖E‖_∞ ≤ γ_W r and ‖E‖₁ ≤ γ_W c, so ‖E‖₂ ≤ γ_W √(rc). For the symmetrised
matrix: ‖fl(Ĥ) − H_spec‖₂ ≤ ‖Ĥ − H_spec‖₂ + ‖fl(Ĥ) − Ĥ‖₂ ≤ ‖E‖₂ + u(1 + γ)‖(|H|_abs + |H|_absᵀ)/2‖₂
≤ γ_{m+2L_max−1}√(rc) + u(1 + γ_{m+2L_max−1})√(rc) ≤ γ_W √(rc), since ‖(|H|_abs + |H|_absᵀ)/2‖₂ ≤ (‖|H|_abs‖₂ +
‖|H|_absᵀ‖₂)/2 = ‖|H|_abs‖₂ ≤ √(rc). ∎

**Theorem A4.** ‖H_h − H_spec‖₂ ≤ ‖H_h − H_ref‖₂ + ‖H_ref − H_spec‖₂ ≤ √(‖D‖₁‖D‖_∞) + Δ_builder ≤ max(‖D‖₁, ‖D‖_∞) +
Δ_builder by Lemma A3 twice. Without a producer matrix, H_h is fl(Ĥ) (D is not formed; Lemma A3's
symmetrised case) or the exact spec operator applied matrix-free (then the "operator" of S2 *is* H_spec up to
the rounding of its application, which is part of the Theorem 1 accounting, and Δ_builder is an upper
bound on nothing that is needed — it is still added, harmlessly). ∎

**Theorem A5.** Weyl's inequality (Horn–Johnson, *Matrix Analysis*, 2nd ed., Cor. 4.3.15). For S2: if
|λ_i(H_h) − μ| ≤ r then |λ_i(H_spec) − μ| ≤ r + Δ for the same index i, so an injection into distinct indices of
H_h is an injection into distinct indices of H_spec. For S3: λ_j(H_spec) < σ − β − Δ ⇒ λ_j(H_h) < σ − β, so the
count below σ − β − Δ for H_spec is ≤ the count below σ − β for H_h ≤ ν; and λ_j(H_h) ≤ σ + β ⇒
λ_j(H_spec) ≤ σ + β + Δ, so ν ≤ #{λ(H_spec) ≤ σ + β + Δ}. ∎

## 3 Where the code implements it

`certify_ed/certify.py` (branch `track/backlog-proofs`).

| quantity | code | direction |
|---|---|---|
| m, L_max, W | `_builder_shift`: `max_len = max(len(t.op))`, `width = spec.n_terms * 2 ** max_len + 2 * max_len + 2` | W = m + 2L_max + 2 ≥ m + 2L_max (Lemma A2) with two spare roundings for `gamma()`'s own |
| r, c | `row = max(op.abs_matvec(ones))`, `col = max(op.abs_rmatvec(ones))` (`construct.SpecOperator`) | computed sums ≥ exact (1 − γ_{m+2L_max−1}) (Lemma A1); the division by `(1 - gamma(width))` restores an upper bound |
| Δ_builder | `up(g * sqrt(row * col) / (1 - gamma(width)) * (1 + 4 U))`, `g = gamma(width)` | six roundings covered by (1 + 4u)(1 + 8u); `gamma(width)` ≥ γ_W(1 − u)², absorbed by the two spare units of W (before this branch the factor was `(1 + gamma(width))`, which is ≥ 1/(1 − γ_{W−3}) only for Wu ≲ 10⁻⁸) |
| D, ‖D‖_∞, ‖D‖₁ | `Dm = (Hh - Href).tocsr()`; `inf_norm_upper(Dm)`, `inf_norm_upper(Dm.conj().T)` | the entrywise subtraction rounds once (relative u); `inf_norm_upper` = max row sum × (1 + γ_{width+1})(1 + 8u) covers the sum's γ_{width−1} and that u |
| Δ | `ev["model_shift_bound"] = up(max(inf_norm_upper(Dm), inf_norm_upper(Dm.conj().T.tocsr())) + ev["model_shift_bound"])` (line 469) | one addition inside `up` |
| H_h | `certify()`: `op = (opm + opm.conj().T) / 2` when `opm` is not exactly Hermitian; `OperatorAdapter(op)` | S2–S4 concern H_h |
| no producer matrix | `op = Href` (then symmetrised by the same lines) or `SpecOperator`; `model_shift_bound = _builder_shift(spec, states)` | Lemma A2/A3 symmetrised case |
| sector claim | `H_full_ref = hermitian_part(H_full_ref)`; `_builder_shift(spec, st_u1 or None)` | Theorem 8 / Theorem 2 need a Hermitian full operator |
| S2 transfer | `eps_shift = model_shift`; `enc.radius.max() + eps_shift` (accuracy presumption), `add_up(enc.radius, eps_s3)` in the exported radii (INT-03) and in `gaps.py` | Theorem A5 (S2) |
| S3 transfer | `_certify_lowest(..., eps_shift)`: `rad = add_up(radius, eps_shift)`, `beta = add_up(r.beta, eps_shift)`; `_certify_window`, `_certify_window_vectorless` likewise | Theorem A5 (S3), outward rounding |
| compose | `m_build = max(_builder_shift(spec, st_a))` over audit spaces; `eps = up(rho + m_build)` | Theorem 8 Step 4 |

Rounding direction: Δ is rounded towards +∞ relative to ‖H_h − H_spec‖₂ at every step.

**What is rounded in the builder (`construct.py`, `localops.py`).** `shift_components`: `np.sqrt(S(S+1) − m(m±1))`
for spins (the argument is an exact dyadic number; one rounding of the root; exact for S = ½ where the roots
are 0 and 1), `np.sqrt(k + 1)`, `np.sqrt(k)` for bosons, exact entries for fermions, exact scalings by 2, ½, ∓i/2.
`_apply_string`: `amp = amp * factor`, one complex × (real or imaginary) multiplication per letter; the sign
`np.where(parity, −1, 1)` is exact. `build_bitwise`: `coo_matrix(...).tocsr()` sums duplicate (row, col) pairs
in floating point. `hermitian_part`: `(H + H.conj().T) / 2`, one rounding per entry. Nothing else rounds.

## 4 Computational verification

`docs/proofs/verify_appA_model-shift.py`:

- N1 (exact rationals): random real matrices with rational entries: ‖E‖₂² (mpmath, 60 digits, from the exact
  rationals) ≤ ‖E‖₁ ‖E‖_∞ (exact Fractions) and `inf_norm_upper` ≥ the exact ∞-norm; complex matrices in mp;
- B1 (Lemma A1/A3, planted specs): spin-1 and boson chains with random non-dyadic couplings and strings of
  length 1–4 (including S^x S^y products, so that both two-component letters and imaginary factors occur):
  H_spec from `checker.exact_matrix` (the rational matrix K = D⁻¹HD, exact), H_ref from `build_bitwise`;
  entrywise |H_ref − H_spec| ≤ γ_{m+2L_max−1}|H|_abs (Fractions for the entries, |H|_abs from the exact monomials
  via mp), ‖H_ref − H_spec‖₂ (mp) ≤ `_builder_shift(spec)`; the same for `hermitian_part(H_ref)` (Lemma A2);
- W1 (Theorem A5): sorted eigenvalues of H_ref and of H_spec differ by at most `_builder_shift` (mp);
- D1 (Theorem A4): a planted producer matrix H_p = H_ref + P with P within `product_tolerance` (so that S1 is
  CERTIFIED), through `check_model_fidelity`: the reported `model_shift_bound` ≥ ‖H_h − H_spec‖₂ (mp); negative
  control: P beyond the tolerance → S1 FAILED (no shift is reported as a bound);
- adversarial: couplings 10^{±100} (scale), a spec with a single term, L_max = 4 strings, a boson chain with
  n_max = 4 (√k roundings), a coupling that is a huge integer (exact) next to one that is not.

Counts: RUN PENDING (to be filled from the script's output).

## 5 Analytic cross-check

(i) *Spin-½ Heisenberg with dyadic couplings.* All amplitudes are exact (0, 1, ½) and all monomial products
and sums are dyadic numbers of bounded size, so H_ref = H_spec exactly; Lemma A1's bound is then slack by its
whole value, and Δ_builder is a pure guard (it is still > 0, so every exported radius moves by one ulp — INT-03).

(ii) *A single monomial per entry (m = 1, L = 1, one irrational amplitude).* The entry is fl(c · fl(√a)) with two
roundings: exact error ≤ (2u + u²)|c√a| ≤ γ₂|c√a|; Lemma A1 gives γ_{m+2L−1} = γ₂ — tight.

(iii) *Diagonal E.* For a diagonal error matrix ‖E‖₂ = ‖E‖₁ = ‖E‖_∞ = max|E_ii|: Lemma A3 is an equality.

(iv) *Rank-one check of Lemma A3.* E = xyᴴ: ‖E‖₂ = ‖x‖₂‖y‖₂, ‖E‖_∞ = ‖x‖_∞‖y‖₁, ‖E‖₁ = ‖x‖₁‖y‖_∞, and
‖x‖₂² ≤ ‖x‖_∞‖x‖₁ gives ‖E‖₂² ≤ ‖E‖₁‖E‖_∞.

(v) *Weyl transfer, k = 1.* A single enclosure |λ − μ| ≤ r for H_h and ‖H_h − H_spec‖ ≤ Δ give, for the same
sorted index, |λ′ − μ| ≤ r + Δ: the statement of Theorem 1 for the declared model with the radius `add_up(r, Δ)`
reported since INT-03.

## 6 Status

STATUS LINE PENDING: to be written from the completed verification run (format: "Derived by Claude on <date>;
machine-checked (<N> random + <M> adversarial cases, 0 violations); independent re-derivation: pending
(Phase 2); Sarang's check (D-12): pending.").
