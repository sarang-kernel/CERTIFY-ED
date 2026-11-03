# D-02: matrix-free residual bounds from the spec (S2 without a stored matrix)

Derivation document in the D-59 format. A spec operator H is applied term by term from its local terms
(`construct.SpecOperator._apply`: for every coupling c_t and every monomial of its string, one vectorised pass
over the basis states) and never stored. The Theorem 1 radius (`docs/proofs/thm1_fp-accounting.md`) needs, for
such an operator, (i) a count W of the roundings that one output entry of fl(H x) can accumulate relative to the
*exact* spec operator, (ii) the vector |H|_abs |x| that scales them, and (iii) a bound on ‖H‖₂ for the cluster
bounds. This document derives (i) and (iii) from the spec alone; (ii) is the second pass the code already makes
(`abs_matvec`, Appendix A). The result makes S2 a statement about the exact spec operator, with the roundings
of the local amplitudes (square roots for spin ≥ 1 and bosons) counted inside W, so that no separate model shift
is needed for a matrix-free claim.

Notation as in `docs/theory.md` §1 and `thm1_fp-accounting.md`. Spec: N sites, local space of dimension q,
couplings t = 1 … T (T = `spec.n_terms`), each a complex or real coefficient c_t (stored binary64, taken exactly)
and a string o_{t,1} ⋯ o_{t,L_t} of letters on sites (possibly repeated); L := max_t L_t. For a letter o the code
uses its *shift components* (δ, a) with a[k] = ⟨k+δ|o|k⟩ (`localops.shift_components`); the exact components are
the same with exact amplitudes â[k]. A *monomial* of coupling t is one choice of component per letter; it maps a
basis state to at most one basis state (each digit is shifted by a fixed amount), with amplitude
c_t · Π_l a_l[k_l] · (JW sign). H_spec denotes the exact operator (exact c_t, exact â); |H_spec|_abs the entrywise
sum of the moduli of the exact monomial amplitudes (an entrywise upper bound on |H_spec|; Appendix A).
u = 2⁻⁵³, γ_k = ku/(1 − ku); fl(·) a binary64 round-to-nearest operation.

## 1 Statement

**Error model.** Standard model (Higham 2002, eq. 2.4) with the INT-05 additive terms entering only through the
existing helpers of `enclosure.residuals` (`residual_additive_term`, `norm2_upper`), which are not changed here.
NumPy evaluates `amp = amp * factor` and `a * x[src]` one rounding per real operation (complex products as four
real products and two additions), and `y[rr] += contrib` as one rounded addition per target entry.

**Definitions (exact combinatorics on the spec, `matrixfree.py`).**

- c(o) := max over target digits k of #{components (δ, a) of o : 0 ≤ k − δ < q and a[k − δ] ≠ 0}
  (`letter_counts`). For q = 2 every letter has c(o) = 1; for spin 1 (q = 3) c(x) = c(y) = 2, c(z) = c(±) = 1.
- w_row := Σ_t Π_{l ≤ L_t} c(o_{t,l})  (`terms_per_row`): an upper bound on the number of monomial contributions
  that can land in one output entry (Lemma M1).
- ρ := max_t #{l : the stored amplitudes of o_{t,l} are not all exact} (`rho`; a letter is *exact* when every stored
  nonzero amplitude equals its exact value — decided in rational arithmetic by `_exact_components`: spin-1/2 and
  fermion letters, spin-S S^z, boson n are exact; spin ≥ 1 ladder/x/y letters and boson ladders are rounded once
  per amplitude).
- W := w_row + L + ρ + 1 (real data), W := w_row + 3L + ρ + 3 (complex data)  (`W`); `rounding_width` := W − 2.

**Theorem D-02 (entrywise rounding of the matrix-free product).** For every stored x and every output entry i,

    (a)  |fl(H x)_i − (H_spec x)_i| ≤ γ_W (|H_spec|_abs |x|)_i,

where fl(H x) is what `MatrixFreeOperator.matvec` (= `SpecOperator.matvec`, bit for bit) returns. Consequently
the Theorem 1-fl radius of `enclosure.residuals`, evaluated with `adapter.width = W − 2` (so that
`gamma_dot(width + 2)` = γ_W for real data and 2γ_{W+4} ≥ γ_W for complex data, and `gamma(width + 1)` = γ_{W−1}
covers the roundings of the second pass |H|_abs|x|, Lemma M3), satisfies

    (b)  rel ≥ ‖H_spec x − μx‖ / ‖x‖      for the exact spec operator (no model shift).

**Lemma N (spec norm bound).** ‖H_spec‖₂ ≤ Σ_t |c_t| Π_l ‖o_{t,l}‖₂ ≤ Σ_t |c_t| Π_l √(‖o_{t,l}‖₁ ‖o_{t,l}‖_∞) =: N_spec,
and `norm_bound()` ≥ N_spec. `OperatorAdapter` keeps min(`norm_bound()`, row-sum bound of Appendix A): both are
proven upper bounds.

**Lemma M2 (w_row sharpness for q = 2).** For spin-1/2 and fermion specs w_row = T exactly: every monomial of a
string on q = 2 is a bijection of the digit patterns it acts on, and exactly one monomial of each coupling reaches a
given target state. The 3.1 width T·2^L + 2L + 2 (`OperatorAdapter`, `certify._builder_shift`) exceeds W by about
a factor 2^L for such specs (Heisenberg chain: 4T + 6 against T + 3).

**Hypotheses recorded in the certificate** (`amplitude_hypothesis`): the list of exact letters, ρ, w_row, L, W.
The claim's S1 statement names them ("every local amplitude of the letters … is exactly representable in binary64;
the letters … have amplitudes rounded once each, counted in W").

## 2 Proof

**Lemma M1 (contributions per output entry).** Fix an output basis state i (digits k_1 … k_N). A monomial of
coupling t contributes to y_i from the source state j whose digits are k_s − δ for each site s touched by the
monomial (k_s − Σδ when a site is repeated), provided every source digit is in range and every amplitude is
nonzero. Hence, for a string on distinct sites, the monomials of t that contribute to y_i are the choices of one
component per letter with a valid source digit at that letter's site — at most Π_l c(o_{t,l}) of them. For repeated
sites the per-letter count c(o) is still an upper bound on the valid choices of that letter (its own component must
have a valid source digit from the digit the previous letters produced). Summing over couplings gives ≤ w_row
contributions to y_i. (The code's passes: `_apply_string` yields one (s_fin, amp, ok) array per monomial; a
target index appears in at most one position of each monomial's `s_fin[m]`, since the monomial is injective on
states — this is also why `y[rr] += contrib` and `np.add.at` give the same numbers.)

**Lemma M2** is the case c(o) = 1 for every letter on q = 2: the two components of x and y have δ = ±1, and a
target digit k ∈ {0, 1} admits exactly one valid source digit k − δ ∈ {0, 1}; the single-component letters have
c = 1 by definition when their amplitude at the valid source is nonzero (z: a[k] = ±1/2 ≠ 0; ±, fermion ±, n: one
component with one valid source). So each coupling reaches i by exactly one monomial, or by none when an
amplitude vanishes (fermion n on an empty site), and w_row = T is attained by a state every coupling reaches.

**Step 1: one contribution.** The computed amplitude of a monomial at a source j is formed by `amp = c_t` (exact),
then for each of the L_t letters `amp = amp * factor` with `factor` = a_l[k_l] (times an exact ±1 Jordan–Wigner
sign, `np.where(..., -1.0, 1.0)`, exact): L_t roundings, each relative u (real) — for complex data a complex
product has |fl(zw) − zw| ≤ γ₃|z||w| (Higham Lemma 3.5), i.e. (1 + θ₃) in modulus. The stored a_l[k_l] equals the
exact â_l[k_l] when the letter is exact, and â_l[k_l](1 + δ′), |δ′| ≤ u, otherwise (`shift_components`: one `np.sqrt`
of an exactly representable argument S(S+1) − m(m±1) for spin — a half-integer times a half-integer, exact in
binary64 for S ≤ 2⁵⁰ — then a product with the exact scale 1/2, 2 or 1: the scale is a power of two, exact; the
`(-0.5j)` factor of y is exact; bosons: `np.sqrt(k)` one rounding). So the computed amplitude is
c_t Π_l â_l (1 + θ), with |θ| ≤ γ_{L_t + ρ} (real; Λ1 of `thm1_fp-accounting.md`) or ≤ γ_{3L_t + ρ} (complex).
The product with x_j (`a * x[src]`) adds one rounding (real) or (1 + θ₃) (complex); `.real` for a real output is
exact.

**Step 2: the sum.** y_i starts at 0 and receives at most w_row contributions (Lemma M1), each by one rounded
addition (`y[rr] += contrib`, distinct targets within a pass): the k-th addend carries at most w_row − k + 1 further
roundings (Higham eq. (3.4) for a recursive sum, any order). Each exact contribution is therefore multiplied by
(1 + θ) with |θ| ≤ γ_{L_t + ρ + 1 + w_row − 1} ≤ γ_{w_row + L + ρ} ≤ γ_{W − 1} ≤ γ_W (real), and with
|θ| ≤ γ_{3L_t + ρ + 3 + w_row − 1} ≤ γ_{w_row + 3L + ρ + 2} ≤ γ_{W − 1} (complex; a complex addition rounds both
components, relative u in modulus). Summing the moduli,

    |fl(H x)_i − (H_spec x)_i| ≤ γ_W Σ_{contributions} |c_t Π_l â_l| |x_j| = γ_W (|H_spec|_abs |x|)_i,

which is (a). (The slack of one index, γ_{W−1} ≤ γ_W, is kept for the real case; the complex case uses
`gamma_dot(W − 2 + 2)` = 2γ_{W+4} ≥ γ_W in `_rounding_bound`.)

**Lemma M3 (the second pass).** `abs_matvec(|x|)` computes Σ |a_l …| |x_j| by the same passes with `np.abs`:
the moduli of real amplitudes are exact; of complex ones one rounding each (hypot) on an amplitude that already
carries (1 + θ_{3L_t+ρ}); then the product with |x_j| and the ≤ w_row − 1 additions of non-negative terms. The
computed entry is therefore ≥ (|H_spec|_abs|x|)_i (1 − γ_{w_row + L + ρ}) (real) and ≥ … (1 − γ_{w_row + 3L + ρ + 2})
(complex), both ≥ (1 − γ_{W−1}); `_rounding_bound` divides by (1 − `gamma(width + 1)`) = (1 − γ_{W−1}(1 + θ₂)) and
multiplies by (1 + 4u), exactly the step (Λ5) of `thm1_fp-accounting.md` with w + 1 replaced by W − 1, so that
`absHX_up_i` ≥ (|H_spec|_abs|x|)_i. The rest of Theorem 1-fl (Steps 3–5 of that document) is unchanged, which
gives (b): the radius bounds ‖H_spec x − μx‖/‖x‖ for the exact operator, so the certified statement concerns the
declared model directly.

**Lemma N.** A coupling's operator is c_t times a product of local operators, each acting on one site (tensor
factor) possibly multiplied by the Jordan–Wigner parity string, which is a product of unitary diagonal factors
(±1) on other sites. ‖A ⊗ B‖₂ = ‖A‖₂‖B‖₂ and ‖UA‖₂ = ‖A‖₂ for unitary U, so ‖c_t o_{t,1} ⋯ o_{t,L_t}‖₂ ≤ |c_t| Π_l ‖o_{t,l}‖₂
(sub-multiplicativity for repeated sites). The triangle inequality over couplings gives the first inequality;
‖o‖₂ ≤ √(‖o‖₁‖o‖_∞) (Horn–Johnson Cor. 5.6.10) the second, and `local_norm_bound` evaluates the latter on the
moduli of the exact local matrix with the guard (1 + 8u)²; `norm_bound()` sums T non-negative products with the
factor (1 + γ_{T+L+2})(1 + 8u) covering the ≤ T + L + 1 roundings of the sum and products.

**Why no model shift.** `certify._builder_shift` bounds ‖H_ref − H_spec‖ for a *stored* reference matrix built
once (its entries carry the amplitude roundings and the symmetrisation). For the matrix-free operator no matrix
exists: the amplitude roundings are re-incurred in every product and are counted in W by Step 1 (the ρ term). The
S1 statement for a matrix-free claim therefore records `model_shift_bound = 0` with the amplitude hypothesis
named, and S2 adds nothing. S3 (inertia) needs a factorization and stays NOT_CHECKED (`NEEDS_MATRIX`), as before.

## 3 Where the code implements it

| proof object | code | rounding direction |
|---|---|---|
| c(o), w_row (Lemma M1/M2) | `certify_ed/matrixfree.py:letter_counts`, `MatrixFreeOperator.__init__` (`terms_per_row`) | exact integers |
| exact letters, ρ | `_exact_components`, `rho` | rational comparison, no rounding |
| W, `rounding_width` = W − 2 | `MatrixFreeOperator.__init__` | integers |
| the passes (Step 1–2) | `MatrixFreeOperator._apply` (same arithmetic as `construct.SpecOperator._apply`; `y[rr] += contrib`, full-space index map) | what (a) bounds |
| γ_W in the radius (b) | `enclosure.OperatorAdapter.__init__` (`width = min(3.1 count, rounding_width)`), `_rounding_bound` (`gamma_dot(w + 2)`, `gamma(w + 1)`) | upward, as in `thm1_fp-accounting.md` |
| Lemma N | `local_norm_bound`, `norm_bound()`; `OperatorAdapter` takes min with the Appendix-A row-sum bound (its γ index raised to `width + 2` ≥ W − 1 for this operator) | upward (`up`, (1 + γ)) |
| amplitude hypothesis in the record | `amplitude_hypothesis`; `certify.check_model_fidelity` (matrix-free branch, `Options(matrix_free=True)`) | — |

## 4 Computational verification

`docs/proofs/verify_D-02.py` (run from the repository root; 150 random specs by default):

- **Random cases:** specs drawn from six families (spin-1/2 spin and Pauli conventions, spin-1, fermions with
  both Jordan–Wigner conventions, bosons with n_max = 2, complex couplings), N = 2–5 sites (2–3 for q = 3),
  1–6 couplings, strings of length 1–3 with repeated sites allowed, both state orders. For each: (V1) the entrywise
  inequality (a) against H_spec x evaluated at 60 digits from the exact local matrices (`exact_H`: Kronecker
  products with the Jordan–Wigner parity factors; square roots exact to 60 digits, with 2⁻¹⁰⁶⁰·W slack per entry
  for the mp evaluation), with |H_spec|_abs from the products of the local moduli; (V2) the instrumented count of
  contributions per output entry ≤ `terms_per_row`; (V3) the 60-digit spectral norm ≤ `norm_bound()` and ≤
  `OperatorAdapter(op).norm_upper()`; (V5) bit identity with `SpecOperator.matvec`. On the Hermitian library
  specs (Heisenberg, TFIM, J1–J2, spin-1 with D, spinless fermions, Bose–Hubbard): (V4) the implemented radius
  `residuals(...).rel_bound` ≥ ‖H_spec x − μx‖/‖x‖ for random x with the Rayleigh quotient and for exact
  eigenvectors (residual at rounding level).
- **Adversarial:** exactly cancelling couplings (+J and −J on one bond); 24 three-letter strings on four sites
  (many terms per site, w_row = 24); couplings 10¹⁵⁰ and 10⁻¹⁵⁰; vector entries 10⁻¹⁶⁰ (the INT-05 additive terms
  of `residuals`); spin-1 x/y (ρ = 2); a zero coupling; repeated sites in one string; n = 2; periodic fermions
  (Jordan–Wigner string across the boundary).
- **Negative controls:** the exact ground state of the 6-site chain under the correct spec has rel ≈ 10⁻¹⁵; the
  same vector under a spec with one coefficient changed by 0.1 % has rel > 10³ times larger (the bound stays
  valid for the wrong operator — it bounds the residual of the vector *under that spec*, which is what makes S2
  fail); a planted vector V₀ + 10⁻³V₁ has rel ≈ 7·10⁻⁴ > 10⁻⁴.
- **Result:** 182 random + 18 adversarial cases, 0 skipped, **0 violations** (435 s; `--cases 150`, the default).
- **Scale (timing, not a correctness check; `experiments/d02_matrixfree_scale.py`):** S2 CERTIFIED at every point with the radius at rounding level and the memory under 3.2 GB: d = 2^16: S2 8 s (one matvec 3 s), radius 3.96e-14, peak 67 MB, W = 51; d = 2^20: S2 169 s (one matvec 40 s), radius 6.34e-14, peak 257 MB, W = 63; d = 2^22: S2 653 s (one matvec 210 s), radius 7.71e-14, peak 869 MB, W = 69; d = 2^24: S2 1896 s (one matvec 734 s), radius 9.21e-14, peak 3160 MB, W = 75. The time is the Python-level term-by-term application (about 60 vectorised passes per product at N = 20–24); the bound itself costs one extra pass (|H|_abs|x|) and the norm pass. Records: `results/d02_matrixfree_scale.json`.

## 5 Analytic cross-check

(i) *Diagonal spec.* For a spec of z-letters only (an Ising model), every monomial is diagonal, fl(H x)_i is a
sum of T products c_t Π m_l x_i with exact amplitudes (ρ = 0): W = T + L + 1, and the classical bound for a
T-term inner product with L + 1 products per term is γ_{T+L} (Higham eq. 3.4 with the extra products) ≤ γ_W.
(ii) *Reduction to the stored-matrix count.* For q = 2 (Lemma M2) w_row = T is the number of nonzeros a row of
the stored matrix would have at most (one per coupling, before merging equal targets), and W − (L + ρ + 1) = T
is the `adapter.width` the stored-matrix path uses for a row with T nonzeros: the matrix-free count equals the
stored-matrix count plus the L + ρ + 1 roundings that the stored path pays once, at build time, through the
model shift. (iii) *Lemma N for a single Pauli string* reduces to ‖c σ^{a}⊗⋯‖ = |c|, exact (‖σ‖₁ = ‖σ‖_∞ = 1).

## 6 Status

Derived by Claude on 2026-10-31; machine-checked (182 random + 18 adversarial cases, 0 violations); independent re-derivation: done 2026-10-30 (fresh-context session, verdict GAP, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
