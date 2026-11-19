# Theorem 1: floating-point evaluation of the residual radius (code-facing accounting)

Derivation document in the D-59 format. It turns the code-facing list of `docs/theory.md` §2 (INT-04) into a
proof with an explicit error model. Notation: A the stored Hermitian matrix (or the exact spec operator, see
Remark 3), x the stored claimed vector (n entries), μ the stored claimed value, r := Ax − μx (exact),
a_i := (|A||x|)_i, q_i := |μ x_i|; u = 2⁻⁵³, γ_k := ku/(1 − ku); w := `adapter.width` (an upper bound on the
number of nonzero terms in any row sum of A x; for a spec operator the monomial count, `docs/proofs/appA_model-shift.md`);
g := `gamma_dot(w + 2)` = γ_{w+2} for real data and 2γ_{w+6} for complex data; fl(·) a round-to-nearest binary64
operation; ‖·‖ the Euclidean norm. Rc denotes the computed residual vector and E the computed entrywise bound.

## 1 Statement

**Error model (standard, this document).** fl(z) = z(1 + δ) with |δ| ≤ u for every operation whose exact result
is a normal binary64 number or zero; the additive term for subnormal results is treated in
`docs/proofs/int05_underflow.md`, whose additive quantities (A_{n,w}, T_n, ULP_MIN) enter the lines below only as
non-negative summands and are set to zero in this document's inequalities. BLAS and sparse kernels evaluate
each entry of a product as a sum of the exact products in some order, with or without fused multiply-add
(FMA), each operation rounded once (Higham 2002, §3.1: the bound γ_k for a k-term inner product holds for any
order and with FMA).

**Hypotheses.** A finite, x ≠ 0 stored, μ stored; n ≤ 2²⁶ and w ≤ 2²⁶ (the second-order terms (nu)², (wu)² are
then ≤ u/2; every quantity the code forms is a normal number or zero, or else the INT-05 document applies).

**Theorem 1 (exact).** For Hermitian A, x ≠ 0, real μ: some eigenvalue λ satisfies |λ − μ| ≤ ‖Ax − μx‖/‖x‖
(Parlett 1998, Thm 4.5.1; proof: expand x in an eigenbasis, ‖Ax − μx‖² = Σ|c_k|²(λ_k − μ)² ≥ min_k(λ_k − μ)² ‖x‖²).

**Theorem 1-fl (what `enclosure.residuals` returns).** With `rel` the value of `rel_bound` for the pair (μ, x),

    rel ≥ ‖Ax − μx‖ / ‖x‖      (the exact Theorem 1 radius for the stored numbers; rel = ∞ iff `xl` = 0),

and, stage by stage: (S2) `absHX_up_i` ≥ a_i; (S3) E_i ≥ |Rc_i − r_i|; (S4) `norm2_upper(v)` ≥ ‖v‖(1 + 3u) and
`norm2_lower(v)` ≤ ‖v‖(1 − 3u) for any stored v with 1 ≤ n ≤ 2²⁶ entries; (S5) `upper` ≥ ‖Rc‖ + ‖E‖ (plus the
INT-05 term) and `rel` ≥ `upper`/‖x‖.

## 2 Proof

**Scalar lemmas** (Higham 2002, Lemma 3.1 and Lemma 3.3; the inequalities with numerical constants are also
checked exactly over n, w ∈ {1, …, 2·10⁴} ∪ {2^j : 15 ≤ j ≤ 26} by the verification script, section 4).

- (Λ1) A product of k factors (1 + δ_i)^{±1}, |δ_i| ≤ u, ku < 1, equals 1 + θ_k with |θ_k| ≤ γ_k. (Higham Lemma 3.1.)
- (Λ2) (1 + γ_a)(1 + γ_b) ≤ 1 + γ_{a+b}; γ_a + u(1 + γ_a) ≤ γ_{a+1}. (Higham Lemma 3.3 and its proof: γ_{a+1} − γ_a =
  u/((1 − au)(1 − (a+1)u)) ≥ u(1 + au + …) ≥ u(1 + γ_a) since γ_a = au/(1 − au) ≤ 2au for au ≤ 1/2.)
- (Λ3) `gamma(k)` = fl(fl(ku)/fl(1 − ku)) = γ_k(1 + θ₂), |θ₂| ≤ γ₂ (ku is exact: an integer times a power of two).
- (Λ4, norm guards) for 1 ≤ n ≤ 2²⁶ and N ∈ {n, 2n}:
  (1 − γ_N)^{1/2}(1 − u)·(1 + γ_{n+2}(1 − u)²)·(1 + 8u)(1 − u)⁴ ≥ 1 + 3u  and
  (1 + γ_N)^{1/2}(1 + u)·(1 − γ_{n+2}(1 + u)²)·(1 + u)³/(1 + 8u) ≤ 1 − 3u.
  First order: γ_{n+2} − γ_N/2 ≥ (n + 2)u − nu(1 + 2nu + …) ≥ 2u − 2(nu)² ≥ 1.5u for n ≤ 2²⁶ ((nu)² ≤ 2⁻⁵⁴ = u/2), and the
  guards contribute 8u − 5u = 3u; the remaining second-order terms are ≤ 4(nu)² + 50u² ≤ 2.5u in total, which leaves
  ≥ 1 + 2u in the worst case of the exact evaluation — the script checks the sharper constant 3u exactly on the
  grid. (The slack is what absorbs the extra rounding of step 5.)
- (Λ5, |A||x| guard) for 1 ≤ w ≤ 2²⁶: (1 − γ_w)(1 − u)³(1 + 4u)/(1 − γ_{w+1}(1 − u)²) ≥ 1 + u. First order:
  (1 − γ_w)/(1 − γ_{w+1}(1 − 2u)) ≥ 1 + (γ_{w+1} − γ_w) − 2uγ_{w+1} ≥ 1 + u(1 + γ_w) − 2uγ_{w+1} ≥ 1 + u − 2uγ_{w+1};
  times (1 − 3u + 3u²)(1 + 4u) ≥ 1 + u − 12u²: product ≥ 1 + 2u − O(wu²) ≥ 1 + u.
- (Λ6, E guard) for 1 ≤ w ≤ 2²⁶: (1 − u)⁶(1 + 4u)·(1 + u/γ_{w+2}) ≥ 1, since (1 − u)⁶(1 + 4u) ≥ 1 − 2u − 16u² and
  u/γ_{w+2} = (1 − (w+2)u)/(w + 2) ≥ 2u·(1 + 16u)/(1 − 2u − 16u²) whenever (w + 2)·2u(1 + …) ≤ 1 − (w + 2)u, i.e. for
  (w + 2)u ≤ 1/3.

  *Erratum (10 Nov 2026, found by the Lean formalization `CertifyCore.Theorems.Thm1`, theorems `lambda6` and
  `lambda6_fails_at_boundary`, docs/lean/CORE.md section 8).* The closing condition
  (w + 2)u ≤ 1/3 admits exactly one value, w + 2 = 3002399751580330 = ⌊2⁵³/3⌋, at which Λ6 fails; a correct closing
  condition is (w + 2)u ≤ (1 − 5u)/3. Λ6 as stated, for 1 ≤ w ≤ 2²⁶, holds (kernel-checked). Likewise the upper
  norm guard Λ4 with N = 2n holds for n ≤ 2²⁶ as stated and fails from n = 84886744 (exact computation). The code
  (`enclosure.py` `_rounding_bound`, `fp.py` norm guards) does not check w ≤ 2²⁶ or n ≤ 2²⁶; no stored operator comes
  near these sizes (largest dimension in the run records 1116229), and enforcing the range is a candidate for 3.2.2.

**Step 1: the residual vector** (`enclosure.py` lines 343–346: `HX = adapter.matvec(Xb)`, `Rb = HX − Xb * mub`).
Real data. fl(Ax)_i is a sum of at most w exact products a_{ij}x_j in some order with w products and ≤ w − 1
additions (or ≤ w FMAs): each term carries a factor (1 + θ_w) (Λ1), so |fl(Ax)_i − (Ax)_i| ≤ γ_w a_i (Higham eq.
(3.4)). fl(μx_i) = μx_i(1 + δ), |δ| ≤ u. The subtraction: Rc_i = (fl(Ax)_i − fl(μx_i))(1 + δ′). Hence

    |Rc_i − r_i| ≤ (1 + u)(γ_w a_i + u q_i) + u|(Ax)_i − μx_i| ≤ (γ_w(1 + u) + u) a_i + (2u + u²) q_i =: e_i,

using |(Ax)_i − μx_i| ≤ a_i + q_i. By (Λ2) γ_w(1 + u) + u ≤ γ_{w+1} ≤ γ_{w+2} and 2u + u² ≤ γ₂ ≤ γ_{w+2}, so
e_i ≤ E*_i := γ_{w+2}(a_i + q_i) = g(a_i + q_i), with slack E*_i − e_i ≥ (γ_{w+2} − γ_{w+1}) a_i + (γ_{w+2} − γ₂) q_i ≥ u(a_i + q_i)
(γ_{w+2} − γ_{w+1} ≥ u and γ_{w+2} − γ₂ ≥ wu ≥ u for w ≥ 1), i.e. E*_i ≥ e_i(1 + u/γ_{w+2}).
Complex data. A product of two complex binary64 numbers computed as (ac − bd, ad + bc) with four real products
and two additions satisfies |fl(zw) − zw| ≤ √2(2u + u²)|z||w| ≤ γ₃|z||w| (expand both components; (|ac| + |bd|)² +
(|ad| + |bc|)² ≤ 2|z|²|w|²; Higham Lemma 3.5 gives the same constant); a complex addition rounds each component,
a relative error ≤ u in modulus. Each term of fl(Ax)_i therefore carries (1 + θ₃)(1 + θ_{w−1}) ≤ 1 + θ_{w+2} (Λ2), so
|fl(Ax)_i − (Ax)_i| ≤ γ_{w+2} a_i; with the real μ the rest is as above: e_i ≤ (γ_{w+2}(1 + u) + u) a_i + γ₂ q_i ≤
γ_{w+4}(a_i + q_i) ≤ 2γ_{w+6}(a_i + q_i) = g(a_i + q_i), slack ≥ γ_{w+6}(a_i + q_i) ≥ E*_i/2.

**Step 2: the bound on |A||x|** (`_rounding_bound`, lines 293–296: `E = adapter.abs_matvec(X)`, `E /= (1.0 − gw)`,
`E *= (1 + 4U)` with `gw = gamma(w + 1)`). `absHX_i` = fl(Σ_j |a_{ij}||x_j|) is a sum of w non-negative products
(the moduli |a_{ij}|, |x_j| of complex data are themselves rounded once each, a relative u, absorbed below in
the complex g), so absHX_i ≥ a_i(1 − γ_w) (all terms have one sign: (Λ1) with no cancellation). `gw` = γ_{w+1}(1 + θ₂)
≥ γ_{w+1}(1 − u)² (Λ3). The three roundings of the line (1 − gw, the division, the product) each lose at most a
factor (1 − u). Therefore

    absHX_up_i ≥ a_i (1 − γ_w)(1 − u)³(1 + 4u) / (1 − γ_{w+1}(1 − u)²) ≥ a_i(1 + u) ≥ a_i      by (Λ5).

**Step 3: the entrywise bound E** (lines 297–301: `aX = |X| |mu|`, `E += aX`, `E *= g`, `E *= (1 + 4U)`). Four
roundings (the product |x_i||μ|, the addition, the product with g, the product with (1 + 4u)) and the two inside
`gamma_dot` (g ≥ γ-value·(1 − u)²), all on non-negative quantities, so

    E_i ≥ g (absHX_up_i + q_i)(1 − u)⁶(1 + 4u) ≥ g (a_i + q_i)(1 − u)⁶(1 + 4u) = E*_i (1 − u)⁶(1 + 4u) ≥ e_i ≥ |Rc_i − r_i|,

the last two by (Λ6) with E*_i ≥ e_i(1 + u/γ_{w+2}) (real data) or by the slack E*_i/2 (complex data).

**Step 4: the norms** (`fp.norm2_upper`, `fp.norm2_lower`, lines 149–162; used at lines 351–352 and 356).
`np.linalg.norm(v)` = fl(√fl(Σ_j v_j²)) with the sum of N squares (N = n real entries, or 2n real components) as a
BLAS dot product: S := fl(Σ v_j²) = Σ v_j²(1 + θ_N^{(j)}) ∈ ‖v‖²[1 − γ_N, 1 + γ_N] (non-negative terms, (Λ1)), and
s := fl(√S) = √S(1 + δ) ∈ ‖v‖[(1 − γ_N)^{1/2}(1 − u), (1 + γ_N)^{1/2}(1 + u)]. `norm2_upper(v)` = fl(fl(fl(s·fl(1 + γ̂))·(1 + 8u))
+ T_n) with γ̂ = `gamma(n + 2)` ≥ γ_{n+2}(1 − u)² and T_n ≥ 0: three roundings on the product and one on the final
addition, each ≥ (1 − u). Hence

    norm2_upper(v) ≥ ‖v‖ (1 − γ_N)^{1/2}(1 − u)(1 + γ_{n+2}(1 − u)²)(1 + 8u)(1 − u)⁴ ≥ ‖v‖(1 + 3u)      by (Λ4).

Likewise `norm2_lower(v)` = max(0, fl(fl(fl(s·fl(1 − γ̂))/(1 + 8u)) − T_n)) with γ̂ ≤ γ_{n+2}(1 + u)² and three roundings
each ≤ (1 + u) (the subtraction of T_n ≥ 0 only lowers the value, and fl is monotone):

    norm2_lower(v) ≤ ‖v‖ (1 + γ_N)^{1/2}(1 + u)(1 − γ_{n+2}(1 + u)²)(1 + u)³/(1 + 8u) ≤ ‖v‖(1 − 3u)      by (Λ4),

and norm2_lower(v) = 0 exactly when the formula is ≤ 0, in which case `rel` = ∞ (line 375), a valid bound.

**Step 5: the radius** (lines 368 and 375: `upper = (R_up + E_up) + A_{n,w}`, `rel = upper / xl · (1 + 4u) + 2 ULP_MIN`).
Two roundings in `upper`, two in `rel` (the final `+ 2 ULP_MIN` is exact or rounds up, and is ≥ 0):

    upper ≥ (R_up + E_up)(1 − u)² ≥ (‖Rc‖ + ‖E‖)(1 + 3u)(1 − u)² ≥ (‖Rc‖ + ‖E‖)(1 + u − 5u²),
    rel ≥ upper (1 − u)²(1 + 4u) / xl ≥ (‖Rc‖ + ‖E‖)(1 + u − 5u²)(1 + 2u − 7u²) / (‖x‖(1 − 3u)) ≥ (‖Rc‖ + ‖E‖)/‖x‖.

(The count of roundings on the path from the norms to `rel` is four; (1 + 4u) covers three (Λ1: (1 − γ₃)(1 + 4u) ≥ 1)
and the fourth is paid by the 3u slack of Step 4, which is why the slack is stated as part of (S4).) Finally,
by the triangle inequality and Step 3, ‖r‖ ≤ ‖Rc‖ + ‖Rc − r‖ ≤ ‖Rc‖ + ‖E‖ (E_i ≥ |Rc_i − r_i| entrywise), so

    rel ≥ (‖Rc‖ + ‖E‖)/‖x‖ ≥ ‖Ax − μx‖/‖x‖.  ∎

**Remarks.** (1) The model shift is added afterwards with `add_up` (`docs/proofs/appA_model-shift.md`, Theorem A5);
nothing here depends on it. (2) The 3.1 manuscript appendix ("k ≤ 6 roundings, factor 1 + 8u, shift multiplied
in") does not correspond to these lines; the scaffold `docs/proofs/appendixA_scaffold.tex` states the chain
above. (3) For a spec operator (`OperatorAdapter.kind == "spec"`), Ax is applied matrix-free: each entry of the
result is a sum of at most m = n_terms 2^{L_max} monomial contributions, each a product of the coupling, ≤ L_max
local amplitudes (one rounding each where irrational) and x_j (one more product): ≤ 2L_max + 1 roundings per
contribution and m − 1 in the sum, i.e. (1 + θ_{m+2L_max}); with w := m + 2L_max + 2 (`enclosure.OperatorAdapter`,
changed on this branch from m + L_max + 2, which was short for L_max ≥ 5) the Step 1 bound γ_w a_i holds with
a_i the row sums of the exact |H|_abs, and `abs_matvec` returns ≥ a_i(1 − γ_w) as in Step 2. (4) Real data with
an exactly zero residual (an exact eigenpair of a dyadic matrix) gives Rc = 0 and rel = ‖E‖/xl·(1 + 4u) > 0:
the radius is never zero for a nonzero x (E_i ≥ g q_i > 0 unless μx = 0 and Ax = 0).

## 3 Where the code implements it

`certify_ed/enclosure.py` and `certify_ed/fp.py`, branch `track/backlog-proofs` (line numbers of this branch).

| step | lines | code | proof constant |
|---|---|---|---|
| 1 | 343, 346 | `HX = adapter.matvec(Xb)`; `Rb = HX - Xb * mub[None, :]` | e_i ≤ g(a_i + q_i) |
| 2 | 293–296 | `E = adapter.abs_matvec(X)`; `E /= (1.0 - gw)`; `E *= (1 + 4 * U)`; `gw = gamma(w + 1)` | (Λ5): absHX_up ≥ a_i |
| 3 | 297–301 | `aX = abs(X) * abs(mu)`; `E += aX`; `E *= g`; `E *= (1 + 4 * U)`; `g = gamma_dot(w + 2, cplx)` | (Λ6): E_i ≥ e_i |
| 4 | 149–162 (fp), 351–352, 356 | `norm2_upper(v) = s (1 + gamma(n + 2)) _GUARD + norm_underflow_term(n)`; `norm2_lower(v) = max(0, s (1 - gamma(n + 2)) / _GUARD - norm_underflow_term(n))` | (Λ4): ≥ ‖v‖(1 + 3u), ≤ ‖v‖(1 − 3u) |
| 5 | 368, 375 | `upper = (R_up + E_up) + residual_additive_term(n, w, cplx)`; `rel = where(xl > 0, upper / xl, inf) * (1 + 4 * U) + 2 * ULP_MIN` | four roundings: (1 + 4u) and the 3u slack |
| constants | fp 30, 50, 58, 93 | `_GUARD = 1 + 8u`; `gamma(k) = ku/(1 - ku)`; `gamma_dot(k, cplx) = 2 gamma(k + 4)` or `gamma(k)`; `up(x) = x _GUARD` | (Λ3) |
| spec width | enclosure `OperatorAdapter.__init__` (spec branch) | `width = n_terms 2^max_len + 2 max_len + 2` | Remark 3 |

Rounding direction: every quantity is an upper bound on what it stands for (absHX_up on a_i, E on the entrywise
error, norm2_upper on the norm, rel on the radius) and `xl` a lower bound on ‖x‖. The test
`tests/test_theorem1_accounting.py` pins the stages on a fixed 3 × 3 case and random cases in exact rational
arithmetic; `tests/test_backlog_proofs.py::test_theorem1_accounting_slack_constants` pins (Λ4)–(Λ6) at a few n, w.


**Correction after the independent re-derivation (2026-11-03, `docs/proofs/VERDICTS.md`).** For complex data the moduli |a_ij| and |x_j| are each rounded once (hypot), so (S2) `absHX_up_i >= a_i` needs the division constant γ_{w+3} in place of γ_{w+1}; `enclosure._rounding_bound` now uses `gamma(width + 3)` when the data are complex (looser bound; real data unchanged). The (Λ5) argument then applies with w + 2 in place of w. Section 1 (S2) and Step 2 of section 2 are to be read with this constant for complex data; Sarang's check (D-12) remains pending.

**Correction (2026-11-18), upper moduli; UNVERIFIED (D-12).** The correction above counts one rounding per modulus.
NumPy's complex absolute value is not within one rounding: its relative error reaches 2.47u on the reference host,
and no bound on it is established (`docs/proofs/fp_modulus-up.md`, section 1). Since 3.3.0.dev0 the moduli of complex data come from
`fp.modulus_up` and `fp.abs_up`, which return m ≥ |z| (Lemma M there). Step 2 then reads, for complex data as for real
data: `absHX_i` = fl(Σ_j m_ij m_j) ≥ a_i(1 − γ_w), with no count for the moduli (the corollary of Lemma M). The code
keeps `gamma(width + 3)` for complex data. (Λ5) is stated with the division index w + 1; its left side increases with
the index, since γ_k does, so

    (Λ5′)  (1 − γ_w)(1 − u)³(1 + 4u)/(1 − γ_k(1 − u)²) ≥ 1 + u   for every k ≥ w + 1, in particular k = w + 3,

and (S2) `absHX_up_i` ≥ a_i(1 + u) ≥ a_i holds for complex data with two spare counts. The sentence "The (Λ5) argument
then applies with w + 2 in place of w" of the correction above is superseded, and so is the parenthesis of Step 2 on
moduli "rounded once each". In Step 3, |x_i||μ| is formed from m_i ≥ |x_i| with one rounding, as written; the complex
slack E*_i/2 is no longer needed for the moduli. `verify_thm1_fp-accounting.py` now checks (S2) with the code's
constant (w + 3 for complex data) and the upper moduli, requires its replica of E to equal `_rounding_bound` bit for
bit, checks (Λ5′) at k = w + 3 on the grid, and adds complex cases whose entries are the arguments on which `np.abs`
is too small: run of 2026-11-18 (`--cases 200`, `docs/proofs/runs/thm1_fp-accounting_2026-11-18.txt`): Λ 20 000 grid
points + 12 powers of two; R 200 random + 606 adversarial cases; S 400 random + 1 200 adversarial checks; 0 violations.

## 4 Computational verification

`docs/proofs/verify_thm1_fp-accounting.py` (exact rationals: `fractions.Fraction` for every stored number,
`decimal` square roots with directed rounding for the norms):

- Λ: the scalar lemmas (Λ2), (Λ4), (Λ5), (Λ6) exactly for every n, w ≤ 2·10⁴ and for 2^j, 15 ≤ j ≤ 26 (and (Λ4) with N = 2n);
- R: random dense real and complex Hermitian A (n ≤ 7), random x and μ (exact eigenpairs, perturbed eigenpairs,
  random vectors, μ far from the spectrum): per case the stage inequalities (S2) absHX_up ≥ |A||x|, (S3) E ≥ |Rc − r|,
  (S4) norm2_upper ≥ ‖v‖(1 + 3u) and norm2_lower ≤ ‖v‖(1 − 3u) for v ∈ {Rc, E, x}, (S5) upper ≥ ‖Rc‖ + ‖E‖ and
  rel ≥ ‖r‖/‖x‖, all with exact ‖·‖ bracketed by directed-rounded square roots; sparse A through the same adapter;
- adversarial: μ = 0 with Ax = 0 (zero residual), x with entries 10^{±150} (scaling), x with a single nonzero
  entry (w effectively 1), A = 0, A diagonal, a 1 × 1 case, μ = 10¹⁰ (q_i ≫ a_i), x with cancelling components
  (|Rc| ≪ |A||x|), complex A with real x and real A with complex x, a spec operator (`SpecOperator`) against
  its dense build for a spin-1 chain (Remark 3).

Counts (run of 2026-10-30, `--cases 20000 --seed 6`, output in `docs/proofs/runs/thm1_fp-accounting.txt`): Λ 20 000
grid points + 12 powers of two; R (rel ≥ exact radius) 20 000 random + 246 adversarial cases (12 families × 20
one-ulp jitters, and 6 spec-operator pairs); S (stage inequalities) 40 000 random + 480 adversarial checks;
0 violations.

## 5 Analytic cross-check

(i) *A 1 × 1 case.* A = (a), x = (ξ), μ: r = (a − μ)ξ, Rc = fl(fl(aξ) − fl(μξ)), e = (γ₁(1 + u) + u)|aξ| + (2u + u²)|μξ|,
E* = γ₃(|aξ| + |μξ|) (w = 1). rel ≥ (|Rc| + E)/|ξ| ≥ |a − μ|: for a = μ the exact radius is 0 while rel ≈ 2γ₃|a| ≈ 6u|a|,
the price of the entrywise bound, as expected.

(ii) *Exact eigenpair of a dyadic matrix.* Heisenberg spin-½ with exact eigenvector entries (e.g. the singlet of
two sites, x = (0, 1, −1, 0)/1 with μ = −3/4): Ax = μx exactly in binary64, Rc = 0, and rel = ‖E‖(1 + 4u)/xl with
E_i = g(absHX_up_i + |μx_i|)(1 + 4u) ≈ γ₆(3/4 + 3/4)·… — the radius is γ-small but positive (Remark 4), consistent
with the recorded fixtures where exact eigenpairs carry radii of order 10⁻¹⁵.

(iii) *Scaling.* Replacing (A, μ) by (tA, tμ), t a power of two, scales every stored quantity of the chain
exactly by t (no new rounding), and rel by t: the bound is homogeneous, as Theorem 1 is.

(iv) *The INT-04 list.* Steps 1–5 are the six items of `docs/theory.md` §2 with the constants named: three factors
(1 + 4u) (steps 2, 3, 5), the guards (1 ± γ_{n+2})(1 + 8u)^{±1} (step 4), and the slacks of γ_{w+1} over γ_w (step 2),
of γ_{w+2} over the entrywise error (step 3) and of γ_{n+2} over γ_N/2 + u (step 4); the one change on this branch
is the fourth rounding of step 5 (the INT-05 term), paid by the step-4 slack.

## 6 Status

Derived on 2026-10-30; machine-checked (20 000 (40 000 stage checks, 20 000 lemma grid points) random + 246 (480 stage checks, 12 lemma grid points) adversarial cases, 0 violations); independent re-derivation: done 2026-10-30 (written before the original proof was read, not a human check; verdict GAP, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
Correction for upper moduli on 2026-11-18 (end of section 3; UNVERIFIED, D-12).
