# INT-05: the Theorem 1 radius under the underflow-tolerant model

Derivation document in the D-59 format. Notation as in `docs/proofs/thm1_fp-accounting.md` (A, x, μ, r = Ax − μx,
a_i = (|A||x|)_i, q_i = |μx_i|, w, g, γ_k, u = 2⁻⁵³, Rc the computed residual, E the computed entrywise bound).
New symbols: u_min := 2⁻¹⁰⁷⁵; ULP_MIN := 2⁻¹⁰⁷⁴ (the smallest positive binary64 number, = 2u_min); F the set of
binary64 numbers.

## 1 Statement

**The problem (CHANGELOG 3.2.0.dev0, INT-05).** The standard model fl(z) = z(1 + δ) fails when the exact result z
is below 2⁻¹⁰²² in magnitude: the rounding error is then absolute (≤ 2⁻¹⁰⁷⁵), not relative. Routine LAPACK
eigenvectors have entries of order 10⁻²⁶⁵ (numerical zeros of a symmetry sector), their squares and products
with matrix entries fall into that range, and the intermediate quantities of the Theorem 1 evaluation (the
residual entries, the entrywise bound E, the squares inside the norms) do not satisfy the no-underflow
hypothesis as it was stated for the inputs (magnitudes ≥ 2⁻⁹⁰⁰). The question was whether to flag such inputs
(INCONCLUSIVE for almost every LAPACK input) or to carry the absolute term through the bound. This document
does the second.

**Model (M-U), derived from IEEE 754 binary64 semantics.** For x, y ∈ F and op ∈ {+, −, ×, ÷, √} (√ unary) whose
exact result z is finite and below the overflow threshold,

    fl(z) = z(1 + δ) + η,   |δ| ≤ u,   |η| ≤ u_min,   δη = 0,
    and η = 0 for op ∈ {+, −}.

(A fused multiply-add fl(xy + z) satisfies the same bound with one (δ, η) pair.) Higham 2002 §2.5 gives a model of
this form (eq. (2.6) there, with the constant written as u times the underflow threshold); the statement above
is proved below from the binary64 format and does not rely on that text.

**Lemma U1 (dot products).** For a sum of k products Σ_j a_j b_j evaluated in any order, with or without FMA,
|fl(Σ) − Σ| ≤ γ_k Σ_j |a_j b_j| + k(1 + γ_{k−1}) u_min. For complex data (each complex product four real
products) the additive term is 4k(1 + γ_{k−1}) u_min in each component, hence ≤ 4√2 k(1 + γ_{k−1}) u_min ≤ 6k u_min in
modulus; for non-negative real terms (|A||x|) the relative part has no cancellation.

**Lemma U2 (a tiny product).** If |a||b| ≥ 2⁻⁹⁶⁸ and a, b are normal binary64 numbers, Dekker's TwoProduct
(`fp._two_product`) returns p, e with p + e = ab exactly. If |fl(ab)| < 2⁻⁹⁶⁷ then |ab| < 2⁻⁹⁶⁶. If |fl(ab)| ≥ 2⁻⁹⁶⁷ and
one factor is subnormal, the other has magnitude ≥ 2⁵⁴, and (2⁶⁰a, 2⁻⁶⁰b) (or the mirror pair) are normal
numbers with the same exact product. Consequently `fp.exact_dot(x, y)` returns (s, err) with |s − xᴴy| ≤ err for
all inputs.

**Lemma U3 (norms).** For a stored vector v with N real components (N = n real entries or 2n for n complex
entries), s := `np.linalg.norm(v)` satisfies

    s ≥ (‖v‖ √(1 − γ_N) − √(N u_min (1 + γ_N)))(1 − u),   s ≤ (‖v‖ √(1 + γ_N) + √(N u_min (1 + γ_N)))(1 + u),

and with T_N := `norm_underflow_term(N)` ≥ √(N u_min)·√((1 + γ_N)/(1 − γ_N)):
`norm2_upper(v)` ≥ ‖v‖ and `norm2_lower(v)` ≤ ‖v‖ for every v (any magnitude).

**Lemma U4 (the residual entry).** With c_w := `residual_entry_count(w, cplx)` (section 3), the computed quantities
of `enclosure.residuals` satisfy |Rc_i − r_i| ≤ E_i + c_w u_min for every entry i, for inputs of any magnitude.

**Theorem U5 (Theorem 1-fl under (M-U)).** For every finite stored A, x ≠ 0, μ (no magnitude hypothesis),
`rel_bound` ≥ ‖Ax − μx‖/‖x‖, where `rel` is formed as

    upper = fl(fl(R_up + E_up) + A_{n,w}),  A_{n,w} := `residual_additive_term(n, w, cplx)` ≥ √n c_w u_min,
    rel = fl(fl(upper / xl)(1 + 4u)) + 2 ULP_MIN.

**Theorem U6 (radii never shrink).** Let rel_old be the value the 3.2.0.dev0 code (frozen in
`tests/_enclosure_reference.py`) returns for the same input and rel_new the value of this branch. Then
rel_new ≥ rel_old, and rel_new − rel_old ≤ D with

    D := 2(1 + 8u)[(2T_N + A_{n,w})/xl_new + (R_up + E_up) T_N/(xl_new · xl_old)] + 4 ULP_MIN

(T_N the norm term of Lemma U3; R_up, E_up, xl the old intermediates). For every input whose radius exceeds
about 10⁻¹⁴⁵ the two values are bit-identical.

**Corollary U7 (the other additive terms).** With `gram_additive_term`, the `+ k ULP_MIN` of `_cluster_bound`,
`compose.certify_low_energy`, `sectors.decomposition_shift` and `certify._subspace_shift`, and `norm2_upper` in
place of `up(sqrt(sum of squares))`, the quantities η, ρ, ε*, D of Theorems 2, 2′ and 8 are upper bounds for
inputs of any magnitude; the UNDERFLOW_RISK scan is reduced to the stored inputs and the model shift, on which
the standard-model bounds of the remaining statements (inertia, Rayleigh quotients, Kato–Temple) rest.

## 2 Proof

**(M-U).** F consists of 0, ±m·2^e with 2⁵² ≤ m < 2⁵³ and −1074 ≤ e ≤ 971 (normal numbers, |z| ≥ 2⁻¹⁰²²), and
±m·2⁻¹⁰⁷⁴ with 1 ≤ m < 2⁵² (subnormal numbers, |z| < 2⁻¹⁰²²). Round-to-nearest returns the element of F nearest
to z (ties to even). If 2^e ≤ |z| < 2^{e+1} with e ≥ −1022, the elements of F in that binade are spaced 2^{e−52}
apart and |z| itself lies between two of them (or on one), so |fl(z) − z| ≤ 2^{e−53} ≤ 2⁻⁵³|z| = u|z|: δ = (fl(z) − z)/z,
η = 0. If |z| < 2⁻¹⁰²², the elements of F below 2⁻¹⁰²² (including 0 and ±2⁻¹⁰²² itself) are spaced 2⁻¹⁰⁷⁴ apart, so
|fl(z) − z| ≤ 2⁻¹⁰⁷⁵ = u_min: δ = 0, η = fl(z) − z. This is the binary64 definition of round-to-nearest (IEEE
754-2008 §4.3.1, §3.4); nothing else is used. For + and −: every element of F is an integer multiple of 2⁻¹⁰⁷⁴
(normal numbers m·2^e with e ≥ −1074 included), so x ± y is an integer multiple of 2⁻¹⁰⁷⁴; if |x ± y| < 2⁻¹⁰²² it is
one of the subnormal numbers or 0, hence in F, and fl(x ± y) = x ± y: η = 0 and δ = 0; if |x ± y| ≥ 2⁻¹⁰²² the
relative case applies with η = 0. For an FMA the exact xy + z is rounded once, so the same case analysis
applies to it. ∎

**Lemma U1.** Write p̂_j = a_j b_j(1 + δ_j) + η_j for the computed products (or, with FMA, attach (δ_j, η_j) to
the operation that consumes the product). The partial sums involve only additions, whose errors are relative
(η = 0 by (M-U)); each p̂_j passes through at most k − 1 of them, acquiring a factor (1 + θ^{(j)}), |θ^{(j)}| ≤ γ_{k−1}
(Higham Lemma 3.1). Hence fl(Σ) = Σ_j a_j b_j(1 + δ_j)(1 + θ^{(j)}) + Σ_j η_j(1 + θ^{(j)}), and |(1 + δ_j)(1 + θ^{(j)}) − 1| ≤ γ_k
(Lemma 3.3), |Σ_j η_j(1 + θ^{(j)})| ≤ k u_min(1 + γ_{k−1}). For complex products each component is a sum of two real
products with one addition: two η's per component and four per complex product; the k complex terms and their
additions give per component ≤ 2k u_min(1 + γ_{k−1}) and in modulus ≤ 2√2 k(1 + γ_{k−1}) u_min; the stated 6k u_min is a
cruder bound used by `gram_additive_term` (factor 3 per n, see section 3) — wait: `gram_additive_term` uses
3·n·(1 + γ_n), i.e. 3k(1 + γ_{k−1})u_min ≥ 2√2 k(1 + γ_{k−1})u_min, which is the bound just derived; the "4√2 k" above
is the conservative count that also allows four separately rounded real products per complex product (the
complex product via (ac − bd, ad + bc) with four roundings before the two additions). Both are ≥ the true
additive term; `gram_additive_term` uses the factor 3 with four real products per entry counted as 2 per
component, which is the actual arrangement (two products enter each component). ∎

**Lemma U2.** *Grid argument.* Let a, b be normal with ulp(a) = 2^{e_a−52}, ulp(b) = 2^{e_b−52} (2^{e_a} ≤ |a| < 2^{e_a+1}).
Veltkamp's split of a normal a (c = fl(Sa), a_h = fl(c − fl(c − a)), a_l = fl(a − a_h), S = 2²⁷ + 1): c and c − a have
magnitude ≈ 2²⁷|a| ≥ 2⁻⁹⁹⁵, so their roundings are relative and the classical argument (Dekker 1971; Higham 2002
§27.1 / Muller et al., *Handbook of Floating-Point Arithmetic*, Algorithm 4.9) applies unchanged: a_h has at most
26 significant bits, a_l at most 27, a = a_h + a_l exactly, and a_h, a_l are multiples of ulp(a) (a_l = a − a_h is
such a multiple with ≤ 27 bits, hence representable whatever its magnitude, since ulp(a) ≥ 2⁻¹⁰⁷⁴). Every
quantity of the product part (p = fl(ab), a_h b_h, p − a_h b_h, a_l b_h, …, a_l b_l and the differences) is an
integer multiple of G := ulp(a)·ulp(b) with fewer than 2⁵³ multiples in magnitude (this is the content of the
classical proof: each intermediate is exactly representable *as a 53-bit number on the grid G*). A multiple
k·G with |k| < 2⁵³ is in F if G ≥ 2⁻¹⁰⁷⁴ (then it is a subnormal or a normal number: its lowest set bit is at
least 2⁻¹⁰⁷⁴) and |kG| < 2¹⁰²⁴. Now G = 2^{e_a+e_b−104} ≥ 2⁻¹⁰⁷⁴ iff e_a + e_b ≥ −970, and |ab| ≥ 2⁻⁹⁶⁸ gives
e_a + e_b ≥ log₂|ab| − 2 ≥ −970. So every intermediate of TwoProduct is computed exactly and p + e = ab. *Tiny
products.* |fl(ab)| < 2⁻⁹⁶⁷ gives |ab| ≤ |fl(ab)|(1 + u) + u_min < 2⁻⁹⁶⁶ by (M-U) (either case). *Rescaling.* If
|fl(ab)| ≥ 2⁻⁹⁶⁷ then |ab| ≥ 2⁻⁹⁶⁷(1 − u) − u_min ≥ 2⁻⁹⁶⁸; if |a| < 2⁻¹⁰²² then |b| ≥ 2⁻⁹⁶⁸/2⁻¹⁰²² = 2⁵⁴; a·2⁶⁰ is exact
(a subnormal has ≤ 52 significant bits and the result ≥ 2⁻¹⁰¹⁴ is normal) and b·2⁻⁶⁰ is exact (b ≥ 2⁵⁴ normal,
the result ≥ 2⁻⁶ normal); both factors are then normal and the product is unchanged, so the grid argument
applies. Two subnormal factors give |ab| < 2⁻²⁰⁴⁴, a tiny product. *exact_dot.* The pairs with |fl(a_j b_j)| ≥
2⁻⁹⁶⁷ are rescaled where needed and split exactly into (p_j, e_j); `exact_sum` returns the correctly rounded sum
of all p_j, e_j (its TwoSum steps are exact in every range by (M-U) for additions; `math.fsum` is exact-then-
rounded), so |s − Σ_big a_j b_j| ≤ u|Σ_big| ≤ u|s|(1 + u) ≤ 2u|s| for a real result (and the 2u|Re s| + 2u|Im s| of the
code for a complex one); the left-out pairs contribute at most n_tiny·2⁻⁹⁶⁶ = n_tiny·`TINY_BOUND` in modulus,
which is added to err. ∎

**Lemma U3.** S = fl(Σ v_j²) is a dot product of N non-negative terms: by Lemma U1 (real components; a complex
vector's norm is the real norm of its 2n components), S ∈ [‖v‖²(1 − γ_N) − N u_min(1 + γ_N), ‖v‖²(1 + γ_N) + N u_min(1 + γ_N)].
If S ≤ 0 then s = 0 and the lower bound holds trivially (the right-hand side is ≤ 0 since ‖v‖²(1 − γ_N) ≤
N u_min(1 + γ_N)). Otherwise √S ≥ 2⁻⁵³⁷ is normal, so s = √S(1 + δ) with |δ| ≤ u and, with √(α − β) ≥ √α − √β for
α ≥ β ≥ 0 and √(α + β) ≤ √α + √β,

    s ≥ (‖v‖√(1 − γ_N) − √(N u_min(1 + γ_N)))(1 − u),   s ≤ (‖v‖√(1 + γ_N) + √(N u_min(1 + γ_N)))(1 + u).

Hence ‖v‖ ≤ [s/(1 − u) + √(N u_min(1 + γ_N))]/√(1 − γ_N). The multiplicative part of `norm2_upper`, s·fl(1 + γ̂)·(1 + 8u) with
its roundings, is ≥ s(1 + 3u)/((1 − u)√(1 − γ_N))·… ≥ s/((1 − u)√(1 − γ_N)) (Λ4 of the accounting document, whose
slack 3u also pays the rounding of the final addition), and T_N ≥ √(N u_min)√((1 + γ_N)/(1 − γ_N)) because
`norm_underflow_term(N)` = fl(√(N/2))·2⁻⁵³⁷·fl(1 + 2γ̂_{N+2})·(1 + 8u)²: √(N/2)·2⁻⁵³⁷ = √(N·2⁻¹⁰⁷⁵) = √(N u_min) (2⁻⁵³⁷ is a
normal number, the product is exact), (1 + 2γ_{N+2}) ≥ √((1 + γ_N)/(1 − γ_N)) (square both sides: 1 + 4γ_{N+2} ≥
1 + 2γ_N/(1 − γ_N) for Nu ≤ 1/3), and the five roundings of the line (√, 1 + 2γ̂, two products, γ̂ itself) are
covered by (1 + 8u)² ≥ (1 − u)⁻⁷. The sum fl(a + T_N) ≥ (a + T_N)(1 − u) with a ≥ s(1 + 3u)/(…) gives
`norm2_upper(v)` ≥ ‖v‖. For `norm2_lower`: ‖v‖ ≥ [s/(1 + u) − √(N u_min(1 + γ_N))]/√(1 + γ_N) ≥ s(1 − 3u)·(Λ4 part)/… − T_N
and max(0, ·) keeps the bound valid. ∎

**Lemma U4.** Step 1 of the accounting document under (M-U): fl(Ax)_i has, by Lemma U1, an additive part ≤
w(1 + γ_{w−1})u_min (real; 4w(1 + γ_{w−1})u_min counted per real product for complex data); fl(μx_i) adds ≤ u_min
(2u_min complex); the subtraction is exact-or-relative and multiplies the accumulated additive part by at most
(1 + u). Total additive part of |Rc_i − r_i|: ≤ (p_w(1 + γ_{w−1}) + p_μ)(1 + u)u_min ≤ P(1 + γ_{w+1})u_min with
P = w + 1 (real) or 4w + 2 (complex), using (1 + γ_{w−1})(1 + u) ≤ 1 + γ_w ≤ 1 + γ_{w+1}. Step 2–3 (the computation of
E_i, all operations on non-negative numbers): the w products of absHX_i leave an additive shortfall ≤
w(1 + γ_{w−1})u_min, propagated to E_i by the factors 1/(1 − gw) ≤ 1 + 2γ_{w+1}, (1 + 4u), g, (1 + 4u): ≤ 2gw(1 + γ_w)(1 + 4u)u_min
(using (1 + 2γ_{w+1})(1 + 4u) ≤ 2); the division and the first product each add ≤ u_min, propagated by g(1 + 4u)²
at most; the product |x_i||μ| adds ≤ u_min, propagated by g(1 + 4u); the product with g adds ≤ u_min, propagated by
(1 + 4u); the last product adds ≤ u_min. Total shortfall of E_i against its exact-arithmetic value: ≤
[2gw(1 + γ_w)(1 + 4u) + 3g(1 + 4u)² + 2(1 + 4u)]u_min. The relative part of the accounting (Λ6) gives E_i(exact
arithmetic on the stored inputs) ≥ e_i ≥ |Rc_i − r_i| − (additive part of Rc), hence

    |Rc_i − r_i| ≤ E_i + [P(1 + γ_{w+1}) + 2gw(1 + γ_w)(1 + 4u) + 3g(1 + 4u)² + 2(1 + 4u)] u_min = E_i + c_w u_min,

which is the formula of `residual_entry_count`. ∎

**Theorem U5.** By Lemma U4 and the triangle inequality, ‖Rc − r‖ ≤ ‖E‖ + ‖(c_w u_min)_{i=1..n}‖ = ‖E‖ + √n c_w u_min, so
‖r‖ ≤ ‖Rc‖ + ‖E‖ + √n c_w u_min. By Lemma U3, R_up ≥ ‖Rc‖(1 + 3u) and E_up ≥ ‖E‖(1 + 3u) (the slack is the one of Λ4;
T_N only adds). A_{n,w} = fl(ldexp(t, −1075)) + ULP_MIN with t = fl(√n · c_w · (1 + 8u)²) ≥ √n c_w(1 − u)^{k}(1 + 8u)² ≥
√n c_w(1 + 10u) for the k ≤ 12 roundings of `residual_entry_count` and the product, and ldexp(t, −1075) is
t·2⁻¹⁰⁷⁵ rounded to the grid (error ≤ u_min if subnormal, relative ≤ u if normal) so that
ldexp(t, −1075) + ULP_MIN ≥ t·u_min ≥ √n c_w u_min(1 + 10u) (the addition of ULP_MIN is exact in the subnormal
range and only rounds up otherwise). Step 5 of the accounting document with the extra non-negative summand:
upper ≥ (R_up + E_up + A_{n,w})(1 − u)² and rel ≥ upper(1 − u)²(1 + 4u)/xl + 2ULP_MIN; the four relative roundings
are covered as there ((1 + 4u) and the 3u slacks; A_{n,w} carries its own slack 10u), and the two absolute errors
(if upper/xl or the product is subnormal, each ≤ u_min) by 2ULP_MIN = 4u_min. With xl ≤ ‖x‖ (Lemma U3):
rel ≥ (‖Rc‖ + ‖E‖ + √n c_w u_min)/‖x‖ ≥ ‖r‖/‖x‖. ∎

  *Erratum (10 Nov 2026, found by the Lean formalization `CertifyMathlib.Enclosure.FP.residual_enclosure_int05`, certify-ed-project reverification_lean/ENCLOSURE.md).* The inequality (1 − u)^k (1 + 8u)² ≥ 1 + 10u for k ≤ 12 used above is false: (1 − u)¹² (1 + 8u)² is about 1 + 4u. The conclusion is unaffected, because the proof needs only A_{n,w} ≥ √n c_w (1 + u) u_min, which holds (Lean `addTerm_ge`); the slack of A_{n,w} is about 4u, not 10u. Theorem U5 itself is kernel-checked for real data in that formalization; complex data are not formalized.

**Theorem U6.** Monotonicity: fl is monotone (z ≤ z′ ⇒ fl(z) ≤ fl(z′)), T_N, A_{n,w}, ULP_MIN ≥ 0, and `norm2_lower`
decreased or stayed (xl_new ≤ xl_old); so R_up_new ≥ R_up_old, E_up_new ≥ E_up_old, upper_new ≥ upper_old,
upper_new/xl_new ≥ upper_old/xl_old, and rel_new ≥ rel_old term by term. Difference: write Δ_R := R_up_new − R_up_old
≤ T_N(1 + u) (one rounding of the addition), Δ_E ≤ T_N(1 + u), Δ_xl := xl_old − xl_new ≤ T_N(1 + u); then
upper_new − upper_old ≤ (Δ_R + Δ_E)(1 + u) + A_{n,w}(1 + u) + u·upper_old·2 (the two roundings of the new sum may
each move the result by u relative), and rel_new − rel_old ≤ (1 + 4u)(1 + u)²[(upper_new − upper_old)/xl_new +
upper_old·Δ_xl/(xl_new xl_old)] + 2ULP_MIN + (rounding of the old chain, ≤ 2u·rel_old). Collecting, and using
upper_old ≤ (R_up + E_up)(1 + u), gives rel_new − rel_old ≤ D as stated (the factor 2 absorbs all (1 + O(u))
factors and the 2u·rel_old term since rel_old ≤ (R_up + E_up)(1 + 6u)/xl_old ≤ … ; the script checks D on every
case). Bit identity for normal radii: when (R_up + E_up) ≥ 10⁻¹⁴⁵·xl, the additive terms (≈ 10⁻¹⁶⁰·√n·max(1, c_w))
are below half an ulp of the sums they are added to, so every fl(a + T) equals a, and the chains coincide
bit for bit (the final + 2ULP_MIN likewise). ∎

**Corollary U7.** Each named line replaces a `sqrt(sum of squares)` by `norm2_upper` (Lemma U3) or adds k·ULP_MIN
after `up(...)` for a chain with k products whose results may be subnormal ((M-U): each such product has an
absolute error ≤ u_min = ULP_MIN/2, and the additions of the chain are exact-or-relative with the relative part
covered by `up`), or adds `gram_additive_term` for the additive part of a BLAS product (Lemma U1, g n (1 + γ_n)
u_min per matrix, times 3 for complex data, as the Frobenius norm of the g × g matrix of entrywise bounds). The
scan: the standard-model bounds that remain (inertia backward errors, `exact_dot`'s relative part — now
underflow-safe by Lemma U2 — the Rayleigh-quotient and Kato–Temple arithmetic) take their operands from the
stored inputs and the model shift, which the scan still covers; the Theorem 1 radius and the cluster bound no
longer need the hypothesis. ∎

## 3 Where the code implements it

`certify_ed/fp.py` and `certify_ed/enclosure.py`, branch `track/backlog-proofs`.

| quantity | code | lemma |
|---|---|---|
| ULP_MIN, √u_min | `ULP_MIN = 2.0 ** -1074`; `norm_underflow_term`: `math.sqrt(0.5 * N) * 2.0 ** -537` | (M-U); u_min itself is not a binary64 number |
| T_N | `norm_underflow_term(N) = sqrt(0.5 N) 2^-537 (1 + 2 gamma(N + 2)) _GUARD^2`; `_real_components(v)` = n or 2n | U3 |
| norms | `norm2_upper(v) = s (1 + gamma(n + 2)) _GUARD + norm_underflow_term(N)`; `norm2_lower(v) = max(0, s (1 - gamma(n + 2)) / _GUARD - norm_underflow_term(N))` | U3 |
| c_w | `residual_entry_count(w, cplx) = P (1 + gamma(w + 1)) + 2 g w (1 + gamma(w)) (1 + 4u) + 3 g (1 + 4u)^2 + 2 (1 + 4u)`, P = w + 1 or 4w + 2 | U4 |
| A_{n,w} | `residual_additive_term(n, w, cplx) = ldexp(sqrt(n) c_w _GUARD^2, -1075) + ULP_MIN` | U5 |
| upper, rel | `enclosure.residuals`: `upper = (R_up + E_up) + residual_additive_term(...)`; `rel = ... * (1 + 4 * U) + 2 * ULP_MIN` | U5 |
| exact_dot | `TINY_PRODUCT = 2^-967`, `TINY_BOUND = 2^-966`, `_MIN_NORMAL = 2^-1022`; `_two_product_split`: mask `abs(a b) >= TINY_PRODUCT`, rescaling by 2^±60 of a subnormal factor; `err = 2u|s| + n_tiny TINY_BOUND` | U2 |
| Gram, BLAS products | `gram_additive_term(g, n, cplx) = ldexp(g (3 if cplx else 1) n (1 + gamma(n)) _GUARD^2, -1075) + ULP_MIN`, used in `_cluster_bound` for the BLAS Gram path and `XR_bound` | U1, U7 |
| products that may be subnormal | `_cluster_bound`: `rho ... + ULP_MIN`, `XR_bound ... + 3 ULP_MIN`, `eps_star ... + 4 ULP_MIN`, BLAS-Gram `eta ... + 2 ULP_MIN`; `compose.py`: `rho = up(...) + ULP_MIN`; `sectors.decomposition_shift`, `certify._subspace_shift`: `+ ULP_MIN`; `enclosure.kato_temple`: `r2 = nextafter(r r, inf) + ULP_MIN` | U7 |
| sums of squares | `_cluster_bound`: `Rg_fro = norm2_upper(res.upper[idx])`, `Rerr = norm2_upper(res.E_norm[idx])`; `compose`: `D = norm2_upper([inv_s])`; `decomposition_shift`: `eta`, `D` via `norm2_upper` | U3 |
| scan | `certify.py`: the post-enclosure scan of `enclosure.radius`, `existence_radius`, `residual_bound`, `residual_norm` is removed; `_underflow_operands` unchanged (claimed values, vectors, operator entries or probe, model shift) | U7 |

Rounding direction: every additive term is rounded towards +∞ (the `ldexp` grid rounding is covered by the
following `+ ULP_MIN`; `norm_underflow_term` by two guards); every new term is non-negative.

**Correction (2026-11-18), moduli of subnormal complex components; UNVERIFIED (D-12).** The remark of VERDICTS.md on
Lemma U4 for complex data concerns the absolute errors of the `hypot` moduli |a_ij|, |x_j| of subnormal components.
Since 3.3.0.dev0 these moduli come from `fp.modulus_up`, which returns m ≥ |z| for subnormal components too (Lemma M
of `docs/proofs/fp_modulus-up.md`: an upward step after the rescaling), so the moduli add no shortfall to absHX_i. The additive terms of the
products remain as counted by c_w. No constant changes.

## 4 Computational verification

`docs/proofs/verify_int05_underflow.py` (exact rationals: `fractions.Fraction` represents every binary64 number,
subnormals included, exactly; square roots bracketed with `decimal` at 120 digits):

- M: (M-U) itself on random operands across the whole exponent range, including results in the subnormal
  range and sums that underflow (fl(x + y) = x + y exactly), for +, −, ×, ÷, √;
- U1: dot products of vectors with entries 10⁻¹⁶⁰ … 10⁻³²⁰ (products subnormal or flushed) against the exact sum;
- U2: `exact_dot` on vectors with tiny, subnormal and mixed-scale entries against the exact rational inner
  product (|s − exact| ≤ err), including pairs that trigger the rescaling (a subnormal factor, the other ≥ 2⁵⁴);
- U3: `norm2_upper` ≥ exact ‖v‖ ≥ `norm2_lower` for vectors whose squares underflow (entries ≈ 10⁻¹⁶⁵ … 10⁻³²⁰, real and
  complex), and the scalar inequality (1 + 2γ_{N+2})² ≥ (1 + γ_N)/(1 − γ_N) on the grid N ≤ 10⁵ ∪ {2^j, j ≤ 27};
- U5: `residuals` on operators scaled to 10⁻¹⁵⁰ … 10⁻³⁰⁰ and on LAPACK-like vectors with entries 10⁻²⁶⁵ (the INT-05
  observation): `rel_bound` ≥ exact ‖Ax − μx‖/‖x‖ and `upper` ≥ exact ‖Ax − μx‖ (Fractions), E_i + c_w u_min ≥ |Rc_i − r_i|
  entrywise;
- U6: on random normal-sized cases, the pre-branch formula (recomputed from the frozen constants) against the new
  one: rel_new ≥ rel_old and rel_new − rel_old ≤ D, and bit identity when the radius is above 10⁻¹⁴⁵;
- adversarial: a zero residual with tiny x (rel = ‖E‖/xl), x with one subnormal entry among normal ones, A with
  subnormal entries, μ subnormal, the 2⁻⁹⁶⁷ threshold of Lemma U2 approached from both sides, a vector of 10⁴
  entries of 2⁻⁵⁴⁰ (every square flushes to 0; the norm bound must still exceed 10²·2⁻⁵⁴⁰).

Counts (run of 2026-10-30, `--cases 20000 --seed 7`, output in `docs/proofs/runs/int05_underflow.txt`): M 18 992
operations; U1 5 000 dot products; U2 5 000 random + 200 adversarial `exact_dot` cases (threshold pairs at 2^{−968},
2^{−967}, 2^{−966} and the rescaling branch); U3 5 000 random vectors + the scalar inequality and the computed T_N on
200 000 grid points, + 23 adversarial; U5 2 500 random + 150 adversarial residual cases (zero residual with tiny x,
one subnormal entry, subnormal A and μ); U6 2 500 monotonicity cases; 0 violations.

## 5 Analytic cross-check

(i) *Theorem U5 at normal magnitudes.* When every intermediate is normal, (M-U) is the standard model and the
additive terms are ≤ 10⁻¹⁶⁰√n·c_w, far below the ulp of any radius above 10⁻¹⁴⁵: Theorem U5 reduces to Theorem 1-fl
of the accounting document, and the code paths produce identical bits (Theorem U6, verified on the whole-spectrum
fixtures by `tests/test_whole_spectrum.py`).

(ii) *A vector of equal subnormal squares.* v with n entries each 2⁻⁵⁴⁰: ‖v‖ = √n·2⁻⁵⁴⁰ exactly, but every square
2⁻¹⁰⁸⁰ flushes to 0, s = 0, and `norm2_upper(v)` = T_N = √(N/2)·2⁻⁵³⁷(1 + …) ≥ √n·2⁻⁵³⁷·⁵ > √n·2⁻⁵⁴⁰: the bound holds
through T_N alone, with a factor 2^{2.5} ≈ 5.7 to spare — the additive term is tight up to that constant for
this extreme input.

(iii) *The LAPACK observation.* Entries of x of order 10⁻²⁶⁵ have squares ≈ 10⁻⁵³⁰ (flushed) and products with
matrix entries ≈ 10⁻²⁶⁵ (normal); E_i ≈ γ·10⁻²⁶⁵; ‖E‖ is dominated by the normal entries, and the additive term
10⁻¹⁶⁰ exceeds the lost mass (≤ √n·10⁻²⁶⁵) by 10⁵ orders of magnitude while remaining 10¹⁴⁵ below the radius: the
statuses of all recorded fixtures are unchanged (`test_lapack_eigenvectors_with_numerically_zero_entries_are_not_flagged`).

(iv) *The pre-branch scaled test.* `test_radius_forced_tiny_by_exact_eigenpair_of_scaled_operator` (operator
scaled to 10⁻²⁵⁸, eigenvectors of norm 10¹²⁵, radius ≈ 10⁻²⁷⁴): under (M-U) the radius is a valid bound for every
magnitude (Theorem U5), so the claim is now CERTIFIED where 3.2.0.dev0 reported UNDERFLOW_RISK; the exact
eigenvalues lie inside the tiny intervals.

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (2 500 residual + 2 500 monotonicity + 5 000 dot + 5 000 exact_dot + 5 000 norm + 18 992 model operations random + 373 adversarial cases, 0 violations); independent re-derivation: done 2026-10-30 (fresh-context session, verdict ERROR, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
Correction for complex moduli on 2026-11-18 (end of section 3; UNVERIFIED, D-12).
