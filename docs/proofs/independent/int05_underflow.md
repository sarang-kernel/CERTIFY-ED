# Independent re-derivation: INT-05 (Theorem 1 radius under the underflow-tolerant model)

Fresh-context session, 2026-10-30. Sections 1 and 3 of `../int05_underflow.md` read first; proof written before
section 2 was opened. Notation as in the original and in `independent/thm1_fp-accounting.md`. u = 2⁻⁵³,
u_min = 2⁻¹⁰⁷⁵, ULP_MIN = 2⁻¹⁰⁷⁴. Classical facts: IEEE 754-2008 §3.3–3.4 (binary64 format: 53-bit significand,
exponent range, subnormals with fixed spacing 2⁻¹⁰⁷⁴, round-to-nearest); Higham (2002) Lemma 3.1, 3.3, §2.5 eq.
(2.6) (model with an additive underflow term), §4.2 (summation in any order); Dekker (1971) / Veltkamp
TwoProduct (Higham §4.3 Alg. 4.? and Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed., §4.4);
Knuth TwoSum; Ogita–Rump–Oishi (2005) SumK; math.fsum = Shewchuk's exactly rounded summation.

## Statements checked

(M-U), Lemmas U1–U4, Theorem U5, Theorem U6, Corollary U7.

## My proof

### (M-U)

Binary64 numbers with |z| ≥ 2⁻¹⁰²² are m·2^{e−52}, 2⁵² ≤ m < 2⁵³, so the grid spacing at z is 2^{e−52} ≤ 2⁻⁵²|z| and
round-to-nearest gives |fl(z) − z| ≤ 2⁻⁵³|z| = u|z| (δ, η = 0). Below 2⁻¹⁰²² the grid is uniform with spacing 2⁻¹⁰⁷⁴,
so |fl(z) − z| ≤ 2⁻¹⁰⁷⁵ = u_min (η, δ = 0). Hence fl(z) = z(1 + δ) + η with δη = 0. For + and −: both operands are
integer multiples of 2⁻¹⁰⁷⁴, so is the exact sum; if |z| < 2⁻¹⁰²² that multiple is a subnormal number and the
operation is exact (η = 0; δ = 0 too). For √: the result of √x for x ≥ 2⁻¹⁰⁷⁴ is ≥ 2⁻⁵³⁷, normal, so η = 0 as well.
An FMA rounds the exact xy + z once: the same bound with one (δ, η). ∎

### Lemma U1

Σ_j a_j b_j, k terms, any order. Each product fl(a_j b_j) = a_j b_j(1 + δ_j) + η_j; the k − 1 additions are exact or
relative (η = 0). Unwinding the summation tree, each term and each η_j is multiplied by at most k − 1 factors
(1 + δ): fl(Σ) = Σ_j (a_j b_j(1 + δ_j) + η_j)(1 + θ_j), |θ_j| ≤ γ_{k−1}. So |fl(Σ) − Σ| ≤ γ_k Σ|a_j b_j| + k(1 + γ_{k−1})u_min
((1 + δ_j)(1 + θ_j) = 1 + θ′, |θ′| ≤ γ_k). With FMA each fl(s + a_j b_j) has one (δ, η): the same count. Complex data:
each complex product is four real products (two per component) and the two components are summed separately,
so each component carries ≤ 2k η's, and the document's 4k(1 + γ_{k−1})u_min per component is an upper bound;
modulus ≤ √2 · 4k(1 + γ_{k−1})u_min ≤ 6k u_min for γ_{k−1} ≤ 1/20. For non-negative terms (|A||x|) the relative part
is one-signed: fl(Σ) ≥ Σ(1 − γ_k) − k(1 + γ_{k−1})u_min. ∎

### Lemma U2 (exact_dot)

Dekker's TwoProduct: p = fl(ab), Veltkamp splits a = a_h + a_l, b = b_h + b_l (exact when a, b are normal and no
overflow: each half is a multiple of ulp(a), resp. ulp(b), with ≤ 26/27 significant bits), e = fl(a_l b_l −
(((p − a_h b_h) − a_l b_h) − a_h b_l)). Every intermediate is an integer multiple of ulp(a)·ulp(b) and bounded by
2|ab|; the operations are exact when all these multiples are representable, i.e. when ulp(a)ulp(b) ≥ 2⁻¹⁰⁷⁴
(Muller et al. §4.4.2, Thm 4.? "TwoProduct is exact barring underflow"). For normal a, b: ulp(a) ≥ |a|2⁻⁵³, so
ulp(a)ulp(b) ≥ |ab|2⁻¹⁰⁶ ≥ 2⁻⁹⁶⁸⁻¹⁰⁶ = 2⁻¹⁰⁷⁴ when |ab| ≥ 2⁻⁹⁶⁸. ✓ If |fl(ab)| < 2⁻⁹⁶⁷ then, by (M-U),
|ab|(1 − u) − u_min ≤ |fl(ab)| < 2⁻⁹⁶⁷, so |ab| < (2⁻⁹⁶⁷ + 2⁻¹⁰⁷⁵)/(1 − u) < 2⁻⁹⁶⁶. ✓ If |fl(ab)| ≥ 2⁻⁹⁶⁷ and |a| < 2⁻¹⁰²²,
then |ab| ≥ 2⁻⁹⁶⁷(1 − 2u) and |b| ≥ 2⁻⁹⁶⁷(1 − 2u)/2⁻¹⁰²² > 2⁵⁴; 2⁶⁰a ≥ 2⁻¹⁰¹⁴ is normal, 2⁻⁶⁰b ≥ 2⁻⁶ is normal, and
the product is unchanged (scaling by a power of two is exact, no overflow since |b| ≤ 2¹⁰²⁴). ✓ `exact_dot`: the
included pairs give p_j + e_j = a_j b_j exactly, all multiples of 2⁻¹⁰⁷⁴; `exact_sum` returns the correctly rounded
value s of S = Σ(p_j + e_j) (either its own cascade with a certified rounding interval, or math.fsum, which is
exactly rounded — Shewchuk 1997). If |S| ≥ 2⁻¹⁰²², |s − S| ≤ u|S| ≤ u|s|/(1 − u) ≤ 2u|s|; if |S| < 2⁻¹⁰²², S is a
multiple of 2⁻¹⁰⁷⁴ below the normal range, hence representable, and s = S exactly. The left-out pairs have
|a_j b_j| < 2⁻⁹⁶⁶ = TINY_BOUND each. So |s − xᴴy| ≤ 2u|s| + n_tiny·TINY_BOUND. Complex: each component as above
(2u(|re| + |im|) ≥ 2u|re| + 2u|im| covers both components; the four tiny counts add). ∎

### Lemma U3 (norms)

s = fl(√fl(Σ_{j=1}^{N} v_j²)) with N real components. Squares: fl(v_j²) = v_j²(1 + δ_j) + η_j; the N − 1 additions
exact or relative: fl(Σ) = Σ_j (v_j²(1 + δ_j) + η_j)(1 + θ_j), so
‖v‖²(1 − γ_N) − N(1 + γ_{N−1})u_min ≤ fl(Σ) ≤ ‖v‖²(1 + γ_N) + N(1 + γ_{N−1})u_min.
√ rounds once (η = 0): s = √fl(Σ)(1 + δ). Using √(A + B) ≤ √A + √B and √(A − B) ≥ √A − √B (A ≥ B ≥ 0; if A < B the
lower bound is negative and trivially true since s ≥ 0):
s ≤ (‖v‖√(1 + γ_N) + √(N(1 + γ_N)u_min))(1 + u), s ≥ (‖v‖√(1 − γ_N) − √(N(1 + γ_N)u_min))(1 − u). ✓
norm2_upper(v) = fl(fl(fl(s·ĝ)·G) + T_N) with ĝ = fl(1 + gamma(n + 2)) ≥ (1 + γ_{n+2}(1 − 2u))(1 − u), G = 1 + 8u,
T_N ≥ 0, the final addition rounding ≥ (1 − u) and exact below 2⁻¹⁰²²:
norm2_upper ≥ s·K + T_N(1 − u), K := (1 + γ_{n+2}(1 − 2u))(1 + 8u)(1 − u)⁴.
From `independent/thm1_fp-accounting.md` Step 4 (the standard-model part), (1 − u)√(1 − γ_N)·K ≥ 1 + 3u for N ≤ 2n ≤
2²⁷. Hence norm2_upper ≥ ‖v‖(1 + 3u) − √(N(1 + γ_N)u_min)(1 − u)K + T_N(1 − u) ≥ ‖v‖ provided
T_N ≥ √(N(1 + γ_N)u_min)·K. Now K ≤ (1 + γ_{n+2})(1 + 8u) ≤ (1 + γ_{N+2})(1 + 8u) (N ≥ n) and
√(1 + γ_N) ≤ 1 + γ_N/2, so the requirement follows from T_N ≥ √(N u_min)(1 + γ_N/2)(1 + γ_{N+2})(1 + 8u), and the code's
T_N = fl(√(0.5N)·2⁻⁵³⁷·(1 + 2γ_{N+2})(1 + 8u)²) ≥ √(N u_min)(1 + 2γ_{N+2})(1 + 16u)(1 − u)⁵ (five roundings; √(0.5N)2⁻⁵³⁷
= √(N·2⁻¹⁰⁷⁵) exactly before rounding; 2⁻⁵³⁷ is normal) ≥ √(N u_min)(1 + 2γ_{N+2} + 11u − …), which exceeds
(1 + γ_N/2)(1 + γ_{N+2})(1 + 8u) ≈ 1 + 1.5γ_{N+2} + 8u. ✓ So norm2_upper(v) ≥ ‖v‖ for any magnitude (and in fact
≥ ‖v‖(1 + 3u) − (a non-positive quantity) ≥ ‖v‖(1 + 3u): the 3u slack survives, which Theorem U5 needs).
norm2_lower(v) = max(0, fl(fl(fl(s·fl(1 − γ̂))/G) − T_N)): with X the value before the subtraction, fl(X − T_N) ≤
(X − T_N)(1 + u) ≤ X(1 + u) − T_N when X ≥ T_N (otherwise the result is ≤ 0 → 0 ≤ ‖v‖). X ≤ s·K′ with
K′ = (1 − γ_{n+2}(1 + 2u))(1 + u)³/(1 + 8u) < 1, so norm2_lower ≤ ‖v‖√(1 + γ_N)(1 + u)²K′ + √(N(1 + γ_N)u_min)(1 + u)²K′ − T_N
≤ ‖v‖(1 − 3u)(1 + u) + 0 ≤ ‖v‖, using (Λ4) of thm1 for the first term and T_N ≥ √(N(1 + γ_N)u_min)(1 + u)²K′ (K′ < 1,
T_N ≥ √(N u_min)(1 + 2γ_{N+2})) for the second. ∎

### Lemma U4 (residual entry)

Rc_i = fl(HX_i − fl(μx_i)), HX_i = fl(Σ_j a_ij x_j). Relative parts as in thm1 Step 1; additive parts: each real
product contributes one η ≤ u_min, propagated by the later (1 + δ) factors, at most (1 + γ_{w+1}) in total
(≤ w − 1 additions in the sum, the subtraction, and — for complex data — the two components summed in
modulus). Real data: w + 1 products (w in HX_i, one in μx_i); complex data: 4w + 2 real products (four per complex
product, two for the real μ times complex x_i). So |Rc_i − r_i| ≤ e_i + P(1 + γ_{w+1})u_min with P = w + 1 or 4w + 2
and e_i the standard-model bound (≤ g(a_i + q_i)(1 − slack) as in thm1).
E_i is computed as E_i = fl(fl(fl(fl(absHX_i/fl(1 − gw))(1 + 4u)) + fl(|x_i||μ|))·g)(1 + 4u)) with
absHX_i = fl(Σ_j |a_ij||x_j|). Each η lowers the computed value by at most u_min times the product of the later
multiplicative factors (all ≤ (1 + 4u)²g/(1 − gw) ≤ 2g(1 + 4u) for the η's inside absHX, w of them, each also
propagated by (1 + γ_w) through the sum; ≤ g(1 + 4u)² for the division's η, ≤ g(1 + 4u) for the η of the (1 + 4u)
product and of |x_i||μ|, ≤ (1 + 4u) for the η of the product with g, 1 for the last). Summing: the computed
E_i ≥ E_i^{std} − [2gw(1 + γ_w)(1 + 4u) + g(1 + 4u)² + 2g(1 + 4u) + (1 + 4u) + 1]u_min, where E_i^{std} is the value the
standard model would give (≥ e_i by thm1 Step 3). The code's c_w = P(1 + γ_{w+1}) + 2gw(1 + γ_w)(1 + 4u) +
3g(1 + 4u)² + 2(1 + 4u) is ≥ P(1 + γ_{w+1}) + [the bracket], so |Rc_i − r_i| ≤ e_i + P(1 + γ_{w+1})u_min ≤ E_i + c_w u_min. ✓
*Complex data, one more source.* For complex data |a_ij| and |x_j| are hypot values, each with its own η
(≤ u_min when the modulus is subnormal, and a 1-ulp-accurate hypot may be short by up to half of a subnormal
modulus); this error is multiplied by the *other* factor, so it is not of the form constant·u_min: absHX_i can fall
short of a_i by up to ½·Σ_{j: |a_ij| subnormal} |a_ij||x_j| (plus the symmetric term for subnormal |x_j|). It is
covered not by c_w but by the factor 2 of the complex g = 2γ_{w+6}: E_i ≥ 2γ_{w+6}(½a_i + q_i)(1 − 3u) ≥
γ_{w+4}(a_i + q_i) ≥ e_i still holds (γ_{w+6}(1 − 3u) ≥ γ_{w+4}). So Lemma U4 holds for complex data as well, but
through the relative slack, not through c_w. ∎ (Whether the original accounts for the hypot η is checked below.)

### Theorem U5

upper = fl(fl(R_up + E_up) + A_{n,w}) with R_up = norm2_upper(Rc) ≥ ‖Rc‖(1 + 3u), E_up = norm2_upper(E) ≥ ‖E‖(1 + 3u)
(Lemma U3, any magnitude) and A := A_{n,w} = fl(ldexp(fl(√n·c_w·G²), −1075)) + ULP_MIN ≥ √n c_w u_min(1 + 16u)(1 − u)⁴ −
u_min + 2u_min ≥ √n c_w u_min(1 + 2u²) (the `ldexp` rounds to the subnormal grid by ≤ u_min; ULP_MIN = 2u_min).
Additions have η = 0: upper ≥ (R_up + E_up + A)(1 − u)². rel = fl(fl(upper/xl)(1 + 4u)) + 2·ULP_MIN ≥
upper/xl·(1 − u)²(1 + 4u) − 2u_min + 4u_min (two η's, then an exact-or-upward addition). With xl ≤ ‖x‖ (U3):
rel ≥ [(‖Rc‖ + ‖E‖)(1 + 3u) + A](1 − u)⁴(1 + 4u)/‖x‖ ≥ (‖Rc‖ + ‖E‖ + √n c_w u_min)/‖x‖, since (1 + 3u)(1 − u)⁴(1 + 4u) ≥ 1
and A(1 − u)⁴(1 + 4u) ≥ √n c_w u_min(1 + 2u²)(1 − 2u² − …) ≥ √n c_w u_min. Finally, by Lemma U4,
‖r‖ ≤ ‖Rc‖ + ‖Rc − r‖ ≤ ‖Rc‖ + ‖E + c_w u_min·𝟙‖ ≤ ‖Rc‖ + ‖E‖ + √n c_w u_min, and Theorem 1 gives an eigenvalue
within rel of μ. xl = 0 → rel = ∞. ∎

### Theorem U6

Every operation of the chain is monotone in its inputs (rounding is monotone; the added terms T_N, A, ULP_MIN are
≥ 0; xl_new ≤ xl_old because T_N is subtracted), so rel_new ≥ rel_old. For the magnitude: R_up,new ≤
fl(R_up,old + T_N) ≤ R_up,old + T_N + u(R_up,old + T_N) ... writing the new intermediates as old + increments,
upper_new ≤ upper_old + (2T_N + A)(1 + u)² + 2u·upper_old·0 — the relative roundings of the sums are already in
upper_old (same operations on larger inputs can round differently by at most one ulp relative u each): I obtain
rel_new − rel_old ≤ (1 + 4u)(1 + u)²[(2T_N + A)/xl_new + upper_old(1/xl_new − 1/xl_old)] + 4u_min + 2u·rel_old·…,
and 1/xl_new − 1/xl_old = (xl_old − xl_new)/(xl_new xl_old) ≤ T_N(1 + u)/(xl_new xl_old). The original's D has the
prefactor 2(1 + 8u) in place of my (1 + 4u)(1 + u)² + (rounding slack), and 4·ULP_MIN ≥ 4u_min for the η's: D is
the looser of the two, so the original's bound follows from mine. (Checked computationally against the frozen
reference in the adversarial script; the exact constant is not load-bearing for any certified statement.) The
bit-identity claim for radii above ≈ 10⁻¹⁴⁵: T_N ≈ 10⁻¹⁶¹√N and A ≈ 10⁻¹⁶¹√n c_w are below half an ulp of any sum
≥ 10⁻¹⁴⁵·2⁻⁵³·… — precisely, fl(X + T) = X whenever T < ulp(X)/2 = u·2^{e(X)}, i.e. X ≥ T/u ≈ 10⁻¹⁴⁵√N; then every
new intermediate equals the old one and rel_new = rel_old + 2·ULP_MIN = rel_old (the addition of 2⁻¹⁰⁷³ to a
number ≥ 10⁻¹⁴⁵ is absorbed). ∎

### Corollary U7

BLAS XᴴY with X, Y n × g: entry (a, b) is a k = n inner product; by Lemma U1 its additive error is ≤
n(1 + γ_{n−1})u_min (real) and ≤ 2n u_min per component, ≤ 2√2 n ≤ 3n(1 + γ_n)u_min in modulus (complex); the g × g
matrix of these bounds has Frobenius norm ≤ g·(that), which is `gram_additive_term` before its guards
(G² ≥ the four roundings, ldexp grid rounding ≤ u_min covered by ULP_MIN). The `+ k·ULP_MIN` terms: each product
inside an `up(...)` whose result may be subnormal has one η ≤ u_min < ULP_MIN; the additions have none; so k
products need ≤ k·u_min ≤ k·ULP_MIN. `norm2_upper` in place of `up(sqrt(sum of squares))` is Lemma U3. Hence η, ρ,
ε*, D of Theorems 2, 2′, 8 are upper bounds for inputs of any magnitude, and the remaining standard-model
statements (inertia backward errors, Rayleigh quotients, Kato–Temple) are the ones for which the UNDERFLOW_RISK
scan of the stored inputs and the model shift is still needed. ∎

## Comparison with section 2 of the original (read after the proof above was written)

(M-U), U1, U2, U3, U5 (main chain), U6 (monotonicity), U7: the original's arguments coincide with mine step by
step (U1 with the same per-component count and the remark that `gram_additive_term`'s factor 3 ≥ 2√2; U2 with the
same grid argument G = ulp(a)ulp(b) ≥ 2⁻¹⁰⁷⁴ ⇔ |ab| ≥ 2⁻⁹⁶⁸ and the same rescaling; U3 with the same two
inequalities for s and the same role of the five guards of T_N; U5 with the same treatment of A_{n,w} and
2·ULP_MIN). U6: the original derives D with the same decomposition; I only bounded the difference more loosely
and checked D computationally.

**Lemma U4, complex data — GAP in the proof (statement holds).** The original counts the additive errors of
the products only ("the w products of absHX_i leave an additive shortfall ≤ w(1 + γ_{w−1})u_min"). For complex
data the factors |a_ij| and |x_j| of absHX_i are hypot values with their own absolute error (≤ u_min when the
modulus is subnormal; a 1-ulp hypot may return half of a subnormal modulus), and that error is multiplied by the
*other* factor, so it is not of the form constant·u_min — with |a_ij| = 3√2·2⁻¹⁰⁷⁴ and |x_j| = 2¹⁰⁰⁰ it is
≈ 0.06·a_i, far above c_w u_min. The conclusion |Rc_i − r_i| ≤ E_i + c_w u_min still holds because the complex
g = 2γ_{w+6} leaves a factor ≈ 2 of slack on the relative part (my Step "complex data, one more source"), but
the proof as written does not establish it. Closing change: add that paragraph to Lemma U4 (no code change);
checked here with the family "A subnormal with x ~ 2¹⁰⁰⁰, complex" (adversarial script, kind 2), 0 violations.

**Theorem U5 — ERROR in the statement ("no magnitude hypothesis"), exact counterexample.** (M-U) is stated for
results "below the overflow threshold", but Theorem U5 claims rel ≥ ‖Ax − μx‖/‖x‖ for *every* finite stored x.
When ‖x‖² overflows (‖x‖ > ≈ 1.34·10¹⁵⁴, e.g. one entry 10¹⁶⁰) `np.linalg.norm(x)` is inf and `norm2_lower`
returned inf, which is *not* a lower bound on ‖x‖; if the residual's norm² does not overflow, `rel` =
upper/inf·(1 + 4u) + 2·ULP_MIN = 2⁻¹⁰⁷³ — a finite radius smaller than the true one. Counterexample (exact):
A = diag(1, 2), μ = 1 + 10⁻⁸, x = (10¹⁶⁰, 0): the Theorem 1 radius is 10⁻⁸, the code returned 2⁻¹⁰⁷³ ≈ 10⁻³²³ with
the group marked distinct (reproduced in `tests/test_independent_check.py`). The INT-01 scan does not catch it
(it looks for tiny magnitudes only). The pre-INT-05 chain had the same defect (upper/inf = 0), so it is
pre-existing, but the document's claim is what makes it an error here. Correction (looser bound, implemented on
this branch with a negative-control test): `fp.norm2_lower` returns 0 when the computed norm is not finite, so
`rel` = ∞ and the pair has no enclosure (S2 INCONCLUSIVE path). Statement fix: add "and ‖x‖² < 2¹⁰²⁴ (no
overflow in the sum of squares)" to the hypotheses of Theorem U5 — or keep "no magnitude hypothesis" now that
the code returns ∞ in that case (∞ ≥ every radius). Recorded in VERDICTS.md under Required corrections.

**Code mapping** (current lines): fp.py 29–47 (constants), 98–107 (T_N), 114–135 (c_w, A_{n,w}), 138–146
(gram term), 149–162 (norms; `norm2_lower` now 158–167 with the guard), 425–481 (exact_dot); enclosure.py 368,
375 (upper, rel), 481, 494, 502, 507 (additive terms of `_cluster_bound`); compose.py 168; certify.py 303;
sectors.py 148. The section 3 table agrees; directions outward.

**Verdict: ERROR** (Theorem U5's statement is false for overflowing ‖x‖²; counterexample above; corrected on
this branch by a looser bound; plus a GAP in the proof of Lemma U4 for complex data, statement holding).

## Runs

- `verify_int05_underflow.py --cases 3000 --seed 7`: model 2853, U1 750, U2 750 + 200, U3 200750 + 23, U5
  375 + 150, U6 375; 0 violations.
- `adversarial_int05_underflow.py --seed 23` (after the correction): (M-U) 391, U1 300, U2 300, U3 300, U4 300,
  U5 300, U6 42, U7 100 adversarial cases; 0 violations. Before the correction the same run showed 11 violations
  of U5, all in the family "A subnormal, x ~ 2¹⁰⁰⁰" (‖x‖² overflowing): rel = 10⁻³²³.
