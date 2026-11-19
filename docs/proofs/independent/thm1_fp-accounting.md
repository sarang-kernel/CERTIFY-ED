# Independent re-derivation: Theorem 1-fl (floating-point evaluation of the residual radius)

Written on 2026-10-30 before the original proof was read (not a human check). Sections 1 and 3 of `../thm1_fp-accounting.md` read first; proof written before
section 2 was opened. (`docs/theory.md` §2, read for the notation, carries the code-facing list from which the
original document was written; the derivations below are mine.) Notation as in the original: r = Ax − μx exact,
a_i = (|A||x|)_i, q_i = |μx_i|, Rc the computed residual, E the computed entrywise bound, w = `adapter.width`,
g = `gamma_dot(w + 2)` (γ_{w+2} real, 2γ_{w+6} complex), standard model fl(z) = z(1 + δ), |δ| ≤ u, additive
underflow terms set to zero. Higham (2002): Lemma 3.1, Lemma 3.3, eq. (3.4)–(3.5) (inner products, any order,
with or without FMA), Lemma 3.5 (complex arithmetic: fl(x ± y) relative error ≤ u, fl(xy) relative error ≤ √2γ₂).

## Statement checked

rel ≥ ‖Ax − μx‖/‖x‖ for the stored numbers, with (S2) absHX_up_i ≥ a_i, (S3) E_i ≥ |Rc_i − r_i|, (S4)
norm2_upper(v) ≥ ‖v‖(1 + 3u), norm2_lower(v) ≤ ‖v‖(1 − 3u) for 1 ≤ n ≤ 2²⁶, (S5) upper ≥ ‖Rc‖ + ‖E‖ and
rel ≥ upper/‖x‖. Hypotheses n, w ≤ 2²⁶ (so nu, wu ≤ 2⁻²⁷ and second-order terms are ≤ u/2 relative to first-order).

## My proof

Throughout, "k roundings covered by (1 + cu)" means (1 − u)^k(1 + cu) ≥ 1; (1 − u)³(1 + 4u) = 1 + u − 6u² + … ≥ 1 and
(1 − u)⁴(1 + 4u) = 1 − 2u² + … < 1 (four roundings are *not* covered by (1 + 4u) alone).

### Step 1: entrywise error of the computed residual

HX_i = fl(Σ_j A_ij x_j), a sum of at most w products. Real data: |HX_i − (Ax)_i| ≤ γ_w a_i (Higham (3.5), any
order, with FMA). Complex data: each product carries |θ| ≤ √2γ₂ ≤ γ₃ and the w − 1 additions γ_{w−1}, so
|HX_i − (Ax)_i| ≤ γ_{w+2}a_i (Lemma 3.3). fl(μx_i) = μx_i(1 + δ), |δ| ≤ u (μ real: each component rounds once).
Rc_i = (HX_i − fl(μx_i))(1 + δ′), |δ′| ≤ u (complex subtraction rounds each component once; modulus error ≤ u).
Hence Rc_i − r_i = (HX_i − (Ax)_i − (fl(μx_i) − μx_i))(1 + δ′) + r_iδ′ and, with |r_i| ≤ a_i + q_i,
e_i := |Rc_i − r_i| ≤ (γ_w a_i + u q_i)(1 + u) + u(a_i + q_i) ≤ γ_{w+1}a_i + γ₂q_i ≤ γ_{w+2}(a_i + q_i) (real);
≤ (γ_{w+2}(1 + u) + u)a_i + γ₂q_i ≤ γ_{w+3}(a_i + q_i) ≤ 2γ_{w+6}(a_i + q_i) (complex). So e_i ≤ g(a_i + q_i),
with a relative slack of at least (γ_{w+2} − γ_{w+1})a_i ≥ u a_i and (γ_{w+2} − γ₂)q_i ≥ w u q_i in the real case
(≈ 1/(w + 2) of g(a_i + q_i)), and of about a factor 2 in the complex case.

### Step 2: absHX_up ≥ a_i

absHX_i = fl(Σ_j |A_ij| |x_j|) over at most w terms. Real data: |A_ij|, |x_j| exact, the w products and w − 1
additions give absHX_i ≥ a_i(1 − γ_w). Complex data: |A_ij| and |x_j| are each one rounding (hypot; error ≤ u
relative), so absHX_i ≥ a_i(1 − γ_{w+2}). The code forms `absHX / fl(1 − gw) * (1 + 4u)` with gw = `gamma(w + 1)`,
whose two internal roundings give gw ∈ γ_{w+1}[1 − 2u − 4u², 1 + 2u + 4u²]; fl(1 − gw) ≤ (1 − gw)(1 + u); the
division and the product round once each. Hence
absHX_up_i ≥ a_i (1 − γ_w)(1 − u)³(1 + 4u)/(1 − γ_{w+1}(1 − 2u − 4u²)).
Real data: (1 − γ_w)/(1 − γ_{w+1}(1 − 3u)) ≥ 1 + (γ_{w+1} − γ_w − 3uγ_{w+1}) ≥ 1 + u(1 − 3γ_{w+1}) ≥ 1 + u/2, so
absHX_up_i ≥ a_i(1 + u/2)(1 + u − 6u²) ≥ a_i. ✓
Complex data: (1 − γ_{w+2})/(1 − γ_{w+1}(1 − 3u)) ≤ 1 − (γ_{w+2} − γ_{w+1}) + 3uγ_{w+1} + O(u²) ≈ 1 − u − (2w + 3)u² +
3(w + 1)u² ≈ 1 − u + wu², and (1 − u)³(1 + 4u) = 1 + u − 6u² + O(u³), so the product is 1 − (6 − w)u² + O(u³) and
**the lower bound can fall below 1 by O(wu²)**. The route above therefore does not establish (S2) for complex
data; whether (S2) actually fails needs a worst case in which the hypot and product roundings all go the same
way (searched for in `adversarial_thm1_fp-accounting.py`). This does not affect the radius (Step 3 has a factor
≈ 2 of slack in the complex case), but (S2) as stated is a claim about an intermediate quantity.

### Step 3: E_i ≥ e_i

E_i = fl(fl(fl(absHX_up_i + fl(|x_i||μ|)) · g_c) · (1 + 4u)) with g_c = `gamma_dot(w + 2)` ≥ g(1 − 2u − 4u²) (two
internal roundings; the factor 2 of the complex case is exact). Complex |x_i| adds one rounding. So
E_i ≥ g(1 − 2u − 4u²)(absHX_up_i + q_i)(1 − u)^k(1 + 4u), k = 4 (real) or 5 (complex), i.e.
E_i ≥ g(a_i + q_i)(1 − 2u)(1 − 4u)(1 + 4u) ≥ g(a_i + q_i)(1 − 2u − 16u²) (real), ≥ g(a_i + q_i)(1 − 3u − …)
(complex; here absHX_up_i ≥ a_i(1 − 7u²) from Step 2 is enough). Against Step 1: real,
g(1 − 2u − 16u²)(a_i + q_i) − e_i ≥ (γ_{w+2} − γ_{w+1} − 2uγ_{w+2} − 16u²)a_i + (γ_{w+2} − γ₂ − 2uγ_{w+2})q_i ≥ 0,
since γ_{w+2} − γ_{w+1} ≥ u ≥ 2uγ_{w+2} + 16u² (γ_{w+2} ≤ 1/4 when (w + 2)u ≤ 1/5) and γ_{w+2} − γ₂ ≥ wu ≥ 2uγ_{w+2}
for w ≥ 1 (for w = 0 there is no residual: A = 0 and e_i = uq_i ≤ γ₂q_i(1 − 3u) still). Complex: g(1 − 3u) =
2γ_{w+6}(1 − 3u) ≥ γ_{w+3} trivially. So E_i ≥ e_i. ✓ (S3)

### Step 4: the norm guards

s = fl(‖v‖) = fl(√fl(Σ v_j²)) (numpy: a BLAS dot for real v; for complex v two real dots of the components and one
addition, 2n squares). The sum of N = n (real) or 2n (complex) squares: fl(Σ) = Σ(1 + θ), |θ| ≤ γ_N (Higham (3.5))
— for complex, γ_{n} per dot and one more addition, |θ| ≤ γ_{n+1}. The square root rounds once. So
‖v‖(1 − γ_{n+1})^{1/2}(1 − u) ≤ s ≤ ‖v‖(1 + γ_{n+1})^{1/2}(1 + u).
norm2_upper(v) = fl(fl(fl(s · fl(1 + γ_c)) · (1 + 8u))) + T ≥ s(1 + γ_{n+2}(1 − 2u))(1 + 8u)(1 − u)³
≥ ‖v‖(1 − γ_{n+1}/2)(1 − u)(1 + γ_{n+2} − 2uγ_{n+2})(1 + 5u − 3u²) ≥ ‖v‖(1 + (γ_{n+2} − γ_{n+1}/2 − u) + 5u − O(nu²))
≥ ‖v‖(1 + 3u + (n/2 + 1)u − O(n²u²)) ≥ ‖v‖(1 + 3u) for n ≤ 2²⁶ (n²u² ≤ u/2). ✓
norm2_lower(v) = max(0, fl(fl(fl(s · fl(1 − γ_c)) / (1 + 8u))) − T) ≤ s(1 − γ_{n+2}(1 + 2u))(1 + u)³/(1 + 8u)
≤ ‖v‖(1 + γ_{n+1}/2 + u)(1 − γ_{n+2})(1 − 5u + …) ≤ ‖v‖(1 − (γ_{n+2} − γ_{n+1}/2) − 4u + O(nu²)) ≤ ‖v‖(1 − 3u). ✓ (S4)
(√(1 ± γ) ≤ 1 ± γ/2 upper / ≥ 1 − γ/2 − γ²/8 lower; the γ² terms are O(n²u²).)

### Step 5: upper and rel

upper = fl(fl(R_up + E_up) + A_{n,w}) ≥ (R_up + E_up)(1 − u)² (A ≥ 0; set to 0 here); by Step 4 R_up ≥ ‖Rc‖(1 + 3u),
E_up ≥ ‖E‖(1 + 3u). rel = fl(fl(upper/xl) · (1 + 4u)) + 2·ULP_MIN ≥ upper/xl · (1 − u)²(1 + 4u). With xl ≤ ‖x‖
(Step 4, xl > 0): rel ≥ (‖Rc‖ + ‖E‖)/‖x‖ · (1 + 3u)(1 − u)⁴(1 + 4u) ≥ (‖Rc‖ + ‖E‖)/‖x‖ (since
(1 + 3u)(1 − 2u²) ≥ 1). Finally ‖r‖ ≤ ‖Rc‖ + ‖Rc − r‖ ≤ ‖Rc‖ + ‖E‖ (Step 3, entrywise |Rc_i − r_i| ≤ E_i, so
‖Rc − r‖ ≤ ‖E‖), and Theorem 1 (exact: expand x in an eigenbasis, ‖r‖² = Σ|c_k|²(λ_k − μ)² ≥ min_k(λ_k − μ)²‖x‖²)
gives an eigenvalue within rel of μ. xl = 0 gives rel = ∞ (`np.where`). ✓ (S5)

### Remark 3 (spec operator)

For the matrix-free spec operator each entry of H x is a sum of at most m = n_terms·2^{L_max} monomial
contributions, each a product of the coupling, ≤ L_max local amplitudes (one rounding each where irrational) and
x_j — at most 2L_max + 1 roundings per contribution and m − 1 in the sum: the inner-product bound γ_w holds with
w = m + 2L_max + 2 (two spare units), which is what `OperatorAdapter` sets (enclosure.py 112); `abs_matvec`
sums the |t̂| |x_j| (construct.py 263–265), which is what Step 2 needs.

## Comparison with section 2 of the original (read after the proof above was written)

Step 1: identical (same decomposition, same constants e_i ≤ g(a_i + q_i); the original's complex bound is
γ_{w+4}, mine γ_{w+3}; both ≤ 2γ_{w+6}). Step 3: identical ((Λ6) is my inequality with the roundings counted as
six including `gamma_dot`'s two; same use of the Step 1 slack for real data and of the factor 2 for complex
data). Step 4: identical ((Λ4), N ∈ {n, 2n}; the original also pins the inequalities exactly on a grid in the
verification script). Step 5: identical (four roundings, (1 + 4u) covers three, the 3u slack of Step 4 pays the
fourth). Remarks (3), (4) as mine.

**Step 2 — GAP for complex data.** The original writes: "the moduli |a_ij|, |x_j| of complex data are themselves
rounded once each, a relative u, absorbed below in the complex g", and then concludes absHX_up_i ≥ a_i(1 + u)
"by (Λ5)". (Λ5) is proved from absHX_i ≥ a_i(1 − γ_w), which holds for real data only; for complex data the
hypot roundings give absHX_i ≥ a_i(1 − γ_{w+2}), and (1 − γ_{w+2})(1 − u)³(1 + 4u)/(1 − γ_{w+1}(1 − u)²) is
1 − O(wu²), not ≥ 1 (my Step 2). The statement (S2) is therefore *not established* for complex data, and the
sentence "absorbed below in the complex g" is a correct remark about Step 3 (where the factor 2 of
g = 2γ_{w+6} covers everything) but does not justify (S2) itself. The certified radius is unaffected: (S3) and
(S5) hold for complex data through the Step 3 slack, as the original's own Step 3 argument shows. Whether (S2)
is *false* for complex data I could not decide: the worst case needs six roundings to go the same way, and the
relative rounding error of each product/quotient is bounded by u divided by its mantissa, so the first-order
budget (w + 5)u against (w + 5)u is tight only for mantissas near 1, where the division and multiplication
round almost exactly; a targeted search (150 tuned cases, moduli just below rounding midpoints, mantissas
1.0–2.0) found the smallest margin positive. Closing the gap: either (a) restate (S2) for real data and, for
complex data, state only absHX_up_i ≥ a_i(1 − γ_{w+2})(1 − u)³(1 + 4u)/(1 − γ_{w+1}(1 − u)²) ≥ a_i(1 − 8wu²),
which is what Step 3 needs (no code change), or (b) use `gw = gamma(w + 3)` for complex data in
`_rounding_bound` (a looser bound, allowed without a proof change; it restores (S2) with margin ≥ u). Recorded in
VERDICTS.md under Required corrections (statement-level; no certified bound affected).

**Remark (outside the hypotheses).** When ‖x‖² overflows (‖x‖ > ≈10¹⁵⁴), `np.linalg.norm` returns inf,
`norm2_lower` returns inf and `rel` = inf/inf = NaN; `enclose` marks the group non-distinct and `certify()`
treats a non-finite radius fail-safe (IMP-22, certify.py 1054–1059). No finite too-small bound is produced;
worth a sentence in the document's hypotheses ("every quantity the code forms is a normal number" excludes it).

**Code mapping (current `dev/3.2`).** Step 1: enclosure.py 343, 346; Step 2–3: `_rounding_bound` 286–302 (gw 291,
division 295, (1 + 4u) 296, |x||μ| 297–298, addition 299, g 300, (1 + 4u) 301); Step 4: fp.py 149–162, used at
351–352, 356; Step 5: 368, 375; constants fp.py 29–30, 50–66, 93–95; spec width enclosure.py 112. All as the
table of section 3 states; rounding directions outward.

**Verdict: GAP** (sub-claim (S2) for complex data not justified; the main statement rel ≥ ‖Ax − μx‖/‖x‖ and
(S3)–(S5) hold; closing change given above).

## Runs

- `verify_thm1_fp-accounting.py --cases 2000 --seed 7`: scalar lemmas 20000 grid points + 12 adversarial,
  rel ≥ exact radius 2000 random + 246 adversarial, stage inequalities 4000 + 480; 0 violations.
- `adversarial_thm1_fp-accounting.py --cases 600 --seed 19`: (S2) real 50 random + 350 adversarial (50 subnormal-x
  cases skipped: outside the standard model, INT-05's domain), (S2) complex 150 tuned cases (no violation found),
  (S3) 50 + 500 (50 skipped), (S4) 100 + 1075, (S5) + Theorem 1 50 + 475 (includes the subnormal-x cases, where
  the INT-05 terms carry the bound), overflow controls 100 (inf/NaN, never a finite too-small bound);
  0 violations.
