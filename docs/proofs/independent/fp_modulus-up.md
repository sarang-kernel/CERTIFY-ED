# Independent re-derivation: fp_modulus-up (Lemma M, upper bounds on complex moduli, and its corollary)

Fresh-context session, 2026-11-18; the third leg of rule D-12 (a derivation, a machine check, an independent
re-derivation from the statement alone). Written from section 1 of `docs/proofs/fp_modulus-up.md` as it stood on
2026-11-18 (statement-only extract, SHA-256 d79c905c...f6ed) and the code excerpt of `certify_ed/fp.py` at branch
`fix/complex-moduli` (`modulus_up`, `_modulus_up_scaled`, `abs_up`, `_GUARD`; SHA-256 915a39cc...5ead). The author's
proof (section 2 there) was not read and not asked for. The script of section 5 is
`docs/proofs/independent/adversarial_fp_modulus-up.py`.
**UNVERIFIED (D-12):** this document is the independent leg only. It is not checked by Sarang, and nothing below says
it is.

Notation: as in the statement, u = 2⁻⁵³, γ_k = ku/(1 − ku), Ω the largest binary64 number. In addition T := 2⁻¹⁰⁷⁴,
the spacing of the subnormal grid, and fl(t) is the binary64 number nearest to the real number t, ties to even. For
z = x′ + iy′ the code works with x := |x′| and y := |y′| (exact), M := max(x, y), N := min(x, y); |z| = √(M² + N²).
δ with an index is a relative rounding error, |δ| ≤ u.

## 1 Result

**Verdicts.** AGREE = I reach the same statement by my own proof and the exact check finds no violation. GAP = true
in substance, but the statement needs a hypothesis or a restriction it does not state. DISAGREE = false. No claim is
filed as DISAGREE. The inequality of the corollary is false as literally written when a product underflows (or when
an infinite bound meets a zero one); I file it under GAP because the missing hypothesis is the usual one of such
rounding counts and because what the corollary is there for, no count for the moduli, is right.

| Claim | Verdict | Evidence |
|---|---|---|
| Lemma M (i): m ≥ \|z\| | AGREE | Sections 2.2 to 2.6; in fact m ≥ \|z\|(1 + 4u) whenever both components are nonzero. 0 violations of (i) in 1,009,411 exact cases, and 0 of the stronger bound in 963,014. |
| Lemma M (ii): m ≤ \|z\|(1 + 12u) + 2⁻¹⁰⁷³ for finite m; no additive term when \|z\| ≥ 2⁻¹⁰²² | AGREE | Section 2.6. 12 is the least integer the rounding count gives (the count gives 1 + 11u + 27u² + ...); 2⁻¹⁰⁷³ is right for the ldexp clause of FP-M as stated. 0 violations in 1,009,411 cases (907,125 with \|z\| ≥ 2⁻¹⁰²²). Largest ratio seen 1 + 10.513u. |
| Lemma M (iii): m = ∞ only if \|z\|(1 + 12u) > Ω | AGREE | Section 2.6; in fact only if \|z\|(1 + 12u) ≥ 2¹⁰²⁴. 0 violations in 6,027 infinite results. |
| Lemma M (iv): a zero component gives m = \|z\| | AGREE | Section 2.5. 0 violations in 40,370 cases (+0 and −0), and 272 more in the special-input test. |
| Corollary: fl(Σ_j m_ij m_j) ≥ (1 − γ_w) Σ_j \|a_ij\|\|x_j\|, nothing counted for the moduli | GAP | Section 2.7. True when no product m_ij m_j underflows and all moduli are finite (0 violations in 6,784 sums). As stated, with FP-M only, it fails for a = x = 2⁻⁶⁰⁰ (the product is 0) and gives nan for m_ij = ∞, m_j = 0. Fix: add the hypothesis, or the term − w·2⁻¹⁰⁷⁵. |
| Corollary, last sentence (other expressions built by sums and products) | GAP | Same proof and the same two provisos. |
| FP-M is sufficient for Lemma M | AGREE | Section 3: each hypothesis is used; none is missing for Lemma M itself. Some are stronger than (i) needs. |
| Statement: "for real input np.abs(z), which is exact" | GAP (small) | Exact for floating dtypes. np.abs wraps at the most negative integer, and an object-dtype array of complex numbers also takes this path and is not bounded from above (Section 4). |
| Statement, "Why": the np.abs errors −2.466u, −2.462u, +2.269u and np.abs(2⁻¹⁰⁷⁴(1 + i)) = 2⁻¹⁰⁷⁴ | AGREE | Reproduced on this host: −2.4661u, −2.4617u, +2.2687u, and 1·2⁻¹⁰⁷⁴. |
| Code = "the computation" of the statement | AGREE | Read line by line (Section 4); bit-identical with an exact model written from the statement in 1,009,411 cases. |

**What the statement should add.** (1) In the corollary: "no product underflows and the moduli are finite", or the
additive term. (2) For real input: "floating dtypes". (3) Either a guard in the code or a sentence in the statement
for inputs whose components are not binary64 numbers (object arrays; `np.clongdouble` where it is wider than binary64).

## 2 Derivation

**2.1 Rounding facts.** (F1) If the real t satisfies 2⁻¹⁰²² ≤ t ≤ Ω, then fl(t) = t(1 + δ), |δ| ≤ u. (F2) If
0 ≤ t < 2⁻¹⁰²², then |fl(t) − t| ≤ T/2 = 2⁻¹⁰⁷⁵. (F3) fl is monotone and leaves binary64 numbers unchanged; so
c ≤ t ≤ d with binary64 c, d gives c ≤ fl(t) ≤ d. (F4) A number k·2^j with integers 0 ≤ k < 2⁵³, j ≥ −1074, below
2¹⁰²⁴, is a binary64 number. F1 and F2 are "correctly rounded, to nearest" for one operation; F2 is the place where
gradual underflow enters. The operations |·|, max, the comparisons, negation of the integer exponent and the widening
of binary32 to binary64 are exact.

**2.2 The five roundings.** Let α > 0 be a binary64 number with 2⁻¹⁰⁰⁰ ≤ α² ≤ 2¹⁰⁰⁰, let β be real with 0 ≤ β ≤ α,
ζ := √(α² + β²), and let β̂ be a binary64 number in [0, 2⁵⁰⁰]. Both branches compute

    p = fl(α·α),   q = fl(β̂·β̂),   s = fl(p + q),   r = fl(√s),   g = fl(r·(1 + 8u)),

five roundings; 1 + 8u = 1 + 2⁻⁵⁰ is a binary64 number, so the guard itself is exact. Suppose

    (H_θ)   |q − β²| ≤ uβ² + θα²   for some θ with 0 ≤ θ ≤ 2⁻⁷⁰.

*Claim.* ζ(1 + 4u) ≤ g ≤ ζ(1 + 12u), and p, s, r, g are finite and at least 2⁻¹⁰⁰⁰, 2⁻¹⁰⁰⁰, 2⁻⁵⁰⁰, 2⁻⁵⁰⁰.

*Proof.* Ranges first. By F3, p ∈ [2⁻¹⁰⁰⁰, 2¹⁰⁰⁰] and q ∈ [0, 2¹⁰⁰⁰]; so p + q and, by F3, s lie in [2⁻¹⁰⁰⁰, 2¹⁰⁰¹];
r ∈ [2⁻⁵⁰⁰, 2⁵⁰¹]; r(1 + 8u) < 2⁵⁰². So F1 applies to four of the five roundings: p = α²(1 + δ₁), s = (p + q)(1 + δ₃),
r = √s (1 + δ₄), g = r(1 + 8u)(1 + δ₅). The fifth, q, is covered by (H_θ). Then |p + q − ζ²| ≤ uα² + uβ² + θα² ≤
(u + θ)ζ², so

    ζ²(1 − u − θ)(1 − u) ≤ s ≤ ζ²(1 + u + θ)(1 + u),   hence   ζ(1 − u − θ) ≤ √s ≤ ζ(1 + u + θ),

because (1 − u − θ)² ≤ (1 − u − θ)(1 − u) and (1 + u + θ)(1 + u) ≤ (1 + u + θ)². Two more roundings give

    g ≥ ζ(1 − u − θ)(1 − u)²(1 + 8u) ≥ ζ[(1 − u)³(1 + 8u) − θ(1 + 8u)] = ζ[1 + 5u − 21u² + 23u³ − 8u⁴ − θ(1 + 8u)],
    g ≤ ζ(1 + u + θ)(1 + u)²(1 + 8u) = ζ[1 + 11u + 27u² + 25u³ + 8u⁴ + θ(1 + u)²(1 + 8u)].

For θ ≤ 2⁻⁷⁰ = 2⁻¹⁷u, 21u² + θ(1 + 8u) < u and 27u² + 25u³ + 8u⁴ + 2θ < u. This is the claim. ∎

*Count.* The roundings of p and q share one level (their relative errors are averaged with weights α²/ζ² and β²/ζ²),
s is a second level, and the root halves both: to first order they cost u/2 + u/2. The root and the guard product
cost u each. So g/ζ lies in 1 + 8u ± 3u to first order. The guard therefore has to exceed 3u (plus second-order terms)
for a lower bound, and the upper constant is 8 + 3 = 11 plus second-order terms; 12 is the next integer.

**2.3 The unscaled branch** (2⁻⁵⁰⁰ ≤ M ≤ 2⁵⁰⁰, x, y > 0). Here α = M and β = β̂ = N. If N² ≥ 2⁻¹⁰²², F1 gives
q = N²(1 + δ₂), which is (H_θ) with θ = 0. If N² < 2⁻¹⁰²² (N small or subnormal; the square is subnormal or rounds to
0), F2 gives |q − N²| ≤ 2⁻¹⁰⁷⁵ ≤ 2⁻⁷⁵M², because M² ≥ 2⁻¹⁰⁰⁰: (H_θ) with θ = 2⁻⁷⁵. So

    |z|(1 + 4u) ≤ m ≤ |z|(1 + 12u),   m finite,   |z| ≥ M ≥ 2⁻⁵⁰⁰.

Both ends of the interval are safe, with room. At the low end the argument needs M² in the normal range and
2⁻¹⁰⁷⁵/M² small against u; it still goes through at M = 2⁻⁵¹⁰, with θ = 2⁻⁵⁵ = u/4 in place of 2⁻⁷⁵ (both bounds of 2.2 survive a θ of u/4). At
the high end it needs s < 2¹⁰²⁴, which holds up to M = 2⁵¹¹. The limits 2^±500 leave a factor 2¹⁰ and 2¹¹.

**2.4 The scaled branch** (M < 2⁻⁵⁰⁰ or M > 2⁵⁰⁰, M finite, x, y > 0).

*Scaling.* By the frexp hypothesis M = f·2^e with f ∈ [1/2, 1); e ∈ [−1073, 1024]. a := ldexp(M, −e) equals f, a
binary64 number, so it is exact. Let b := N·2⁻ᵉ ∈ (0, f] (real) and b̂ := ldexp(N, −e). Write N = n·2^t with an integer
n < 2⁵³ and t ≥ −1074. If t − e ≥ −1074, b = n·2^(t−e) is a binary64 number (F4) and b̂ = b; this is always the case
when M < 2⁻⁵⁰⁰, where the scaling is upward. If t − e < −1074, which needs M > 2⁵⁰⁰, then b < 2⁵³·2⁻¹⁰⁷⁵ = 2⁻¹⁰²² and
by the ldexp hypothesis b̂ is a point of the subnormal grid within T of b, so 0 ≤ b̂ ≤ 2⁻¹⁰²². Hence: b̂ ≠ b only if
b < 2⁻¹⁰²², and then b̂ ≤ 2⁻¹⁰²².

*(H_θ) for α = a, β = b, β̂ = b̂* (a² ∈ [1/4, 1), b̂ < 1). Three cases. (α) b ≥ 2⁻⁵¹¹: b̂ = b, b² ≥ 2⁻¹⁰²², q =
b²(1 + δ₂). (β) 2⁻¹⁰²² ≤ b < 2⁻⁵¹¹: b̂ = b, and F2 gives |q − b²| ≤ 2⁻¹⁰⁷⁵ ≤ 2⁻¹⁰⁷³a². (γ) b < 2⁻¹⁰²²: b̂ ≤ 2⁻¹⁰²²,
b̂² ≤ 2⁻²⁰⁴⁴ < 2⁻¹⁰⁷⁵, so q = 0, and |q − b²| = b² < 2⁻²⁰⁴⁴ ≤ 2⁻²⁰⁴²a². So (H_θ) holds with θ = 2⁻¹⁰⁷³, and by 2.2

    ζ(1 + 4u) ≤ m_s ≤ ζ(1 + 12u),   ζ = √(a² + b²) = |z|·2⁻ᵉ ∈ [1/2, √2).

The rounding of the scaled smaller component to the subnormal grid (case γ) costs nothing that can be measured: it
happens only where b²/a² < 2⁻²⁰⁴². The guard does not have to pay for it.

*Rescaling.* Let v := m_s·2^e (real); |z|(1 + 4u) ≤ v ≤ |z|(1 + 12u). m_s is a binary64 number in [1/2, 2), so
v = k·2^j with an integer k < 2⁵³, and v cannot lie strictly between Ω and 2¹⁰²⁴.
(a) v ≥ 2¹⁰²⁴: ldexp returns ∞, and m = ∞.
(b) 2⁻¹⁰²² ≤ v < 2¹⁰²⁴: v is a binary64 number (F4; j ≥ −1074 follows from v ≥ 2⁻¹⁰²² and k < 2⁵³), so m′ = v, the test
m′ < 2⁻¹⁰²² is false, and m = v.
(c) v < 2⁻¹⁰²²: by the ldexp hypothesis m′ is a grid point with |m′ − v| < T; so m′ ≤ 2⁻¹⁰²². If m′ = 2⁻¹⁰²² no step is
taken and m = m′ > v. If m′ < 2⁻¹⁰²², the nextafter hypothesis gives m = m′ + T > v. In both cases v < m < v + 2T.
Since v ≥ |z|, case (c) needs |z| < 2⁻¹⁰²².

**2.5 Zero, infinite and nan components.** The code evaluates the unscaled formula for every entry, with the
floating-point warnings off, and overwrites the entries whose M is not in [2⁻⁵⁰⁰, 2⁵⁰⁰] with the scaled result; for
M = 0, ∞ or nan both comparisons of `fast` are false, so these entries are among them. Then every entry with an
infinite component is set to ∞. Then, entries with x = 0 get y, and entries with y = 0 get x. For finite z this is
(iv): x = 0 gives m = y = |z| and y = 0 gives m = x = |z|, exactly, since |Re z| and |Im z| are exact. It holds for
−0 as well (−0 = 0 is true and |−0| = +0), in both branches, and for 0 + 0i (m = +0). The scaled results that are
overwritten are harmless: frexp(0) = (0, 0) gives 0, one step up gives T, and the zero override replaces it.
frexp(∞) and frexp(nan) return an exponent that C leaves unspecified (−1 on the reference host), but ldexp(∞, k) = ∞
and ldexp(nan, k) = nan for every k; so an entry with a nan component is nan after the scaled formula whatever the
exponent was (p + q is nan), it stays nan through the step (nan < 2⁻¹⁰²² is false), and the ∞ override then acts if
the other component is infinite. The ∞ override and the zero overrides commute: on (0, ∞) both give ∞; on (0, nan)
the zero override gives nan; on (∞, nan) only the ∞ override acts. So the result is ∞ if a component is infinite,
otherwise nan if a component is nan, as the statement says.

**2.6 Lemma M.** Let z have finite components.
(i) With a zero component m = |z|. Unscaled branch: m ≥ |z|(1 + 4u). Scaled branch: (a) m = ∞; (b) m = v ≥
|z|(1 + 4u); (c) m > v ≥ |z|(1 + 4u). So m ≥ |z|, and more: m ≥ |z|(1 + 4u) whenever both components are nonzero,
which the statement does not claim.
(ii) Let m be finite. With a zero component m = |z|. Unscaled branch: m ≤ |z|(1 + 12u). Scaled branch, (b): m = v ≤
|z|(1 + 12u); (c): m < v + 2T ≤ |z|(1 + 12u) + 2⁻¹⁰⁷³. If |z| ≥ 2⁻¹⁰²², case (c) cannot occur, and in the unscaled
branch |z| ≥ 2⁻⁵⁰⁰ anyway; so then m ≤ |z|(1 + 12u).
(iii) m = ∞ occurs only in (a), where |z|(1 + 12u) ≥ v ≥ 2¹⁰²⁴ > Ω.
(iv) Section 2.5. ∎

*Remarks.* (1) Since m ≥ |z|(1 + 4u) for two nonzero components, the bound is never exact there;
`modulus_up(3 + 4i)` is 5 + 5 ulp, not 5.
(2) (iii) can be read in both directions: m = ∞ only if |z|(1 + 12u) ≥ 2¹⁰²⁴, and m = ∞ whenever both components are
nonzero and |z|(1 + 4u) ≥ 2¹⁰²⁴. Example: x + i·2⁻¹⁰⁷⁴ gives ∞ for every x ≥ Ω − 7 ulp, while x + 0i gives x. This is
allowed by (iii), but a caller should know that a nonzero imaginary part of any size makes the bound infinite just
below Ω. (3) With a correctly rounded ldexp, case (c) gives m ≤ v + 3T/2; with a truncating one, m ≤ v + T. The term
2⁻¹⁰⁷³ = 2T covers every rounding direction, which is what FP-M allows. (4) (ii) is also a lower bound on |z| in terms
of m; it could serve where a lower bound on a modulus is needed (the open question of the statement). Not pursued here.

**2.7 The corollary.** Let P_j := m_ij·m_j (real). Then P_j ≥ |a_ij||x_j| ≥ 0, by Lemma M (i) or exactness. Two
hypotheses are needed that the statement does not list: (C-a) every modulus bound is finite; (C-b) no product
underflows, that is, P_j = 0 or P_j ≥ 2⁻¹⁰²². Then fl(P_j) = P_j(1 + δ_j) by F1 (an overflow gives ∞, which only
helps). A sum of two non-negative binary64 numbers c, d satisfies fl(c + d) = (c + d)(1 + δ) without exception at the
bottom: below 2⁻¹⁰²² the sum of two multiples of T is a binary64 number and is exact (and an overflow at the top
gives ∞, which again only helps). Any order of summation is a
binary tree with w leaves, and a leaf has at most w − 1 additions above it. By induction over the tree the computed
sum is Σ_j P_j·Π(1 + δ) with at most w factors for each term (one product, at most w − 1 additions), so

    fl(Σ_j m_ij m_j) ≥ (1 − u)^w Σ_j P_j ≥ (1 − wu) Σ_j P_j ≥ (1 − γ_w) Σ_j |a_ij||x_j|,

the last step because γ_w ≥ wu. Nothing is counted for the moduli: they enter only through P_j ≥ |a_ij||x_j|. A
fused multiply-add in the summation removes roundings and the bound stays.

Without (C-b) the inequality is false. a = x = 2⁻⁶⁰⁰ (real, or with a zero imaginary part): the moduli are exact,
fl(2⁻⁶⁰⁰·2⁻⁶⁰⁰) = 0, and |a||x| = 2⁻¹²⁰⁰ > 0. With gradual underflow, F2 gives fl(P_j) ≥ P_j(1 − u) − 2⁻¹⁰⁷⁵ for every
product, and the same induction gives the statement that is true without (C-b):

    fl(Σ_j m_ij m_j) ≥ (1 − γ_w) Σ_j |a_ij||x_j| − w·2⁻¹⁰⁷⁵.

Without (C-a): Lemma M (iii) allows m_ij = ∞ for finite a_ij; if then m_j = 0, the product ∞·0 is nan and the sum is
nan. A bound that is nan must be treated as a failure by the caller; the inequality itself is not true for nan.
If the analyses that use the corollary already carry a no-underflow hypothesis or an underflow term, the corollary
inherits it and should say so.

*Other expressions.* Let E be built from non-negative arguments by sums and products along a fixed tree in which
every path from the root to a leaf passes at most d operations. Under (C-a) and (C-b) for every product in the tree,
the same induction gives Ê(m) ≥ (1 − u)^d E(m) ≥ (1 − γ_d)E(m), and E is non-decreasing in each argument, so
E(m) ≥ E(|a|). The count d holds the sums and products and nothing for the moduli. So the last sentence of the
corollary is right with the same two provisos.

## 3 The hypotheses FP-M: where each is used, what is stronger than needed, what is missing

**Round to nearest, correctly rounded ×, +, √.** Used in F1 for δ₁, δ₃, δ₄, δ₅ and for δ₂ when the square is normal;
in F2 for the square that underflows (cases N² < 2⁻¹⁰²², β and γ). "For subnormal results too" is needed for the
multiplication only: every sum here is at least 2⁻¹⁰⁰⁰ and every root at least 2⁻⁵⁰⁰. Round to nearest is stronger
than (i) needs: with any rounding direction (|δ| < 2u, absolute error below T) the same steps give
g ≥ ζ[(1 − 2u)³(1 + 8u) − ...] = ζ(1 + 2u − 36u² + ...) > ζ, and the rescaling argument (c) does not depend on the
direction. The constants of (ii) and (iii) do need it: with directed rounding the count gives 1 + 14u + ..., not 12u.

**Gradual underflow.** Used at one place that matters: the unscaled branch with N² < 2⁻¹⁰²² (θ = 2⁻⁷⁵). With products
flushed to zero, M = 2⁻⁴⁹⁹ and N = 2⁻⁵¹² give q = 0 in place of 2⁻¹⁰²⁴, a relative loss of 2⁻²⁷ in |z|, far above
the guard; (i) is then false (control NC_flush_to_zero_products, Section 5). It is also what gives subnormal inputs
and results a meaning. In the scaled branch a flushed square would be harmless (θ ≤ 2⁻¹⁰²⁰).

**np.frexp.** The exact exponent places a in [1/2, 1), which gives a² ∈ [1/4, 1) and m_s ∈ [1/2, 2). It is used for
subnormal M (e down to −1073). On this host it is exact on 200,000 samples that include subnormal numbers.

**np.ldexp.** Three clauses, three uses. Exact when the result is a binary64 number: a always, b̂ when b is on the
grid, m′ in case (b). ∞ above Ω: case (a). A neighbour on the subnormal grid, error below T: for b̂ in case (γ), where
any value in [0, 2⁻¹⁰²²] would do, and for m′ in case (c), where this clause is exactly what the one step up
compensates. The clause is weaker than C99 Annex F, as the statement says, and it is all the proof uses. On this host
np.ldexp returned the correctly rounded value in all 197,010 inexact subnormal results sampled, never anything
further than a neighbour. The exact model of the code passes (i) to (iii) with a truncating and with an upward ldexp
as well.

**np.nextafter.** t + T for 0 ≤ t < 2⁻¹⁰²²: case (c).

**One rounding per ufunc call.** Identifies the five roundings. A fused operation would round less often; agreed.
Evidence on this host: the code is bit-identical with a model that rounds each operation once, in 1,009,411 cases
evaluated by array calls (the SIMD loops).

**The thresholds 2^±500.** Conservative by 2¹⁰ and 2¹¹ (Section 2.3).

**The guard 1 + 8u.** Stronger than needed: by the count of 2.2, 1 + 4u already gives (i) ((1 − u)³(1 + 4u) =
1 + u − 9u² + ... > 1), and (ii) would then hold with 1 + 8u in place of 1 + 12u. 1 + 2u does not give (i)
((1 − u)³(1 + 2u) < 1; the control finds violations). The extra 4u is harmless and buys margin.

**The constant 1 + 12u.** Right, and not sharp. The count gives 1 + 11u + 27u² + 25u³ + 8u⁴ + θ-terms, which is above
1 + 11u, so 12 is the least integer this proof yields. The largest ratio found is 1 + 10.513u; no case above 1 + 11u.

**The term 2⁻¹⁰⁷³.** Right for FP-M as stated: an ldexp error below T plus the step T. It cannot be dropped (z =
2⁻¹⁰⁷⁴(1 + i) gives m = 2T against |z| = 1.414T) and T alone is not enough when ldexp rounds to nearest or upward
(largest excess found on this host: 1.5000T; with round to nearest the bound is m ≤ v + 3T/2).

**Missing, small.** (1) The excerpt does not define U; the proof and the check assume U = 2⁻⁵³, so that _GUARD is
1 + 2⁻⁵⁰ exactly. (2) The lemma is for binary64 (or binary32) components; the code does not enforce this (Section 4).
(3) The corollary needs (C-a) and (C-b).

## 4 The code against the statement

Read line by line, the code does what "the computation" says, operation by operation, one ufunc call each: the
branch test is inclusive at both ends as in the statement; the scaled function applies ldexp to x and y rather than
to M and N, which is the same pair; the step up is taken exactly when m′ < 2⁻¹⁰²². The points below are the inputs
on which the code does something the statement does not describe, or describes only in part.

1. *Entries that are overwritten.* The scaled function also receives entries with M = 0, ∞, nan; their results are
   replaced (Section 2.5). Nothing depends on the unspecified exponent that frexp returns for ∞ and nan.
2. *Negative zero.* Treated as zero; the result is +0 or the exact other component (272 checks).
3. *Order of the overrides.* ∞ first, zero second. The order is immaterial, as the two rules never disagree
   (Section 2.5). nan with ∞ gives ∞; 0 with ∞ gives ∞; 0 with nan gives nan. This is also what np.abs returns (51
   checks, and no floating-point error is raised under `np.errstate(all="raise")`).
4. *complex64.* The components are widened to float64 exactly; the result is float64 and equals the complex128 call
   bit for bit; Lemma M applies (50,025 checks). This is about the modulus only: a bound formed in single precision
   is not covered (the author's document, Risks and limits).
5. *Scalars.* A Python complex, a NumPy complex scalar and a 0-d array give a Python float equal to the array result.
6. *Shapes and layouts.* Fortran order, strided views, read-only, big-endian and empty arrays give the same values,
   and the input is not modified.
7. *Large arrays.* No chunking; about five float64 temporaries of the size of the input. The values do not depend on
   the position in the array (2,000,003 entries; offsets, reversal, strides, and 20,000 one-entry calls agree bit for
   bit). Arrays near the memory limit were not run.
8. *Not covered by the statement: object arrays.* An array of dtype object that holds complex numbers has kind "O",
   not "c", and is sent to np.abs, which calls abs() of each element. The result is not an upper bound: m < |z| in
   49,842 of 100,000 random entries, and in 1 of the 3 NumPy counterexamples. The code should reject or convert it.
9. *Not covered: integers.* For integer input np.abs wraps at the most negative value (−2⁶³ stays −2⁶³). "Exact"
   holds for floating dtypes.
10. *Not covered: np.clongdouble wider than binary64.* `astype(np.float64)` then rounds the components to nearest.
    On this host long double is binary64, so this could not be run; by reasoning only: with a zero component the
    override returns the rounded other component, which can be below the true one, so (i) fails there; with two
    nonzero components in the normal range the margin m ≥ |z′|(1 + 4u) absorbs the rounding (|z′| ≥ |z|(1 − u)).
11. *abs_up, sparse input.* The duplicates of the argument are summed in place (a COO matrix with 600 stored entries
    had 495 afterwards), as the docstring says. The moduli bounded are those of the summed, rounded entries. The
    pattern is kept, the result is real, and every entry is ≥ the modulus of the stored entry, exactly, for coo, csr,
    csc, bsr, dia, lil, dok, matrices and arrays (16 checks).
12. *The step up is also taken when the rescaling was exact* (a subnormal m′ that is a grid point). This is
    harmless for (i) and is inside the term 2⁻¹⁰⁷³ of (ii).

## 5 Machine check

Script `docs/proofs/independent/adversarial_fp_modulus-up.py` (written in this session as `adversarial_modulus_up.py`,
SHA-256 641df231...75f4; in the repository it tests `certify_ed.fp` itself, or an excerpt given with `--code`). Host:
Python 3.11.17, NumPy 2.4.4, SciPy 1.17.1, Windows (build 26200), AVX512F; run of 18 Nov 2026 on the excerpt, 620 s.
The run was made twice (the second after the lower-margin count below was added); every other count was identical in
both. The code under test was the excerpt itself (executed with U = 2⁻⁵³). Every pass/fail decision on Lemma M is a
comparison of rational numbers (`fractions.Fraction` on the binary64 values): m² ≥ x² + y² for (i);
(m − 2⁻¹⁰⁷³)² ≤ (x² + y²)(1 + 12u)² and, for x² + y² ≥ 2⁻²⁰⁴⁴, m² ≤ (x² + y²)(1 + 12u)² for (ii);
(x² + y²)(1 + 12u)² > Ω² for (iii); equality for (iv).

**Cases.** 1,009,411 in 17 sets: A, all ordered pairs of 183 boundary magnitudes (2⁻¹⁰⁷⁴ to 5·2⁻¹⁰⁷⁴, 2⁻¹⁰²² ± 3 ulp,
2⁻⁵³⁷, 2⁻⁵¹¹, 2⁻⁵⁰⁰ ± 3 ulp, 2⁵⁰⁰ ± 3 ulp, 2⁵¹¹, 2¹⁰²³, Ω − 0..3 ulp, and others), 33,489; B, component
ratios 2^±k for k = 0..1100 from 33 bases, 97,880; C, equal components at every exponent,
16,668; D, the three NumPy counterexamples with all signs and orders and scaled by 2^j, 12,596;
F, Pythagorean triples scaled by 2^j, 16,782; G, the border of the subnormal step, 9,626; Z,
one component zero, 40,370; E1 to E8b, random, 770,000 in nine families aimed at the two branches, the
thresholds, the subnormal range, overflow, and underflow of the smaller square; E9, 12,000 cases picked from
5,000,000 random z by a double-double screen (the screen is a selector, not exact; the picked cases are checked
exactly).

**Lemma M on the code.** Violations: (i) 0 of 1,009,411; (ii) with the term 2⁻¹⁰⁷³, 0 of 1,009,411; (ii) without it
for |z| ≥ 2⁻¹⁰²², 0 of 907,125; (iii) 0 of 6,027 infinite results; (iv) 0 of
40,370. 96,113 results were below 2⁻¹⁰²² and 96,259 cases had
|z| < 2⁻¹⁰²². Every case of the proof was reached: unscaled with a normal square 326,967, with an
underflowing square 128,102; scaled, with b̂ = b and a normal square (case α) 393,248, with
b̂ = b and an underflowing square (case β, and case γ when b is on the grid) 72,553, with b̂ ≠ b (case γ) 48,171;
rescaling to ∞ 6,027, to a subnormal value inexactly 90,953, exactly
4,118, and rounded up to 2⁻¹⁰²² 4.
The lower margin of Section 2.6, which the statement does not claim: m < |z|(1 + 4u) in 0 of 963,014 cases with a
finite m and two nonzero components; m < |z|(1 + 5u) in 0; m < |z|(1 + 6u) in 4,157 (a control).

**Exact model.** A model of "the computation", written from the statement with integer arithmetic (each rounding
done on integers, no libm), agrees with the code bit for bit in 1,009,411 of 1,009,411 cases. Run with a truncating
ldexp (differs from the code in 13,639 cases) and with an upward ldexp
(77,314 cases) it has 0 violations of (i), (ii), (iii).

**Negative controls** (violations of (i) unless said otherwise; all on the same cases, so the checks can fail):
guard 1 + 2u, 169 (127 of them in E9); no guard, 661,951; no subnormal
step, 74,728 (ldexp to nearest) and 86,824
(truncating); no scaling, 192,055, and ∞ against (iii) in 213,912; products
flushed to zero, 21,279; numpy.abs itself, below |z| in 769,984 cases
(any to-nearest modulus is below |z| about half the time; this control shows that the check sees values that do not
come from the model). For (ii): m > |z|(1 + 8u) in 382,140 of 907,125 cases, > 1 + 9u in
31,248, > 1 + 10u in 2,602, > 1 + 11u in 0; with no
additive term, 90,279 of 96,259 cases with |z| < 2⁻¹⁰²² violate; with T in
place of 2T, 11,320 (code, ldexp to nearest), 87,618
(model, upward ldexp), 0 (model, truncating ldexp). For information: guard 1 + 4u, 0
violations of (i); guard 1 + 6u, 0.

**Extremes** (floats evaluated from the exact rationals, not pass/fail). (m/|z| − 1)/u lies between 5.4586 and
10.5130 over the cases with |z| ≥ 2⁻¹⁰²² and two nonzero components. The largest m − |z|(1 + 12u) for |z| < 2⁻¹⁰²² is
1.5000 T.

**FP-M on this host** (200,000 samples each, compared with integer reference arithmetic): np.multiply (1,332
subnormal results among them), np.add, np.sqrt, np.frexp (subnormal arguments included), np.ldexp with a normal
result, np.ldexp above Ω, np.nextafter below 2⁻¹⁰²²: 0 deviations each. np.ldexp with a subnormal result: 197,010
inexact, all equal to the correctly rounded value, 0 further than a neighbour; 2,990 exact.

**Corollary.** With moduli from `modulus_up`, w ∈ {1, 2, 3, 5, 10, 100, 1000}, four orders (left to right, random,
np.sum, np.dot), and a rational upper bound for Σ|a_ij||x_j|: 0 violations in 6,784 sums without underflow.
Controls: with products that underflow, 432 of 1,696 sums violate the inequality as stated; with numpy.abs
in place of `modulus_up`, 51 of 1,696 sums violate it (so the moduli would need a rounding count there, as
the statement says); the three explicit cases of Section 2.7 behave as described.

**Limits of this check.** Not run: `np.clongdouble` wider than binary64 (not available on this host; item 10 is
reasoning), arrays larger than 2,000,003 entries, hosts other than this one (for other ldexp roundings the exact
model stands in). The excerpt does not show the definition of U. The double-double screen and the extremes are not
exact; everything else is.

## 6 Suggested edits (for the author; none changes Lemma M)

1. Corollary: add "if no product underflows and the moduli are finite", or state it with the term − w·2⁻¹⁰⁷⁵ and say
   that a nan bound counts as a failure.
2. Statement, first sentence of "the computation": "for real floating input".
3. Code: raise for object arrays and for complex types wider than complex128 (or round their components upward), so
   that every accepted input is covered by Lemma M.
4. Optional, for the record: m ≥ |z|(1 + 4u) when both components are nonzero; ∞ is returned from
   |z| ≈ Ω(1 − 7u) on when both components are nonzero; "for subnormal results too" is needed for the multiplication
   only.

## 7 Runs and resolution (added in the repository)

* Run of 18 Nov 2026 in this session, on the excerpt: the counts of section 5 (1,009,411 exact cases, 620 s).
* `python docs/proofs/independent/adversarial_fp_modulus-up.py --workers 6`, on `certify_ed.fp` at commit 249b8ae of
  `fix/upper-moduli` (after two review rounds; the second added passes
  without a copy of the input, 0 + 0i on the unscaled branch, a float for every scalar, integer sparse data stepped
  up; output in `docs/proofs/runs/adversarial_fp_modulus-up.txt`): the same 1,009,411 exact cases; (i) 0 violations,
  (ii) 0 (907,125 with |z| ≥ 2⁻¹⁰²²), (iii) 0 in 6,027 infinite results, (iv) 0 in 40,370; lower margin 1 + 4u: 0
  violations in 963,014; code bit-identical with the exact model; all negative controls detected; corollary 0
  violations in 6,784 sums without underflow, controls 432 of 1,696 (products that underflow) and 51 of 1,696
  (`numpy.abs`); special inputs 0 failures (integer input returns [2⁶³ stepped up, 5]; an object array is rejected
  with TypeError); expectations not met: 0 (454 s).
* Resolution by the author of `docs/proofs/fp_modulus-up.md` (2026-11-18): the corollary is stated for finite moduli
  with the term w·2⁻¹⁰⁷⁵, absent when no product underflows, and its general form with the same provisos (suggested
  edit 1); "for real input" reads "real floating input", and integers are converted to float64 and stepped up from 2⁵³
  (edit 2); object arrays, sparse input and complex types wider than complex128 are rejected with TypeError (edit 3);
  the lower margin m ≥ |z|(1 + 4u) is recorded as a remark (edit 4). A non-finite bound makes the records that rest on
  it INCONCLUSIVE (S2, S3, S1 on a sector isometry, the variational and moment records; `tests/test_nonfinite_bounds.py`
  and, with an infinite upper modulus, `tests/test_nonfinite_moduli.py`), as edit 1 asks for a nan bound. Lemma M is
  unchanged.
