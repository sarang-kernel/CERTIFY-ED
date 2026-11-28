# C-07 (second part) Certified Chebyshev moments and bounds on the spectral distribution on a certified level

Status: derived on 24 November 2026; UNVERIFIED until Sarang's check (D-12). Machine check: `docs/proofs/verify_C-07b.py`.
Code: `certify_ed/spectral_bounds.py` (statements M5 and M6 of the moments record family).

Notation as in `docs/proofs/C-07_moments.md`: H Hermitian n × n and O Hermitian n × n (both stored), X (n × g) the stored
vectors of a certified S4 cluster, P the spectral projector of H onto the cluster's g eigenvalues, η = ‖XᴴX − I‖₂ < 1,
s the certified sin θ bound, u = 2⁻⁵³, γ_m and γ̇_m as in `fp.py`, w_A the largest number of nonzeros per row of A.
E(·) is the spectral measure of H, T_j and U_j the Chebyshev polynomials of the first and second kind.
Two letters have a second meaning below: in Lemma 5 (c), Lemma 7 and Proposition 3, s is the scale of the map onto
[−1, 1] (s_c in the code), not the sin θ bound of Lemma 6; in Lemma 8, s is the variable of the Bernstein
polynomials and A a polynomial with integer coefficients, not the matrix (H − cI)s.

## 1 Statement

**Lemma 5 (the spectral interval and the map onto [−1, 1]).** (a) With d_i = h_ii (real) and
R_i = Σ_{k≠i} |h_ik|, every eigenvalue of H lies in [min_i(d_i − R_i), max_i(d_i + R_i)] (Gershgorin). The code takes
the off-diagonal part H − diag(H) (exact), the row sums ρ_i = fl(Σ_k m_ik) of the upper moduli m_ik ≥ |h_ik|
(`abs_up`) in any order, R̄_i = up(ρ_i/(1 − γ_w)) ≥ R_i, and the ends `sub_down`(d_i, R̄_i), `add_up`(d_i, R̄_i).
(b) If λ_min(H) ≥ ℓ and λ_max(H) ≤ h are proved otherwise (for ℓ: a 'lowest' claim with S2 and S3 CERTIFIED, whose
intervals hold the lowest eigenvalues, gives ℓ = `sub_down`(μ_i, r_i) minimised over the claim, r_i the larger stored
radius), the interval [lo, hi] is the intersection. (c) For binary64 numbers c and s > 0 with (lo − c)s ≥ −1 and
(hi − c)s ≤ 1 (checked in exact rational arithmetic), the exact matrix A := (H − cI)s is Hermitian with σ(A) ⊂ [−1, 1].

**Lemma 6 (Chebyshev moments through the stored vectors).** Put τ_j := tr(P O T_j(A) O)/g. Then
‖T_j(A)‖₂ ≤ 1 and, with Lemma 1 of C-07 applied to the Hermitian M = O T_j(A) O (‖M‖₂ ≤ ‖O‖₂²),

    | τ_j − (1/g) Σ_i ⟨Ox_i, T_j(A) Ox_i⟩ | ≤ (2s + η) ‖O‖₂².

**Lemma 7 (floating-point accounting of the three-term recurrence).** For a stored vector z let
â(z) := fl(s·fl(fl(Hz) − fl(cz))) (componentwise: a sum of the w products h_ik z_k, minus the rounded product c z_i, then
one multiplication by s). Then |â(z)_i − (Az)_i| ≤ s γ̇_{w+2} (Σ_k |h_ik||z_k| + |c||z_i|), and
ε(z) := up(`norm2_upper`(γ̇_{w+2} s (fl(|H|m) + |c|m)/(1 − γ_{w+4}))), m = `modulus_up`(z), bounds ‖â(z) − Az‖₂.
Let ŷ_i = fl(Ox_i), ẑ₀ = ŷ_i, ẑ₁ = â(ẑ₀) and ẑ_{j+1} = fl(2 â(ẑ_j) − ẑ_{j−1}) for j ≥ 1 (multiplication by 2 is
exact); ε_j := ε(ẑ_{j−1}), δ₁ = ε₁ and, for j ≥ 2,

    δ_j = up( 2 ε_j + u/(1 − u) · `norm2_upper`(ẑ_j) ).

Then f_j := ẑ_j − (2 A ẑ_{j−1} − ẑ_{j−2}) (j ≥ 2), f₁ := ẑ₁ − A ẑ₀ satisfy ‖f_j‖ ≤ δ_j, and the error
e_j := ẑ_j − T_j(A) ŷ_i satisfies

    e_j = Σ_{l=1}^{j} U_{j−l}(A) f_l,      ‖e_j‖ ≤ E_j := Σ_{l=1}^{j} (j − l + 1) δ_l.

E_j is computed as S_j = up(S_{j−1} + δ_j), E_j = up(E_{j−1} + S_j) (E_j − E_{j−1} = Σ_{l≤j} δ_l). With
(v_i, d_i) = `exact_dot`(ŷ_i, ẑ_j) and τ̂_j = fl(Σ_i Re v_i)/g,

    | τ̂_j − (1/g) Σ_i ⟨Ox_i, T_j(A) Ox_i⟩ | ≤ (1/g) Σ_i [ (2‖ŷ_i‖ + e_y,i) e_y,i + ‖ŷ_i‖ E_j,i + d_i ]
                                               + γ_{g+1} Σ_i |Re v_i| / g + u|τ̂_j|,

with e_y,i the bound of Lemma 3 of C-07 on ‖ŷ_i − Ox_i‖. The certified interval of M5 is τ̂_j ± b_j with
b_j = up((projector term + rounding terms)/g) + 2u|τ̂_j|, the last addition rounded upward and the ends rounded
outward.

**Proposition 3 (M6, bounds on the distribution function).** The set function μ(S) := tr(P O E_A(S) O)/g is a
positive Borel measure, carried by σ(A) ⊂ [−1, 1], with ∫ T_j dμ = τ_j. For ξ ∈ [−1, 1) put F(ξ) := μ([−1, ξ]). If
p = Σ_j d_j T_j satisfies

    p(u) ≥ 1 for u ∈ [−1, ξ]   and   p(u) ≥ 0 for u ∈ [ξ, 1],

then F(ξ) ≤ Σ_j d_j τ_j ≤ Σ_j max(d_j τ_j^lo, d_j τ_j^hi); if q = Σ_j d_j T_j satisfies q ≤ 1 on [−1, ξ] and q ≤ 0 on
[ξ, 1], then F(ξ) ≥ Σ_j min(d_j τ_j^lo, d_j τ_j^hi). Always 0 ≤ F(ξ) ≤ τ₀^hi, F(ξ) = 0 for ξ < −1 and F(ξ) = τ₀ for ξ ≥ 1.
In energy units, the distribution of H is F_H(E) = μ_H((−∞, E]) = F((E − c)s), and the energies of a query are mapped
to ξ = (E − c)s in exact rational arithmetic (a dyadic number, since E, c and s are binary64 numbers).

**Corollary (monotone envelope).** F is nondecreasing, so for energies E₁ < … < E_m with bounds L_k ≤ F(E_k) ≤ U_k,
also max_{l ≤ k} L_l ≤ F(E_k) ≤ min_{l ≥ k} U_l.

**Lemma 8 (proof of the sign conditions).** Let A be a polynomial with integer coefficients of degree at most m (the
d_j are rounded to multiples of 2⁻⁴⁰ first, and T_j has integer monomial coefficients) and Q(t) = A(2t − 1)
= Σ_k q_k t^k, so that t ∈ [0, 1] ↦ u = 2t − 1 ∈ [−1, 1] and the q_k are integers. Write B_{i,m}(t) = C(m, i) tⁱ(1 − t)^{m−i}.

(a) Q = Σ_i β_i B_{i,m} with β_i = Σ_{k ≤ i} C(i, k)/C(m, k) q_k = N_i / C(m, i), N_i = Σ_{k ≤ i} C(m − k, i − k) q_k.

(b) For coefficients b_0..b_m and t ∈ [0, 1], Σ_i b_i B_{i,m}(t) ≥ min_i b_i.

(c) (de Casteljau with rounding down) For t* ∈ [0, 1] let b⁽⁰⁾ = b and let b⁽ᵏ⁾_i be any numbers with
b⁽ᵏ⁾_i ≤ (1 − t*) b⁽ᵏ⁻¹⁾_i + t* b⁽ᵏ⁻¹⁾_{i+1} (k = 1..m, i = 0..m − k). If b_i ≤ β_i for all i, the numbers
l_k = b⁽ᵏ⁾_0 and r_k = b⁽ᵐ⁻ᵏ⁾_k satisfy Σ_k l_k B_{k,m}(s) ≤ Q(t* s) and Σ_k r_k B_{k,m}(s) ≤ Q(t* + (1 − t*) s) for
s ∈ [0, 1]: the coefficients of each part are lowered, and the polynomials they represent lie below Q on that part.

The code takes b_i = ⌊2⁶⁴ N_i / C(m, i)⌋ (so b_i ≤ 2⁶⁴ β_i), splits at the dyadic t* = (ξ + 1)/2 with integer
arithmetic and floor shifts, keeps the part [0, t*] for
the condition on [−1, ξ] and [t*, 1] for the one on [ξ, 1], cuts each part first near the free nodes of the polynomial
(where it touches its bound; the cut points are multiples of 2⁻³⁰ of the remaining part, and any cut is admissible by
(c)), and bisects each piece (t* = 1/2, again rounding down) until
every piece has all its coefficients ≥ 0, an end coefficient is negative, or the depth 40 is reached (then the
polynomial is not used and the trivial bound is reported). By (c) applied along the chain of splits, every piece's
coefficients represent a polynomial below 2⁶⁴ A on that piece, and by (b) all of them ≥ 0 proves A ≥ 0 there.

## 2 Proof

**Lemma 5.** (a) Gershgorin's theorem for the Hermitian H. The off-diagonal part is exact: H − diag(H) subtracts each
stored diagonal entry from itself (0) and leaves the others. For nonnegative terms a sum computed in any order is
fl(Σ m_k) = Σ m_k(1 + θ_k) with |θ_k| ≤ γ_{w−1}, so Σ m_k ≤ ρ/(1 − γ_w); `up` covers the rounding of the division,
and `sub_down`/`add_up` (one step outward after a rounded subtraction or addition) give ends on the safe side. (b) The
spectrum lies in each of the two intervals, hence in their intersection; for the certificate, S3 CERTIFIED states that
the claimed intervals μ_i ± r_i hold exactly the lowest eigenvalues, so the lowest eigenvalue is at least the smallest
left end. (c) σ(A) = {(λ − c)s : λ ∈ σ(H)} ⊂ [(lo − c)s, (hi − c)s] ⊂ [−1, 1] because s > 0. ∎

**Lemma 6.** For λ ∈ [−1, 1], |T_j(λ)| = |cos(j arccos λ)| ≤ 1; by the spectral theorem ‖T_j(A)‖₂ = max over σ(A) of
|T_j(λ)| ≤ 1. M = O T_j(A) O is Hermitian (T_j has real coefficients) with ‖M‖₂ ≤ ‖O‖₂²‖T_j(A)‖₂ ≤ ‖O‖₂². Lemma 1 of
C-07 gives |tr(PM) − tr(XXᴴM)| ≤ (2gs + gη)‖M‖₂, and tr(XXᴴM) = Σ_i ⟨Ox_i, T_j(A) Ox_i⟩. Divide by g. ∎

**Lemma 7.** The product â(z): component i of fl(fl(Hz) − fl(cz)) is the inner product of the row (h_i1, …, h_in, −c)
with (z_1, …, z_n, z_i), at most w + 1 nonzero terms, evaluated in a particular order with one rounding per product
and per addition; its error is at most γ̇_{w+1} times the sum of the moduli of the terms (Higham 2002, eq. 3.5 and §3.6;
γ̇ = γ for real data, the conservative 2γ_{w+5} of `gamma_dot` for complex data). The multiplication by the real s adds
one relative rounding u, and γ̇_{w+1} + u(1 + γ̇_{w+1}) ≤ γ̇_{w+2}; with |(H − cI)z|_i at most the same sum of moduli,
|â(z)_i − (Az)_i| ≤ s γ̇_{w+2} (Σ_k |h_ik||z_k| + |c||z_i|). The bound vector is formed from nonnegative numbers with at
most w + 4 roundings per component (the products and sums of fl(|H|m), the product |c|m_i, the addition, the products
by s and by γ̇), so the exact vector is at most the computed one divided by 1 − γ_{w+4}. That quotient is itself
evaluated in floating point: γ̇_{w+2} and γ_{w+4} are rounded, and so are 1 − γ_{w+4} and the division, at most four
further relative errors u in each component. They are covered by `up` applied to the 2-norm (the factor 1 + 8u,
itself rounded once, so at least 1 + 6u; ‖v(1 + θ)‖ ≤ (1 + max|θ_i|)‖v‖); for e_y,i (Lemma 3 of C-07) the divisor
1 − γ_{w_O} does not cover the product by γ̇, so there are at most five, which the same factor covers. `norm2_upper`
bounds the 2-norm of the stored vector, and m ≥ |z| componentwise. The subtraction: componentwise fl(a − b) = (a − b)(1 + θ),
|θ| ≤ u (for complex data on the real and the imaginary part separately, which gives the same modulus bound), so
‖fl(a − b) − (a − b)‖ ≤ u‖a − b‖ ≤ u/(1 − u)‖fl(a − b)‖; with a = 2 â(ẑ_{j−1}) the product error enters twice.
The true sequence z_j = T_j(A) ŷ_i satisfies z₀ = ŷ_i, z₁ = A z₀ and z_{j+1} = 2 A z_j − z_{j−1}; subtracting,
e₀ = 0, e₁ = f₁ and e_{j+1} = 2 A e_j − e_{j−1} + f_{j+1}. Induction on j with U₀ = 1, U₁(x) = 2x and
U_{m+1} = 2x U_m − U_{m−1} gives e_j = Σ_{l=1}^{j} U_{j−l}(A) f_l. For λ = cos θ ∈ [−1, 1],
|U_m(λ)| = |sin((m + 1)θ)/sin θ| ≤ m + 1, so ‖U_m(A)‖₂ ≤ m + 1 and ‖e_j‖ ≤ Σ_l (j − l + 1) δ_l. The inner product:
⟨ŷ, ẑ_j⟩ − ⟨y, T_j(A) y⟩ = ⟨ŷ, e_j⟩ + ⟨ŷ − y, T_j(A) ŷ⟩ + ⟨y, T_j(A)(ŷ − y)⟩ with y = Ox_i; the three terms are
at most ‖ŷ‖E_j, e_y‖ŷ‖ and (‖ŷ‖ + e_y) e_y. The exact_dot error, the sum and the division are those of Lemma 3 of
C-07; the three rounding terms of each member are added upward (`add_up`), their sum over the g members is formed
with `math.fsum` (correctly rounded, one relative error u, covered by the `up` of b_j), and the last addition of b_j
is rounded upward. The argument of that `up` carries at most four relative errors u (the `math.fsum`, the two
additions of the projector, rounding and summation terms, and the division by g), and `up` multiplies by 1 + 8u with
one rounding of its own, which covers them; the summation term γ_{g+1} Σ|Re v_i| (spare while the sum is
correctly rounded) reaches the argument through up to six roundings, also within that factor, since
1 + 7u − 8u² ≥ (1 − u)⁻⁶. ∎

**Proposition 3.** For a Borel set S, E(S) is an orthogonal projector commuting with H, and
tr(P O E(S) O) = tr(P P O E(S) E(S) O) = tr((E(S) O P)ᴴ (E(S) O P)) = ‖E(S) O P‖_F² ≥ 0 (cyclicity, P = P² = Pᴴ,
E(S) = E(S)² = E(S)ᴴ). Countable additivity follows from that of E. μ(ℝ \ σ(A)) = 0 because E vanishes there.
By the functional calculus, ∫ f dμ = tr(P O f(A) O)/g for every polynomial f, in particular ∫ T_j dμ = τ_j. If
p ≥ 1_{[−1, ξ]} on [−1, 1] (which the two conditions state: on (ξ, 1] the indicator is 0), then
F(ξ) = ∫ 1_{[−1,ξ]} dμ ≤ ∫ p dμ = Σ_j d_j τ_j, and each term d_j τ_j is at most max(d_j τ_j^lo, d_j τ_j^hi) because
τ_j ∈ [τ_j^lo, τ_j^hi]. The lower bound is the same with the inequalities reversed (q ≤ 0 on [ξ, 1] is stronger than
q ≤ 0 on (ξ, 1], so q ≤ 1_{[−1,ξ]} also at ξ). 0 ≤ F ≤ μ([−1,1]) = τ₀ is positivity; the cases ξ < −1 and ξ ≥ 1 are
the support. The map E ↦ (E − c)s is increasing (s > 0, the scale of Lemma 5 (c)), so
F_H(E) = μ_H((−∞, E]) = μ([−1, (E − c)s]). ∎

**Lemma 8.** (a) The standard change of basis tᵏ = Σ_{i ≥ k} C(i, k)/C(m, k) B_{i,m}(t) gives β_i; and
C(i, k)/C(m, k) = C(m − k, i − k)/C(m, i) (both equal i!(m − k)!/(m!(i − k)!)). (b) The B_{i,m} are nonnegative on
[0, 1] and sum to 1, so Σ_i b_i B_{i,m}(t) is a convex combination of the b_i. (c) With exact arithmetic (equality in
the recursion) and b = β, l and r are the Bernstein coefficients of Q on the two parts (the subdivision property of the
de Casteljau algorithm). Write b̄⁽ᵏ⁾ for the exact recursion from β. By induction on k, b⁽ᵏ⁾_i ≤ b̄⁽ᵏ⁾_i: true for
k = 0 by hypothesis, and b⁽ᵏ⁾_i ≤ (1 − t*) b⁽ᵏ⁻¹⁾_i + t* b⁽ᵏ⁻¹⁾_{i+1} ≤ (1 − t*) b̄⁽ᵏ⁻¹⁾_i + t* b̄⁽ᵏ⁻¹⁾_{i+1} = b̄⁽ᵏ⁾_i
because the weights 1 − t* and t* are nonnegative. So l_k ≤ l̄_k and r_k ≤ r̄_k, and since B_{k,m}(s) ≥ 0,
Σ_k l_k B_{k,m}(s) ≤ Σ_k l̄_k B_{k,m}(s) = Q(t* s), and the same for r. For a chain of splits, apply (c) to the
polynomial R represented by the lowered coefficients of the part being split (R ≤ Q there): its parts lie below R and
hence below Q. ∎

**Remark (why the Hermite polynomial is a majorant; not used by any bound).** Let x₁ < … < x_L < ξ < y₁ < … < y_R be
distinct, n = L + R ≥ 1, and let p of degree ≤ 2n satisfy p(x_k) = 1, p(y_k) = 0, p′ = 0 at all n nodes and p(ξ) = 1
(2n + 1 conditions; the interpolant exists and is unique). If R ≥ 1, p′ has the n zeros at the nodes and, by Rolle,
one zero strictly inside each of the L intervals between consecutive points of x₁, …, x_L, ξ and each of the R − 1
intervals between consecutive y_k: 2n − 1 distinct zeros. p is not constant (p(ξ) = 1, p(y₁) = 0), so p′ has degree
exactly 2n − 1, these are all its zeros, and all are simple. p′ has no zero in (z_L, y₁), z_L the Rolle zero between
x_L and ξ (or (−∞, y₁) when L = 0), and p(ξ) = 1 > 0 = p(y₁), so p′ < 0 there; p′ changes sign at each of its zeros,
so going right from y₁ and left from z_L every node is a local minimum and every Rolle zero a local maximum, and
p′ < 0 left of the first zero, p′ > 0 right of the last. Hence p ≥ 1 on (−∞, ξ] (minima equal to 1 at the x_k, and
p decreasing from z_L to p(ξ) = 1) and p ≥ 0 on [ξ, ∞) (p decreasing from 1 to 0 on [ξ, y₁], minima equal to 0 at the
y_k). If R = 0, p ≡ 1.
The lower polynomial (p(ξ) = 0) is 1 minus the upper one for the reflected step. With the Gauss–Radau nodes of μ
through ξ, ∫ p dμ is the Chebyshev–Markov–Stieltjes bound. The code builds these nodes from the computed moments (the
modified Chebyshev algorithm and a Radau modification of the Jacobi matrix) and solves the Hermite conditions in the
Chebyshev basis, all in floating point, so the polynomial it proves is only near this one; the touching at the nodes
is why a margin is still needed.

## 3 What is and is not claimed

The choice of p and q is a heuristic: the bound reported rests only on the exact proof of Lemma 8 and on the moment
intervals of M5. The candidates are Hermite polynomials of the Remark with two kinds of free nodes: the Gauss–Radau
nodes of the moments (orders n, 3n/4, n/2, n/4, n/8 of the n the moment recursion allows; the moments of a measure
that fills only part of [−1, 1] determine few recurrence coefficients, about 20 for the experiment's structure factors,
and the full order grows by orders of magnitude where the measure has no mass), and the Chebyshev points of the first
kind, n = K/2 and K/4 (the one nearest to ξ dropped when closer than a quarter of the local spacing). For the second
kind the Hermite interpolant is of Hermite–Fejér type, bounded on [−1, 1], with a resolution of one node spacing
around ξ. The candidates are ranked by a floating-point estimate of their bound (Σ d_j τ̂_j ± (Σ|d_j|ρ_j + margin τ₀))
and the best three are proved in that order; when none is proved, a linear program
solved on max(512, 16(K + 1)) Chebyshev points plus points near ξ, which keeps the polynomial between −1 and 2 on its
points (without this, directions of the coefficients on which the moments are nearly zero are free and the polynomial
oscillates between the points) and adds the most violated local minima of a fine scan in up to eight exchange rounds.
A bound is as good as the polynomial and the margin added before the proof (the largest violation found on a scan of
about 4·10⁴ points uniform in u and in arccos u and dense near ξ, refined around its 64 lowest local minima, times
1.01, plus 10⁻⁹(1 + Σ|d_j|), added to d₀; doubled up to eleven times when the proof fails). The bounds are not claimed
to be the optimal ones of the Chebyshev–Markov–Stieltjes inequalities; the experiment compares them with the exact
distribution where it is known. Continued-fraction (Lanczos) bounds on the spectral function itself (statement M4)
remain not implemented.

The derivation is new and stays UNVERIFIED until Sarang's check (D-12). Lemma 7 rests on the same standard-model
hypotheses as Lemma 3 of C-07 (no underflow of nonzero operands). The scan covers the stored vectors and the entries
of H and O and makes M5 INCONCLUSIVE when one of them is below the threshold; for the products formed later
(ŷ = fl(Ox), the iterates ẑ_j and the products of b_j, such as the step-1 term (2‖ŷ‖ + e_y) e_y) the hypothesis is
assumed, not scanned, as for the other standard-model bounds (`docs/limitations.md`). The probes of the round-10
readings (not recorded in the repository) found inputs that pass the scan and make such products subnormal (‖O‖ near
1e-170), with every exact moment inside its interval; why (perhaps the additive term of `exact_dot`) is not derived (UNVERIFIED, D-12), and an upward step for
every product is an open item.
