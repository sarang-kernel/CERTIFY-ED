# C-03 Entanglement entropy and entanglement spectrum of a certified eigenspace

Notation as in `docs/theory.md`: A is the Hermitian n × n operator of the declared model with eigenvalues
λ₁ ≤ … ≤ λₙ; X is the n × g matrix of claimed eigenvectors of a cluster C, η = ‖XᴴX − I‖₂ < 1, Q̂ the polar
factor of X (X = Q̂M, M = (XᴴX)^{1/2}), P the spectral projector of A onto the g eigenvalues of the cluster
(Theorem 6: ‖Q̂Q̂ᴴ − P‖₂ ≤ s, with s = ρ/sep the Davis–Kahan sin Θ bound computed by S4). ‖·‖₁ is the trace
norm, ‖·‖₂ the spectral norm, ‖·‖_F the Frobenius norm. The Hilbert space is a tensor product
ℋ = ℋ_A ⊗ ℋ_B of the local spaces of the sites in A (a user-declared proper, non-empty subset) and of the
remaining sites B; d_A = dim ℋ_A = q^{|A|}. tr_B is the partial trace over ℋ_B. S(ρ) = −tr ρ ln ρ (nats;
the record converts to another base by dividing by the enclosure of ln(base)). h(T) = −T ln T − (1−T) ln(1−T).

## 1 Statement

**Hypotheses.** (H1) S4 has certified the cluster: ‖Q̂Q̂ᴴ − P‖₂ ≤ s < 1 and η = ‖XᴴX − I‖₂ < 1 (Theorem 6,
`eigenspace.certify_eigenspace`, s = `ec.sin_theta`, η = `ec.eta`). (H2) The claim is in the computational
basis (full space or a U(1) block given by `Claim.states`); site 0 is the most significant base-q digit of the
integer encoding (`ModelSpec.site_order = "site0_msb"`).

**Definitions.** ρ_A := tr_B(P)/g (the normalised projector of the certified eigenspace; for g = 1 the reduced
state of the eigenvector), ρ̂_A := tr_B(XXᴴ)/t with t := tr(XᴴX) = ‖X‖_F² (the reduced state of the stored
vectors, normalised; for g = 1, ρ̂_A = tr_B(qqᴴ) with q = x/‖x‖). T := ‖ρ_A − ρ̂_A‖₁/2.

**Proposition 1 (trace distance).** Under (H1),

    T ≤ T* := s + η/(1 − η),   and for g = 1:  T ≤ s   (the second term is identically zero).

**Proposition 2 (entropy).** Under (H1) and if T* ≤ 1 − 1/d_A,

    |S(ρ_A) − S(ρ̂_A)| ≤ g_{d_A}(T*),   g_d(T) := T ln(d − 1) + h(T).

(The inequality |S(ρ) − S(σ)| ≤ g_d(‖ρ − σ‖₁/2) is the Fannes–Audenaert inequality, Audenaert 2007,
Theorem 1, valid for all pairs of d-dimensional density matrices; the range condition is what allows the
upper bound T* to replace T, see Lemma 4.) When T* > 1 − 1/d_A no interval is stated (NOT_CHECKED).

**Proposition 3 (entanglement spectrum).** Under (H1), with λ_k(·) the k-th smallest eigenvalue,

    |λ_k(ρ_A) − λ_k(ρ̂_A)| ≤ 2 T*    for k = 1 … d_A.

**Lemma 3 (exact enclosure of the spectrum of ρ̂_A).** Let ρ̂ be a Hermitian positive semidefinite d × d matrix
with rational entries and trace 1, V any d × d matrix with η_V := ‖VᴴV − I‖_F < 1, p₁ ≤ … ≤ p_d real numbers,
and w := ‖Vᴴ ρ̂ V − diag(p)‖_F. Then for every k

    max(0, p_k − w)/(1 + η_V) ≤ λ_k(ρ̂) ≤ min(1, (p_k + w)/(1 − η_V)).

**Lemma 4 (monotonicity).** g_d is increasing on [0, 1 − 1/d] (and decreasing on [1 − 1/d, 1]).

**Conclusion (what the certificate states).** With the enclosures [l_k, u_k] of λ_k(ρ̂_A) from Lemma 3, the
enclosure Ŝ of S(ρ̂_A) = Σ_k f(λ_k(ρ̂_A)), f(p) = −p ln p, obtained by interval evaluation of f on each
[l_k, u_k] (Lemma 5 below), and F := an upper bound on g_{d_A}(T*):

    S(ρ_A) ∈ [max(0, Ŝ_lo − F), min(ln d_A, Ŝ_hi + F)],
    λ_k(ρ_A) ∈ [max(0, l_k − 2T*), min(1, u_k + 2T*)].

For g > 1 the statement is about ρ_A = tr_B(P)/g, which depends only on the eigenspace, not on the basis the
producer chose inside it (the per-vector reduced states are not well defined for a degenerate level, D-14).

## 2 Proof

**Lemma A (rank bound on the trace norm).** For any matrix Z of rank r, ‖Z‖₁ ≤ r ‖Z‖₂. *Proof.* ‖Z‖₁ is the sum of
the r nonzero singular values, each ≤ ‖Z‖₂ (Horn & Johnson, *Topics in Matrix Analysis*, 1991, §3.5). ∎

**Lemma B (contractivity of the partial trace).** For Hermitian Z on ℋ_A ⊗ ℋ_B, ‖tr_B Z‖₁ ≤ ‖Z‖₁.
*Proof.* Write Z = Z₊ − Z₋ with Z₊, Z₋ ≥ 0 and tr Z₊ + tr Z₋ = ‖Z‖₁ (Jordan decomposition). The partial trace
is positive and trace preserving (tr_B Z_± ≥ 0, tr(tr_B Z_±) = tr Z_±), so by the triangle inequality
‖tr_B Z‖₁ ≤ ‖tr_B Z₊‖₁ + ‖tr_B Z₋‖₁ = tr Z₊ + tr Z₋ = ‖Z‖₁. ∎ (Nielsen & Chuang, *Quantum Computation and
Quantum Information*, Theorem 9.2, states the same for density matrices.)

**Lemma C (rank-one differences).** For unit vectors ψ, φ with ‖ψ − φ‖ = ε: ‖ψψᴴ − φφᴴ‖₁ ≤ 2ε. More precisely
‖ψψᴴ − φφᴴ‖₁ = 2 sin θ and ‖ψψᴴ − φφᴴ‖₂ = sin θ, where cos θ = |⟨ψ, φ⟩|, and sin θ ≤ ε.
*Proof.* ψψᴴ − φφᴴ = ψ(ψ − φ)ᴴ + (ψ − φ)φᴴ; each term is rank one with trace norm ‖ψ‖‖ψ − φ‖ = ε, so
‖·‖₁ ≤ 2ε. For the exact values: ψψᴴ − φφᴴ is Hermitian, traceless, of rank ≤ 2, supported on span{ψ, φ}, so its
eigenvalues are ±μ with μ² = tr(ψψᴴ − φφᴴ)²/2 = (2 − 2|⟨ψ,φ⟩|²)/2 = sin²θ; hence ‖·‖₂ = sin θ, ‖·‖₁ = 2 sin θ.
Finally 1 − |⟨ψ,φ⟩| ≥ (1 − |⟨ψ,φ⟩|²)/2, i.e. ε² = 2 − 2 Re⟨ψ,φ⟩ ≥ 2 − 2|⟨ψ,φ⟩| ≥ sin²θ. ∎

*Remark (which error enters).* The S2 existence radius r (Theorem 1) bounds |λ − μ| only; two different unit
vectors can have the same residual and arbitrary mutual angle, so r gives no bound on ‖ψ − ψ̂‖ without a gap.
The vector error must come from S4: Theorem 6's s = ρ/sep (Davis–Kahan sin Θ) bounds ‖Q̂Q̂ᴴ − P‖₂, which for
g = 1 is sin θ between q = x/‖x‖ and the exact eigenvector ψ (Lemma C). By Lemma C, sin θ ≤ ‖ψ − e^{iφ}q‖ for
every phase φ, so the bound T ≤ s is at least as sharp as the bound 2ε/2 = ε obtained from any vector error ε
through the route "‖ρ_A − ρ̂_A‖₁ ≤ ‖ψψᴴ − ψ̂ψ̂ᴴ‖₁ ≤ 2ε" (Lemmas B and C). The code uses T ≤ s.

**Proof of Proposition 1.** By Lemma B applied to Z = P/g − XXᴴ/t (Hermitian),

    2T = ‖tr_B(P/g − XXᴴ/t)‖₁ ≤ ‖P/g − XXᴴ/t‖₁ ≤ ‖P − Q̂Q̂ᴴ‖₁/g + ‖Q̂Q̂ᴴ/g − XXᴴ/t‖₁.

(i) P − Q̂Q̂ᴴ has rank ≤ 2g (the sum of two rank-g operators), so by Lemma A and (H1) ‖P − Q̂Q̂ᴴ‖₁ ≤ 2g s, and the
first term is ≤ 2s.
(ii) XXᴴ = Q̂M²Q̂ᴴ with M² = XᴴX, whose eigenvalues m₁ … m_g lie in [1 − η, 1 + η] because ‖XᴴX − I‖₂ = η;
t = tr XᴴX = Σ_i m_i. Then Q̂Q̂ᴴ/g − XXᴴ/t = Q̂ (I/g − M²/t) Q̂ᴴ, and since Q̂ is an isometry the singular values of
this operator are |1/g − m_i/t|, so

    ‖Q̂Q̂ᴴ/g − XXᴴ/t‖₁ = Σ_i |1/g − m_i/t| = (1/t) Σ_i |t/g − m_i|.

t/g is the mean of the m_i and lies in [1 − η, 1 + η] as well, so |t/g − m_i| ≤ 2η for each i, Σ_i ≤ 2gη, and
with t ≥ g(1 − η): ‖·‖₁ ≤ 2η/(1 − η). For g = 1, m₁ = t and the sum is exactly 0.
Together 2T ≤ 2s + 2η/(1 − η), i.e. T ≤ s + η/(1 − η), and T ≤ s for g = 1. ∎

**Proof of Lemma 3.** V is nonsingular: the eigenvalues of VᴴV lie in [1 − ‖VᴴV − I‖₂, 1 + ‖VᴴV − I‖₂] ⊂
[1 − η_V, 1 + η_V] with η_V < 1 (‖·‖₂ ≤ ‖·‖_F). Ostrowski's theorem (Horn & Johnson, *Matrix Analysis*,
Theorem 4.5.9): for Hermitian ρ̂ and nonsingular S, λ_k(Sρ̂Sᴴ) = θ_k λ_k(ρ̂) with θ_k ∈ [λ_min(SSᴴ), λ_max(SSᴴ)];
with S = Vᴴ, SSᴴ = VᴴV, so θ_k ∈ [1 − η_V, 1 + η_V] ⊂ (0, 2) and the sorted orders agree. Weyl's inequality
(Horn & Johnson, Theorem 4.3.1) for the Hermitian matrices Vᴴρ̂V and diag(p), whose k-th eigenvalue is p_k
(sorted): |λ_k(Vᴴρ̂V) − p_k| ≤ ‖Vᴴρ̂V − diag(p)‖₂ ≤ w. Hence λ_k(ρ̂) = λ_k(Vᴴρ̂V)/θ_k ∈ [(p_k − w)/θ_k, (p_k + w)/θ_k].
Since λ_k(ρ̂) ≥ 0 (positive semidefinite) and ≤ 1 (trace 1, nonnegative eigenvalues), when p_k − w ≥ 0 the
smallest value of (p_k − w)/θ_k is at θ_k = 1 + η_V and the largest value of (p_k + w)/θ_k is at θ_k = 1 − η_V;
when p_k − w < 0 the lower bound 0 is used. ∎

**Proof of Lemma 4.** g_d′(T) = ln(d − 1) + ln((1 − T)/T), which is ≥ 0 iff (1 − T)/T ≥ 1/(d − 1) iff
T ≤ 1 − 1/d. ∎

**Proof of Proposition 2.** Audenaert (2007), Theorem 1: for density matrices ρ, σ on ℂ^d with
T = ‖ρ − σ‖₁/2, |S(ρ) − S(σ)| ≤ T ln(d − 1) + h(T) (the paper states the result in bits; dividing by ln 2
gives the nats form). Apply it to ρ_A, ρ̂_A (both are density matrices on ℋ_A: positive, trace 1 — P/g and
XXᴴ/t have trace 1 and the partial trace preserves the trace). By Proposition 1, T ≤ T*, and by Lemma 4,
g_{d_A}(T) ≤ g_{d_A}(T*) when T* ≤ 1 − 1/d_A (both T and T* lie in the interval where g is increasing). ∎
*Why the range is enforced.* Beyond 1 − 1/d the function decreases, so g(T*) with T < T* is not an upper bound
on g(T): for d = 2, ρ = I/2, σ = |0⟩⟨0|, T = 1/2 and |S(ρ) − S(σ)| = ln 2 = 0.693, while g₂(0.9) = h(0.9) = 0.325
(`verify_C-03.py`, part C, reproduces this). A trivial bound ln d_A would remain true but is not reported as
a certified interval: the record is NOT_CHECKED with the reason.

**Proof of Proposition 3.** ρ_A − ρ̂_A is Hermitian, so Weyl's inequality gives |λ_k(ρ_A) − λ_k(ρ̂_A)| ≤
‖ρ_A − ρ̂_A‖₂ ≤ ‖ρ_A − ρ̂_A‖₁ = 2T ≤ 2T*. ∎

**Lemma 5 (interval evaluation of f(p) = −p ln p).** f is continuous on [0, 1] with f(0) = f(1) = 0,
f′(p) = −ln p − 1, so f increases on [0, 1/e] and decreases on [1/e, 1], with maximum f(1/e) = 1/e. Hence for
0 ≤ a ≤ b ≤ 1: f([a, b]) = [min(f(a), f(b)), max(f(a), f(b))] when 1/e ∉ [a, b], and the upper end is 1/e when
1/e ∈ [a, b]. The code decides "1/e may lie in [a, b]" against an enclosure of 1/e, which can only enlarge the
upper end. Each f(a) is enclosed as −(a · ln a) with the outward dyadic arithmetic of `ivl.py` (Lemma 17 of
`docs/theory.md`: ln by the atanh series with explicit remainder, products and sums rounded outward). Sums of
enclosures enclose the sum. The binary entropy is h(T) = f(T) + f(1 − T) and ln(d − 1) is `ivl.ln_interval` of
the integer; g_d(T*) is enclosed as the product and sum of these enclosures, upper end taken. ∎

## 3 Where the code implements it

File `certify_ed/entanglement.py` (branch `track/new-statements`); every rounding below is outward (upper
bounds rounded up, lower bounds down):

* Bipartition and exact reduced matrix: `bipartition_indices` (lines 75–96) maps each basis integer to its A and
  B digit patterns (site 0 the most significant digit); `reduced_density_exact` (99–142) forms
  M = Σ_j tr_B(x_j x_jᴴ) as integers at a common dyadic scale (each binary64 entry is m·2^e exactly,
  `_ints_common_scale`, 53–72) and returns Fractions M/2^{2K} and t = tr M. No rounding: these are the exact
  rationals of (H2).
* Lemma 3: `enclose_psd_eigenvalues` (179–221). The floating-point V and p come from `numpy.linalg.eigh` of the
  rounded ρ̂ (used only as a guess); η_V (line 204) and w (line 215) are exact rationals, their square roots
  bounded above by `_frac_sqrt_upper` (42–50: (isqrt(ab) + 1)/b, exact for perfect squares); the enclosure
  ends are lines 219–220 (lower end divided by 1 + η_V, upper by 1 − η_V, clamped to [0, 1]).
* Lemma 5 and the entropy: `_f_point` (226–233), `_f_interval` (236–251; the 1/e test is at 249–250 against the
  enclosure `exp_interval(−1)`), `entropy_interval` (254–256), `binary_entropy_upper` (259–261).
* Lemma 4 / Proposition 2: `fannes_audenaert_bound` (264–275) returns None outside [0, 1 − 1/d_A] and otherwise
  the upper end of T* ln(d_A − 1) + h(T*).
* Proposition 1: `certify_entanglement`, lines 318–325: T* = s for g = 1 (line 319), s + η/(1 − η) for g > 1
  (line 324), as exact Fractions of the stored binary64 bounds s = `ec.sin_theta` and η = `ec.eta`.
* Conclusion: lines 345–349 (Ŝ ± F, clamped to [0, ln d_A] with the upper enclosure of ln d_A) and 356–360 (the
  spectrum intervals l_k − 2T*, u_k + 2T*, clamped to [0, 1]). Conversion to binary64 at the end only, outward
  (`ivl.to_float_down/up`). The computed value `value` (floating point) is reported for display and is not a
  bound.
* Hook: `certify.py` S4 loop (`r["entanglement"] = entanglement_records(...)`, one block after the correlation
  records); `entanglement_records` (384–411) handles the NOT_CHECKED paths (no spec: NEEDS_SPEC; sector
  claims: UNSUPPORTED_CLAIM; d_A above `max_dim_a`: SIZE_LIMIT; T* out of range or A not a proper subset:
  NOT_APPLICABLE with the reason) and attaches `next_action` (`diagnostics.entanglement_next_action`).

The model shift (S1) enters through s: `certify.py` adds the shift's effect to ρ before s = ρ/sep is formed
(S4 block), so T* refers to the declared model whenever S4 does.

## 4 Computational verification

`docs/proofs/verify_C-03.py` (seed 20261030, mpmath at 50 digits for the non-rational quantities, exact
rationals for η, t, the reduced matrix and the enclosures; comparison margin 10⁻⁴⁰). Run of 2026-10-30:

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | Proposition 1: exact T against s + η/(1 − η) (s = exact sin θ_max from the polar factor) | 120 random + 9 adversarial (g = 1 with ‖x‖ ∈ {0.1, 3, 50}; angles 1.2, 1.5, 1.56 rad; η = 0.91 and 0.9975 for g = 2; an exact rotated basis) | 0 |
| B | Lemma 3: the exact enclosure contains the mpmath eigenvalues of ρ̂ | 60 random (d_A ≤ 6, d_B ≤ 6, g ≤ 2, real and complex, scales 10⁻³ … 10³) + 5 adversarial (maximally entangled, product state, scales 10⁻¹⁵⁰ and 10¹⁵⁰, Schmidt weights 10⁻¹² apart) | 0 |
| C | Fannes–Audenaert on random pairs (d ≤ 6, including pairs 10⁻⁸ apart), the 12 equality cases ρ pure, σ = (1 − T)ρ + T(I − ρ)/(d − 1) (equality to 10⁻³⁸), monotonicity of g_d on a 2000-point grid of [0, 1 − 1/d] for d = 2 … 11, and the beyond-range example (reproduced) | 162 pairs + 20 000 grid steps | 0 |
| D | Proposition 3 (Weyl) on random pairs of states | 80 | 0 |
| E | enclosures of f, h, g against mpmath (700 cases incl. range enclosures over random [a, b]); end to end: the interval built from the exact sin θ of a planted rotation contains the exact eigenvector's entropy (40 random Hermitian cases, d_A, d_B ≤ 4, rotation angles up to 0.6 rad, vectors scaled by 0.5 … 2) | 740 | 0 |

Total printed by the script: 21 176 cases, 0 violations. The tests `tests/test_entanglement.py` (11 tests) check
the implementation through `certify()` against 40-digit mpmath references on the 6-site Heisenberg ring
(ground state, triplet cluster and a rotated basis of it, U(1) block with complex phases, one-site entropy
= 1 bit), planted rotations (interval contains the exact entropy; the trace-distance bound covers sin φ) and
the negative controls (T* above the range → NOT_CHECKED with reason and next_action and no interval; size
limit; bad subsets; no spec; sector claim).

## 5 Analytic cross-check

1. *Singlet, one site.* For any SU(2)-singlet eigenvector and A = one spin-1/2 site, ρ_A = I/2 exactly, so
   S(ρ_A) = ln 2 (1 bit). The 6-site ring test checks that the certified interval (log base 2) contains 1 and has
   width below 10⁻⁹; the 8-site run of the development log gave [1 − 1.6·10⁻¹¹, 1 + 2·10⁻¹⁶].
2. *Equality case of Fannes–Audenaert.* For ρ pure and σ = (1 − T)ρ + T(I − ρ)/(d − 1): S(σ) = h(T) + T ln(d − 1)
   (the eigenvalues of σ are 1 − T once and T/(d − 1) with multiplicity d − 1), so |S(ρ) − S(σ)| = g_d(T) exactly:
   the inequality is sharp and the verification reproduces the equality to 10⁻³⁸ (part C). At T = 1 − 1/d,
   σ = I/d and g_d(1 − 1/d) = ln d, the maximal possible entropy difference.
3. *Product state.* In `test_planted_rotation_inside_the_range_keeps_the_exact_entropy_inside_the_interval`
   the exact eigenvector is a product state (S = 0) and the claimed vector is rotated by sin φ = 0.15 into a state
   whose reduced density matrix across the cut A = {1} is also pure (the rotation acts on site 0), so Ŝ = 0 and
   the interval is [0, g₂(s)] with s ≥ 0.15: it contains 0, as it must; across A = {0} the same reasoning gives
   Ŝ = 0 too. The adversarial part of the verification instead rotates into the complementary subspace, where
   Ŝ ≠ S(ρ_A).
4. *Limiting case g = 1, η → 0, s → 0.* T* → 0, g(T*) → 0 and the interval collapses onto the enclosure Ŝ of the
   computed state, whose width is of the order of the exact-stage defects w and η_V (10⁻¹⁵ … 10⁻¹⁶ in the tests)
   propagated through f; the 8-site ground-state interval (A = half the chain) has width 2.4·10⁻¹¹, dominated
   by the Fannes–Audenaert term at s = 3.7·10⁻¹³ (g₈(s) ≈ s ln 7 + h(s) ≈ 10⁻¹¹ because h(s) ≈ −s ln s).

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (1 150 random + 20 026 adversarial cases, 0 violations);
independent re-derivation: pending (Phase 2); Sarang's check (D-12): pending.
