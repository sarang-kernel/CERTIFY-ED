# Theory

What each statement of CERTIFY-ED rests on. Proofs are in the appendices of the manuscript scaffold
(`ms_v3`: Appendix A for §1–§6, Appendix B for §7, Appendix C for §8). Results that are not
textbook material are marked **UNVERIFIED** until a second person has re-derived them (project rule
D-12).

Throughout, A is Hermitian, n × n, with eigenvalues λ₁ ≤ … ≤ λₙ. In a certificate, A is the operator
of the declared model on the producer's basis.

## 1. Arithmetic model (S1–S5)

Every quantity that enters a bound is computed in binary64 and then replaced by a bound on its exact
value. The model is fl(a∘b) = (a∘b)(1+δ) with |δ| ≤ u = 2⁻⁵³, which gives γ_m = mu/(1 − mu) for
length-m sums and products (Higham 2002). These bounds hold for any summation order and with FMA. They
fail for Strassen-type products, reduced-precision BLAS, flush-to-zero/denormals-are-zero and
non-default rounding. `fp.probe_float_environment` tests these properties on every run and downgrades
CERTIFIED statements when one fails.

**No-underflow hypothesis (enforced since 3.2.0.dev0; INT-01) and the underflow-tolerant Theorem 1 radius
(INT-05, track/backlog-proofs).** The model fl(a∘b) = (a∘b)(1+δ) fails for results in the subnormal range, where
the rounding error is absolute, not relative. Since track/backlog-proofs the Theorem 1 radius (`enclosure.residuals`)
and the cluster bound are evaluated under the underflow-tolerant model fl(a∘b) = (a∘b)(1+δ) + η, |δ| ≤ u,
|η| ≤ u_min = 2⁻¹⁰⁷⁵, with η = 0 for additions and subtractions (a sum that underflows is exact; derived from the
binary64 format in `docs/proofs/int05_underflow.md`): `fp.norm2_upper`/`norm2_lower` carry the term
√(N u_min)·√((1+γ_N)/(1−γ_N)) for the squares that may underflow, the residual bound carries √n·c_w·u_min for the
per-entry additive errors of A x, μ x and the bound E, and `fp.exact_dot` leaves products below 2⁻⁹⁶⁷ out of the
error-free sum and adds their possible total to its error bound. These terms are ≈ 10⁻¹⁶⁰ and change no stored
bound of a normal-sized quantity (bit identity on the recorded fixtures); for inputs with tiny magnitudes the
radii grow instead of being reported as UNDERFLOW_RISK, and they never shrink. The remaining standard-model
bounds (inertia backward errors, Rayleigh quotients, Kato–Temple arithmetic) still assume that every nonzero
operand has magnitude at least 2⁻⁹⁰⁰ (`fp.UNDERFLOW_THRESHOLD`): `certify()` scans the claimed eigenvalues, the
eigenvector entries, the operator's stored entries (for a spec or matrix-free operator: its norm bound and one
matvec on a seeded random probe) and the model shift. Exact zeros are allowed. Any finding makes S2
INCONCLUSIVE with reason code `UNDERFLOW_RISK`, and S3, S4, S7 and S8 INCONCLUSIVE with the same code, because
each rests on the S2 enclosure; S1 and the exact layer (S6) are untouched. A user who prefers to assume the
hypothesis can pass `Options(underflow_check=False)` (CLI `--assume-no-underflow`); the choice is recorded in the
certificate's `scope.options.underflow_check` and `arithmetic.underflow_check`. The radii, existence radii,
residual bounds and residual norms are no longer scanned (they were until this branch; the INT-05 known issue of
CHANGELOG 3.2.0.dev0 is closed by the additive terms). Two quantities are computed more accurately: Gram entries
and Rayleigh quotients use error-free products and exactly rounded sums, so each has error ≤ 2u|·| (plus the
tiny-product term of `exact_dot`).

**Endpoints are rounded outward (since 3.1).** The certified statement |λ − μ| ≤ r is about the stored
numbers μ and r. An endpoint μ − r computed with round-to-nearest can land half an ulp *inside* the
exact interval: for example, 1 − 2⁻⁶⁰ rounds to 1.0. Every endpoint, shift margin (σ ∓ β), separation
and cut comparison is therefore formed with `fp.sub_down` / `fp.add_up`, which step one ulp outward.
The exact layer does not need this.

## 2. Enclosures (S2)

**Theorem 1 (residual).** For x ≠ 0 and real μ, some eigenvalue satisfies |λ − μ| ≤ ‖Ax − μx‖/‖x‖.

This gives existence only: two copies of one vector satisfy it with the same eigenvalue.

**Floating-point evaluation of the Theorem 1 radius (code-facing; `enclosure.residuals`, INT-04).** The
manuscript's Appendix A is to be rewritten from this list (D-16); the code's bound is what is described
here, not what the appendix said (k ≤ 6 roundings, factor 1 + 8u, shift multiplied in). The list is proved,
with the exact error model and the scalar inequalities behind each factor, in `docs/proofs/thm1_fp-accounting.md`
(standard model) and `docs/proofs/int05_underflow.md` (additive underflow terms); a statement-level scaffold of
the appendix is `docs/proofs/appendixA_scaffold.tex`. Since track/backlog-proofs step 5 has one more rounding (the
addition of the INT-05 term), paid by the 3u slack of the norm guards, and `upper` ends with that term. Notation: A the
stored matrix, x and μ the stored claim, w = `adapter.width` the largest number of nonzeros in a row of
A, n the dimension, u = 2⁻⁵³, γ_m = mu/(1 − mu), g = `gamma_dot(w + 2)` (= γ_{w+2} for real data,
2γ_{w+6} for complex data), ‖·‖ the Euclidean norm. Every operation below is round-to-nearest; "(1 + 4u)
covers k roundings" means (1 − γ_k)(1 + 4u) ≥ 1, which holds for k ≤ 3 and for k = 4 only together with
the slack named in that step. The lines are those of `enclosure.py` on this branch.

1. *Residual vectors* (lines 149–152). R = fl(A x − fl(μ x)); each entry of fl(A x) is a BLAS or sparse
   inner product of at most w terms (error ≤ γ_w (|A||x|)_i), the product μ x_i and the subtraction add
   one rounding each. Hence |R_i − (Ax − μx)_i| ≤ γ_w (|A||x|)_i + u(1 + γ_w)(|A||x|)_i + (2u + u²)|μ x_i|,
   which is ≤ E*_i := γ_{w+2} ((|A||x|)_i + |μ||x_i|) (Higham 2002, Lemma 3.1 and §3.1; the complex case
   uses 2γ_{w+6}, which dominates Higham §3.6). The slack E*_i − |R_i − (Ax − μx)_i| is at least
   (γ_{w+2} − γ_w − u − uγ_w)(|A||x|)_i + (γ_{w+2} − 2u − u²)|μ x_i| ≥ u (|A||x|)_i + u |μ x_i| up to O(u²),
   a relative slack of order 1/(w + 2) of E*_i.
2. *Bound on |A||x|* (lines 159–163). absHX = fl(|A||x|) is a sum of w non-negative products, so
   (|A||x|)_i ≤ absHX_i/(1 − γ_w). The code divides by (1 − γ_{w+1}) instead (one extra u of slack) and
   multiplies by (1 + 4u): `absHX_up = absHX / (1 − gw) * (1 + 4u)`, gw = `gamma(w + 1)`. Roundings in
   this line: fl(1 − gw), the division, the product — three, covered by (1 + 4u). The two roundings inside
   `gamma()` (fl(1 − mu), the division) change gw by a relative 2u, i.e. 1 − gw by a relative 2u·gw/(1 − gw),
   which the γ_{w+1} slack absorbs. Result: absHX_up_i ≥ (|A||x|)_i.
3. *Entrywise rounding bound* (line 164). `E = g * (absHX_up + |X| |μ|) * (1 + 4u)`. Roundings: the
   product |x_i||μ|, the addition, the product with g, the product with (1 + 4u) — four — and two inside
   `gamma_dot` (relative 2u in g). (1 + 4u) alone covers three; the remaining relative shortfall, about
   2u + 16u², is absorbed by the slack of step 1 (E* exceeds the true entrywise error by a relative
   ≥ 1/(w + 2) ≫ 2u). Result: E_i ≥ |R_i − (Ax − μx)_i|, so ‖Ax − μx‖ ≤ ‖R‖ + ‖E‖ (triangle inequality).
4. *Norms* (line 165, `fp.norm2_upper`; line 155, `fp.norm2_lower`). For a stored vector v of length n,
   s = fl(‖v‖) = fl(sqrt(fl(Σ v_j²))) satisfies ‖v‖ (1 − γ_n)^{1/2} (1 − u) ≤ s ≤ ‖v‖ (1 + γ_n)^{1/2} (1 + u).
   `norm2_upper(v) = s · (1 + γ_{n+2}) · (1 + 8u)` and `norm2_lower(v) = max(0, s · (1 − γ_{n+2}) / (1 + 8u))`:
   the factor (1 ± γ_{n+2}) undoes the BLAS error (γ_{n+2} ≥ γ_n/2 + u + O(u²)), and (1 + 8u) = `fp._GUARD`
   covers the roundings of fl(1 ± γ_{n+2}), the product and the final product or division (three; the two
   inside `gamma()` are absorbed by the slack of γ_{n+2} over γ_n/2 + u). Hence
   `upper_i = fl(norm2_upper(R_i) + norm2_upper(E_i))` is ≥ (‖R_i‖ + ‖E_i‖)(1 − u) — the addition is the one
   rounding of this line not covered on it — and `xl_i = norm2_lower(x_i)` ≤ ‖x_i‖. (The two norms per pair
   are formed separately; the appendix's single factor on ‖R + E‖ does not correspond to a code line.)
5. *Radius* (line 170). `rel = upper / xl * (1 + 4u)`: the division and the product, plus the addition of
   step 4 — three roundings, covered by (1 + 4u). Result: rel_i ≥ (‖R_i‖ + ‖E_i‖)/‖x_i‖ ≥ ‖Ax_i − μ_i x_i‖/‖x_i‖,
   the Theorem 1 radius for the stored numbers. `xl_i = 0` (a zero vector) gives rel_i = ∞.
6. *Model shift* (certify.py; not part of `residuals`). When S1 certified the producer's matrix against the
   spec with a shift bound σ (‖H_used − H_spec‖₂ ≤ σ), the radius that refers to the declared model is
   `add_up(rel_i, σ)`, i.e. fl(rel_i + σ) stepped one ulp towards +∞ — an upper bound on the exact sum —
   wherever the radius enters a statement: the S3 enclosures (`_certify_lowest`, `_certify_window`), the
   S4 separations, the exported `radius` and `existence_radius` arrays (INT-03). The shift is *added*, not
   multiplied in; no factor (1 + 8u) is applied to it. The comparison with the accuracy target in S2
   uses round-to-nearest rel.max() + σ, since the target is a presumption, not a certified bound.

In total the code applies three factors (1 + 4u) (steps 2, 3, 5), two guards (1 + γ_{n+2})(1 + 8u) and one
guard (1 − γ_{n+2})/(1 + 8u) (step 4), and relies on the slack of γ_{w+1} over γ_w, of γ_{w+2} over the
entrywise error, and of γ_{n+2} over γ_n/2 + u where a line has more roundings than its factor covers.
Test `tests/test_theorem1_accounting.py` recomputes a fixed 3 × 3 case in exact rational arithmetic and
pins, stage by stage, that each computed quantity is at least what it bounds and at most that value
times the factors and rounding envelopes listed here.

**Theorem 2 (clusters with non-orthonormal vectors).** Let X be n × g, Λ = diag(μ) real,
R = AX − XΛ and η = ‖Xᴴ X − I‖ < 1. Set δ = 1 − √(1 − η) and ρ = ‖R‖ + (‖A‖ + ‖Λ‖)δ. Then there are g
*distinct* indices i_j with |λ_{i_j} − μ_j| ≤ ρ. Proof: the polar factor Q̂ = X(XᴴX)^{−1/2} is an isometry with
‖X − Q̂‖ ≤ δ, so ‖AQ̂ − Q̂Λ‖ ≤ ρ (polar-factor lemma), and Kahan's inclusion theorem — for an orthonormal Q and
*any* Hermitian g × g block H, the eigenvalues of H are matched to distinct eigenvalues of A within ‖AQ − QH‖
(proof by the Courant–Fischer interval count and Hall's theorem) — gives the matching. For orthonormal
Ritz pairs this is Kahan's theorem with ρ = ‖R‖. Until track/backlog-proofs the radius was
ε = (w + √(w² + 4ρ²))/2 ≥ ρ (the Weyl-perturbation proof of the manuscript appendix); ρ is never worse.
Derivation, rounding accounting and verification: `docs/proofs/thm2_polar-factor.md` (D-59; Sarang's check
D-12 pending).

**Theorem 2′ (sharper cluster radius; since 3.1.0, review finding IMP-13).** Under the hypotheses of
Theorem 2 (η < 1; Λ Hermitian, in the implementation Λ = diag(μ) real), with δ = 1 − √(1 − η) and
spread(Λ) = λ_max(Λ) − λ_min(Λ), there are g distinct indices i_j with

    |λ_{i_j} − μ_j| ≤ ε* := ‖R‖/√(1 − η) + ‖Xᴴ R‖/(1 − η) + δ·spread(Λ)/√(1 − η).

`enclosure._cluster_bound` computes both ρ and ε* (each term an upper bound, guarded with `fp.up` and the
INT-05 additive terms) and uses min(ρ, ε*) as the cluster radius; both are proven, so the min is sound. For
exact eigenvectors of one degenerate level ε* is of the order of the residual, where ρ is dominated by
(‖A‖ + ‖Λ‖)δ. Derivation and verification: `docs/proofs/thm2prime_eps-star.md` (D-59; Sarang's check D-12
pending; the earlier numerical evidence was 15 876 random cases, `thm2_checks.py`, review of 2026-10-21).

*Proof sketch.* Polar decomposition X = Q̂M with M = (Xᴴ X)^{1/2}; the eigenvalues of M lie in
[√(1−η), √(1+η)], so ‖M − I‖ ≤ 1 − √(1−η) = δ and ‖M⁻¹‖ ≤ (1−η)^{−1/2}.
(i) *Kahan.* H′ := Q̂ᴴ A Q̂ is Hermitian. From AX = R + XΛ, A Q̂ = R M⁻¹ + Q̂ M Λ M⁻¹, hence
(I − Q̂Q̂ᴴ) A Q̂ = (I − Q̂Q̂ᴴ) R M⁻¹ and ‖A Q̂ − Q̂ H′‖ ≤ ‖R‖/√(1−η). Kahan's theorem matches the eigenvalues
θ_1 ≤ … ≤ θ_g of H′ to g *distinct* eigenvalues of A within ‖R‖/√(1−η).
(ii) *Weyl.* Using Q̂ᴴ = M⁻¹Xᴴ, H′ − Λ = M⁻¹ Xᴴ R M⁻¹ + (MΛM⁻¹ − Λ), and MΛM⁻¹ − Λ = [M, Λ]M⁻¹ =
[M − I, Λ − cI] M⁻¹ for any scalar c; with c the midpoint of the spectrum of Λ, ‖Λ − cI‖ = spread/2, so
‖H′ − Λ‖ ≤ ‖Xᴴ R‖/(1−η) + δ·spread/√(1−η). H′ − Λ is Hermitian, so Weyl's inequality pairs the sorted
eigenvalues of H′ and Λ within this bound.
(iii) Composing the two matchings gives distinct eigenvalues of A within ε* of the eigenvalues of Λ. ∎

Groups whose Theorem 1 intervals overlap get Theorem 2 / 2′. If η ≥ 1, the group keeps existence only and
S2 is INCONCLUSIVE.

**What each radius certifies (since 3.1.0).** A cluster member i carries two radii. Its *existence
radius* is the Theorem 1 radius r_i (an eigenvalue lies within r_i of μ_i). The *cluster radius* ε is the
one under which the members are matched to *distinct* eigenvalues; it is also the radius used by every
indexed statement (S3's "λ_K lies within … of μ_K", Theorem 8, Proposition 9), because the matching of
Theorem 2 need not be the one that puts the eigenvalue of r_i's interval at the sorted position of μ_i.
The accuracy presumption of S2 is therefore applied in two steps: if ε meets the target, S2 is
CERTIFIED; if only the existence radii meet it, the claim is accurate but its vectors are too far from
orthonormal to certify the count at that accuracy, and S2 is INCONCLUSIVE with that diagnosis (a correct
claim with valid non-orthogonal eigenvectors of a degenerate level was FAILED before 3.1.0, IMP-13);
if an existence radius misses the target, S2 is FAILED.

**Theorem 3 (Kato–Temple).** With ‖x‖ = 1, θ = x^H A x, r = ‖Ax − θx‖, and (α, β) containing exactly one
eigenvalue λ: θ − r²/(β − θ) ≤ λ ≤ θ + r²/(θ − α). The isolating interval comes from S3, and the
adjacent levels are used (tests pin this).

**Matrix-free operators (D-02, since 3.2; `matrixfree.MatrixFreeOperator`, `Options(matrix_free=True)`).** When
the claim carries no operator and the spec is applied term by term, the Theorem 1-fl accounting uses a rounding
count derived from the spec: W = w_row + L + ρ + 1 (real data; w_row + 3L + ρ + 3 complex), where w_row bounds the
number of monomial contributions to one output entry (Σ_t Π_l c(o_{t,l}); c(o) = 1 for every spin-1/2 and fermion
letter, so w_row = n_terms there), L the longest string and ρ the number of letters per string whose stored
amplitudes are rounded (spin ≥ 1 ladder/x/y, boson ladders; exact for spin-1/2, fermions, S^z, n). The radius then
bounds ‖H_spec x − μx‖/‖x‖ for the *exact* spec operator (the amplitude roundings are inside W), so no model shift is
added; S1 records the amplitude hypothesis by name. ‖H‖₂ ≤ Σ_t |c_t| Π_l ‖o_{t,l}‖₂ (tensor-product and unitary
invariance of the Jordan–Wigner string) is combined with the Appendix-A row-sum bound by a minimum (both proven).
Derivation and verification (182 random specs + 18 adversarial cases at 60 digits, 0 violations):
`docs/proofs/D-02_matrix-free.md`. S3 and S8 keep needing a stored matrix (NEEDS_MATRIX).

## 3. Counting (S3)

**Lemma 4 (inertia with backward error).** If ‖LDL^H − P(A − σI)P^T‖ ≤ β for unit lower-triangular L
and block-diagonal D with ν negative eigenvalues, then #{λ < σ − β} ≤ ν ≤ #{λ ≤ σ + β}. ν is counted
exactly: 2 × 2 blocks by the sign of a rational determinant.

**Corollary 5 (completeness).** If k claimed values have distinct enclosures inside (−∞, σ − β) and ν = k,
they enclose λ₁…λ_k, and λ_{k+1} ≥ σ − β. The pairing is the sorted one — λ_j in the j-th sorted enclosure —
because the enclosure groups have disjoint union intervals and the members of a cluster group share the
cluster radius (sorted-pairing lemma; it fails for non-uniform radii). With a model shift Δ the radii and β
grow by Δ (Weyl). Derivation and verification: `docs/proofs/cor5_pairing.md` (D-59; Sarang's check D-12 pending).

**Backward error β of the sparse path (D-01, since 3.2; `inertia.inertia_sparse`, `ldl_residual.py`).** For the
SuperLU symmetric-mode factorization the 3.1 code bounded β a priori by γ_{w+1}‖L‖_F‖U‖_F + ‖L‖_F‖U − DLᴴ‖_F + …
(Higham 2002, Thm 9.3), which assumes that SuperLU forms every entry of L and U by a standard inner-product update
— a property of the library's supernodal kernels that is not established (known issue MS-M11). Since 3.2 β is
measured: the residual E = L D Lᴴ − P(A − σI)Pᵀ of the *stored* factors is formed in floating point in row blocks and
its rounding bounded entrywise by |E_ij| ≤ |E_c,ij|/(1 − u) + γ_{t_ij+1}(|L||D||L|ᵀ)_ij, t_ij ≤ min(c_i, c_j) (the number
of stored nonzeros in the rows); ‖E‖₂ ≤ ‖E‖_∞ (E Hermitian) gives β. Lemma 4 then applies with no hypothesis on how
the factors were computed; an unstable factorization (no pivoting) shows as a large β. Derivation, code map and
verification (400 random + 17 adversarial cases in exact rational arithmetic, 0 violations):
`docs/proofs/D-01_sparse-residual.md`. The a-priori bound remains available as `inertia_method='sparse-apriori'`
(conditional). The factorization is not attempted when `cost.sparse_inertia_mb` (fill model plus the residual's
arrays) exceeds the memory budget: NOT_CHECKED, never a count.

**Vectorless window count (C-02, since 3.2.0.dev0; `certify._certify_window_vectorless`).** For a window
claim without eigenvectors there are no enclosures, so Corollary 5 does not apply; the code certifies only
the *number* of eigenvalues in (a, b), with no pairing to the claimed values. It uses Lemma 4 and the count
inequality of Proposition 14, nothing else:

1. Lemma 4 at an edge e gives ν_e with #{λ < e − β_e} ≤ ν_e ≤ #{λ ≤ e + β_e}; the two ends agree, and equal
   #{λ < e}, as soon as the collar [e − β_e, e + β_e] holds no eigenvalue.
2. Two more counts at σ₁ = e − δ and σ₂ = e + δ (δ > β) give, by the first inequality of Proposition 14 on the
   window (σ₁, σ₂), #{λ ∈ (σ₁ + β₁, σ₂ − β₂)} ≤ ν₂ − ν₁. When ν₂ = ν₁ and the collar of step 1 lies inside
   (σ₁ + β₁, σ₂ − β₂), the collar is eigenvalue-free and ν_e = #{λ < e} exactly; e is then not an eigenvalue.
   When ν₂ > ν₁, an eigenvalue lies in [σ₁ − β₁, σ₂ + β₂] and the statement is INCONCLUSIVE ("window edge
   within ... of an eigenvalue"). δ starts at max(8 β_e, 32 u·scale) and is widened (at most three retries)
   when a backward error is not smaller than the distance to the collar.
3. With both edges exact, ν_b − ν_a = #{λ ∈ [a, b)} = #{λ ∈ (a, b)}. CERTIFIED when this equals the number of
   claimed values, FAILED otherwise (`missing_at_least` or `spurious_at_least`). Endpoint arithmetic uses
   `fp.add_up` / `fp.sub_down`; the S1 model shift is added to every β as in the other S3 paths.

## 4. Degenerate levels (S4)

**Theorem 6 (projector).** If an interval contains the enclosures of a cluster and exactly g
eigenvalues (spectral projector P), at distance ≥ sep from the rest, and s = ρ/sep < 1, then
‖Q̂Q̂^H − P‖ ≤ s, ‖XX^H − P‖ ≤ s + η, and |tr(X^H O X) − tr(PO)| ≤ 2g‖O‖(s + η).

**Proposition 7 (quantum numbers).** For [A, S] = 0 with eigenvalues q_j of S on range P, the
eigenvalues m_j of X^H S X satisfy |m_j − q_j| ≤ 2s²‖S‖ + (2κ + κ²)(1 + η)‖S‖, κ = (1 − η)^(−1/2) − 1.
On a lattice of spacing h, a bound below h/2 snaps to the exact label. Total spin S(S+1) is certified
through the Casimir (`su2.py`). Momentum labels are audited through ‖GV − λV‖ with a leakage bound
(`sectors.audit_label`).

## 5. Symmetry sectors

**Theorem 8 (assembly).** Let V₁…V_m be sector isometries with Σd_s = n, W = [V₁ ⋯ V_m] and B = ⊕H_s,
with η = ‖W^H W − I‖ < 1. Let every sector's claims below E_cut be certified and counted. Then the pooled
claimed values match λ₁…λ_K within r + ε, and λ_{K+1} ≥ E_cut − max β_s − ε (`compose.py`).

*ε for a square W (since 3.1.0, review IMP-12/IMP-20).* Because Σd_s = n is checked first, W is square
and its polar factor Q̂ is unitary. With R̂ = HQ̂ − Q̂B = (HW − WB) + H(Q̂ − W) − (Q̂ − W)B and
‖Q̂ − W‖ = ‖I − M‖ ≤ δ, ‖R̂‖ ≤ ‖HW − WB‖ + (‖H‖ + ‖B‖)δ = ρ. Q̂ᴴ H Q̂ − B = Q̂ᴴ R̂ is Hermitian and
Q̂ᴴ H Q̂ has the spectrum of H, so Weyl's inequality pairs the sorted eigenvalues of H and B within ρ.
`compose.py` therefore uses ε = ρ (+ the builder shift) where it used (1 + √5)/2 · ρ before. Here H is the
symmetrised built matrix (`construct.hermitian_part`), whose distance to the declared model is the builder
shift (Appendix A). Derivation (with Corollary 5 for each sector and the sorted-pairing lemma) and
verification: `docs/proofs/thm8_rho.md` (D-59; Sarang's check D-12 pending). `sectors.decomposition_shift` and
`certify._subspace_shift` (a single sector against the full operator, Theorem 2 with X = V) also return ρ since
track/backlog-proofs.

## 6. Whole-spectrum enclosure (`spectrum.py`, formerly symveig)

`enclose_spectrum(H, S_diag)` encloses every eigenvalue.

1. **Sectors.** If S_diag is given, conservation is checked *exactly*: no stored entry H_ij ≠ 0 may join
   S_i ≠ S_j. Without this check, the sector spectra are not the spectrum of H. symveig 1.0.0 did not
   check it (see `symveig_audit.md`, P2).
2. Each block is diagonalised and enclosed by §2, with |A||x| rounding bounds.
3. **Counts.** A merged chain of intervals is *counted* only if Theorem 2 applies. The enclosure is
   *complete* when every interval is counted and the counts add up to n. Then the union contains every
   eigenvalue, and each interval holds exactly its count (pigeonhole). Without Theorem 2 a merged
   interval holds *at least one* eigenvalue. symveig 1.0.0 reported its count without proof (P1).
4. Intervals from different sectors that overlap are merged, and their counts add.

Why sectors tighten worst-case bounds: a bound of the form γ₄ₙ‖A‖_F‖x‖ grows with the dimension
(h ≈ 6 u d ‖A‖_F, symveig's Figure 1). The rounding term of a |A||x| bound with row-width γ factors
does not, so what remains is mostly the eigensolver's own residual. Experiment E8 measures both. The
widths still grow with d, but more slowly: the fitted median growth is about d^1.45 for symveig against d^0.97
(run of 2026-10-18; `fig_widths.pdf`).

## 7. Exact layer (S6, `checker.py`, `exact_sectors.py`) — **UNVERIFIED**

**Scaled basis.** For each local space there are integer weights w_k such that D = diag(√w_k) makes
every operator letter rational in D⁻¹·D:
- spin S: w_k = binom(2S, k), so S⁺ → k+1 and S⁻ → 2S−k+1;
- bosons: w_k = k!, so b† → 1 and b → k;
- fermions and spin-1/2: w_k = 1.

The many-body weights are products over sites, and every weight is ≥ 1. The declared model, with its
binary64 couplings taken exactly, becomes K = D⁻¹HD = (Re + i Im)/den with integer entries.
WK = DHD is Hermitian and is checked exactly. The spin weights are symmetric under k ↔ 2S − k, so
the spin flip remains a symmetry of K.

**Exact radii.** For a claimed vector x, take y = fl(x/√w) and the test vector x′ = Dy, which equals x
up to one rounding. Then ‖Hx′ − λx′‖² = Σ_r w_r|(Ky − λy)_r|² and ‖x′‖² = Σ_r w_r|y_r|² are rational.
The radius is bounded above with an integer square root. Theorem 1 applies.

**Exact counts.** At g = a/b, M = den·D(bH − aI)D = b·W·K_den − a·den·W is an integer symmetric matrix
congruent to b·den(H − g). A floating-point LDL^T proposes factors, and they are moved to dyadic grids.
The residual LtDtLt^T − M is computed exactly with FLINT, block of rows by block of rows, giving β.
Because D ≥ I, Lemma 4 gives #{λ < g − δ} ≤ ν ≤ #{λ < g + δ} with δ = β/(b·den).

**Exact 'k lowest' (S6; Proposition 13 in ms_v3; corrected in 3.1).** Merge the exact radii into
components C_1 < … < C_m, with m_j claimed values in C_j and K_j = m_1 + … + m_j. Count at one point
in every gap and one above C_m. Each component with m_j ≥ 2 is also bracketed by a point just below and
a point just above it, including below C_1. If every count equals the number of claimed values below
its point, and no component meets any window [g − δ, g + δ], then:
- for m_j = 1, λ_{K_j} ∈ C_j. The counts on the two sides pin the index, so no distinctness is needed;
- for m_j ≥ 2, λ_{K_{j−1}+1..K_j} lie in the bracket [a_j − δ, b_j + δ] (a "level enclosure");
- λ_{k+1} ≥ g_top − δ.

The per-vector residual intervals give *an* eigenvalue each, not distinct ones. Before 3.1 a single
count above a multi-value component was taken to mean that the component held all its claimed
eigenvalues. Two copies of the first-excited vector claimed as "the two lowest" were then certified,
although the ground state is missing (now a test). FAILED is reported only when an unclaimed
eigenvalue lies under a claimed one. An excess above the top component triggers one count just
below it; a cut multiplet is INCONCLUSIVE.

**Exact sectors.** Symmetry generators are site permutations and the spin flip, with ±1 phases, and
they must preserve the weights; commutation with K and weight invariance are checked exactly. For a
one-dimensional character with values in {±1, ±i}, the orbit sums v = Σ_g conj(χ(g)) φ_g(s) e_{gs} are
Gaussian-integer vectors. They are mutually orthogonal, with ‖Dv‖² ≥ 1. The blocks v^H (W K_den) v′
are Gaussian-integer Hermitian; a complex block is counted through its real embedding (every
eigenvalue twice). Writing D·V = QN with N ≥ I keeps the δ of Lemma 4. Non-abelian groups are
allowed when every irrep met is one-dimensional, for example k = 0 or π with reflection and spin
flip. Completeness is checked by summing the sector dimensions. `exact_count_below` gives certified
lower bounds on λ_min in this way. This is how the pyrochlore application works.

## 8. Applications

Pyrochlore crown, S = 1/2 (`apps/pyrochlore/spin_half.py`):
- Kránitz & Penc's argument gives e_GS ≥ λ_min(H_crown). That argument is taken from the paper.
- We use K·H with integer couplings, K = 8·10⁷, which satisfies the embedding constraint exactly.
- SU(2): every multiplet has an S^z = 0 member.
- Z₂⁷ blocks (128 characters) and exact counts (§7) give λ_min ≥ −0.55015829882… (E9, run of
  2026-10-21, shift at relative margin 1e-10; earlier certificates gave −0.550158313 and
  −0.55015831338…; λ_min ≈ −0.5501582979).

Pyrochlore crown, S = 1 (`spin_one.py`):
- The outer-pair spins S̃ are conserved, so the all-singlet block equals −12c₁ + c₁₇·H_ring(spin 1, 6
  sites).
- E₀(ring) is enclosed by S2 and S3, which gives a certified eigenvalue below the published bound.

AKLT (`apps/aklt`):
- The criterion η < 1/z̃ (square-octagon: 4η⊥ + 2η∥ < 1) is taken from Pomata & Wei.
- The kernel lemma, ker H_Γ = VBS range, is a Schwinger-boson divisibility argument (`docs/applications.md`,
  **UNVERIFIED** here). It is confirmed by exact mod-p nullities for the star, square-octagon and
  decorated-honeycomb subgraphs.
- η is bounded through the bordered pencil matrix [[−pG_A, qC], [qCᵀ, −qG_B]] (Haynsworth inertia):
  t′ = (1 + β/μ)(t + β/ν).
- The number of unit cosines is bounded below by a mod-p rank of the union VBS states.
- Symmetry blocking is an exact congruence (`apps/aklt/eta.py` docstring).

## 9. Statements added in 3.1 (written before their code, rule D-46) — derived in 3.2, independent re-derivation pending

Every result in this section was marked **[UNVERIFIED – re-derivation pending, D-12]** in 3.1. In 3.2 (decision
D-59) each has a derivation document in `docs/proofs/` with a verification script in exact rational arithmetic
or mpmath at 200 digits, and the proofs below are superseded by those documents: Proposition 14 →
`docs/proofs/P14_degeneracy_bracketing.md`, Proposition 15 → `docs/proofs/P15_summed_bounds.md`, Proposition 16 →
`docs/proofs/P16_thermo_intervals.md`, Lemma 17 → `docs/proofs/L17_dyadic_exp_ln.md`. The independent
re-derivation (Phase 2) and Sarang's check (D-12) are pending for all four; the code strings therefore keep the
word UNVERIFIED with that meaning. Two corrections found by the derivations are already in the code: the atanh
remainder constant of Lemma 17 (8/7 for u ≤ 0.35 instead of 9/8 for u ≤ 1/3, because the rounded upper argument
can exceed 1/3 by one grid unit) and the moduli |a_k| of Proposition 15 (an IEEE sqrt bound instead of a libm
hypot). The code in `degeneracy.py`, `correlations.py`, `ivl.py` and `thermo.py` implements exactly the
statements below; the tests check the implementation against the statements, not the other way round.

### 9.1 Degeneracy count (S8, `degeneracy.py`)

Setting: a cluster C of g claimed pairs whose enclosures are certified distinct (Theorem 2), with
level enclosure J = [J_L, J_U] containing all g enclosures. Two inertia counts (Lemma 4) at shifts
σ_lo < J_L and σ_hi > J_U give ν_lo, β_lo and ν_hi, β_hi.

**Proposition 14 (bracketing count).** Put I_in = (σ_lo + β_lo, σ_hi − β_hi) and
I_out = [σ_lo − β_lo, σ_hi + β_hi], and assume σ_lo + β_lo < σ_hi − β_hi (I_in is not empty; without this
hypothesis the first inequality fails, see the 1 × 1 example in the Lean formalization,
docs/lean/COUNTING.md; corrected 10 Nov 2026, the hypothesis is in docs/proofs/P14_degeneracy_bracketing.md and the
code uses the inequality only when J ⊂ I_in, which implies it). Then

  #{λ ∈ I_in} ≤ ν_hi − ν_lo ≤ #{λ ∈ I_out}.

If J ⊂ I_in and ν_hi − ν_lo = g, then I_in contains exactly g eigenvalues, and they are the g distinct
eigenvalues matched to C by Theorem 2. If ν_hi − ν_lo > g, at least one further eigenvalue lies in
I_out, and the degeneracy of the level is INCONCLUSIVE.

*Proof.* Lemma 4 gives #{λ < σ_hi − β_hi} ≤ ν_hi ≤ #{λ ≤ σ_hi + β_hi} and
#{λ < σ_lo − β_lo} ≤ ν_lo ≤ #{λ ≤ σ_lo + β_lo}. Subtracting the upper bound on ν_lo from the lower bound
on ν_hi gives ν_hi − ν_lo ≥ #{λ < σ_hi − β_hi} − #{λ ≤ σ_lo + β_lo} = #{λ ∈ I_in}; subtracting the other
way gives ν_hi − ν_lo ≤ #{λ ∈ I_out}. If J ⊂ I_in, the g distinct eigenvalues of Theorem 2 lie in I_in,
so #{λ ∈ I_in} ≥ g; with ν_hi − ν_lo = g the first inequality gives #{λ ∈ I_in} ≤ g. ∎

The certified statement is "exactly g eigenvalues in I_in" (a count in an interval of stated width),
never "g exactly equal eigenvalues": exact equality is not decidable in floating point and is left to
the exact layer. Shifts are moved outward (at most three retries, as in S3) when a backward error β is
not smaller than the distance from the shift to J. All endpoint arithmetic uses `fp.add_up`/`sub_down`.
When the operator used for the counts differs from the declared model by at most ε (S1 model shift),
ε is added to β_lo and β_hi, as in S3.

### 9.2 Linear combinations of certified observables (S4 extension, `correlations.py`)

**Proposition 15 (summed bounds).** Let P be the spectral projector of a certified cluster
(Theorem 6) and O_1 … O_m Hermitian operators with certified values v_k = tr(X^H O_k X) and bounds
b_k ≥ |tr(P O_k) − v_k| (Theorem 6, including the rounding term of `certify_observable`). For complex
coefficients a_1 … a_m,

  | Σ_k a_k tr(P O_k) − Σ_k a_k v_k | ≤ Σ_k |a_k| b_k.

*Proof.* Triangle inequality termwise. ∎ The sum Σ|a_k| b_k is formed with `fp.add_up` after rounding
each product |a_k| b_k upward; the computed value Σ a_k v_k carries its own rounding, bounded by
γ_{m+1} Σ |a_k||v_k| (standard model), which is added to the bound.

Uses:

* **Batch of observables.** m independent Theorem 6 records; no new mathematics.
* **Two-point functions.** For site operators O_i, O_j acting on different sites, O_i O_j is Hermitian
  when O_i and O_j are Hermitian (they commute). C_ij = tr(P O_i O_j) is then a Theorem 6 observable
  with O = O_i O_j; for i = j, O = O_i². For a single eigenvector (g = 1) tr(P O) is the expectation
  value; for g > 1 it is the trace over the level (basis independent, D-14), and the per-state
  average is tr(P O)/g with bound b/g.
* **Static structure factor.** S(q) = (1/N) Σ_{i,j} e^{iq·(r_i − r_j)} C_ij is Proposition 15 with
  a_ij = e^{iq·(r_i−r_j)}/N, so |a_ij| = 1/N and the bound is (1/N) Σ_ij b_ij. The real part is reported;
  S(q) is real when C_ij = C_ji, which holds for commuting Hermitian site operators. Only the pairs i ≤ j
  are computed; C_ji = C_ij is used for the rest (exact for commuting Hermitian O_i, O_j).

* **Non-Hermitian operators (C-05, since 3.2.0.dev0).** For a non-Hermitian O (S⁺_i S⁻_j, or a user
  observable) the classical decomposition O = H + iK with H = (O + Oᴴ)/2 and K = (O − Oᴴ)/(2i), both
  Hermitian, gives tr(P O) = tr(P H) + i tr(P K); each trace is a Theorem 6 observable and the recombination
  is Proposition 15 with coefficients 1 and i (bound b_H + b_K). The code forms H and K in floating point and
  checks, entry by entry, that the sums are exact (Knuth's TwoSum error term is zero; the halving and the
  multiplication by −i are exact), so that the statement refers to O itself; when a sum rounds, the record is
  INCONCLUSIVE. For two-letter correlation requests C_ji = conj(C_ij) (tr(P Oᴴ) = conj tr(P O)).

### 9.3 Thermodynamics from a complete whole-spectrum enclosure (S7, `thermo.py`)

**Hypothesis (H).** Intervals [l_k, h_k] with counts c_k, k = 1 … m, Σ c_k = n, such that every
eigenvalue of A lies in some interval and interval k contains exactly c_k eigenvalues (counted with
multiplicity). This is the `complete = True` output of Proposition 9 (`spectrum.py`); without it no
thermodynamic statement is made. k_B = 1, T > 0, β = 1/T.

Write E_0 = min_k l_k and e = E − E_0 ≥ 0 for every eigenvalue, so each eigenvalue in interval k has
e ∈ [a_k, b_k] with a_k = l_k − E_0 ≥ 0 and b_k = h_k − E_0.

**Proposition 16 (interval bounds).** Under (H), with the shifted Boltzmann weight intervals
W_k = c_k · [exp(−β b_k), exp(−β a_k)] (exp is decreasing, so the ends are correct) and interval
arithmetic (+, ×, ÷ on intervals; Z > 0):

  Z̃ = Σ_k W_k ∋ Z e^{βE_0},         Z = Σ_i e^{−βE_i},
  F = −T ln Z = E_0 − T ln Z̃,
  ⟨e⟩ ∈ (Σ_k W_k [a_k, b_k]) / Z̃,   U = ⟨E⟩ = E_0 + ⟨e⟩,
  Var(E) = ⟨e²⟩ − ⟨e⟩² ∈ (Σ_k W_k [a_k², b_k²]) / Z̃ − ⟨e⟩²,  intersected with [0, ∞),
  C = β² Var(E).

*Proof.* Each eigenvalue E_i of interval k contributes e^{−βE_i} = e^{−βE_0} e^{−βe_i} with
e_i ∈ [a_k, b_k], so its weight lies in e^{−βE_0}·[exp(−βb_k), exp(−βa_k)]; summing the c_k members of
interval k and then over k gives Z e^{βE_0} ∈ Σ_k W_k. The same termwise argument gives
Σ_i e_i e^{−βe_i} ∈ Σ_k W_k [a_k, b_k] and Σ_i e_i² e^{−βe_i} ∈ Σ_k W_k [a_k², b_k²] (a_k ≥ 0, so
e ↦ e² is monotone on the interval). ⟨e⟩ and ⟨e²⟩ are the quotients by Z e^{βE_0}; the quotient of
enclosures encloses the quotient. Var(E) = Var(e) ≥ 0 and C = β² Var(E) are the textbook identities.
F follows from Z = e^{−βE_0} Z̃ and monotonicity of ln. ∎

The bound on Var(E) and C can be loose: the quotient and the difference ⟨e²⟩ − ⟨e⟩² are formed with
intervals that ignore the correlation between numerator and denominator, so the width of the C
interval is not of the order of the eigenvalue widths but of β² (b_k − a_k) times the energy scale,
summed. The certificate states this. When every interval is a single point up to rounding (widths of
order 1e−13 in units of ‖A‖), the loss is negligible at moderate β; at very large β (T → 0) the
enclosure of C remains correct but may be wide.

**Lemma 17 (dyadic enclosures of exp and ln, `ivl.py`).** All arithmetic in Proposition 16 is done on
intervals whose ends are dyadic rationals p·2^{−P} (P = 192 bits), with every operation rounded outward
(floor for lower ends, ceiling for upper ends). For x ≥ 0 the enclosure of exp(x) is obtained from the
Taylor polynomial of degree N at t = x/2^s ≤ 1/2 with remainder 0 ≤ R ≤ t^{N+1}/(N+1)! · 1/(1 − t/(N+2)),
followed by s squarings (monotone on [1, ∞)); exp(−x) = 1/exp(x) with outward division. For z > 0 write
z = 2^k·m with m ∈ [1, 2) and ln z = k ln 2 + 2 atanh(u), u = (m − 1)/(m + 1) ∈ [0, 1/3); the series
2 Σ_{j≤N} u^{2j+1}/(2j+1) has remainder 0 ≤ R ≤ 2u^{2N+3}/((2N+3)(1 − u²)) ≤ (16/7) u^{2N+3}/(2N+3) for the
rounded upper argument u ≤ 0.35; ln 2 = 2 atanh(1/3) likewise.
Each enclosure is rounded outward to the dyadic grid at every step. Conversion of a dyadic end to
binary64 rounds outward (`ivl.to_float_down/up`). No floating-point operation enters S7 except this
final conversion and the reading of the input floats l_k, h_k, which are taken as the exact binary64
values they are.

*Proof.* Standard Taylor remainders for exp (alternating-free, all terms positive) and for atanh on
[0, 1/3); outward dyadic rounding preserves enclosure at each step; squaring and division of
positive intervals are monotone. ∎ (Bit lengths stay bounded because every intermediate is rounded to
P bits; the enclosures are therefore not the tightest possible, only correct.)

### 9.4 Open: continued-fraction bounds from certified moments (C-07, second part; not implemented)

The first part of C-07 (`moments.py`, §10.7) certifies the moments m_k = tr(P O Hᵏ O)/g of a local operator on a
certified level for k ≤ K. What remains open is turning K + 1 certified moment intervals into certified bounds on the
spectral function A(ω) = Σ_n |⟨n|O|φ⟩|² δ(ω − E_n) (its Stieltjes transform, or its integrated weight on an interval). The
mathematics needed: the Lanczos three-term recurrence coefficients (a_j, b_j) are rational functions of the moments
(Hankel determinants), so interval moments give interval coefficients only while the Hankel matrices stay certified
positive definite; the Chebyshev–Markov–Stieltjes inequalities then bound the integrated spectral weight between
consecutive Gauss nodes, and a Gauss quadrature of the truncated continued fraction gives a two-sided bound on
∫ f dA for functions f with a sign-definite (2K+2)-th derivative. Each step has a floating-point accounting of its own
(the recurrence is numerically unstable from moments; a stable route works from the Krylov vectors H^k Oφ themselves
with certified orthogonalisation defects). None of this is implemented; the record's statement M4 says so
(NOT_IMPLEMENTED), and `docs/SCOPE_MATRIX.md` lists it as the open slot.

## 10. Statements added in 3.2 (derived under D-59; documents in `docs/proofs/`)

Each statement below has a derivation document with exactly the sections Statement, Proof, Where the code
implements it, Computational verification, Analytic cross-check and Status, and a verification script
`docs/proofs/verify_<id>.py`. All are **[derived; independent re-derivation done (not a human check, `docs/proofs/VERDICTS.md`); Sarang's check pending, D-12]**.

### 10.1 Entanglement entropy and spectrum (C-03, `entanglement.py`, `docs/proofs/C-03_entanglement.md`)

For a certified cluster (Theorem 6: ‖Q̂Q̂ᴴ − P‖₂ ≤ s, η = ‖XᴴX − I‖₂) and a declared cut A|B, with
ρ_A = tr_B(P)/g and ρ̂_A = tr_B(XXᴴ)/tr(XᴴX): T = ‖ρ_A − ρ̂_A‖₁/2 ≤ T* = s + η/(1 − η) (= s for g = 1), by the
rank-2g trace-norm bound on P − Q̂Q̂ᴴ and the contractivity of the partial trace; |S(ρ_A) − S(ρ̂_A)| ≤
T* ln(d_A − 1) + h(T*) (Fannes–Audenaert, Audenaert 2007) provided T* ≤ 1 − 1/d_A, where the right-hand side is
increasing in T; |λ_k(ρ_A) − λ_k(ρ̂_A)| ≤ 2T* (Weyl). The reduced matrix of the stored vectors is exact
(rational), its spectrum is enclosed by Ostrowski + Weyl around a floating-point eigendecomposition with exactly
computed defects, and x ln x, h and ln are evaluated on the dyadic intervals of Lemma 17.

### 10.2 Correlations through the sector isometry (C-04, `correlations.py`, `docs/proofs/C-04_correlation_lift.md`)

For a symmetry-adapted claim (V, H_s) audited by S1 (ι ≥ ‖VᴴV − I‖, ν ≥ ‖HV − VH_s‖) and a cluster certified
by Theorem 6 in the block (P_s, ‖XXᴴ − P_s‖ ≤ s + η): tr(P_s VᴴOV) = tr(Π_V O) with Π_V = VP_sVᴴ (exact), and
with Q the polar factor of V and Π_Q = QP_sQᴴ (an orthogonal projector of rank g),
|tr(Π_V O) − tr(Π_Q O)| ≤ 2g‖O‖₂δ(1 + √(1+ι)), δ = 1 − √(1−ι) (rank-2g bound on ‖Π_V − Π_Q‖₁ and Hölder). The
compression VᴴOV is formed in floating point with the standard-model bound γ̇_{w_O+w_V}‖|V|ᵀ|O||V|‖_F +
u‖Õ_fl‖_F on its error, which enters multiplied by ‖P_s‖₁ = g. The certified statement is about Π_Q; whether
range Π_Q is an H-eigenspace rests on ν (recorded), and other sectors may share the level.

### 10.3 Certified parameter intervals (E-04, `sweep.py`, `docs/proofs/E-04_parameter_intervals.md`)

For the affine interpolant F_k(c) = H(c_k) + (c − c_k)H1 of the certified grid Hamiltonians, with B ≥ ‖H1‖₂ (row-sum
bound of the built derivative plus the builder shift, or the triangle bound Σ|b_t|Π‖o‖_∞ over the terms), Weyl's
inequality gives |λ_j(F_k(c)) − λ_j(H(c_k))| ≤ B|c − c_k| for every sorted index j, so the S2 enclosures at c_k widen
to tubes over the segment with S3's ranks preserved (the count below a shifting threshold moves by at most the
widening), and the neighbouring certificate's tube, shifted by the affinity defect d_k ≥ ‖H(c_{k+1}) − F_k(c_{k+1})‖₂
(zero for coefficients affine in binary64; exact rational check), is intersected with it. Two tubes are certified
apart on a sample interval of width D when max of the end margins + BD < 0 (the margin is 2B-Lipschitz); a
not-separated run flanked by opposite certified relations contains a crossing (continuity of the sorted
eigenvalues).

### 10.4 Davis–Kahan statements (E-07, C-12, C-13; `davis_kahan.py`, `docs/proofs/E-07_davis_kahan.md`)

*E-07.* With H′ = Q̂ᴴAQ̂, ‖AQ̂ − Q̂H′‖_F ≤ ‖R‖_F/√(1−η) and ‖H′ − Λ‖₂ ≤ w′ = ‖XᴴR‖₂/(1−η) + δ·spread/√(1−η) (the two
polar identities of the Theorem 2′ sketch, proved in the document), and the Frobenius form of the sin Θ theorem
(for Y = (I−P)Q̂: A_⊥Y − YH′ = (I−P)R′, entrywise division by the eigenvalue differences ≥ δ′) gives
‖Q̂Q̂ᴴ − P‖₂ ≤ s_DK = ‖R‖_F/(√(1−η)(dist − w′)). S4 uses min(s₆, s_DK), both recorded (`sin_theta_bounds`).
*C-12.* For ‖[A,S]‖ ≤ c (computed, recorded): every eigenvalue m of QᴴSQ has an eigenvalue of S within
(c + ‖S‖w/2)/γ₀ (Theorem 1 for S with the residual ‖(I−P)Sφ‖), and the sorted eigenvalues of XᴴSX are within
(2s + 2s² + η)‖S‖ of those of QᴴSQ (alignment ‖Q̂ − QU‖ ≤ s + s², Weyl, Ostrowski).
*C-13.* |tr(P₁P₂) − ‖X₁ᴴX₂‖_F²| ≤ g(s₁+η₁) + g(1+η₁)(s₂+η₂) (Theorem 6 for both clusters, Hölder with ‖P‖₁ = g).

### 10.5 Variational certificates for non-ED states (E-05, `variational.py`, `docs/proofs/E-05_variational.md`)

For a trial vector ψ (dense, an MPS contracted to a vector, or a sparse CI vector) and H Hermitian: λ₁ ≤ θ = ⟨ψ, Hψ⟩/⟨ψ, ψ⟩
(Rayleigh–Ritz, no hypothesis); an eigenvalue within r = ‖Hψ − θψ‖/‖ψ‖ of θ (Theorem 1), r² = ⟨ψ, H²ψ⟩/⟨ψ, ψ⟩ − θ²; and
Temple's inequality λ₁ ≥ θ − r²/(ρ − θ) for any ρ with θ < ρ ≤ λ₂ (from (H − λ₁)(H − ρ) ⪰ 0; an equality for two-level
mixtures). θ is enclosed from the error-free inner products of `fp.exact_dot` and the entrywise bound γ̇_{w+2}|H||ψ| on
fl(Hψ) (the four corner quotients, 4u outward), r by the Theorem 1 radius at θ̂ plus the width of the θ enclosure; for a
spec the model shift σ enters as λ₁(H_spec) ≤ θ_hi + σ, radius + σ (Weyl), and Temple applied to H_ref with ρ − σ, then
− σ. The Temple statement is CERTIFIED only when ρ comes from a certificate of a 'lowest' claim on the same operator
(second smallest lower endpoint of its S2 enclosures for k ≥ 2, S3's next-eigenvalue bound for k = 1); a declared ρ gives
the distinct status CERTIFIED_UNDER_HYPOTHESIS with the hypothesis `declared_gap` named in the record and in the
statement text. An MPS is contracted left to right with |ψ̂ − ψ| ≤ γ_M |A⁽¹⁾|⋯|A⁽ᴸ⁾| entrywise (M = Σ D_{j−1}, Higham
Lemma 3.3); the certified statements are about ψ̂, and |θ(ψ) − θ(ψ̂)| ≤ (2r‖ψ̂‖e + (‖H‖ + |θ|)e²)/(‖ψ̂‖ − e)² is recorded.
Producer-declared stochastic estimates are DECLARED (DECLARED_ESTIMATE) and nothing rests on them.

### 10.6 Generalized eigenproblems (E-03, `generalized.py`, `docs/proofs/E-03_generalized.md`)

For A x = λ B x with B ≻ 0 and B = LLᴴ, the pencil is similar to C = L⁻¹AL⁻ᴴ (Lᴴx are C's eigenvectors). Theorem 1 for C
gives |λ − μ| ≤ ‖Ax − μBx‖_{B⁻¹}/‖x‖_B ≤ ‖Ax − μBx‖₂/(√β_c ‖x‖_B) for any certified β_c ≤ λ_min(B) (‖L⁻¹‖₂² = 1/λ_min(B)); β_c
comes from the inertia of B − βI (ν = 0 with backward error b ⇒ λ_min(B) ≥ β − b), and a B whose inertia at 0 is
negative is refused. The residual s = fl(Ax − μ fl(Bx)) carries the entrywise bound γ̇_n|A||x| + |μ|γ̇_{n+1}|B||x| +
u(1 + 2u)|ŝ|, xᴴBx the error-free inner product of `fp.exact_dot` minus γ̇_n|x|ᵀ|B||x|. Completeness of a 'lowest' claim:
A − σB = L(C − σI)Lᴴ, so ν(A − σB) = #{λ < σ} (Sylvester); a factorization of M̂ = fl(A − σB) with total backward error b
(LDLᴴ plus the forming error ‖u|σ||B| + u|M̂|‖_∞) satisfies #{λ < σ − b/β_c} ≤ ν ≤ #{λ ≤ σ + b/β_c}, because bI ⪯ (b/β_c)B;
ν = k with pairwise disjoint enclosures above which σ − b/β_c lies makes the enclosures the k lowest eigenvalues and
gives λ_{k+1} ≥ σ − b/β_c. A declared λ_min(B) is a named hypothesis (CERTIFIED_UNDER_HYPOTHESIS).

### 10.7 Certified moments and sum rules (C-07 first part, `moments.py`, `docs/proofs/C-07_moments.md`)

For a certified cluster (P, g stored vectors X, s, η) and Hermitian O: |tr(PM) − tr(XXᴴM)| ≤ (2gs + gη)‖M‖₂ for every
Hermitian M (trace norm of P − Q̂Q̂ᴴ ≤ 2gs by rank, of Q̂(M² − I)Q̂ᴴ ≤ gη; Hölder), applied to M = OHᵏO with ‖M‖ ≤ ‖O‖²‖H‖ᵏ.
tr(XXᴴOHᵏO) = Σ_i ⟨Ox_i, HᵏOx_i⟩ is evaluated by one application of O and k of H to the stored vectors with the Theorem 1
style entrywise bounds γ̇_w|O||x|, γ̇_w|H||z| propagated as D_k = ‖H‖D_{k−1} + ε_k, the polarisation identity for the error
of the quadratic form, and `fp.exact_dot` for the final inner product; Hᵏ is never formed. Sum rules: m₀ = tr(PO²)/g and
m₁ − tr(PHPO²)/g = tr(P[O,[H,O]])/(2g) (HP = PH, cyclicity), with tr(PHPO²)/g ∈ [a, b]·m₀; the right-hand sides are
certified by Theorem 6 on the explicitly formed matrices (product chains with the entrywise bound γ_M|F₁|⋯|F_m| and the
Hermitian average), and the record states whether the two certified intervals overlap (CERTIFIED) or not (FAILED). The
continued-fraction part is open (§9.4).
