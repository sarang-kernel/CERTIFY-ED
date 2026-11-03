# E-07, C-12, C-13 Davis–Kahan sin Θ statements: sharper eigenspace bound, approximate-symmetry labels, fidelities

Notation as in `docs/theory.md`: A Hermitian n × n; X (n × g) the claimed vectors of a cluster with claimed
values μ (Λ = diag(μ)), R = AX − XΛ, η = ‖XᴴX − I‖₂ < 1, X = Q̂M the polar decomposition (M = (XᴴX)^{1/2}, Q̂ᴴQ̂ = I),
δ = 1 − √(1−η) ≥ ‖M − I‖₂, ‖M⁻¹‖₂ ≤ (1−η)^{−1/2}, ‖M‖₂ ≤ √(1+η). P is the spectral projector of A onto the g
cluster eigenvalues (certified distinct by Theorem 2, isolated by S3: every other eigenvalue of A lies at
distance ≥ dist from [a, b] = [min μ, max μ]); Q (n × g) an orthonormal basis of range P. H′ := Q̂ᴴAQ̂ (Hermitian,
the Ritz matrix). ‖·‖₂ spectral norm, ‖·‖_F Frobenius, ‖·‖₁ trace norm. sin θ_max := ‖Q̂Q̂ᴴ − P‖₂ = ‖(I − P)Q̂‖₂
(equal-rank projectors).

## 1 Statement

**Lemma 1 (sin Θ, Frobenius form; Davis & Kahan 1970, Stewart & Sun 1990 Thm V.3.4 / Parlett 1998 Thm 11.7.1).**
Let Q̂ be orthonormal (n × g), H′ = Q̂ᴴAQ̂, R′ = AQ̂ − Q̂H′. Let P be the spectral projector of A onto a set of
eigenvalues and suppose every eigenvalue of A *not* in that set lies at distance ≥ δ′ > 0 from every eigenvalue
of H′. Then ‖(I − P)Q̂‖_F ≤ ‖(I − P)R′‖_F/δ′ ≤ ‖R′‖_F/δ′.

**Lemma 2 (polar identities).** (i) (I − Q̂Q̂ᴴ)AQ̂ = (I − Q̂Q̂ᴴ)RM⁻¹, hence ‖AQ̂ − Q̂H′‖_F = ‖(I − Q̂Q̂ᴴ)AQ̂‖_F ≤
‖R‖_F/√(1−η). (ii) ‖H′ − Λ‖₂ ≤ w′ := ‖XᴴR‖₂/(1−η) + δ·spread(μ)/√(1−η), spread(μ) = max μ − min μ.

**Proposition 1 (E-07, Davis–Kahan alternative to Theorem 6).** If dist > w′ then

    sin θ_max = ‖Q̂Q̂ᴴ − P‖₂ ≤ s_DK := ‖R‖_F / (√(1−η) (dist − w′)).

Theorem 6's bound s₆ = ρ/(dist − ε) (ρ = ‖R‖ + (‖A‖ + ‖Λ‖)δ, ε the cluster radius; `docs/theory.md` §4 and the
manuscript appendix) remains valid; S4 uses s = min(s₆, s_DK) and records both. Everything downstream of s
(‖XXᴴ − P‖₂ ≤ s + η, Theorem 6 observables, Proposition 7 labels, C-03, C-04, C-13) is unchanged.

**Lemma 3 (alignment).** With U the unitary polar factor of QᴴQ̂ and s ≥ sin θ_max: ‖Q̂ − QU‖₂ ≤ s + s².

**Proposition 2 (C-12, approximate symmetry).** Let S be Hermitian with ‖[A, S]‖₂ ≤ c, J = [J_L, J_U] an interval
containing the g cluster eigenvalues (width w = J_U − J_L), and γ₀ the distance from the midpoint of J to every
other eigenvalue of A (γ₀ ≥ sep + w/2 where sep is S4's separation of J from the rest). Let m₁ ≤ … ≤ m_g be the
eigenvalues of the exact compression QᴴSQ and m̂₁ ≤ … ≤ m̂_g those of the Hermitian matrix XᴴSX. Then
(a) for every i there is an eigenvalue q of S with |q − m_i| ≤ (c + ‖S‖₂ w/2)/γ₀;
(b) |m̂_i − m_i| ≤ (2s + 2s² + η)‖S‖₂ for every i;
hence every m̂_i has an eigenvalue of S within D := (2s + 2s² + η)‖S‖₂ + (c + ‖S‖₂w/2)/γ₀ (+ the rounding of the
computed m̂_i). If D is below half the spacing of a declared lattice of eigenvalues of S, the labels snap to it.
The statement is conditional on the recorded hypothesis ‖[A, S]‖₂ ≤ c; the code computes c from the stored
matrices with a rounding bound (Lemma 5). For an exact symmetry (c = rounding only) it recovers Proposition 7's
conclusion with a first-order instead of second-order geometric term.

**Proposition 3 (C-13, overlap of two certified eigenspaces).** For two clusters (P₁, X₁, s₁, η₁), (P₂, X₂, s₂, η₂)
of the same dimension g on the same space, with ‖X_iX_iᴴ − P_i‖₂ ≤ s_i + η_i (Theorem 6),

    |tr(P₁P₂) − tr(X₁X₁ᴴX₂X₂ᴴ)| ≤ g(s₁ + η₁) + g(1 + η₁)(s₂ + η₂),   tr(X₁X₁ᴴX₂X₂ᴴ) = ‖X₁ᴴX₂‖_F².

For g = 1, tr(P₁P₂) = |⟨ψ₁|ψ₂⟩|² is the fidelity of the two exact eigenvectors; for g > 1, tr(P₁P₂)/g ∈ [0, 1] is the
mean overlap of the two subspaces (basis independent).

**Lemma 5 (rounding).** (a) Commutator: for stored H, S and C = fl(fl(HS) − fl(SH)), |C − [H, S]| ≤
γ̇_{w_H+1}|H||S| + γ̇_{w_S+1}|S||H| entrywise, and ‖[H,S]‖₂ ≤ ‖[H,S]‖_∞ (anti-Hermitian), bounded by the ∞-norm of
|C| plus the entrywise bound. (b) Labels: ‖fl(XᴴSX) − XᴴSX‖₂ ≤ γ̇_{w_S+n}‖X‖_F²‖S‖_∞ ≤ γ̇_{w_S+n} g(1+η)‖S‖_∞;
eigvalsh's backward error is assumed ≤ 10g²u‖M‖ (LAPACK, recorded as an assumption as in Proposition 7).
(c) Fidelity: with Ĉ = fl(X₁ᴴX₂), e := γ̇_n‖X₁‖_F‖X₂‖_F ≥ ‖Ĉ − X₁ᴴX₂‖_F, |‖Ĉ‖_F² − ‖X₁ᴴX₂‖_F²| ≤ (2‖Ĉ‖_F + e)e, and the
floating-point sum of the g² squared moduli adds ≤ γ_{g²+2}‖Ĉ‖_F².

## 2 Proof

**Lemma 1.** Let Y = (I − P)Q̂. Since A commutes with P, (I − P)R′ = (I − P)AQ̂ − (I − P)Q̂H′ = A_⊥Y − YH′, where A_⊥ is
the restriction of A to range(I − P), with eigenvalues λ_k at distance ≥ δ′ from the eigenvalues h_j of H′. Write
Y in the eigenbases: A_⊥ = Σ_k λ_k u_ku_kᴴ, H′ = Σ_j h_j v_jv_jᴴ; then (A_⊥Y − YH′)_{kj} := u_kᴴ(A_⊥Y − YH′)v_j =
(λ_k − h_j) u_kᴴYv_j, so |u_kᴴYv_j| ≤ |u_kᴴ(I − P)R′v_j|/δ′ for every (k, j), and summing the squares over the
orthonormal bases, ‖Y‖_F ≤ ‖(I − P)R′‖_F/δ′ (the Frobenius norm is unitarily invariant). ‖(I − P)R′‖_F ≤ ‖R′‖_F
because ‖I − P‖₂ ≤ 1. ∎ (This is the Frobenius version of the Davis–Kahan sin Θ theorem; the spectral-norm
version needs the stronger hypothesis that the spectra lie in an interval and its complement and is not used.)

**Lemma 2.** From AX = R + XΛ and X = Q̂M (M nonsingular): AQ̂ = RM⁻¹ + Q̂MΛM⁻¹, so (I − Q̂Q̂ᴴ)AQ̂ = (I − Q̂Q̂ᴴ)RM⁻¹
because (I − Q̂Q̂ᴴ)Q̂ = 0. Also AQ̂ − Q̂H′ = (I − Q̂Q̂ᴴ)AQ̂ (definition of H′). Hence ‖AQ̂ − Q̂H′‖_F ≤ ‖I − Q̂Q̂ᴴ‖₂‖R‖_F‖M⁻¹‖₂ ≤
‖R‖_F/√(1−η): (i). For (ii), Q̂ᴴ = M⁻¹Xᴴ, so H′ − Λ = Q̂ᴴ(RM⁻¹ + Q̂MΛM⁻¹) − Λ = M⁻¹XᴴRM⁻¹ + (MΛM⁻¹ − Λ), and
MΛM⁻¹ − Λ = [M, Λ]M⁻¹ = [M − I, Λ − cI]M⁻¹ for every scalar c; with c the midpoint of the spectrum of Λ,
‖Λ − cI‖₂ = spread/2 and ‖[M − I, Λ − cI]‖₂ ≤ 2‖M − I‖₂‖Λ − cI‖₂ ≤ δ·spread. Thus ‖H′ − Λ‖₂ ≤ ‖XᴴR‖₂/(1−η) +
δ·spread/√(1−η). The eigenvalue bounds for M: the eigenvalues of M² = XᴴX lie in [1−η, 1+η], so those of M lie
in [√(1−η), √(1+η)], ‖M⁻¹‖₂ ≤ (1−η)^{−1/2}, ‖M − I‖₂ ≤ max(1 − √(1−η), √(1+η) − 1) = 1 − √(1−η) = δ (concavity of
√). ∎

**Proposition 1.** By Lemma 2 (ii) and Weyl's inequality, every eigenvalue of H′ lies within w′ of an eigenvalue of
Λ, i.e. in [a − w′, b + w′]. Every eigenvalue of A outside range P lies at distance ≥ dist from [a, b], hence at
distance ≥ dist − w′ =: δ′ > 0 from every eigenvalue of H′. Lemma 1 with this P and δ′, then Lemma 2 (i):
‖(I − P)Q̂‖₂ ≤ ‖(I − P)Q̂‖_F ≤ ‖AQ̂ − Q̂H′‖_F/δ′ ≤ ‖R‖_F/(√(1−η)(dist − w′)). Finally ‖Q̂Q̂ᴴ − P‖₂ = ‖(I − P)Q̂‖₂ for
projectors of equal rank g (rank P = g because the cluster's g eigenvalues are distinct eigenvalues of A counted
with multiplicity, S2/S3; e.g. Stewart & Sun 1990, Thm I.5.5). The min with s₆: both are upper bounds on the same
quantity, so their minimum is. The code's inputs are upper bounds on ‖R‖_F, ‖XᴴR‖₂, δ, spread and a lower bound
on dist, which can only increase s_DK. ∎

**Lemma 3.** Let C = QᴴQ̂ (g × g) with polar factor U (unitary, since C is nonsingular: CᴴC = Q̂ᴴPQ̂ = I − Q̂ᴴ(I−P)Q̂
has eigenvalues in [1 − s², 1] because ‖Q̂ᴴ(I−P)Q̂‖₂ = ‖(I−P)Q̂‖₂² ≤ s² < 1). Q̂ − QU = (I − QQᴴ)Q̂ + Q(C − U).
‖(I − QQᴴ)Q̂‖₂ = ‖(I − P)Q̂‖₂ ≤ s. C − U = U((CᴴC)^{1/2} − I), so ‖C − U‖₂ = ‖(CᴴC)^{1/2} − I‖₂ ≤ 1 − √(1 − s²) ≤ s²
(for t ∈ [0, 1], 1 − √(1−t) ≤ t). Hence ‖Q̂ − QU‖₂ ≤ s + s². ∎

**Proposition 2.** (a) Let y be a unit eigenvector of QᴴSQ with eigenvalue m and φ = Qy ∈ range P, ‖φ‖ = 1. Then
PSφ = QQᴴSQy = mφ, so Sφ − mφ = (I − P)Sφ. Let λ₀ be the midpoint of J; (A − λ₀) restricted to range(I − P) has
eigenvalues of modulus ≥ γ₀, so for any vector v, ‖(I − P)v‖ ≤ ‖(A − λ₀)(I − P)v‖/γ₀ = ‖(I − P)(A − λ₀)v‖/γ₀ ≤
‖(A − λ₀)v‖/γ₀ (A commutes with P). With v = Sφ: (A − λ₀)Sφ = S(A − λ₀)φ + [A, S]φ, and ‖(A − λ₀)φ‖ ≤ w/2 since φ
is a combination of eigenvectors with eigenvalues in J. Thus ‖Sφ − mφ‖ ≤ (‖S‖₂w/2 + c)/γ₀, and Theorem 1 (the
residual theorem, `docs/theory.md` §2) applied to the Hermitian S and the unit vector φ gives an eigenvalue of S
within that distance of m. (b) The eigenvalues of QᴴSQ equal those of UᴴQᴴSQU (unitary similarity). By Lemma 3,
‖Q̂ᴴSQ̂ − UᴴQᴴSQU‖₂ ≤ ‖Q̂ᴴS(Q̂ − QU)‖₂ + ‖(Q̂ − QU)ᴴSQU‖₂ ≤ 2‖S‖₂(s + s²) (‖Q̂‖₂ = ‖QU‖₂ = 1), so by Weyl
|λ_i(Q̂ᴴSQ̂) − m_i| ≤ 2‖S‖₂(s + s²). Further XᴴSX = M(Q̂ᴴSQ̂)M with M Hermitian positive definite, so by Ostrowski's
theorem (Horn & Johnson Thm 4.5.9) λ_i(XᴴSX) = θ_iλ_i(Q̂ᴴSQ̂) with θ_i ∈ [1−η, 1+η], whence |λ_i(XᴴSX) − λ_i(Q̂ᴴSQ̂)| ≤
η|λ_i(Q̂ᴴSQ̂)| ≤ η‖S‖₂. Adding: |m̂_i − m_i| ≤ (2s + 2s² + η)‖S‖₂. The computed m̂_i differ from the exact λ_i(XᴴSX) by
the rounding of Lemma 5 (b) (Weyl again). Snapping: if every m̂_i is within D of an eigenvalue of S and D is less
than half the minimal spacing of the declared eigenvalue set, the nearest declared value is that eigenvalue. ∎

**Proposition 3.** tr(P₁P₂) − tr(X₁X₁ᴴX₂X₂ᴴ) = tr((P₁ − X₁X₁ᴴ)P₂) + tr(X₁X₁ᴴ(P₂ − X₂X₂ᴴ)). Hölder: |tr(AB)| ≤ ‖A‖₂‖B‖₁;
‖P₂‖₁ = g (orthogonal projector of rank g), ‖X₁X₁ᴴ‖₁ = tr(X₁ᴴX₁) ≤ g(1+η₁) (positive semidefinite; eigenvalues of
X₁ᴴX₁ in [1−η₁, 1+η₁]); the projector errors are ≤ s_i + η_i (Theorem 6: ‖XXᴴ − P‖₂ ≤ ‖Q̂Q̂ᴴ − P‖₂ + ‖Q̂(M² − I)Q̂ᴴ‖₂ ≤
s + η). tr(X₁X₁ᴴX₂X₂ᴴ) = tr((X₁ᴴX₂)(X₁ᴴX₂)ᴴ) = ‖X₁ᴴX₂‖_F² by cyclicity. For g = 1 and unit ψ_i, tr(ψ₁ψ₁ᴴψ₂ψ₂ᴴ) =
|⟨ψ₁|ψ₂⟩|². 0 ≤ tr(P₁P₂) = ‖P₁P₂‖_F² ≤ ‖P₁‖_F² = g gives the clipping. ∎

**Lemma 5.** (a) Standard model: each entry of fl(HS) is an inner product of ≤ w_H nonzero terms (error ≤ γ̇_{w_H}(|H||S|)_{ij}),
likewise fl(SH), and the subtraction adds one rounding ≤ u|fl(HS) − fl(SH)|_{ij} ≤ u((|H||S|)_{ij}(1+γ̇) + (|S||H|)_{ij}(1+γ̇)):
together ≤ γ̇_{w_H+1}|H||S| + γ̇_{w_S+1}|S||H| entrywise (γ̇_a + u(1 + γ̇_a) ≤ γ̇_{a+1}). The computed absolute products
underestimate the exact ones by at most a relative γ̇ (non-negative terms), covered by the factor 1/(1 − γ̇). For
anti-Hermitian C, ‖C‖₂ ≤ (‖C‖₁‖C‖_∞)^{1/2} = ‖C‖_∞, and the ∞-norm of |C_exact| ≤ |C| + E is ≤ ‖|C| + E‖_∞. (b) fl(Xᴴ(SX)):
two inner products of ≤ w_S and n terms, entrywise error ≤ γ̇_{w_S+n}(|X|ᵀ|S||X|)_{ij}; ‖|X|ᵀ|S||X|‖_F ≤ ‖|X|‖_F‖|S|‖₂‖|X|‖_F ≤
‖X‖_F²‖S‖_∞ (‖|S|‖₂ ≤ ‖|S|‖_∞ = ‖S‖_∞ for Hermitian S); ‖X‖_F² = tr XᴴX ≤ g(1+η). The Hermitian part adds ≤ u‖M̂‖_F,
inside the index shift +4 used by the code. (c) |‖Ĉ‖² − ‖C‖²| = (‖Ĉ‖ + ‖C‖)|‖Ĉ‖ − ‖C‖| ≤ (2‖Ĉ‖ + e)e with ‖C‖ ≤ ‖Ĉ‖ + e
and |‖Ĉ‖ − ‖C‖| ≤ ‖Ĉ − C‖_F ≤ e (Higham 2002, eq. (3.5) for the inner products); the sum of g² squared moduli in
floating point has relative error ≤ γ_{g²+1} plus the squaring, ≤ γ_{g²+2}. ∎

## 3 Where the code implements it

`certify_ed/davis_kahan.py`, `certify_ed/eigenspace.py`, `certify_ed/enclosure.py`, `certify_ed/certify.py`
(branch `track/new-statements`):

* Proposition 1: `sin_theta_davis_kahan` (davis_kahan.py 47–69). Inputs from `enclosure._cluster_bound`, which now
  exports its upper bounds `R_fro` (= Rg_fro ≥ ‖R‖_F, a Frobenius sum of the per-column residual bounds), `XR`
  (≥ ‖XᴴR‖₂), `delta`, `spread` in its result dict (additive). Model shift σ (S1): lines 57–58 add √(1+η)σ to ‖R‖_F
  and (1+η)σ to ‖XᴴR‖₂ (the replacement of the operator by the declared model changes R by at most σ‖X‖₂ ≤ σ√(1+η)
  and XᴴR by at most σ‖X‖₂² ≤ σ(1+η)). w′ at line 61 (`up`), the gap dist − w′ at 62 (`sub_down`), s_DK at 65 (`up`).
  The hook: `certify.py` S4 loop, `dist` (line 1326, the outward-rounded distance of [a, b] to the other claimed
  enclosures and the window edges — S3's certified window is what makes dist a bound for *every* other eigenvalue),
  `s_dk` (1332) and `certify_eigenspace(..., sin_theta_alt=s_dk)` (1334). `eigenspace.certify_eigenspace`
  (90–111): s₆ at line 98, s = min(s₆, s_DK) at 102, `used` at 105; both bounds and the Davis–Kahan record are
  stored in `sin_theta_bounds` of the cluster record. A cluster with s₆ ≥ 1 but s_DK < 1 is now certified (the
  statement rests on Proposition 1 alone); every downstream quantity uses s.
* Proposition 2 and Lemma 5 (a), (b): `commutator_bound` (72–92: E at 87, `inf_norm_upper(|C| + E)` at 91, guarded);
  `approximate_symmetry_label` (95–165: rounding 121–122, w rounded up 124, γ₀ = sep + w/2 rounded down 125, the
  three terms 130–131, D at 132 with `add_up`). `labels_hook` (232–250) is called from the S4 loop (certify.py
  line 1368) with the reference matrix (the spec's, else the stored operator) and J = (JL, JU), the cluster's level
  enclosure; without a stored matrix the record is NOT-CHECKED (NEEDS_MATRIX). Records live in
  `quantum_numbers[name]` with `approximate_symmetry: true` and `hypothesis.commutator_bound`.
* Proposition 3 and Lemma 5 (c): `certified_fidelity` (168–201: e at 188, Ĉ's norm 189, rounding 190, projector term
  191, bound 192 with `add_up`; interval clipped to [0, g]); `fidelity_from_certificates` (212–229) reads the stored
  eigenvectors and the S4 cluster records (`sin_theta_bound`, `eta`) of two certificates.

Rounding directions: every bound is rounded up (`up`, `add_up`, index shifts), gaps and γ₀ down (`sub_down`).

## 4 Computational verification

`docs/proofs/verify_E-07.py` (seed 20261030, mpmath at 50 digits; comparison margin 10⁻⁴⁰, 10⁻¹⁴ where the code's
binary64 output is compared). Run of 2026-10-30:

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | Lemma 1 on random Hermitian A with a planted cluster and gap, random orthonormal Q̂ rotated towards the complement: ‖(I−P)Q̂‖_F ≤ ‖(I−P)R′‖_F/δ′ with δ′ the exact Ritz-to-rest distance | 60 | 0 |
| B | Lemma 2 (i) and (ii) with exact polar factors, η up to 0.5 | 110 | 0 |
| C | Proposition 1: exact sin θ_max ≤ s_DK (the code's function fed exact ingredients, n ≤ 11, g ≤ 3, rotations 10⁻⁶ … 10⁻¹, η up to 10⁻¹; adversarial: η ≈ 0.3–0.5, a 0.4-rad rotation, g = n − 1, exact eigenvectors); s_DK ≤ the Theorem 6 quotient ρ/(dist − ε*) in 100/100 random cases; **negative control**: dist overstated to 50 (an eigenvalue inside the claimed gap) undercuts sin θ in 20/20 cases | 100 random + 4 adversarial | 0 |
| D | Proposition 2 on random A, S with a planted commutator (S = commuting part + perturbation 10⁻⁶…10⁻¹): every eigenvalue of XᴴSX within D of an eigenvalue of S; Lemma 3 (alignment) on the same cases; the tight case A = diag(0, γ), S = c′σ_x, X = e₀ where the deviation equals c/γ exactly (equality, 20 cases); **negative control**: c halved in the tight case breaks the bound in 20/20 cases | 60 + 60 + 20 | 0 |
| E | Proposition 3 with exact projectors of random subspaces, claimed vectors rotated by up to 0.5 rad and scaled | 79 | 0 |
| F | Lemma 5 (a), (c): the code's commutator bound against the exact ‖[H,S]‖_∞ of random sparse real/complex H, S (scales 10⁻³…10³); the fidelity value against the exact trace, within the rounding term | 60 | 0 |

Total: 553 cases, 0 violations. Tests `tests/test_davis_kahan.py` (7): on the 8-site Heisenberg ring both bounds
are recorded and s_DK < s₆ for the singlet and the triplet (used: davis_kahan); the function bounds the exact
sin θ on 40 planted matrices and the planted-gap negative control breaks it; approximate-symmetry labels for the
exact S^z_total (c = rounding, 4·10⁻¹⁴), for S^z_total + 10⁻³S^x₀ and with declared allowed values (all snap to the
exact labels), the hypothesis c covers the exact commutator norm, a 0.05 perturbation stays CERTIFIED with every
label within D of an eigenvalue of S, a 0.6 perturbation is INCONCLUSIVE (D ≥ 1/2), no reference matrix →
NOT-CHECKED; fidelity of the ground states at J2 = 0 and 0.1 contains |⟨ψ₁|ψ₂⟩|² (bound < 10⁻¹¹), the triplet
subspaces' overlap contains tr(P₁P₂), shape and missing-vector errors, and 20 planted rotations.

## 5 Analytic cross-check

1. *g = 1, exact normalisation.* η = 0, δ = 0, w′ = |xᴴr| (r the residual, which is orthogonal to x when μ is the
   Rayleigh quotient, so w′ = 0 up to rounding) and s_DK = ‖r‖/dist: the classical sin θ ≤ ‖r‖/gap for a single
   Ritz vector (Parlett 1998, Thm 11.7.1), while s₆ = (‖r‖ + (‖A‖ + |μ|)·0)/(dist − ‖r‖) = ‖r‖/(dist − ‖r‖) ≥ s_DK.
   For the 8-site singlet the run shows s_DK = 1.5·10⁻¹³ against s₆ = 3.7·10⁻¹³ with the same ‖R‖_F; the difference
   comes from the model-shift allowance (Theorem 6 enlarges ρ by 2.5σ, the Lipschitz constant of the S4 block,
   while ‖R‖_F grows by √(1+η)σ ≈ σ) and from the (‖A‖ + ‖Λ‖)δ term of ρ (≈ 10 · 8·10⁻¹⁶). Both are visible in
   `sin_theta_bounds`.
2. *Exact symmetry.* c = 0 (rounding only), so Proposition 2 gives D = (2s + 2s² + η)‖S‖ + ‖S‖w/(2γ₀) + rounding; for
   the singlet (s ≈ 10⁻¹³, w ≈ 10⁻¹³) D ≈ 10⁻¹² and the S^z label 0 snaps (test). Proposition 7 gives the second-order
   term 2s²‖S‖ instead of 2s‖S‖: Proposition 2 is weaker for exact symmetries, as expected from a first-order
   argument, and is the only one available when c > 0.
3. *Tight label case.* A = diag(0, γ), S = c′σ_x: [A, S] = γc′(e₁e₀ᵀ − e₀e₁ᵀ), ‖[A,S]‖₂ = γc′; the exact
   eigenvector e₀ has label 0 and the eigenvalues of S are ±c′: deviation c′ = c/γ exactly, so the bound (a) is
   attained (part D, 20 cases to 10⁻⁴⁰) and cannot be improved without further hypotheses.
4. *Fidelity of identical states.* X₁ = X₂ exact eigenvectors (s = η = 0): F̂ = g, tr(P₁P₂) = g, bound = rounding
   only; for orthogonal subspaces F̂ = 0. The planted-rotation test covers intermediate angles, where
   tr(P₁P₂) − F̂ is first order in the rotations and the bound g(s₁ + η₁) + g(1+η₁)(s₂ + η₂) is first order too.

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (529 random + 24 adversarial cases, 0 violations; both negative
controls detected 20/20); independent re-derivation: done 2026-10-30 (fresh-context session, verdict GAP, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
