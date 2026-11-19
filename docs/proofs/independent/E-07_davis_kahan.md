# Independent re-derivation: E-07 / C-12 / C-13 (Davis–Kahan sin Θ, approximate-symmetry labels, fidelities)

Written from sections 1 and 3 of `docs/proofs/E-07_davis_kahan.md` only (section 2 read afterwards; comparison at
the end). Notation as there. Classical facts: Davis–Kahan sin Θ theorem (Davis & Kahan 1970; Stewart & Sun 1990
Thm V.3.6: if the spectrum of H′ lies in [α, β] and the eigenvalues of A outside the set lie outside (α − δ′, β + δ′),
then ‖(I − P)Q̂‖ ≤ ‖R′‖/δ′ in every unitarily invariant norm; the Frobenius form needs only pointwise separation);
Weyl; Ostrowski (Horn & Johnson Thm 4.5.9); the residual theorem (Parlett Thm 4.5.1: a unit v with ‖Sv − mv‖ ≤ r
has an eigenvalue of S within r of m); Hölder |tr(AB)| ≤ ‖A‖₁‖B‖₂; ‖P₁ − P₂‖₂ = ‖(I − P₂)P₁‖₂ for equal-rank
orthogonal projectors (Stewart & Sun Thm I.5.5).

## Proofs

**Lemma 1.** Write A = A₁ ⊕ A₂ on range P ⊕ range(I − P). Multiply R′ = AQ̂ − Q̂H′ by (I − P): (I − P)R′ = A₂Y − YH′ with
Y := (I − P)Q̂ (A commutes with P). In eigenbases of A₂ (eigenvalues α_i) and H′ (eigenvalues h_j) this reads
Y_ij(α_i − h_j) = ((I − P)R′)_ij with |α_i − h_j| ≥ δ′, so ‖Y‖_F ≤ ‖(I − P)R′‖_F/δ′ ≤ ‖R′‖_F/δ′ (unitary invariance of
‖·‖_F; ‖(I − P)R′‖_F ≤ ‖R′‖_F). ∎ (Pointwise separation suffices for the Frobenius norm; the 2-norm version needs
the interval hypothesis, which Proposition 1's setting also provides — used below for the model shift.)

**Lemma 2.** (i) R = AQ̂M − Q̂MΛ and (I − Q̂Q̂ᴴ)Q̂ = 0 give (I − Q̂Q̂ᴴ)R = (I − Q̂Q̂ᴴ)AQ̂M, i.e. (I − Q̂Q̂ᴴ)AQ̂ = (I − Q̂Q̂ᴴ)RM⁻¹;
AQ̂ − Q̂H′ = (I − Q̂Q̂ᴴ)AQ̂, so ‖R′‖_F ≤ ‖R‖_F‖M⁻¹‖₂ ≤ ‖R‖_F/√(1−η). (ii) Q̂ᴴAX = Q̂ᴴR + Q̂ᴴXΛ, i.e. H′M = Q̂ᴴR + MΛ, so
H′ = Q̂ᴴRM⁻¹ + MΛM⁻¹ and H′ − Λ = M⁻¹XᴴRM⁻¹ + [M − I, Λ − ζI]M⁻¹ for any scalar ζ (Q̂ᴴ = M⁻¹Xᴴ; [M, Λ] = [M − I, Λ − ζI]).
With ζ the midpoint of [min μ, max μ]: ‖[M − I, Λ − ζI]‖₂ ≤ 2δ·spread/2, hence ‖H′ − Λ‖₂ ≤ ‖XᴴR‖₂/(1−η) +
δ·spread/√(1−η) = w′. ∎

**Proposition 1.** H′ and Λ are Hermitian g × g with ‖H′ − Λ‖₂ ≤ w′, so (Weyl) every eigenvalue of H′ lies in
[min μ − w′, max μ + w′]; every eigenvalue of A outside the cluster is at distance ≥ dist from [min μ, max μ], hence
≥ dist − w′ =: δ′ > 0 from every eigenvalue of H′ (and outside the interval's δ′-neighbourhood). Lemma 1 with
Lemma 2 (i): ‖(I − P)Q̂‖_F ≤ ‖R‖_F/(√(1−η)(dist − w′)). Since rank P = g = rank Q̂Q̂ᴴ, sin θ_max = ‖Q̂Q̂ᴴ − P‖₂ =
‖(I − P)Q̂‖₂ ≤ ‖(I − P)Q̂‖_F. ∎
*Model shift.* The statement is about H_spec = A + Δ, ‖Δ‖₂ ≤ σ, while R, XᴴR are computed for A. R_spec = R + ΔX,
XᴴR_spec = XᴴR + XᴴΔX, so ‖XᴴR_spec‖₂ ≤ ‖XᴴR‖₂ + (1+η)σ (as the code adds), but ‖ΔX‖_F ≤ σ‖X‖_F ≤ σ√(g(1+η)) — the
Frobenius increment is √g times the σ√(1+η) that section 3 states (that is a 2-norm bound). The code's bound is
nevertheless valid: use the 2-norm sin Θ theorem (interval separation holds as shown), ‖R′_spec‖₂ =
‖(I − Q̂Q̂ᴴ)(A + Δ)Q̂‖₂ ≤ ‖R‖₂/√(1−η) + σ ≤ (‖R‖_F + σ√(1+η))/√(1−η), with δ′ computed from the shifted w′. So
sin θ_max ≤ (‖R‖_F + σ√(1+η))/(√(1−η)(dist − w′_spec)) holds, by a different route than the document's.

**Lemma 3.** C := QᴴQ̂ = UK, K = (CᴴC)^{1/2} ⪰ 0, singular values of C = cos θ_i ≥ cos θ_max ≥ √(1 − s²).
Q̂ − QU = (I − P)Q̂ + Q(C − U) = (I − P)Q̂ + QU(K − I); ‖(I − P)Q̂‖₂ = sin θ_max ≤ s and ‖K − I‖₂ = 1 − cos θ_max ≤
1 − √(1 − s²) ≤ s². ∎

**Proposition 2.** (a) Let y be a unit eigenvector of QᴴSQ, QᴴSQy = m y, v := Qy (unit, in range P). PSv = Q(QᴴSQ)y =
mv, so Sv − mv = (I − P)Sv. Put A′ := A − ζI (ζ the midpoint of J): A′ maps range P to itself with ‖A′v‖ ≤ w/2 (its
eigenvalues there lie in [−w/2, w/2]) and on range(I − P) its eigenvalues have modulus ≥ γ₀. Then
(I − P)[A′, S]v = (I − P)A′Sv − (I − P)SA′v = A′(I − P)Sv − (I − P)SA′v, so γ₀‖(I − P)Sv‖ ≤ ‖A′(I − P)Sv‖ ≤ c + ‖S‖₂w/2,
and the residual theorem gives an eigenvalue q of S with |q − m| ≤ (c + ‖S‖₂w/2)/γ₀. (b) XᴴSX = M(Q̂ᴴSQ̂)M with M ≻ 0,
spectrum in [√(1−η), √(1+η)]: Ostrowski gives |λ_i(XᴴSX) − λ_i(Q̂ᴴSQ̂)| ≤ η|λ_i(Q̂ᴴSQ̂)| ≤ η‖S‖₂ for each sorted
index. With E := Q̂ − QU (Lemma 3), Q̂ᴴSQ̂ − Uᴴ(QᴴSQ)U = Q̂ᴴSE + EᴴSQU, of norm ≤ 2(s + s²)‖S‖₂, and Uᴴ(QᴴSQ)U has the
eigenvalues m_i; Weyl gives |λ_i(Q̂ᴴSQ̂) − m_i| ≤ 2(s + s²)‖S‖₂. Together |m̂_i − m_i| ≤ (2s + 2s² + η)‖S‖₂. ∎
*γ₀ in the code.* γ₀ = sep + w/2 bounds the midpoint distance only if J = [a − ε, b + ε] with ε the quantity
subtracted in sep = dist − ε (dist measured from [a, b] = [min μ, max μ]). The code's J also contains the members'
S2 enclosures μ_i ± (rad_i + σ); for S4 clusters these radii are the cluster radius ε (S2 'cluster' method) and
ε is enlarged by lip·σ ≥ 3.5σ > σ, so J = [a − ε, b + ε] and γ₀ is correct. A direct γ₀ = min(ζ − U_below, L_above − ζ)
would not depend on this coincidence (remark).

**Proposition 3.** tr(P₁P₂) − tr(X₁X₁ᴴX₂X₂ᴴ) = tr((P₁ − X₁X₁ᴴ)P₂) + tr(X₁X₁ᴴ(P₂ − X₂X₂ᴴ)); Hölder with ‖P₂‖₁ = g and
‖X₁X₁ᴴ‖₁ = tr(X₁ᴴX₁) = Σ m_i² ≤ g(1+η₁) gives the two terms. tr(X₁X₁ᴴX₂X₂ᴴ) = tr((X₁ᴴX₂)(X₁ᴴX₂)ᴴ) = ‖X₁ᴴX₂‖_F²;
0 ≤ tr(P₁P₂) = ‖P₁P₂‖_F² ≤ g. ∎

**Lemma 5.** (a) |fl(HS) − HS| ≤ γ̇_{w_H}|H||S| entrywise (rows of H have ≤ w_H nonzeros; complex via Higham's
constants as in my C-04 note), the subtraction adds a relative u: total ≤ γ̇_{w_H+1}|H||S| + γ̇_{w_S+1}|S||H|
(γ_a + u + uγ_a ≤ γ_{a+1}; for the complex constant 2γ_{a+4} + u(1 + 2γ_{a+4}) ≤ 2γ_{a+5}). [H, S] is anti-Hermitian
for Hermitian H, S, hence normal: ‖·‖₂ = ρ ≤ ‖·‖_∞ ≤ ‖|C| + E‖_∞. (b) fl(Xᴴ(SX)): |·− XᴴSX| ≤ γ̇_{w_S+n}|X|ᵀ|S||X|
(two stages, Higham constants), ‖|X|ᵀ|S||X|‖₂ ≤ ‖X‖_F²‖|S|‖₂ ≤ ‖X‖_F²‖S‖_∞ (|S| symmetric), ‖X‖_F² ≤ g(1+η); the
Hermitian-part rounding adds u relative. (c) |Ĉ − X₁ᴴX₂| ≤ γ̇_n|X₁|ᵀ|X₂| entrywise, ‖·‖_F ≤ γ̇_n‖X₁‖_F‖X₂‖_F = e;
|‖Ĉ‖² − ‖C‖²| = (‖Ĉ‖ + ‖C‖)|‖Ĉ‖ − ‖C‖| ≤ (2‖Ĉ‖_F + e)e; summing g² squared moduli each formed as re² + im² (two
products, one addition) and accumulated: relative ≤ (1+u)²(1+γ_{g²−1}) − 1 ≤ γ_{g²+1} ≤ γ_{g²+2}. ∎
*Remark.* The code forms the squared moduli as `np.abs(C) ** 2` (a `hypot` followed by a square); if the
platform's `hypot` is only faithfully rounded (≤ 1 ulp, i.e. up to 2u relative), the per-term error is up to
(1+2u)²(1+u) − 1 ≈ 5u against the 3u that γ_{g²+2} leaves beyond the summation — a formal shortfall of ~2u on a
term that is itself ~u‖Ĉ‖²; `C.real**2 + C.imag**2` matches the lemma exactly.

## Code (current files)

* `sin_theta_davis_kahan` (davis_kahan.py 47–67): inputs from `enclosure._cluster_bound` 486–513 (`R_fro` = Rg_fro,
  the 2-norm of the vector of per-column residual upper bounds ≥ ‖R‖_F; `XR` ≥ ‖XᴴR‖₂ (Theorem 2′ accounting,
  checked in my thm2prime note); `delta` = up(1 − √(1−η)) + 4u; `spread` rounded up). Shift 56–58 (`add_up`, `up`),
  inv_sqrt/inv 59–60 (`up`, three roundings each), w′ 61 (`up`, four roundings), gap 62 (`sub_down`), s 65 (`up`).
  Hook: certify.py 1326–1338: `dist` from [a_, b_] to the outward ends of the other enclosures / window edges
  (S3's complete window makes every other eigenvalue lie there), `sub_down`; `sep = dist − eps` (1331);
  `certify_eigenspace(..., sin_theta_alt=s_dk)` (1338) → eigenspace.py 90–112: s₆ = up(ρ/sep), s = min, both
  recorded; `projector_error = up(s + η)`.
* `commutator_bound` 72–92: as Lemma 5 (a) with the absolute products' underestimation undone by up(1/(1 − γ̇));
  `inf_norm_upper(|C| + E)`, `up`. The reference H is `Href` (spec matrix) or the stored operator (certify.py 1372) —
  for a stored operator that is Hermitian only up to rounding the anti-Hermiticity of the exact commutator, and
  hence ‖·‖₂ ≤ ‖·‖_∞, holds only up to that rounding (remark).
* `approximate_symmetry_label` 95–163: M symmetrised (112), rounding 121–122 (γ̇_{w_S+n+4}·‖S‖_∞(1+η)g(1+4u) +
  10g²u‖S‖(1+η), the eigvalsh assumption recorded in the basis text), w rounded up (124), γ₀ = sep + w/2 rounded
  down (125), geom/sym with `up` (130–131), D by `add_up` (132); snapping requires D < half spacing (144–148).
* `certified_fidelity` 168–201: e (188; two products, covered by the (1+4u) of 190), cf = fro_upper(Ĉ) (189), rnd
  (190, `up`), proj (191, `up`), bound `add_up` (192), interval clipped to [0, g] with `sub_down`/`add_up`
  (193–194). `fidelity_from_certificates` 212–229 reads `sin_theta_bound`, `eta` of CERTIFIED S4 clusters.
* Section 3's line numbers are those of `track/new-statements`; the current ones are given above.

### Comparison with section 2 (read after the above)

Lemmas 1–3, Propositions 1–3 and Lemma 5 agree with my derivations step by step (same eigenbasis argument for
Lemma 1, same polar identities, the same commutator trick for Proposition 2 (a), Ostrowski + alignment for (b),
Hölder for Proposition 3).

**GAP (model shift in Proposition 1).** Section 2 does not treat the model shift; section 3 justifies the code's
increments by "R changes by at most σ‖X‖₂ ≤ σ√(1+η)". For the *Frobenius* norm used by Lemma 1 the increment is
‖ΔX‖_F ≤ σ‖X‖_F ≤ σ√(g(1+η)) — larger by √g for g ≥ 2, so the written route does not prove the implemented bound.
The bound is nevertheless valid: the 2-norm sin Θ theorem applies (the spectrum of H′_spec lies in
[a − w′, b + w′] and the other eigenvalues outside its (dist − w′)-neighbourhood — exactly the interval hypothesis
section 2 says "is not used"), and ‖R′_spec‖₂ ≤ ‖R‖_F/√(1−η) + σ ≤ (‖R‖_F + σ√(1+η))/√(1−η). My adversarial run
includes the Frobenius worst case (ΔX = σY with Y orthonormal, g up to 4): 67 cases, 0 violations. Correction:
proof text only (add the 2-norm argument, or change the Frobenius increment to σ√(g(1+η)) — the latter would
loosen s_DK and is not needed).

**Remarks.** (1) `certified_fidelity` forms the squared moduli as `np.abs(C) ** 2`; Lemma 5 (c)'s γ_{g²+2} assumes
re² + im² (a faithfully rounded `hypot` can cost ~2u more per term) — a formal shortfall on a term of relative size
u; `C.real**2 + C.imag**2` matches the lemma. (2) γ₀ = sep + w/2 is the midpoint distance only because the code's J
equals [a − ε, b + ε] (the members' S2 radii are the cluster radius and ε carries ≥ 3.5σ ≥ σ); a direct
γ₀ = min(ζ − U_below, L_above − ζ) would not rest on that. (3) The two-stage complex rounding constant γ̇_{w_S+n} in
Lemma 5 (b) needs the Higham-constant composition (my C-04 note); the code's index shift +4 covers it. (4) For a
stored reference operator that is Hermitian only up to rounding, the anti-Hermiticity of the exact commutator —
and ‖·‖₂ ≤ ‖·‖_∞ — holds up to that rounding. (5) Section 3 line numbers are those of `track/new-statements`.

**Verdict: GAP** (model-shift increment of ‖R‖_F in Proposition 1 not justified as written; the implemented bound
holds by the 2-norm sin Θ theorem; no certified bound affected).

### Runs

* `python docs/proofs/verify_E-07.py 11` (positional seed; fixed sizes): A 60, B 114, C 104 (+ negative control
  20/20), D 60 + 20 tight + 60 alignment (+ negative control 20/20), E 79, F 60; TOTAL 557 cases, 0 violations (49 s).
* `python docs/proofs/independent/adversarial_E-07_davis_kahan.py --cases 100 --seed 83`: Proposition 1 90 cases
  (10 skipped: η ≥ 1 or Ritz values not separated) + 67 model-shift cases with ‖ΔX‖_F = σ√g, Lemma 3 100 (s up to
  ~1, g = n − 1), Proposition 2 97 (c from 0 to 1e-2·‖S‖, w = 0 and w > 0) + `approximate_symmetry_label` end to end
  97, Proposition 3 / `certified_fidelity` 99 (identical, orthogonal and generic subspaces, g ≤ 3), Lemma 5 (a)
  `commutator_bound` 100 (±1 and 1 + 2⁻⁵² entries, scales 10^±30, diagonal S); 0 violations (79 s). Two oracle
  defects fixed on the way (binary64-orthonormal planted bases; mp roundoff of an exactly zero angle) — test
  artefacts.
