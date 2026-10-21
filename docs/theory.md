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
CERTIFIED statements when one fails. Two quantities are computed more accurately: Gram entries and
Rayleigh quotients use error-free products and exactly rounded sums, so each has error ≤ 2u|·|.

**Endpoints are rounded outward (since 3.1).** The certified statement |λ − μ| ≤ r is about the stored
numbers μ and r. An endpoint μ − r computed with round-to-nearest can land half an ulp *inside* the
exact interval: for example, 1 − 2⁻⁶⁰ rounds to 1.0. Every endpoint, shift margin (σ ∓ β), separation
and cut comparison is therefore formed with `fp.sub_down` / `fp.add_up`, which step one ulp outward.
The exact layer does not need this.

## 2. Enclosures (S2)

**Theorem 1 (residual).** For x ≠ 0 and real μ, some eigenvalue satisfies |λ − μ| ≤ ‖Ax − μx‖/‖x‖.

This gives existence only: two copies of one vector satisfy it with the same eigenvalue.

**Theorem 2 (clusters with non-orthonormal vectors).** Let X be n × g, Λ = diag(μ) real,
R = AX − XΛ and η = ‖X^H X − I‖ < 1. Set δ = 1 − √(1 − η) and ρ = ‖R‖ + (‖A‖ + ‖Λ‖)δ. Let Q̂ be the
polar factor of X, w ≥ ‖Q̂^H(AQ̂ − Q̂Λ)‖ and ε = (w + √(w² + 4ρ²))/2. Then there are g *distinct*
indices i_j with |λ_{i_j} − μ_j| ≤ ε. For orthonormal Ritz pairs this reduces to Kahan's theorem,
ε = ‖R‖. **UNVERIFIED** in this form.

**Theorem 2′ (sharper cluster radius; since 3.1.0, review finding IMP-13).** Under the hypotheses of
Theorem 2 (η < 1; Λ Hermitian, in the implementation Λ = diag(μ) real), with δ = 1 − √(1 − η) and
spread(Λ) = λ_max(Λ) − λ_min(Λ), there are g distinct indices i_j with

    |λ_{i_j} − μ_j| ≤ ε* := ‖R‖/√(1 − η) + ‖X^H R‖/(1 − η) + δ·spread(Λ)/√(1 − η).

`enclosure._cluster_bound` computes both ε and ε* (each term an upper bound, guarded with `fp.up`) and
uses min(ε, ε*) as the cluster radius. For exact eigenvectors of one degenerate level ε* is of the
order of the residual, where ε is dominated by (‖A‖ + ‖Λ‖)δ.
**[UNVERIFIED — re-derivation pending, D-12].** Numerical evidence: 15 876 random cases with 0
violations (`thm2_checks.py`, review of 2026-10-21).

*Proof sketch.* Polar decomposition X = Q̂M with M = (X^H X)^{1/2}; the eigenvalues of M lie in
[√(1−η), √(1+η)], so ‖M − I‖ ≤ 1 − √(1−η) = δ and ‖M⁻¹‖ ≤ (1−η)^{−1/2}.
(i) *Kahan.* H′ := Q̂^H A Q̂ is Hermitian. From AX = R + XΛ, A Q̂ = R M⁻¹ + Q̂ M Λ M⁻¹, hence
(I − Q̂Q̂^H) A Q̂ = (I − Q̂Q̂^H) R M⁻¹ and ‖A Q̂ − Q̂ H′‖ ≤ ‖R‖/√(1−η). Kahan's theorem (Parlett 1998,
Thm 11.5.1; valid for any Hermitian H′, not only diagonal) matches the eigenvalues θ_1 ≤ … ≤ θ_g of H′
to g *distinct* eigenvalues of A within ‖R‖/√(1−η).
(ii) *Weyl.* Using Q̂^H = M⁻¹X^H, H′ − Λ = M⁻¹ X^H R M⁻¹ + (MΛM⁻¹ − Λ), and MΛM⁻¹ − Λ = [M, Λ]M⁻¹ =
[M − I, Λ − cI] M⁻¹ for any scalar c; with c the midpoint of the spectrum of Λ, ‖Λ − cI‖ = spread/2, so
‖H′ − Λ‖ ≤ ‖X^H R‖/(1−η) + δ·spread/√(1−η). H′ − Λ is Hermitian, so Weyl's inequality pairs the sorted
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

## 3. Counting (S3)

**Lemma 4 (inertia with backward error).** If ‖LDL^H − P(A − σI)P^T‖ ≤ β for unit lower-triangular L
and block-diagonal D with ν negative eigenvalues, then #{λ < σ − β} ≤ ν ≤ #{λ ≤ σ + β}. ν is counted
exactly: 2 × 2 blocks by the sign of a rational determinant.

**Corollary 5 (completeness).** If k claimed values have distinct enclosures inside (−∞, σ − β) and ν = k,
they enclose λ₁…λ_k, and λ_{k+1} ≥ σ − β.

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
and its polar factor Q̂ is unitary. With R̂ = AQ̂ − Q̂B = (AW − WB) + A(Q̂ − W) − (Q̂ − W)B and
‖Q̂ − W‖ = ‖I − M‖ ≤ δ, ‖R̂‖ ≤ ‖AW − WB‖ + (‖A‖ + ‖B‖)δ = ρ. Q̂^H A Q̂ − B = Q̂^H R̂ is Hermitian and
Q̂^H A Q̂ has the spectrum of A, so Weyl's inequality pairs the sorted eigenvalues of A and B within ρ.
`compose.py` therefore uses ε = ρ (+ the builder shift) where it used (1 + √5)/2 · ρ before.
**[UNVERIFIED — re-derivation pending, D-12]** (a one-line consequence of Weyl, but new in the project).

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
