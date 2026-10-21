# Applications: numerical lemmas of published proofs

Two published rigorous results depend on floating-point exact-diagonalisation numbers. In both cases
CERTIFY-ED replaces those numbers with certified ones. Run them with `python run.py --only E9,E10`
(add `--full` for the full set), or call `certify_ed.apps` directly.

## 1. Pyrochlore Heisenberg antiferromagnet (Kránitz & Penc, arXiv:2601.07800)

Their Anderson-type bound uses the lowest eigenvalue of an 18-site "crown" cluster, which they
obtained with Lanczos. We use the reduced model with c3 = c12 and c5 = 0. The couplings are
c1 = n1/10⁷ and c3 = n3/10⁷, and c17 is fixed by the embedding constraint
2c1 + 4c3 + 4c12 + 2c17 = 1.

| Spin | Result | Status | Code |
|---|---|---|---|
| 1/2 | λ_min(H_crown) ≥ −0.550158299 (exact rational bound −0.55015829882…) at n1/n3 = 381414/817957 (published −0.550158(2); λ_min ≈ −0.5501582979) | certified, exact arithmetic (Z₂⁷ blocks, largest 3890) | `apps.pyrochlore.spin_half.certify_crown` |
| 1 | at the printed couplings (c1 = 0.053466, c3 = 0.082726) the all-singlet sector has eigenvalue ≤ −1.6380246 < −1.632985 (published bound); margin 5.0e-3 | certified in exact arithmetic (six-ring energy re-derived in the scaled spin-1 basis; radius below 1e-14) | `apps.pyrochlore.spin_one.printed_coupling_check` |
| 1 | re-optimised max-min over all 92 pair-spin classes ≈ −1.6343596 | floating-point estimate (the competitive all-S̃=2 class has 1.1M states) | `apps.pyrochlore.spin_one.landscape` |

The S = 1 discrepancy may come from an unprinted coupling (c5). The authors should be asked before
anything about it is stated publicly.

## 2. AKLT spectral gaps on 2D lattices (Pomata & Wei, arXiv:1911.01410)


Source: Pomata & Wei, "Demonstrating the AKLT spectral gap on 2D degree-3 lattices",
arXiv:1911.01410 (PRL 2020). In that paper the gap of each model follows from a finite-size criterion
η < 1/z̃ (square-octagon: 4η⊥ + 2η∥ < 1). η is the largest cosine, other than 1, of the principal angles
between the ground spaces of two overlapping subgraph Hamiltonians, and it was computed in floating
point (full diagonalisation or Lanczos). Here every η is replaced by an upper bound proved in exact
arithmetic.

### What is certified, per lattice

| Lattice | Criterion | η (paper) | Certified upper bound | Holds | Kernel equality | `certify_lattice(...)` |
|---|---|---|---|---|---|---|
| star [3.12.12] | η < 1/4 | 0.1110430220 | 0.1110430221 | yes | lemma + exact check (6 sites) | `"star"` |
| square-octagon | 4η⊥ + 2η∥ < 1 | 0.1589663310 / 0.1061446858 | 0.1589663312 / 0.1061446859 (sum 0.84816) | yes | lemma + exact check (8 sites) | `"square_octagon_perp"`, `"square_octagon_par"` |
| cross [4.6.12] | η < 1/3 | 0.1997384500 | 0.1997384508 | yes | lemma (12 sites, 16.7M states: not checked separately) | `"cross"` |
| decorated honeycomb (n=1) | η < 1/4 | 0.1530329085 | 0.1530329086 | yes | lemma + exact check (both Γ, 9 sites) | `"decorated_honeycomb"` |
| decorated square (n=1) | η < 1/4 | 0.2203543174 | 0.2203543194 | yes | lemma (12 sites, 4.1M states) | `"decorated_square"` (mirror × spin flip; largest exact block 4034) |
| honeycomb | η < 1/6 (Method III) | 0.1445124916 | not attempted | — | — | see below |

Negative control: on the decorated honeycomb, the same code at t = 0.0234 < η² (η = 0.15297) finds
218 > 216 eigenvalues above the threshold and reports "not certified". It does not certify a bound that
is false.

### Chain of rigour (all lattices)
1. **VBS maps.** These are exact integer matrices in the scaled basis D = diag(√binom(z,k)). Boundary
   configurations are reduced to per-site down counts, and U_X is checked to be injective (exact rank of
   its Gram matrix).
2. **Ground spaces.** ker H_Γ = range(U_Γ) by the kernel lemma (§3 below,
   re-derived, needs a human check), confirmed independently by exact mod-p nullity counts where the
   size allows.
3. **Reduction to V = range U_L ⊗ range U_C ⊗ range U_R.** The parts of A and B outside V are
   orthogonal to the other subspace (argument in the `apps/aklt/eta.py` docstring), so this is exact.
4. **Cosines.** The cosines are the eigenvalues of the pencil (C G_B⁻¹ Cᵀ, G_A). One formulation for all lattices: the bordered matrix [[−pG_A, qC],[qCᵀ, −qG_B]] with Haynsworth
   inertia additivity, which needs no inverse (`apps/aklt/eta.py`). The first scripts (2026-10-17,
   in `legacy/`) used an adjugate-scaled matrix for star, square-octagon and cross. The unified code
   reproduces their bounds to 9–10 digits.

   Everything is split into S^z blocks. The lattice mirror (decorated square) and the spin flip act by
   verified signed permutations and give an exact integer congruence into character blocks.
5. **Inertia.** A floating-point LDLᵀ is re-checked exactly (FLINT integers). With residual bound β,
   Weyl's inequality and the bound (G_B + β)⁻¹ ≥ G_B⁻¹/(1 + β/μ) give #(cos² > t′) ≤ #positive pivots,
   where t′ = (1 + β/μ)(t + β/ν) and μ, ν are certified lower bounds on λ_min of the Gram factors.
6. **Unit cosines.** dim(A ∩ B) ≥ the rank of the union VBS states. The rank is computed exactly, or
   as a mod-p rank, which is a valid lower bound.

No step depends on the floating-point model.

Bounds are from results/run_full_20261021 and are rounded up at the 10th decimal. Each criterion is established at a sharp threshold t = η_float²(1 + 1e-9), with fallbacks 1e-7 and 1e-5 that were not needed. The certified η then exceeds the published η by 2e-11 to 2e-9; on the cross and the decorated square most of the excess is the factorization term t′ − t.
The criterion itself is decided by comparing the certified t′ with 1/z̃² exactly, in rationals. The older
certificate files in `legacy/applications_2026-10-17` also contain runs at t = 1/z̃². Their t′ exceeds
1/z̃² by a few 1e-8, so those runs do not by themselves prove the strict inequality. Their "certified"
flag only records the eigenvalue count. In every run the floating-point factorizations only propose
factors, and an exact computation checks them.

### Not attempted: honeycomb (Method III)
Its V has dimension 2²⁶ (paper, Table I). The final gap bound 0.00266 also uses three further
numerical constants (γ_p, γ^[2p], γ^[4p]), which are separate eigenvalue computations on plaquette
clusters. Each of those would itself need a certificate. Too large for this machine. It is listed as
the natural next target.

### Remaining caveats
* The subgraph geometry for each lattice is inferred from the paper's text. It is confirmed by
  reproducing the paper's η to 7–10 digits in floating point before certifying.
* The kernel lemma is a standard argument re-derived here. It is marked for Sarang's check (D-12).
* Pomata–Wei's criterion itself (the operator inequality H̃² ≥ (1 − z̃η)H and its finite-size
  reduction) is taken from the paper and not re-proved here.


## 3. The kernel lemma

Status: standard argument re-derived here (Schwinger-boson / polynomial form of VBS states, as in
Affleck–Kennedy–Lieb–Tasaki 1988 and Kirillov–Korepin 1989). Per D-12 it is marked
**RE-DERIVED, NEEDS HUMAN CHECK** until Sarang has read it. It is also checked by exact computation
for three subgraphs (below), which do not depend on the argument.

### Statement
Let Γ be a finite simple graph. Give vertex v spin S_v with 2S_v ≥ deg_Γ(v), and let
H_Γ = Σ_{(u,v) ∈ E(Γ)} P^{(uv)}_{S_u+S_v}, where P_J is the projector onto total spin J of the pair.
Then ker H_Γ is exactly the span of the VBS states of Γ with arbitrary boundary spins, i.e. the range
of the reduced VBS map U_Γ used in `apps/aklt/graph.py` (one column per pattern of down-spin counts on each
site's free virtual spins). Its dimension is ∏_v (2S_v − deg_Γ(v) + 1).

### Proof
1. *Polynomial picture.* The spin-S_v space is the space of homogeneous polynomials of degree 2S_v in
   two variables (a_v, b_v), with SU(2) acting linearly on (a_v, b_v). The many-site space is the
   space of polynomials in all variables that are homogeneous of degree 2S_v in each pair.
2. *One edge.* ω_uv = a_u b_v − b_u a_v is SU(2)-invariant. The polynomials divisible by ω_uv form an
   invariant subspace, ω_uv · (bidegree (2S_u−1, 2S_v−1)), isomorphic to V_{S_u−1/2} ⊗ V_{S_v−1/2}, of
   dimension 2S_u · 2S_v. Its complement has dimension (2S_u+1)(2S_v+1) − 4S_uS_v = 2(S_u+S_v)+1,
   which is the dimension of the J = S_u+S_v irrep. V_{S_u−1/2} ⊗ V_{S_v−1/2} contains only J ≤ S_u+S_v−1,
   and the decomposition of V_{S_u} ⊗ V_{S_v} has no repeated J. So the divisible subspace is
   exactly the sum of the J < S_u+S_v components: ker P_{S_u+S_v} = ω_uv · (…).
3. *All edges.* H_Γ is a sum of positive semidefinite terms, so ker H_Γ = ∩_e ker P_e: the
   polynomials divisible by every ω_e. Each ω_e is an irreducible quadratic (a rank-4 quadratic form),
   and distinct edges give non-associate factors. The polynomial ring is a UFD, so divisibility by
   all of them is divisibility by Ψ = ∏_e ω_e. Hence ψ = Ψ · φ, with φ homogeneous of degree
   2S_v − deg_Γ(v) in each site's variables.
4. *Identification.* Ψ is the VBS state of Γ: one singlet per edge, symmetrised at each site. A
   monomial φ = ∏_v a_v^{α_v} b_v^{β_v} is a boundary configuration with α_v up and β_v down free virtual
   spins at v, and depends only on these counts. Multiplication by Ψ is injective, since the ring
   has no zero divisors. So ker H_Γ = range(U_Γ), with the stated dimension. ∎

### Exact computational confirmation (independent of the proof)
`apps/aklt/kernel.py`. The '≥' direction uses exact integer VBS states with H U = 0 and their exact rank.
The '≤' direction uses nullity over Q ≤ nullity mod p, per S^z block, with p = 2^31 − 1. No floating
point is used.

| Γ | sites | dim | rank U | nullity mod p | equal |
|---|---|---|---|---|---|
| star (two triangles + bridge) | 6 | 4 096 | 16 | 16 | yes |
| square-octagon (two squares + bridge) | 8 | 65 536 | 64 | 64 | yes |
| decorated honeycomb Γ_i ("H") | 9 | 62 208 | 36 | 36 | yes |
| decorated honeycomb Γ_j | 9 | 62 208 | 36 | 36 | yes |

The cross (12 sites, 16.7M states) and decorated-square (12 sites, 4.1M states) subgraphs rely on the
lemma alone.
