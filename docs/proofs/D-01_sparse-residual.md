# D-01: a-posteriori backward error of the sparse L D L^H factorization used for inertia counts (S3)

Derivation document in the D-59 format. It replaces, for the sparse S3 path, the a-priori SuperLU bound of 3.1
(known issue MS-M11: `certify_ed/inertia.py` 3.1.0, `inertia_sparse`, lines 150–167 of that version:
β = γ_{w+1}‖L‖_F‖U‖_F + ‖L‖_F‖U − DL^H‖_F + …, which rests on Higham 2002, Theorem 9.3, i.e. on the hypothesis that
every entry of L and U was produced by a standard inner-product update — a property of the factorization
*algorithm* that SuperLU's supernodal BLAS-3 kernels are not documented to have in that form). The new bound
rests only on the *stored* factors and on the standard model for the operations this code performs itself.

Notation as in `docs/theory.md` §1 and §3. S := A − σI (Hermitian, stored binary64, dimension n); P the permutation
SuperLU returns (`perm_c`, with `perm_r == perm_c` verified); S_P := P S P^T (a re-indexing of stored numbers, exact);
L the stored unit lower-triangular factor (c_i nonzeros in row i, w := max_i c_i); D the stored real diagonal of U
(the pivots); E := L D L^H − S_P the exact residual of the stored factors (Hermitian, since both terms are);
G := |L| |D| |L|^T (entrywise moduli, exact); u = 2⁻⁵³; γ_k := ku/(1 − ku); g_k := `gamma_dot(k)` (= γ_k real data,
2γ_{k+4} complex data); fl(·) one round-to-nearest binary64 operation. F_c := fl(L · fl(D L^H)) the computed
product and E_c := fl(F_c − S_P) the computed residual, both as SciPy's CSR kernels form them.

## 1 Statement

**Error model.** Standard model fl(a ∘ b) = (a ∘ b)(1 + δ), |δ| ≤ u, for every operation whose exact result is a
normal number or zero (Higham 2002, eq. 2.4); the sparse kernels of SciPy evaluate each entry of a product as a
sum of the exact products in some order, each operation rounded once (the γ_k bound of an inner product of k terms
holds for any order and with FMA; Higham §3.1). The underflow hypothesis of `docs/theory.md` §1 (every nonzero
operand ≥ 2⁻⁹⁰⁰) is assumed for S, L and D, as for the other standard-model bounds of the S3 path.

**Hypotheses.** (H1) S finite and Hermitian as stored; (H2) L finite with unit diagonal and zero strict upper
triangle after the permutation (checked at `inertia.py:158`), D finite real with no zero entry (`inertia.py:164`,
`:168`); (H3) n ≤ 2²⁶ and w ≤ 2²⁶ (second-order terms (nu)², (wu)² ≤ u/2, as in `thm1_fp-accounting.md`).

**Theorem D-01 (a-posteriori backward error).** Under (H1)–(H3), with t_ij := #{k : L_ik ≠ 0 and L_jk ≠ 0} ≤
min(c_i, c_j),

    (a)  |E_ij| ≤ |E_c,ij| / (1 − u) + g_{t_ij+1} G_ij          for every i, j   (entrywise, exact |E_ij|),

    (b)  ‖E‖₂ ≤ ‖E‖_∞ = max_i Σ_j |E_ij| ≤ max_i Σ_j ( |E_c,ij|/(1 − u) + g_{c_i+1} G_ij ),

    (c)  the value β returned by `ldl_residual.sparse_ldl_residual_bound` satisfies
         β ≥ ‖E‖₂ + 2u max_i |S_ii|,

and therefore, by Lemma 4 of `docs/theory.md` applied to the exact matrices L D L^H and S_P with the exact residual
E (and Weyl's inequality for the shift rounding, Appendix A of `docs/theory.md`, `appA_model-shift.md`):

    #{λ(A) < σ − β} ≤ ν ≤ #{λ(A) ≤ σ + β},      ν := #{k : D_k < 0}.

**Consequence for S3 (collar).** The count ν is usable exactly as before: `certify._certify_lowest` and
`_certify_window` require the claimed enclosures to lie outside the collar [σ − β, σ + β] (formed with
`fp.sub_down`/`fp.add_up`); nothing in the S3 logic changes, only the number β. The bound is tighter whenever the
measured residual is small — on the Heisenberg S^z = 0 blocks of E2 it is 4–50 times smaller than the 3.1 bound
(section 4) — and it is *valid without any hypothesis on SuperLU's arithmetic*: a factorization computed by any
means (another library, another precision, a corrupted memory) gives a correct sandwich or a large β.

**Soundness rule.** The a-priori bound is not combined with the new one by a minimum (the old bound is conditional
on an unproven property of SuperLU; the soundness rule of D-59 allows min(old, new) only when both are proven). It
remains available as `Options(inertia_method="sparse-apriori")` for comparison and is labelled conditional in
`docs/SCOPE_MATRIX.md`.

## 2 Proof

**Lemma R1 (exact algebra).** E is Hermitian: (L D L^H)^H = L D L^H (D real) and S_P^H = (P S P^T)^H = P S^H P^T = S_P.
For a Hermitian matrix ‖E‖₂ ≤ ‖E‖_∞ (Horn–Johnson, Matrix Analysis, Cor. 5.6.10: ‖E‖₂² ≤ ‖E‖₁‖E‖_∞ = ‖E‖_∞²).
Hence the row-sum bound (b) follows from the entrywise bound (a) and max_i Σ_j |E_ij| = max_i Σ_j |E_ji| (the two
triangles are mirror images), which is why the code forms the lower triangle only (lines 126–133).

**Lemma R2 (pattern).** (L D L^H)_ij = Σ_{k ≤ min(i,j)} L_ik D_k conj(L_jk) has at most t_ij ≤ min(c_i, c_j) nonzero
terms, and the product L[I, :] · (D L^H)[:, :b] for a row block I = [a, b) has at most Σ_{i∈I} c_i · (fill of the
rows) entries, every one inside the filled pattern of L L^H restricted to the block: the block memory bound of
`_row_blocks` (line 65, 2 nnz(L[I, :]) + |I| entries per temporary, three temporaries plus the S block) is an upper
bound on what SciPy allocates for the block (the CSR product allocates its exact output pattern).

**Step 1: the product fl(D L^H)** (line 106). Each entry fl(D_k conj(L_jk)) = D_k conj(L_jk)(1 + δ), |δ| ≤ u (a real
times a complex number: both components are rounded once; the conjugate is exact).

**Step 2: the computed product F_c** (line 126, SciPy `csr_matmat`). F_c,ij is the floating-point sum, in some order,
of the t_ij exact products L_ik · fl(D_k conj(L_jk)); real data: each term carries (1 + θ_{t_ij}) from the t_ij − 1
additions and one product, times the (1 + δ) of Step 1: in total 1 + θ_{t_ij+1} by Higham Lemma 3.1 (Λ1 of
`thm1_fp-accounting.md`), so

    |F_c,ij − (L D L^H)_ij| ≤ γ_{t_ij+1} Σ_k |L_ik| |D_k| |L_jk| = γ_{t_ij+1} G_ij.

Complex data: a complex product costs (1 + θ₃) in modulus (Higham Lemma 3.5, as in Step 1 of
`thm1_fp-accounting.md`), each complex addition one (1 + δ) per component, i.e. a relative u in modulus; each term
therefore carries at most 1 + θ_{t_ij+3} together with Step 1's (1 + δ): 1 + θ_{t_ij+4} ≤ 1 + θ_{t_ij+5}, and
γ_{t+5} ≤ 2γ_{t+5} = g_{t+1}. In both cases |F_c,ij − (L D L^H)_ij| ≤ g_{t_ij+1} G_ij.

**Step 3: the subtraction** (line 127). E_c,ij = (F_c,ij − S_P,ij)(1 + δ′), |δ′| ≤ u, so F_c,ij − S_P,ij =
E_c,ij/(1 + δ′) and |F_c,ij − S_P,ij| ≤ |E_c,ij|/(1 − u). With Step 2 and the triangle inequality,

    |E_ij| = |(L D L^H)_ij − S_P,ij| ≤ |F_c,ij − S_P,ij| + |(L D L^H)_ij − F_c,ij| ≤ |E_c,ij|/(1 − u) + g_{t_ij+1} G_ij,

which is (a). (For complex data the subtraction rounds each component: the modulus of the error is ≤ u times the
modulus of the exact difference, the same inequality.)

**Step 4: the row sums in floating point** (lines 108–115, 131–133, 143–156). The code bounds Σ_j |E_ij| by
Σ_j |E_c,ij|/(1 − u) + g_{c_i+1} (G 1)_i, using t_ij + 1 ≤ c_i + 1 and the monotonicity of γ. It computes:

- `rows_abs_i` = fl(Σ_j |E_c,ij|), assembled from the lower triangle (r_sum: row sums of the block's lower triangle,
  c_sum: column sums of its strict lower part, mirrored into the rows < b; lines 131–144). For complex data |E_c,ij|
  is one rounded square root of a rounded sum of two rounded squares: fl(|z|) ≥ |z|(1 − u)³ ≥ |z|(1 − γ₃). Each row
  total is a sum of at most n non-negative terms with at most len(blocks) + 1 further partial-sum additions, so
  fl(Σ) ≥ Σ_j |E_c,ij| (1 − γ₃)(1 − γ_{n+len(blocks)}) ≥ Σ_j |E_c,ij| (1 − γ_{m+2}) with m := n + len(blocks) + 1
  (Higham Lemma 3.3). Line 152 forms `resid_inf` = fl(fl(fl(max_i rows_abs_i · fl(1 + γ̂_m)) · (1 + 4u)) · (1 + 8u))
  with γ̂_m = `gamma(m)` ≥ γ_m(1 − u)² (Λ3 of `thm1_fp-accounting.md`) and three roundings. The inequality needed,

      (Λ-R1)  (1 − γ_{m+2}) (1 + γ_m(1 − u)²) (1 + 4u)(1 + 8u)(1 − u)³ ≥ 1/(1 − u)      for 2 ≤ m ≤ 2²⁶ + 2²⁰,

  holds: first order −(m + 2)u + mu + 12u − 3u − u = 5u ≥ 0 with margin 4u − O((mu)² + u²) > 0 for mu ≤ 1/8; the
  verification script checks it exactly on a grid of m (section 4). Hence `resid_inf` ≥ max_i Σ_j |E_c,ij|/(1 − u).
- `rounding_rows_i` = fl(g_{c_i+1} · G1_i) with G1 = fl(|L| (|D| ⊙ fl(|L|^T 1))) (lines 108–115): the column sums carry
  (1 − γ_{n−1}), the scaling by |D_k| one (1 − u), the matvec with c_i ≤ n terms (1 − γ_{c_i}), and the moduli of
  complex entries (1 − u)³ each (two per term): G1_i ≥ (G 1)_i (1 − γ_{2n+6}). g_{c_i+1} ≥ g-value · (1 − u)² (Λ3) and
  the product adds one (1 − u). Line 154 inflates by fl(1 + γ̂_{2n+8})(1 + 8u)² with three roundings, and

      (Λ-R2)  (1 − γ_{2n+6}) (1 − u)³ (1 + γ_{2n+8}(1 − u)²) (1 + 8u)² (1 − u)³ ≥ 1      for 1 ≤ n ≤ 2²⁶

  (first order 2u + 16u − 6u = 12u > 0; checked exactly by the script). So `round_inf` ≥ max_i g_{c_i+1} (G 1)_i ≥
  max_i Σ_j g_{t_ij+1} G_ij.
- `shift_term` = fl(2u · max|S_ii|) ≥ 2u max_i |S_ii| (1 − u), and line 156 `beta = up(resid_inf + round_inf +
  shift_term)`: two additions and the (1 + 8u) of `up` ≥ (1 − u)⁻³. Therefore β ≥ ‖E‖_∞ + 2u max|S_ii| ≥ ‖E‖₂ +
  2u max|S_ii|, which is (c).

**Step 5: from the stored S to A − σI.** The S3 caller forms S = fl(A − σI) (`inertia.py:146`): only the diagonal is
rounded, |S_ii − (A_ii − σ)| ≤ u|S_ii|/(1 − u) ≤ 2u|S_ii|; the diagonal perturbation has 2-norm ≤ 2u max_i|S_ii| =: s.
Thus ‖L D L^H − P(A − σI)P^T‖₂ ≤ ‖E‖₂ + s ≤ β, and Lemma 4 (Sylvester's law for the congruence L D L^H with
invertible L, Weyl's inequality for the perturbation of norm ≤ β) gives the sandwich. The count ν of negative
pivots is exact (signs of stored floats, `inertia.py:160–166`; no 2 × 2 blocks occur: SuperLU's symmetric mode
produces a diagonal D).

**Step 6: inconclusive paths.** A zero pivot (`inertia.py:164`), a row pivot (perm_r ≠ perm_c, `:153`), a non-unit
diagonal of L (`:158`) or a non-finite factor (`:168`; SuperLU returns inf/nan on overflow, raising no error) make
the result inconclusive; `sparse_ldl_residual_bound` raises `FloatingPointError` on non-finite factors as a second
guard. None of these paths produces a count.

## 3 Where the code implements it

| proof object | code | rounding direction |
|---|---|---|
| factorization, permutation check, unit diagonal, zero pivot | `certify_ed/inertia.py:148–166` | exact checks |
| S_P = P S P^T (ip = argsort(perm_c)) | `certify_ed/ldl_residual.py:104–105` | exact re-indexing |
| fl(D L^H) (Step 1) | `ldl_residual.py:106` | one rounding per entry, bounded in Step 2 |
| (G 1)_i, g_{c_i+1} (Step 4, rounding rows) | `ldl_residual.py:108–115` | inflated at line 154 by (1 + γ_{2n+8})(1 + 8u)² (Λ-R2) |
| row blocks (Lemma R2, memory) | `ldl_residual.py:65–84`, `DEFAULT_BLOCK_MB = 64` | partition independent of the thread count |
| F_c, E_c, lower triangle (Steps 2–3) | `ldl_residual.py:126–129` | the rounding is what Theorem D-01(a) bounds |
| row/column sums of |E_c| (Lemma R1) | `ldl_residual.py:131–144` | inflated at line 152 by (1 + γ_m)(1 + 4u)(1 + 8u), m = n + blocks + 1 (Λ-R1) |
| shift term s = 2u max|S_ii| (Step 5) | `ldl_residual.py:155` | upward (`up` at line 156) |
| β = up(resid + round + shift) (c) | `ldl_residual.py:156` | upward |
| result record (`residual` dict), method name `sparse-superlu-residual` | `inertia.py:187–189` | — |
| memory gate (fill model, `cost.sparse_inertia_mb`) | `inertia.py:214–223` | NOT_CHECKED (`resource_failure`), never a count |
| 3.1 a-priori bound kept as `sparse-apriori` | `inertia.py:171–185` | conditional (SCOPE_MATRIX) |

The collar logic that consumes β (`certify._certify_lowest`, `_certify_window`, `_edge_count_exact`,
`degeneracy.degeneracy_counts`) is unchanged.

## 4 Computational verification

`docs/proofs/verify_D-01.py` (run from the repository root, `python docs/proofs/verify_D-01.py`, 207 s):

- **Random cases (400):** sparse Hermitian S, n ∈ [2, 13], real and complex alternating, densities 0.15/0.3/0.6,
  entry scales 1, 10⁻³, 10³, diagonal shifts 0, ±3·scale (so that the inertia varies). For each: SuperLU as in
  `inertia_sparse`; (V1) ‖E‖_∞ ≤ β with E computed in exact rational arithmetic (`fractions.Fraction`, complex as
  pairs, moduli bounded above with a rational square root of relative slack 2⁻⁴⁰ — a slack in the *tested*
  quantity, which can only produce spurious violations, never hide one); (V2) the entrywise inequality (a), with
  E_c recomputed by the same SciPy operations and t_ij counted exactly; (V3) Lemma 4's sandwich against 60-digit
  mpmath eigenvalues; and the sanity check β ≤ 2⁻³⁰ max(1, max|S_ij|). The direct call uses `block_mb = 0.001` so
  that every case runs through many row blocks (the mirrored column sums and the block partition are exercised),
  and the script requires its β to equal the one `inertia_sparse` returns.
- **Scalar inequalities (Λ-R1), (Λ-R2):** checked exactly in rational arithmetic for m, n ∈ {1, …, 2·10⁴} ∪
  {2^j : 15 ≤ j ≤ 27} (reported as part of the adversarial count: one case per inequality).
- **Adversarial cases (15, 4 skipped as expected-inconclusive):** exact zero pivot [[0,1],[1,0]] (inconclusive);
  the zero matrix (inconclusive); pivots with growth 1/ε for ε = 10⁻⁶, 10⁻¹², 10⁻¹⁶ (β grows to the measured
  residual, sandwich holds); an eigenvalue exactly at σ (rank-1 [[1,1],[1,1]]: exact zero second pivot,
  inconclusive); a cancelling pivot (inconclusive); an eigenvalue 2⁻⁵² from σ with a nonzero pivot (sandwich
  holds: the eigenvalue lies inside the collar); entry scales 10¹⁵⁰ (overflow in the product → non-finite factors,
  inconclusive, counted as skipped), 10⁻¹⁵⁰ and 10³⁰⁰; n = 1; a diagonal matrix; a complex Hermitian tridiagonal;
  five dense-pattern n = 12 matrices.
- **Result:** 400 random + 17 adversarial cases (15 matrices + the two scalar inequalities), 4 skipped (hypothesis not met), **0 violations** (207 s).
- **Scale (not a correctness check):** on the Heisenberg chain S^z = 0 blocks, σ = −0.4N: n = 924, β = 4.4·10⁻¹²
  (3.1: 1.7·10⁻¹⁰); n = 3432, β = 4.1·10⁻⁹ (3.1: 3.1·10⁻⁸); n = 12870, β = 1.6·10⁻⁸ (3.1: 6.4·10⁻⁸); the measured
  ‖E_c‖_∞ itself is 10⁻¹⁴ … 10⁻¹¹ — the bound is dominated by the rounding term g_{c_i+1}(G 1)_i, i.e. by the fill.
  Times: factorization 0.0 / 0.6 / 20 s, residual 0.1 / 4 / 24 s (8 threads; 88 s on one thread). See
  `experiments/e2_s3_reach.py` and RESULT_scale.md for the E2 points.

## 5 Analytic cross-check

(i) *Exact factors.* If L and D are the exact factors of S_P (e.g. S diagonal: L = I, D = diag(S), or any S whose
factorization is exact in binary64), then E = 0, E_c = 0 and the theorem reduces to β ≥ 2u max|S_ii| ≥ s, the
shift rounding alone — the same term the dense path carries (`inertia_dense`, `U * max|S_ii| * 2`). The diagonal
adversarial case realises this: β = s·(1 + 8u) exactly up to the guard.

(ii) *Reduction to the classical bound.* Dropping the measured term, (a) gives |E_ij| ≤ γ_{t_ij+1} G_ij ≤ γ_{w+1}
(|L||D||L|^T)_ij when F_c happens to equal S_P exactly — the entrywise form of Higham's Theorem 9.3 for a
factorization whose *product* was evaluated by standard inner products. The new bound thus contains the structure
of the old one, but its hypothesis is now about the product this code forms (which it controls), not about
SuperLU.

(iii) *Limiting case n = 1.* L = [1], D = [S₁₁], E = 0: β = 2u|S₁₁|(1 + 8u), and Lemma 4 states #{λ < −β} ≤ [S₁₁ < 0]
≤ #{λ ≤ β}, trivially true for the single eigenvalue λ = S₁₁ (the n = 1 adversarial case).

## 6 Status

Derived by Claude on 2026-10-31; machine-checked (400 random + 17 adversarial cases, 0 violations); independent
re-derivation: pending (Phase 2); Sarang's check (D-12): pending.
