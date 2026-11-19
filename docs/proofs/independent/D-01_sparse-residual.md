# Independent re-derivation: D-01 (a-posteriori backward error of the sparse L D Lᴴ factorisation)

Not a human check. Written from sections 1 and 3 of `docs/proofs/D-01_sparse-residual.md` only (section 2 read
afterwards; comparison at the end). Notation as there. Classical facts: Higham (2002) eq. (3.5) (inner products, any
order, FMA allowed), Lemma 3.3; Sylvester's law of inertia; Weyl; ‖E‖₂ ≤ ‖E‖_∞ for Hermitian E; theory.md Lemma 4
(inertia sandwich from a backward error).

## Proofs

**What is bounded.** Only the stored numbers enter: L (unit lower triangular after the permutation), D_r := Re of
the stored pivots (the code discards imaginary parts; the factorisation whose inertia is counted is the Hermitian
L D_r Lᴴ, whose inertia is #{k : D_r,k < 0} by Sylvester, L being unit triangular hence nonsingular), S_P := P S Pᵀ.
Nothing about how L, D were produced is used; in particular U = D Lᴴ is never assumed, and the permutation
convention is irrelevant for soundness — the inertia of S_P equals that of S for every permutation P, and the
residual is measured against the S_P actually used (a wrong P would only inflate E_c, never shrink the true error).

**Theorem D-01 (a).** F_c,ij = fl(Σ_k L_ik fl(D_k conj L_jk)) over the t_ij indices k with L_ik L_jk ≠ 0. Each inner
factor fl(D_k conj L_jk) = D_k conj(L_jk)(1 + δ_k), |δ_k| ≤ u (D_k real: componentwise rounding, modulus error ≤ u);
the inner product of t_ij terms then carries, per term, (1 + δ_k)(1 + θ_k) − 1 with |θ_k| ≤ γ_{t_ij} (real data) —
together ≤ γ_{t_ij+1} (Lemma 3.3) — or, for complex data, the complex inner-product constant ≤ γ_{t+2} plus the
pre-rounding: ≤ γ_{t+3} ≤ 2γ_{t+5} = g_{t+1}. Hence |F_c − L D Lᴴ|_ij ≤ g_{t_ij+1} G_ij. E_c = fl(F_c − S_P) entrywise:
|E_c,ij − (F_c,ij − S_P,ij)| ≤ u|F_c,ij − S_P,ij| ≤ u|E_c,ij|/(1 − u). Therefore
E_ij = (L D Lᴴ − S_P)_ij = E_c,ij + [(F_c − S_P) − E_c]_ij − [F_c − L D Lᴴ]_ij and
|E_ij| ≤ |E_c,ij|(1 + u/(1 − u)) + g_{t_ij+1}G_ij = |E_c,ij|/(1 − u) + g_{t_ij+1}G_ij. ∎
**(b).** E is Hermitian (both L D_r Lᴴ and S_P are), so ‖E‖₂ ≤ ‖E‖_∞ = max_i Σ_j |E_ij|, and t_ij ≤ c_i with g monotone.
E_c itself need not be Hermitian (the sparse kernel may sum the k's in different orders for (i, j) and (j, i)); the
code uses, for j > i, the mirrored entry E_c,ji: valid because |E_ij| = |E_ji| ≤ |E_c,ji|/(1 − u) + g_{t_ji+1}G_ji and
G_ji = G_ij, t_ji = t_ij. ∎
**(c).** The code's `resid_inf` is max_i of floating-point row sums of |E_c| (lower triangle + mirrored column sums):
each row sum has ≤ n terms accumulated in ≤ (blocks + 1) partial sums, relative error ≤ γ_{n+blocks} (any order),
the moduli |E_c,ij| (hypot for complex) ≤ 2u each, the division by (1 − u) and two inflation products ≤ 3u: the
factor (1 + γ_m)(1 + 4u)(1 + 8u), m = n + blocks + 1, covers (1 + γ_{n+blocks})(1 + 2u)(1 + u)(1 + 2u). `round_inf`:
(G1)_i = Σ_k |L_ik||D_k| Σ_j|L_jk| is computed as column sums (≤ n non-negative terms, γ_n), a scaling (u), a matvec
(≤ c_i ≤ n terms, γ_n) and complex moduli (u each): shortfall ≤ γ_{2n+3} < γ_{2n+8}; g_{c_i+1} carries two roundings and
the product one: (1 + 8u)². The shift term: S = fl(A − σI) differs from A − σI only on the diagonal, by ≤ u|S_ii|, so
‖S − (A − σI)‖₂ ≤ u max_i|S_ii| ≤ 2u max|S_ii|. β = up(resid + round + shift) ≥ ‖E‖_∞ + ‖S − (A − σI)‖₂. ∎
**Sandwich.** ν = ν(L D_r Lᴴ) (Sylvester). ‖L D_r Lᴴ − P(A − σI)Pᵀ‖₂ ≤ ‖E‖₂ + ‖S − (A − σI)‖₂ ≤ β, and P(A − σI)Pᵀ has
the eigenvalues λ_j(A) − σ. Weyl: ν ≥ #{λ_j(A) − σ < −β} and ν ≤ #{λ_j(A) − σ ≤ β}, i.e. #{λ(A) < σ − β} ≤ ν ≤
#{λ(A) ≤ σ + β}. ∎

**Soundness remark.** The bound is self-correcting: corrupted factors, pivots with discarded imaginary parts, a
different permutation convention or a non-standard factorisation algorithm change E_c and ν together and leave the
sandwich valid — only the hypotheses (H1) (S Hermitian, finite), the standard model for SciPy's own kernels, and the
absence of underflow in the residual computation are needed. Underflow: a product L_ik D_k conj L_jk below 2⁻⁹⁰⁰ in
modulus would violate the standard model for that operation; the document assumes the underflow hypothesis for S,
L, D (the S3 scan of the operator does not scan L and D — remark).

## Code (current files)

* `inertia.inertia_sparse` 137–189: SuperLU symmetric mode (148–149), perm_r = perm_c required (153–155), unit
  diagonal of L (158–160), real parts of the pivots and the zero-pivot refusal (161–167), finiteness (168–170), the
  3.1 a-priori bound kept under `residual="apriori"` (171–184, labelled conditional), the new bound (185–189).
* `ldl_residual.sparse_ldl_residual_bound`: ip = argsort(perm), S_P = S[ip][:, ip] (exact re-indexing); fl(D Lᴴ) via
  `diags(Dr) @ L.conj().T`; (G1)_i by column sums, scaling and one matvec; g_{c_i+1} = `gamma_dot(c_i + 1)`; row blocks
  from `_row_blocks` (L and block_mb only — thread-count independent); per block F_b = L[a:b] @ (D Lᴴ)[:, :b], E_b =
  F_b − S_P[a:b, :b], lower triangle `tril(k=a)`, row sums and strict-lower column sums mirrored; inflations
  (1 + γ_m)(1 + 4u)(1 + 8u) and (1 + γ_{2n+8})(1 + 8u)²; shift term 2u max|S_ii|; β = up(sum). The record carries
  residual_inf, rounding_inf, shift_term, nnz, blocks, threads.
* Section 3's table matches the current code (line numbers of `inertia.py` are 148–170 for the checks; the module
  layout of `ldl_residual.py` agrees with the description).

## Comparison with section 2 (read after the proof above)

Section 2 (Lemma R1, R2, Steps 1–6) follows the same route as my proof: entrywise standard-model accounting of
fl(D Lᴴ), the CSR product and the subtraction (Steps 1–3 = my (a)), the Hermitian row-sum argument with the mirrored
lower triangle (Lemma R1 + Step 4 = my (b)), the floating-point evaluation of the two row-sum bounds with the
inequalities (Λ-R1), (Λ-R2) and the shift term (Step 4–5 = my (c) and the sandwich), and the inconclusive paths
(Step 6). The constants agree with mine: Step 2 gives |F_c,ij − (L D Lᴴ)_ij| ≤ g_{t_ij+1} G_ij with g = `gamma_dot`
(real γ_{t+1}; complex 2γ_{t+5} ≥ γ_{t+4}), Step 4's accounting of the row sums (γ₃ for the complex moduli,
γ_{n+len(blocks)} for the summation, three roundings of the inflations) and of G1 (γ_{2n+6}, (1 − u)³) matches my
shortfall bookkeeping, and the sandwich is Sylvester + Weyl as in mine.

**Verdict: AGREES-WITH-REMARKS.**

1. *(Λ-R1) first-order margin.* The text computes "−(m + 2)u + mu + 12u − 3u − u = 5u"; the sum is 6u. Harmless
   (the margin is larger than stated; the script checks the inequality exactly), but the arithmetic should be fixed.
2. *Complex moduli.* Step 4 models fl(|z|) as a rounded square root of a rounded sum of two rounded squares,
   (1 − u)³. NumPy evaluates |z| with the C library `hypot`, whose accuracy is implementation-defined (typically
   ≤ 1 ulp, i.e. within (1 − 2u), inside the (1 − u)³ budget) — the standard-model clause of section 1 ("each
   operation rounded once") does not cover a library function; the assumption |fl(|z|) − |z|| ≤ 3u|z| should be stated
   explicitly. The same applies to the moduli |L_ik|, |L_jk| in G1 (Step 4, second item) and to the moduli in
   `rows_abs`.
3. *Pivots.* D := Re(diag U) (`inertia.py:161`): for complex data SuperLU's pivots may carry a rounding-level
   imaginary part; the counted factorisation is L·Re(D)·Lᴴ, and the discarded imaginary parts land in E (my
   adversarial part A checks the exact residual of exactly this factorisation). Section 2 writes "D real" without
   saying where the real part is taken — a sentence in Lemma R1 would make the Hermiticity of L D Lᴴ visibly a
   consequence of the code, not of SuperLU.
4. *E_c is not Hermitian.* Lemma R1 says "the two triangles are mirror images": true for the exact E, not for E_c
   (F_c,ij and F_c,ji are rounded sums in different orders). The code is right — it applies (a) to the lower entry
   (i ≥ j) and uses E_ji = conj E_ij exactly — but the sentence "which is why the code forms the lower triangle only"
   should say so (the bound on row i uses |E_c,ij| for j ≤ i and |E_c,ji| for j > i, with G_ji = G_ij).
5. *Hypotheses carried, not checked.* (H1) Hermiticity of S is inherited from the caller (the S3 callers pass
   `construct.hermitian_part(H)`, exactly Hermitian — `certify.py:413`, `compose.py:54, 78, 81`; `inertia_sparse`
   itself does not check it), and the underflow
   hypothesis for L and D is assumed (section 1 says so) but not scanned — the S3 operand scan covers the operator
   only. Both are stated in section 1; I record them as the hypotheses a reader of the code must supply.
6. *U is discarded.* The factorisation whose residual is measured is L·D·Lᴴ, not SuperLU's L·U; nothing about the
   off-diagonal of U is used, so the result does not depend on SuperLU's arithmetic at all (as section 1 says), only
   on SciPy's product/subtraction kernels and NumPy's reductions — the line of `inertia.py` that keeps the 3.1
   a-priori bound under `residual="apriori"` is the only place where SuperLU's arithmetic matters.

Code mapping (step 3 of the protocol): `ldl_residual.py` lines cited in section 2 (65, 106, 108–115, 126–133,
143–156) agree with the current file; `inertia.py:146, 153, 158, 160–168` as cited. Rounding directions: every
inflation is upward ((1 + γ̂)(1 + 4u)(1 + 8u), (1 + γ̂)(1 + 8u)², `up`), no downward rounding is needed anywhere,
and the shift term 2u·max|S_ii| is exact (power-of-two scaling). No code change.

## Runs

* `python docs/proofs/verify_D-01.py --cases 60 --seed 11` (reduced; default 400 cases): 60 random + 17 adversarial cases,
  4 skipped (hypothesis not met), 0 violations.
* `python docs/proofs/independent/adversarial_D-01_sparse-residual.py --cases 40 --seed 103`: 0 violations
  (exact-rational residual of SuperLU's factors, n ≤ 60, real and complex, dynamic range up to 10⁻⁸..10⁸, three
  block sizes; entrywise (a) with t_ij ≤ min(c_i, c_j); mp inertia sandwich at random shifts and at shifts 10⁻⁹/10⁻¹³
  from an eigenvalue; β identical for threads 1 and 4; corrupted L / flipped pivot / wrong permutation keep the
  sandwich); re-run of the committed file after a cosmetic edit with `--cases 16 --seed 104`: 0 violations.
