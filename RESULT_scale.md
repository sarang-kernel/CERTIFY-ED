# RESULT: track/scale (D-01, D-02)

Branch `track/scale` on `dev/3.2` (ac52920). Head: the commit that adds this file (`git log`). Bundle `track-scale.bundle`,
patches `patches_scale.zip` (`git format-patch dev/3.2..track/scale`).

Deviations from the plan: (1) the E2 reach is a reproduction on the Heisenberg chain S^z = 0 blocks
(N = 12–18, d = 924–48620; no record of `results/run_full_20261022` is in the clone), with S1–S3 only and the
eigenpairs from `scipy.sparse.linalg.eigsh` instead of QuSpin; d = 184756 was not attempted (the fill model
predicts far above 8 GiB, as it already does at 48620). (2) The D-01 verification uses n ≤ 13 for the exact
rational residual (the check is O(n⁴) in rational arithmetic); the sandwich on random sparse matrices up to
n = 160 (dense `eigh` ground truth) and the chain blocks up to 12870 is covered by the tests and the reach run; an
independent Sylvester-on-blocks count above d = 3000 was not done. (3) The D-02 scale run uses an exact one-magnon
eigenvector (certified radius at rounding level), not a Lanczos result; it measures time and memory.

## Item 1 — D-01 a-posteriori residual for the sparse S3 path (MS-M11): done

- Document: `docs/proofs/D-01_sparse-residual.md` (Statement, Proof with Lemmas R1–R2 and Steps 1–6, code map,
  verification, analytic cross-check, status line). Script `docs/proofs/verify_D-01.py`: **400 random + 17
  adversarial cases (15 matrices + the two scalar inequalities Λ-R1/Λ-R2), 4 skipped (expected-inconclusive:
  zero pivot, zero matrix, rank-1 eigenvalue at σ, cancelling pivot), 0 violations**, 202 s.
- Implementation: new `certify_ed/ldl_residual.py` (`sparse_ldl_residual_bound`: residual of the stored factors
  in deterministic row blocks, lower triangle only, mirrored row sums, γ_{c_i+1}(|L||D||L|ᵀ1)_i rounding rows,
  outward inflation (1 + γ_m)(1 + 4u)(1 + 8u) and (1 + γ_{2n+8})(1 + 8u)²; threads over the blocks, partition
  independent of the thread count so the recorded β is machine-independent). `inertia.inertia_sparse` defaults
  to it (method `sparse-superlu-residual`, record field `residual`); the 3.1 bound is `inertia_method=
  'sparse-apriori'` (conditional on SuperLU's arithmetic, not used by default, not min-combined — soundness
  rule). `inertia()` refuses a sparse factorization whose predicted peak exceeds `budget_mb` (resource failure →
  S3 NOT_CHECKED). `cost.sparse_inertia_mb` gained `sparse_residual_mb` (resident D Lᴴ, |L|, P S Pᵀ plus
  1.2 × block budget per live thread; measured peaks 105 / 1484 MB at d = 3432 / 12870 against predictions
  172 / 1900).
- Collar: unchanged logic in `certify._certify_lowest/_certify_window/_edge_count_exact` and
  `degeneracy.degeneracy_counts`; only β changes. Out-of-core factorization: not implemented (stated in the
  document, SCOPE_MATRIX and the NOT_CHECKED note).
- E2 reach reproduction (`experiments/e2_s3_reach.py`, `results/e2_s3_reach.json`; budget 7000 MB, 8 threads,
  k = 4 lowest plus the multiplet closure, S1–S3):

| N | d | method | S3 | S3 seconds | β | peak / predicted MB |
|---|---|---|---|---|---|---|
| 12 | 924 | sparse | CERTIFIED | 3.0 | 8.98e-11 | 68 / 13 |
| 12 | 924 | sparse-apriori | CERTIFIED | 2.7 | 3.73e-10 | 70 / 13 |
| 14 | 3432 | sparse | CERTIFIED | 15.5 | 2.39e-09 | 167 / 172 |
| 14 | 3432 | sparse-apriori | CERTIFIED | 8.9 | 7.19e-09 | 143 / 172 |
| 16 | 12870 | sparse | CERTIFIED | 95.4 | 1.19e-06 | 1554 / 1900 |
| 16 | 12870 | sparse-apriori | CERTIFIED | 61.6 | 5.84e-06 | 1059 / 1900 |
| 18 | 48620 | sparse | NOT_CHECKED | 0.0 |  | 180 / 19250 |

  The a-posteriori β is 4–5× below the 3.1 bound at the E2 shifts (up to 50× at other shifts, e.g. 4.4e-12 against
  1.7e-10 at d = 924, σ = −4.8); d = 48620 is refused by the memory gate (19.25 GB predicted; both the 4.5 GiB cap
  of E2 and the 8 GiB of this task are exceeded), so the S3 reach in dimension is unchanged on this machine —
  what changed is that the bound at the certified points no longer rests on an unproven property of SuperLU, and
  that the refusal is predicted before any allocation.
- Tests: `tests/test_sparse_residual.py` (46 passed): sandwich on 30 random sparse Hermitian matrices (4 shifts
  each) and 4 planted-inertia matrices, exact rational entrywise check with many blocks, determinism across
  threads/block sizes and through `inertia()`, block coverage, a-priori method still available and never
  tighter than the measured one on the chain, memory gate, adversarial (zero pivot, eigenvalue at σ, 2⁻⁵² from σ,
  growth 1/ε pivots, scales 1e160/1e300, non-finite factors refused). Negative controls: a corrupted L entry, a
  flipped pivot sign and a wrong permutation raise β (≥ 10³×); a claim omitting the lowest eigenvalue is not
  certified through `certify()`. `tests/test_planner.py` updated for the method name (its memory-peak test
  passes with the new term).

## Item 2 — D-02 matrix-free residual bounds from the spec: done

- Document: `docs/proofs/D-02_matrix-free.md` (Theorem D-02 (a)/(b), Lemmas M1–M3, N; code map; verification;
  cross-checks; status). Script `docs/proofs/verify_D-02.py`: **182 random + 18 adversarial cases, 0 violations**,
  435 s (60-digit exact spec operators from Kronecker products with the Jordan–Wigner parity; exact rationals
  where the amplitudes are rational). One script revision during the work: the adapter's Appendix-A row-sum
  norm bound is checked only for Hermitian specs (it needs Hermiticity, the standing hypothesis of `certify()`;
  Lemma N holds for any spec) — the three non-Hermitian random specs that exposed this were not violations of
  anything the code uses.
- Implementation: new `certify_ed/matrixfree.py` (`MatrixFreeOperator`: bit-identical application, full-space
  index map, `terms_per_row` = Σ_t Π_l c(o_l), `rho`, `W`, `rounding_width = W − 2`, `norm_bound()` =
  Σ|c_t|Π√(‖o‖₁‖o‖_∞), `amplitude_hypothesis`); `enclosure.OperatorAdapter` spec branch takes `width = min(3.1
  count, rounding_width)` and `norm = min(row-sum bound, spec bound)` (both proven); `certify.Options.matrix_free`
  (default False): a claim without an operator is certified at any dimension with no reference matrix, S1
  CERTIFIED with `model_shift_bound = 0` and the amplitude hypothesis in the evidence and statement text, S3/S8
  NEEDS_MATRIX as before. For spin-1/2 Heisenberg the rounding width drops from 4T + 6 to T + 1 (radii ≈ 3.6×
  smaller than the 3.1 spec path at N = 10; the stored-matrix path is still ≈ 2× tighter).
- Scale (`experiments/d02_matrixfree_scale.py`, `results/d02_matrixfree_scale.json`; Heisenberg chain, full
  space, k = π one-magnon state as a float64 memmap, `Options(matrix_free=True, only=("S1","S2"))`, one process
  per point, Windows 8 cores 16 GiB):

| d | S2 | S2 seconds | one matvec (s) | radius | W | ‖H‖ bound used | peak MB |
|---|---|---|---|---|---|---|---|
| 2^16 = 65536 | CERTIFIED | 8 | 3 | 3.96e-14 | 51 | 12 | 67 |
| 2^20 = 1048576 | CERTIFIED | 169 | 40 | 6.34e-14 | 63 | 15 | 257 |
| 2^22 = 4194304 | CERTIFIED | 653 | 210 | 7.71e-14 | 69 | 16.5 | 869 |
| 2^24 = 16777216 | CERTIFIED | 1896 | 734 | 9.21e-14 | 75 | 18 | 3160 |

  Memory stays under 3.2 GB at 2²⁴ (the per-pass temporaries of the bitwise application). The time is the
  Python-level term-by-term application (about 60 vectorised passes per product; S2 costs one product, one
  |H|_abs pass, the norm pass and the underflow probe); a compiled kernel would change the constant, not the bound.
- Tests: `tests/test_matrixfree.py` (14 passed): bit identity with `SpecOperator` on six model families and
  agreement with the stored matrix, counts (w_row = n_terms for spin-1/2, 2^L for spin-1 x/y, ρ = 2 for spin-1),
  norm bound vs the dense spectral norm and the adapter's minimum, radii ≤ the 3.1 spec path and containing the
  eigenvalues, end to end on a memmap vector (S1 hypothesis text, S2 CERTIFIED, S3 NOT_CHECKED/NEEDS_MATRIX,
  radius ≤ the reference-matrix path's), spin-1 hypothesis recorded, cancelling terms, couplings 1e±150 (radius
  at the INT-05 floor). Negative controls: a 1 % wrong coefficient raises the radii by > 10³ and S2 is not
  CERTIFIED; a planted V₀ + 10⁻²V₅ fails S2.

## Shared files and expected conflicts

`certify_ed/certify.py` (Options field `matrix_free`, one branch at the top of `check_model_fidelity`'s
non-sector path, three lines in the operator selection), `certify_ed/enclosure.py` (spec branch of
`OperatorAdapter.__init__` only), `certify_ed/cost.py` (new functions after `sparse_fill_estimate`,
`sparse_inertia_mb` extended, `__all__`), `certify_ed/inertia.py` (sparse path), `CHANGELOG.md` (one new
`### Scale track` section before the new-statements section), `docs/theory.md` (one paragraph at the end of §2,
one at the start of §3), `docs/SCOPE_MATRIX.md` (ground-state and producer-format rows extended),
`tests/test_planner.py` (method name). No schema change (no new reason code, statement id or schema-level
evidence key; the inertia record's `residual` dict and the S1 evidence `amplitude_hypothesis` are free-form
evidence). New files: `certify_ed/ldl_residual.py`, `certify_ed/matrixfree.py`,
`docs/proofs/D-01_sparse-residual.md`, `docs/proofs/verify_D-01.py`, `docs/proofs/D-02_matrix-free.md`,
`docs/proofs/verify_D-02.py`, `tests/test_sparse_residual.py`, `tests/test_matrixfree.py`,
`experiments/e2_s3_reach.py`, `experiments/d02_matrixfree_scale.py`, `results/e2_s3_reach.json`,
`results/d02_matrixfree_scale.json`. Other tracks touching certify.py / CHANGELOG / theory.md / SCOPE_MATRIX
merge additively; the CHANGELOG section and the theory.md paragraphs are self-contained blocks.

## Not done

- Out-of-core sparse factorization (D-01 target d = 10⁶): not implemented; the memory gate reports it.
- An independent Sylvester-on-blocks count above d = 3000 as ground truth for the sandwich (the mathematics is
  covered by the exact rational check at small n and the dense `eigh` ground truth to n = 160).
- A CLI flag for `matrix_free` (API option only); `cost.plan_certification` does not model the matrix-free S2 time.
- D-02 for sector-isometry claims (the operator acts on the full space or a U(1) state set).
- Two safety notes on the shared `certify.py`: the matrix-free branch is reached only with `Options(matrix_free=True)`
  and `claim.operator is None`; nothing else in `certify()` changes.

## Suite

`python -m pytest -q` on track/scale (Windows, PYTHONSAFEPATH removed): **1107 passed, 8 skipped, 5 warnings in 5228.10s (1:27:08)** (dev/3.2 at ac52920: 1047 passed, 8 skipped; +60 tests from this track). `ruff check certify_ed tests experiments conformance corpus docs/proofs`: All checks passed!.
