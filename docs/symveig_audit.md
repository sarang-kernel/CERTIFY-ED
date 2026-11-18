# symveig 1.0.0 and its CPC manuscript: audit and merge into CERTIFY-ED (v0.1, 2026-10-18)

Inputs: `symveig-1.0.0.zip` (package), `symveig-manuscript.zip` (elsarticle source), `symveig_cpc.pdf`
(18 pp.); the package is kept unchanged in `legacy/symveig_1_0_0/`. Status: findings are reproducible. The scripts are
`experiments/e8_symveig_comparison.py` and `tests/test_spectrum.py`, and the snippets
quoted below.

## 1. What symveig is
It encloses all eigenvalues of a Hermitian matrix. Each computed pair (μ, x) gets the per-vector
residual bound |λ − μ| ≤ ‖Ax − μx‖/‖x‖. The floating-point error is bounded with
γ₄ₙ‖A‖_F‖x‖ for the matrix-vector product. Overlapping intervals are merged and reported with a
multiplicity. An optional sector path splits A by the diagonal of a conserved quantity (S_z) and
encloses each block. The package's own suite passes (11 tests), and the paper's Table 1 numbers
reproduce here (for example, sector h_max 2.77e-11 at L = 12, PBC).

## 2. Problems found (each one reproduced)
| # | Claim in paper/code | Finding | Evidence |
|---|---|---|---|
| P1 | "The interval … contains exactly `multiplicity` true eigenvalues … the union over all returned objects contains the full spectrum" (§2.4, §3.2, `Enclosure` docstring) | Not proved. A per-vector interval contains *at least one* eigenvalue, not a distinct one. The count needs a certified bound on the orthogonality of the vectors (a Kahan-type cluster theorem), and symveig never checks X^H X. With producer-supplied eigenpairs (`eigh_result`, the advertised use for "validating approximate eigensolvers") it is false: A = diag(0, 1) with two copies of e₁ gives one interval [−1.4e-15, 1.4e-15], "multiplicity 2", and the eigenvalue 1 lies outside every interval. With LAPACK's own vectors the claim happens to hold, because X is orthonormal to ~nu, but nothing in the code establishes it. | `test_duplicated_eigenvector_is_not_counted` |
| P2 | `enclose_sectors(H, S_diag)` "verifies each sector" | It never checks that H conserves S. It slices out the blocks and encloses them. With a small staggered transverse field (0.05) on the L = 6 Heisenberg chain, 58 of 64 true eigenvalues lie outside every "verified" enclosure, and the reported ground-state interval [−2.80277564, −2.80277564] excludes the true −2.86389024. | `test_non_conserved_quantity_rejected` |
| P3 | Validation: "every reference eigenvalue … computed by an independent LAPACK call is contained" (§4.1) | For the global path the reference is the same `eigh` result whose eigenvalues are the interval midpoints (`benchmark.py`: `ref = mu_global`), so containment is automatic. The sector-path check is genuine. A float reference is also only accurate to ~n u‖A‖, which is *wider* than tight enclosures (E8 shows offsets up to 9e-15). Containment against a float reference passes here only because symveig's bounds are loose. | `benchmark.py` l. 58–93; `tests/test_spectrum.py::_contained` |
| P4 | "to our knowledge, the first such tool in the scientific-Python stack"; "existing verified eigensolvers … require MATLAB and the commercial INTLAB toolbox" | Too strong. Arb, via python-flint (`acb_mat.eig`, Rump's algorithm), gives rigorous eigenvalue enclosures in Python. In our test it fails to isolate degenerate lattice spectra: Heisenberg L = 8 and 10 S_z = 0 sectors fail even at 256 bits with `multiple=True`. That is the defensible version of the gap. | snippet in session 2026-10-18 |
| P5 | Sector path "tighter by 3–9×" | True for symveig's own bound, but it is an artefact of the γ₄ₙ‖A‖_F form: the bound grows with dimension while the true error does not. On the same sectors, a |A||x| bound with row-width γ factors (CERTIFY-ED) gives median half-widths 11–29× smaller at L = 12 (run of 10-21). The "133× speedup" is the standard O(n³) → Σd_s³ gain of block diagonalisation, measured against a complex-arithmetic `eigh` of a real matrix. | E8 |
| P6 | Tier 2 (Rump–Lange Lemma 6.1, opt-in) | Unverified. The gap is taken from midpoints, the formula "ρ + 2ρ²/gap + spread" is not the lemma as stated in the paper, and orthonormality is assumed (`g_orth = k γ_n`), not checked. Off by default, but it is shipped. Recommend removing it, or marking it UNVERIFIED. | `certify_eig._cluster_tighten` |
| P7 | No check of the floating-point environment | FTZ/DAZ, rounding mode and reduced-precision BLAS would silently void the γ bounds. CERTIFY-ED's run-time probe covers this. | — |
| P8 | Minor | Hermitian symmetrisation 0.5(H + H^H) is applied silently, so the enclosure refers to the symmetrised matrix. Model builders use complex128 for real models. The manuscript still contains placeholders (REPLACE_WITH_EMAIL, REPLACE_WITH_USER). | — |

## 3. What carries into the one paper (D-33)
* **Function, rebuilt correctly.** `certify_ed.spectrum.enclose_spectrum(H, S_diag=None, eigh_result=None)`
  gives the same whole-spectrum, optionally sectored enclosure, with these changes:
  - counts are certified by the cluster theorem (η < 1), otherwise reported as `existence-only`;
  - `complete` holds only when all counts are certified and sum to n;
  - an exact check that no stored entry couples two sectors;
  - an exact Hermiticity requirement (no silent symmetrisation);
  - |A||x| rounding bounds.

  Tests: `tests/test_spectrum.py` (9: symveig's synthetic classes plus both counterexamples).
* **E8.** symveig against certify_ed on symveig's 17 benchmark configurations, with the same
  matrices. Every certify_ed enclosure is complete (counts certified). In the run of 2026-10-21, at
  L = 12 (d = 4096): median half-widths are 11–29× smaller; max half-widths are 1.3–12× smaller; and
  certify_ed takes 0.9–2.3× symveig's time (|A||x| products and Gram matrices). At d ≤ 64, the max
  half-widths can be up to 3× larger, because cluster intervals carry the count guarantee. (Run of
  10-18: median 11–28×; the earlier "max 1.7–12×, 1.4–4.3× slower" came from the first E8 run, 10-18 13:32.)
* **Contributions of symveig to keep in the paper:**
  1. the plain statement that sector decomposition shrinks *dimension-dependent* rounding bounds;
  2. the observation that the form of the rounding bound decides how widths scale (γ₄ₙ‖A‖_F gives
     h ≈ 6 u d ‖A‖_F, while |A||x| tracks row width) — a small figure;
  3. the Arb/INTLAB positioning, corrected per P4;
  4. the "verify an approximate eigensolver" use case, which is exactly where P1 bites. It makes a
     good worked example of why counts need the cluster theorem.

## 4. Decisions needed from Sarang
1. **Do not submit the CPC manuscript as it stands.** P1–P4 are claims a referee can falsify with a
   10-line script. Under D-33 the content goes into the one paper instead. If a separate CPC
   submission is still wanted, the fixes are P1 (add the Gram check plus cluster theorem, or drop
   "exactly"), P2 (exact commutation check), P3 (independent reference, or remove the global-path
   claim), P4 (reword) and P6 (remove Tier 2). Effort: about 1–2 days of code and text.
2. **arXiv:2606.16217 v1** was posted on 2026-06-15 and is the only version. Its abstract matches the CPC version, except that it says "up to 130×". Its body was not checked here. Options: post a
   v2 noting P1/P2, or replace it when the merged paper is posted. This is Sarang's call; not urgent,
   but before anyone cites it.
3. Whether the merged repository should keep the symveig 1.0.0 API names as a thin compatibility
   layer (`enclose_global`, `enclose_sectors` over `enclose_spectrum`). Not done yet. Cheap if wanted.
