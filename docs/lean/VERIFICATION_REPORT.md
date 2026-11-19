# VERIFICATION_REPORT.md: the Lean re-verification of CERTIFY-ED

This report states what the Lean 4 work in `lean/` checked, how strongly, and what it did not check. It was generated on 12 November 2026 by `tools/make_verification_report.py` from the files in `lean/results/`; every number below is copied from one of them, and the last section lists them. The toolchain is Lean 4.35.0 and Mathlib v4.35.0 (commit 56a005855ea1). The objects of the check are the certificates, derivation documents and run records of CERTIFY-ED as they are in the project archive v5 (`archive_v5/`, code at `dev/3.2` commit `a814d28`, source hash `cb73dec3e4433a3b`), and the documentation corrections of the code repository that this work led to (commits `40d59f4` and `fa86456`, `docs/project/sources/SOURCES.md`). The parts added on 11 November 2026 (the non-Hermitian statements of E-02, the refinement of the kernel checks, the certificate driver and the applications N-11 and N-12) refer to the code branches `fix/3.2.2` (tag `v3.2.2`) and `dev/3.3`, also described in `docs/project/sources/SOURCES.md`; their accounts are `NONHERMITIAN.md`, `LEAN_PRODUCT.md` and `APPS_NEXT.md`.

## How to read the levels

Every result carries one level of the scale defined in `docs/project/README.md`. Level K is a Lean theorem accepted by the Lean kernel, with no `sorry`, no `native_decide` and no added axiom; the words used for it are "kernel-checked in Lean". Level K2 is a level K theorem that the independent kernel nanoda also accepted, after the theorem and everything it depends on (Lean core and Mathlib included) were exported with `leanexport`. Level K+ is a level K theorem whose module `leanchecker` replayed; this repeats Lean's own kernel on the compiled files and guards against a defect of the elaborator or of the files, not against a defect of the kernel. Level N is a compiled Lean program that recomputed a number or a status from raw data and agrees with the record; the program itself is not a theorem. A level applies to a statement as it is written in Lean. How far the Lean statement says what the document says is described in the column `correspondence` of the result files and in `ENCLOSURE.md`, `COUNTING.md`, `CORE.md` and `APPS.md`; that comparison was made by reading and is not part of any level. None of these levels is a check by a person in the sense of rule D-12; the last sections explain what that means for the derivation documents.

## Summary

| Part | What | Result | Level |
|---|---|---|---|
| build | `lake build CertifyCore CertifyMathlib CertifyApps cecheck ceapps` | exit code 0, 2888 jobs, 0 errors; warnings: 23 about Mathlib's module system, 6 about the exponent threshold of `FloatUnderflow.lean`; modules compiled by this build: `CertifyMathlib`, the others replayed | K |
| axiom audit | `Audit/AxiomAudit.lean`: every theorem of the three libraries | CertifyCore: 164 named and 265 auxiliary theorems, axioms Classical.choice, Quot.sound, propext, 0 with any other axiom; CertifyMathlib: 916 named and 558 auxiliary theorems, axioms Classical.choice, Quot.sound, propext, 0 with any other axiom; CertifyApps: 87 named and 0 auxiliary theorems, axioms propext, 0 with any other axiom | K |
| text search | the library files of CertifyCore, CertifyMathlib, CertifyApps | CertifyCore: 21 files, 0 `sorry`, 0 `admit`, 0 `native_decide`, 0 `axiom`, 0 `debug` options; CertifyMathlib: 48 files, 0 `sorry`, 0 `admit`, 0 `native_decide`, 0 `axiom`, 0 `debug` options; CertifyApps: 14 files, 0 `sorry`, 0 `admit`, 0 `native_decide`, 0 `axiom`, 0 `debug` options | K |
| exact checker `cecheck` | the 55 validation certificates, their 18 FAILED statuses and 80 forgeries | 55 of 55 certificates pass; 18 of 18 FAILED statuses reproduced (15 also refuted by exact counts); 80 of 80 forgeries rejected; agreement with the C verifier on 135 of 135 rows | N |
| CertifyCore theorems | the finite arithmetic behind the bounds (Theorem 1 guards, INT-05, Lemma 17 constants, E-04, counting) and three certificates checked in the kernel | 76 theorems; nanoda accepted 76 of 76 | K2 |
| CertifyMathlib.Enclosure | Theorems 1 to 3, the floating-point model and accounting, Appendix A, Kahan, polar factor, E-03, E-05, D-02, Lemma 17 | 22 statements: 18 complete, 4 partial; 355 theorems; nanoda accepted 63 of 63 main theorems | K2 |
| CertifyMathlib.Counting | Sylvester, Lemma 4, Corollary 5, D-01, Theorems 6 and 8, E-07, P14 to P16, C-04, C-06, C-07, C-12, C-13, E-04 | 27 statements: 18 complete, 7 partial, 2 not formalized; 291 theorems; nanoda accepted 71 of 71 main theorems | K2 |
| CertifyMathlib.Apps | bordered-pencil lemma, eta bound, finite-size criterion, AKLT kernel certificate | 37 theorems; nanoda accepted 15 of 15 main theorems | K2 |
| CertifyMathlib.NonHermitian | Bauer-Fike, its residual form, and the disc inclusion of statements N2 and N3 of E-02 (UNVERIFIED in the code) | 9 statements: 4 complete, 2 partial, 3 not formalized; 16 theorems; nanoda accepted 6 of 6 main theorems | K2 |
| CertifyMathlib.Refine | acceptance by the kernel checks `pairOK` and `checkLowest` of CertifyCore implies the conclusion of `residual_enclosure` for the decoded rational data; the count from the pivots of a rational elimination | 17 theorems; nanoda accepted 3 of 5 theorems tried (2 stopped at the memory limit of 4096 MiB); the others replayed by leanchecker | K2 for 3, K+ for 11 |
| certificate driver | `Verify/cecheck_bundle.py`: export of any certificate, `cecheck`, one verdict per statement | 9 of 9 genuine certificates pass (exit code 0), 3 of 3 forgeries rejected (exit code 1); exports byte-identical to `export/export_certs.py` on 9 of 9 | N |
| `ceapps` | the certified application numbers (E7, E9, E10) | 14 of 14 targets agree with the 3.2.1 records; a final re-run with one binary reproduced 14 of 14 results exactly | N |
| final replay | `leanchecker` on all 78 modules of the three libraries as they were on 11 November 2026; the modules added since were replayed by their tracks (rows above and the table of second checkers) | exit code 0, 5961.9 s | K+ |
| nanoda recheck | the main theorems of CertifyMathlib.Counting.D01Float and P15Float, which the build of 11 November 2026 rebuilt because enclosure modules they import had changed | 14 of 14 accepted | K2 |

A replay of all 83 modules after the build of 12 November 2026 did not run: stalled at start: leanchecker held about 24 MiB of private memory, one thread spent its time in system calls and no module was read; stopped after the time given (exit code 15). A probe on the single module CertifyCore.Kernel.Defs, also with an absolute LEAN_PATH, stalled the same way, while lake build and lake env lean ran normally in the same environment (18171.4 s). The compiled files of the 78 modules replayed on 11 November 2026 are the ones this build reused (it compiled only the modules listed in the first row), and the modules added since, `CertifyMathlib.NonHermitian` and `CertifyMathlib.Refine`, were replayed by their tracks with exit code 0 and 0.

The second checker `lean4lean` could not be used: its build reads no compiled file of Lean 4.35.0 ("incompatible header", `results/core/second_kernel.json`), so K2 rests on nanoda alone.

## The statements of the documents

The three tables list every statement that was taken up, with the main Lean theorems and what is not formalized. "Complete" means that the whole statement of the document has a Lean counterpart at the given level, under the correspondence described in the track document; "partial" means that a named part is missing, and the last column says which. The source paragraph of each statement is given in the result file.

### Enclosures and the floating-point model (`CertifyMathlib.Enclosure`, `ENCLOSURE.md`)

| Id | Statement | Lean | Level | Main theorems | Not formalized, and notes |
|---|---|---|---|---|---|
| T1 | Theorem 1 (residual enclosure) | complete | K2 | `residual_enclosure` |  |
| T1-family | Theorem 1, multi-pair form used by S2 | complete | K2 | `residual_enclosure_family` |  |
| T1-shift | Theorem 1 with the S1 model shift sigma | complete | K2 | `residual_enclosure_of_shift`, `residual_enclosure_of_l2_opNorm` |  |
| FP-model | Floating-point model: standard model, INT-05 additive term, directed rounding; Higham Lemmas 3.1 and 3.3, sums in any order, dot product and matrix-vector bounds, complex product (Higham Lemma 3.5) | complete | K2 | `FP.higham_lemma_3_1`, `FP.gam_succ`, `FP.STree.comp_error`, `FP.dot_error`, `FP.cmul_err` |  |
| T1-fl | Theorem 1 floating-point accounting, real data | complete | K2 | `FP.residual_radius_fl`, `FP.residual_enclosure_fl` |  |
| T1-fl-complex | Theorem 1 floating-point accounting, complex data | complete | K2 | `FP.residual_radius_fl_c`, `FP.residual_enclosure_fl_c` |  |
| T1-fl-int05 | Theorem 1 floating-point accounting under the underflow-tolerant model (INT-05, Theorem U5), real data | complete | K2 | `FP.residual_radius_int05`, `FP.residual_enclosure_int05` | Complex data under the underflow-tolerant model (residual_entry_count with 4w + 2 real products, T_N with N = 2n) is not formalized; the complex accounting T1-fl-complex is proved in the standard model only. That binary64 arithmetic and NumPy satisfy UModel (in particular that the sum of two binary64 values is exact when it underflows, and that ldexp rounds to the subnormal grid with error at most u_min) is part of the correspondence and is not checked by Lean. |
| T3 | Theorem 3 (Kato-Temple) | complete | K2 | `kato_temple` |  |
| Gershgorin | Gershgorin for Hermitian matrices and the norm bound \|\|A\|\|_2 <= \|\|A\|\|_inf | complete | K2 | `eigenvalue_mem_gershgorin`, `l2_opNorm_le_of_row_sum` |  |
| Weyl | Weyl's inequality (sorted and counting forms) | complete | K2 | `weyl_eigenvalues₀`, `card_le_le_card_le_add` |  |
| A1-A2 | Appendix A, Lemmas A1 and A2 (builder rounding, symmetrisation rounding) | complete | K2 | `FP.builder_error`, `FP.symmetrised_error`, `FP.mono_error`, `FP.abs_sum_bounds_real`, `FP.abs_sum_bounds_complex` | Document note: for complex amplitudes the factor of the second sentence of Lemma A1 is 1 +- gamma_(m+2L), one index more than the document states, because np.abs of a complex number is rounded. That a row or column of the operator has at most m monomials is the size of the summation tree (a hypothesis, as for an entry). |
| A3 | Appendix A, Lemma A3 (\|\|E\|\|_2^2 <= \|\|E\|\|_1 \|\|E\|\|_inf <= max^2) | complete | K2 | `schur_bound`, `FP.builder_shift`, `FP.symmetrised_shift` |  |
| A4 | Appendix A, Theorem A4 (the model shift Delta) | complete | K2 | `FP.model_shift_fl`, `model_shift_bound` | The theorem takes r, c, R, C as upper bounds. For r and c computed from abs_matvec(ones) and abs_rmatvec(ones), the second sentence of Lemma A1 (A1-A2, abs_sum_bounds_real, abs_sum_bounds_complex) gives exact sum <= computed sum / (1 - gamma_(m+2L-1)) (gamma_(m+2L) for complex amplitudes); the rounding of up(g sqrt(row col)/(1 - gamma(width))(1 + 4u)) and inf_norm_upper of the computed D are not formalized. |
| A5 | Appendix A, Theorem A5 (Weyl transfer for S2 and S3) | complete | K2 | `weyl_transfer_enclosure`, `weyl_transfer_count`, `FP.weyl_transfer_fl` |  |
| polar | Polar-factor lemma (a) to (d) | complete | K2 | `polarQ_norm`, `sub_polarQ_le`, `polarQ_residual_le` |  |
| Kahan | Kahan's inclusion theorem (general Hermitian block) | complete | K2 | `kahan`, `kahan_antitone` |  |
| T2 | Theorem 2 (clusters, radius rho) | complete | K2 | `cluster_enclosure` |  |
| T2prime | Theorem 2' (sharper radius eps*) and min(rho, eps*) | complete | K2 | `cluster_enclosure_eps_star`, `cluster_min` |  |
| E-05 | E-05 variational certificates | partial | K2 | `rayleigh_ritz_matrix`, `temple_matrix`, `rayleigh_perturb` | Lemma 3: the computed bound E_N with fl(\|psi\|^T a_hat)(1 + gamma_(n+2))/(1 - gamma_(w+1)) and its guard, the outward 4u of the corners, r_stored and the rounding of the Temple bound (v), and complex data; Lemma 4 (MPS contraction) is not formalized. |
| D-02 | D-02 matrix-free residual bounds | partial | K2 | `FP.matrixfree_entry_error`, `FP.matrixfree_entry_error_c`, `FP.norm_kronecker_le`, `FP.d02b_key`, `FP.d02b_radius` | Lemma M1 (at most w_row contributions per output entry) enters as the size of the summation tree; Lemma M3 (the second pass, computed \|H\|\|x\| >= a (1 - gamma_(W-1))) is a hypothesis of d02b_radius; part (b) is proved for real data in the standard model only (complex data and the INT-05 additive terms on the matrix-free path are not formalized). The reverse inequality \|\|A (x) B\|\| >= \|\|A\|\| \|\|B\|\| of the Kronecker identity and the rounding of norm_bound() are not formalized. Document note: the argument of the document that (b) holds by the slack of one gamma index should be replaced by the second-order inequality d02b_key: with adapter.width = W - 2 the residual error index gamma_(W-1) + u(1 + gamma_(W-1)) is covered by gamma_W and the rounding factors of the code only with a margin of about W u^2. |
| E-03 | E-03 generalized eigenproblems (Cholesky reduction) | partial | K2 | `generalized_residual_enclosure_beta`, `pencil_roots_eq_eigenvalues`, `pencil_inertia`, `pencil_count_perturbed` | Propositions 1 to 3 (the Bunch-Kaufman count of B - beta I and of fl(A - fl(sigma B)), and the floating-point residual radii) are not formalized; the step from \|\|M' - (A - sigma B)\|\|_2 <= b to the quadratic-form bound is not stated in Lean. |
| L17 | Lemma 17 (dyadic exp and ln enclosures) | partial | K2 | `Dyadic.exp_nonneg_mem`, `Dyadic.exp_neg_mem`, `Dyadic.lnSeriesDn_le`, `Dyadic.lnSeriesUp_ge`, `Dyadic.ln2_mem`, `Dyadic.ln_point_mem` and 4 more | Not formalized: the bit-length decomposition of _ln_point (k = bitlength(v) - 1 - P and m_lo <= m 2^P <= m_hi <= 2 2^P), which enters ln_point_mem and u_grid_bounds as hypotheses; the binary64 facts of (v) (float(Fraction) correctly rounded to nearest, nextafter the adjacent binary64), which enter as hypotheses on F; the argument guards of exp_interval (the shortcut for arguments <= -1000 and the limit 2^16). P >= 6 is a hypothesis of ln2_mem (the code has P = 192). Document note: from_float is exact (width 0) only when f 2^P is an integer; for a binary64 below the grid resolution, for example 2^-1074 with P = 192, from_fraction returns (0, 1) (from_fraction_tiny, two_pow_neg_1074_grid); the enclosure property is unaffected. |

### Counting, assembly and eigenspaces (`CertifyMathlib.Counting`, `COUNTING.md`)

| Id | Statement | Lean | Level | Main theorems | Not formalized |
|---|---|---|---|---|---|
| SYL | Sylvester's law of inertia | complete | K2 | `sylvester_inertia`, `cntLt_eq_of_factor`, `sylvester_card` |  |
| MINMAX | eigenvalue count = largest dimension of a negative subspace | complete | K2 | `finrank_le_cntLt`, `exists_negOn_cntLt`, `finrank_le_cntLe`, `exists_nonposOn_cntLe`, `finrank_add_finrank_le` |  |
| WEYL | Weyl's inequality, counting and sorted form | complete | K2 | `weyl_cntLt`, `weyl_cntLe`, `weyl_opNorm`, `weyl_sorted`, `weyl_sorted_opNorm` |  |
| NORM | spectral-norm hypotheses imply quadratic-form hypotheses | complete | K2 | `abs_qf_le_of_toEuclideanLin`, `lemma4_opNorm`, `thm8_opNorm` |  |
| L4 | Lemma 4, inertia with backward error | complete | K2 | `lemma4`, `lemma4_doc`, `lemma4_opNorm`, `lemma4_hermitian`, `lemma4_perm` |  |
| L4-2x2 | count of a 2 x 2 pivot block | complete | K2 | `cntLt_fin_two` |  |
| COR5 | Corollary 5, completeness and sorted pairing | complete | K2 | `image_eq_of_match`, `injective_of_groups`, `cor5_le`, `cor5_lt`, `cor5_sorted`, `cor5_model` and 5 more |  |
| D-01 | a-posteriori backward error of the sparse factorization | partial | K2 | `abs_qf_le_rowSum`, `lemma4_rowSum`, `nsq_conjTranspose_mulVec`, `ldl_apply`, `lemma4_perm_rowSum`, `abs_le_rnd_div` and 20 more | part (c) and the sandwich with the computed beta for complex data (the rounded moduli fl(\|z\|) of the complex residual entries and the complex matrix-vector products of G1); the underflow hypothesis of the document is replaced by the standard model without underflow (eta = 0) |
| BLOCK | spectrum of a block-diagonal matrix is the union of the block spectra | complete | K2 | `cnt_blockDiagonal'`, `isHermitian_blockDiagonal'`, `card_filter_sigma` |  |
| THM8 | Theorem 8, sector assembly | complete | K2 | `thm8_blocks`, `thm8`, `thm8_opNorm`, `cnt_conj`, `isHermitian_conj`, `polarMat` and 11 more |  |
| SORTED | sorted-index forms of the j-th eigenvalue statements | complete | K2 | `eigAsc`, `ascEquiv`, `eigAsc_monotone`, `eigAsc_eq`, `cntLe_eq`, `cntLt_eq` and 8 more |  |
| THM6 | Theorem 6, projector bound | complete | K2 | `sinTheta_core`, `spec`, `specProj`, `eigCols`, `eigCols_conjTranspose_mul`, `eigCols_mul_conjTranspose` and 9 more |  |
| E-07-L1 | E-07 Lemma 1, Frobenius sin Theta | complete | K2 | `frob_sylvester`, `mul_one_sub_specProj_comm`, `coord_one_sub_specProj`, `e07_lemma1_sharp`, `e07_lemma1` |  |
| E-07-L2 | E-07 Lemma 2, polar identities | complete | K2 | `polar_identity_i`, `e07_lemma2_i`, `polar_identity_ii`, `e07_lemma2_ii` |  |
| E-07-P1 | E-07 Proposition 1, s_DK | complete | K2 | `e07_prop1`, `e07_prop1_proj` |  |
| E-07-L3 | E-07 Lemma 3, alignment | complete | K2 | `e07_lemma3` |  |
| C-12 | E-07 Proposition 2 (C-12), approximate-symmetry labels | partial | K2 | `ostrowski_pos`, `ostrowski_neg`, `ostrowski_polar`, `ostrowski_sorted`, `eigAsc_conj`, `abs_eigAsc_le` and 1 more | part (a) (an eigenvalue of S within (c + \|\|S\|\| w/2)/gamma_0 of each m_i, from the commutator bound) and the combined bound D |
| C-13 | E-07 Proposition 3 (C-13), overlap of two certified eigenspaces | complete | K2 | `norm_apply_le_sqrt`, `norm_trace_le_card`, `c13_prop3` |  |
| E-07-L5 | E-07 Lemma 5, rounding | not formalized | none |  | Lemma 5 (floating-point evaluation of the commutator, the labels and the fidelity); not attempted, the floating-point parts formalized so far are those of D-01 and Proposition 15, and the standard model of CertifyMathlib.Enclosure.FloatModel would serve for it |
| P14 | Proposition 14, degeneracy bracketing | complete | K2 | `bracket_count`, `bracket_exact`, `bracket_ge`, `card_Ioo_add_cntLe`, `cntLe_le_cntLt_add` |  |
| C-16 | C-16 exact bracket | not formalized | none |  | Sturm's theorem and Yun's squarefree factorization on Mathlib; hypothesis (H2) (exactness of python-flint) is outside Lean |
| P15 | Proposition 15, summed bounds | complete | K2 | `summed_bound`, `le_upFl`, `norm_eq_sqrt_re_im`, `modulusUpper_ge`, `modulusUpper_nonneg`, `dotFlC_error` and 5 more |  |
| P16 | Proposition 16, thermodynamic intervals | partial | K2 | `thermo_Z`, `thermo_energy`, `thermo_second`, `quotient_bounds`, `free_energy_bounds`, `sum_fiber_bounds` and 6 more | dyadic interval arithmetic of Lemma 17 (outward-rounded exp and ln; enclosure part) |
| C-06 | C-06 truncated thermodynamics (Proposition 16') | partial | K2 | `tail_weight`, `phiT`, `psiT`, `psiT_eq`, `phiT_le_max`, `phiT_at_peak` and 17 more | the outward-rounded evaluation of the interval ends by Lemma 17 (dyadic exp and ln) in (iii), and the statement that phi and psi tend to 0 at infinity (not needed by the Lean form of the case without M) |
| C-04 | C-04 correlation lift | partial | K2 | `c04_lemma0`, `mul_conjTranspose_eq_polar`, `c04_prop1`, `c04_conclusion` | Lemma 1 (compression rounding, a binary64 statement; not attempted, the standard model of CertifyMathlib.Enclosure.FloatModel would serve for it) |
| C-07 | C-07 moments and sum rules | partial | K2 | `norm_trace_herm_le`, `rank_conj_le`, `c07_lemma1`, `trace_mul_conjTranspose_eq`, `norm_pow_apply_le`, `c07_lemma2` and 2 more | Lemmas 3 and 4 (floating-point accounting of the repeated products, binary64 statements; not attempted, the standard model of CertifyMathlib.Enclosure.FloatModel would serve for them) |
| E-04 | E-04 parameter intervals | partial | K2 | `weyl_affine`, `weyl_affine_right`, `qf_triangle`, `sampling_criterion`, `qf_smul`, `weyl_affine_sorted` | Lemma 4 of the document (term-wise bound on \|\|H1\|\|_2, norms of Kronecker products) and the continuity argument for crossings |

### The AKLT application (`CertifyMathlib.Apps`, `APPS.md` section 9)

| Statement | Lean | Level | Main theorems | Not formalized, and notes |
|---|---|---|---|---|
| bordered-pencil lemma and Haynsworth inertia | complete | K2 | `bordered_pencil`, `bordered_inertia` |  |
| eta bound of `eta.py` | complete | K2 | `eta_count`, `qf_inv_shift`, `pencil_count_eq`, `eig_le_of_count`, `e10_eta_bound` | the identification of the eigenvalues of K with squared principal-angle cosines is taken as definition |
| finite-size criterion as the code uses it | complete | K2 | `e10_criterion`, `criterion_of_eta_sq`, `square_octagon_criterion`, `sqrt_le_isqrt_div` |  |
| AKLT kernel lemma | partial | K2 | `aklt_kernel_certificate`, `rank_zmod_le_rank_rat`, `psd_sum_mulVec_eq_zero` | formalized as a certificate over the rationals with the projector similarity as hypothesis; the general SU(2) lemma for arbitrary graphs is not formalized and is not needed for the six lattices |
| Pomata and Wei, criterion implies a gap | not formalized | none |  | cited from the paper |

### Non-Hermitian eigenpairs (`CertifyMathlib.NonHermitian`, `NONHERMITIAN.md`)

The statements are those of `docs/proofs/E-02_non_hermitian.md` on the code branch `dev/3.3`. The code marks the claim kind UNVERIFIED, and the derivation document has the status of rule D-12 described below.

| Id | Statement | Lean | Level | Main theorems | Not formalized |
|---|---|---|---|---|---|
| E02-L1 | Lemma 1, backward error of a right residual | partial | K2 | `backward_error` | equality of the norm of E (Lean has the upper bound), the singular value form, the left residual form and the two by two example |
| E02-T2 | Theorem 2, Bauer-Fike | complete | K2 | `bauerFike_core`, `bauerFike`, `bauerFike_matrix`, `bauerFike_matrix_sharp` |  |
| E02-T2r | Theorem 2, residual form | complete | K2 | `bauerFike_residual`, `bauerFike_residual_matrix` |  |
| E02-L3 | Lemma 3, the inverse from a computed inverse | partial | K2 | `neumann_bound`, `neumann_inverse` | invertibility of W and the formula for the inverse of V (the code uses only the bound) |
| E02-T4a | Theorem 4 (a), similarity | not formalized | none |  | not needed by the Lean proof of (b) |
| E02-T4b | Theorem 4 (b), discs | complete | K2 | `similarity_inclusion`, `disc_inclusion_matrix` |  |
| E02-N2N3 | Theorem 4 (b) with Lemma 3, statements N2 and N3 of the code | complete | K2 | `disc_inclusion_certified` |  |
| E02-T4c | Theorem 4 (c) to (e), counting | not formalized | none |  | continuity of polynomial roots in a real parameter with multiplicity |
| E02-P5 | Propositions 5 to 7, floating-point accounting | not formalized | none |  | floating-point model of ENCLOSURE.md not yet applied |

### Refinement of the kernel checks (`CertifyMathlib.Refine`, `LEAN_PRODUCT.md`)

These theorems connect two parts of the work. The kernel-reducible checks `pairOK` and `checkLowest` of `CertifyCore/Kernel/Defs.lean` decide, for decoded rational data, the inequalities that `cecheck` tests; the theorems below prove on Mathlib that acceptance implies the conclusion of `residual_enclosure` (`CertifyMathlib.Enclosure`), so that the S2 re-check of the three certificates checked in the kernel holds as a theorem about the decoded data. They concern the kernel definitions, not the compiled code of `cecheck` itself, which stays at level N.

| Statement | Lean | Level | Main theorems | Not formalized, and notes |
|---|---|---|---|---|
| per-pair radius check implies Theorem 1, also with model shift and radius | complete | K2 | `pairOK_theorem1`, `pairOK_theorem1_shift_rad` | cluster radii (Theorems 2 and 2') not refined |
| the lowest-window check implies Theorem 1 and the S2 conclusion | complete | K+ | `checkLowest_theorem1`, `checkLowest_S2`, `pairOK_theorem1_shift` |  |
| the three kernel certificates (matrix-free, dense arrays, npz) | complete | K+ | `matrixFree_theorem1`, `denseArrays_theorem1`, `npz_theorem1`, `matrixFree_S2`, `denseArrays_S2`, `npz_S2` | nanoda stopped at the memory limit for `npz_S2`, `npz_theorem1` |
| the added checks of symmetry and nonzero vectors | complete | K+ | `symOK_rejects_asymmetric`, `vecOK_rejects_zero_vector` |  |
| pivots of an elimination give the Sylvester count | partial | K2 | `count_of_gaussQ` | N-05: per-certificate count theorems (Drafts/LeanProduct/SmallCount.lean) and the refinement of CertifyCore.K.bareiss (Sylvester's determinant identity) |

## The certificates

`cecheck` (compiled from `CertifyCore`, level N) reads a certificate, its array file and the exported operator, decodes every binary64 value exactly, recomputes the hashes, and re-checks the S2 radii and the S3 counts in exact rational arithmetic. On the validation set of the C verifier (63 bundles, of which 8 were refused by certify-ed and have no certificate) it passed 55 of 55 certificates, as the C verifier did. It confirmed 31 CERTIFIED S2 statements and 24 CERTIFIED S3 statements exactly, recomputed 70 recorded inertia counts (all equal) and 211 exact counts in total, and confirmed 42 S8 brackets. It reproduced all 18 FAILED statuses, and for 15 of them an exact inertia count contradicts the claim outright. It rejected all 80 forged certificates (four kinds, twenty each); certify-ed's own integrity check rejects 40 of them, because a lowered radius or a promoted status with resealed hashes is consistent as a record and is caught only by recomputation. The row-by-row comparison with the C verifier agrees on 135 of 135 rows (`results/core/agreement.csv`).

What `cecheck` does not check is listed in `CORE.md`: S1 model fidelity (model shift taken from the record); S4 eigenspace bounds; S5 to S7; Theorems 1, 2, 2', Sylvester's law and Weyl's inequality (Mathlib part); complex operators and sector certificates (none in the set); lean4lean (the toolchain's build reads no 4.35.0 .olean file).

## The application numbers

`ceapps` (compiled from `CertifyApps`, level N) rebuilds each model from its specification in exact integer arithmetic, checks it entrywise against the Python builder, and recomputes the certified number from the recorded vectors and shifts with exact rational arithmetic and exact LDL inertia counts. A target agrees when the Lean value is at least as strong as the recorded certified value and every recorded count is reproduced. The final re-run used one binary (SHA-256 `635dfbc225f351ee...`, `results/apps/final_rerun.json`).

| Target | Quantity | Lean (`ceapps`) | 3.2.1 record | Result |
|---|---|---|---|---|
| e7_chain_J2_0 | lowest eigenvalue of the block enclosed around the recorded E0 with the recorded exact radius; eigenvalues below the recorded count point | Rayleigh quotient -8.9043865298764423e0, radius 2.1599193648251390e-23, count 1 | E0 -8.9043865298764171e0, exact radius 3.7193004277411412e-14, count 1 | agrees |
| e7_chain_J2_0.5 | lowest eigenvalue of the block enclosed around the recorded E0 with the recorded exact radius; eigenvalues below the recorded count point | Rayleigh quotient -7.5000000000000000e0, radius 6.2805712975047203e-23, count 1 | E0 -7.4999999999999662e0, exact radius 3.3926633543218007e-14, count 1 | agrees |
| e7_square_J2_0 | lowest eigenvalue of the block enclosed around the recorded E0 with the recorded exact radius; eigenvalues below the recorded count point | Rayleigh quotient -1.1228483208428860e1, radius 6.6295588784963786e-31, count 1 | E0 -1.1228483208428859e1, exact radius 1.7685219221982841e-14, count 1 | agrees |
| e7_square_J2_0.5 | lowest eigenvalue of the block enclosed around the recorded E0 with the recorded exact radius; eigenvalues below the recorded count point | Rayleigh quotient -8.4579233513948467e0, radius 3.8644449557691711e-23, count 1 | E0 -8.4579233513948395e0, exact radius 1.0998626604929780e-14, count 1 | agrees |
| e9_crown_spin_half_lower_bound | certified lower bound on lambda_min of the reduced spin-1/2 crown | lower bound -5.5015829796302939e-1 | lower bound -5.5015829882120382e-1 | agrees |
| e9_crown_spin_one_upper_bound | certified upper bound -12 c1 + c17 (E0 + r) on an eigenvalue of the spin-1 crown at the printed couplings | eigenvalue at most -1.6380246425131775e0 | eigenvalue at most -1.6380246425131773e0 | agrees |
| e10_star | certified upper bound t' on eta^2 at the recorded threshold, and the criterion | eta at most 1.1104302202114303e-1 | eta at most 1.1104302202338649e-1 | agrees |
| e10_square_octagon_perp | certified upper bound t' on eta^2 at the recorded threshold, and the criterion | eta at most 1.5896633112606854e-1 | eta at most 1.5896633115208958e-1 | agrees |
| e10_square_octagon_par | certified upper bound t' on eta^2 at the recorded threshold, and the criterion | eta at most 1.0614468582666172e-1 | eta at most 1.0614468587603454e-1 | agrees |
| e10_cross | certified upper bound t' on eta^2 at the recorded threshold, and the criterion | eta at most 1.9973845006935519e-1 | eta at most 1.9973845072185567e-1 | agrees |
| e10_decorated_honeycomb | certified upper bound t' on eta^2 at the recorded threshold, and the criterion | eta at most 1.5303290854841137e-1 | eta at most 1.5303290857599913e-1 | agrees |
| e10_decorated_square | certified upper bound t' on eta^2 at the recorded threshold, and the criterion | eta at most 2.2035431746216704e-1 | eta at most 2.2035431930729685e-1 | agrees |
| e10_kernels | ker H_X = range U_X for Gamma_i = L u C and Gamma_j = C u R of the six lattices | 12 subgraphs, kernel equals the VBS range in all: yes; largest dimension 16777216 | 4 subgraphs checked in the record | agrees |
| e10_square_octagon_criterion | 4 eta_perp + 2 eta_par < 1 | 4 eta_perp + 2 eta_par at most 8.4815469615759761e-1 | at most 8.4815469636042740e-1 | agrees |

The models were also compared with the Python builder entry by entry, including the two 20-site chains of E7 (`results/apps/compare_aklt.json`, `results/apps/compare_chain20.json`, `results/apps/compare_models.json`, `results/apps/selftest.txt`; all equal).

Two application items of the development roadmap were taken up with the code of `dev/3.3`; neither has a Lean counterpart yet (`APPS_NEXT.md`, `results/apps_next/`). For N-11, the nine couplings that Kránitz and Penc print for their headline bound fix the crown Hamiltonian once the digit in parentheses is appended. At those couplings the floating-point layer of certify-ed gives -0.5498329145683885 <= lambda_min <= -0.5498329144473251 for the S^z = 0 crown (dimension 48620, 8 blocks, the largest of dimension 6194, 3018 s). This layer rests on the binary64 rounding model; the exact re-check without that model counted 3 of 8 blocks (count 0, 0, 0) before it was stopped, so the exact layer is not complete. The printed value -0.549832(8), read as -0.5498328, lies above the whole interval, 1.15e-07 above its lower end, and the floating-point value -0.54983289 of the earlier review lies outside it. What this means for the paper's statement depends on how its digit in parentheses is meant, which is for Sarang Vehale to judge. For N-12, the honeycomb criterion of Pomata and Wei (Method III, 30 sites) needs a bordered block of 1293292 rows, 12462 GiB for one binary64 copy, so it was not run. The exact product of the printed gap constants is 0.0026603333944698595; the value 0.002660333395 of the final equation agrees with it, and the value 0.002550333395 of the conclusion differs by -0.00011.

## Second checkers and axioms

| Library | leanchecker | nanoda | Axioms |
|---|---|---|---|
| CertifyCore | `leanchecker CertifyCore`, exit code 0, 473.3 s | 76 of 76 theorems accepted | Classical.choice, Quot.sound, propext |
| CertifyMathlib.Enclosure | 4 runs, exit codes 0, 0, 0, 0, 2541.8 + 233.5 + 1956.2 + 1324.4 s | 63 of 63 main theorems accepted; one theorem (`FP.residual_enclosure_int05`) needed a memory limit of 6 GiB for its export and was run after the final build | Classical.choice, Quot.sound, propext |
| CertifyMathlib.Counting | 4 runs, exit codes 0, 0, 0, 0, 2091.1 + 875.6 + 2101.7 + 1316.6 s | 71 of 71 main theorems accepted | Classical.choice, Quot.sound, propext |
| CertifyMathlib.Apps | exit code 0, 1087.6 s | 15 of 15 main theorems accepted | Classical.choice, Quot.sound, propext |
| CertifyMathlib.NonHermitian | exit code 0, 2989 s | 6 of 6 main theorems accepted | Classical.choice, Quot.sound, propext |
| CertifyMathlib.Refine | exit code 0, 1844 s | 3 of 5 theorems accepted, 2 stopped at 4096 MiB | Classical.choice, Quot.sound, propext |
| all three libraries | exit code 0, 5961.9 s, 78 modules (11 November 2026) | 14 of 14 rechecked (the main theorems of CertifyMathlib.Counting.D01Float and P15Float, which the build of 11 November 2026 rebuilt because enclosure modules they import had changed) | see the axiom audit |

The axiom audit (`Audit/AxiomAudit.lean`, output `results/integration_logs/axiom_audit.log`) asks Lean for the axioms of every theorem of the three libraries, auxiliary theorems included. A `sorry` would appear there as `sorryAx` and a `native_decide` as `Lean.ofReduceBool`; neither appears. The `decide +kernel` proofs of `CertifyCore` (the three certificates checked in the kernel, and some guards) are evaluated by the kernel itself and add no axiom. Each nanoda run checks the exported theorem together with every declaration it depends on, about 3000 to 36000 declarations per run.

## What the formalization found in the documents

Formalizing a statement forces every hypothesis and every range to be written down. Eleven places of the documents were found where the text and the Lean statement differ. None of them changes a stored certificate, a status or a published number; seven are corrected in the code repository by the documentation commits `40d59f4` and `fa86456`, one (L-3) is closed by the range check of 3.2.2, and three confirm a remark of the documents or record a difference of form. `docs/project/failures/07_found_while_formalizing.md` describes each in prose.

| Id | Place | The document says | What holds | Found by | Effect | Correction |
|---|---|---|---|---|---|---|
| L-1 | code/certify-ed/docs/proofs/thm1_fp-accounting.md, lines 54 to 56 (guard Lambda6) | the guard holds whenever (w + 2)u <= 1/3 | the guard fails at w + 2 = 3002399751580330 = floor(2^53/3), which that condition admits (margin -7.0 u^2); a correct condition is (w + 2)u <= (1 - 5u)/3; for the stated range w <= 2^26 the guard holds | CertifyCore.Theorems.Thm1 (`lambda6`, `lambda6_fails_at_boundary`), level K2; results/core/findings.json | none: no stored operator has a row width near 2^26 | erratum in the code repository, commit `40d59f4` |
| L-2 | code/certify-ed/docs/proofs/thm1_fp-accounting.md, lines 44 to 50 (guard Lambda4) | the upper norm guard holds for n <= 2^26 | true as stated (margin 0.75u at n = 2^26 with N = 2n); the guard fails from n = 84886744 (exact bisection, not a Lean theorem) | CertifyCore export/findings_values.py; results/core/findings.json | none: the largest stored dimension is 1116229 | noted in the erratum of commit `40d59f4` |
| L-3 | code/certify-ed/certify_ed/enclosure.py (`_rounding_bound`) and certify_ed/fp.py lines 149 to 167 | the code applies Theorem 1 and the norm guards to any row width w and dimension n | the proofs need w <= 2^26 and n <= 2^26 (findings L-1 and L-2), and the code does not check either bound | reading the code against the Lean statements of CertifyCore.Theorems.Thm1 | none for the stored records (largest dimension 1116229) | closed in 3.2.2: commit 1f47d4c (branch fix/3.2.2, tag v3.2.2) refuses to form a bound outside n, w <= 2^26; S2 and the statements that rest on it become INCONCLUSIVE with the reason code FP_GUARD_RANGE |
| L-4 | code/certify-ed/docs/proofs/L17_dyadic_exp_ln.md, lines 112 to 125 (remainder constant) | the 3.1 code used the constant 9/8, valid only for u <= 1/3; the 3.2 code uses 8/7 | confirmed: 1/(1 - x^2) = 1.1396 at x = 0.35 exceeds 9/8 and 8/7 suffices; at the largest argument the tail is at most 0.00118 grid units while both versions add one grid unit, so no 3.1 enclosure was wrong | CertifyCore.Theorems.L17, level K2; CertifyMathlib.Enclosure.Dyadic (`lnSeriesUp_ge`) | none | none needed; the document already states it |
| L-5 | code/certify-ed/docs/theory.md, section 9.1, lines 362 to 375 (Proposition 14) | the count of eigenvalues in the inner interval is at most nu_hi - nu_lo | only when sigma_lo + beta_lo < sigma_hi - beta_hi; a 1 x 1 example with an empty inner interval violates the display; the code uses the inequality only for J inside the inner interval, which implies the hypothesis | CertifyMathlib.Counting (`bracket_count`), level K2; COUNTING.md | none | hypothesis added in the code repository, commit `40d59f4` |
| L-6 | code/certify-ed/docs/proofs/D-02_matrix-free.md, lines 47 to 49 and 106 to 107 (part b) | part (b) follows because the real case keeps a slack of one gamma index | part (b) holds, but by a second-order inequality with a margin of about W u^2 (Lean `FP.d02b_key`), not by the slack of one index | CertifyMathlib.Enclosure.FP (`d02b_key`, `d02b_radius`), level K2; ENCLOSURE.md | none | erratum in the code repository, commit `fa86456` |
| L-7 | code/certify-ed/docs/proofs/int05_underflow.md, lines 160 to 166 (Theorem U5) | (1 - u)^k (1 + 8u)^2 >= 1 + 10u for k <= 12 | false: the product is about 1 + 4u; the proof needs only A_{n,w} >= sqrt(n) c_w (1 + u) u_min, which holds (Lean `addTerm_ge`) | CertifyMathlib.Enclosure.FP (`residual_enclosure_int05`), level K2; ENCLOSURE.md | none | erratum in the code repository, commit `fa86456` |
| L-8 | code/certify-ed/docs/proofs/L17_dyadic_exp_ln.md, line 55 (conversion (v)) | `from_float(f)` is exact (width 0) | exact only when f 2^P is an integer; for 2^-1074 with P = 192 the result is the interval (0, 1) in grid units, which still contains f | CertifyMathlib.Enclosure.Dyadic (`from_fraction_tiny`, `two_pow_neg_1074_grid`), level K2 | none: only the enclosure property is used | erratum in the code repository, commit `fa86456` |
| L-9 | code/certify-ed/docs/proofs/appA_model-shift.md, lines 25 and 67 to 69 (Lemma A1, second sentence) | the computed row and column sums of \|H\|_abs are within a factor (1 +- gamma_{m+2L_max-1}) of the exact sums | for complex amplitudes the factor is (1 +- gamma_{m+2L_max}), because `np.abs` of a computed amplitude is rounded; the code divides by (1 - gamma(width)) with width = m + 2L_max + 2, which covers it | CertifyMathlib.Enclosure.FP (`abs_sum_bounds_complex`), level K2 | none | erratum in the code repository, commit `fa86456` |
| L-10 | code/certify-ed/docs/proofs/D-01_sparse-residual.md, part (a), complex data | the constant g_{t+1} = 2 gamma_{t+5} | the Lean model gives gamma_{t+3}, smaller than the document's constant; the document's constant is the more conservative one | CertifyMathlib.Counting (`d01_a_c`), level K2 | none | none needed |
| L-11 | manuscript_v4/sections/s3_math.tex (`thm:dk`, Theorem 6) | the separation is measured from the Ritz values | the usual sin Theta theorem measures it from an interval containing them; under the hypotheses of `thm:dk` the two agree, and the Lean form `thm6_ritz` takes the location of the Ritz values as a hypothesis instead of the two-line argument of the manuscript | CertifyMathlib.Counting (`thm6_ritz`), level K2 | none | none needed |

The refinement of the kernel checks (`LEAN_PRODUCT.md`) found two places where the kernel-reducible checks of `CertifyCore/Kernel/Defs.lean` accept more than `cecheck` does: checkLowest does not test symmetry of the decoded matrix (cecheck does, H5.symmetric); the refinement theorems take symOK as a hypothesis; pairOK and the single-group check accept a zero vector (cecheck rejects it in E2); the refinement theorems take vecOK as a hypothesis. No stored certificate is affected, because `cecheck` makes both tests.

## What is not formalized, and why

Four kinds of gap remain. First, the link between the Lean model of floating point and the machine: that binary64 arithmetic, NumPy and SciPy round as the standard model (and, for INT-05, the underflow model) assumes, that `float(Fraction)` rounds correctly and that a sum or a product in the code has the summation tree the theorem assumes. These are hypotheses of the Lean theorems, named in each correspondence text, and they are supported by the runtime probe of certify-ed, not by Lean. Second, the programs: `cecheck` and `ceapps` are compiled Lean programs (level N), and no theorem states that a program computes what a Mathlib theorem assumes; their site-by-site elimination for the AKLT kernels, for example, is argued in a docstring. Third, statements that need mathematics that Mathlib v4.35.0 does not have: C-16 needs Sturm's theorem and squarefree factorization, C-04 and C-07 need the trace norm and Hölder's inequality for traces (the Lean proofs use a rank bound instead, with the documents' constants), and the Pomata and Wei gap theorem was cited, not formalized. Fourth, parts that were not attempted for time: Theorem U5 and D-01 part (c) for complex data, the floating-point lemmas of E-07 Lemma 5, C-04 Lemma 1 and C-07 Lemmas 3 and 4, C-12 part (a), E-03 Propositions 1 to 3, E-05 Lemma 3 (the computed bound) and Lemma 4, E-04 Lemma 4 and the continuity argument, the code-level evaluation of the model shift in A4, and the bit-length decomposition of Lemma 17. Statement S1 (that the operator is the declared physical model) is outside the reach of any of these checks beyond the entrywise comparison of the rebuilt models. Lemmas 11 and 12 and Proposition 13 of the exact layer (`manuscript_v4/sections/appB_exact.tex`) have no Lean theorem; `cecheck` recomputes the counts they justify.

Of the parts added on 11 November 2026, the following are not formalized: from E-02, Lemma 1, backward error of a right residual (equality of the norm of E (Lean has the upper bound), the singular value form, the left residual form and the two by two example); Lemma 3, the inverse from a computed inverse (invertibility of W and the formula for the inverse of V (the code uses only the bound)); Theorem 4 (a), similarity (not needed by the Lean proof of (b)); Theorem 4 (c) to (e), counting (continuity of polynomial roots in a real parameter with multiplicity); Propositions 5 to 7, floating-point accounting (floating-point model of ENCLOSURE.md not yet applied); from the refinement of the kernel checks, N-05: per-certificate count theorems (Drafts/LeanProduct/SmallCount.lean) and the refinement of CertifyCore.K.bareiss (Sylvester's determinant identity); N-06: Theorem U5 complex, D-01 (c) complex, C-12 (a), E-07 Lemma 5: not attempted. The applications N-11 and N-12 have no Lean counterpart.

## Relation to rule D-12

Rule D-12 says that new mathematics counts as established only after Sarang Vehale has re-derived it. A level K2 theorem removes the proof as a source of error but not the statement, and the findings above show that statements and documents can differ. The derivation documents therefore keep the status "derived, machine-checked, independently re-derived (not a human check), Sarang's check pending", with the Lean level reported next to it. A proposal for how a statement review could take the place of a re-derivation for the statements at level K2 is in `docs/project/decisions/05_proposed_d62_lean.md`; it is not adopted.

## How to rebuild and re-run

From `lean/`, with the toolchain directory and `C:\Program Files\Git\cmd` on `PATH` and `MATHLIB_CACHE_DIR` set as in `docs/lean/README.md`:

```
lake exe cache get
lake build CertifyCore CertifyMathlib CertifyApps cecheck ceapps
lake env lean Audit/AxiomAudit.lean
lake env leanchecker CertifyApps CertifyApps.AkltHam CertifyApps.Apps ...   (all 78 modules, see Audit/run_final_checks.py)
python apps_export/run_all.py --exe .lake/build/bin/ceapps.exe --archive ../../archive/archive_v5 --out results/apps --work work/apps
python ../tools/make_verification_report.py
```

The certificate checks with `cecheck` and the C verifier are the commands of `CORE.md` section 1; they need the certify-ed source of `archive_v5/code/certify-ed` on `PYTHONPATH` to regenerate the validation certificates. A single certificate is re-checked with `python Verify/cecheck_bundle.py CERT.json --bundle DIR` (or `--operator OPERATOR.npz`); from the code branch `dev/3.3` the same check runs as `certify-ed verify CERT.json --bundle DIR --external "python Verify/cecheck_bundle.py {certificate} --bundle {bundle} --json {report}"` (with the full path of the driver), which writes a separate record and changes no status (`LEAN_PRODUCT.md`).

The nanoda drivers are `export/second_kernel.py` (CertifyCore), `Drafts/Enclosure/nanoda_run.py`, `Drafts/Counting/nanoda_counting.py` and `apps_export/nanoda_apps.py`; each runs one theorem per process and stops a run above 4 GiB of private memory (6 GiB for the one re-run made by `Audit/run_final_checks.py`).

## Files

| File | Contents |
|---|---|
| `results/core/summary.json`, `agreement.csv`, `findings.json`, `second_kernel.json` | cecheck results, agreement with the C verifier, findings, second-kernel runs of CertifyCore |
| `results/mathlib_enclosure.json` | statements, theorems, axioms and checker runs of the enclosure library |
| `results/mathlib_counting.json` | the same for the counting library |
| `results/apps/summary.json`, `mathlib_apps.json`, `final_rerun.json` | application targets, the Mathlib part of the application, the final re-run |
| `results/findings_documents.json` | the findings table above |
| `results/integration.json` | final build, axiom audit, text search, final leanchecker run, nanoda recheck |
| `results/mathlib_nonhermitian.json` | statements, theorems, axioms and checker runs of the non-Hermitian library |
| `results/lean_product.json` | the certificate driver and its runs, the refinement theorems, their axioms and checker runs, what is not done, the two findings on the kernel checks |
| `results/apps_next/` | the records of the applications N-11 and N-12 and the values read from the papers |
| `CORE.md`, `ENCLOSURE.md`, `COUNTING.md`, `APPS.md`, `NONHERMITIAN.md`, `LEAN_PRODUCT.md`, `APPS_NEXT.md` | the full account of each part |
