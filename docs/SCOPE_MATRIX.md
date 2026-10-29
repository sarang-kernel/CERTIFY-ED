# Scope matrix: what a certificate says about the quantities labs report

Rows are the quantities an ED paper or lab notebook reports. Columns say what a CERTIFY-ED 3.1
certificate states about each today (statement id), what the user can narrow, what is out of scope and
why, and the mathematics that would be needed to bring it in. "Out of scope" is a design slot with a named
reason code, not a promise: the slot says what is missing, and nothing here claims the missing piece is
easy. Statements marked UNVERIFIED (Prop 14-16) await a second derivation (D-12).

Reason codes are those of `certify_ed.scope.REASON_CODES`; every certificate lists the standing
out-of-scope rows of this table in `scope.out_of_scope` with these codes.

| quantity labs report | certificate states today | narrowable (how) | out of scope, and why | mathematics that would be needed |
|---|---|---|---|---|
| **ground-state energy** | S2: \|E₀ − μ\| ≤ r for a distinct eigenvalue (Thm 1-3); S3: it is the lowest (Lemma 4, Cor 5); S6: both without floating point (Prop 13); S1: of the declared model | `--only S1,S2` drops the "lowest" claim; `--skip S1` drops the model statement; budgets | nothing | — |
| **gaps** | S3: certified lower bound on λ_{k+1} − λ_k (`gap_lower_bound`); Kato-Temple sharpening (Thm 3); S6 exact gap for 'lowest' claims; since 3.2 the body record `gaps` (C-01): intervals for every difference of consecutive levels read off the S2 enclosures ordered by S3 (Cor 5, pigeonhole step of theory §6), the gap above the highest claimed level with S3's lower bound and no upper bound (`S3.evidence.gap_interval`); claim k + 1 pairs to bound λ_{k+1} − λ_k on both sides | S3 can be skipped (then no gap and no `gaps` record) | gap *upper* bounds from an unconverged excited vector without S3: Thm 1 on that vector gives an eigenvalue ≤ μ + r, which is an upper bound on λ_{k+1} only when S3 certifies the count below | none new: the record is a presentation of S2 + S3 (`certify_ed.gaps`) |
| **excited states** | S2 per pair; S3 inside the claimed window (`kind='window'`); S4 for clusters | window claims; `--skip S3` | interior windows without eigenvectors: S3 needs certified intervals (NEEDS_EIGENVECTORS) | counts alone (two inertia counts) say how many eigenvalues lie in (a, b) without vectors; a vectorless window-count statement is a straightforward addition (Lemma 4 only) |
| **degeneracies** | S8 (Prop 14, UNVERIFIED): exactly g eigenvalues in a bracket around the level; S2/S4 distinctness within the cluster; S6 level enclosures | `degeneracy=False`, `--skip S8` | exact equality of the g eigenvalues (two eigenvalues 1e−14 apart are "degenerate" only up to the bracket width): not a floating-point statement | exact arithmetic on a symmetry-reduced block (S6 path), or a group-theoretic argument supplied as an assumption |
| **quantum numbers / labels** | S4: multiset of S-eigenvalues on the cluster (Prop 7), total spin through the Casimir, momentum by `sectors.audit_label`; S1 sector audit for symmetry-adapted bases | `eigenspaces='none'`; `symmetries={}` | labels of operators that do not commute exactly with the declared model (UNSUPPORTED_MODEL); labels inside a cluster whose separation is not certified (INCONCLUSIVE, not a scope matter) | approximate symmetries need a perturbative label bound (‖[H, S]‖ enters Davis-Kahan) |
| **correlation functions** ⟨O_i O_j⟩ | S4 extension (Prop 15, UNVERIFIED): tr(P O_i O_j) with Thm 6 bound per pair, per-state value tr(P·)/g; `Options.correlations` | `correlations={}` | operators that leave the producer's U(1) block (the statement would refer to a truncated operator); symmetry-adapted (isometry) claims (UNSUPPORTED_CLAIM: Thm 6 is applied in the sector block and the lift is not implemented); non-Hermitian products such as S⁺_i S⁻_j (use the Hermitian combinations) | lift of tr(P O) through the sector isometry (Thm 8 machinery) |
| **static structure factors** S(q) | S4 extension (Prop 15): (1/N) Σ e^{iq·(r_i−r_j)} C_ij with summed bounds; `Options.correlations[...]["q"]` | as above | as above; dynamical S(q, ω) is in the dynamics row | — |
| **order parameters** | S4 observables: tr(P O) for any Hermitian O the user supplies (`Options.observables`), batches included; squared order parameters are correlation sums (row above) | `observables={}` | order parameters defined by a limit (N → ∞) or by symmetry breaking in the thermodynamic limit (OUT_OF_SCOPE_EXTRAPOLATION) | — |
| **entanglement entropy and spectrum** | nothing (NOT_IMPLEMENTED); the eigenvector error is certified (S4 projector error, S2 residual) but is not propagated to ρ_A | — | the map from a vector error to an entropy error is not implemented | for g = 1: ‖ψ − ψ̂‖ ≤ ε gives ‖ρ_A − ρ̂_A‖₁ ≤ 2ε, then the Fannes-Audenaert inequality bounds \|S(ρ_A) − S(ρ̂_A)\| ≤ ε' log(d_A − 1) + h(ε'); for clusters, the entropy is basis dependent (only projector-level quantities are well defined) |
| **thermodynamics** Z, F, U, C | S7 (Prop 16, UNVERIFIED) from a complete whole-spectrum enclosure, dyadic interval arithmetic (Lemma 17); `--preset whole-spectrum --temperatures` | temperatures list; `--skip S7` | from a partial spectrum (NEEDS_WHOLE_SPECTRUM): low-T truncation needs a certified bound on the omitted weight; stochastic / FTLM estimates carry sampling error that no deterministic statement covers | truncated spectrum: with S3's λ_{k+1} lower bound and the dimension, the tail weight is ≤ (n − k) e^{−βλ_{k+1}}, giving two-sided bounds at low T — a direct extension of Prop 16 |
| **dynamical response, spectral functions** | nothing (OUT_OF_SCOPE_DYNAMICS) | — | continued-fraction / Krylov spectral functions are statements about a resolvent, not about one eigenpair set; broadening is a modelling choice | certified Lanczos error bounds (Kaniel-Paige-Saad with a posteriori residuals), sum-rule checks with certified moments tr(P O H^k O), interval evaluation of the continued fraction |
| **time evolution** | nothing (OUT_OF_SCOPE_DYNAMICS) | — | ψ(t) = e^{−iHt}ψ₀ is not an eigenproblem result; a certified full spectrum (S2+S3, kind 'all') makes e^{−iHt} computable with Prop 16-style interval arithmetic, but that path is not implemented | a posteriori bounds for Krylov/Chebyshev propagators; or interval exponentiation on the certified spectral decomposition for small dimensions |
| **level statistics** (r-ratios, spacing distributions) | nothing (NOT_IMPLEMENTED); the ingredient — a complete, counted spectrum per symmetry sector — is S2+S3 with `kind='all'` and the sector audit of S1 | — | the unfolding step has no certified form; sector purity is a precondition the certificate can supply but the statistic cannot check | interval arithmetic on r-ratios from the enclosed spectrum (gap ratios of intervals) is mechanical; the unfolding is a modelling choice to be declared, not certified |
| **finite-size extrapolation** | nothing (OUT_OF_SCOPE_EXTRAPOLATION); certified inputs at each size are what the tool provides | — | an extrapolation is a statement about a model of the sequence of sizes, not about any finite matrix | none within this tool; an extrapolation report can cite per-size certificates by hash |
| **phase boundaries, critical points** | nothing (OUT_OF_SCOPE_EXTRAPOLATION); since 3.2 `certify_ed.crossings.certified_crossing(A, B, level)` / `certify-ed crossing` (C-11) orders the rank-`level` eigenvalue intervals of two certificates (two couplings, or two sectors) when they do not overlap, with both hashes and the margin; without S3 CERTIFIED the record is about the certified interval of the claimed value and says so | — | derived from extrapolated quantities or crossings of fitted curves | none new for the crossing (interval comparison of certified data); a phase boundary remains an extrapolation |
| **matrix fidelity / "is the Hamiltonian right"** | S1: producer matrix = declared model (entrywise, sector audit, or probes) | `--skip S1` | correctness of the declared model itself (never verified: the spec is the declaration); producer codes with no exportable matrix and no spec (black box: S2 bounds concern the black box, see `operator_note`) | — |
| **producer formats** | any of npy/npz/HDF5/CSV/TSV/text/JSON/Matrix Market, matrix-free Python (`docs/producer_bundle.md`) | — | compressed states (MPS) and non-Hermitian problems (UNSUPPORTED_MODEL): the statements assume a Hermitian matrix and explicit vectors | non-Hermitian: left/right residuals, pseudospectra; MPS: residual via MPO contraction, completeness undefined |

## Reason codes used in `scope.out_of_scope` and `evidence.reason_code`

| code | meaning |
|---|---|
| `NARROWED_BY_USER` | the user excluded the statement (`--only` / `--skip` / preset) |
| `DEFAULT_SELECTION` | not part of the default selection; switched on by its option, a preset or `--only` |
| `DEPENDENCY_NARROWED` | the statement needs another statement that the user excluded |
| `BUDGET_TIME`, `BUDGET_MEMORY` | the time budget was spent, or the memory estimate exceeds the budget |
| `NEEDS_EIGENVECTORS`, `NEEDS_SPEC`, `NEEDS_MATRIX`, `NEEDS_WHOLE_SPECTRUM`, `NEEDS_TEMPERATURES`, `NEEDS_OBSERVABLES`, `NEEDS_DEPENDENCY` | an input the statement needs is missing (`NEEDS_DEPENDENCY`: a corroborating solver or a file library was not named) |
| `MISSING_DEPENDENCY` | an optional package is not installed: python-flint for S6 and the other exact statements (`pip install 'certify-ed[exact]'`), QuSpin for `claim_from_quspin` (`certify-ed[quspin]`); the statement is NOT_CHECKED, the evidence names the package and the extra, and `evidence.next_action` says how to install it (3.2) |
| `NO_COMPLETENESS_CLAIM` | a `subset` claim: nothing is said about other eigenvalues |
| `NO_CLUSTERS` | no (quasi-)degenerate clusters among the claimed pairs |
| `SIZE_LIMIT` | dimension above the configured limit (`inertia_max_dim`, `exact_max_dim`, ...) |
| `UNSUPPORTED_CLAIM`, `UNSUPPORTED_MODEL` | the statement is not implemented for this claim kind / sector form / model class |
| `NOT_IMPLEMENTED` | a design slot: the mathematics is identified in this table but not implemented |
| `OUT_OF_SCOPE_DYNAMICS` | dynamical response and time evolution are not statements about eigenpairs of one matrix |
| `OUT_OF_SCOPE_EXTRAPOLATION` | extrapolations and phase boundaries are not statements about any finite matrix |
| `NOT_APPLICABLE` | the statement does not apply to this input |
| `UNDERFLOW_RISK` | (since 3.2.0.dev0) a nonzero operand, radius or residual bound has magnitude below 2^-900: the no-underflow hypothesis of the floating-point model is not established for this input, so S2 and the statements that rest on it are INCONCLUSIVE; `Options(underflow_check=False)` assumes the hypothesis instead and records that |

## How to read a certificate against this table

1. `scope.preset` and `scope.selection` say what was asked for.
2. `scope.in_scope` lists the statements that ran, with their status; `scope.narrowed` lists what the user
   removed and why (the user's own words under `reason`); `scope.out_of_scope` lists statements that could
   not run (missing inputs, budgets, limits) and the standing rows above, each with a code.
3. Every statement id appears exactly once across the three lists. A quantity not in this table is not
   addressed by any statement; say so in the paper rather than inferring it from a certificate.
4. Every statement that is INCONCLUSIVE or NOT_CHECKED carries `evidence.next_action` (3.2): one sentence
   saying what would let the statement be decided (more eigenpairs, eigenvectors, a larger budget or limit, a
   window claim, an optional package). It is advice derived from the reason code and the summary; it never
   changes a status.
