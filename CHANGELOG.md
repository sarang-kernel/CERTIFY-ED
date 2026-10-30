# Changelog

## [3.2.0.dev0] (unreleased; branch `dev/3.2`)

`__version__` = 3.2.0.dev0. The computational source changes (certify.py, fp.py, scope.py), so records made
with it are not comparable by `source_hash` with the 3.1.0 evidence run; no statement that 3.1.0 certified
becomes *more* certified here (soundness rule), and the only status changes are downgrades to INCONCLUSIVE
under the new reason code below.

Integration (2026-10-29): seven tracks of roadmap Waves 0 and 1 merged in the order known-issues, core-api,
format-packaging, producers, statements, whole-spectrum (its commits re-written to LF line endings, content
unchanged), viewer-corpus. Conflicts were additive except two: `certify.py`'s observable loop now applies the
Hermitian decomposition (C-05) to the parsed operators of C-15, and an inverted window is refused when the
Claim is built (C-14) rather than reported by S3 (C-02 test adjusted). `Certificate.load` validates the schema
before the hashes. Full suite on the merged branch: see `RESULT_integration_3.2.md`. Sixteen CLI commands:
env, certify, import, plan, statement, crossing, levelstats, extrapolate, view, presets, conform, spectrum,
list, test, reproduce, numbers.

### New statements track (roadmap C-03, C-04, E-04, C-12, C-13, E-07; track/new-statements)

Every statement of this track has a derivation in `docs/proofs/<id>_<name>.md` with its verification script
`docs/proofs/verify_<id>.py` (decision D-59: derived here, machine-checked in exact rational arithmetic where the
quantity is rational and at 40-50 mpmath digits where it is not; independent re-derivation pending, D-12). No
existing bound became tighter; new bounds are combined with old ones as min(old, new) only where both are proven.

- **C-03 entanglement entropy and spectrum.** `Options.entanglement = {name: {"sites": [...], "log_base": "e" | 2,
  "max_dim_a": 64}}` adds, to every S4 cluster record, `entanglement[name]`: a certified interval for S(rho_A)
  with rho_A = tr_B(P)/g (the reduced state of the certified eigenvector for g = 1; for a cluster the normalised
  projector, the only basis-independent object), the entanglement spectrum as intervals, the computed value, the
  trace-distance bound T* = s (+ eta/(1 - eta) for g > 1) and the Fannes-Audenaert term. Mathematics
  (`docs/proofs/C-03_entanglement.md`): ||rho_A - rho_hat_A||_1 <= 2 T* from the rank-2g trace-norm bound on
  P - Qhat Qhat^H and the contractivity of the partial trace; |S(rho_A) - S(rho_hat_A)| <= T* ln(d_A - 1) + h(T*)
  (Audenaert 2007; the upper bound T* may replace T only for T* <= 1 - 1/d_A, where the function is increasing:
  outside that range the record is NOT_CHECKED with the reason, never an interval); |lambda_k(rho_A) -
  lambda_k(rho_hat_A)| <= 2 T* (Weyl). The reduced matrix of the stored vectors is formed in exact rational
  arithmetic, its spectrum enclosed by Ostrowski's and Weyl's theorems around a floating-point
  eigendecomposition with exactly computed defects, and -p ln p, h and ln evaluated on dyadic intervals
  (`ivl.py`). New module `certify_ed/entanglement.py`; one hook in the S4 loop; `scope.OUT_OF_SCOPE_QUANTITIES`
  lists the quantity as DEFAULT_SELECTION (not NOT_IMPLEMENTED) and omits it when requested. Symmetry-adapted
  claims are not lifted (UNSUPPORTED_CLAIM). Verification: 1 150 random + 20 026 adversarial cases, 0
  violations; tests `tests/test_entanglement.py` (11) against 40-digit references, planted rotations and the
  negative controls.

- **C-04 correlations for symmetry-adapted claims.** A claim with `sector_isometry` V used to leave
  `Options.correlations` NOT_CHECKED (UNSUPPORTED_CLAIM). Now `correlations.certify_correlations_lifted` builds
  the site operators in the full space and certifies tr(Pi_Q O) through the isometry: Theorem 6 in the block on
  the compressed Hermitian operator (V^H O V)_h, plus g times the compression rounding bound (standard model,
  `compress_operator`), plus the isometry term 2 g ||O|| delta (1 + sqrt(1 + iota)), delta = 1 - sqrt(1 - iota),
  iota = S1's isometry defect, where Pi_Q = Q P_s Q^H is the orthogonal projector onto the lift of the certified
  block eigenspace by the polar factor Q of V (`docs/proofs/C-04_correlation_lift.md`). Each pair record carries
  `bound` (for Pi_Q) and `bound_V` (for Pi_V = V P_s V^H, the exact identity tr(P_s V^H O V) = tr(Pi_V O)), the
  defects and `refers_to`; two-letter requests go through the Hermitian decomposition; structure factors apply
  unchanged. A full-space matrix in `Options.observables` is lifted the same way for sector claims. Without
  S1's audit the records are NOT_CHECKED (DEPENDENCY_NARROWED); when the audit failed they are INCONCLUSIVE.
  The lift does not claim that range(Pi_Q) is an H-eigenspace (that rests on the invariance defect, recorded)
  nor that other sectors are absent from the level. Verification: 240 random + 14 adversarial cases, 0
  violations; tests `tests/test_correlation_lift.py` (7) against full-space values on the 8-site ring (k = 0 and
  parity sectors, S^z S^z and S+ S-, S(q)), with a planted 0.1 % stretch (INCONCLUSIVE), S1 skipped
  (NOT_CHECKED) and a planted iota = 0.05 where only the isometry term makes the interval contain tr(Pi_Q O).

### Statements track (roadmap C-01, C-11, C-02, C-05, C-09, C-10; track/statements)

- **C-01 gap record.** `body.gaps` (`certify_ed/gaps.py`): for a claim whose S2 and S3 are both CERTIFIED, the
  S2 enclosures (radii including the model shift) are merged into levels (maximal chains of overlapping
  intervals), each holding exactly its count of eigenvalues (Corollary 5 for 'lowest', the window counts for
  'window', count = dimension for 'all', then the pigeonhole step of theory §6), and every difference of
  consecutive levels gets an outward-rounded interval. For 'lowest' claims the gap above the highest claimed
  level has S3's `gap_lower_bound` as lower end and no upper end (the next eigenvalue is not claimed); it is
  repeated as `S3.evidence.gap_interval`. To bound λ_{k+1} − λ_k on both sides, claim k + 1 pairs: the upper
  end is then μ_{k+1} + r_{k+1} − (μ_k − r_k) (Theorem 1). No new statement id (the scope matrix calls this a
  presentation task); the schema has `$defs/gaps_record`. Negative controls: a planted wrong (k+1)-th
  eigenvalue gives S2 FAILED and no record; without S3 there is no record (`tests/test_gaps_crossings.py`).
- **C-11 certified crossings.** `certify_ed.crossings.certified_crossing(cert_a, cert_b, level=0)` and
  `certify-ed crossing A.json B.json [--level L] [--json out]`: the rank-`level` eigenvalue intervals of two
  certificates are compared; relation `<`, `>` or `not separated`, the margin (rounded down), both intervals,
  both certificate hashes and spec hashes, and the caveat when a certificate has no S3 CERTIFIED (then the
  interval is the S2 enclosure of the claimed value at that sorted position and the statement says so). A
  certificate without S2 CERTIFIED is refused (`ValueError`). Pure interval comparison of certified data.
- **C-02 vectorless window count.** An eigenvalue-only claim with `kind='window'` used to leave S3 NOT_CHECKED
  (NEEDS_EIGENVECTORS). Now `_certify_window_vectorless` certifies the number of eigenvalues in (a, b) from
  inertia counts alone: Lemma 4 at the edge, plus two counts at e ± δ whose agreement makes the edge collar
  eigenvalue-free through the count inequality of Proposition 14 (docs/theory.md §3, "Vectorless window
  count"). S3 is CERTIFIED when the count equals the number of claimed values, FAILED otherwise
  (`missing_at_least` / `spurious_at_least`), INCONCLUSIVE when an edge is within the backward error of an
  eigenvalue (the summary says "window edge ... of an eigenvalue"; `next_action` says to move the edge).
  Evidence: `vectorless: true`, `count_in_window`, `edges`, `count_statement`, `pairing`. No claimed value is
  paired with an eigenvalue: S2, S4, S8 stay NOT_CHECKED and the summary says so. Six factorizations per
  window. Tests: `tests/test_vectorless_window.py` (count matches the dense spectrum at four windows, planted
  wrong counts FAILED both ways, edge on an eigenvalue INCONCLUSIVE, random-window negative control, gates).
- **C-05 Hermitian decomposition.** `Options.observables` and `Options.correlations` accept non-Hermitian
  operators: `correlations.certify_observable_general` splits O = H + iK (H = (O + O^H)/2, K = (O - O^H)/(2i)),
  certifies tr(P H) and tr(P K) by Theorem 6 and records the recombination tr(P O) = tr(P H) + i tr(P K) with
  bound b_H + b_K (Proposition 15); the record names both parts (`parts`, `definition`, `recombine`,
  `value` = [re, im]). The parts are checked to be formed exactly in floating point (TwoSum); otherwise the
  record is INCONCLUSIVE. Correlation requests take a two-letter string (`{"letter": "+-", ...}` for
  S+_i S-_j, built as one two-site term from the spec; C_ji = conj(C_ij) in the structure factor). A
  non-Hermitian observable used to reach `certify_observable` unchecked; `certify_observables_batch` used to
  raise. Tests: `tests/test_hermitian_decomposition.py` against the exact N = 4 ring values (-1/3, 1/6, 1/2;
  S(pi) = 4/3), with a rounded-parts negative control.
- **C-09 level statistics.** `certify_ed.levelstats.level_statistics(cert, unfolding='none'|degree, bins, bin_range)`
  and `certify-ed levelstats CERT.json [--unfold poly:D]`: from a kind-'all' certificate with S2 and S3
  CERTIFIED, the levels of `gaps.merge_enclosures` carry ranks, and interval arithmetic (`ivl.py`) gives the
  spacings s_n, the r-ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) (undefined when the larger spacing
  may vanish, as inside a multiplet), the mean r as an interval, and a spacing histogram in units of the mean
  spacing with per-bin at-least / at-most counts. The unfolding is a declared parameter: none, or a polynomial
  fitted to the staircase in floating point, evaluated on the intervals by interval Horner, recorded with its
  coefficients and marked "declared, not certified"; the unfolded spacings are clipped to >= 0 only when the
  derivative of the polynomial is positive on the hull of the spectrum (interval check). A certificate
  without a counted spectrum is refused (`LevelStatisticsError`, CLI exit 2 with a message). The scope block's
  "level statistics" row keeps NOT_IMPLEMENTED (no certificate statement) with the detail naming the tool.
- **C-10 extrapolation report.** `certify_ed.extrapolation.extrapolation_report(certs, model="1/N"|"1/N^2"|callable,
  sizes=None, per_site=False, level=0)` and `certify-ed extrapolate C1.json C2.json ... [--model] [--per-site]`:
  Markdown and JSON listing the per-size certified intervals by certificate hash (through
  `crossings.level_interval`, with the caveat when S3 is not CERTIFIED), a numpy least-squares fit of the
  declared model to the midpoints, and the fitted limit with the sentence that it is NOT certified and
  depends on the declared model. The extrapolated record has no status field and its report section contains
  no status word (test). `certify_ed/extrapolation.py` is in `provenance.NON_COMPUTATIONAL`; `gaps.py`,
  `crossings.py` and `levelstats.py` are not (their intervals are claimed-certified numbers).
- New public names: `certified_crossing`, `level_statistics`, `extrapolation_report` (docs/api.md regenerated).
  Tests: `tests/test_levelstats_extrapolation.py`.
### Whole spectrum, interior windows, streaming vectors, planner, operator text (roadmap C-17, C-14, A-06, D-08, C-15; track/whole-spectrum)
- C-17 whole-spectrum path. `quick(H, w, V, spec, kind="all")` and `certify-ed certify --kind all` use the
  `whole-spectrum` preset unless a preset, `--only` or `--skip` is given (recorded in the scope block as any
  preset); `temperatures=[...]` adds S7. The evaluation of the S2 bounds was reorganised without changing a
  bound: `enclosure.residuals` works in column blocks (bit-identical for sparse and spec operators; a dense
  operator keeps one `gemm` call because OpenBLAS `gemm` is not invariant under column blocking), the rounding
  bound `E` is reduced to its column norms and not stored, the dense `|A|` is formed transiently, the exact
  Gram inner products of a cluster are cached across merges, the Kato-Temple products `H x`, `|H||x|` are
  formed for a block of isolated pairs at once (for a sparse operator the intervals are bit-identical to the
  per-pair ones; for a dense operator they differ in the last bits because `gemm` columns and `gemv` are
  different floating-point evaluations of the same bound), and the S4 per-cluster Python loops over all other
  pairs became array operations. `fp.exact_sum` (cascaded error-free summation, Ogita-Rump-Oishi SumK with
  Knuth's TwoSum) returns `math.fsum`'s value and falls back to it whenever a rounding boundary is within the
  slack; `exact_dot` uses it for long vectors. Timings and peak memory, whole-spectrum preset with two
  temperatures, Heisenberg chain Sz = 0 blocks, dense operator unless noted (8 cores, OpenBLAS; before -> after):
  the 924 block takes 3-4 s either way (the git provenance call dominates);
  dimension 3432 (dense operator): total 34 s -> 22 s, S2 18 s -> 14 s, peak RSS 0.9 GB -> 0.7 GB (radii sha 7b132274 = 7b132274)
  dimension 12870 (sparse operator): total 568 s -> 398 s, S2 536 s -> 366 s, peak RSS 8.7 GB -> 2.9 GB (radii sha def5ba16 = def5ba16)
  dimension 12870 (dense operator): total 798 s -> 635 s, S2 657 s -> 585 s, peak RSS 9.8 GB -> 8.5 GB (radii sha 01b8df55 = 01b8df55)
  (the 12870 sparse runs overlapped with the test suite; the dense ones ran alone). The dense 12870 run of dev/3.2
  needed 10.1 GB on a 16 GB machine and paged; the remaining S2 time is two n^2 k gemms (about 4 TFlop each at
  40 GFlop/s) plus the exact Gram inner products of the 6400 clusters.
  The radii are bit-identical before and after on every fixture (`tests/test_whole_spectrum.py` against the
  frozen `tests/_enclosure_reference.py` and against the recorded radii in `tests/fixtures/`).
- C-14 interior windows everywhere. `Claim` validates `kind` and `window` (`certify.parse_window`: a pair
  a < b of finite numbers, from a tuple, a list or 'a,b' / 'a b'); the same rule applies in
  `claim_from_arrays/npz/hdf5/file/bundle`, in the bundle's `meta.json` and on the command line, where
  `--window A B` (or `--window=A,B`) overrides the bundle. The certificate's `claim.window` is a list.
  Cookbook: `docs/cookbook/shift_invert_window.md` (`eigsh(sigma=..., which="LM")`, executed output).
- A-06 streaming vectors. `claim.eigenvectors` may be a numpy memmap or an h5py dataset: the residuals, the
  Gram matrices and the cluster/Kato-Temple columns are read in blocks of `Options(block_mb=...)` (default
  `budget_mb`, else 512 MB); the certificate is identical to the in-memory one (memmap and HDF5, C and F
  layouts, tested). Limits: a dense operator reads the vectors whole (no restriction: it is larger than the
  vectors); S3 by inertia for `lowest`/`window` claims still factorizes the stored matrix in memory
  (`docs/cookbook/streaming_vectors.md`). `array_sha256` and the underflow scan work in row chunks.
- D-08 planner. `cost.plan_certification` / `plan_for_claim` predict seconds and peak memory per statement from
  n, nnz and k (dense LDL^T workspace 8.5 n^2 entries; sparse LU from a fitted fill model; residual arrays
  counted one by one; rates measured on this machine, `cost.RATES`, `experiments/planner_calibration.py`);
  `Options(plan=True)` / `--plan` print and record it (`body["plan"]`, schema re-rendered); `certify-ed plan`
  prints it without running. `inertia_method="auto"` keeps the 3.1 rule and additionally sends a sparse matrix
  whose dense workspace would exceed `budget_mb` (or the available memory) to SuperLU
  (`cost.recommend_inertia_method`); the S3/S8 memory gate estimates the path that will be taken (3.1 estimated
  the dense path only, so a budget below the dense workspace made S3 NOT_CHECKED even when SuperLU fitted).
  `tests/test_planner.py`: predicted memory within a factor 2 of the measured peak (dense LDL^T at 3432, SuperLU
  at 3432 and 12870, the recorded whole-spectrum runs); statuses of `auto` equal those of the explicit method
  on the fixtures; `plan=True` changes nothing in the certificate. Advisory throughout: nothing here changes a
  status.
- C-15 operator text. `certify_ed.operators.parse_operator`: Pauli strings ("XIZY", "X0 Z2"), letter_site
  shorthand ("Sz_0 Sz_3", "S+_0 S-_3", "n_0 n_1"), linear combinations ("0.5 S+_0 S-_1 + 0.5 S-_0 S+_1"),
  dictionaries and QuSpin-style operator lists become the sparse operators `Options.observables` certifies;
  `Options.correlations` accepts "Sz_0 Sz_3" for `{"letter": "z", "sites": [0, 3]}`. Non-Hermitian results
  ("S+_0 S-_3" alone) are refused with the Theorem-6 message (the C-05 Hermitian split is not in this branch).
  Tests against Kronecker-built matrices on 4 sites.

### Usability (roadmap A-01, A-02, A-09, A-04, A-05; track/core-api)
- `certify_ed.quick(H, w, V=None, spec=None, kind="lowest", **options)`: builds the Claim and Options and calls
  `certify()` (no check of its own). `Certificate.summary()` is now one sentence; the former multi-line text is
  `Certificate.report()`; `_repr_html_` / `_repr_markdown_` render a statement table in Jupyter.
- `Certificate.methods_paragraph(style="plain"|"latex")` and `certify-ed statement CERT.json` (`certify_ed/report.py`,
  outside the fingerprint): a methods-section paragraph whose every number is a certificate field, rounded in the safe
  direction (radii up, gap bounds down), with hypotheses, run-stack caveat, body hash and a BibTeX entry (concept DOI;
  version DOI placeholder).
- `evidence["next_action"]` on every INCONCLUSIVE and NOT_CHECKED statement (`certify_ed/diagnostics.py`).
- New reason code `MISSING_DEPENDENCY`: without python-flint, S6 is NOT_CHECKED instead of an import error; the QuSpin
  importer names the extra to install. `certify_ed.apps` and the exact checker itself still need python-flint.
- `Options(budget_seconds="auto")` / `--budget auto`: a default budget from a cost model fitted to seven E2 numbers of the
  3.1.0 evidence run (`certify_ed/cost.py`; advisory, never changes a status); `Options(progress=True)` / `--progress`.

### Producers (roadmap F-01, A-07, F-02, F-05; track/producers)
- F-01 `certify_ed.integrations.quspin.certify_hamiltonian(H, w, V, basis=None, static=None, spec=None, kind="lowest",
  **options)`: one call from a QuSpin `hamiltonian` and the output of `H.eigsh` / `H.eigh` to a Certificate. The
  basis and the operator-string list are taken from the object; eigenvalues are sorted for `lowest` / `window`
  claims; the Claim and ModelSpec come from `claim_from_quspin`, so a symmetry block (kblock, pblock, ...) keeps
  the S1 sector audit through `basis.get_proj`. Time-dependent Hamiltonians are refused. `spec=` certifies
  against an independently written model. Docs: `docs/integrations/quspin.md`; a draft upstream pull-request
  text proposing `quspin.tools.certify` is `integrations/quspin_PR.md` (text only). Tests:
  `tests/test_integrations_quspin.py` (full, U(1), kblock and kblock+pblock bases give the statuses of the direct
  route; unsorted eigsh output; spec override with a wrong coupling -> S1 FAILED; negative controls: a planted
  eigenvalue and a swapped eigenvector are not certified).
- A-07 producer record hook `certify_ed.record(spec, path, kind=..., window=..., fmt=...)`: `with record(...) as
  rec: ...; rec.capture(w, V, H)` (or the attributes `rec.eigenvalues`, `rec.eigenvectors`, `rec.operator`,
  `rec.basis`) writes a producer bundle (schema 1.0) on exit with `meta.environment` (Python, numpy, scipy,
  certify-ed versions, platform) and `meta.timing` (UTC start/end, wall and process seconds); the decorator
  `certify_ed.recorded(spec, path)` wraps a function returning `(w, V)` or `(w, V, H)`. Nothing is written when
  the block raises; exiting without eigenvalues raises. The bundle schema (`certify_ed/schema.py`,
  `schema/producer-bundle-1.0.schema.json`) documents the two new optional fields. Module `certify_ed/recorder.py`
  (NON_COMPUTATIONAL). Tests: `tests/test_recorder.py` (bundle passes `check_bundle` and the CLI `conform` and
  `certify --from`; statuses equal the direct Claim; decorator; U(1) basis; window kind and csv format; meta
  validates with jsonschema; negative controls: a planted eigenvalue and a planted operator entry written through
  the hook are not certified).
- F-02 file importers for other ED codes (`certify_ed/io/hphi.py`, `diagham.py`, `spinpack.py`, `alps.py`;
  `claim_from_hphi`, `claim_from_diagham`, `claim_from_spinpack`, `claim_from_alps`; docs/integrations/ed_codes.md).
  Layout sources: HPhi `energy.dat` / `Eigenvalue.dat` / binary `eigenvec` (manual 3.5: int iterations, unsigned long
  local_size, complex double x (local_size + 1), first component unused; the 4-byte-long variant is detected from the
  file size); DiagHam binary real vectors (wiki: int dimension + doubles) with complex vectors ASSUMED Re/Im
  interleaved, spectrum text files (quantum-number columns, energy last: observed convention); SPINPACK: ASSUMED
  text layouts (`table`, `keyvalue`), recorded as `layout_status` in the claim's producer record, no vectors; ALPS
  `/spectrum/sectors/<n>/energies`, `quantumnumbers`, and the `SAVE_EIGENVECTORS` eigenvector and basis tables
  (reordered into the spec encoding when a spec is given). Every reader checks header against file size and column
  counts and raises `FormatError`. The basis order of HPhi/DiagHam/SPINPACK relative to a ModelSpec is not
  established: the caller passes `operator=` and `states=`. Fixture generators: `tests/fixtures/ed_codes.py`.
  Tests: `tests/test_io_ed_codes.py` (22: layouts, truncation and header checks, round trips through certify()
  on a 6-site chain with S1-S3 CERTIFIED, MPI rank concatenation, negative controls: planted energy and swapped
  vector in the producer's files are not certified).
- F-05 `ModelSpec.from_pauli_strings({"XXI": 0.5, ...})`, `ModelSpec.from_openfermion(QubitOperator)`,
  `ModelSpec.from_qiskit(SparsePauliOp)` (`certify_ed/integrations/pauli.py`, docs/integrations/pauli.md): spin-1/2
  Pauli-convention specs with character j on site j, basis index = big-endian bit string (the ordering of
  `get_sparse_operator` and `SparsePauliOp.to_matrix()`; Qiskit's little-endian labels give site = n-1-qubit,
  recorded in `parameters`), real coefficients required, identity term kept. The packages are read through their
  object interfaces (`.terms`; `.to_list()`, `.num_qubits`) and never imported. Tests: `tests/test_pauli_specs.py`
  (15: TFIM from strings equals `tfim_chain` entry by entry, a 2-qubit H2-like operator certifies with kind='all'
  against numpy, complex odd-Y operator, OpenFermion/Qiskit-like objects, JSON round trip; negative controls:
  a producer matrix with one coefficient off by 1e-3 -> S1 FAILED, planted eigenvalue not certified, wrong
  endianness -> S1 FAILED).
- `schema/certificate-2.1.schema.json` re-rendered: the file on dev/3.2 lacked the reason codes
  `MISSING_DEPENDENCY` and `UNDERFLOW_RISK` that `scope.REASON_CODES` already carried (the Wave 0 merge did not
  re-render).

### INT-01: the no-underflow hypothesis is enforced (E-06; design decision D-57, conservative option)
- The standard model behind every floating-point bound assumes no underflow: nonzero operands, radii and
  residual bounds of magnitude at least 2^-900 (`fp.UNDERFLOW_THRESHOLD`). 3.1.0 stated this and probed
  flush-to-zero, but never looked at the input. `certify()` now scans, before S2, the claimed eigenvalues,
  the eigenvector entries, the operator's stored entries (sparse data array; for a spec or matrix-free
  operator the norm bound and one matvec on a seeded random probe) and the model shift, and after the
  enclosure the radii, existence radii, residual bounds and residual norms. Exact zeros are fine.
- Any finding makes S2 INCONCLUSIVE with the new reason code `UNDERFLOW_RISK` (evidence: where, first
  indices, minimal magnitude, threshold; the summary keeps what the status would otherwise have been), and
  S3, S4, S7, S8 INCONCLUSIVE with the same code, since each rests on the S2 enclosure. S1 and S6 (exact
  arithmetic) are not affected. Statuses of inputs without a finding are identical to a run without the
  check (test `test_standard_fixtures_unchanged_by_underflow_check`); the existing fixtures have none.
- This is the conservative choice and it is reversible: `Options(underflow_check=False)` (CLI
  `--assume-no-underflow`) skips the scan and records `underflow_check: false` in the certificate's
  `scope.options` and `arithmetic.underflow_check.performed`; the certificate then carries the hypothesis
  as the user's. Rescaling the input instead of rejecting it was not implemented (it would need a
  statement about the rescaled operator). Alternatives remain open to Sarang (D-57).
- `docs/theory.md` section 1 names the hypothesis and the enforcement; `docs/SCOPE_MATRIX.md` and
  `scope.REASON_CODES` carry the code. Tests: `tests/test_underflow_check.py` (19: planted subnormal
  eigenvector entry, planted 2^-950 dense and sparse matrix entries, a radius of 1e-274 from an exact
  eigenpair of a 1e-258-scaled operator with eigenvectors of norm 1e125, tiny claimed eigenvalue,
  matrix-free probe, spec operator probe, complex components, opt-out recorded, CLI flag, fixtures
  unchanged, LAPACK numerical zeros not flagged, negative control: a finding never leaves S2 or a
  dependent statement CERTIFIED).
- **INT-05 (new known issue, found while implementing INT-01; needs a design decision).** The scan covers
  the stored inputs and the reported outputs. The *intermediate* quantities of the Theorem 1 bound are not
  covered by the hypothesis as stated, and routine inputs violate it there: `scipy.linalg.eigh` returns
  eigenvector entries of order 1e-265 (numerical zeros of a symmetry sector; above 2^-900), for which the
  residual vector R and the entrywise rounding bound E = gamma (|H||x| + |x||mu|) have entries of order
  1e-278 (below 2^-900), and their squares in `np.linalg.norm` underflow to zero. The lost mass is tiny
  compared with the guards (1 + 8u) of `fp.norm2_upper`, but saying so is a bound the theory does not
  contain (D-12): either the hypothesis is strengthened to cover the intermediates (which would make
  almost every LAPACK input INCONCLUSIVE), or the standard model is replaced by the underflow-tolerant
  model fl(a op b) = (a op b)(1 + delta) + eta, |eta| <= u lambda (Higham 2002, eq. 2.6; Demmel 1984)
  with an absolute term carried through `norm2_upper`, `residuals` and the Gram bounds. An early version
  of this branch scanned R and E and flagged `test_wrong_bond_sign_is_localized`; it was withdrawn. Test
  `test_lapack_eigenvectors_with_numerically_zero_entries_are_not_flagged` pins the observation.

### INT-04: code-facing accounting of the Theorem 1 radius (documentation)
- `docs/theory.md` section 2 now lists, line by line, how `enclosure.residuals` evaluates the Theorem 1
  radius in floating point: three roundings each covered by a factor (1 + 4u), the norm bounds of
  `fp.norm2_upper` / `fp.norm2_lower` with their (1 + gamma_{n+2}) and (1 + 8u) guards, and the model shift
  added afterwards with `add_up`. The code's bound is unchanged. The manuscript's Appendix A (k <= 6
  roundings, 1 + 8u, shift multiplied in) is to be rewritten from this list by Sarang (D-16).
  Test `tests/test_theorem1_accounting.py` recomputes the radius of a fixed 3 x 3 case with `Fraction`
  arithmetic and pins the code's value between the exact radius and the documented-factor evaluation.

### Merged from `fix/3.1.1` (f441227)

Two of the known issues of 3.1.0, fixed without touching what any statement asserts (merged with a merge commit;
under D-44 a version the paper cites needs its own full run).

- **INT-02.** `Certificate.verify_scope()` checks the scope block: every statement id exactly once (in scope, narrowed,
  or out of scope), known reason codes, every statement in scope present, a narrowed statement that is present is
  `NOT_CHECKED`, nothing reported outside the scope lists; `None` for certificates written before the scope block.
  `verify_integrity()` reports `scope_ok`/`scope_problems` and fails on `False`, so `Certificate.load()` refuses such a
  certificate even when its body hash was recomputed.
- **INT-03.** The exported `radius` and `existence_radius` arrays add the model shift with upward rounding (`add_up`), as
  the statements do; without a shift the radii are exported unchanged. The builder's rounding bound makes the shift
  nonzero for almost every certificate, so this changes almost every exported radius by at most one ulp.
- Tests: `tests/test_fix_3_1_1.py` (12; including a writer that drops a statement and recomputes the hash).
  Full suite 556 passed, 1 skipped; ruff clean.
### Format, packaging and governance (roadmap B-02, F-09, A-03, H-01, H-05, B-09, G-05, G-12; track/format-packaging)

No statement, bound or number changes in this track; the version is bumped in the known-issues track.
`certify_ed/schema.py` and `certify_ed/conformance.py` are listed in `provenance.NON_COMPUTATIONAL`.

- **B-02.** Published JSON Schema (draft 2020-12) of the certificate body, `schema/certificate-2.1.schema.json`,
  and of the producer bundle, `schema/producer-bundle-1.0.schema.json`, rendered from `certify_ed/schema.py`
  (the enumerations are the code's; a test keeps the files in sync). `Certificate.load(validate_schema=None|True|False)`
  validates when `jsonschema` is installed (extra `schema`) and otherwise runs the structural checks (every
  statement id exactly once in the scope block, reason codes declared and known, in_scope statuses equal to the
  statements); a non-conforming body raises `SchemaValidationError`. This also closes INT-02 independently of
  `fix/3.1.1`. `docs/schema.md`: field by field, stability promise, how a third-party writer conforms.
- **F-09.** `certify-ed conform DIR...` (`certify_ed/conformance.py`): bundle schema, import, certify, certificate
  schema, expected statuses. Three reference bundles under `conformance/reference/` (dense text, npz, matrix-free)
  with `expected.json`; `conformance/README.md` for third-party writers. Negative controls: unknown status,
  unknown reason code, CORROBORATED outside S5, duplicated/missing scope entries, tampered files with a recomputed
  hash, a bundle with a planted wrong eigenvalue.
- **A-03.** Extras `exact`, `quspin`, `hdf5`, `plots`, `schema`, `all` (the union), `dev`; `MANIFEST.in` ships the
  schemas, the reference bundles, docs and tests in the sdist. `python -m build` gives an sdist and a wheel; the
  wheel installs into a fresh venv and `certify-ed env` runs. `packaging/conda-forge/meta.yaml` (noarch draft),
  `packaging/RELEASE_CHECKLIST.md` (version bump, annotated tag, full run per D-44, checksums, sigstore, Zenodo
  new version under 10.5281/zenodo.20066565, GitHub release; never a force-push).
- **H-01.** `.github/workflows/ci.yml` replaces `tests.yml`: lint, build, schema sync; ubuntu/macos/windows x
  Python 3.11/3.12 with `.[all]` and QuSpin 1.0.1 (`--no-deps` after its real dependencies; the wheel declares the
  obsolete `importlib` backport; documented in `docs/reproducibility.md`), test log uploaded; `nightly-mutation.yml`
  runs a 32-mutant E6 sample without a failure gate.
- **H-05.** `containers/Dockerfile` and `containers/certify-ed.def` pin the 3.1.0 evidence-run stack (Python 3.11,
  numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0, QuSpin 1.0.1, h5py, matplotlib, mpmath).
- **B-09.** `packaging/SIGNING.md`: keyless sigstore signing and verification of release assets and `SHA256SUMS`.
  The certificate schema reserves `environment.distribution_sha256`; 3.1 does not write it (the hash of an installed
  distribution is not well defined; `source_hash` identifies the code from a checkout).
- **G-05 / G-12.** `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `GOVERNANCE.md` (maintainers,
  decisions, release cadence, deprecation policy: schema fields never removed within a major version),
  `docs/SCHEMA_RFC_PROCESS.md`, issue templates (bug, feature, *I could not certify X* feeding `docs/SCOPE_MATRIX.md`),
  `ROADMAP.md` summarising the scored expansion roadmap.
- Tests: `tests/test_schema_conformance.py` (17), `tests/test_packaging.py` (7).

### Viewer and corpus (roadmap A-08, B-07; track/viewer-corpus)
- `certify-ed view CERT.json [-o CERT.html]` and `certify_ed.viewer.view_file` (A-08): one self-contained HTML
  page per certificate (inline CSS, no scripts, no external resources): header (model, claim kind and k,
  version, hashes, environment, floating-point probe, no-underflow scan), the statements in plain words
  (status, summary, next action, key evidence), the scope block (in scope / narrowed with the user's reason /
  not checked and out of scope with the reason code and its scope-matrix explanation), the enclosure table
  (per pair: claimed value, radius and existence radius for the producer operator and, from the sidecar, with
  the model shift; enclosure interval; cluster), the Kato-Temple intervals, the S3 gap and inertia data, the
  S4/S8 cluster tables, S7 intervals, S6 exact data, the methods paragraph of `report.py`, and an integrity
  footer (body hash stored and recomputed, array hashes, `verify_scope`, schema validation). Every number is
  the certificate's value rounded by `report.round_decimal` in the safe direction (radii and error bounds up,
  gap and separation bounds down, certified windows inward, enclosures outward; claimed values and counts
  exact) and carries the exact value in `data-exact`. A tampered body, a modified or missing sidecar, or a
  scope block that drops a statement produces a red INTEGRITY CHECK FAILED banner and exit code 1 (the page is
  still written). User strings are HTML-escaped. `certify_ed/viewer.py` is in `provenance.NON_COMPUTATIONAL`.
  Tests: `tests/test_viewer.py` (20: every schema fixture certificate renders with every statement id and
  status; every displayed number checked against its exact value and direction; enclosure rows against the
  body and the sidecar; tampered body / sidecar / scope block banners; `<script>` in the narrowing reason and
  the claim description escaped; pre-2.1 certificate without scope block; CLI exit codes). `docs/viewer.md`.

- Planted-error corpus `corpus/` (B-07): `corpus/make_corpus.py` writes 60 producer bundles (1.7 MB; Heisenberg chain
  N = 8 and 10, J1-J2 chain N = 8 at J2 = 0.5, TFIM N = 8; dense `eigh` references, deterministic) with one planted
  error each in fourteen classes (wrong eigenvalue by 1e-5 and past the next level, unnormalised isolated vector and
  cluster member, swapped pairs, missing multiplet member, duplicated vector, wrong coupling, wrong matrix entry,
  non-Hermitian producer matrix, NaN eigenvector entry and eigenvalue, mislabeled sector, subnormal entry, wrong
  window count) plus one control per model, each with an `expected.json` naming the statements a correct checker
  must not certify (derived from the statement definitions, not from a checker). `corpus/run_corpus.py` runs any
  checker command on every case, reads a certificate or a flat `{statement: status}` result, and scores it
  (detected / unsound / silent / refused / conditional / error; false and weak alarms on the controls) into
  `results/<name>/report.md` and `report.json`. `corpus/README.md` describes every class, what a correct checker
  must say, the scoring rules, the revision history of the expectations and how to submit a case.
  Score of CERTIFY-ED 3.2.0.dev0 on the corpus (`corpus/results/certify-ed/`): detected 53 / missed 0 (unsound 0,
  silent 0, error 0) / false alarms 0 (weak 0) of 53 planted cases, 7 controls clean; the 8 NaN cases are detected
  by refusal at import. Disclosure: the first run, with `unnormalised_cluster_member` expecting only S4, scored
  detected 49 / missed 4 (silent 4: S4 NOT_CHECKED behind an INCONCLUSIVE S3, S2 INCONCLUSIVE on distinctness);
  the expectation was then revised to S2 and S4 (Theorem 2 and Theorem 3 share the near-orthonormality
  hypothesis, as the case note already said) and both reports are committed
  (`results/certify-ed_expectations-before-revision/`). No change was made to the certifier for the corpus.
  Tests: `tests/test_corpus.py` (71: every committed case is a readable bundle with a consistent expectation,
  index hashes, in-process determinism, the planted changes are present in the arrays, scorer verdicts, summary
  counts, result reader, runner end to end with a checker that certifies everything (unsound) and one that
  refuses, certify-ed on one planted case and its control, the committed report has no unsound case).
  `docs/corpus.md`. `corpus/` is outside the computational source fingerprint (it is neither `certify_ed/` nor
  `experiments/`); `experiments/e1_failure_matrix.py` is unchanged.

## [3.1.0] - 2026-10-22

`__version__` = 3.1.0 (tag `v3.1.0`; evidence run `results/run_full_20261022`, made from the tagged commit on the
Windows workstation with the reference library versions, D-54). Two soundness fixes found
while writing the proofs for the manuscript scaffold (`ms_v3`), sharper application bounds, and the
fixes of the independent review of 2026-10-21 (ids IMP-xx: code audit; SS-xx: second stack; MS-Nxx:
manuscript check). The computational source hash changes with this release, so the records of the run
of 2026-10-21 (source 7fb63586baf7ea6f) are superseded by the next full run.

### Known issues of 3.1.0 (found while filling the manuscript's evidence pointers, 2026-10-22)
- **INT-01.** The no-underflow hypothesis of the floating-point statements is assumed, not enforced: `fp.py` probes
  gradual underflow and denormals-are-zero, but no input check rejects operands or radii below 2^-900 (the
  manuscript said it did). Needs a design decision (reject, rescale, or state the hypothesis as the user's).
- **INT-02.** No validator rejects a certificate whose scope block omits a statement: `verify_integrity` checks the
  hashes only, and the completeness check of the scope block exists only as a test helper. Fixed on `fix/3.1.1`.
- **INT-03.** The exported `radius` and `existence_radius` arrays are `enc.radius + eps_shift` in round-to-nearest
  (`certify.py` 1161-1162) and can understate the sum by half an ulp; the statements themselves use `add_up`.
  Fixed on `fix/3.1.1`.
- **INT-04.** Appendix A's accounting of the Theorem 1 radius (k <= 6 roundings, factor 1 + 8u, model shift
  multiplied in) does not describe the code (three roundings, factor 1 + 4u, model shift added with upward
  rounding). The code's bound is sound; the text is to be rewritten (D-16).

### Evidence run of 3.1.0 (2026-10-22)
- `results/run_full_20261022`: tag `v3.1.0` (`e860a83`), computational source `acbef11af991c860` for every step, no step
  dirty; Windows workstation, Python 3.11.17, numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0
  (FLINT 3.6.0), QuSpin 1.0.1 (the library versions of the Linux 3.0.0 run). Tests: 541 passed, 1 skipped, 1 warning.
  E0–E10 ok. Numbers: 61/61 and 45/45 checks, after one change to `make_numbers.py` (`55c2ef9`): the W4 check
  asserted the Linux outcome for the 4×4 J2 = 0.5 completeness (NOT_CHECKED); here S3 is CERTIFIED (gap ≥ 0.2870)
  because a job object counts committed memory where RLIMIT_AS counts address space. E6: 38 of 64 mutants killed;
  15 survivors await the human verdict of MS-N03 (`results/run_full_20261022/E6_inspection_worksheet.md`).
- Two stacks: against `run_full_20261021` (Linux, 3.0.0) `compare_runs.py` finds no MISMATCH leaf
  (`RESULT_evidence_run_3.1.0.md`). With the library versions pinned, the E9 exact lower bound and every E10
  factorization residual are bit-identical across the two operating systems (the up-to-6 % differences of the
  candidate run came from library versions); E7 exact radii still differ (ratios 0.25–1.47).
- `compare_runs.py`: named classes for prose with run-stack digits (floating-point numbers only; counts stay strict),
  records extended by a newer version, records drawn from the stack (E5 ARPACK rows, E6 mutant samples; compared by
  their invariants in the RESULT document), explained differences (`EXPLAINED`, with reasons), and timing/memory
  leaves; negative controls in `tests/test_compare_runs.py`.

### Windows portability (2026-10-22, found by the first attempt of the evidence run)
The first full run on the Windows workstation (`results/run_full_20261022_attempt1`, kept with an ATTEMPT_NOTE)
failed in four places that the Linux runs never exercised. None touched a certified statement.
- **Matrix-free importer.** scipy 1.17's `LinearOperator.matmat` calls the producer's `matvec` with (n, 1)
  columns, which a function written for vectors broadcasts to (n, n). `certify_ed/io/formats.py` now always calls
  the producer's `matvec`/`rmatvec` with a vector of shape (n,), checks that n numbers come back (FormatError
  naming the file otherwise) and builds `matmat` column by column. Test: `test_matrix_free_matvec_is_called_with_vectors_and_checked`.
- **E2, E4: memory cap and peak memory.** `resource` and `preexec_fn` do not exist on Windows. New
  `experiments/portable.py`: RLIMIT_AS on POSIX (unchanged), a job object with a per-process memory limit on
  Windows; peak working set on Windows; CPU model and RAM from the registry and GlobalMemoryStatusEx (the harness
  recorded neither on Windows, and `make_numbers.py` read /proc unguarded). E2 rows and E4 settings record the cap
  mechanism (`memory_cap`). Tests: `tests/test_portable.py` (a capped child must raise MemoryError).
- **E6: the mutants face the shipped suite.** The mutation harness copied only `certify_ed/`, `tests/`,
  `examples/`; six 3.1.0 tests read `pyproject.toml` or `experiments/` (so the baseline failed on every platform) and
  25 tests skipped in the copy against 1 in the repository. The copy is now every tracked file (`git ls-files`).
  `tests/test_driver.py`'s `needs_repo` also requires a git checkout, as `test_imp06` already did.

### Candidate run and integration (2026-10-22)
- `release/3.1.0` at `v3.1.0-candidate`: fix track + scope framework + survey/scope generators + v4 macro emission +
  `compare_runs.py` + `make_result_candidate.py` + E9 named-sector step. Full suite on the Windows integration stack
  (Python 3.12.15, numpy 2.5.4, scipy 1.18.1, python-flint 0.8.0): 502 passed, 31 skipped (30 need QuSpin, 1 needs gcc);
  ruff clean. Computational source `1d4e33529dfce716`.
- `results/run_candidate_20261021` (E7, E9, E10, E8 with the pre-merge source `e707838f6852d474`) reproduces every statement
  of `run_full_20261021`; see `RESULT_candidate_run.md`. The 3.1.0 evidence run (all experiments, tests enabled, one
  commit, no edits in flight) is still to be made on the reference stack.

### Fixed (review of 2026-10-21)
- **IMP-18 (critical).** `checker.radius_exact` returned radius 0 for a zero test vector, so a wrong
  eigenvalue claimed with a zero vector (or, in `exact_recheck_sector_lowest`, with a vector from another
  sector whose exact projection vanishes) was CERTIFIED "with no floating-point assumption". The function
  now returns None for ‖x′‖ = 0 and `exact_recheck_lowest`, `exact_recheck_sector_lowest` and S6 report
  INCONCLUSIVE; the hypothesis x ≠ 0 is stated in the docstrings.
- **IMP-21, IMP-22 (critical).** A NaN claimed eigenvalue passed every comparison: `certify()` reported S2
  and S3 CERTIFIED for a full-spectrum claim and `enclose_spectrum` returned `complete=True` with an
  eigenvalue in no interval. `enclose()` and `enclose_spectrum()` refuse non-finite claims (ValueError);
  `certify()` and the exact re-checks report S2/S3/S6 INCONCLUSIVE; the accuracy comparison is written
  fail-safe (`not (radius <= target)`); a group with a non-finite radius is never 'distinct'. A zero
  claimed vector no longer crashes S3 (IMP-23): INCONCLUSIVE with the pair named.
- **IMP-13 (major).** A correct claim with valid but non-orthogonal eigenvectors of a degenerate level was
  S2 FAILED (cluster radius 4.9 against a target 4.5e-8). Theorem 2′ (`docs/theory.md`, UNVERIFIED, D-12)
  gives the cluster radius ε* = ‖R‖/√(1−η) + ‖XᴴR‖/(1−η) + δ·spread(Λ)/√(1−η); `_cluster_bound` uses
  min(ε, ε*) and records both. Every cluster member keeps its Theorem-1 radius as `existence_radius`
  (records, arrays); the cluster radius remains the radius of the distinct matching and of every
  indexed statement. When only the cluster radius misses the accuracy target, S2 is INCONCLUSIVE with
  the diagnosis, not FAILED. The review's case is now CERTIFIED (radius 8e-14).
- **IMP-12 / IMP-20.** `compose.certify_low_energy` uses ε = ρ (Weyl for the square W) instead of
  (1+√5)/2·ρ, and records symmetrised producer blocks (`hermitian_part` per sector, `symmetrised_blocks`).
- **IMP-08.** The certified window of S3 'window' claims is rounded inward (`add_up(a, β_a)`,
  `sub_down(b, β_b)`); the same window feeds Kato–Temple and the S4 separations.
- **IMP-14.** `inertia_dense` checks that the permuted LDLᵀ factor is lower triangular, as
  `inertia_exact` does.
- **IMP-24.** `fp.probe_float_environment` detects round-toward-zero (1 + 3·2⁻⁵⁴ must give 1 + 2⁻⁵²,
  and the mirror). The rounding-mode tests run on Windows through `ucrtbase` as well as on glibc.
- **IMP-05 / SS-02.** Provenance calls git with `core.filemode=false`: a lost executable bit (Windows
  checkouts) no longer marks the computational source as dirty.
- **IMP-06.** The numbers step records its provenance (`provenance.numbers_fingerprint`: commit, blob
  hashes of `make_numbers*.py`, source and tests hashes) in `manifest.steps.numbers.provenance`.
- **IMP-07 / MS-N02.** `\NtestsTotal`, `\NtestsVtwo` (and new `\NtestsPassed`, `\NtestsSkipped`) come
  from the run's tests step (`manifest.steps.tests.summary`), not from a live collection.
- **IMP-26.** `__version__` = 3.1.0.dev0 (pyproject, `.zenodo.json`); every certificate's `environment`
  carries `source_hash`, `source_dirty`, `git_commit`, `python_flint` and `flint_c_library`.
- **SS-04.** `cli.NEEDS["E3"]` includes QuSpin (part d imports it); `experiments/harness.Run` logs
  "ABORTED" and records `_aborted` when the experiment raises, instead of "finished".
- **MS-N12.** The FLINT C-library version (`flint.__FLINT_VERSION__`) is recorded in
  `environment.json`, `certify-ed env` and certificates; macros `\FlintCVersion`, `\PythonFlintVersion`,
  `\CertifyEdVersion`.
- **MS-N03.** `e6_recheck.py` (`_provenance`) and `e6_classify.py` (`provenance`) stamp their outputs with
  the commit, `source_hash` and `tests_hash`; `make_numbers_apps.py` requires the stamps on
  `classification.json`, `recheck.json` and `inspection.json` (CHECK FAILED otherwise), checks that they
  refer to the E6 run's source, and that every "test added, mutant now killed" verdict has a recheck
  entry (`\MutRecheckKilled`, `\MutRecheckTestsHash`).
- **IMP-28 / MS-N01.** `\AKLTCritMin` is rounded down (0.44, was 0.45); a check verifies that the
  printed range contains every criterion value.
- **IMP-27 / SS-15.** The AKLT agreement check tests exactly the printed `\AKLTMaxExcess` (the fixed
  5e-9 threshold is gone).
- **MS-N04 / MS-N05.** E2 text macros (`\CSoneUoneMax`, `\CStwoMax`, `\CkTwentyTwoSone`,
  `\CkTwentyTwoMem`) come from the same run as the table cells; `\CSoneUoneMax` is the maximum over
  all S^z rows (new `\CSoneUoneAtMaxDim` for the largest block).
- **MS-N06.** S3 feasibility is stated through nnz/fill: `\CnotFitAtOrBelowMaxDim`, `\CnotFitDim`,
  `\CnotFitNnzMin` and a check that every block of dimension ≤ `\CmaxFeasibleDim` whose S3 did not fit
  has more nonzeros than the largest certified block.
- **MS-N07.** `\TcShiftFraction` names the 0.9 of E3(c) as the shift-placement fraction; the check says so;
  `\TcCasesRecorded` keeps the count of recorded rows (one duplicate) next to `\TcCases`.
- **MS-N13.** `\VaMatricesGenerated` (matrices with and without an Arb reference) next to `\VaMatrices`
  and `\VaNoRef`; `\TbN` gives n for the E3(b) ratios, whose range is now rounded outward.
- Statement check "same computational source" falls back to the records' `source_hash` when the
  recorded commits are not in the repository (rewritten history).

### Added (review of 2026-10-21)
- `tests/test_adversarial.py`: the 14 attack cases of the review (7 failed on the reviewed HEAD).
- `tests/test_review_fixes.py`: one or more regression tests per finding above.

### Added (E9 named sectors, branch feat/e9-named-sectors; changes the computational source hash)
- `e9_pyrochlore.py` step S1N: the lowest eigenvalue of the two pair-spin sectors Kranitz & Penc name,
  S~ = (0,0,0,2,0,2) and (0,2,2,0,2,2), at the printed couplings (S^z = 0 block, ARPACK, floating point),
  recorded under `spin1_printed.named_sectors`; `make_numbers_apps.py` emits `\CrownOneNamedSectorA/B`
  from it (review B3-2) and checks that both reproduce the published value to 1e-5 and lie above the
  certified all-singlet eigenvalue. `tests/test_e9_named_sectors.py`. To be merged after the candidate
  run at source e707838f6852d474 has finished (the step adds about half a minute to E9).

### Added (manuscript v4 macros, branch feat/v4-macros; post-processing only, source hash unchanged)
- `make_numbers.py` / `make_numbers_apps.py` emit every macro the v4 scaffold used from its hand-written
  `generated/numbers_todo.tex`, each with a comment naming the record and field it comes from:
  `\AKLTCritMinDown`, `\AKLTMaxExcessObserved`, `\AKLTExcessCheck` (the check's own threshold, RA-23),
  `\TbRatioMinDown`, `\TbRatioMaxUp`, `\TbPairs`, `\TcCasesUnique` (MS-N13, MS-N07), `\CSoneSzMaxTable`,
  `\CkTwentyTwoSoneTable`, `\CnnzSquareJtwoNotChecked` (MS-N04/05/06), `\NtestsRun`,
  `\NtestsAddedAfterRun` (MS-N02), `\FnPoneOrPtwo`, `\FnBothMissed`, `\FnBothMissedList` and the
  "P1 or P2" column of Table e1 (MS-H02), `\CrownOneRingEzero`, `\CrownOneRingRadiusExact`,
  `\CrownOneRingRadiusFP` (RA-09), `\CrownOneMinClassDim`, `\CrownOneBigClassDim`,
  `\CrownOneBigClassAbove` (MS-N09), `\SourceHashNew` (IMP-26), `\FlintLibVersion` (MS-N12),
  `\ExactCheckerLines` (+ per-file, MS-N11), `\DecSquareNoSymDim` (from the E10 run.log),
  `\SymveigPtwoOutside` (from docs/symveig_audit.md, RA-32), the literature values
  `\CrownHalfReducedPaper`, `\CrownHalfHeadlinePaper` (marked as such, B2-4/RA-21) and, when the record
  carries them, `\MutGapsClosedStamped` (MS-N03) and `\CrownOneNamedSectorA/B` (B3-2).
- `generated/numbers_todo.tex` is now written by `make_numbers_apps.py`: the macros whose value no record of
  the run contains, as `\todo{reason}` (nothing is substituted); empty when every macro has a record.
- `make_numbers.py` checks that no macro is defined twice across the generated files; the environment
  macros that both scripts know are written by `make_numbers_apps.py` with `\providecommand` (before this,
  `\FlintCVersion`, `\PythonFlintVersion` and `\CertifyEdVersion` were defined twice, which would have stopped LaTeX with "command already defined"; found by the duplicate-macro check, no LaTeX build was run).
- `tests/test_make_numbers_helpers.py`: the helpers behind these macros (code-line counter, pytest collection
  parser, duplicate detection, P1-or-P2 verdicts, E10 log and symveig-audit parsers,
  pending-file writer).

### Added (scope framework, branch feat/scope-framework)
- **Universal producer import** (`certify_ed.io`, `docs/producer_bundle.md`): eigenvalues, eigenvectors,
  operators and sector bases from .npy, .npz (dense, `save_npz`, CSR/COO triplets), HDF5, CSV/TSV/text,
  JSON and Matrix Market; matrix-free operators from a Python file (opt-in `--allow-code`); the
  *producer bundle* directory convention (`eigenvalues.*`, `eigenvectors.*`, `operator.*`, `basis.*`,
  `spec.json`, `meta.json`) so that any code is certified by writing files; `claim_from_file`,
  `claim_from_bundle`, `write_bundle`; non-finite input is rejected at the door (review IMP-21).
  `certify-ed certify --from DIR` / `certify-ed import DIR`.
- **Statement selection and presets** (`presets.py`, `docs/presets.md`): `Options.preset / only / skip /
  narrowing_reason / budget_seconds / budget_mb`; presets `lab-minimal`, `benchmark`, `proof-lemma`,
  `whole-spectrum`, `labels`, `full` as data; `certify-ed presets`. Budgets turn statements into
  NOT_CHECKED (reason `BUDGET_TIME` / `BUDGET_MEMORY`), never into exceptions.
- **Scope block** (`scope.py`, `docs/SCOPE_MATRIX.md`): every certificate (schema 2.1) carries `scope`
  with the preset, the statements in scope, the narrowed statements with the user's reason, and the
  out-of-scope quantities with reason codes; `Certificate.summary()` prints it.
- **New statements, all UNVERIFIED (D-12) with proofs written before the code (D-46, `docs/theory.md` §9):**
  `S7.thermodynamics` (Prop 16: interval bounds on Z, F, U, C from a complete whole-spectrum enclosure,
  dyadic interval arithmetic with enclosed exp/ln, `ivl.py`, `thermo.py`); S4 extensions (Prop 15:
  batches of observables, two-point functions, static structure factors, `correlations.py`,
  `Options.correlations`); `S8.degeneracy` (Prop 14: number of eigenvalues in each cluster by bracketing
  inertia counts, `degeneracy.py`). `spectrum.intervals_from_enclosure` / `merge_intervals` expose the
  Prop 9 interval list, recorded as `spectrum_enclosure` for whole-spectrum claims.
- Tests: 58 (import round trips), 18 (presets/scope), 8 (S7), 8 (S4 extensions), 8 (S8).

### Fixed
- **Exact 'k lowest' re-check (S6, `exact_recheck_lowest`, `exact_recheck_sector_lowest`).** A component
  of overlapping residual intervals with m ≥ 2 claimed values was taken to hold m eigenvalues
  whenever one count above it agreed. Residual intervals give *an* eigenvalue each, not distinct
  ones: two copies of the first-excited vector, claimed as "the two lowest", were certified although
  the ground state is missing. Such components are now bracketed by counts on both sides, and their
  eigenvalues are certified inside the bracket ("level enclosure"). Single-value components keep
  their residual interval: the counts on the two sides pin the index. Results carry `enclosures`
  and `next_lower_bound`. FAILED now means an unclaimed eigenvalue lies under a claimed one; a cut
  multiplet is INCONCLUSIVE (it was FAILED). `checker.certify_by_counts` is shared by both routines.
- **Outward rounding of reported endpoints.** μ − r, μ + r, σ ∓ β, separations and cut comparisons
  are formed with `fp.sub_down` / `fp.add_up` (one ulp outward). Round-to-nearest could place an
  endpoint half an ulp inside the exact interval. This affects enclosures and their records,
  cluster merging, S3 margins and gaps, Kato–Temple, S4 separations, sector assembly and
  whole-spectrum intervals.
- `inertia_exact` checks that the permuted LDLᵀ factor is exactly unit lower triangular.

### Changed
- AKLT: thresholds are placed just above the floating-point η² (margins 1e-9, 1e-7, 1e-5). The
  certified η now exceeds the published values by 2e-11 to 2e-9 (it was about 2e-6). On the cross
  and the decorated square, the remaining excess comes from the factorization term t′ − t (β/μ, β/ν).
  Reported η and the square-octagon sum are floats rounded upward from the exact rationals.
- Pyrochlore S = 1/2 crown: a ladder of shifts (relative margins 1e-10, 1e-9, 2e-8). The certified bound
  is λ_min ≥ −0.550158299 (it was −0.550158314), within 1e-9 of the Lanczos value.
- Provenance: the post-processing scripts left the computational fingerprint; the test suite has its
  own `tests_hash`, and `--resume` re-runs E6 when the tests changed. E6 is classified
  automatically after it runs.
- `make_numbers_apps.py`: E6 table, E8 width-scaling figure (`fig_widths.pdf`), criterion values for
  AKLT, run-manifest table, tighter statement checks.
- Tests: 370 before the review fixes; 7 of them close the 8 test gaps found by the mutation analysis of
  the run of 2026-10-21: 2×2 pivots with det > 0, the Dekker splitter, exact residuals with huge entries,
  the round-up probe, the SU(2) check in mixed Pauli letters, the sector compression diagnostic and the
  TFIM bonds. The review fixes add `test_adversarial.py` (14) and `test_review_fixes.py` (see the
  README for the current count).

## [3.0.0] - 2026-10-18

One package for the combined CERTIFY-ED / symveig paper.

### Added
- **Exact re-check layer** (no floating-point assumption), for every local space:
  - `checker.exact_matrix`: the declared model in the scaled basis D⁻¹HD with integer weights;
    spin-S binom(2S, k), bosons k!.
  - `exact_eigenpair_radii`, `exact_recheck_lowest`.
  - `exact_sectors.exact_recheck_sector_lowest` and `exact_count_below`: symmetry sectors with
    Gaussian characters, non-abelian groups allowed when the irreps met are one-dimensional.
  - `checker.inertia_exact`: lean exact LDLᵀ check.
- **S6.exact_recheck** statement (`Options(exact_recheck=True)`).
- **`certify_ed.spectrum.enclose_spectrum`**: whole-spectrum enclosure by sectors (formerly
  symveig). Counts are certified by the cluster theorem, conservation and Hermiticity are checked
  exactly, and rounding bounds use |A||x|.
- **`certify_ed.compat.symveig`**: the symveig 1.0.0 API on the new engine, with symveig's tests ported.
- **`certify_ed.apps.aklt`**: certified Pomata–Wei criteria for five lattices, one pipeline;
  kernel-lemma checks.
- **`certify_ed.apps.pyrochlore`**: exact S = 1/2 crown bound (now through `exact_count_below`);
  S = 1 printed-coupling check; pair-spin landscape.
- **CLI** `certify-ed` (`env`, `certify`, `spectrum`, `list`, `test`, `reproduce`, `numbers`) and
  `run.py`.
- Experiments E7 (exact benchmarks), E8 (symveig comparison), E9 (pyrochlore), E10 (AKLT).
  `make_numbers_apps.py` reads their records.
- `fp.probe_float_environment` is recorded in every experiment's environment.
- Run-time float probe, total-spin and momentum label certification, manufactured and metamorphic
  tests, mutation analysis E6 with recheck and inspection tooling (2026-10-17).
- **Provenance** (`certify_ed.provenance`): every experiment record and every reproduction step
  carries a fingerprint of the computational source (`certify_ed/` and `experiments/` without the
  command line, test helpers and this module). Both numbers scripts check that all records they read
  share one fingerprint, with no uncommitted changes.
- `certify-ed reproduce` / `run.py`:
  - `--jobs N` runs experiments in parallel, longest first; E2, E6 and E8 always run alone;
  - `--resume` keeps the steps of an interrupted run that finished with the same source;
  - unknown experiment names are rejected.
- `apps.pyrochlore.spin_one.printed_coupling_check` also re-derives the six-ring energy in exact
  arithmetic (scaled spin-1 basis). The S = 1 contradiction now needs no floating-point assumption.
- Tests (356) for every AKLT lattice geometry, the spin-1 pair-spin tools, the sector-decomposition
  audits (coverage, orthogonality, trace moments, certified spectral shift) and the driver
  (parallel jobs, resume, numbers on partial runs), and for six gaps found by mutation analysis of the
  final tree (internal builder-disagreement path, 1-D eigenvectors, dense non-Hermitian producers,
  fermionic translation sign, float-model caveat, square-lattice default).
- Tooling: `ruff check .` is clean (configuration in `pyproject.toml`). CI runs lint, builds the
  wheel, runs the suite on Python 3.10–3.12 and does a quick reproduction. SPDX license metadata.

### Changed
- Package metadata in `pyproject.toml` (`setup.py` removed). Version 3.0.0. Authors: Sarang Vehale,
  Ritu Goel.
- Tests moved from `tests/v2/` to `tests/`. Helpers moved to `certify_ed.testing`.
- `Certificate` is the v2 certificate class (`CertificateV2` is kept as an alias).
- Experiment results can be directed with `CERTIFY_RESULTS`.

### Removed (moved to `legacy/`)
- v1 modules (`hamiltonian`, `oracles`, `certificates`, `observables`, `symmetries`), `validators/`,
  v1 tests and examples. Tag `v1.0.0-baseline`.
- symveig 1.0.0 (kept unchanged in `legacy/symveig_1_0_0` for E8).
- The script versions of the applications and their certificate files (2026-10-17).

### Fixed
- The numbers step compared whole commits, so a run that crossed a commit touching only the command
  line failed its provenance check. It now compares computational sources.
- The E2 check "the N=22 momentum block has the largest S1 time" compared runs instead of blocks. It
  failed when the same block's run with S3 took longer than its run without.
- Mutation workspaces now include `examples/`, which the test suite runs.

### Fixed (found while merging symveig; see docs/symveig_audit.md)
- Merged-interval multiplicities are reported as certified counts only when proved.
- Sector decompositions are rejected when the quantity is not conserved.

## [2.0.0.dev0] - 2026-10-04
v2 verification core S1–S5, sector assembly, importers, certificate v2, experiments E0–E5.

## [1.0.0] - 2026-05
CERTIFY-ED v1 (arXiv:2605.11787), now in `legacy/v1`.
