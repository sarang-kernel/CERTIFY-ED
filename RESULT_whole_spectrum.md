# RESULT: track/whole-spectrum (roadmap C-17, C-14, D-08, A-06, C-15)

Base: `dev/3.2` = f080586 (Wave 0 merged). Branch `track/whole-spectrum`, head: 07036b1 (this file was added in that commit; the code is e9a1857 + e24f45c).
Environment: certifyed-311 (Python 3.11, numpy 2.4.4 with OpenBLAS 0.3.31 Haswell kernels, scipy 1.17.1,
python-flint 0.9.0, QuSpin 1.0.1), Windows, 8 cores, 16 GB. Commits: e9a1857 (C-17, C-14, C-15, D-08
scaffolding), e24f45c (D-08 calibrated and tested, A-06 documentation, changelog and docs), plus the final
commit of this file.

## Item status

| item | status | one line |
|---|---|---|
| C-17 whole-spectrum path | done | `quick(..., kind="all")` / `--kind all` take the `whole-spectrum` preset; S2 evaluation reorganised without changing a bound; radii bit-identical before/after on every fixture; timings below |
| C-14 interior windows everywhere | done | `kind="window"` validated once (`parse_window`) for Claim, every importer, meta.json, `--window A B`; shift-invert cookbook page executed; tests per importer path with a planted missing eigenvalue |
| D-08 cost model and planner | done | `cost.plan_certification` / `plan_for_claim`, `Options(plan=True)`, `--plan`, `certify-ed plan`; calibrated on the 924/3432/12870 blocks; memory within factor 2 of measured peaks; `auto` statuses identical to the explicit method |
| A-06 streaming vectors | done (with the documented limit) | numpy memmap and h5py datasets processed in column blocks; certificate identical to the in-memory one (tested at 252 with memmap/HDF5 in both layouts, and at 12870 with a memmap on disk: `test_memmap_12870_fixture_identical_to_in_memory`, 811 s, passed); dense operators read the vectors whole (see below) |
| C-15 operator ergonomics | done (C-05 split not available) | Pauli strings, letter_site shorthand, linear combinations, dictionaries, QuSpin operator lists -> sparse operators; non-Hermitian products refused with the Theorem-6 message; tests against Kronecker-built 4-site matrices |

## C-17: what changed and what did not

Bounds unchanged: the residual matrix `H X - X diag(mu)`, its componentwise rounding bound `E`, the Theorem-1
and Theorem-2/2' radii, the Kato-Temple formula (all in `enclosure.py`). What changed is the evaluation:

- `enclosure.residuals` processes columns in blocks (`column_blocks`, never a block of width 1 since numpy
  reduces a one-column array along a contiguous axis with a different summation order). Sparse and spec
  operators are column-block invariant (one ordered sum per entry) and the test suite checks it; **dense BLAS
  gemm is not** (verified on this OpenBLAS: `H @ X[:, a:b]` differs in the last bits from the columns of
  `H @ X` for every block width tried), so a dense operator always takes one block, exactly the pre-change call.
- `E` is reduced to its column norms (`ResidualReport.E_norm`, the quantity the cluster bound uses) and not
  stored; `ResidualReport.E` is a property that recomputes it (two tests use it). `R` is stored for in-memory
  vectors and recomputed per cluster for streamed ones (`residual_columns`, F-ordered like `R[:, idx]`:
  the small gemm `X_g^H R_g` is layout-sensitive, which the HDF5 path exposed).
- The dense `|A|` is no longer held by `OperatorAdapter` (formed inside `abs_matvec`, same gemm operands).
- The exact Gram inner products of a cluster are memoised within one `enclose` call (`_exact_gram_cached`).
- `fp.exact_sum`: cascaded error-free summation (Ogita-Rump-Oishi SumK with Knuth's TwoSum, vectorised through
  `cumsum`) that returns the correctly rounded sum, i.e. `math.fsum`'s value, and falls back to `math.fsum`
  whenever the folded remainder plus every slack is not strictly inside the rounding interval of the result,
  the result is zero, the input is non-finite or has fewer than 4096 terms. 1500 random cases (including
  heavy cancellation, mixed exponents, exact ties) and all Gram products of the fixtures agree with `fsum`
  bit for bit; the test `test_exact_sum_equals_fsum` keeps a sample. D-12: no new bound; the error-free
  transformations are the ones `exact_dot` already relied on (Dekker, Knuth), the cascade is Ogita-Rump-Oishi 2005.
- Kato-Temple: `H x` and `|H||x|` for all isolated pairs of a block in one product each (`certify.py`), passed to
  `rayleigh_interval` / `kato_temple` (new optional `Ax`, `absAx`). For a sparse operator the intervals are
  bit-identical to the per-pair ones (tested); for a dense operator 3 of 44 intervals at 924 differ by
  <= 2e-16 relative (gemm column vs gemv: two valid evaluations of the same bound). Records are sorted by index.
- S4: the per-cluster Python loops over all other pairs (`others`, `clash`, `U_below`, `L_above`) became
  array operations with the same `add_up` / `sub_down` (elementwise, same bits).
- `fp.tiny_nonzero` and `certificate_v2.array_sha256` work in row chunks (same results, no whole copy).

Timings (whole-spectrum preset, temperatures [0.5, 1.0], Heisenberg chain Sz = 0 blocks, `experiments`-style
subprocess per run with `portable.peak_rss_mb`; 12870 sparse runs overlapped with the test suite, the others ran
alone; before = dev/3.2 f080586 worktree, after = this branch at e9a1857):

| dimension | operator | code | total s | S1 s | S2 s | S7 s | Kato-Temple records | peak RSS MB | radii sha256[:16] |
|---|---|---|---|---|---|---|---|---|---|
| 3432 | dense | track | 21.6 | 0.42 | 14.11 | 1.84 | 127 | 699 | 7b132274a2bf5b46 |
| 3432 | dense | dev/3.2 f080586 | 33.9 | 0.42 | 17.76 | 1.87 | 127 | 968 | 7b132274a2bf5b46 |
| 12870 | sparse | track | 397.8 | 0.59 | 365.81 | 7.23 | 0 | 2974 | def5ba165b29da51 |
| 12870 | sparse | dev/3.2 f080586 | 568.2 | 0.61 | 535.61 | 7.06 | 0 | 8905 | def5ba165b29da51 |
| 12870 | dense | track | 635.3 | 4.78 | 584.97 | 7.08 | 251 | 8715 | 01b8df55a12e059a |
| 12870 | dense | dev/3.2 f080586 | 798.0 | 4.42 | 656.87 | 7.96 | 251 | 10059 | 01b8df55a12e059a |

Dimension 924 (dense, in-process): 4.9 s -> 3.3 s total, S2 1.59 s -> 1.17 s (the git provenance call, 1.4 s,
dominates there). The remaining S2 time at 12870 is the two `n^2 k` gemms (4.3 TFlop each, about 40 GFlop/s
here) and the exact Gram inner products of the ~6400 clusters; S7 (7 s) and the per-column norms are small.
The dense 12870 run of the base code needed 10.1 GB on this 16 GB machine and paged.

Bit-identity tests (`tests/test_whole_spectrum.py`): (i) `tests/_enclosure_reference.py` is a verbatim copy of
the pre-change `enclosure.py`; radii, existence radii, residual bounds, computed norms, estimates, group
structure and `E` norms must be equal for dims 70, 252, 924, dense and sparse, with forced narrow blocks, and
for a complex operator; (ii) `tests/fixtures/whole_spectrum_252_{claim,sparse,dense}.npz` hold the pre-change
radii of a fixed claim (vectors stored as float32, eigenvalues dyadic, so the inputs are identical on every
machine); exact on this machine, asserted at 1e-12 with a warning when a different BLAS rounds a 2-norm
differently (stated in the test); (iii) the recorded 924 sparse and dense certificates of the base code equal
the new ones array for array (checked once during development, not a committed test: the 924 inputs come from `eigh`).

Deliberately not done: S8 was not added to the `whole-spectrum` preset (one factorization pair per cluster,
thousands at 12870; `labels` has it), and S8 for `kind="all"` was not re-derived from the Prop-9 spectrum
enclosure (it would make S8 CERTIFIED without inertia where dev/3.2 could say INCONCLUSIVE: soundness rule
as written). `enclose_spectrum` (per-sector S2 by a conserved diagonal) is unchanged.

## C-14

`Claim.__post_init__` validates `kind` (`CLAIM_KINDS`) and `window` through `certify.parse_window` (pair a < b
of finite numbers from a tuple, list, "a,b" or "a b"); `claim_from_arrays/npz/hdf5/file/bundle` pass `window`
through to `Claim`, `io.bundle.read_bundle` uses the same parser for `meta.json`, the CLI `--window` takes two
arguments or one comma-joined (`--window=A,B` when A is negative, an argparse rule) for `certify`, `import` and
`plan`, and overrides the bundle. The certificate records `claim.window` as a list. Cookbook:
`docs/cookbook/shift_invert_window.md` + `snippets/shift_invert_window.py` (open J1-J2 chain, Sz = 0 block of
dimension 924, `eigsh(k=10, sigma=-1, which="LM")`, window edges halfway into the gaps shown by the two
outermost values, S3 CERTIFIED; the same list with a window 0.2 wider: S3 FAILED, 41 unclaimed). Tests
(`tests/test_windows.py`, 10): every importer path and `quick`, CLI two-argument and comma form, `import` with
meta.json window and override, validation errors; negative control on every path: one claimed pair dropped
inside the window -> S3 FAILED with `missing_at_least >= 1`.

## D-08

`certify_ed/cost.py`: `plan_certification(n, nnz, k, kind, dense_operator, ...)` -> `Plan` with one row per
statement (runs?, method, seconds, peak MB, basis); `plan_for_claim(claim, spec, options)` reads the shapes;
`recommend_inertia_method`; `dense_inertia_mb`, `sparse_inertia_mb`, `sparse_fill_estimate`. Memory is counted
array by array; constants measured by `experiments/planner_calibration.py` (run on 2026-10-29, output in the
cost.py header): dense LDL^T peak 9.28 / 8.75 n^2 entries at 924 / 3432 (constant 9.5); SuperLU fill 97156 /
1669784 / 19164106 at 924 / 3432 / 12870 (fit 0.117 n^2.006), working storage 4.2-4.4 x fill x 12 bytes
(constant 4.5); gemm 22 / 41 GFlop/s, sparse H X 0.07 / 0.20 x 10^9 entries/s, exact inner product 110 ns per
term at 3432. `Options(plan=True)` / `--plan` print the plan (stderr, or the progress sink) and record it under
`body["plan"]` (schema property added, `schema/certificate-2.1.schema.json` re-rendered); `certify-ed plan`
prints it without running. `inertia_method="auto"` keeps the 3.1 rule (dense for a dense matrix or a sparse one
of dimension <= 2500) and sends a sparse matrix whose dense workspace exceeds `budget_mb` (or the available
physical memory) to SuperLU; the S3/S8 memory gate now estimates the path that will be taken. Default
`inertia_method` stays `"auto"`. Tests (`tests/test_planner.py`, 9): measured peak in a fresh process for dense
LDL^T at 3432 (786 MB, predicted 853) and SuperLU at 3432 and 12870 (83 / 972 MB; predicted within factor 2,
fill within factor 2); the recorded whole-spectrum peaks (3432 dense, 12870 sparse and dense) within factor 2;
`auto` == explicit method statuses and summaries on the 1024 fixture, and == `sparse` under a 50 MB budget
(lowest and window claims); `plan=True` leaves statements and radii unchanged and the saved certificate
validates; CLI. Time predictions are rough (a factor 2-3 on these blocks) and machine-specific; stated as such.
Negative control of the "never changes a status" rule: the existing budget tests (`budget_mb=0.001` -> S3
NOT_CHECKED with BUDGET_MEMORY) still pass with the new sparse estimate.

## A-06

`certify()` accepts `claim.eigenvectors` as a numpy memmap or an h5py dataset (`enclosure.is_streamed`,
`StreamedColumns` wrapper for HDF5 with numpy-style column indexing); `residuals` reads blocks of
`Options(block_mb=...)` (default `budget_mb`, else 512 MB of four n x b temporaries); cluster and Kato-Temple
columns are read on demand; `nonfinite_columns`, `tiny_nonzero`, `array_sha256` and S6 handle the streamed
object; `store_eigenvectors` reads them in. Identity tests: memmap and HDF5, C and F layouts, sparse and dense
operator, whole-spectrum and `labels` presets, whole certificate signature (arrays, statements, enclosures,
Kato-Temple, spectrum, hash); the 12870 memmap of the fixture (1.3 GB on disk in the workspace, test
`test_memmap_12870_fixture_identical_to_in_memory`, gated by `CERTIFY_ED_LARGE_FIXTURE`): identical, 811 s for
both runs. Negative controls: a duplicated vector and a NaN entry in a memmap claim give the same non-CERTIFIED
S2 as in memory. Documented limits (`docs/cookbook/streaming_vectors.md`, `docs/limitations.md`): a dense
operator reads the vectors whole (gemm not block-invariant; a dense n x n operator is larger than n x k vectors
anyway); whole-spectrum S3 for `kind="all"` needs no factorization, but S3/S8 for `lowest`/`window` still
factorize the stored matrix in memory (dense 9.5 n^2 entries, or SuperLU).

## C-15

`certify_ed/operators.py`: `parse_operator(expr, spec, states, name, hermitian=True)` for Pauli strings
("XIZY", upper-case letters are the Pauli matrices, sigma = 2 S in the spin convention, spin-1/2 only),
compact "X0 Z2", letter_site shorthand ("Sz_0 Sz_3", "S+_0 S-_3", "n_0 n_1", "+_0 -_3"), linear combinations
("0.5 S+_0 S-_1 + 0.5 S-_0 S+_1", "Sz_0 Sz_1 - 0.25 I"), dictionaries ({"letter","sites"} as a product,
{"op","sites","coeff"}) and QuSpin operator lists; all built by `build_bitwise` from the spec's local space on
the claim's state set. `certify()` resolves `Options.observables` and `Options.correlations` (string requests
-> `{"letter", "sites"}`) before any statement; errors are `ValueError` / `OperatorSyntaxError` naming the
token. The C-05 Hermitian split is not in this base, so non-Hermitian products are refused with the existing
`correlations._hermitian_part_or_fail` message; the symmetrised sum is the accepted form. Tests
(`tests/test_operators.py`, 8) against Kronecker-built matrices on 4 sites in the spec's digit convention,
Pauli convention of `tfim_chain`, spin-1 refusal of Pauli letters, end-to-end S4 values from text equal to the
matrix path, refusal inside `certify()`.

## Files touched

certify_ed: enclosure.py, certify.py, fp.py, certificate_v2.py, cost.py, inertia.py, degeneracy.py, cli.py,
api.py, importers.py (unchanged logic; window passes through Claim), io/bundle.py, schema.py, operators.py (new),
__init__.py. schema/certificate-2.1.schema.json (re-rendered). tests: test_whole_spectrum.py, test_windows.py,
test_planner.py, test_operators.py (new), _enclosure_reference.py (frozen module), fixtures/whole_spectrum_252_*.npz,
test_review_fixes.py (monkeypatch signature). experiments/planner_calibration.py (new). docs: cookbook/
shift_invert_window.md, cookbook/streaming_vectors.md, cookbook/snippets/{shift_invert_window,streaming_vectors}.py,
cookbook/snippets/run_all.py, cookbook/index.md, mkdocs.yml, presets.md, limitations.md, api.md, make_api.py,
CHANGELOG.md.

## Tests and lint

Full suite: `pytest -q tests` -> `2 failed, 763 passed, 2 skipped, 1 warning in 2118.06s (0:35:18)` (run while the 12870 memmap test was also running).
Both failures are pre-existing at f080586 and recorded as fixed on dev/3.2 68e26d0 (`test_fix_3_1_1::test_scope_validator_rejects_a_statement_dropped_from_the_scope_block`, `test_schema_conformance::test_unknown_status_and_reason_code_fail_at_load`: two validators guard the same tampering, `Certificate.load` order); the other two of the four pre-existing failures (rendered schema out of date) pass here because the schema was re-rendered for the `plan` property. The 2 skips are the `CERTIFY_ED_LARGE_FIXTURE`-gated 12870 memmap test (run separately: passed, 811 s) and one pre-existing skip.
ruff: `ruff check certify_ed tests experiments conformance` -> All checks passed!

## Expected merge conflicts with the other Wave-1 tracks

- Statements track: `certify.py` (S3 window path: this track only adds `budget_mb=` to the `inertia_count` calls in
  `_certify_lowest` / `_certify_window`, and the S4 vectorised loops; the C-15 resolution block after the budget;
  the plan record and `"plan"` body key). `correlations.py` untouched here. If C-05 lands, `operators.
  resolve_correlation_request` should route `+-` requests to the split instead of refusing.
- Producers track: `importers.py` is unchanged by this track (window validation lives in `Claim`), `io/bundle.py`
  has a three-line change (parse_window) in `read_bundle`; `cli.py` gained `_claim_from_args`, `cmd_plan`, the
  `--plan`, `--block-mb` and `--window` changes (textual conflicts likely if that track edits the same flags).
- Viewer and corpus: no shared files except `CHANGELOG.md` and `mkdocs.yml` nav (additive).
- Lead's dev/3.2 68e26d0 (schema re-render, `Certificate.load` schema-before-hash): `schema/certificate-2.1.schema.json`
  conflicts (resolution: re-render with `python -m certify_ed.schema --write schema`); `certificate_v2.py` touches
  different functions (`array_sha256` here).

## Deviations

- The timings at 12870 with the sparse operator overlapped with other work on the machine (test runs); the dense
  12870 and the 3432 runs ran alone. The 924 timing is in-process, not from the benchmark driver (its fixture
  lacked a key during the driver run).
- The planner's time model is calibrated on the three Heisenberg blocks of this machine only; stated as advisory.
- The recorded-radii test asserts bit-identity only on this machine's BLAS (1e-12 elsewhere), by construction.
