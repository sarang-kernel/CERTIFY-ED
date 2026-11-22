# Changelog

## [3.3.0.dev0] — unreleased (dev/3.3)

`dev/3.3` starts at `v3.2.2`, merges `fix/3.2.3` (tag `v3.2.3`) and carries everything of [3.2.3] and [3.2.2] below. The integration sets the version to
3.3.0.dev0, lets `parallel.process_memory_cap_mb` read the cap through `cost.process_memory_cap` (the reader of the
3.2.2 memory gate), so that the worker budgets and the gate see the same limit, and widens the version pattern of
`tests/test_review_fixes.py`.

### Small follow-ups

No constant changes; statuses change only on the inputs named here. The computational source hash changes.

- **Non-finite imaginary parts.** A claimed value whose imaginary part is not finite is now a non-finite value: the
  conversion of claimed eigenvalues (`precision.claimed_reals`) makes it nan at every entry point that converts them,
  which are `certify`, `certify_hamiltonian` (QuSpin), `certified_sweep`, `exact_recheck_lowest`,
  `exact_recheck_sector_lowest`, `exact_eigenpair_radii`, `certify_generalized`, `certify_low_energy`, `enclose`,
  `residuals`, the Ozaki residuals, the interval-arithmetic residuals, `corroborate`, the ALPS reader and
  `write_bundle`. The rule for non-finite claimed values then applies: S2 and the exact re-checks are INCONCLUSIVE
  ("claim contains non-finite values"), `exact_eigenpair_radii` gives an infinite radius and lists the index under
  `non_finite_values`, `corroborate` is INCONCLUSIVE, and `enclose`, `certify_generalized`, the ALPS reader and
  `write_bundle` refuse it. Before, the imaginary part was dropped, and w + ∞i was certified as w by `certify`,
  `certify_hamiltonian`, `certified_sweep` and both exact re-checks. An object array of numpy complex scalars is now
  seen as complex: a non-finite imaginary part makes the value nan, and a finite one is recorded (the float conversion
  read the real parts). `invariants.spectrum_invariants` reads claimed values as certify does, so a non-finite imaginary
  part makes `compare_invariants` decide nothing (it compared the real part).
- **Real non-finite claimed values.** `exact_eigenpair_radii` raised a ValueError from the rational conversion for a
  real nan, and `write_bundle` wrote a real nan or infinity (whose readers then refused the file). `corroborate` is
  INCONCLUSIVE for a non-finite claimed value before any recomputation runs (`non_finite_values`; the tolerance is nan);
  for a real infinity it said CORROBORATED, since its tolerance was infinite. `exact_eigenpair_radii` has the key
  `non_finite_values` only when there is such a value, so the S6 evidence of finite claims is as before.
  `docs/schema.md` and the certificate schema name the S5 evidence key `non_finite_values` (a new optional evidence key:
  no version change, `docs/SCHEMA_RFC_PROCESS.md`).
- **Dropped finite imaginary parts.** The conversion is otherwise that of 3.x: a finite nonzero imaginary part is
  dropped (with numpy's ComplexWarning) and the statements concern the real parts, and a list of Python complex numbers
  is refused with numpy's TypeError. New: the claim record of `certify` then holds `dropped_imaginary_parts` (`count`,
  `max_abs`), an optional field (`certify_ed/schema.py`, `docs/schema.md`, the rendered schema), and `not_verified`
  names those parts, so that the report shows them; claims without such parts are recorded as before.
  `certify_hamiltonian`, `certified_sweep` and `claim_from_file` pass values that hold a nonzero imaginary part (a
  complex array, or an object array of numbers) to the claim as given, and other values in their float form, as at main
  (`precision.carries_imaginary_parts`; the QuSpin wrapper sorts them by their real parts, as before; `claim_from_file`
  converted every value to real, with no record), so the record is made there as well. The sector composition writes it
  into the sector's record (`precision.dropped_imaginary_parts`) and, summed over the sectors, into its own record
  (`count`, `max_abs`, `sectors`) and its summary. Where the record is not made: `certify_generalized`, `write_bundle`
  (which holds the real parts only), the ALPS reader and the exact re-checks drop finite imaginary parts as in 3.x,
  without a record (an open item, `docs/project/failures/05_open_items.md`); the section "Dropped imaginary parts:
  records at the remaining entry points" above says which make it now.
- **Sector composition.** A sector with a non-finite claimed value is INCONCLUSIVE before anything is counted, with the
  first indices and their number in its record (`non_finite_eigenvalues`, `n_non_finite`); a nan value was left out of
  the count, so the sector was reported MISSING and the composition FAILED, and w + ∞i was certified.
- **Symmetry entries.** An (approximate) symmetry entry given as an iterator, or as a mapping with keys besides 0, 1 and
  2, now gets the private copy and the widening record of a tuple entry; an iterator entry of `Options.symmetries` was
  read without either, one of `Options.approximate_symmetries` raised a TypeError, and a mapping with further keys was
  passed on as given. An entry of `Options.symmetries` is read by index, so a mapping {0: S, 1: lattice} is accepted in
  binary64 as well (it raised a TypeError; in float32 it was accepted through its converted copy); an iterator is read
  once, when `certify` starts, and an empty one is refused with that reason. A mapping entry without the key 0 or 1, and
  an entry without a length, are refused with the message of a sequence of the wrong length (a KeyError or a TypeError
  before); a malformed entry of `Options.symmetries` is refused when `certify` starts, whether or not S4 reads it
  (before, only when S4 reached the entry), so an entry that main accepted while S4 did not read it is now refused. An
  entry of `Options.approximate_symmetries` that cannot be read as (S, lattice[, allowed_values]) (empty, a mapping
  without the keys its length asks for, None, a number, a string, a bare sparse matrix) is refused with that message
  where S4 reads it (an IndexError, KeyError or TypeError there before); a dense 2-D array entry is read by its rows, as
  in 3.x. certify still passes such an entry on as given, so a run whose statements do not read it certifies as before
  (`tests/test_input_unchanged.py`). An entry that can be iterated but not indexed (a `dict.values()` view, a class with
  `__iter__` only) is read as its items, as 3.x unpacked it. The planner (`cost.narrow_input_mb`) counts a mapping
  entry, a values view and an iterable-only entry as a tuple entry (an iterator is not read there).
- **Operators of the moments and variational checks.** `certified_moments` and `certify_variational` refuse an operator
  or observable with non-finite entries, or with repeated entries that sum beyond the range, with that reason (they
  reported 'not Hermitian' with a nan difference); the check reads the values a sparse matrix holds, in every format,
  not spare capacity past nnz or DIA padding (`precision._stored_values`).
- **Other fixes.** `sectors.restrict_to_u1` drops stored zeros of the isometry before the row look-up (a stored zero in
  a row after the last row of the support raised an IndexError). S8 gives S2's reason code for a cluster whose pairs S2
  did not certify distinct because a radius is not finite: `NONFINITE_BOUND` for couplings of 1e200,
  `UNRESOLVED_VECTOR_NORM` for vectors scaled by 1e-170 or 1e155 (it had no code; its record marks such a cluster
  `radius_not_finite`); a radius that is infinite for another reason (a zero claimed vector) gives no code, as in S2;
  `docs/SCOPE_MATRIX.md` lists S8 under both codes. The figure helper `experiments/figures/_common.py` is renamed
  `_figcommon.py`: `docs/proofs` has its own `_common`, and in one test session the module imported first was used for
  both (an ImportError in `tests/test_nonhermitian.py`, depending on the test order); `tests/test_figures.py` is skipped
  when `experiments/figures` is absent (the source distribution).
- **Tests.** `tests/test_small_followups.py` (each case fails at the base bf567f9f, run with the new helper's import
  replaced by the 3.x conversion, except three that guard behaviour the base already has: no S8 code for a zero claimed
  vector, and two entry forms read as their items; the conversion control fails only on its assertions about the
  helper).
- **Open** (listed in `docs/project/failures/05_open_items.md`): for a claim with eigenvectors and a non-finite claimed
  value, S3, S4 and S8 give `NEEDS_EIGENVECTORS` ('the claim has eigenvalues only'); certificates with a non-finite
  claimed value, or with an `existence-only` enclosure, do not load under the default schema validation (as at main).

### GPU products: the device probe reads the moduli in the stored order

With the summing of repeated entries (the section below), `fp.abs_up` returns the summed matrix (indices sorted) and no
longer sorts the caller's matrix in place. The device probe (`gpu.probe`) checks the ordered kernels on matrices with
unsorted indices and took their moduli from `abs_up`, so the kernels read the moduli in another order than the values:
on the H100 of the GPU re-test of main (`bf567f9f`) the check `kernels_match_scipy` failed and every call with
`device="gpu"` used the CPU products (the device record says `used: false` with that reason; the certificates are those
of the CPU, and 13 tests of `tests/test_gpu.py` failed). The probe now takes the moduli on the stored pattern
(`gpu._moduli_in_stored_order`), and the ordered kernels and `DeviceCSR` refuse moduli on another stored pattern
(`gpu._same_stored_pattern`: shape, indptr and the first nnz indices, O(nnz) on the host when the arrays are distinct
objects, next to a transfer of the same size). Moduli whose index buffer only carries spare capacity past nnz are now
accepted, where the previous check refused them and the call used the CPU products. The device test of the products
compares with the summed matrix that `DeviceCSR` holds, as the CPU products do. On a host whose device passes the probe,
a certificate of `device="gpu"` now records the device as used in the S2 evidence (`device`: `used: true`, with the
device's product count and spot-checked rows) where it recorded `used: false` with the probe's reason; statements,
bounds and the CPU products do not change. The computational source hash changes.

### Peak memory of the planner measurements on Linux

The planner tests measure the peak memory of a factorization in a child process (`experiments/portable.py`,
`peak_rss_mb`). On Linux it read `ru_maxrss`, which after fork (or vfork) and exec starts at the peak resident size of
the parent: inside a full test run the child reported the peak of the pytest process, and three tests of
`tests/test_planner.py` failed on Linux in the full suite only (measured peaks 129 MiB, 0 MiB and 268 MiB against
predictions of 854, 132 and 1373 MiB; they pass alone). On the Linux host of the GPU runs a child of a process holding
300 MiB reported `ru_maxrss` 324 MiB and VmHWM 12 MiB. The reader now takes VmHWM of `/proc/self/status`, the peak
resident set of the process itself (`ru_maxrss` where `/proc` or the VmHWM line is absent, as before; macOS and other
POSIX systems were not assessed for the inherited peak); the three tests pass in the full suite there. A new Linux test
runs the reader in a child of a process holding 300 MiB (it fails with the previous reader). The reader also serves the
peak-memory fields of the experiments' run records: on Linux, a record of a process started by a larger one may now read
lower than with the previous reader, which reported the larger of the two peaks; records of processes started by a small
driver and every Windows record are unchanged. The planner's constants are unchanged: they were calibrated on the
Windows development machine (`RECORDED_PEAKS` of `tests/test_planner.py`, from the C-17 runs of
`experiments/bench_whole.py`, and the factorization peaks measured there for D-08); no recorded Linux figure was
re-measured. No change to the package.

### Dropped imaginary parts: records at the remaining entry points

No constant or status changes; records gain an optional field. The computational source hash changes.

- `certify_generalized` writes `dropped_imaginary_parts` (`count`, `max_abs`) into the claim of its record when the
  claimed eigenvalues are complex with finite nonzero imaginary parts; the statement-record schema leaves `claim` open,
  so the schema is unchanged. The statements and enclosures are those of the real parts, as before.
- `exact_recheck_lowest`, `exact_recheck_sector_lowest`, `exact_eigenpair_radii` and `corroborate` add the same record
  to their result (a key present only then). `certify` passes real values to them (S6, S5), so certificates are
  unchanged; a test pins the S6 evidence of a claim with a dropped part.
- The ALPS reader (`read_alps_spectrum`) gives each sector the keys `energies_as_given` (the stored values when they
  hold a nonzero imaginary part, otherwise `energies`) and `dropped_imaginary_parts` (the record, or None).
  `claim_from_alps` passes the stored values to the claim, so that `certify` records the dropped parts and lists them
  under `not_verified`. Real energies, and complex energies whose imaginary parts are all zero, are read as before.
- Not recorded: the functions that return numbers only (`enclose`, `residuals`, the Ozaki residuals,
  `arb_residual_radii`, `spectrum_invariants`); `certify` passes them real values, and a direct call with complex values
  gets numpy's ComplexWarning, as in 3.x. `write_bundle` still holds the real parts only, without a record; a manifest
  field is a schema decision (`docs/SCHEMA_RFC_PROCESS.md`, `docs/project/failures/05_open_items.md`).
- Tests: `tests/test_dropped_parts_records.py` (13 tests; three need python-flint, four need h5py). One of them lists
  every caller of `precision.claimed_reals` with the place where its dropped parts are recorded, or why it returns
  numbers only, and fails on a caller that is not listed. `tests/abs_audit_allowlist.txt` names the body `_corroborate`,
  where the two moduli of `corroborate` now are.


