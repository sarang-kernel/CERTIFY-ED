# File importers for other ED codes: HPhi, DiagHam, SPINPACK, ALPS

`certify_ed.claim_from_hphi`, `claim_from_diagham`, `claim_from_spinpack` and `claim_from_alps` read the output
files of these codes into a `Claim`. They live in `certify_ed/io/{hphi,diagham,spinpack,alps}.py`; each
module's docstring states which parts of the layout come from documentation and which are assumptions, and
every reader checks the file against the layout (header against file size, column counts, state numbering) and
raises `certify_ed.io.FormatError` naming the observed and the expected quantity when they disagree.

What none of the readers knows is the order of the code's basis states relative to a certify-ed `ModelSpec`
encoding (digit k of site i, site 0 most significant). The vectors are returned in the code's order; to run
S1 you supply the code's matrix in the same order (`operator=`) and, for a U(1) block, the state list in the
spec's encoding (`states=`). The ALPS reader is the exception: when the file carries the basis table written by
`SAVE_EIGENVECTORS=1`, the vectors are reordered into the spec encoding and the Claim gets the block's
`StateSet`. Without an operator, S2-S4 run against the spec's own matrix only when the vectors are in the spec
encoding; otherwise pass the operator.

| code | files | layout source | reader |
|---|---|---|---|
| HPhi | `##_energy.dat` (Lanczos/LOBCG keyword blocks), `Eigenvalue.dat` (full diagonalization), `##_eigenvec_&&_rank_$$.dat` (binary) | HPhi manual 3.5, file specification: energy.dat, Eigenvalue.dat, eigenvec.dat (the C `fwrite` sequence: `int` iterations, `unsigned long` local_size, `complex double` x (local_size + 1), first component unused) | `read_hphi_energy`, `read_hphi_eigenvalue_file`, `read_hphi_eigenvec`, `claim_from_hphi(energy, eigenvectors="output/zvo_eigenvec_{state}_rank_{rank}.dat", operator=, states=)` |
| DiagHam | `*.vec` (binary), spectrum `*.dat` (text) | DiagHam wiki "Binary vectors": little endian, `int` dimension then `double` x dimension (real, < 2^31 components). Complex vectors: ASSUMED Re, Im interleaved after the same header (by analogy with the documented binary matrix format). Spectrum columns: quantum numbers first, energy last, `#` header: observed convention, not a manual statement | `read_diagham_vector`, `read_diagham_spectrum`, `claim_from_diagham(spectrum, vectors=[...], select={"Sz": 0}, operator=, states=)` |
| SPINPACK | energy text output, `daten.def` | ASSUMED (no file specification was available): `layout="table"` (energy in a text column, other columns are sector labels) or `layout="keyvalue"` (`E=<value>` tokens); `daten.def` parsed as key/value text for provenance only. Eigenvectors are not read | `read_spinpack_energies`, `read_spinpack_def`, `claim_from_spinpack(energies, layout=, select=, definition=)` |
| ALPS | `*.task*.out.h5` | pyalps layout `/spectrum/sectors/<n>/energies` and `quantumnumbers/<name>`; eigenvectors and basis tables (`eigenvectors/{shape,values}`, `basis/{shape,values}`) from the `SAVE_EIGENVECTORS` addition (ALPSim/ALPS PR 168) | `read_alps_spectrum`, `states_from_alps_basis`, `claim_from_alps(path, sector=0, spec=, operator=)` |

The Claim's `producer` record names the code, the files, the selected sector or lines, and for SPINPACK the
layout assumption (`layout_status`), so the certificate says what the import rested on.

## HPhi

```python
import certify_ed as ce
spec = ce.heisenberg_chain(16)                      # the model HPhi was given (standard mode: model = "Spin", L = 16)
claim = ce.claim_from_hphi("output/zvo_energy.dat", "output/zvo_eigenvec_{state}_rank_{rank}.dat",
                           operator=H_hphi, states=states, kind="lowest")
cert = ce.certify(claim, spec)
```

`H_hphi` is HPhi's matrix in HPhi's state order (for instance from `OutputHam=1`, read with
`certify_ed.io.read_operator` as Matrix Market) and `states` the `StateSet` of the block in the spec's encoding.
With MPI output the rank files of one state concatenate in rank order; `ranks=` pins their number. `n_states=`
keeps the first n of the `exct` states. The energies of the Lanczos / LOBCG output are the states HPhi
converged, in HPhi's order (`kind="lowest"` claims they are the lowest; S3 checks that).

## DiagHam

```python
claim = ce.claim_from_diagham("spin_1_2_chain_n_6.dat", vectors=[f"spin_1_2_chain_n_6_sz_0.{i}.vec" for i in range(3)],
                              select={"Sz": 0}, n_states=3, operator=H_diagham, states=states)
```

`select` keys are column names from the `#` header line or 0-based column indices; the energy is the last
column. One `.vec` file per selected eigenvalue, in order. `complex=True/False` forces the interpretation of a
vector file whose size would be ambiguous (it never is for dimension >= 1; the option exists for the record).

## SPINPACK

```python
claim = ce.claim_from_spinpack("ev.dat", layout="table", select={0: 0}, definition="daten.def", kind="subset")
```

The default `kind` is `subset` because the assumed layout does not say that the file lists the lowest states in
order. The certificate's `claim.producer.layout_status` records that the layout is assumed. When a real SPINPACK
run is at hand, check the two layouts against it; the energy file can always be converted by one line into a
producer bundle (`docs/producer_bundle.md`) if neither matches.

## ALPS

```python
claim = ce.claim_from_alps("parm.task1.out.h5", sector=0, spec=spec)      # SAVE_EIGENVECTORS=1 output
cert = ce.certify(claim, spec)
```

The sector index is the position under `/spectrum/sectors`; `claim.producer.all_sector_quantumnumbers` lists
the quantum numbers of every sector so the right one can be chosen. `states_from_alps_basis` maps the basis
table assuming ALPS's local-state index equals the spec digit (spin-S sites listed Sz = -S .. S; occupation
numbers for bosons and fermions). Files without the basis table need `operator=` and `states=` as for HPhi.

## Tests

`tests/test_io_ed_codes.py` writes each format with the generators in `tests/fixtures/ed_codes.py` (which follow
the layouts above, including the assumed ones), round-trips a 6-site Heisenberg chain through `certify()` with
S1, S2, S3 CERTIFIED, checks the rejection of truncated and inconsistent files, and carries negative controls
(a planted energy or a swapped vector written into the producer's files is not certified). A test passing here
shows that reader and generator agree on the documented layout; it is not a test against the codes' own output.
