# The modules of the certify_ed package

This document gives the role of every module of the package `certify_ed`, grouped by what it does, at `a814d28` of
`code/certify-ed/` (computationally identical to tag `v3.2.1`). The roles are taken from the module docstrings and
from `report/04_software.md`; `docs/CODE_INDEX.md` lists every file with its line count, its definitions and the files
it reads and writes where the code names them, and `code/certify-ed/docs/api.md` lists every public function with its
signature. The last column says whether the module is part of the computational source hash (`05_provenance.md`): a
module marked "no" is listed in `NON_COMPUTATIONAL` in `certify_ed/provenance.py` because it cannot change a status, a
bound or a count.

The package has a layered structure. The model layer (`spec.py`, `localops.py`, `construct.py`, `basis.py`,
`operators.py`) says what the declared model is and builds reference matrices from it. The floating-point layer
(`fp.py`, `enclosure.py`, `inertia.py`, `ldl_residual.py`, `matrixfree.py`, `eigenspace.py`, `sectors.py`,
`compose.py`, `spectrum.py`) computes the bounds under the standard floating-point model. The exact layer
(`checker.py`, `exact_sectors.py`, `exact_degeneracy.py`, `rigorous.py`, `ivl.py`) re-derives or validates results
with no floating-point assumption. The orchestration layer (`certify.py`, `presets.py`, `scope.py`, `cost.py`,
`certificate_v2.py`, `schema.py`) runs the statements and writes the certificate. The record layer (`gaps.py`,
`crossings.py`, `levelstats.py`, `sweep.py`, `thermo.py`, `degeneracy.py`, `correlations.py`, `entanglement.py`,
`davis_kahan.py`, `moments.py`, `variational.py`, `generalized.py`) adds statements and records built on certified
enclosures and counts. The presentation and input layers (`report.py`, `diagnostics.py`, `viewer.py`,
`extrapolation.py`, `api.py`, `cli.py`, `importers.py`, `io/`, `integrations/`, `recorder.py`, `conformance.py`)
connect the package to producers and readers. The applications (`apps/`) and the compatibility layer (`compat/`) sit
on top.

## Model layer

| Module | Role | Serves | In source hash |
|---|---|---|---|
| `spec.py` | declarative `ModelSpec`: local space, operator strings in QuSpin's `static` format, canonical JSON and SHA-256; factories such as `heisenberg_chain`, `tfim_chain`, `heisenberg_square`, `spinless_fermion_chain`, `bose_hubbard_chain`, `spin1_heisenberg_chain`; `spec_from_quspin` | S1, every statement through the model | yes |
| `localops.py` | local Hilbert spaces (spin S, spinless fermions with a Jordan-Wigner order, bosons with a cutoff) and single-site operators with QuSpin-compatible letters, including the exact scaled forms used by the exact layer | S1, S6 | yes |
| `construct.py` | two independent builders of the matrix of a spec (`build_bitwise` by digit arithmetic, `build_kron` by Kronecker products), `cross_construct` comparing them, the matrix-free `SpecOperator`, the rounding bound of the build, and `hermitian_part` | S1, model shift of S2 to S4 | yes |
| `basis.py` | reference symmetry sectors as explicit isometries: translation, reflection, spin flip, U(1) state sets, `cyclic_isometry`, `sector_matrix` | S1 for sectors, E4, E7 | yes |
| `operators.py` | observables and correlations written as text or operator lists, parsed into sparse operators in the claim space (`parse_operator`); refuses non-Hermitian operators where Theorem 6 needs Hermitian ones | S4 inputs | yes |
| `invariants.py` | spectral invariants computed from the specification alone, without building the matrix: the trace of H and its squared Frobenius norm; the cheapest model-fidelity checks, and the only ones for a producer that exposes neither a matrix nor a matrix-vector product | S1 | yes |
| `su2.py` | structural SU(2) invariance of a specification and the total-spin Casimir on a basis | S4 quantum numbers | yes |

## Floating-point layer

| Module | Role | Serves | In source hash |
|---|---|---|---|
| `fp.py` | the floating-point error model behind every bound: unit roundoff, γ constants, error-free products, the run-time probe of the standard model, directed rounding (`sub_down`, `add_up`), the underflow scan (INT-01) and the additive underflow terms (INT-05) | all floating-point statements | yes |
| `enclosure.py` | eigenvalue enclosures for claimed eigenpairs: residual radii (Theorem 1), the cluster bound with radius min(ρ, ε\*) (Theorem 2 and the lemma of decision D-52), grouping into clusters, Kato-Temple intervals (Theorem 3), outward-rounded endpoints; `OperatorAdapter`, `enclose`, `kato_temple` | S2 | yes |
| `inertia.py` | eigenvalue counts by Sylvester's law of inertia: dense Bunch-Kaufman with an explicit residual, sparse SuperLU in symmetric mode with a backward error β; memory failures become inconclusive results with `resource_failure=True` | S3, S8 | yes |
| `ldl_residual.py` | the a-posteriori backward-error bound of a sparse L D L^H factorization, computed by row blocks; one thread by default since 3.2.1, more through `CERTIFY_ED_RESIDUAL_THREADS` (roadmap item D-01, `docs/proofs/D-01_sparse-residual.md`) | S3 on the sparse path | yes |
| `matrixfree.py` | a matrix-free spec operator with an a-priori rounding count and a norm bound, so that S2 radii can be certified for claims whose matrix is never stored (roadmap item D-02, `docs/proofs/D-02_matrix-free.md`) | S2 at dimensions up to 2^24 | yes |
| `eigenspace.py` | basis-independent certification of (quasi-)degenerate eigenspaces: projector distance, observables tr(PO), quantum numbers (Theorem 6, Proposition 7) | S4 | yes |
| `sectors.py` | audit of a producer's symmetry-sector decomposition (isometry and invariance defects), label audits, and composition of per-sector statements into statements about the full operator | S1 for sectors, S4 labels | yes |
| `compose.py` | certified low-energy spectrum of H assembled from symmetry sectors (`certify_low_energy`, Theorem 8) | sector assembly, E3(d), E4 W1 | yes |
| `spectrum.py` | enclosure of the whole spectrum of a Hermitian matrix, optionally sector by sector, with counts certified by the cluster theorem and exact checks of conservation and Hermiticity (`enclose_spectrum`; replaces the engine of symveig) | whole-spectrum claims, Proposition 9, E8 | yes |
| `corroborate.py` | optional recomputation by algorithmically distinct eigensolvers, including a cyclic Jacobi method, with a declared table of shared components | S5 | yes |

## Exact layer

| Module | Role | Serves | In source hash |
|---|---|---|---|
| `checker.py` | independent checks of stored factorizations in exact integer and ball arithmetic: the scaled-basis exact matrix, exact radii, exact inertia (`inertia_exact`), `certify_by_counts`, `exact_recheck_lowest` | S6, E7, applications | yes |
| `exact_sectors.py` | exact re-check of "lowest eigenvalues in a symmetry sector" claims: orbit-sum bases with characters in {±1, ±i}, `exact_count_below`, `exact_recheck_sector_lowest` | S6 for sectors, E7, E9 | yes |
| `exact_degeneracy.py` | exact eigenvalue counts and multiplicities in a rational bracket with no floating-point assumption (roadmap item C-16) | S8 exact variant | yes |
| `rigorous.py` | reference enclosures in ball arithmetic (Arb, through python-flint) that do not rely on the floating-point model, used to validate the float64 bounds | E0, E3 | yes |
| `ivl.py` | interval arithmetic on dyadic rationals with outward rounding, including exp and ln (Lemma 17) | S7 | yes |

## Orchestration and the certificate

| Module | Role | Serves | In source hash |
|---|---|---|---|
| `certify.py` | `certify(claim, spec, options)`: runs S1 to S8 in order, sets statuses, writes the not-checked list and the certificate body; holds `Claim` and `Options` | every statement | yes |
| `presets.py` | statement selection and the presets (`lab-minimal`, `benchmark`, `proof-lemma`, `whole-spectrum`, `labels`, `full`), dependency closure of `only` and `skip` | scope | yes |
| `scope.py` | the scope block of a certificate, the reason codes (`REASON_CODES`) and the budget guard | scope, budgets | yes |
| `cost.py` | advisory cost model of S1 to S3 against the dimension, fitted to E2 records; the planner; the default residual thread count `residual_threads` | `--budget auto`, `--plan` | yes |
| `certificate_v2.py` | the certificate format: canonical JSON, body hash, array side file with per-array hashes, HDF5 export, `Certificate.load` with integrity and scope checks, summary and report methods | every certificate | yes |
| `schema.py` | JSON Schemas (draft 2020-12) of the certificate body and of the producer bundle, rendered from the code's enumerations, and optional validation | format checks | no |
| `provenance.py` | source and test fingerprints (`source_fingerprint`, `numbers_fingerprint`) and the `NON_COMPUTATIONAL` list | provenance of every record | no |

## Records built on certified results

| Module | Role | Serves | In source hash |
|---|---|---|---|
| `gaps.py` | certified intervals for differences of consecutive eigenvalues (roadmap item C-01) | gap records | yes |
| `crossings.py` | whether level k of certificate A lies below or above that of certificate B (roadmap item C-11) | `certify-ed crossing` | yes |
| `levelstats.py` | interval r-ratios and spacing histograms from an enclosed and counted spectrum (roadmap item C-09) | `certify-ed levelstats` | yes |
| `sweep.py` | certified eigenvalue tubes over a coupling interval from certificates at grid points (roadmap item E-04) | `certify-ed sweep` | yes |
| `thermo.py` | interval bounds on Z, F, U and C from a complete whole-spectrum enclosure (Proposition 16) in dyadic interval arithmetic, and the truncated form for a partial spectrum (roadmap item C-06) | S7 | yes |
| `degeneracy.py` | the number of eigenvalues in each certified cluster by bracketing inertia counts (Proposition 14) | S8 | yes |
| `correlations.py` | batches of certified observables, two-point functions and static structure factors, and their lift from sectors (roadmap item C-04) | S4 extensions | yes |
| `entanglement.py` | entanglement entropy and entanglement spectrum of a certified eigenvector (roadmap item C-03) | S4 extension | yes |
| `davis_kahan.py` | Davis-Kahan sin Θ statements, a residual-based alternative to the Theorem 6 bound for labels, fidelities and clusters (roadmap items C-12, C-13, E-07) | S4 | yes |
| `moments.py` | certified moments and sum rules on a certified eigenspace (roadmap item C-07, first part) | S4 extension | yes |
| `variational.py` | variational certificates for trial states that are not ED eigenvectors: Rayleigh upper bound, existence radius and Temple lower bound with a declared or certified gap (roadmap item E-05) | `certify-ed variational` | yes |
| `generalized.py` | generalized Hermitian eigenproblems A x = λ B x with B positive definite, dense path (roadmap item E-03) | generalized claims | yes |

## Presentation, input and output

| Module | Role | Serves | In source hash |
|---|---|---|---|
| `api.py` | `quick(H, w, V, spec, kind, **options)`: builds the `Claim` and the `Options` and returns the certificate | one-call use | no |
| `cli.py` | the `certify-ed` command line and the reproduction driver (`reproduce`, `numbers`) | `03_cli_and_presets.md` | no |
| `report.py` | methods paragraphs with every number taken from a certificate, in plain text or LaTeX with BibTeX | `certify-ed statement` | no |
| `diagnostics.py` | the `next_action` text of every statement that is not CERTIFIED or FAILED | diagnostics | no |
| `viewer.py` | offline HTML page of a certificate: one self-contained file with no scripts and no network access | `certify-ed view` | no |
| `extrapolation.py` | finite-size extrapolation report from per-size certificates; presentation only, nothing certified | `certify-ed extrapolate` | no |
| `testing.py` | helpers shared by the test suite and the experiments | tests, experiments | no |
| `importers.py` | `claim_from_quspin`, `claim_from_npz`, `claim_from_hdf5`, `export_npz` | input | yes |
| `io/__init__.py` | the universal producer import: file formats and the producer bundle convention | input | yes |
| `io/formats.py` | readers and writers for NPY, NPZ, HDF5, CSV and text, JSON, Matrix Market and the opt-in matrix-free Python file (`read_array`, `read_operator`) | input | yes |
| `io/bundle.py` | the producer bundle: `claim_from_bundle`, `write_bundle` | input | yes |
| `io/hphi.py` | readers for the output files of HΦ (roadmap item F-02) | input | yes |
| `io/diagham.py` | readers for DiagHam output (roadmap item F-02) | input | yes |
| `io/spinpack.py` | readers for SPINPACK text output (roadmap item F-02) | input | yes |
| `io/alps.py` | reader for the HDF5 output of the ALPS diagonalization applications (roadmap item F-02) | input | yes |
| `integrations/__init__.py` | integrations with producer codes and quantum-computing libraries | input | no |
| `integrations/quspin.py` | one call from a QuSpin `hamiltonian` and its eigensolver output to a certificate (roadmap item F-01) | input | no |
| `integrations/pauli.py` | Pauli-string Hamiltonians as model specs: plain strings, OpenFermion `QubitOperator`, Qiskit `SparsePauliOp` (roadmap item F-05) | input | no |
| `recorder.py` | the producer record hook (`record`, `recorded`): captures a solver's output in a Python process and writes a producer bundle unchanged (roadmap item A-07) | input | no |
| `conformance.py` | conformance check of a producer bundle: schema, import, certify, and the schema of the result | `certify-ed conform` | no |
| `__init__.py` | the public namespace of the package and `__version__` | import | yes |

## Applications and compatibility

| Module | Role | Serves | In source hash |
|---|---|---|---|
| `apps/aklt/__init__.py` | certified Pomata-Wei spectral-gap criteria for two-dimensional AKLT models | E10 | yes |
| `apps/aklt/graph.py` | VBS geometry on general graphs in exact integer arithmetic | E10 | yes |
| `apps/aklt/lattices.py` | the unions L, C, R of two overlapping subgraphs for each lattice, with geometry inferred from the paper's text and confirmed by reproducing its η to 7 to 10 digits | E10 | yes |
| `apps/aklt/eta.py` | the exact certificate for η on a union, through the bordered pencil and exact symmetry blocks (decision D-38) | E10 | yes |
| `apps/aklt/kernel.py` | the exact check that the kernel of the subgraph Hamiltonian equals the range of the VBS map (decision D-37) | E10 | yes |
| `apps/pyrochlore/__init__.py` | certified Anderson-type lower bounds for the pyrochlore Heisenberg antiferromagnet from the crown cluster, reduced model with c5 = 0 | E9 | yes |
| `apps/pyrochlore/model.py` | geometry and couplings of the crown cluster (reduced model c3 = c12, c5 = 0) | E9 | yes |
| `apps/pyrochlore/spin_half.py` | the S = 1/2 crown: certified lower bound on the lowest eigenvalue in exact arithmetic | E9 | yes |
| `apps/pyrochlore/spin_one.py` | the S = 1 crown, decomposed by the conserved spin of each outer pair | E9 | yes |
| `compat/__init__.py` | compatibility layers for code written against earlier packages of the project | compatibility | yes |
| `compat/symveig.py` | the symveig 1.0.0 API (`enclose_global`, `enclose_sectors`) on the certify-ed engine, with the corrected behaviour | compatibility | yes |
