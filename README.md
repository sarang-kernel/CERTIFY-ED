# CERTIFY-ED

**Certificates for exact-diagonalization results: rigorous enclosures, completeness, labels and
exact re-checks, with applications to published proofs.**

CERTIFY-ED checks a result produced by any ED code (QuSpin, XDiag, HΦ, your own) against a declared
model. It returns a *certificate*: a machine-readable list of statements, each marked `CERTIFIED`,
`FAILED` (with a diagnosis), `INCONCLUSIVE` (with the reason) or `NOT_CHECKED` (with the reason). It
never calls your code. Since 3.0 it also contains the whole-spectrum enclosure of the former
**symveig** package, an **exact re-check layer** that uses no floating-point assumption, and two
**applications** to numerical lemmas in published rigorous proofs.

| Statement | What it says when CERTIFIED |
|---|---|
| **S1** model fidelity | your matrix (or symmetry block) is the matrix of the declared model, built independently; on failure, a least-squares fit names the terms whose coefficients differ |
| **S2** enclosures | each claimed eigenvalue lies within a stated radius of a *distinct* eigenvalue of the declared model |
| **S3** completeness | no eigenvalue below the top claimed one is missing; certified lower bound on the next one |
| **S4** eigenspaces | for degenerate levels: projector error, basis-independent observables tr(PO), certified quantum numbers (total spin, momentum) |
| **S5** corroboration | (optional) independent solvers reproduce the values; their shared components are recorded |
| **S6** exact re-check | (optional) S2 and S3 of a 'lowest' claim re-derived in integer arithmetic, with no floating-point assumption |
| **S7** thermodynamics | (whole-spectrum claims, `--temperatures`) interval bounds on Z, F, U and C from the complete enclosure; UNVERIFIED (D-12), see `docs/theory.md` §9 |
| **S8** degeneracy | number of eigenvalues in each cluster, certified by bracketing inertia counts; UNVERIFIED (D-12) |

S1–S5 are rigorous under the IEEE 754 standard model. A run-time probe checks that model on every
run (rounding mode, no flush-to-zero, full-precision BLAS), and certificates are downgraded when it
fails. S6 and the `exact_*` functions do not rely on that model at all: floating point only
*proposes* factorizations, and integer arithmetic (FLINT) checks them.

## Install

```bash
pip install -e .            # numpy, scipy: S1-S5, whole-spectrum enclosure
pip install -e .[exact]     # python-flint: S6, exact re-checks, Arb references, applications
pip install -e .[all]       # + QuSpin, h5py, matplotlib, pytest
pip install -e .[schema]    # jsonschema: validate certificates against schema/certificate-2.1.schema.json
```

Tested with Python 3.11, NumPy 2.4, SciPy 1.17, python-flint 0.9, QuSpin 1.0.1.

## Use

**Certify files written by any code (the `dd if=` route)**

Write a directory with `eigenvalues.*`, optionally `eigenvectors.*`, `operator.*`, `basis.*`, `spec.json`
and `meta.json` in any of npy / npz / HDF5 / CSV / text / JSON / Matrix Market (`docs/producer_bundle.md`), then

```bash
certify-ed certify --from my_run                       # everything that fits the inputs
certify-ed certify --from my_run --preset lab-minimal  # S1 + S2 only; `certify-ed presets` lists the presets
certify-ed certify --from my_run --only S1,S2,S3 --budget-seconds 600 --reason "vectors of excited states not stored"
```

Every certificate carries a `scope` block: the statements in scope, the ones narrowed (with the reason) and
the quantities out of scope with a reason code (`docs/presets.md`, `docs/SCOPE_MATRIX.md`).


**Certify a QuSpin result**

```python
import numpy as np, certify_ed as ce
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

L = 16
spec = ce.heisenberg_chain(L, J2=0.3)                 # the model you intend, written independently
static = spec.to_quspin_static()
basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pauli=0)
H = hamiltonian(static, [], basis=basis, dtype=np.float64)
E, V = H.eigsh(k=6, which="SA")

claim, _ = ce.claim_from_quspin(H, basis, static, E, V, kind="lowest")
cert = ce.certify(claim, spec)
print(cert.summary())                       # one sentence; cert.report() is the full text
cert.save("result.json")                              # + result.arrays.npz, hashed
```

**One call from arrays** (any producer: a matrix `H`, eigenvalues `w`, eigenvectors `V`)

```python
cert = ce.quick(H, w[:4], V[:, :4], spec)            # kind="lowest"; preset=, budget_seconds="auto", progress=True, ...
cert.summary()                                        # one sentence: what was certified and what was not
cert                                                  # a table in Jupyter; every open statement carries evidence["next_action"]
print(cert.methods_paragraph("latex"))                # methods paragraph, numbers as \newcommand macros, BibTeX
```

`certify-ed statement result.json` prints the same paragraph from a saved certificate;
`certify-ed view result.json` writes `result.html`, a self-contained page (no scripts, no network) with every
statement in plain words, the scope block, the enclosure table and the integrity report (`docs/viewer.md`).
`corpus/` is a planted-error corpus (60 cases, 14 error classes) with a runner that scores any checker
(`corpus/README.md`).

A reader re-checks the file with `ce.Certificate.load("result.json", verify=True)`. Other producers
export to NPZ/HDF5 (`ce.export_npz`, `ce.claim_from_npz`, `ce.claim_from_hdf5`). A matrix-free
producer passes a `LinearOperator`.

**Re-check with no floating-point assumption** (any local space: spin-S, fermions, bosons)

```python
cert = ce.certify(claim, spec, ce.Options(exact_recheck=True))      # adds S6
r = ce.exact_recheck_lowest(spec, E[:k], V[:, :k], states)            # dense blocks up to ~4000
r = ce.exact_recheck_sector_lowest(spec, states, [ce.translation(spec), ce.reflection(spec),
                                   ce.spin_flip(spec)], {0: 0}, E[:k], X)   # momentum k = 0, any size that blocks
```

**Enclose a whole spectrum** (the former symveig; counts certified, conservation checked exactly)

```python
enc = ce.enclose_spectrum(H, S_diag=Sz)              # sector by sector; never diagonalises H as a whole
enc.complete, enc.max_halfwidth, enc.intervals[0]
from certify_ed.compat.symveig import enclose_global, enclose_sectors   # symveig 1.0.0 API
```

Runnable versions of these snippets are in `examples/`.

**Command line**

```bash
certify-ed env                                  # float environment probe and versions
certify-ed certify claim.npz --spec spec.json --kind lowest --exact
certify-ed spectrum matrix.npz --sectors Sz     # whole-spectrum enclosure
certify-ed list                                 # experiments E0-E10
```

## Applications (`certify_ed.apps`)

* **`apps.pyrochlore`**: Kránitz & Penc's Anderson-type bound for the pyrochlore Heisenberg
  antiferromagnet (arXiv:2601.07800).
  - S = 1/2: λ_min of the 18-site crown cluster is ≥ −0.550158299 (the exact rational bound is
    −0.55015829882…), certified in exact arithmetic; the cluster's λ_min itself is ≈ −0.5501582979.
  - S = 1: at the printed couplings, the cluster has an eigenvalue ≤ −1.6380246, below the published
    bound −1.632985. This is certified.
* **`apps.aklt`**: Pomata & Wei's spectral-gap criteria for 2D AKLT models (arXiv:1911.01410). η is
  bounded in exact arithmetic for the star, square-octagon, cross, decorated-honeycomb and
  decorated-square lattices, and all five criteria hold. The kernel lemma (ground space = VBS
  states) is confirmed exactly where the size allows.

See `docs/applications.md`.

## Reproduce the paper

```bash
python run.py              # quick: tests + E0, E1, E3, E7, E8, E9, E10 at reduced size (~12 min; 8 min with --jobs 2)
python run.py --full       # every experiment at paper size (about 2.7 h on one core; ~5 GB peak memory)
python run.py --full --jobs 2   # two at a time: 1 h 46 min on a 2-core machine (about 2.5 GB per job)
python run.py --only E10   # one experiment
python run.py --full --out results/run_full_X --resume   # finish an interrupted run, keeping finished steps
```

Each run writes `results/run_<mode>_<timestamp>/`. It contains `manifest.json` (commit, environment,
float probe, per-step status and time), `SUMMARY.md`, logs, one record per experiment
(`data.json`, `environment.json`, `run.log`) and `paper/` (numbers, tables and checks of every
statement the text makes about them). Every record carries a fingerprint of the code that produced
it, and the numbers step refuses to mix records from different computational sources. See
`docs/reproducibility.md`, which includes how to compare two machines.

## Limits

* **S3 completeness** needs a sparse LDLᵀ factorization (floating-point path) or a dense exact one
  (exact path). On a workstation this reaches blocks of about 10⁴ (floating point) or a few thousand
  per symmetry block (exact). Larger blocks get `NOT_CHECKED`, never an uncertified count.
* **Exact sectors** need one-dimensional characters with values in {±1, ±i}: momenta 0 and π on
  any lattice, all momenta on lattices of side 1, 2 or 4, reflections and spin flip.
* Hermitian, time-independent operator-string Hamiltonians; single process; native importer for
  QuSpin only.
* The proofs in `docs/theory.md` marked UNVERIFIED have not yet been re-derived by a second person.

## Layout

```
certify_ed/
  spec.py, localops.py       model specification, local spaces (exact scaled forms included)
  construct.py, basis.py     two independent reference builders, matrix-free operator, sectors
  enclosure.py, fp.py        S2 bounds, floating-point error model and run-time probe
  inertia.py                 S3 inertia counts with backward-error bounds
  eigenspace.py, sectors.py  S4, sector and label audits;  su2.py total spin
  compose.py                 certified global spectrum from symmetry blocks
  certify.py                 certify(claim, spec, options): S1-S6
  certificate_v2.py          certificate format (canonical JSON + hashed NPZ, HDF5)
  importers.py               QuSpin, NPZ, HDF5
  spectrum.py                whole-spectrum enclosure (former symveig)
  checker.py, exact_sectors.py   exact re-check layer (FLINT integers)
  corroborate.py, rigorous.py    S5 solvers; Arb references
  apps/aklt, apps/pyrochlore     applications to published proofs
  compat/symveig.py          symveig 1.0.0 API
  cli.py, provenance.py      command line and reproduction driver; source fingerprints
  testing.py                 shared test helpers
tests/                       416 tests (410 without QuSpin; pytest; `ruff check .` is clean)
examples/                    01-04: QuSpin certificate, exact re-check, whole spectrum, applications
experiments/                 E0-E10, harness, make_numbers*.py
docs/                        theory, reproducibility, applications, symveig audit, API notes
docs/project/, docs/blog/    the project record: documentation of the whole project, the blog, the project changelog
docs/lean/, lean/            the Lean 4 re-verification (reports in docs/lean/, sources and results in lean/)
tools/                       generators and the style check of the project record
schema/                      JSON Schema of the certificate (2.1) and of the producer bundle (1.0); docs/schema.md
conformance/                 reference producer bundles and `certify-ed conform` (conformance/README.md)
legacy/                      CERTIFY-ED v1, symveig 1.0.0, first application scripts (not installed)
run.py                       reproduce everything
```

## Contributing and governance

`CONTRIBUTING.md` (the soundness rule: a negative control per status-affecting change; the mathematics
rule D-12), `GOVERNANCE.md` (maintainers, decisions, release cadence, deprecation policy), `CODE_OF_CONDUCT.md`,
`docs/roadmap/ROADMAP.md`, `docs/SCHEMA_RFC_PROCESS.md`; issue templates include *I could not certify X*, which feeds
`docs/SCOPE_MATRIX.md`. Releases: `packaging/RELEASE_CHECKLIST.md`, `packaging/SIGNING.md`; containers: `containers/`.

## License and citation

MIT (see `LICENSE`). Citation metadata: `CITATION.cff`. Preprints: CERTIFY-ED v1, arXiv:2605.11787;
symveig, arXiv:2606.16217. A combined paper is in preparation.
