# Roadmap

This is the summary of the scored expansion roadmap (`ROADMAP_expansion.md`, v0.1, 2026-10-23, with the
items as a sortable table in `ROADMAP_expansion_items.csv`; both are kept with the project archive). It
lists 78 items in eight themes, each scored on **Adoption** (A, 0-5: how much it removes a reason not to
use the tool) and **Rigor/coverage** (R, 0-5: how much more, or more soundly, a certificate states), with
Score = A + R and an effort class S (days) / M (weeks) / L (months) / XL (a year or more). The scores are
one session's judgment and are meant to be re-weighted by the maintainers; the CSV is the place to do it.

## What adoption depends on

The practice survey (67 papers, provisional coding) finds that the field's evidence of correctness is
comparison with another code (57 %) or with an analytic limit (51 %); 4 % state a residual tolerance and
18 % give nothing; no surveyed paper uses an ED number as a lemma in a rigorous argument. Adoption
therefore depends on four things: **friction** (a certificate from one call, one `pip install`), **trust
that does not depend on the authors** (a published format and a verifier written by someone else),
**coverage of what papers report**, and **norms** (a citable paper, documentation, pilot users, editors).

## Themes

| theme | items | points | what it covers |
|---|---:|---:|---|
| A Friction | 9 | 52 | one-call API, paper-statement generator, PyPI/conda packaging, budgets, streaming vectors, producer hook, HTML viewer, diagnostics |
| B Trust | 10 | 66 | independent verifier, JSON Schema, 3.1.1, trusted computing base, machine-checked proofs, rounding calculus, planted-error corpus, mutation testing, signed releases, external audit |
| C Coverage | 17 | 103 | gaps, vectorless counts, entanglement, correlations through isometries, Hermitian decomposition, truncated thermodynamics, dynamics, time evolution, level statistics, extrapolation reports, crossings, approximate symmetries, fidelity, windows, operator input, exact degeneracy, whole-spectrum default |
| D Scale | 8 | 50 | completeness beyond d ~ 5e4, matrix-free S2 at 1e8-1e9, error-free dot products, parallelism, GPU, larger exact layer, macOS caps, cost model |
| E Mathematics | 7 | 47 | the UNVERIFIED backlog (D-12), non-Hermitian, generalized problems, parameter intervals, variational certificates, underflow policy, sharper clusters |
| F Ecosystem | 9 | 54 | QuSpin upstream, HPhi/DiagHam/SPINPACK/ALPS importers, Julia package, PETSc/SLEPc, Pauli-string specs, TeNPy/ITensor, NetKet, C/Fortran writers, reference bundles |
| G Community | 12 | 70 | public release, software paper, documentation site, tutorials, governance, journal engagement, certificate deposit, reproducibility harness, survey hand-coding, benchmark page, pilot users, scope intake |
| H Engineering | 6 | 28 | CI matrix, property-based tests, typing and semver, performance regression tests, containers, environment capture |

## Waves

**Wave 0, ship what exists (weeks; no new mathematics).** G-01 public release of 3.1.0; B-03 release 3.1.1
with the known issues fixed; E-06 underflow policy decision; A-03 PyPI wheels and conda-forge recipe;
A-01 one-call API; A-02 paper-statement generator; B-02 published JSON Schema with validation at load;
G-03 documentation site; G-05 governance; A-09 explanatory diagnostics; H-01 CI matrix. 66 points.

**Wave 1, cover what the survey says labs report (1-3 months; engineering only).** C-17 whole-spectrum
default path; C-14 first-class interior windows; C-15 operator input from Pauli strings and operator lists;
C-01 gap record; C-02 vectorless window count; C-05 Hermitian decomposition of correlations; C-10
extrapolation report; C-11 certified level crossings; C-09 level statistics; F-01 QuSpin upstream; F-02
HPhi/DiagHam/SPINPACK/ALPS importers; F-05 Pauli-string/OpenFermion specs; A-07 producer hook; A-08
offline HTML viewer; D-08 cost model; A-05 default budget; B-07 planted-error corpus. 99 points.

**Wave 2, trust and scale (3-9 months).** B-01 independent verifier in a second language; F-03 Julia
package; D-01 completeness beyond d ~ 5e4 (new mathematics: verified sparse LDL with an a posteriori
bound); D-02 matrix-free S2 at d ~ 1e8-1e9; D-03 error-free dot products; D-04 parallelism; B-04 trusted
computing base; B-06 rounding calculus; H-02 property-based tests; G-10 benchmark page; G-08 harness as
a package. 76 points.

**Wave 3, new statements (needs D-12 time).** E-01 the UNVERIFIED backlog; C-03 entanglement; C-04
correlations through sector isometries; C-06 truncated-spectrum thermodynamics; E-04 certified parameter
intervals; C-07 dynamical response; E-05 variational certificates; F-06/F-07 tensor-network and NQS
adapters; E-03 generalized eigenproblems; E-02 non-Hermitian; C-12, C-13, C-16, E-07, C-08, D-05, D-06.
120 points. Every item here is UNVERIFIED until a second person re-derives it; this is the schedule's
bottleneck.

**Wave 4, the field (continuous).** G-02 software paper; G-09 survey hand-coding; G-06 journal
engagement; G-11 pilot users; G-04 tutorials; G-07 certificate deposit; G-12 scope intake; B-10 external
audit; B-08 mutation testing as a process; B-09 signed releases; H-03..H-06; D-07; F-04, F-08, F-09;
A-04, A-06; B-05 machine-checked proofs last. 109 points.

## The top items

Best score per unit of effort (small items first): A-01, A-02, B-02 (7 each, S); A-09, B-10, C-02, F-09,
G-01, G-06, G-10 (6, S); C-01, C-05, C-10, C-11, G-05 (5, S).

Largest improvements (score >= 8): B-01 independent verifier (10, L), D-01 completeness at scale (10, L),
C-07 dynamics (9, L), D-02 matrix-free S2 (9, L), E-05 variational certificates (9, L), B-05 machine-checked
proofs (8, XL), B-07 planted-error corpus (8, M), C-03 entanglement (8, M), C-04 correlations through
isometries (8, M), C-17 whole-spectrum default (8, M), E-01 UNVERIFIED backlog (8, L), E-04 parameter
intervals (8, M).

## Status of the Wave 0 items this repository tracks

| item | status | where |
|---|---|---|
| B-02 JSON Schema with validation at load | done (3.2.0-dev) | `schema/`, `certify_ed/schema.py`, `docs/schema.md` |
| F-09 reference bundles and conformance | done (3.2.0-dev) | `conformance/`, `certify-ed conform` |
| A-03 packaging extras, sdist/wheel, conda-forge recipe | done except the PyPI upload and the feedstock | `pyproject.toml`, `packaging/` |
| H-01 CI matrix and nightly mutation sample | written, runs once the repository is public | `.github/workflows/` |
| H-05 containers | written, images not built in CI | `containers/` |
| B-09 signed releases | process documented; distribution hash in certificates deferred | `packaging/SIGNING.md` |
| G-05 governance, G-12 scope intake | done | `CONTRIBUTING.md`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, `docs/SCHEMA_RFC_PROCESS.md`, `.github/ISSUE_TEMPLATE/` |

Other Wave 0 and Wave 1 items are tracked in their own branches; the CHANGELOG records what each release
ships.

## Caveats

Scores are a draft to be re-weighted. Items needing new mathematics cannot be finished by code alone.
Nothing in Waves 1-4 should delay Wave 0. The survey numbers are provisional until the hand coding (G-09)
is finished. The two items that could change practice are B-01 (a verifier the authors do not control)
and G-06/G-11 (editors and pilot users); everything else makes those two credible.
