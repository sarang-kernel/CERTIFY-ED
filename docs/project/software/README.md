# The software documentation

These documents describe certify-ed, the software that issues certificates for exact-diagonalization results, as it
stands in the code repository `code/certify-ed/` of the project archive v5 (branch `dev/3.2` at `a814d28`; the
computational source is that of tag `v3.2.1` = `f3c478b`). The cited version of the manuscript is still 3.1.0
(decision D-61); where 3.1.0 differs from 3.2.1 in a way that matters for a record, the documents say so. Nothing in
this folder changes the code repository.

| Document | Contents |
|---|---|
| `01_architecture.md` | what the software takes and returns, the statements S1 to S8, the statuses, the hypotheses and the not-checked list, the data flow, what 3.2 added, the applications |
| `02_modules.md` | the role of every module of `certify_ed`, grouped by layer, and whether it is part of the computational source hash |
| `03_cli_and_presets.md` | every subcommand of `certify-ed` with its options, the presets, the selection rules, the budgets and the reason codes |
| `04_api.md` | the Python interface: specifying a model, certifying a claim, reading a certificate, the exact layer, whole spectra, producers, planning |
| `05_provenance.md` | the source fingerprint, the checks built on it, the lessons of the evidence runs, and the hashes of the tagged versions recomputed from the commits |
| `06_formats_and_schema.md` | the certificate body and its side file, the integrity hashes, the JSON Schemas and their stability promise, the producer bundle |
| `07_importers_and_producers.md` | every route from a producer's output into a claim, the record hook, the conformance check |
| `08_tests_and_mutation.md` | the test suite and its growth, mutation analysis (E6), the planted-error corpus |
| `09_verifier.md` | the independent C verifier: what it recomputes, what it does not, how to build it, how it was validated |

A reader new to the project can take them in order. A reader who wants to run the software can start with
`03_cli_and_presets.md` and `04_api.md`, and then follow `docs/reproduction/`. `docs/CODE_INDEX.md` lists every
program file of the code repository with what it does, the files it reads and writes where the code names them, and
the statements or experiments its docstring names.

The vocabulary of the documents is as follows. The producer is the ED code whose output is checked. A claim is that
output together with what it asserts (which eigenvalues, and whether they are complete). The declared model, or spec,
is the Hamiltonian the producer was meant to diagonalize, written independently of the producer. A certificate is the
record of statements about the claim and the declared model, each with a status. The model shift is the certified
distance between the producer's matrix and the matrix of the declared model, which every radius includes. A sector is
a block of the Hamiltonian fixed by a symmetry quantum number. The computational source hash is the fingerprint of the
code that can change a recorded number. An evidence run is a full run of every experiment at one computational source,
as decision D-44 requires for a cited version.
