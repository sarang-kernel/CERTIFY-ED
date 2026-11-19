# Pre-registration: survey of exact-diagonalization practice in recent papers (v0.1, 2026-10-21; v0.2 changelog appended after the run)

Written before any paper was fetched. Changes after this point go in the changelog at the end.

## Question

What do laboratories that publish exact-diagonalization (ED) results actually report, what evidence of correctness do they give for those numbers, and how are the numbers used? The answer fixes the scope that a certificate tool has to cover (the "wide" default) and the dimensions along which a user narrows it.

## Population and sampling

- Source: arXiv API (`export.arxiv.org/api/query`), 3 s between requests, exponential back-off on HTTP 429.
- Query: `abs:"exact diagonalization" OR abs:"exact diagonalisation" OR abs:"Lanczos" OR abs:"full diagonalization"` restricted to categories `cond-mat.str-el`, `cond-mat.stat-mech`, `cond-mat.quant-gas`, `cond-mat.supr-con`, `quant-ph`, submitted 2024-01-01 to 2026-10-21.
- Full sample: all results returned up to 600 records (metadata and abstracts).
- Coded subsample: 80 papers drawn uniformly at random from the full sample (seed 20261021), restricted to papers for which the LaTeX source or PDF can be fetched. If a draw cannot be fetched, the next random paper replaces it; replacements are counted and reported.
- Exclusions (recorded, not silently dropped): papers where "exact diagonalization" is only mentioned as a comparison method without the authors running it themselves; review articles; papers in which ED is applied to non-Hermitian or classical problems (kept in a separate count).

## Coding fields (one row per paper)

| Field | Values |
|---|---|
| `uses_ed` | yes / comparison-only / no |
| `producer` | named code (QuSpin, XDiag, HPhi, ALPS, DanceQ, ITensor, ...) / in-house / not stated |
| `method` | full diagonalization / Lanczos-Arnoldi / shift-invert / other / not stated |
| `max_dim` | largest Hilbert-space (or block) dimension stated, or `not stated` |
| `sectors` | symmetry sectors used (translation, point group, U(1), SU(2), parity, none, not stated) |
| `quantities` | multi-label: ground energy, gaps, excited levels, degeneracies, quantum numbers/labels, correlation functions, structure factor, order parameter, entanglement entropy/spectrum, fidelity/susceptibility, thermodynamics (full spectrum), dynamical response/spectral function, time evolution, level statistics, finite-size extrapolation, other |
| `evidence` | multi-label: residual or convergence tolerance stated, comparison with another code, comparison with literature value, comparison with analytic limit, symmetry/sum-rule check, none stated |
| `use` | multi-label: benchmark anchor for another method, lemma in a rigorous argument, standalone physics result |
| `eigenvectors_used` | yes / no / unclear |
| `precision_stated` | yes (digits or tolerance) / no |
| `data_available` | raw ED data or code published: yes / no / partial |

## Procedure

1. Fetch and store metadata for the full sample (`survey_sample.json`).
2. For the coded subsample, fetch the LaTeX source (preferred) or PDF text.
3. First pass: one call of an automatic coder per paper with the coding scheme as a fixed schema, on the full text (truncated to the first 60k characters if longer, with the methods and results sections prioritised when a `\section` structure is present).
4. Check reading: 12 papers (15 %) drawn at random with seed 20261022 are re-coded in a second automatic reading of the extracted text (not a human check); per-field agreement is reported. Fields with agreement below 80 % are re-coded for the whole subsample with revised coding instructions, and the revision is recorded.
5. Outputs: `survey_sample.json`, `survey_coded.csv`, `survey_handcheck.csv`, one generated LaTeX table (quantities and evidence frequencies), one figure, `RESULT_practice_survey.md`.

## Stop rules and what each outcome means

- If fewer than 40 papers of the subsample can be fetched and coded, report the survey as a pilot and do not use its percentages in the manuscript.
- Percentages are reported with Wilson 95 % intervals and the subsample size; they describe arXiv preprints 2024-2026 in the listed categories, not "the community".
- No outcome of the survey changes which statements CERTIFY-ED certifies; it changes the default scope and the presets, and it decides which quantities get a NOT CHECKED slot with a reason.

## Known limits

- Abstract-level query misses papers that use ED without naming it in the abstract.
- The automatic first pass can mislabel; the check reading measures this, not corrects it everywhere.
- Full text is used only as far as arXiv provides source or PDF text; supplementary material is not fetched.

## Changelog

- v0.1 (2026-10-21): first version, before any fetch.
- v0.2 (2026-10-21, written after the run; nothing above this line was edited). Deviations and specifications:
  1. Population cap. The query returned 878 records, i.e. more than the pre-registered cap of 600. Instead of
     keeping the 600 most recent (which would have dropped most 2024 papers), all 878 records were stored and
     the 80-paper subsample was drawn from all of them. The cap was a budget guard, not a scientific choice; the population
     is therefore the full set of API matches, sorted by submission date, fetched 9 pages of 100.
  2. Sampling implementation. Subsample order = numpy.random.default_rng(20261021).permutation over the stored record order;
     the first 80 fetchable papers in that order form the coded subsample. 0 fetch failures, 0 replacements.
  3. Source host. export.arxiv.org/e-print answered HTTP 429 persistently (three back-offs of 10, 20, 40 s) and the sandbox
     proxy truncates responses at 1 MiB; sources were fetched from arxiv.org/e-print with HTTP Range requests (3 s between
     requests). The API query itself used export.arxiv.org as pre-registered. All 80 texts came from LaTeX sources; no PDF
     fallback was needed.
  4. Coding schema. The pre-registered fields were kept; two bookkeeping fields were added (paper_type: research/review, and
     problem_class: hermitian-quantum/non-hermitian/classical/other) so that the pre-registered exclusions could be recorded
     per paper, plus max_sites, quotes, confidence and notes as free text. `sectors` was coded as multi-label (the
     pre-registration lists sector types but does not say single-label).
  5. Check reading. Drawn as random.Random(20261022).sample(range(80), 12) over the subsample in draw order. Coding was done
     in a second automatic reading of keyword-focused excerpts of the extracted text plus targeted full-text searches, not a
     human check and not a complete linear reading. Agreement on multi-label fields is reported as the Jaccard index between label sets
     (label-wise presence/absence agreement is given alongside).
  6. Re-coding. Pass-1 agreement was below 80 % on 10 of 13 fields. As pre-registered, those fields
     were re-coded for the whole subsample with revised coding instructions (coding_scheme2.py: every positive label must carry a
     verbatim quote; labels without a quote are dropped; tightened definitions for sectors, evidence, producer, precision).
     uses_ed, paper_type, max_dim and max_sites were kept from pass 1. Two papers (2601.03902v2, 2408.00576v2) keep pass-1
     values on the re-coded fields because the coding budget was exhausted before their pass-2 call.
  7. After re-coding, 9 fields remain below 80 % (producer_category, method, eigenvectors_used, precision_stated, data_available, sectors, quantities, evidence, use). The pre-registration
     allows one revision; the remaining disagreement is reported, not corrected, and the result file marks the affected
     percentages as provisional. The subset coded in the check reading is reported alongside.
- v0.3 (2026-11-19, wording only): steps 3 and 4, the known limits and items 5, 6 and 7 of v0.2 name the coder and the
  check reading in neutral terms. The check reading was a second automatic reading, not a human check. The procedure,
  the data and every number are unchanged.
