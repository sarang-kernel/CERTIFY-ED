# RESULT: survey of exact-diagonalization practice in recent arXiv preprints

Generated 2026-10-21 19:27 UTC by `survey/analyze.py` + `survey/make_result.py`.
Pre-registration: PREREG_practice_survey.md (v0.1 written before any fetch; v0.2 changelog records every deviation).
Wording revised 2026-11-19 (pre-registration v0.3: neutral terms for the coder and the check reading); the numbers are unchanged.
Status: **full survey per the pre-registered stop rule (n_analysis >= 40)**.

## 1. Sample

| Stage | Count |
|---|---:|
| arXiv API records matched by the pre-registered query (totalResults) | 878 |
| Records stored in `survey_sample.json` (deduplicated) | 878 |
| Population by year of first submission (2024 / 2025 / 2026) | 233 / 274 / 371 |
| Subsample draws attempted (seed 20261021) | 80 |
| Fetch failures -> replacements | 0 |
| Coded subsample (full text obtained and coded) | 80 |
| Subsample by year (2024 / 2025 / 2026) | 14 / 39 / 27 |
| Text source | latex-tar: 80 |
| Text truncation to 60 k chars | full: 55, section-prioritised: 25 |

Population primary categories (top): cond-mat.str-el 389, quant-ph 193, hep-th 58, cond-mat.quant-gas 56, cond-mat.mes-hall 54, cond-mat.stat-mech 37.

One `comment` field of `survey_sample.json` was edited on 18 November 2026, after this report was generated: an
author's e-mail address in the arXiv comment of one record was replaced by `[e-mail address removed]`. No count or
coded field of this report reads that field.

## 2. Exclusions (recorded, not dropped from the CSV)

| Reason | Count |
|---|---:|
| comparison-only | 3 |
| no-ED | 5 |
| non-hermitian | 3 |
| other | 2 |
| **Analysis set** (uses ED themselves, research article, Hermitian quantum problem) | **67** |

## 3. What is reported, what evidence is given, how the numbers are used

**Quantities reported from ED** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| excited levels | 42 | 63 | [51, 73] |
| ground energy | 38 | 57 | [45, 68] |
| other | 29 | 43 | [32, 55] |
| correlation functions | 28 | 42 | [31, 54] |
| order parameter | 25 | 37 | [27, 49] |
| dynamical response/spectral function | 10 | 15 | [8, 25] |
| finite-size extrapolation | 10 | 15 | [8, 25] |
| entanglement entropy/spectrum | 9 | 13 | [7, 24] |
| gaps | 8 | 12 | [6, 22] |
| structure factor | 6 | 9 | [4, 18] |
| degeneracies | 4 | 6 | [2, 14] |
| time evolution | 3 | 4 | [2, 12] |
| level statistics | 3 | 4 | [2, 12] |
| quantum numbers/labels | 2 | 3 | [1, 10] |
| thermodynamics (full spectrum) | 2 | 3 | [1, 10] |
| fidelity/susceptibility | 1 | 1 | [0, 8] |

**Evidence of correctness given for the ED numbers** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| comparison with another code | 38 | 57 | [45, 68] |
| comparison with analytic limit | 34 | 51 | [39, 62] |
| symmetry/sum-rule check | 13 | 19 | [12, 30] |
| none stated | 12 | 18 | [11, 29] |
| comparison with literature value | 7 | 10 | [5, 20] |
| residual or convergence tolerance stated | 3 | 4 | [2, 12] |

Papers giving at least one kind of evidence: 55/67 = 82 %.
Median number of distinct quantity labels per paper: 3.0.

**Use of the ED numbers** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| standalone physics result | 55 | 82 | [71, 89] |
| benchmark anchor for another method | 29 | 43 | [32, 55] |
| lemma in a rigorous argument | 0 | 0 | [0, 5] |

**Symmetry sectors used** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| not stated | 40 | 60 | [48, 71] |
| U(1) | 24 | 36 | [25, 48] |
| translation | 19 | 28 | [19, 40] |
| SU(2) | 6 | 9 | [4, 18] |
| parity | 4 | 6 | [2, 14] |
| point group | 1 | 1 | [0, 8] |
| none | 0 | 0 | [0, 5] |

**Method** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| full diagonalization | 27 | 40 | [29, 52] |
| not stated | 26 | 39 | [28, 51] |
| Lanczos-Arnoldi | 12 | 18 | [11, 29] |
| shift-invert | 1 | 1 | [0, 8] |
| other | 1 | 1 | [0, 8] |

**Producer** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| not stated | 43 | 64 | [52, 75] |
| named code | 15 | 22 | [14, 34] |
| in-house | 9 | 13 | [7, 24] |

Named codes credited (first-named, analysis set, labels normalised by case and with parentheticals and co-named libraries dropped; the verbatim free-text values are in survey_coded.csv): QuSpin (4), ALPS (2), SciPy (1), public code repository (1), NumPy (1), LAPACK (1), PETSc (1), ARPACK (1), Expokit.jl (1), Sunny.jl (1), DiagHam (1); total 15.
Largest Hilbert-space / block dimension stated explicitly: 16/67.

**Eigenvectors needed for a reported quantity** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| yes | 58 | 87 | [76, 93] |
| no | 7 | 10 | [5, 20] |
| unclear | 2 | 3 | [1, 10] |

**Numerical precision of the ED numbers stated** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| no | 45 | 67 | [55, 77] |
| yes | 22 | 33 | [23, 45] |

**ED data or code published** (analysis set, n = 67; Wilson 95 % interval in brackets)

| Label | k | % | 95 % CI |
|---|---:|---:|---|
| no | 51 | 76 | [65, 85] |
| partial | 10 | 15 | [8, 25] |
| yes | 6 | 9 | [4, 18] |

## 4. Check reading (seed 20261022, n = 12 papers re-coded in a second automatic reading of the extracted text, not a human check)

Agreement between the final automatic coding (pass 2, after the pre-registered revision of the coding instructions) and the check reading; the pass-1 figures are given below the table. Single-valued fields: exact match. Multi-label fields:
mean Jaccard index between label sets (1 when both empty); the label-wise presence/absence agreement is given in the
last column for comparison. The pre-registered threshold is 80 %.

| Field | Agreement | Label-wise |
|---|---:|---:|
| uses_ed | 92 % |  |
| paper_type | 100 % |  |
| problem_class | 92 % |  |
| producer_category | 67 % |  |
| method | 50 % |  |
| eigenvectors_used | 75 % |  |
| precision_stated | 75 % |  |
| data_available | 75 % |  |
| max_dim_stated | 100 % |  |
| sectors | 49 % | 81 % |
| quantities | 52 % | 86 % |
| evidence | 39 % | 75 % |
| use | 67 % | 81 % |

Fields below 80 %: producer_category, method, eigenvectors_used, precision_stated, data_available, sectors, quantities, evidence, use.

Pass-1 agreement (before the pre-registered revision): uses_ed 92 %, paper_type 100 %, problem_class 75 %, producer_category 58 %, method 50 %, eigenvectors_used 75 %, precision_stated 58 %, data_available 58 %, max_dim_stated 100 %, sectors 43 %, quantities 46 %, evidence 38 %, use 54 %.
Fields re-coded with the revised instructions (pass 2, quote-backed labels): problem_class, producer_category, method, eigenvectors_used, precision_stated, data_available, sectors, quantities, evidence, use.
Residual disagreement direction on the check set (label counts summed over the 12 papers): the coder added labels the check
reading did not — sectors 10, quantities 14, evidence 11, use 5; the check reading had labels the coder missed —
sectors 6, quantities 13, evidence 7, use 2. The coder still over-codes `evidence` (it counts a benchmark in which
ED is the reference as evidence *for* the ED) and under-codes `method` (it answers 'not stated' where Lanczos is named in the text).

**Consequence.** Percentages for `evidence`, `method`, `sectors`, `producer_category`, `precision_stated`, `data_available`,
`use` and `quantities` (paper-level) are **provisional**: they rest on an automatic coding whose paper-level agreement with the
check reading is below the pre-registered 80 %. Label-wise agreement for `quantities` is 86 %, so the *ranking* of quantities is
usable for the scope matrix; the evidence percentages should be read with the re-coded subset below.

**Analysis-set papers of the check reading (n = 8):** quantities — ground energy 3, gaps 2, excited levels 3, degeneracies 1, quantum numbers/labels 1, correlation functions 2, structure factor 2, thermodynamics (full spectrum) 1, dynamical response/spectral function 5, time evolution 1, level statistics 1, other 4;
evidence — residual or convergence tolerance stated 2, comparison with another code 2, comparison with literature value 1, comparison with analytic limit 1, symmetry/sum-rule check 1, none stated 4; use — benchmark anchor for another method 3, standalone physics result 6.
In this subset, 4/8 papers state no evidence for the correctness of their ED numbers and
2/8 state a residual or convergence tolerance.


## 5. Reading for the scope framework (feeds `survey_for_scope.json`)

- Fractions above are of the analysis set (n = 67); they describe arXiv preprints 2024-2026 in the listed categories that
  mention ED/Lanczos in the abstract, not "the community".
- `survey_for_scope.json` carries, for each quantity label, the fraction of papers reporting it with its Wilson interval,
  for the SCOPE_MATRIX and the preset design. No outcome changes which statements CERTIFY-ED certifies (pre-registration,
  stop rules).

## 6. Limits

- Abstract-level query: papers that use ED without naming it in the abstract are missed; papers that mention Lanczos in
  other senses (Krylov complexity, Lanczos time evolution) enter the population and are excluded at coding.
- The first pass is an automatic coder reading LaTeX-stripped text (equations replaced by [EQ]); numbers that live
  only in equations or figures are invisible to it. The check reading measures the error, it does not correct it everywhere.
- The check reading was a second automatic reading of keyword-focused excerpts of the extracted text plus targeted
  searches of the full text: not a human check, and not a complete linear reading of every paper.
- Supplementary material is not fetched; text is truncated at 60 k characters with methods/results sections prioritised.

## 7. Deviations from the pre-registration

See PREREG_practice_survey.md v0.2 changelog (population cap lifted to the full 878 matches; arxiv.org source host with
Range requests; bookkeeping fields added; two papers without pass-2 coding; residual agreement below 80 % after the single
pre-registered revision).

