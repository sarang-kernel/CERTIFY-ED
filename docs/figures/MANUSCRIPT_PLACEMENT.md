# Proposed placement of the figure set in manuscript v4 and in the consolidated report

Nothing in `manuscript_v4` is changed by this document; it proposes where each figure of `INDEX.md` goes
(manuscript sections as in `manuscript_v4/sections/*.tex`, report chapters as in `report/NN_*.md`) and gives
a `\caption` scaffold for each. Numbers in the scaffolds are the macros `make_numbers.py`, `make_numbers_apps.py`
and `make_survey_table.py` define (`generated/numbers*.tex`); where no macro exists for a number the figure
shows, the scaffold says `needs macro \Name from <record field>` and the figure's `.numbers.json` carries the
value. The three figures the manuscript already has (`fig_tightness`, `fig_widths`, `fig_cost`) stay where they
are; `fig_cost_memory` complements `fig_cost` (two stacks, memory) and does not replace it.

| figure | manuscript section (file) | report chapter | role |
|---|---|---|---|
| C1 `fig_certificate_anatomy` | §2 What a certificate states (`s2_certificates.tex`), first figure of the section | 04_software | the object the paper is about; replaces the prose walk through the JSON keys |
| C2 `fig_enclosure_geometry` | §3 Mathematical basis, "Enclosures" and "Counting" (`s3_math.tex`) | 05_math | Theorems 1–3, Lemma 4, Corollary 5 on one line |
| C3 `fig_inertia_completeness` | §3 "Counting" (`s3_math.tex`), after Corollary 5; or Appendix A next to the proof of Lemma 4 if §3 is tight | 05_math | the sandwich, the vectorless window count (C-02), the refusals |
| C4 `fig_rounding_accounting` | Appendix A "Computed quantities" (`appA_proofs.tex`), where the Theorem 1 accounting is rewritten from theory.md (D-16, INT-04) | 05_math; 08_defects (INT-04) | the code's accounting, not the former appendix text |
| E1 `fig_failure_matrix` | §5.3 Planted errors (`s5_verifier.tex`, `sec:e1`), next to Table `tab:e1` (the table keeps the mechanisms, the figure the outcomes) | 11_experiments (E1) | what each practice catches |
| E2 `fig_cost_memory` | §8 Cost and reach (`s8_cost.tex`), after `fig_cost` | 11_experiments (E2); 04_software | the same blocks on two stacks; memory decides S3 |
| E3 `fig_two_stack` | Appendix D Reproducing every number (`appD_reproducibility.tex`) | 03_chronology or 11_experiments (the comparison section of RESULT_evidence_run_3.1.0) | 0 MISMATCH across 3 885 leaves |
| E4 `fig_mutation` | §5.5 Mutation analysis (`s5_verifier.tex`, `sec:e6`) | 11_experiments (E6); 08_defects | killed/survived per module and the survivor categories |
| E5 `fig_corpus_detection` | §5 Verifying the verifier: a new subsection "Planted-error corpus" after `sec:e1` (the corpus is 3.2, not in v4; add when the manuscript moves to the 3.2 run) | 11_experiments (new corpus section); 04_software | detection by statement and class |
| E6 `fig_scope_vs_survey` | §2b Scope (`s2b_scope.tex`), between `sec:survey` and `sec:scope-matrix`, replacing or complementing the survey table | 01_summary (the pitch); 07_applications | what labs report against what is certified |
| E7 `fig_roadmap_progress` | not in the manuscript (project status, D-16); report only | 12_open (roadmap status after Waves 0–1) | 30 of 78 items |
| A1 `fig_crown_sectors` | §6.1 Pyrochlore (`s6_applications.tex`, `sec:app-crown`), with the landscape panel referred from Appendix C "what is computed and what is not" | 07_applications | the sector landscape and the certified all-singlet bound |
| A2 `fig_aklt_eta` | §6.2 AKLT (`s6_applications.tex`, `sec:app-aklt`), next to Table `apps_aklt` | 07_applications | certified η against the published values and the criteria |

Manuscript figure numbering, in reading order: C1 (Fig. 1), E6 (Fig. 2), C2 (Fig. 3), C3 (Fig. 4), fig_tightness
(Fig. 5, existing), E1 (Fig. 6), E4 (Fig. 7), A1 (Fig. 8), fig_widths (Fig. 9, existing), A2 (Fig. 10), fig_cost
(Fig. 11, existing), E2 (Fig. 12); C4, E3 in the appendices; E5 when the 3.2 run is the manuscript's run.

## Caption scaffolds

```latex
% C1 -- s2_certificates.tex
\begin{figure}[t]\centering
\includegraphics[width=\textwidth]{generated/figures/fig_certificate_anatomy.pdf}
\caption{What a certificate contains and how its statements depend on each other. (a)~The JSON body: subject
and claim, one record per statement S1--S8 with a status, the scope block, integrity and environment, and the
hashed arrays sidecar; the statuses shown are an example. (b)~S1 fixes the matrix the enclosures refer to (its
shift is added to every radius), S2 encloses, S3 counts and orders, S4 and S8 read clusters and brackets off
S2 and S3; S6 re-checks S2 and S3 in exact arithmetic from the claim and the specification alone; S7 needs the
whole-spectrum enclosure of a kind \texttt{all} claim; S5 is corroboration, not a certified statement.}
\label{fig:anatomy}\end{figure}

% C2 -- s3_math.tex
\caption{The enclosures on the real line (positions illustrative). Theorem~\ref{thm:residual} gives each
claimed value an interval of radius $r=\|Ax-\mu x\|/\|x\|$ (existence only); overlapping intervals form a
cluster with one radius $\varepsilon$ (Theorem~2/2$'$) under which the members are matched to distinct
eigenvalues; Theorem~\ref{thm:kato} sharpens an isolated value inside its isolating interval $(\alpha,\beta)$;
Lemma~\ref{lem:inertia} bounds the count below the shift $\sigma$ with backward error $\beta$, and
Corollary~\ref{cor:complete} turns $\nu=k$ into $\lambda_{k+1}\ge\sigma-\beta$.}
\label{fig:geometry}

% C3 -- s3_math.tex (or appA_proofs.tex)
\caption{How inertia counts certify ``exactly $k$ eigenvalues below $\sigma$'' ($N(x)=\#\{\lambda<x\}$,
illustrative spectra). (a)~$k$ distinct enclosures below $\sigma-\beta$ give $N(\sigma-\beta)\ge k$, the
inertia $\nu=k$ gives $N(\sigma-\beta)\le\nu$ (Lemma~\ref{lem:inertia}); the sandwich closes. (b)~A window
claim without eigenvectors: $\nu_b-\nu_a=\#\{\lambda\in(a,b)\}$ once two further counts per edge show the
collar eigenvalue-free. (c)~$\nu>k$ is FAILED (a missed eigenvalue below $\sigma$); an eigenvalue inside the
collar is INCONCLUSIVE.}
\label{fig:inertia}

% C4 -- appA_proofs.tex
\caption{The floating-point accounting of the Theorem~\ref{thm:residual} radius as implemented
(\texttt{enclosure.residuals}): three factors $(1+4u)$, the norm guards $(1\pm\gamma_{n+2})(1+8u)$, and the
model shift $\sigma$ from S1 added with \texttt{add\_up} where the radius enters S3, S4 and the exported
arrays. The slack of $\gamma_{w+1}$ over $\gamma_w$, of $\gamma_{w+2}$ over the entrywise error and of
$\gamma_{n+2}$ over $\gamma_n/2+u$ covers the roundings no factor names;
\texttt{tests/test\_theorem1\_accounting.py} pins each stage in exact arithmetic.}
\label{fig:rounding}

% E1 -- s5_verifier.tex, next to tab:e1
\caption{Fifteen results checked by three practices ($N=\FnN$ sites, $K=5$ claimed pairs; \FnErr{} of the
results are wrong). P1 (residual check with the producer's matrix) flags \FnPone, P2 (recomputation from the
intended model) flags \FnPtwo, at least one of the two flags \FnPoneOrPtwo; \pkg{} marks \FnPthreeFailed{}
FAILED and one INCONCLUSIVE (F7), and localizes the error in 12 cases
(needs macro \FnPthreeLocalizes from the E1 record, count of \texttt{P3\_localizes = true}). The
``statement named'' column is read off the diagnosis text of the record. F12 is correct and certified at the
multiplet level.}
\label{fig:e1}
% K = 5: needs macro \FnK from E1 settings.K (the table caption writes the number).

% E2 -- s8_cost.tex, after fig_cost
\caption{Certifier wall time (a) and peak memory (b) against block dimension for the E2 blocks on two stacks
(Linux, 3.0.0; Windows, 3.1.0). Filled: S3 CERTIFIED; open: S3 NOT\_CHECKED under the \CmemCap~GiB cap. On both
stacks S3 is certified up to dimension \CmaxFeasibleDim{} and not for the \CnotFitAtOrBelowMaxDim{} denser block
of the same dimension (\CnotFitDim, $\CnotFitNnzMin$ nonzeros). Number of blocks, longest run and peak memory:
needs macros \CtwoBlocks, \CtwoMaxSecondsWin, \CtwoMaxSecondsLinux, \CtwoMaxMiBWin, \CtwoMaxMiBLinux from the
two \texttt{e2\_cost} records (\texttt{certifier.total}, \texttt{certifier.maxrss\_mb}).}
\label{fig:costmem}

% E3 -- appD_reproducibility.tex
\caption{Record-by-record comparison of the two runs: every leaf of the eleven \texttt{data.json} files is in
one class; no MISMATCH. Counts: needs macros \CmpLeaves, \CmpSame, \CmpFloatDiffers, \CmpStackDependent,
\CmpSampleDependent, \CmpExplained, \CmpTiming, \CmpMissing, \CmpMismatch from
\texttt{compare\_vs\_run\_full\_20261021.csv} (column \texttt{class}).}
\label{fig:twostack}

% E4 -- s5_verifier.tex, sec:e6
\caption{Mutation analysis. (a)~\MutN{} sampled mutants: \MutKilled{} killed, \MutSurvived{} survived,
\MutTimeouts{} timeouts (score \MutScore); of the survivors \MutNeedInspection{} need a human verdict, the rest
are classified by rule (heuristic parameter, rounding allowance, no effect on statements, strict versus
non-strict comparison; counts: needs macros \MutCatHeuristic, \MutCatRounding, \MutCatNoEffect,
\MutCatStrict from \texttt{classification.json}, \texttt{survivor\_categories}). (b)~The 16-mutant sample on
the merged 3.2 branch: 13 killed (needs macros \MutSampleN, \MutSampleKilled from the 3.2 E6 record once the
3.2 evidence run exists; until then from RESULT\_integration\_3.2.md). Different code and sample sizes; the
scores are not compared.}
\label{fig:mutation}

% E5 -- s5_verifier.tex, new subsection (3.2 run)
\caption{\pkg{} on the planted-error corpus: statuses per planted-error class and statement, refusals, and
the statements the corpus expects not to be certified (frames). Counts: needs macros \CorpusPlanted,
\CorpusControls, \CorpusDetected, \CorpusDetectedByRefusal, \CorpusMissed, \CorpusFalseAlarms from
\texttt{corpus/results/certify-ed/report.json} (\texttt{summary}).}
\label{fig:corpus}

% E6 -- s2b_scope.tex, between sec:survey and sec:scope-matrix
\caption{What ED papers report against what a certificate states ($n=\SurveyN$ papers of the analysis set,
\SurveyProvisional; Wilson 95\,\% intervals). Excited levels (\SurveyFracExcitedLevels) and ground energies
(\SurveyFracGroundEnergy) are certified statements (S2, S3); correlation functions (\SurveyFracCorrelations)
and order parameters (\SurveyFracOrderParameter) through S4; dynamical response (\SurveyFracDynamics) and
finite-size extrapolation (\SurveyFracFiniteSize) are out of scope by design; entanglement
(\SurveyFracEntanglement) has nothing certified. Share of papers reporting only covered quantities: needs
macro \SurveyFullyCovered from \texttt{survey\_coded.csv} (the rule of
\texttt{fig\_scope\_vs\_survey.py}).}
\label{fig:scope}

% A1 -- s6_applications.tex, sec:app-crown
\caption{The pyrochlore-crown $S=1$ model. (a)~The lowest eigenvalue of each pair-spin class of the $S^z=0$
block at the re-optimised couplings, sorted; the lowest class (\CrownOneLandscapeMin, dimension
\CrownOneMinClassDim) and the largest class (dimension \CrownOneBigClassDim, \CrownOneBigClassAbove{} above
it) are marked (number of classes and the couplings: needs macros \CrownOneNclasses, \CrownOneLandscapeCone,
\CrownOneLandscapeCthree from \texttt{spin1\_landscape}). (b)~At the printed couplings the all-singlet-sector
eigenvalue is certified $\le\CrownOneEig$ in exact arithmetic (ring radius \CrownOneRingRadiusExact),
\CrownOneMargin{} below the published bound \CrownOnePublished; the two named sectors have lowest eigenvalues
\CrownOneNamedSectorA{} and \CrownOneNamedSectorB{} (floating point, no certified interval).}
\label{fig:crown}

% A2 -- s6_applications.tex, sec:app-aklt
\caption{Certified upper bounds on $\eta$ for the Pomata--Wei lattices against the published values (the
largest excess is \AKLTMaxExcess): star \EtaStar, cross \EtaCross, decorated honeycomb \EtaDecHoney, decorated
square \EtaDecSquare; the square-octagon pair: needs macros \EtaSqOctPerp, \EtaSqOctPar from
\texttt{lattices.square\_octagon\_perp.eta\_upper} and \texttt{.square\_octagon\_par.eta\_upper}. Orange
ticks: the gap-criterion thresholds $1/z$; the square-octagon rows share $4\eta_\perp+2\eta_\parallel<1$.
(b)~The certified criterion values lie between \AKLTCritMin{} and \AKLTCritMax{} (the square-octagon sum:
needs macro \AKLTSqOctSum from \texttt{square\_octagon.sum\_upper}); every one is below 1. Largest block
\DecSquareLargestBlock{} (decorated square, \DecSquareMinutes~min).}
\label{fig:aklt}
```

## Report placement notes

- 01_summary: E6 as the one figure of the summary (the pitch: what labs report, what is certified).
- 04_software: C1, E2, E5.
- 05_math: C2, C3, C4.
- 07_applications: A1, A2, E6.
- 08_defects: C4 (INT-04), E4 (survivors as test gaps).
- 11_experiments: E1, E2, E3, E4, E5 under their experiment headings.
- 12_open: E7.

The report's `build_report.py` embeds figures by path; the PNGs in `docs/figures/` are the files to copy into
the report tree (the PDFs for the LaTeX build).
