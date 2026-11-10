# Decisions

The project keeps an append-only decision log, `project_docs/DECISIONS_LOG.md`, in which every entry records a
decision, the alternatives rejected, the reason and the condition under which it would be reopened. Version 1.19 of 6
November 2026 holds 61 entries, D-01 to D-61. The four files of this folder explain every entry in plain prose: what
was decided, why, what was rejected and what followed. They paraphrase the log and do not replace it; where the two
differ, the log is the record.

The identifiers need one warning. The expansion roadmap of 29 October 2026 (`code/certify-ed/ROADMAP.md`) also uses
identifiers of the form letter, hyphen, two digits (A-01, C-03, D-01, E-05 and so on). Roadmap item D-01 (the
a-posteriori residual of the sparse factorization) is unrelated to decision D-01 (the end of the research direction
C2). In these documents a bare D-NN is a decision; a roadmap item is always called a roadmap item.

The decisions fall into four periods. D-01 to D-16 (May to July 2026) close the first version and set the rules that
still govern the project, above all D-12 (new mathematics counts as established only after Sarang Vehale re-derives
it) and D-16 (manuscript prose is Sarang's). D-17 to D-31 (September to October 2026) design version 2 and its
evidence policy. D-32 to D-47 (15 to 21 October 2026) merge the project into one paper and one codebase, add the two
applications and release 3.0.0. D-48 to D-61 (21 October to 6 November 2026) follow the independent review, the 3.1.0
release, the expansion roadmap and the 3.2 evidence run. Three decisions define how every other document of this
repository talks about results: D-12 and D-59 fix the status of the derivation documents, D-44 fixes what counts as
one evidence run, and D-61 records that the cited version is still 3.1.0 until Sarang decides.

| Entry | Date | Subject | File |
|---|---|---|---|
| D-01 | May 2026 | direction C2 is stopped | [01_before_v2.md](01_before_v2.md) |
| D-02 | May to June 2026 | the target of direction C1 | [01_before_v2.md](01_before_v2.md) |
| D-03 | 13 July 2026 | C1 is put aside | [01_before_v2.md](01_before_v2.md) |
| D-04 | June to July 2026 | no rebuttal to the SciPost referees | [01_before_v2.md](01_before_v2.md) |
| D-05 | 13 July 2026 | the failure of v1 was a mismatch of scope and claim | [01_before_v2.md](01_before_v2.md) |
| D-06 | 13 July 2026 | broaden to a general checking layer | [01_before_v2.md](01_before_v2.md) |
| D-07 | 13 July 2026 | enumerate everything first | [01_before_v2.md](01_before_v2.md) |
| D-08 | 13 July 2026 | no constraints in the scope document | [01_before_v2.md](01_before_v2.md) |
| D-09 | 13 July 2026 | no preliminary list from Sarang | [01_before_v2.md](01_before_v2.md) |
| D-10 | 13 July 2026 | the engineering plan follows the enumeration | [01_before_v2.md](01_before_v2.md) |
| D-11 | 13 July 2026 | talk to producer authors after the scope exists | [01_before_v2.md](01_before_v2.md) |
| D-12 | 13 July 2026 | new mathematics needs a person who re-derives it | [01_before_v2.md](01_before_v2.md) |
| D-13 | 13 July 2026 | documentation is mandatory | [01_before_v2.md](01_before_v2.md) |
| D-14 | 13 July 2026 | one idea taken from an unsolicited e-mail | [01_before_v2.md](01_before_v2.md) |
| D-15 | 21 July 2026 | no response to the third rejecting report | [01_before_v2.md](01_before_v2.md) |
| D-16 | 21 July 2026 | Sarang drafts the prose of every manuscript | [01_before_v2.md](01_before_v2.md) |
| D-17 | 25 September to 4 October 2026 | a revised manuscript and a v2 core, as an exception to D-10 | [02_v2_design.md](02_v2_design.md) |
| D-18 | September 2026 | five statements, four statuses and a not-checked list | [02_v2_design.md](02_v2_design.md) |
| D-19 | September 2026 | a default accuracy target for S2 | [02_v2_design.md](02_v2_design.md) |
| D-20 | September 2026 | completeness by Sylvester's law of inertia | [02_v2_design.md](02_v2_design.md) |
| D-21 | 4 October 2026 | the direction of the Jordan-Wigner string is an option | [02_v2_design.md](02_v2_design.md) |
| D-22 | 4 October 2026 | radii refer to the declared model | [02_v2_design.md](02_v2_design.md) |
| D-23 | 4 October 2026 | an ordering condition for the assembly of sectors | [02_v2_design.md](02_v2_design.md) |
| D-24 | 4 October 2026 | every number of the paper is generated from records | [02_v2_design.md](02_v2_design.md) |
| D-25 | 4 October 2026 | no claim of mathematical novelty | [02_v2_design.md](02_v2_design.md) |
| D-26 | 4 October 2026 | every iterative solver call is seeded | [02_v2_design.md](02_v2_design.md) |
| D-27 | 4 October 2026 | the ARPACK multiplet truncation becomes experiment E5 | [02_v2_design.md](02_v2_design.md) |
| D-28 | 4 October 2026 | the below-top probe accounts for the backward error | [02_v2_design.md](02_v2_design.md) |
| D-29 | 11 October 2026 | records of interrupted runs are never read | [02_v2_design.md](02_v2_design.md) |
| D-30 | 11 October 2026 | a table of the contribution by kind | [02_v2_design.md](02_v2_design.md) |
| D-31 | 11 October 2026 | a visual check of every rendered figure | [02_v2_design.md](02_v2_design.md) |
| D-32 | 15 October 2026 | no target venue; a survey of neighbouring practice | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-33 | 17 October 2026 | symveig belongs to the project and joins one paper | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-34 | 17 October 2026 | the first application is the pyrochlore bound of Kránitz and Penc | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-35 | 17 October 2026 | mutation analysis is evidence, with classified survivors | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-36 | 17 October 2026 | the second application is the AKLT gap criteria | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-37 | 17 October 2026 | the AKLT kernel equality is a lemma, also checked by computation | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-38 | 17 October 2026 | the bordered pencil for the decorated lattices | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-39 | 17 October 2026 | all inspected survivors of the first E6 run resolved | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-40 | 17 October 2026 | the S = 1 contradiction is certified, the corrected optimum is an estimate | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-41 | 17 October 2026 | the decorated square is certified | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-42 | 18 October 2026 | symveig is merged as a capability | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-43 | 18 October 2026 | one codebase, certify-ed 3.0.0 | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-44 | 18 October 2026 | provenance by computational source, and one run per cited version | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-45 | 21 October 2026 | the one-paper manuscript as a scaffold | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-46 | 21 October 2026 | writing the proofs against the code found two soundness gaps | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-47 | 21 October 2026 | Sarang's decisions on four open items | [03_one_paper_and_3.0.md](03_one_paper_and_3.0.md) |
| D-48 | 21 October 2026, evening | the history of the code repository is rewritten before the 3.1.0 run | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-49 | 21 October 2026, evening | certify everything a laboratory run produces, and declare the scope | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-50 | 21 October 2026, evening | a pre-registered survey of published ED practice | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-51 | 21 October 2026, evening | the note to Kránitz and Penc is corrected before sending | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-52 | 21 October 2026, evening | Theorem 2 is restated through Kahan's theorem | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-53 | 21 October 2026, evening | a corrected replacement of the symveig preprint and deprecation of its code | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-54 | 22 October 2026 | the Windows workstation is the primary machine | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-55 | 22 October 2026 | the four v1 figures are not carried into ms_v4 | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-56 | 22 October 2026 | the public repository is extended, not rewritten | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-57 | 22 October 2026, night | the 3.1.0 run stands; four integration findings are known issues | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-58 | 29 October 2026 | the expansion roadmap, executed on dev/3.2 | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-59 | 30 October 2026, recorded 3 November 2026 | clearing the D-12 backlog with derivation documents | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-60 | 3 November 2026 | certify-ed 3.2.0 gets its own full run | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
| D-61 | 6 November 2026 | the 3.2 evidence run is the 3.2.1 run | [04_review_3.1_and_3.2.md](04_review_3.1_and_3.2.md) |
