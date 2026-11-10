# Applications and the independent verifier: map of the documents

This folder describes what CERTIFY-ED has computed outside its own test suite: the numerical lemmas of two published
proofs (the pyrochlore crown of Kránitz and Penc and the AKLT lattices of Pomata and Wei), the benchmark ground energies
certified with no floating-point assumption (experiment E7), the audit of the earlier package symveig and the comparison
with it (experiment E8), and the independent verifier written in C. The mathematics they use is in `docs/math/`. Every
path is relative to `sources/archive_v5/`; the code is `code/certify-ed/` at commit `a814d28` of `dev/3.2`.

| file | contents |
|---|---|
| [crown_kranitz_penc.md](crown_kranitz_penc.md) | the 18-site pyrochlore crown: the certified lower bound for the reduced spin-1/2 model, what it confirms and what it does not (the reduced-model value, not the nine-coupling headline value), the certified spin-1 eigenvalue below the published bound at the printed couplings, and the floating-point values of the named sectors and of the 92 pair-spin classes |
| [aklt_pomata_wei.md](aklt_pomata_wei.md) | the AKLT gap criteria of Pomata and Wei: certified upper bounds on $\eta$ for five lattices, the kernel lemma and the bordered-pencil lemma, the checks, the first certification scripts of 17 October 2026 and their two formulations, the honeycomb lattice (not attempted) and a numerical discrepancy in the paper's honeycomb proof |
| [exact_benchmarks_e7.md](exact_benchmarks_e7.md) | experiment E7: four benchmark ground energies of dimension 9 252 and 12 870 certified by the exact symmetry-sector path, with the radii and times of the earlier runs on two platforms |
| [symveig_e8.md](symveig_e8.md) | symveig 1.0.0: what it does, the audit, the conditions under which a whole-spectrum enclosure may claim counts, its replacement `certify_ed.spectrum`, experiment E8, and the decision about the preprint |
| [c_verifier.md](c_verifier.md) | the C verifier `verifier/src/cev.c`: scope, checks, what it does not check, and its validation (55 of 55 certificates, 18 of 18 FAILED statuses reproduced, 80 of 80 forgeries caught) |

## Status of the application results

Each number below is one of three kinds, and the documents keep them apart: a certified bound (from a certificate or an
exact computation, with the lemmas it rests on named), a floating-point estimate (computed without an error bound), or a
published value (copied from the paper). The application results of the two evidence runs, 3.1.0
(`code/certify-ed/results/run_full_20261022`) and 3.2.1 (`code/certify-ed/results/run_full_20261105`), agree in every
compared leaf except timings: 893 of 893 for E9, 333 of 333 for E10 and 336 of 336 for E7
(`code/certify-ed/RESULT_evidence_run_3.2.1.md`). The certified application bounds also rest on mathematics that awaits
Sarang Vehale's check: the exact-layer Lemmas 11 and 12 and Proposition 13 (crown and E7) and the bordered-pencil lemma
(AKLT), all marked UNVERIFIED and without a derivation document. The arguments of the two papers (the Anderson
decomposition and the gap criterion) are taken from the papers and are not re-derived.

## Where each number comes from

| quantity | value | kind | record |
|---|---|---|---|
| crown, spin 1/2, reduced model: $\lambda_{\min}$ is at least | -0.5501582988212038 | certified (exact arithmetic) | `code/certify-ed/results/run_full_20261105/e9_pyrochlore_20261105_084835/data.json`, `spin_half.lower_bound` |
| crown, spin 1/2, Lanczos estimate | -0.550158297907944 | floating-point estimate | same file, `spin_half.e0_estimate` |
| crown, spin 1/2, paper's reduced-model value | -0.550158(2) | published | `manuscript_v4/generated_3.1.0/numbers_apps.tex`, `CrownHalfReducedPaper` |
| crown, spin 1/2, paper's headline (nine couplings) | -0.549832(8) | published; reproduced in floating point (-0.54983289), not certified | same file, `CrownHalfHeadlinePaper`; `review/REVIEW_B_applications.md`, B2-4 |
| crown, spin 1, printed couplings: an eigenvalue is at most | -1.6380246425131773 | certified (exact re-check of the ring) | `code/certify-ed/results/run_full_20261105/e9_pyrochlore_20261105_084835/data.json`, `spin1_printed.eigenvalue_upper` |
| crown, spin 1, published bound | -1.632985 | published | same file, `spin1_printed.published_bound` |
| crown, spin 1, named sector [0,0,0,2,0,2] | -1.6329866581559116 | floating-point estimate | same file, `spin1_printed.named_sectors[0].energy` |
| crown, spin 1, named sector [0,2,2,0,2,2] | -1.632984890229642 | floating-point estimate | same file, `spin1_printed.named_sectors[1].energy` |
| crown, spin 1, lowest of 92 classes at re-optimised couplings | -1.6343595995104314 | floating-point estimate | same file, `spin1_landscape.classes[0].energy` |
| crown, spin 1, best value over $(c_1, c_3, c_5)$ is at most | -1.6342449 | floating-point upper bound over 34 classes | `review/REVIEW_B_applications.md`, B3-5 |
| AKLT, $\eta$ upper bounds (five lattices, six values) | 0.1110430221 to 0.2203543194 | certified (exact arithmetic) | `code/certify-ed/results/run_full_20261105/e10_aklt_20261105_071106/data.json`, `lattices.<name>.eta_upper`; rounded in `manuscript_v4/generated_3.1.0/tables/apps_aklt.tex` |
| AKLT, published $\eta$ | 0.1110430220 to 0.2203543174 | published | same file, `lattices.<name>.eta_published` |
| AKLT, square-octagon criterion $4\eta_\perp + 2\eta_\parallel$ is at most | 0.8481546963604274 | certified | same file, `square_octagon.sum_upper` |
| E7, four ground energies and exact radii | -8.904386529876417 to -11.228483208428859; radii 1.099862660492978e-14 to 3.719300427741141e-14 | certified (exact arithmetic) | `code/certify-ed/results/run_full_20261105/e7_exact_benchmarks_20261105_092337/data.json` |
| E8, half-widths of symveig and certify-ed | 17 rows | certified enclosures (certify-ed) and symveig's intervals | `code/certify-ed/results/run_full_20261105/E8.log`, `code/certify-ed/results/run_full_20261022/E8.log` |
| C verifier, certificates passed, FAILED statuses reproduced, forgeries caught | 55 of 55, 18 of 18, 80 of 80 | validation of a second program | `code/certify-ed/verifier/validation/RESULTS.md` |

## Reproducing

All four experiments run from the root of the code clone with `python run.py --only E7,E8,E9,E10` (or one at a time),
and the full evidence run with `certify-ed reproduce --full --with-mutation --jobs 2 --out results/run_full_<date>`; in
the 3.2.1 run E7 took 1821 s, E8 284 s, E9 2101 s and E10 5848 s on the development machine (8 logical processors, 16
GiB). The verifier is built and validated as described in [c_verifier.md](c_verifier.md).
