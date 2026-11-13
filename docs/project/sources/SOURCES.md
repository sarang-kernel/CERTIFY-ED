# sources/

`archive_v5/` is the unpacked project archive v5, `CERTIFY-ED_project_archive_v5_2026-11-06.tar.gz` (47 MB, 2045
members), SHA-256 `a0aaea142448f396db5adbefd602b7000641441cb1f9bdbbb3616bdea7d1b056`. It is not tracked by git because it contains a complete git clone of the code
repository (`archive_v5/code/certify-ed/`, branch `dev/3.2` at `a814d28`, tags `v3.1.0`, `v3.2.0`, `v3.2.1`) and its run
records (`archive_v5/code/certify-ed/results/`). Every path cited in docs/ and blog/ is relative to `archive_v5/`.

`certify-ed_dev-3.2.bundle` is the code repository as a git bundle (all branches and tags), SHA-256
`627313031d6686ef90cb2ad7602d5aed6abb31e0c7786361ab36a5810fe5d04c`. It is two commits ahead of `archive_v5/code/certify-ed`: `dev/3.2` is at `fa86456`.
Commit `40d59f4` adds the documentation corrections of 10 November 2026 (two explanations in
`RESULT_evidence_run_3.2.1.md`, the hypothesis of Proposition 14 in `docs/theory.md`, an erratum to
`docs/proofs/thm1_fp-accounting.md`). Commit `fa86456` adds four errata found by the Lean formalization (one inequality in
the proof of Theorem U5 in `docs/proofs/int05_underflow.md`, the argument for part (b) in `docs/proofs/D-02_matrix-free.md`,
the width of `from_float` in `docs/proofs/L17_dyadic_exp_ln.md`, the index of Lemma A1 for complex amplitudes in
`docs/proofs/appA_model-shift.md`) and a note in `CHANGELOG.md`. Neither commit changes a computational file; the source
hash stays `cb73dec3e4433a3b`. To recreate the clone: `git clone certify-ed_dev-3.2.bundle certify-ed`.

`certify-ed_dev-3.3.bundle` is the code repository as a git bundle (all branches and tags) after the first development
phase of 11 to 13 November 2026, SHA-256 `dae67f79290a2251c717ae2480008b6b7e8a3f015758d5215ac2c907c9991b45`. It holds every ref of `certify-ed_dev-3.2.bundle` and in
addition `fix/3.2.2` at `a9b9838` (tag `v3.2.2`, source hash `746958b63a65b43e`), `fix/3.2.3` at `fec6f8c` (tag `v3.2.3` at
`9f42adc`, source hash `bfc115e824386443`), `dev/3.3` at `6939734` (version 3.3.0.dev0, the merges of the five tracks and of
`fix/3.2.3`; its full test suite at `6939734`: 1265 passed, 19 skipped, 0 failed) and the track branches `track/scale-wave8` at `eb82eb9`, `track/hardening-3.2.2` at `740d824`,
`track/non-hermitian` at `e21eda8`, `track/external-verifier` at `d704471` and `track/applications` at `bf530ca`. On
`fix/3.2.3`, commit `5dedf9d` changes only the number script `experiments/make_numbers.py` (the source hash stays
`bfc115e824386443`) and commit `fec6f8c` adds `RESULT_evidence_run_3.2.3.md`. To recreate the clone:
`git clone certify-ed_dev-3.3.bundle certify-ed`, then `git checkout dev/3.3`.

`runs/` holds the run records of the two evidence runs of this phase, which the code repository does not track
(`results/` is in its `.gitignore`). `runs/run_full_20261112_v3.2.2.tar.gz` (SHA-256
`129fa6695034d4053a8f6b176a9805db850849039e9c717e4af6c3aba352a40e`) is the 3.2.2 run `results/run_full_20261112`, with its
comparison against the 3.2.1 run; it is incomplete (see `RESULT_evidence_run_3.2.2.md` in the bundle).
`runs/run_full_20261112b_v3.2.3.tar.gz` (SHA-256 `cb87c3b756bb90d11191907988e6506bdae5c4deef4d53274dd77d3299b39ed6`) is the 3.2.3 run
`results/run_full_20261112b`, with its comparison against the 3.2.1 run and the manifest and summary as they were before
the numbers step was resumed at `5dedf9d` (see `RESULT_evidence_run_3.2.3.md`). The 3.2.1 run `results/run_full_20261105`
is in `archive_v5/code/certify-ed/results/`.

To recreate `archive_v5/` from the tarball: extract it and rename the top folder `CERTIFY-ED_project_archive_v5_2026-11-06`
to `archive_v5`. Its own `SHA256SUMS.txt` lists every file.
