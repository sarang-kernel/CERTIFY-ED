# docs/project/sources/

`archive_v5/` is the unpacked project archive v5, `CERTIFY-ED_project_archive_v5_2026-11-06.tar.gz` (47 MB, 2045
members), SHA-256 `a0aaea142448f396db5adbefd602b7000641441cb1f9bdbbb3616bdea7d1b056`. It is not part of this
repository: it contains a complete git clone of the code repository (`archive_v5/code/certify-ed/`, branch `dev/3.2` at
`a814d28`, tags `v3.1.0`, `v3.2.0`, `v3.2.1`) with its run records, and material that is not for publication (draft
manuscripts and referee reports). It is kept next to the clone, in `../archive/archive_v5` seen from the repository
root; the generators in `tools/`, `lean/` and `docs/blog/` look for it there, or at the path in the environment variable
`CERTIFYED_ARCHIVE`. Every path cited in `docs/project/` and `docs/blog/` that does not start with a folder of this
repository is relative to `archive_v5/`.

The code history is part of this repository. Until 18 November 2026 the project record cited it through two git bundles,
`certify-ed_dev-3.2.bundle` (SHA-256 `627313031d6686ef90cb2ad7602d5aed6abb31e0c7786361ab36a5810fe5d04c`) and
`certify-ed_dev-3.3.bundle` (SHA-256 `52a2e64efa76bf0a0bbe1084616f4cd7308cb412fc4b0d709888bb8076cf9849`); every commit they
held has the same id here. The two documentation commits after archive v5, `40d59f4` (the corrections of 10 November 2026)
and `fa86456` (four errata found by the Lean formalization), change no computational file; the source hash stays
`cb73dec3e4433a3b`. The first development phase of 11 to 13 November 2026 added tag `v3.2.2` at `a9b9838` (source hash
`746958b63a65b43e`), tag `v3.2.3` at `9f42adc` (source hash `bfc115e824386443`) and the line 3.3.0.dev0 at `c29e968`, whose
full test suite at `6939734` gave 1265 passed, 19 skipped, 0 failed.

`runs/` holds the run records of the two evidence runs of this phase, which the code repository does not track
(`results/` is in its `.gitignore`). `runs/run_full_20261112_v3.2.2.tar.gz` (SHA-256
`129fa6695034d4053a8f6b176a9805db850849039e9c717e4af6c3aba352a40e`) is the 3.2.2 run `results/run_full_20261112`, with its
comparison against the 3.2.1 run; it is incomplete (see `RESULT_evidence_run_3.2.2.md`).
`runs/run_full_20261112b_v3.2.3.tar.gz` (SHA-256 `cb87c3b756bb90d11191907988e6506bdae5c4deef4d53274dd77d3299b39ed6`) is the 3.2.3 run
`results/run_full_20261112b`, with its comparison against the 3.2.1 run and the manifest and summary as they were before
the numbers step was resumed at `5dedf9d` (see `RESULT_evidence_run_3.2.3.md`). The 3.2.1 run `results/run_full_20261105`
is in `archive_v5/code/certify-ed/results/`.

To recreate `archive_v5/` from the tarball: extract it and rename the top folder `CERTIFY-ED_project_archive_v5_2026-11-06`
to `archive_v5`. Its own `SHA256SUMS.txt` lists every file.
