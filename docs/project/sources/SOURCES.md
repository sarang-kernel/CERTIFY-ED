# sources/

`archive_v5/` is the unpacked project archive v5, `CERTIFY-ED_project_archive_v5_2026-11-06.tar.gz` (47 MB, 2045
members), SHA-256 `a0aaea142448f396db5adbefd602b7000641441cb1f9bdbbb3616bdea7d1b056`. It is not tracked by git because it contains a complete git clone of the code
repository (`archive_v5/code/certify-ed/`, branch `dev/3.2` at `a814d28`, tags `v3.1.0`, `v3.2.0`, `v3.2.1`) and its run
records (`archive_v5/code/certify-ed/results/`). Every path cited in docs/ and blog/ is relative to `archive_v5/`.

`certify-ed_dev-3.2.bundle` is the code repository as a git bundle (all branches and tags), SHA-256
`8647baf7498de56c6db2242a48a8d3b54e8988ca19855ded5779b14842581511`. It is one commit ahead of `archive_v5/code/certify-ed`: `dev/3.2` is at `40d59f4`, which adds the
documentation corrections of 10 November 2026 (two explanations in `RESULT_evidence_run_3.2.1.md`, the hypothesis of
Proposition 14 in `docs/theory.md`, an erratum to `docs/proofs/thm1_fp-accounting.md`; no computational file changed, the source
hash stays `cb73dec3e4433a3b`). To recreate the clone: `git clone certify-ed_dev-3.2.bundle certify-ed`.

To recreate `archive_v5/` from the tarball: extract it and rename the top folder `CERTIFY-ED_project_archive_v5_2026-11-06`
to `archive_v5`. Its own `SHA256SUMS.txt` lists every file.
