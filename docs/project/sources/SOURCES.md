# sources/

`archive_v5/` is the unpacked project archive v5, `CERTIFY-ED_project_archive_v5_2026-11-06.tar.gz` (47 MB, 2045
members), SHA-256 `a0aaea142448f396db5adbefd602b7000641441cb1f9bdbbb3616bdea7d1b056`. It is not tracked by git because it contains a complete git clone of the code
repository (`archive_v5/code/certify-ed/`, branch `dev/3.2` at `a814d28`, tags `v3.1.0`, `v3.2.0`, `v3.2.1`) and its run
records (`archive_v5/code/certify-ed/results/`). Every path cited in docs/ and blog/ is relative to `archive_v5/`.

`certify-ed_dev-3.2.bundle` is the same code repository as a git bundle (all branches and tags), SHA-256
`aa17a7f68bde15d7fb008ceeb0d0e1317b1622e976319e8a19d3d6bffa70a95c`. To recreate the clone: `git clone certify-ed_dev-3.2.bundle certify-ed`.

To recreate `archive_v5/` from the tarball: extract it and rename the top folder `CERTIFY-ED_project_archive_v5_2026-11-06`
to `archive_v5`. Its own `SHA256SUMS.txt` lists every file.
