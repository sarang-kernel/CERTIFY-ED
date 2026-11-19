# Provenance and the computational source hash

Every record of certify-ed says which code produced it. The rule, decision D-44 of 18 October 2026, is that a version
the paper cites needs one full run at a single computational source, and the mechanism that enforces it is the source
fingerprint of `certify_ed/provenance.py`. This document explains the fingerprint, the checks that use it, the lessons
the evidence runs taught about it, and how to recompute it from the repository. Facts are from
`code/certify-ed/certify_ed/provenance.py`, `code/certify-ed/docs/reproducibility.md` and the run manifests in
`code/certify-ed/results/`.

## The fingerprint

`source_fingerprint()` returns five fields. `git_commit` is the commit of the checkout. `source_hash` is the first 16
hexadecimal digits of the SHA-256 of the output of `git ls-files -s -- certify_ed experiments`, with the files of the
list `NON_COMPUTATIONAL` excluded; each output line holds the file mode, the blob hash and the path, so the hash
changes exactly when the content of a computational file changes, is added or is removed. `source_dirty` says whether
`git status --porcelain` reports uncommitted changes in those paths. `tests_hash` and `tests_dirty` are the same over
`tests/`. git is always called with `core.filemode=false`, so a lost executable bit (common on Windows checkouts) does
not count as a change; this was finding IMP-05 of the review of 21 October 2026.

The list `NON_COMPUTATIONAL` names the files that cannot change a status, a bound or a count. At tag `v3.1.0` it had
12 entries; at `v3.2.0` and `v3.2.1` it has 27. They are the command line (`cli.py`), the test helpers (`testing.py`),
the provenance module itself, the presentation and convenience modules added in 3.2 (`api.py`, `report.py`,
`diagnostics.py`, `viewer.py`, `schema.py`, `conformance.py`, `recorder.py`, the three files of `integrations/`,
`extrapolation.py`), the post-processing scripts that only read records (`make_numbers.py`, `make_numbers_apps.py`,
`e6_classify.py`, `e6_recheck.py`, `compare_runs.py`, `make_result_candidate.py`, `make_survey_table.py`,
`make_scope_table.py`, the folders `experiments/survey` and `experiments/figures`), and the documentation tooling
(`docs/cookbook/snippets`, `docs/make_api.py`, `mkdocs.yml`). The comments in the list give the reason for each group.
Modules whose output is a claimed certified number stay in the fingerprint even when they look like reporting code:
`gaps.py`, `crossings.py` and `levelstats.py` are deliberately not listed.

Two records with the same `source_hash` and `source_dirty = false` ran the same computational code, even if their
commits differ. A commit that changes only documentation, tests, post-processing or the command line therefore leaves
a run valid, and the 3.2.1 run stayed valid through the non-computational commits made after it (the number-script
change `44d318c5`, the figures and the result files).

## Where the fingerprint is recorded and checked

Every experiment record writes the fingerprint into its `environment.json`, every step of `manifest.json` carries it,
and every certificate carries the version of `certify_ed` and the `source_hash` in its `environment` block. The number
scripts `make_numbers.py` and `make_numbers_apps.py` refuse records with more than one `source_hash`, with
`source_dirty = true`, or an E6 record that ran against uncommitted tests. The numbers step records its own provenance
(`numbers_fingerprint`: the commit, the blob hash and the content hash of each number script, whether either is
modified, and the fingerprints at that time), so the commit of the scripts that produced a set of manuscript macros
can be read from the run directory (finding IMP-06). The test counts of the manuscript come from the run's own tests
step, never from a test collection on the machine that regenerates the numbers (IMP-07). The post-run files of an E6
record (`classification.json`, `recheck.json`, `inspection.json`) must carry a provenance stamp; a hand-written file
without one makes `make_numbers_apps.py` report a failed check (MS-N03). `--resume` uses the same rule to keep
finished steps, and re-runs E6 when `tests_hash` changed.

## Lessons from the evidence runs

The rule replaced an older check that all records came from one commit. That check failed spuriously when the full run
of 18 October 2026 crossed a commit that changed only the command line, which is why D-44 moved from commits to the
content of the computational files.

The fingerprint of each step is computed by the running driver with the `NON_COMPUTATIONAL` list it imported when it
started, over the working tree at the start of that step, and `git status --porcelain` counts untracked files as
uncommitted changes. Two consequences follow. Committing a change to the list while a run is in flight changes the
hash of later steps, and creating any file under `experiments/` while a run is in flight marks later steps dirty;
either makes the checks "one computational source" and "ran without uncommitted changes" fail. The 3.1.0 and 3.2.1
evidence runs were therefore each started from one commit with no edits in flight, and the documents of this
repository were written outside the code repository for the same reason: nothing here changes `code/certify-ed/`, so
its source hash at `a814d28` is still `cb73dec3e4433a3b`.

The history rewrite of decision D-48 changed every commit hash quoted before 21 October 2026. The file
`code/certify-ed/HASH_MAP.csv` maps each old hash to its new hash, so a hash quoted in an older record can still be
resolved.

## The hashes of the cited and the evidence versions

| Tag | Commit | `source_hash` | `tests_hash` | Entries in `NON_COMPUTATIONAL` | Run with this source |
|---|---|---|---|---|---|
| `v3.1.0` | `e860a83` | `acbef11af991c860` | `37eb92860974bd3d` | 12 | `run_full_20261022` |
| `v3.2.0` | `55d9f61` | `47446976d4dbfdb1` | `e21564ee2f7be8d3` | 27 | `run_full_20261104` |
| `v3.2.1` | `f3c478b` | `cb73dec3e4433a3b` | `9da2ba19ca572962` | 27 | `run_full_20261105` |
| `dev/3.2` head | `a814d28` | `cb73dec3e4433a3b` | `9da2ba19ca572962` | 27 | none (same source as `v3.2.1`) |

The hashes in this table were recomputed from the commits by `tools/codeindex/source_hash.py` of this repository, and
each agrees with the `source_hash` and `tests_hash` in the `manifest.json` of the run named in the last column. The
script reads `NON_COMPUTATIONAL` from `provenance.py` as committed at the given revision, lists the blobs with `git
ls-tree`, writes them in the form of `git ls-files -s` and hashes them in index order; it never checks out a file or
touches the index, so it can be run on the read-only source tree. The command is in `docs/project/reproduction/README.md`.
