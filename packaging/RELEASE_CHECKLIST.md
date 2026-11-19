# Release checklist

A release is a tag, a full evidence run, an sdist and a wheel with checksums and signatures, a Zenodo
version under the concept DOI, and a GitHub release. Nothing on this list is a force-push; a release
that went wrong gets a new patch version, never a rewritten tag.

## 0. Before the version bump

- [ ] `main` (or the release branch) contains every merged track; `git log --oneline origin/main..HEAD` is empty.
- [ ] `ruff check certify_ed tests experiments conformance` is clean.
- [ ] `pytest -q` passes on the release machine with QuSpin and python-flint installed (the full suite,
      about 8 minutes; the number of tests and skips goes into the CHANGELOG).
- [ ] `python -m certify_ed.schema --check schema` says `in sync` for both files. A schema change was
      proposed through `docs/SCHEMA_RFC_PROCESS.md` and `docs/schema.md` was updated.
- [ ] `certify-ed conform conformance/reference/dense-arrays conformance/reference/npz conformance/reference/matrix-free`
      prints `CONFORMANT` three times.
- [ ] `docs/SCOPE_MATRIX.md` lists every reason code in `certify_ed/scope.py` and every statement.
- [ ] Every statement of new mathematics in `docs/theory.md` is either verified or marked UNVERIFIED (D-12);
      the CHANGELOG names the UNVERIFIED items.
- [ ] Known issues found after the evidence run are listed in the CHANGELOG under the version they belong to.

## 1. Version bump (one commit)

- [ ] `certify_ed/__init__.py` `__version__`, `pyproject.toml` `version`, `CITATION.cff` `version` and
      `date-released`, `.zenodo.json` `version`, `packaging/conda-forge/meta.yaml` `version`.
- [ ] `CHANGELOG.md`: the section for this version, with the test count, the computational source hash
      (`python -c "from certify_ed.provenance import source_fingerprint as f; print(f())"` on the clean tree),
      and the list of UNVERIFIED mathematics.
- [ ] Commit message: `Release X.Y.Z` plus the one-paragraph summary from the CHANGELOG.

## 2. Tag

```
git tag -a vX.Y.Z -m "certify-ed X.Y.Z"
git push origin vX.Y.Z
```

The tag is annotated and points at the version-bump commit. It is never moved: a mistake in the tagged
tree is fixed in `X.Y.(Z+1)`.

## 3. The full run (D-44)

- [ ] On the reference stack (`containers/README.md`: Python 3.11, numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0,
      QuSpin 1.0.1): `python run.py --full --with-mutation --jobs 4 --out results/run_full_<date>`.
- [ ] `certify-ed numbers --run results/run_full_<date>` regenerates every number of the manuscript
      (`make_numbers` and `make_numbers_apps` with zero CHECK FAILED); the RESULT file written by
      `experiments/make_result_candidate.py` goes into the repository as `RESULT_evidence_run_X.Y.Z.md`.
- [ ] `experiments/compare_runs.py` against the previous release's run: no MISMATCH leaf, every difference
      explained in the RESULT file.
- [ ] The run directory's `environment.json` records `source_dirty: false` and the tag's commit.
- [ ] Archive the run directory (tar.zst) with the release assets; it is the evidence, not the code.

## 4. Build the distributions

```
python -m pip install --upgrade build
rm -rf dist/
python -m build                         # dist/certify_ed-X.Y.Z.tar.gz and dist/certify_ed-X.Y.Z-py3-none-any.whl
```

- [ ] `tar tzf dist/certify_ed-X.Y.Z.tar.gz | grep -c schema/` is 2 and the sdist contains `conformance/reference/`.
- [ ] Fresh venv test: `python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl && /tmp/v/bin/certify-ed env`.
- [ ] `twine check dist/*`.

## 5. Checksums and signatures (`packaging/SIGNING.md`)

```
cd dist && sha256sum * > SHA256SUMS && cd ..
python -m sigstore sign dist/*.tar.gz dist/*.whl dist/SHA256SUMS
```

- [ ] Verify once before publishing:
      `python -m sigstore verify identity --cert-identity <release email> --cert-oidc-issuer <issuer> dist/certify_ed-X.Y.Z.tar.gz`.

## 6. Publish

- [ ] PyPI: `twine upload dist/*.tar.gz dist/*.whl` (TestPyPI first for a new major version).
- [ ] Zenodo: a **new version** of the existing record under the concept DOI 10.5281/zenodo.20066565
      (never a new concept record). Upload the sdist, the wheel, `SHA256SUMS`, the `.sigstore.json`
      bundles and the archived run directory; `.zenodo.json` supplies the metadata. Note the version DOI
      Zenodo mints and add it to `CITATION.cff` (`identifiers`) in the next commit.
- [ ] GitHub release from the tag: title `certify-ed X.Y.Z`, body = the CHANGELOG section, assets = the
      files uploaded to Zenodo. Mark pre-releases as such.
- [ ] conda-forge: update `version` and `sha256` (of the PyPI sdist) in the feedstock, or submit
      `packaging/conda-forge/meta.yaml` to staged-recipes for the first release.

## 7. After the release

- [ ] Open the next development version: `__version__ = "X.Y+1.0.dev0"` in `certify_ed/__init__.py` and
      `pyproject.toml`; CHANGELOG gets an `Unreleased` heading.
- [ ] Close the milestone; move unfinished roadmap items to the next one (`docs/roadmap/ROADMAP.md`).
- [ ] If anything on this list was skipped, say so in the CHANGELOG entry rather than in a private note.
