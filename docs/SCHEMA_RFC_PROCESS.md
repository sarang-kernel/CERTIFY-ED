# Changing the certificate format: the RFC process

The certificate (`schema/certificate-2.1.schema.json`), the producer bundle
(`schema/producer-bundle-1.0.schema.json`), the model spec (`certify-ed/model-spec/1.0`) and the scope
vocabulary (`certify_ed/scope.py`) are read by people and programs that are not this repository. A change
to any of them is a change to a published interface and goes through the steps below. `docs/schema.md`
states the stability promise the process protects.

## When an RFC is needed

| change | version effect | RFC |
|---|---|---|
| a new optional field, a new evidence key | none (2.1 readers ignore unknown fields) | no: a pull request that updates `certify_ed/schema.py`, `docs/schema.md` and the rendered files |
| a new enumeration value: statement id, status, reason code, claim kind | minor (2.1 -> 2.2) | yes, short form |
| a new required field, a change of a field's type or meaning, removal of anything, a change of the canonicalization or the array hash | major (2.x -> 3.0) | yes, full form |
| a new producer-bundle field | none | no |
| a change of what an existing bundle field means | major of the bundle schema | yes |

## The RFC

An RFC is a pull request adding `docs/rfcs/NNNN-short-title.md` with these sections:

1. **Summary**: one paragraph; the field or value being added, changed or removed.
2. **Motivation**: which quantity or producer needs it; a link to the issue (often an
   *I could not certify X* report) and the row of `docs/SCOPE_MATRIX.md` it changes.
3. **Specification**: the exact JSON Schema fragment, the enumeration value with its one-line meaning (for
   a reason code: the `REASON_CODES` entry), and which code writes it.
4. **Mathematics**: for a new statement or status, the theorem in `docs/theory.md` it rests on, or the
   UNVERIFIED entry added (D-12). A status that no theorem supports is not added.
5. **Compatibility**: what a reader of the previous schema version sees; the migration, if any; the
   `Certificate.load` behaviour on old files.
6. **Alternatives considered**: including "put it in `evidence`" (which needs no RFC).
7. **Reference implementation**: the pull request with the code, the schema change, the rendered files
   (`python -m certify_ed.schema --write schema`), the documentation, and a fixture certificate that
   exercises the new field together with the negative control for any new enumeration value.

## Review

* An RFC is open for comments for at least **two weeks** (short form) or **four weeks** (full form) after
  it is announced in the repository's discussions and, for a major version, in the release notes of the
  preceding minor release.
* The lead maintainer decides (`GOVERNANCE.md`); a major-version RFC additionally needs a written
  statement from at least one external writer or reader of the format (a producer code, a verifier) that
  the change is implementable on their side.
* The decision and its reasons go into the RFC file (status `accepted`, `rejected` or `withdrawn`); an
  accepted RFC's number is cited in the CHANGELOG entry of the release that ships it.

## After acceptance

* `certify_ed/schema.py` is the source of truth; the rendered files and `docs/schema.md` are updated in
  the same commit, and `tests/test_schema_conformance.py` keeps them in sync.
* The schema version string (`certificate_v2.CERT_SCHEMA`, `scope.SCOPE_SCHEMA`, `io.bundle.META_SCHEMA`)
  is bumped; for a major version, `Certificate.load` keeps reading the previous major version and says
  in `summary()` which version it read.
* `conformance/reference/` gets a bundle or certificate that exercises the change when the change is
  visible to third-party writers.
* The reason-code table in `docs/SCOPE_MATRIX.md` is updated in the same release.
