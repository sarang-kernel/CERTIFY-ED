# Governance

## Maintainers

| role | who | contact |
|---|---|---|
| lead maintainer, release manager, mathematics (D-12 second reader for code written by others) | Sarang Vehale | sarangvehale2@gmail.com |
| co-author, mathematics | Ritu Goel | through the lead maintainer |

Maintainers merge pull requests, cut releases and sign them (`packaging/SIGNING.md`). A contributor
becomes a maintainer by invitation after sustained contributions that passed review, recorded in this
file by a pull request.

## Decision process

* Ordinary changes (bug fixes, documentation, new importers, tooling) are decided by the reviewing
  maintainer; one approval is enough. The contributor may not approve their own change.
* **Status-affecting changes** (anything that can turn a statement CERTIFIED, or change a bound) need
  the lead maintainer's approval and the negative control described in `CONTRIBUTING.md`.
* **New mathematics** follows D-12: written in `docs/theory.md` as UNVERIFIED, re-derived by a second
  person, and only then relied on in a release note. A disagreement about a derivation is resolved by
  writing it out, not by majority.
* **Format changes** (certificate schema, producer bundle, scope vocabulary) go through
  `docs/SCHEMA_RFC_PROCESS.md`.
* Anything else contentious is discussed in an issue and decided by the lead maintainer, who writes the
  reason into the issue before closing it.

## Release cadence

* A **patch** release (x.y.Z) whenever a known issue of the current release has a fix and its own run
  (`CHANGELOG.md` known-issues sections), typically within weeks.
* A **minor** release (x.Y.0) when a wave of the roadmap (`docs/roadmap/ROADMAP.md`) lands, roughly every three to
  six months, with a full evidence run (D-44) on the reference stack and a two-stack comparison.
* A **major** release (X.0.0) only for a certificate-schema major version or a change of what
  CERTIFIED means; it is announced one minor release ahead.

Every release follows `packaging/RELEASE_CHECKLIST.md`. Tags are never moved; a bad release gets a
successor.

## Deprecation policy

* **Certificate schema**: within a major version a field is never removed or re-typed and an enumeration
  value is never removed. New fields are optional. A field that is no longer written is kept in the schema,
  documented as such in `docs/schema.md`, and still accepted by `Certificate.load`. Readers of 2.x
  certificates written by 3.1 must be able to read every 2.x certificate written later.
* **Python API**: names exported from `certify_ed` (the `__all__` list) are kept for at least two minor
  releases after a deprecation warning is added, with the replacement named in the warning and in the
  CHANGELOG. `certify_ed.compat` holds the API of absorbed packages (symveig 1.0.0) indefinitely.
* **Command line**: a subcommand or flag is deprecated the same way (a warning for two minor releases);
  its output format is part of the API when it is JSON.
* **Reason codes and statement ids**: never removed; a retired code stays in `REASON_CODES` with its
  documentation marked retired.
* **Reproduction**: a release's evidence run must remain reproducible with that release's container
  recipe (`containers/`); the recipes of past releases are kept under their tag.

## What governance does not decide

Whether a statement is CERTIFIED is decided by the mathematics and the code, not by a vote. A maintainer
who believes a CERTIFIED statement is unsound has the standing to downgrade it (to INCONCLUSIVE with a
reason) in a patch release without waiting for consensus; restoring it needs the derivation.
