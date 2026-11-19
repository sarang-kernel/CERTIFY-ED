# Contributing to CERTIFY-ED

CERTIFY-ED issues certificates; a contribution that makes a certificate say more than the mathematics
supports is worse than no contribution. The rules below exist for that reason. They are short because
they are binding.

## Setting up

```
git clone https://github.com/sarang-kernel/CERTIFY-ED
cd CERTIFY-ED
pip install -e .[dev,all]           # + QuSpin: docs/reproducibility.md, "Installing QuSpin 1.0.1"
pytest -q                           # about 8 minutes with QuSpin and python-flint; skips without them
ruff check certify_ed tests experiments conformance
```

On Windows, run pytest with `TEMP` pointing at a directory inside the checkout if the system temp
directory is on another drive (the experiments copy the tree). Targeted runs while developing
(`pytest -q tests/test_bounds.py`), the full suite once before a pull request.

## The soundness rule

No change may make a statement CERTIFIED where the previous release said less (FAILED, INCONCLUSIVE or
NOT_CHECKED) unless the change implements a theorem that is already in `docs/theory.md`, and the pull
request names it.

Every new code path that can affect a status needs a **negative control**: a test with a planted error
(a wrong eigenvalue, a perturbed matrix, a missing vector, a tampered certificate) that must *not* come out
CERTIFIED. Tests of the form "the right answer is certified" show liveness; only the planted error shows
soundness. `tests/test_adversarial.py`, `tests/test_manufactured.py` and `tests/test_schema_conformance.py`
have examples; `experiments/e1_failure_matrix.py` is the systematic version.

Floating-point bounds use the helpers in `certify_ed/fp.py` (`add_up`, `sub_down`, `up`, `gamma`), never
raw round-to-nearest arithmetic in a bound. A bound computed with `+` is a bug even when the test passes.

## The mathematics rule (D-12)

You may not introduce a new theorem, lemma or bound in code. Everything implemented rests either on a
statement in `docs/theory.md` (Theorems 1-3, Lemma 4, Corollary 5, Theorems 6-8, Propositions 7, 9, 13-16,
Lemma 17) or on a classical result cited by name (Weyl, Kato-Temple, Davis-Kahan, Sylvester's law of
inertia, Fannes-Audenaert, ...). A new statement goes into `docs/theory.md` first, marked **UNVERIFIED**
with the derivation written against the code, and stays marked until a second person has re-derived it.
Code may implement an UNVERIFIED statement only if the statement record it writes says so in
`assumptions` (as S7 and S8 do today). If a step would need new mathematics and you cannot write the
derivation, stop and open an issue: a blocked item with a reason is a result, a silent approximation is not.

## Scope and reason codes

Statuses are `CERTIFIED`, `FAILED`, `INCONCLUSIVE`, `NOT_CHECKED` (and `CORROBORATED` for S5 only). A new
reason for not checking something is a new entry in `certify_ed/scope.py` `REASON_CODES`, a row in
`docs/SCOPE_MATRIX.md`, and a schema minor version (`docs/SCHEMA_RFC_PROCESS.md`). Every statement id
appears exactly once in a certificate's scope block; `tests/test_presets_scope.py` and the structural
checks in `certify_ed/schema.py` enforce it.

## Provenance

`certify_ed/provenance.py` `NON_COMPUTATIONAL` lists the files that cannot change a computed number
(command line, test helpers, schema validation, post-processing scripts). Put presentation and tooling
modules there; leave computational modules out, so that the `source_hash` recorded in certificates
changes when, and only when, a number can.

## Commits and pull requests

* One concern per commit. The message says what changed and why, in plain prose; the first line is a
  summary under 100 characters, the body explains the reason and names the tests added.
* No trailers (co-author lines, sign-off lines, tool stamps): authorship is the git author field.
* A pull request that touches a status-affecting path lists: the theorem it rests on, the negative control
  it adds, and the result of the full suite on your machine.
* Documentation and scaffolding are welcome without tests; manuscript prose is the authors' (D-16).
* Register of prose: plain, precise, no marketing language, no emoji.

## Reporting a problem

Use the issue templates: a bug (what you ran, what the certificate said), a feature, or
*I could not certify X* (`.github/ISSUE_TEMPLATE/could-not-certify.md`), which feeds `docs/SCOPE_MATRIX.md`.
A soundness problem (a CERTIFIED statement you can show to be false) should be reported directly to the
maintainers (`GOVERNANCE.md`) before it is public, so that a fixed version can be released with it.
