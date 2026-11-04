# CERTIFY-ED

Certificates for exact-diagonalization results. A claimed spectrum (eigenvalues, optionally eigenvectors,
from any producer) is checked against a declared model and receives statements — model fidelity (S1),
eigenpair enclosures (S2), completeness and gap (S3), eigenspaces and labels (S4), corroboration (S5), an
exact integer re-check (S6), thermodynamics (S7), degeneracy counts (S8) — each CERTIFIED, FAILED,
INCONCLUSIVE or NOT_CHECKED with its evidence and its assumptions. This site documents version 3.2.1.

- [Quickstart](quickstart.md) — install, one call, read the certificate, write the sentence.
- [Cookbook](cookbook/index.md) — numpy arrays, `eigsh`, files, QuSpin, matrix-free, producer bundle.
- [What certified means](what_certified_means.md) — statuses, hypotheses, what is not claimed, run-stack
  values, UNVERIFIED results.
- [Limitations](limitations.md) — sizes, budgets, the exact layer, known issues.
- [FAQ: INCONCLUSIVE](faq_inconclusive.md) — every summary and reason code with its next action.
- Reference: [theory](theory.md), [scope matrix](SCOPE_MATRIX.md), [presets and scope](presets.md),
  [producer bundle](producer_bundle.md), [API](api.md), [applications](applications.md),
  [reproducibility](reproducibility.md).

Source: `github.com/sarang-kernel/CERTIFY-ED`. Preprints: arXiv:2605.11787 (v1), arXiv:2606.16217 (symveig).
