---
name: I could not certify X
about: A quantity or a claim the tool would not certify; feeds docs/SCOPE_MATRIX.md and the reason codes
title: "Could not certify: "
labels: scope
---

<!-- Fill in what you can. This template exists so that the scope matrix stays a record of what labs
     actually try to certify. An honest "the tool said NOT_CHECKED and I do not know why" is a useful report. -->

**What was computed**

The quantity (ground energy, a gap, a correlation function, an entanglement entropy, a dynamical
response, ...), the model (or the `spec.json`), the dimension, the producer code, and whether
eigenvectors were available.

**What you asked the tool to do**

The command line or the `Options` (preset, `--only/--skip`, budgets), and the claim kind
(`subset`, `lowest`, `window`, `all`).

**What the certificate said**

For each statement that did not come out CERTIFIED: its id, status and summary, and the reason code from
the scope block (`scope.out_of_scope[].reason_code` / `scope.narrowed[].reason_code`), e.g.
`S3.completeness NOT_CHECKED NEEDS_EIGENVECTORS`. Attach the certificate JSON if you can (no arrays needed).

**Which row of docs/SCOPE_MATRIX.md this is**

If the quantity is in the matrix, which row, and whether the stated reason matched what happened. If it
is not in the matrix, say so: that is the finding.

**What would have been enough**

What statement, with what hypotheses, would have served your purpose (an upper bound only? an interval
with a declared gap hypothesis? a count without vectors?).

<!-- Maintainers: triage into (a) a documentation gap, (b) an engineering item (scope matrix "narrowable"
     column), (c) new mathematics (D-12; an UNVERIFIED entry in docs/theory.md before any code), or
     (d) out of scope with the reason code confirmed. Update docs/SCOPE_MATRIX.md and, when a reason
     code is missing, follow docs/SCHEMA_RFC_PROCESS.md. -->
