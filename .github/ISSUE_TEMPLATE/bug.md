---
name: Bug
about: Something fails, raises, or produces a certificate that contradicts itself
title: ""
labels: bug
---

**What you ran**

The command line or the Python call (`certify(...)`, `Options(...)`), and how the claim was produced
(a producer bundle, `claim_from_quspin`, arrays).

**What the certificate said**

The relevant statements with their status and summary (`cert.summary()`), or the traceback.

**What you expected**

**Environment**

Paste the output of `certify-ed env` (versions, floating-point probe).

**Minimal reproduction**

A bundle directory or a few lines building the operator and the claim; the smaller the dimension the
better. If the problem depends on your data and you cannot share it, say what shape and kind it has.
