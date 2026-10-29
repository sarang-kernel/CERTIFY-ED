# Offline certificate viewer

`certify-ed view CERT.json [-o CERT.html]` writes one HTML file from a saved certificate (and its
`.arrays.npz` sidecar when present). The page is self-contained: inline CSS, no scripts, no external
resources; it opens offline in any browser and can be attached to a paper's supplementary material next
to the certificate. The same page comes from Python:

```python
from certify_ed.viewer import view_file
view_file("result.json", "result.html")                 # {'html', 'integrity', 'schema', 'notes'}
```

## What the page shows

| section | content | source in the certificate |
|---|---|---|
| banner | INTEGRITY VERIFIED, or INTEGRITY CHECK FAILED with the reason; SCHEMA VALIDATION FAILED when the body does not conform | `Certificate.verify_integrity()`, `verify_scope()`, `schema.validate_certificate_body` |
| header | model (name, sites, parameters), operator kind and dimension, claim kind and k, package version, hashes (body, spec, operator, eigenvectors, computational source, commit), run environment, floating-point probe, no-underflow scan, sidecar arrays | `subject`, `claim`, `environment`, `integrity`, `arithmetic`, `arrays` |
| statements | one row per statement: id and a plain-words title, status and its meaning, the summary, the next action, key evidence | `statements[]` (`summary`, `evidence`, `evidence.next_action`) |
| scope | preset; statements in scope; statements narrowed by the user with the reason given and the reason code; statements not checked with the reason code and its explanation from the scope matrix; quantities out of scope of this version; budgets; options | `scope` (docs/SCOPE_MATRIX.md) |
| enclosures | one row per claimed pair: claimed value, radius, existence radius (producer operator), the same radii with the certified model shift (sidecar, what the methods paragraph quotes), enclosure interval, cluster, distinctness, residual bound | `enclosures[]`, `arrays.radius`, `arrays.existence_radius` |
| Kato-Temple | pairs whose Kato-Temple interval is narrower than the residual enclosure, with the isolation interval | `kato_temple[]` |
| completeness | lower bound on the next eigenvalue, gap, certified window, inertia counts at each shift | `statements[S3].evidence` |
| clusters | S4: separation, sin-theta and projector error bounds per cluster; S8: level enclosures, brackets, counts | `statements[S4|S8].evidence.clusters` |
| S7, S6 | thermodynamic intervals per temperature; exact enclosures and counts | their evidence |
| methods | the methods paragraph of `certify_ed.report` and the BibTeX entry | computed from the certificate |
| integrity report | body hash (stored and, on a mismatch, recomputed), array hashes, scope block, schema validation, the not-verified list | as above |

## Numbers

Every number on the page is a value of the certificate, rounded for display by `report.round_decimal`
in the direction that cannot overstate the result:

* rounded **up**: radii, existence radii, residual bounds, error bounds (sin-theta, projector, eta, rho),
  factorization error bounds beta, the model shift, the unit roundoff, widths, upper ends of enclosures;
* rounded **down**: gap bounds, lower bounds on the next eigenvalue, separation bounds, lower ends of
  enclosures;
* rounded **inward**: certified windows and isolation intervals; rounded **outward**: level enclosures,
  brackets and thermodynamic intervals;
* printed **exactly** (`repr`): claimed values, shifts, counts, estimates, thresholds.

Each number element carries the exact value in a `data-exact` attribute (`<td class="num up"
data-exact="1.2878781504338143e-14">1.288e-14</td>`), which is what `tests/test_viewer.py` compares
with the certificate for every number of every fixture certificate. The column headers say the
direction.

## Integrity

`view` loads the file without raising: a modified body, a modified or missing sidecar, or a scope block
that no longer accounts for every statement produces the page with a red banner at the top and the exit
code 1; the footer shows the stored and the recomputed body hash. A hash shows that the file was not
modified after it was saved; it does not show that the computation behind it was honest (see
*What certified means*). Re-running `certify` on the stored claim is the check that does not trust the
producer.

User-supplied strings (the narrowing reason, the claim description, producer records) are HTML-escaped.

`certify_ed/viewer.py` reads a certificate and writes text; it is listed in
`provenance.NON_COMPUTATIONAL` and cannot change a number.
