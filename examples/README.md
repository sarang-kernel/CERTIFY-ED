# Examples

| Script | Shows | Needs |
|---|---|---|
| `01_certify_quspin.py` | certify a QuSpin result (S1–S4) and save the certificate | quspin |
| `02_exact_recheck.py` | S6 exact re-check for a spin-1 chain; exact sector counts at k = 0 | python-flint |
| `03_whole_spectrum.py` | whole-spectrum enclosure by sectors; the symveig 1.0.0 API | — |
| `04_applications.py` | AKLT star criterion; spin-1 pyrochlore printed-coupling check | python-flint |

Each runs in seconds. `tests/test_examples.py` runs them all.
