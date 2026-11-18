# legacy/

Code kept for provenance and for comparisons. It is not part of the installed `certify_ed` package.

* `v1/`: CERTIFY-ED v1.0.0, the code described in arXiv:2605.11787. This covers the symbolic
  Hamiltonian builders, the oracles, the 13 validators, the v1 tests and examples. It was superseded
  by the v2 certification core (2026-10); the git tag `v1.0.0-baseline` marks the import. The files are kept
  for reading, not for running: their tests import `certify_ed`, which is now the 3.x package, and the v1
  packaging and run records are not here. The complete v1.0.0 is tag `v1.0.0`, on the first-parent line of
  the commit that joined the two histories.
* `symveig_1_0_0/`: symveig 1.0.0 (Vehale & Goel, arXiv:2606.16217), unchanged. Experiment E8
  compares it with `certify_ed.spectrum` and `certify_ed.compat.symveig`, which replace it. See
  `docs/symveig_audit.md` for the problems the merge fixes.
* `applications_2026-10-17/`: the first, script-based versions of the AKLT and pyrochlore
  applications, with the certificate files and logs they produced on 2026-10-17. The applications
  now live in `certify_ed.apps.aklt` and `certify_ed.apps.pyrochlore` and are run by experiments
  E9 and E10. The unified code reproduces every certified value (see CHANGELOG 3.0.0).
