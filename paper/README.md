# The combined paper

This folder holds the LaTeX source of the combined paper on CERTIFY-ED (`certify_ed_paper.tex`), its bibliography
(`certify_ed_refs.bib`), the two figures that are not in `docs/figures/` (`m6_spectral_bounds.png`, the Heisenberg ring
of 25 November 2026, and `m6_square_lattice.png`, the J1-J2 model on the 4 x 4 torus at J2 = 0.5 of 26 November 2026,
both drawn from the records of the M5 and M6 experiments in `docs/results/RESULT_m6_spectral_bounds.md`) and the
compiled PDF.
Every number in the text is copied from a record of this repository, which the text names where the number is used.
The evidence numbers are those of the 3.2.1 evidence run, and the 3.2.3 run reproduces every status.

To build the PDF, run `pdflatex certify_ed_paper`, `bibtex certify_ed_paper` and `pdflatex certify_ed_paper` twice in
this folder. The other eight figures are read from `../docs/figures/`. The packages used are amsmath, amssymb,
amsthm, mathtools, graphicx, booktabs, multirow, array, geometry, xcolor, hyperref, url and lmodern.

The source has two switches near the top, `\showSone` for the spin 1 crown finding and `\showCrownComparison` for the
comparison of the nine-coupling enclosure with the printed digits of arXiv:2601.07800. Both are on. A copy for a
publisher is a single folder with the source, the bibliography and the ten figure files, with the comment lines and
the switch definitions removed and each switched block resolved as the switch is set.
