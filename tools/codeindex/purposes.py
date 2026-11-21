"""Hand-written sentences for docs/project/CODE_INDEX.md, used by make_code_index.py instead of the docstring's first sentence.

A file is listed here when it has no module docstring, when the first sentence of its docstring is only a heading
or too vague to say what the file does, or when that sentence uses a word the documentation's style rules exclude
(tools/lint_docs.py). Each sentence was written by reading the file at commit a814d28.
"""

PURPOSE = {
    # no module docstring
    "experiments/run_all.sh": (
        "Legacy shell driver that runs the experiments in two single-threaded queues (E2, E4 and E5 one after "
        "another, E0, E1 and E3 beside them) and leaves `make_numbers.py` to be run afterwards; replaced by "
        "`certify-ed reproduce`."),
    "legacy/applications_2026-10-17/aklt_gap/cross_certify.py": (
        "Script that certifies eta for the cross lattice [4.6.12] with `pw_general.certify_eta_blocks` at two "
        "thresholds and writes `cross_certificate.json`."),
    "legacy/applications_2026-10-17/aklt_gap/so_certify.py": (
        "Script that certifies the two eta values of the square-octagon lattice (parallel and perpendicular unions) "
        "with `pw_general` and evaluates the joint criterion."),
    "legacy/applications_2026-10-17/pyrochlore_crown/crown_opt.py": (
        "Grid search around the integer couplings (381414, 817957) of a crown record for the couplings with the "
        "largest lowest eigenvalue (sparse `eigsh` on the S^z = 0 block); prints the best grid point."),
    "legacy/applications_2026-10-17/pyrochlore_crown/run_exact.py": (
        "Script that runs `crown_certify.exact_check` on a stored crown record and writes the exact-check lower "
        "bound to a JSON file."),
    "tests/conftest.py": (
        "Test configuration: puts the repository on the import path, defines the `needs_quspin` and `needs_flint` "
        "skip markers, and re-exports the local spaces and the random model generator of `certify_ed.testing`."),
    "tests/test_label_audit.py": (
        "Tests of `certify_ed.sectors.audit_label`: QuSpin momentum blocks are labelled correctly and a mislabelled "
        "block is detected."),
    "tests/test_total_spin.py": (
        "Tests of the structural SU(2) check (`certify_ed.su2`) and of the total-spin labels of S4 clusters, "
        "including the spin-1 triplet and singlet."),
    "verifier/src/cev.c": (
        "The independent C verifier: recomputes the hashes, the S2 radii from the stored vectors and operator, "
        "the S3 count and ordering, and the consistency of the recorded statuses, with directed rounding and no "
        "code shared with certify_ed."),
    # style rules of the documentation
    "certify_ed/spectrum.py": (
        "Enclosure of the whole spectrum of a Hermitian matrix, optionally sector by sector: the symveig "
        "functionality rebuilt on the CERTIFY-ED engine, with three corrections."),
    "experiments/figures/fig_crown_sectors.py": (
        "Figure of the pyrochlore-crown S = 1 pair-spin classes (lowest eigenvalue of each class at the "
        "re-optimised couplings) and of the named sectors at the printed couplings."),
    "experiments/harness.py": (
        "Shared reproducibility layer of all experiments: each run writes a record directory with `data.json`, "
        "`environment.json`, `run.log` and figures."),
    "legacy/symveig_1_0_0/symveig/certify_eig.py": (
        "symveig 1.0.0 core: eigenvalue enclosures for Hermitian matrices, with optional sector decomposition under "
        "a commuting diagonal observable."),
    "legacy/symveig_1_0_0/symveig/__init__.py": (
        "symveig 1.0.0: eigenvalue enclosures for symmetry-decomposed Hermitian matrices, kept for the record."),
    # first sentence too vague on its own
    "experiments/e4_workflows.py": (
        "E4: four research workflows: W1 a certified low-energy spectrum assembled from QuSpin symmetry blocks, W2 a "
        "degenerate multiplet with certified quantum numbers, W3 a matrix-free producer at d = 2^20, W4 certified "
        "reference values for commonly used benchmark systems."),
    "experiments/e5_arpack_multiplet.py": (
        "E5: an error that nobody planted: ARPACK, asked through QuSpin for the eight lowest states of the 4x4 "
        "Heisenberg S^z = 0 block, sometimes returns three of the four members of a fourfold level; the script "
        "records what certify-ed states for seeded start vectors."),
    "experiments/e3_tightness.py": (
        "E3: how tight the certified statements are: radius and Kato-Temple interval against the actual error, "
        "float64 against interval-arithmetic residuals, the bound on the next eigenvalue against the true one, and "
        "the sector-composition radius against the actual deviation."),
    "certify_ed/__init__.py": (
        "Package entry point of CERTIFY-ED (certificates for exact-diagonalization results): the public names "
        "of the package."),
    "legacy/symveig_1_0_0/symveig/benchmark.py": (
        "symveig 1.0.0 benchmark engine: global and sector-decomposed enclosures of one Hamiltonian, the per-sector "
        "breakdown, reference eigenvalues, timings and the width and speed ratios, as a dictionary for JSON."),
    "legacy/symveig_1_0_0/symveig/cli.py": (
        "symveig 1.0.0 driver (`run.py` of that package): the test suite and the benchmarks of every model, "
        "with JSON results, a summary table and the figures."),
    "legacy/v1/run_all_benchmarks.py": (
        "Driver of version 1: the pytest suite, the independent validators across the physics models, and "
        "the export of the results as JSON, figures and an archive."),
    "legacy/v1/validators/analytic_validator.py": (
        "Validator of version 1 that compares with closed-form results for small models (Heisenberg clusters, "
        "transverse-field Ising limits, free fermions, the Majumdar-Ghosh point, AKLT and others)."),
    "legacy/v1/validators/orthonormality_validator.py": (
        "Validator of version 1 that checks orthonormality and completeness of the eigenvectors and the spectral "
        "decomposition H = V D V^dag."),
    "legacy/v1/validators/thermal_validator.py": (
        "Validator of version 1 that checks the known high-temperature and low-temperature limits of thermal "
        "averages."),
    "legacy/v1/validators/unitarity_validator.py": (
        "Validator of version 1 that checks that the time-evolution operator built from the eigendecomposition is "
        "unitary and has the group property."),
    # first docstring sentence is a heading
    "certify_ed/precision.py": (
        "Exact conversion of stored operators, vectors and matrices to binary64 (`binary64`, `require_binary64`) and "
        "the checks of integer and object values, applied by every entry point before a product or a format "
        "conversion."),
}
