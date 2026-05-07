# CERTIFY-ED

[![DOI:10.5281/zenodo.20066566](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.5281/zenodo.20066566)
**Verified exact diagonalization for quantum many-body systems.**

CERTIFY-ED is a Python framework for exact diagonalization (ED) of quantum
many-body Hamiltonians that ships with **thirteen independent validation
layers**. The goal is not to be the fastest ED code, but to produce
results that come with strong, machine-checkable certificates of
correctness.

## What's in the box

- **16 physics models** — TFIM, Heisenberg, XXZ, XYZ, SSH, J1–J2, Majumdar-Ghosh,
  cluster model, free fermions, Kitaev chain, Hubbard, AKLT, Haldane,
  2D TFIM, 2D Heisenberg, Kitaev honeycomb.
- **Multi-oracle eigendecomposition** — NumPy LAPACK-DSYEVD, SciPy LAPACK-DSYEVD,
  SciPy LAPACK-DSYEVR, plus an ARPACK Lanczos sparse oracle for cross-checks.
- **13 independent validators** — analytic, QuSpin, mpmath arbitrary-precision,
  sparse-vs-dense, Jordan-Wigner free-fermion, sum rules, orthonormality,
  unitarity, conservation laws, symmetry-resolved spectra, thermal limits,
  finite-size scaling, error-injection.
- **SHA-256 verified certificates** — JSON output with hash chain so any
  downstream consumer can detect tampering.
- **Master runner with embedded pytest** — single command runs the whole
  pipeline, streams progress to the terminal, and exports JSON + figures
  - a tar.gz archive.

## Quick install

```bash
git clone <repo>
cd certify-ed
pip install -e .
```

Optional extras:

```bash
pip install -e .[validation]      # QuSpin (Python 3.10 only)
pip install -e .[high_precision]  # mpmath
pip install -e .[plotting]        # matplotlib
pip install -e .[dev]             # pytest
```

## One-command verification

```bash
python run_all_benchmarks.py
```

This runs all 53 pytest tests, all 13 validators (≈80 individual checks),
generates four figures, and packages everything into a timestamped
`results/run_YYYYMMDD_HHMMSS.tar.gz` you can attach to a paper submission.

Total run time: about 30 seconds on a modern laptop.

## Three-line usage

```python
from certify_ed import build_model, MultiOracle, Certificate

H = build_model('heisenberg', n_sites=6)
evals, evecs, report = MultiOracle().diagonalize_with_consensus(H)
Certificate(evals, evecs, H, consensus_report=report).save('cert.json')
```

## Why thirteen validators?

A single comparison with one external code is one data point. If both codes
share a bug — say, a common LAPACK driver — neither will catch it. Real
verification needs many _independent_ checks that cannot all fail in the
same way. The thirteen validators here cover algebraic invariants, analytic
limits, alternative algorithms, alternative arithmetic precisions,
alternative codes, conservation laws, dynamical consistency, asymptotic
limits, and error injection. Each catches a different class of failure.

The full failure-mode coverage matrix is in `docs/VALIDATORS.md`.

## Repository layout

```
certify-ed/
├── certify_ed/                 Core package
│   ├── hamiltonian.py          16 model builders
│   ├── oracles.py              Multi-oracle consensus
│   ├── certificates.py         Hashed JSON certificates
│   ├── observables.py          Expectation, thermal averages
│   └── symmetries.py           Conservation operators
├── validators/                 13 independent validation layers
├── tests/                      Pytest suite (53 tests)
├── examples/                   Usage examples
├── docs/                       API + validator documentation
├── run_all_benchmarks.py       Master pipeline runner
├── QUICKSTART.md               Five-minute introduction
└── README.md                   You are here
```

## Testing

```bash
pytest tests/ -v
```

53 tests covering Hamiltonian construction (parameterised over all 16
models), oracle consensus, certificate hashing and tamper detection,
symmetry operators, and validator smoke tests.

## License

MIT. See `LICENSE`.

## Citation

This package was built to support the manuscript **"CERTIFY-ED: A Symbolic
Validation Framework for Exact Diagonalization of Quantum Many-Body
Systems."** A draft is being prepared for submission.
