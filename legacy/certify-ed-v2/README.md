# CERTIFY-ED: Verification Framework for Exact Diagonalization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A verification framework for exact diagonalization of quantum many-body systems, providing:

- **Symbolic Hermiticity verification** before numerical computation
- **Multi-oracle consensus validation** using independent LAPACK eigensolvers
- **Exportable verification certificates** with SHA-256 integrity protection
- **Comprehensive benchmarking suite** including QuSpin cross-validation
- **Master runner script** that produces a single archive of all results

## Installation

```bash
git clone https://github.com/sarangvehale/certify-ed.git
cd certify-ed
pip install -e .
```

## Quick Start

```python
from certify_ed import build_tfim, MultiOracle, Certificate

# Build 4-site TFIM
H = build_tfim(n_sites=4, J=1.0, h=0.5)

# Diagonalize with multi-oracle consensus
oracle = MultiOracle()
evals, evecs, report = oracle.diagonalize_with_consensus(H)

# Generate verification certificate
cert = Certificate(evals, evecs, H, metadata={'model': 'TFIM'})
cert.save('result.json')
print(cert.summary())
```

## Run All Benchmarks

To generate complete benchmark data for publication:

```bash
python run_all_benchmarks.py
```

This produces a `.tar.gz` archive with:
- Analytic validation results
- QuSpin cross-validation (if installed)
- Performance scaling data
- Platform reproducibility data
- Error injection tests
- Generated figures
- Aggregated manuscript data

## Documentation

- [Code Documentation](docs/CODE_DOCUMENTATION.md) — Complete API reference

## Examples

- `examples/example_01_basic_workflow.py` — Complete TFIM workflow
- `examples/example_02_validation.py` — Validation against analytic solutions

## Testing

```bash
pytest tests/ -v
```

## License

MIT License
