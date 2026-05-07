# CERTIFY-ED: Certified Exact Diagonalization

A Python framework for verified exact diagonalization of quantum many-body systems with multi-oracle consensus validation and exportable verification certificates.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

CERTIFY-ED applies established numerical verification techniques to exact diagonalization workflows, providing:

- **Symbolic Hermiticity verification** before numerical computation
- **Multi-oracle consensus validation** comparing independent eigensolvers
- **Exportable verification certificates** with eigenvalues, residuals, and metadata
- **Systematic benchmarking** against analytic solutions

## Installation

```bash
git clone https://github.com/sarangvehale/certify-ed.git
cd certify-ed
pip install -e .
```

## Quick Start

```python
from certify_ed import build_tfim, MultiOracle, Certificate

# Build 4-site transverse-field Ising model
H = build_tfim(n_sites=4, J=1.0, h=0.5)

# Diagonalize with multi-oracle consensus
oracle = MultiOracle()
eigenvalues, eigenvectors, consensus = oracle.diagonalize_with_consensus(H)

# Generate verification certificate
cert = Certificate(eigenvalues, eigenvectors, H)
cert.save('results.json')
print(cert.summary())
```

See `examples/` for complete workflows and `README.md` for full documentation.

## License

MIT License - see LICENSE file.
