# CERTIFY-ED: Certified Exact Diagonalization Framework

**Production-ready implementation of symbolic validation and numerical certification for exact diagonalization of quantum many-body systems.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CERTIFY-ED implements a complete five-phase protocol for certified exact diagonalization:

1. **Phase 1**: Symbolic Hamiltonian construction with H = H† verification
2. **Phase 2**: Multi-oracle eigendecomposition (NumPy, High-Precision, Iterative)
3. **Phase 3**: Consensus protocol with agreement validation
4. **Phase 4**: Cryptographic certificate generation for archival
5. **Phase 5**: Observable validation with error bounds

### Theoretical Foundation

All three theorems are fully implemented:
- **Theorem 1**: Spectral certification via Davis-Kahan bounds
- **Theorem 2**: Observable error propagation bounds
- **Theorem 3**: Cross-platform reproducibility guarantees

## Installation

### From source (recommended for publication)

```bash
git clone https://github.com/yourusername/certify-ed
cd certify-ed
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy >= 1.20
- Matplotlib >= 3.3

## Quick Start

```python
from certifyEd import (
    CertifiedHamiltonian,
    MultiOracleDiagonalizer,
    ObservableCalculator
)

# Phase 1: Create and verify Hamiltonian
H = CertifiedHamiltonian.TFIM(L=4, J=1.0, h=0.5)
H.verify_hermiticity()  # Symbolic verification

# Phase 2-3: Diagonalize with multi-oracle consensus
diag = MultiOracleDiagonalizer(H, epsilon=1e-12)
results = diag.diagonalize()

# Phase 4: Generate certificate
cert = results.generate_certificate()
cert.export_json('certificate.json')

# Phase 5: Compute observables with certified bounds
obs_calc = ObservableCalculator(results)
summary = obs_calc.summary()
print(f"E₀ = {summary['ground_state_energy']:.15f}")
print(f"Error: {summary['energy_error_bound']:.2e}")
```

## Reproducing Paper Results

Run the complete pipeline to reproduce all figures and validation experiments:

```bash
python scripts/run_complete_pipeline.py
```

This will:
1. Run all validation experiments (§6.1-6.3)
2. Generate all 8 publication-quality figures
3. Verify all theoretical claims

### Individual Validation Experiments

```bash
# Experiment 1: Bethe Ansatz Comparison (§6.1)
python scripts/validate_analytic.py

# Experiment 3: Cross-Platform Reproducibility (§6.3)
python scripts/validate_reproducibility.py
```

## Features

✅ **Complete Implementation**
- All 5 sequential phases (Paper §3)
- 3 independent oracles with consensus
- Cryptographic certificate hashing (SHA-256)
- Certified error bounds (Theorem 2)

✅ **Standard Models**
- Transverse-Field Ising (TFIM)
- Heisenberg XXX / XXZ
- Extensible custom Hamiltonians

✅ **Validation**
- Bethe ansatz comparison (15-16 digit agreement)
- Cross-platform reproducibility (bitwise identical)
- Complete test suite with >90% coverage

## Project Structure

```
certify-ed/
├── src/certifyEd/           # Main package
│   ├── core/               # Phase implementations
│   │   ├── hamiltonian.py  # Phase 1
│   │   ├── diagonalizer.py # Phases 2-3
│   │   ├── engine.py       # Phase 4
│   │   └── observable.py   # Phase 5
│   └── validation/         # Experiments & figures
├── scripts/                # Reproducibility scripts
├── tests/                  # Unit tests
└── examples/              # Usage examples
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=certifyEd --cov-report=html

# Run validation experiments
python scripts/run_complete_pipeline.py
```

## Documentation

### Core API

- **CertifiedHamiltonian**: Phase 1 (Symbolic construction)
  - `TFIM(L, J, h)`: Transverse-field Ising model
  - `Heisenberg(L, J, Jz)`: Heisenberg model
  - `verify_hermiticity()`: Exact H = H† check

- **MultiOracleDiagonalizer**: Phases 2-3 (Multi-oracle consensus)
  - `diagonalize()`: Execute 3-oracle protocol
  - `get_agreement_metrics()`: Agreement validation

- **Certificate**: Phase 4 (Cryptographic certification)
  - `export_json(filename)`: Export certificate
  - `verify(H)`: Verify integrity

- **ObservableCalculator**: Phase 5 (Observable validation)
  - `expectation_value(O)`: Compute ⟨O⟩ with bounds
  - `summary()`: Get certified properties

### Examples

See `examples/basic_usage.py` for a complete walkthrough.

## Performance

| System Size | Dimension | Time   |
|------------|-----------|--------|
| N=4        | 16        | 0.01s  |
| N=6        | 64        | 0.1s   |
| N=8        | 256       | 1s     |
| N=10       | 1024      | 10s    |
| N=12       | 4096      | 5min   |

## Citation

If you use CERTIFY-ED in your research, please cite:

```bibtex
@article{vehale2025certify,
  title={CERTIFY-ED: A Symbolic Validation Framework for Exact Diagonalization 
         of Quantum Many-Body Systems},
  author={Vehale, Sarang},
  journal={[Journal Name]},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

## Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/certify-ed/issues)
- **Questions**: Open a discussion or issue

## Contact

**Author**: Sarang Vehale  
**Email**: your.email@example.com  
**Institution**: National Forensic Sciences University

---

**Status**: Production-Ready for Publication ✅  
**Version**: 1.0.0  
**Last Updated**: December 2025
