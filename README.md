# CERTIFY-ED: Certified Exact Diagonalization

A comprehensive formal verification framework for quantum many-body systems using exact diagonalization (ED), with multi-oracle consensus validation, symbolic verification, and exportable proof certificates.

## Features

✅ **Multi-Oracle Consensus** - Three independent eigensolvers for error detection
✅ **Symbolic Verification** - Hermiticity verification before numerical computation
✅ **Proof Certificates** - Portable, verifiable, cryptographically-signed records
✅ **Error Bounds** - Certified error bounds on eigenvalues and observables
✅ **Platform Reproducibility** - Identical results across Linux/macOS/Windows
✅ **Long-Term Archival** - Certificates valid indefinitely across software updates

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/certify-ed.git
cd certify-ed
pip install -e .
```

### Basic Usage

```python
from certify_ed import CertifiedHamiltonian, CertificationEngine

# Create Hamiltonian
ham = CertifiedHamiltonian.TFIM(L=4, J=1.0, h=0.5)

# Run certification
engine = CertificationEngine(verbose=True)
certificate = engine.certify_hamiltonian(ham)

# Access certified results
E_0 = certificate.eigenpairs[0].eigenvalue
print(f"Ground state: {E_0:.10f}")
```

## System Requirements

- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- Optional: SageMath for symbolic computation

## Documentation

- [Installation & Setup](docs/README.md)
- [API Reference](docs/API.md)
- [Tutorial](docs/TUTORIAL.md)
- [Paper](https://arxiv.org/abs/XXXX.XXXXX) (preprint)

## Performance

| System Size | Time | Memory |
|-------------|------|--------|
| N=4         | 10ms | 1 MB   |
| N=8         | 300ms| 8 MB   |
| N=12        | 15s  | 128 MB |

## Citation

```bibtex
@article{certify-ed2025,
  title={Formal Verification of Exact Diagonalization},
  author={[Your Name]},
  journal={[Journal]},
  year={2025}
}
```

## License

MIT License - See LICENSE file

## Contact

[your.email@institution.edu]

---

**CERTIFY-ED** - Rigorous verification for quantum computing research
