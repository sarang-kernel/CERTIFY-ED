# CERTIFY-ED: Code Documentation

Complete reference documentation for the CERTIFY-ED framework.

## Table of Contents

1. [Installation](#installation)
2. [Architecture Overview](#architecture-overview)
3. [Module Reference](#module-reference)
4. [Benchmark Suite](#benchmark-suite)
5. [Master Runner](#master-runner)
6. [Output Format](#output-format)
7. [Reproducing Results](#reproducing-results)

---

## Installation

### Requirements

- Python 3.8 or later
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- (Optional) matplotlib ≥ 3.5 for figures
- (Optional) pytest ≥ 7.0 for testing
- (Optional) QuSpin ≥ 0.3.6 for cross-validation

### Install from source

```bash
git clone https://github.com/sarangvehale/certify-ed.git
cd certify-ed
pip install -e .
```

### Install with all extras

```bash
pip install -e ".[dev,validation,plotting]"
```

---

## Architecture Overview

```
certify-ed/
├── certify_ed/                 # Core package
│   ├── __init__.py            # Public API
│   ├── hamiltonian.py         # Hamiltonian construction
│   ├── oracles.py             # Multi-oracle diagonalization
│   ├── certificates.py        # Certificate generation
│   └── observables.py         # Physical observables
│
├── benchmarks/                # Benchmark suite
│   ├── __init__.py
│   ├── analytic_benchmarks.py     # Validation vs known solutions
│   ├── quspin_validation.py       # Cross-validation with QuSpin
│   ├── performance_benchmarks.py  # Timing and scaling
│   ├── platform_benchmarks.py     # Platform reproducibility
│   └── error_injection.py         # Error detection tests
│
├── tests/                     # Unit tests
│   ├── test_hamiltonian.py
│   ├── test_oracles.py
│   └── test_certificates.py
│
├── examples/                  # Usage examples
│   ├── example_01_basic_workflow.py
│   └── example_02_validation.py
│
├── results/                   # Generated benchmark outputs
│   └── run_TIMESTAMP/         # Per-run results
│
├── run_all_benchmarks.py      # Master runner script
├── setup.py
├── requirements.txt
└── README.md
```

---

## Module Reference

### `certify_ed.hamiltonian`

Hamiltonian construction and Hermiticity verification.

#### Functions

##### `pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
Return Pauli matrices `(I, X, Y, Z)` as 2x2 complex arrays.

##### `tensor_product(*operators) -> np.ndarray`
Compute tensor (Kronecker) product of multiple operators.

##### `build_tfim(n_sites, J=1.0, h=0.5, boundary='open') -> np.ndarray`
Build transverse-field Ising model: `H = -J Σ Z_i Z_{i+1} - h Σ X_i`.

##### `build_heisenberg(n_sites, J=1.0, boundary='open') -> np.ndarray`
Build isotropic Heisenberg model: `H = J Σ S_i · S_{i+1}` where `S = σ/2`.

##### `build_xxz(n_sites, J=1.0, Delta=1.0, boundary='open') -> np.ndarray`
Build XXZ model: `H = (J/4) Σ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})`.

#### Classes

##### `SymbolicHamiltonian`

Construct Hamiltonians term-by-term with Hermiticity verification.

```python
ham = SymbolicHamiltonian(n_sites=3)
ham.add_term(-1.0, [('Z', 0), ('Z', 1)])  # -Z_0 Z_1
ham.add_term(-0.5, [('X', 0)])             # -0.5 X_0
H = ham.build()
assert ham.verify_hermiticity()
```

**Methods:**
- `add_term(coefficient, operators)`: Add a term. `operators` is a list of `(name, site_index)`.
- `build()`: Construct numerical matrix.
- `verify_hermiticity(tolerance=1e-14)`: Check `H = H†`.

---

### `certify_ed.oracles`

Multi-oracle eigendecomposition with consensus validation.

#### Classes

##### `Oracle` (abstract base)

Base class for eigendecomposition oracles. Subclasses must implement:
- `diagonalize(H) -> (eigenvalues, eigenvectors)`
- `name() -> str`

##### `NumPyOracle`

Uses `numpy.linalg.eigh` (LAPACK DSYEVD via NumPy wrapper).

##### `ScipyOracle(driver='evd')`

Uses `scipy.linalg.eigh` with selectable LAPACK driver.

**Drivers:**
- `'evd'` — DSYEVD (divide-and-conquer, fastest)
- `'evr'` — DSYEVR (relatively robust representations)
- `'ev'`  — DSYEV (classical QR)

##### `MultiOracle(oracles=None, tolerance=1e-10)`

Orchestrates multiple oracles for consensus validation.

**Default oracles:** `[NumPyOracle(), ScipyOracle('evd'), ScipyOracle('evr')]`

**Methods:**
- `diagonalize_with_consensus(H, return_all=False)`: Run all oracles, validate agreement.
  Returns `(eigenvalues, eigenvectors, consensus_report)`.
- `compute_residuals(H, evals, evecs)`: Compute `||H|ψ_n⟩ - E_n|ψ_n⟩||` for each pair.

**Consensus report contents:**
```python
{
    'consensus': bool,              # True if all oracles agree within tolerance
    'max_disagreement': float,      # Maximum eigenvalue difference
    'oracle_names': List[str],      # Names of all oracles used
    'n_oracles': int,
    'tolerance': float,
    'pairwise_max_diffs': List[List[float]]  # Pairwise max differences
}
```

---

### `certify_ed.certificates`

Verification certificate generation and validation.

#### Classes

##### `Certificate`

Generate verification certificates from eigendecomposition results.

```python
cert = Certificate(
    eigenvalues=evals,
    eigenvectors=evecs,
    hamiltonian=H,
    metadata={'model': 'TFIM', 'J': 1.0, 'h': 0.5},
    consensus_report=report
)
cert.save('result.json')
print(cert.summary())
```

**Attributes:**
- `eigenvalues`, `eigenvectors`, `H` — Input data
- `residuals` — Computed residual norms
- `normalization_errors` — `|⟨ψ|ψ⟩ - 1|` for each eigenvector
- `is_certified` — True if max residual < 1e-10 and max norm error < 1e-12

**Methods:**
- `save(filename, include_eigenvectors=True)` — Save to JSON with SHA-256 hash
- `to_dict(include_eigenvectors=True)` — Convert to dictionary
- `summary() -> str` — Human-readable summary

#### Functions

##### `load_certificate(filename, verify_hash=True) -> dict`

Load and optionally verify a certificate. Raises `ValueError` if hash mismatch.

---

### `certify_ed.observables`

Physical observable computation.

#### Classes

##### `ObservableCalculator(eigenvalues, eigenvectors, residuals=None)`

Compute expectation values from eigendecomposition.

**Methods:**
- `expectation_value(operator, state_index=0)` — `⟨ψ_n|O|ψ_n⟩`
- `all_expectation_values(operator)` — Array of expectation values for all states
- `correlation(op_a, op_b, state_index=0)` — Connected correlation `⟨AB⟩ - ⟨A⟩⟨B⟩`
- `thermal_average(operator, beta)` — `⟨O⟩ = Σ exp(-βE_n)⟨n|O|n⟩ / Z`

---

## Benchmark Suite

### `benchmarks.analytic_benchmarks.AnalyticBenchmarks`

Runs validation against known exact solutions.

**Tests included:**
1. Single qubit in transverse field
2. Two-site Heisenberg (singlet/triplet)
3. Three-site Heisenberg (PBC, Bethe ansatz comparison)
4. TFIM classical limit (h=0)
5. TFIM quantum limit (J=0)
6. XX chain (free fermion solvable)

**Usage:**
```python
from benchmarks.analytic_benchmarks import AnalyticBenchmarks

bench = AnalyticBenchmarks(tolerance=1e-12)
results = bench.run_all()
summary = bench.summary()
```

### `benchmarks.quspin_validation.QuSpinValidator`

Cross-validates with QuSpin (if installed). Falls back gracefully if not.

**Tests included:**
1. TFIM parameter sweep (4-site, 20 h-values)
2. Heisenberg comparison (4-site)

### `benchmarks.performance_benchmarks.PerformanceBenchmarks`

Measures timing and scaling.

**Configuration:**
```python
bench = PerformanceBenchmarks(n_runs=5, max_n_sites=10)
```

**Tests included:**
1. Performance scaling (N=2 to max_n_sites)
2. Oracle comparison (individual oracle timing at fixed size)

### `benchmarks.platform_benchmarks.PlatformBenchmarks`

Generates platform-specific reference data for cross-platform comparison.

**Output includes:**
- Full platform metadata (OS, BLAS, NumPy/SciPy versions)
- Reference eigenvalues at full precision (20 digits)
- SHA-256 hash of eigenvalue array (for quick comparison)

To check cross-platform reproducibility, run on multiple platforms and compare:
- Eigenvalue float values (expected agreement: < 1e-13)
- Hash values (will differ due to bit-level rounding, but eigenvalues should match physically)

### `benchmarks.error_injection.ErrorInjectionTests`

Validates that the framework correctly detects errors.

**Tests included:**
1. Non-Hermitian term injection (should be detected)
2. Matrix corruption (residuals should reveal it)
3. Oracle disagreement (consensus should fail)
4. Eigenvector perturbation (residuals should grow)

---

## Master Runner

### `run_all_benchmarks.py`

Single script to run everything and produce a packaged archive.

#### Usage

```bash
# Full run (all benchmarks)
python run_all_benchmarks.py

# Quick run (smaller sizes, fewer iterations)
python run_all_benchmarks.py --quick

# Skip pytest tests
python run_all_benchmarks.py --no-tests

# Skip QuSpin (if installation issues)
python run_all_benchmarks.py --no-quspin

# Custom output directory
python run_all_benchmarks.py --output-dir /path/to/results
```

#### Phases

1. **Platform Information** — Collects system metadata
2. **Test Suite** — Runs pytest on all unit tests
3. **Analytic Validation** — Tests against known exact solutions
4. **QuSpin Cross-Validation** — Compares with QuSpin (if available)
5. **Error Injection Tests** — Verifies error detection capability
6. **Performance Benchmarks** — Times scaling and oracle comparison
7. **Figure Generation** — Creates plots (if matplotlib available)
8. **Manuscript Data Aggregation** — Builds `manuscript_data.json` with key tables/numbers
9. **Archive Creation** — Packages everything into `.tar.gz`

#### Output Structure

```
results/run_YYYYMMDD_HHMMSS/
├── manifest.json              # Summary of all phases
├── platform_results.json      # Platform info + reference computations
├── test_results.json          # pytest summary
├── test_output.txt            # Full pytest output
├── analytic_results.json      # Analytic benchmark results
├── quspin_results.json        # QuSpin validation (if run)
├── error_injection_results.json
├── performance_results.json   # Timing data
├── manuscript_data.json       # Aggregated for manuscript
└── figures/
    ├── performance_scaling.png
    ├── analytic_residuals.png
    └── consensus_quality.png

certify_ed_results_YYYYMMDD_HHMMSS.tar.gz  # Full archive
```

---

## Output Format

### `manifest.json`

Top-level summary:

```json
{
    "run_timestamp": "2026-05-06T11:13:26",
    "total_elapsed_seconds": 4.9,
    "mode": "quick",
    "phases": {
        "platform": {"status": "success", "elapsed_seconds": 0.3},
        "tests": {"status": "success", "elapsed_seconds": 0.0},
        ...
    },
    "output_files": [...],
    "output_directories": [...]
}
```

### `manuscript_data.json`

Aggregated data ready for manuscript inclusion:

```json
{
    "summary": {
        "analytic": {
            "tests_run": 6,
            "tests_passed": 6,
            "max_residual": 3.24e-15,
            "max_disagreement": 9.33e-15
        },
        "quspin": {...},
        "platform": {...}
    },
    "tables": {
        "analytic_validation": [...],     # Ready-to-format table rows
        "performance_scaling": [...]
    },
    "key_numbers": {
        "quspin_max_abs_diff": 1.2e-13,
        "mean_consensus_overhead": 2.85
    }
}
```

### Certificate format

Each certificate is a JSON file with:
- `format_version` — Schema version
- `timestamp` — ISO 8601 generation time
- `platform` — System metadata
- `spectrum` — Eigenvalues, ground state, gap
- `verification` — Residuals, normalization errors
- `certification` — Pass/fail status
- `metadata` — User-provided info
- `consensus` — Multi-oracle report (if available)
- `eigenvectors_real`, `eigenvectors_imag` — Eigenvectors (if `include_eigenvectors=True`)
- `sha256` — Hash of entire certificate (excluding hash field) for tamper detection

---

## Reproducing Results

### Step 1: Run benchmarks

```bash
cd certify-ed
python run_all_benchmarks.py
```

This produces a `.tar.gz` archive with all results.

### Step 2: Extract and inspect

```bash
tar -xzf certify_ed_results_*.tar.gz
cd results/run_*
cat manuscript_data.json | python -m json.tool
```

### Step 3: Verify reproducibility

Run on different platforms and compare:

```bash
# On Platform A
python run_all_benchmarks.py
# Save: results/run_*/platform_results.json

# On Platform B
python run_all_benchmarks.py
# Compare: eigenvalues_float values, max difference should be < 1e-13
```

### Step 4: Use in manuscript

The `manuscript_data.json` file contains:
- `summary` — High-level pass/fail counts and key error magnitudes
- `tables` — Pre-formatted rows for inclusion in tables
- `key_numbers` — Specific values referenced in text

Insert these directly into the manuscript template.

---

## Troubleshooting

### Tests fail on certificate save/load

Ensure you have the latest version with `_NumpyJSONEncoder` (handles numpy bool/int types).

### QuSpin installation fails

QuSpin requires Python 3.10 specifically and certain BLAS configurations. If unavailable:
```bash
python run_all_benchmarks.py --no-quspin
```
The benchmark suite handles this gracefully and notes that QuSpin validation was skipped.

### Performance benchmarks slow

Use quick mode:
```bash
python run_all_benchmarks.py --quick
```
This runs N=2..8 instead of N=2..10 with fewer iterations.

### Memory errors at large N

For N > 12, memory usage exceeds 1 GB. Either reduce `max_n_sites` in performance benchmarks or run on a machine with more RAM.

---

## Citing

If you use CERTIFY-ED in research, please cite:

```bibtex
@software{vehale2026certify,
    author = {Vehale, Sarang},
    title = {CERTIFY-ED: A Verification Framework for Exact Diagonalization},
    year = {2026},
    url = {https://github.com/sarangvehale/certify-ed},
    version = {1.0.0}
}
```

---

## License

MIT License — see LICENSE file.
