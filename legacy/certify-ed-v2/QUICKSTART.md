# CERTIFY-ED Quick Start Guide

## What This Package Does

CERTIFY-ED provides verified exact diagonalization of quantum many-body systems with:
- Multi-oracle consensus validation
- Cryptographically-signed verification certificates
- Comprehensive benchmark suite

## To Generate Manuscript Data: 3 Steps

### Step 1: Install

```bash
cd certify-ed
pip install -e .
```

If you want QuSpin cross-validation (optional):
```bash
pip install quspin
```

### Step 2: Run Everything

Single command generates all benchmark data, runs tests, creates figures, and packages into archive:

```bash
python run_all_benchmarks.py
```

This takes ~1-3 minutes and produces:

- **`certify_ed_results_TIMESTAMP.tar.gz`** ← Send this for manuscript revision

The archive contains:
```
results/run_TIMESTAMP/
├── manifest.json                  Summary of all phases
├── platform_results.json          OS, BLAS, NumPy versions
├── test_results.json              Pytest results
├── analytic_results.json          Validation vs known solutions
├── quspin_results.json            QuSpin cross-validation (if installed)
├── performance_results.json       Timing benchmarks
├── error_injection_results.json   Error detection tests
├── manuscript_data.json           Aggregated tables ready for manuscript
└── figures/
    ├── performance_scaling.png
    ├── analytic_residuals.png
    └── consensus_quality.png
```

### Step 3: Send Archive

The `.tar.gz` file is everything needed to revise the manuscript with real data.

## Optional: Quick Mode

For faster runs (smaller systems, fewer iterations):
```bash
python run_all_benchmarks.py --quick
```

## Optional: Custom Runs

```bash
# Skip pytest
python run_all_benchmarks.py --no-tests

# Skip QuSpin (if not installed)
python run_all_benchmarks.py --no-quspin

# Custom output location
python run_all_benchmarks.py --output-dir /path/to/results
```

## Verifying Everything Works

```bash
# Run unit tests
pytest tests/ -v

# Run example
python examples/example_01_basic_workflow.py
```

## Troubleshooting

### "Module not found" errors
Make sure you ran `pip install -e .` from the package root.

### Slow performance benchmarks
Use `--quick` mode or reduce `max_n_sites` in `benchmarks/performance_benchmarks.py`.

### QuSpin installation issues
QuSpin requires Python 3.10. Skip with `--no-quspin`.

### Memory errors
Don't run on systems with less than 8 GB RAM with default settings.

## Files Description

| File | Purpose |
|------|---------|
| `run_all_benchmarks.py` | Master runner — execute everything |
| `certify_ed/` | Core package |
| `benchmarks/` | Benchmark suite (one class per benchmark type) |
| `tests/` | Unit tests |
| `examples/` | Usage examples |
| `docs/CODE_DOCUMENTATION.md` | Full API reference |

## Multi-Platform Reproducibility

To verify cross-platform consistency, run on multiple machines:

```bash
# Platform A (e.g., Linux):
python run_all_benchmarks.py
# Save the archive

# Platform B (e.g., macOS):
python run_all_benchmarks.py
# Save the archive

# Compare:
# - Open both platform_results.json
# - Compare eigenvalues_float arrays
# - Differences should be < 1e-13 (expected due to BLAS variations)
```

## Next Steps

After running benchmarks, send `certify_ed_results_TIMESTAMP.tar.gz` to revise the manuscript with real data.
