# Installation Guide

## Requirements

- Python 3.8 or later
- NumPy >= 1.20
- Matplotlib >= 3.3
- pytest >= 6.0 (for testing)

## Installation Methods

### Method 1: Development Install (Recommended for Development)

For active development and contribution:

```bash
git clone https://github.com/sarang-kernel/CERTIFY-ED.git
cd certify-ed
pip install -e .
```

This installs the package in "editable" mode, so changes to the source code are immediately reflected.

### Method 2: Production Install

For end users:

```bash
git clone https://github.com/sarang-kernel/CERTIFY-ED.git
cd certify-ed
pip install .
```

### Method 3: With Development Dependencies

To install with testing and development tools:

```bash
pip install -e ".[dev]"
```

This includes pytest, pytest-cov, black, and flake8.

## Verification

Verify the installation:

```python
python -c "import certifyEd; print(certifyEd.__version__)"
```

Expected output: `1.0.0`

## Running Tests

After installation, run the test suite:

```bash
pytest tests/ -v
```

All tests should pass.

## Running Validation Experiments

Reproduce paper results:

```bash
# Complete pipeline
python scripts/run_complete_pipeline.py

# Individual experiments
python scripts/validate_analytic.py
python scripts/validate_reproducibility.py
```

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'certifyEd'`:

```bash
# Ensure you're in the certify-ed directory
cd certify-ed

# Reinstall
pip install -e .
```

### Missing Dependencies

If NumPy or Matplotlib are missing:

```bash
pip install -r requirements.txt
```

### Permission Errors

Use `--user` flag:

```bash
pip install --user -e .
```

Or use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Next Steps

1. Run `python examples/basic_usage.py` for a quick demo
2. See README.md for API documentation
3. Run validation experiments to verify installation
