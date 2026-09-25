# CERTIFY-ED Quickstart

Five-minute introduction to verified exact diagonalization.

## Install

```bash
pip install -e .
pip install -e .[high_precision,plotting,dev]   # recommended optionals
```

## Hello world

```python
from certify_ed import build_model, MultiOracle

H = build_model('tfim', n_sites=4, J=1.0, h=0.5)
oracle = MultiOracle()
evals, evecs, report = oracle.diagonalize_with_consensus(H)

print(f'Ground state:    {evals[0]:.10f}')
print(f'Spectral gap:    {evals[1] - evals[0]:.10f}')
print(f'Oracle consensus: {report["consensus"]}')
```

The `MultiOracle` runs three independent LAPACK paths (NumPy DSYEVD,
SciPy DSYEVD, SciPy DSYEVR) and reports the worst pairwise disagreement.
For double-precision LAPACK on a well-conditioned Hamiltonian, you should
see disagreement around 1e-14 to 1e-15.

## Available models

```python
from certify_ed import list_models

print(list_models())
# ['tfim', 'heisenberg', 'xxz', 'xyz', 'ssh', 'j1j2', 'majumdar_ghosh',
#  'cluster', 'free_fermion', 'kitaev_chain', 'hubbard', 'aklt', 'haldane',
#  'tfim_2d', 'heisenberg_2d', 'kitaev_honeycomb']
```

Each takes physically meaningful keyword arguments — see `docs/CODE_DOCUMENTATION.md`.

## Certificates

A `Certificate` is the unit of trustworthy ED output. It bundles the
spectrum, residuals, normalization errors, oracle consensus, and a
SHA-256 hash:

```python
from certify_ed import Certificate, load_certificate

cert = Certificate(evals, evecs, H, consensus_report=report,
                   metadata={'model': 'tfim', 'N': 4, 'J': 1, 'h': 0.5})
cert.save('cert.json')

# Anyone receiving cert.json can verify the hash:
loaded = load_certificate('cert.json', verify_hash=True)
```

If a single byte of the JSON changes, `load_certificate` raises a
`ValueError`.

## Running validators

A validator probes one specific aspect of correctness:

```python
from validators import AnalyticValidator, ErrorInjectionValidator

# Compare against closed-form solutions
v = AnalyticValidator()
print(v.summary())
# {'validator': 'AnalyticValidator', 'n_total': 12, 'n_passed': 12, ...}

# Inject known errors and check the framework catches them
v = ErrorInjectionValidator()
print(v.summary())
# {'validator': 'ErrorInjectionValidator', 'n_total': 6, 'all_detected': True, ...}
```

The 13 validators all follow the same interface: `validator.summary()`
returns a dict with `n_total`, `n_passed`, and `individual_results`.

## Run everything

```bash
python run_all_benchmarks.py
```

You'll see something like:

```
==============================================================================
                     CERTIFY-ED MASTER VALIDATION RUN
==============================================================================

------------------------------------------------------------------------------
  STAGE 1: pytest test suite
------------------------------------------------------------------------------
[pytest output streams here in real time]
============================= 53 passed in 16s ==============================

------------------------------------------------------------------------------
  STAGE 2: 13 independent validators
------------------------------------------------------------------------------
>> [analytic] Closed-form analytic results (Bethe ansatz, SU(2), exact GS)
   PASS: 12/12 tests, 0.4s
>> [quspin] Cross-validation against QuSpin
   SKIPPED: 0/0 tests, 0.0s
>> [high_precision] mpmath arbitrary-precision (50 digits) reference
   PASS: 3/3 tests, 0.6s
... (10 more validators) ...

Validator suite total: 78/78 tests passed

==============================================================================
                                 RUN COMPLETE
==============================================================================
  Total time:                 33s
  Output directory:           results/run_20260507_041350
  Archive:                    results/run_20260507_041350.tar.gz
```

The archive is what you could attach to a submission. It contains:

- `pytest_output.txt` — full pytest log
- `validators/*.json` — per-validator detailed results
- `figures/*.png` — sum-rule errors, scaling plots, validator summary,
  error-injection chart
- `manuscript_data.json` — aggregated headline numbers
- `platform_info.json` — Python/NumPy/SciPy versions
- `manifest.json` — index pointing to all of the above

## Examples directory

- `examples/example_01_basic_workflow.py` — build, diagonalize, certify
- `examples/example_02_validation.py` — run individual validators
- `examples/example_03_symmetry_resolved.py` — sector decomposition

## Next reading

- `README.md` — overview
- `docs/CODE_DOCUMENTATION.md` — full API reference
- `docs/VALIDATORS.md` — what each validator checks and why
