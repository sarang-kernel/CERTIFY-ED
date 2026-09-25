# CERTIFY-ED API Reference

Full API reference for the `certify_ed` package and `validators` package.

## `certify_ed.hamiltonian`

### Operator helpers

```python
pauli_matrices() -> (I, X, Y, Z)
spin_half_operators() -> (I, Sx, Sy, Sz, S+, S-)
spin_one_operators() -> (I, Sx, Sy, Sz, S+, S-)
fermion_operators() -> (c, c†, n, F)        # F is JW string (-1)^n
tensor_product(*ops) -> ndarray              # iterated Kronecker
site_operator(op, site, n_sites, local_dim=2) -> ndarray
jordan_wigner_operator(op, F, site, n_sites) -> ndarray
```

### Symbolic builder

```python
class SymbolicHamiltonian:
    def __init__(self, n_sites: int): ...
    def add_term(self, coefficient: complex,
                 operators: List[Tuple[str, int]]) -> None: ...
    def build(self) -> ndarray: ...
    def verify_hermiticity(self, tolerance: float = 1e-14) -> bool: ...
```

`operators` is a list of (op_name, site) pairs from {'I','X','Y','Z'}.

### Model registry

```python
list_models() -> List[str]                       # 16 models
build_model(name: str, **kwargs) -> ndarray      # dispatcher
MODEL_REGISTRY: Dict[str, Callable]              # raw registry
```

### Model builders

| Model | Builder | Key arguments |
|---|---|---|
| Transverse-field Ising | `build_tfim(n_sites, J=1.0, h=0.5, boundary='open')` | J, h |
| Heisenberg (XXX) | `build_heisenberg(n_sites, J=1.0, boundary='open')` | J |
| XXZ | `build_xxz(n_sites, J=1.0, Delta=1.0, boundary='open')` | J, Δ |
| XYZ | `build_xyz(n_sites, Jx, Jy, Jz, boundary='open')` | Jx, Jy, Jz |
| SSH | `build_ssh(n_sites, t1=1.0, t2=0.5, boundary='open')` | t₁, t₂ |
| J1-J2 chain | `build_j1j2_chain(n_sites, J1, J2, boundary)` | J₁, J₂ |
| Majumdar-Ghosh | `build_majumdar_ghosh(n_sites)` | (PBC, J₂/J₁ = ½) |
| Cluster model | `build_cluster_model(n_sites, h=1.0, boundary)` | h |
| Free fermion chain | `build_free_fermion_chain(n_sites, t, mu, boundary)` | t, μ |
| Kitaev chain | `build_kitaev_chain(n_sites, t, mu, Delta, boundary)` | t, μ, Δ |
| Hubbard chain | `build_hubbard_chain(n_sites, t, U, boundary)` | t, U |
| AKLT | `build_aklt_chain(n_sites, boundary='open')` | (fixed) |
| Haldane chain | `build_haldane_chain(n_sites, J, D, E, boundary)` | J, D, E |
| 2D TFIM | `build_tfim_2d(Lx, Ly, J, h, boundary)` | J, h |
| 2D Heisenberg | `build_heisenberg_2d(Lx, Ly, J, boundary)` | J |
| Kitaev honeycomb | `build_kitaev_honeycomb(Lx, Ly, Jx, Jy, Jz)` | Jx, Jy, Jz |

`boundary` is `'open'` or `'periodic'` for 1D; 2D supports both for each direction.

## `certify_ed.oracles`

### Single oracles

```python
class NumPyOracle(Oracle):
    """numpy.linalg.eigh — LAPACK DSYEVD via NumPy."""

class ScipyOracle(Oracle):
    """scipy.linalg.eigh with selectable LAPACK driver."""
    def __init__(self, driver: str = 'evd'):  # 'evd' | 'evr' | 'ev'

class SparseOracle(Oracle):
    """ARPACK Lanczos via scipy.sparse.linalg.eigsh — lowest-k only."""
    def __init__(self, k: Optional[int] = None, max_dim_full: int = 50):
```

All oracles expose `diagonalize(H) -> (evals, evecs)` and `name() -> str`.

### Multi-oracle

```python
class MultiOracle:
    def __init__(self, oracles: Optional[List[Oracle]] = None,
                 tolerance: float = 1e-10):
    def diagonalize_with_consensus(self, H) -> (evals, evecs, report)
    def compute_residuals(self, H, evals, evecs) -> ndarray
```

The default oracle list is `[NumPyOracle, ScipyOracle('evd'), ScipyOracle('evr')]`
giving three LAPACK paths. The `report` dict contains:

- `consensus: bool`
- `max_disagreement: float`
- `oracle_names: List[str]`
- `pairwise_max_diffs: List[List[float]]`
- (optional) `sparse_cross_check: dict` if a `SparseOracle` is in the list

## `certify_ed.certificates`

```python
class Certificate:
    def __init__(self, eigenvalues, eigenvectors, hamiltonian,
                 metadata: Optional[Dict] = None,
                 consensus_report: Optional[Dict] = None):
    def to_dict(self, include_eigenvectors: bool = True) -> dict
    def save(self, filename: str, include_eigenvectors: bool = True) -> None
    def summary(self) -> str
    # Attributes:
    eigenvalues, eigenvectors, H
    residuals, normalization_errors, is_certified
    metadata, consensus_report
```

```python
def load_certificate(filename: str, verify_hash: bool = True) -> dict
```

The on-disk JSON includes `format_version`, `timestamp`, `platform` (Python /
NumPy versions, OS), `spectrum`, `verification`, `certification`, and
`sha256` (over the rest of the document, sorted-key encoding).

## `certify_ed.symmetries`

```python
total_sz_operator(n_sites)         # sum_i Z_i / 2
total_sx_operator(n_sites)         # sum_i X_i / 2
parity_operator(n_sites)           # prod_i X_i  (spin-flip parity)
z_parity_operator(n_sites)         # prod_i Z_i
translation_operator(n_sites)      # cyclic site translation
fermion_number_operator(n_sites)   # sum_i (1 - Z_i)/2 (JW)
fermion_parity_operator(n_sites)   # prod_i Z_i

project_onto_sector(evecs, evals, S, target_value, tolerance) -> (evals, evecs)
commutator(A, B) -> ndarray
commutator_norm(A, B) -> float
check_conservation(H, S, tolerance=1e-12) -> dict
```

## `certify_ed.observables`

```python
class ObservableCalculator:
    def __init__(self, eigenvalues, eigenvectors, residuals=None):
    def expectation_value(self, operator, state_index=0) -> float
    def all_expectation_values(self, operator) -> ndarray
    def correlation(self, op_a, op_b, state_index=0) -> float
    def thermal_average(self, operator, beta) -> float
    def partition_function(self, beta) -> float
    def free_energy(self, beta) -> float
```

## `validators` package

Each validator class has the same interface:

```python
v = ValidatorClass()
v.run_all() -> List[Dict]    # individual results
v.summary() -> Dict           # aggregated summary
```

The summary always contains keys `validator`, `n_total`, `n_passed`,
`individual_results`. Some add `available` (for optional-dependency
validators) or `all_passed` / `all_detected`.

The thirteen validators:

```
AnalyticValidator             - Closed-form analytic results
QuSpinValidator               - Cross-check against QuSpin
HighPrecisionValidator        - mpmath at 50 digits
SparseDenseValidator          - ARPACK Lanczos vs LAPACK direct
JordanWignerValidator         - Free-fermion analytic spectra
SpectralSumRuleValidator      - tr(H), tr(H²), tr(H³), ‖H‖
OrthonormalityValidator       - V†V, VV†, VDV†
UnitarityValidator            - exp(-iHt) is unitary, group property
ConservationLawValidator      - [H, S] = 0 and quantum-number block-diag
SymmetrySectorValidator       - Sector-by-sector vs full spectrum
ThermalLimitValidator         - β → 0 and β → ∞ limits
FiniteSizeScalingValidator    - Trend toward thermodynamic limits
ErrorInjectionValidator       - Framework's own error-detection
```

See `docs/VALIDATORS.md` for what each one tests in detail.

## Master pipeline

`run_all_benchmarks.py` is a single entry point that:

1. runs `pytest tests/ -v` with output streamed to the terminal,
2. runs all 13 validators sequentially,
3. generates four matplotlib figures,
4. aggregates everything into `manuscript_data.json`,
5. tars the whole output directory.

Output goes under `results/run_<timestamp>/` and a sibling
`results/run_<timestamp>.tar.gz` archive.

The exit code is 0 only if every test in every layer passes.
