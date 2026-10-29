# Draft pull request to QuSpin: `quspin.tools.certify`

Status: draft text, not submitted. Target repository: QuSpin (weinbe58/QuSpin), branch `master`.
Nothing in certify-ed depends on this; the wrapper it proposes calls
`certify_ed.integrations.quspin.certify_hamiltonian`, which exists in certify-ed 3.2.

---

## Title

Add `quspin.tools.certify`: an optional a-posteriori certificate for `hamiltonian.eigsh` / `eigh` results

## Summary

This adds a thin optional module `quspin/tools/certify.py` with one function,

```python
def certify(H, E, V=None, kind="lowest", **options):
    """A-posteriori certificate for eigenpairs of a static `hamiltonian` (needs the certify-ed package)."""
    try:
        from certify_ed.integrations.quspin import certify_hamiltonian
    except ImportError as e:
        raise ImportError("quspin.tools.certify needs the certify-ed package: pip install certify-ed") from e
    return certify_hamiltonian(H, E, V, kind=kind, **options)
```

and a short section in the documentation. The dependency is optional and imported inside the function; QuSpin's
import time, dependencies and test suite are unchanged when certify-ed is absent.

## What the user gets

```python
H = hamiltonian(static, [], basis=basis, dtype=np.float64)
E, V = H.eigsh(k=4, which="SA")
cert = quspin.tools.certify(H, E, V)
print(cert.summary())
# 3 of 5 statements CERTIFIED (S1, S2, S3), ...; max certified radius 5.2e-13 (rounded up); ...
```

The certificate contains, for the claim "E are the 4 lowest eigenvalues of H":

* S1, model fidelity: the matrix QuSpin built for `static` equals, entry by entry, an independently built matrix
  for the same operator strings; for a symmetry-reduced basis (`kblock`, `pblock`, ...), that the block is an
  invariant, orthonormal restriction of the full operator (isometry and invariance defects reported);
* S2, eigenpairs: for each (E_i, V[:, i]) a rigorous radius around E_i that contains an eigenvalue of H
  (residual bound with directed rounding, Theorem 1 of the certify-ed theory notes), and the matching to distinct
  eigenvalues;
* S3, completeness: that there are exactly k eigenvalues below the claimed cut, with a certified gap to the next
  (inertia count of a shifted LDL^T factorization with an error bound);
* S4, S8: eigenspace and degeneracy statements for clusters of claimed values.

Each statement is CERTIFIED, FAILED, INCONCLUSIVE or NOT_CHECKED with its evidence; INCONCLUSIVE statements carry
a `next_action`. The certificate is a JSON document with a schema and a body hash.

## Why in QuSpin

QuSpin already exposes `check_symm`, `check_herm` and `check_pcon` as construction-time checks. This adds the
corresponding a-posteriori check of the solver output, at the point where users have `H`, `E` and `V` in hand.
The wrapper is one function and one optional import; the cost of a certificate for a sector of a few thousand
states is seconds.

## Scope and limits (stated in the docstring)

* Static Hamiltonians only (`dynamic_list` empty); `quantum_operator` and `quantum_LinearOperator` are not
  covered by the wrapper (they can be certified through `certify_ed.quick` with a matrix and a model spec).
* The model is rebuilt from the operator-string list QuSpin received, so S1 compares two constructions of the
  same strings; a separately written model can be passed as `spec=`.
* Bounds hold under the IEEE 754 standard model (checked at run time by certify-ed); a no-underflow hypothesis on
  the input is enforced by a scan and reported if violated.
* The certificate certifies the eigenpairs of the matrix QuSpin built, in the basis QuSpin built. It does not
  certify the physics of the model.

## Files

* `quspin/tools/certify.py` (new, ~20 lines)
* `docs/sources/tools.rst`: one entry
* `tests/test_certify_tool.py`: skipped when certify-ed is not installed; with it, a 10-site Heisenberg chain in
  the `Nup=5, kblock=0` block, four lowest states, all of S1, S2, S3 CERTIFIED; and a planted wrong eigenvalue
  that is not certified.

## Open questions for the maintainers

1. Name: `quspin.tools.certify` or `quspin.tools.verification.certify`?
2. Whether the optional dependency should appear in `setup.py` extras (`quspin[certify]`).
3. Whether the function should accept the `hamiltonian`'s own `static` list argument explicitly instead of
   reading `H._static_opstr_list` (the wrapper reads the private attribute because `hamiltonian` does not expose
   the list; a public `hamiltonian.static_list` property would remove that).
