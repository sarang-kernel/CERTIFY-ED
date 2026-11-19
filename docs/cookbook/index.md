# Cookbook

One page per way of getting a result into `certify()`. Every example on these pages was run with
certify-ed 3.2.0.dev0 (Python 3.11, numpy 2.4.4, scipy 1.17.1, QuSpin 1.0.1, python-flint 0.9.0, Windows) and its
output is pasted as printed; the scripts are in `docs/cookbook/snippets/` and `run_all.py` there re-runs them.

| producer | page | what the example certifies |
|---|---|---|
| dense arrays from `scipy.linalg.eigh` | [numpy arrays](numpy_eigh.md) | the whole spectrum of a TFIM chain; the k lowest, including an ill-posed k |
| `scipy.sparse.linalg.eigsh`, lowest k | [eigsh](eigsh_lowest.md) | five lowest of a J1–J2 chain, dimension 4096 |
| `eigsh(sigma=...)`, interior window | [shift-invert window](shift_invert_window.md) | eight eigenvalues around E = -1 of an open J1–J2 chain, `kind="window"`; a window that reaches too far is FAILED |
| `.npz` / HDF5 files, command line | [files](files_import.md) | `certify-ed certify file.npz --spec` and `certify-ed import bundle/` |
| QuSpin | [QuSpin](quspin.md) | a momentum block (`kblock=0`), with the certified lift to the full operator |
| a `matvec` only | [matrix-free](matrix_free.md) | with a declared spec (S1 by probes) and without one |
| any code that writes files | [producer bundle](producer_bundle.md) | a CSV bundle with `meta.json`, a preset, `certify-ed import` |
| eigenvectors on disk (`.npy` memmap, HDF5) | [vectors larger than RAM](streaming_vectors.md) | the whole spectrum of a 924-block from a memmap, identical to the in-memory certificate; `--plan` |

The inputs every path ends with are the same: a `Claim` (eigenvalues, optionally eigenvectors, the
producer's operator, the kind of completeness claimed: `subset`, `lowest`, `window`, `all`) and a
`ModelSpec`. Without eigenvectors only counts can be certified; without a spec nothing ties the matrix to a
model (S1 is NOT_CHECKED and the radii refer to the producer's matrix as stored).
