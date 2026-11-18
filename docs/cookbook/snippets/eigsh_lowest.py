"""Sparse matrix, scipy.sparse.linalg.eigsh for the k lowest eigenpairs."""
import numpy as np
import scipy.sparse.linalg as spla

import certify_ed as ce

spec = ce.heisenberg_chain(12, 1.0, J2=0.4)            # J1-J2 chain, L = 12 (dimension 4096)
H = ce.build_bitwise(spec)                             # scipy.sparse CSR
k = 5
rng = np.random.default_rng(0)
w, V = spla.eigsh(H, k=k, which="SA", tol=1e-12, v0=rng.standard_normal(H.shape[0]))
order = np.argsort(w)                                  # eigsh does not sort
w, V = w[order], V[:, order]

claim = ce.Claim(eigenvalues=w, eigenvectors=V, operator=H, kind="lowest",
                 producer={"name": "scipy.sparse.linalg.eigsh", "tol": 1e-12})
cert = ce.certify(claim, spec)
print(cert.summary())
print("radii", np.array2string(cert.arrays["radius"], precision=2))
