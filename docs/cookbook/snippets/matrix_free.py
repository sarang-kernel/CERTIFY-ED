"""Matrix-free producers: a LinearOperator with a declared spec (S1 by probes), and without one."""
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

import certify_ed as ce

spec = ce.heisenberg_chain(8, 1.0)
H = ce.build_bitwise(spec)                             # stands in for the producer's internal representation
w, V = la.eigh(H.toarray())

def matvec(x):                                         # the producer exposes only this
    return H @ x

L = spla.LinearOperator(H.shape, matvec=matvec, dtype=np.float64)
claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=L, kind="lowest",
                 producer={"name": "my_matvec"})

# with a spec: S1 compares the matvec with the spec operator on random probes; S2-S4 then bound the spec operator
cert = ce.certify(claim, spec)
print(cert.summary())
print("operator note:", cert.body["operator_note"])

# without a spec: nothing ties the matvec to a model; the rounding inside it cannot be bounded
cert2 = ce.certify(claim, None)
for s in cert2.statements:
    print(f"no spec  {s['id']:20s} {s['status']:12s} {s['evidence'].get('reason_code')}")
print("no spec  S2 summary:", cert2.statements[1]["summary"])
