"""Interior window by shift-invert: scipy.sparse.linalg.eigsh(sigma=..., which="LM"), then kind="window"."""
import numpy as np
import scipy.sparse.linalg as spla

import certify_ed as ce

spec = ce.heisenberg_chain(12, 1.0, J2=0.4, boundary="open")     # J1-J2 chain, L = 12, open ends
states = ce.u1_state_set(spec, 6, label="Sz=0")                 # the Sz = 0 block (dimension 924)
H = ce.build_bitwise(spec, states)                               # scipy.sparse CSR on that block
sigma = -1.0                                                     # the energy the eigenvalues are wanted around
rng = np.random.default_rng(0)
# shift-invert: the eigenvalues closest to sigma are the largest in magnitude of (H - sigma)^-1
w, V = spla.eigsh(H, k=10, sigma=sigma, which="LM", tol=1e-12, v0=rng.standard_normal(H.shape[0]))
order = np.argsort(w)
w, V = w[order], V[:, order]
# A window claim says: the list holds EVERY eigenvalue in (a, b). The two outermost values computed are
# only there to show where the gaps are; the window edges go into those gaps and the claim is the inner 8.
a, b = 0.5 * (w[0] + w[1]), 0.5 * (w[-2] + w[-1])
w_in, V_in = w[1:-1], V[:, 1:-1]
print("eigsh around sigma =", sigma, ":", np.array2string(w, precision=6))
print("window (%.6f, %.6f), %d claimed inside" % (a, b, w_in.size))

cert = ce.quick(H, w_in, V_in, spec, kind="window", window=(a, b), states=states,
                producer={"name": "scipy.sparse.linalg.eigsh", "sigma": sigma, "which": "LM", "tol": 1e-12})
print(cert.summary())
for s in cert.statements[:3]:
    print(f"  {s['id']:18s} {s['status']:12s} {s['summary']}")

# the same list with a window that reaches past it: FAILED (eigenvalues exist there), not an error
cert2 = ce.quick(H, w_in, V_in, spec, kind="window", window=(a, b + 0.2), states=states, only="S2,S3")
s3 = [s for s in cert2.statements if s["id"] == "S3.completeness"][0]
print("wider window:", s3["status"], "-", s3["summary"])
