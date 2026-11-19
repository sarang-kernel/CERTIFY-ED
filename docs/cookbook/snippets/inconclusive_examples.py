"""What INCONCLUSIVE and FAILED look like: a duplicated vector, a missing eigenvalue, an ill-posed k, a tiny entry."""
import numpy as np
import scipy.linalg as la

import certify_ed as ce

spec = ce.tfim_chain(8, 1.0, 0.8)
H = ce.build_bitwise(spec)
w, V = la.eigh(H.toarray())
print("lowest six:", np.array2string(w[:6], precision=6))

def show(tag, cert):
    for s in cert.statements[:3]:
        print(f"{tag:10s} {s['id']:18s} {s['status']:12s} {s['evidence'].get('reason_code') or ''}")
    s = cert.statements[1] if cert.statements[1]["status"] != "CERTIFIED" else cert.statements[2]
    print(f"{tag:10s} {s['id'].split('.')[0]} says: {s['summary'][:170]}")

# the same vector claimed twice: existence holds, distinctness does not
X = V[:, :3].copy(); X[:, 1] = X[:, 0]; mu = w[:3].copy(); mu[1] = mu[0]
show("duplicate", ce.certify(ce.Claim(mu, X, operator=H, kind="lowest"), spec))

# the third lowest eigenvalue left out of a 'lowest' claim (indices 0, 1, 3, 4): the inertia count catches it
idx = [0, 1, 3, 4]
show("missing", ce.certify(ce.Claim(w[idx], V[:, idx], operator=H, kind="lowest"), spec))

# 'the four lowest' when the fourth level is a doublet: the claim is complete but k is ill-posed
show("ill-posed", ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec))

# a subnormal entry planted in an eigenvector: the no-underflow hypothesis is not established
X = V[:, :3].copy(); X[5, 2] = 2.0 ** -1060
show("underflow", ce.certify(ce.Claim(w[:3], X, operator=H, kind="lowest"), spec))
