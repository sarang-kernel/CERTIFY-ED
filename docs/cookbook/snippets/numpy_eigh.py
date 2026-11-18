"""Dense arrays from scipy.linalg.eigh: the whole spectrum (kind 'all') and the k lowest (kind 'lowest')."""
import numpy as np
import scipy.linalg as la

import certify_ed as ce

spec = ce.tfim_chain(8, 1.0, 0.8)                      # transverse-field Ising chain, L = 8
H = ce.build_bitwise(spec).toarray()
w, V = la.eigh(H)
print("lowest ten:", np.array2string(w[:10], precision=6))

# the whole spectrum: every eigenvalue claimed, count must equal the dimension
cert_all = ce.certify(ce.claim_from_arrays(w, V, operator=H, kind="all"), spec)
for s in cert_all.statements:
    print(f"all      {s['id']:20s} {s['status']}")
print("all      max radius", f"{cert_all.arrays['radius'].max():.2e}")

# 'the six lowest': the sixth level is a quadruplet, so 'the k lowest' is ill-posed for k = 6
for k in (6, 5):
    cert = ce.certify(ce.claim_from_arrays(w[:k], V[:, :k], operator=H, kind="lowest"), spec)
    s3 = cert.statements[2]
    print(f"k={k}      S3.completeness      {s3['status']}")
    if s3["status"] == "CERTIFIED":
        print(f"k={k}      gap >= {s3['evidence']['gap_lower_bound']:.6f}; next eigenvalue >= {s3['evidence']['next_eigenvalue_lower_bound']:.6f}")
    else:
        print(f"k={k}      reason: {s3['evidence']['reason']}")
