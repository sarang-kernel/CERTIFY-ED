"""Five minutes: a Heisenberg chain, scipy.linalg.eigh, one call, the certificate."""
import scipy.linalg as la

import certify_ed as ce

spec = ce.heisenberg_chain(8, 1.0)                     # the model you intend (declared independently of the solver)
H = ce.build_bitwise(spec)                             # your matrix; here CERTIFY-ED's own builder stands in
w, V = la.eigh(H.toarray())                            # the producer: any solver, here LAPACK

claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
cert = ce.certify(claim, spec)                        # Options() by default: S1-S4 and S8
print(cert.summary())
path = cert.save("quickstart_certificate.json")       # certificate.json + certificate.arrays.npz
print(path)
