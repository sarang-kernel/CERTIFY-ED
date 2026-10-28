"""Certify a QuSpin result: the six lowest states of a J1-J2 chain in the S^z = 0, k = 0 block."""
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

import certify_ed as ce

L = 14
spec = ce.heisenberg_chain(L, J2=0.3)                  # the intended model, written independently
static = spec.to_quspin_static()
basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pauli=0)
H = hamiltonian(static, [], basis=basis, dtype=np.float64, check_symm=False, check_herm=False, check_pcon=False)
E, V = H.eigsh(k=6, which="SA", v0=np.random.default_rng(1).standard_normal(basis.Ns))
order = np.argsort(E)
claim, _ = ce.claim_from_quspin(H, basis, static, E[order], V[:, order], kind="lowest")
cert = ce.certify(claim, spec)
print(cert.summary())                       # one sentence; cert.report() is the full text
cert.save("example_01_certificate.json")
