"""QuSpin: a J1-J2 chain in the S^z = 0, k = 0 block; the claim and the spec come from claim_from_quspin."""
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

import certify_ed as ce

L = 10
spec = ce.heisenberg_chain(L, J2=0.3)                  # the intended model, written independently of QuSpin
static = spec.to_quspin_static()
basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pauli=0)
H = hamiltonian(static, [], basis=basis, dtype=np.float64,
                check_symm=False, check_herm=False, check_pcon=False)
E, V = H.eigsh(k=4, which="SA", v0=np.random.default_rng(1).standard_normal(basis.Ns))
order = np.argsort(E)
claim, spec_q = ce.claim_from_quspin(H, basis, static, E[order], V[:, order], kind="lowest")
print("sector basis size", basis.Ns, "; full dimension", spec.dim)
cert = ce.certify(claim, spec)
print(cert.summary())
print("sector dimension", cert.body["subject"]["sector"]["dimension"],
      " lift shift", f"{cert.body['subject']['sector']['lift_shift']:.2e}")
