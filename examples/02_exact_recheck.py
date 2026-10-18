"""Re-check a claim with no floating-point assumption (S6), for a spin-1 chain (irrational matrix
elements; exact through the scaled basis), and certify a k = 0 ground state of a longer chain through
exact symmetry sectors."""
import numpy as np
import scipy.linalg as la

import certify_ed as ce

spec = ce.spin1_heisenberg_chain(6)
st = ce.u1_state_set(spec, 6)                          # S^z = 0
H = ce.build_bitwise(spec, st)
w, V = la.eigh(H.toarray())
cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest", states=st), spec,
                  ce.Options(exact_recheck=True))
for s in cert.statements:
    print(f"{s['id']:22s} {s['status']:13s} {s['summary'][:90]}")

# k = 0 ground state of a 16-site Heisenberg chain, counted exactly in the (P, Z) sectors of k = 0
spec = ce.heisenberg_chain(16)
st = ce.u1_state_set(spec, 8)
H = ce.build_bitwise(spec, st)
T = ce.translation(spec)
V0 = ce.cyclic_isometry(st, T, 0)
wk, Y = la.eigh((V0.conj().T @ H @ V0).toarray())
r = ce.exact_recheck_sector_lowest(spec, st, [T, ce.reflection(spec), ce.spin_flip(spec)], {0: 0},
                                   wk[:1], (V0 @ Y[:, :1]).real, next_hint=wk[1])
print(f"\nN=16, k=0: E0 = {wk[0]:.12f}, exact radius {r['radii'][0]:.1e}, status {r['status']}, "
      f"sectors {[s['dim'] for s in r['sectors']]}")
