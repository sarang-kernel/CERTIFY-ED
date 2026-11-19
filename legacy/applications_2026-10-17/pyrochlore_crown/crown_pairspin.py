"""Spin-S crown cluster (reduced model, c3 = c12, c5 = 0) decomposed by conserved outer-pair spins."""
import itertools, sys, time
import numpy as np, scipy.sparse.linalg as spla
from mixed_spin import heisenberg

HEXRING = 6   # hexagon sites 0..5 in ring order; tetrahedron k touches hexagon sites k and k+1


def config_hamiltonian(S2, pair_spins2, c1, c3, c17):
    """S2 = 2S of the lattice spins; pair_spins2 = 2*S~ per outer pair (6 entries)."""
    two = [S2] * 6
    bonds = [(c17, i, (i + 1) % 6) for i in range(6)]
    const = 0.0
    S = S2 / 2
    for k, ts in enumerate(pair_spins2):
        St = ts / 2
        const += c1 * (St * (St + 1) - 2 * S * (S + 1)) / 2
        if ts == 0:
            continue
        two.append(ts)
        idx = len(two) - 1
        bonds += [(c3, idx, k), (c3, idx, (k + 1) % 6)]
    return heisenberg(two, bonds, const)


def canonical(cfg):
    """Representative of a configuration under the dihedral group of the ring of 6 tetrahedra."""
    rots = [tuple(cfg[(i + r) % 6] for i in range(6)) for r in range(6)]
    refl = [tuple(c[::-1]) for c in rots]
    return min(rots + refl)


def landscape(S2, c1, c3, k=1):
    c17 = (1 - 2 * c1 - 8 * c3) / 2
    allowed = list(range(0, 2 * S2 + 1, 2))          # 2*S~ for S~ = 0 .. 2S
    classes = {}
    for cfg in itertools.product(allowed, repeat=6):
        classes.setdefault(canonical(cfg), 0)
        classes[canonical(cfg)] += 1
    out = []
    for cfg, mult in sorted(classes.items()):
        H, st = config_hamiltonian(S2, cfg, c1, c3, c17)
        d = H.shape[0]
        if d <= 400:
            e = np.linalg.eigvalsh(H.toarray())[0]
        else:
            v0 = np.random.default_rng(1).standard_normal(d)
            e = spla.eigsh(H, k=1, which="SA", tol=1e-12, v0=v0, return_eigenvectors=False)[0]
        out.append((float(e), cfg, d, mult))
    out.sort()
    return out


if __name__ == "__main__":
    S2 = int(sys.argv[1])
    c1, c3 = float(sys.argv[2]), float(sys.argv[3])
    t = time.time()
    L = landscape(S2, c1, c3)
    print(f"{len(L)} classes, {time.time()-t:.0f}s")
    for e, cfg, d, m in L[:12]:
        print(f"{e:.8f}  pair spins {tuple(x/2 for x in cfg)}  dim {d}  mult {m}")
    print("largest dim", max(x[2] for x in L))
