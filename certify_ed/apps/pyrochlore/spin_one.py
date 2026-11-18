"""S = 1 crown, reduced model (c3 = c12, c5 = 0), decomposed by the conserved spin S~ of each outer pair.

Each outer pair (a, b) with S~ in {0, 1, 2} acts as one composite spin: c1 S_a.S_b = c1 (S~(S~+1) -
2 S(S+1))/2 is constant on the block and c3 (S_a + S_b).S_h = c3 S~.S_h. The cluster Hamiltonian is
block diagonal over the 3^6 pair-spin configurations (92 classes under the dihedral group of the ring).

* printed_coupling_check: at the printed couplings, the all-singlet configuration has the exact
  eigenvalue -12 c1 + c17 E0(spin-1 ring of 6); E0 is enclosed by certify() (S2 + S3), so the cluster's
  lambda_min is rigorously below the published bound.
* landscape: floating-point lowest energy per class (estimates; the all-S~=2 class has 1.1M states).
"""
import itertools
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PRINTED = {"c1": "0.053466", "c3": "0.082726", "published_bound": "-1.632985"}


def sz0_basis(two_spins):
    """All tuples (2m_1, ..., 2m_n) with sum 0, as an int array (states x n)."""
    ranges = [np.arange(-ts, ts + 1, 2) for ts in two_spins]
    maxrest = np.concatenate([np.cumsum(list(two_spins)[::-1])[::-1][1:], [0]])
    states = np.zeros((1, 0), dtype=np.int64)
    for k, r in enumerate(ranges):
        cand = np.concatenate([np.repeat(states, len(r), axis=0), np.tile(r, states.shape[0])[:, None]], axis=1)
        states = cand[np.abs(cand.sum(axis=1)) <= maxrest[k]]
    return states[states.sum(axis=1) == 0]


def _encode(states, two_spins):
    base = np.array([ts + 1 for ts in two_spins], dtype=np.int64)
    digits = (states + np.array(two_spins)) // 2
    code = np.zeros(states.shape[0], dtype=np.int64)
    for k in range(states.shape[1]):
        code = code * base[k] + digits[:, k]
    return code


def heisenberg_mixed(two_spins, bonds, const=0.0):
    """H = sum c S_i.S_j + const on S^z = 0 for sites of different spins (floating point)."""
    states = sz0_basis(two_spins)
    codes = _encode(states, two_spins)
    order = np.argsort(codes)
    states, codes = states[order], codes[order]
    n = len(codes)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, float(const))]
    diag = np.zeros(n)
    for c, i, j in bonds:
        mi, mj = states[:, i] / 2.0, states[:, j] / 2.0
        diag += c * mi * mj
        Si, Sj = two_spins[i] / 2.0, two_spins[j] / 2.0
        for di, dj in ((1, -1), (-1, 1)):
            ok = (np.abs(states[:, i] + 2 * di) <= two_spins[i]) & (np.abs(states[:, j] + 2 * dj) <= two_spins[j])
            src = np.nonzero(ok)[0]
            if src.size == 0:
                continue
            t = states[src].copy()
            t[:, i] += 2 * di
            t[:, j] += 2 * dj
            dst = np.searchsorted(codes, _encode(t, two_spins))
            a = np.sqrt(Si * (Si + 1) - mi[src] * (mi[src] + di))
            b = np.sqrt(Sj * (Sj + 1) - mj[src] * (mj[src] + dj))
            rows.append(dst); cols.append(src); vals.append(0.5 * c * a * b)
    vals[0] = vals[0] + diag
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    H.sum_duplicates()
    return H, states


def config_hamiltonian(S2, pair_spins2, c1, c3, c17):
    """Block of one pair-spin configuration; S2 = 2S, pair_spins2 = 2 S~ per outer pair."""
    two = [S2] * 6
    bl = [(c17, i, (i + 1) % 6) for i in range(6)]
    const = 0.0
    S = S2 / 2
    for k, ts in enumerate(pair_spins2):
        St = ts / 2
        const += c1 * (St * (St + 1) - 2 * S * (S + 1)) / 2
        if ts == 0:
            continue
        two.append(ts)
        idx = len(two) - 1
        bl += [(c3, idx, k), (c3, idx, (k + 1) % 6)]
    return heisenberg_mixed(two, bl, const)


def canonical(cfg):
    rots = [tuple(cfg[(i + r) % 6] for i in range(6)) for r in range(6)]
    return min(rots + [tuple(c[::-1]) for c in rots])


def landscape(S2, c1, c3, classes=None):
    """Floating-point lowest eigenvalue of every pair-spin class (or of the given classes)."""
    c17 = (1 - 2 * c1 - 8 * c3) / 2
    allowed = list(range(0, 2 * S2 + 1, 2))
    counts = {}
    for cfg in itertools.product(allowed, repeat=6):
        counts[canonical(cfg)] = counts.get(canonical(cfg), 0) + 1
    out = []
    for cfg, mult in sorted(counts.items()):
        if classes is not None and cfg not in classes:
            continue
        H, _ = config_hamiltonian(S2, cfg, c1, c3, c17)
        d = H.shape[0]
        if d <= 400:
            e = float(np.linalg.eigvalsh(H.toarray())[0])
        else:
            v0 = np.random.default_rng(1).standard_normal(d)
            e = float(spla.eigsh(H, k=1, which="SA", tol=1e-12, v0=v0, return_eigenvectors=False)[0])
        out.append((e, list(cfg), d, mult))
    out.sort()
    return out


def printed_coupling_check(c1=PRINTED["c1"], c3=PRINTED["c3"], published=PRINTED["published_bound"],
                           exact=True):
    """Rigorous: the all-singlet configuration has eigenvalue -12 c1 + c17 E0(ring). E0 is enclosed by
    certify() (S2 + S3, standard floating-point model) and, with exact=True and python-flint, again by
    exact_recheck_lowest (integer arithmetic only, scaled spin-1 basis); the bound used is the exact one
    when available. Returns the certified upper bound on that eigenvalue and whether it lies below the
    published bound."""
    import scipy.linalg as la
    from ...certify import Claim, certify
    from ...construct import build_bitwise
    from ...spec import spin1_heisenberg_chain
    c1, c3, pub = Fraction(c1), Fraction(c3), Fraction(published)
    c17 = (1 - 2 * c1 - 8 * c3) / 2
    spec = spin1_heisenberg_chain(6)
    H = build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    cert = certify(Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec)
    st = {s["id"]: s for s in cert.statements}
    assert st["S2.eigenpairs"]["status"] == "CERTIFIED" and st["S3.completeness"]["status"] == "CERTIFIED"
    ev = st["S2.eigenpairs"]["evidence"]
    rad = Fraction(ev["max_radius"]) + Fraction(ev["model_shift"])
    out = {"E0_radius_standard_model": float(rad)}
    if exact:
        try:
            from ...checker import exact_recheck_lowest
            r = exact_recheck_lowest(spec, w[:1], V[:, :1], next_hint=w[1])
            out["exact_recheck"] = r["status"]
            if r["status"] == "CERTIFIED":
                rad = Fraction(r["radii"][0])
                out["E0_radius_exact"] = float(rad)
        except ImportError:
            out["exact_recheck"] = "NOT_CHECKED (python-flint missing)"
    E0 = Fraction(float(w[0]))
    val_hi = -12 * c1 + c17 * (E0 + rad if c17 > 0 else E0 - rad)
    out.update({"c1": str(c1), "c3": str(c3), "c17": str(c17), "E0_ring": float(w[0]), "E0_radius": float(rad),
                "eigenvalue_upper": float(val_hi), "published_bound": float(pub),
                "contradiction": bool(val_hi < pub), "margin": float(pub - val_hi),
                "no_floating_point_assumption": out.get("exact_recheck") == "CERTIFIED"})
    return out
