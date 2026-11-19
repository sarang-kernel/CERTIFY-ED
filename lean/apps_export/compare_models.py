"""Entrywise comparison of the models rebuilt in Lean (ceapps export ...) with the Python builder of certify-ed.

For each model the Lean program writes the integer matrix Hint = den_L H on its S^z state set; this script
builds the same model with certify_ed (the spec builders the experiments use) and its exact form with
certify_ed.checker.exact_matrix, and compares the two as sets of (row state, column state, rational value).
For sector matrices, certify_ed.exact_sectors builds the orbit-sum columns and the Gaussian-integer sector
matrix; the two bases differ by the choice of representative and by a factor per column, so the comparison
is A_py[a, b] = conj(l_a) l_b (den_py / den_L) A_lean[a, b] with l_a the value of the Python column at the
Lean representative, together with D_py[a] = |l_a|^2 N_lean[a]. Every comparison is exact (Python ints and
Fractions).

usage: python compare_models.py --exe <ceapps.exe> --code <certify-ed clone> --work <dir> --out <json> [--big]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np


def read_lean_heis(path):
    with open(path) as f:
        N, n, den = (int(x) for x in f.readline().split())
        states = [int(f.readline()) for _ in range(n)]
        ent = {}
        for line in f:
            r, c, v = line.split()
            ent[(states[int(r)], states[int(c)])] = int(v)
    return N, states, den, ent


def py_model(key):
    import certify_ed as ce
    from certify_ed.apps.pyrochlore.model import spec_for, K_SCALE
    table = {
        "chain8": (lambda: ce.heisenberg_chain(8, J2=0.0), 4, 1),
        "chain8_J2": (lambda: ce.heisenberg_chain(8, J2=0.5), 4, 1),
        "chain12_J2": (lambda: ce.heisenberg_chain(12, J2=0.5), 6, 1),
        "chain20": (lambda: ce.heisenberg_chain(20, J2=0.0), 10, 1),
        "chain20_J2": (lambda: ce.heisenberg_chain(20, J2=0.5), 10, 1),
        "square4": (lambda: ce.heisenberg_square(4, 4, J2=0.0), 8, 1),
        "square4_J2": (lambda: ce.heisenberg_square(4, 4, J2=0.5), 8, 1),
        "square42_J2": (lambda: ce.heisenberg_square(4, 2, J2=0.5), 4, 1),
        # the crown spec is K H with K = K_SCALE (model.spec_for, scale=K_SCALE), as certify_crown uses it
        "crown": (lambda: spec_for(381414, 817957, scale=K_SCALE)[0], 9, K_SCALE),
    }
    build, up, scale = table[key]
    spec = build()
    st = ce.u1_state_set(spec, up)
    return spec, st, scale


def compare_heis(exe, key, work):
    from certify_ed.checker import exact_matrix
    path = os.path.join(work, f"exp_{key}.txt")
    t0 = time.time()
    subprocess.run([exe, "export", key, path], check=True)
    N, states, den_l, ent_l = read_lean_heis(path)
    spec, st, scale = py_model(key)
    ex = exact_matrix(spec, st)
    order = st.states_in_order()
    ok_states = sorted(int(s) for s in order) == states
    # Python: (scale H) = (re + i im) / den_py; Lean: H = Hint / den_l
    imag_zero = all(int(b) == 0 for b in ex.im)
    weights_one = all(int(w) == 1 for w in ex.weights)
    ent_p = {}
    for r, c, a in zip(ex.rows.tolist(), ex.cols.tolist(), ex.re.tolist()):
        ent_p[(int(order[r]), int(order[c]))] = Fraction(int(a), int(ex.den) * scale)
    ent_lq = {k: Fraction(v, den_l) for k, v in ent_l.items()}
    same = ent_p == ent_lq
    return {"model": key, "sites": N, "states": len(states), "nonzeros_lean": len(ent_l),
            "nonzeros_python": len(ent_p), "state_sets_equal": ok_states, "python_imaginary_part_zero": imag_zero,
            "python_weights_one": weights_one, "entries_equal": same, "den_lean": den_l, "den_python": int(ex.den),
            "python_scale": scale, "seconds": round(time.time() - t0, 1)}


def py_gens(key, spec):
    from certify_ed.basis import translation, reflection, spin_flip
    from certify_ed.exact_sectors import site_permutation
    from certify_ed.apps.pyrochlore.model import symmetry_generators
    if key.startswith("chain"):
        return [translation(spec), reflection(spec), spin_flip(spec)]
    if key.startswith("square"):
        Lx, Ly = (4, 2) if key.startswith("square42") else (4, 4)
        site = lambda x, y: (y % Ly) * Lx + (x % Lx)
        px = [0] * (Lx * Ly); py = [0] * (Lx * Ly)
        for y in range(Ly):
            for x in range(Lx):
                px[site(x, y)] = site(x + 1, y); py[site(x, y)] = site(x, y + 1)
        return [site_permutation(spec, px, "Tx"), site_permutation(spec, py, "Ty"), spin_flip(spec)]
    if key == "crown":
        return symmetry_generators(spec)
    raise KeyError(key)


def compare_sector(exe, key, exps, work):
    """exps: exponent k per generator (character value i^k)."""
    from certify_ed.checker import exact_matrix
    from certify_ed.exact_sectors import FiniteGroup, orbit_columns, sector_integer_matrix
    tag = "_".join(str(k) for k in exps)
    path = os.path.join(work, f"exp_sector_{key}_{tag}.txt")
    t0 = time.time()
    subprocess.run([exe, "export-sector", key, ",".join(str(k) for k in exps), path], check=True)
    with open(path) as f:
        head = f.readline().split()
        dim, den_l, order_l = int(head[0]), int(head[1]), int(head[2])
        commute_l, herm_l = head[3] == "true", head[4] == "true"
        reps = [int(x) for x in f.readline().split()]
        nrm = [int(x) for x in f.readline().split()]
        A = {}
        for line in f:
            t, a, b, v = line.split()
            re_, im_ = A.get((int(a), int(b)), (0, 0))
            A[(int(a), int(b))] = (re_ + int(v), im_) if t == "R" else (re_, im_ + int(v))
    spec, st, scale = py_model(key)
    gens = py_gens(key, spec)
    ex = exact_matrix(spec, st)
    G = FiniteGroup(gens, st)
    units = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    chi = G.character([units[k % 4] for k in exps])
    if chi is None:
        return {"model": key, "exponents": exps, "error": "not a character in Python", "entries_equal": False}
    cols = orbit_columns(G, chi)
    Are, Aim, D = sector_integer_matrix(ex, cols)
    asc = sorted(int(s) for s in st.states_in_order())
    lean_rep_states = np.array([asc[r] for r in reps], dtype=np.int64)
    idx, found = st.index_of(lean_rep_states)
    owner = {}
    for o, col in enumerate(cols):
        for t in col:
            owner[t] = o
    perm = [owner.get(int(i)) for i in idx]
    ok_cover = bool(np.all(found)) and None not in perm and sorted(perm) == list(range(len(cols)))
    same = ok_cover and len(cols) == dim
    if same:
        lam = [cols[perm[a]][int(idx[a])] for a in range(dim)]
        ratio = Fraction(int(ex.den) * scale, den_l)       # A_py = V^H (den_py K H) V, A_lean = den_l w^H H w
        for a in range(dim):
            if int(D[perm[a]]) != (lam[a][0] ** 2 + lam[a][1] ** 2) * nrm[a]:
                same = False
                break
        if same:
            for a in range(dim):
                ar, ai = lam[a]
                for b in range(dim):
                    br, bi = lam[b]
                    fr, fi = ar * br + ai * bi, ar * bi - ai * br      # conj(l_a) l_b
                    lr, li = A.get((a, b), (0, 0))
                    pr, pi = fr * lr - fi * li, fr * li + fi * lr
                    if Fraction(int(Are[perm[a], perm[b]])) != ratio * pr or \
                            Fraction(int(Aim[perm[a], perm[b]])) != ratio * pi:
                        same = False
                        break
                if not same:
                    break
    return {"model": key, "exponents": exps, "dim_lean": dim, "dim_python": len(cols), "group_order_lean": order_l,
            "group_order_python": G.order, "lean_commutes": commute_l, "lean_hermitian": herm_l,
            "columns_matched": ok_cover, "entries_equal": bool(same), "seconds": round(time.time() - t0, 1)}


def compare_spin1_ring(exe, work):
    import certify_ed as ce
    from certify_ed.checker import exact_matrix
    from certify_ed.construct import full_state_set
    path = os.path.join(work, "exp_spin1_ring.txt")
    t0 = time.time()
    subprocess.run([exe, "export-spin1-ring", path], check=True)
    with open(path) as f:
        N, n, den_l = (int(x) for x in f.readline().split())
        ent_l = {}
        for line in f:
            r, c, v = (int(x) for x in line.split())
            ent_l[(r, c)] = v
    spec = ce.spin1_heisenberg_chain(6)
    ex = exact_matrix(spec)
    w = [int(x) for x in ex.weights]
    sio = full_state_set(spec).states_in_order()
    # certify-ed works with K = D^-1 H D, D = diag(sqrt(w)); then H_rc = K_rc sqrt(w_c / w_r) up to the
    # orientation of D, so the exact test is H_rc^2 w_r = K_rc^2 w_c... both orientations are tried
    ok_a, ok_b = True, True
    seen = set()
    for r, c, a in zip(ex.rows.tolist(), ex.cols.tolist(), ex.re.tolist()):
        K = Fraction(int(a), int(ex.den))
        R, C = int(sio[r]), int(sio[c])
        h = ent_l.get((R, C), 0)
        sign_ok = (K > 0) == (h > 0) and (K == 0) == (h == 0)
        if not (sign_ok and K * K * w[r] == h * h * w[c]):
            ok_a = False
        if not (sign_ok and K * K * w[c] == h * h * w[r]):
            ok_b = False
        seen.add((R, C))
    ok = (ok_a or ok_b) and seen == set(ent_l)
    return {"model": "spin1_ring6", "states": n, "nonzeros_lean": len(ent_l), "nonzeros_python": len(ex.rows),
            "entries_equal": ok, "orientation": "K_rc^2 w_r = H_rc^2 w_c" if ok_a else "K_rc^2 w_c = H_rc^2 w_r",
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--big", action="store_true", help="also the full E7 chains (184756 states)")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    os.makedirs(a.work, exist_ok=True)
    res = {"heisenberg": [], "sectors": [], "spin1": []}
    keys = ["chain8", "chain8_J2", "chain12_J2", "square42_J2", "square4", "square4_J2", "crown"]
    if a.big:
        keys += ["chain20", "chain20_J2"]
    for k in keys:
        r = compare_heis(a.exe, k, a.work)
        print(json.dumps(r), flush=True)
        res["heisenberg"].append(r)
    sectors = [("chain8", [0, 0, 2]), ("chain8_J2", [0, 2, 2]), ("chain12_J2", [0, 0, 0]), ("chain12_J2", [0, 2, 2]),
               ("square42_J2", [1, 0, 2]), ("square42_J2", [2, 2, 0]), ("square4_J2", [1, 2, 0]),
               ("square4", [0, 0, 0]), ("crown", [0, 2, 0, 2, 2, 0, 0]), ("crown", [2, 2, 2, 2, 2, 2, 2])]
    for k, e in sectors:
        r = compare_sector(a.exe, k, e, a.work)
        print(json.dumps(r), flush=True)
        res["sectors"].append(r)
    r = compare_spin1_ring(a.exe, a.work)
    print(json.dumps(r), flush=True)
    res["spin1"].append(r)
    res["all_equal"] = all(x.get("entries_equal") for g in ("heisenberg", "sectors", "spin1") for x in res[g]) and \
        all(x.get("state_sets_equal", True) for x in res["heisenberg"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("all equal:", res["all_equal"])


if __name__ == "__main__":
    main()
