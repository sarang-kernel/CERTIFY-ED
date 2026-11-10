"""Entrywise comparison of the AKLT ingredients rebuilt in Lean with certify_ed.apps.aklt (graph.py, kernel.py).

For each lattice of E10: the graph (sites, degrees, edge list in order, regions L, C, R), the Gram matrices
G_L, G_C, G_R (reduced_region), G_i, G_j (union_coords + gram_of_coords), the reduced down counts, the
singlet counts and the rank modulo 2^31 - 1 of the union VBS states (rank_mod_p_lower). For the star and the
square-octagon, also the scaled AKLT Hamiltonian den H of the subgraph L u C (kernel.hamiltonian_region),
entrywise, and the Lean check H U = 0 on the reduced VBS columns. Every comparison is exact.

usage: python compare_aklt.py --exe <ceapps.exe> --code <certify-ed clone> --work <dir> --out <json>
"""
import argparse
import json
import os
import subprocess
import sys
import time


def compare_lattice(exe, name, work):
    from certify_ed.apps.aklt import LATTICES
    from certify_ed.apps.aklt.graph import reduced_region, union_coords, gram_of_coords, rank_mod_p_lower
    path = os.path.join(work, f"exp_aklt_{name}.json")
    t0 = time.time()
    subprocess.run([exe, "export-aklt", name, path], check=True)
    lean = json.load(open(path))
    g, L, C, R = LATTICES[name][0]()
    rL, rC, rR = (reduced_region(g, X) for X in (L, C, R))
    Ti, di, si, _ = union_coords(g, [L, C], [rL, rC])
    Tj, dj, sj, _ = union_coords(g, [C, R], [rC, rR])
    Gi = gram_of_coords(Ti, [rL["Gn"], rC["Gn"]])
    Gj = gram_of_coords(Tj, [rC["Gn"], rR["Gn"]])
    TU, _, _, _ = union_coords(g, [L, C, R], [rL, rC, rR])
    n_unit = rank_mod_p_lower(TU)
    tolist = lambda G: [[int(x) for x in row] for row in G]
    checks = {
        "graph": lean["edges"] == [list(e) for e in g.edges] and lean["z"] == list(g.z) and
                 lean["L"] == list(L) and lean["C"] == list(C) and lean["R"] == list(R),
        "GL": lean["GL"] == tolist(rL["Gn"]), "GC": lean["GC"] == tolist(rC["Gn"]), "GR": lean["GR"] == tolist(rR["Gn"]),
        "Gi": lean["Gi"] == tolist(Gi), "Gj": lean["Gj"] == tolist(Gj),
        "downs": lean["downsL"] == [int(x) for x in rL["downs"]] and lean["downsR"] == [int(x) for x in rR["downs"]]
                 and lean["downs_i"] == [int(x) for x in di] and lean["downs_j"] == [int(x) for x in dj],
        "singlets": lean["singlets_i"] == si and lean["singlets_j"] == sj and lean["innerL"] == rL["inner"]
                    and lean["innerR"] == rR["inner"],
        "unit_cosines": lean["unit_cosines_at_least"] == n_unit,
    }
    return {"lattice": name, "dims": [rL["n"], rC["n"], rR["n"], len(Ti), len(Tj)], "unit_cosines": n_unit,
            "checks": checks, "entries_equal": all(checks.values()), "seconds": round(time.time() - t0, 1)}


def compare_ham(exe, name, work):
    from certify_ed.apps.aklt import LATTICES
    from certify_ed.apps.aklt.kernel import hamiltonian_region
    path = os.path.join(work, f"exp_aklt_ham_{name}.txt")
    t0 = time.time()
    subprocess.run([exe, "export-aklt-ham", name, path], check=True)
    with open(path) as f:
        N, den, nv, ok = f.readline().split()
        ent = {}
        for line in f:
            r, c, v = (int(x) for x in line.split())
            ent[(r, c)] = v
    g, L, C, R = LATTICES[name][0]()
    H, den_p, _ = hamiltonian_region(g, L + C)
    Hc = H.tocoo()
    py = {}
    for r, c, v in zip(Hc.row.tolist(), Hc.col.tolist(), Hc.data.tolist()):
        if v:
            py[(r, c)] = py.get((r, c), 0) + int(v)
    py = {k: v for k, v in py.items() if v}
    return {"lattice": name, "region": "L u C", "dim": int(N), "den_lean": int(den), "den_python": int(den_p),
            "nonzeros": len(ent), "entries_equal": ent == py and int(den) == int(den_p),
            "lean_H_U_zero": ok == "true", "vbs_columns": int(nv), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    os.makedirs(a.work, exist_ok=True)
    res = {"lattices": [], "hamiltonians": []}
    for name in ["star", "square_octagon_par", "square_octagon_perp", "cross", "decorated_honeycomb", "decorated_square"]:
        r = compare_lattice(a.exe, name, a.work)
        print(json.dumps(r), flush=True)
        res["lattices"].append(r)
    for name in ["star", "square_octagon_par"]:
        r = compare_ham(a.exe, name, a.work)
        print(json.dumps(r), flush=True)
        res["hamiltonians"].append(r)
    res["all_equal"] = all(x["entries_equal"] for x in res["lattices"]) and \
        all(x["entries_equal"] and x["lean_H_U_zero"] for x in res["hamiltonians"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("all equal:", res["all_equal"])


if __name__ == "__main__":
    main()
