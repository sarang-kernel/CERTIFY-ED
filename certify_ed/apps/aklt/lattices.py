"""Unions L - C - R of two overlapping subgraphs for the Pomata-Wei lattices (geometry inferred from
the paper's text and confirmed by reproducing its eta to 7-10 digits)."""
from .graph import Graph


def square(k):              # N, E, S, W of square k
    return 4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3


def square_edges(k):
    N, E, S, W = square(k)
    return [(N, E), (E, S), (S, W), (W, N)]


def square_octagon(kind):
    """kind 'par': squares 0-1-2 in a row; 'perp': square 2 above square 1."""
    edges = square_edges(0) + square_edges(1) + square_edges(2)
    edges.append((square(0)[1], square(1)[3]))              # E0 - W1
    if kind == "par":
        edges.append((square(1)[1], square(2)[3]))          # E1 - W2
    else:
        edges.append((square(1)[0], square(2)[2]))          # N1 - S2
    g = Graph([3] * 12, edges)
    return g, list(square(0)), list(square(1)), list(square(2))


def cross():
    """Hexagons h1 (a0..a5) and h2 (b0..b5) sharing the square a0-a1-b1-b0; squares on (a2,a3), (a4,a5)
    with extra sites p2..p5 and on (b2,b3), (b4,b5) with q2..q5."""
    names = ["a0", "a1", "a2", "a3", "a4", "a5", "p2", "p3", "p4", "p5",
             "b0", "b1", "b2", "b3", "b4", "b5", "q2", "q3", "q4", "q5"]
    ix = {n: i for i, n in enumerate(names)}
    E = []
    for h in "ab":
        for i in range(6):
            E.append((f"{h}{i}", f"{h}{(i + 1) % 6}"))
    E = [e for e in E if set(e) != {"b0", "b1"}]           # (b0,b1) added with the shared square
    E += [("a0", "b0"), ("a1", "b1"), ("b0", "b1")]          # shared square a0-a1-b1-b0
    for h, x in (("a", "p"), ("b", "q")):
        for i, j in ((2, 3), (4, 5)):
            E += [(f"{h}{i}", f"{x}{i}"), (f"{x}{i}", f"{x}{j}"), (f"{x}{j}", f"{h}{j}")]
    edges = [(ix[u], ix[v]) for u, v in E]
    g = Graph([3] * 20, edges)
    C = [ix[n] for n in ("a0", "a1", "b0", "b1")]
    L = [ix[n] for n in ("a2", "a3", "a4", "a5", "p2", "p3", "p4", "p5")]
    R = [ix[n] for n in ("b2", "b3", "b4", "b5", "q2", "q3", "q4", "q5")]
    return g, L, C, R


def decorated_honeycomb():
    """Once-decorated honeycomb: spin-3/2 vertices (z=3), spin-1 decorations (z=2).
    Gamma_i = H(v, w): v, w, d_vw, d_v_up, d_w_up, d_v_dn, u_v, d_w_dn, u_w ; Gamma_j = H(x, u_v), whose
    top arm at u_v is (d_v_dn, u_v) = C."""
    names = ["v", "w", "d_vw", "d_vu", "d_wu", "d_wd", "u_w",      # L
             "d_vd", "u",                                           # C
             "x", "d_ux", "d_xu", "d_ud", "d_xd", "uu", "ux"]       # R
    ix = {n: i for i, n in enumerate(names)}
    E = [("v", "d_vw"), ("d_vw", "w"), ("v", "d_vu"), ("w", "d_wu"), ("v", "d_vd"), ("d_vd", "u"),
         ("w", "d_wd"), ("d_wd", "u_w"), ("u", "d_ux"), ("d_ux", "x"), ("x", "d_xu"), ("u", "d_ud"),
         ("d_ud", "uu"), ("x", "d_xd"), ("d_xd", "ux")]
    z = [3 if n in ("v", "w", "u_w", "u", "x", "uu", "ux") else 2 for n in names]
    g = Graph(z, [(ix[a], ix[b]) for a, b in E])
    L = [ix[n] for n in names[:7]]; C = [ix[n] for n in names[7:9]]; R = [ix[n] for n in names[9:]]
    return g, L, C, R


def decorated_square():
    """Once-decorated square lattice: spin-2 vertices (z=4), spin-1 decorations (z=2) on every edge.
    Gamma_i (A1 type): plaquette with lower-left corner (0,0) plus the two outward decorations at its
    upper-right (1,1) and lower-left (0,0) corners. Gamma_j (A2 type, rotated by pi/2): plaquette with
    lower-left corner (1,1) plus the outward decorations at its upper-left (1,2) and lower-right (2,1)
    corners. C = Gamma_i cap Gamma_j = vertex (1,1) and its decorations towards (1,2) and (2,1)."""
    def plaquette_edges(x, y):
        return [((x, y), (x + 1, y)), ((x, y), (x, y + 1)), ((x + 1, y), (x + 1, y + 1)), ((x, y + 1), (x + 1, y + 1))]
    gi_v = [(0, 0), (1, 0), (0, 1), (1, 1)]
    gi_e = plaquette_edges(0, 0) + [((1, 1), (1, 2)), ((1, 1), (2, 1)), ((0, -1), (0, 0)), ((-1, 0), (0, 0))]
    gj_v = [(1, 1), (2, 1), (1, 2), (2, 2)]
    gj_e = plaquette_edges(1, 1) + [((0, 2), (1, 2)), ((1, 2), (1, 3)), ((2, 1), (3, 1)), ((2, 0), (2, 1))]
    sites, z = [], []
    def add(s, zz):
        if s not in sites:
            sites.append(s); z.append(zz)
        return sites.index(s)
    edges = []
    members = {"i": set(), "j": set()}
    for tag, vs, es in (("i", gi_v, gi_e), ("j", gj_v, gj_e)):
        for v in vs:
            members[tag].add(add(("v", v), 4))
        for (a, b) in es:
            d = add(("d", (a, b)), 2)
            members[tag].add(d)
            for end in (a, b):
                if end in vs:
                    e = (sites.index(("v", end)), d)
                    if e not in edges:
                        edges.append(e)
    g = Graph(z, edges)
    g.labels = list(sites)
    Cset = members["i"] & members["j"]
    L = sorted(members["i"] - Cset); C = sorted(Cset); R = sorted(members["j"] - Cset)
    return g, L, C, R


def decorated_square_reflection(g):
    """Site permutation of decorated_square() for the mirror (x, y) -> (y, x)."""
    lab = g.labels
    sw = lambda p: (p[1], p[0])
    perm = []
    for kind, key in lab:
        if kind == "v":
            perm.append(lab.index(("v", sw(key))))
        else:
            a, b = key
            img = ("d", (sw(a), sw(b))) if ("d", (sw(a), sw(b))) in lab else ("d", (sw(b), sw(a)))
            perm.append(lab.index(img))
    return perm


def star():
    """Star lattice [3.12.12]: triangles T_L - T - T_R joined by inter-triangle edges at different
    vertices of T. Gamma = an inter-triangle edge with the two triangles it joins."""
    tri = lambda a: [(a, a + 1), (a + 1, a + 2), (a, a + 2)]
    g = Graph([3] * 9, tri(0) + tri(3) + tri(6) + [(2, 3), (4, 6)])
    return g, [0, 1, 2], [3, 4, 5], [6, 7, 8]
