"""Union graphs for the Pomata-Wei lattices (geometry reconstructed from the paper's description)."""
from pw_general import Graph


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
