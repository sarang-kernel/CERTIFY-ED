"""Geometry and couplings of the crown cluster (Kranitz & Penc, Eq. D3; reduced model c3 = c12, c5 = 0)."""
from fractions import Fraction


from ...localops import LocalSpace
from ...spec import ModelSpec

N = 18
PAIRS = [(1, 2), (6, 12), (9, 16), (17, 18), (7, 13), (3, 10)]          # outer pairs (c1 bonds)
HEX = [4, 5, 11, 15, 14, 8]                                            # hexagon ring (c17 bonds)
TET_HEX = {(1, 2): (4, 5), (6, 12): (5, 11), (9, 16): (11, 15), (17, 18): (15, 14), (7, 13): (14, 8),
           (3, 10): (8, 4)}                                             # pair -> its hexagon edge
C3_SITES = (1, 12, 9, 18, 7, 10)    # outer site whose two hexagon bonds carry c3 (the other carries c12)
K_SCALE = 8 * 10 ** 7               # K H is an integer matrix for couplings n/10^7 (S.S in {±1/4, 1/2})


def bonds(c1, c3, c12, c17):
    """The 36 bonds (coupling, site i, site j), sites 1-based as in the paper."""
    b = [(c1, p[0], p[1]) for p in PAIRS]
    for p, (h1, h2) in TET_HEX.items():
        for o in p:
            for h in (h1, h2):
                b.append((c3 if o in C3_SITES else c12, o, h))
    b += [(c17, HEX[i], HEX[(i + 1) % 6]) for i in range(6)]
    assert len(b) == 36 and len({tuple(sorted((i, j))) for _, i, j in b}) == 36
    return b


def couplings(n1, n3):
    """c1 = n1/10^7, c3 = c12 = n3/10^7, c17 fixed by the embedding constraint 2c1 + 4c3 + 4c12 + 2c17 = 1."""
    c1, c3 = Fraction(n1, 10 ** 7), Fraction(n3, 10 ** 7)
    c17 = (1 - 2 * c1 - 8 * c3) / 2
    assert 2 * c1 + 4 * c3 + 4 * c3 + 2 * c17 == 1
    return c1, c3, c17


def spec_for(n1, n3, scale=1):
    """ModelSpec of scale * H_crown for spin 1/2. With scale = K_SCALE every coupling of the spec is an
    integer, so the binary64 couplings are exact and the exact layer sees the intended rational model."""
    c1, c3, c17 = couplings(n1, n3)
    bl = bonds(c1, c3, c3, c17)
    zz = [[float(scale * c), i - 1, j - 1] for c, i, j in bl]
    pm = [[float(scale * c / 2), i - 1, j - 1] for c, i, j in bl]
    if scale == K_SCALE:
        for (c, _, _), a, b in zip(bl, zz, pm):
            assert Fraction(a[0]) == scale * c and Fraction(b[0]) == scale * c / 2
    spec = ModelSpec.from_static(N, LocalSpace.spin_space("1/2", "spin"), [["zz", zz], ["+-", pm], ["-+", pm]],
                                 name="pyrochlore_crown_reduced", parameters={"n1": n1, "n3": n3, "scale": scale})
    return spec, (c1, c3, c17)


def symmetry_generators(spec):
    """Z2^7: the swap of the two outer sites of each tetrahedron (a symmetry because c3 = c12) and the
    global spin flip, as CyclicSymmetry objects for certify_ed.exact_sectors."""
    from ...basis import spin_flip
    from ...exact_sectors import site_permutation
    gens = []
    for a, b in PAIRS:
        p = list(range(N))
        p[a - 1], p[b - 1] = b - 1, a - 1
        gens.append(site_permutation(spec, p, f"swap{a}-{b}"))
    gens.append(spin_flip(spec))
    return gens
