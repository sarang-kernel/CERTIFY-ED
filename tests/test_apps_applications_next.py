"""N-11 (nine-coupling crown of Kranitz & Penc) and N-12 (cost plan of an AKLT eta certificate) on small instances."""
from fractions import Fraction

import numpy as np
import pytest

from conftest import needs_flint


def test_nine_coupling_printed_values_and_embedding():
    from certify_ed.apps.pyrochlore import nine_coupling as nc
    assert nc.printed_value("0.037416(7)") == Fraction(374167, 10 ** 7)
    assert nc.printed_value("0.037416(7)", "truncated") == Fraction(37416, 10 ** 6)
    assert nc.printed_value("-0.000546(9)") == Fraction(-5469, 10 ** 7)
    assert nc.printed_value("0.0000415(5)") == Fraction(4155, 10 ** 8)
    with pytest.raises(ValueError):
        nc.printed_value("0.0374167")
    c = nc.couplings()
    assert c[13] == -c[6] / 2 and c[15] == -c[7] and c[19] == -2 * c[4]
    assert c[17] == Fraction(1, 2) - c[1] - 2 * c[3] - 2 * c[12]
    assert c[18] == -2 * (c[2] + c[5] + c[11] / 2)
    assert all(c[k] == 0 for k in nc.ZERO)
    bad = dict(c); bad[17] += Fraction(1, 10 ** 9)                    # negative control: constraint violated
    with pytest.raises(ValueError):
        nc.check_embedding(bad)


def test_nine_coupling_orbit_table_matches_reduced_model():
    from certify_ed.apps.pyrochlore import nine_coupling as nc
    from certify_ed.apps.pyrochlore.model import bonds as nn_bonds, couplings as nn_couplings
    sizes = nc.check_orbit_table()
    assert sum(sizes.values()) == 153 and len(sizes) == 19
    # the reduced model (c3 = c12, c5 = 0, all further couplings 0) is a special case of the nine-coupling model
    c1, c3, c17 = nn_couplings(381414, 817957)
    free = {k: Fraction(0) for k in nc.FREE}
    free.update({1: c1, 3: c3, 12: c3})
    c = nc.couplings(free=free)
    assert c[17] == c17
    nine = sorted((x, min(i, j), max(i, j)) for x, i, j in nc.bonds(c))
    reduced = sorted((x, min(i, j), max(i, j)) for x, i, j in nn_bonds(c1, c3, c3, c17))
    assert nine == reduced


def test_nine_coupling_symmetry_group_is_d3d():
    from certify_ed.apps.pyrochlore import nine_coupling as nc
    G = nc.automorphisms()
    assert len(G) == 12
    orders = sorted(nc._order(p) for p in G)
    assert orders == [1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 6, 6]
    a, b = nc.c2h_generators()
    assert tuple(a[b[i] - 1] for i in range(18)) == tuple(b[a[i] - 1] for i in range(18))


def _small_sector(nc, c, n_down=2):
    from certify_ed.basis import magnetization_state_set
    K = nc.scale_for(c)
    spec = nc.spec_for(c, scale=K)
    st = magnetization_state_set(spec, 18 - 2 * n_down)     # 2 S^z = 18 - 2 n_down: C(18, n_down) states
    return K, spec, st


@needs_flint
def test_count_by_block_agrees_with_exact_count_below():
    from certify_ed.apps.pyrochlore import nine_coupling as nc
    from certify_ed.construct import build_bitwise
    from certify_ed.exact_sectors import exact_count_below, site_permutation
    c = nc.couplings()
    K, spec, st = _small_sector(nc, c)
    assert st.size == 153
    a, b = nc.c2h_generators()
    gens = [site_permutation(spec, [x - 1 for x in a], "C2prime"), site_permutation(spec, [x - 1 for x in b], "inv")]
    w = np.linalg.eigvalsh(build_bitwise(nc.spec_for(c), st).toarray())
    for x in (w[0] - 1e-6, (w[0] + w[1]) / 2, (w[4] + w[5]) / 2):
        g = Fraction(int(np.floor(x * K * 2 ** 20)), 2 ** 20)
        mine = nc.count_below_by_block(spec, st, gens, g)
        ref, info = exact_count_below(spec, st, gens, {}, [g])
        assert mine["count"] == ref[0]["count"] == int(np.sum(w * K < float(g)))
        assert sum(s["dim"] for s in mine["sectors"]) == st.size == info["dim"]
    # negative control: the swap of an outer pair is a symmetry of the reduced model only
    swap = list(range(18)); swap[0], swap[1] = 1, 0
    with pytest.raises(ValueError):
        nc.count_below_by_block(spec, st, [site_permutation(spec, swap, "swap1-2")], Fraction(0))


@needs_flint
def test_rayleigh_upper_is_an_upper_bound():
    from certify_ed.apps.pyrochlore import nine_coupling as nc
    from certify_ed.construct import build_bitwise
    c = nc.couplings()
    K, spec, st = _small_sector(nc, c)
    H = build_bitwise(nc.spec_for(c), st).toarray()
    w, V = np.linalg.eigh(H)
    ub = nc.rayleigh_upper(spec, st, V[:, 0]) / K
    assert Fraction(float(w[0])) - Fraction(1, 10 ** 12) <= ub <= Fraction(float(w[0])) + Fraction(1, 10 ** 12)
    rng = np.random.default_rng(1)
    for _ in range(5):                                       # any trial vector gives an upper bound
        assert nc.rayleigh_upper(spec, st, rng.standard_normal(st.size)) / K >= Fraction(float(w[0])) - Fraction(1, 10 ** 12)


def test_aklt_plan_reproduces_recorded_block_sizes():
    from certify_ed.apps.aklt import lattices
    from certify_ed.apps.aklt.estimate import plan_eta
    for name, largest in (("star", 70), ("decorated_honeycomb", 673)):      # E10 records (largest block)
        g, L, C, R = getattr(lattices, name)()
        assert plan_eta(g, L, C, R)["largest_block_rows"] == largest


def test_honeycomb_method3_geometry():
    from certify_ed.apps.aklt import lattices
    from certify_ed.apps.aklt.estimate import _lite
    g, L, C, R = lattices.honeycomb_method3()
    assert len(g.z) == 30 and len(g.edges) == 36 and (len(L), len(C), len(R)) == (12, 6, 12)
    assert [_lite(g, X)["n"] for X in (L, C, R)] == [2 ** 10, 2 ** 6, 2 ** 10]         # dim V = 2^26 (paper)
    assert _lite(g, sorted(set(L) | set(C)))["n"] == 2 ** 12
