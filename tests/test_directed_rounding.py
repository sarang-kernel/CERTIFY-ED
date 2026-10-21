"""Interval endpoints are rounded outward: a reported endpoint is on the safe side of the exact one.

The certified statement |lambda - mu| <= r is about the exact reals mu and r. Endpoints mu -/+ r computed
with round-to-nearest can land half an ulp inside the exact interval; sub_down/add_up step outward."""
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from certify_ed.enclosure import OperatorAdapter, enclose
from certify_ed.fp import add_down, add_up, sub_down, sub_up
from certify_ed.spectrum import enclose_spectrum


def test_helpers_bound_the_exact_result():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(2000) * 10.0 ** rng.integers(-3, 4, 2000)
    b = rng.standard_normal(2000) * 10.0 ** rng.integers(-18, 2, 2000)
    for x, y in zip(a, b):
        X, Y = Fraction(float(x)), Fraction(float(y))
        assert Fraction(float(sub_down(x, y))) <= X - Y <= Fraction(float(sub_up(x, y)))
        assert Fraction(float(add_down(x, y))) <= X + Y <= Fraction(float(add_up(x, y)))


def test_round_to_nearest_would_land_inside():
    # 1 - 2^-60 rounds to 1.0 under round-to-nearest: the naive lower end exceeds the exact one
    mu, r = 1.0, 2.0 ** -60
    assert Fraction(mu - r) > Fraction(mu) - Fraction(r)
    assert Fraction(float(sub_down(mu, r))) <= Fraction(mu) - Fraction(r)


def test_enclosure_endpoints_contain_the_exact_interval():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((40, 40))
    A = A + A.T
    w, V = np.linalg.eigh(A)
    enc = enclose(OperatorAdapter(A), w[:10], V[:, :10])
    for i, rec in enumerate(enc.to_records()):
        mu, r = Fraction(float(enc.mu[i])), Fraction(float(enc.radius[i]))
        assert Fraction(rec["lower"]) <= mu - r and Fraction(rec["upper"]) >= mu + r
        lo, hi = enc.interval(i)
        assert Fraction(lo) <= mu - r and Fraction(hi) >= mu + r


def test_spectrum_intervals_contain_every_member_interval():
    n = 30
    H = sp.random(n, n, density=0.2, random_state=11)
    H = (H + H.T).tocsr()
    sp_enc = enclose_spectrum(H)
    assert sp_enc.complete
    w = np.linalg.eigvalsh(H.toarray())
    assert all(sp_enc.contains(x) for x in w)
    for iv in sp_enc.intervals:
        assert iv.lo <= iv.hi
