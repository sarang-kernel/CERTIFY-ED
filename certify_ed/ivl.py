"""Interval arithmetic on dyadic rationals with outward rounding, including exp and ln (Lemma 17).

Numbers are Python integers at a fixed scale 2^PREC (PREC = 192 bits). Every operation rounds its lower
end down (floor) and its upper end up (ceil), so an interval always contains the exact result of the
exact operation on any members of its operands. exp and ln are enclosed by truncated series with explicit
remainder bounds (docs/theory.md, Section 9.3, Lemma 17). No floating-point arithmetic is used except in
`from_float` (exact: a binary64 is a dyadic rational) and in `to_float_down/up` (outward).

This module exists for S7 (thermodynamic intervals), where the standard-model floating-point bounds of
`fp.py` would need a rounding analysis of exp; it is self-contained and needs no optional dependency.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Tuple

PREC = 192
ONE = 1 << PREC

__all__ = ["PREC", "Interval", "from_float", "from_fraction", "to_float_down", "to_float_up", "exp_interval",
           "ln_interval", "sum_intervals"]


def _floor_div(a: int, b: int) -> int:
    return a // b                     # Python floors


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def from_fraction(q: Fraction) -> "Interval":
    """Smallest dyadic interval at PREC containing the rational q."""
    n, d = q.numerator, q.denominator
    return Interval(_floor_div(n << PREC, d), _ceil_div(n << PREC, d))


def from_float(x: float) -> "Interval":
    """The binary64 x as an exact dyadic (x is a dyadic rational)."""
    if not math.isfinite(x):
        raise ValueError("non-finite float")
    return from_fraction(Fraction(x))


def _to_fraction(v: int) -> Fraction:
    return Fraction(v, ONE)


def to_float_down(v: int) -> float:
    """Largest binary64 <= v / 2^PREC."""
    q = _to_fraction(v)
    f = float(q)                      # correctly rounded to nearest
    if Fraction(f) > q:
        f = math.nextafter(f, -math.inf)
    return f


def to_float_up(v: int) -> float:
    q = _to_fraction(v)
    f = float(q)
    if Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    return f


class Interval:
    """[lo, hi] with integer ends at scale 2^PREC."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo: int, hi: int):
        if lo > hi:
            raise ValueError("empty interval")
        self.lo = int(lo)
        self.hi = int(hi)

    # -------------------------------------------------------------- arithmetic
    def __add__(self, o: "Interval") -> "Interval":
        return Interval(self.lo + o.lo, self.hi + o.hi)

    def __sub__(self, o: "Interval") -> "Interval":
        return Interval(self.lo - o.hi, self.hi - o.lo)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, o: "Interval") -> "Interval":
        ps = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(_floor_div(min(ps), ONE), _ceil_div(max(ps), ONE))

    def scale_int(self, c: int) -> "Interval":
        """Exact multiplication by an integer."""
        return Interval(self.lo * c, self.hi * c) if c >= 0 else Interval(self.hi * c, self.lo * c)

    def __truediv__(self, o: "Interval") -> "Interval":
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError("division by an interval containing 0")
        qs = (_floor_div(self.lo << PREC, o.lo), _floor_div(self.lo << PREC, o.hi),
              _floor_div(self.hi << PREC, o.lo), _floor_div(self.hi << PREC, o.hi))
        qs2 = (_ceil_div(self.lo << PREC, o.lo), _ceil_div(self.lo << PREC, o.hi),
               _ceil_div(self.hi << PREC, o.lo), _ceil_div(self.hi << PREC, o.hi))
        return Interval(min(qs), max(qs2))

    def square(self) -> "Interval":
        if self.lo >= 0:
            return Interval(_floor_div(self.lo * self.lo, ONE), _ceil_div(self.hi * self.hi, ONE))
        if self.hi <= 0:
            return Interval(_floor_div(self.hi * self.hi, ONE), _ceil_div(self.lo * self.lo, ONE))
        return Interval(0, _ceil_div(max(self.lo * self.lo, self.hi * self.hi), ONE))

    def intersect_nonneg(self) -> "Interval":
        """Intersection with [0, inf) (used for a variance, which is >= 0 by theorem)."""
        if self.hi < 0:
            raise ValueError("interval entirely negative")
        return Interval(max(self.lo, 0), self.hi)

    # -------------------------------------------------------------- queries
    def contains(self, q) -> bool:
        q = Fraction(q)
        return _to_fraction(self.lo) <= q <= _to_fraction(self.hi)

    def width(self) -> float:
        return to_float_up(self.hi - self.lo)

    def floats(self) -> Tuple[float, float]:
        return to_float_down(self.lo), to_float_up(self.hi)

    def fractions(self) -> Tuple[Fraction, Fraction]:
        return _to_fraction(self.lo), _to_fraction(self.hi)

    def __repr__(self):
        lo, hi = self.floats()
        return f"Interval[{lo!r}, {hi!r}]"


ZERO = Interval(0, 0)
UNIT = Interval(ONE, ONE)


def sum_intervals(items) -> Interval:
    lo = hi = 0
    for iv in items:
        lo += iv.lo
        hi += iv.hi
    return Interval(lo, hi)


# ---------------------------------------------------------------------------------------- exp
_EXP_TERMS = 40           # degree N of the Taylor polynomial at |t| <= 1/2: remainder < 2^-41/41! ~ 1e-62


def _exp_point_nonneg(x: int) -> Interval:
    """Enclosure of exp(x / 2^PREC) for x >= 0 (Lemma 17): argument halving, Taylor polynomial, squarings."""
    if x == 0:
        return UNIT
    s = 0
    t = x
    half = ONE >> 1
    while t > half:
        t = _ceil_div(t, 2)            # t_up >= x / 2^s: upper
        s += 1
    # lower and upper t after halving: exact when x is divisible, else bracket
    t_lo = x >> s if s else x
    t_hi = _ceil_div(x, 1 << s) if s else x
    # polynomial with directed rounding, evaluated termwise: term_k = t^k / k!
    def poly(tv: int, up: bool) -> int:
        acc = ONE
        term = ONE
        for k in range(1, _EXP_TERMS + 1):
            term = (term * tv) // ONE if not up else _ceil_div(term * tv, ONE)
            term = term // k if not up else _ceil_div(term, k)
            acc += term
        return acc, term
    lo, _ = poly(t_lo, False)
    hi, last = poly(t_hi, True)
    # remainder for the upper end: term_{N+1} / (1 - t/(N+2)) <= last * t / (N+1) * 2 (t <= 1/2 < (N+2)/2)
    rem = _ceil_div(_ceil_div(last * t_hi, ONE) * 2, _EXP_TERMS + 1)
    hi += rem
    iv = Interval(lo, hi)
    for _ in range(s):
        iv = iv.square()               # exp(x) = exp(t)^(2^s); both ends >= 1 so squaring is monotone
    return iv


def exp_interval(iv: Interval) -> Interval:
    """Enclosure of exp over the interval (exp is increasing)."""
    def exp_end(v: int, up: bool) -> int:
        if v >= 0:
            e = _exp_point_nonneg(v)
            return e.hi if up else e.lo
        e = _exp_point_nonneg(-v)      # exp(-|v|) = 1 / exp(|v|)
        return _ceil_div(ONE << PREC, e.lo) if up else _floor_div(ONE << PREC, e.hi)
    return Interval(exp_end(iv.lo, False), exp_end(iv.hi, True))


# ---------------------------------------------------------------------------------------- ln
_LN_TERMS = 60            # atanh series terms for u <= 1/3: remainder <= 2 u^(2N+3)/((2N+3)(1-u^2)) ~ 1e-59


def _atanh2_series(u_lo: int, u_hi: int) -> Interval:
    """Enclosure of 2*atanh(u) = ln((1+u)/(1-u)) for 0 <= u_lo <= u_hi <= 1/3 (scaled ints)."""
    def series(u: int, up: bool) -> int:
        acc = 0
        pw = u
        u2 = (u * u) // ONE if not up else _ceil_div(u * u, ONE)
        for j in range(_LN_TERMS + 1):
            term = pw // (2 * j + 1) if not up else _ceil_div(pw, 2 * j + 1)
            acc += term
            pw = (pw * u2) // ONE if not up else _ceil_div(pw * u2, ONE)
        if up:
            # remainder: sum_{j>N} u^(2j+1)/(2j+1) <= pw / ((2N+3)(1-u^2)); 1-u^2 >= 8/9 for u <= 1/3
            acc += _ceil_div(_ceil_div(pw * 9, 8), 2 * _LN_TERMS + 3)
        return 2 * acc
    return Interval(series(u_lo, False), series(u_hi, True))


_LN2_CACHE = None


def ln2_interval() -> Interval:
    global _LN2_CACHE
    if _LN2_CACHE is None:
        third = Interval(_floor_div(ONE, 3), _ceil_div(ONE, 3))
        _LN2_CACHE = _atanh2_series(third.lo, third.hi)
    return _LN2_CACHE


def _ln_point(v: int, up: bool) -> int:
    """One end of an enclosure of ln(v / 2^PREC), v > 0."""
    if v <= 0:
        raise ValueError("ln of a non-positive number")
    # v / 2^PREC = 2^k * m with m in [1, 2)
    k = v.bit_length() - 1 - PREC
    m = (v << -k) if k < 0 else (v >> k)          # m scaled: in [ONE, 2*ONE)
    exact_shift = (k <= 0) or (v == (m << k))
    m_lo = m
    m_hi = m if exact_shift else m + 1
    # u = (m - 1)/(m + 1) in [0, 1/3)
    u_lo = _floor_div((m_lo - ONE) << PREC, m_lo + ONE)
    u_hi = _ceil_div((m_hi - ONE) << PREC, m_hi + ONE)
    a = _atanh2_series(max(u_lo, 0), max(u_hi, 0))
    l2 = ln2_interval()
    kl = l2.scale_int(k)
    tot = a + kl
    return tot.hi if up else tot.lo


def ln_interval(iv: Interval) -> Interval:
    """Enclosure of ln over an interval with lo > 0 (ln is increasing)."""
    if iv.lo <= 0:
        raise ValueError("ln needs a positive interval")
    return Interval(_ln_point(iv.lo, False), _ln_point(iv.hi, True))
