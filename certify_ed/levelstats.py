"""Level statistics from an enclosed, counted spectrum (roadmap C-09): interval r-ratios and spacing histogram.

Input: a certificate of a whole-spectrum claim (kind 'all') whose S2 and S3 are CERTIFIED, so that the S2
enclosures (radii including the model shift) contain every eigenvalue and the levels of
`gaps.merge_enclosures` carry eigenvalue ranks (count = dimension, then the pigeonhole step of docs/theory.md
Section 6). Everything below is interval arithmetic (certify_ed/ivl.py, dyadic with outward rounding) on
those intervals:

* lambda_j in L_j = [lo, hi] of the level holding rank j (a cluster of c eigenvalues gives c copies of its
  level interval);
* spacings s_j = lambda_{j+1} - lambda_j in L_{j+1} - L_j, intersected with [0, inf) (the eigenvalues are
  sorted);
* r-ratios r_j = min(s_j, s_{j+1}) / max(s_j, s_{j+1}) by interval min, max and division, intersected with
  [0, 1] (true by definition); undefined when the larger spacing may be zero (its interval contains 0);
* the mean r as the interval sum divided by the number of defined ratios;
* the spacing histogram in units of the mean spacing (an interval): per bin the number of spacings whose
  interval lies inside the bin ("at_least") and the number whose interval meets the bin ("at_most").

The unfolding is a declared parameter: 'none', or a polynomial degree d. With a degree, the staircase
N(E) = j + 1/2 at the midpoints is fitted by a least-squares polynomial p (numpy, floating point), and the
statistics are computed on p(L_j), evaluated by interval Horner. The polynomial is a modelling choice: the
output records its coefficients and marks it "declared, not certified". The intervals are certified *given*
the declared polynomial; a different unfolding gives different numbers.

No new mathematics: interval arithmetic on certified intervals (Lemma 17 machinery) and the ordering of the
levels from S3.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .certificate_v2 import Certificate
from .crossings import _radii, certificate_hash
from .gaps import merge_enclosures
from .ivl import Interval, from_float, sum_intervals

__all__ = ["level_statistics", "format_level_statistics", "LevelStatisticsError"]


class LevelStatisticsError(ValueError):
    """The certificate does not carry a complete, counted spectrum."""


def _iv(lo: float, hi: float) -> Interval:
    a, b = from_float(float(lo)), from_float(float(hi))
    return Interval(a.lo, b.hi)


def _iv_min(x: Interval, y: Interval) -> Interval:
    return Interval(min(x.lo, y.lo), min(x.hi, y.hi))


def _iv_max(x: Interval, y: Interval) -> Interval:
    return Interval(max(x.lo, y.lo), max(x.hi, y.hi))


def _clip(x: Interval, lo: Optional[int], hi: Optional[int]) -> Interval:
    """Intersection with [lo, hi] (scaled ints); the true value is known to lie there."""
    a = x.lo if lo is None else max(x.lo, lo)
    b = x.hi if hi is None else min(x.hi, hi)
    if a > b:
        raise LevelStatisticsError("interval arithmetic produced an empty intersection (inconsistent input)")
    return Interval(a, b)


def _horner(coeffs: Sequence[float], x: Interval) -> Interval:
    """p(x) = c_0 x^d + ... + c_d on an interval (outward rounding at every step)."""
    acc = from_float(float(coeffs[0]))
    for c in coeffs[1:]:
        acc = acc * x + from_float(float(c))
    return acc


def _rank_intervals(cert: Certificate):
    """One interval per eigenvalue rank, from the merged levels of the whole-spectrum enclosure."""
    got = _radii(cert)
    if got is None:
        raise LevelStatisticsError("certificate carries no enclosures")
    mu, rad = got
    levels = merge_enclosures(mu, rad)
    out: List[Interval] = []
    for lv in levels:
        out += [_iv(lv["lower"], lv["upper"])] * lv["count"]
    mids = np.sort(np.asarray(mu, dtype=float))
    return out, mids, levels


def _check_input(cert: Certificate) -> None:
    kind = (cert.body.get("claim") or {}).get("kind")
    s2, s3 = cert.status_of("S2.eigenpairs"), cert.status_of("S3.completeness")
    problems = []
    if kind != "all":
        problems.append(f"the claim kind is {kind!r}, not 'all' (a whole-spectrum claim)")
    if s2 != "CERTIFIED":
        problems.append(f"S2.eigenpairs is {s2}, not CERTIFIED (no certified enclosures)")
    if s3 != "CERTIFIED":
        problems.append(f"S3.completeness is {s3}, not CERTIFIED (the spectrum is not counted)")
    if problems:
        raise LevelStatisticsError("level statistics need an enclosed, counted spectrum: kind 'all' with S2 and S3 "
                                   "CERTIFIED; this certificate: " + "; ".join(problems))
    dim = int(((cert.body.get("subject") or {}).get("operator") or {}).get("dimension", 0))
    n = int((cert.body.get("claim") or {}).get("n_eigenvalues", 0))
    if dim and n != dim:
        raise LevelStatisticsError(f"claim has {n} eigenvalues but the operator has dimension {dim}")


def level_statistics(cert: Certificate, unfolding="none", bins: int = 12, bin_range=(0.0, 3.0)) -> Dict[str, Any]:
    """Interval level statistics of a certified whole spectrum. See the module docstring.

    unfolding: 'none' or an integer polynomial degree (declared, not certified). bins / bin_range: the spacing
    histogram, in units of the mean spacing. Raises LevelStatisticsError (a ValueError) when the certificate
    is not a kind-'all' claim with S2 and S3 CERTIFIED."""
    _check_input(cert)
    L, mids, levels = _rank_intervals(cert)
    n = len(L)
    if n < 3:
        raise LevelStatisticsError(f"at least 3 eigenvalues are needed for one spacing ratio (got {n})")
    unf: Dict[str, Any] = {"kind": "none", "status": "declared, not certified",
                           "note": "no unfolding: statistics of the raw spectrum"}
    E = L
    e_mid = mids
    if unfolding not in (None, "none"):
        d = int(unfolding)
        if d < 1:
            raise LevelStatisticsError("polynomial unfolding needs degree >= 1")
        coeffs = np.polyfit(mids, np.arange(n) + 0.5, d)
        p_mid = np.polyval(coeffs, mids)
        monotone = bool(np.all(np.diff(p_mid) >= 0))
        unf = {"kind": "polynomial", "degree": d, "coefficients": [float(c) for c in coeffs],
               "fit": "least squares of the staircase N(E) = j + 1/2 at the midpoints (numpy.polyfit, floating point)",
               "monotone_on_midpoints": monotone, "status": "declared, not certified",
               "note": ("the polynomial is a modelling choice; the intervals below are certified given this polynomial, "
                        "evaluated on the level intervals by interval Horner")}
        E = [_horner(coeffs, x) for x in L]
        e_mid = p_mid
        # p is increasing on the spectrum when its derivative is positive on the hull of the level intervals
        # (interval Horner of p' on the hull); only then are the unfolded spacings known to be >= 0
        dcoeffs = np.polyder(coeffs)
        hull = Interval(min(x.lo for x in L), max(x.hi for x in L))
        dp = _horner(dcoeffs, hull)
        increasing = dp.lo > 0
        unf["derivative_on_hull"] = list(dp.floats())
        unf["increasing_certified"] = bool(increasing)
    else:
        increasing = True
    # spacings
    S: List[Interval] = []
    s_float = np.diff(e_mid)
    for j in range(n - 1):
        s = E[j + 1] - E[j]
        S.append(_clip(s, 0, None) if increasing else s)
    # r-ratios
    one = from_float(1.0)
    ratios: List[Optional[Interval]] = []
    r_float: List[Optional[float]] = []
    for j in range(n - 2):
        a, b = S[j], S[j + 1]
        mx = _iv_max(a, b)
        fa, fb = float(s_float[j]), float(s_float[j + 1])
        r_float.append(min(fa, fb) / max(fa, fb) if max(fa, fb) > 0 else None)
        mn = _iv_min(a, b)
        if mx.lo <= 0 or mn.lo < 0:          # the larger spacing may vanish, or a spacing may be negative
            ratios.append(None)
            continue
        r = mn / mx
        ratios.append(_clip(r, 0, one.hi))
    defined = [r for r in ratios if r is not None]
    mean_r = None
    if defined:
        tot = sum_intervals(defined)
        mean_r = tot / from_float(float(len(defined)))
    # histogram in units of the mean spacing (an interval)
    span = E[-1] - E[0]
    hist = None
    if span.lo > 0 and increasing:
        mean_s = span / from_float(float(n - 1))
        lo_e, hi_e = float(bin_range[0]), float(bin_range[1])
        edges = np.linspace(lo_e, hi_e, int(bins) + 1)
        edge_iv = [from_float(float(e)) for e in edges]
        at_least = [0] * int(bins)
        at_most = [0] * int(bins)
        above = 0
        for s in S:
            x = s / mean_s
            for k in range(int(bins)):
                e0, e1 = edge_iv[k], edge_iv[k + 1]
                if x.lo >= e0.hi and x.hi < e1.lo:
                    at_least[k] += 1
                if x.hi >= e0.lo and x.lo < e1.hi:
                    at_most[k] += 1
            if x.lo >= edge_iv[-1].lo:
                above += 1
        hist = {"unit": "mean spacing (interval)", "mean_spacing": list(mean_s.floats()),
                "edges": [float(e) for e in edges], "at_least": at_least, "at_most": at_most,
                "above_range_at_least": above, "n_spacings": len(S),
                "note": ("at_least: spacings whose interval lies inside the bin; at_most: spacings whose interval meets "
                         "the bin; the true histogram lies between the two")}
    rf = [None if x is None or iv is None else float(x) for x, iv in zip(r_float, ratios)]
    rec = {"schema": "certify-ed/levelstats/1.0", "certificate_hash": certificate_hash(cert),
           "dimension": n, "n_levels": len(levels), "unfolding": unf,
           "basis": ("interval arithmetic (certify_ed.ivl, outward rounding) on the S2 enclosures ordered by S3 "
                     "(count = dimension; pigeonhole step of docs/theory.md Section 6); r in [0, 1] by definition"),
           "spacings": [list(s.floats()) for s in S],
           "spacings_float": [float(x) for x in s_float],
           "r_ratios": [None if r is None else list(r.floats()) for r in ratios],
           "r_ratios_float": rf,
           "n_ratios_defined": len(defined), "n_ratios_undefined": len(ratios) - len(defined),
           "mean_r": None if mean_r is None else list(mean_r.floats()),
           "mean_r_float": (float(np.mean([x for x in rf if x is not None])) if any(x is not None for x in rf) else None),
           "max_level_width": float(max(x.width() for x in L)),
           "histogram": hist}
    return rec


def format_level_statistics(rec: Dict[str, Any]) -> str:
    out = [f"level statistics of certificate {rec['certificate_hash']}: {rec['dimension']} eigenvalues, "
           f"{rec['n_levels']} levels, max level width {rec['max_level_width']:.3e}",
           f"unfolding: {rec['unfolding']['kind']}" + (f" degree {rec['unfolding']['degree']}" if rec['unfolding']['kind'] == 'polynomial' else "")
           + f" ({rec['unfolding']['status']})"]
    if rec["mean_r"] is not None:
        lo, hi = rec["mean_r"]
        out.append(f"mean r in [{lo:.12g}, {hi:.12g}] over {rec['n_ratios_defined']} defined ratios "
                   f"({rec['n_ratios_undefined']} undefined: the larger spacing may be zero); float value {rec['mean_r_float']:.12g}")
    else:
        out.append("mean r: no defined ratio")
    h = rec["histogram"]
    if h:
        out.append(f"spacing histogram in units of the mean spacing {h['mean_spacing']} ({h['n_spacings']} spacings):")
        for k in range(len(h["at_least"])):
            out.append(f"  [{h['edges'][k]:.3g}, {h['edges'][k + 1]:.3g}): {h['at_least'][k]} .. {h['at_most'][k]}")
        if h["above_range_at_least"]:
            out.append(f"  >= {h['edges'][-1]:.3g}: at least {h['above_range_at_least']}")
    return "\n".join(out)
