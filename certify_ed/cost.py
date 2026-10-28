"""Advisory cost model: wall seconds of S1+S2+S3 against the dimension of the claim.

The model is a table of coefficients fitted to the E2 records of the 3.1.0 evidence run
(results/run_full_20261022, Windows, 8 cores, one BLAS thread per experiment; the records themselves are
not part of the repository because results/ is git-ignored). Measured certify seconds for S1+S2+S3:

    Heisenberg chain, U(1) block (Sz = 0)      dim   924: 0.7 s     3432: 2.9 s     12870: 65.5 s
    momentum k = 0 blocks of the chain          dim   810: 0.9 s     2704: 2.7 s      9252: 17.8 s
    square lattice 4x4, Sz = 0                  dim 12870: 175.4 s

Each row is fitted as log(seconds) = a + b log(dim) by least squares (two parameters, three points for the
chain and the block rows; the square row is the chain row shifted to pass through its single point). The
estimate for an unknown model is the maximum over the rows, so it is an upper envelope of what was
measured, not a prediction for a particular machine.

The estimate is advisory. It never changes a status: `budget_seconds="auto"` only turns it into the
default time budget (`auto_budget`), which is then enforced like any other budget (statements not started
within it are NOT_CHECKED with reason BUDGET_TIME, scope.py). A user who wants a statement to run
regardless passes a number or None.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# (dimension, seconds) of the 3.1.0 evidence run (E2), per model family
E2_RECORDS: Dict[str, Tuple[Tuple[int, float], ...]] = {
    "chain_u1": ((924, 0.7), (3432, 2.9), (12870, 65.5)),
    "k0_blocks": ((810, 0.9), (2704, 2.7), (9252, 17.8)),
    "square_u1": ((12870, 175.4),),
}

# the auto budget is this factor times the envelope estimate, never below the floor
AUTO_FACTOR = 5.0
AUTO_FLOOR_SECONDS = 60.0


def _fit_loglog(points) -> Tuple[float, float]:
    """Least-squares a, b in log(s) = a + b log(d)."""
    xs = [math.log(d) for d, _ in points]
    ys = [math.log(s) for _, s in points]
    n = len(points)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def _coefficients() -> Dict[str, Tuple[float, float]]:
    a_c, b_c = _fit_loglog(E2_RECORDS["chain_u1"])
    a_k, b_k = _fit_loglog(E2_RECORDS["k0_blocks"])
    d_sq, s_sq = E2_RECORDS["square_u1"][0]
    a_s = math.log(s_sq) - b_c * math.log(d_sq)          # chain exponent through the square point
    return {"chain_u1": (a_c, b_c), "k0_blocks": (a_k, b_k), "square_u1": (a_s, b_c)}


COEFFICIENTS: Dict[str, Tuple[float, float]] = _coefficients()
"""model family -> (a, b) with  seconds = exp(a) * dim ** b  (fitted at import time from E2_RECORDS)."""


def estimate_seconds(dimension: int, model: str = "auto") -> float:
    """Estimated wall seconds of S1+S2+S3 for a claim of this dimension.

    `model` is a key of COEFFICIENTS or "auto" (the maximum over all rows). Monotone increasing in the
    dimension (every fitted exponent is positive). Dimensions below 1 are treated as 1."""
    d = max(1.0, float(dimension))
    rows = COEFFICIENTS.values() if model == "auto" else (COEFFICIENTS[model],)
    return max(math.exp(a) * d ** b for a, b in rows)


def auto_budget(dimension: int, model: str = "auto", factor: float = AUTO_FACTOR,
                floor: float = AUTO_FLOOR_SECONDS) -> float:
    """Default time budget: max(floor, factor * estimate_seconds(dimension, model)); always finite."""
    est = estimate_seconds(dimension, model)
    return float(max(floor, factor * est))


def resolve_budget_seconds(value, dimension: Optional[int]) -> Tuple[Optional[float], Optional[dict]]:
    """Turn Options.budget_seconds (None, a number, or "auto") into (seconds or None, record or None).

    The record says how an "auto" budget was derived, for the certificate's scope block."""
    if value is None:
        return None, None
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            raise ValueError(f"budget_seconds must be a number, None or 'auto' (got {value!r})")
        dim = int(dimension or 0)
        est = estimate_seconds(dim)
        sec = auto_budget(dim)
        return sec, {"mode": "auto", "dimension": dim, "estimate_seconds": est, "factor": AUTO_FACTOR,
                     "floor_seconds": AUTO_FLOOR_SECONDS, "budget_seconds": sec,
                     "model": "upper envelope of the 3.1.0 E2 rows (certify_ed.cost)"}
    sec = float(value)
    if not math.isfinite(sec) or sec < 0:
        raise ValueError(f"budget_seconds must be finite and non-negative (got {value!r})")
    return sec, None


__all__ = ["E2_RECORDS", "COEFFICIENTS", "AUTO_FACTOR", "AUTO_FLOOR_SECONDS", "estimate_seconds", "auto_budget",
           "resolve_budget_seconds"]
