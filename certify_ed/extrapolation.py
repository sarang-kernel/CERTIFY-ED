"""Finite-size extrapolation report (roadmap C-10). Presentation only; nothing here is certified.

`extrapolation_report(certs, model)` lists, per certificate, the certified interval of the chosen eigenvalue
(by certificate hash; the interval is the one `crossings.level_interval` reads off S2/S3, divided by N when
`per_site`), fits the declared model to the interval midpoints by least squares (numpy, floating point) and
reports the fitted limit with the explicit statement that it is NOT certified and depends on the declared
model. The report never attaches a certificate status to the extrapolated value: the extrapolation is a
statement about a model of the sequence of sizes, not about any finite matrix (docs/SCOPE_MATRIX.md,
OUT_OF_SCOPE_EXTRAPOLATION). Listed in provenance.NON_COMPUTATIONAL: it cannot change a certified number.

Models: "1/N" fits y = c0 + c1/N; "1/N^2" fits y = c0 + c1/N^2; a callable f(N) returns the regressors
(without the constant) and the fitted limit is c0, which is the N -> inf value only if every regressor
vanishes there (the report says so and records the model's name).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .certificate_v2 import Certificate
from .crossings import level_interval
from .ivl import from_float, Interval

__all__ = ["extrapolation_report", "NOT_CERTIFIED_SENTENCE"]

NOT_CERTIFIED_SENTENCE = ("The extrapolated value is NOT certified: it depends on the declared model and on the set of "
                          "sizes, being the constant term of a least-squares fit of that model to the midpoints of the "
                          "per-size certified intervals. Only the per-size intervals are certificate values.")

_MODELS = {"1/N": (lambda N: [1.0 / N], "y = c0 + c1 / N"),
           "1/N^2": (lambda N: [1.0 / N ** 2], "y = c0 + c1 / N^2")}


def _size_of(cert: Certificate) -> Optional[int]:
    spec = (cert.body.get("subject") or {}).get("spec")
    if isinstance(spec, dict) and spec.get("n_sites"):
        return int(spec["n_sites"])
    return None


def extrapolation_report(certs: Sequence[Certificate], model: Union[str, Callable[[float], Sequence[float]]] = "1/N",
                         sizes: Optional[Sequence[int]] = None, per_site: bool = False, level: int = 0,
                         quantity_name: Optional[str] = None) -> Dict[str, Any]:
    """Markdown and JSON report of per-size certified intervals and a declared-model fit (not certified).

    certs: one certificate per size (S2 CERTIFIED; S3 CERTIFIED makes the rank certified, else the row carries the
    caveat of `crossings.level_interval`). sizes: N per certificate (default: the spec's n_sites). per_site: divide
    the intervals by N (interval arithmetic). level: eigenvalue rank (0 = lowest). Returns a dict with "rows",
    "fit", "extrapolated", "markdown"; the "extrapolated" record carries no status field."""
    if callable(model):
        regress, formula = model, f"custom callable {getattr(model, '__name__', 'f')}(N): y = c0 + sum c_k f_k(N)"
        model_name = f"custom:{getattr(model, '__name__', 'f')}"
    else:
        if model not in _MODELS:
            raise ValueError(f"unknown model {model!r}; known: {', '.join(_MODELS)} or a callable")
        regress, formula = _MODELS[model]
        model_name = model
    if sizes is not None and len(sizes) != len(certs):
        raise ValueError("sizes must have one entry per certificate")
    rows: List[Dict[str, Any]] = []
    for k, cert in enumerate(certs):
        N = int(sizes[k]) if sizes is not None else _size_of(cert)
        if N is None:
            raise ValueError(f"certificate {k}: no spec with n_sites; pass sizes=[...]")
        li = level_interval(cert, level)
        lo, hi = li["interval"]
        if per_site:
            iv = Interval(from_float(lo).lo, from_float(hi).hi) / from_float(float(N))
            lo, hi = iv.floats()
        rows.append({"N": N, "hash": li["hash"], "hash_source": li["hash_source"], "interval": [float(lo), float(hi)],
                     "midpoint": 0.5 * (float(lo) + float(hi)), "halfwidth": 0.5 * (float(hi) - float(lo)),
                     "rank_certified": li["rank_certified"], "caveat": li["caveat"], "S2": li["S2"], "S3": li["S3"],
                     "subject": li["subject"], "spec_sha256": li["spec_sha256"], "parameters": li["parameters"]})
    rows.sort(key=lambda r: r["N"])
    Ns = np.array([r["N"] for r in rows], dtype=float)
    y = np.array([r["midpoint"] for r in rows], dtype=float)
    A = np.array([[1.0] + list(regress(float(N))) for N in Ns], dtype=float)
    n_par = A.shape[1]
    fit: Dict[str, Any] = {"model": model_name, "formula": formula, "n_points": int(len(rows)), "n_parameters": int(n_par),
                           "method": "numpy.linalg.lstsq on the interval midpoints (floating point, unweighted)"}
    extrapolated: Dict[str, Any] = {"value": None, "model": model_name, "sentence": NOT_CERTIFIED_SENTENCE,
                                    "depends_on": {"model": model_name, "sizes": [int(N) for N in Ns],
                                                   "certificates": [r["hash"] for r in rows]}}
    if len(rows) >= n_par:
        coef, res, rank, _sv = np.linalg.lstsq(A, y, rcond=None)
        fit.update(coefficients=[float(c) for c in coef], rank=int(rank),
                   residual_norm=float(np.linalg.norm(A @ coef - y)),
                   fitted=[float(v) for v in (A @ coef)])
        extrapolated["value"] = float(coef[0])
        extrapolated["limit_is_c0"] = ("c0 is the N -> inf value of the model only if every regressor vanishes "
                                       "there (true for 1/N and 1/N^2)")
        # how far the midpoints could move inside their intervals: a sensitivity note, not a bound on the limit
        extrapolated["midpoint_halfwidths_max"] = float(max(r["halfwidth"] for r in rows))
    else:
        fit["note"] = f"{len(rows)} point(s) for {n_par} parameters: no fit"
    qname = quantity_name or (f"lambda_{level}" + (" / N" if per_site else ""))
    md = _markdown(rows, fit, extrapolated, qname, level)
    return {"schema": "certify-ed/extrapolation-report/1.0", "quantity": qname, "level": int(level), "per_site": bool(per_site),
            "rows": rows, "fit": fit, "extrapolated": extrapolated, "markdown": md}


def _markdown(rows, fit, extrapolated, qname, level) -> str:
    out = [f"# Finite-size report for {qname}", "",
           "Per-size values are certificate intervals (S2 enclosures; the rank is certified by S3 where the row says so). "
           "Cite them by the certificate hash. The last section is a fit of a declared model and is not a certificate "
           "statement.", "",
           "## Per-size certified intervals", "",
           "| N | lower | upper | half-width | rank | certificate (body SHA-256) |", "|---|---|---|---|---|---|"]
    for r in rows:
        rank = f"rank {level}, S3-certified" if r["rank_certified"] else f"sorted position {level}, rank not certified"
        out.append(f"| {r['N']} | {r['interval'][0]!r} | {r['interval'][1]!r} | {r['halfwidth']:.3e} | {rank} | `{r['hash']}` |")
    cav = [f"- N = {r['N']}: {r['caveat']}" for r in rows if r["caveat"]]
    if cav:
        out += ["", "Caveats:", *cav]
    out += ["", "## Declared model and fit", "",
            f"Model: `{fit['model']}` ({fit['formula']}); {fit['method']}; {fit['n_points']} points, {fit['n_parameters']} parameters."]
    if "coefficients" in fit:
        out.append("Coefficients: " + ", ".join(f"c{k} = {c!r}" for k, c in enumerate(fit["coefficients"]))
                   + f"; residual norm {fit['residual_norm']:.3e}.")
    else:
        out.append(fit.get("note", "no fit"))
    out += ["", "## Extrapolated value (not certified)", ""]
    if extrapolated["value"] is not None:
        out.append(f"Fitted limit of {qname} under the declared model `{extrapolated['model']}`: {extrapolated['value']!r}.")
        out.append(extrapolated["limit_is_c0"] + ".")
    else:
        out.append("No fitted limit (too few sizes for the model).")
    out += ["", extrapolated["sentence"], ""]
    return "\n".join(out)
