"""Producer record hook (roadmap A-07): capture a solver's output and write a producer bundle.

For an in-house code that already has the arrays in memory, the context manager writes the bundle on exit:

    import certify_ed

    with certify_ed.record(spec, "run42.bundle") as rec:
        w, V = scipy.sparse.linalg.eigsh(H, k=4, which="SA")
        rec.capture(w, V, H)                  # or: rec.eigenvalues = w; rec.eigenvectors = V; rec.operator = H

and afterwards `certify-ed certify --from run42.bundle` (or `certify_ed.claim_from_bundle`) certifies it. The
decorator form wraps a function that returns (w, V) or (w, V, H):

    @certify_ed.recorded(spec, "run42.bundle")
    def solve(H, k):
        return scipy.sparse.linalg.eigsh(H, k=k, which="SA")

What is written (certify_ed/io/bundle.py, schema certify-ed/producer-bundle/1.0): eigenvalues.<fmt>,
eigenvectors.<fmt>, operator.<fmt>, basis.<fmt> (a state list or an isometry), spec.json and meta.json with
`kind`, `window`, `producer`, `environment` (Python, numpy, scipy, certify-ed versions and the platform string)
and `timing` (UTC start and end, wall and process seconds). The arrays are written as given, in the
producer's own floating-point values (npz by default); nothing is rounded, sorted or filtered.

If the block raises, nothing is written and the exception propagates. If it exits without eigenvalues,
`ValueError` is raised: an empty bundle would certify nothing. This module computes no bound
(provenance.NON_COMPUTATIONAL).
"""
from __future__ import annotations

import datetime as _dt
import functools
import platform
import time
from typing import Any, Dict, Optional

import numpy as np

from .io.bundle import write_bundle
from .spec import ModelSpec

__all__ = ["Recorder", "record", "recorded", "environment_record"]


def environment_record() -> Dict[str, Any]:
    """Versions and platform of the producing process (no host name, no user name, no paths)."""
    import scipy
    from . import __version__
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "certify_ed": __version__, "platform": platform.platform(), "machine": platform.machine()}


def _utc(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc).isoformat(timespec="seconds")


class Recorder:
    """Collects a producer's arrays inside `with certify_ed.record(...) as rec:` and writes the bundle on exit.

    Attributes set by the producer: eigenvalues (required), eigenvectors, operator, basis (a 1-D state list in
    the spec's encoding, or an n_full x n isometry), kind, window, producer (dict), meta (dict, merged into
    meta.json). `capture(w, V, H, basis)` sets them in one call. After exit, `path` is the bundle directory and
    `meta` the meta.json content that was written."""

    def __init__(self, spec: Optional[ModelSpec], path: str, kind: str = "lowest", window=None, fmt: str = "npz",
                 producer: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None,
                 accuracy_target: Optional[float] = None, description: str = ""):
        if spec is not None and isinstance(spec, dict):
            spec = ModelSpec.from_dict(spec)
        if spec is not None and not isinstance(spec, ModelSpec):
            raise TypeError(f"spec must be a ModelSpec, a dict or None (got {type(spec).__name__})")
        if kind not in ("subset", "lowest", "window", "all"):
            raise ValueError(f"kind must be subset, lowest, window or all (got {kind!r})")
        if kind == "window" and window is None:
            raise ValueError("kind='window' needs window=(a, b)")
        self.spec = spec
        self.path = path
        self.kind = kind
        self.window = window
        self.fmt = fmt
        self.producer: Dict[str, Any] = {"name": "record hook"}
        self.producer.update(producer or {})
        self.meta: Dict[str, Any] = dict(meta or {})
        self.accuracy_target = accuracy_target
        self.description = description
        self.eigenvalues = None
        self.eigenvectors = None
        self.operator = None
        self.basis = None
        self.written = False
        self._t0 = self._p0 = None

    # ------------------------------------------------------------------ API
    def capture(self, eigenvalues, eigenvectors=None, operator=None, basis=None) -> "Recorder":
        """Set the arrays in one call; returns self."""
        self.eigenvalues = eigenvalues
        if eigenvectors is not None:
            self.eigenvectors = eigenvectors
        if operator is not None:
            self.operator = operator
        if basis is not None:
            self.basis = basis
        return self

    def __enter__(self) -> "Recorder":
        self._t0, self._p0 = time.time(), time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False                      # nothing written; the producer's exception propagates
        self.write()
        return False

    def write(self) -> str:
        """Write the bundle now (called by __exit__); returns the directory path."""
        if self.eigenvalues is None:
            raise ValueError(f"record({self.path!r}): no eigenvalues were captured (set rec.eigenvalues or call "
                             "rec.capture(w, ...)); nothing written")
        t1, p1 = time.time(), time.process_time()
        w = np.asarray(self.eigenvalues)
        if w.ndim != 1:
            w = w.reshape(-1)
        if np.iscomplexobj(w):
            if np.any(w.imag != 0):
                raise ValueError("eigenvalues must be real (a Hermitian claim)")
            w = w.real
        X = None if self.eigenvectors is None else np.asarray(self.eigenvectors)
        if X is not None:
            if X.ndim == 1:
                X = X[:, None]
            if X.shape[1] != w.size:
                if X.shape[0] == w.size and X.shape[0] != X.shape[1]:
                    raise ValueError(f"eigenvectors have shape {X.shape} for {w.size} eigenvalues: one column per "
                                     "eigenvalue is expected (transpose the array)")
                raise ValueError(f"eigenvectors have shape {X.shape} for {w.size} eigenvalues")
        meta = dict(self.meta)
        meta["producer"] = dict(self.producer)
        meta["environment"] = environment_record()
        meta["timing"] = {"started_utc": _utc(self._t0 if self._t0 is not None else t1), "finished_utc": _utc(t1),
                          "wall_seconds": round(t1 - (self._t0 if self._t0 is not None else t1), 6),
                          "process_seconds": round(p1 - (self._p0 if self._p0 is not None else p1), 6)}
        if self.accuracy_target is not None:
            meta["accuracy_target"] = float(self.accuracy_target)
        if self.description:
            meta["description"] = str(self.description)
        if self.spec is not None and self.basis is not None and np.ndim(self.basis) == 1:
            meta.setdefault("state_order", self.spec.state_order)
        self.path = write_bundle(self.path, w, X, operator=self.operator, spec=self.spec, basis=self.basis,
                                 fmt=self.fmt, meta=meta, kind=self.kind, window=self.window)
        self.meta = meta
        self.written = True
        return self.path


def record(spec: Optional[ModelSpec], path: str, kind: str = "lowest", window=None, fmt: str = "npz",
           producer: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None,
           accuracy_target: Optional[float] = None, description: str = "") -> Recorder:
    """Context manager: `with certify_ed.record(spec, path) as rec: ...; rec.capture(w, V, H)` writes a producer
    bundle at `path` on exit (see the module docstring). `spec` may be None when the model is declared later
    (`certify-ed certify --from path --spec model.json`)."""
    return Recorder(spec, path, kind=kind, window=window, fmt=fmt, producer=producer, meta=meta,
                    accuracy_target=accuracy_target, description=description)


def recorded(spec: Optional[ModelSpec], path: str, operator=None, basis=None, **record_kw):
    """Decorator form: the wrapped function returns (w,), (w, V) or (w, V, H) [or a bare array w]; the bundle is
    written after it returns and the function's result is returned unchanged. `operator=` / `basis=` supply
    what the function does not return."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with record(spec, path, **record_kw) as rec:
                out = fn(*args, **kwargs)
                parts = out if isinstance(out, tuple) else (out,)
                w = parts[0]
                V = parts[1] if len(parts) > 1 else None
                H = parts[2] if len(parts) > 2 else operator
                rec.capture(w, V, H, basis)
            wrapper.last_bundle = rec.path
            return out
        wrapper.last_bundle = None
        return wrapper
    return deco
