"""Certified crossings (roadmap C-11): is lambda_level of certificate A below or above that of certificate B?

Pure interval logic on certified data. Each certificate contributes one interval, the S2 enclosure of the
eigenvalue of the requested rank (radii include the model shift, so the interval refers to the operator the
certificate's statements refer to). Two intervals that do not overlap order the two eigenvalues strictly
(if a <= x <= b and c <= y <= d with b < c then x < y); overlapping intervals decide nothing ("not
separated"). No new mathematics: the intervals are the certificate's, the comparison is of their ends.

What "the eigenvalue of rank `level`" means depends on S3:

* S3 CERTIFIED for a 'lowest' or 'all' claim: the levels of `gaps.merge_enclosures` carry eigenvalue ranks
  (Corollary 5 and the pigeonhole step of docs/theory.md Section 6, as in `certify_ed.gaps`); the interval
  of rank `level` (0 = lowest) is that of the level containing it.
* otherwise: the interval is the S2 enclosure of the claimed value at sorted position `level`; it does
  contain an eigenvalue (Theorem 1-2), but its rank in the spectrum is not certified. The record says so
  (`rank_certified: false`, `caveat`), and the statement is about "the certified interval of the claimed
  value", not about lambda_level.

Typical use: the same model at two couplings (level crossing versus the coupling), or two symmetry sectors at
one coupling (which sector holds the ground state). The record carries both certificate hashes, both spec
hashes and whether they agree; it does not check that the two certificates describe the same model.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np

from .certificate_v2 import Certificate, canonical_json, _jsonable
from .fp import add_up, sub_down
from .gaps import merge_enclosures, rank_interval

__all__ = ["certificate_hash", "level_interval", "certified_crossing", "format_crossing"]

RELATIONS = ("<", ">", "not separated")


def certificate_hash(cert: Certificate) -> str:
    """body_sha256 of a saved certificate. For a certificate that was never saved, the digest of the body as it
    stands (arrays hashed, no sidecar name); it differs from the hash `save` writes, because the sidecar file
    name enters the saved body. `hash_source` in the level record says which one it is."""
    integ = cert.body.get("integrity") or {}
    if integ.get("body_sha256"):
        return str(integ["body_sha256"])
    body = cert._finalize_body((cert.body.get("arrays") or {}).get("sidecar"))
    return str(body["integrity"]["body_sha256"])


def _hash_source(cert: Certificate) -> str:
    return "saved certificate (integrity.body_sha256)" if (cert.body.get("integrity") or {}).get("body_sha256") else \
        "unsaved certificate: digest of the body in memory (the saved hash differs: the sidecar name enters the body)"


def _as_float(x) -> float:
    if isinstance(x, str):
        return float(x)                # 'inf', '-inf', 'nan' as written by the JSON writer
    return float(x)


def _radii(cert: Certificate):
    """(claimed values, radii including the model/lift shift) or None when the certificate has no enclosures."""
    mu = cert.arrays.get("eigenvalues")
    rad = cert.arrays.get("radius")
    if mu is not None and rad is not None and len(mu) == len(rad):
        return np.asarray(mu, dtype=float), np.asarray(rad, dtype=float)
    encs = cert.body.get("enclosures") or []
    if not encs:
        return None
    s2 = cert.statement("S2.eigenpairs") or {}
    ev = s2.get("evidence") or {}
    shift = _as_float(ev.get("sector_lift_shift") if ev.get("sector_lift_shift") is not None else ev.get("model_shift", 0.0))
    mu = np.array([_as_float(e["claimed"]) for e in encs], dtype=float)
    rad = np.array([_as_float(e["radius"]) for e in encs], dtype=float)
    if shift > 0:
        rad = np.asarray(add_up(rad, shift), dtype=float)
    return mu, rad


def level_interval(cert: Certificate, level: int = 0) -> Dict[str, Any]:
    """The certified interval for the eigenvalue of rank `level` (0 = lowest) of a certificate.

    Raises ValueError when the certificate has no CERTIFIED S2 (no certified intervals) or fewer claimed
    pairs than `level` + 1. See the module docstring for what the interval means with and without S3."""
    s2 = cert.status_of("S2.eigenpairs")
    if s2 != "CERTIFIED":
        raise ValueError(f"no certified intervals: S2.eigenpairs is {s2}")
    got = _radii(cert)
    if got is None:
        raise ValueError("certificate carries no enclosures")
    mu, rad = got
    if not (0 <= int(level) < mu.size):
        raise ValueError(f"level {level} out of range: the certificate has {mu.size} claimed pairs")
    if not np.all(np.isfinite(rad)):
        raise ValueError("some certified radius is not finite")
    kind = (cert.body.get("claim") or {}).get("kind")
    s3 = cert.status_of("S3.completeness")
    levels = merge_enclosures(mu, rad)
    rank_certified = s3 == "CERTIFIED" and kind in ("lowest", "all")
    if rank_certified:
        r = 1
        for lv in levels:
            lv["ranks"] = [r, r + lv["count"] - 1]
            r += lv["count"]
        lv = rank_interval(levels, int(level) + 1)
        caveat = None
        what = f"lambda_{level} (rank {level}, 0 = lowest, certified by S3)"
    else:
        pos = int(np.argsort(mu, kind="stable")[int(level)])
        lv = next(x for x in levels if pos in x["members"])
        caveat = (f"S3.completeness is {s3}" + (f" (claim kind {kind!r})" if kind not in ("lowest", "all") else "")
                  + f": the interval is the S2 enclosure of the claimed value at sorted position {level}; it contains "
                    "an eigenvalue, but that eigenvalue's rank in the spectrum is not certified")
        what = f"the certified interval of the claimed value at sorted position {level}"
    s2_ev = (cert.statement("S2.eigenpairs") or {}).get("evidence") or {}
    subj = cert.body.get("subject") or {}
    return {"interval": [float(lv["lower"]), float(lv["upper"])], "level": int(level), "what": what,
            "rank_certified": bool(rank_certified), "caveat": caveat, "members": list(lv["members"]),
            "level_count": int(lv["count"]), "claim_kind": kind, "S2": s2, "S3": s3,
            "refers_to": s2_ev.get("radius_refers_to"), "hash": certificate_hash(cert), "hash_source": _hash_source(cert),
            "subject": subj.get("description"), "spec_sha256": subj.get("spec_sha256"),
            "parameters": ((subj.get("spec") or {}).get("parameters") if isinstance(subj.get("spec"), dict) else None),
            "sector": subj.get("sector")}


def certified_crossing(cert_a: Certificate, cert_b: Certificate, level: int = 0) -> Dict[str, Any]:
    """Compare the rank-`level` eigenvalue intervals of two certificates.

    Returns a record with `relation` '<' (lambda_level(a) < lambda_level(b)), '>' or 'not separated',
    both intervals with their certificate hashes, the margin (distance between the intervals, rounded down;
    for overlapping intervals the negative of the overlap, informational) and the caveats when a rank is not
    certified. Raises ValueError when a certificate has no certified interval for the level."""
    A = level_interval(cert_a, level)
    B = level_interval(cert_b, level)
    a_lo, a_hi = A["interval"]
    b_lo, b_hi = B["interval"]
    caveats = [f"{tag}: {c}" for tag, c in (("A", A["caveat"]), ("B", B["caveat"])) if c]
    if a_hi < b_lo:
        relation, margin = "<", float(sub_down(b_lo, a_hi))
    elif b_hi < a_lo:
        relation, margin = ">", float(sub_down(a_lo, b_hi))
    else:
        relation, margin = "not separated", float(-(min(a_hi, b_hi) - max(a_lo, b_lo)))
    both_ranked = A["rank_certified"] and B["rank_certified"]
    name = f"lambda_{level}" if both_ranked else f"the certified interval of the claimed value at sorted position {level}"
    if relation == "not separated":
        stmt = (f"{name} of A and of B are not separated: the intervals [{a_lo!r}, {a_hi!r}] and "
                f"[{b_lo!r}, {b_hi!r}] overlap by {-margin:.3e}")
    else:
        stmt = (f"{name}(A) {relation} {name}(B)" if both_ranked else
                f"{name} of A lies {'below' if relation == '<' else 'above'} that of B") + \
               f": [{a_lo!r}, {a_hi!r}] {relation} [{b_lo!r}, {b_hi!r}], margin >= {margin:.6e}"
    if caveats:
        stmt += "; " + "; ".join(caveats)
    same_spec = (A["spec_sha256"] is not None and A["spec_sha256"] == B["spec_sha256"])
    rec = {"schema": "certify-ed/crossing/1.0", "level": int(level), "relation": relation, "margin": margin,
           "statement": stmt, "rank_certified": bool(both_ranked), "caveats": caveats,
           "a": A, "b": B, "same_spec_sha256": bool(same_spec),
           "basis": ("interval comparison of the S2 enclosures (Theorems 1-2; radii include the model shift); "
                     "ranks from S3 (Corollary 5) where certified; the record does not check that the two "
                     "certificates describe the same model: compare spec_sha256 and parameters")}
    rec["record_sha256"] = hashlib.sha256(canonical_json(_jsonable({k: v for k, v in rec.items()}))
                                          .encode()).hexdigest()
    return rec


def format_crossing(rec: Dict[str, Any]) -> str:
    """Plain-text rendering of a crossing record."""
    lines = [rec["statement"], f"  A: {rec['a']['hash']}  {rec['a']['subject'] or ''}  params {rec['a']['parameters']}",
             f"  B: {rec['b']['hash']}  {rec['b']['subject'] or ''}  params {rec['b']['parameters']}",
             f"  same spec hash: {rec['same_spec_sha256']}; relation: {rec['relation']}; margin: {rec['margin']!r}"]
    return "\n".join(lines)


def _default(o):   # pragma: no cover - json fallback
    return str(o)


def crossing_main(a: Certificate, b: Certificate, level: int = 0, json_path: Optional[str] = None) -> Dict[str, Any]:
    rec = certified_crossing(a, b, level)
    if json_path:
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(rec), f, indent=1, default=_default)
    return rec
