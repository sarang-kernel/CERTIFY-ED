"""Certified parameter intervals (roadmap E-04): eigenvalue tubes over a coupling interval from certificates at
grid couplings and a bound on the derivative of the Hamiltonian.

Setting (docs/proofs/E-04_parameter_intervals.md). The declared model is affine in one coupling c on each grid
segment: F_k(c) = H(c_k) + (c - c_k) H1, where H(c_k) is the spec at the grid coupling c_k (certified by S1) and
H1 is the derivative spec (declared, or inferred from the coefficients at two grid couplings and checked at all
of them). With B >= ||H1||_2 (a row-sum bound of the built H1, or the triangle-inequality bound over its terms,
both rounded up), Weyl's inequality gives, for every sorted index j,

    |lambda_j(F_k(c)) - lambda_j(H(c_k))| <= B |c - c_k|,

so an S2 enclosure [mu - r, mu + r] of lambda_j(H(c_k)) widens to the tube [mu - r - B|c - c_k|, mu + r + B|c - c_k|]
over the whole segment; when S3 at c_k certifies the rank j (Corollary 5), the tube is about lambda_j(F_k(c)) with
its rank preserved (Weyl is an inequality per sorted index; equivalently the count below a shifting threshold is
preserved); otherwise it contains an eigenvalue of F_k(c) of a fixed but unidentified index. The neighbouring
certificate at c_{k+1} concerns H(c_{k+1}) = F_k(c_{k+1}) + E_k with ||E_k||_2 <= d_k (the affinity defect, zero
when the coefficients are exactly affine in binary64), so its tube, shifted by d_k, is also a tube for F_k(c) on
the segment, and the two are intersected. S3's lower bound on lambda_{k+1} widens the same way.

`certified_crossing_interval` locates where two tubes (from one scan or two scans, e.g. two symmetry sectors)
are certified apart ('<' or '>'), and brackets any crossing: an interval of c on whose left the relation is
certified one way and on whose right the other way contains a crossing of the two eigenvalue branches (sorted
eigenvalues are continuous in c). Certification between sample points uses the 2B-Lipschitz bound on the
margin, so no point of the interval is missed.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import Certificate, _jsonable, canonical_json
from .construct import build_bitwise
from .fp import U, abs_up, add_up, inf_norm_upper, modulus_up, sub_down, up
from .precision import binary64, check_values, summed
from .spec import ModelSpec, Term

__all__ = ["derivative_spec", "affinity_defect", "matrix_affinity_defect", "term_norm_bound", "h1_norm_bound", "certified_sweep",
           "tube_at", "certified_crossing_interval", "format_sweep", "format_crossing_interval"]

SWEEP_SCHEMA = "certify-ed/sweep/1.0"


# ------------------------------------------------------------------------------------------------ the family

def _check_same_structure(a: ModelSpec, b: ModelSpec) -> None:
    if a.n_sites != b.n_sites or a.local.to_dict() != b.local.to_dict() or a.state_order != b.state_order:
        raise ValueError("the spec's sites, local space or encoding change with the coupling: not one family")
    if len(a.terms) != len(b.terms) or any(ta.op != tb.op or [s for _, s in ta.couplings] != [s for _, s in tb.couplings]
                                           for ta, tb in zip(a.terms, b.terms)):
        raise ValueError("the spec's operator strings or bonds change with the coupling: the family is not affine in "
                         "one parameter with a fixed term structure")


def derivative_spec(spec_fn: Callable[[float], ModelSpec], c0: float, c1: float) -> ModelSpec:
    """H1 inferred from the coefficients at two couplings: b_t = (coef_t(c1) - coef_t(c0)) / (c1 - c0) (binary64).
    The result is a declaration; `affinity_defect` measures how far the family deviates from H(c0) + (c - c0) H1
    at every other grid coupling."""
    if not c1 != c0:
        raise ValueError("two distinct couplings are needed to infer the derivative spec")
    a, b = spec_fn(c0), spec_fn(c1)
    _check_same_structure(a, b)
    terms = []
    for ta, tb in zip(a.terms, b.terms):
        cps = [((complex(cb) - complex(ca)) / (c1 - c0), s) for (ca, s), (cb, _) in zip(ta.couplings, tb.couplings)]
        terms.append(Term(ta.op, cps))
    h1 = ModelSpec(a.n_sites, a.local, terms, site_order=a.site_order, state_order=a.state_order,
                   name=(a.name + "_derivative") if a.name else "derivative",
                   parameters={"derived_from": [float(c0), float(c1)]})
    h1.validate()
    return h1


def term_norm_bound(spec: ModelSpec, op: str) -> float:
    """Upper bound on ||o_1 ... o_m||_2 for an operator string on distinct sites: the product of the local letters'
    infinity norms (tensor product; Jordan-Wigner signs are unitary), rounded up. The float local matrices differ
    from the exact letters by a relative u per entry, covered by the factor (1 + 4u)."""
    bnd = 1.0
    for letter in op:
        M = spec.local.matrix(letter)
        bnd *= inf_norm_upper(modulus_up(M))
    return float(up(bnd * (1 + 4 * U)))


def h1_norm_bound(h1: ModelSpec, H1_matrix=None, builder_shift: float = 0.0) -> Tuple[float, str]:
    """B >= ||H1||_2 for the exact spec operator H1: min of the row-sum bound of the built matrix plus the builder's
    rounding shift (Hermitian: ||.||_2 <= ||.||_inf; `certify._builder_shift` bounds ||H1_built - H1||_2) and the
    triangle bound sum_t |b_t| ||O_t||_2 over the terms; both are proven upper bounds. Returns (B, which)."""
    tri = 0.0
    for t in h1.terms:
        nb = term_norm_bound(h1, t.op)
        for c, _ in t.couplings:
            tri = float(add_up(tri, up(modulus_up(complex(c)) * nb)))
    if H1_matrix is None:
        return tri, "triangle inequality over the terms"
    row = float(add_up(inf_norm_upper(H1_matrix), builder_shift))
    return (row, "row-sum bound of the built H1 (+ builder shift)") if row <= tri else (tri, "triangle inequality over the terms")


def matrix_affinity_defect(Ha, Hb, dc: float, H1) -> float:
    """Upper bound on ||H_b - H_a - dc H1||_2 for stored Hermitian matrices (a sector block family): the row-sum
    bound of the computed difference plus the rounding of its three operations per entry, gamma_3 (|H_b| + |H_a| +
    |dc||H1|) entrywise. Data in a narrower format than binary64 are converted exactly first, before the conversion to
    CSR (precision.py)."""
    Ha, Hb, H1 = (sp.csr_matrix(summed(binary64(M if sp.issparse(M) else np.asarray(M), name), name))
                  for M, name in ((Ha, "Ha"), (Hb, "Hb"), (H1, "H1")))
    g3 = 3 * U / (1 - 3 * U)
    diff = (Hb - Ha - dc * H1).tocsr()
    rnd = (abs_up(Hb) + abs_up(Ha) + abs(dc) * abs_up(H1)).tocsr() * (g3 * (1 + 4 * U))   # upper moduli (fp.abs_up)
    return float(up(inf_norm_upper((abs_up(diff) + rnd).tocsr())))


def affinity_defect(spec_a: ModelSpec, spec_b: ModelSpec, dc: float, h1: ModelSpec) -> float:
    """Upper bound on ||H(c_b) - H(c_a) - dc H1||_2 from the coefficients: sum_t |e_t| ||O_t||_2 with
    e_t = coef_t(c_b) - coef_t(c_a) - dc b_t computed exactly in rationals (binary64 coefficients and dc are dyadic),
    |e_t| rounded up. Exactly zero when the coefficients are affine in binary64 (the usual case, e.g. a field or an
    anisotropy that enters one coefficient linearly)."""
    from fractions import Fraction
    _check_same_structure(spec_a, spec_b)
    _check_same_structure(spec_a, h1)
    tot = 0.0
    dcf = Fraction(float(dc))
    for ta, tb, th in zip(spec_a.terms, spec_b.terms, h1.terms):
        nb = term_norm_bound(spec_a, ta.op)
        for (ca, _), (cb, _), (bt, _) in zip(ta.couplings, tb.couplings, th.couplings):
            ca, cb, bt = complex(ca), complex(cb), complex(bt)
            # exact: binary64 coefficients and dc are dyadic rationals
            er = Fraction(cb.real) - Fraction(ca.real) - dcf * Fraction(bt.real)
            ei = Fraction(cb.imag) - Fraction(ca.imag) - dcf * Fraction(bt.imag)
            if er == 0 and ei == 0:
                continue
            e_up = up(math.sqrt(float(er * er + ei * ei)) * (1 + 4 * U))     # |e| rounded up (two roundings)
            tot = float(add_up(tot, up(e_up * nb)))
    return float(tot)


# ------------------------------------------------------------------------------------------------ the scan

def _default_solver(H, k: int):
    n = H.shape[0]
    if n <= 3000:
        import scipy.linalg as la
        w, Y = la.eigh(H.toarray() if sp.issparse(H) else np.asarray(H))
        return w[:k], Y[:, :k]
    import scipy.sparse.linalg as spla
    w, Y = spla.eigsh(H, k=k, which="SA", tol=1e-13)
    o = np.argsort(w)
    return w[o], Y[:, o]


def _intervals(cert: Certificate, block: bool = False):
    """Sorted (mu, radius incl. the model shift) of a certificate with S2 CERTIFIED, or None. For a sector claim
    (block=True) the block radii of body["enclosures"] are used (the tube concerns the stored block family), not
    the lifted radii of the arrays."""
    from .crossings import _radii
    if cert.status_of("S2.eigenpairs") != "CERTIFIED":
        return None
    if block:
        encs = cert.body.get("enclosures") or []
        if not encs:
            return None
        mu = np.array([float(e["claimed"]) for e in encs], dtype=float)
        rad = np.array([float(e["radius"]) for e in encs], dtype=float)
    else:
        got = _radii(cert)
        if got is None:
            return None
        mu, rad = got
    o = np.argsort(mu, kind="stable")
    return mu[o], rad[o]


def certified_sweep(spec_fn: Callable[[float], ModelSpec], couplings: Sequence[float], k: int = 1,
                    solver: Optional[Callable] = None, options=None, h1_spec: Optional[ModelSpec] = None,
                    kind: str = "lowest", states_fn: Optional[Callable[[ModelSpec], Any]] = None,
                    sector_fn: Optional[Callable] = None, accuracy_target: Optional[float] = None,
                    save_dir: Optional[str] = None, parameter_name: str = "c") -> Dict[str, Any]:
    """Certify the k lowest eigenpairs at every coupling and assemble the certified tubes between them.

    spec_fn(c) -> ModelSpec (the declared family), couplings an increasing sequence, solver(H, k) -> (w, Y) the
    producer (default: dense eigh below n = 3000, else ARPACK), options the certify.Options for every point,
    h1_spec the derivative spec (inferred from the first two couplings when None), states_fn(spec) -> StateSet to
    work in a U(1) block (H1 must conserve it: zero leak), sector_fn(spec, states) -> V0 a symmetry-sector isometry
    within the block (e.g. `cyclic_isometry(states, translation(spec), j)`): the scan then concerns the stored
    block family H_s(c_k) + (c - c_k) H1_s (H_s = V0^H H V0, H1_s = V0^H H1 V0, floating point), with ranks in the
    block, S1 auditing V0 at every grid point and the lift to the full operator certified at the grid points only
    (recorded per point). Returns the scan record (JSON-able; certificates in `record["_certificates"]`, not
    serialised)."""
    from .certify import Claim, Options, certify, _builder_shift
    from .basis import sector_matrix
    from .construct import full_state_set
    cs = [float(c) for c in couplings]
    if len(cs) < 2 or any(b <= a for a, b in zip(cs, cs[1:])):
        raise ValueError("couplings must be an increasing sequence of at least two values")
    opts = options or Options()
    solver = solver or _default_solver
    specs = [spec_fn(c) for c in cs]
    for s_ in specs[1:]:
        _check_same_structure(specs[0], s_)
    h1 = h1_spec if h1_spec is not None else derivative_spec(spec_fn, cs[0], cs[1])
    _check_same_structure(specs[0], h1)
    states = states_fn(specs[0]) if states_fn is not None else None
    H1, leak1 = build_bitwise(h1, states, return_leak=True)
    if leak1 > 0:
        raise ValueError("the derivative operator H1 does not conserve the block defined by states_fn (leak "
                         f"{leak1:.2e}); the tube would refer to a truncated operator")
    V0 = sector_fn(specs[0], states) if sector_fn is not None else None
    V_full = None
    if V0 is not None:
        V0 = sp.csr_matrix(summed(binary64(V0, "sector isometry"), "sector isometry"))   # exact; summed
        if states is not None:
            full_idx, _ = full_state_set(specs[0]).index_of(states.states_in_order())
            P = sp.csr_matrix((np.ones(states.size), (full_idx, np.arange(states.size))), shape=(specs[0].dim, states.size))
            V_full = (P @ V0).tocsr()
        else:
            V_full = V0
        H1_s = sector_matrix(H1, V0)
        B, B_source = float(inf_norm_upper(H1_s)), "row-sum bound of the stored block H1_s = V0^H H1 V0"
    else:
        B, B_source = h1_norm_bound(h1, H1, _builder_shift(h1, states))
    if V0 is None:
        defects = [affinity_defect(specs[i], specs[i + 1], cs[i + 1] - cs[i], h1) for i in range(len(cs) - 1)]
    points: List[Dict[str, Any]] = []
    certs: List[Certificate] = []
    blocks: List[Any] = []
    for i, (c, spec) in enumerate(zip(cs, specs)):
        H, leak = build_bitwise(spec, states, return_leak=True)
        if leak > 0:
            raise ValueError(f"H({c}) does not conserve the block defined by states_fn (leak {leak:.2e})")
        if V0 is not None:
            H = sector_matrix(H, V0)
            blocks.append(H)
        w, Y = solver(H, k)
        check_values(w, "solver eigenvalues")          # before the conversion (precision.py)
        w = np.asarray(w, dtype=float)
        Y = np.asarray(Y)
        claim = Claim(w[:k], Y[:, :k], operator=H, kind=kind, states=states if V0 is None else None,
                      sector_isometry=V_full, accuracy_target=accuracy_target,
                      description=f"{spec.name or 'model'} at {parameter_name} = {c!r}")
        cert = certify(claim, spec, opts)
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            cert.save(os.path.join(save_dir, f"point_{i:03d}"))
        certs.append(cert)
        from .crossings import certificate_hash, _hash_source
        iv = _intervals(cert, block=V0 is not None)
        s3 = cert.status_of("S3.completeness")
        s3_ev = (cert.statement("S3.completeness") or {}).get("evidence") or {}
        nlb = s3_ev.get("next_eigenvalue_lower_bound")
        rank_certified = bool(iv is not None and s3 == "CERTIFIED" and kind in ("lowest", "all"))
        pt = {"index": i, parameter_name: c, "hash": certificate_hash(cert), "hash_source": _hash_source(cert),
              "spec_sha256": spec.sha256(), "S1": cert.status_of("S1.model_fidelity"),
              "S2": cert.status_of("S2.eigenpairs"), "S3": s3, "rank_certified": rank_certified,
              "claimed": None if iv is None else [float(x) for x in iv[0]],
              "radius": None if iv is None else [float(x) for x in iv[1]],
              "lower": None if iv is None else [float(x) for x in sub_down(iv[0], iv[1])],
              "upper": None if iv is None else [float(x) for x in add_up(iv[0], iv[1])],
              "next_eigenvalue_lower_bound": None if (nlb is None or not rank_certified) else float(nlb)}
        if V0 is not None:
            s2_ev = (cert.statement("S2.eigenpairs") or {}).get("evidence") or {}
            pt["sector_lift_shift"] = s2_ev.get("sector_lift_shift")
            pt["rank_scope"] = "sector block"
        else:
            pt["rank_scope"] = "full operator" if states is None else f"U(1) block {getattr(states, 'label', '')}"
        points.append(pt)
    if V0 is not None:
        defects = [matrix_affinity_defect(blocks[i], blocks[i + 1], cs[i + 1] - cs[i], H1_s) for i in range(len(cs) - 1)]
    rec = {"schema": SWEEP_SCHEMA, "parameter": parameter_name, "couplings": cs, "k": int(k), "kind": kind,
           "family": (("F_k(c) = H(c_k) + (c - c_k) H1 on each segment [c_k, c_{k+1}]; H1 "
                       + ("declared by the user" if h1_spec is not None else "inferred from the coefficients at the first two couplings"))
                      if V0 is None else
                      ("stored block family F_k(c) = H_s(c_k) + (c - c_k) H1_s on each segment, H_s = V0^H H V0 and "
                       "H1_s = V0^H H1 V0 in floating point; ranks and tubes refer to the block; the lift to the full "
                       "operator is certified at the grid points only (sector_lift_shift per point)")),
           "h1_spec": h1.to_dict(), "h1_spec_sha256": h1.sha256(), "B": float(B), "B_source": B_source,
           "affinity_defects": [float(d) for d in defects], "exactly_affine": bool(all(d == 0.0 for d in defects)),
           "block": None if states is None else {"label": getattr(states, "label", ""), "dimension": int(states.size)},
           "sector": None if V0 is None else {"dimension": int(V0.shape[1])},
           "points": points,
           "basis": ("Weyl's inequality |lambda_j(A + E) - lambda_j(A)| <= ||E||_2 per sorted index with E = (c - c_k) H1 "
                     "(+ the affinity defect for the neighbouring certificate); ranks from S3 (Corollary 5) where certified; "
                     "docs/proofs/E-04_parameter_intervals.md"),
           "caveats": ["the tubes are about the affine interpolant F_k(c) of the certified grid Hamiltonians; a spec "
                       "built at an intermediate coupling may round its coefficients and then differs from F_k(c) by at "
                       "most the rounding of each coefficient times the term norm (not computed here)",
                       "a point whose S2 is not CERTIFIED contributes no tube; a point whose S3 is not CERTIFIED "
                       "contributes tubes that contain an eigenvalue of unidentified rank"]}
    rec["record_sha256"] = _record_hash(rec)
    rec["_certificates"] = certs
    return rec


def _record_hash(rec: dict) -> str:
    import hashlib
    body = {k: v for k, v in rec.items() if not k.startswith("_") and k != "record_sha256"}
    return hashlib.sha256(canonical_json(_jsonable(body)).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ tubes

def _segment(rec: dict, c: float) -> int:
    cs = rec["couplings"]
    if c < cs[0] or c > cs[-1]:
        raise ValueError(f"coupling {c!r} outside the scanned range [{cs[0]!r}, {cs[-1]!r}]")
    k = int(np.searchsorted(cs, c, side="right")) - 1
    return min(max(k, 0), len(cs) - 2)


def tube_at(rec: dict, level: int, c: float) -> Dict[str, Any]:
    """Certified interval for the eigenvalue of sorted index `level` (0 = lowest) of F_k(c) at coupling c
    (intersection of the tubes of the two grid points of the segment; the right point's tube includes the
    segment's affinity defect). Raises ValueError when neither grid point has a certified interval."""
    k = _segment(rec, c)
    B = rec["B"]
    d = rec["affinity_defects"][k]
    lo, hi = -math.inf, math.inf
    ranked = True
    used = []
    for side, pt in (("left", rec["points"][k]), ("right", rec["points"][k + 1])):
        if pt["lower"] is None or level >= len(pt["lower"]):
            continue
        dist = abs(c - pt[rec["parameter"]])
        widen = up(B * dist)
        if side == "right":
            widen = float(add_up(widen, d))
        lo_s = float(sub_down(pt["lower"][level], widen))
        hi_s = float(add_up(pt["upper"][level], widen))
        lo, hi = max(lo, lo_s), min(hi, hi_s)
        ranked = ranked and pt["rank_certified"]
        used.append(side)
    if not used:
        raise ValueError(f"no certified interval for level {level} on segment {k}")
    if lo > hi:
        raise ValueError(f"empty intersection of tubes at c = {c!r} for level {level}: the inputs are inconsistent "
                         "(an affinity defect or a certificate is wrong)")
    return {"interval": [lo, hi], "level": int(level), rec["parameter"]: float(c), "segment": k, "from": used,
            "rank_certified": bool(ranked)}


def next_level_lower_bound_at(rec: dict, c: float) -> Optional[float]:
    """Lower bound on lambda_{k+1}(F_k(c)) from S3's bounds at the segment's grid points (None if neither has one)."""
    k = _segment(rec, c)
    best = None
    for side, pt in (("left", rec["points"][k]), ("right", rec["points"][k + 1])):
        L = pt.get("next_eigenvalue_lower_bound")
        if L is None:
            continue
        widen = up(rec["B"] * abs(c - pt[rec["parameter"]]))
        if side == "right":
            widen = float(add_up(widen, rec["affinity_defects"][k]))
        v = float(sub_down(L, widen))
        best = v if best is None else max(best, v)
    return best


# ------------------------------------------------------------------------------------------------ crossings

def certified_crossing_interval(rec_a: dict, level_a: int, rec_b: Optional[dict] = None, level_b: Optional[int] = None,
                                samples_per_segment: int = 200) -> Dict[str, Any]:
    """Where the tube of level_a (scan rec_a) and the tube of level_b (scan rec_b, default rec_a) are certified apart.

    The margin m_<(c) = upper_a(c) - lower_b(c) is 2B-Lipschitz (B = max of the two scans' B); on a sample interval
    [c_i, c_{i+1}] of width D the relation a < b holds throughout when max(m_<(c_i), m_<(c_{i+1})) + B D + d_a + d_b < 0
    (docs/proofs/E-04_parameter_intervals.md, Lemma 3; d_a, d_b the affinity defects of the segments of the two
    scans that end at c_{i+1}, zero unless c_{i+1} is an interior grid point of that scan), and likewise for '>'. Returns the maximal runs where '<'
    or '>' is certified, the remaining 'not separated' runs, and `crossing_intervals`: not-separated runs with '<'
    certified on one side and '>' on the other (each contains a crossing of the two branches)."""
    rec_b = rec_a if rec_b is None else rec_b
    level_b = level_a if level_b is None else level_b
    pa, pb = rec_a["parameter"], rec_b["parameter"]
    lo_c = max(rec_a["couplings"][0], rec_b["couplings"][0])
    hi_c = min(rec_a["couplings"][-1], rec_b["couplings"][-1])
    if not lo_c < hi_c:
        raise ValueError("the two scans do not overlap in coupling")
    grid = sorted({float(x) for x in rec_a["couplings"] + rec_b["couplings"] if lo_c <= x <= hi_c} | {lo_c, hi_c})
    cs: List[float] = []
    for a, b in zip(grid, grid[1:]):
        cs += [float(x) for x in np.linspace(a, b, samples_per_segment + 1)[:-1]]
    cs.append(float(grid[-1]))
    B = max(float(rec_a["B"]), float(rec_b["B"]))
    m_lt, m_gt = [], []
    for c in cs:
        ta = tube_at(rec_a, level_a, c)["interval"]
        tb = tube_at(rec_b, level_b, c)["interval"]
        m_lt.append(float(np.nextafter(ta[1] - tb[0], np.inf)))      # upper_a - lower_b, rounded up
        m_gt.append(float(np.nextafter(tb[1] - ta[0], np.inf)))      # upper_b - lower_a, rounded up
    # At an interior grid point c_{k+1} of a scan, tube_at switches from the segment-k family F_k to F_{k+1}, whose
    # value at c_{k+1} is H(c_{k+1}); the margin of the segment-k branches at that end is therefore only known up to
    # the segment's affinity defect d_k (|lambda_j(F_k(c_{k+1})) - lambda_j(H(c_{k+1}))| <= d_k, Weyl). The sample
    # interval ending there gets d_k of every scan with a grid point there added to its slack (independent
    # re-derivation 2026-10-30, docs/proofs/VERDICTS.md, E-04: without it a crossing of the segment-k branches
    # inside the last sample interval before a grid point with d_k > 0 was certified away).
    def _end_defect(rec: dict, c_end: float) -> float:
        cps = rec["couplings"]
        for k in range(1, len(cps) - 1):
            if cps[k] == c_end:
                return float(rec["affinity_defects"][k - 1])
        return 0.0

    labels = []
    for i in range(len(cs) - 1):
        D = float(np.nextafter(cs[i + 1] - cs[i], np.inf))
        slack = float(up(B * D))
        slack = float(add_up(slack, add_up(_end_defect(rec_a, cs[i + 1]), _end_defect(rec_b, cs[i + 1]))))
        if max(m_lt[i], m_lt[i + 1]) + slack < 0:
            labels.append("<")
        elif max(m_gt[i], m_gt[i + 1]) + slack < 0:
            labels.append(">")
        else:
            labels.append("not separated")
    runs = []
    for i, lab in enumerate(labels):
        if runs and runs[-1]["relation"] == lab:
            runs[-1]["interval"][1] = cs[i + 1]
        else:
            runs.append({"relation": lab, "interval": [cs[i], cs[i + 1]]})
    crossings = []
    for j, r in enumerate(runs):
        if r["relation"] == "not separated" and 0 < j < len(runs) - 1:
            left, right = runs[j - 1]["relation"], runs[j + 1]["relation"]
            if {left, right} == {"<", ">"}:
                crossings.append({"interval": list(r["interval"]), "from": left, "to": right})
    ranked = all(p["rank_certified"] for p in rec_a["points"] if p["lower"] is not None) and \
        all(p["rank_certified"] for p in rec_b["points"] if p["lower"] is not None)
    out = {"schema": "certify-ed/sweep-crossing/1.0", "level_a": int(level_a), "level_b": int(level_b),
           "scan_a": rec_a.get("record_sha256"), "scan_b": rec_b.get("record_sha256"), "same_scan": rec_b is rec_a,
           "range": [lo_c, hi_c], "samples_per_segment": int(samples_per_segment), "B": B,
           "runs": runs, "crossing_intervals": crossings, "ranks_certified": bool(ranked),
           "statement": _crossing_statement(runs, crossings, level_a, level_b, pa if pa == pb else f"{pa}/{pb}"),
           "basis": ("interval comparison of the certified tubes; between sample points the 2B-Lipschitz bound on the "
                     "margin certifies the relation on the whole sample interval (Lemma 3 of "
                     "docs/proofs/E-04_parameter_intervals.md); a crossing lies in every not-separated run flanked by "
                     "opposite certified relations (continuity of the sorted eigenvalues in c)")}
    return out


def _crossing_statement(runs, crossings, la, lb, pname) -> str:
    parts = []
    for r in runs:
        a, b = r["interval"]
        if r["relation"] == "<":
            parts.append(f"lambda_{la}(A) < lambda_{lb}(B) for {pname} in [{a!r}, {b!r}]")
        elif r["relation"] == ">":
            parts.append(f"lambda_{la}(A) > lambda_{lb}(B) for {pname} in [{a!r}, {b!r}]")
        else:
            parts.append(f"not separated for {pname} in [{a!r}, {b!r}]")
    if crossings:
        parts.append("crossing interval(s): " + ", ".join(f"[{c['interval'][0]!r}, {c['interval'][1]!r}]" for c in crossings))
    else:
        parts.append("no certified crossing")
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------ text

def format_sweep(rec: dict, n_eval: int = 5) -> str:
    p = rec["parameter"]
    lines = [f"certified sweep over {p} in [{rec['couplings'][0]!r}, {rec['couplings'][-1]!r}] ({len(rec['couplings'])} points, "
             f"k = {rec['k']}, kind {rec['kind']}); B = {rec['B']:.6g} ({rec['B_source']}); "
             f"affinity defects max {max(rec['affinity_defects']):.3g}" + (" (exactly affine)" if rec["exactly_affine"] else "")]
    for pt in rec["points"]:
        if pt["lower"] is None:
            lines.append(f"  {p} = {pt[p]!r}: S2 {pt['S2']} (no certified intervals); hash {pt['hash'][:16]}")
            continue
        ivs = ", ".join(f"[{lo:.12g}, {hi:.12g}]" for lo, hi in zip(pt["lower"], pt["upper"]))
        lines.append(f"  {p} = {pt[p]!r}: {ivs}; S3 {pt['S3']}" + (" (ranks certified)" if pt["rank_certified"] else "")
                     + f"; hash {pt['hash'][:16]}")
    cs = rec["couplings"]
    for c in np.linspace(cs[0], cs[-1], n_eval):
        try:
            tubes = [tube_at(rec, j, float(c))["interval"] for j in range(rec["k"])]
            lines.append(f"  tube at {p} = {float(c):.6g}: " + ", ".join(f"[{a:.10g}, {b:.10g}]" for a, b in tubes))
        except ValueError as e:
            lines.append(f"  tube at {p} = {float(c):.6g}: {e}")
    return "\n".join(lines)


def format_crossing_interval(rec: dict) -> str:
    return rec["statement"]
