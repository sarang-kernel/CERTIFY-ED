"""Eigenvalue enclosures for claimed eigenpairs of a Hermitian operator.

Statements produced here concern the operator AS STORED (the floating-point
matrix or the exact operator defined by a spec, see `operator_kind`).

Theorem 1 (single pair; e.g. Parlett 1998, Thm 4.5.1). For Hermitian A, any
nonzero x and any real mu, some eigenvalue lambda_j satisfies
    |lambda_j - mu| <= ||A x - mu x|| / ||x||.

Theorem 2 (cluster; proved in the paper's appendix). Let X (n x g) satisfy
||X^* X - I||_2 <= eta < 1, let Lambda = diag(mu_1..mu_g) be real, and let
R = A X - X Lambda. Put delta = 1 - sqrt(1 - eta),
    rho = ||R||_2 + (||A||_2 + ||Lambda||_2) delta,
    w   = any upper bound on ||Qhat^* Rhat||_2 (w <= rho),
    eps = (w + sqrt(w^2 + 4 rho^2)) / 2,
where Qhat is the orthonormal polar factor of X. Then there is an injective
map i -> j(i) with |lambda_{j(i)} - mu_i| <= eps for i = 1..g.

Theorem 2' (sharper cluster radius; docs/theory.md section 2, UNVERIFIED, D-12).
Under the same hypotheses, with spread(Lambda) = max(mu) - min(mu),
    eps* = ||R||_2 / sqrt(1 - eta) + ||X^* R||_2 / (1 - eta) + delta spread(Lambda) / sqrt(1 - eta)
also admits such an injection (Kahan's theorem for Qhat^* A Qhat, then Weyl for
Qhat^* A Qhat - Lambda). The cluster radius used is min(eps, eps*).

Each cluster member keeps its Theorem-1 radius as its *existence radius*; the
cluster radius is the one under which the members are matched to distinct
eigenvalues and the one used by every indexed statement downstream.

Disjoint groups therefore certify distinct eigenvalues; overlapping groups are
merged and re-bounded until the enlarged intervals of different groups are
disjoint. When vectors are missing or too far from orthonormal (eta >= 1),
the group only certifies existence (at least one eigenvalue per interval),
not multiplicity; multiplicities then come from inertia counts.

Theorem 3 (Kato 1949; Temple 1928). If the open interval (alpha, beta) contains
exactly one distinct eigenvalue lambda of A, x is a unit vector with Rayleigh
quotient theta in (alpha, beta) and r = ||A x - theta x||, then
    theta - r^2/(beta - theta) <= lambda <= theta + r^2/(theta - alpha).
Used to sharpen isolated enclosures once inertia counts certify isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .fp import (U, gamma, gamma_dot, up, norm2_upper, norm2_lower, fro_upper,
                 inf_norm_upper, max_nnz_per_row, safe_sqrt_upper, exact_dot, exact_gram,
                 sub_down, add_up, sub_up)

EXACT_GRAM_BUDGET = 20_000_000   # n*g*(g+1)/2 above which the gamma_n Gram bound is used


# ---------------------------------------------------------------------------
# operator adapters
# ---------------------------------------------------------------------------

class OperatorAdapter:
    """Uniform access to H x, |H| |x| bounds and ||H||_2 bounds.

    kind:
      'matrix'   : dense ndarray or scipy.sparse matrix (bounds rigorous)
      'spec'     : certify_ed.construct.SpecOperator (bounds rigorous, via abs_matvec)
      'blackbox' : LinearOperator / callable without |H| access (rounding of
                   the producer's matvec cannot be bounded; statements are
                   marked 'floating-point estimate')
    """

    def __init__(self, op, norm_bound: Optional[float] = None):
        self.op = op
        if sp.issparse(op) or isinstance(op, np.ndarray):
            self.kind = "matrix"
            self.A = op.tocsr() if sp.issparse(op) else np.asarray(op)
            self.shape = self.A.shape
            self.complex = bool(np.iscomplexobj(self.A.data if sp.issparse(self.A) else self.A))
            self.width = max_nnz_per_row(self.A)
            self._absA = abs(self.A) if sp.issparse(self.A) else np.abs(self.A)
            self._norm = inf_norm_upper(self.A)
        elif hasattr(op, "abs_matvec") and hasattr(op, "spec"):
            self.kind = "spec"
            self.shape = op.shape
            self.complex = op.dtype == np.complex128
            spec = op.spec
            max_len = max((len(t.op) for t in spec.terms), default=1)
            # each matrix element is a sum over couplings of products of <= max_len+1 factors
            self.width = spec.n_terms * (2 ** max_len) + max_len + 2
            ones = np.ones(op.shape[0])
            self._norm = up(float(np.max(op.abs_matvec(ones)))) * (1 + gamma(self.width))
        else:
            self.kind = "blackbox"
            self.shape = op.shape
            self.complex = np.dtype(getattr(op, "dtype", np.complex128)).kind == "c"
            self.width = None
            self._norm = norm_bound
        if norm_bound is not None:
            self._norm = norm_bound if self._norm is None else min(self._norm, norm_bound)

    @property
    def rigorous(self) -> bool:
        return self.kind in ("matrix", "spec")

    def matvec(self, X):
        if self.kind == "matrix":
            return self.A @ X
        if self.kind == "spec":
            return self.op.matvec(X)
        if hasattr(self.op, "matmat") and X.ndim == 2:
            return self.op.matmat(X)
        if X.ndim == 2:
            return np.column_stack([self.op.matvec(X[:, i]) for i in range(X.shape[1])])
        return self.op.matvec(X)

    def abs_matvec(self, X):
        if self.kind == "matrix":
            return self._absA @ np.abs(X)
        if self.kind == "spec":
            return self.op.abs_matvec(np.abs(X))
        raise RuntimeError("no |H| access for black-box operator")

    def norm_upper(self) -> Optional[float]:
        return self._norm


# ---------------------------------------------------------------------------
# residuals
# ---------------------------------------------------------------------------

@dataclass
class ResidualReport:
    computed: np.ndarray        # ||fl(Hx - mu x)||_2 as computed
    upper: np.ndarray           # rigorous upper bound on exact ||Hx - mu x||_2 (inf if not rigorous)
    xnorm_lower: np.ndarray
    rel_bound: np.ndarray       # upper / xnorm_lower  (Theorem 1 radius; inf if not rigorous)
    rel_estimate: np.ndarray    # computed / ||x|| (floating-point estimate, always finite)
    R: np.ndarray               # computed residual vectors (n x k)
    E: np.ndarray               # componentwise rounding bound on R (n x k), zeros if not rigorous
    rigorous: bool


def residuals(adapter: OperatorAdapter, mu: np.ndarray, X: np.ndarray) -> ResidualReport:
    mu = np.asarray(mu, dtype=float)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    HX = adapter.matvec(X)
    if HX.ndim == 1:
        HX = HX[:, None]
    R = HX - X * mu[None, :]
    n, k = X.shape
    computed = np.linalg.norm(R, axis=0)
    xl = np.array([norm2_lower(X[:, i]) for i in range(k)])
    if adapter.rigorous:
        cplx = adapter.complex or np.iscomplexobj(X)
        g = gamma_dot(adapter.width + 2, cplx)
        absHX = adapter.abs_matvec(X)
        if absHX.ndim == 1:
            absHX = absHX[:, None]
        gw = gamma(adapter.width + 1)
        absHX_up = absHX / (1.0 - gw) * (1 + 4 * U)
        E = g * (absHX_up + np.abs(X) * np.abs(mu)[None, :]) * (1 + 4 * U)
        upper = np.array([norm2_upper(R[:, i]) + norm2_upper(E[:, i]) for i in range(k)])
    else:
        E = np.zeros_like(np.abs(R))
        upper = np.full(k, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(xl > 0, upper / xl, np.inf) * (1 + 4 * U)
        est = computed / np.linalg.norm(X, axis=0)
    return ResidualReport(computed, upper, xl, rel, est, R, E, adapter.rigorous)


# ---------------------------------------------------------------------------
# enclosures
# ---------------------------------------------------------------------------

@dataclass
class Group:
    members: List[int]           # indices into the claimed list (sorted by claimed value)
    radius: float                # certified radius for every member
    distinct: bool               # True: members matched to distinct eigenvalues
    method: str                  # 'single' | 'cluster' | 'existence-only' | 'kato-temple'
    eta: Optional[float] = None  # orthonormality defect bound (cluster)
    rho: Optional[float] = None  # ||Rhat||_2 bound (cluster)
    note: str = ""
    bound: Optional[dict] = None # cluster: {'eps': Theorem 2, 'eps_star': Theorem 2', 'used': which is smaller}

    @property
    def size(self):
        return len(self.members)


@dataclass
class EnclosureSet:
    mu: np.ndarray               # claimed values
    radius: np.ndarray           # per-pair certified radius (cluster radius for cluster members: the
    #                              radius of the distinct matching, used by every indexed statement)
    group_of: np.ndarray         # group index per pair
    groups: List[Group]
    residual: ResidualReport
    rigorous: bool
    all_distinct: bool
    existence_radius: Optional[np.ndarray] = None   # per-pair Theorem-1 radius: an eigenvalue lies within
    #                              it of the claimed value (<= radius); None only for legacy construction

    def __post_init__(self):
        if self.existence_radius is None:
            self.existence_radius = np.array(self.radius, dtype=float, copy=True)

    def interval(self, i):
        """(lower, upper): bounds on the exact endpoints mu_i -/+ r_i (rounded outward)."""
        return (float(sub_down(self.mu[i], self.radius[i])), float(add_up(self.mu[i], self.radius[i])))

    def to_records(self) -> List[dict]:
        out = []
        for i in range(self.mu.size):
            g = self.groups[self.group_of[i]]
            out.append({"index": int(i), "claimed": float(self.mu[i]), "radius": float(self.radius[i]),
                        "lower": float(sub_down(self.mu[i], self.radius[i])),
                        "upper": float(add_up(self.mu[i], self.radius[i])),
                        "existence_radius": float(self.existence_radius[i]),
                        "group": int(self.group_of[i]), "group_size": g.size,
                        "method": g.method, "distinct": bool(g.distinct),
                        "residual_computed": float(self.residual.computed[i]),
                        "residual_bound": float(self.residual.upper[i])})
        return out


def _cluster_bound(adapter: OperatorAdapter, mu: np.ndarray, X: np.ndarray,
                   res: ResidualReport, idx: Sequence[int]):
    """Cluster radius for the group idx: min of the Theorem 2 radius eps and the Theorem 2' radius
    eps* = ||R||/sqrt(1-eta) + ||X^H R||/(1-eta) + delta*spread(mu)/sqrt(1-eta) (docs/theory.md).
    Returns (radius, eta, rho, {'eps', 'eps_star', 'used'}) or None when eta >= 1 or ||A|| is unknown."""
    Xg = X[:, idx]
    n, g = Xg.shape
    cplx = np.iscomplexobj(Xg)
    gd = gamma_dot(n, cplx)
    if n * g * (g + 1) // 2 <= EXACT_GRAM_BUDGET:
        # accurately rounded Gram entries: error <= 2u|G_ij| each (error-free transformations)
        G, Eg = exact_gram(Xg)
        eta = up(fro_upper(G - np.eye(g)) + fro_upper(Eg) + 4 * U * g)
    else:
        G = Xg.conj().T @ Xg
        xf2 = fro_upper(Xg) ** 2
        eta = up(fro_upper(G - np.eye(g)) + gd * xf2)
    if not eta < 1.0:
        return None
    delta = 1.0 - math.sqrt(1.0 - eta)
    delta = up(delta) + 4 * U
    Rg_fro = safe_sqrt_upper(float(np.sum(res.upper[list(idx)] ** 2)))
    Hn = adapter.norm_upper()
    if Hn is None or not np.isfinite(Hn):
        return None
    lam_n = float(np.max(np.abs(mu[list(idx)])))
    rho = up(Rg_fro + (Hn + lam_n) * delta)
    # w bound: ||Qhat^* Rhat|| <= ||X^* R_exact|| + ||X|| ||Rhat - R|| + delta ||Rhat||
    Rc = res.R[:, idx]
    XR = Xg.conj().T @ Rc
    xnorm = math.sqrt(1.0 + eta)
    Rerr = safe_sqrt_upper(float(sum(norm2_upper(res.E[:, j]) ** 2 for j in idx)))
    # ||X^* R_exact||_2 <= ||fl(X^* Rc)||_F + rounding of that product + ||X|| ||R_exact - Rc||
    XR_bound = up(fro_upper(XR) + gd * fro_upper(Xg) * fro_upper(Rc) + xnorm * Rerr)
    w = XR_bound + xnorm * (Hn + lam_n) * delta + delta * rho
    w = min(up(w), rho)
    eps = up(0.5 * (w + math.sqrt(w * w + 4.0 * rho * rho)))
    # Theorem 2' (IMP-13): every factor an upper bound; up() covers the few roundings of each product
    inv_sqrt = up(1.0 / math.sqrt(1.0 - eta))
    inv = up(1.0 / (1.0 - eta))
    mu_g = mu[list(idx)]
    spread = up(float(sub_up(float(np.max(mu_g)), float(np.min(mu_g)))))
    eps_star = up(Rg_fro * inv_sqrt + XR_bound * inv + delta * spread * inv_sqrt)
    used = "eps_star" if eps_star < eps else "eps"
    return min(eps, eps_star), eta, rho, {"eps": float(eps), "eps_star": float(eps_star), "used": used}


def enclose(adapter: OperatorAdapter, mu: Sequence[float], X: Optional[np.ndarray]) -> EnclosureSet:
    """Certified intervals for claimed eigenpairs (mu_i, x_i) of the stored operator."""
    mu = np.asarray(mu, dtype=float)
    k = mu.size
    if X is None:
        raise ValueError("eigenvectors are required for enclosures (eigenvalue-only claims "
                         "are handled by inertia counts)")
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    # IMP-21/IMP-22: NaN compares False everywhere, so a non-finite claimed value would pass every
    # overlap and accuracy test; refuse it at the entry
    if not np.all(np.isfinite(mu)):
        raise ValueError("claimed eigenvalues must be finite (found NaN/inf at indices "
                         f"{np.nonzero(~np.isfinite(mu))[0].tolist()[:8]})")
    if not np.all(np.isfinite(X)):
        raise ValueError("claimed eigenvectors must be finite (non-finite entries in columns "
                         f"{np.nonzero(~np.all(np.isfinite(X), axis=0))[0].tolist()[:8]})")
    order = np.argsort(mu, kind="stable")
    res = residuals(adapter, mu, X)
    radius = res.rel_bound.copy() if res.rigorous else res.rel_estimate.copy()

    # start: every pair its own group; merge by overlap of current intervals
    groups: List[List[int]] = [[int(i)] for i in order]
    rad_of_group = [float(radius[g[0]]) for g in groups]
    kinds = ["single"] * len(groups)
    meta = [None] * len(groups)

    def overlaps(ga, ra, gb, rb):
        hi_a = add_up(max(mu[i] for i in ga), ra)
        lo_b = sub_down(min(mu[i] for i in gb), rb)
        return hi_a >= lo_b          # outward-rounded ends: touching intervals are merged

    changed = True
    while changed:
        changed = False
        merged, mr, mk, mm = [groups[0]], [rad_of_group[0]], [kinds[0]], [meta[0]]
        for g, r, kd, me in zip(groups[1:], rad_of_group[1:], kinds[1:], meta[1:]):
            if overlaps(merged[-1], mr[-1], g, r):
                newg = merged[-1] + g
                cb = _cluster_bound(adapter, mu, X, res, newg) if res.rigorous else None
                if cb is None:
                    # existence only: each member keeps its own Theorem-1 radius
                    merged[-1] = newg
                    mr[-1] = float(max(radius[i] for i in newg))
                    mk[-1] = "existence-only"
                    mm[-1] = None
                else:
                    eps, eta, rho, info = cb
                    merged[-1] = newg
                    mr[-1] = eps
                    mk[-1] = "cluster"
                    mm[-1] = (eta, rho, info)
                changed = True
            else:
                merged.append(g); mr.append(r); mk.append(kd); mm.append(me)
        groups, rad_of_group, kinds, meta = merged, mr, mk, mm

    group_objs: List[Group] = []
    group_of = np.zeros(k, dtype=int)
    final_radius = np.zeros(k)
    for gi, (g, r, kd, me) in enumerate(zip(groups, rad_of_group, kinds, meta)):
        members = sorted(g, key=lambda i: mu[i])
        if kd == "existence-only":
            rads = [float(radius[i]) for i in members]
        else:
            rads = [float(r)] * len(members)
        # a non-finite radius (zero vector) encloses nothing and never counts as a distinct match
        distinct = kd in ("single", "cluster") and res.rigorous and all(math.isfinite(x) for x in rads)
        for i, ri in zip(members, rads):
            group_of[i] = gi
            final_radius[i] = ri
        group_objs.append(Group(members, float(max(rads)), distinct, kd if res.rigorous else "estimate",
                                eta=None if me is None else float(me[0]),
                                rho=None if me is None else float(me[1]),
                                bound=None if me is None else me[2]))
    # safety net: a group may claim distinct eigenvalues only if its union interval is
    # disjoint from every other group's union interval
    if len(group_objs) > 1:
        L = np.array([min(sub_down(mu[i], final_radius[i]) for i in g.members) for g in group_objs])
        Uu = np.array([max(add_up(mu[i], final_radius[i]) for i in g.members) for g in group_objs])
        order_g = np.argsort(L, kind="stable")
        Ls, Us = L[order_g], Uu[order_g]
        prefmax = np.maximum.accumulate(np.r_[-np.inf, Us[:-1]])
        nextL = np.r_[Ls[1:], np.inf]
        clash = (Ls <= prefmax) | (Us >= nextL)
        for pos, gi in enumerate(order_g):
            if clash[pos] and group_objs[gi].distinct:
                group_objs[gi].distinct = False
                group_objs[gi].note = "union interval overlaps another group"
    all_distinct = all(g.distinct for g in group_objs)
    # existence radius: Theorem 1 per member (never larger than the cluster radius; min() guards rounding)
    existence = np.minimum(np.asarray(radius, dtype=float), final_radius)
    return EnclosureSet(mu, final_radius, group_of, group_objs, res, res.rigorous, all_distinct, existence)


def cluster_groups(enc: EnclosureSet, cluster_tol: float) -> List[List[int]]:
    """Partition claimed pairs into clusters: members of the same enclosure group,
    or consecutive claimed values closer than cluster_tol."""
    order = np.argsort(enc.mu, kind="stable")
    parent = list(range(enc.mu.size))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for g in enc.groups:
        for m in g.members[1:]:
            parent[find(m)] = find(g.members[0])
    for a, b in zip(order[:-1], order[1:]):
        if enc.mu[b] - enc.mu[a] <= cluster_tol:
            parent[find(b)] = find(a)
    out = {}
    for i in order:
        out.setdefault(find(i), []).append(int(i))
    return list(out.values())


# ---------------------------------------------------------------------------
# Kato-Temple sharpening
# ---------------------------------------------------------------------------

def rayleigh_interval(adapter: OperatorAdapter, x: np.ndarray):
    """Rigorous enclosure [lo, hi] of theta = x^* A x / x^* x and the Theorem-1 radius at theta_hat.

    A x is computed in floating point (componentwise error <= gamma_w |A||x|); the two inner
    products are evaluated with error-free transformations (see fp.exact_dot).
    """
    x = np.asarray(x).ravel()
    cplx = adapter.complex or np.iscomplexobj(x)
    Ax = adapter.matvec(x)
    num_c, num_e = exact_dot(x, Ax)
    num = float(np.real(num_c))
    den_c, den_e = exact_dot(x, x)
    den = float(np.real(den_c))
    gdw = gamma_dot(adapter.width + 2, cplx)
    gw = gamma(adapter.width + 1)
    absAx = adapter.abs_matvec(x) / (1 - gw)
    xa = np.abs(x)
    # |x^*(Ax - fl(Ax))| <= |x|^T gdw |A||x|, bounded with a guarded float dot of non-negatives
    num_err = up(num_e + gdw * float(xa @ absAx) * (1 + gamma(x.size + 2)))
    den_err = up(den_e)
    den_lo = den - den_err
    if den_lo <= 0:
        return None
    cands = [(num - num_err) / (den + den_err), (num - num_err) / den_lo,
             (num + num_err) / (den + den_err), (num + num_err) / den_lo]
    lo, hi = min(cands), max(cands)
    lo -= abs(lo) * 4 * U
    hi += abs(hi) * 4 * U
    theta_hat = num / den
    res = residuals(adapter, np.array([theta_hat]), x[:, None])
    return lo, hi, theta_hat, float(res.rel_bound[0])


def kato_temple(adapter: OperatorAdapter, x: np.ndarray, alpha: float, beta: float):
    """Certified [lo, hi] for the unique eigenvalue in (alpha, beta), or None if
    the hypotheses cannot be verified (theta not safely inside, or margins
    smaller than the residual)."""
    ri = rayleigh_interval(adapter, x)
    if ri is None:
        return None
    th_lo, th_hi, th, r_hat = ri
    # r = ||Ax - theta x|| <= ||Ax - th x|| + |theta - th| (unit-norm normalisation in rel_bound)
    r = up(r_hat + max(th_hi - th, th - th_lo))
    if not (alpha < sub_down(th_lo, r) and add_up(th_hi, r) < beta):
        return None
    r2 = np.nextafter(r * r, np.inf)
    lower = float(sub_down(th_lo, np.nextafter(r2 / sub_down(beta, th_lo), np.inf)))
    upper = float(add_up(th_hi, np.nextafter(r2 / sub_down(th_hi, alpha), np.inf)))
    lower -= abs(lower) * 8 * U
    upper += abs(upper) * 8 * U
    return float(lower), float(upper), float(th), float(r)
