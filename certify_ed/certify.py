"""Orchestration: turn a claimed ED result into a certificate of explicit statements.

Statements (ids are stable and appear in certificates):

  S1.model_fidelity   the producer's operator equals the operator defined by the
                      ModelSpec (entrywise, or invariance + isometry for a
                      symmetry sector, or agreement on random probes for a
                      matrix-free producer)
  S2.eigenpairs       each claimed eigenvalue lies within a certified radius of a
                      true eigenvalue, distinct claimed pairs matched to distinct
                      eigenvalues (enclosure.Theorems 1-2, optionally 3)
  S3.completeness     no eigenvalue is missing from the claimed set inside the
                      claimed window (inertia counts), and a certified lower
                      bound on the first unclaimed eigenvalue (gap)
  S4.eigenspaces      basis-independent statements about claimed clusters:
                      projector error, certified observables, quantum numbers
  S5.corroboration    optional recomputation with algorithmically distinct
                      solvers, with their shared components declared
  S6.exact_recheck    optional (Options.exact_recheck): S2 radii and the S3 count of a
                      'lowest' claim re-derived in integer arithmetic, with no
                      floating-point assumption (checker.exact_recheck_lowest)
  S7.thermodynamics   whole-spectrum claims with Options.temperatures: interval bounds on
                      Z, F, U, C from the complete enclosure (thermo.py, Prop 16; UNVERIFIED)
  S8.degeneracy       number of eigenvalues in each cluster by bracketing inertia counts
                      (degeneracy.py, Prop 14; UNVERIFIED)

Every statement records its evidence, its assumptions and its cost in seconds. Which statements run is
decided by `presets.resolve_selection` (Options.preset / only / skip) and by the budget (Options.
budget_seconds / budget_mb); a statement that does not run is NOT_CHECKED with a reason code, and the
certificate's `scope` block lists the decision (scope.py, docs/presets.md, docs/SCOPE_MATRIX.md).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import Certificate, environment_record, matrix_sha256, array_sha256
from .cost import resolve_budget_seconds
from .diagnostics import annotate_next_actions
from .construct import (StateSet, build_bitwise, cross_construct, full_state_set, hermitian_part, matrix_difference,
                        product_tolerance, SpecOperator)
from .eigenspace import (certify_eigenspace, certify_quantum_numbers,
                         commutes_diagonal, rotation_spread)
from .enclosure import (OperatorAdapter, StreamedColumns, enclose, kato_temple, _cluster_bound, cluster_groups,
                        is_streamed, nonfinite_columns, read_columns)
from .fp import (U, ULP_MIN, UNDERFLOW_THRESHOLD, GuardRangeError, add_up, float_env_record,
                 guard_range_findings, inf_norm_upper, probe_float_environment, sub_down, underflow_scan, up)
from .inertia import inertia as inertia_count
from .invariants import compare_invariants, matrix_invariants, spec_invariants
from .precision import binary64, check_values, widening_record
from .presets import STATEMENTS, preset_options, resolve_selection
from .scope import Budget, build_scope_block
from .sectors import audit_sector, restrict_to_u1
from .spec import ModelSpec


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A claimed spectral result, as handed over by a producer.

    kind:
      'lowest' : the claimed eigenvalues are the len(eigenvalues) lowest of the operator
      'window' : the claimed eigenvalues are all eigenvalues in `window` = (a, b)
      'all'    : the claim is the full spectrum
      'subset' : no completeness claim (only accuracy of the given pairs)
    operator: the producer's operator (dense/sparse matrix or LinearOperator);
      None means "use the matrix CERTIFY-ED builds from the spec".
    states: basis states of the producer's operator when it is a U(1)-restricted
      block expressed in the spec's integer encoding (e.g. QuSpin basis.states).
    sector_isometry: for symmetry-adapted blocks (momentum, parity, ...), the
      producer's isometry V (rows in the spec's full-space encoding).
    """
    eigenvalues: Sequence[float]
    eigenvectors: Optional[np.ndarray] = None
    operator: Any = None
    kind: str = "subset"
    window: Optional[Tuple[float, float]] = None
    states: Optional[StateSet] = None
    sector_isometry: Optional[sp.spmatrix] = None
    producer: Dict[str, Any] = field(default_factory=dict)
    accuracy_target: Optional[float] = None   # None -> default 1e-8*max(1,||H||); inf -> no target
    description: str = ""

    def __post_init__(self):
        if self.kind not in CLAIM_KINDS:
            raise ValueError(f"claim kind must be one of {', '.join(CLAIM_KINDS)} (got {self.kind!r})")
        if self.window is not None:
            self.window = parse_window(self.window)
        if self.kind == "window" and self.window is None:
            raise ValueError("a 'window' claim needs window=(a, b): the claim is that the eigenvalues are ALL "
                             "eigenvalues in (a, b)")


CLAIM_KINDS = ("subset", "lowest", "window", "all")


def parse_window(value) -> Tuple[float, float]:
    """(a, b) with a < b, both finite, from a pair, a list, or a string 'a,b' / 'a b' (C-14: the same rule for
    every importer, the bundle's meta.json and the command line)."""
    if isinstance(value, str):
        parts = [p for p in value.replace(";", ",").replace(" ", ",").split(",") if p]
    else:
        try:
            parts = list(value)
        except TypeError:
            raise ValueError(f"window must be a pair (a, b), got {value!r}") from None
    if len(parts) != 2:
        raise ValueError(f"window must be a pair (a, b), got {value!r}")
    try:
        a, b = float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        raise ValueError(f"window endpoints must be numbers, got {value!r}") from None
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"window endpoints must be finite, got ({a}, {b}); a 'lowest' or 'all' claim needs no window")
    if not a < b:
        raise ValueError(f"window must satisfy a < b, got ({a}, {b})")
    return (a, b)


@dataclass
class Options:
    model_fidelity: bool = True
    kron_cross_check_max_dim: int = 2 ** 14
    reference_max_dim: int = 2 ** 22
    inertia: bool = True
    inertia_method: str = "auto"
    inertia_max_dim: int = 60000
    gap_fractions: Sequence[float] = (0.9, 0.5, 0.2, 0.05)
    kato_temple: bool = True
    eigenspaces: str = "clusters"            # 'none' | 'clusters' | 'all'
    cluster_tol: float = 0.0                 # claimed values closer than this form one cluster (S4)
    observables: Dict[str, Any] = field(default_factory=dict)       # name -> operator in claim space
    symmetries: Dict[str, Tuple[np.ndarray, Optional[float]]] = field(default_factory=dict)
    #   name -> (diagonal of S in the claim space, lattice spacing of its eigenvalues or None)
    corroborate: Sequence[str] = ()
    store_eigenvectors: bool = False
    probe_vectors: int = 3
    exact_recheck: bool = False              # S6: re-derive S2 radii and S3 count in exact arithmetic
    exact_max_dim: int = 4000
    seed: int = 12345
    # ---- scope (3.1): selection, presets, budgets; see presets.py / scope.py
    preset: Optional[str] = None             # name in presets.PRESETS; sets the statement list
    only: Optional[Sequence[str]] = None     # restrict to these statements (dependencies are added)
    skip: Sequence[str] = ()                 # remove these statements
    narrowing_reason: str = ""               # the user's reason, copied into the scope block
    budget_seconds: Any = None               # statements not started within this wall time are NOT_CHECKED;
    #   a number, None, or "auto" (default budget from the E2 cost model, certify_ed.cost; advisory)
    budget_mb: Optional[float] = None        # statements whose estimated memory exceeds this are NOT_CHECKED
    # ---- new statements (3.1, UNVERIFIED)
    temperatures: Sequence[float] = ()       # S7 for whole-spectrum claims
    whole_spectrum: bool = False             # record the Prop 9 interval list in the certificate (kind 'all')
    correlations: Dict[str, Any] = field(default_factory=dict)
    #   name -> {"letter": "z", "sites": [...], "positions": [[x, y], ...] (optional), "q": [[...], ...] (optional)}
    degeneracy: bool = True                  # S8 for the S4 clusters
    # ---- entanglement (C-03, 3.2): name -> {"sites": [...], "log_base": "e" | 2, "max_dim_a": int}; S4 evidence
    # `entanglement` per cluster: certified interval for S(rho_A) and the entanglement spectrum (entanglement.py)
    entanglement: Dict[str, Any] = field(default_factory=dict)
    # ---- approximate symmetries (C-12, 3.2): name -> (S, lattice_or_None[, allowed_values]); S a matrix or a diagonal in
    # the claim space; the commutator bound ||[H, S]|| is computed from the stored reference matrix and recorded as the
    # hypothesis of the label record (davis_kahan.py)
    approximate_symmetries: Dict[str, Any] = field(default_factory=dict)
    # ---- no-underflow hypothesis (INT-01, 3.2.0.dev0): scan operands, radii and residual bounds for nonzero
    # magnitudes below fp.UNDERFLOW_THRESHOLD; a finding makes S2 and its dependents INCONCLUSIVE
    # (UNDERFLOW_RISK). False assumes the hypothesis; the choice is recorded in the scope block's options.
    underflow_check: bool = True
    # ---- feedback (3.2)
    progress: Any = False                    # True: one line per statement on stderr; or a callable taking the line
    # ---- streaming vectors (A-06, 3.2): claim.eigenvectors may be a numpy memmap or an h5py dataset; the residuals
    # are then formed in column blocks of about block_mb megabytes of temporaries (None: budget_mb if given, else
    # enclosure.DEFAULT_BLOCK_MB). The results do not depend on the block width (sparse/spec operators; a dense
    # operator reads the vectors whole). Advisory for memory only: never changes a number.
    block_mb: Optional[float] = None
    # ---- planner (D-08, 3.2): print the predicted seconds and peak memory per statement (cost.plan_for_claim) before
    # running, and record it under body["plan"]. Advisory: it never changes a status.
    plan: bool = False
    # ---- matrix-free S2 (D-02, 3.2, docs/proofs/D-02_matrix-free.md): for a claim without an operator, apply the
    # spec term by term (matrixfree.MatrixFreeOperator) at any dimension instead of building a reference matrix;
    # the roundings of the local amplitudes are counted in the S2 radius (no model shift), the amplitude hypothesis
    # is recorded in S1; S3 and S8 need a stored matrix and are NOT_CHECKED (NEEDS_MATRIX).
    matrix_free: bool = False
    # ---- GPU products (D-05, 3.3.0.dev0, gpu.py): device="gpu" forms H X and |H| |X| for S2 and Kato-Temple on a
    # CUDA device with the ordered kernels of gpu.py, bit for bit equal to the CPU products (device probe and spot
    # checks), for a stored sparse operator; otherwise, or on any failure, the CPU computes them and S2's evidence
    # `device` says why. It never changes a number of the certificate.
    device: str = "cpu"

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "Options":
        """Options with a preset's values (presets.PRESETS[name]['options']), then the overrides."""
        kw = preset_options(name)
        kw.update(overrides)
        return cls(**kw)


DEFAULT_REL_ACCURACY = 1e-8


def _stmt(sid, status, summary, evidence=None, assumptions=None, seconds=0.0, reason_code=None):
    assert status in ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED") or \
        (sid == "S5.corroboration" and status == "CORROBORATED")
    ev = dict(evidence or {})
    if reason_code is not None:
        ev["reason_code"] = reason_code
    return {"id": sid, "status": status, "summary": summary, "evidence": ev,
            "assumptions": assumptions or [], "seconds": round(float(seconds), 6)}


class _StatementLog(list):
    """The statement list; with Options.progress it reports each statement as it is appended (status and
    elapsed wall time). Presentation only: it changes nothing in the records."""

    def __init__(self, progress, t_start):
        super().__init__()
        self._sink = (print if progress is True else progress) if progress else None
        self._t0 = t_start

    def append(self, stmt):
        super().append(stmt)
        if self._sink is not None:
            line = f"[{time.time() - self._t0:8.2f} s] {stmt['id']:20s} {stmt['status']:13s} {stmt.get('summary', '')}"
            if self._sink is print:
                print(line, file=sys.stderr, flush=True)
            else:
                self._sink(line)


def _gate(sel, budget, short, estimate_mb=None, what="", use_cap=True):
    """None when the statement may run; else a NOT_CHECKED statement with its reason code."""
    sid = STATEMENTS[short]["id"]
    nar = sel.narrowing_of(short)
    if nar is not None:
        code = "DEPENDENCY_NARROWED" if nar["by"] == "dependency" else "NARROWED_BY_USER"
        return _stmt(sid, "NOT_CHECKED", f"not selected ({nar['by']}: {nar['reason']})", {"narrowed": nar},
                     reason_code=code)
    ok, note = budget.time_left(sid)
    if not ok:
        return _stmt(sid, "NOT_CHECKED", note, {"budget": budget.to_dict()}, reason_code="BUDGET_TIME")
    ok, note = budget.memory_ok(sid, estimate_mb, what, use_cap=use_cap)
    if not ok:
        return _stmt(sid, "NOT_CHECKED", note, {"estimate_mb": estimate_mb}, reason_code="BUDGET_MEMORY")
    return None


def _mb(n_entries: float, bytes_per: float) -> float:
    return float(n_entries) * bytes_per / 2 ** 20


def _kt_block(n: int, cplx: bool, block_mb: Optional[float]) -> int:
    """Columns per Kato-Temple product block: three dense n x b temporaries within block_mb (enclosure.DEFAULT_BLOCK_MB)."""
    from .enclosure import DEFAULT_BLOCK_MB
    mb = DEFAULT_BLOCK_MB if block_mb is None else float(block_mb)
    return int(max(1, (mb * 2 ** 20) // (max(1, n) * (16 if cplx else 8) * 3)))


def _underflow_operands(mu, X, adapter, seed: int):
    """The stored operands of the S2 bounds, for the no-underflow scan (fp.underflow_scan).

    (name, array, threshold) tuples: the claimed eigenvalues and eigenvector entries; for a stored matrix
    its entries (the sparse data array); for a spec or black-box operator the norm bound and the result of
    one matvec on a fixed random probe (the only access to its entries; the probe uses the certificate's
    seed). Neither the intermediate quantities of the Theorem 1 bound (residual vectors, entrywise rounding
    bound) nor its outputs (radii, residual bounds) are scanned: that bound holds under the underflow-tolerant
    model for every magnitude (INT-05, docs/proofs/int05_underflow.md). The hypothesis is stated for the
    stored inputs and the model shift, which the standard-model bounds of the other statements use.
    """
    T = UNDERFLOW_THRESHOLD
    parts = [("claim.eigenvalues", mu, T)]
    if X is not None:
        parts.append(("claim.eigenvectors", X, T))
    if adapter.kind == "matrix":
        parts.append(("operator.entries", adapter.A.data[:adapter.A.nnz] if sp.issparse(adapter.A) else adapter.A, T))
    else:
        nb = adapter.norm_upper()
        if nb is not None:
            parts.append(("operator.norm_bound", np.array([nb], dtype=float), T))
        n = int(adapter.shape[0])
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(n)
        if adapter.complex:
            v = v + 1j * rng.standard_normal(n)
        try:
            parts.append(("operator.probe_matvec", np.asarray(adapter.matvec(v)), T))
        except Exception:      # a producer matvec that fails on a probe is reported by S1/S2 themselves
            pass
    return parts


def _underflow_statement_note(findings):
    where = ", ".join(f"{f['where']} (min {f['min_magnitude']:.2e}, {f['count']} entr{'y' if f['count'] == 1 else 'ies'})"
                      for f in findings)
    return (f"no-underflow hypothesis not established: nonzero magnitudes below 2^-900 in {where}; the "
            "floating-point bounds assume operands and radii above 2^-900 (Options(underflow_check=False) "
            "assumes it)")


def _guard_range_note(findings):
    what = ", ".join(f"{f['quantity']} = {f['value']} ({f['where']})" for f in findings)
    return (f"outside the range of the floating-point guards: {what} exceeds {findings[0]['limit']}; the guards of "
            "Theorem 1 (Lambda4 to Lambda6, docs/proofs/thm1_fp-accounting.md) and of the sparse residual (D-01, H3) "
            "are established for n, w <= 2^26 only")


_GUARD_RANGE_MARK = "the range of the floating-point guards"


def _subspace_shift(iso: float, inv: float, H_norm: float, M_norm: float) -> Optional[float]:
    """Theorem 2 with X = V (sector isometry, g = d_s) and Hermitian M = H_s: every eigenvalue of H_s is
    within rho = inv + (||H|| + ||H_s||) delta of a distinct eigenvalue of H (Kahan's theorem on the polar
    factor of V, docs/proofs/thm2_polar-factor.md). Was (1 + sqrt 5)/2 rho before track/backlog-proofs."""
    if not iso < 1:
        return None
    delta = up(1 - math.sqrt(1 - iso)) + 4 * U
    return up(inv + (H_norm + M_norm) * delta) + ULP_MIN


# ---------------------------------------------------------------------------
# S1: model fidelity
# ---------------------------------------------------------------------------

def _diagnose_mismatch(spec: ModelSpec, states: StateSet, Hp: sp.spmatrix, Href: sp.spmatrix,
                       max_couplings: int = 400) -> dict:
    """Explain a matrix mismatch: global scale factor, or per-coupling attribution."""
    D = (sp.csr_matrix(Hp) - Href).tocsr()
    out = {}
    num = (Href.multiply(Hp.conj())).sum()
    den = (Href.multiply(Href.conj())).sum()
    if abs(den) > 0:
        alpha = complex(num / den)
        resid = sp.linalg.norm(sp.csr_matrix(Hp) - alpha.real * Href) / max(sp.linalg.norm(Href), 1e-300)
        out["best_scale_factor"] = alpha.real
        out["relative_residual_after_scaling"] = float(resid)
    if spec.n_terms <= max_couplings:
        # least-squares fit  D ~ sum_t a_t O_t  over all couplings of the spec (the O_t need not be
        # orthogonal once restricted to a sector, so a joint fit is required)
        n = D.shape[0]
        labels, cols = [], []
        for t in spec.terms:
            for c, s in t.couplings:
                sub = ModelSpec(spec.n_sites, spec.local, [type(t)(t.op, [(1.0, s)])],
                                state_order=spec.state_order)
                O = sp.coo_matrix(build_bitwise(sub, states))
                labels.append((t.op, list(s), complex(c)))
                cols.append(O)
        keys_all = np.unique(np.concatenate([o.row.astype(np.int64) * n + o.col for o in cols] +
                                            [sp.coo_matrix(D).row.astype(np.int64) * n + sp.coo_matrix(D).col]))
        def vec(M):
            M = sp.coo_matrix(M)
            v = np.zeros(keys_all.size, dtype=complex)
            v[np.searchsorted(keys_all, M.row.astype(np.int64) * n + M.col)] += M.data
            return v
        Mmat = np.column_stack([vec(o) for o in cols])
        d = vec(D)
        a, _, rank, sv = np.linalg.lstsq(Mmat, d, rcond=None)
        resid = float(np.linalg.norm(Mmat @ a - d) / max(np.linalg.norm(d), 1e-300))
        attrib = []
        for (op_, s_, c_), ai in zip(labels, a):
            if abs(ai) > 1e-9 * max(1.0, abs(c_)):
                attrib.append({"op": op_, "sites": s_, "spec_coefficient": [c_.real, c_.imag],
                               "producer_minus_spec": [float(ai.real), float(ai.imag)]})
        attrib.sort(key=lambda r: -abs(complex(*r["producer_minus_spec"])))
        out["coupling_fit"] = {
            "method": "least squares over all spec couplings on the producer's basis",
            "rank": int(rank), "n_couplings": len(labels),
            "relative_residual": resid,
            "interpretation": ("difference fully explained by changed coupling coefficients"
                               if resid < 1e-8 else
                               "difference NOT expressible as changed coefficients of the spec's operator "
                               "strings (wrong operator content, basis/ordering convention, or missing term types)"),
            "couplings": attrib[:16], "n_couplings_implicated": len(attrib)}
    return out


def _builder_shift(spec: ModelSpec, states: Optional[StateSet] = None) -> float:
    """Upper bound on ||H_ref - H_spec||_2, where H_spec is the spec operator on `states` in exact
    arithmetic (couplings taken as the given binary floating-point numbers) and H_ref is CERTIFY-ED's
    floating-point build of it.

    Every entry of H_ref is a sum of at most m = n_terms * 2^Lmax contributions, each a product of the
    coupling and at most Lmax local amplitudes; each local amplitude carries one rounding when it is not
    exactly representable (square roots for S >= 1 and bosons), each of the Lmax multiplications one more,
    and the sum m - 1: a relative perturbation theta with |theta| <= gamma_{m + 2 Lmax - 1} per contribution
    (one more rounding when the matrix is symmetrised, construct.hermitian_part). Hence
    |H_ref - H_spec| <= gamma_W |H|_abs entrywise with W = m + 2 Lmax + 2, where |H|_abs sums the absolute
    values of the exact contributions, and ||H_ref - H_spec||_2 <= gamma_W sqrt(||H|_abs||_1 ||H|_abs||_inf).
    Both norms are evaluated with the matrix-free operator (row sums and column sums of the computed
    nonnegative terms, which are at least the exact ones times (1 - gamma_{m + 2 Lmax - 1}); the division by
    (1 - gamma_W) undoes that). No symmetry of |H|_abs is assumed (docs/proofs/appA_model-shift.md)."""
    from .fp import gamma
    op = SpecOperator(spec, states)
    n = op.shape[0]
    if n == 0 or spec.n_terms == 0:
        return 0.0
    max_len = max((len(t.op) for t in spec.terms), default=1)
    width = spec.n_terms * (2 ** max_len) + 2 * max_len + 2
    ones = np.ones(n)
    row = float(np.max(op.abs_matvec(ones)))
    col = float(np.max(op.abs_rmatvec(ones)))
    g = gamma(width)
    # six roundings (product, sqrt, product, 1 - gamma, division, product) covered by (1 + 4u)(1 + 8u) of up()
    return up(g * math.sqrt(row * col) / (1 - gamma(width)) * (1 + 4 * U))


def check_model_fidelity(claim: Claim, spec: ModelSpec, opts: Options, H_full_ref=None):
    t0 = time.time()
    ev = {"spec_sha256": spec.sha256(), "spec_name": spec.name}
    op = claim.operator
    if claim.sector_isometry is not None:
        V_audit = claim.sector_isometry
        red = restrict_to_u1(spec, claim.sector_isometry) if H_full_ref is None else None
        if red is not None:
            H_full_ref, V_audit, st_u1 = red
            ev["audit_space"] = {"kind": "U(1) block (exactly invariant; leak 0)", "label": st_u1.label,
                                 "dimension": int(st_u1.size)}
        elif H_full_ref is None:
            H_full_ref = build_bitwise(spec)
            ev["audit_space"] = {"kind": "full Hilbert space", "dimension": int(spec.dim)}
        # Theorem 2 with X = V needs a Hermitian full operator; the builder shift covers the symmetrisation
        H_full_ref = hermitian_part(H_full_ref)
        Hs = sp.csr_matrix(op)
        herm = float(abs(Hs - Hs.conj().T).max()) if Hs.nnz else 0.0
        ev["hermiticity_max_abs"] = herm
        # the Hermitian part is what S2-S4 certify; audit exactly that matrix
        Hs = ((Hs + Hs.conj().T) / 2).tocsr() if herm > 0 else Hs
        audit = audit_sector(H_full_ref, V_audit, Hs, label="producer sector")
        ev["sector_audit"] = audit.to_dict()
        ev["model_shift_bound"] = _builder_shift(spec, st_u1 if red is not None else None)
        status = "CERTIFIED" if audit.passed else "FAILED"
        summ = (f"producer sector block is an invariant, orthonormal restriction of the spec "
                f"operator (isometry defect {audit.isometry_defect:.1e}, invariance defect "
                f"{audit.invariance_defect:.1e})" if status == "CERTIFIED" else
                f"sector audit failed (isometry defect {audit.isometry_defect:.1e}, invariance "
                f"defect {audit.invariance_defect:.1e}, hermiticity {herm:.1e})")
        return _stmt("S1.model_fidelity", status, summ, ev,
                     ["full-space reference built by CERTIFY-ED from the spec (bitwise path)"],
                     time.time() - t0), H_full_ref, audit

    n_ref = claim.states.size if claim.states is not None else spec.dim
    if opts.matrix_free and op is None:
        # D-02: no reference matrix at any dimension; the S2 radius concerns the exact spec operator directly
        from .matrixfree import MatrixFreeOperator
        mf = MatrixFreeOperator(spec, claim.states)
        hyp = mf.amplitude_hypothesis
        ev["reference"] = {"path": "matrix-free from the spec (D-02)", "dimension": int(n_ref),
                           "note": "no stored matrix; local amplitudes' roundings are counted in the S2 residual bound"}
        ev["model_shift_bound"] = 0.0
        ev["operator_source"] = "applied matrix-free from the spec by CERTIFY-ED (D-02)"
        ev["amplitude_hypothesis"] = hyp
        return _stmt("S1.model_fidelity", "CERTIFIED",
                     "operator applied matrix-free from the spec by CERTIFY-ED; " + hyp["statement"],
                     ev, [hyp["statement"], "docs/proofs/D-02_matrix-free.md (derived, independent re-derivation pending)"],
                     time.time() - t0), None, None
    if n_ref > opts.reference_max_dim:
        # too large to store a reference matrix: work with the matrix-free reference operator
        ref = SpecOperator(spec, claim.states)
        ev["reference"] = {"path": "bitwise, matrix-free", "dimension": int(n_ref),
                           "note": f"dimension > reference_max_dim {opts.reference_max_dim}: no stored reference"}
        ev["model_shift_bound"] = _builder_shift(spec, claim.states)
        if op is None:
            ev["operator_source"] = "applied matrix-free from the spec by CERTIFY-ED"
            return _stmt("S1.model_fidelity", "CERTIFIED",
                         "operator applied matrix-free from the spec by CERTIFY-ED (no producer matrix to compare)",
                         ev, [], time.time() - t0), None, None
        return _probe_comparison(op, ref.matvec, int(n_ref), spec, opts, ev, t0), None, None
    states = claim.states if claim.states is not None else full_state_set(spec)
    Href, leak = build_bitwise(spec, states, return_leak=True)
    ev["reference"] = {"path": "bitwise", "dimension": int(states.size), "nnz": int(Href.nnz),
                       "state_set": states.label, "leak_out_of_state_set": leak}
    if states.label == "full" and spec.dim <= opts.kron_cross_check_max_dim:
        cc = cross_construct(spec, kron_max_dim=opts.kron_cross_check_max_dim)
        ev["internal_cross_construction"] = {k: cc[k] for k in ("agree", "max_abs_diff", "tolerance") if k in cc}
        if cc.get("agree") is False:
            return _stmt("S1.model_fidelity", "INCONCLUSIVE",
                         "CERTIFY-ED's two reference builders disagree (internal error)", ev,
                         [], time.time() - t0), Href, None
    if states.label == "full":
        si = spec_invariants(spec) if spec.n_terms <= 400 else None
        if si is not None:
            ev["spec_invariants_vs_reference"] = compare_invariants(si, matrix_invariants(Href))
    if leak > 0:
        return _stmt("S1.model_fidelity", "FAILED",
                     f"the spec does not conserve the quantity defining the producer's basis "
                     f"(amplitude leaving the state set: {leak:.3e})", ev, [], time.time() - t0), Href, None
    ev["model_shift_bound"] = _builder_shift(spec, states)   # replaced below when a producer matrix is compared
    if op is None:
        ev["operator_source"] = "built by CERTIFY-ED from the spec"
        return _stmt("S1.model_fidelity", "CERTIFIED",
                     "operator built from the spec by CERTIFY-ED (no producer matrix to compare)",
                     ev, [], time.time() - t0), Href, None
    if sp.issparse(op) or isinstance(op, np.ndarray):
        Hp = sp.csr_matrix(op)
        herm = float(abs(Hp - Hp.conj().T).max()) if Hp.nnz else 0.0
        ev["hermiticity_max_abs"] = herm
        diff = matrix_difference(Hp, Href)
        tol = product_tolerance(spec, diff.get("max_abs_entry", 1.0))
        ev["comparison"] = dict(diff, tolerance=tol)
        ok = diff["shape_match"] and diff["max_abs_diff"] <= tol
        if ok:
            Hh = ((Hp + Hp.conj().T) / 2).tocsr() if herm > 0 else Hp
            Dm = (Hh - Href).tocsr()
            # ||D||_2 <= sqrt(||D||_1 ||D||_inf) <= max(||D||_1, ||D||_inf)
            ev["model_shift_bound"] = up(max(inf_norm_upper(Dm), inf_norm_upper(Dm.conj().T.tocsr()))
                                         + ev["model_shift_bound"])
            return _stmt("S1.model_fidelity", "CERTIFIED",
                         f"producer matrix equals the spec matrix entrywise (max |diff| "
                         f"{diff['max_abs_diff']:.1e} <= {tol:.1e})", ev,
                         ["spec encoding (site order, state order, local conventions) matches the producer's"],
                         time.time() - t0), Href, None
        if diff["shape_match"]:
            ev["diagnosis"] = _diagnose_mismatch(spec, states, Hp, Href)
        return _stmt("S1.model_fidelity", "FAILED",
                     f"producer matrix differs from the spec matrix (max |diff| {diff['max_abs_diff']:.3e}, "
                     f"{diff.get('n_differing_entries', '?')} entries)", ev, [], time.time() - t0), Href, None
    # black-box operator: probabilistic agreement on random vectors
    return _probe_comparison(op, lambda z: Href @ z, states.size, spec, opts, ev, t0), Href, None


def _probe_comparison(op, ref_apply, n: int, spec: ModelSpec, opts: Options, ev: dict, t0: float):
    """S1 by random probes: the producer's operator (black box, or a stored matrix too large for a
    stored reference) against the reference operator. Probabilistic; the bounds of S2-S4 are then
    computed with the reference operator, so the statements refer to the declared model."""
    rng = np.random.default_rng(opts.seed)
    worst = 0.0
    for _ in range(opts.probe_vectors):
        z = rng.standard_normal(n) + (1j * rng.standard_normal(n) if not spec.is_real() else 0.0)
        a = op.matvec(z) if hasattr(op, "matvec") else op @ z
        b = ref_apply(z)
        worst = max(worst, float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)))
    ev["probe"] = {"n_vectors": opts.probe_vectors, "max_relative_difference": worst}
    tol = 1e3 * U * max(1, spec.n_terms)
    st = "CERTIFIED" if worst <= tol else "FAILED"
    kind = "stored producer matrix" if (sp.issparse(op) or isinstance(op, np.ndarray)) else "matrix-free producer"
    return _stmt("S1.model_fidelity", st,
                 f"{kind} agrees with the spec on {opts.probe_vectors} random vectors "
                 f"(max rel. diff {worst:.1e})" if st == "CERTIFIED" else
                 f"{kind} disagrees with the spec (max rel. diff {worst:.3e})",
                 ev, ["probabilistic: a difference orthogonal to all probes would be missed "
                      "(probability zero for Gaussian probes in exact arithmetic)"],
                 time.time() - t0)


# ---------------------------------------------------------------------------
# S3: completeness
# ---------------------------------------------------------------------------

def _certify_lowest(A, mu_sorted, enc, opts, eps_shift=0.0):
    """Completeness of a 'k lowest' claim. A: matrix used for inertia.

    Probe order (each probe = one sparse/dense LDL^T factorization): first near a
    floating-point estimate of the (k+1)-th eigenvalue (a hint; correctness rests only
    on the inertia counts), then progressively closer to the highest claimed value,
    finally at a minimal margin. Typical cost when the claim is complete: one factorization.
    The first probe that counts more than k eigenvalues triggers one count just below the
    highest claimed value; if eigenvalues are missing there the claim fails at once
    (typical cost of a FAILED statement: two factorizations).
    """
    k = mu_sorted.size
    rad = add_up(enc.radius[np.argsort(enc.mu, kind="stable")], eps_shift)
    hi = add_up(mu_sorted, rad)
    lo = sub_down(mu_sorted, rad)
    hi_k = float(hi.max())
    counts = []
    scale = max(1.0, float(np.max(np.abs(mu_sorted))))
    margin = max(1e-8 * scale, 16 * float(rad.max()))
    distinct = enc.all_distinct
    est = _estimate_next(A, k, seed=opts.seed)
    sigmas = []
    if est is not None and est > hi_k + margin:
        sigmas += [hi_k + f * (est - hi_k) for f in opts.gap_fractions if f * (est - hi_k) > margin]
    sigmas.append(hi_k + margin)
    ev = {"counts": counts, "next_eigenvalue_estimate": est}
    r, sigma = None, None
    last_reason = None
    low_probe = {}

    lo_top = float(lo[np.argmax(mu_sorted)])

    def _below_top():
        """Inertia count just below the lower end of the highest claimed enclosure (cached).

        The shift starts at a minimal margin below; if the factorization's backward error is not
        smaller than the distance to the enclosure, the shift is moved down to four times the
        backward error (at most three retries), as for the probe above the claim."""
        if not low_probe:
            s_low = lo_top - margin
            for _ in range(4):
                r_low = inertia_count(A, s_low, method=opts.inertia_method, budget_mb=opts.budget_mb)
                counts.append(r_low.to_dict())
                if not r_low.conclusive or r_low.beta + eps_shift < lo_top - s_low:
                    break
                s_low = lo_top - 4 * (r_low.beta + eps_shift)
            low_probe.update(s_low=s_low, r=r_low)
        return low_probe["s_low"], low_probe["r"]

    def _missing_below(s_low, r_low):
        """Lower bound on eigenvalues missing below the top claim, or None if not conclusive.

        Eigenvalues <= s_low + beta number at least nu; claimed enclosures reaching below s_low + beta
        number c; if nu > c, at least nu - c eigenvalues below the top enclosure are unclaimed."""
        if r_low.conclusive and add_up(r_low.beta, eps_shift) < sub_down(lo_top, s_low):
            bl = add_up(r_low.beta, eps_shift)
            return int(r_low.negative - np.sum(lo <= add_up(s_low, bl)))
        return None

    for idx, sigma in enumerate(sigmas):
        tries = 0
        while True:
            r = inertia_count(A, sigma, method=opts.inertia_method, budget_mb=opts.budget_mb)
            counts.append(r.to_dict())
            if r.resource_failure:
                return "NOT_CHECKED", dict(ev, reason=f"inertia count not attempted to completion: {r.note}"), None
            # the final (minimal-margin) probe sits close to a claimed eigenvalue, where a
            # no-pivoting factorization can lose accuracy: widen the margin to the observed beta
            if (idx == len(sigmas) - 1 and r.conclusive and r.beta + eps_shift >= (sigma - hi_k) / 2
                    and tries < 3):
                sigma = hi_k + 4 * (r.beta + eps_shift)
                tries += 1
                continue
            break
        if not r.conclusive:
            last_reason = f"factorization at sigma={sigma:.10g} inconclusive ({r.note})"
            continue
        beta = float(add_up(r.beta, eps_shift))
        s_minus_b = float(sub_down(sigma, beta))
        if r.negative == k and distinct and hi_k < s_minus_b:
            # exactly k eigenvalues below sigma - beta, and they are the claimed ones
            ev.update(top_shift=sigma, beta_effective=beta, next_eigenvalue_lower_bound=s_minus_b,
                      certified_window=["-inf", s_minus_b], gap_lower_bound=float(sub_down(s_minus_b, hi_k)))
            return "CERTIFIED", ev, (float("-inf"), s_minus_b)
        if r.negative == k and distinct:
            last_reason = (f"factorization backward error ({beta:.2e}) too large relative to the "
                           f"margin above the highest claimed value")
        if r.negative <= k:
            break           # descending further cannot help
        # more than k eigenvalues below sigma: decide at once whether any lies below the top claim
        s_low, r_low = _below_top()
        miss = _missing_below(s_low, r_low)
        if miss is not None and miss > 0:
            ev.update(top_shift=sigma, beta_effective=beta, missing_below_top_claim_at_least=miss,
                      witness_shift=s_low)
            return "FAILED", ev, None
    if r is None or not r.conclusive:
        return "INCONCLUSIVE", dict(ev, reason=last_reason or "no conclusive inertia count"), None
    beta = float(add_up(r.beta, eps_shift))
    ev.update(top_shift=sigma, beta_effective=beta)
    if r.negative == k and not distinct:
        return "INCONCLUSIVE", dict(ev, reason="claimed pairs not all certified distinct (S2); "
                                               "the count matches but cannot be attributed"), None
    if r.negative == k:
        return "INCONCLUSIVE", dict(ev, reason=last_reason), None
    if r.negative < k:
        return "INCONCLUSIVE", dict(ev, reason=f"inertia counts {r.negative} eigenvalues below "
                                               f"{sigma:.10g} but {k} claimed pairs lie below it: "
                                               "claimed pairs are not all distinct eigenpairs"), None
    # Something is not accounted for. Is anything missing BELOW the top claimed value?
    s_low, r_low = _below_top()
    miss = _missing_below(s_low, r_low)
    if miss is not None:
        bl = float(add_up(r_low.beta, eps_shift))
        if miss > 0:
            ev["missing_below_top_claim_at_least"] = miss
            ev["witness_shift"] = s_low
            return "FAILED", ev, None
        c_lt = int(np.sum(hi < sub_down(s_low, bl)))
        if distinct and r_low.negative == c_lt:
            ev["complete_below"] = float(sub_down(s_low, bl))
    if r.negative > k:
        ev["unclaimed_near_top"] = int(r.negative - k)
        if "complete_below" in ev:
            n_top_claimed = int(np.sum(lo > ev["complete_below"]))
            ev["top_multiplet"] = {"claimed": n_top_claimed,
                                   "at_least": int(r.negative - r_low.negative),
                                   "interval": [ev["complete_below"], float(add_up(sigma, beta))]}
            ev["reason"] = (f"the claim is complete below {ev['complete_below']:.10g}, but the highest "
                            f"claimed level is a (quasi-)degenerate multiplet of at least "
                            f"{r.negative - r_low.negative} eigenvalues within "
                            f"[{ev['complete_below']:.10g}, {sigma + beta:.10g}], of which only "
                            f"{n_top_claimed} are claimed: 'the k lowest' is ill-posed for this k")
        else:
            ev["reason"] = ("at least one unclaimed eigenvalue lies within "
                            f"{margin + beta:.2e} above the highest claimed one")
        return "INCONCLUSIVE", ev, None
    ev["reason"] = "count above the claim could not be classified"
    return "INCONCLUSIVE", ev, None


def _estimate_next(A, k, seed=12345):
    """Floating-point estimate of the (k+1)-th smallest eigenvalue (a hint only).

    The Lanczos start vector is drawn from a generator seeded with ``seed`` so that the shifts, and
    hence the certified gap and the run time, are reproducible (SciPy otherwise seeds from OS entropy)."""
    try:
        n = A.shape[0]
        if k + 1 > n:
            return None
        if not sp.issparse(A) or n <= 1500:
            M = A.toarray() if sp.issparse(A) else np.asarray(A)
            import scipy.linalg as _la
            return float(_la.eigvalsh(M, subset_by_index=[k, k])[0])
        import scipy.sparse.linalg as _spla
        v0 = np.random.default_rng(seed).standard_normal(n)
        w = _spla.eigsh(A, k=min(k + 6, n - 2), which="SA", return_eigenvectors=False, tol=1e-10, v0=v0)
        return float(np.sort(w)[k])
    except Exception:
        return None


def _certify_window(A, enc, a, b, opts, eps_shift=0.0):
    ra = inertia_count(A, a, method=opts.inertia_method, budget_mb=opts.budget_mb)
    if ra.resource_failure:
        return "NOT_CHECKED", {"counts": [ra.to_dict()], "window": [a, b],
                               "reason": f"inertia count not attempted to completion: {ra.note}"}, None
    rb = inertia_count(A, b, method=opts.inertia_method, budget_mb=opts.budget_mb)
    ev = {"counts": [ra.to_dict(), rb.to_dict()], "window": [a, b]}
    if rb.resource_failure:
        return "NOT_CHECKED", dict(ev, reason=f"inertia count not attempted to completion: {rb.note}"), None
    if not (ra.conclusive and rb.conclusive):
        return "INCONCLUSIVE", dict(ev, reason="inertia inconclusive at a window edge"), None
    ba, bb = add_up(ra.beta, eps_shift), add_up(rb.beta, eps_shift)
    n = rb.negative - ra.negative
    lo = sub_down(enc.mu, add_up(enc.radius, eps_shift))
    hi = add_up(enc.mu, add_up(enc.radius, eps_shift))
    inside = (lo > add_up(a, ba)) & (hi < sub_down(b, bb))
    touching = (hi >= sub_down(a, ba)) & (lo <= add_up(b, bb))
    c_in = int(np.sum(inside)) if enc.all_distinct else None
    c_all = int(np.sum(touching))
    ev.update({"count_from_inertia": int(n), "claimed_certified_inside": c_in,
               "claimed_touching": c_all, "beta_a": ba, "beta_b": bb})
    if c_in is not None and n == c_in:
        # inward rounding of the certified window (IMP-08): the exact a + beta_a and b - beta_b may be rounded
        # the unsafe way by round-to-nearest; the same window feeds Kato-Temple and the S4 separations
        lo_w, hi_w = float(add_up(a, ba)), float(sub_down(b, bb))
        ev["certified_window"] = [lo_w, hi_w]
        return "CERTIFIED", ev, (lo_w, hi_w)
    if n > c_all:
        ev["missing_at_least"] = int(n - c_all)
        return "FAILED", ev, None
    return "INCONCLUSIVE", ev, None


def _edge_count_exact(A, e, opts, eps_shift, counts, scale):
    """Exact number of eigenvalues below a window edge e, from inertia counts alone (C-02).

    Lemma 4 at e gives nu_e with #{lambda < e - beta_e} <= nu_e <= #{lambda <= e + beta_e}: the count is exact
    when the collar [e - beta_e, e + beta_e] holds no eigenvalue. Two further counts at sigma_1 = e - delta and
    sigma_2 = e + delta (delta > beta) decide that with the count inequality of Proposition 14 on the window
    (sigma_1, sigma_2): #{lambda in (sigma_1 + beta_1, sigma_2 - beta_2)} <= nu_2 - nu_1. If nu_2 = nu_1 and the
    collar lies inside (sigma_1 + beta_1, sigma_2 - beta_2), the collar is eigenvalue-free, so nu_e = #{lambda < e}
    and e itself is not an eigenvalue. If nu_2 > nu_1, an eigenvalue lies in [sigma_1 - beta_1, sigma_2 + beta_2]
    around the edge and the count at e is not decided (INCONCLUSIVE). delta is widened (at most three retries) when a
    backward error is not smaller than the distance to the collar, as the other S3 paths do.

    Returns (status, record): status 'EXACT' with record {'nu', ...}, or 'NOT_CHECKED' / 'INCONCLUSIVE' with a
    reason. Every count is appended to `counts`."""
    r = inertia_count(A, e, method=opts.inertia_method)
    counts.append(r.to_dict())
    rec = {"edge": float(e)}
    if r.resource_failure:
        return "NOT_CHECKED", dict(rec, reason=f"inertia count not attempted to completion: {r.note}")
    if not r.conclusive:
        return "INCONCLUSIVE", dict(rec, reason=f"inertia inconclusive at the window edge {e:.10g} ({r.note})")
    beta_e = float(add_up(r.beta, eps_shift))
    rec.update(nu=int(r.negative), beta=beta_e)
    delta = max(8.0 * beta_e, 32.0 * U * scale)
    for _ in range(4):
        s1, s2 = float(e - delta), float(e + delta)
        r1 = inertia_count(A, s1, method=opts.inertia_method)
        counts.append(r1.to_dict())
        if r1.resource_failure:
            return "NOT_CHECKED", dict(rec, reason=f"inertia count not attempted to completion: {r1.note}")
        r2 = inertia_count(A, s2, method=opts.inertia_method)
        counts.append(r2.to_dict())
        if r2.resource_failure:
            return "NOT_CHECKED", dict(rec, reason=f"inertia count not attempted to completion: {r2.note}")
        if not (r1.conclusive and r2.conclusive):
            return "INCONCLUSIVE", dict(rec, delta=delta, reason=f"inertia inconclusive next to the window edge {e:.10g}")
        b1, b2 = float(add_up(r1.beta, eps_shift)), float(add_up(r2.beta, eps_shift))
        rec.update(delta=delta, shifts=[s1, s2], nu_below=int(r1.negative), nu_above=int(r2.negative),
                   beta_below=b1, beta_above=b2)
        if add_up(s1, b1) < sub_down(e, beta_e) and sub_down(s2, b2) > add_up(e, beta_e):
            if r2.negative == r1.negative:
                rec.update(exact=True, eigenvalue_free=[float(add_up(s1, b1)), float(sub_down(s2, b2))])
                return "EXACT", rec
            rec.update(exact=False)
            return "INCONCLUSIVE", dict(rec, reason=(f"window edge {e:.10g} is within {delta + max(b1, b2):.2e} of an "
                                                     f"eigenvalue (counts {r1.negative} below {s1:.10g}, {r2.negative} "
                                                     f"below {s2:.10g}): the count at the edge is not decided"))
        delta = max(2.0 * delta, 4.0 * max(b1, b2, beta_e))
    return "INCONCLUSIVE", dict(rec, exact=False,
                                reason=f"factorization backward errors too large next to the window edge {e:.10g}")


def _certify_window_vectorless(A, mu, a, b, opts, eps_shift=0.0):
    """C-02: the number of eigenvalues in (a, b) from inertia counts alone (eigenvalue-only window claim).

    No enclosures exist, so nothing pairs a claimed value with an eigenvalue: the statement is only that the
    window holds exactly `count_in_window` eigenvalues (Lemma 4 at each edge, made exact by the edge check of
    `_edge_count_exact`), compared with the number of claimed values."""
    counts = []
    n_claimed = int(np.asarray(mu).size)
    ev = {"counts": counts, "window": [float(a), float(b)], "vectorless": True, "n_claimed": n_claimed}
    if not (np.isfinite(a) and np.isfinite(b) and a < b):
        return "INCONCLUSIVE", dict(ev, reason=f"window ({a!r}, {b!r}) is not a finite interval with a < b"), None
    scale = max(1.0, abs(float(a)), abs(float(b)))
    edges = {}
    for name, e in (("a", float(a)), ("b", float(b))):
        st, rec = _edge_count_exact(A, e, opts, eps_shift, counts, scale)
        edges[name] = rec
        ev["edges"] = edges
        if st != "EXACT":
            return st, dict(ev, reason=rec.get("reason", "")), None
    n = int(edges["b"]["nu"] - edges["a"]["nu"])
    ev["count_in_window"] = n
    ev["count_statement"] = (f"exactly {n} eigenvalues in ({a:.17g}, {b:.17g}); neither edge is an eigenvalue "
                             f"(eigenvalue-free collars {edges['a']['eigenvalue_free']} and {edges['b']['eigenvalue_free']})")
    ev["pairing"] = "none: no eigenvectors, so no claimed value is matched to an eigenvalue"
    if n == n_claimed:
        return "CERTIFIED", ev, None
    if n > n_claimed:
        ev["missing_at_least"] = int(n - n_claimed)
    else:
        ev["spurious_at_least"] = int(n_claimed - n)
    return "FAILED", ev, None


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def _have_flint() -> bool:
    """Whether python-flint can be imported now (checked at call time so a test can simulate its absence)."""
    try:
        import flint  # noqa: F401
    except Exception:
        return False
    return True


def _exact_statement(claim, spec, mu, X, opts, statements, t0):
    """S6: the claim re-derived in exact arithmetic (checker.exact_recheck_lowest)."""
    sid = "S6.exact_recheck"
    if spec is None:
        return _stmt(sid, "NOT_CHECKED", "needs a model specification", {}, [], time.time() - t0,
                     reason_code="NEEDS_SPEC")
    if X is None:
        return _stmt(sid, "NOT_CHECKED", "eigenvalue-only claim: the exact re-check needs eigenvectors", {}, [],
                     time.time() - t0, reason_code="NEEDS_EIGENVECTORS")
    if claim.kind != "lowest":
        return _stmt(sid, "NOT_CHECKED", f"implemented for 'lowest' claims (this claim: {claim.kind!r})", {}, [],
                     time.time() - t0, reason_code="UNSUPPORTED_CLAIM")
    if claim.sector_isometry is not None:
        return _stmt(sid, "NOT_CHECKED", "symmetry-adapted claims: use exact_recheck_sector_lowest", {}, [],
                     time.time() - t0, reason_code="UNSUPPORTED_CLAIM")
    n = X.shape[0]
    if n > opts.exact_max_dim:
        return _stmt(sid, "NOT_CHECKED", f"dimension {n} above exact_max_dim {opts.exact_max_dim}", {}, [],
                     time.time() - t0, reason_code="SIZE_LIMIT")
    if is_streamed(X):                 # within exact_max_dim: the exact layer works on in-memory arrays
        X = read_columns(X, slice(0, int(X.shape[1])))
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(X))):
        return _stmt(sid, "INCONCLUSIVE", "claim contains non-finite values", {}, [], time.time() - t0)
    if not _have_flint():
        return _stmt(sid, "NOT_CHECKED", "python-flint not installed (pip install 'certify-ed[exact]')",
                     {"missing": "python-flint", "extra": "certify-ed[exact]"}, [], time.time() - t0,
                     reason_code="MISSING_DEPENDENCY")
    from .checker import exact_recheck_lowest
    s3 = next((s for s in statements if s["id"] == "S3.completeness"), None)
    hint = (s3 or {}).get("evidence", {}).get("next_eigenvalue_estimate")
    try:
        r = exact_recheck_lowest(spec, mu, X, claim.states, next_hint=hint, max_dim=opts.exact_max_dim)
    except ImportError as e:
        return _stmt(sid, "NOT_CHECKED", f"python-flint not installed ({' '.join(str(e).split())[:120]})",
                     {"missing": "python-flint", "extra": "certify-ed[exact]"}, [], time.time() - t0,
                     reason_code="MISSING_DEPENDENCY")
    st = r["status"] if r["status"] in ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED") else "INCONCLUSIVE"
    if st == "CERTIFIED":
        lv = [e for e in r["enclosures"] if e["kind"] != "residual interval"]
        ok_txt = (f"{mu.size} lowest eigenvalues re-derived in exact arithmetic: "
                  f"{mu.size - sum(e['eigenvalue_indices'][1] - e['eigenvalue_indices'][0] + 1 for e in lv)} "
                  f"within their residual radii (max {max(r['radii']):.2e})")
        if lv:
            ok_txt += (f", {len(lv)} multiplet(s) located by exact counts (widest level enclosure "
                       f"{max(e['upper'] - e['lower'] for e in lv):.2e})")
        ok_txt += f"; next eigenvalue >= {r['next_lower_bound']:.10g}; no floating-point assumption"
    summ = {"CERTIFIED": ok_txt if st == "CERTIFIED" else "",
            "FAILED": "exact inertia counts contradict the claim",
            "INCONCLUSIVE": r.get("reason", "exact re-check could not separate the claimed values from the rest"),
            "NOT_CHECKED": r.get("reason", "not checked")}[st]
    return _stmt(sid, st, summ, r, ["Krylov-Bogolyubov; Sylvester's law of inertia; Weyl's inequality"],
                 time.time() - t0)


def _private_copies(widened=None):
    """own(m, what): the matrix certify() works on in place of the caller's matrix m.

    A matrix in a narrower floating format than binary64 (float16, float32, complex64) becomes a private exact binary64
    copy (precision.binary64), listed in `widened` under `what`; a wider format is refused (PrecisionError). Integer
    matrices are checked (every value a binary64 number) and used as given, as in 3.x: the steps that multiply them
    convert them exactly, and their certificates are those of 3.x. Boolean matrices (3.x raised a TypeError) are
    converted exactly, not listed. A CSR matrix whose indices are unsorted or repeated becomes a private copy. Any
    other value is returned unchanged.

    A matrix given in several places (the operator also as an observable, say) gets one copy, so that a summation done
    through one place is seen through the others, as it was on the caller's matrix. A new own() for each run."""
    copies = {}

    def own(m, what="matrix"):
        hit = copies.get(id(m))
        if hit is None:
            note = []
            c = binary64(m, what, note, keep_integers=True)   # narrower floats: binary64 copies; integers checked
            if c is m:
                if not (sp.issparse(m) and m.format == "csr") or m.has_canonical_format:
                    return m
                c = m.copy()
                if (c.indices.dtype, c.indptr.dtype) != (m.indices.dtype, m.indptr.dtype):   # copy() may narrow int64
                    c.indices, c.indptr = m.indices.copy(), m.indptr.copy()
            hit = copies[id(m)] = (m, c, note)    # m is kept, so that its id is not reused during the run
        if widened is not None:
            widened.extend(dict(w, input=what) for w in hit[2])
        return hit[1]
    return own


def _as_given(claim: Claim, own, widened=None) -> Claim:
    """The claim with the operator, the sector isometry and the eigenvectors certify() works on: private copies (own)
    of a CSR operator and a CSR sector isometry whose indices are unsorted or repeated, and exact binary64 copies of
    every one of them (and of the vectors) stored in a narrower format.

    Several steps (the S1 comparison, the sector audit, the absolute value taken by the adapter) sum duplicates and
    sort indices in place, through arrays they share with these matrices. On copies the caller's matrices stay as given,
    and certifying the same claim again gives the same certificate. Each copy costs one more matrix in memory.

    Every bound counts binary64 arithmetic (fp.U = 2^-53); numpy would form the products of single-precision data in
    single precision, so such data are widened here, before anything runs (precision.py). The widening is exact: the
    statements concern the values as stored. Claimed eigenvalues are converted with dtype=float as before; a type for
    which that conversion would round is refused."""
    check_values(claim.eigenvalues, "claim.eigenvalues")
    op, iso = own(claim.operator, "claim.operator"), own(claim.sector_isometry, "claim.sector_isometry")
    X = binary64(claim.eigenvectors, "claim.eigenvectors", widened, stream=True, keep_integers=True)
    if op is claim.operator and iso is claim.sector_isometry and X is claim.eigenvectors:
        return claim
    return replace(claim, operator=op, sector_isometry=iso, eigenvectors=X)


def _options_as_given(opts: Options, own) -> Options:
    """Options with private copies (own) of the CSR matrices among the observables and approximate symmetries whose
    indices are unsorted or repeated (the S4 norm bounds and the label records take absolute values of them in place),
    and exact binary64 copies of the observables, approximate symmetries and symmetry diagonals stored in a narrower
    format (precision.py)."""
    def entry(v, what):                    # (approximate) symmetry: (S, lattice[, allowed]) as a sequence
        if sp.issparse(v) or isinstance(v, (str, bytes)):
            return v
        try:
            S = v[0]
        except (TypeError, IndexError, KeyError):
            return v
        S_own = own(S, what)
        if S_own is S:
            return v
        rest = list(v)[1:]
        return [S_own, *rest] if type(v) is list else (S_own, *rest)
    obs = {k: own(v, f"options.observables[{k}]") for k, v in opts.observables.items()}
    asym = {k: entry(v, f"options.approximate_symmetries[{k}]") for k, v in opts.approximate_symmetries.items()}
    syms = {k: entry(v, f"options.symmetries[{k}]") for k, v in opts.symmetries.items()}
    if all(obs[k] is opts.observables[k] for k in obs) and \
            all(asym[k] is opts.approximate_symmetries[k] for k in asym) and \
            all(syms[k] is opts.symmetries[k] for k in syms):
        return opts
    return replace(opts, observables=obs, approximate_symmetries=asym, symmetries=syms)


def _certify_as_given(claim: Claim, spec: Optional[ModelSpec], opts: Options) -> Certificate:
    """_certify on private copies of the caller's non-canonical CSR matrices and on exact binary64 copies of inputs
    stored in a narrower format; the copies go when it returns."""
    widened = []
    own = _private_copies(widened)
    narrow = None
    if opts.plan:                     # the planner sees the binary64 copies only: add the caller's narrower arrays
        from .cost import narrow_input_mb
        narrow = narrow_input_mb(claim, opts)
    return _certify(_as_given(claim, own, widened), spec, _options_as_given(opts, own), widened, narrow_mb=narrow)


def certify(claim: Claim, spec: Optional[ModelSpec] = None, options: Optional[Options] = None) -> Certificate:
    """Certify a claimed spectrum against a declared model: statements S1-S6 (see the module docstring), returned as a Certificate."""
    opts = options or Options()
    if opts.device not in ("cpu", "gpu"):
        raise ValueError(f"Options.device must be 'cpu' or 'gpu', not {opts.device!r}")
    cert = _certify_as_given(claim, spec, opts)
    s2 = cert.statement("S2.eigenpairs")
    rec = None if s2 is None else (s2.get("evidence") or {}).get("device")
    if rec is None:
        return cert
    if rec.get("gpu_products_at_fallback", 0) > 0:
        # D-05: the device failed (a spot check found a product unequal to the CPU product, or CuPy raised) after
        # products had been taken from it. Those products are not trusted, so the whole certificate is computed again
        # with the CPU products; the record keeps the failure.
        cert = _certify_as_given(claim, spec, replace(opts, device="cpu"))
        s2 = cert.statement("S2.eigenpairs")
        if s2 is not None:
            s2["evidence"]["device"] = dict(rec, used=False, recomputed_on_cpu=True)
        return cert
    rec["used"] = int((rec.get("gpu") or {}).get("products", 0)) > 0     # products of this certificate from the GPU
    return cert


def _certify(claim: Claim, spec: Optional[ModelSpec], opts: Options, widened=None, narrow_mb=None) -> Certificate:
    """The body of certify() for one device setting. `widened`: the inputs _as_given converted to binary64 (recorded
    under arithmetic.input_widening when there are any); `narrow_mb`: the caller's arrays in a narrower format, for
    the plan (cost.narrow_input_mb)."""
    t_start = time.time()
    statements = _StatementLog(opts.progress, t_start)
    mu = np.asarray(claim.eigenvalues)
    mu = np.asarray(mu.real if mu.dtype.kind == "c" else mu, dtype=float)    # imaginary parts are 0 (check_values)
    X = claim.eigenvectors
    streamed = X is not None and is_streamed(X)
    if streamed and not isinstance(X, (np.memmap, StreamedColumns)):
        X = StreamedColumns(X)                 # h5py dataset: numpy-style column indexing for every statement
    if X is not None and not streamed:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
    block_mb = opts.block_mb if opts.block_mb is not None else opts.budget_mb
    sel = resolve_selection(opts.preset, opts.only, opts.skip, opts.narrowing_reason,
                            corroborate=bool(opts.corroborate), exact_recheck=bool(opts.exact_recheck),
                            temperatures=bool(len(opts.temperatures)), degeneracy=bool(opts.degeneracy))
    n_claim = (claim.states.size if claim.states is not None else
               (claim.sector_isometry.shape[1] if claim.sector_isometry is not None else
                (spec.dim if spec is not None else (claim.operator.shape[0] if claim.operator is not None else 0))))
    budget_seconds, budget_auto = resolve_budget_seconds(opts.budget_seconds, n_claim)
    budget = Budget(budget_seconds, opts.budget_mb, t_start)
    plan_record = None
    if opts.plan:
        from .cost import plan_for_claim
        plan_ = plan_for_claim(claim, spec, opts, narrow_mb=narrow_mb)
        plan_record = plan_.to_dict()
        sink = (print if opts.progress is True else opts.progress) if opts.progress else print
        for line in plan_.format().splitlines():
            if sink is print:
                print(line, file=sys.stderr, flush=True)
            else:
                sink(line)
    # C-15: observables and correlations given as text, dictionaries or operator lists become sparse operators
    # in the claim space before anything runs (an unreadable or non-Hermitian operator is an error here, not a
    # statement); matrices pass through unchanged
    from .operators import resolve_correlation_request, resolve_observables
    if claim.sector_isometry is not None and any(not (sp.issparse(o) or isinstance(o, np.ndarray))
                                                 for o in opts.observables.values()):
        raise ValueError("observables given as text or operator lists need a computational-basis claim (states= or the "
                         "full space); for a symmetry-adapted block pass the operator as a matrix in the block")
    observables = resolve_observables(spec, opts.observables, claim.states)
    correlations = {name: resolve_correlation_request(spec, name, req) for name, req in opts.correlations.items()}
    is_cplx_claim = (X is not None and np.dtype(X.dtype).kind == "c") or (spec is not None and not spec.is_real())

    # ---------------- S1
    H_full_ref, audit = None, None
    Href = None
    s1_gate = _gate(sel, budget, "S1",
                    estimate_mb=(_mb((claim.states.size if claim.states is not None else spec.dim) * max(1, spec.n_terms)
                                     * 2, 32) if spec is not None else None),
                    what="reference matrix from the spec (estimate: dim x terms x 2 entries)",
                    # 3.2.3: the estimate is a coarse upper bound (the 3.2.2 run refused S1 at 16896 MB for a block the
                    # 3.2.1 run certified at a peak of 2989 MB under a 4608 MB cap), and a reference build that does not
                    # fit ends in a caught MemoryError (NOT_CHECKED); so the process memory cap does not gate S1, only
                    # an explicit budget does
                    use_cap=False)
    if s1_gate is not None and spec is not None:
        statements.append(s1_gate)
    elif spec is not None and opts.model_fidelity:
        t_s1 = time.time()
        try:
            s1, Href_or_full, audit = check_model_fidelity(claim, spec, opts)
        except GuardRangeError as e:
            # 3.2.2 (finding L-3): a bound of the comparison would use a floating-point guard outside its range
            s1 = _stmt("S1.model_fidelity", "INCONCLUSIVE", f"model fidelity bound not formed: {e}",
                       {"spec_sha256": spec.sha256(), "guard_range": {"findings": [e.to_dict()]}},
                       [], time.time() - t_s1, reason_code="FP_GUARD_RANGE")
            Href_or_full, audit = None, None
        except MemoryError as e:
            s1 = _stmt("S1.model_fidelity", "NOT_CHECKED",
                       f"reference build did not fit in memory ({type(e).__name__})",
                       {"spec_sha256": spec.sha256(), "error": " ".join(str(e).split())[:160]}, [],
                       time.time() - t_s1)
            Href_or_full, audit = None, None
        statements.append(s1)
        if claim.sector_isometry is not None:
            H_full_ref = Href_or_full
        else:
            Href = Href_or_full
    else:
        statements.append(_stmt("S1.model_fidelity", "NOT_CHECKED",
                                "no model specification supplied" if spec is None else "disabled",
                                reason_code="NEEDS_SPEC" if spec is None else "NARROWED_BY_USER"))

    s1_status = statements[0]["status"]
    s1_probe = "probe" in statements[0].get("evidence", {})
    op_note = None
    is_matrix = lambda o: sp.issparse(o) or isinstance(o, np.ndarray)
    if claim.operator is None:
        if Href is not None:
            op = Href
        elif spec is not None and opts.matrix_free:
            from .matrixfree import MatrixFreeOperator
            op = MatrixFreeOperator(spec, claim.states)
            op_note = ("operator applied matrix-free from the spec (no stored matrix); rounding counted per row "
                       "from the spec, D-02")
        elif spec is not None:
            op = SpecOperator(spec, claim.states)
            op_note = "operator applied matrix-free from the spec (no stored matrix)"
        else:
            raise ValueError("no operator: supply claim.operator or a spec")
    elif is_matrix(claim.operator) and not s1_probe:
        op = claim.operator
    else:
        # black-box producer matvec (or a stored matrix compared by probes): bound the spec operator
        # instead, if S1 tied them together
        if spec is not None and s1_status == "CERTIFIED":
            op = Href if Href is not None else SpecOperator(spec, claim.states)
            op_note = ("producer operator compared with the spec on random probes (S1); S2-S4 bounds are "
                       "computed for the spec operator" + ("" if Href is not None else ", applied matrix-free"))
        else:
            op = claim.operator
            op_note = "black-box operator without a verified spec: rounding in the producer's matvec is not bounded"
    herm_note = None
    if sp.issparse(op) or isinstance(op, np.ndarray):
        opm = sp.csr_matrix(op) if sp.issparse(op) else np.asarray(op)
        asym = opm - opm.conj().T
        dev = float(abs(asym).max()) if sp.issparse(asym) else float(np.abs(asym).max())
        if dev > 0:
            op = (opm + opm.conj().T) / 2
            herm_note = {"max_abs_asymmetry": dev,
                         "note": "producer matrix not exactly Hermitian; statements concern its Hermitian part"}
    adapter = OperatorAdapter(op, device=opts.device)
    # model shift: bound on ||H_operator_used - H_spec||_2 (exact spec operator), so that S2/S3
    # statements refer to the declared model; 0 when there is no spec or S1 did not certify it
    model_shift = 0.0
    if spec is not None and statements and statements[0]["status"] == "CERTIFIED":
        model_shift = float(statements[0]["evidence"].get("model_shift_bound", 0.0))
    eps_shift = model_shift
    if audit is not None:
        e = _subspace_shift(audit.isometry_defect, audit.invariance_defect,
                            inf_norm_upper(H_full_ref), adapter.norm_upper())
        eps_shift = float("inf") if e is None else up(e + model_shift)

    # ---------------- S2
    t0 = time.time()
    enc = None
    s2_gate = _gate(sel, budget, "S2")
    bad_mu = np.nonzero(~np.isfinite(mu))[0].tolist()
    bad_x = [] if X is None else nonfinite_columns(X, block_mb)
    nonfinite = bool(bad_mu or bad_x)
    # INT-01: the no-underflow hypothesis of the standard model (fp.UNDERFLOW_THRESHOLD) is checked on the
    # operands before S2 and on the radii and residual bounds after the enclosure; a finding makes S2 and
    # every statement that depends on it INCONCLUSIVE with reason code UNDERFLOW_RISK (never CERTIFIED)
    uf_findings = []
    uf_record = {"performed": bool(opts.underflow_check), "threshold": UNDERFLOW_THRESHOLD,
                 "scanned": [], "findings": uf_findings}
    if opts.underflow_check and not nonfinite:
        parts = _underflow_operands(mu, X, adapter, opts.seed)
        if eps_shift > 0 and np.isfinite(eps_shift):
            parts.append(("model_shift", np.array([eps_shift], dtype=float), UNDERFLOW_THRESHOLD))
        uf_record["scanned"] += [p[0] for p in parts]
        uf_findings += underflow_scan(parts)
    underflow = bool(uf_findings)
    # 3.2.2 (roadmap N-01, finding L-3): the guards of Theorem 1 (docs/proofs/thm1_fp-accounting.md, Lambda4 to
    # Lambda6) and of the sparse residual (D-01, H3) are established for dimensions n <= 2^26 and row widths
    # w <= 2^26 only. Above that S2 and every statement that rests on its enclosures or on the factorization bounds
    # are INCONCLUSIVE with reason code FP_GUARD_RANGE (never CERTIFIED); fp.GuardRangeError raised by a guarded
    # function inside the range of n and w (a norm of more than 2^26 entries) is reported the same way.
    gr_findings = guard_range_findings(n=int(adapter.shape[0]), w=adapter.width if adapter.rigorous else None,
                                       where="certify: dimension and row width of the operator")
    if s2_gate is not None:
        statements.append(s2_gate)
    elif nonfinite:
        # IMP-21/IMP-22: NaN passes every comparison written as 'radius > target'; such a claim gets no
        # enclosure at all and no statement that depends on one can be CERTIFIED
        statements.append(_stmt("S2.eigenpairs", "INCONCLUSIVE",
                                f"claim contains non-finite values (eigenvalue indices {bad_mu[:8]}, eigenvector "
                                f"columns {bad_x[:8]}): no enclosure can be formed",
                                {"non_finite_eigenvalues": bad_mu, "non_finite_eigenvector_columns": bad_x},
                                [], time.time() - t0))
    elif X is None:
        statements.append(_stmt("S2.eigenpairs", "NOT_CHECKED",
                                "eigenvalue-only claim: no vectors to bound (see S3 for counts)",
                                reason_code="NEEDS_EIGENVECTORS"))
    else:
        if not gr_findings:
            try:
                enc = enclose(adapter, mu, X, block_mb=block_mb)
            except GuardRangeError as e:
                gr_findings = [e.to_dict()]
    if s2_gate is None and not nonfinite and X is not None and gr_findings:
        statements.append(_stmt("S2.eigenpairs", "INCONCLUSIVE", _guard_range_note(gr_findings),
                                {"n_pairs": int(mu.size), "guard_range": {"findings": gr_findings}},
                                [], time.time() - t0, reason_code="FP_GUARD_RANGE"))
    elif enc is not None:
        ev = {"n_pairs": int(mu.size), "max_radius": float(enc.radius.max()),
              "median_radius": float(np.median(enc.radius)),
              "max_existence_radius": float(enc.existence_radius.max()),
              "groups": [{"members": g.members, "method": g.method, "radius": g.radius,
                          "distinct": g.distinct, "eta": g.eta, "bound": g.bound}
                         for g in enc.groups if g.size > 1],
              "rigorous": enc.rigorous, "sector_lift_shift": eps_shift if audit is not None else None,
              "model_shift": model_shift,
              "radius_refers_to": ("declared model (spec)" if model_shift > 0 or (spec is not None and
                                   statements[0]["status"] == "CERTIFIED") else "producer operator")}
        if not enc.rigorous:
            st, summ = "INCONCLUSIVE", "residuals computed but rounding not boundable for this operator (estimates only)"
        elif not enc.all_distinct:
            st, summ = "INCONCLUSIVE", ("each claimed value has an eigenvalue within its radius, but some "
                                        "claimed pairs could not be matched to distinct eigenvalues "
                                        "(duplicate or non-orthogonal vectors)")
        else:
            st = "CERTIFIED"
            summ = (f"{mu.size} claimed values matched to distinct eigenvalues; max radius "
                    f"{enc.radius.max():.2e}" + (f" (sector block); {enc.radius.max() + eps_shift:.2e} "
                                                 f"for the full operator" if audit is not None else
                                                 (f" (+{model_shift:.1e} model shift)" if model_shift > 0 else "")))
        # accuracy presumption: unless the producer states otherwise, a claimed eigenpair is
        # presumed accurate to DEFAULT_REL_ACCURACY * ||H|| (a true but loose enclosure is FAILED)
        Hn_ = adapter.norm_upper() or 1.0
        if claim.accuracy_target is None:
            target, tsrc = DEFAULT_REL_ACCURACY * max(1.0, Hn_), "default (1e-8 * max(1, ||H||))"
        else:
            target, tsrc = float(claim.accuracy_target), "stated by producer"
        ev["accuracy_target"] = {"value": target, "source": tsrc}
        if st == "CERTIFIED" and not (enc.radius.max() + eps_shift <= target):   # fail-safe for NaN (IMP-22)
            worst = int(np.argmax(enc.radius))
            g = enc.groups[enc.group_of[worst]]
            if not np.isfinite(enc.radius.max() + eps_shift):
                st = "INCONCLUSIVE"
                summ = f"certified radius of pair {worst} is not finite: no accuracy statement"
            elif enc.existence_radius.max() + eps_shift <= target and g.eta is not None:
                # IMP-13: the claim is accurate (every Theorem-1 interval meets the target) but the vectors
                # are too far from orthonormal to certify distinct eigenvalues at that accuracy
                st = "INCONCLUSIVE"
                summ = (f"each claimed value has an eigenvalue within its Theorem-1 radius (max "
                        f"{enc.existence_radius.max() + eps_shift:.2e}, within the accuracy target {target:.1e} "
                        f"[{tsrc}]), but the cluster radius {enc.radius.max() + eps_shift:.2e} (pair {worst}, "
                        f"orthonormality defect eta = {g.eta:.2e}) exceeds it: the pairs are matched to "
                        f"distinct eigenvalues only within the cluster radius")
                ev["diagnosis"] = "accuracy target missed only through the orthonormality defect of the cluster"
            else:
                st = "FAILED"
                summ = (f"certified radius {enc.radius.max() + eps_shift:.2e} (pair {worst}) exceeds the "
                        f"accuracy target {target:.1e} [{tsrc}]")
        # INT-05 (track/backlog-proofs): the radii, existence radii and residual bounds are no longer scanned. The
        # Theorem 1 radius is evaluated under the underflow-tolerant model fl(z) = z(1 + delta) + eta with the
        # additive terms of fp.norm2_upper/norm2_lower and fp.residual_additive_term, so it is an upper bound for
        # every magnitude of the intermediate quantities (docs/proofs/int05_underflow.md); the scan covers the
        # stored inputs and the model shift, on which the remaining standard-model bounds (inertia, Gram,
        # Rayleigh quotients) rest.
        code2 = None
        if underflow:
            st, code2 = "INCONCLUSIVE", "UNDERFLOW_RISK"
            summ = _underflow_statement_note(uf_findings) + "; would otherwise be: " + summ
            ev["underflow"] = {"threshold": UNDERFLOW_THRESHOLD, "findings": uf_findings}
        statements.append(_stmt("S2.eigenpairs", st, summ, ev,
                                ["standard floating-point model (see 'arithmetic')"] if enc.rigorous else [],
                                time.time() - t0, reason_code=code2))

    # ---------------- S3
    t0 = time.time()
    window = None
    inertia_op = op if is_matrix(op) else Href
    # S3/S4 concern the declared model when S1 certified the producer's matrix (model shift added);
    # for a symmetry sector they concern the producer's sector block (see compose.py for the lift)
    eps_s3 = model_shift if audit is None else 0.0
    n_op = int(adapter.shape[0])
    # memory estimate of the factorizations for the budget gate, for the path the planner would take (D-08):
    # dense LDL^T (the 3.1 estimate of 2 n^2 entries) or SuperLU (cost.sparse_inertia_mb from the fill model)
    s3_est = None
    if inertia_op is not None:
        from .cost import recommend_inertia_method, sparse_inertia_mb
        _dense_op = not sp.issparse(inertia_op)
        _meth = (recommend_inertia_method(n_op, None if _dense_op else int(inertia_op.nnz), _dense_op, is_cplx_claim,
                                          opts.budget_mb)[0] if opts.inertia_method == "auto" else opts.inertia_method)
        if _meth == "dense":
            s3_est = _mb(n_op * n_op, (16 if is_cplx_claim else 8) * 2)
        elif not _dense_op:
            s3_est = sparse_inertia_mb(n_op, int(inertia_op.nnz), is_cplx_claim)
    s3_gate = _gate(sel, budget, "S3", estimate_mb=s3_est if claim.kind in ("lowest", "window") else None,
                    what="dense LDL^T factorization (estimate: 2 n^2 entries)" if s3_est is not None and _meth == "dense"
                    else "sparse LU factorization (estimate from the fill model, certify_ed.cost)")
    if s3_gate is not None:
        statements.append(s3_gate)
    elif nonfinite:
        statements.append(_stmt("S3.completeness", "INCONCLUSIVE",
                                "claim contains non-finite values: counts cannot be matched to the claimed set",
                                {"n_claimed": int(mu.size)}))
    elif underflow and enc is not None and claim.kind in ("lowest", "window", "all"):
        statements.append(_stmt("S3.completeness", "INCONCLUSIVE",
                                "S2 inconclusive under UNDERFLOW_RISK: the enclosures and the factorization bounds "
                                "rest on the no-underflow hypothesis, which is not established for this input; "
                                "counts are not matched to the claimed set",
                                {"n_claimed": int(mu.size), "underflow": {"findings": uf_findings}},
                                reason_code="UNDERFLOW_RISK"))
    elif gr_findings and claim.kind in ("lowest", "window", "all"):
        statements.append(_stmt("S3.completeness", "INCONCLUSIVE",
                                "S3 rests on bounds outside the range of the floating-point guards: "
                                + _guard_range_note(gr_findings),
                                {"n_claimed": int(mu.size), "guard_range": {"findings": gr_findings}},
                                reason_code="FP_GUARD_RANGE"))
    elif enc is not None and not np.all(np.isfinite(enc.radius)) and claim.kind in ("lowest", "window", "all"):
        # a zero claimed vector has an infinite Theorem-1 radius: there is no enclosure to count against
        # (the shift sigma = inf would otherwise crash the factorization, IMP-23)
        statements.append(_stmt("S3.completeness", "INCONCLUSIVE",
                                "some claimed pair has no finite enclosure (zero eigenvector): "
                                "counts cannot be matched to the claimed set",
                                {"pairs_without_enclosure": np.nonzero(~np.isfinite(enc.radius))[0].tolist()[:16]}))
    elif claim.kind == "all":
        if enc is not None and enc.all_distinct and mu.size == adapter.shape[0]:
            statements.append(_stmt("S3.completeness", "CERTIFIED",
                                    f"{mu.size} distinct certified eigenvalues = dimension: spectrum complete",
                                    {"dimension": int(adapter.shape[0])}, [], time.time() - t0))
            window = (float("-inf"), float("inf"))
        else:
            statements.append(_stmt("S3.completeness", "INCONCLUSIVE",
                                    "full-spectrum claim but pairs not all certified distinct or count != dimension",
                                    {"n_claimed": int(mu.size), "dimension": int(adapter.shape[0])}))
    elif claim.kind in ("lowest", "window"):
        if not opts.inertia:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED", "inertia counting disabled",
                                    reason_code="NARROWED_BY_USER"))
        elif inertia_op is None:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED",
                                    "no stored matrix to factorize (matrix-free producer without spec)",
                                    reason_code="NEEDS_MATRIX"))
        elif inertia_op.shape[0] > opts.inertia_max_dim:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED",
                                    f"factorization skipped: dimension {inertia_op.shape[0]} > inertia_max_dim "
                                    f"{opts.inertia_max_dim}", {"dimension": int(inertia_op.shape[0])},
                                    reason_code="SIZE_LIMIT"))
        elif enc is None and claim.kind == "window" and underflow:
            statements.append(_stmt("S3.completeness", "INCONCLUSIVE",
                                    "UNDERFLOW_RISK: the factorization bounds rest on the no-underflow hypothesis, "
                                    "which is not established for this operator; no vectorless count",
                                    {"n_claimed": int(mu.size), "underflow": {"findings": uf_findings}},
                                    reason_code="UNDERFLOW_RISK"))
        elif enc is None and claim.kind == "window":
            # C-02: eigenvalue-only window claim; the count in (a, b) from inertia alone, no pairing
            a, b = claim.window
            st, ev, _ = _certify_window_vectorless(inertia_op, mu, a, b, opts, eps_s3)
            if st == "CERTIFIED":
                summ = (f"exactly {ev['count_in_window']} eigenvalues in ({a:.10g}, {b:.10g}) by inertia counts, as "
                        f"many as claimed; no eigenvectors, so no claimed value is paired with an eigenvalue")
            elif st == "FAILED":
                summ = (f"inertia counts exactly {ev['count_in_window']} eigenvalues in ({a:.10g}, {b:.10g}) but the "
                        f"claim lists {mu.size}: " +
                        (f"at least {ev['missing_at_least']} eigenvalue(s) in the window are not claimed"
                         if "missing_at_least" in ev else
                         f"at least {ev['spurious_at_least']} claimed value(s) are not eigenvalues in the window"))
            else:
                summ = ev.get("reason", "counts at the window edges could not be made exact")
            code3 = ("FP_GUARD_RANGE" if st != "CERTIFIED" and any(_GUARD_RANGE_MARK in str(c.get("note", ""))
                                                                    for c in ev.get("counts", [])) else None)
            if st == "NOT_CHECKED":
                # a factorization that was not attempted (memory budget, or since 3.2.2 the process memory cap) or
                # could not be completed (MemoryError): the code is explicit, not read from the wording
                code3 = "BUDGET_MEMORY"
            statements.append(_stmt("S3.completeness", st, summ, ev,
                                    ["Sylvester inertia of a computed LDL^T factorization with explicitly bounded "
                                     "backward error (Lemma 4 at three shifts per window edge; the count inequality "
                                     "of Proposition 14 on the edge collars); standard floating-point model; no "
                                     "eigenvectors: the count is not paired with the claimed values"],
                                    time.time() - t0, reason_code=code3))
        elif enc is None:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED",
                                    "eigenvalue-only claim: completeness needs certified intervals",
                                    reason_code="NEEDS_EIGENVECTORS"))
        else:
            if claim.kind == "lowest":
                st, ev, window = _certify_lowest(inertia_op, np.sort(mu), enc, opts, eps_s3)
                if st == "CERTIFIED":
                    summ = (f"exactly {mu.size} eigenvalues below {ev['next_eigenvalue_lower_bound']:.10g}; "
                            f"certified gap to the next >= {ev['gap_lower_bound']:.3e}")
                elif st == "FAILED":
                    summ = (f"at least {ev['missing_at_least'] if 'missing_at_least' in ev else ev['missing_below_top_claim_at_least']}"
                            f" eigenvalue(s) below the highest claimed value are missing from the claim")
                else:
                    summ = ev.get("reason", "inconclusive")
            else:
                a, b = claim.window
                st, ev, window = _certify_window(inertia_op, enc, a, b, opts, eps_s3)
                summ = {"CERTIFIED": f"claim contains every eigenvalue in ({window[0] if window else a:.10g}, "
                                     f"{window[1] if window else b:.10g})" if st == "CERTIFIED" else "",
                        "FAILED": f"at least {ev.get('missing_at_least')} eigenvalue(s) in the window are not claimed",
                        "INCONCLUSIVE": "counts could not be matched to the claimed set",
                        "NOT_CHECKED": ev.get("reason", "")}[st]
            code3 = ("FP_GUARD_RANGE" if st != "CERTIFIED" and any(_GUARD_RANGE_MARK in str(c.get("note", ""))
                                                                    for c in ev.get("counts", [])) else None)
            if st == "NOT_CHECKED":
                # a factorization that was not attempted (memory budget, or since 3.2.2 the process memory cap) or
                # could not be completed (MemoryError): the code is explicit, not read from the wording
                code3 = "BUDGET_MEMORY"
            statements.append(_stmt("S3.completeness", st, summ, ev,
                                    ["Sylvester inertia of a computed LDL^T factorization with explicitly "
                                     "bounded backward error; standard floating-point model"],
                                    time.time() - t0, reason_code=code3))
    else:
        statements.append(_stmt("S3.completeness", "NOT_CHECKED", "claim kind 'subset' makes no completeness claim",
                                reason_code="NO_COMPLETENESS_CLAIM"))

    # ---------------- gap records (C-01): a presentation of S2 + S3, no statement of its own (gaps.py)
    gaps_record = None
    _s2 = next((s for s in statements if s["id"] == "S2.eigenpairs"), None)
    if _s2 is not None and getattr(adapter, "device_record", None) is not None:
        _s2["evidence"]["device"] = adapter.device_record     # D-05: where H X and |H| |X| were formed
    _s3 = next((s for s in statements if s["id"] == "S3.completeness"), None)
    if (enc is not None and window is not None and _s2 is not None and _s3 is not None
            and _s2["status"] == "CERTIFIED" and _s3["status"] == "CERTIFIED"
            and claim.kind in ("lowest", "window", "all") and np.all(np.isfinite(enc.radius))):
        from .gaps import gap_records
        _rad = enc.radius if eps_s3 == 0.0 else add_up(enc.radius, eps_s3)
        gaps_record = gap_records(mu, _rad, claim.kind, _s3["evidence"],
                                  ("sector block" if audit is not None else
                                   _s2["evidence"].get("radius_refers_to", "producer operator")))
        if gaps_record["top_gap"] is not None:
            _s3["evidence"]["gap_interval"] = list(gaps_record["top_gap"]["interval"])
    # ---------------- Kato-Temple sharpening for isolated pairs (Theorem 3)
    kt = []
    if (enc is not None and window is not None and opts.kato_temple and enc.rigorous
            and enc.all_distinct and audit is None):
        order = np.argsort(enc.mu, kind="stable")
        iso = [int(i) for i in order if enc.groups[enc.group_of[i]].size == 1]
        # fl(H x) and |H| |x| for every isolated pair in one product each (C-17): a column of H X_iso is a valid
        # floating-point evaluation of H x under the same standard-model bound as a single matvec; the Theorem-3
        # interval is the same function of it (enclosure.rayleigh_interval). Formed in column blocks of the
        # memory budget so that the whole-spectrum case does not hold a second n x k array.
        pos_of = {i: p for p, i in enumerate(order)}
        for a0 in range(0, len(iso), max(1, _kt_block(int(adapter.shape[0]), is_cplx_claim, block_mb))):
            chunk = iso[a0:a0 + max(1, _kt_block(int(adapter.shape[0]), is_cplx_claim, block_mb))]
            Xc = read_columns(X, chunk)
            HXc, AXc = adapter.both_products(Xc)
            if HXc.ndim == 1:
                HXc, AXc = HXc[:, None], AXc[:, None]
            for c, i in enumerate(chunk):
                pos = pos_of[i]
                j_lo = order[pos - 1] if pos > 0 else None
                j_hi = order[pos + 1] if pos + 1 < len(order) else None
                # isolation interval for the operator in `adapter`; the S3 window refers to the declared
                # model, so it is shrunk by the model shift
                alpha = float(add_up(enc.mu[j_lo], enc.radius[j_lo]) if j_lo is not None else add_up(window[0], eps_s3))
                beta_ = float(sub_down(enc.mu[j_hi], enc.radius[j_hi]) if j_hi is not None
                              else sub_down(window[1], eps_s3))
                if not np.isfinite(beta_):
                    continue
                res = kato_temple(adapter, Xc[:, c], alpha, beta_, Ax=HXc[:, c], absAx=AXc[:, c])
                if res is not None:
                    lower, upper, th, r = res
                    lower, upper = float(sub_down(lower, eps_s3)), float(add_up(upper, eps_s3))   # declared model
                    if upper - lower < 2 * float(enc.radius[i]):
                        kt.append({"index": int(i), "lower": lower, "upper": upper,
                                   "width": upper - lower, "residual_width": 2 * float(enc.radius[i]),
                                   "isolation_interval": [float(alpha), float(beta_)]})
            del Xc, HXc, AXc
        kt.sort(key=lambda r: r["index"])
    # ---------------- S4 eigenspaces
    t0 = time.time()
    eig_records = []
    s4_gate = _gate(sel, budget, "S4")
    s4_clusters = []
    if s4_gate is not None:
        statements.append(s4_gate)
    elif gr_findings and X is not None and opts.eigenspaces != "none":
        statements.append(_stmt("S4.eigenspaces", "INCONCLUSIVE",
                                "S2 inconclusive under FP_GUARD_RANGE: no certified separations or projector bounds",
                                {"guard_range": {"findings": gr_findings}},
                                reason_code="FP_GUARD_RANGE"))
    elif enc is None or opts.eigenspaces == "none":
        statements.append(_stmt("S4.eigenspaces", "NOT_CHECKED", "not requested or no eigenvectors",
                                reason_code="NEEDS_EIGENVECTORS" if enc is None else "NARROWED_BY_USER"))
    elif underflow:
        statements.append(_stmt("S4.eigenspaces", "INCONCLUSIVE",
                                "S2 inconclusive under UNDERFLOW_RISK: no certified separations or projector bounds",
                                {"underflow": {"findings": uf_findings}}, reason_code="UNDERFLOW_RISK"))
    elif window is None:
        statements.append(_stmt("S4.eigenspaces", "NOT_CHECKED",
                                "requires a certified complete window (S3) to bound separations",
                                reason_code="NEEDS_WHOLE_SPECTRUM"))
    else:
        clusters = cluster_groups(enc, opts.cluster_tol)
        groups = [c for c in clusters if (len(c) > 1 or opts.eigenspaces == "all")]
        n_ok = 0
        rad = enc.radius
        for C in groups:
            rec = None
            try:
                Cset = set(C)
                a_ = min(enc.mu[i] for i in C)
                b_ = max(enc.mu[i] for i in C)
                rec = {"members": C, "claimed": [float(enc.mu[i]) for i in C], "splitting": float(b_ - a_)}
                cb = _cluster_bound(adapter, enc.mu, X, enc.residual, C)
                if cb is None:
                    rec.update(status="INCONCLUSIVE", reason="cluster vectors too far from orthonormal")
                    eig_records.append(rec)
                    continue
                eps, eta, rho, _bound = cb
                if eps_s3 > 0:
                    # statements refer to the declared model: replacing the operator used for the bounds by
                    # H_spec changes the residual by at most shift*||X|| <= shift*sqrt(1+eta), ||A|| by at most
                    # shift, hence rho by at most (sqrt(1+eta) + delta) * shift < 2.5 * shift, and w by at most
                    # shift; epsilon is 1-Lipschitz in w and in rho, so it grows by at most 3.5 * shift.
                    # eps* (Theorem 2') grows by at most (sqrt((1+eta)/(1-eta)) + (1+eta)/(1-eta)) * shift
                    # (||R|| by sqrt(1+eta) shift, ||X^H R|| by (1+eta) shift); min(eps, eps*) by the larger factor
                    lip = max(3.5, math.sqrt((1 + eta) / (1 - eta)) + (1 + eta) / (1 - eta))
                    rho = up(rho + 2.5 * eps_s3)
                    eps = up(eps + lip * eps_s3)
                rad_m = add_up(rad, eps_s3)
                JL = min(sub_down(a_, eps), min(sub_down(enc.mu[i], rad_m[i]) for i in C))
                JU = max(add_up(b_, eps), max(add_up(enc.mu[i], rad_m[i]) for i in C))
                # the other claimed pairs' outward-rounded endpoints, as arrays (C-17: the per-pair Python loops of
                # 3.1 were O(k) numpy-scalar calls per cluster; add_up / sub_down are elementwise, so the values are
                # the same bit for bit)
                mask = np.ones(enc.mu.size, dtype=bool)
                mask[list(Cset)] = False
                mu_o, rad_o = enc.mu[mask], rad_m[mask]
                hi_o, lo_o = add_up(mu_o, rad_o), sub_down(mu_o, rad_o)
                clash = int(np.count_nonzero(~((hi_o < JL) | (lo_o > JU))))
                if clash or not (JL > window[0] and JU < window[1]):
                    rec.update(status="INCONCLUSIVE",
                               reason="cluster interval overlaps other claimed pairs or the window edge")
                    eig_records.append(rec)
                    continue
                U_below = max(float(hi_o[mu_o < a_].max()) if np.any(mu_o < a_) else -np.inf, window[0])
                L_above = min(float(lo_o[mu_o > b_].min()) if np.any(mu_o > b_) else np.inf, window[1])
                # Theorem 4 needs the distance from the Ritz values of range(X), which lie within w <= eps
                # of the claimed values, to the remaining eigenvalues
                dist = float(min(sub_down(a_, U_below), sub_down(L_above, b_)))
                sep = float(sub_down(dist, eps))
                rec["separation_lower_bound"] = float(sep)
                # E-07: the Davis-Kahan sin(theta) bound with the residual ||R||_F / sqrt(1 - eta) and the Ritz-value
                # deviation w' (davis_kahan.py); S4 uses min(Theorem 6, Davis-Kahan), both proven and both recorded
                from .davis_kahan import sin_theta_davis_kahan
                s_dk = sin_theta_davis_kahan(_bound["R_fro"], _bound["XR"], eta, _bound["delta"], _bound["spread"], dist,
                                             model_shift=eps_s3)
                ec = certify_eigenspace(X, C, enc.mu, rho, eta, sep, sin_theta_alt=s_dk)
                if ec is None:
                    rec.update(status="INCONCLUSIVE", reason="residual not small relative to separation")
                    eig_records.append(rec)
                    continue
                for name, O in observables.items():
                    # C-05: a non-Hermitian observable is certified through its Hermitian decomposition
                    from .correlations import certify_observable_general, certify_observable_lifted
                    if (claim.sector_isometry is not None and audit is not None and audit.passed
                            and O.shape[0] == claim.sector_isometry.shape[0] != X.shape[0]):
                        # C-04: an observable given in the full space is lifted through the sector isometry
                        certify_observable_lifted(ec, X, claim.sector_isometry, O, name, audit.isometry_defect,
                                                  audit.invariance_defect)
                        continue
                    r_obs = certify_observable_general(ec, X, O, name)
                    O_spread = O if r_obs.get("hermitian", True) else (sp.csr_matrix(O) + sp.csr_matrix(O).conj().T) / 2
                    ec.observables[name]["per_vector_spread_under_rotation"] = rotation_spread(X, C, O_spread)
                corr_recs = _correlations(ec, X, spec, claim, correlations, rec, audit=audit)
                for name, (sdiag, lattice) in opts.symmetries.items():
                    if isinstance(sdiag, str) and sdiag == "su2":
                        _total_spin(ec, X, name, spec, claim, rec)
                        continue
                    # [A, S] = 0 is needed for the operator the statement refers to: the reference built
                    # from the spec when there is one, else the operator used for the bounds
                    comm_op = Href if Href is not None else (op if is_matrix(op) else None)
                    okc = commutes_diagonal(comm_op, sdiag) if comm_op is not None else False
                    certify_quantum_numbers(ec, X, sp.diags(sdiag), name, lattice=lattice,
                                            commutation_certified=okc)
                    ec.quantum_numbers[name]["commutation_checked_on"] = (
                        "reference matrix built from the spec" if Href is not None else
                        "operator used for the bounds" if comm_op is not None else "not checked")
                if opts.approximate_symmetries:
                    # C-12: labels of approximately conserved operators, hypothesis ||[H, S]|| <= c recorded (davis_kahan.py)
                    from .davis_kahan import labels_hook
                    labels_hook(ec, X, opts.approximate_symmetries, Href if Href is not None else (op if is_matrix(op) else None),
                                (JL, JU))
                r = ec.to_dict()
                r["splitting"] = rec["splitting"]
                r["status"] = "CERTIFIED"
                if corr_recs:
                    r["correlations"] = corr_recs
                if opts.entanglement:
                    # C-03: certified entanglement entropy and spectrum across declared cuts (entanglement.py)
                    from .entanglement import entanglement_records
                    r["entanglement"] = entanglement_records(ec, X, spec, claim, opts.entanglement)
                n_ok += 1
                eig_records.append(r)
                s4_clusters.append(C)
            except GuardRangeError as e:
                # 3.2.2 (finding L-3): a bound of this cluster would use a floating-point guard outside its range
                rec = rec if rec is not None else {"members": C}
                rec.update(status="INCONCLUSIVE", reason=f"bound not formed: {e}", reason_code="FP_GUARD_RANGE")
                if not any(r is rec for r in eig_records):
                    eig_records.append(rec)
        if not groups:
            statements.append(_stmt("S4.eigenspaces", "NOT_CHECKED", "no clusters among the claimed pairs",
                                    {}, [], time.time() - t0, reason_code="NO_CLUSTERS"))
        else:
            st = "CERTIFIED" if n_ok == len(groups) else "INCONCLUSIVE"
            statements.append(_stmt("S4.eigenspaces", st,
                                    f"{n_ok}/{len(groups)} clusters certified at projector level",
                                    {"clusters": eig_records}, ["Davis-Kahan sin(theta) theorem; Ostrowski's theorem"],
                                    time.time() - t0))

    # ---------------- S5 corroboration
    if sel.wants("S5") or opts.corroborate:
        s5_gate = _gate(sel, budget, "S5")
        if s5_gate is not None:
            statements.append(s5_gate)
        elif not opts.corroborate:
            statements.append(_stmt("S5.corroboration", "NOT_CHECKED", "no corroborating solvers named "
                                    "(Options.corroborate)", reason_code="NEEDS_DEPENDENCY"))
        else:
            from .corroborate import corroborate
            t0 = time.time()
            cor = corroborate(op, mu, enc, list(opts.corroborate))
            statements.append(_stmt("S5.corroboration", cor["status"], cor["summary"], cor, [], time.time() - t0))

    # ---------------- S6 exact re-check (no floating-point assumption)
    if sel.wants("S6") or opts.exact_recheck:
        n6 = X.shape[0] if X is not None else n_claim
        s6_gate = _gate(sel, budget, "S6", estimate_mb=_mb(n6 * n6, 48) if n6 else None,
                        what="exact integer matrix and residual (estimate: n^2 x 48 bytes)")
        if s6_gate is not None:
            statements.append(s6_gate)
        else:
            t0 = time.time()
            statements.append(_exact_statement(claim, spec, mu, X, opts, statements, t0))

    # ---------------- S7 thermodynamics from a complete whole-spectrum enclosure (Prop 16, UNVERIFIED)
    spectrum_record = None
    if (sel.wants("S7") or opts.whole_spectrum) and claim.kind == "all" and enc is not None and window is not None:
        from .spectrum import intervals_from_enclosure, merge_intervals
        se = merge_intervals(intervals_from_enclosure(enc, eps_s3 if audit is None else 0.0), int(adapter.shape[0]))
        spectrum_record = {"complete": se.complete, "n_intervals": len(se.intervals), "max_halfwidth": se.max_halfwidth,
                           "intervals": [{"lo": iv.lo, "hi": iv.hi, "count": iv.count, "counted": iv.counted,
                                          "method": iv.method} for iv in se.intervals]}
    # C-06: a 'lowest' claim whose S3 is CERTIFIED gives the k lowest intervals, a lower bound on lambda_{k+1}
    # (window[1]) and the dimension; the omitted weight is bounded by (n - k) exp(-beta L) (Proposition 16')
    truncated_record = None
    if (sel.wants("S7") and claim.kind == "lowest" and enc is not None and window is not None
            and np.isfinite(window[1]) and len(opts.temperatures)):
        from .spectrum import intervals_from_enclosure, merge_intervals
        se_t = merge_intervals(intervals_from_enclosure(enc, eps_s3 if audit is None else 0.0), int(adapter.shape[0]))
        nb = adapter.norm_upper()
        truncated_record = {"intervals": [{"lo": iv.lo, "hi": iv.hi, "count": iv.count} for iv in se_t.intervals],
                            "counted": all(iv.counted for iv in se_t.intervals),
                            "next_lower_bound": float(window[1]),
                            "upper_bound": None if nb is None else float(add_up(nb, eps_s3))}
    if sel.wants("S7"):
        s7_gate = _gate(sel, budget, "S7")
        if s7_gate is not None:
            statements.append(s7_gate)
        elif gr_findings and claim.kind in ("all", "lowest") and X is not None:
            statements.append(_stmt("S7.thermodynamics", "INCONCLUSIVE",
                                    "S2 inconclusive under FP_GUARD_RANGE: no complete enclosure to sum over",
                                    {"guard_range": {"findings": gr_findings}},
                                    reason_code="FP_GUARD_RANGE"))
        elif underflow and claim.kind in ("all", "lowest") and enc is not None:
            statements.append(_stmt("S7.thermodynamics", "INCONCLUSIVE",
                                    "S2 inconclusive under UNDERFLOW_RISK: no complete enclosure to sum over",
                                    {"underflow": {"findings": uf_findings}}, reason_code="UNDERFLOW_RISK"))
        elif claim.kind == "lowest" and not len(opts.temperatures) and enc is not None and window is not None:
            statements.append(_stmt("S7.thermodynamics", "NOT_CHECKED", "no temperatures given (Options.temperatures)",
                                    reason_code="NEEDS_TEMPERATURES"))
        elif claim.kind == "lowest" and truncated_record is not None and truncated_record["counted"]:
            from .thermo import thermodynamics_truncated
            t0 = time.time()
            try:
                th = thermodynamics_truncated(truncated_record["intervals"], truncated_record["next_lower_bound"],
                                              int(adapter.shape[0]), opts.temperatures,
                                              upper_bound=truncated_record["upper_bound"])
                st7 = "CERTIFIED"
                tb = th["tail_bound"]
                summ7 = (f"Z, F, U, C enclosed at {len(th['rows'])} temperature(s) from the {th.get('claimed_count', th['dimension'])} "
                         f"lowest eigenvalues and the tail bound on the {tb['omitted_count']} omitted ones "
                         f"(sum_(j>k) exp(-beta lambda_j) <= (n-k) exp(-beta L), L = {tb.get('next_lower_bound')!r}; "
                         f"method {tb['method']!r}); max C width {max(r['C'][1] - r['C'][0] for r in th['rows']):.2e}")
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                th, st7, summ7 = {"error": str(e)}, "INCONCLUSIVE", f"thermodynamic intervals not formed: {e}"
            statements.append(_stmt("S7.thermodynamics", st7, summ7, th,
                                    ["S2 enclosures and the S3 lower bound on lambda_{k+1} for the operator the "
                                     "statements refer to; the dimension n of the operator; k_B = 1; dyadic interval "
                                     "arithmetic with outward rounding (Lemma 17); Proposition 16' [derived: "
                                     "docs/proofs/C06_truncated_thermodynamics.md; UNVERIFIED by independent "
                                     "re-derivation, D-12]"],
                                    time.time() - t0))
        elif claim.kind != "all" or spectrum_record is None or not spectrum_record["complete"]:
            statements.append(_stmt("S7.thermodynamics", "NOT_CHECKED",
                                    "needs a whole-spectrum claim (kind 'all') whose enclosure is complete (S3), or a "
                                    "'lowest' claim with S3 CERTIFIED (truncated spectrum, C-06)",
                                    reason_code="NEEDS_WHOLE_SPECTRUM"))
        elif not len(opts.temperatures):
            statements.append(_stmt("S7.thermodynamics", "NOT_CHECKED", "no temperatures given (Options.temperatures)",
                                    reason_code="NEEDS_TEMPERATURES"))
        else:
            from .thermo import thermodynamics
            t0 = time.time()
            try:
                th = thermodynamics(spectrum_record["intervals"], opts.temperatures, dimension=int(adapter.shape[0]))
                st7 = "CERTIFIED"
                summ7 = (f"Z, F, U, C enclosed at {len(th['rows'])} temperature(s) from {th['n_intervals']} counted "
                         f"intervals; max C width {max(r['C'][1] - r['C'][0] for r in th['rows']):.2e}")
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                th, st7, summ7 = {"error": str(e)}, "INCONCLUSIVE", f"thermodynamic intervals not formed: {e}"
            statements.append(_stmt("S7.thermodynamics", st7, summ7, th,
                                    ["complete whole-spectrum enclosure (Prop 9) for the operator the S2/S3 "
                                     "statements refer to; k_B = 1; dyadic interval arithmetic with outward rounding "
                                     "(Lemma 17); Proposition 16 [derived: docs/proofs/P16_thermo_intervals.md; UNVERIFIED by "
                                     "independent re-derivation, D-12]"],
                                    time.time() - t0))

    # ---------------- S8 degeneracy counts by bracketing inertia (Prop 14, UNVERIFIED)
    if sel.wants("S8"):
        s8_gate = _gate(sel, budget, "S8", estimate_mb=s3_est,
                        what="dense LDL^T factorizations (estimate: 2 n^2 entries)" if s3_est is not None and _meth == "dense"
                        else "sparse LU factorizations (estimate from the fill model, certify_ed.cost)")
        if s8_gate is not None:
            statements.append(s8_gate)
        elif gr_findings and X is not None:
            statements.append(_stmt("S8.degeneracy", "INCONCLUSIVE",
                                    "S2 inconclusive under FP_GUARD_RANGE: no level enclosures to count against",
                                    {"guard_range": {"findings": gr_findings}},
                                    reason_code="FP_GUARD_RANGE"))
        elif enc is None:
            statements.append(_stmt("S8.degeneracy", "NOT_CHECKED", "eigenvalue-only claim: no clusters to count",
                                    reason_code="NEEDS_EIGENVECTORS"))
        elif underflow:
            statements.append(_stmt("S8.degeneracy", "INCONCLUSIVE",
                                    "S2 inconclusive under UNDERFLOW_RISK: no level enclosures to count against",
                                    {"underflow": {"findings": uf_findings}}, reason_code="UNDERFLOW_RISK"))
        elif inertia_op is None:
            statements.append(_stmt("S8.degeneracy", "NOT_CHECKED",
                                    "no stored matrix to factorize (matrix-free producer without spec)",
                                    reason_code="NEEDS_MATRIX"))
        elif inertia_op.shape[0] > opts.inertia_max_dim:
            statements.append(_stmt("S8.degeneracy", "NOT_CHECKED",
                                    f"factorization skipped: dimension {inertia_op.shape[0]} > inertia_max_dim "
                                    f"{opts.inertia_max_dim}", reason_code="SIZE_LIMIT"))
        else:
            from .degeneracy import degeneracy_counts
            t0 = time.time()
            cl = [c for c in cluster_groups(enc, opts.cluster_tol) if len(c) > 1 or opts.eigenspaces == "all"]
            if not cl:
                statements.append(_stmt("S8.degeneracy", "NOT_CHECKED", "no clusters among the claimed pairs", {}, [],
                                        time.time() - t0, reason_code="NO_CLUSTERS"))
            else:
                recs = degeneracy_counts(inertia_op, enc, cl, method=opts.inertia_method, eps_shift=eps_s3, budget_mb=opts.budget_mb)
                n_cert = sum(1 for r in recs if r["status"] == "CERTIFIED")
                n_fail = sum(1 for r in recs if r["status"] == "FAILED")
                st8 = "CERTIFIED" if n_cert == len(recs) else ("FAILED" if n_fail else "INCONCLUSIVE")
                assumptions8 = ["Lemma 4 at two shifts; Theorem 2 distinctness; Proposition 14 "
                                "[derived: docs/proofs/P14_degeneracy_bracketing.md; UNVERIFIED by "
                                "independent re-derivation, D-12]"]
                summ8 = (f"{n_cert}/{len(recs)} clusters: eigenvalue count in the level enclosure certified "
                         f"by bracketing inertia counts" + (f"; degeneracies {[r.get('degeneracy') for r in recs]}"
                                                            if st8 == "CERTIFIED" else ""))
                # C-16: exact count and multiplicities in the inner bracket, no floating-point assumption
                # (Options.exact_recheck, spec declared, python-flint, plain or U(1)-block claims)
                exact_note = None
                if opts.exact_recheck and spec is not None and claim.sector_isometry is None and any("bracket_inner" in r for r in recs):
                    if not _have_flint():
                        exact_note = {"status": "NOT_CHECKED", "reason": "python-flint not installed", "reason_code": "MISSING_DEPENDENCY"}
                    else:
                        from .checker import exact_matrix, is_hermitian_exact
                        from .exact_degeneracy import MAX_DIM, exact_brackets_for_clusters
                        n_ex = claim.states.size if claim.states is not None else spec.dim
                        if n_ex > min(opts.exact_max_dim, MAX_DIM):
                            exact_note = {"status": "NOT_CHECKED", "reason_code": "SIZE_LIMIT",
                                          "reason": f"dimension {n_ex} above the exact bracket cap {min(opts.exact_max_dim, MAX_DIM)}"}
                        else:
                            ex_m = exact_matrix(spec, claim.states)
                            if not is_hermitian_exact(ex_m):
                                exact_note = {"status": "FAILED", "reason": "declared model is not Hermitian"}
                            else:
                                xb = exact_brackets_for_clusters(ex_m, recs)
                                for r, x in zip(recs, xb):
                                    if x is not None:
                                        r["exact_bracket"] = x
                                n_eq = sum(1 for x in xb if x and x.get("status") == "CERTIFIED" and x.get("equal"))
                                n_neq = sum(1 for x in xb if x and x.get("status") == "CERTIFIED" and x.get("equal") is False)
                                n_xf = sum(1 for x in xb if x and x.get("status") == "FAILED")
                                if n_xf:
                                    st8 = "FAILED"
                                summ8 += (f"; exact brackets (C-16): {n_eq} cluster(s) exactly degenerate, {n_neq} not "
                                          f"degenerate, {n_xf} contradicted")
                                assumptions8.append("exact bracket (C-16): characteristic polynomial of the exact declared "
                                                    "model, Sturm counts; no floating-point assumption "
                                                    "[derived: docs/proofs/C16_exact_bracket.md; UNVERIFIED by "
                                                    "independent re-derivation, D-12]")
                ev8 = {"clusters": recs}
                if exact_note is not None:
                    ev8["exact_bracket"] = exact_note
                statements.append(_stmt("S8.degeneracy", st8, summ8, ev8, assumptions8, time.time() - t0))

    # ---------------- assemble
    subject = {"description": (spec.name if spec else "operator supplied without spec"),
               "spec": spec.to_dict() if spec else None,
               "spec_sha256": spec.sha256() if spec else None,
               "operator": {"kind": adapter.kind, "dimension": int(adapter.shape[0]),
                            "matrix_sha256": matrix_sha256(op) if is_matrix(op) else None,
                            "norm_upper_bound": adapter.norm_upper()},
               "sector": None if claim.sector_isometry is None else {
                   "isometry_sha256": matrix_sha256(claim.sector_isometry),
                   "dimension": int(claim.sector_isometry.shape[1]),
                   "lift_shift": eps_shift}}
    claim_rec = {"description": claim.description or f"{claim.kind} claim, {mu.size} eigenvalues",
                 "kind": claim.kind, "window": None if claim.window is None else [float(claim.window[0]), float(claim.window[1])],
                 "n_eigenvalues": int(mu.size),
                 "has_eigenvectors": X is not None, "producer": claim.producer,
                 "accuracy_target": claim.accuracy_target,
                 "eigenvalues": mu.tolist(),
                 "eigenvectors_sha256": array_sha256(X) if X is not None else None}
    not_verified = [
        "that the model specification is the Hamiltonian the authors intend (only producer operator == spec is checked)",
        "correctness of the producer's code in general (only this result is checked)",
    ]
    s6_ok = any(s["id"] == "S6.exact_recheck" and s["status"] == "CERTIFIED" for s in statements)
    if enc is not None and enc.rigorous and not s6_ok:
        not_verified.append("interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')")
    if claim.kind == "subset":
        not_verified.append("completeness: no claim that other eigenvalues are absent")
    elif window is not None:
        not_verified.append(f"anything outside the certified window {list(window)}")
    if opts.eigenspaces == "clusters":
        not_verified.append("eigenvector subspaces of isolated (non-cluster) pairs beyond their residuals")
    fp_probe = probe_float_environment()
    if not fp_probe["standard_model_hypotheses_hold"]:
        why = ("the floating-point environment of this process violates the standard model "
               f"(failed probes: {', '.join(fp_probe['failed'])}); bounds computed under it are not certified")
        for s in statements:
            if s["status"] == "CERTIFIED" and s["id"] != "S6.exact_recheck":   # S6 uses no floating point
                s["status"] = "INCONCLUSIVE"
                s["summary"] = why + "; would otherwise be: " + s["summary"]
        not_verified.append("bounds: " + why)
    arithmetic = float_env_record()
    arithmetic["runtime_probe"] = fp_probe
    arithmetic["underflow_check"] = uf_record
    widening = widening_record(widened or [])
    if widening is not None:                  # only when an input was converted: binary64 inputs keep their record
        arithmetic["input_widening"] = widening
    annotate_next_actions(statements, {"kind": claim.kind, "has_eigenvectors": X is not None, "has_spec": spec is not None,
                                       "eigenspaces": opts.eigenspaces, "budget_seconds": budget_seconds})
    body = {"subject": subject, "claim": claim_rec, "statements": list(statements),
            "enclosures": enc.to_records() if enc is not None else [],
            "kato_temple": kt,
            "arithmetic": arithmetic, "environment": environment_record(),
            "hermitian_part": herm_note,
            "operator_note": op_note,
            "spectrum_enclosure": spectrum_record,
            "gaps": gaps_record,
            "plan": plan_record,
            "scope": build_scope_block(
                sel, statements, budget,
                {"preset": opts.preset, "only": None if opts.only is None else list(opts.only), "skip": list(opts.skip),
                 "budget_seconds": budget_seconds, "budget_mb": opts.budget_mb, "budget_auto": budget_auto,
                 "temperatures": [float(t) for t in opts.temperatures], "eigenspaces": opts.eigenspaces,
                 "observables": sorted(opts.observables), "correlations": sorted(opts.correlations),
                 "symmetries": sorted(opts.symmetries), "corroborate": list(opts.corroborate),
                 "exact_recheck": bool(opts.exact_recheck), "degeneracy": bool(opts.degeneracy),
                 "underflow_check": bool(opts.underflow_check), "entanglement": sorted(opts.entanglement),
                 "approximate_symmetries": sorted(opts.approximate_symmetries)},
                ("sector block: S2-S4 concern the producer's sector matrix; S2 radii in 'arrays' "
                 "include the certified shift to the full operator") if audit is not None else "operator",
                requested={"kind": claim.kind, "has_eigenvectors": X is not None, "has_spec": spec is not None,
                           "has_matrix": is_matrix(op)}),
            "not_verified": not_verified,
            "timing": {"total_seconds": time.time() - t_start,
                       "per_statement": {s_["id"]: s_["seconds"] for s_ in statements}}}
    arrays = {"eigenvalues": mu}
    if enc is not None:
        # INT-03: the exported radii are rounded upward like the statements' (add_up), never to nearest
        _shift = eps_shift if np.isfinite(eps_shift) else 0.0
        if _shift == 0.0:                         # adding zero is exact
            arrays["radius"] = np.array(enc.radius, dtype=float)
            arrays["existence_radius"] = np.array(enc.existence_radius, dtype=float)
        else:
            arrays["radius"] = np.asarray(add_up(enc.radius, _shift), dtype=float)
            arrays["existence_radius"] = np.asarray(add_up(enc.existence_radius, _shift), dtype=float)
        arrays["residual_bound"] = enc.residual.upper
    if opts.store_eigenvectors and X is not None:
        arrays["eigenvectors"] = np.asarray(X) if not is_streamed(X) else read_columns(X, slice(0, int(X.shape[1])))
    if getattr(adapter, "_gpu", None) is not None:      # D-05: free the device copy of H for the next certificate
        try:
            adapter._gpu.close()
        except Exception:
            pass
        adapter._gpu = None
    return Certificate(body, arrays)


def _total_spin(ec, X, name, spec, claim, rec):
    """Certified total-spin content of a cluster: eigenvalues of X^+ S^2 X snapped to S(S+1).
    [H, S^2] = 0 for the declared model is established structurally (certify_ed.su2)."""
    from .su2 import allowed_casimir_values, casimir_spec, su2_structural
    if spec is None or claim.sector_isometry is not None:
        ec.quantum_numbers[name] = {"status": "NOT-CHECKED",
                                    "reason": "needs a specification and a computational-basis claim"}
        return
    ok, why = su2_structural(spec)
    if not ok:
        ec.quantum_numbers[name] = {"status": "NOT-CHECKED", "reason": "SU(2) invariance not established: " + why}
        return
    states = claim.states if claim.states is not None else full_state_set(spec)
    S2 = build_bitwise(casimir_spec(spec), states)
    av = allowed_casimir_values(spec)
    r = certify_quantum_numbers(ec, X, S2, name, commutation_certified=True, allowed_values=av)
    r["commutation_checked_on"] = "structural SU(2) invariance of the specification: " + why
    if "certified_values" in r:
        r["total_spin"] = [float((-1 + np.sqrt(1 + 4 * v)) / 2) for v in r["certified_values"]]


def _correlations(ec, X, spec, claim, requests, rec, audit=None):
    """S4 extension: two-point functions and structure factors requested in Options.correlations (resolved to
    dictionaries by operators.resolve_correlation_request; correlations.py, Proposition 15). Records are
    returned for the cluster; errors become INCONCLUSIVE records rather than exceptions. For a symmetry-adapted
    claim (sector_isometry) the site operators are built in the full space and lifted through the isometry
    (C-04, certify_correlations_lifted) using S1's audit (isometry and invariance defects); without that audit
    the records are NOT_CHECKED, and when the audit failed they are INCONCLUSIVE."""
    if not requests:
        return {}
    out = {}
    if spec is None:
        for name in requests:
            out[name] = {"status": "NOT_CHECKED", "reason": "needs a model specification",
                         "reason_code": "NEEDS_SPEC"}
        return out
    lifted = claim.sector_isometry is not None
    if lifted and audit is None:
        for name in requests:
            out[name] = {"status": "NOT_CHECKED", "reason_code": "DEPENDENCY_NARROWED",
                         "reason": ("symmetry-adapted claim: the lift of tr(P O) through the sector isometry needs S1's "
                                    "sector audit (isometry and invariance defects); S1 did not run or produced no audit")}
        return out
    if lifted and not audit.passed:
        for name in requests:
            out[name] = {"status": "INCONCLUSIVE",
                         "reason": ("symmetry-adapted claim whose S1 sector audit failed: the block is not certified to be an "
                                    "invariant orthonormal restriction of the declared model, so the lifted quantity is not "
                                    "certified"), "isometry_defect": float(audit.isometry_defect),
                         "invariance_defect": float(audit.invariance_defect)}
        return out
    from .correlations import certify_correlations, certify_correlations_lifted, structure_factor
    for name, req in requests.items():
        try:
            letter = req.get("letter", "z")
            sites = list(req.get("sites", range(spec.n_sites)))
            if lifted:
                cr = certify_correlations_lifted(ec, X, spec, claim.sector_isometry, letter, sites,
                                                 audit.isometry_defect, audit.invariance_defect, name=name)
            else:
                cr = certify_correlations(ec, X, spec, letter, sites, states=claim.states, name=name)
            cr["status"] = "CERTIFIED"
            if req.get("q") is not None:
                pos = req.get("positions") or [[float(i)] for i in sites]
                cr["structure_factor"] = structure_factor(cr, pos, req["q"], n_sites=req.get("n_sites"))
            out[name] = cr
        except ValueError as e:
            out[name] = {"status": "INCONCLUSIVE", "reason": str(e)}
    return out

