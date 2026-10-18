"""Declarative model specifications (the "what was meant" side of a certificate).

A `ModelSpec` states the Hamiltonian as a sum of operator strings on a finite
set of sites, using the same operator-string format as QuSpin's static lists:

    terms = [["zz", [[J, 0, 1], [J, 1, 2], ...]],
             ["x",  [[h, 0], [h, 1], ...]]]

Each coupling [c, i_1, ..., i_m] contributes c * o_1(i_1) o_2(i_2) ... o_m(i_m)
with the rightmost factor acting first. The spec also fixes the integer
encoding of basis states so that matrices built from it can be compared with a
producer's matrix entry by entry:

    site_order  = "site0_msb"   site 0 is the most significant base-q digit
    state_order = "descending"  basis index 0 is the largest integer (QuSpin)
                  "ascending"   basis index 0 is the all-zero-digit state

The spec is serialised canonically and hashed (SHA-256) so that a certificate
binds to exactly one model definition.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


from .localops import LocalSpace

SPEC_SCHEMA = "certify-ed/model-spec/1.0"


def _coeff_to_json(c) -> Any:
    c = complex(c)
    if c.imag == 0.0:
        return float(c.real)
    return [float(c.real), float(c.imag)]


def _coeff_from_json(c) -> complex:
    if isinstance(c, (list, tuple)):
        return complex(float(c[0]), float(c[1]))
    return complex(float(c))


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, fixed separators, shortest round-trip floats."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


@dataclass
class Term:
    op: str
    couplings: List[Tuple[complex, Tuple[int, ...]]]

    def to_json(self):
        return [self.op, [[_coeff_to_json(c)] + list(s) for c, s in self.couplings]]


@dataclass
class ModelSpec:
    n_sites: int
    local: LocalSpace
    terms: List[Term]
    site_order: str = "site0_msb"
    state_order: str = "descending"
    name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # ----------------------------------------------------------- construction
    @staticmethod
    def from_static(n_sites: int, local: LocalSpace, static: Sequence, **kw) -> "ModelSpec":
        terms = []
        for entry in static:
            op, coupling_list = entry[0], entry[1]
            cps = []
            for cp in coupling_list:
                c = _coeff_from_json(cp[0]) if isinstance(cp[0], (list, tuple)) else complex(cp[0])
                sites = tuple(int(s) for s in cp[1:])
                cps.append((c, sites))
            terms.append(Term(str(op), cps))
        spec = ModelSpec(n_sites, local, terms, **kw)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.site_order != "site0_msb":
            raise ValueError("only site_order='site0_msb' is implemented")
        if self.state_order not in ("descending", "ascending"):
            raise ValueError("state_order must be 'descending' or 'ascending'")
        allowed = set(self.local.letters)
        for t in self.terms:
            bad = set(t.op) - allowed
            if bad:
                raise ValueError(f"letters {sorted(bad)} not valid for {self.local.kind} space")
            for _c, s in t.couplings:
                if len(s) != len(t.op):
                    raise ValueError(f"term {t.op!r}: coupling {s} has wrong number of sites")
                if any(not (0 <= i < self.n_sites) for i in s):
                    raise ValueError(f"term {t.op!r}: site out of range in {s}")

    # ------------------------------------------------------------- properties
    @property
    def q(self) -> int:
        return self.local.q

    @property
    def dim(self) -> int:
        return self.q ** self.n_sites

    @property
    def n_terms(self) -> int:
        return sum(len(t.couplings) for t in self.terms)

    def is_real(self) -> bool:
        """True if all coefficients are real and no letter has imaginary matrix elements."""
        coeff_real = all(complex(c).imag == 0 for t in self.terms for c, _ in t.couplings)
        # 'y' has imaginary elements; an even number of y's per string is real
        strings_real = all(t.op.count("y") % 2 == 0 for t in self.terms)
        return coeff_real and strings_real

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "schema": SPEC_SCHEMA,
            "n_sites": self.n_sites,
            "local_space": self.local.to_dict(),
            "encoding": {"site_order": self.site_order, "state_order": self.state_order,
                         "base": self.q},
            "terms": [t.to_json() for t in self.terms],
            "name": self.name,
            "parameters": self.parameters,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "ModelSpec":
        if d.get("schema") != SPEC_SCHEMA:
            raise ValueError(f"unsupported spec schema {d.get('schema')!r}")
        local = LocalSpace.from_dict(d["local_space"])
        enc = d.get("encoding", {})
        terms = []
        for op, cps in d["terms"]:
            terms.append(Term(op, [(_coeff_from_json(cp[0]), tuple(int(s) for s in cp[1:]))
                                   for cp in cps]))
        spec = ModelSpec(int(d["n_sites"]), local, terms,
                         site_order=enc.get("site_order", "site0_msb"),
                         state_order=enc.get("state_order", "descending"),
                         name=d.get("name", ""), parameters=d.get("parameters", {}),
                         notes=d.get("notes", ""))
        spec.validate()
        return spec

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def to_quspin_static(self) -> list:
        return [[t.op, [[c if complex(c).imag else complex(c).real] + list(s)
                        for c, s in t.couplings]] for t in self.terms]

    def with_state_order(self, order: str) -> "ModelSpec":
        s = copy.deepcopy(self)
        s.state_order = order
        s.validate()
        return s


# ---------------------------------------------------------------------------
# QuSpin interoperability
# ---------------------------------------------------------------------------

def local_space_from_quspin_basis(basis) -> LocalSpace:
    """Infer the LocalSpace (and convention) of a QuSpin 1-d basis object."""
    cls = type(basis).__name__
    if "spin" in cls and "fermion" not in cls:
        sps = int(basis.sps)
        S = f"{sps - 1}/2" if (sps - 1) % 2 else str((sps - 1) // 2)
        pauli = getattr(basis, "_pauli", None)
        if pauli is None:
            pauli = getattr(basis, "pauli", 0)
        conv = {0: "spin", 1: "pauli", -1: "pauli_pm1"}[int(pauli)] if sps == 2 else "spin"
        return LocalSpace.spin_space(S, conv)
    if "spinless_fermion" in cls:
        return LocalSpace.fermion_space()
    if "boson" in cls:
        return LocalSpace.boson_space(int(basis.sps) - 1)
    raise NotImplementedError(f"QuSpin basis type {cls} not supported by the importer")


def spec_from_quspin(static: Sequence, basis, n_sites: int = None, **kw) -> ModelSpec:
    """Build a ModelSpec from a QuSpin static list and basis (full-space semantics).

    Only the local space and the operator strings are taken from QuSpin; the
    matrix is then rebuilt independently by CERTIFY-ED. Symmetry blocks of the
    QuSpin basis are audited separately (see certify_ed.sectors).
    """
    N = n_sites if n_sites is not None else int(basis.L if hasattr(basis, "L") else basis.N)
    local = local_space_from_quspin_basis(basis)
    return ModelSpec.from_static(N, local, static, **kw)


# ---------------------------------------------------------------------------
# Spec factories for common models (declarative; used by examples/experiments)
# ---------------------------------------------------------------------------

def _bonds_1d(N: int, r: int, boundary: str):
    bonds = [(i, i + r) for i in range(N - r)]
    if boundary == "periodic":
        bonds += [(i, (i + r) % N) for i in range(N - r, N)]
    return bonds


def heisenberg_chain(N: int, J: float = 1.0, J2: float = 0.0, Delta: float = 1.0,
                     boundary: str = "periodic", convention: str = "spin") -> ModelSpec:
    """H = sum_<ij> J [ (S+_i S-_j + h.c.)/2 + Delta Sz_i Sz_j ] (+ J2 next-nearest).

    In the 'spin' convention this is J sum S_i.S_j at Delta=1.
    """
    local = LocalSpace.spin_space("1/2", convention)
    pm_scale = {"spin": 0.5, "pauli": 0.125, "pauli_pm1": 0.5}[convention]
    zz_scale = {"spin": 1.0, "pauli": 0.25, "pauli_pm1": 0.25}[convention]
    static = []
    for Jr, r in ((J, 1), (J2, 2)):
        if Jr == 0.0:
            continue
        bonds = _bonds_1d(N, r, boundary)
        static.append(["+-", [[Jr * pm_scale, i, j] for i, j in bonds]])
        static.append(["-+", [[Jr * pm_scale, i, j] for i, j in bonds]])
        static.append(["zz", [[Jr * Delta * zz_scale, i, j] for i, j in bonds]])
    return ModelSpec.from_static(N, local, static, name="heisenberg_chain",
                                 parameters=dict(J=J, J2=J2, Delta=Delta, boundary=boundary,
                                                 convention=convention,
                                                 physical_form="J sum S_i.S_j (+J2 nnn), S=sigma/2"))


def tfim_chain(N: int, J: float = 1.0, h: float = 1.0, boundary: str = "periodic") -> ModelSpec:
    """H = -J sum sigma^z_i sigma^z_{i+1} - h sum sigma^x_i (Pauli convention)."""
    local = LocalSpace.spin_space("1/2", "pauli")
    bonds = _bonds_1d(N, 1, boundary)
    static = [["zz", [[-J, i, j] for i, j in bonds]], ["x", [[-h, i] for i in range(N)]]]
    return ModelSpec.from_static(N, local, static, name="tfim_chain",
                                 parameters=dict(J=J, h=h, boundary=boundary,
                                                 physical_form="-J sum s^z s^z - h sum s^x (Pauli)"))


def heisenberg_square(Lx: int, Ly: int, J: float = 1.0, J2: float = 0.0,
                      boundary: str = "periodic") -> ModelSpec:
    """H = J sum_<ij> S_i.S_j (+ J2 on next-nearest neighbours) on an Lx x Ly square lattice, sites y*Lx + x."""
    local = LocalSpace.spin_space("1/2", "spin")
    N = Lx * Ly
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)
    nn, nnn = set(), set()
    for x in range(Lx):
        for y in range(Ly):
            for dx, dy, S in ((1, 0, nn), (0, 1, nn), (1, 1, nnn), (1, -1, nnn)):
                xx, yy = x + dx, y + dy
                if boundary == "open" and not (0 <= xx < Lx and 0 <= yy < Ly):
                    continue
                a, b = idx(x, y), idx(xx, yy)
                if a != b:
                    S.add((min(a, b), max(a, b)))
    static = []
    for Jr, bonds in ((J, sorted(nn)), (J2, sorted(nnn))):
        if Jr == 0.0:
            continue
        static.append(["+-", [[0.5 * Jr, i, j] for i, j in bonds]])
        static.append(["-+", [[0.5 * Jr, i, j] for i, j in bonds]])
        static.append(["zz", [[Jr, i, j] for i, j in bonds]])
    return ModelSpec.from_static(N, local, static, name="heisenberg_square",
                                 parameters=dict(Lx=Lx, Ly=Ly, J=J, J2=J2, boundary=boundary))


def spinless_fermion_chain(N: int, t: float = 1.0, V: float = 0.0, mu: float = 0.0,
                           boundary: str = "open") -> ModelSpec:
    """H = -t sum (c+_i c_j + h.c.) + V sum n_i n_j - mu sum n_i."""
    local = LocalSpace.fermion_space()
    bonds = _bonds_1d(N, 1, boundary)
    static = [["+-", [[-t, i, j] for i, j in bonds]], ["-+", [[t, i, j] for i, j in bonds]]]
    if V:
        static.append(["nn", [[V, i, j] for i, j in bonds]])
    if mu:
        static.append(["n", [[-mu, i] for i in range(N)]])
    return ModelSpec.from_static(N, local, static, name="spinless_fermion_chain",
                                 parameters=dict(t=t, V=V, mu=mu, boundary=boundary))


def bose_hubbard_chain(N: int, n_max: int, t: float = 1.0, U: float = 1.0,
                       boundary: str = "open") -> ModelSpec:
    """H = -t sum (b+_i b_j + h.c.) + U/2 sum n_i (n_i - 1), bosons with cutoff n_max."""
    local = LocalSpace.boson_space(n_max)
    bonds = _bonds_1d(N, 1, boundary)
    static = [["+-", [[-t, i, j] for i, j in bonds]], ["-+", [[-t, i, j] for i, j in bonds]],
              ["nn", [[0.5 * U, i, i] for i in range(N)]], ["n", [[-0.5 * U, i] for i in range(N)]]]
    return ModelSpec.from_static(N, local, static, name="bose_hubbard_chain",
                                 parameters=dict(n_max=n_max, t=t, U=U, boundary=boundary))


def spin1_heisenberg_chain(N: int, J: float = 1.0, D: float = 0.0,
                           boundary: str = "periodic") -> ModelSpec:
    """Spin-1 chain H = J sum S_i.S_{i+1} + D sum (S^z_i)^2."""
    local = LocalSpace.spin_space("1", "spin")
    bonds = _bonds_1d(N, 1, boundary)
    static = [["+-", [[0.5 * J, i, j] for i, j in bonds]], ["-+", [[0.5 * J, i, j] for i, j in bonds]],
              ["zz", [[J, i, j] for i, j in bonds]]]
    if D:
        static.append(["zz", [[D, i, i] for i in range(N)]])
    return ModelSpec.from_static(N, local, static, name="spin1_heisenberg_chain",
                                 parameters=dict(J=J, D=D, boundary=boundary))
