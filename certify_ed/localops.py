"""Local Hilbert spaces and single-site operators with QuSpin-compatible letters.

A local space fixes (i) the local dimension q, (ii) the meaning of each digit
0..q-1 of the integer encoding of basis states, and (iii) the matrix of every
operator letter. The conventions below reproduce QuSpin 1.0 exactly
(verified in tests/v2/test_construction.py and experiments/e0_verifier_validation.py):

  spin S          digit k  <->  m = k - S          letters I x y z + -
      convention "spin"     : S^a operators (QuSpin pauli=0)
      convention "pauli"    : sigma^a = 2 S^a for a in {x,y,z,+,-} (QuSpin pauli=1,
                              so sigma^+ has matrix element 2)
      convention "pauli_pm1": sigma^{x,y,z} = 2 S^{x,y,z}, sigma^+- = S^+- (QuSpin pauli=-1)
  fermion         digit n  = occupation 0/1     letters I + - n z   (z = n - 1/2)
      '+' and '-' carry a Jordan-Wigner sign (-1)^(number of occupied sites on
      one side of the site), evaluated on the state the operator acts on.
      jw_string = "higher" (default, QuSpin 1.0): sites with larger index;
      jw_string = "lower": sites with smaller index (c_j = prod_{i<j}(-1)^{n_i} a_j).
      Number-conserving strings (hopping, density, interactions) have identical
      matrices in both conventions. Strings that change the particle number differ:
      pairing terms c+c+ change sign, odd strings differ by the total parity.
  boson (n_max)   digit n  = occupation 0..n_max letters I + - n

Every operator letter is decomposed into "shift components": o = sum_delta
D_delta S^delta, where S^delta raises the digit by delta and D_delta is diagonal.
The bitwise builder uses this decomposition; the Kronecker builder uses the
dense local matrix. The two are therefore independent implementations of the
same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple

import numpy as np


def _parse_spin(S) -> Fraction:
    f = Fraction(S) if not isinstance(S, str) else Fraction(S)
    if f <= 0 or (2 * f).denominator != 1:
        raise ValueError(f"spin must be a positive (half-)integer, got {S}")
    return f


@dataclass(frozen=True)
class LocalSpace:
    kind: str                    # 'spin' | 'fermion' | 'boson'
    q: int                       # local dimension
    spin: str = ""               # e.g. '1/2' (spin only)
    convention: str = ""         # spin: 'spin' | 'pauli' | 'pauli_pm1'
    n_max: int = 0               # boson cutoff
    jw_string: str = ""          # fermion: 'higher' (QuSpin) | 'lower'

    # ------------------------------------------------------------------ build
    @staticmethod
    def spin_space(S="1/2", convention="spin") -> "LocalSpace":
        f = _parse_spin(S)
        if convention not in ("spin", "pauli", "pauli_pm1"):
            raise ValueError(f"unknown spin convention {convention!r}")
        if convention != "spin" and f != Fraction(1, 2):
            raise ValueError("Pauli conventions are only defined for S=1/2")
        return LocalSpace("spin", int(2 * f + 1), spin=str(f), convention=convention)

    @staticmethod
    def fermion_space(jw_string: str = "higher") -> "LocalSpace":
        if jw_string not in ("higher", "lower"):
            raise ValueError(f"jw_string must be 'higher' or 'lower', got {jw_string!r}")
        return LocalSpace("fermion", 2, jw_string=jw_string)

    @staticmethod
    def boson_space(n_max: int) -> "LocalSpace":
        if n_max < 1:
            raise ValueError("boson cutoff n_max must be >= 1")
        return LocalSpace("boson", n_max + 1, n_max=n_max)

    # ------------------------------------------------------------ properties
    @property
    def letters(self) -> Tuple[str, ...]:
        if self.kind == "spin":
            return ("I", "x", "y", "z", "+", "-")
        if self.kind == "fermion":
            return ("I", "+", "-", "n", "z")
        return ("I", "+", "-", "n")

    @property
    def is_fermionic(self) -> bool:
        return self.kind == "fermion"

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "q": self.q}
        if self.kind == "spin":
            d.update(spin=self.spin, convention=self.convention)
        if self.kind == "boson":
            d.update(n_max=self.n_max)
        if self.kind == "fermion":
            d.update(jw_string=self.jw_string)
        return d

    @staticmethod
    def from_dict(d: dict) -> "LocalSpace":
        k = d["kind"]
        if k == "spin":
            return LocalSpace.spin_space(d.get("spin", "1/2"), d.get("convention", "spin"))
        if k == "fermion":
            return LocalSpace.fermion_space(d.get("jw_string", "higher"))
        if k == "boson":
            return LocalSpace.boson_space(int(d["n_max"]))
        raise ValueError(f"unknown local space kind {k!r}")

    # ------------------------------------------------------------- operators
    def matrix(self, letter: str) -> np.ndarray:
        """Dense q x q matrix <k'|o|k> of a letter (no Jordan-Wigner string)."""
        q = self.q
        if letter == "I":
            return np.eye(q, dtype=complex)
        if self.kind == "spin":
            S = Fraction(self.spin)
            m = np.array([float(k - S) for k in range(q)])
            Sp = np.zeros((q, q), dtype=complex)
            Sfl = float(S)
            for k in range(q - 1):
                mk = m[k]
                Sp[k + 1, k] = np.sqrt(Sfl * (Sfl + 1) - mk * (mk + 1))
            Sm = Sp.conj().T
            Sz = np.diag(m).astype(complex)
            Sx = (Sp + Sm) / 2
            Sy = (Sp - Sm) / 2j
            base = {"x": Sx, "y": Sy, "z": Sz, "+": Sp, "-": Sm}
            if letter not in base:
                raise KeyError(f"letter {letter!r} not defined for spin space")
            M = base[letter]
            if self.convention == "pauli":
                return 2 * M
            if self.convention == "pauli_pm1":
                return M if letter in "+-" else 2 * M
            return M
        if self.kind == "fermion":
            c = np.array([[0, 1], [0, 0]], dtype=complex)  # annihilator |1> -> |0>
            base = {"+": c.T.copy(), "-": c, "n": np.diag([0, 1]).astype(complex),
                    "z": np.diag([-0.5, 0.5]).astype(complex)}
            if letter not in base:
                raise KeyError(f"letter {letter!r} not defined for fermion space")
            return base[letter]
        # boson
        a = np.zeros((q, q), dtype=complex)
        for n in range(1, q):
            a[n - 1, n] = np.sqrt(n)
        base = {"+": a.T.copy(), "-": a, "n": np.diag(np.arange(q)).astype(complex)}
        if letter not in base:
            raise KeyError(f"letter {letter!r} not defined for boson space")
        return base[letter]

    def shift_components(self, letter: str) -> List[Tuple[int, np.ndarray]]:
        """Decompose a letter into [(delta, amp)] with amp[k] = <k+delta|o|k>.

        amp has length q; entries whose output digit is out of range are zero.
        Derived from explicit formulas (not from `matrix`) so that the bitwise
        builder does not share code with the Kronecker builder.
        """
        q = self.q
        k = np.arange(q)
        if letter == "I":
            return [(0, np.ones(q, dtype=complex))]
        if self.kind == "spin":
            S = float(Fraction(self.spin))
            m = k - S
            raise_amp = np.where(k < q - 1, np.sqrt(np.maximum(S * (S + 1) - m * (m + 1), 0.0)), 0.0)
            lower_amp = np.where(k > 0, np.sqrt(np.maximum(S * (S + 1) - m * (m - 1), 0.0)), 0.0)
            pm_scale = 2.0 if self.convention == "pauli" else 1.0
            xyz_scale = 2.0 if self.convention in ("pauli", "pauli_pm1") else 1.0
            if letter == "z":
                return [(0, xyz_scale * m.astype(complex))]
            if letter == "+":
                return [(+1, pm_scale * raise_amp.astype(complex))]
            if letter == "-":
                return [(-1, pm_scale * lower_amp.astype(complex))]
            if letter == "x":   # S^x = (S^+ + S^-)/2
                return [(+1, xyz_scale * 0.5 * raise_amp.astype(complex)),
                        (-1, xyz_scale * 0.5 * lower_amp.astype(complex))]
            if letter == "y":   # S^y = (S^+ - S^-)/(2i)
                return [(+1, xyz_scale * (-0.5j) * raise_amp),
                        (-1, xyz_scale * (0.5j) * lower_amp)]
            raise KeyError(letter)
        if self.kind == "fermion":
            if letter == "+":
                return [(+1, np.array([1.0, 0.0], dtype=complex))]
            if letter == "-":
                return [(-1, np.array([0.0, 1.0], dtype=complex))]
            if letter == "n":
                return [(0, np.array([0.0, 1.0], dtype=complex))]
            if letter == "z":
                return [(0, np.array([-0.5, 0.5], dtype=complex))]
            raise KeyError(letter)
        # boson
        if letter == "+":
            return [(+1, np.where(k < q - 1, np.sqrt(k + 1.0), 0.0).astype(complex))]
        if letter == "-":
            return [(-1, np.sqrt(k.astype(float)).astype(complex))]
        if letter == "n":
            return [(0, k.astype(complex))]
        raise KeyError(letter)

    # ------------------------------------------------------- exact (scaled) form
    def exact_weights(self) -> List[int]:
        """Integer weights w_k = d_k^2 of the diagonal scaling D = diag(d_k) in which every letter has
        rational matrix elements: spin S: binom(2S, k) (symmetric under the flip k <-> q-1-k);
        boson: k!; fermion: 1. All weights are >= 1."""
        from math import comb, factorial
        if self.kind == "spin":
            twoS = self.q - 1
            return [comb(twoS, k) for k in range(self.q)]
        if self.kind == "boson":
            return [factorial(k) for k in range(self.q)]
        return [1] * self.q

    def _exact_amp2(self, delta: int) -> List[Fraction]:
        """Exact squares of the shift amplitudes <k+delta|S^+ or S^-|k> (spin) or <k+delta|b^+-|k>
        (boson), unscaled; zero where the target digit is out of range."""
        q = self.q
        out = []
        for k in range(q):
            if not (0 <= k + delta < q):
                out.append(Fraction(0)); continue
            if self.kind == "spin":
                S = Fraction(self.spin); m = k - S
                out.append(S * (S + 1) - m * (m + delta))
            elif self.kind == "boson":
                out.append(Fraction(k + 1) if delta > 0 else Fraction(k))
            else:
                out.append(Fraction(1))
        return out

    def exact_scaled_components(self, letter: str):
        """Exact shift components of D^-1 o D: [(delta, [(re, im) Fractions per digit k])], where
        entry k is <k+delta|D^-1 o D|k>. With the weights of exact_weights every entry is rational
        (spin: S^+ -> k+1, S^- -> 2S-k+1; boson: b^+ -> 1, b -> k). Raises ValueError if a scaled
        element is irrational (cannot happen for the spaces defined here)."""
        from math import isqrt
        q = self.q
        w = self.exact_weights()
        Z = Fraction(0)

        def sqrt_rational(f: Fraction) -> Fraction:
            a, b = f.numerator, f.denominator
            ra, rb = isqrt(a), isqrt(b)
            if ra * ra != a or rb * rb != b:
                raise ValueError(f"scaled element sqrt({f}) is irrational")
            return Fraction(ra, rb)

        def scaled(delta):
            a2 = self._exact_amp2(delta)
            return [sqrt_rational(a2[k] * w[k] / w[k + delta]) if a2[k] != 0 else Z for k in range(q)]

        if letter == "I":
            return [(0, [(Fraction(1), Z)] * q)]
        if self.kind == "fermion":
            comps = self.shift_components(letter)       # all rational for fermions
            return [(d, [(Fraction(float(a.real)), Fraction(float(a.imag))) for a in amp]) for d, amp in comps]
        if self.kind == "spin":
            S = Fraction(self.spin)
            pm = Fraction(2) if self.convention == "pauli" else Fraction(1)
            xyz = Fraction(2) if self.convention in ("pauli", "pauli_pm1") else Fraction(1)
            up_, dn = scaled(+1), scaled(-1)
            if letter == "z":
                return [(0, [(xyz * (k - S), Z) for k in range(q)])]
            if letter == "+":
                return [(+1, [(pm * v, Z) for v in up_])]
            if letter == "-":
                return [(-1, [(pm * v, Z) for v in dn])]
            if letter == "x":
                h = xyz / 2
                return [(+1, [(h * v, Z) for v in up_]), (-1, [(h * v, Z) for v in dn])]
            if letter == "y":
                h = xyz / 2
                return [(+1, [(Z, -h * v) for v in up_]), (-1, [(Z, h * v) for v in dn])]
            raise KeyError(letter)
        # boson
        if letter == "+":
            return [(+1, [(v, Z) for v in scaled(+1)])]
        if letter == "-":
            return [(-1, [(v, Z) for v in scaled(-1)])]
        if letter == "n":
            return [(0, [(Fraction(k), Z) for k in range(q)])]
        raise KeyError(letter)

    def jw_letter(self, letter: str) -> bool:
        """Whether this letter carries a Jordan-Wigner string."""
        return self.kind == "fermion" and letter in ("+", "-")

    def jw_on(self, letter: str, l: int, site: int) -> bool:
        """Whether site l carries a parity factor in the string of `letter` at `site`."""
        if not self.jw_letter(letter):
            return False
        return l > site if self.jw_string == "higher" else l < site

    def local_parity(self) -> np.ndarray:
        """(-1)^n on the local space (fermions only)."""
        return np.diag([1.0, -1.0]).astype(complex)
