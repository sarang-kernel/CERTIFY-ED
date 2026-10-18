"""Helpers shared by the test suite and the experiments: the local spaces exercised everywhere and a
random Hermitian operator-string model generator."""
import numpy as np

from .localops import LocalSpace
from .spec import ModelSpec

LOCALS = [
    ("spin1/2-spin", lambda: LocalSpace.spin_space("1/2", "spin"), 7),
    ("spin1/2-pauli", lambda: LocalSpace.spin_space("1/2", "pauli"), 7),
    ("spin1/2-pauli_pm1", lambda: LocalSpace.spin_space("1/2", "pauli_pm1"), 7),
    ("spin1", lambda: LocalSpace.spin_space("1", "spin"), 4),
    ("spin3/2", lambda: LocalSpace.spin_space("3/2", "spin"), 3),
    ("fermion", lambda: LocalSpace.fermion_space(), 7),
    ("boson3", lambda: LocalSpace.boson_space(3), 4),
]


def random_spec(local: LocalSpace, N: int, rng, n_terms: int = 6, max_len: int = 3,
                hermitian: bool = True, allow_complex: bool = True) -> ModelSpec:
    """Random Hermitian operator-string Hamiltonian (each string paired with its adjoint)."""
    letters = [l for l in local.letters if l != "I"]
    adj = {"+": "-", "-": "+"}
    static = []
    for _ in range(n_terms):
        L = int(rng.integers(1, max_len + 1))
        op = "".join(rng.choice(letters, L))
        sites = [int(s) for s in rng.choice(N, L, replace=L > N)]
        c = float(rng.normal())
        if allow_complex and rng.random() < 0.3:
            c = complex(c, float(rng.normal()))
        static.append([op, [[c] + sites]])
        if hermitian:
            op_adj = "".join(adj.get(l, l) for l in op[::-1])
            static.append([op_adj, [[np.conj(c)] + sites[::-1]]])
    return ModelSpec.from_static(N, local, static)
