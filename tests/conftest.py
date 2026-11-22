import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)


try:
    import quspin  # noqa: F401
    HAVE_QUSPIN = True
except Exception:
    HAVE_QUSPIN = False

try:
    import flint  # noqa: F401
    HAVE_FLINT = True
except Exception:
    HAVE_FLINT = False

needs_quspin = pytest.mark.skipif(not HAVE_QUSPIN, reason="QuSpin not installed")
needs_flint = pytest.mark.skipif(not HAVE_FLINT, reason="python-flint not installed")


def eigh_dense(a):
    """scipy.linalg.eigh of a dense Hermitian matrix, with the 1 x 1 case done directly: scipy 1.8 cannot size the
    LAPACK workspace for n = 1 (a sector of dimension 1). The same result as scipy.linalg.eigh otherwise."""
    import numpy as np
    import scipy.linalg as la
    a = np.asarray(a)
    if a.shape == (1, 1):
        return np.real(a).reshape(1).astype(float), np.ones((1, 1), dtype=np.result_type(a.dtype, np.float64))
    return la.eigh(a)


def pytest_addoption(parser):
    parser.addoption("--perf", action="store_true", default=False,
                     help="run the performance regression tests (tests/test_perf_regression.py, roadmap H-04)")


from certify_ed.testing import LOCALS, random_spec  # noqa: E402,F401  (re-exported for the tests)
