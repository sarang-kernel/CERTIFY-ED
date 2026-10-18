import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

from certify_ed.localops import LocalSpace
from certify_ed.spec import ModelSpec

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


from certify_ed.testing import LOCALS, random_spec  # noqa: E402,F401  (re-exported for the tests)
