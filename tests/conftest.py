"""Shared pytest configuration and compatibility shims."""

import numpy as np

# numpy 2.x removed np.isscalar, but pytest 9.0.x still uses it internally
# in pytest.approx. Restore it until pytest ships a fix.
if not hasattr(np, "isscalar"):
    np.isscalar = lambda x: np.ndim(x) == 0
