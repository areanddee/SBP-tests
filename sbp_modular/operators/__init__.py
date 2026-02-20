"""
operators — SBP-SAT operator construction and dispatch
=======================================================

Public API:
    make_operators(N, dx, backend='auto')  → OperatorSet
    sbp_42(N, dx)                          → raw SBP operator namedtuple

Usage in solver:
    from operators import make_operators

    op = make_operators(N, dx)          # auto-selects dense/sparse
    op = make_operators(N, dx, 'dense') # force dense
    op = make_operators(N, dx, 'sparse')# force sparse

    # Apply operators:
    dv1_dt = -g * op.Dvc.ax0(h)        # Dvc @ h  (batched axis -2)
    dv2_dt = -g * op.Dvc.ax1(h)        # h @ Dvc.T (batched axis -1)

    # Access raw matrices:
    mat = op.Dvc.matrix

    # Quadrature & extrapolation:
    Hv_diag = op.Hv_diag
    l, r = op.l, op.r
"""

from .operator_dispatch import make_operators, OperatorSet, detect_backend
from .sbp_staggered_1d import sbp_42

__all__ = ['make_operators', 'OperatorSet', 'detect_backend', 'sbp_42']
