"""
system.py — Full Cubed-Sphere Linearized SWE Assembly
=======================================================

Assembles all components (operators, grid, metric, SAT, projection)
into a complete right-hand side function for the linearized shallow
water equations on the cubed sphere (f=0).

Equations (Shashkin Eq. 50 with F=0):
    dv1/dt = -g Dvc @ (Ah h)                         (gradient x1)
    dv2/dt = -g (Ah h) @ Dvc.T                       (gradient x2)
    dh/dt  = -H0 Ah J_h^{-1} [Dcv@(J1 v^1) + (J2 v^2)@Dcv.T + SAT]  (continuity)

where (v^1, v^2) = Q(v1, v2) are contravariant velocities and
Ah denotes the h-projection operator.

Reference: Shashkin 2025, Section 5
"""

import jax
import jax.numpy as jnp

from .operators import sbp_42
from .geometry import compute_metric
from .mesh import make_staggered_grids, make_all_metrics, compute_contravariant
from .projection import build_projection_fn
from .sat import build_cartesian_sat_fn


def make_cubed_sphere_swe(N, H0, g):
    """
    Build cubed-sphere linearized SWE system (f=0).

    Args:
        N:  Grid resolution (cells per panel per direction)
        H0: Mean fluid height
        g:  Gravity

    Returns:
        dict with keys:
            rhs:        JIT-compiled function (h, v1, v2) → (dh, dv1, dv2)
            project_h:  h-projection function
            grids:      dict of coordinate arrays
            metrics:    dict of metric quantities at grid points
            ops:        SBP operator bundle
            corners:    list of corner groups
            Wh, W1, W2: quadrature weights
            Jh, J1, J2: Jacobians
            Pvc, Pcv:   interpolation matrices
            N, dx:      resolution info
    """
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)
    dx = float(grids['dx'])

    ops = sbp_42(N, dx)
    Dvc = ops.Dvc
    Dcv = ops.Dcv
    Pvc = ops.Pvc
    Pcv = ops.Pcv
    Hv_diag = jnp.diag(ops.Hv)
    Hc_diag = jnp.diag(ops.Hc)

    project_h, corners = build_projection_fn(N, metrics['Jh'], Hv_diag)
    add_sat = build_cartesian_sat_fn(N, ops, compute_metric)

    Jh = metrics['Jh']
    J1 = metrics['J1']
    J2 = metrics['J2']
    Jh_inv = 1.0 / Jh

    # 2D quadrature weights (per panel)
    Wh = jnp.outer(Hv_diag, Hv_diag)   # (N+1, N+1)
    W1 = jnp.outer(Hc_diag, Hv_diag)   # (N, N+1)
    W2 = jnp.outer(Hv_diag, Hc_diag)   # (N+1, N)

    def rhs(h, v1, v2):
        """
        RHS of linearized SWE on cubed sphere (f=0).

        Args:
            h:  (6, N+1, N+1) height perturbation
            v1: (6, N, N+1)   covariant velocity component 1
            v2: (6, N+1, N)   covariant velocity component 2

        Returns:
            dh_dt, dv1_dt, dv2_dt: tendencies
        """
        # Project h at all interfaces
        h_proj = project_h(h)

        # Gradient (momentum equations)
        dv1_dt = -g * jnp.einsum('ij,pjk->pik', Dvc, h_proj)   # (6,N,N+1)
        dv2_dt = -g * jnp.einsum('pij,kj->pik', h_proj, Dvc)   # (6,N+1,N)

        # Contravariant velocity (vmap over panels)
        v1c, v2c = _contra_vmap(v1, v2)
        u1_all = J1 * v1c   # (6, N, N+1) mass flux
        u2_all = J2 * v2c   # (6, N+1, N) mass flux

        # Divergence
        div = (jnp.einsum('ij,pjk->pik', Dcv, u1_all) +
               jnp.einsum('pij,kj->pik', u2_all, Dcv))   # (6,N+1,N+1)

        # SAT corrections at all 12 edges
        div = add_sat(div, u1_all, u2_all, v1, v2)

        # Continuity: dh/dt = -H0 · project(J_h^{-1} · div)
        dh_dt = project_h(-H0 * Jh_inv * div)

        return dh_dt, dv1_dt, dv2_dt

    # vmap contravariant velocity over panel axis
    def _contra_single(v1_p, v2_p):
        return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
    _contra_vmap = jax.vmap(_contra_single)

    rhs_jit = jax.jit(rhs)

    return {
        'rhs': rhs_jit,
        'project_h': project_h,
        'grids': grids,
        'metrics': metrics,
        'ops': ops,
        'corners': corners,
        'Wh': Wh, 'W1': W1, 'W2': W2,
        'Jh': Jh, 'J1': J1, 'J2': J2,
        'Pvc': Pvc, 'Pcv': Pcv,
        'N': N, 'dx': dx,
    }
