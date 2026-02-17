"""
mesh.py — Staggered Grid Construction and Metric Evaluation
=============================================================

Builds the staggered Arakawa C-grid for the cubed sphere and evaluates
metric quantities at all grid point locations.

Grid layout on each panel (xi1, xi2) ∈ [-π/4, π/4]²:
    h-points:  (N+1, N+1) at vertices  (xi_v × xi_v)
    v1-points: (N, N+1)   at x1-faces  (xi_c × xi_v)
    v2-points: (N+1, N)   at x2-faces  (xi_v × xi_c)

where xi_v are vertex coordinates and xi_c are cell-center coordinates.

Reference: Shashkin 2025, Section 3
"""

import jax
import jax.numpy as jnp

from .geometry import compute_metric


# ============================================================
# Grid construction
# ============================================================

def make_staggered_grids(N):
    """
    Create coordinate arrays for staggered grid on [-π/4, π/4]².

    Args:
        N: Number of cells per direction per panel

    Returns:
        dict with keys:
            xi_v:   (N+1,)   vertex coordinates
            xi_c:   (N,)     cell-center coordinates
            dx:     scalar   grid spacing
            xi1_h, xi2_h:    (N+1, N+1) h-point coordinates
            xi1_v1, xi2_v1:  (N, N+1)   v1-point coordinates
            xi1_v2, xi2_v2:  (N+1, N)   v2-point coordinates
    """
    L = jnp.pi / 2
    dx = L / N
    xi_v = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, N + 1)
    xi_c = (jnp.arange(N) + 0.5) * dx - jnp.pi / 4

    # 2D coordinates
    xi1_h, xi2_h = jnp.meshgrid(xi_v, xi_v, indexing='ij')     # (N+1, N+1)
    xi1_v1, xi2_v1 = jnp.meshgrid(xi_c, xi_v, indexing='ij')   # (N, N+1)
    xi1_v2, xi2_v2 = jnp.meshgrid(xi_v, xi_c, indexing='ij')   # (N+1, N)

    return {
        'xi_v': xi_v, 'xi_c': xi_c, 'dx': dx,
        'xi1_h': xi1_h, 'xi2_h': xi2_h,
        'xi1_v1': xi1_v1, 'xi2_v1': xi2_v1,
        'xi1_v2': xi1_v2, 'xi2_v2': xi2_v2,
    }


# ============================================================
# Metric evaluation at grid points
# ============================================================

def make_all_metrics(grids):
    """
    Compute Jacobian and contravariant metric at all staggered locations.

    Returns:
        dict with keys:
            Jh:    (N+1, N+1) Jacobian at h-points
            J1:    (N, N+1)   Jacobian at v1-points
            J2:    (N+1, N)   Jacobian at v2-points
            Q11_1: (N, N+1)   G^11 at v1-points
            Q12_h: (N+1, N+1) G^12 at h-points
            Q22_2: (N+1, N)   G^22 at v2-points
    """
    Jh, _, Q12_h, _ = compute_metric(grids['xi1_h'], grids['xi2_h'])
    J1, Q11_1, _, _ = compute_metric(grids['xi1_v1'], grids['xi2_v1'])
    J2, _, _, Q22_2 = compute_metric(grids['xi1_v2'], grids['xi2_v2'])

    return {
        'Jh': Jh, 'J1': J1, 'J2': J2,
        'Q11_1': Q11_1, 'Q12_h': Q12_h, 'Q22_2': Q22_2,
    }


# ============================================================
# Contravariant velocity (Eq. 56)
# ============================================================

def compute_contravariant(v1, v2, metrics, Pvc, Pcv):
    """
    Compute contravariant velocities v¹, v² from covariant v1, v2.

    Uses SBP interpolation operators to handle the off-diagonal metric
    coupling between staggered grid locations.

    Args:
        v1: (N, N+1) or (6, N, N+1) covariant velocity component 1
        v2: (N+1, N) or (6, N+1, N) covariant velocity component 2
        metrics: dict from make_all_metrics
        Pvc: (N, N+1) vertex-to-center interpolation matrix
        Pcv: (N+1, N) center-to-vertex interpolation matrix

    Returns:
        v1_contra, v2_contra: contravariant velocities at v1, v2 points

    Reference: Shashkin 2025, Eq. 56
    """
    Q11_1 = metrics['Q11_1']
    Q12_h = metrics['Q12_h']
    Q22_2 = metrics['Q22_2']
    Jh = metrics['Jh']
    J1 = metrics['J1']
    J2 = metrics['J2']

    JQ12 = Jh * Q12_h  # (N+1, N+1)

    # v¹: diagonal + off-diagonal via h-points
    v2_at_h = v2 @ Pcv.T          # P2h: interpolate v2 to h-points
    cross_at_h = JQ12 * v2_at_h
    cross_at_v1 = Pvc @ cross_at_h  # Ph1: interpolate to v1-points
    v1_contra = Q11_1 * v1 + cross_at_v1 / J1

    # v²: off-diagonal via h-points + diagonal
    v1_at_h = Pcv @ v1             # P1h: interpolate v1 to h-points
    cross_at_h2 = JQ12 * v1_at_h
    cross_at_v2 = cross_at_h2 @ Pvc.T  # Ph2: interpolate to v2-points
    v2_contra = cross_at_v2 / J2 + Q22_2 * v2

    return v1_contra, v2_contra
