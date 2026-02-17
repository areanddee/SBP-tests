"""
geometry.py — Equiangular Gnomonic Projection and Metric Tensor
================================================================

Coordinate-system geometry for the cubed sphere, independent of
any particular grid discretization.

The equiangular gnomonic projection maps each cube face to the unit
sphere via (xi1, xi2) ∈ [-π/4, π/4]².  The metric tensor is the
same for all 6 panels in these coordinates.

Reference: Shashkin 2025, Section 2
"""

import jax.numpy as jnp


# ============================================================
# Equiangular → Cartesian mapping
# ============================================================

def equiangular_to_cartesian(xi1, xi2, face_id):
    """
    Map equiangular coordinates (xi1, xi2) to Cartesian (X, Y, Z)
    on the unit sphere for the given cube face.

    Face numbering (Ronchi et al.):
        0: +Z (top)     1: +X (front)   2: +Y (left)
        3: -X (back)    4: -Y (right)   5: -Z (bottom)

    Args:
        xi1, xi2: Equiangular coordinates in [-π/4, π/4]
        face_id: Integer 0-5

    Returns:
        X, Y, Z: Cartesian coordinates on unit sphere
    """
    t1 = jnp.tan(xi1)
    t2 = jnp.tan(xi2)
    r = jnp.sqrt(1.0 + t1**2 + t2**2)

    if face_id == 0:    # +Z
        X, Y, Z = t1 / r, t2 / r, 1.0 / r
    elif face_id == 1:  # +X
        X, Y, Z = 1.0 / r, t1 / r, t2 / r
    elif face_id == 2:  # +Y
        X, Y, Z = -t1 / r, 1.0 / r, t2 / r
    elif face_id == 3:  # -X
        X, Y, Z = -1.0 / r, -t1 / r, t2 / r
    elif face_id == 4:  # -Y
        X, Y, Z = t1 / r, -1.0 / r, t2 / r
    elif face_id == 5:  # -Z
        X, Y, Z = -t1 / r, t2 / r, -1.0 / r
    else:
        raise ValueError(f"face_id must be 0-5, got {face_id}")

    return X, Y, Z


# ============================================================
# Metric tensor
# ============================================================

def compute_metric(xi1, xi2):
    """
    Compute metric terms at arbitrary (xi1, xi2) points.

    The metric is identical for all 6 panels in equiangular coordinates.

    Returns:
        J:   Jacobian √G = 1/(r³ cos²ξ1 cos²ξ2)
        Q11: Contravariant metric G^11
        Q12: Contravariant metric G^12 = G^21
        Q22: Contravariant metric G^22

    where r² = 1 + tan²ξ1 + tan²ξ2

    Reference: Shashkin 2025, Eq. 5-8
    """
    t1 = jnp.tan(xi1)
    t2 = jnp.tan(xi2)
    c1 = jnp.cos(xi1)
    c2 = jnp.cos(xi2)
    r2 = 1.0 + t1**2 + t2**2
    r = jnp.sqrt(r2)

    J = 1.0 / (r**3 * c1**2 * c2**2)
    alpha = r**4 * c1**2 * c2**2
    Q11 = alpha * (1.0 - t1**2 / r2)
    Q12 = alpha * (-t1 * t2 / r2)
    Q22 = alpha * (1.0 - t2**2 / r2)

    return J, Q11, Q12, Q22
