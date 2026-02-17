"""
Velocity Transforms for Cubed-Sphere

Convert between contravariant velocity (v^1, v^2), covariant velocity (v_1, v_2),
and Cartesian velocity (Vx, Vy, Vz).

Key equations:
  V = v^1 * a_1 + v^2 * a_2       (contravariant to Cartesian)
  v_i = V . a_i                     (Cartesian to covariant)
  v^i = g^{ij} * v_j               (covariant to contravariant, index raising)
  v_i = g_{ij} * v^j               (contravariant to covariant, index lowering)

where a_i = dr/dxi^i are covariant basis vectors (tangent to coordinate lines)
and g_{ij} = a_i . a_j is the covariant metric tensor.

NAMING CONVENTION:
  contravariant = v^i (upper index) -- components in the direction of basis vectors
  covariant     = v_i (lower index) -- projections onto basis vectors
  
  The SWE momentum equation evolves COVARIANT v_i (Shashkin Eq. 50).
  The mass flux uses CONTRAVARIANT v^i via u = J * v^contra.
  compute_contravariant() in the SWE code takes covariant v_i and returns v^i.

CRITICAL: The original velocity_transforms.py had BOTH functions misnamed:
  old "covariant_to_cartesian"  did V = u*a_1 + v*a_2  (actually contravariant->Cartesian)
  old "cartesian_to_covariant"  returned g^{ij}(V.a_j)  (actually Cartesian->contravariant)
  
This file has FOUR correctly named functions:
  contravariant_to_cartesian:  V = v^i * a_i            (no metric needed)
  cartesian_to_contravariant:  v^i = g^{ij} (V . a_j)   (needs metric inverse)
  covariant_to_cartesian:      raise first, then expand   (needs metric inverse)
  cartesian_to_covariant:      v_i = V . a_i             (no metric needed)
"""

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from .geometry import equiangular_to_cartesian


def get_covariant_basis(xi1, xi2, face_id):
    """
    Compute covariant basis vectors a_1, a_2 at point (xi1, xi2) on face.
    
    a_i = dr/dxi^i where r = (X, Y, Z) is position on sphere.
    
    Returns:
        a1: (a1_x, a1_y, a1_z) - tangent vector in xi1 direction
        a2: (a2_x, a2_y, a2_z) - tangent vector in xi2 direction
    """
    t1 = jnp.tan(xi1)
    t2 = jnp.tan(xi2)
    s1 = 1.0 / jnp.cos(xi1)**2  # sec^2(xi1)
    s2 = 1.0 / jnp.cos(xi2)**2  # sec^2(xi2)
    
    d2 = 1.0 + t1**2 + t2**2
    d = jnp.sqrt(d2)
    d3 = d2 * d
    
    # Derivatives of (t1/d), (t2/d), (1/d) with respect to xi1 and xi2
    dt1d_dxi1 = s1 * (1 + t2**2) / d3
    dt2d_dxi1 = -t1 * t2 * s1 / d3
    d1d_dxi1 = -t1 * s1 / d3
    
    dt1d_dxi2 = -t1 * t2 * s2 / d3
    dt2d_dxi2 = s2 * (1 + t1**2) / d3
    d1d_dxi2 = -t2 * s2 / d3
    
    # Apply chain rule based on face transform
    if face_id == 0:  # X = t1/d, Y = t2/d, Z = 1/d
        a1 = (dt1d_dxi1, dt2d_dxi1, d1d_dxi1)
        a2 = (dt1d_dxi2, dt2d_dxi2, d1d_dxi2)
    elif face_id == 1:  # X = -t1/d, Y = 1/d, Z = t2/d
        a1 = (-dt1d_dxi1, d1d_dxi1, dt2d_dxi1)
        a2 = (-dt1d_dxi2, d1d_dxi2, dt2d_dxi2)
    elif face_id == 2:  # X = -1/d, Y = -t1/d, Z = t2/d
        a1 = (-d1d_dxi1, -dt1d_dxi1, dt2d_dxi1)
        a2 = (-d1d_dxi2, -dt1d_dxi2, dt2d_dxi2)
    elif face_id == 3:  # X = t1/d, Y = -1/d, Z = t2/d
        a1 = (dt1d_dxi1, -d1d_dxi1, dt2d_dxi1)
        a2 = (dt1d_dxi2, -d1d_dxi2, dt2d_dxi2)
    elif face_id == 4:  # X = 1/d, Y = t1/d, Z = t2/d
        a1 = (d1d_dxi1, dt1d_dxi1, dt2d_dxi1)
        a2 = (d1d_dxi2, dt1d_dxi2, dt2d_dxi2)
    elif face_id == 5:  # X = -t1/d, Y = t2/d, Z = -1/d
        a1 = (-dt1d_dxi1, dt2d_dxi1, -d1d_dxi1)
        a2 = (-dt1d_dxi2, dt2d_dxi2, -d1d_dxi2)
    else:
        raise ValueError(f"Invalid face_id: {face_id}")
    
    return a1, a2


def _compute_metric_and_inverse(a1, a2):
    """
    Compute covariant metric g_{ij} and its inverse g^{ij}.
    
    Returns: g11, g12, g22, ginv11, ginv12, ginv22
    """
    g11 = a1[0]**2 + a1[1]**2 + a1[2]**2
    g12 = a1[0]*a2[0] + a1[1]*a2[1] + a1[2]*a2[2]
    g22 = a2[0]**2 + a2[1]**2 + a2[2]**2
    
    det_g = g11 * g22 - g12**2
    ginv11 = g22 / det_g
    ginv12 = -g12 / det_g
    ginv22 = g11 / det_g
    
    return g11, g12, g22, ginv11, ginv12, ginv22


# ============================================================
# Contravariant <-> Cartesian  (v^i <-> V)
# ============================================================

def contravariant_to_cartesian(v1, v2, xi1, xi2, face_id):
    """
    Convert CONTRAVARIANT velocity (v^1, v^2) to Cartesian (Vx, Vy, Vz).
    
    V = v^1 * a_1 + v^2 * a_2
    
    This is the standard expansion of a vector in terms of contravariant
    components and covariant basis vectors. No metric needed.
    """
    a1, a2 = get_covariant_basis(xi1, xi2, face_id)
    
    Vx = v1 * a1[0] + v2 * a2[0]
    Vy = v1 * a1[1] + v2 * a2[1]
    Vz = v1 * a1[2] + v2 * a2[2]
    
    return Vx, Vy, Vz


def cartesian_to_contravariant(Vx, Vy, Vz, xi1, xi2, face_id):
    """
    Convert Cartesian velocity (Vx, Vy, Vz) to CONTRAVARIANT (v^1, v^2).
    
    v^i = g^{ij} * (V . a_j)
    
    First projects V onto covariant basis to get covariant components v_i,
    then raises the index with the inverse metric g^{ij}.
    """
    a1, a2 = get_covariant_basis(xi1, xi2, face_id)
    
    # Covariant components: v_i = V . a_i
    V_dot_a1 = Vx * a1[0] + Vy * a1[1] + Vz * a1[2]
    V_dot_a2 = Vx * a2[0] + Vy * a2[1] + Vz * a2[2]
    
    # Raise index: v^i = g^{ij} v_j
    _, _, _, ginv11, ginv12, ginv22 = _compute_metric_and_inverse(a1, a2)
    
    v1 = ginv11 * V_dot_a1 + ginv12 * V_dot_a2
    v2 = ginv12 * V_dot_a1 + ginv22 * V_dot_a2
    
    return v1, v2


# ============================================================
# Covariant <-> Cartesian  (v_i <-> V)
# ============================================================

def cartesian_to_covariant(Vx, Vy, Vz, xi1, xi2, face_id):
    """
    Convert Cartesian velocity (Vx, Vy, Vz) to COVARIANT (v_1, v_2).
    
    v_i = V . a_i
    
    These are the natural projections onto the covariant basis.
    The SWE momentum equation (Shashkin Eq. 50) evolves these components.
    NO metric inverse needed -- this is just a dot product.
    """
    a1, a2 = get_covariant_basis(xi1, xi2, face_id)
    
    v1 = Vx * a1[0] + Vy * a1[1] + Vz * a1[2]
    v2 = Vx * a2[0] + Vy * a2[1] + Vz * a2[2]
    
    return v1, v2


def covariant_to_cartesian(v1, v2, xi1, xi2, face_id):
    """
    Convert COVARIANT velocity (v_1, v_2) to Cartesian (Vx, Vy, Vz).
    
    Two steps:
      1. Raise index: v^i = g^{ij} v_j
      2. Expand: V = v^i * a_i
    
    This is equivalent to V = v_i * a^i where a^i = g^{ij} a_j are the
    contravariant basis vectors.
    
    REQUIRES metric inverse (unlike cartesian_to_covariant which is just a dot product).
    """
    a1, a2 = get_covariant_basis(xi1, xi2, face_id)
    
    # Step 1: Raise index  v^i = g^{ij} v_j
    _, _, _, ginv11, ginv12, ginv22 = _compute_metric_and_inverse(a1, a2)
    
    v1_up = ginv11 * v1 + ginv12 * v2
    v2_up = ginv12 * v1 + ginv22 * v2
    
    # Step 2: Expand  V = v^i a_i
    Vx = v1_up * a1[0] + v2_up * a2[0]
    Vy = v1_up * a1[1] + v2_up * a2[1]
    Vz = v1_up * a1[2] + v2_up * a2[2]
    
    return Vx, Vy, Vz

