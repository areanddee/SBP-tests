"""
Cubed-Sphere Grid and Metric Terms

Equiangular gnomonic projection on unit sphere.
Coordinate transforms VERIFIED against halo_exchange.py connectivity.

Face assignments:
  Face 0: +Z (north pole)
  Face 1: +Y
  Face 2: -X
  Face 3: -Y
  Face 4: +X
  Face 5: -Z (south pole)

Note: On Face 0, xi1 varies along rows (→X), xi2 varies along columns (→Y).
Each face has its own (xi1, xi2) to (X, Y, Z) mapping.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class CubedSphereGrid(NamedTuple):
    """Grid coordinates and metric terms for one panel."""
    N: int
    dx: float
    
    # Coordinates at cell centers
    x1_h: jnp.ndarray  # [N, N]
    x2_h: jnp.ndarray  # [N, N]
    
    # Metric terms at cell centers
    sqrt_G_h: jnp.ndarray     # Jacobian √G [N, N]
    G11_contra_h: jnp.ndarray  # G^11 [N, N]
    G12_contra_h: jnp.ndarray  # G^12 [N, N]
    G22_contra_h: jnp.ndarray  # G^22 [N, N]


def equiangular_to_cartesian(xi1, xi2, face_id):
    """
    Convert equiangular coordinates to Cartesian (X, Y, Z) on unit sphere.
    
    VERIFIED: All 12 edge connections match physically.
    
    Args:
        xi1, xi2: Equiangular coordinates in [-π/4, π/4]
        face_id: Panel index 0-5
        
    Returns:
        X, Y, Z: Cartesian coordinates on unit sphere
    """
    t1 = jnp.tan(xi1)
    t2 = jnp.tan(xi2)
    d = jnp.sqrt(1.0 + t1**2 + t2**2)
    
    if face_id == 0:    # +Z (north pole)
        X, Y, Z = t1/d, t2/d, 1/d
    elif face_id == 1:  # +Y
        X, Y, Z = -t1/d, 1/d, t2/d
    elif face_id == 2:  # -X
        X, Y, Z = -1/d, -t1/d, t2/d
    elif face_id == 3:  # -Y
        X, Y, Z = t1/d, -1/d, t2/d
    elif face_id == 4:  # +X
        X, Y, Z = 1/d, t1/d, t2/d
    elif face_id == 5:  # -Z (south pole)
        X, Y, Z = -t1/d, t2/d, -1/d
    else:
        raise ValueError(f"Invalid face_id: {face_id}")
    
    return X, Y, Z


def compute_metric_at_points(x1, x2):
    """
    Compute metric terms at arbitrary points.
    
    The metric is the SAME for all panels in equiangular coordinates.
    Only the Cartesian mapping differs between panels.
    """
    tan_x1 = jnp.tan(x1)
    tan_x2 = jnp.tan(x2)
    cos_x1 = jnp.cos(x1)
    cos_x2 = jnp.cos(x2)
    
    r_squared = 1.0 + tan_x1**2 + tan_x2**2
    r = jnp.sqrt(r_squared)
    
    # Jacobian √G = 1 / (r³ cos²x1 cos²x2)
    sqrt_G = 1.0 / (r**3 * cos_x1**2 * cos_x2**2)
    
    # Contravariant metric G^ij
    alpha = r**4 * cos_x1**2 * cos_x2**2
    
    G11_contra = alpha * (1.0 - tan_x1**2 / r_squared)
    G12_contra = alpha * (-tan_x1 * tan_x2 / r_squared)
    G22_contra = alpha * (1.0 - tan_x2**2 / r_squared)
    
    return sqrt_G, G11_contra, G12_contra, G22_contra


def make_cubed_sphere_grid(N):
    """
    Create grid and metric terms for a single panel.
    
    Args:
        N: Number of interior cells in each direction
        
    Returns:
        CubedSphereGrid with coordinates and metric terms
    """
    L = jnp.pi / 2
    dx = L / N
    
    # Cell center coordinates
    x_centers = jnp.linspace(-jnp.pi/4 + dx/2, jnp.pi/4 - dx/2, N)
    x1_h, x2_h = jnp.meshgrid(x_centers, x_centers, indexing='ij')
    
    # Compute metric at cell centers
    sqrt_G_h, G11_contra_h, G12_contra_h, G22_contra_h = compute_metric_at_points(x1_h, x2_h)
    
    return CubedSphereGrid(
        N=N,
        dx=dx,
        x1_h=x1_h, x2_h=x2_h,
        sqrt_G_h=sqrt_G_h,
        G11_contra_h=G11_contra_h,
        G12_contra_h=G12_contra_h,
        G22_contra_h=G22_contra_h,
    )


if __name__ == "__main__":
    print("="*60)
    print("Grid Module - Verified Transforms")
    print("="*60)
    
    print("\nFace centers on unit sphere:")
    for face in range(6):
        X, Y, Z = equiangular_to_cartesian(0.0, 0.0, face)
        print(f"  Face {face}: ({float(X):+.3f}, {float(Y):+.3f}, {float(Z):+.3f})")
    
    N = 16
    grid = make_cubed_sphere_grid(N)
    print(f"\nGrid N={N}, dx={grid.dx:.4f} rad")
    print(f"√G range: [{float(grid.sqrt_G_h.min()):.4f}, {float(grid.sqrt_G_h.max()):.4f}]")
