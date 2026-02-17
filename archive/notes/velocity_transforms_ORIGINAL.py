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

from grid import equiangular_to_cartesian


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


# ============================================================
# Unit Tests
# ============================================================

def test_basis_vectors_orthogonal_to_normal():
    """Basis vectors a_1, a_2 should be tangent to the sphere (r . a_i = 0)."""
    print("Test: Basis vectors orthogonal to normal...")
    
    max_err = 0.0
    for face_id in range(6):
        for xi1 in [-0.5, 0.0, 0.5]:
            for xi2 in [-0.5, 0.0, 0.5]:
                X, Y, Z = equiangular_to_cartesian(xi1, xi2, face_id)
                a1, a2 = get_covariant_basis(xi1, xi2, face_id)
                
                r_dot_a1 = X * a1[0] + Y * a1[1] + Z * a1[2]
                r_dot_a2 = X * a2[0] + Y * a2[1] + Z * a2[2]
                
                err = max(abs(float(r_dot_a1)), abs(float(r_dot_a2)))
                max_err = max(max_err, err)
    
    print(f"  Max |r . a| = {max_err:.2e}")
    assert max_err < 1e-10, f"Basis not tangent to sphere"
    print("  PASSED")


def test_contravariant_roundtrip():
    """(v^1,v^2) -> (Vx,Vy,Vz) -> (v^1,v^2) should round-trip."""
    print("\nTest: Contravariant roundtrip...")
    
    max_err = 0.0
    for face_id in range(6):
        for xi1 in [-0.5, 0.0, 0.5]:
            for xi2 in [-0.5, 0.0, 0.5]:
                v1_orig, v2_orig = 1.5, -0.7
                
                Vx, Vy, Vz = contravariant_to_cartesian(v1_orig, v2_orig, xi1, xi2, face_id)
                v1_back, v2_back = cartesian_to_contravariant(Vx, Vy, Vz, xi1, xi2, face_id)
                
                err = max(abs(float(v1_back - v1_orig)), abs(float(v2_back - v2_orig)))
                max_err = max(max_err, err)
    
    print(f"  Max roundtrip error = {max_err:.2e}")
    assert max_err < 1e-10
    print("  PASSED")


def test_covariant_roundtrip():
    """(v_1,v_2) -> (Vx,Vy,Vz) -> (v_1,v_2) should round-trip."""
    print("\nTest: Covariant roundtrip...")
    
    max_err = 0.0
    for face_id in range(6):
        for xi1 in [-0.5, 0.0, 0.5]:
            for xi2 in [-0.5, 0.0, 0.5]:
                v1_orig, v2_orig = 1.5, -0.7
                
                Vx, Vy, Vz = covariant_to_cartesian(v1_orig, v2_orig, xi1, xi2, face_id)
                v1_back, v2_back = cartesian_to_covariant(Vx, Vy, Vz, xi1, xi2, face_id)
                
                err = max(abs(float(v1_back - v1_orig)), abs(float(v2_back - v2_orig)))
                max_err = max(max_err, err)
    
    print(f"  Max roundtrip error = {max_err:.2e}")
    assert max_err < 1e-10
    print("  PASSED")


def test_covariant_vs_contravariant_consistency():
    """Starting from the same Cartesian V, covariant and contravariant
    should be related by the metric: v^i = g^{ij} v_j."""
    print("\nTest: Covariant/contravariant consistency via metric...")
    
    max_err = 0.0
    for face_id in range(6):
        for xi1 in [-0.5, 0.0, 0.5]:
            for xi2 in [-0.5, 0.0, 0.5]:
                Vx, Vy, Vz = 1.0, -0.5, 0.3
                
                # Get both representations
                v1_cov, v2_cov = cartesian_to_covariant(Vx, Vy, Vz, xi1, xi2, face_id)
                v1_con, v2_con = cartesian_to_contravariant(Vx, Vy, Vz, xi1, xi2, face_id)
                
                # Raise covariant to get contravariant via metric
                a1, a2 = get_covariant_basis(xi1, xi2, face_id)
                _, _, _, ginv11, ginv12, ginv22 = _compute_metric_and_inverse(a1, a2)
                
                v1_raised = ginv11 * v1_cov + ginv12 * v2_cov
                v2_raised = ginv12 * v1_cov + ginv22 * v2_cov
                
                err = max(abs(float(v1_raised - v1_con)),
                         abs(float(v2_raised - v2_con)))
                max_err = max(max_err, err)
    
    print(f"  Max |v^i_raised - v^i_direct| = {max_err:.2e}")
    assert max_err < 1e-10
    print("  PASSED")


def test_covariant_continuity_at_edges():
    """
    CRITICAL TEST: At the SAME physical location on a shared edge,
    the Cartesian velocity must be identical whether computed from
    covariant components on either panel.
    """
    print("\nTest: Covariant continuity at edges...")
    
    pi4 = jnp.pi / 4
    test_xi_values = [-0.3, 0.0, 0.3]
    
    max_err = 0.0
    
    # Test P0-E <-> P4-N (T operation, axis-swap)
    for xi2_a in test_xi_values:
        xi1_a = pi4
        xi1_b = xi2_a  # T: column->row
        xi2_b = pi4
        
        # Verify same position
        Xa, Ya, Za = equiangular_to_cartesian(xi1_a, xi2_a, 0)
        Xb, Yb, Zb = equiangular_to_cartesian(xi1_b, xi2_b, 4)
        pos_err = jnp.sqrt((Xa-Xb)**2 + (Ya-Yb)**2 + (Za-Zb)**2)
        assert float(pos_err) < 1e-10
        
        # A Cartesian velocity at this point
        Vx, Vy, Vz = 1.0, -0.5, 0.3
        
        # Covariant in both panels
        v1_a, v2_a = cartesian_to_covariant(Vx, Vy, Vz, xi1_a, xi2_a, 0)
        v1_b, v2_b = cartesian_to_covariant(Vx, Vy, Vz, xi1_b, xi2_b, 4)
        
        # Reconstruct Cartesian from each panel's covariant
        Vx_a, Vy_a, Vz_a = covariant_to_cartesian(v1_a, v2_a, xi1_a, xi2_a, 0)
        Vx_b, Vy_b, Vz_b = covariant_to_cartesian(v1_b, v2_b, xi1_b, xi2_b, 4)
        
        err = float(jnp.sqrt((Vx_a-Vx_b)**2 + (Vy_a-Vy_b)**2 + (Vz_a-Vz_b)**2))
        max_err = max(max_err, err)
    
    print(f"  Max Cartesian velocity error at axis-swap edge = {max_err:.2e}")
    assert max_err < 1e-10
    print("  PASSED")


def test_old_vs_new_naming():
    """
    Verify that the OLD function names would give WRONG results for covariant input.
    
    The old code did: V = v_i * a_i (treating covariant as contravariant).
    The new code does: v^i = g^{ij} v_j, then V = v^i * a_i.
    These differ when g != identity (i.e., everywhere on the cubed sphere).
    """
    print("\nTest: Old vs new naming difference...")
    
    xi1, xi2, face_id = 0.3, 0.4, 0
    v1_cov, v2_cov = 1.5, -0.7  # These are COVARIANT components
    
    # CORRECT: new covariant_to_cartesian (raises index first)
    Vx_correct, Vy_correct, Vz_correct = covariant_to_cartesian(
        v1_cov, v2_cov, xi1, xi2, face_id)
    
    # WRONG: what old code did (no index raising, treated as contravariant)
    Vx_wrong, Vy_wrong, Vz_wrong = contravariant_to_cartesian(
        v1_cov, v2_cov, xi1, xi2, face_id)
    
    diff = float(jnp.sqrt((Vx_correct - Vx_wrong)**2 + 
                           (Vy_correct - Vy_wrong)**2 + 
                           (Vz_correct - Vz_wrong)**2))
    
    print(f"  |V_correct - V_wrong| = {diff:.2e}")
    assert diff > 0.01, "Should differ significantly away from panel center"
    print(f"  PASSED (they differ by {diff:.2e}, confirming fix matters)")


def run_all_tests():
    """Run all velocity transform tests."""
    print("="*60)
    print("VELOCITY TRANSFORM UNIT TESTS (corrected naming)")
    print("="*60)
    
    test_basis_vectors_orthogonal_to_normal()
    test_contravariant_roundtrip()
    test_covariant_roundtrip()
    test_covariant_vs_contravariant_consistency()
    test_covariant_continuity_at_edges()
    test_old_vs_new_naming()
    
    print()
    print("="*60)
    print("ALL VELOCITY TRANSFORM TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
