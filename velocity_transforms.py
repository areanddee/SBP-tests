"""
Velocity Transforms for Cubed-Sphere

Convert between covariant velocity (u, v) and Cartesian velocity (Vx, Vy, Vz).

Key equations:
  V = u * a_1 + v * a_2           (covariant to Cartesian)
  u = G^{ij} * (V · a_j)          (Cartesian to covariant)

where a_i = ∂r/∂ξ^i are covariant basis vectors (tangent to coordinate lines).
"""

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from grid import equiangular_to_cartesian


def get_covariant_basis(xi1, xi2, face_id):
    """
    Compute covariant basis vectors a_1, a_2 at point (xi1, xi2) on face.
    
    a_i = ∂r/∂ξ^i where r = (X, Y, Z) is position on sphere.
    
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


def covariant_to_cartesian(u, v, xi1, xi2, face_id):
    """
    Convert covariant velocity (u, v) to Cartesian (Vx, Vy, Vz).
    
    V = u * a_1 + v * a_2
    """
    a1, a2 = get_covariant_basis(xi1, xi2, face_id)
    
    Vx = u * a1[0] + v * a2[0]
    Vy = u * a1[1] + v * a2[1]
    Vz = u * a1[2] + v * a2[2]
    
    return Vx, Vy, Vz


def cartesian_to_covariant(Vx, Vy, Vz, xi1, xi2, face_id):
    """
    Convert Cartesian velocity (Vx, Vy, Vz) to covariant (u, v).
    
    u = G^{11} * (V·a_1) + G^{12} * (V·a_2)
    v = G^{12} * (V·a_1) + G^{22} * (V·a_2)
    """
    a1, a2 = get_covariant_basis(xi1, xi2, face_id)
    
    # V · a_i
    V_dot_a1 = Vx * a1[0] + Vy * a1[1] + Vz * a1[2]
    V_dot_a2 = Vx * a2[0] + Vy * a2[1] + Vz * a2[2]
    
    # Covariant metric G_{ij} = a_i · a_j
    G11 = a1[0]**2 + a1[1]**2 + a1[2]**2
    G12 = a1[0]*a2[0] + a1[1]*a2[1] + a1[2]*a2[2]
    G22 = a2[0]**2 + a2[1]**2 + a2[2]**2
    
    # Contravariant metric G^{ij} = inverse of G_{ij}
    det_G = G11 * G22 - G12**2
    G11_inv = G22 / det_G
    G12_inv = -G12 / det_G
    G22_inv = G11 / det_G
    
    # u^i = G^{ij} * (V · a_j)
    u = G11_inv * V_dot_a1 + G12_inv * V_dot_a2
    v = G12_inv * V_dot_a1 + G22_inv * V_dot_a2
    
    return u, v


# =============================================================================
# Unit Tests
# =============================================================================

def test_basis_vectors_orthogonal_to_normal():
    """Basis vectors a_1, a_2 should be tangent to the sphere (r · a_i = 0)."""
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
    
    print(f"  Max |r · a| = {max_err:.2e}")
    assert max_err < 1e-10, f"Basis not tangent to sphere"
    print("  ✓ PASSED")


def test_velocity_roundtrip():
    """Converting (u,v) → (Vx,Vy,Vz) → (u,v) should give back original values."""
    print("\nTest: Velocity roundtrip...")
    
    max_err = 0.0
    for face_id in range(6):
        for xi1 in [-0.5, 0.0, 0.5]:
            for xi2 in [-0.5, 0.0, 0.5]:
                u_orig, v_orig = 1.5, -0.7
                
                Vx, Vy, Vz = covariant_to_cartesian(u_orig, v_orig, xi1, xi2, face_id)
                u_back, v_back = cartesian_to_covariant(Vx, Vy, Vz, xi1, xi2, face_id)
                
                err = max(abs(float(u_back - u_orig)), abs(float(v_back - v_orig)))
                max_err = max(max_err, err)
    
    print(f"  Max roundtrip error = {max_err:.2e}")
    assert max_err < 1e-10
    print("  ✓ PASSED")


def test_velocity_continuity_at_edges():
    """
    CRITICAL TEST: At the SAME physical location on a shared edge,
    the Cartesian velocity must be identical when computed from either panel.
    """
    print("\nTest: Velocity continuity at edges...")
    
    # Test multiple points along edges
    pi4 = jnp.pi / 4
    test_xi1_values = [-0.3, 0.0, 0.3]
    
    max_err = 0.0
    
    # Test (0,N) ↔ (1,N) [R]: Face 0 N connects to Face 1 N with reversal
    # Face 0 N: xi2 = pi/4, X = t1/d, Y = 1/d * t2_max, Z = 1/d
    # Face 1 N: xi2 = pi/4, but xi1 maps to -xi1 due to [R]
    for xi1_a in test_xi1_values:
        xi2_a = pi4
        xi1_b = -xi1_a  # [R] reversal
        xi2_b = pi4
        
        # Verify same position
        Xa, Ya, Za = equiangular_to_cartesian(xi1_a, xi2_a, 0)
        Xb, Yb, Zb = equiangular_to_cartesian(xi1_b, xi2_b, 1)
        pos_err = jnp.sqrt((Xa-Xb)**2 + (Ya-Yb)**2 + (Za-Zb)**2)
        assert float(pos_err) < 1e-10, f"Position mismatch: {float(pos_err)}"
        
        # Test velocity
        u_a, v_a = 1.0, 0.5
        Vx_a, Vy_a, Vz_a = covariant_to_cartesian(u_a, v_a, xi1_a, xi2_a, 0)
        u_b, v_b = cartesian_to_covariant(Vx_a, Vy_a, Vz_a, xi1_b, xi2_b, 1)
        Vx_b, Vy_b, Vz_b = covariant_to_cartesian(u_b, v_b, xi1_b, xi2_b, 1)
        
        err = float(jnp.sqrt((Vx_a-Vx_b)**2 + (Vy_a-Vy_b)**2 + (Vz_a-Vz_b)**2))
        max_err = max(max_err, err)
    
    # Test (0,S) ↔ (3,N) [N]: Same xi1 mapping
    for xi1_a in test_xi1_values:
        xi2_a = -pi4
        xi1_b = xi1_a  # [N] no change
        xi2_b = pi4
        
        Xa, Ya, Za = equiangular_to_cartesian(xi1_a, xi2_a, 0)
        Xb, Yb, Zb = equiangular_to_cartesian(xi1_b, xi2_b, 3)
        pos_err = jnp.sqrt((Xa-Xb)**2 + (Ya-Yb)**2 + (Za-Zb)**2)
        assert float(pos_err) < 1e-10, f"Position mismatch (0,S)↔(3,N): {float(pos_err)}"
        
        u_a, v_a = 1.0, 0.5
        Vx_a, Vy_a, Vz_a = covariant_to_cartesian(u_a, v_a, xi1_a, xi2_a, 0)
        u_b, v_b = cartesian_to_covariant(Vx_a, Vy_a, Vz_a, xi1_b, xi2_b, 3)
        Vx_b, Vy_b, Vz_b = covariant_to_cartesian(u_b, v_b, xi1_b, xi2_b, 3)
        
        err = float(jnp.sqrt((Vx_a-Vx_b)**2 + (Vy_a-Vy_b)**2 + (Vz_a-Vz_b)**2))
        max_err = max(max_err, err)
    
    # Test (0,E) ↔ (4,N) [T]: Transpose - xi1_a maps to xi2_b
    for xi2_a in test_xi1_values:  # varying along column
        xi1_a = pi4
        xi1_b = xi2_a  # [T] transpose: column becomes row
        xi2_b = pi4
        
        Xa, Ya, Za = equiangular_to_cartesian(xi1_a, xi2_a, 0)
        Xb, Yb, Zb = equiangular_to_cartesian(xi1_b, xi2_b, 4)
        pos_err = jnp.sqrt((Xa-Xb)**2 + (Ya-Yb)**2 + (Za-Zb)**2)
        assert float(pos_err) < 1e-10, f"Position mismatch (0,E)↔(4,N): {float(pos_err)}"
        
        u_a, v_a = 1.0, 0.5
        Vx_a, Vy_a, Vz_a = covariant_to_cartesian(u_a, v_a, xi1_a, xi2_a, 0)
        u_b, v_b = cartesian_to_covariant(Vx_a, Vy_a, Vz_a, xi1_b, xi2_b, 4)
        Vx_b, Vy_b, Vz_b = covariant_to_cartesian(u_b, v_b, xi1_b, xi2_b, 4)
        
        err = float(jnp.sqrt((Vx_a-Vx_b)**2 + (Vy_a-Vy_b)**2 + (Vz_a-Vz_b)**2))
        max_err = max(max_err, err)
    
    # Test (0,W) ↔ (2,N) [TR]: Transpose + Reverse
    for xi2_a in test_xi1_values:
        xi1_a = -pi4
        xi1_b = -xi2_a  # [TR] transpose and reverse
        xi2_b = pi4
        
        Xa, Ya, Za = equiangular_to_cartesian(xi1_a, xi2_a, 0)
        Xb, Yb, Zb = equiangular_to_cartesian(xi1_b, xi2_b, 2)
        pos_err = jnp.sqrt((Xa-Xb)**2 + (Ya-Yb)**2 + (Za-Zb)**2)
        assert float(pos_err) < 1e-10, f"Position mismatch (0,W)↔(2,N): {float(pos_err)}"
        
        u_a, v_a = 1.0, 0.5
        Vx_a, Vy_a, Vz_a = covariant_to_cartesian(u_a, v_a, xi1_a, xi2_a, 0)
        u_b, v_b = cartesian_to_covariant(Vx_a, Vy_a, Vz_a, xi1_b, xi2_b, 2)
        Vx_b, Vy_b, Vz_b = covariant_to_cartesian(u_b, v_b, xi1_b, xi2_b, 2)
        
        err = float(jnp.sqrt((Vx_a-Vx_b)**2 + (Vy_a-Vy_b)**2 + (Vz_a-Vz_b)**2))
        max_err = max(max_err, err)
    
    # Test equatorial belt: (1,E) ↔ (2,W) [N]
    for xi2_a in test_xi1_values:
        xi1_a = pi4
        xi1_b = -pi4  # E connects to W
        xi2_b = xi2_a  # [N] no change
        
        Xa, Ya, Za = equiangular_to_cartesian(xi1_a, xi2_a, 1)
        Xb, Yb, Zb = equiangular_to_cartesian(xi1_b, xi2_b, 2)
        pos_err = jnp.sqrt((Xa-Xb)**2 + (Ya-Yb)**2 + (Za-Zb)**2)
        assert float(pos_err) < 1e-10, f"Position mismatch (1,E)↔(2,W): {float(pos_err)}"
        
        u_a, v_a = 1.0, 0.5
        Vx_a, Vy_a, Vz_a = covariant_to_cartesian(u_a, v_a, xi1_a, xi2_a, 1)
        u_b, v_b = cartesian_to_covariant(Vx_a, Vy_a, Vz_a, xi1_b, xi2_b, 2)
        Vx_b, Vy_b, Vz_b = covariant_to_cartesian(u_b, v_b, xi1_b, xi2_b, 2)
        
        err = float(jnp.sqrt((Vx_a-Vx_b)**2 + (Vy_a-Vy_b)**2 + (Vz_a-Vz_b)**2))
        max_err = max(max_err, err)
    
    print(f"  Max Cartesian velocity error = {max_err:.2e}")
    assert max_err < 1e-10, f"Velocity not continuous"
    print("  ✓ PASSED (tested 5 edge types)")


def run_all_tests():
    """Run all velocity transform tests."""
    print("="*60)
    print("VELOCITY TRANSFORM UNIT TESTS")
    print("="*60)
    
    test_basis_vectors_orthogonal_to_normal()
    test_velocity_roundtrip()
    test_velocity_continuity_at_edges()
    
    print()
    print("="*60)
    print("✓ ALL VELOCITY TRANSFORM TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
