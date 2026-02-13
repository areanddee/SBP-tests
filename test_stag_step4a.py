"""
test_stag_step4a.py — Coriolis Operator (Eq. 63) Unit Tests
============================================================

GOAL: Build and validate the energy-conserving, edge-continuous Coriolis
operator from Shashkin Eq. 63 before integrating into the full RHS.

Eq. 63: F = J_v^{-1} P_hv V J_h^2 C P_vh

where:
  P_vh: interpolate covariant velocity from v-points to h-points
  C:    Coriolis rotation [[0, f], [-f, 0]] at h-points  (Eq. 58)
  J_h^2: Jacobian squared at h-points
  V:    edge-continuity operator Y [Ah; Ah; Ah] X  (Eq. 60)
        X = covariant → Cartesian, Ah = scalar projection, Y = Cartesian → covariant
  P_hv: interpolate from h-points back to v-points
  J_v^{-1}: inverse Jacobian at v-points

Algorithm for Coriolis tendency on covariant velocity v:
  1. w_h = P_vh · v           (interp to h-points)
  2. c_h = C · w_h            (Coriolis rotation)
  3. c_h *= J_h^2             (scale)
  4. c_h = V(c_h)             (project to edge-continuous via Cartesian)
  5. result = P_hv · c_h      (interp back to v-points)
  6. result *= J_v^{-1}       (divide by Jacobian)

TESTS:
  1. Basis vectors: a_i tangent to sphere, covariant metric consistent
  2. V operator: makes vector field edge-continuous
  3. Energy neutrality: v^T Hv Jv Q (Fv) = 0  (the key property!)
  4. Zero velocity → zero tendency
  5. Coriolis with time integration: energy still dt^5

Reference: Shashkin 2025, Sections 4.5, 5.4, Eq. 57-63

Usage:
    python test_stag_step4a.py
"""
import sys, os
project_dir = '/mnt/project'
if os.path.isdir(project_dir):
    sys.path.insert(0, project_dir)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from sbp_staggered_1d import sbp_42
from grid import equiangular_to_cartesian
from velocity_transforms import get_covariant_basis

# Import Step 4 infrastructure
from test_stag_step4 import (
    EDGES, make_staggered_grids, make_all_metrics,
    compute_metric, compute_contravariant,
    build_projection_fn, build_sat_fn,
    make_cubed_sphere_swe, make_rk4_step,
    compute_mass, compute_energy, _reverses,
)


# ============================================================
# Covariant basis vectors on 2D grids (per panel)
# ============================================================

def compute_basis_on_grid(xi1_2d, xi2_2d, face_id):
    """
    Compute covariant basis vectors a1, a2 at all grid points on a panel.

    Args:
        xi1_2d, xi2_2d: 2D coordinate arrays (any shape)
        face_id: panel index 0-5

    Returns:
        dict with 'a1x','a1y','a1z','a2x','a2y','a2z' arrays (same shape as input)
    """
    a1, a2 = get_covariant_basis(xi1_2d, xi2_2d, face_id)
    return {
        'a1x': a1[0], 'a1y': a1[1], 'a1z': a1[2],
        'a2x': a2[0], 'a2y': a2[1], 'a2z': a2[2],
    }


def compute_all_panel_bases(xi1_2d, xi2_2d):
    """
    Compute basis vectors at h-points for all 6 panels.

    Returns: list of 6 dicts, each from compute_basis_on_grid.
    """
    return [compute_basis_on_grid(xi1_2d, xi2_2d, p) for p in range(6)]


# ============================================================
# V operator (Eq. 60): edge-continuity via Cartesian
# ============================================================

def apply_V(w1_h, w2_h, bases_h, project_h):
    """
    Apply V = Y [Ah; Ah; Ah] X to a covariant vector field at h-points.

    w1_h, w2_h: (6, N+1, N+1) covariant components at h-points
    bases_h: list of 6 dicts with basis vectors at h-points
    project_h: scalar projection function (averaging at edges/corners)

    X: covariant → Cartesian via V = v_i a^i (contravariant basis)
       Equivalently: raise index v^i = G^{ij} v_j, then W = v^i a_i
    Ah: project each Cartesian component
    Y: Cartesian → covariant via w_i = W · a_i (dot product, no metric)

    Y·X = I because a_i · a^j = δ_i^j
    """
    npanels = w1_h.shape[0]

    # Step 1: X — covariant → Cartesian
    # First raise index: v^i = G^{ij} v_j (computed from basis vectors)
    # Then: W = v^1 a_1 + v^2 a_2
    Wx = jnp.zeros_like(w1_h)
    Wy = jnp.zeros_like(w1_h)
    Wz = jnp.zeros_like(w1_h)

    for p in range(npanels):
        b = bases_h[p]
        # Covariant metric g_{ij} = a_i · a_j
        g11 = b['a1x']**2 + b['a1y']**2 + b['a1z']**2
        g12 = b['a1x']*b['a2x'] + b['a1y']*b['a2y'] + b['a1z']*b['a2z']
        g22 = b['a2x']**2 + b['a2y']**2 + b['a2z']**2
        det_g = g11 * g22 - g12**2

        # Contravariant metric G^{ij}
        Ginv11 = g22 / det_g
        Ginv12 = -g12 / det_g
        Ginv22 = g11 / det_g

        # Raise index: v^i = G^{ij} v_j
        v_up1 = Ginv11 * w1_h[p] + Ginv12 * w2_h[p]
        v_up2 = Ginv12 * w1_h[p] + Ginv22 * w2_h[p]

        # Cartesian: W = v^1 a_1 + v^2 a_2
        Wx = Wx.at[p].set(v_up1 * b['a1x'] + v_up2 * b['a2x'])
        Wy = Wy.at[p].set(v_up1 * b['a1y'] + v_up2 * b['a2y'])
        Wz = Wz.at[p].set(v_up1 * b['a1z'] + v_up2 * b['a2z'])

    # Step 2: Ah — project each Cartesian component
    Wx = project_h(Wx)
    Wy = project_h(Wy)
    Wz = project_h(Wz)

    # Step 3: Y — Cartesian → covariant (just dot product, no metric)
    # w_i = W · a_i
    w1_proj = jnp.zeros_like(w1_h)
    w2_proj = jnp.zeros_like(w2_h)

    for p in range(npanels):
        b = bases_h[p]
        w1_proj = w1_proj.at[p].set(
            Wx[p] * b['a1x'] + Wy[p] * b['a1y'] + Wz[p] * b['a1z'])
        w2_proj = w2_proj.at[p].set(
            Wx[p] * b['a2x'] + Wy[p] * b['a2y'] + Wz[p] * b['a2z'])

    return w1_proj, w2_proj


# ============================================================
# Full Coriolis operator (Eq. 63)
# ============================================================

def coriolis_tendency(v1, v2, f_h, Jh, J1, J2, Pcv, Pvc,
                      bases_h, project_h, use_V=True):
    """
    Compute Coriolis tendency.

    Eq. 62 (use_V=True):  F = P_hv V C P_vh J_v
      Derived from energy-neutral Eq. 59 on cubed sphere (J commutes with V).
      1. w = J_v · v           (multiply by J at v-points)
      2. w_h = P_vh · w        (interpolate to h-points)
      3. c_h = C · w_h         (Coriolis rotation)
      4. c_h = V(c_h)          (edge-continuity projection)
      5. F = P_hv · c_h        (interpolate back to v-points)

    Eq. 57 (use_V=False): F = P_hv C P_vh J_v
      Same but skip V (edge-discontinuous).
    """
    # Step 1: multiply by J at v-points
    w1 = v1 * J1[None, :, :]
    w2 = v2 * J2[None, :, :]

    # Step 2: interpolate to h-points
    w1_h = jnp.einsum('ij,pjk->pik', Pcv, w1)    # (6, N+1, N+1)
    w2_h = jnp.einsum('pij,kj->pik', w2, Pcv)     # (6, N+1, N+1)

    # Step 3: Coriolis rotation at h-points
    c1_h = f_h[None, :, :] * w2_h      # +f * w2
    c2_h = -f_h[None, :, :] * w1_h     # -f * w1

    # Step 4: V operator (edge-continuity via Cartesian)
    if use_V:
        c1_h, c2_h = apply_V(c1_h, c2_h, bases_h, project_h)

    # Step 5: interpolate back to v-points
    dv1 = jnp.einsum('ij,pjk->pik', Pvc, c1_h)   # (6, N, N+1)
    dv2 = jnp.einsum('pij,kj->pik', c2_h, Pvc)    # (6, N+1, N)

    return dv1, dv2


# ============================================================
# Tests
# ============================================================

def test_basis_vectors():
    """
    Test 1: Verify covariant basis vectors.
    - a1, a2 tangent to sphere (r · ai = 0)
    - Covariant metric g_{ij} positive definite (det > 0)
    - V roundtrip: apply_V to already-continuous field → unchanged
    """
    print("\n" + "=" * 65)
    print("TEST 1: Basis vectors & V roundtrip")
    print("=" * 65)

    N = 8
    grids = make_staggered_grids(N)
    xi1 = grids['xi1_h']
    xi2 = grids['xi2_h']

    max_tangent_err = 0.0
    min_det = 1e10

    for face_id in range(6):
        X, Y, Z = equiangular_to_cartesian(xi1, xi2, face_id)
        b = compute_basis_on_grid(xi1, xi2, face_id)

        # Tangent: r · a_i = 0
        dot1 = X * b['a1x'] + Y * b['a1y'] + Z * b['a1z']
        dot2 = X * b['a2x'] + Y * b['a2y'] + Z * b['a2z']
        max_tangent_err = max(max_tangent_err,
                              float(jnp.max(jnp.abs(dot1))),
                              float(jnp.max(jnp.abs(dot2))))

        # Positive definite: det(g) > 0
        g11 = b['a1x']**2 + b['a1y']**2 + b['a1z']**2
        g12 = b['a1x']*b['a2x'] + b['a1y']*b['a2y'] + b['a1z']*b['a2z']
        g22 = b['a2x']**2 + b['a2y']**2 + b['a2z']**2
        det_g = g11 * g22 - g12**2
        min_det = min(min_det, float(jnp.min(det_g)))

    print(f"  Max |r · a_i| (tangent check): {max_tangent_err:.2e}")
    print(f"  Min det(g_ij) (pos. definite): {min_det:.6f}")

    # V roundtrip: uniform Cartesian field V=(1,0,0) → apply V → unchanged
    metrics = make_all_metrics(grids)
    Hv_diag = jnp.diag(sbp_42(N, float(grids['dx'])).Hv)
    project_h, _ = build_projection_fn(N, metrics['Jh'], Hv_diag)
    bases_h = compute_all_panel_bases(xi1, xi2)

    # Set covariant components for V=(1,0,0) on each panel
    w1_h = jnp.zeros((6, N + 1, N + 1))
    w2_h = jnp.zeros((6, N + 1, N + 1))
    for p in range(6):
        b = bases_h[p]
        # Covariant component: w_i = V · a_i (projection onto basis)
        w1_h = w1_h.at[p].set(b['a1x'])
        w2_h = w2_h.at[p].set(b['a2x'])

    # Apply V (should be identity for a globally continuous field)
    w1_proj, w2_proj = apply_V(w1_h, w2_h, bases_h, project_h)

    # Interior points should be unchanged (boundaries get averaged)
    rt_err = max(
        float(jnp.max(jnp.abs(w1_proj[:, 1:-1, 1:-1] - w1_h[:, 1:-1, 1:-1]))),
        float(jnp.max(jnp.abs(w2_proj[:, 1:-1, 1:-1] - w2_h[:, 1:-1, 1:-1]))),
    )
    print(f"  V roundtrip error (interior):   {rt_err:.2e}")

    ok = max_tangent_err < 1e-13 and min_det > 0.01 and rt_err < 1e-12
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_V_edge_continuity():
    """
    Test 2: V operator makes a vector field edge-continuous.

    Start with a discontinuous covariant vector field, apply V,
    check that shared edges now agree.
    """
    print("\n" + "=" * 65)
    print("TEST 2: V operator edge continuity")
    print("=" * 65)

    N = 8
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)
    Hv_diag = jnp.diag(sbp_42(N, float(grids['dx'])).Hv)
    project_h, corners = build_projection_fn(N, metrics['Jh'], Hv_diag)
    bases_h = compute_all_panel_bases(grids['xi1_h'], grids['xi2_h'])

    # Create discontinuous vector field: set to physical velocity V = (1,0,0)
    # Convert to covariant on each panel → naturally discontinuous at edges
    w1_h = jnp.zeros((6, N + 1, N + 1))
    w2_h = jnp.zeros((6, N + 1, N + 1))

    for p in range(6):
        b = bases_h[p]
        # V = (1, 0, 0) Cartesian → covariant: w1 = V·a1, w2 = V·a2
        w1_h = w1_h.at[p].set(b['a1x'])
        w2_h = w2_h.at[p].set(b['a2x'])

    # Check discontinuity BEFORE V
    max_disc_before = 0.0
    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        # Extract Cartesian at boundaries and compare
        for comp_name, arr in [('x', 'a1x'), ('y', 'a1y'), ('z', 'a1z')]:
            pass  # We'll check via Cartesian reconstruction

    # Convert to Cartesian, check edge agreement
    def to_cartesian_boundary(w1, w2, bases, panel, edge, N):
        b = bases[panel]
        Wx = w1[panel] * b['a1x'] + w2[panel] * b['a2x']
        Wy = w1[panel] * b['a1y'] + w2[panel] * b['a2y']
        Wz = w1[panel] * b['a1z'] + w2[panel] * b['a2z']
        if edge == 'N':
            return Wx[:, N], Wy[:, N], Wz[:, N]
        elif edge == 'S':
            return Wx[:, 0], Wy[:, 0], Wz[:, 0]
        elif edge == 'E':
            return Wx[N, :], Wy[N, :], Wz[N, :]
        elif edge == 'W':
            return Wx[0, :], Wy[0, :], Wz[0, :]

    # Check edge continuity in CARTESIAN SPACE (should be continuous after V)
    w1_proj, w2_proj = apply_V(w1_h, w2_h, bases_h, project_h)

    max_cart_err = 0.0
    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        Wxa, Wya, Wza = to_cartesian_boundary(w1_proj, w2_proj, bases_h, pa, ea, N)
        Wxb, Wyb, Wzb = to_cartesian_boundary(w1_proj, w2_proj, bases_h, pb, eb, N)
        if rev:
            Wxb, Wyb, Wzb = Wxb[::-1], Wyb[::-1], Wzb[::-1]
        err = max(float(jnp.max(jnp.abs(Wxa - Wxb))),
                  float(jnp.max(jnp.abs(Wya - Wyb))),
                  float(jnp.max(jnp.abs(Wza - Wzb))))
        max_cart_err = max(max_cart_err, err)

    print(f"  Max Cartesian edge discontinuity after V: {max_cart_err:.2e}")

    ok = max_cart_err < 1e-13
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_energy_neutrality_flat():
    """
    Test 3a: Coriolis energy neutrality on a SINGLE FLAT PANEL.

    J=1, Q11=Q22=1, Q12=0. No boundaries, no V, no metric complications.
    Eq. 57 reduces to: F = P_hv C P_vh v
    Energy neutrality: v^T H_v (Fv) = 0
      follows from Pcv^T Hv = Hc Pvc and anti-symmetry of C.

    If this fails → code bug in SBP operators or Coriolis rotation.
    If this passes → issue is cubed-sphere geometry.
    """
    print("\n" + "=" * 65)
    print("TEST 3a: Coriolis energy neutrality — FLAT PANEL")
    print("=" * 65)

    N = 12
    dx = jnp.pi / (2 * N)
    ops = sbp_42(N, float(dx))
    Pcv = ops.Pcv
    Pvc = ops.Pvc
    Hv_diag = jnp.diag(ops.Hv)
    Hc_diag = jnp.diag(ops.Hc)

    # Quadrature weights for staggered 2D grid (flat, J=1)
    W1 = jnp.outer(Hc_diag, Hv_diag)  # (N, N+1)
    W2 = jnp.outer(Hv_diag, Hc_diag)  # (N+1, N)

    f_val = 1.0  # Large f to amplify any leak
    f_h = f_val * jnp.ones((N + 1, N + 1))

    def flat_coriolis(v1, v2):
        """F = P_hv C P_vh v on single panel, J=1."""
        # Interpolate to h-points
        w1_h = Pcv @ v1       # (N+1, N+1)
        w2_h = v2 @ Pcv.T    # (N+1, N+1)
        # Coriolis rotation
        c1_h = f_h * w2_h
        c2_h = -f_h * w1_h
        # Interpolate back
        dv1 = Pvc @ c1_h     # (N, N+1)
        dv2 = c2_h @ Pvc.T   # (N+1, N)
        return dv1, dv2

    def flat_KE(v1, v2):
        """KE = (1/2) sum(v1^2 W1 + v2^2 W2) with J=1, Q=I."""
        return 0.5 * (jnp.sum(v1**2 * W1) + jnp.sum(v2**2 * W2))

    max_ratio = 0.0
    for seed in [42, 123, 999, 2025]:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        v1 = jax.random.normal(k1, (N, N + 1))
        v2 = jax.random.normal(k2, (N + 1, N))

        dv1, dv2 = flat_coriolis(v1, v2)

        # Finite difference
        eps = 1e-5
        KE_plus  = flat_KE(v1 + eps * dv1, v2 + eps * dv2)
        KE_minus = flat_KE(v1 - eps * dv1, v2 - eps * dv2)
        dKE = float((KE_plus - KE_minus) / (2 * eps))
        KE = float(flat_KE(v1, v2))
        ratio = abs(dKE) / KE if KE > 0 else 0.0
        max_ratio = max(max_ratio, ratio)

    # Also test via exact inner product: v^T H Fv
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    v1 = jax.random.normal(k1, (N, N + 1))
    v2 = jax.random.normal(k2, (N + 1, N))
    dv1, dv2 = flat_coriolis(v1, v2)
    exact_ip = float(jnp.sum(v1 * dv1 * W1) + jnp.sum(v2 * dv2 * W2))
    KE = float(flat_KE(v1, v2))

    print(f"  FD:    max|dKE/dt|/KE = {max_ratio:.2e}")
    print(f"  Exact: v^T H Fv / KE  = {abs(exact_ip)/KE:.2e}")

    ok = max_ratio < 1e-12 and abs(exact_ip)/KE < 1e-13
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_energy_neutrality():
    """
    Test 3b: Coriolis energy neutrality on CUBED SPHERE.

    Test Eq. 57 and Eq. 62 with full and diagonal KE norms.
    """
    print("\n" + "=" * 65)
    print("TEST 3b: Coriolis energy neutrality — CUBED SPHERE")
    print("=" * 65)

    N = 12
    H0 = 1.0; g = 1.0
    sys_d = make_cubed_sphere_swe(N, H0, g)
    grids = sys_d['grids']
    metrics = sys_d['metrics']
    Pvc = sys_d['Pvc']; Pcv = sys_d['Pcv']
    W1 = sys_d['W1']; W2 = sys_d['W2']
    Jh = sys_d['Jh']; J1 = sys_d['J1']; J2 = sys_d['J2']
    project_h = sys_d['project_h']

    bases_h = compute_all_panel_bases(grids['xi1_h'], grids['xi2_h'])

    f_val = 1e-4
    f_h = f_val * jnp.ones((N + 1, N + 1))

    def kinetic_energy_full(v1, v2):
        """Full KE with Q12 cross-terms (Eq. 64)."""
        v1c, v2c = jax.vmap(
            lambda v1p, v2p: compute_contravariant(v1p, v2p, metrics, Pvc, Pcv)
        )(v1, v2)
        return 0.5 * H0 * (
            jnp.sum(v1 * J1 * v1c * W1[None]) +
            jnp.sum(v2 * J2 * v2c * W2[None])
        )

    def kinetic_energy_diag(v1, v2):
        """Diagonal-only KE (Q12=0)."""
        Q11 = metrics['Q11_1']
        Q22 = metrics['Q22_2']
        return 0.5 * H0 * (
            jnp.sum(v1**2 * J1 * Q11 * W1[None]) +
            jnp.sum(v2**2 * J2 * Q22 * W2[None])
        )

    all_ok = True
    for label, use_V, ke_fn in [
        ("Eq57, full KE",  False, kinetic_energy_full),
        ("Eq57, diag KE",  False, kinetic_energy_diag),
        ("Eq62, full KE",  True,  kinetic_energy_full),
        ("Eq62, diag KE",  True,  kinetic_energy_diag),
    ]:
        max_ratio = 0.0
        for seed in [42, 123, 999, 2025]:
            key = jax.random.PRNGKey(seed)
            k1, k2 = jax.random.split(key)
            v1 = 0.01 * jax.random.normal(k1, (6, N, N + 1))
            v2 = 0.01 * jax.random.normal(k2, (6, N + 1, N))

            dv1_c, dv2_c = coriolis_tendency(v1, v2, f_h, Jh, J1, J2,
                                              Pcv, Pvc, bases_h, project_h,
                                              use_V=use_V)
            eps = 1e-5
            KE_plus  = ke_fn(v1 + eps * dv1_c, v2 + eps * dv2_c)
            KE_minus = ke_fn(v1 - eps * dv1_c, v2 - eps * dv2_c)
            dKE = float((KE_plus - KE_minus) / (2 * eps))
            KE = float(ke_fn(v1, v2))
            ratio = abs(dKE) / KE if KE > 0 else 0.0
            max_ratio = max(max_ratio, ratio)

        status = '✓' if max_ratio < 1e-12 else '✗'
        print(f"  {status} {label:20s}  max|dKE/dt|/KE = {max_ratio:.2e}")
        if max_ratio > 1e-12:
            all_ok = False

    print(f"\n  {'✓ PASS' if all_ok else '✗ FAIL (see diagnostics above)'}")
    return all_ok


def test_zero_velocity():
    """
    Test 4: Zero velocity gives zero Coriolis tendency.
    """
    print("\n" + "=" * 65)
    print("TEST 4: Zero velocity → zero tendency")
    print("=" * 65)

    N = 8
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)
    dx = float(grids['dx'])
    ops = sbp_42(N, dx)
    Hv_diag = jnp.diag(ops.Hv)
    project_h, _ = build_projection_fn(N, metrics['Jh'], Hv_diag)
    bases_h = compute_all_panel_bases(grids['xi1_h'], grids['xi2_h'])

    f_h = 1e-4 * jnp.ones((N + 1, N + 1))
    v1 = jnp.zeros((6, N, N + 1))
    v2 = jnp.zeros((6, N + 1, N))

    dv1, dv2 = coriolis_tendency(v1, v2, f_h, metrics['Jh'],
                                  metrics['J1'], metrics['J2'],
                                  ops.Pcv, ops.Pvc, bases_h, project_h)

    max_dv1 = float(jnp.max(jnp.abs(dv1)))
    max_dv2 = float(jnp.max(jnp.abs(dv2)))
    print(f"  max|dv1| = {max_dv1:.2e}")
    print(f"  max|dv2| = {max_dv2:.2e}")

    ok = max_dv1 < 1e-15 and max_dv2 < 1e-15
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_energy_dt_scaling_with_coriolis():
    """
    Test 5: Energy CFL sweep with Coriolis included.

    Since Coriolis is energy-neutral in the semi-discrete system,
    the energy error should still scale as dt^5 (RK4 on Hamiltonian).
    """
    print("\n" + "=" * 65)
    print("TEST 5: Energy dt-scaling with Coriolis (f = 1e-4)")
    print("=" * 65)

    N = 12; H0 = 1.0; g = 1.0
    sys_d = make_cubed_sphere_swe(N, H0, g)
    grids = sys_d['grids']
    metrics = sys_d['metrics']
    ops = sys_d['ops']
    Dvc = ops.Dvc; Dcv = ops.Dcv
    Pvc = sys_d['Pvc']; Pcv = sys_d['Pcv']
    Wh = sys_d['Wh']; W1 = sys_d['W1']; W2 = sys_d['W2']
    Jh = sys_d['Jh']; J1 = sys_d['J1']; J2 = sys_d['J2']
    Jh_inv = 1.0 / Jh
    project_h = sys_d['project_h']
    add_sat = build_sat_fn(N, ops, metrics)
    bases_h = compute_all_panel_bases(grids['xi1_h'], grids['xi2_h'])

    f_h = 1e-4 * jnp.ones((N + 1, N + 1))

    # RHS with Coriolis
    def rhs_cori(h, v1, v2):
        h_proj = project_h(h)

        # Gradient
        dv1_dt = -g * jnp.einsum('ij,pjk->pik', Dvc, h_proj)
        dv2_dt = -g * jnp.einsum('pij,kj->pik', h_proj, Dvc)

        # Coriolis tendency
        dv1_c, dv2_c = coriolis_tendency(v1, v2, f_h, Jh, J1, J2,
                                          Pcv, Pvc, bases_h, project_h)
        dv1_dt = dv1_dt + dv1_c
        dv2_dt = dv2_dt + dv2_c

        # Contravariant velocity + divergence (unchanged from step 4)
        v1c, v2c = jax.vmap(
            lambda v1p, v2p: compute_contravariant(v1p, v2p, metrics, Pvc, Pcv)
        )(v1, v2)
        u1 = J1 * v1c
        u2 = J2 * v2c

        div = (jnp.einsum('ij,pjk->pik', Dcv, u1) +
               jnp.einsum('pij,kj->pik', u2, Dcv))
        div = add_sat(div, u1, u2)

        dh_dt = project_h(-H0 * Jh_inv * div)
        return dh_dt, dv1_dt, dv2_dt

    rhs_jit = jax.jit(rhs_cori)
    step_fn = make_rk4_step(rhs_jit)

    # IC: Gaussian on panel 0
    xi_v = grids['xi_v']
    xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')
    h0 = jnp.zeros((6, N + 1, N + 1))
    h0 = h0.at[0].set(0.1 * jnp.exp(-(xi1_2d**2 + xi2_2d**2) / (2 * 0.2**2)))
    v1_0 = jnp.zeros((6, N, N + 1))
    v2_0 = jnp.zeros((6, N + 1, N))

    dx = float(grids['dx'])
    c = np.sqrt(g * H0)
    T_end = jnp.pi / 3  # short integration

    print(f"  N = {N}, T = {float(T_end):.4f}")

    # JIT warmup
    import time as _time
    print(f"  JIT compiling...", flush=True)
    t0 = _time.time()
    _h, _v1, _v2 = step_fn(h0, v1_0, v2_0, 0.01 * dx / c)
    jax.block_until_ready(_h)
    print(f"  JIT compiled in {_time.time()-t0:.1f}s", flush=True)

    CFLs = [0.4, 0.2, 0.1, 0.05]
    dE_list = []

    print(f"\n  {'CFL':>6} {'dt':>12} {'steps':>7} {'ΔE/E':>12} {'rate':>8}")
    print(f"  " + "-" * 50)

    for CFL in CFLs:
        dt = CFL * dx / c
        nsteps = int(np.ceil(T_end / dt))
        dt = T_end / nsteps

        h, v1, v2 = h0, v1_0, v2_0
        E0 = compute_energy(h, v1, v2, Wh, W1, W2, Jh, J1, J2, g, H0,
                            metrics, Pvc, Pcv)

        for s in range(nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)

        E_f = compute_energy(h, v1, v2, Wh, W1, W2, Jh, J1, J2, g, H0,
                             metrics, Pvc, Pcv)
        dE = abs(E_f - E0) / abs(E0)
        dE_list.append(dE)

        if len(dE_list) > 1:
            rate = np.log(dE_list[-2] / dE_list[-1]) / np.log(2)
        else:
            rate = float('nan')

        print(f"  {CFL:6.3f} {float(dt):12.6e} {nsteps:7d}   {dE:12.4e} {rate:8.2f}")

    final_rate = np.log(dE_list[-2] / dE_list[-1]) / np.log(2)
    print(f"\n  dt-scaling rate: {final_rate:.2f}")

    ok = final_rate > 4.0
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  STEP 4a: Coriolis Operator (Eq. 63) — Unit Tests")
    print("  Reference: Shashkin 2025, Sections 4.5, 5.4")
    print("=" * 65)

    results = {}
    results['basis']    = test_basis_vectors()
    results['V_cont']   = test_V_edge_continuity()
    results['flat_cor'] = test_energy_neutrality_flat()
    results['energy_n'] = test_energy_neutrality()
    results['zero_v']   = test_zero_velocity()
    results['dt_scale'] = test_energy_dt_scaling_with_coriolis()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    all_pass = True
    for name, ok in results.items():
        status = '✓ PASS' if ok else '✗ FAIL'
        print(f"  {name:15s} {status}")
        all_pass = all_pass and ok

    if all_pass:
        print(f"\n  All tests passed. Ready for Step 4b.")
    else:
        print(f"\n  Some tests FAILED. Debug before proceeding.")
    print("=" * 65)
