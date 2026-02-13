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

    Algorithm:
      1. X: covariant → Cartesian per panel (Jacobian: J = [a1|a2])
         (Wx, Wy, Wz) = w1 * a1 + w2 * a2
      2. Ah: project each Cartesian component across all panels
      3. Y: Cartesian → covariant per panel (pseudoinverse: (J^T J)^{-1} J^T)
         proj_i = W · a_i, then w_i = G^{ij} proj_j
         where G^{ij} is computed FROM the basis vectors to ensure consistency.
    """
    npanels = w1_h.shape[0]

    # Step 1: X — covariant → Cartesian (per panel)
    Wx = jnp.zeros_like(w1_h)
    Wy = jnp.zeros_like(w1_h)
    Wz = jnp.zeros_like(w1_h)

    for p in range(npanels):
        b = bases_h[p]
        Wx = Wx.at[p].set(w1_h[p] * b['a1x'] + w2_h[p] * b['a2x'])
        Wy = Wy.at[p].set(w1_h[p] * b['a1y'] + w2_h[p] * b['a2y'])
        Wz = Wz.at[p].set(w1_h[p] * b['a1z'] + w2_h[p] * b['a2z'])

    # Step 2: Ah — project each Cartesian component
    Wx = project_h(Wx)
    Wy = project_h(Wy)
    Wz = project_h(Wz)

    # Step 3: Y — Cartesian → covariant (pseudoinverse of Jacobian)
    # Compute G^{ij} from basis vectors directly (ensures X-Y consistency)
    w1_proj = jnp.zeros_like(w1_h)
    w2_proj = jnp.zeros_like(w2_h)

    for p in range(npanels):
        b = bases_h[p]
        # Dot products: proj_i = W · a_i
        proj1 = Wx[p] * b['a1x'] + Wy[p] * b['a1y'] + Wz[p] * b['a1z']
        proj2 = Wx[p] * b['a2x'] + Wy[p] * b['a2y'] + Wz[p] * b['a2z']

        # Covariant metric g_{ij} = a_i · a_j
        g11 = b['a1x']**2 + b['a1y']**2 + b['a1z']**2
        g12 = b['a1x']*b['a2x'] + b['a1y']*b['a2y'] + b['a1z']*b['a2z']
        g22 = b['a2x']**2 + b['a2y']**2 + b['a2z']**2
        det_g = g11 * g22 - g12**2

        # Contravariant metric G^{ij} = inverse of g_{ij}
        Ginv11 = g22 / det_g
        Ginv12 = -g12 / det_g
        Ginv22 = g11 / det_g

        # w_i = G^{ij} * proj_j
        w1_proj = w1_proj.at[p].set(Ginv11 * proj1 + Ginv12 * proj2)
        w2_proj = w2_proj.at[p].set(Ginv12 * proj1 + Ginv22 * proj2)

    return w1_proj, w2_proj


# ============================================================
# Full Coriolis operator (Eq. 63)
# ============================================================

def coriolis_tendency(v1, v2, f_h, Jh, J1, J2, Pcv, Pvc,
                      bases_h, project_h):
    """
    Compute Coriolis tendency using Eq. 63:
      F·v = J_v^{-1} P_hv V J_h^2 C P_vh v

    Args:
        v1: (6, N, N+1) covariant velocity component 1
        v2: (6, N+1, N) covariant velocity component 2
        f_h: (N+1, N+1) Coriolis parameter at h-points
        Jh: (N+1, N+1) Jacobian at h-points
        J1: (N, N+1) Jacobian at v1-points
        J2: (N+1, N) Jacobian at v2-points
        Pcv: (N+1, N) interpolation matrix center→vertex
        Pvc: (N, N+1) interpolation matrix vertex→center
        bases_h: list of 6 dicts with basis vectors at h-points
        project_h: scalar projection function

    Returns:
        dv1_cori: (6, N, N+1) Coriolis tendency for v1
        dv2_cori: (6, N+1, N) Coriolis tendency for v2
    """
    # Step 1: P_vh · v (interpolate covariant v to h-points)
    w1_h = jnp.einsum('ij,pjk->pik', Pcv, v1)    # (6, N+1, N+1)
    w2_h = jnp.einsum('pij,kj->pik', v2, Pcv)     # (6, N+1, N+1)

    # Step 2: C · w_h (Coriolis rotation, Eq. 58)
    c1_h = f_h[None, :, :] * w2_h     # +f * w2
    c2_h = -f_h[None, :, :] * w1_h    # -f * w1

    # Step 3: multiply by J_h^2
    Jh2 = Jh**2
    c1_h = c1_h * Jh2[None, :, :]
    c2_h = c2_h * Jh2[None, :, :]

    # Step 4: V operator (edge-continuity via Cartesian)
    c1_h, c2_h = apply_V(c1_h, c2_h, bases_h, project_h)

    # Step 5: P_hv (interpolate back to v-points)
    dv1 = jnp.einsum('ij,pjk->pik', Pvc, c1_h)   # (6, N, N+1)
    dv2 = jnp.einsum('pij,kj->pik', c2_h, Pvc)    # (6, N+1, N)

    # Step 6: J_v^{-1}
    dv1 = dv1 / J1[None, :, :]
    dv2 = dv2 / J2[None, :, :]

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


def test_energy_neutrality():
    """
    Test 3: Coriolis is energy-neutral.

    v^T Hv Jv Q (F·v) = 0    (Eq. 76, Section 5.4)

    This is the KEY property: Coriolis force does no work.
    Test with random fields and constant f.
    """
    print("\n" + "=" * 65)
    print("TEST 3: Coriolis energy neutrality")
    print("=" * 65)

    N = 12
    H0 = 1.0; g = 1.0
    sys_d = make_cubed_sphere_swe(N, H0, g)
    grids = sys_d['grids']
    metrics = sys_d['metrics']
    Pvc = sys_d['Pvc']; Pcv = sys_d['Pcv']
    Wh = sys_d['Wh']; W1 = sys_d['W1']; W2 = sys_d['W2']
    Jh = sys_d['Jh']; J1 = sys_d['J1']; J2 = sys_d['J2']
    project_h = sys_d['project_h']

    bases_h = compute_all_panel_bases(grids['xi1_h'], grids['xi2_h'])

    # Constant Coriolis parameter
    f_val = 1e-4
    f_h = f_val * jnp.ones((N + 1, N + 1))

    # Random covariant velocity field
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    v1 = 0.01 * jax.random.normal(k1, (6, N, N + 1))
    v2 = 0.01 * jax.random.normal(k2, (6, N + 1, N))

    # Compute Coriolis tendency
    dv1_c, dv2_c = coriolis_tendency(v1, v2, f_h, Jh, J1, J2,
                                      Pcv, Pvc, bases_h, project_h)

    # Energy rate: dKE/dt = H0 * v^T Hv Jv Q (Fv)
    # Apply Q to Fv (the Coriolis tendency), then dot with v
    dv1c_c, dv2c_c = jax.vmap(
        lambda dv1p, dv2p: compute_contravariant(dv1p, dv2p, metrics, Pvc, Pcv)
    )(dv1_c, dv2_c)

    dKE_cori = H0 * (
        float(jnp.sum(v1 * J1 * dv1c_c * W1[None])) +
        float(jnp.sum(v2 * J2 * dv2c_c * W2[None]))
    )

    # KE for normalization: apply Q to v
    v1c, v2c = jax.vmap(
        lambda v1p, v2p: compute_contravariant(v1p, v2p, metrics, Pvc, Pcv)
    )(v1, v2)

    KE = 0.5 * H0 * (
        float(jnp.sum(v1 * J1 * v1c * W1[None])) +
        float(jnp.sum(v2 * J2 * v2c * W2[None]))
    )

    print(f"  KE              = {KE:.6e}")
    print(f"  dKE/dt (Coriolis) = {dKE_cori:.6e}")
    if KE > 0:
        print(f"  |dKE/dt| / KE   = {abs(dKE_cori)/KE:.6e}")

    # Test with DIFFERENT random fields to make sure it's not accidental
    results = []
    for seed in [42, 123, 999, 2025]:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        v1_t = 0.01 * jax.random.normal(k1, (6, N, N + 1))
        v2_t = 0.01 * jax.random.normal(k2, (6, N + 1, N))

        dv1_t, dv2_t = coriolis_tendency(v1_t, v2_t, f_h, Jh, J1, J2,
                                          Pcv, Pvc, bases_h, project_h)
        dv1c_t, dv2c_t = jax.vmap(
            lambda dv1p, dv2p: compute_contravariant(dv1p, dv2p, metrics, Pvc, Pcv)
        )(dv1_t, dv2_t)

        dKE_t = H0 * (
            float(jnp.sum(v1_t * J1 * dv1c_t * W1[None])) +
            float(jnp.sum(v2_t * J2 * dv2c_t * W2[None]))
        )

        v1c_t, v2c_t = jax.vmap(
            lambda v1p, v2p: compute_contravariant(v1p, v2p, metrics, Pvc, Pcv)
        )(v1_t, v2_t)
        KE_t = 0.5 * H0 * (
            float(jnp.sum(v1_t * J1 * v1c_t * W1[None])) +
            float(jnp.sum(v2_t * J2 * v2c_t * W2[None]))
        )
        ratio = abs(dKE_t) / KE_t if KE_t > 0 else 0.0
        results.append(ratio)

    max_ratio = max(results)
    print(f"\n  Max |dKE/dt|/KE across 4 seeds: {max_ratio:.2e}")

    ok = max_ratio < 1e-12
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


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
