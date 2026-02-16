"""
Test: SAT flux continuity for an analytic vector field on the cubed sphere.

Strategy:
  1. Define a smooth Cartesian vector field V = (Vx, Vy, Vz) on the sphere
     (e.g., solid body rotation, or a dipole field).
  2. Project to covariant velocities (v1, v2) on each panel's staggered grid.
  3. Compute contravariant velocity and mass flux u1 = J1*v1c, u2 = J2*v2c.
  4. Extrapolate flux to each panel edge using l, r operators.
  5. For each shared edge: check that the extrapolated fluxes from the two
     panels represent the SAME physical normal flux (after alignment/sign).

If the SAT averaging has incorrect sign handling for axis-swapped edges
(T, TR operations), the fluxes will disagree and this test will catch it.

The physical normal flux at a shared edge should satisfy:
  flux_from_panel_A + sign_a * sign_b * flux_from_panel_B_aligned = 0
because the outward normals point in opposite directions.
"""

import sys
sys.path.insert(0, '/mnt/project')
sys.path.insert(0, '/mnt/user-data/outputs')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from sbp_staggered_1d import sbp_42
from grid import equiangular_to_cartesian
from velocity_transforms import cartesian_to_covariant, get_covariant_basis
from test_stag_step4 import (
    EDGES, _reverses, make_staggered_grids, make_all_metrics,
    compute_metric, compute_contravariant, _hv_inv_index
)


# ============================================================
# Analytic vector fields on the sphere
# ============================================================

def solid_body_rotation(X, Y, Z, axis='z'):
    """Solid body rotation V = Omega x r.  |V| = cos(lat) for z-axis."""
    if axis == 'z':
        return -Y, X, jnp.zeros_like(Z)      # Omega = (0,0,1)
    elif axis == 'x':
        return jnp.zeros_like(X), -Z, Y       # Omega = (1,0,0)
    elif axis == 'y':
        return Z, jnp.zeros_like(Y), -X       # Omega = (0,1,0)


def smooth_dipole(X, Y, Z):
    """Smooth divergence-free field: V = grad(Y) x r (on unit sphere).
    Gives a field with structure near ALL cube vertices."""
    # grad(Y) on the sphere = Y_hat - Y*r_hat (projection to tangent plane)
    # V = (grad(Y) x r) has components derived from Y = sin(lat)*cos(lon)
    # Use a simpler approach: V = (-Z*Y, Z*X, 0) projected to sphere tangent
    Vx = -Z * Y
    Vy = Z * X
    Vz = X * Y  # This makes it non-trivial on all panels
    # Project to tangent plane: V_tangent = V - (V·r)*r
    Vr = Vx * X + Vy * Y + Vz * Z
    return Vx - Vr * X, Vy - Vr * Y, Vz - Vr * Z


# ============================================================
# Compute mass flux from analytic field
# ============================================================

def compute_analytic_mass_flux(N, field_fn):
    """
    Given an analytic Cartesian vector field, compute the covariant
    velocity and mass flux on the staggered cubed-sphere grid.
    
    Returns u1_all (6,N,N+1), u2_all (6,N+1,N), and ops.
    """
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)
    dx = float(grids['dx'])
    ops = sbp_42(N, dx)
    Pvc = ops.Pvc
    Pcv = ops.Pcv

    # v1 lives at (xi_c, xi_v), shape (N, N+1)
    # v2 lives at (xi_v, xi_c), shape (N+1, N)
    xi1_v1 = grids['xi1_v1']
    xi2_v1 = grids['xi2_v1']
    xi1_v2 = grids['xi1_v2']
    xi2_v2 = grids['xi2_v2']

    v1_all = []
    v2_all = []

    for panel in range(6):
        # Cartesian coords at v1 grid points
        X1, Y1, Z1 = equiangular_to_cartesian(xi1_v1, xi2_v1, panel)
        Vx1, Vy1, Vz1 = field_fn(X1, Y1, Z1)
        v1_cov, _ = cartesian_to_covariant(Vx1, Vy1, Vz1, xi1_v1, xi2_v1, panel)

        # Cartesian coords at v2 grid points
        X2, Y2, Z2 = equiangular_to_cartesian(xi1_v2, xi2_v2, panel)
        Vx2, Vy2, Vz2 = field_fn(X2, Y2, Z2)
        _, v2_cov = cartesian_to_covariant(Vx2, Vy2, Vz2, xi1_v2, xi2_v2, panel)

        v1_all.append(v1_cov)
        v2_all.append(v2_cov)

    v1 = jnp.stack(v1_all)  # (6, N, N+1)
    v2 = jnp.stack(v2_all)  # (6, N+1, N)

    # Compute contravariant velocity
    def contra_single(v1_p, v2_p):
        return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
    contra_vmap = jax.vmap(contra_single)
    v1c, v2c = contra_vmap(v1, v2)

    # Mass flux
    J1 = metrics['J1']
    J2 = metrics['J2']
    u1 = J1 * v1c  # (6, N, N+1)
    u2 = J2 * v2c  # (6, N+1, N)

    return u1, u2, v1, v2, ops, grids, metrics


# ============================================================
# TEST 1: Physical flux continuity at shared edges
# ============================================================

def test_flux_continuity(N, field_fn, field_name):
    """
    For each shared edge, extrapolate the mass flux from both panels
    and check that they represent the same physical flux.
    
    The SAT formula uses:
      sat_a = -sign_a * 0.5 * Hv_inv * (flux_a + ss * flux_b_aligned)
    
    For energy conservation, we need:  flux_a + ss * flux_b_aligned ≈ 0
    when the solution is exact (continuous, divergence cancels).
    
    The "SAT mismatch" = flux_a + ss * flux_b_aligned tells us how well
    the fluxes cancel. For a smooth field, this should be O(h^p) where
    p is the extrapolation order at the boundary.
    """
    u1, u2, v1, v2, ops, grids, metrics = compute_analytic_mass_flux(N, field_fn)

    l = ops.l
    r = ops.r
    Hv_diag = jnp.diag(ops.Hv)

    def extrapolate_flux(u1, u2, panel, edge):
        if edge == 'E':
            return jnp.einsum('c,cj->j', r, u1[panel])
        elif edge == 'W':
            return jnp.einsum('c,cj->j', l, u1[panel])
        elif edge == 'N':
            return jnp.einsum('ic,c->i', u2[panel], r)
        elif edge == 'S':
            return jnp.einsum('ic,c->i', u2[panel], l)

    def boundary_sign(edge):
        return +1.0 if edge in ('E', 'N') else -1.0

    results = []

    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        sign_a = boundary_sign(ea)
        sign_b = boundary_sign(eb)
        ss = sign_a * sign_b

        flux_a = extrapolate_flux(u1, u2, pa, ea)
        flux_b = extrapolate_flux(u1, u2, pb, eb)

        flux_b_aligned = flux_b[::-1] if rev else flux_b

        # The SAT "mismatch" - should be small for smooth analytic field
        mismatch = flux_a + ss * flux_b_aligned

        # Also compute the flux magnitude for reference
        flux_mag = jnp.maximum(jnp.abs(flux_a), jnp.abs(flux_b_aligned))
        flux_scale = float(jnp.max(flux_mag))

        max_mismatch = float(jnp.max(jnp.abs(mismatch)))
        mean_mismatch = float(jnp.mean(jnp.abs(mismatch)))

        # Relative error
        rel = max_mismatch / flux_scale if flux_scale > 1e-15 else 0.0

        is_swap = op in ('T', 'TR')
        comp_a = 'u1' if ea in ('E', 'W') else 'u2'
        comp_b = 'u1' if eb in ('E', 'W') else 'u2'

        results.append({
            'edge': f"P{pa}-{ea} <-> P{pb}-{eb}",
            'op': op,
            'swap': is_swap,
            'comp': f"{comp_a}/{comp_b}",
            'max_mismatch': max_mismatch,
            'mean_mismatch': mean_mismatch,
            'flux_scale': flux_scale,
            'rel_err': rel,
            'mismatch_vec': np.array(mismatch),
        })

    return results


def print_results(results, field_name, N):
    """Pretty-print the edge flux continuity results."""
    print(f"\n  {'Edge':>20s}  {'op':>3s}  {'comp':>6s}  "
          f"{'max |mismatch|':>14s}  {'flux scale':>12s}  {'rel err':>10s}  {'type':>10s}")
    print("  " + "-" * 90)

    for r in results:
        etype = "AXIS-SWAP" if r['swap'] else "aligned"
        flag = " <<<" if r['rel_err'] > 0.1 else ""
        print(f"  {r['edge']:>20s}  {r['op']:>3s}  {r['comp']:>6s}  "
              f"{r['max_mismatch']:14.6e}  {r['flux_scale']:12.6e}  "
              f"{r['rel_err']:10.4e}  {etype:>10s}{flag}")


# ============================================================
# TEST 2: Convergence of the mismatch
# ============================================================

def test_mismatch_convergence(field_fn, field_name):
    """Check that the mismatch converges with refinement.
    
    For correct averaging: mismatch -> 0 as N -> inf.
    For INCORRECT sign: mismatch ~ O(1), doesn't converge.
    """
    print(f"\n{'='*70}")
    print(f"CONVERGENCE TEST: {field_name}")
    print(f"{'='*70}")

    Ns = [10, 20, 40]
    all_results = {}
    for N in Ns:
        all_results[N] = test_flux_continuity(N, field_fn, field_name)

    # For each edge, print convergence
    print(f"\n  {'Edge':>20s}  {'op':>3s}  ", end='')
    for N in Ns:
        print(f"{'N='+str(N):>12s}  ", end='')
    for i in range(len(Ns) - 1):
        print(f"{'rate':>6s}  ", end='')
    print()
    print("  " + "-" * (30 + 14 * len(Ns) + 8 * (len(Ns) - 1)))

    for edge_idx in range(len(EDGES)):
        pa, ea, pb, eb, op = EDGES[edge_idx]
        label = f"P{pa}-{ea}<->P{pb}-{eb}"
        is_swap = op in ('T', 'TR')

        print(f"  {label:>20s}  {op:>3s}  ", end='')

        mismatches = []
        for N in Ns:
            m = all_results[N][edge_idx]['max_mismatch']
            mismatches.append(m)
            print(f"{m:12.4e}  ", end='')

        for i in range(len(Ns) - 1):
            if mismatches[i] > 1e-15 and mismatches[i + 1] > 1e-15:
                rate = np.log2(mismatches[i] / mismatches[i + 1])
                print(f"{rate:6.2f}  ", end='')
            else:
                print(f"  ---  ", end='')

        if is_swap:
            print(" AXIS-SWAP", end='')
        print()


# ============================================================
# TEST 3: Direct check - what is the physical normal velocity?
# ============================================================

def test_physical_normal_flux(N, field_fn, field_name):
    """
    At each shared edge, compute the PHYSICAL normal velocity from
    Cartesian V·n, and compare with the extrapolated mass flux.
    
    This decouples the flux computation from the SAT formula.
    If the extrapolated flux doesn't match the physical flux,
    the problem is in how we compute contravariant velocity.
    """
    print(f"\n{'='*70}")
    print(f"PHYSICAL NORMAL FLUX CHECK: {field_name} (N={N})")
    print(f"{'='*70}")
    print()
    print("  Compare extrapolated mass flux with analytic V·n at boundary.\n")

    u1, u2, v1, v2, ops, grids, metrics = compute_analytic_mass_flux(N, field_fn)

    l = ops.l
    r = ops.r
    xi_v = np.array(grids['xi_v'])
    xi_c = np.array(grids['xi_c'])
    pi4 = np.pi / 4

    def extrapolate_flux(u1, u2, panel, edge):
        if edge == 'E':
            return np.array(jnp.einsum('c,cj->j', r, u1[panel]))
        elif edge == 'W':
            return np.array(jnp.einsum('c,cj->j', l, u1[panel]))
        elif edge == 'N':
            return np.array(jnp.einsum('ic,c->i', u2[panel], r))
        elif edge == 'S':
            return np.array(jnp.einsum('ic,c->i', u2[panel], l))

    print(f"  {'Edge':>20s}  {'op':>3s}  {'flux_a vs phys':>14s}  "
          f"{'flux_b vs phys':>14s}  {'note':>15s}")
    print("  " + "-" * 80)

    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        is_swap = op in ('T', 'TR')

        flux_a = extrapolate_flux(u1, u2, pa, ea)
        flux_b = extrapolate_flux(u1, u2, pb, eb)

        # Compute analytic physical normal flux at the h-grid boundary
        # The boundary of the h-grid (N+1 points along the edge)
        if ea in ('E', 'W'):
            xi1_bnd = np.full(N + 1, pi4 if ea == 'E' else -pi4)
            xi2_bnd = xi_v  # along the edge
        else:
            xi1_bnd = xi_v
            xi2_bnd = np.full(N + 1, pi4 if ea == 'N' else -pi4)

        X_a, Y_a, Z_a = equiangular_to_cartesian(xi1_bnd, xi2_bnd, pa)
        Vx_a, Vy_a, Vz_a = field_fn(X_a, Y_a, Z_a)

        # Compute outward unit normal at the boundary
        # For panel pa, edge ea, get the covariant basis and compute normal
        a1, a2 = get_covariant_basis(
            jnp.array(xi1_bnd), jnp.array(xi2_bnd), pa)

        # Outward normal at E/W: along a1 direction (±ξ¹)
        # Outward normal at N/S: along a2 direction (±ξ²)
        if ea in ('E', 'W'):
            # Normal is in a1 direction: n = a1 / |a1|
            n_mag = jnp.sqrt(a1[0]**2 + a1[1]**2 + a1[2]**2)
            sign = 1.0 if ea == 'E' else -1.0
            nx = sign * a1[0] / n_mag
            ny = sign * a1[1] / n_mag
            nz = sign * a1[2] / n_mag
        else:
            n_mag = jnp.sqrt(a2[0]**2 + a2[1]**2 + a2[2]**2)
            sign = 1.0 if ea == 'N' else -1.0
            nx = sign * a2[0] / n_mag
            ny = sign * a2[1] / n_mag
            nz = sign * a2[2] / n_mag

        # Physical V·n at the h-grid boundary
        Vn_phys = np.array(Vx_a * nx + Vy_a * ny + Vz_a * nz)

        # The extrapolated flux is the mass flux, not V·n directly.
        # Mass flux = J * v_contra. To compare with V·n, we need to
        # account for the metric. Actually, let's just check that flux_a 
        # from the two panels are consistent, not compare with V·n.
        # 
        # Better approach: compute J * v_contra ANALYTICALLY at the boundary.
        # v_contra at v1 points = Q11*v1_cov + Q12*v2_interp
        # This is what compute_contravariant does, but we can compute it
        # exactly at the boundary.

        # For now, let's compute the analytic covariant velocity at the
        # h-grid boundary point and compare the flux representations.
        v1_cov_a, v2_cov_a = cartesian_to_covariant(
            Vx_a, Vy_a, Vz_a, jnp.array(xi1_bnd), jnp.array(xi2_bnd), pa)

        # Compute metric at boundary
        J_bnd, Q11_bnd, Q12_bnd, Q22_bnd = compute_metric(
            jnp.array(xi1_bnd), jnp.array(xi2_bnd))

        # Analytic contravariant velocity at boundary
        v1_contra_exact = Q11_bnd * v1_cov_a + Q12_bnd * v2_cov_a
        v2_contra_exact = Q12_bnd * v1_cov_a + Q22_bnd * v2_cov_a

        # Analytic mass flux at boundary
        J_bnd_val = np.array(J_bnd)
        if ea in ('E', 'W'):
            # flux should be J1 * v1_contra, but at h-grid we don't have J1
            # J1 is at v1 grid. Use J_h * v1_contra as proxy
            flux_analytic = np.array(J_bnd * v1_contra_exact)
        else:
            flux_analytic = np.array(J_bnd * v2_contra_exact)

        # Compare extrapolated vs analytic
        err_a = np.max(np.abs(flux_a - flux_analytic))
        scale = max(np.max(np.abs(flux_analytic)), 1e-15)
        rel_a = err_a / scale

        # Do the same for panel b
        if eb in ('E', 'W'):
            xi1_bnd_b = np.full(N + 1, pi4 if eb == 'E' else -pi4)
            xi2_bnd_b = xi_v
        else:
            xi1_bnd_b = xi_v
            xi2_bnd_b = np.full(N + 1, pi4 if eb == 'N' else -pi4)

        X_b, Y_b, Z_b = equiangular_to_cartesian(xi1_bnd_b, xi2_bnd_b, pb)
        Vx_b, Vy_b, Vz_b = field_fn(X_b, Y_b, Z_b)
        v1_cov_b, v2_cov_b = cartesian_to_covariant(
            Vx_b, Vy_b, Vz_b, jnp.array(xi1_bnd_b), jnp.array(xi2_bnd_b), pb)
        J_bnd_b, Q11b, Q12b, Q22b = compute_metric(
            jnp.array(xi1_bnd_b), jnp.array(xi2_bnd_b))

        v1_contra_b = Q11b * v1_cov_b + Q12b * v2_cov_b
        v2_contra_b = Q12b * v1_cov_b + Q22b * v2_cov_b

        if eb in ('E', 'W'):
            flux_analytic_b = np.array(J_bnd_b * v1_contra_b)
        else:
            flux_analytic_b = np.array(J_bnd_b * v2_contra_b)

        err_b = np.max(np.abs(flux_b - flux_analytic_b))
        scale_b = max(np.max(np.abs(flux_analytic_b)), 1e-15)
        rel_b = err_b / scale_b

        note = "AXIS-SWAP" if is_swap else "aligned"
        print(f"  P{pa}-{ea}<->P{pb}-{eb}  {op:>3s}  {rel_a:14.6e}  {rel_b:14.6e}  {note:>15s}")


# ============================================================
# TEST 4: The smoking gun - do extrapolated fluxes from both 
# sides represent the same or opposite physical flux?
# ============================================================

def test_flux_sign_at_edge(N, field_fn, field_name):
    """
    At each edge, compute the ANALYTIC contravariant mass flux from
    both panels. Show whether the SAT sign convention (ss = sign_a*sign_b)
    correctly accounts for the relative orientation.
    
    For a correctly oriented pair:
      flux_a + ss * flux_b_aligned ≈ 0   (for analytic fields)
    
    If the sign is WRONG on axis-swapped edges:
      flux_a + ss * flux_b_aligned ≈ 2 * flux_a   (double instead of cancel)
    """
    print(f"\n{'='*70}")
    print(f"SMOKING GUN: Analytic flux sign check ({field_name}, N={N})")
    print(f"{'='*70}")
    print()
    print("  If mismatch is O(1) on axis-swap edges but O(h^p) on aligned")
    print("  edges, the sign convention is wrong for axis swaps.\n")

    u1, u2, v1, v2, ops, grids, metrics = compute_analytic_mass_flux(N, field_fn)

    l = ops.l
    r = ops.r
    xi_v = np.array(grids['xi_v'])
    pi4 = np.pi / 4

    def extrapolate_flux(u1, u2, panel, edge):
        if edge == 'E':
            return np.array(jnp.einsum('c,cj->j', r, u1[panel]))
        elif edge == 'W':
            return np.array(jnp.einsum('c,cj->j', l, u1[panel]))
        elif edge == 'N':
            return np.array(jnp.einsum('ic,c->i', u2[panel], r))
        elif edge == 'S':
            return np.array(jnp.einsum('ic,c->i', u2[panel], l))

    def boundary_sign(edge):
        return +1.0 if edge in ('E', 'N') else -1.0

    print(f"  {'Edge':>20s}  {'op':>3s}  {'max|flux_a|':>12s}  "
          f"{'max|mismatch|':>14s}  {'ratio':>8s}  {'type':>10s}  {'verdict':>10s}")
    print("  " + "-" * 100)

    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        sign_a = boundary_sign(ea)
        sign_b = boundary_sign(eb)
        ss = sign_a * sign_b

        flux_a = extrapolate_flux(u1, u2, pa, ea)
        flux_b = extrapolate_flux(u1, u2, pb, eb)
        flux_b_aligned = flux_b[::-1] if rev else flux_b

        mismatch = flux_a + ss * flux_b_aligned
        max_flux = max(np.max(np.abs(flux_a)), 1e-15)
        max_mismatch = np.max(np.abs(mismatch))
        ratio = max_mismatch / max_flux

        is_swap = op in ('T', 'TR')
        etype = "AXIS-SWAP" if is_swap else "aligned"

        if ratio > 0.5:
            verdict = "SIGN BUG!"
        elif ratio > 0.01:
            verdict = "suspect"
        else:
            verdict = "OK"

        print(f"  P{pa}-{ea}<->P{pb}-{eb}  {op:>3s}  {max_flux:12.4e}  "
              f"{max_mismatch:14.4e}  {ratio:8.4f}  {etype:>10s}  {verdict:>10s}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    N = 20

    print("=" * 70)
    print("SAT FLUX CONTINUITY TEST ON CUBED SPHERE")
    print("=" * 70)

    # Test with multiple vector fields to ensure robustness
    fields = [
        (lambda X, Y, Z: solid_body_rotation(X, Y, Z, 'z'), "SolidBody-Z"),
        (lambda X, Y, Z: solid_body_rotation(X, Y, Z, 'x'), "SolidBody-X"),
        (smooth_dipole, "Dipole"),
    ]

    for field_fn, field_name in fields:
        print(f"\n{'#'*70}")
        print(f"# Field: {field_name}")
        print(f"{'#'*70}")

        # Test 1: Flux mismatch at each edge
        print(f"\n--- Flux mismatch at edges (N={N}) ---")
        results = test_flux_continuity(N, field_fn, field_name)
        print_results(results, field_name, N)

        # Test 4: Smoking gun
        test_flux_sign_at_edge(N, field_fn, field_name)

    # Convergence test with z-rotation
    test_mismatch_convergence(
        lambda X, Y, Z: solid_body_rotation(X, Y, Z, 'z'),
        "SolidBody-Z")

    test_mismatch_convergence(
        lambda X, Y, Z: solid_body_rotation(X, Y, Z, 'x'),
        "SolidBody-X")

    # Physical normal flux check
    test_physical_normal_flux(
        20,
        lambda X, Y, Z: solid_body_rotation(X, Y, Z, 'x'),
        "SolidBody-X")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
