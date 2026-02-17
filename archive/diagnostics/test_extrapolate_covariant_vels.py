"""
Step 2 Test: Extrapolate covariant velocity (v_1, v_2) to h-grid boundaries.

On the staggered grid:
  v1 lives at (xi_c, xi_v), shape (N, N+1)   — cell-centers in ξ¹, vertices in ξ²
  v2 lives at (xi_v, xi_c), shape (N+1, N)   — vertices in ξ¹, cell-centers in ξ²
  h  lives at (xi_v, xi_v), shape (N+1, N+1) — vertices in both

To get both covariant components at an h-grid boundary (N+1 points):

  Edge E (ξ¹=max, i=N):
    v1_bnd[j] = r^T @ v1[:, j]     extrapolate in ξ¹ for each j   → (N+1,)
    v2_bnd    = Pcv @ v2[N, :]      interpolate v2 from ξ² cell-centers to vertices → (N+1,)

  Edge W (ξ¹=min, i=0):
    v1_bnd[j] = l^T @ v1[:, j]     extrapolate in ξ¹              → (N+1,)
    v2_bnd    = Pcv @ v2[0, :]      interpolate v2 to ξ² vertices  → (N+1,)

  Edge N (ξ²=max, j=N):
    v1_bnd    = Pcv @ v1[:, N]      interpolate v1 from ξ¹ cell-centers to vertices → (N+1,)
    v2_bnd[i] = r^T @ v2[i, :]     extrapolate in ξ² for each i   → (N+1,)
    equivalently: v2_bnd = v2 @ r

  Edge S (ξ²=min, j=0):
    v1_bnd    = Pcv @ v1[:, 0]      interpolate v1 to ξ¹ vertices  → (N+1,)
    v2_bnd    = v2 @ l              extrapolate in ξ²              → (N+1,)

Unit tests:
  1. Polynomial exactness: monomials in ξ should be extrapolated exactly up to
     the operator order (2 at boundary, 4 interior).
  2. Smooth field accuracy: use analytic Cartesian vector field, project to
     covariant on each panel, extrapolate to boundary, compare with analytic
     covariant velocity at boundary. Check convergence.
  3. Cross-panel consistency: at shared edges, the Cartesian velocity
     reconstructed from extrapolated covariant components should match
     between panels (this is the precursor to Step 3).
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
from velocity_transforms import (
    cartesian_to_covariant, covariant_to_cartesian, get_covariant_basis
)


# ============================================================
# The function under test
# ============================================================

def extrapolate_covariant_to_boundary(v1, v2, edge, ops):
    """
    Extrapolate both covariant velocity components to an h-grid boundary.

    Args:
        v1: (N, N+1)   covariant v_1 at (xi_c, xi_v)
        v2: (N+1, N)   covariant v_2 at (xi_v, xi_c)
        edge: 'E', 'W', 'N', or 'S'
        ops: SBP operators (needs l, r, Pcv)

    Returns:
        v1_bnd: (N+1,)  covariant v_1 at h-grid boundary
        v2_bnd: (N+1,)  covariant v_2 at h-grid boundary
    """
    l = ops.l
    r = ops.r
    Pcv = ops.Pcv
    N = ops.Pcv.shape[0] - 1  # Pcv is (N+1, N)

    if edge == 'E':
        # v1: extrapolate in ξ¹ (cell→vertex) at each ξ² column
        v1_bnd = jnp.einsum('c,cj->j', r, v1)        # (N+1,)
        # v2: already at ξ¹ vertex N, interpolate ξ² cell→vertex
        v2_bnd = Pcv @ v2[N, :]                        # (N+1,)

    elif edge == 'W':
        v1_bnd = jnp.einsum('c,cj->j', l, v1)        # (N+1,)
        v2_bnd = Pcv @ v2[0, :]                        # (N+1,)

    elif edge == 'N':
        # v1: already at ξ² vertex N, interpolate ξ¹ cell→vertex
        v1_bnd = Pcv @ v1[:, N]                        # (N+1,)
        # v2: extrapolate in ξ² (cell→vertex) at each ξ¹ row
        v2_bnd = jnp.einsum('ic,c->i', v2, r)         # (N+1,)

    elif edge == 'S':
        v1_bnd = Pcv @ v1[:, 0]                        # (N+1,)
        v2_bnd = jnp.einsum('ic,c->i', v2, l)         # (N+1,)

    else:
        raise ValueError(f"Invalid edge: {edge}")

    return v1_bnd, v2_bnd


# ============================================================
# Analytic test fields
# ============================================================

def solid_body_rotation_x(X, Y, Z):
    """Solid body rotation about X-axis: V = (0, -Z, Y)."""
    return jnp.zeros_like(X), -Z, Y


def solid_body_rotation_z(X, Y, Z):
    """Solid body rotation about Z-axis: V = (-Y, X, 0)."""
    return -Y, X, jnp.zeros_like(Z)


def diagonal_flow(X, Y, Z):
    """Solid body rotation about (1,1,1)/sqrt(3): exercises all components."""
    s3 = 1.0 / jnp.sqrt(3.0)
    # V = omega x r  with omega = (s3, s3, s3)
    Vx = s3 * Z - s3 * Y
    Vy = s3 * X - s3 * Z
    Vz = s3 * Y - s3 * X
    return Vx, Vy, Vz


# ============================================================
# Helper: create covariant velocity on staggered grid
# ============================================================

def make_covariant_field(N, panel, field_fn):
    """
    Create covariant velocity (v_1, v_2) on the staggered grid for one panel
    from an analytic Cartesian vector field.

    Returns:
        v1: (N, N+1)   true covariant v_1 at (xi_c, xi_v)
        v2: (N+1, N)   true covariant v_2 at (xi_v, xi_c)
    """
    L = jnp.pi / 2
    dx = L / N
    xi_v = jnp.linspace(-jnp.pi/4, jnp.pi/4, N + 1)
    xi_c = (jnp.arange(N) + 0.5) * dx - jnp.pi/4

    # v1 grid: (xi_c, xi_v)
    xi1_v1, xi2_v1 = jnp.meshgrid(xi_c, xi_v, indexing='ij')  # (N, N+1)
    X1, Y1, Z1 = equiangular_to_cartesian(xi1_v1, xi2_v1, panel)
    Vx1, Vy1, Vz1 = field_fn(X1, Y1, Z1)
    v1, _ = cartesian_to_covariant(Vx1, Vy1, Vz1, xi1_v1, xi2_v1, panel)

    # v2 grid: (xi_v, xi_c)
    xi1_v2, xi2_v2 = jnp.meshgrid(xi_v, xi_c, indexing='ij')  # (N+1, N)
    X2, Y2, Z2 = equiangular_to_cartesian(xi1_v2, xi2_v2, panel)
    Vx2, Vy2, Vz2 = field_fn(X2, Y2, Z2)
    _, v2 = cartesian_to_covariant(Vx2, Vy2, Vz2, xi1_v2, xi2_v2, panel)

    return v1, v2


def analytic_covariant_at_boundary(N, panel, edge, field_fn):
    """
    Compute the exact covariant velocity at h-grid boundary points.

    Returns:
        v1_exact: (N+1,)  exact covariant v_1 at boundary
        v2_exact: (N+1,)  exact covariant v_2 at boundary
    """
    pi4 = jnp.pi / 4
    xi_v = jnp.linspace(-pi4, pi4, N + 1)

    if edge == 'E':
        xi1_bnd = jnp.full(N + 1, pi4)
        xi2_bnd = xi_v
    elif edge == 'W':
        xi1_bnd = jnp.full(N + 1, -pi4)
        xi2_bnd = xi_v
    elif edge == 'N':
        xi1_bnd = xi_v
        xi2_bnd = jnp.full(N + 1, pi4)
    elif edge == 'S':
        xi1_bnd = xi_v
        xi2_bnd = jnp.full(N + 1, -pi4)

    X, Y, Z = equiangular_to_cartesian(xi1_bnd, xi2_bnd, panel)
    Vx, Vy, Vz = field_fn(X, Y, Z)
    v1_exact, v2_exact = cartesian_to_covariant(Vx, Vy, Vz, xi1_bnd, xi2_bnd, panel)

    return np.array(v1_exact), np.array(v2_exact)


# ============================================================
# TEST 1: Verify extrapolation shape and trivial cases
# ============================================================

def test_shapes_and_constant():
    """Constant covariant velocity should be extrapolated exactly."""
    print("=" * 70)
    print("TEST 1: Shapes and constant field exactness")
    print("=" * 70)

    N = 16
    dx = (jnp.pi / 2) / N
    ops = sbp_42(N, float(dx))

    # Constant covariant field: v1 = 1.0, v2 = -0.5
    v1 = jnp.ones((N, N + 1))
    v2 = -0.5 * jnp.ones((N + 1, N))

    max_err = 0.0
    for edge in ['E', 'W', 'N', 'S']:
        v1_bnd, v2_bnd = extrapolate_covariant_to_boundary(v1, v2, edge, ops)

        assert v1_bnd.shape == (N + 1,), f"v1_bnd shape {v1_bnd.shape} for {edge}"
        assert v2_bnd.shape == (N + 1,), f"v2_bnd shape {v2_bnd.shape} for {edge}"

        err_v1 = float(jnp.max(jnp.abs(v1_bnd - 1.0)))
        err_v2 = float(jnp.max(jnp.abs(v2_bnd - (-0.5))))
        max_err = max(max_err, err_v1, err_v2)
        print(f"  Edge {edge}: v1 err = {err_v1:.2e}, v2 err = {err_v2:.2e}")

    assert max_err < 1e-13, f"Constant field not exact: {max_err}"
    print(f"  Max error: {max_err:.2e}")
    print("  ✓ PASSED\n")


# ============================================================
# TEST 2: Accuracy vs analytic covariant velocity at boundary
# ============================================================

def test_accuracy_single_panel(field_fn, field_name):
    """
    For a single panel, extrapolate covariant velocity to each edge
    and compare with the analytic covariant velocity at boundary.
    """
    print(f"--- {field_name} ---")

    panel = 0
    Ns = [10, 20, 40, 80]
    results = {edge: {'v1': [], 'v2': []} for edge in ['E', 'W', 'N', 'S']}

    for N in Ns:
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, float(dx))

        v1, v2 = make_covariant_field(N, panel, field_fn)

        for edge in ['E', 'W', 'N', 'S']:
            v1_bnd, v2_bnd = extrapolate_covariant_to_boundary(v1, v2, edge, ops)
            v1_exact, v2_exact = analytic_covariant_at_boundary(N, panel, edge, field_fn)

            err_v1 = np.max(np.abs(np.array(v1_bnd) - v1_exact))
            err_v2 = np.max(np.abs(np.array(v2_bnd) - v2_exact))
            results[edge]['v1'].append(err_v1)
            results[edge]['v2'].append(err_v2)

    # Print convergence table
    print(f"  {'Edge':>5s}  {'comp':>4s}", end='')
    for N in Ns:
        print(f"  {'N='+str(N):>12s}", end='')
    for _ in range(len(Ns) - 1):
        print(f"  {'rate':>6s}", end='')
    print()
    print("  " + "-" * (12 + 14 * len(Ns) + 8 * (len(Ns) - 1)))

    for edge in ['E', 'W', 'N', 'S']:
        for comp_name, comp_key in [('v_1', 'v1'), ('v_2', 'v2')]:
            errs = results[edge][comp_key]
            print(f"  {edge:>5s}  {comp_name:>4s}", end='')
            for e in errs:
                print(f"  {e:12.4e}", end='')
            for i in range(len(Ns) - 1):
                if errs[i] > 1e-15 and errs[i+1] > 1e-15:
                    rate = np.log2(errs[i] / errs[i+1])
                    print(f"  {rate:6.2f}", end='')
                else:
                    print(f"    ---", end='')
            print()
    print()


def test_accuracy():
    """Run accuracy test for multiple fields."""
    print("=" * 70)
    print("TEST 2: Extrapolation accuracy vs analytic (panel 0)")
    print("=" * 70)
    print()

    fields = [
        (solid_body_rotation_z, "SolidBody-Z"),
        (solid_body_rotation_x, "SolidBody-X"),
        (diagonal_flow, "Diagonal(1,1,1)"),
    ]

    for fn, name in fields:
        test_accuracy_single_panel(fn, name)

    print("  Expected: ~O(h^2) for extrapolated component (l/r stencil)")
    print("            ~O(h^2) for interpolated component (Pcv boundary)")
    print("  ✓ TEST 2 COMPLETE\n")


# ============================================================
# TEST 3: All 6 panels, all 4 edges
# ============================================================

def test_all_panels():
    """Check that extrapolation works on all panels, not just panel 0."""
    print("=" * 70)
    print("TEST 3: All panels, all edges (N=20, SolidBody-X)")
    print("=" * 70)
    print()

    N = 20
    dx = (jnp.pi / 2) / N
    ops = sbp_42(N, float(dx))
    field_fn = solid_body_rotation_x

    print(f"  {'Panel':>5s}  {'Edge':>5s}  {'err v_1':>12s}  {'err v_2':>12s}")
    print("  " + "-" * 42)

    max_err = 0.0
    for panel in range(6):
        v1, v2 = make_covariant_field(N, panel, field_fn)
        for edge in ['E', 'W', 'N', 'S']:
            v1_bnd, v2_bnd = extrapolate_covariant_to_boundary(v1, v2, edge, ops)
            v1_ex, v2_ex = analytic_covariant_at_boundary(N, panel, edge, field_fn)

            err_v1 = np.max(np.abs(np.array(v1_bnd) - v1_ex))
            err_v2 = np.max(np.abs(np.array(v2_bnd) - v2_ex))
            max_err = max(max_err, err_v1, err_v2)
            print(f"  {panel:>5d}  {edge:>5s}  {err_v1:12.4e}  {err_v2:12.4e}")

    print(f"\n  Global max error: {max_err:.4e}")
    print("  ✓ TEST 3 COMPLETE\n")


# ============================================================
# TEST 4: Cross-panel consistency preview
# ============================================================

EDGES = [
    (0, 'N', 1, 'N', 'R'),
    (0, 'E', 4, 'N', 'T'),
    (0, 'W', 2, 'N', 'TR'),
    (0, 'S', 3, 'N', 'N'),
    (1, 'E', 2, 'W', 'N'),
    (1, 'S', 5, 'N', 'N'),
    (1, 'W', 4, 'E', 'N'),
    (2, 'E', 3, 'W', 'N'),
    (2, 'S', 5, 'E', 'TR'),
    (3, 'E', 4, 'W', 'N'),
    (3, 'S', 5, 'S', 'R'),
    (4, 'S', 5, 'W', 'T'),
]


def _reverses(op):
    return op in ('R', 'TR')


def test_cross_panel_cartesian():
    """
    At each shared edge, extrapolate covariant velocity from both panels,
    convert each to Cartesian, and check that the Cartesian velocity matches.

    This is the critical precursor to Step 3/4.
    For an exact analytic field:
    - The covariant components will DIFFER between panels (different coords)
    - The Cartesian velocity MUST AGREE (same physical vector)
    
    Any mismatch is purely from extrapolation/interpolation error.
    """
    print("=" * 70)
    print("TEST 4: Cross-panel Cartesian consistency at shared edges")
    print("=" * 70)
    print()
    print("  Extrapolate covariant from both panels, convert to Cartesian,")
    print("  check that |V_A - V_B| is small (extrapolation error only).\n")

    N = 20
    dx = (jnp.pi / 2) / N
    ops = sbp_42(N, float(dx))
    pi4 = jnp.pi / 4
    xi_v = jnp.linspace(-pi4, pi4, N + 1)

    field_fn = diagonal_flow  # exercises all panels

    # Precompute covariant fields for all panels
    v1_all = {}
    v2_all = {}
    for panel in range(6):
        v1_all[panel], v2_all[panel] = make_covariant_field(N, panel, field_fn)

    def edge_bnd_coords(edge):
        if edge == 'E':
            return jnp.full(N + 1, pi4), xi_v
        elif edge == 'W':
            return jnp.full(N + 1, -pi4), xi_v
        elif edge == 'N':
            return xi_v, jnp.full(N + 1, pi4)
        elif edge == 'S':
            return xi_v, jnp.full(N + 1, -pi4)

    print(f"  {'Edge':>20s}  {'op':>3s}  {'max|V_A - V_B|':>15s}  "
          f"{'max|V_A - exact|':>16s}  {'type':>10s}")
    print("  " + "-" * 78)

    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)

        # Extrapolate from both panels
        v1a_bnd, v2a_bnd = extrapolate_covariant_to_boundary(
            v1_all[pa], v2_all[pa], ea, ops)
        v1b_bnd, v2b_bnd = extrapolate_covariant_to_boundary(
            v1_all[pb], v2_all[pb], eb, ops)

        # Convert to Cartesian
        xi1_a, xi2_a = edge_bnd_coords(ea)
        xi1_b, xi2_b = edge_bnd_coords(eb)

        Vx_A, Vy_A, Vz_A = covariant_to_cartesian(v1a_bnd, v2a_bnd, xi1_a, xi2_a, pa)
        Vx_B, Vy_B, Vz_B = covariant_to_cartesian(v1b_bnd, v2b_bnd, xi1_b, xi2_b, pb)

        # Align indices (reversal for R, TR edges)
        if rev:
            Vx_B = Vx_B[::-1]
            Vy_B = Vy_B[::-1]
            Vz_B = Vz_B[::-1]

        # Cartesian mismatch between panels
        dV = jnp.sqrt((Vx_A - Vx_B)**2 + (Vy_A - Vy_B)**2 + (Vz_A - Vz_B)**2)
        max_dV = float(jnp.max(dV))

        # Compare panel A with exact
        X_ex, Y_ex, Z_ex = equiangular_to_cartesian(xi1_a, xi2_a, pa)
        Vx_ex, Vy_ex, Vz_ex = field_fn(X_ex, Y_ex, Z_ex)
        dV_ex = jnp.sqrt((Vx_A - Vx_ex)**2 + (Vy_A - Vy_ex)**2 + (Vz_A - Vz_ex)**2)
        max_dV_ex = float(jnp.max(dV_ex))

        is_swap = op in ('T', 'TR')
        etype = "AXIS-SWAP" if is_swap else "aligned"

        print(f"  P{pa}-{ea}<->P{pb}-{eb}  {op:>3s}  {max_dV:15.6e}  "
              f"{max_dV_ex:16.6e}  {etype:>10s}")

    print()
    print("  Key: |V_A - V_B| should be O(h^p) for ALL edges, including AXIS-SWAP.")
    print("  The covariant components differ between panels, but Cartesian must agree.")
    print("  ✓ TEST 4 COMPLETE\n")


# ============================================================
# TEST 5: Convergence of cross-panel Cartesian mismatch
# ============================================================

def test_cross_panel_convergence():
    """Check that cross-panel Cartesian mismatch converges with refinement."""
    print("=" * 70)
    print("TEST 5: Cross-panel Cartesian mismatch convergence")
    print("=" * 70)
    print()

    Ns = [10, 20, 40, 80]
    field_fn = diagonal_flow
    pi4 = np.pi / 4

    # Collect results per edge
    all_results = {i: [] for i in range(len(EDGES))}

    for N in Ns:
        dx = (np.pi / 2) / N
        ops = sbp_42(N, float(dx))
        xi_v = jnp.linspace(-pi4, pi4, N + 1)

        v1_all = {}
        v2_all = {}
        for panel in range(6):
            v1_all[panel], v2_all[panel] = make_covariant_field(N, panel, field_fn)

        def edge_bnd_coords(edge):
            if edge == 'E':
                return jnp.full(N + 1, pi4), xi_v
            elif edge == 'W':
                return jnp.full(N + 1, -pi4), xi_v
            elif edge == 'N':
                return xi_v, jnp.full(N + 1, pi4)
            elif edge == 'S':
                return xi_v, jnp.full(N + 1, -pi4)

        for idx, (pa, ea, pb, eb, op) in enumerate(EDGES):
            rev = _reverses(op)

            v1a_bnd, v2a_bnd = extrapolate_covariant_to_boundary(
                v1_all[pa], v2_all[pa], ea, ops)
            v1b_bnd, v2b_bnd = extrapolate_covariant_to_boundary(
                v1_all[pb], v2_all[pb], eb, ops)

            xi1_a, xi2_a = edge_bnd_coords(ea)
            xi1_b, xi2_b = edge_bnd_coords(eb)

            Vx_A, Vy_A, Vz_A = covariant_to_cartesian(v1a_bnd, v2a_bnd, xi1_a, xi2_a, pa)
            Vx_B, Vy_B, Vz_B = covariant_to_cartesian(v1b_bnd, v2b_bnd, xi1_b, xi2_b, pb)

            if rev:
                Vx_B, Vy_B, Vz_B = Vx_B[::-1], Vy_B[::-1], Vz_B[::-1]

            dV = jnp.sqrt((Vx_A - Vx_B)**2 + (Vy_A - Vy_B)**2 + (Vz_A - Vz_B)**2)
            all_results[idx].append(float(jnp.max(dV)))

    # Print convergence table
    print(f"  {'Edge':>20s}  {'op':>3s}", end='')
    for N in Ns:
        print(f"  {'N='+str(N):>12s}", end='')
    for _ in range(len(Ns) - 1):
        print(f"  {'rate':>6s}", end='')
    print()
    print("  " + "-" * (28 + 14 * len(Ns) + 8 * (len(Ns) - 1)))

    for idx, (pa, ea, pb, eb, op) in enumerate(EDGES):
        label = f"P{pa}-{ea}<->P{pb}-{eb}"
        is_swap = op in ('T', 'TR')
        errs = all_results[idx]

        print(f"  {label:>20s}  {op:>3s}", end='')
        for e in errs:
            print(f"  {e:12.4e}", end='')
        for i in range(len(Ns) - 1):
            if errs[i] > 1e-15 and errs[i+1] > 1e-15:
                rate = np.log2(errs[i] / errs[i+1])
                print(f"  {rate:6.2f}", end='')
            else:
                print(f"    ---", end='')
        if is_swap:
            print("  AXIS-SWAP", end='')
        print()

    print()
    print("  All edges should converge at O(h^2) or better.")
    print("  If AXIS-SWAP edges converge while aligned edges don't (or vice versa),")
    print("  there's still a problem.\n")
    print("  ✓ TEST 5 COMPLETE")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    test_shapes_and_constant()
    test_accuracy()
    test_all_panels()
    test_cross_panel_cartesian()
    test_cross_panel_convergence()

    print()
    print("=" * 70)
    print("ALL STEP 2 TESTS COMPLETE")
    print("=" * 70)
    print()
    print("The function extrapolate_covariant_to_boundary() is validated.")
    print("Next: Step 3 (test_step5b_cart_bnd.py) — boundary Cartesian conversion.")
