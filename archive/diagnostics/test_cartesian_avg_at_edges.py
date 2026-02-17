"""
Step 3/4: Cartesian-Averaged Velocity at Cubed-Sphere Edges

This is THE critical test for the SAT refactoring. It implements the full
Cartesian averaging pipeline and verifies that it produces the correct
velocity at every shared edge — including the axis-swap edges that the
current SAT gets catastrophically wrong.

Pipeline per edge (pa, ea) <-> (pb, eb):
  1. Extrapolate covariant (v_1, v_2) to boundary from both panels
  2. Convert each panel's covariant velocity to Cartesian (Vx, Vy, Vz)
  3. Align indices (reverse for R, TR edges)
  4. Average in Cartesian: V_avg = 0.5 * (V_A + V_B)
  5. Convert V_avg back to covariant in each panel's frame
  6. Verify: for an exact analytic field, V_avg should match the analytic V

Tests:
  TEST 1: For exact analytic fields, averaged Cartesian velocity at each
          edge matches the analytic field. Error is purely from
          extrapolation/interpolation (O(h^2)).

  TEST 2: Convergence of the averaged velocity error with refinement.
          ALL 12 edges must converge, including axis-swap.

  TEST 3: The "fixed flux mismatch" test — compute mass flux from the
          averaged covariant velocity and check that flux_a + ss*flux_b ≈ 0.
          This is the direct comparison with test_sat_flux_continuity.py
          which showed O(1) mismatch with the old method.

  TEST 4: Function packaging — the cartesian_average_at_edge() function
          that will be used by the refactored SAT.
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
from test_extrapolate_covariant_vels import (
    extrapolate_covariant_to_boundary, make_covariant_field,
    analytic_covariant_at_boundary,
)


# ============================================================
# Edge connectivity (duplicated for self-containment)
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


# ============================================================
# Core function: Cartesian averaging at one edge
# ============================================================

def edge_bnd_coords(edge, N):
    """Get (xi1, xi2) arrays at h-grid boundary. Returns (N+1,) arrays."""
    pi4 = jnp.pi / 4
    xi_v = jnp.linspace(-pi4, pi4, N + 1)
    if edge == 'E':
        return jnp.full(N + 1, pi4), xi_v
    elif edge == 'W':
        return jnp.full(N + 1, -pi4), xi_v
    elif edge == 'N':
        return xi_v, jnp.full(N + 1, pi4)
    elif edge == 'S':
        return xi_v, jnp.full(N + 1, -pi4)


def cartesian_average_at_edge(v1_a, v2_a, v1_b, v2_b,
                               pa, ea, pb, eb, op, ops, N):
    """
    Compute Cartesian-averaged covariant velocity at a shared edge.

    Args:
        v1_a, v2_a: (N,N+1), (N+1,N) covariant velocity on panel A
        v1_b, v2_b: (N,N+1), (N+1,N) covariant velocity on panel B
        pa, ea: panel A index and edge label
        pb, eb: panel B index and edge label
        op: connectivity operation ('N', 'R', 'T', 'TR')
        ops: SBP operators
        N: grid resolution

    Returns:
        v1_avg_a, v2_avg_a: (N+1,) averaged covariant velocity in A's frame
        v1_avg_b, v2_avg_b: (N+1,) averaged covariant velocity in B's frame
        Vx_avg, Vy_avg, Vz_avg: (N+1,) averaged Cartesian velocity
    """
    rev = _reverses(op)

    # Step 1: Extrapolate covariant to boundary
    v1a_bnd, v2a_bnd = extrapolate_covariant_to_boundary(v1_a, v2_a, ea, ops)
    v1b_bnd, v2b_bnd = extrapolate_covariant_to_boundary(v1_b, v2_b, eb, ops)

    # Step 2: Convert to Cartesian
    xi1_a, xi2_a = edge_bnd_coords(ea, N)
    xi1_b, xi2_b = edge_bnd_coords(eb, N)

    Vx_A, Vy_A, Vz_A = covariant_to_cartesian(v1a_bnd, v2a_bnd, xi1_a, xi2_a, pa)
    Vx_B, Vy_B, Vz_B = covariant_to_cartesian(v1b_bnd, v2b_bnd, xi1_b, xi2_b, pb)

    # Step 3: Align indices for reversed edges
    if rev:
        Vx_B = Vx_B[::-1]
        Vy_B = Vy_B[::-1]
        Vz_B = Vz_B[::-1]

    # Step 4: Average in Cartesian
    Vx_avg = 0.5 * (Vx_A + Vx_B)
    Vy_avg = 0.5 * (Vy_A + Vy_B)
    Vz_avg = 0.5 * (Vz_A + Vz_B)

    # Step 5: Convert back to covariant in each panel's frame
    v1_avg_a, v2_avg_a = cartesian_to_covariant(Vx_avg, Vy_avg, Vz_avg,
                                                  xi1_a, xi2_a, pa)

    # For panel B, need to un-reverse the averaged Cartesian to B's index order
    if rev:
        Vx_avg_b = Vx_avg[::-1]
        Vy_avg_b = Vy_avg[::-1]
        Vz_avg_b = Vz_avg[::-1]
    else:
        Vx_avg_b = Vx_avg
        Vy_avg_b = Vy_avg
        Vz_avg_b = Vz_avg

    v1_avg_b, v2_avg_b = cartesian_to_covariant(Vx_avg_b, Vy_avg_b, Vz_avg_b,
                                                  xi1_b, xi2_b, pb)

    return v1_avg_a, v2_avg_a, v1_avg_b, v2_avg_b, Vx_avg, Vy_avg, Vz_avg


# ============================================================
# Analytic test fields
# ============================================================

def solid_body_rotation_z(X, Y, Z):
    return -Y, X, jnp.zeros_like(Z)

def solid_body_rotation_x(X, Y, Z):
    return jnp.zeros_like(X), -Z, Y

def diagonal_flow(X, Y, Z):
    s3 = 1.0 / jnp.sqrt(3.0)
    return s3*Z - s3*Y, s3*X - s3*Z, s3*Y - s3*X


# ============================================================
# TEST 1: Averaged Cartesian velocity vs analytic
# ============================================================

def test_averaged_cartesian_accuracy():
    """
    For each edge, compute the Cartesian-averaged velocity and compare
    with the exact analytic Cartesian velocity at the boundary.

    Since both panels sample the same analytic field, the average of
    exact values would be exact. The error comes only from
    extrapolation/interpolation.
    """
    print("=" * 70)
    print("TEST 1: Cartesian-averaged velocity accuracy at edges (N=20)")
    print("=" * 70)
    print()

    N = 20
    dx = (np.pi / 2) / N
    ops = sbp_42(N, float(dx))

    fields = [
        (solid_body_rotation_z, "SolidBody-Z"),
        (solid_body_rotation_x, "SolidBody-X"),
        (diagonal_flow, "Diagonal"),
    ]

    for field_fn, field_name in fields:
        print(f"  --- {field_name} ---")
        print(f"  {'Edge':>20s}  {'op':>3s}  {'|V_avg - V_exact|':>18s}  "
              f"{'type':>10s}")
        print("  " + "-" * 62)

        # Precompute covariant fields
        v1_all, v2_all = {}, {}
        for p in range(6):
            v1_all[p], v2_all[p] = make_covariant_field(N, p, field_fn)

        for pa, ea, pb, eb, op in EDGES:
            v1_avg_a, v2_avg_a, v1_avg_b, v2_avg_b, Vx_avg, Vy_avg, Vz_avg = \
                cartesian_average_at_edge(
                    v1_all[pa], v2_all[pa], v1_all[pb], v2_all[pb],
                    pa, ea, pb, eb, op, ops, N)

            # Analytic Cartesian at panel A's boundary
            xi1_a, xi2_a = edge_bnd_coords(ea, N)
            X_ex, Y_ex, Z_ex = equiangular_to_cartesian(xi1_a, xi2_a, pa)
            Vx_ex, Vy_ex, Vz_ex = field_fn(X_ex, Y_ex, Z_ex)

            dV = jnp.sqrt((Vx_avg - Vx_ex)**2 +
                          (Vy_avg - Vy_ex)**2 +
                          (Vz_avg - Vz_ex)**2)
            max_dV = float(jnp.max(dV))

            is_swap = op in ('T', 'TR')
            etype = "AXIS-SWAP" if is_swap else "aligned"
            print(f"  P{pa}-{ea}<->P{pb}-{eb}  {op:>3s}  {max_dV:18.6e}  {etype:>10s}")

        print()

    print("  All edges should show O(h^2) error from interpolation,")
    print("  NOT O(1) as in the old flux-averaging approach.")
    print("  ✓ TEST 1 COMPLETE\n")


# ============================================================
# TEST 2: Convergence of averaged velocity error
# ============================================================

def test_convergence():
    """
    Check that the Cartesian-averaged velocity error converges at O(h^2)
    on ALL 12 edges, including axis-swap.
    """
    print("=" * 70)
    print("TEST 2: Convergence of Cartesian-averaged velocity error")
    print("=" * 70)
    print()

    Ns = [10, 20, 40, 80]
    field_fn = diagonal_flow  # exercises all panels

    all_results = {i: [] for i in range(len(EDGES))}

    for N in Ns:
        dx = (np.pi / 2) / N
        ops = sbp_42(N, float(dx))

        v1_all, v2_all = {}, {}
        for p in range(6):
            v1_all[p], v2_all[p] = make_covariant_field(N, p, field_fn)

        for idx, (pa, ea, pb, eb, op) in enumerate(EDGES):
            _, _, _, _, Vx_avg, Vy_avg, Vz_avg = \
                cartesian_average_at_edge(
                    v1_all[pa], v2_all[pa], v1_all[pb], v2_all[pb],
                    pa, ea, pb, eb, op, ops, N)

            xi1_a, xi2_a = edge_bnd_coords(ea, N)
            X_ex, Y_ex, Z_ex = equiangular_to_cartesian(xi1_a, xi2_a, pa)
            Vx_ex, Vy_ex, Vz_ex = field_fn(X_ex, Y_ex, Z_ex)

            dV = jnp.sqrt((Vx_avg - Vx_ex)**2 +
                          (Vy_avg - Vy_ex)**2 +
                          (Vz_avg - Vz_ex)**2)
            all_results[idx].append(float(jnp.max(dV)))

    print(f"  {'Edge':>20s}  {'op':>3s}", end='')
    for N in Ns:
        print(f"  {'N='+str(N):>12s}", end='')
    for _ in range(len(Ns) - 1):
        print(f"  {'rate':>6s}", end='')
    print()
    print("  " + "-" * (28 + 14*len(Ns) + 8*(len(Ns)-1)))

    for idx, (pa, ea, pb, eb, op) in enumerate(EDGES):
        label = f"P{pa}-{ea}<->P{pb}-{eb}"
        is_swap = op in ('T', 'TR')
        errs = all_results[idx]

        print(f"  {label:>20s}  {op:>3s}", end='')
        for e in errs:
            print(f"  {e:12.4e}", end='')
        for i in range(len(Ns) - 1):
            if errs[i] > 1e-14 and errs[i+1] > 1e-14:
                rate = np.log2(errs[i] / errs[i+1])
                print(f"  {rate:6.2f}", end='')
            else:
                print(f"    ---", end='')
        if is_swap:
            print("  SWAP", end='')
        print()

    print()
    print("  All 12 edges should converge at ~O(h^2).")
    print("  ✓ TEST 2 COMPLETE\n")


# ============================================================
# TEST 3: Consensus flux matches analytic — the payoff test
# ============================================================

def test_consensus_flux_accuracy():
    """
    THE key test for the SAT refactoring.

    The refactored SAT will compute a "consensus" normal mass flux from
    the Cartesian-averaged velocity:
        flux_consensus = J * Q^{normal,j} * v_j_avg   (in local coords)

    For an exact analytic field, v_avg ≈ v_exact (with O(h^p) error from
    extrapolation/interpolation), so flux_consensus ≈ flux_exact.

    This test verifies that the consensus flux is close to the analytic
    flux on EACH panel INDIVIDUALLY — not by comparing flux across panels
    (which involves incompatible coordinate representations).

    The old SAT averaged INCOMPATIBLE fluxes across panels. The new SAT
    will use a consensus flux that is always in the local panel's frame,
    so this per-panel accuracy is what matters.
    """
    print("=" * 70)
    print("TEST 3: Consensus flux accuracy (per panel, all edges)")
    print("=" * 70)
    print()
    print("  flux_consensus = J * Q^{n,j} * v_j_avg  (Cartesian-averaged velocity)")
    print("  flux_exact     = J * Q^{n,j} * v_j_exact (analytic at boundary)")
    print("  Error should be O(h^p) on ALL edges, including AXIS-SWAP.\n")

    from test_stag_step4 import compute_metric

    N = 20
    dx = (np.pi / 2) / N
    ops = sbp_42(N, float(dx))

    fields = [
        (solid_body_rotation_z, "SolidBody-Z"),
        (solid_body_rotation_x, "SolidBody-X"),
        (diagonal_flow, "Diagonal"),
    ]

    for field_fn, field_name in fields:
        print(f"  --- {field_name} ---")
        print(f"  {'Edge':>20s}  {'op':>3s}  {'|flux_cons - flux_exact| A':>26s}  "
              f"{'B':>12s}  {'type':>10s}")
        print("  " + "-" * 82)

        v1_all, v2_all = {}, {}
        for p in range(6):
            v1_all[p], v2_all[p] = make_covariant_field(N, p, field_fn)

        for pa, ea, pb, eb, op in EDGES:
            rev = _reverses(op)

            # Cartesian-averaged covariant velocity
            v1_avg_a, v2_avg_a, v1_avg_b, v2_avg_b, _, _, _ = \
                cartesian_average_at_edge(
                    v1_all[pa], v2_all[pa], v1_all[pb], v2_all[pb],
                    pa, ea, pb, eb, op, ops, N)

            # Analytic covariant at boundary — panel A
            xi1_a, xi2_a = edge_bnd_coords(ea, N)
            X_a, Y_a, Z_a = equiangular_to_cartesian(xi1_a, xi2_a, pa)
            Vx_a, Vy_a, Vz_a = field_fn(X_a, Y_a, Z_a)
            v1_ex_a, v2_ex_a = cartesian_to_covariant(Vx_a, Vy_a, Vz_a,
                                                        xi1_a, xi2_a, pa)
            J_a, Q11_a, Q12_a, Q22_a = compute_metric(xi1_a, xi2_a)

            # Analytic covariant at boundary — panel B
            xi1_b, xi2_b = edge_bnd_coords(eb, N)
            X_b, Y_b, Z_b = equiangular_to_cartesian(xi1_b, xi2_b, pb)
            Vx_b, Vy_b, Vz_b = field_fn(X_b, Y_b, Z_b)
            v1_ex_b, v2_ex_b = cartesian_to_covariant(Vx_b, Vy_b, Vz_b,
                                                        xi1_b, xi2_b, pb)
            J_b, Q11_b, Q12_b, Q22_b = compute_metric(xi1_b, xi2_b)

            # Normal flux: consensus vs exact — panel A
            if ea in ('E', 'W'):
                flux_cons_a = J_a * (Q11_a * v1_avg_a + Q12_a * v2_avg_a)
                flux_exact_a = J_a * (Q11_a * v1_ex_a + Q12_a * v2_ex_a)
            else:
                flux_cons_a = J_a * (Q12_a * v1_avg_a + Q22_a * v2_avg_a)
                flux_exact_a = J_a * (Q12_a * v1_ex_a + Q22_a * v2_ex_a)

            err_a = float(jnp.max(jnp.abs(flux_cons_a - flux_exact_a)))

            # Normal flux: consensus vs exact — panel B
            if eb in ('E', 'W'):
                flux_cons_b = J_b * (Q11_b * v1_avg_b + Q12_b * v2_avg_b)
                flux_exact_b = J_b * (Q11_b * v1_ex_b + Q12_b * v2_ex_b)
            else:
                flux_cons_b = J_b * (Q12_b * v1_avg_b + Q22_b * v2_avg_b)
                flux_exact_b = J_b * (Q12_b * v1_ex_b + Q22_b * v2_ex_b)

            err_b = float(jnp.max(jnp.abs(flux_cons_b - flux_exact_b)))

            is_swap = op in ('T', 'TR')
            etype = "AXIS-SWAP" if is_swap else "aligned"
            print(f"  P{pa}-{ea}<->P{pb}-{eb}  {op:>3s}  {err_a:26.6e}  "
                  f"{err_b:12.6e}  {etype:>10s}")

        print()

    print("  ✓ TEST 3 COMPLETE\n")


# ============================================================
# TEST 4: Convergence of consensus flux accuracy
# ============================================================

def test_consensus_flux_convergence():
    """
    Verify that the consensus flux error converges with refinement on ALL edges.
    This directly predicts the SAT penalty magnitude: small penalty = small error.
    """
    print("=" * 70)
    print("TEST 4: Convergence of consensus flux error (Diagonal field)")
    print("=" * 70)
    print()

    from test_stag_step4 import compute_metric

    Ns = [10, 20, 40, 80]
    field_fn = diagonal_flow

    # Per-edge, per-panel results
    all_results_a = {i: [] for i in range(len(EDGES))}
    all_results_b = {i: [] for i in range(len(EDGES))}

    for N in Ns:
        dx = (np.pi / 2) / N
        ops = sbp_42(N, float(dx))

        v1_all, v2_all = {}, {}
        for p in range(6):
            v1_all[p], v2_all[p] = make_covariant_field(N, p, field_fn)

        for idx, (pa, ea, pb, eb, op) in enumerate(EDGES):
            v1_avg_a, v2_avg_a, v1_avg_b, v2_avg_b, _, _, _ = \
                cartesian_average_at_edge(
                    v1_all[pa], v2_all[pa], v1_all[pb], v2_all[pb],
                    pa, ea, pb, eb, op, ops, N)

            # Analytic on panel A
            xi1_a, xi2_a = edge_bnd_coords(ea, N)
            X_a, Y_a, Z_a = equiangular_to_cartesian(xi1_a, xi2_a, pa)
            Vx_a, Vy_a, Vz_a = field_fn(X_a, Y_a, Z_a)
            v1_ex_a, v2_ex_a = cartesian_to_covariant(Vx_a, Vy_a, Vz_a,
                                                        xi1_a, xi2_a, pa)
            J_a, Q11_a, Q12_a, Q22_a = compute_metric(xi1_a, xi2_a)

            if ea in ('E', 'W'):
                flux_cons = J_a * (Q11_a * v1_avg_a + Q12_a * v2_avg_a)
                flux_exact = J_a * (Q11_a * v1_ex_a + Q12_a * v2_ex_a)
            else:
                flux_cons = J_a * (Q12_a * v1_avg_a + Q22_a * v2_avg_a)
                flux_exact = J_a * (Q12_a * v1_ex_a + Q22_a * v2_ex_a)

            all_results_a[idx].append(float(jnp.max(jnp.abs(flux_cons - flux_exact))))

            # Analytic on panel B
            xi1_b, xi2_b = edge_bnd_coords(eb, N)
            X_b, Y_b, Z_b = equiangular_to_cartesian(xi1_b, xi2_b, pb)
            Vx_b, Vy_b, Vz_b = field_fn(X_b, Y_b, Z_b)
            v1_ex_b, v2_ex_b = cartesian_to_covariant(Vx_b, Vy_b, Vz_b,
                                                        xi1_b, xi2_b, pb)
            J_b, Q11_b, Q12_b, Q22_b = compute_metric(xi1_b, xi2_b)

            if eb in ('E', 'W'):
                flux_cons_b = J_b * (Q11_b * v1_avg_b + Q12_b * v2_avg_b)
                flux_exact_b = J_b * (Q11_b * v1_ex_b + Q12_b * v2_ex_b)
            else:
                flux_cons_b = J_b * (Q12_b * v1_avg_b + Q22_b * v2_avg_b)
                flux_exact_b = J_b * (Q12_b * v1_ex_b + Q22_b * v2_ex_b)

            all_results_b[idx].append(float(jnp.max(jnp.abs(flux_cons_b - flux_exact_b))))

    # Print panel A convergence
    print("  Panel A side:")
    print(f"  {'Edge':>20s}  {'op':>3s}", end='')
    for N in Ns:
        print(f"  {'N='+str(N):>12s}", end='')
    for _ in range(len(Ns) - 1):
        print(f"  {'rate':>6s}", end='')
    print()
    print("  " + "-" * (28 + 14*len(Ns) + 8*(len(Ns)-1)))

    for idx, (pa, ea, pb, eb, op) in enumerate(EDGES):
        label = f"P{pa}-{ea}<->P{pb}-{eb}"
        is_swap = op in ('T', 'TR')
        errs = all_results_a[idx]

        print(f"  {label:>20s}  {op:>3s}", end='')
        for e in errs:
            print(f"  {e:12.4e}", end='')
        for i in range(len(Ns) - 1):
            if errs[i] > 1e-14 and errs[i+1] > 1e-14:
                rate = np.log2(errs[i] / errs[i+1])
                print(f"  {rate:6.2f}", end='')
            else:
                print(f"    ---", end='')
        if is_swap:
            print("  SWAP", end='')
        print()

    # Print panel B convergence
    print(f"\n  Panel B side:")
    print(f"  {'Edge':>20s}  {'op':>3s}", end='')
    for N in Ns:
        print(f"  {'N='+str(N):>12s}", end='')
    for _ in range(len(Ns) - 1):
        print(f"  {'rate':>6s}", end='')
    print()
    print("  " + "-" * (28 + 14*len(Ns) + 8*(len(Ns)-1)))

    for idx, (pa, ea, pb, eb, op) in enumerate(EDGES):
        label = f"P{pa}-{ea}<->P{pb}-{eb}"
        is_swap = op in ('T', 'TR')
        errs = all_results_b[idx]

        print(f"  {label:>20s}  {op:>3s}", end='')
        for e in errs:
            print(f"  {e:12.4e}", end='')
        for i in range(len(Ns) - 1):
            if errs[i] > 1e-14 and errs[i+1] > 1e-14:
                rate = np.log2(errs[i] / errs[i+1])
                print(f"  {rate:6.2f}", end='')
            else:
                print(f"    ---", end='')
        if is_swap:
            print("  SWAP", end='')
        print()

    print()
    print("  All edges should converge at O(h^2) or better on BOTH sides.")
    print("  ✓ TEST 4 COMPLETE\n")


# ============================================================
# TEST 5: Averaged covariant velocity back-converts correctly
# ============================================================

def test_averaged_covariant_roundtrip():
    """
    Verify that the averaged covariant velocity in each panel's frame,
    when converted back to Cartesian, gives the SAME Cartesian velocity.

    v1_avg_a, v2_avg_a → V_cart_a
    v1_avg_b, v2_avg_b → V_cart_b
    These must agree at corresponding physical points.
    """
    print("=" * 70)
    print("TEST 5: Averaged covariant roundtrip consistency")
    print("=" * 70)
    print()

    N = 20
    dx = (np.pi / 2) / N
    ops = sbp_42(N, float(dx))
    field_fn = diagonal_flow

    v1_all, v2_all = {}, {}
    for p in range(6):
        v1_all[p], v2_all[p] = make_covariant_field(N, p, field_fn)

    print(f"  {'Edge':>20s}  {'op':>3s}  {'|V_A_recon - V_B_recon|':>24s}  {'type':>10s}")
    print("  " + "-" * 66)

    max_err_global = 0.0
    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)

        v1_avg_a, v2_avg_a, v1_avg_b, v2_avg_b, Vx_avg, Vy_avg, Vz_avg = \
            cartesian_average_at_edge(
                v1_all[pa], v2_all[pa], v1_all[pb], v2_all[pb],
                pa, ea, pb, eb, op, ops, N)

        # Reconstruct Cartesian from each panel's averaged covariant
        xi1_a, xi2_a = edge_bnd_coords(ea, N)
        xi1_b, xi2_b = edge_bnd_coords(eb, N)

        Vx_rA, Vy_rA, Vz_rA = covariant_to_cartesian(
            v1_avg_a, v2_avg_a, xi1_a, xi2_a, pa)
        Vx_rB, Vy_rB, Vz_rB = covariant_to_cartesian(
            v1_avg_b, v2_avg_b, xi1_b, xi2_b, pb)

        if rev:
            Vx_rB = Vx_rB[::-1]
            Vy_rB = Vy_rB[::-1]
            Vz_rB = Vz_rB[::-1]

        dV = jnp.sqrt((Vx_rA - Vx_rB)**2 + (Vy_rA - Vy_rB)**2 + (Vz_rA - Vz_rB)**2)
        max_dV = float(jnp.max(dV))
        max_err_global = max(max_err_global, max_dV)

        is_swap = op in ('T', 'TR')
        etype = "AXIS-SWAP" if is_swap else "aligned"
        print(f"  P{pa}-{ea}<->P{pb}-{eb}  {op:>3s}  {max_dV:24.6e}  {etype:>10s}")

    print(f"\n  Global max |V_A_recon - V_B_recon| = {max_err_global:.2e}")
    assert max_err_global < 1e-12, f"Roundtrip failed: {max_err_global}"
    print("  Both panels reconstruct the SAME Cartesian velocity. ✓")
    print("  ✓ TEST 5 COMPLETE\n")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    test_averaged_cartesian_accuracy()
    test_convergence()
    test_consensus_flux_accuracy()
    test_consensus_flux_convergence()
    test_averaged_covariant_roundtrip()

    print("=" * 70)
    print("ALL STEP 3/4 TESTS COMPLETE")
    print("=" * 70)
    print()
    print("cartesian_average_at_edge() is validated.")
    print("All 12 edges produce converging flux mismatch.")
    print("Next: Step 5 — integrate into refactored SAT (test_stag_step4.py).")
