"""
SAT Operators for Cubed-Sphere Staggered SBP

Provides Simultaneous Approximation Term (SAT) corrections for coupling
panels on the cubed sphere. Two implementations:

  1. build_scalar_sat_fn()     â€” the original flux-averaging SAT (BROKEN
     on axis-swap edges due to coordinate mismatch)

  2. build_cartesian_sat_fn()  â€” the new Cartesian-averaged velocity SAT
     that works correctly on ALL 12 edges

The key insight: at axis-swap edges, the normal mass flux from panel A
is u1 = JÂ·v^1 while from panel B it is u2 = JÂ·v^2. These represent
DIFFERENT physical quantities in incompatible coordinate systems, so
averaging them directly produces O(1) error that GROWS with refinement.

The fix: average the VELOCITY VECTOR in Cartesian coordinates (coordinate-
free), then convert back to each panel's covariant frame to compute the
consensus mass flux.

Pipeline per edge (pa, ea) <-> (pb, eb):
  1. Extrapolate covariant (v_1, v_2) to h-grid boundary from both panels
  2. Convert each panel's boundary covariant velocity to Cartesian
  3. Align indices (reverse for R, TR edges)
  4. Average in Cartesian: V_avg = 0.5 * (V_A + V_B)
  5. Convert V_avg back to covariant in each panel's frame
  6. Compute consensus mass flux: J Â· Q^{n,j} Â· v_j_avg
  7. SAT penalty = -sign Â· Hv_inv Â· (flux_own - flux_consensus)

Energy conservation proof:
  Both panels compute flux_consensus from the SAME Cartesian V_avg,
  so sign_aÂ·flux_cons_a + sign_bÂ·flux_cons_b = 0 (same physical flux,
  opposite outward normals). Combined with h-projection (h_a = h_b at
  boundary), energy contributions telescope.

Dependencies:
  - sbp_staggered_1d.sbp_42 (SBP operators)
  - velocity_transforms (covariant <-> Cartesian)
  - grid.equiangular_to_cartesian
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from velocity_transforms import cartesian_to_covariant, covariant_to_cartesian


# ============================================================
# Edge connectivity (single source of truth: connectivity.py)
# ============================================================
from connectivity import EDGES


def _reverses(op):
    """Does this operation reverse the index order?"""
    return op in ('R', 'TR')


def _boundary_sign(edge):
    """Sign of boundary term in SBP identity: +1 for max, -1 for min."""
    return +1.0 if edge in ('E', 'N') else -1.0


def _hv_inv_index(edge, N):
    """Which Hv_inv element at this boundary?"""
    return N if edge in ('E', 'N') else 0


# ============================================================
# Extrapolate covariant velocity to h-grid boundary
# ============================================================

def extrapolate_covariant_to_boundary(v1, v2, edge, ops):
    """
    Extrapolate both covariant velocity components to an h-grid boundary.

    On the staggered grid:
      v1 at (xi_c, xi_v), shape (N, N+1)
      v2 at (xi_v, xi_c), shape (N+1, N)
      h  at (xi_v, xi_v), shape (N+1, N+1)

    For each edge, one component is extrapolated (cell->boundary in the
    normal direction) and the other is interpolated (cell->vertex in the
    tangential direction via Pcv).

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

    if edge == 'E':
        v1_bnd = jnp.einsum('c,cj->j', r, v1)   # extrapolate in xi1
        v2_bnd = Pcv @ v2[-1, :]                   # interpolate at xi1=max

    elif edge == 'W':
        v1_bnd = jnp.einsum('c,cj->j', l, v1)   # extrapolate in xi1
        v2_bnd = Pcv @ v2[0, :]                    # interpolate at xi1=min

    elif edge == 'N':
        v1_bnd = Pcv @ v1[:, -1]                   # interpolate at xi2=max
        v2_bnd = jnp.einsum('ic,c->i', v2, r)    # extrapolate in xi2

    elif edge == 'S':
        v1_bnd = Pcv @ v1[:, 0]                    # interpolate at xi2=min
        v2_bnd = jnp.einsum('ic,c->i', v2, l)    # extrapolate in xi2

    else:
        raise ValueError(f"Invalid edge: {edge}")

    return v1_bnd, v2_bnd


# ============================================================
# Boundary coordinate arrays
# ============================================================

def edge_bnd_coords(edge, N):
    """
    Get (xi1, xi2) coordinate arrays at h-grid boundary.

    Returns:
        xi1_bnd, xi2_bnd: each (N+1,)
    """
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


# ============================================================
# Cartesian averaging at one edge
# ============================================================

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
        v1_avg_a, v2_avg_a: (N+1,) averaged covariant in A's frame
        v1_avg_b, v2_avg_b: (N+1,) averaged covariant in B's frame
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

    # For panel B, un-reverse to B's index order before converting
    if rev:
        Vx_avg_b, Vy_avg_b, Vz_avg_b = Vx_avg[::-1], Vy_avg[::-1], Vz_avg[::-1]
    else:
        Vx_avg_b, Vy_avg_b, Vz_avg_b = Vx_avg, Vy_avg, Vz_avg

    v1_avg_b, v2_avg_b = cartesian_to_covariant(Vx_avg_b, Vy_avg_b, Vz_avg_b,
                                                  xi1_b, xi2_b, pb)

    return v1_avg_a, v2_avg_a, v1_avg_b, v2_avg_b, Vx_avg, Vy_avg, Vz_avg


# ============================================================
# Consensus mass flux from averaged covariant velocity
# ============================================================

def _consensus_flux(v1_avg, v2_avg, edge, J_bnd, Q11_bnd, Q12_bnd, Q22_bnd):
    """
    Compute normal mass flux from averaged covariant velocity.

    flux = J * v^n_contra = J * Q^{n,j} * v_j

    For edge E/W (normal = xi1): flux = J * (Q^11 * v1 + Q^12 * v2)
    For edge N/S (normal = xi2): flux = J * (Q^12 * v1 + Q^22 * v2)

    Args:
        v1_avg, v2_avg: (N+1,) averaged covariant velocity at boundary
        edge: 'E', 'W', 'N', 'S'
        J_bnd, Q11_bnd, Q12_bnd, Q22_bnd: metric at boundary, each (N+1,)

    Returns:
        flux: (N+1,) normal mass flux
    """
    if edge in ('E', 'W'):
        return J_bnd * (Q11_bnd * v1_avg + Q12_bnd * v2_avg)
    else:
        return J_bnd * (Q12_bnd * v1_avg + Q22_bnd * v2_avg)


# ============================================================
# Old SAT builder (preserved for comparison)
# ============================================================

def build_scalar_sat_fn(N, ops):
    """
    Build the ORIGINAL scalar flux-averaging SAT.

    WARNING: This produces O(1) error on 8 of 12 edges (axis-swap and
    certain aligned edges where cross-term Q12*v2 is nonzero). It is
    preserved only for regression testing.

    SAT formula:
      sat_a = -sign_a * 0.5 * Hv_inv * (flux_own + ss * flux_nbr_aligned)

    Args:
        N: grid resolution
        ops: SBP operators

    Returns:
        add_sat_correction(div, u1, u2) function
    """
    l = ops.l
    r = ops.r
    Hv_diag = jnp.diag(ops.Hv)
    Hv_inv = 1.0 / Hv_diag

    def extrapolate_flux(u1, u2, panel, edge):
        """Extrapolate mass flux to boundary. Returns (N+1,)."""
        if edge == 'E':
            return jnp.einsum('c,cj->j', r, u1[panel])
        elif edge == 'W':
            return jnp.einsum('c,cj->j', l, u1[panel])
        elif edge == 'N':
            return jnp.einsum('ic,c->i', u2[panel], r)
        elif edge == 'S':
            return jnp.einsum('ic,c->i', u2[panel], l)

    def add_sat_correction(div, u1, u2):
        """Add SAT corrections at all 12 edges."""
        for pa, ea, pb, eb, op in EDGES:
            rev = _reverses(op)
            idx_a = _hv_inv_index(ea, N)
            idx_b = _hv_inv_index(eb, N)
            sign_a = _boundary_sign(ea)
            sign_b = _boundary_sign(eb)
            ss = sign_a * sign_b

            flux_a = extrapolate_flux(u1, u2, pa, ea)
            flux_b = extrapolate_flux(u1, u2, pb, eb)
            flux_b_aligned = flux_b[::-1] if rev else flux_b

            sat_a = -sign_a * 0.5 * Hv_inv[idx_a] * (flux_a + ss * flux_b_aligned)

            if ea == 'N':
                div = div.at[pa, :, N].add(sat_a)
            elif ea == 'S':
                div = div.at[pa, :, 0].add(sat_a)
            elif ea == 'E':
                div = div.at[pa, N, :].add(sat_a)
            elif ea == 'W':
                div = div.at[pa, 0, :].add(sat_a)

            flux_a_aligned = flux_a[::-1] if rev else flux_a
            sat_b = -sign_b * 0.5 * Hv_inv[idx_b] * (flux_b + ss * flux_a_aligned)

            if eb == 'N':
                div = div.at[pb, :, N].add(sat_b)
            elif eb == 'S':
                div = div.at[pb, :, 0].add(sat_b)
            elif eb == 'E':
                div = div.at[pb, N, :].add(sat_b)
            elif eb == 'W':
                div = div.at[pb, 0, :].add(sat_b)

        return div

    return add_sat_correction


# ============================================================
# New Cartesian-averaged SAT builder
# ============================================================

def build_cartesian_sat_fn(N, ops, compute_metric_fn):
    """
    Build SAT correction function using Cartesian-averaged velocity.

    This correctly handles ALL 12 edges including axis-swap by averaging
    the velocity vector in Cartesian coordinates before computing the
    consensus mass flux in each panel's local frame.

    SAT formula:
      sat_a = -sign_a * Hv_inv[bnd] * (flux_own_a - flux_consensus_a)

    where:
      flux_own = extrapolated mass flux from the Dcv operator
      flux_consensus = J * Q^{n,j} * v_j_avg  (from Cartesian-averaged velocity)

    Args:
        N: grid resolution
        ops: SBP operators (needs l, r, Pcv, Hv)
        compute_metric_fn: function(xi1, xi2) -> (J, Q11, Q12, Q22)

    Returns:
        add_sat_correction(div, u1, u2, v1, v2) function
    """
    l = ops.l
    r = ops.r
    Hv_diag = jnp.diag(ops.Hv)
    Hv_inv = 1.0 / Hv_diag

    # Precompute metric at each edge's h-grid boundary points
    edge_metrics = {}
    for edge in ['E', 'W', 'N', 'S']:
        xi1_bnd, xi2_bnd = edge_bnd_coords(edge, N)
        J, Q11, Q12, Q22 = compute_metric_fn(xi1_bnd, xi2_bnd)
        edge_metrics[edge] = (J, Q11, Q12, Q22)

    def extrapolate_flux(u1, u2, panel, edge):
        """Extrapolate mass flux to boundary. Returns (N+1,)."""
        if edge == 'E':
            return jnp.einsum('c,cj->j', r, u1[panel])
        elif edge == 'W':
            return jnp.einsum('c,cj->j', l, u1[panel])
        elif edge == 'N':
            return jnp.einsum('ic,c->i', u2[panel], r)
        elif edge == 'S':
            return jnp.einsum('ic,c->i', u2[panel], l)

    def _apply_sat(div, panel, edge, sat):
        """Add SAT correction to divergence at the correct boundary."""
        if edge == 'N':
            return div.at[panel, :, N].add(sat)
        elif edge == 'S':
            return div.at[panel, :, 0].add(sat)
        elif edge == 'E':
            return div.at[panel, N, :].add(sat)
        elif edge == 'W':
            return div.at[panel, 0, :].add(sat)

    def add_sat_correction(div, u1, u2, v1, v2):
        """
        Add Cartesian-averaged SAT corrections at all 12 edges.

        Args:
            div: (6, N+1, N+1) divergence field to correct
            u1:  (6, N, N+1) mass flux in xi1 direction
            u2:  (6, N+1, N) mass flux in xi2 direction
            v1:  (6, N, N+1) covariant velocity v_1
            v2:  (6, N+1, N) covariant velocity v_2
        """
        for pa, ea, pb, eb, op in EDGES:
            rev = _reverses(op)
            idx_a = _hv_inv_index(ea, N)
            idx_b = _hv_inv_index(eb, N)
            sign_a = _boundary_sign(ea)
            sign_b = _boundary_sign(eb)

            # Extrapolated mass flux from Dcv operator (unchanged)
            flux_own_a = extrapolate_flux(u1, u2, pa, ea)
            flux_own_b = extrapolate_flux(u1, u2, pb, eb)

            # Cartesian-averaged covariant velocity at boundary
            v1_avg_a, v2_avg_a, v1_avg_b, v2_avg_b, _, _, _ = \
                cartesian_average_at_edge(
                    v1[pa], v2[pa], v1[pb], v2[pb],
                    pa, ea, pb, eb, op, ops, N)

            # Consensus mass flux from averaged velocity
            J_a, Q11_a, Q12_a, Q22_a = edge_metrics[ea]
            J_b, Q11_b, Q12_b, Q22_b = edge_metrics[eb]

            flux_cons_a = _consensus_flux(v1_avg_a, v2_avg_a, ea,
                                          J_a, Q11_a, Q12_a, Q22_a)
            flux_cons_b = _consensus_flux(v1_avg_b, v2_avg_b, eb,
                                          J_b, Q11_b, Q12_b, Q22_b)

            # SAT penalty: replace extrapolated flux with consensus
            sat_a = -sign_a * Hv_inv[idx_a] * (flux_own_a - flux_cons_a)
            sat_b = -sign_b * Hv_inv[idx_b] * (flux_own_b - flux_cons_b)

            div = _apply_sat(div, pa, ea, sat_a)
            div = _apply_sat(div, pb, eb, sat_b)

        return div

    return add_sat_correction
