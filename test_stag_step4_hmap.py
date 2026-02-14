"""
test_stag_step4.py â€” 6-Panel Cubed Sphere with Metrics (f=0)
============================================================

GOAL: Full cubed-sphere SWE with metric terms, 12-edge connectivity,
and SAT-Projection. No Coriolis (f=0) to isolate geometric effects.

PHYSICS (Shashkin Eq. 49, f=0):
    dv1/dt = -g âˆ‚h/âˆ‚x1
    dv2/dt = -g âˆ‚h/âˆ‚x2
    dh/dt  = -(H/J)(âˆ‚(JvÂ¹)/âˆ‚x1 + âˆ‚(JvÂ²)/âˆ‚x2)

  where (vÂ¹,vÂ²) = Q(v1,v2) are contravariant velocities.

GRID: Equiangular gnomonic, xi1,xi2 âˆˆ [-Ï€/4, Ï€/4]
  h:  (6, N+1, N+1) at vertices (xi_v, xi_v)
  v1: (6, N, N+1)   at x-faces  (xi_c, xi_v)
  v2: (6, N+1, N)   at y-faces  (xi_v, xi_c)

METRIC (same for all panels):
  J   = 1/(rÂ³ cosÂ²Î¾1 cosÂ²Î¾2),  rÂ² = 1+tanÂ²Î¾1+tanÂ²Î¾2
  QÂ¹Â¹ = râ´cosÂ²Î¾1cosÂ²Î¾2(1 - tanÂ²Î¾1/rÂ²)
  QÂ¹Â² = -râ´cosÂ²Î¾1cosÂ²Î¾2(tanÎ¾1Â·tanÎ¾2/rÂ²)
  QÂ²Â² = râ´cosÂ²Î¾1cosÂ²Î¾2(1 - tanÂ²Î¾2/rÂ²)

CONNECTIVITY: 12 edges from halo_exchange schedule
  h-projection: average at shared vertices (Eq. 51-52)
  SAT: average mass flux at interfaces (Eq. 53-55)

TARGETS:
  - Steady state: uniform h, zero v â†’ zero tendency
  - Mass conservation: machine precision
  - Energy: spatial exact (dt^p scaling)
  - Stable gravity wave propagation

Reference: Shashkin 2025, Sections 3-5

Usage:
    python test_stag_step4.py
"""
import sys
import os

project_dir = '/mnt/project'
if os.path.isdir(project_dir):
    sys.path.insert(0, project_dir)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial

from sbp_staggered_1d import sbp_42
from grid import equiangular_to_cartesian


# ============================================================
# Connectivity: 12 cubed-sphere edges
# ============================================================
# Format: (panel_a, edge_a, panel_b, edge_b, op)
# op: 'N'=identity, 'R'=reverse, 'T'=identity(axis swap), 'TR'=reverse(axis swap)
# For index mapping: 'N','T' â†’ kâ†”k; 'R','TR' â†’ kâ†”(N-k)
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

# 8 corners: each shared by 3 panels
# Format: [(panel, i, j), ...] for each corner
CORNERS = [
    [(0, 0, 0), (2, 0, 0), (3, 0, 0)],       # corner where F0-SW, F2-SW(?), F3-SW(?) meet -- we need to figure these out from the edges
]
# We'll compute corners from edge connectivity below


def _reverses(op):
    """Does this operation reverse the index order?"""
    return op in ('R', 'TR')


def _get_h_boundary(h_panel, edge, N):
    """Extract (N+1,) boundary from h of shape (N+1, N+1)."""
    if edge == 'N':
        return h_panel[:, N]     # along x1
    elif edge == 'S':
        return h_panel[:, 0]     # along x1
    elif edge == 'E':
        return h_panel[N, :]     # along x2
    elif edge == 'W':
        return h_panel[0, :]     # along x2


def _set_h_boundary(h, panel, edge, vals, N):
    """Set (N+1,) boundary in h of shape (6, N+1, N+1)."""
    if edge == 'N':
        return h.at[panel, :, N].set(vals)
    elif edge == 'S':
        return h.at[panel, :, 0].set(vals)
    elif edge == 'E':
        return h.at[panel, N, :].set(vals)
    elif edge == 'W':
        return h.at[panel, 0, :].set(vals)


def _edge_is_x1_boundary(edge):
    """Is this edge along the x1 boundary (normal = x1)?"""
    return edge in ('E', 'W')


def _flux_vel_component(edge):
    """Which velocity component provides normal flux?"""
    # E/W: normal is x1 â†’ flux from v1 (u1 = J1Â·v1_contra)
    # N/S: normal is x2 â†’ flux from v2 (u2 = J2Â·v2_contra)
    return 'v1' if edge in ('E', 'W') else 'v2'


def _extrap_vector(edge):
    """Which extrapolation vector (l or r) for this edge?"""
    # E (x1=max) â†’ r along x1
    # W (x1=min) â†’ l along x1
    # N (x2=max) â†’ r along x2
    # S (x2=min) â†’ l along x2
    return 'r' if edge in ('E', 'N') else 'l'


def _hv_inv_index(edge, N):
    """Which Hv_inv element at this boundary?"""
    return N if edge in ('E', 'N') else 0


# ============================================================
# Metric computation at staggered locations
# ============================================================

def compute_metric(xi1, xi2):
    """
    Compute metric terms at arbitrary (xi1, xi2) points.
    Same for all panels (equiangular property).

    Returns: J, Q11, Q12, Q22
    """
    t1 = jnp.tan(xi1)
    t2 = jnp.tan(xi2)
    c1 = jnp.cos(xi1)
    c2 = jnp.cos(xi2)
    r2 = 1.0 + t1**2 + t2**2
    r = jnp.sqrt(r2)

    J = 1.0 / (r**3 * c1**2 * c2**2)
    alpha = r**4 * c1**2 * c2**2
    Q11 = alpha * (1.0 - t1**2 / r2)
    Q12 = alpha * (-t1 * t2 / r2)
    Q22 = alpha * (1.0 - t2**2 / r2)

    return J, Q11, Q12, Q22


def make_staggered_grids(N):
    """
    Create coordinate arrays for staggered grid on [-Ï€/4, Ï€/4]Â².

    Returns: xi_v (N+1,), xi_c (N,), dx,
             and 2D coordinate arrays for h, v1, v2 grids.
    """
    L = jnp.pi / 2
    dx = L / N
    xi_v = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, N + 1)
    xi_c = (jnp.arange(N) + 0.5) * dx - jnp.pi / 4

    # 2D coordinates
    xi1_h, xi2_h = jnp.meshgrid(xi_v, xi_v, indexing='ij')   # (N+1, N+1)
    xi1_v1, xi2_v1 = jnp.meshgrid(xi_c, xi_v, indexing='ij')  # (N, N+1)
    xi1_v2, xi2_v2 = jnp.meshgrid(xi_v, xi_c, indexing='ij')  # (N+1, N)

    return {
        'xi_v': xi_v, 'xi_c': xi_c, 'dx': dx,
        'xi1_h': xi1_h, 'xi2_h': xi2_h,
        'xi1_v1': xi1_v1, 'xi2_v1': xi2_v1,
        'xi1_v2': xi1_v2, 'xi2_v2': xi2_v2,
    }


def make_all_metrics(grids):
    """Compute J and Q at all staggered locations."""
    Jh, _, Q12_h, _ = compute_metric(grids['xi1_h'], grids['xi2_h'])
    J1, Q11_1, _, _ = compute_metric(grids['xi1_v1'], grids['xi2_v1'])
    J2, _, _, Q22_2 = compute_metric(grids['xi1_v2'], grids['xi2_v2'])

    return {
        'Jh': Jh, 'J1': J1, 'J2': J2,
        'Q11_1': Q11_1, 'Q12_h': Q12_h, 'Q22_2': Q22_2,
    }


# ============================================================
# Contravariant velocity (Eq. 56)
# ============================================================

def compute_contravariant(v1, v2, metrics, Pvc, Pcv):
    """
    Compute contravariant velocities vÂ¹, vÂ² from covariant v1, v2.

    Eq. 56 â€” works for single panel (N,N+1) or batched (6,N,N+1).
    """
    Q11_1 = metrics['Q11_1']
    Q12_h = metrics['Q12_h']
    Q22_2 = metrics['Q22_2']
    Jh = metrics['Jh']
    J1 = metrics['J1']
    J2 = metrics['J2']

    JQ12 = Jh * Q12_h  # (N+1, N+1)

    # vÂ¹: off-diagonal term
    v2_at_h = v2 @ Pcv.T          # P2h
    cross_at_h = JQ12 * v2_at_h
    cross_at_v1 = Pvc @ cross_at_h  # Ph1
    v1_contra = Q11_1 * v1 + cross_at_v1 / J1

    # vÂ²: off-diagonal term
    v1_at_h = Pcv @ v1             # P1h
    cross_at_h2 = JQ12 * v1_at_h
    cross_at_v2 = cross_at_h2 @ Pvc.T  # Ph2
    v2_contra = cross_at_v2 / J2 + Q22_2 * v2

    return v1_contra, v2_contra


# ============================================================
# h-projection on cubed sphere (Eq. 51-52)
# ============================================================

def build_projection_fn(N, Jh, Hv_diag):
    """
    Build h-projection function for all 12 edges + 8 corners.

    For equiangular cubed sphere, J is the same on both sides of a shared
    vertex, so the weighted average (Eq. 51) simplifies to:
      (AhÂ·h)_m = (JhÂ·HvÂ·h_m + JhÂ·HvÂ·h_m*) / (2Â·JhÂ·Hv)
               = (h_m + h_m*) / 2

    Corner points (Eq. 52): average of 3 panels.
    """
    # Precompute corner assignments from edge connectivity
    # A corner is where two edges meet. Each panel has 4 corners at (i,j) = {0,N}Ã—{0,N}.
    # Build: for each (panel, i, j) corner, find all panels sharing that corner.

    corner_map = {}  # (panel, i, j) â†’ set of (panel, i, j) sharing this physical point

    # First, identify which panel corners are connected via edges
    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        # Get the two endpoint indices on each edge
        for k_a in [0, N]:
            k_b = (N - k_a) if rev else k_a
            # Convert (panel, edge, k) to (panel, i, j)
            ij_a = _edge_k_to_ij(ea, k_a, N)
            ij_b = _edge_k_to_ij(eb, k_b, N)
            key_a = (pa,) + ij_a
            key_b = (pb,) + ij_b
            # Union these corners
            if key_a not in corner_map:
                corner_map[key_a] = {key_a}
            if key_b not in corner_map:
                corner_map[key_b] = {key_b}
            corner_map[key_a].add(key_b)
            corner_map[key_b].add(key_a)

    # Transitively close: if Aâ†”B and Bâ†”C, then Aâ†”Bâ†”C
    changed = True
    while changed:
        changed = False
        for key in list(corner_map.keys()):
            group = set(corner_map[key])
            for member in list(group):
                if member in corner_map:
                    new = corner_map[member]
                    if not new.issubset(group):
                        group.update(new)
                        changed = True
            corner_map[key] = group

    # Deduplicate corner groups
    seen = set()
    corner_groups = []
    for key, group in corner_map.items():
        frozen = frozenset(group)
        if frozen not in seen and len(group) == 3:
            seen.add(frozen)
            corner_groups.append(sorted(group))

    # Build set of all corner (panel, i, j) for fast lookup
    corner_set = set()
    for group in corner_groups:
        for pij in group:
            corner_set.add(pij)

    def project_h(h):
        """
        Project h at all shared interfaces (Eq. 51-52).

        CRITICAL: Corner points (shared by 3 panels) must use original
        values for the 3-panel average (Eq. 52), not values already
        modified by 2-panel edge averaging (Eq. 51). We save originals
        first, then apply edge and corner averages independently.
        """
        h_orig = h  # save for corner averaging

        # 1. Edge averaging for non-corner points (Eq. 51)
        #    Corner endpoints will be overwritten in step 2, so it's
        #    OK to include them here — just don't read them back.
        for pa, ea, pb, eb, op in EDGES:
            rev = _reverses(op)
            # Always read from ORIGINAL h to avoid sequential contamination
            bnd_a = _get_h_boundary(h_orig[pa], ea, N)
            bnd_b = _get_h_boundary(h_orig[pb], eb, N)
            if rev:
                bnd_b = bnd_b[::-1]
            avg = 0.5 * (bnd_a + bnd_b)
            h = _set_h_boundary(h, pa, ea, avg, N)
            avg_b = avg[::-1] if rev else avg
            h = _set_h_boundary(h, pb, eb, avg_b, N)

        # 2. Corner averaging from ORIGINAL values (Eq. 52)
        for group in corner_groups:
            vals = jnp.array([h_orig[p, i, j] for p, i, j in group])
            avg = jnp.mean(vals)
            for p, i, j in group:
                h = h.at[p, i, j].set(avg)

        return h

    return project_h, corner_groups


def _edge_k_to_ij(edge, k, N):
    """Convert edge + index k to (i, j) in the (N+1, N+1) h-grid."""
    if edge == 'N':
        return (k, N)
    elif edge == 'S':
        return (k, 0)
    elif edge == 'E':
        return (N, k)
    elif edge == 'W':
        return (0, k)


# ============================================================
# SAT flux corrections on cubed sphere (Eq. 53-55)
# ============================================================

def build_sat_fn(N, ops, metrics):
    """
    Build SAT correction function for all 12 edges.

    CRITICAL SIGN CONVENTION:
    The SBP boundary term at each edge has a sign:
      East  (i=N): +1 (max boundary)
      West  (i=0): -1 (min boundary)
      North (j=N): +1 (max boundary)
      South (j=0): -1 (min boundary)

    For mass conservation, boundary contributions from adjacent panels
    must cancel: sign_a * flux_a + sign_b * flux_b â†’ 0 with SAT.

    For max-min connections (Eâ†”W): signs are +1,-1 â†’ naturally oppose
    For max-max (Nâ†”N, Eâ†”N): signs are +1,+1 â†’ SAT must negate neighbor
    For min-min (Sâ†”S, Sâ†”W): signs are -1,-1 â†’ SAT must negate neighbor
    """
    l = ops.l   # (N,) left extrapolation
    r = ops.r   # (N,) right extrapolation
    Hv_diag = jnp.diag(ops.Hv)  # (N+1,)
    Hv_inv = 1.0 / Hv_diag      # (N+1,)

    def _boundary_sign(edge):
        """Sign of boundary term in SBP identity: +1 for max, -1 for min."""
        return +1.0 if edge in ('E', 'N') else -1.0

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
        """Add SAT corrections at all 12 edges.
        
        Correct formula derived from SBP identity:
          At max boundary (sign=+1): SAT = -(1/2)*Hv_inv*(own + s_a*s_b*nbr)
          At min boundary (sign=-1): SAT = +(1/2)*Hv_inv*(own + s_a*s_b*nbr)
          Unified: SAT = -sign_own * (1/2) * Hv_inv * (own + s_a*s_b * nbr_aligned)
        """
        for pa, ea, pb, eb, op in EDGES:
            rev = _reverses(op)
            idx_a = _hv_inv_index(ea, N)
            idx_b = _hv_inv_index(eb, N)
            sign_a = _boundary_sign(ea)
            sign_b = _boundary_sign(eb)
            ss = sign_a * sign_b  # product of boundary signs

            flux_a = extrapolate_flux(u1, u2, pa, ea)
            flux_b = extrapolate_flux(u1, u2, pb, eb)

            if rev:
                flux_b_aligned = flux_b[::-1]
            else:
                flux_b_aligned = flux_b

            # SAT for panel a
            sat_a = -sign_a * 0.5 * Hv_inv[idx_a] * (flux_a + ss * flux_b_aligned)

            if ea == 'N':
                div = div.at[pa, :, N].add(sat_a)
            elif ea == 'S':
                div = div.at[pa, :, 0].add(sat_a)
            elif ea == 'E':
                div = div.at[pa, N, :].add(sat_a)
            elif ea == 'W':
                div = div.at[pa, 0, :].add(sat_a)

            # SAT for panel b
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
# Build full cubed-sphere SWE system
# ============================================================

def make_cubed_sphere_swe(N, H0, g):
    """
    Build cubed-sphere linearized SWE system (f=0).

    Equations (Shashkin Eq. 50 with F=0):
      dv1/dt = -g Dvc @ (Ah h)                    (gradient x1)
      dv2/dt = -g (Ah h) @ Dvc.T                  (gradient x2)
      dh/dt  = -H0 Ah Jhâ»Â¹ [Dcv@(J1vÂ¹) + (J2vÂ²)@Dcv.T + SAT]  (continuity)
    """
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)
    dx = float(grids['dx'])

    ops = sbp_42(N, dx)
    Dvc = ops.Dvc
    Dcv = ops.Dcv
    Pvc = ops.Pvc
    Pcv = ops.Pcv
    Hv_diag = jnp.diag(ops.Hv)
    Hc_diag = jnp.diag(ops.Hc)

    project_h, corners = build_projection_fn(N, metrics['Jh'], Hv_diag)
    add_sat = build_sat_fn(N, ops, metrics)

    Jh = metrics['Jh']
    J1 = metrics['J1']
    J2 = metrics['J2']
    Jh_inv = 1.0 / Jh

    # 2D quadrature weights (per panel)
    Wh = jnp.outer(Hv_diag, Hv_diag)   # (N+1, N+1)
    W1 = jnp.outer(Hc_diag, Hv_diag)   # (N, N+1)
    W2 = jnp.outer(Hv_diag, Hc_diag)   # (N+1, N)

    def rhs(h, v1, v2):
        """
        RHS of linearized SWE on cubed sphere (f=0).

        h:  (6, N+1, N+1)
        v1: (6, N, N+1)
        v2: (6, N+1, N)
        """
        # Project h at all interfaces
        h_proj = project_h(h)

        # === Gradient (momentum equations) ===
        # Batch over panels: Dvc @ h_proj[p] for all p
        dv1_dt = -g * jnp.einsum('ij,pjk->pik', Dvc, h_proj)  # (6,N,N+1)
        dv2_dt = -g * jnp.einsum('pij,kj->pik', h_proj, Dvc)  # (6,N+1,N)

        # === Contravariant velocity (vmap over panels) ===
        v1c, v2c = _contra_vmap(v1, v2)
        u1_all = J1 * v1c   # (6, N, N+1) mass flux
        u2_all = J2 * v2c   # (6, N+1, N) mass flux

        # === Divergence ===
        div = (jnp.einsum('ij,pjk->pik', Dcv, u1_all) +
               jnp.einsum('pij,kj->pik', u2_all, Dcv))  # (6,N+1,N+1)

        # Add SAT corrections at all 12 edges
        div = add_sat(div, u1_all, u2_all)

        # Continuity: dh/dt = -H0 Â· project(Jhâ»Â¹ Â· div)
        dh_dt = project_h(-H0 * Jh_inv * div)

        return dh_dt, dv1_dt, dv2_dt

    # vmap contravariant velocity over panel axis
    def _contra_single(v1_p, v2_p):
        return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
    _contra_vmap = jax.vmap(_contra_single)

    # JIT compile the RHS (static control flow in project_h/add_sat is fine)
    rhs_jit = jax.jit(rhs)

    return {
        'rhs': rhs_jit,
        'project_h': project_h,
        'grids': grids,
        'metrics': metrics,
        'ops': ops,
        'corners': corners,
        'Wh': Wh, 'W1': W1, 'W2': W2,
        'Jh': Jh, 'J1': J1, 'J2': J2,
        'Pvc': Pvc, 'Pcv': Pcv,
        'N': N, 'dx': dx,
    }


# ============================================================
# RK4
# ============================================================

def make_rk4_step(rhs_fn):
    @jax.jit
    def step(h, v1, v2, dt):
        k1h, k1v1, k1v2 = rhs_fn(h, v1, v2)
        k2h, k2v1, k2v2 = rhs_fn(h + 0.5*dt*k1h, v1 + 0.5*dt*k1v1, v2 + 0.5*dt*k1v2)
        k3h, k3v1, k3v2 = rhs_fn(h + 0.5*dt*k2h, v1 + 0.5*dt*k2v1, v2 + 0.5*dt*k2v2)
        k4h, k4v1, k4v2 = rhs_fn(h + dt*k3h, v1 + dt*k3v1, v2 + dt*k3v2)
        return (h  + (dt/6)*(k1h  + 2*k2h  + 2*k3h  + k4h),
                v1 + (dt/6)*(k1v1 + 2*k2v1 + 2*k3v1 + k4v1),
                v2 + (dt/6)*(k1v2 + 2*k2v2 + 2*k3v2 + k4v2))
    return step


# ============================================================
# Diagnostics
# ============================================================

def compute_mass(h, Wh, Jh):
    """Global mass = sum_panels âˆ« hÂ·J dA = sum h Â· Jh Â· Wh"""
    return float(jnp.sum(h * Jh[None, :, :] * Wh[None, :, :]))


def compute_energy(h, v1, v2, Wh, W1, W2, Jh, J1, J2, g, H0,
                   metrics, Pvc, Pcv):
    """
    Total energy E = (g/2)âˆ«hÂ²J dA + (H0/2)âˆ«(v1Â·vÂ¹ + v2Â·vÂ²)J dA
    """
    # PE: (g/2) sum_p hÂ²Â·JhÂ·Wh
    PE = 0.5 * g * float(jnp.sum(h**2 * Jh[None, :, :] * Wh[None, :, :]))

    # KE: vmap contravariant velocity
    def _contra_single(v1_p, v2_p):
        return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
    v1c, v2c = jax.vmap(_contra_single)(v1, v2)

    KE = 0.5 * H0 * (
        float(jnp.sum(v1 * J1 * v1c * W1[None, :, :])) +
        float(jnp.sum(v2 * J2 * v2c * W2[None, :, :]))
    )

    return PE + KE


def compute_energy_simple(h, v1, v2, Wh, W1, W2, Jh, J1, J2, g, H0):
    """Simplified energy using diagonal metric only (Q12â‰ˆ0 near center)."""
    PE = 0.5 * g * float(jnp.sum(h**2 * Jh[None, :, :] * Wh[None, :, :]))
    # Approximate KE with diagonal metric only
    KE = 0.5 * H0 * (float(jnp.sum(v1**2 * J1[None, :, :] * W1[None, :, :])) +
                       float(jnp.sum(v2**2 * J2[None, :, :] * W2[None, :, :])))
    return PE + KE


# ============================================================
# Test 1: Metric verification
# ============================================================

def test_metrics():
    """
    Verify metric terms:
    1. J > 0 everywhere
    2. Q symmetric: Q12 = Q21
    3. Q positive definite: Q11Â·Q22 - Q12Â² > 0
    4. det(Q) = 1/JÂ²
    5. Same metric at shared vertices across panels
    """
    print("\n" + "=" * 65)
    print("TEST 1: Metric Verification")
    print("=" * 65)

    N = 16
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)

    # Check J > 0
    Jh = metrics['Jh']
    J1 = metrics['J1']
    J2 = metrics['J2']
    j_min = min(float(jnp.min(Jh)), float(jnp.min(J1)), float(jnp.min(J2)))
    print(f"  J > 0:  min(J) = {j_min:.6e}  {'âœ“' if j_min > 0 else 'âœ—'}")

    # Check Q positive definite at h-points
    Q11_h, Q12_h, Q22_h = [compute_metric(grids['xi1_h'], grids['xi2_h'])[i] for i in [1, 2, 3]]
    det_Q = Q11_h * Q22_h - Q12_h**2
    det_min = float(jnp.min(det_Q))
    print(f"  Q p.d.: min(det Q) = {det_min:.6e}  {'âœ“' if det_min > 0 else 'âœ—'}")

    # Check det(Q) = 1/JÂ²
    inv_J2 = 1.0 / Jh**2
    det_err = float(jnp.max(jnp.abs(det_Q - inv_J2) / inv_J2))
    print(f"  det(Q) = 1/JÂ²: max rel err = {det_err:.2e}  {'âœ“' if det_err < 1e-12 else 'âœ—'}")

    # Check shared vertices have same metric (equiangular property)
    # Corner (-Ï€/4, -Ï€/4) is the same physical point for panels 0(SW), 2(SW?), 3(SW?)
    # Since metric depends only on (xi1, xi2), all panels share the same values
    J_corner = float(Jh[0, 0])
    J_center = float(Jh[N // 2, N // 2])
    print(f"  J at corner: {J_corner:.6f}, at center: {J_center:.6f}")
    J_corner_exact = 4 * np.sqrt(3) / 9  # 1/(3âˆš3Â·(1/4)) at (Ï€/4,Ï€/4)
    print(f"  J_corner/J_center = {J_corner/J_center:.4f} (expect 4âˆš3/9 â‰ˆ {J_corner_exact:.4f})")

    passed = j_min > 0 and det_min > 0 and det_err < 1e-12
    print(f"  {'âœ“ PASS' if passed else 'âœ— FAIL'}")
    return passed


# ============================================================
# Test 2: Steady state (uniform h, zero v)
# ============================================================

def test_steady_state():
    """
    Uniform h with zero velocity should give zero tendency.

    This tests that the gradient of a constant is zero and
    the divergence of zero flux is zero, even with metric terms.
    """
    print("\n" + "=" * 65)
    print("TEST 2: Steady State (uniform h, v=0)")
    print("=" * 65)

    N = 16
    sys_d = make_cubed_sphere_swe(N, H0=1.0, g=1.0)

    h = jnp.ones((6, N + 1, N + 1))
    v1 = jnp.zeros((6, N, N + 1))
    v2 = jnp.zeros((6, N + 1, N))

    dh, dv1, dv2 = sys_d['rhs'](h, v1, v2)

    max_dh = float(jnp.max(jnp.abs(dh)))
    max_dv1 = float(jnp.max(jnp.abs(dv1)))
    max_dv2 = float(jnp.max(jnp.abs(dv2)))

    print(f"  max|dh/dt|  = {max_dh:.2e}")
    print(f"  max|dv1/dt| = {max_dv1:.2e}")
    print(f"  max|dv2/dt| = {max_dv2:.2e}")

    passed = max_dh < 1e-12 and max_dv1 < 1e-12 and max_dv2 < 1e-12
    print(f"  {'âœ“ PASS' if passed else 'âœ— FAIL'}")
    return passed


# ============================================================
# Test 3: h-projection connectivity
# ============================================================

def test_projection():
    """
    Verify h-projection produces continuous field at all 12 edges.

    Set h = smooth function of (X,Y,Z) on the sphere.
    After projection, values at shared vertices should match.
    """
    print("\n" + "=" * 65)
    print("TEST 3: h-Projection Connectivity")
    print("=" * 65)

    N = 16
    sys_d = make_cubed_sphere_swe(N, H0=1.0, g=1.0)
    grids = sys_d['grids']
    project_h = sys_d['project_h']

    xi_v = grids['xi_v']

    # Set h = Yâ‚â° âˆ Z on the sphere (continuous, smooth)
    h = jnp.zeros((6, N + 1, N + 1))
    for p in range(6):
        xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')
        X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
        h = h.at[p].set(Z)

    h_proj = project_h(h)

    # Check continuity at all 12 edges
    max_jump = 0.0
    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        bnd_a = _get_h_boundary(h_proj[pa], ea, N)
        bnd_b = _get_h_boundary(h_proj[pb], eb, N)
        if rev:
            bnd_b = bnd_b[::-1]
        jump = float(jnp.max(jnp.abs(bnd_a - bnd_b)))
        max_jump = max(max_jump, jump)

    print(f"  Max edge discontinuity after projection: {max_jump:.2e}")

    # Check corners
    corners = sys_d['corners']
    max_corner = 0.0
    for group in corners:
        vals = [float(h_proj[p, i, j]) for p, i, j in group]
        spread = max(vals) - min(vals)
        max_corner = max(max_corner, spread)

    print(f"  Max corner spread: {max_corner:.2e}")
    print(f"  Found {len(corners)} corner groups of 3")

    passed = max_jump < 1e-14 and max_corner < 1e-14
    print(f"  {'âœ“ PASS' if passed else 'âœ— FAIL'}")
    return passed


# ============================================================
# Test 4: Mass conservation
# ============================================================

def test_mass_conservation():
    """
    Gaussian perturbation on one panel â†’ gravity waves.
    Check mass conservation over ~50 time steps.
    """
    print("\n" + "=" * 65)
    print("TEST 4: Mass Conservation (Gaussian perturbation)")
    print("=" * 65)

    N = 16; H0 = 1.0; g = 1.0
    sys_d = make_cubed_sphere_swe(N, H0, g)
    Wh = sys_d['Wh']; Jh = sys_d['Jh']
    grids = sys_d['grids']
    xi_v = grids['xi_v']
    dx = sys_d['dx']

    # Gaussian centered on panel 0
    h = jnp.zeros((6, N + 1, N + 1))
    xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')
    sigma = 0.2  # broad enough to be well-resolved
    h = h.at[0].set(0.1 * jnp.exp(-(xi1_2d**2 + xi2_2d**2) / (2 * sigma**2)))
    v1 = jnp.zeros((6, N, N + 1))
    v2 = jnp.zeros((6, N + 1, N))

    mass0 = compute_mass(h, Wh, Jh)
    step_fn = make_rk4_step(sys_d['rhs'])

    c = np.sqrt(g * H0)
    CFL = 0.3
    dt = CFL * dx / c
    nsteps = 100

    print(f"  N = {N}, CFL = {CFL}, dt = {dt:.6e}, steps = {nsteps}")
    print(f"  Initial mass: {mass0:.10e}")
    print(f"  JIT compiling RK4 step (first call)...", flush=True)

    # Warmup JIT
    import time as _time
    t0 = _time.time()
    h, v1, v2 = step_fn(h, v1, v2, dt)
    jax.block_until_ready(h)
    print(f"  JIT compiled in {_time.time()-t0:.1f}s", flush=True)

    max_merr = 0.0
    for s in range(1, nsteps):  # already did step 0 as warmup
        h, v1, v2 = step_fn(h, v1, v2, dt)
        if (s + 1) % 25 == 0:
            mass = compute_mass(h, Wh, Jh)
            merr = abs(mass - mass0)
            max_merr = max(max_merr, merr)
            max_h = float(jnp.max(jnp.abs(h)))
            print(f"  Step {s+1:4d}: mass_err = {merr:.2e}, max|h| = {max_h:.4e}")

    stable = float(jnp.max(jnp.abs(h))) < 1.0  # shouldn't blow up
    passed = max_merr < 1e-10 and stable
    print(f"\n  Max mass error: {max_merr:.2e}")
    print(f"  Stable: {'yes' if stable else 'NO'}")
    print(f"  {'âœ“ PASS' if passed else 'âœ— FAIL'}")
    return passed


# ============================================================
# Test 5: Energy CFL sweep
# ============================================================

def test_energy():
    """
    Verify energy error scales with dt (spatial energy-exact).
    """
    print("\n" + "=" * 65)
    print("TEST 5: Energy Conservation â€” CFL sweep")
    print("=" * 65)

    N = 12; H0 = 1.0; g = 1.0
    sys_d = make_cubed_sphere_swe(N, H0, g)
    Wh = sys_d['Wh']; W1 = sys_d['W1']; W2 = sys_d['W2']
    Jh = sys_d['Jh']; J1 = sys_d['J1']; J2 = sys_d['J2']
    metrics = sys_d['metrics']
    Pvc = sys_d['Pvc']; Pcv = sys_d['Pcv']
    grids = sys_d['grids']
    xi_v = grids['xi_v']
    dx = sys_d['dx']

    # IC: Gaussian on panel 0
    h0 = jnp.zeros((6, N + 1, N + 1))
    xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')
    h0 = h0.at[0].set(0.1 * jnp.exp(-(xi1_2d**2 + xi2_2d**2) / (2 * 0.2**2)))
    v1_0 = jnp.zeros((6, N, N + 1))
    v2_0 = jnp.zeros((6, N + 1, N))

    E0 = compute_energy(h0, v1_0, v2_0, Wh, W1, W2, Jh, J1, J2, g, H0,
                        metrics, Pvc, Pcv)

    c = np.sqrt(g * H0)
    step_fn = make_rk4_step(sys_d['rhs'])

    CFLs = [0.4, 0.2, 0.1, 0.05]
    T_end = 20 * dx / c  # short integration
    results = []

    print(f"  N = {N}, T = {T_end:.4f}")
    print(f"  JIT compiling...", flush=True)

    # Warmup JIT with smallest CFL (most steps, compile once)
    import time as _time
    t0 = _time.time()
    _h, _v1, _v2 = step_fn(h0, v1_0, v2_0, 0.01 * dx / c)
    jax.block_until_ready(_h)
    print(f"  JIT compiled in {_time.time()-t0:.1f}s", flush=True)

    print(f"\n  {'CFL':>6} {'dt':>12} {'steps':>7} {'Î”E/E':>12} {'rate':>8}")
    print("  " + "-" * 50)

    for CFL in CFLs:
        dt = CFL * dx / c
        nsteps = max(int(np.ceil(T_end / dt)), 1)
        dt = T_end / nsteps

        h, v1, v2 = h0.copy(), v1_0.copy(), v2_0.copy()
        for _ in range(nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)

        Ef = compute_energy(h, v1, v2, Wh, W1, W2, Jh, J1, J2, g, H0,
                            metrics, Pvc, Pcv)
        rel = abs(Ef - E0) / abs(E0)

        if results:
            rate = np.log2(results[-1]['err'] / max(rel, 1e-16)) / \
                   np.log2(results[-1]['dt'] / dt)
            rate_str = f"{rate:8.2f}"
        else:
            rate_str = "     ---"

        results.append({'CFL': CFL, 'dt': dt, 'err': rel})
        print(f"  {CFL:6.3f} {dt:12.6e} {nsteps:7d} {rel:12.4e} {rate_str}")

    if len(results) >= 2 and results[-1]['err'] > 1e-15:
        final_rate = np.log2(results[-2]['err'] / max(results[-1]['err'], 1e-16)) / \
                     np.log2(results[-2]['dt'] / results[-1]['dt'])
    else:
        final_rate = 0.0

    print(f"\n  dt-scaling rate: {final_rate:.2f}")
    passed = final_rate > 3.0
    print(f"  {'âœ“ PASS' if passed else 'âœ— FAIL'}")
    return passed


# ============================================================
# Test 6: Gravity wave stability
# ============================================================

def gaussian_hill_ic(N, grids, variant=1):
    """
    Gaussian hill initial condition from Shashkin Section 6.2.

    h(t=0) = exp(-16 r^2 / a^2) on unit sphere (a=1).
    r = great-circle arc distance from center point.

    Variant 1: center at face 0 center = (0, 0, 1)
               Tests wave propagation through panel edges.
    Variant 2: center at cube vertex = (1,1,1)/sqrt(3)
               Tests wave propagation through panel corners (3-panel junctions).
               Faces 0 (+Z), 1 (+Y), 4 (+X) share this vertex.

    Returns: h (6, N+1, N+1), center (3,) Cartesian
    """
    if variant == 1:
        center = jnp.array([0.0, 0.0, 1.0])
    elif variant == 2:
        center = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)
    else:
        raise ValueError(f"Unknown variant {variant}")

    xi1_h = grids['xi1_h']
    xi2_h = grids['xi2_h']

    def _panel_h(p):
        X, Y, Z = equiangular_to_cartesian(xi1_h, xi2_h, p)
        dot = jnp.clip(X * center[0] + Y * center[1] + Z * center[2], -1.0, 1.0)
        return jnp.exp(-16.0 * jnp.arccos(dot)**2)

    h = jnp.stack([_panel_h(p) for p in range(6)])  # (6, N+1, N+1)
    return h, center


def _plot_error_heatmaps(diff, h_fine, N_c, N_f, variant, T_end):
    """
    Plot self-convergence error on all 6 panels + solution.

    diff:   (6, N_c+1, N_c+1) — error field on coarse grid
    h_fine: (6, N_f+1, N_f+1) — fine solution for context
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    face_names = ['+Z (0)', '+Y (1)', '-X (2)', '-Y (3)', '+X (4)', '-Z (5)']

    # --- Figure 1: Error heatmaps on all 6 panels ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f'Gaussian Hill Variant {variant}: Self-convergence error '
        f'(N={N_c} vs N={N_f}, T={T_end:.2f})',
        fontsize=14)

    vmax_err = float(jnp.max(jnp.abs(diff)))
    for p in range(6):
        ax = axes[p // 3, p % 3]
        err_p = np.array(diff[p])
        im = ax.imshow(err_p.T, origin='lower', cmap='RdBu_r',
                        vmin=-vmax_err, vmax=vmax_err,
                        aspect='equal', extent=[-45, 45, -45, 45])
        ax.set_title(f'Panel {face_names[p]}', fontsize=11)
        ax.set_xlabel('ξ₁ (deg)')
        ax.set_ylabel('ξ₂ (deg)')
        # Mark corners
        for ci in [-45, 45]:
            for cj in [-45, 45]:
                ax.plot(ci, cj, 'k+', markersize=8, markeredgewidth=1.5)

    fig.colorbar(im, ax=axes, shrink=0.6, label='h error')
    plt.tight_layout()
    plt.savefig(f'/mnt/user-data/outputs/gauss{variant}_error_panels.png', dpi=150)
    plt.close()

    # --- Figure 2: Solution heatmaps on all 6 panels ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle(
        f'Gaussian Hill Variant {variant}: Solution h at T={T_end:.2f} '
        f'(N={N_f})',
        fontsize=14)

    vmax_sol = float(jnp.max(jnp.abs(h_fine)))
    for p in range(6):
        ax = axes2[p // 3, p % 3]
        sol_p = np.array(h_fine[p])
        im2 = ax.imshow(sol_p.T, origin='lower', cmap='RdBu_r',
                         vmin=-vmax_sol, vmax=vmax_sol,
                         aspect='equal', extent=[-45, 45, -45, 45])
        ax.set_title(f'Panel {face_names[p]}', fontsize=11)
        ax.set_xlabel('ξ₁ (deg)')
        ax.set_ylabel('ξ₂ (deg)')
        for ci in [-45, 45]:
            for cj in [-45, 45]:
                ax.plot(ci, cj, 'k+', markersize=8, markeredgewidth=1.5)

    fig2.colorbar(im2, ax=axes2, shrink=0.6, label='h')
    plt.tight_layout()
    plt.savefig(f'/mnt/user-data/outputs/gauss{variant}_solution_panels.png', dpi=150)
    plt.close()

    # --- Figure 3: Error along edges and diagonals ---
    # Extract error along panel boundaries to see if edges dominate
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle(
        f'Gaussian Hill Variant {variant}: Error structure (N={N_c} vs N={N_f})',
        fontsize=14)

    # Top-left: max |error| per row/col for each panel (edge proximity)
    ax = axes3[0, 0]
    for p in range(6):
        err_p = np.abs(np.array(diff[p]))
        # Max error in each row (along ξ₁)
        max_per_row = err_p.max(axis=1)
        xi = np.linspace(-45, 45, len(max_per_row))
        ax.semilogy(xi, max_per_row, label=f'P{p}')
    ax.set_xlabel('ξ₂ (deg)')
    ax.set_ylabel('max|error| along ξ₁')
    ax.set_title('Error profile along ξ₂ (each panel)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Top-right: max |error| along each edge of all panels
    ax = axes3[0, 1]
    edge_labels = []
    edge_errors = []
    for p in range(6):
        err_p = np.abs(np.array(diff[p]))
        N_c_loc = err_p.shape[0] - 1
        edges_data = {
            'S': err_p[:, 0],
            'N': err_p[:, N_c_loc],
            'W': err_p[0, :],
            'E': err_p[N_c_loc, :],
        }
        for ename, evals in edges_data.items():
            edge_labels.append(f'P{p}-{ename}')
            edge_errors.append(evals.max())
    ax.barh(range(len(edge_labels)), edge_errors, color='steelblue')
    ax.set_yticks(range(len(edge_labels)))
    ax.set_yticklabels(edge_labels, fontsize=7)
    ax.set_xlabel('max|error| on edge')
    ax.set_title('Max error per panel edge')
    ax.grid(True, axis='x', alpha=0.3)

    # Bottom-left: error at 8 corners
    ax = axes3[1, 0]
    corner_labels = []
    corner_errors = []
    for p in range(6):
        err_p = np.abs(np.array(diff[p]))
        N_c_loc = err_p.shape[0] - 1
        for (i, j), label in [((0, 0), 'SW'), ((N_c_loc, 0), 'SE'),
                                ((0, N_c_loc), 'NW'), ((N_c_loc, N_c_loc), 'NE')]:
            corner_labels.append(f'P{p}-{label}')
            corner_errors.append(err_p[i, j])
    ax.barh(range(len(corner_labels)), corner_errors, color='coral')
    ax.set_yticks(range(len(corner_labels)))
    ax.set_yticklabels(corner_labels, fontsize=6)
    ax.set_xlabel('|error| at corner')
    ax.set_title('Error at panel corners')
    ax.grid(True, axis='x', alpha=0.3)

    # Bottom-right: error histogram (interior vs boundary)
    ax = axes3[1, 1]
    for p in range(6):
        err_p = np.abs(np.array(diff[p]))
        N_c_loc = err_p.shape[0] - 1
        interior = err_p[2:-2, 2:-2].ravel()
        boundary = np.concatenate([err_p[0, :], err_p[N_c_loc, :],
                                    err_p[1:-1, 0], err_p[1:-1, N_c_loc]])
        if p == 0:
            ax.hist(interior, bins=50, alpha=0.5, label='Interior', color='steelblue')
            ax.hist(boundary, bins=50, alpha=0.5, label='Boundary', color='coral')
        else:
            ax.hist(interior, bins=50, alpha=0.3, color='steelblue')
            ax.hist(boundary, bins=50, alpha=0.3, color='coral')
    ax.set_xlabel('|error|')
    ax.set_ylabel('count')
    ax.set_title('Error distribution: interior vs boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/mnt/user-data/outputs/gauss{variant}_error_structure.png', dpi=150)
    plt.close()

    print(f"\n  Plots saved: gauss{variant}_error_panels.png")
    print(f"               gauss{variant}_solution_panels.png")
    print(f"               gauss{variant}_error_structure.png")


def test_gaussian_hill(variant=1):
    """
    Shashkin Section 6.2: Gaussian hill gravity wave propagation (f=0).

    Variant 1: Panel center — waves cross edges.
    Variant 2: Cube vertex — waves cross 3-panel corners from t=0.

    Runs at multiple N for self-convergence (N-to-2N subsampling).
    Reports mass conservation, energy conservation, stability, convergence rates.
    """
    print("\n" + "=" * 65)
    print(f"TEST: Gaussian Hill Variant {variant} (Shashkin 6.2, f=0)")
    loc = "panel center (0,0,1)" if variant == 1 else "cube vertex (1,1,1)/sqrt(3)"
    print(f"  Center: {loc}")
    print("=" * 65)

    g = 1.0; H0 = 1.0
    c = np.sqrt(g * H0)
    T_end = np.pi   # half circumference — waves refocus at antipodal point

    N_list = [10, 20, 40, 80]
    CFL = 0.3
    solutions = {}

    for idx, N in enumerate(N_list):
        sys_d = make_cubed_sphere_swe(N, H0, g)
        grids = sys_d['grids']
        Wh = sys_d['Wh']; Jh = sys_d['Jh']
        dx = sys_d['dx']

        h0, center = gaussian_hill_ic(N, grids, variant)
        v1_0 = jnp.zeros((6, N, N + 1))
        v2_0 = jnp.zeros((6, N + 1, N))

        mass0 = compute_mass(h0, Wh, Jh)
        E0 = compute_energy(h0, v1_0, v2_0, Wh, sys_d['W1'], sys_d['W2'],
                            Jh, sys_d['J1'], sys_d['J2'], g, H0,
                            sys_d['metrics'], sys_d['Pvc'], sys_d['Pcv'])

        step_fn = make_rk4_step(sys_d['rhs'])

        dt = CFL * dx / c
        nsteps = int(np.ceil(T_end / dt))
        dt = T_end / nsteps

        # JIT warmup + timing
        import time as _time
        t0 = _time.time()
        _h, _v1, _v2 = step_fn(h0, v1_0, v2_0, dt)
        jax.block_until_ready(_h)
        jit_time = _time.time() - t0
        if idx == 0:
            print(f"  JIT warmup: {jit_time:.1f}s", flush=True)

        # Time stepping with lax.fori_loop for GPU efficiency
        def scan_step(carry, _):
            h, v1, v2 = carry
            h, v1, v2 = step_fn(h, v1, v2, dt)
            return (h, v1, v2), None

        t0 = _time.time()
        (h, v1, v2), _ = jax.lax.scan(scan_step, (h0, v1_0, v2_0), None, length=nsteps)
        jax.block_until_ready(h)
        wall_time = _time.time() - t0

        mass_f = compute_mass(h, Wh, Jh)
        E_f = compute_energy(h, v1, v2, Wh, sys_d['W1'], sys_d['W2'],
                             Jh, sys_d['J1'], sys_d['J2'], g, H0,
                             sys_d['metrics'], sys_d['Pvc'], sys_d['Pcv'])

        mass_err = abs(mass_f - mass0)
        energy_err = abs(E_f - E0) / abs(E0)
        max_h = float(jnp.max(jnp.abs(h)))
        panel_max = [float(jnp.max(jnp.abs(h[p]))) for p in range(6)]
        n_active = sum(1 for m in panel_max if m > 1e-6)

        print(f"\n  N={N:3d}  steps={nsteps:5d}  dt={float(dt):.3e}  wall={wall_time:.1f}s")
        print(f"    mass err: {mass_err:.2e}   energy err: {energy_err:.2e}")
        print(f"    max|h|: {max_h:.4e}   active panels: {n_active}/6")

        solutions[N] = h

    # Self-convergence: subsample 2N solution onto N grid
    # With K pairs we get K errors and K-1 rates
    print(f"\n  Self-convergence (h at T={T_end:.2f}):")
    print(f"  {'pair':>8s}  {'l2 err':>10s}  {'rate':>6s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*6}")

    errors = []
    pairs = []
    ok = True

    for i in range(len(N_list) - 1):
        N_c = N_list[i]
        N_f = N_list[i + 1]
        assert N_f == 2 * N_c, f"Need doubling: {N_c} -> {N_f}"

        h_c = solutions[N_c]
        h_f = solutions[N_f]

        # Subsample fine to coarse: every other point
        h_f_sub = h_f[:, ::2, ::2]

        # L2 error (weighted by Jh*Wh on coarse grid)
        sys_c = make_cubed_sphere_swe(N_c, H0, g)
        Wh_c = sys_c['Wh']; Jh_c = sys_c['Jh']
        diff = h_c - h_f_sub
        l2_err = float(jnp.sqrt(jnp.sum(diff**2 * Jh_c[None] * Wh_c[None])))

        errors.append(l2_err)
        pairs.append((N_c, N_f))

        rate_str = "   —"
        if len(errors) >= 2:
            rate = np.log(errors[-2] / errors[-1]) / np.log(2.0)
            rate_str = f"{rate:6.2f}"

        print(f"  {N_c:3d}->{N_f:<3d}  {l2_err:10.2e}  {rate_str}")

        if l2_err > 1.0:
            ok = False

    # === Error heat-maps ===
    # Use the finest pair for visualization
    N_c = pairs[-1][0]
    N_f = pairs[-1][1]
    h_c = solutions[N_c]
    h_f_sub = solutions[N_f][:, ::2, ::2]
    diff = h_c - h_f_sub

    # Also get the solution itself for context
    h_sol = solutions[N_f]

    _plot_error_heatmaps(diff, h_sol, N_c, N_f, variant, T_end)

    # Check final solution is stable
    h_final = solutions[N_list[-1]]
    max_final = float(jnp.max(jnp.abs(h_final)))
    stable = max_final < 5.0

    passed = ok and stable
    print(f"\n  Stable: {'yes' if stable else 'NO'}")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_gravity_wave():
    """
    Gaussian perturbation radiates gravity waves across panels.
    Check stability and wave spreading.
    """
    print("\n" + "=" * 65)
    print("TEST 6: Gravity Wave on Cubed Sphere")
    print("=" * 65)

    N = 16; H0 = 1.0; g = 1.0
    sys_d = make_cubed_sphere_swe(N, H0, g)
    Wh = sys_d['Wh']; Jh = sys_d['Jh']
    grids = sys_d['grids']
    xi_v = grids['xi_v']
    dx = sys_d['dx']

    h = jnp.zeros((6, N + 1, N + 1))
    xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')
    h = h.at[0].set(0.1 * jnp.exp(-(xi1_2d**2 + xi2_2d**2) / (2 * 0.15**2)))
    v1 = jnp.zeros((6, N, N + 1))
    v2 = jnp.zeros((6, N + 1, N))

    mass0 = compute_mass(h, Wh, Jh)
    max0 = float(jnp.max(jnp.abs(h)))
    step_fn = make_rk4_step(sys_d['rhs'])

    c = np.sqrt(g * H0)
    CFL = 0.3
    dt = CFL * dx / c
    # Run long enough for waves to reach opposite panel
    T_end = np.pi / c  # half circumference
    nsteps = int(np.ceil(T_end / dt))
    dt = T_end / nsteps

    print(f"  N = {N}, CFL = {CFL}")
    print(f"  T = {T_end:.2f} (half circumference), steps = {nsteps}")

    for s in range(nsteps):
        h, v1, v2 = step_fn(h, v1, v2, dt)
        if (s + 1) % (nsteps // 4) == 0:
            max_h = float(jnp.max(jnp.abs(h)))
            panel_max = [float(jnp.max(jnp.abs(h[p]))) for p in range(6)]
            n_active = sum(1 for m in panel_max if m > 1e-6)
            print(f"  Step {s+1:5d}/{nsteps}: max|h| = {max_h:.4e}, "
                  f"active panels = {n_active}/6")

    mass_f = compute_mass(h, Wh, Jh)
    max_f = float(jnp.max(jnp.abs(h)))
    panel_max = [float(jnp.max(jnp.abs(h[p]))) for p in range(6)]
    n_active = sum(1 for m in panel_max if m > 1e-6)

    stable = max_f < 10 * max0
    mass_err = abs(mass_f - mass0)

    print(f"\n  Final max|h|: {max_f:.4e} (initial: {max0:.4e})")
    print(f"  Mass error: {mass_err:.2e}")
    print(f"  Stable: {'yes' if stable else 'NO'}")
    print(f"  Active panels: {n_active}/6")
    print(f"  Panel maxima: {['%.2e' % m for m in panel_max]}")

    passed = stable and n_active >= 4 and mass_err < 1e-8
    print(f"  {'âœ“ PASS' if passed else 'âœ— FAIL'}")
    return passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Step 4: Cubed Sphere SWE Tests')
    parser.add_argument('--gauss', type=int, choices=[1, 2], default=None,
                        help='Run Gaussian hill variant 1 (panel center) or 2 (vertex)')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests including Gaussian hill variants')
    args = parser.parse_args()

    print("=" * 65)
    print("  STEP 4: 6-Panel Cubed Sphere + Gaussian Hill Tests")
    print("  SBP 4/2, f=0")
    print("  Reference: Shashkin 2025, Sections 3-6")
    print("=" * 65)

    results = {}

    if args.gauss is not None:
        # Run only the specified Gaussian hill variant
        results[f'gauss_{args.gauss}'] = test_gaussian_hill(variant=args.gauss)
    elif args.all:
        # Run everything
        results['metrics']    = test_metrics()
        results['steady']     = test_steady_state()
        results['projection'] = test_projection()
        results['mass']       = test_mass_conservation()
        results['energy']     = test_energy()
        results['gravity']    = test_gravity_wave()
        results['gauss_1']    = test_gaussian_hill(variant=1)
        results['gauss_2']    = test_gaussian_hill(variant=2)
    else:
        # Default: run original tests
        results['metrics']    = test_metrics()
        results['steady']     = test_steady_state()
        results['projection'] = test_projection()
        results['mass']       = test_mass_conservation()
        results['energy']     = test_energy()
        results['gravity']    = test_gravity_wave()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    all_pass = True
    for name, passed in results.items():
        print(f"  {name:<15} {'PASS' if passed else 'FAIL'}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("  All tests passed.")
    else:
        print("  Some tests failed.")
    print("=" * 65)

