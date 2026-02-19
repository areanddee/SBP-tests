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
from sat_operators import build_cartesian_sat_fn

# ============================================================
# Connectivity (single source of truth: connectivity.py)
# ============================================================
from connectivity import EDGES

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

    def project_h(h):
        """Project h at all shared interfaces."""
        # 1. Edge averaging (12 edges)
        h_orig = h 
        for pa, ea, pb, eb, op in EDGES:
            rev = _reverses(op)
            bnd_a = _get_h_boundary(h_orig[pa], ea, N)
            bnd_b = _get_h_boundary(h_orig[pb], eb, N)
            if rev:
                bnd_b = bnd_b[::-1]
            avg = 0.5 * (bnd_a + bnd_b)
            h = _set_h_boundary(h, pa, ea, avg, N)
            avg_b = avg[::-1] if rev else avg
            h = _set_h_boundary(h, pb, eb, avg_b, N)

        # 2. Corner averaging (8 corners, 3 panels each)
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
    add_sat = build_cartesian_sat_fn(N, ops, compute_metric)

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
        div = add_sat(div, u1_all, u2_all, v1, v2)

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

    @partial(jax.jit, static_argnums=(4,))
    def run_n_steps(h, v1, v2, dt, nsteps):
        """Run nsteps RK4 steps entirely on-device via lax.fori_loop."""
        def body(i, state):
            h, v1, v2 = state
            return step(h, v1, v2, dt)
        return jax.lax.fori_loop(0, nsteps, body, (h, v1, v2))

    return step, run_n_steps


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
    step_fn, _ = make_rk4_step(sys_d['rhs'])

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
    step_fn, _ = make_rk4_step(sys_d['rhs'])

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
    step_fn, _ = make_rk4_step(sys_d['rhs'])

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
# Test 7: Richardson spatial convergence
# ============================================================

def test_convergence(variant=1):
    """
    Richardson self-convergence test.

    Runs at N = 24, 48, 96, 192, 384 Compares each coarser solution against
    the finest (N=24) subsampled to the coarse grid.

    variant 1: Gaussian centered at panel center (0,0) on panel 0
    variant 2: Gaussian centered at cube vertex (pi/4,pi/4) on panel 0

    For Ch42 (Shashkin Table 2):
      Variant 1: l2 rate ~3, l_inf rate ~3
      Variant 2: l2 rate drops somewhat (interpolation error dominates)
    """
    import time as _time

    label = "panel center (0,0)" if variant == 1 else "cube vertex (pi/4,pi/4)"
    print("\n" + "=" * 65)
    print(f"TEST 7: Spatial Convergence -- Gaussian variant {variant}")
    print(f"        Center at {label}")
    print("=" * 65)

    Ns = [24, 48, 96, 192, 384]
    H0 = 1.0; g = 1.0; c = np.sqrt(g * H0)
    CFL = 0.05
    T_end = 25.   # fixed physical time (days)

    print(f"  T = {T_end}, CFL = {CFL}, H0 = {H0}, g = {g}")
    print(f"  N values: {Ns}")

    # --- Gaussian IC using great-circle distance (same on all panels) ---
    def make_gaussian_ic(N_loc, grids_loc):
        xi_v_loc = grids_loc['xi_v']
        xi1_2d, xi2_2d = jnp.meshgrid(xi_v_loc, xi_v_loc, indexing='ij')
        h_ic = jnp.zeros((6, N_loc + 1, N_loc + 1))
        v1_ic = jnp.zeros((6, N_loc, N_loc + 1))
        v2_ic = jnp.zeros((6, N_loc + 1, N_loc))
        sigma = 0.3
        amp = 0.01
        if variant == 1:
            X0, Y0, Z0 = 0.0, 0.0, 1.0
        else:
            X0, Y0, Z0 = equiangular_to_cartesian(
                jnp.array(jnp.pi / 4), jnp.array(jnp.pi / 4), 0)
            X0, Y0, Z0 = float(X0), float(Y0), float(Z0)
        for p in range(6):
            X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
            cos_d = jnp.clip(X * X0 + Y * Y0 + Z * Z0, -1.0, 1.0)
            d = jnp.arccos(cos_d)
            h_ic = h_ic.at[p].set(amp * jnp.exp(-d**2 / (2 * sigma**2)))
        return h_ic, v1_ic, v2_ic

    # --- Run all resolutions ---
    run_results = {}
    for N in Ns:
        sys_d = make_cubed_sphere_swe(N, H0, g)
        grids = sys_d['grids']
        dx = sys_d['dx']
        dt = 1./(N*3)       # per Table 1, section 6.1, Shashkin, et al. 2025 
        #dt = CFL * dx / c
        CFL = c*dt/dx
        nsteps = int(np.ceil(T_end / dt))
        dt = T_end / nsteps   # hit T_end exactly

        h, v1, v2 = make_gaussian_ic(N, grids)
        mass0 = compute_mass(h, sys_d['Wh'], sys_d['Jh'])
        step_fn, run_n_steps = make_rk4_step(sys_d['rhs'])

        print(f"\n  N = {N:3d}: dx = {dx:.4e}, dt = {dt:.4e}, CFL = {CFL:.4e}, steps = {nsteps}",
              end='', flush=True)

        t0 = _time.time()
        h, v1, v2 = step_fn(h, v1, v2, dt)  # JIT warmup (1 step)
        jax.block_until_ready(h)
        jit_t = _time.time() - t0
        print(f"  (JIT {jit_t:.1f}s)", end='', flush=True)

        t0 = _time.time()
        h, v1, v2 = run_n_steps(h, v1, v2, dt, nsteps - 1)  # remaining steps on-device
        jax.block_until_ready(h)
        run_t = _time.time() - t0
        print(f"  (run {run_t:.1f}s)", flush=True)

        mass_f = compute_mass(h, sys_d['Wh'], sys_d['Jh'])
        mass_err = abs(mass_f - mass0)
        print(f"         mass_err = {mass_err:.2e}")

        run_results[N] = {
            'h': h, 'dx': dx,
            'Jh': sys_d['Jh'], 'Wh': sys_d['Wh'],
        }

    # --- Compute errors relative to finest ---
    N_fine = max(Ns)
    h_fine = run_results[N_fine]['h']

    print(f"\n  Errors vs N = {N_fine} reference:")
    print(f"  {'N':>5} {'dx':>12} {'L2 err':>12} {'L2 rate':>8} "
          f"{'Linf err':>12} {'Linf rate':>10}")
    print(f"  {'-'*65}")

    errors = []
    for N in Ns[:-1]:
        stride = N_fine // N
        assert stride * N == N_fine, f"N={N} doesn't divide N_fine={N_fine}"

        h_ref = h_fine[:, ::stride, ::stride]
        diff = run_results[N]['h'] - h_ref
        Jh_c = run_results[N]['Jh']
        Wh_c = run_results[N]['Wh']

        l2 = float(jnp.sqrt(jnp.sum(diff**2 * Jh_c[None] * Wh_c[None])))
        linf = float(jnp.max(jnp.abs(diff)))

        errors.append({'N': N, 'dx': run_results[N]['dx'], 'l2': l2, 'linf': linf})

        if len(errors) >= 2:
            l2_rate = (np.log(errors[-2]['l2'] / errors[-1]['l2']) /
                       np.log(errors[-2]['dx'] / errors[-1]['dx']))
            linf_rate = (np.log(errors[-2]['linf'] / errors[-1]['linf']) /
                         np.log(errors[-2]['dx'] / errors[-1]['dx']))
            l2_str = f"{l2_rate:8.2f}"
            linf_str = f"{linf_rate:10.2f}"
        else:
            l2_str = "     ---"
            linf_str = "       ---"

        print(f"  {N:5d} {run_results[N]['dx']:12.4e} {l2:12.4e} {l2_str} "
              f"{linf:12.4e} {linf_str}")

    # --- Summary ---
    if len(errors) >= 2:
        final_l2 = (np.log(errors[-2]['l2'] / errors[-1]['l2']) /
                    np.log(errors[-2]['dx'] / errors[-1]['dx']))
        final_linf = (np.log(errors[-2]['linf'] / errors[-1]['linf']) /
                      np.log(errors[-2]['dx'] / errors[-1]['dx']))
    else:
        final_l2 = 0.0
        final_linf = 0.0

    print(f"\n  Final L2 convergence rate:   {final_l2:.2f}")
    print(f"  Final Linf convergence rate: {final_linf:.2f}")

    # Ch42 expects ~3 for variant 1, somewhat lower for variant 2
    threshold = 2.5 if variant == 1 else 1.8
    passed = final_l2 > threshold
    print(f"  Expected L2 rate > {threshold:.1f}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gauss', type=int, choices=[1, 2], default=None,
                        help='Gaussian variant for convergence test: '
                             '1=panel center, 2=cube vertex')
    args = parser.parse_args()

    print("=" * 65)
    print("  STEP 5: 6-Panel Cubed Sphere -- Metrics + SAT-Projection")
    print("  SBP 4/2, f=0, Cartesian-averaged SAT")
    print("  Reference: Shashkin 2025, Sections 3-5")
    print("=" * 65)

    results = {}
    results['metrics']    = test_metrics()
    results['steady']     = test_steady_state()
    results['projection'] = test_projection()
    results['mass']       = test_mass_conservation()
    results['energy']     = test_energy()
    results['gravity']    = test_gravity_wave()

    # Convergence test(s)
    if args.gauss is not None:
        results[f'converge_v{args.gauss}'] = test_convergence(variant=args.gauss)
    else:
        results['converge_v1'] = test_convergence(variant=1)
        results['converge_v2'] = test_convergence(variant=2)

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<15} {status}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("  All tests passed.")
    else:
        print("  Some tests failed. Fix before proceeding.")
    print("=" * 65)
