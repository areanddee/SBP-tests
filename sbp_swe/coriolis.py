"""
coriolis.py — Energy-Conserving Coriolis Operator (Eq. 63)
============================================================

Implements the Coriolis operator for the linearized SWE on the cubed sphere.
The key feature is the V operator (Eq. 60) which projects vector fields
through Cartesian space to enforce edge-continuity, enabling energy neutrality.

The operator chain is:
    F = P_hv  V  C  P_vh  J_v

where:
    P_vh:  interpolate covariant velocity from v-points to h-points
    C:     Coriolis rotation [[0, f], [-f, 0]] at h-points
    V:     edge-continuity operator Y [Ah; Ah; Ah] X
           X = covariant → Cartesian, Ah = scalar projection, Y = Cartesian → covariant
    P_hv:  interpolate from h-points back to v-points
    J_v:   Jacobian at v-points (premultiplied)

Energy neutrality: v^T Hv Jv Q (Fv) = 0  (antisymmetric by construction)

Reference: Shashkin 2025, Sections 4.5, 5.4, Eq. 57-63
"""

import jax
import jax.numpy as jnp

from .velocity import get_covariant_basis


# ============================================================
# Covariant basis vectors on 2D grids
# ============================================================

def compute_basis_on_grid(xi1_2d, xi2_2d, face_id):
    """
    Compute covariant basis vectors a1, a2 at all grid points on a panel.

    Args:
        xi1_2d, xi2_2d: 2D coordinate arrays (any shape)
        face_id: panel index 0-5

    Returns:
        dict with 'a1x','a1y','a1z','a2x','a2y','a2z' arrays
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

    X: covariant → Cartesian (raise index, then expand in covariant basis)
    Ah: project each Cartesian component (scalar averaging at edges/corners)
    Y: Cartesian → covariant (dot product with covariant basis, no metric)

    Y·X = I because a_i · a^j = delta_i^j, so V is idempotent on
    continuous fields and makes discontinuous fields continuous.

    Args:
        w1_h, w2_h: (6, N+1, N+1) covariant components at h-points
        bases_h: list of 6 dicts with basis vectors at h-points
        project_h: scalar projection function

    Returns:
        w1_proj, w2_proj: (6, N+1, N+1) edge-continuous covariant field
    """
    npanels = w1_h.shape[0]

    # Step 1: X — covariant → Cartesian
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

    # Step 3: Y — Cartesian → covariant (w_i = W · a_i, no metric needed)
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
    Compute Coriolis tendency on covariant velocity.

    With V (use_V=True, Eq. 62):
        F = P_hv  V  C  P_vh  J_v
        Energy-neutral and edge-continuous.

    Without V (use_V=False, Eq. 57):
        F = P_hv  C  P_vh  J_v
        Energy-neutral but edge-discontinuous.

    Args:
        v1: (6, N, N+1)   covariant velocity v_1
        v2: (6, N+1, N)   covariant velocity v_2
        f_h: (N+1, N+1)   Coriolis parameter at h-points
        Jh: (N+1, N+1)    Jacobian at h-points
        J1: (N, N+1)      Jacobian at v1-points
        J2: (N+1, N)      Jacobian at v2-points
        Pcv: (N+1, N)     center-to-vertex interpolation
        Pvc: (N, N+1)     vertex-to-center interpolation
        bases_h: list of 6 dicts with basis vectors at h-points
        project_h: scalar h-projection function
        use_V: whether to apply V operator for edge continuity

    Returns:
        dv1, dv2: Coriolis tendency at v1, v2 points
    """
    # Step 1: multiply by J at v-points
    w1 = v1 * J1[None, :, :]
    w2 = v2 * J2[None, :, :]

    # Step 2: interpolate to h-points
    w1_h = jnp.einsum('ij,pjk->pik', Pcv, w1)    # (6, N+1, N+1)
    w2_h = jnp.einsum('pij,kj->pik', w2, Pcv)     # (6, N+1, N+1)

    # Step 3: Coriolis rotation at h-points
    c1_h = f_h[None, :, :] * w2_h       # +f * w2
    c2_h = -f_h[None, :, :] * w1_h      # -f * w1

    # Step 4: V operator (edge-continuity via Cartesian)
    if use_V:
        c1_h, c2_h = apply_V(c1_h, c2_h, bases_h, project_h)

    # Step 5: interpolate back to v-points
    dv1 = jnp.einsum('ij,pjk->pik', Pvc, c1_h)    # (6, N, N+1)
    dv2 = jnp.einsum('pij,kj->pik', c2_h, Pvc)     # (6, N+1, N)

    return dv1, dv2
