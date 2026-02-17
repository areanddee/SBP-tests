"""
diagnostics.py — Mass and Energy Computation
==============================================

Global integral diagnostics for the cubed-sphere SWE.

Mass should be conserved to machine precision with the conservative
Cartesian SAT.  Energy conservation is spatial-exact; temporal energy
error scales as dt^5 with RK4.
"""

import jax
import jax.numpy as jnp

from .mesh import compute_contravariant


def compute_mass(h, Wh, Jh):
    """
    Global mass integral over all 6 panels.

    mass = sum_p ∫ h · J dA = sum_p sum_{i,j} h[p,i,j] · Jh[i,j] · Wh[i,j]

    Args:
        h:  (6, N+1, N+1)  height perturbation
        Wh: (N+1, N+1)     quadrature weights at h-points
        Jh: (N+1, N+1)     Jacobian at h-points

    Returns:
        float: total mass
    """
    return float(jnp.sum(h * Jh[None, :, :] * Wh[None, :, :]))


def compute_energy(h, v1, v2, Wh, W1, W2, Jh, J1, J2, g, H0,
                   metrics, Pvc, Pcv):
    """
    Total energy: E = PE + KE

    PE = (g/2) ∫ h² J dA
    KE = (H0/2) ∫ (v_1·v¹ + v_2·v²) J dA

    where v^i are contravariant velocities computed from covariant v_i.

    Args:
        h:  (6, N+1, N+1) height
        v1: (6, N, N+1) covariant velocity
        v2: (6, N+1, N) covariant velocity
        Wh, W1, W2: quadrature weights at h, v1, v2 points
        Jh, J1, J2: Jacobian at h, v1, v2 points
        g: gravity
        H0: mean height
        metrics: dict from make_all_metrics
        Pvc, Pcv: SBP interpolation matrices

    Returns:
        float: total energy
    """
    PE = 0.5 * g * float(jnp.sum(h**2 * Jh[None, :, :] * Wh[None, :, :]))

    def _contra_single(v1_p, v2_p):
        return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
    v1c, v2c = jax.vmap(_contra_single)(v1, v2)

    KE = 0.5 * H0 * (
        float(jnp.sum(v1 * J1 * v1c * W1[None, :, :])) +
        float(jnp.sum(v2 * J2 * v2c * W2[None, :, :]))
    )

    return PE + KE
