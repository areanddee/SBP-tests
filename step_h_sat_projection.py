"""
Step H: SBP 4/2 advection with SAT-Projection at panel interfaces.

KEY CHANGE from Steps A-F:
  Old: ghost cells + uniform [-1/16, 9/16, 9/16, -1/16] stencil everywhere
  New: SBP Pcv/Dvc matrix operators + projection at panel boundaries

The SBP 4/2 scheme has:
  - 4th-order interpolation/derivative in interior (rows 3..N-4)
  - 2nd-order boundary stencils (rows 0..2, N-2..N)
  - Non-uniform H-norm (quadrature weights) at boundaries
  - Projection: at each shared panel edge, face values are averaged
    from both sides → ensures continuity (Eq. 51 of Shashkin)

This eliminates ghost cells entirely. The boundary coupling comes
from the projection exchange of face values between panels.

Usage:
    python step_h_sat_projection.py 40                  # cosine bell
    python step_h_sat_projection.py 40 --gaussian       # Gaussian hill
"""
import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
# Also add project dir for sbp_staggered_1d
project_dir = '/mnt/project'
if os.path.isdir(project_dir):
    sys.path.insert(0, project_dir)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

print(f"Platform: {jax.devices()[0].platform}")

from grid import equiangular_to_cartesian
from halo_exchange import create_communication_schedule
from sbp_staggered_1d import sbp_42
from advection_sbp_cubesphere import (
    SBPPanelData,
    cosine_bell_ic,
    total_mass,
)


# ============================================================
# Gaussian hill IC (same as step_e)
# ============================================================

def gaussian_ic(panels, lon0=0.0, lat0=0.0, width=16.0, h0=1.0):
    center = jnp.array([
        jnp.cos(lat0) * jnp.cos(lon0),
        jnp.cos(lat0) * jnp.sin(lon0),
        jnp.sin(lat0)
    ])
    phi_list = []
    for p in panels:
        N = p.N
        dx = p.dx
        pi4 = jnp.pi / 4
        xi_c = jnp.linspace(-pi4 + dx/2, pi4 - dx/2, N)
        XI1, XI2 = jnp.meshgrid(xi_c, xi_c, indexing='ij')
        X, Y, Z = equiangular_to_cartesian(XI1, XI2, p.face_id)
        dot = X * center[0] + Y * center[1] + Z * center[2]
        dot = jnp.clip(dot, -1.0, 1.0)
        r = jnp.arccos(dot)
        phi = h0 * jnp.exp(-width * r**2)
        phi_list.append(phi)
    return jnp.stack(phi_list)


# ============================================================
# Projection exchange at panel boundaries
# ============================================================

def _get_bdy_face(phi_xface, phi_yface, panel, edge, N):
    """Extract N boundary face values for a given panel edge."""
    if edge == 'E':
        return phi_xface[panel, N, :]   # x-face at right boundary
    elif edge == 'W':
        return phi_xface[panel, 0, :]   # x-face at left boundary
    elif edge == 'N':
        return phi_yface[panel, :, N]   # y-face at top boundary
    elif edge == 'S':
        return phi_yface[panel, :, 0]   # y-face at bottom boundary


def _set_bdy_face(phi_xface, phi_yface, panel, edge, val, N):
    """Set N boundary face values for a given panel edge."""
    if edge == 'E':
        phi_xface = phi_xface.at[panel, N, :].set(val)
    elif edge == 'W':
        phi_xface = phi_xface.at[panel, 0, :].set(val)
    elif edge == 'N':
        phi_yface = phi_yface.at[panel, :, N].set(val)
    elif edge == 'S':
        phi_yface = phi_yface.at[panel, :, 0].set(val)
    return phi_xface, phi_yface


def make_projection_exchange(schedule, N):
    """
    Build JIT-compiled projection exchange function.

    At each shared panel edge:
      1. Extract boundary face values from both sides
      2. Apply orientation transform (R/TR = reverse)
      3. Average (Hv-weighted, but Hv[0]=Hv[N] for 4/2 → simple average)
      4. Write projected value back to both sides

    This ensures φ_face is continuous at all 12 panel interfaces.
    """
    def project_faces(phi_xface, phi_yface):
        """
        phi_xface: (6, N+1, N) — face values in x-direction
        phi_yface: (6, N, N+1) — face values in y-direction
        """
        for stage in schedule:
            for (face_a, edge_a), (face_b, edge_b), operation in stage:
                # Get boundary face values from each side
                val_a = _get_bdy_face(phi_xface, phi_yface, face_a, edge_a, N)
                val_b = _get_bdy_face(phi_xface, phi_yface, face_b, edge_b, N)

                # Apply orientation transform for A→B and B→A
                if operation in ('R', 'TR'):
                    val_a_for_b = val_a[::-1]
                    val_b_for_a = val_b[::-1]
                else:  # 'N' or 'T'
                    val_a_for_b = val_a
                    val_b_for_a = val_b

                # Project: weighted average (Eq. 51)
                # For symmetric Hv (Hv[0] = Hv[N]), this is simple averaging
                proj_a = 0.5 * (val_a + val_b_for_a)
                proj_b = 0.5 * (val_b + val_a_for_b)

                # Write back
                phi_xface, phi_yface = _set_bdy_face(
                    phi_xface, phi_yface, face_a, edge_a, proj_a, N)
                phi_xface, phi_yface = _set_bdy_face(
                    phi_xface, phi_yface, face_b, edge_b, proj_b, N)

        return phi_xface, phi_yface

    return jax.jit(project_faces)


# ============================================================
# RHS using SBP matrix operators + projection
# ============================================================

def make_rhs_fn(panels, N, dx):
    """
    Build JIT-compiled RHS using full SBP 4/2 operators.

    Instead of ghost cells + uniform stencil, uses:
      Pcv: (N+1, N) matrix — interpolate φ from centers to faces
      Dvc: (N, N+1) matrix — flux divergence from faces to centers

    Both have proper 2nd-order boundary stencils and H-norm weighting.
    The projection exchange ensures face-value continuity at panel interfaces.
    """
    # Build 1D SBP 4/2 operators
    ops = sbp_42(N, dx)
    Pcv = ops.Pcv   # (N+1, N) — interpolation center→face
    Dvc = ops.Dvc   # (N, N+1) — divergence face→center

    # Pre-stack panel geometry
    sqrt_G_v1_xface = jnp.stack([p.sqrt_G_v1_xface for p in panels])  # (6, N+1, N)
    sqrt_G_v2_yface = jnp.stack([p.sqrt_G_v2_yface for p in panels])  # (6, N, N+1)
    inv_sqrt_G = jnp.stack([p.inv_sqrt_G for p in panels])            # (6, N, N)

    # Build projection exchange
    schedule = create_communication_schedule()
    project_fn = make_projection_exchange(schedule, N)

    @jax.jit
    def rhs(phi):
        # phi: (6, N, N) — cell-center values

        # --- Interpolate to faces using Pcv ---
        # X-direction: Pcv @ phi[p, :, j] for each panel p and column j
        # phi_xface[p, f, j] = sum_i Pcv[f, i] * phi[p, i, j]
        phi_xface = jnp.einsum('fi,pij->pfj', Pcv, phi)  # (6, N+1, N)

        # Y-direction: Pcv @ phi[p, i, :] for each panel p and row i
        # phi_yface[p, i, f] = sum_j Pcv[f, j] * phi[p, i, j]
        phi_yface = jnp.einsum('fj,pij->pif', Pcv, phi)  # (6, N, N+1)

        # --- Project at panel boundaries ---
        phi_xface, phi_yface = project_fn(phi_xface, phi_yface)

        # --- Compute fluxes ---
        Fx = sqrt_G_v1_xface * phi_xface  # (6, N+1, N)
        Fy = sqrt_G_v2_yface * phi_yface  # (6, N, N+1)

        # --- Divergence using Dvc ---
        # X: dFx[p, i, j] = sum_f Dvc[i, f] * Fx[p, f, j]
        dFx = jnp.einsum('if,pfj->pij', Dvc, Fx)  # (6, N, N)

        # Y: dFy[p, i, j] = sum_f Dvc[j, f] * Fy[p, i, f]
        dFy = jnp.einsum('jf,pif->pij', Dvc, Fy)  # (6, N, N)

        return -inv_sqrt_G * (dFx + dFy)

    return rhs


def make_stepper(panels, N, dx, dt):
    """RK4 time stepper using SBP-SAT-Projection RHS."""
    rhs_fn = make_rhs_fn(panels, N, dx)

    @jax.jit
    def step(phi):
        k1 = rhs_fn(phi)
        k2 = rhs_fn(phi + 0.5 * dt * k1)
        k3 = rhs_fn(phi + 0.5 * dt * k2)
        k4 = rhs_fn(phi + dt * k3)
        return phi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return step


# ============================================================
# Error norms
# ============================================================

def compute_error_norms(phi, phi_exact, panels, dx):
    num1 = num2 = den1 = den2 = 0.0
    linf_num = linf_den = 0.0
    for f in range(6):
        err = phi[f] - phi_exact[f]
        sqG = panels[f].sqrt_G
        num1 += float(jnp.sum(jnp.abs(err) * sqG) * dx**2)
        den1 += float(jnp.sum(jnp.abs(phi_exact[f]) * sqG) * dx**2)
        num2 += float(jnp.sum(err**2 * sqG) * dx**2)
        den2 += float(jnp.sum(phi_exact[f]**2 * sqG) * dx**2)
        linf_num = max(linf_num, float(jnp.max(jnp.abs(err))))
        linf_den = max(linf_den, float(jnp.max(jnp.abs(phi_exact[f]))))
    return num1/(den1+1e-30), np.sqrt(num2/(den2+1e-30)), linf_num/(linf_den+1e-30)


# ============================================================
# Main
# ============================================================

def run(N, use_gaussian=False):
    CFL = 0.5
    ic_name = "Gaussian" if use_gaussian else "Cosine bell"

    print("=" * 70)
    print(f"  SBP 4/2 + SAT-Projection")
    print(f"  N = {N}, CFL = {CFL}, IC = {ic_name}, full rotation")
    print("=" * 70)

    dx = float(jnp.pi / (2 * N))
    panels = [SBPPanelData(N, f, u0=1.0, order=4) for f in range(6)]

    if use_gaussian:
        phi0 = gaussian_ic(panels, lon0=0.0, lat0=0.0, width=16.0, h0=1.0)
    else:
        phi0 = cosine_bell_ic(panels, lon0=0.0, lat0=0.0, R=1.0/3, h0=1000.0)

    mass0 = total_mass(phi0, panels, dx)
    max0 = float(jnp.max(phi0))

    max_speed = 1.5
    T = float(2 * jnp.pi)
    dt = CFL * dx / max_speed
    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps
    actual_cfl = max_speed * dt / dx

    print(f"  dx = {dx:.4f}, dt = {dt:.4e}")
    print(f"  steps = {nsteps}, actual CFL = {actual_cfl:.3f}")
    print(f"  Initial: max = {max0:.6f}, mass = {mass0:.6f}")

    step_fn = make_stepper(panels, N, dx, dt)

    print("  JIT warmup...", end=" ", flush=True)
    t0 = time.perf_counter()
    _ = step_fn(phi0)
    jax.block_until_ready(_)
    print(f"({time.perf_counter()-t0:.1f}s)")

    phi = phi0.copy()

    t_start = time.perf_counter()
    report_interval = max(1, nsteps // 8)
    for s in range(nsteps):
        phi = step_fn(phi)
        if (s + 1) % report_interval == 0:
            jax.block_until_ready(phi)
            mass = total_mass(phi, panels, dx)
            mass_err = abs(mass - mass0) / abs(mass0)
            mx = float(jnp.max(phi))
            mn = float(jnp.min(phi))
            print(f"  Step {s+1:6d}/{nsteps}: mass_err={mass_err:.2e}, "
                  f"max={mx:.6f}, min={mn:.6e}")

    jax.block_until_ready(phi)
    elapsed = time.perf_counter() - t_start

    mass_final = total_mass(phi, panels, dx)
    mass_err = abs(mass_final - mass0) / abs(mass0)
    max_final = float(jnp.max(phi))
    min_final = float(jnp.min(phi))
    l1, l2, linf = compute_error_norms(phi, phi0, panels, dx)

    print()
    print("-" * 70)
    print(f"  N = {N}  |  {ic_name}  |  SBP 4/2 + SAT-Projection")
    print("-" * 70)
    print(f"  l1   = {l1:.6e}")
    print(f"  l2   = {l2:.6e}")
    print(f"  linf = {linf:.6e}")
    print(f"  Mass error:      {mass_err:.2e}")
    print(f"  Max final:       {max_final:.6f}  (initial: {max0:.6f})")
    print(f"  Min final:       {min_final:.6e}")
    print(f"  Amplitude loss:  {(max0 - max_final)/max0:.2%}")
    print(f"  Wall time:       {elapsed:.1f}s  ({nsteps/elapsed:.1f} steps/s)")
    print("=" * 70)

    return l1, l2, linf


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    N = int(args[0]) if args else 40
    use_gaussian = '--gaussian' in flags

    run(N, use_gaussian=use_gaussian)
