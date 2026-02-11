"""
Step E v2: Full rotation with VECTORIZED RHS (no Python loops in JIT).

Usage:
    python step_e_full_rotation.py          # default N=40
    python step_e_full_rotation.py 40
    python step_e_full_rotation.py 80
    python step_e_full_rotation.py 160
"""
import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

print(jax.devices()[0].platform)

from halo_exchange import create_communication_schedule
from advection_sbp_cubesphere import (
    SBPPanelData,
    cosine_bell_ic,
    total_mass,
)

# ============================================================
# 2-deep halo exchange (verified in Steps B, C)
# ============================================================

def extend_to_include_ghosts_2deep(field_interior, N):
    return jnp.pad(field_interior, ((0, 0), (2, 2), (2, 2)), mode='edge')

def extract_edge_2deep(field_face, edge, N):
    interior = field_face[2:N+2, 2:N+2]
    if edge == 'E':
        return interior[-1, :], interior[-2, :]
    elif edge == 'W':
        return interior[0, :], interior[1, :]
    elif edge == 'N':
        return interior[:, -1], interior[:, -2]
    elif edge == 'S':
        return interior[:, 0], interior[:, 1]

def insert_ghost_2deep(field_face, edge, ghost1, ghost2, N):
    if edge == 'E':
        field_face = field_face.at[N+2, 2:N+2].set(ghost1)
        field_face = field_face.at[N+3, 2:N+2].set(ghost2)
    elif edge == 'W':
        field_face = field_face.at[1, 2:N+2].set(ghost1)
        field_face = field_face.at[0, 2:N+2].set(ghost2)
    elif edge == 'N':
        field_face = field_face.at[2:N+2, N+2].set(ghost1)
        field_face = field_face.at[2:N+2, N+3].set(ghost2)
    elif edge == 'S':
        field_face = field_face.at[2:N+2, 1].set(ghost1)
        field_face = field_face.at[2:N+2, 0].set(ghost2)
    return field_face

def apply_operation_2deep(data1, data2, operation):
    if operation == 'N':
        return data1, data2
    elif operation == 'R':
        return data1[::-1], data2[::-1]
    elif operation == 'T':
        return data1, data2
    elif operation == 'TR':
        return data1[::-1], data2[::-1]

def exchange_pair_2deep(field, face_a, edge_a, face_b, edge_b, operation, N):
    a_row1, a_row2 = extract_edge_2deep(field[face_a], edge_a, N)
    b_row1, b_row2 = extract_edge_2deep(field[face_b], edge_b, N)
    a_to_b_1, a_to_b_2 = apply_operation_2deep(a_row1, a_row2, operation)
    b_to_a_1, b_to_a_2 = apply_operation_2deep(b_row1, b_row2, operation)
    field = field.at[face_b].set(
        insert_ghost_2deep(field[face_b], edge_b, a_to_b_1, a_to_b_2, N)
    )
    field = field.at[face_a].set(
        insert_ghost_2deep(field[face_a], edge_a, b_to_a_1, b_to_a_2, N)
    )
    return field

def make_halo_exchange_2deep(schedule, N):
    def exchange_all(field_ghosts):
        field = field_ghosts
        for stage in schedule:
            for (face_a, edge_a), (face_b, edge_b), operation in stage:
                field = exchange_pair_2deep(
                    field, face_a, edge_a, face_b, edge_b, operation, N
                )
        return field
    return jax.jit(exchange_all)


# ============================================================
# VECTORIZED RHS — no Python for-loops inside JIT
# ============================================================

def make_rhs_fn(panels, N, dx):
    """
    Build a JIT-compiled RHS function with all panel data baked in.
    No Python loops at execution time.
    """
    # Stack all panel geometry into (6, ...) arrays
    sqrt_G_v1_xface = jnp.stack([p.sqrt_G_v1_xface for p in panels])  # (6, N+1, N)
    sqrt_G_v2_yface = jnp.stack([p.sqrt_G_v2_yface for p in panels])  # (6, N, N+1)
    inv_sqrt_G = jnp.stack([p.inv_sqrt_G for p in panels])            # (6, N, N)

    @jax.jit
    def rhs(phi_ghost_2deep):
        """
        phi_ghost_2deep: (6, N+4, N+4)
        returns: (6, N, N)
        """
        # X-direction interpolation: all 6 panels at once
        phi_xext = phi_ghost_2deep[:, :, 2:N+2]  # (6, N+4, N)
        phi_xface = (-1.0/16 * phi_xext[:, 0:N+1, :] +
                      9.0/16 * phi_xext[:, 1:N+2, :] +
                      9.0/16 * phi_xext[:, 2:N+3, :] +
                     -1.0/16 * phi_xext[:, 3:N+4, :])  # (6, N+1, N)

        Fx = sqrt_G_v1_xface * phi_xface
        dFx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx  # (6, N, N)

        # Y-direction interpolation
        phi_yext = phi_ghost_2deep[:, 2:N+2, :]  # (6, N, N+4)
        phi_yface = (-1.0/16 * phi_yext[:, :, 0:N+1] +
                      9.0/16 * phi_yext[:, :, 1:N+2] +
                      9.0/16 * phi_yext[:, :, 2:N+3] +
                     -1.0/16 * phi_yext[:, :, 3:N+4])  # (6, N, N+1)

        Fy = sqrt_G_v2_yface * phi_yface
        dFy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dx  # (6, N, N)

        return -inv_sqrt_G * (dFx + dFy)

    return rhs


def make_stepper(panels, N, dx, dt, halo_fn):
    """Build a JIT-compiled RK4 stepper."""
    rhs_fn = make_rhs_fn(panels, N, dx)

    @jax.jit
    def step(phi):
        def rhs(p):
            return rhs_fn(halo_fn(extend_to_include_ghosts_2deep(p, N)))
        k1 = rhs(phi)
        k2 = rhs(phi + 0.5 * dt * k1)
        k3 = rhs(phi + 0.5 * dt * k2)
        k4 = rhs(phi + dt * k3)
        return phi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return step


# ============================================================
# Error norms
# ============================================================

def compute_error_norms(phi, phi_exact, panels, dx):
    """l1, l2, linf normalized error norms (area-weighted)."""
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

def run(N):
    CFL = 0.5
    order = 4

    print("=" * 65)
    print(f"  N = {N}, CFL = {CFL}, order = {order}, full rotation")
    print("=" * 65)

    dx = float(jnp.pi / (2 * N))
    panels = [SBPPanelData(N, f, u0=1.0, order=order) for f in range(6)]

    schedule = create_communication_schedule()
    halo_fn = make_halo_exchange_2deep(schedule, N)

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
    print(f"  Initial: max = {max0:.2f}, mass = {mass0:.4f}")

    # Build JIT-compiled stepper
    step_fn = make_stepper(panels, N, dx, dt, halo_fn)

    # JIT warmup
    print("  JIT warmup...", end=" ", flush=True)
    t0 = time.perf_counter()
    _ = step_fn(phi0)
    jax.block_until_ready(_)
    print(f"({time.perf_counter()-t0:.1f}s)")

    phi = phi0.copy()

    # Run
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
                  f"max={mx:.1f}, min={mn:.2f}")

    jax.block_until_ready(phi)
    elapsed = time.perf_counter() - t_start

    mass_final = total_mass(phi, panels, dx)
    mass_err = abs(mass_final - mass0) / abs(mass0)
    max_final = float(jnp.max(phi))
    min_final = float(jnp.min(phi))
    l1, l2, linf = compute_error_norms(phi, phi0, panels, dx)

    print()
    print("-" * 65)
    print(f"  N = {N}  |  Full rotation results")
    print("-" * 65)
    print(f"  l1   = {l1:.6e}")
    print(f"  l2   = {l2:.6e}")
    print(f"  linf = {linf:.6e}")
    print(f"  Mass error:      {mass_err:.2e}")
    print(f"  Max final:       {max_final:.2f}  (initial: {max0:.2f})")
    print(f"  Min final:       {min_final:.4f}")
    print(f"  Amplitude loss:  {(max0 - max_final)/max0:.1%}")
    print(f"  Wall time:       {elapsed:.1f}s  ({nsteps/elapsed:.1f} steps/s)")
    print("=" * 65)

    return l1, l2, linf


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    run(N)
