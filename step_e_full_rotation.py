"""
Step E v3: Full rotation with Gaussian or cosine bell IC.

Usage:
    python step_e_full_rotation.py 40                  # cosine bell (default)
    python step_e_full_rotation.py 40 --gaussian       # Gaussian hill
    python step_e_full_rotation.py 80 --gaussian
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

print(f"Platform: {jax.devices()[0].platform}")

from grid import equiangular_to_cartesian
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
# Gaussian hill IC (C∞ smooth — matches Shashkin Eq. 84 form)
# ============================================================

def gaussian_ic(panels, lon0=0.0, lat0=0.0, width=16.0, h0=1.0):
    """
    Gaussian hill: h = h0 * exp(-width * r^2)
    
    where r is great-circle distance from (lon0, lat0).
    Shashkin uses width=16/a^2 with a=1 → width=16.
    We use h0=1.0 (unit amplitude) for clean error analysis.
    """
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
        r = jnp.arccos(dot)  # great-circle distance in radians
        
        phi = h0 * jnp.exp(-width * r**2)
        phi_list.append(phi)
    
    return jnp.stack(phi_list)


# ============================================================
# VECTORIZED RHS
# ============================================================

def make_rhs_fn(panels, N, dx):
    sqrt_G_v1_xface = jnp.stack([p.sqrt_G_v1_xface for p in panels])
    sqrt_G_v2_yface = jnp.stack([p.sqrt_G_v2_yface for p in panels])
    inv_sqrt_G = jnp.stack([p.inv_sqrt_G for p in panels])

    @jax.jit
    def rhs(phi_ghost_2deep):
        phi_xext = phi_ghost_2deep[:, :, 2:N+2]
        phi_xface = (-1.0/16 * phi_xext[:, 0:N+1, :] +
                      9.0/16 * phi_xext[:, 1:N+2, :] +
                      9.0/16 * phi_xext[:, 2:N+3, :] +
                     -1.0/16 * phi_xext[:, 3:N+4, :])
        Fx = sqrt_G_v1_xface * phi_xface
        dFx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx

        phi_yext = phi_ghost_2deep[:, 2:N+2, :]
        phi_yface = (-1.0/16 * phi_yext[:, :, 0:N+1] +
                      9.0/16 * phi_yext[:, :, 1:N+2] +
                      9.0/16 * phi_yext[:, :, 2:N+3] +
                     -1.0/16 * phi_yext[:, :, 3:N+4])
        Fy = sqrt_G_v2_yface * phi_yface
        dFy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dx

        return -inv_sqrt_G * (dFx + dFy)

    return rhs


def make_stepper(panels, N, dx, dt, halo_fn):
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
    order = 4
    ic_name = "Gaussian" if use_gaussian else "Cosine bell"

    print("=" * 65)
    print(f"  N = {N}, CFL = {CFL}, order = {order}, IC = {ic_name}")
    print(f"  Full rotation (T = 2π)")
    print("=" * 65)

    dx = float(jnp.pi / (2 * N))
    panels = [SBPPanelData(N, f, u0=1.0, order=order) for f in range(6)]

    schedule = create_communication_schedule()
    halo_fn = make_halo_exchange_2deep(schedule, N)

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

    step_fn = make_stepper(panels, N, dx, dt, halo_fn)

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
    print("-" * 65)
    print(f"  N = {N}  |  {ic_name}  |  Full rotation results")
    print("-" * 65)
    print(f"  l1   = {l1:.6e}")
    print(f"  l2   = {l2:.6e}")
    print(f"  linf = {linf:.6e}")
    print(f"  Mass error:      {mass_err:.2e}")
    print(f"  Max final:       {max_final:.6f}  (initial: {max0:.6f})")
    print(f"  Min final:       {min_final:.6e}")
    print(f"  Amplitude loss:  {(max0 - max_final)/max0:.2%}")
    print(f"  Wall time:       {elapsed:.1f}s  ({nsteps/elapsed:.1f} steps/s)")
    print("=" * 65)

    return l1, l2, linf


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    N = int(args[0]) if args else 40
    use_gaussian = '--gaussian' in flags

    run(N, use_gaussian=use_gaussian)
