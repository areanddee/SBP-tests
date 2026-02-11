"""
Step G: Test SAT (upwind correction at panel boundaries).

Compares three variants at N=40, full rotation:
  1. No SAT (current — centered everywhere)
  2. Full upwind at boundary faces only (σ=1)
  3. Half upwind at boundary faces (σ=0.5)

The SAT modification: at the 2 boundary faces per direction (faces 0 and N),
add Lax-Friedrichs dissipation to the flux:

  F = (√G v) * φ_centered - (σ/2) * |√G v| * (φ_R - φ_L)

Interior faces (1..N-1) are unchanged.

Usage:
    python step_g_sat_test.py          # default N=40
    python step_g_sat_test.py 80
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

from halo_exchange import create_communication_schedule
from advection_sbp_cubesphere import (
    SBPPanelData,
    cosine_bell_ic,
    total_mass,
)

# ============================================================
# 2-deep halo exchange (verified)
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
# RHS with SAT parameter
# ============================================================

def make_rhs_fn(panels, N, dx, sigma=0.0):
    """
    Build JIT-compiled RHS with optional SAT dissipation.
    
    sigma = 0.0: pure centered (no SAT)
    sigma = 1.0: full Lax-Friedrichs at boundary faces
    sigma = 0.5: half dissipation at boundary faces
    
    SAT is applied ONLY at face 0 and face N (panel boundaries).
    Interior faces (1..N-1) use pure centered interpolation.
    """
    sqrt_G_v1_xface = jnp.stack([p.sqrt_G_v1_xface for p in panels])  # (6, N+1, N)
    sqrt_G_v2_yface = jnp.stack([p.sqrt_G_v2_yface for p in panels])  # (6, N, N+1)
    inv_sqrt_G = jnp.stack([p.inv_sqrt_G for p in panels])            # (6, N, N)

    @jax.jit
    def rhs(phi_ghost_2deep):
        # --- X-direction ---
        phi_xext = phi_ghost_2deep[:, :, 2:N+2]  # (6, N+4, N)
        
        # Centered 4-point interpolation at all faces
        phi_xface = (-1.0/16 * phi_xext[:, 0:N+1, :] +
                      9.0/16 * phi_xext[:, 1:N+2, :] +
                      9.0/16 * phi_xext[:, 2:N+3, :] +
                     -1.0/16 * phi_xext[:, 3:N+4, :])  # (6, N+1, N)
        
        # Centered flux
        Fx = sqrt_G_v1_xface * phi_xface  # (6, N+1, N)
        
        # SAT: add Lax-Friedrichs dissipation at boundary faces only
        # Face 0: between cells at indices 1 (ghost) and 2 (interior)
        # φ_L = phi_xext[:, 1, :], φ_R = phi_xext[:, 2, :]
        # Face N: between cells at indices N+1 (interior) and N+2 (ghost)
        # φ_L = phi_xext[:, N+1, :], φ_R = phi_xext[:, N+2, :]
        if sigma > 0:
            # Face 0
            phi_L_0 = phi_xext[:, 1, :]   # (6, N) - cell left of face 0
            phi_R_0 = phi_xext[:, 2, :]   # (6, N) - cell right of face 0
            alpha_0 = jnp.abs(sqrt_G_v1_xface[:, 0, :])  # (6, N)
            Fx = Fx.at[:, 0, :].add(-sigma/2 * alpha_0 * (phi_R_0 - phi_L_0))
            
            # Face N
            phi_L_N = phi_xext[:, N+1, :]  # (6, N)
            phi_R_N = phi_xext[:, N+2, :]  # (6, N)
            alpha_N = jnp.abs(sqrt_G_v1_xface[:, N, :])
            Fx = Fx.at[:, N, :].add(-sigma/2 * alpha_N * (phi_R_N - phi_L_N))
        
        dFx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx  # (6, N, N)
        
        # --- Y-direction ---
        phi_yext = phi_ghost_2deep[:, 2:N+2, :]  # (6, N, N+4)
        
        phi_yface = (-1.0/16 * phi_yext[:, :, 0:N+1] +
                      9.0/16 * phi_yext[:, :, 1:N+2] +
                      9.0/16 * phi_yext[:, :, 2:N+3] +
                     -1.0/16 * phi_yext[:, :, 3:N+4])  # (6, N, N+1)
        
        Fy = sqrt_G_v2_yface * phi_yface
        
        if sigma > 0:
            # Face 0
            phi_L_0 = phi_yext[:, :, 1]
            phi_R_0 = phi_yext[:, :, 2]
            alpha_0 = jnp.abs(sqrt_G_v2_yface[:, :, 0])
            Fy = Fy.at[:, :, 0].add(-sigma/2 * alpha_0 * (phi_R_0 - phi_L_0))
            
            # Face N
            phi_L_N = phi_yext[:, :, N+1]
            phi_R_N = phi_yext[:, :, N+2]
            alpha_N = jnp.abs(sqrt_G_v2_yface[:, :, N])
            Fy = Fy.at[:, :, N].add(-sigma/2 * alpha_N * (phi_R_N - phi_L_N))
        
        dFy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dx  # (6, N, N)
        
        return -inv_sqrt_G * (dFx + dFy)

    return rhs


def make_stepper(panels, N, dx, dt, halo_fn, sigma=0.0):
    rhs_fn = make_rhs_fn(panels, N, dx, sigma=sigma)

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
# Run one experiment
# ============================================================

def run_one(N, sigma, label):
    CFL = 0.5
    order = 4

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

    step_fn = make_stepper(panels, N, dx, dt, halo_fn, sigma=sigma)

    # JIT warmup
    _ = step_fn(phi0)
    jax.block_until_ready(_)

    phi = phi0.copy()
    t_start = time.perf_counter()
    for s in range(nsteps):
        phi = step_fn(phi)
    jax.block_until_ready(phi)
    elapsed = time.perf_counter() - t_start

    mass_final = total_mass(phi, panels, dx)
    mass_err = abs(mass_final - mass0) / abs(mass0)
    max_final = float(jnp.max(phi))
    min_final = float(jnp.min(phi))
    l1, l2, linf = compute_error_norms(phi, phi0, panels, dx)
    amp_loss = (max0 - max_final) / max0

    return {
        'label': label, 'sigma': sigma, 'N': N,
        'l1': l1, 'l2': l2, 'linf': linf,
        'mass_err': mass_err, 'max': max_final, 'min': min_final,
        'amp_loss': amp_loss, 'elapsed': elapsed, 'nsteps': nsteps,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    print("=" * 75)
    print(f"  SAT COMPARISON TEST — N = {N}, CFL = 0.5, full rotation")
    print("=" * 75)

    variants = [
        (0.0,  "No SAT (centered)"),
        (0.5,  "SAT σ=0.5 (half upwind)"),
        (1.0,  "SAT σ=1.0 (full upwind)"),
    ]

    results = []
    for sigma, label in variants:
        print(f"\n  Running: {label}...")
        r = run_one(N, sigma, label)
        results.append(r)
        print(f"    l1={r['l1']:.4e}  l2={r['l2']:.4e}  linf={r['linf']:.4e}  "
              f"min={r['min']:.2f}  mass_err={r['mass_err']:.2e}  "
              f"time={r['elapsed']:.1f}s")

    # Comparison table
    print()
    print("=" * 75)
    print(f"  COMPARISON TABLE — N = {N}")
    print("=" * 75)
    print(f"  {'Variant':<25} {'l1':>10} {'l2':>10} {'linf':>10} "
          f"{'min':>10} {'amp_loss':>10} {'mass_err':>10}")
    print("-" * 75)
    for r in results:
        print(f"  {r['label']:<25} {r['l1']:10.4e} {r['l2']:10.4e} {r['linf']:10.4e} "
              f"{r['min']:10.2f} {r['amp_loss']:9.1%} {r['mass_err']:10.2e}")
    
    print()
    print("  FV3 reference (N=40):   1.0300e-01 6.6900e-02 4.9400e-02"
          "      -0.03     -3.5%")
    print("=" * 75)
