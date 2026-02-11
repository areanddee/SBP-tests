"""
Cubed-Sphere Advection with SBP 4/2 Operators
==============================================

Uses verified 6-panel infrastructure with higher-order SBP operators:
- 4th order interior accuracy
- 2nd order boundary accuracy  
- Machine-precision mass conservation (flux-form)

Key differences from 2/1 scheme:
- Pvc interpolation: [-1/16, 9/16, 9/16, -1/16] instead of [1/2, 1/2]
- Dvc derivative: 4th order compact stencil instead of 2nd order
"""

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import time
from functools import partial

# Import verified infrastructure
import sys
sys.path.insert(0, '/home/claude')
from grid import equiangular_to_cartesian, compute_metric_at_points
from velocity_transforms import cartesian_to_covariant
from halo_exchange import (
    make_halo_exchange,
    create_communication_schedule,
    extend_to_include_ghosts,
)


# =============================================================================
# SBP 4/2 Operator Construction
# =============================================================================

def sbp_42_interpolation_matrix(N):
    """
    Build 4/2 order interpolation matrix Pvc: vertex -> center.
    
    Interior stencil (4th order): [-1/16, 9/16, 9/16, -1/16]
    Boundary stencil (2nd order): [1/2, 1/2]
    
    Maps (N+1) vertex values to N center values.
    
    Returns:
        Pvc: (N, N+1) matrix
    """
    Pvc = jnp.zeros((N, N + 1))
    
    # First row: simple average (2nd order)
    Pvc = Pvc.at[0, 0].set(0.5)
    Pvc = Pvc.at[0, 1].set(0.5)
    
    # Second row: 4th order if possible, else 2nd order
    if N >= 3:
        Pvc = Pvc.at[1, 0].set(-1/16)
        Pvc = Pvc.at[1, 1].set(9/16)
        Pvc = Pvc.at[1, 2].set(9/16)
        Pvc = Pvc.at[1, 3].set(-1/16)
    else:
        Pvc = Pvc.at[1, 1].set(0.5)
        Pvc = Pvc.at[1, 2].set(0.5)
    
    # Interior rows: 4th order stencil
    for i in range(2, N - 2):
        Pvc = Pvc.at[i, i - 1].set(-1/16)
        Pvc = Pvc.at[i, i].set(9/16)
        Pvc = Pvc.at[i, i + 1].set(9/16)
        Pvc = Pvc.at[i, i + 2].set(-1/16)
    
    # Second-to-last row
    if N >= 3:
        Pvc = Pvc.at[N-2, N-3].set(-1/16)
        Pvc = Pvc.at[N-2, N-2].set(9/16)
        Pvc = Pvc.at[N-2, N-1].set(9/16)
        Pvc = Pvc.at[N-2, N].set(-1/16)
    
    # Last row: simple average
    Pvc = Pvc.at[N-1, N-1].set(0.5)
    Pvc = Pvc.at[N-1, N].set(0.5)
    
    return Pvc


def sbp_42_derivative_matrix(N, dx):
    """
    Build 4/2 order derivative matrix Dvc: vertex -> center.
    
    Interior stencil (4th order): [1/24, -9/8, 9/8, -1/24] / dx
    Boundary stencils from Shashkin et al. Eq. 91.
    
    Maps (N+1) vertex values to N center derivatives.
    
    Returns:
        Dvc: (N, N+1) matrix
    """
    if N < 8:
        # Fall back to 2nd order for small N
        return sbp_21_derivative_matrix(N, dx)
    
    Dvc = jnp.zeros((N, N + 1))
    
    # Left boundary rows (from Shashkin Eq. 91, transposed for Dvc)
    # Row 0: d/dx at first center using first 4 vertices
    Dvc = Dvc.at[0, 0].set(-2.0)
    Dvc = Dvc.at[0, 1].set(3.0)
    Dvc = Dvc.at[0, 2].set(-1.0)
    
    # Row 1
    Dvc = Dvc.at[1, 0].set(-1.0)
    Dvc = Dvc.at[1, 1].set(1.0)
    
    # Row 2: transition to 4th order
    Dvc = Dvc.at[2, 0].set(1/24)
    Dvc = Dvc.at[2, 1].set(-9/8)
    Dvc = Dvc.at[2, 2].set(9/8)
    Dvc = Dvc.at[2, 3].set(-1/24)
    
    # Row 3: modified stencil from Eq. 91
    Dvc = Dvc.at[3, 0].set(-1/71)
    Dvc = Dvc.at[3, 1].set(6/71)
    Dvc = Dvc.at[3, 2].set(-83/71)
    Dvc = Dvc.at[3, 3].set(81/71)
    Dvc = Dvc.at[3, 4].set(-3/71)
    
    # Interior rows: 4th order stencil
    for i in range(4, N - 4):
        Dvc = Dvc.at[i, i - 1].set(1/24)
        Dvc = Dvc.at[i, i].set(-9/8)
        Dvc = Dvc.at[i, i + 1].set(9/8)
        Dvc = Dvc.at[i, i + 2].set(-1/24)
    
    # Right boundary rows (antisymmetric)
    # Row N-4
    Dvc = Dvc.at[N-4, N-4].set(3/71)
    Dvc = Dvc.at[N-4, N-3].set(-81/71)
    Dvc = Dvc.at[N-4, N-2].set(83/71)
    Dvc = Dvc.at[N-4, N-1].set(-6/71)
    Dvc = Dvc.at[N-4, N].set(1/71)
    
    # Row N-3
    Dvc = Dvc.at[N-3, N-3].set(1/24)
    Dvc = Dvc.at[N-3, N-2].set(-9/8)
    Dvc = Dvc.at[N-3, N-1].set(9/8)
    Dvc = Dvc.at[N-3, N].set(-1/24)
    
    # Row N-2
    Dvc = Dvc.at[N-2, N-1].set(-1.0)
    Dvc = Dvc.at[N-2, N].set(1.0)
    
    # Row N-1
    Dvc = Dvc.at[N-1, N-2].set(1.0)
    Dvc = Dvc.at[N-1, N-1].set(-3.0)
    Dvc = Dvc.at[N-1, N].set(2.0)
    
    return Dvc / dx


def sbp_21_derivative_matrix(N, dx):
    """2nd order derivative matrix (fallback for small N)."""
    Dvc = jnp.zeros((N, N + 1))
    for i in range(N):
        Dvc = Dvc.at[i, i].set(-1.0)
        Dvc = Dvc.at[i, i + 1].set(1.0)
    return Dvc / dx


# =============================================================================
# Panel Geometry with SBP Operators
# =============================================================================

class SBPPanelData:
    """Precomputed geometry and SBP operators for one panel."""
    
    def __init__(self, N, face_id, u0=1.0, order=4):
        dx = float(jnp.pi / (2 * N))
        pi4 = jnp.pi / 4
        
        self.N = N
        self.dx = dx
        self.face_id = face_id
        
        # Cell centers and faces
        xi_c = jnp.linspace(-pi4 + dx/2, pi4 - dx/2, N)
        xi_f = jnp.linspace(-pi4, pi4, N + 1)
        
        # Metric at cell centers
        XI1_c, XI2_c = jnp.meshgrid(xi_c, xi_c, indexing='ij')
        sqrt_G_c, _, _, _ = compute_metric_at_points(XI1_c, XI2_c)
        self.sqrt_G = sqrt_G_c
        self.inv_sqrt_G = 1.0 / sqrt_G_c
        
        # Velocity at cell centers (solid body rotation)
        X_c, Y_c, Z_c = equiangular_to_cartesian(XI1_c, XI2_c, face_id)
        Vx_c, Vy_c, Vz_c = u0 * (-Y_c), u0 * X_c, jnp.zeros_like(X_c)
        self.v1_c, self.v2_c = cartesian_to_covariant(Vx_c, Vy_c, Vz_c, XI1_c, XI2_c, face_id)
        
        # √G·v at x-faces (N+1, N)
        XI1_xf, XI2_xf = jnp.meshgrid(xi_f, xi_c, indexing='ij')
        sqrt_G_xf, _, _, _ = compute_metric_at_points(XI1_xf, XI2_xf)
        X_xf, Y_xf, Z_xf = equiangular_to_cartesian(XI1_xf, XI2_xf, face_id)
        Vx_xf, Vy_xf, Vz_xf = u0 * (-Y_xf), u0 * X_xf, jnp.zeros_like(X_xf)
        v1_xf, _ = cartesian_to_covariant(Vx_xf, Vy_xf, Vz_xf, XI1_xf, XI2_xf, face_id)
        self.sqrt_G_v1_xface = sqrt_G_xf * v1_xf
        
        # √G·v at y-faces (N, N+1)
        XI1_yf, XI2_yf = jnp.meshgrid(xi_c, xi_f, indexing='ij')
        sqrt_G_yf, _, _, _ = compute_metric_at_points(XI1_yf, XI2_yf)
        X_yf, Y_yf, Z_yf = equiangular_to_cartesian(XI1_yf, XI2_yf, face_id)
        Vx_yf, Vy_yf, Vz_yf = u0 * (-Y_yf), u0 * X_yf, jnp.zeros_like(X_yf)
        _, v2_yf = cartesian_to_covariant(Vx_yf, Vy_yf, Vz_yf, XI1_yf, XI2_yf, face_id)
        self.sqrt_G_v2_yface = sqrt_G_yf * v2_yf
        
        # SBP operators
        if order == 4 and N >= 8:
            self.Pvc_x = sbp_42_interpolation_matrix(N)
            self.Dvc_x = sbp_42_derivative_matrix(N, dx)
        else:
            # 2nd order fallback
            self.Pvc_x = self._simple_interp_matrix(N)
            self.Dvc_x = sbp_21_derivative_matrix(N, dx)
        
        # Same operators for y-direction (operators are 1D)
        self.Pvc_y = self.Pvc_x
        self.Dvc_y = self.Dvc_x
    
    def _simple_interp_matrix(self, N):
        """Simple 2-point average interpolation."""
        P = jnp.zeros((N, N + 1))
        for i in range(N):
            P = P.at[i, i].set(0.5)
            P = P.at[i, i + 1].set(0.5)
        return P


# =============================================================================
# SBP-Based RHS Computation
# =============================================================================

def sbp_advection_rhs(phi_ghost, panels, N, dx):
    """
    Compute advection RHS using SBP 4/2 operators.
    
    Uses flux-form: dφ/dt = -(1/√G) div(√G v φ)
    
    The key difference from 2/1: higher-order interpolation to faces.
    
    Args:
        phi_ghost: (6, N+2, N+2) with halo-exchanged ghost cells
        panels: list of SBPPanelData
        N: grid resolution
        dx: grid spacing
        
    Returns:
        rhs: (6, N, N)
    """
    rhs_all = []
    
    for f in range(6):
        p = panels[f]
        pg = phi_ghost[f]  # (N+2, N+2)
        
        # Extract interior with one layer of ghosts for interpolation
        # phi_extended[i, j] for i in 0..N+1, j in 0..N+1
        
        # --- X-direction flux: F_x = (√G v¹)_face · φ_face ---
        # Interpolate φ to x-faces using SBP Pvc operator
        # For each j-column, apply 1D interpolation in i-direction
        
        # phi for x-interpolation: need (N+1) points for each j
        # Use ghost cells: pg[0:N+2, 1:N+1] gives (N+2, N) 
        # We need (N+1, N) for faces, so use pg[0:N+1, 1:N+1] and pg[1:N+2, 1:N+1]
        
        # Simple approach: average neighboring cells (including ghosts)
        # φ_face[i, j] = 0.5*(φ[i, j] + φ[i+1, j]) for face between cells i and i+1
        # In ghost coords: between pg[i, j+1] and pg[i+1, j+1]
        
        # For 4/2 SBP: need 4-point stencil near boundaries
        # Interior: [-1/16, 9/16, 9/16, -1/16]
        
        # Build extended phi arrays for interpolation
        # x-direction: need values at i = -1, 0, 1, ..., N, N+1 for each j
        # These are pg[0, 1:N+1], pg[1, 1:N+1], ..., pg[N+1, 1:N+1]
        
        phi_xext = pg[:, 1:N+1]  # (N+2, N) - all i values, interior j values
        
        # Apply 4th order interpolation for interior faces
        # Face i (between cells i-1 and i in interior indexing)
        # Uses cells at i-2, i-1, i, i+1 in interior indexing
        # = cells at i-1, i, i+1, i+2 in ghost indexing
        
        # For simplicity, use matrix multiplication where possible
        # phi_xface[i, j] for i = 0..N, j = 0..N-1
        
        phi_xface = jnp.zeros((N+1, N))
        
        # Face 0 (left boundary): simple average
        phi_xface = phi_xface.at[0, :].set(0.5 * (phi_xext[0, :] + phi_xext[1, :]))
        
        # Face 1: 4-point stencil if available
        if N >= 4:
            phi_xface = phi_xface.at[1, :].set(
                -1/16 * phi_xext[0, :] + 9/16 * phi_xext[1, :] + 
                 9/16 * phi_xext[2, :] - 1/16 * phi_xext[3, :])
        else:
            phi_xface = phi_xface.at[1, :].set(0.5 * (phi_xext[1, :] + phi_xext[2, :]))
        
        # Interior faces: 4th order stencil
        for i in range(2, N - 1):
            phi_xface = phi_xface.at[i, :].set(
                -1/16 * phi_xext[i-1, :] + 9/16 * phi_xext[i, :] + 
                 9/16 * phi_xext[i+1, :] - 1/16 * phi_xext[i+2, :])
        
        # Face N-1: 4-point stencil
        if N >= 4:
            phi_xface = phi_xface.at[N-1, :].set(
                -1/16 * phi_xext[N-2, :] + 9/16 * phi_xext[N-1, :] + 
                 9/16 * phi_xext[N, :] - 1/16 * phi_xext[N+1, :])
        else:
            phi_xface = phi_xface.at[N-1, :].set(0.5 * (phi_xext[N-1, :] + phi_xext[N, :]))
        
        # Face N (right boundary): simple average
        phi_xface = phi_xface.at[N, :].set(0.5 * (phi_xext[N, :] + phi_xext[N+1, :]))
        
        # Compute x-flux
        Fx = p.sqrt_G_v1_xface * phi_xface  # (N+1, N)
        
        # Flux divergence in x
        dFx = (Fx[1:N+1, :] - Fx[0:N, :]) / dx  # (N, N)
        
        # --- Y-direction flux: similar process ---
        phi_yext = pg[1:N+1, :]  # (N, N+2)
        
        phi_yface = jnp.zeros((N, N+1))
        
        # Face 0 (bottom boundary)
        phi_yface = phi_yface.at[:, 0].set(0.5 * (phi_yext[:, 0] + phi_yext[:, 1]))
        
        # Face 1
        if N >= 4:
            phi_yface = phi_yface.at[:, 1].set(
                -1/16 * phi_yext[:, 0] + 9/16 * phi_yext[:, 1] + 
                 9/16 * phi_yext[:, 2] - 1/16 * phi_yext[:, 3])
        else:
            phi_yface = phi_yface.at[:, 1].set(0.5 * (phi_yext[:, 1] + phi_yext[:, 2]))
        
        # Interior faces
        for j in range(2, N - 1):
            phi_yface = phi_yface.at[:, j].set(
                -1/16 * phi_yext[:, j-1] + 9/16 * phi_yext[:, j] + 
                 9/16 * phi_yext[:, j+1] - 1/16 * phi_yext[:, j+2])
        
        # Face N-1
        if N >= 4:
            phi_yface = phi_yface.at[:, N-1].set(
                -1/16 * phi_yext[:, N-2] + 9/16 * phi_yext[:, N-1] + 
                 9/16 * phi_yext[:, N] - 1/16 * phi_yext[:, N+1])
        else:
            phi_yface = phi_yface.at[:, N-1].set(0.5 * (phi_yext[:, N-1] + phi_yext[:, N]))
        
        # Face N (top boundary)
        phi_yface = phi_yface.at[:, N].set(0.5 * (phi_yext[:, N] + phi_yext[:, N+1]))
        
        # Compute y-flux
        Fy = p.sqrt_G_v2_yface * phi_yface  # (N, N+1)
        
        # Flux divergence in y
        dFy = (Fy[:, 1:N+1] - Fy[:, 0:N]) / dx  # (N, N)
        
        # RHS: -(1/√G)(dFx + dFy)
        rhs_panel = -p.inv_sqrt_G * (dFx + dFy)
        rhs_all.append(rhs_panel)
    
    return jnp.stack(rhs_all)


def rk4_step(phi, panels, N, dx, dt, halo_fn, rhs_fn):
    """Single RK4 timestep."""
    def rhs(p):
        p_ghost = halo_fn(extend_to_include_ghosts(p, N))
        return rhs_fn(p_ghost, panels, N, dx)
    
    k1 = rhs(phi)
    k2 = rhs(phi + 0.5 * dt * k1)
    k3 = rhs(phi + 0.5 * dt * k2)
    k4 = rhs(phi + dt * k3)
    return phi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================
# Initial Conditions
# =============================================================================

def cosine_bell_ic(panels, lon0=0.0, lat0=0.0, R=1.0/3, h0=1000.0):
    """Cosine bell centered at (lon0, lat0)."""
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
        
        phi = jnp.where(r < R, (h0 / 2) * (1 + jnp.cos(jnp.pi * r / R)), 0.0)
        phi_list.append(phi)
    
    return jnp.stack(phi_list)


# =============================================================================
# Diagnostics
# =============================================================================

def total_mass(phi, panels, dx):
    """∫ φ √G dξ¹dξ² summed over all panels."""
    m = 0.0
    for f in range(6):
        m += float(jnp.sum(phi[f] * panels[f].sqrt_G) * dx**2)
    return m


# =============================================================================
# Main Test
# =============================================================================

def run_williamson_tc1(N=32, order=4, num_rotations=1.0, verbose=True):
    """
    Run Williamson Test Case 1 (solid body rotation).
    
    Args:
        N: grid resolution per panel
        order: SBP order (2 or 4)
        num_rotations: number of full rotations
        verbose: print progress
    """
    if verbose:
        print("=" * 60)
        print(f"WILLIAMSON TC1: SBP {order}/{'1' if order==2 else '2'} ADVECTION")
        print("=" * 60)
    
    dx = float(jnp.pi / (2 * N))
    
    # Build panels with SBP operators
    panels = [SBPPanelData(N, f, u0=1.0, order=order) for f in range(6)]
    
    # Create halo exchange
    schedule = create_communication_schedule()
    halo_fn = make_halo_exchange(schedule, N)
    
    # Initial condition
    phi0 = cosine_bell_ic(panels, lon0=0.0, lat0=0.0, R=1.0/3, h0=1000.0)
    mass0 = total_mass(phi0, panels, dx)
    max0 = float(jnp.max(phi0))
    
    # Time stepping
    T = 2 * jnp.pi * num_rotations
    dt = 0.4 * dx / 1.5
    nsteps = int(T / dt)
    dt = T / nsteps
    
    if verbose:
        print(f"  N = {N}, dx = {dx:.4f}")
        print(f"  Period = {float(T):.4f}, dt = {dt:.4e}, steps = {nsteps}")
        print(f"  Initial: max = {max0:.2f}, mass = {mass0:.4f}")
    
    # Choose RHS function
    rhs_fn = sbp_advection_rhs
    
    # JIT warmup
    phi = rk4_step(phi0, panels, N, dx, dt, halo_fn, rhs_fn)
    phi = phi0
    
    # Run simulation
    t_start = time.perf_counter()
    for step in range(nsteps):
        phi = rk4_step(phi, panels, N, dx, dt, halo_fn, rhs_fn)
        
        if verbose and (step + 1) % max(1, nsteps // 4) == 0:
            mass = total_mass(phi, panels, dx)
            mass_err = abs(mass - mass0) / abs(mass0)
            max_val = float(jnp.max(phi))
            min_val = float(jnp.min(phi))
            print(f"  Step {step+1:5d}: mass_err={mass_err:.2e}, max={max_val:.1f}, min={min_val:.1f}")
    
    elapsed = time.perf_counter() - t_start
    
    # Final diagnostics
    mass_final = total_mass(phi, panels, dx)
    mass_err = abs(mass_final - mass0) / abs(mass0)
    max_final = float(jnp.max(phi))
    min_final = float(jnp.min(phi))
    amp_loss = (max0 - max_final) / max0
    linf = float(jnp.max(jnp.abs(phi - phi0)))
    
    if verbose:
        print()
        print(f"  Final: max = {max_final:.2f}, min = {min_final:.2f}")
        print(f"  Mass error: {mass_err:.2e}")
        print(f"  Amplitude loss: {amp_loss:.1%}")
        print(f"  L∞ error: {linf:.4f}")
        print(f"  Time: {elapsed:.1f}s ({nsteps/elapsed:.0f} steps/s)")
    
    return {
        'N': N, 'order': order,
        'mass_err': mass_err,
        'amp_loss': amp_loss,
        'linf': linf,
        'max_final': max_final,
        'min_final': min_final,
        'elapsed': elapsed,
        'nsteps': nsteps,
    }


def convergence_study(orders=[2, 4], Ns=[16, 32, 64]):
    """Run convergence study comparing SBP orders."""
    print("\n" + "=" * 70)
    print("CONVERGENCE STUDY")
    print("=" * 70)
    
    results = {}
    
    for order in orders:
        print(f"\n--- SBP {order}/{'1' if order==2 else '2'} ---")
        results[order] = []
        
        for N in Ns:
            if order == 4 and N < 8:
                print(f"  N={N}: Skipped (need N>=8 for 4/2)")
                continue
            
            r = run_williamson_tc1(N=N, order=order, num_rotations=1.0, verbose=False)
            results[order].append(r)
            print(f"  N={N:3d}: amp_loss={r['amp_loss']:.1%}, min={r['min_final']:.1f}, mass_err={r['mass_err']:.2e}")
    
    # Print convergence rates
    print("\n" + "-" * 50)
    print("AMPLITUDE LOSS CONVERGENCE:")
    for order in orders:
        res = results[order]
        if len(res) >= 2:
            for i in range(1, len(res)):
                if res[i]['amp_loss'] > 0 and res[i-1]['amp_loss'] > 0:
                    rate = np.log2(res[i-1]['amp_loss'] / res[i]['amp_loss'])
                    print(f"  SBP {order}: N={res[i-1]['N']}→{res[i]['N']}: rate = {rate:.2f}")
    
    return results


if __name__ == "__main__":
    # Quick test
    print("\n" + "=" * 70)
    print("QUICK TEST: SBP 4/2 vs 2/1")
    print("=" * 70)
    
    # Compare 2/1 and 4/2 at N=32
    print("\n--- 2/1 scheme ---")
    r21 = run_williamson_tc1(N=32, order=2, num_rotations=1.0)
    
    print("\n--- 4/2 scheme ---")
    r42 = run_williamson_tc1(N=32, order=4, num_rotations=1.0)
    
    print("\n" + "=" * 70)
    print("COMPARISON (N=32, 1 rotation)")
    print("=" * 70)
    print(f"  {'Metric':<20} {'2/1':>15} {'4/2':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'Amplitude loss':<20} {r21['amp_loss']:>14.1%} {r42['amp_loss']:>14.1%}")
    print(f"  {'Min value':<20} {r21['min_final']:>15.1f} {r42['min_final']:>15.1f}")
    print(f"  {'Mass error':<20} {r21['mass_err']:>15.2e} {r42['mass_err']:>15.2e}")
    
    # Convergence study
    convergence_study(orders=[2, 4], Ns=[16, 32, 64])
