"""
test_stag_step3.py — 6-Panel Ring, Multi-Block SAT-Projection
=============================================================

GOAL: Validate multi-panel SAT-Projection coupling with 6 blocks
before introducing cubed-sphere connectivity and metrics.

LAYOUT:
  6 panels arranged in a periodic ring along x:
    [0]-[1]-[2]-[3]-[4]-[5]-[0]  (periodic wrap)

  Each panel is [0, L_panel] × [0, L_panel].
  Global domain: [0, 6*L_panel] × [0, L_panel], doubly periodic.
  Flat geometry: J=1, Q=I (isolates panel coupling errors).

COUPLING (Shashkin Eqs. 50-55, simplified for flat geometry):
  x-direction (inter-panel):
    - h-projection: average h at shared boundary vertices (Eq. 51)
    - SAT correction: average velocity flux at interfaces (Eq. 53-55)
    - Gradient: raw Dvc on projected h
    - Divergence: raw Dcv + SAT, then project result

  y-direction (self-periodic within each panel):
    - h-projection: average h[:,0] and h[:,-1] (Eq. 33)
    - SAT-corrected divergence DS_cv (Eq. 26-27)
    - Gradient: raw Dvc on projected h

STATE:
  h:  (6, N+1, N+1)  height at vertices
  v1: (6, N, N+1)    x-velocity at x-faces
  v2: (6, N+1, N)    y-velocity at y-faces

TARGETS:
  - Convergence rate ≥ 3 (SBP 4/2)
  - Mass conservation: machine precision
  - Energy: spatial exact, temporal dt^5
  - Smooth wave propagation across panel boundaries

Reference: Shashkin 2025, Sections 2.2, 3, 4.2

Usage:
    python test_stag_step3.py
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

from sbp_staggered_1d import (sbp_42, apply_sat_correction,
                               make_projection_matrix)


# ============================================================
# Build 6-panel ring SWE system
# ============================================================

def make_6panel_swe_system(N, L_panel, H0, g):
    """
    Build 6-panel ring SWE system with inter-panel SAT-Projection.

    Panels are connected left-to-right in x (periodic ring of 6).
    Each panel is self-periodic in y.
    Flat geometry: J=1, Q=I.
    """
    dx = L_panel / N
    ops = sbp_42(N, dx)

    # Raw operators (for x-direction inter-panel)
    Dvc = ops.Dvc       # [N, N+1] gradient: vertices→centers
    Dcv = ops.Dcv       # [N+1, N] divergence: centers→vertices
    l = ops.l           # [N] left boundary extrapolation
    r = ops.r           # [N] right boundary extrapolation

    # Quadrature diagonals
    Hv_diag = jnp.diag(ops.Hv)   # (N+1,)
    Hc_diag = jnp.diag(ops.Hc)   # (N,)
    Hv_inv_0 = 1.0 / Hv_diag[0]
    Hv_inv_N = 1.0 / Hv_diag[N]

    # SAT-corrected divergence for y-direction (self-periodic)
    _, DS_cv_y = apply_sat_correction(ops)  # [N+1, N]

    # 2D quadrature weights per panel
    Wh = jnp.outer(Hv_diag, Hv_diag)   # (N+1, N+1) for h
    W1 = jnp.outer(Hc_diag, Hv_diag)   # (N, N+1) for v1
    W2 = jnp.outer(Hv_diag, Hc_diag)   # (N+1, N) for v2

    # ----------------------------------------------------------
    # Projection: enforce h-continuity at all interfaces
    # ----------------------------------------------------------
    def project_h(h):
        """
        Project h at all interfaces.

        x-interfaces: panel p's right (i=N) ↔ panel (p+1)%6's left (i=0)
        y-interfaces: self-periodic within each panel (j=0 ↔ j=N)

        h: (6, N+1, N+1)
        """
        # y-projection (self-periodic within each panel)
        avg_y = 0.5 * (h[:, :, 0] + h[:, :, -1])
        h = h.at[:, :, 0].set(avg_y)
        h = h.at[:, :, -1].set(avg_y)

        # x-projection (inter-panel)
        # Panel p's right boundary h[:,−1,:] connects to panel (p+1)%6's left h[:,0,:]
        rights = h[:, -1, :]                           # (6, N+1)
        lefts_next = jnp.roll(h[:, 0, :], -1, axis=0)  # (6, N+1)

        # Weighted average (for SBP 4/2, boundary Hv weights may differ)
        w_r = Hv_diag[N] / (Hv_diag[N] + Hv_diag[0])
        w_l = Hv_diag[0] / (Hv_diag[N] + Hv_diag[0])
        avg_x = w_r * rights + w_l * lefts_next

        h = h.at[:, -1, :].set(avg_x)
        h = h.at[:, 0, :].set(jnp.roll(avg_x, 1, axis=0))

        return h

    # ----------------------------------------------------------
    # SAT correction for x-direction inter-panel velocity flux
    # ----------------------------------------------------------
    def sat_x_corrections(v1):
        """
        Compute SAT corrections at inter-panel x-boundaries.

        At row 0 of panel p (interface with panel p−1):
          sat_0 = −0.5 · Hv⁻¹[0] · (r·v1[p−1] − l·v1[p])
          (replaces one-sided flux with interface average)

        At row N of panel p (interface with panel p+1):
          sat_N = −0.5 · Hv⁻¹[N] · (r·v1[p] − l·v1[p+1])

        v1: (6, N, N+1)
        Returns: sat_0 (6, N+1), sat_N (6, N+1)
        """
        # Extrapolate v1 to boundaries using l and r
        # l·v1[p] = extrapolation of panel p's velocity to its LEFT boundary
        # r·v1[p] = extrapolation of panel p's velocity to its RIGHT boundary
        extrap_left = jnp.einsum('c,pcj->pj', l, v1)    # (6, N+1)
        extrap_right = jnp.einsum('c,pcj->pj', r, v1)   # (6, N+1)

        # SAT at row 0: interface with panel (p−1)%6
        extrap_right_prev = jnp.roll(extrap_right, 1, axis=0)
        sat_0 = -0.5 * Hv_inv_0 * (extrap_right_prev - extrap_left)

        # SAT at row N: interface with panel (p+1)%6
        extrap_left_next = jnp.roll(extrap_left, -1, axis=0)
        sat_N = -0.5 * Hv_inv_N * (extrap_right - extrap_left_next)

        return sat_0, sat_N

    # ----------------------------------------------------------
    # RHS
    # ----------------------------------------------------------
    def rhs(h, v1, v2):
        """
        RHS of linearized SWE on 6-panel flat ring.

        h:  (6, N+1, N+1)
        v1: (6, N, N+1)
        v2: (6, N+1, N)
        """
        # Project h at all interfaces
        h_proj = project_h(h)

        # === Gradient (momentum) ===
        # x: raw Dvc @ h_projected (Eq. 50a with Dhv·Ah·h)
        dv1_dt = -g * jnp.einsum('ij,pjk->pik', Dvc, h_proj)  # (6,N,N+1)
        # y: raw Dvc @ h_projected along y (h already y-projected)
        dv2_dt = -g * jnp.einsum('pij,kj->pik', h_proj, Dvc)   # (6,N+1,N)

        # === Divergence + SAT (continuity) ===
        # x-divergence: raw Dcv, then add inter-panel SAT
        div_x = jnp.einsum('ij,pjk->pik', Dcv, v1)   # (6,N+1,N+1)
        sat_0, sat_N = sat_x_corrections(v1)
        div_x = div_x.at[:, 0, :].add(sat_0)
        div_x = div_x.at[:, -1, :].add(sat_N)

        # y-divergence: SAT-corrected Dcv for self-periodic y
        div_y = jnp.einsum('pij,kj->pik', v2, DS_cv_y)  # (6,N+1,N+1)

        # Total divergence, project at all interfaces
        dh_dt = -H0 * project_h(div_x + div_y)

        return dh_dt, dv1_dt, dv2_dt

    return {
        'rhs': jax.jit(rhs),
        'project_h': project_h,
        'ops': ops,
        'Dvc': Dvc, 'Dcv': Dcv,
        'DS_cv_y': DS_cv_y,
        'l': l, 'r': r,
        'Wh': Wh, 'W1': W1, 'W2': W2,
        'Hv_diag': Hv_diag, 'Hc_diag': Hc_diag,
        'dx': dx, 'N': N, 'L_panel': L_panel,
    }


# ============================================================
# RK4 time stepper
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
# Grid helpers
# ============================================================

def make_grids(N, L_panel, n_panels=6):
    """Create global coordinate grids for all panels."""
    dx = L_panel / N
    x_v_local = jnp.linspace(0, L_panel, N + 1)   # (N+1,) local vertex coords
    x_c_local = (jnp.arange(N) + 0.5) * dx         # (N,)   local center coords
    y_v = x_v_local                                  # same in y
    y_c = x_c_local

    # Global x-coordinates per panel
    # Panel p: x_global = p*L_panel + x_local
    offsets = jnp.arange(n_panels) * L_panel         # (6,)

    # h-grid: (6, N+1, N+1)
    Xh = offsets[:, None, None] + x_v_local[None, :, None] * jnp.ones(N + 1)[None, None, :]
    Yh = jnp.ones(n_panels)[:, None, None] * jnp.ones(N + 1)[None, :, None] * y_v[None, None, :]

    # v1-grid: (6, N, N+1)
    Xv1 = offsets[:, None, None] + x_c_local[None, :, None] * jnp.ones(N + 1)[None, None, :]
    Yv1 = jnp.ones(n_panels)[:, None, None] * jnp.ones(N)[None, :, None] * y_v[None, None, :]

    # v2-grid: (6, N+1, N)
    Xv2 = offsets[:, None, None] + x_v_local[None, :, None] * jnp.ones(N)[None, None, :]
    Yv2 = jnp.ones(n_panels)[:, None, None] * jnp.ones(N + 1)[None, :, None] * y_c[None, None, :]

    return Xh, Yh, Xv1, Yv1, Xv2, Yv2


# ============================================================
# Diagnostics
# ============================================================

def compute_mass(h, Wh):
    """Global mass = sum over all panels of ∫h dA."""
    return float(jnp.sum(h * Wh[None, :, :]))


def compute_energy(h, v1, v2, Wh, W1, W2, g, H0):
    PE = 0.5 * g * float(jnp.sum(h**2 * Wh[None, :, :]))
    KE = 0.5 * H0 * (float(jnp.sum(v1**2 * W1[None, :, :])) +
                       float(jnp.sum(v2**2 * W2[None, :, :])))
    return PE + KE


# ============================================================
# Test 1: Multi-panel SBP property
# ============================================================

def test_sbp_multi():
    """
    Verify the global SBP identity holds across all 6 panels.

    For each direction, the identity should hold independently:
      x: sum_p v1[p]^T W1 (Dvc @ h_proj[p]) + (Ah_x)[p]^T Wh (Dcv @ v1[p] + SAT_x) = 0
      y: sum_p v2[p]^T W2 (h_proj[p] @ Dvc^T) + (Ah_y)[p]^T Wh (v2[p] @ DS_cv^T) = 0
    """
    print("\n" + "=" * 65)
    print("TEST 1: Multi-Panel SBP Property")
    print("=" * 65)

    passed = True
    for N in [16, 32]:
        L_panel = 1.0
        sys_d = make_6panel_swe_system(N, L_panel, H0=1.0, g=1.0)
        Dvc = sys_d['Dvc']
        Dcv = sys_d['Dcv']
        DS_cv_y = sys_d['DS_cv_y']
        Wh = sys_d['Wh']
        W1 = sys_d['W1']
        W2 = sys_d['W2']
        l = sys_d['l']
        r = sys_d['r']
        Hv_diag = sys_d['Hv_diag']

        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (6, N+1, N+1))
        v1 = jax.random.normal(k2, (6, N, N+1))
        v2 = jax.random.normal(k3, (6, N+1, N))

        # --- x-direction: project h in x only, use SAME h for both terms ---
        rights = h[:, -1, :]
        lefts_next = jnp.roll(h[:, 0, :], -1, axis=0)
        w_r = Hv_diag[N] / (Hv_diag[N] + Hv_diag[0])
        w_l = Hv_diag[0] / (Hv_diag[N] + Hv_diag[0])
        avg_x = w_r * rights + w_l * lefts_next
        h_x = h.at[:, -1, :].set(avg_x).at[:, 0, :].set(jnp.roll(avg_x, 1, axis=0))

        # Gradient: Dvc @ h_x
        grad_x = jnp.einsum('ij,pjk->pik', Dvc, h_x)
        grad_x_sum = jnp.sum(v1 * W1[None,:,:] * grad_x)

        # Divergence + SAT: Dcv @ v1 + SAT, weighted by SAME h_x
        div_x_raw = jnp.einsum('ij,pjk->pik', Dcv, v1)
        extrap_l = jnp.einsum('c,pcj->pj', l, v1)
        extrap_r = jnp.einsum('c,pcj->pj', r, v1)
        Hv_inv_0 = 1.0 / Hv_diag[0]
        Hv_inv_N = 1.0 / Hv_diag[N]
        sat_0 = -0.5 * Hv_inv_0 * (jnp.roll(extrap_r, 1, axis=0) - extrap_l)
        sat_N = -0.5 * Hv_inv_N * (extrap_r - jnp.roll(extrap_l, -1, axis=0))
        div_x = div_x_raw.at[:, 0, :].add(sat_0).at[:, -1, :].add(sat_N)

        div_x_sum = jnp.sum(h_x * Wh[None,:,:] * div_x)
        x_err = float(jnp.abs(grad_x_sum + div_x_sum))

        # --- y-direction: project h in y only, use SAME h for both terms ---
        avg_y = 0.5 * (h[:, :, 0] + h[:, :, -1])
        h_y = h.at[:, :, 0].set(avg_y).at[:, :, -1].set(avg_y)

        # Gradient: h_y @ Dvc^T
        grad_y = jnp.einsum('pij,kj->pik', h_y, Dvc)
        grad_y_sum = jnp.sum(v2 * W2[None,:,:] * grad_y)

        # Divergence: v2 @ DS_cv_y^T, weighted by SAME h_y
        div_y = jnp.einsum('pij,kj->pik', v2, DS_cv_y)
        div_y_sum = jnp.sum(h_y * Wh[None,:,:] * div_y)
        y_err = float(jnp.abs(grad_y_sum + div_y_sum))

        err = max(x_err, y_err)
        ok = err < 1e-9
        passed = passed and ok
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  N = {N:4d}:  x: {x_err:.2e}  y: {y_err:.2e}  {status}")

    return passed


# ============================================================
# Test 2: Plane wave convergence (x-direction, across panels)
# ============================================================

def test_convergence_x():
    """
    Rightward-traveling plane wave across all 6 panels.

    Global domain: [0, 6L] × [0, L], periodic.
    k = 2π·n_waves / (6L),  ω = c·k
    After one period T, solution returns to IC.
    """
    print("\n" + "=" * 65)
    print("TEST 2: Plane Wave Convergence (x, across 6 panels)")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L_panel = 1.0
    L_global = 6 * L_panel
    c = np.sqrt(g * H0)
    n_waves = 3  # 3 wavelengths in 6L → wavelength = 2L = 2 panels
    k = 2 * np.pi * n_waves / L_global
    omega = c * k
    T_period = 2 * np.pi / omega
    A_wave = 0.1
    CFL = 0.3

    print(f"  6 panels × L={L_panel}, global L={L_global}")
    print(f"  c = {c:.2f}, λ = {L_global/n_waves:.3f}, T = {T_period:.4f}")
    print()
    print(f"  {'N':>6} {'steps':>7} {'l2(h)':>12} {'rate':>8} {'mass_err':>10}")
    print("  " + "-" * 50)

    results = []

    for N in [8, 12, 16, 24, 32, 48]:
        sys_d = make_6panel_swe_system(N, L_panel, H0, g)
        step_fn = make_rk4_step(sys_d['rhs'])
        Wh = sys_d['Wh']
        dx = L_panel / N

        Xh, Yh, Xv1, Yv1, Xv2, Yv2 = make_grids(N, L_panel)

        # IC: rightward-traveling plane wave (h = f(x), v2 = 0)
        h0 = A_wave * jnp.cos(k * Xh)
        v1_0 = A_wave * np.sqrt(g / H0) * jnp.cos(k * Xv1)
        v2_0 = jnp.zeros((6, N + 1, N))

        mass0 = compute_mass(h0, Wh)

        dt = CFL * dx / c
        nsteps = int(np.ceil(T_period / dt))
        dt = T_period / nsteps

        h, v1, v2 = h0, v1_0, v2_0
        for _ in range(nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)

        mass_err = abs(compute_mass(h, Wh) - mass0)
        h_err = float(jnp.sqrt(jnp.sum((h - h0)**2 * Wh[None,:,:]) /
                                jnp.sum(h0**2 * Wh[None,:,:])))

        if results:
            rate = np.log2(results[-1]['h_err'] / h_err) / \
                   np.log2(N / results[-1]['N'])
            rate_str = f"{rate:8.2f}"
        else:
            rate_str = "     ---"

        results.append({'N': N, 'h_err': h_err})
        print(f"  {N:6d} {nsteps:7d} {h_err:12.4e} {rate_str} {mass_err:10.2e}")

    final_rate = np.log2(results[-2]['h_err'] / results[-1]['h_err']) / \
                 np.log2(results[-1]['N'] / results[-2]['N'])
    print(f"\n  Final rate: {final_rate:.2f}")
    passed = final_rate > 2.8
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 3: Diagonal plane wave (both directions)
# ============================================================

def test_convergence_diag():
    """
    Diagonal wave: exercises both x (inter-panel) and y (intra-panel).
    """
    print("\n" + "=" * 65)
    print("TEST 3: Diagonal Wave (x across panels + y within panels)")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L_panel = 1.0
    L_global = 6 * L_panel
    c = np.sqrt(g * H0)
    # x: 3 waves in 6L, y: 2 waves in L
    kx = 2 * np.pi * 3 / L_global
    ky = 2 * np.pi * 2 / L_panel
    k_mag = np.sqrt(kx**2 + ky**2)
    omega = c * k_mag
    T_period = 2 * np.pi / omega
    A_wave = 0.1
    CFL = 0.2

    print(f"  kx = {kx:.2f}, ky = {ky:.2f}, |k| = {k_mag:.2f}")
    print(f"  T = {T_period:.4f}, CFL = {CFL}")
    print()
    print(f"  {'N':>6} {'steps':>7} {'l2(h)':>12} {'rate':>8}")
    print("  " + "-" * 40)

    results = []

    for N in [8, 12, 16, 24, 32, 48]:
        sys_d = make_6panel_swe_system(N, L_panel, H0, g)
        step_fn = make_rk4_step(sys_d['rhs'])
        Wh = sys_d['Wh']
        dx = L_panel / N

        Xh, Yh, Xv1, Yv1, Xv2, Yv2 = make_grids(N, L_panel)

        phase_h = kx * Xh + ky * Yh
        phase_v1 = kx * Xv1 + ky * Yv1
        phase_v2 = kx * Xv2 + ky * Yv2

        h0 = A_wave * jnp.cos(phase_h)
        v1_0 = A_wave * np.sqrt(g / H0) * (kx / k_mag) * jnp.cos(phase_v1)
        v2_0 = A_wave * np.sqrt(g / H0) * (ky / k_mag) * jnp.cos(phase_v2)

        dt = CFL * dx / c
        nsteps = int(np.ceil(T_period / dt))
        dt = T_period / nsteps

        h, v1, v2 = h0, v1_0, v2_0
        for _ in range(nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)

        h_err = float(jnp.sqrt(jnp.sum((h - h0)**2 * Wh[None,:,:]) /
                                jnp.sum(h0**2 * Wh[None,:,:])))

        if results:
            rate = np.log2(results[-1]['h_err'] / h_err) / \
                   np.log2(N / results[-1]['N'])
            rate_str = f"{rate:8.2f}"
        else:
            rate_str = "     ---"

        results.append({'N': N, 'h_err': h_err})
        print(f"  {N:6d} {nsteps:7d} {h_err:12.4e} {rate_str}")

    final_rate = np.log2(results[-2]['h_err'] / results[-1]['h_err']) / \
                 np.log2(results[-1]['N'] / results[-2]['N'])
    print(f"\n  Final rate: {final_rate:.2f}")
    passed = final_rate > 2.8
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 4: Mass conservation
# ============================================================

def test_mass():
    """Mass conservation over 10 wave periods."""
    print("\n" + "=" * 65)
    print("TEST 4: Mass Conservation (10 wave periods)")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L_panel = 1.0; N = 16
    c = np.sqrt(g * H0)
    k = 2 * np.pi * 3 / (6 * L_panel)
    T_period = 2 * np.pi / (c * k)
    A_wave = 0.1; CFL = 0.3; dx = L_panel / N

    sys_d = make_6panel_swe_system(N, L_panel, H0, g)
    step_fn = make_rk4_step(sys_d['rhs'])
    Wh = sys_d['Wh']

    Xh, _, Xv1, _, _, _ = make_grids(N, L_panel)
    h = A_wave * jnp.cos(k * Xh)
    v1 = A_wave * np.sqrt(g / H0) * jnp.cos(k * Xv1)
    v2 = jnp.zeros((6, N + 1, N))

    mass0 = compute_mass(h, Wh)
    dt = CFL * dx / c
    T_end = 10 * T_period
    nsteps = int(np.ceil(T_end / dt))
    dt = T_end / nsteps

    max_merr = 0.0
    for s in range(nsteps):
        h, v1, v2 = step_fn(h, v1, v2, dt)
        if (s + 1) % (nsteps // 5) == 0:
            merr = abs(compute_mass(h, Wh) - mass0)
            max_merr = max(max_merr, merr)
            print(f"  Step {s+1:6d}/{nsteps}: mass_err = {merr:.2e}")

    passed = max_merr < 1e-11
    print(f"\n  Max mass error: {max_merr:.2e}")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 5: Energy CFL sweep
# ============================================================

def test_energy():
    """Verify energy error scales with dt (spatial energy-exact)."""
    print("\n" + "=" * 65)
    print("TEST 5: Energy Conservation — CFL sweep")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L_panel = 1.0; N = 12
    c = np.sqrt(g * H0)
    k = 2 * np.pi * 3 / (6 * L_panel)
    T_period = 2 * np.pi / (c * k)
    A_wave = 0.1; dx = L_panel / N

    sys_d = make_6panel_swe_system(N, L_panel, H0, g)
    step_fn = make_rk4_step(sys_d['rhs'])
    Wh = sys_d['Wh']; W1 = sys_d['W1']; W2 = sys_d['W2']

    Xh, _, Xv1, _, _, _ = make_grids(N, L_panel)
    h0 = A_wave * jnp.cos(k * Xh)
    v1_0 = A_wave * np.sqrt(g / H0) * jnp.cos(k * Xv1)
    v2_0 = jnp.zeros((6, N + 1, N))
    E0 = compute_energy(h0, v1_0, v2_0, Wh, W1, W2, g, H0)
    T_end = 10 * T_period

    CFLs = [0.4, 0.2, 0.1, 0.05]
    results = []

    print(f"  N = {N}, T = {T_end:.4f}")
    print(f"\n  {'CFL':>6} {'dt':>12} {'steps':>7} {'ΔE/E':>12} {'rate':>8}")
    print("  " + "-" * 50)

    for CFL in CFLs:
        dt = CFL * dx / c
        nsteps = int(np.ceil(T_end / dt))
        dt = T_end / nsteps

        h, v1, v2 = h0.copy(), v1_0.copy(), v2_0.copy()
        for _ in range(nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)

        E_f = compute_energy(h, v1, v2, Wh, W1, W2, g, H0)
        rel = abs(E_f - E0) / E0

        if results:
            rate = np.log2(results[-1]['err'] / rel) / \
                   np.log2(results[-1]['dt'] / dt)
            rate_str = f"{rate:8.2f}"
        else:
            rate_str = "     ---"

        results.append({'CFL': CFL, 'dt': dt, 'err': rel})
        print(f"  {CFL:6.3f} {dt:12.6e} {nsteps:7d} {rel:12.4e} {rate_str}")

    final_rate = np.log2(results[-2]['err'] / results[-1]['err']) / \
                 np.log2(results[-2]['dt'] / results[-1]['dt'])
    print(f"\n  dt-scaling rate: {final_rate:.2f}")
    passed = final_rate > 3.5
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 6: Gaussian crossing panel boundary
# ============================================================

def test_gaussian_crossing():
    """
    Gaussian pulse launched from near a panel boundary.
    Should cross smoothly without reflection or instability.

    The pulse starts in panel 2 near its right boundary (x = 2.8L)
    and propagates rightward into panel 3.
    """
    print("\n" + "=" * 65)
    print("TEST 6: Gaussian Pulse Crossing Panel Boundary")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L_panel = 1.0; N = 32
    c = np.sqrt(g * H0)
    dx = L_panel / N; CFL = 0.3
    sigma = 0.08

    sys_d = make_6panel_swe_system(N, L_panel, H0, g)
    step_fn = make_rk4_step(sys_d['rhs'])
    Wh = sys_d['Wh']; W1 = sys_d['W1']; W2 = sys_d['W2']

    Xh, Yh, _, _, _, _ = make_grids(N, L_panel)

    # Pulse center: near right boundary of panel 2 (x=2.8, y=0.5)
    x0 = 2.8 * L_panel
    y0 = 0.5 * L_panel
    h0 = 0.1 * jnp.exp(-((Xh - x0)**2 + (Yh - y0)**2) / (2 * sigma**2))
    v1_0 = jnp.zeros((6, N, N + 1))
    v2_0 = jnp.zeros((6, N + 1, N))

    mass0 = compute_mass(h0, Wh)
    E0 = compute_energy(h0, v1_0, v2_0, Wh, W1, W2, g, H0)
    max0 = float(jnp.max(h0))

    # Run long enough for wavefront to cross 2+ panel boundaries
    T_end = 2.5 * L_panel / c
    dt = CFL * dx / c
    nsteps = int(np.ceil(T_end / dt))
    dt = T_end / nsteps

    print(f"  N = {N}, σ = {sigma}")
    print(f"  Pulse at x = {x0:.1f} (panel 2, near right boundary)")
    print(f"  T = {T_end:.2f}, wavefront travel ≈ {c*T_end:.1f}L")
    print(f"  Steps = {nsteps}")

    h, v1, v2 = h0, v1_0, v2_0
    for s in range(nsteps):
        h, v1, v2 = step_fn(h, v1, v2, dt)

    mass_f = compute_mass(h, Wh)
    E_f = compute_energy(h, v1, v2, Wh, W1, W2, g, H0)
    mass_err = abs(mass_f - mass0) / (abs(mass0) + 1e-30)
    e_err = abs(E_f - E0) / E0

    max_f = float(jnp.max(jnp.abs(h)))
    stable = max_f < 10 * max0

    # Check waves spread across multiple panels
    panel_max = [float(jnp.max(jnp.abs(h[p]))) for p in range(6)]
    n_active_panels = sum(1 for m in panel_max if m > 1e-6)

    print(f"\n  max|h|:     {max_f:.6e} (initial: {max0:.6e})")
    print(f"  mass error: {mass_err:.2e}")
    print(f"  ΔE/E:       {e_err:.2e}")
    print(f"  Stable:     {'yes' if stable else 'NO'}")
    print(f"  Active panels: {n_active_panels}/6")
    print(f"  Panel maxima: {['%.2e' % m for m in panel_max]}")

    passed = stable and n_active_panels >= 3 and mass_err < 1e-11
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  STEP 3: 6-Panel Ring — Multi-Block SAT-Projection")
    print("  Flat Geometry, SBP 4/2")
    print("  Reference: Shashkin 2025, Sections 2.2, 3, 4.2")
    print("=" * 65)

    results = {}
    results['sbp_multi']  = test_sbp_multi()
    results['conv_x']     = test_convergence_x()
    results['conv_diag']  = test_convergence_diag()
    results['mass']       = test_mass()
    results['energy']     = test_energy()
    results['gaussian']   = test_gaussian_crossing()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<15} {status}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("  All tests passed. Ready for Step 4 (cubed-sphere metrics).")
    else:
        print("  Some tests failed. Fix before proceeding.")
    print("=" * 65)
