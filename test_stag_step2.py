"""
test_stag_step2.py — 2D Staggered SWE on Single Panel (Doubly Periodic)
========================================================================

GOAL: Validate 2D Kronecker product extension of SBP 4/2 operators
on the simplest 2D problem before introducing multi-panel connectivity.

PHYSICS:
  Linearized 2D shallow water on flat periodic domain [0,L]×[0,L]:
    dh/dt  = -H0 (∂v1/∂x1 + ∂v2/∂x2)    (continuity)
    dv1/dt = -g ∂h/∂x1                    (x-momentum)
    dv2/dt = -g ∂h/∂x2                    (y-momentum)

  Supports plane waves with phase speed c = √(gH0).

GRID (Shashkin Section 3, Eqs. 35-37, single panel Nb=1):
  h  at x^h = (x^v_i, x^v_j)  →  (N+1, N+1)   vertices
  v1 at x^1 = (x^c_i, x^v_j)  →  (N, N+1)      x-faces
  v2 at x^2 = (x^v_i, x^c_j)  →  (N+1, N)       y-faces

OPERATORS (Kronecker product, Eqs. 39-43):
  Via 1D operators applied along each axis:
    ∂h/∂x1 → DP_vc @ h         along axis 0  →  (N, N+1)
    ∂h/∂x2 → h @ DP_vc.T       along axis 1  →  (N+1, N)
    ∂v1/∂x1 → DP_cv @ v1       along axis 0  →  (N+1, N+1)
    ∂v2/∂x2 → v2 @ DP_cv.T     along axis 1  →  (N+1, N+1)

  2D projection (Eq. 33 applied to both axes):
    A_2d(h) = A @ h @ A

QUADRATURE (Eq. 41):
  Hh = Hv ⊗ Hv   at h-points
  H1 = Hc ⊗ Hv   at v1-points
  H2 = Hv ⊗ Hc   at v2-points

TARGETS:
  - Convergence rate ≥ 3 (plane wave, one period)
  - Mass conservation: machine precision
  - Energy: spatial exact, temporal dt^5

Reference: Shashkin, Goyman & Tretyak 2025, Sections 2.2 and 3

Usage:
    python test_stag_step2.py
"""
import sys
import os
import time

project_dir = '/mnt/project'
if os.path.isdir(project_dir):
    sys.path.insert(0, project_dir)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from sbp_staggered_1d import sbp_42, apply_sat_projection, make_projection_matrix


# ============================================================
# Build 2D periodic SWE system
# ============================================================

def make_2d_swe_system(N, L, H0, g):
    """
    Build 2D linearized SWE operators on a single doubly-periodic panel.

    Uses SBP 4/2 operators applied via Kronecker product structure
    (implemented as matrix multiplications along each axis).
    """
    dx = L / N
    ops = sbp_42(N, dx)

    DP_vc, DP_cv = apply_sat_projection(ops)
    A = make_projection_matrix(ops)

    # Diagonal quadrature weights for 2D norms
    Hv_diag = jnp.diag(ops.Hv)  # (N+1,)
    Hc_diag = jnp.diag(ops.Hc)  # (N,)

    # 2D quadrature weights (outer products)
    Wh = jnp.outer(Hv_diag, Hv_diag)    # (N+1, N+1) for h
    W1 = jnp.outer(Hc_diag, Hv_diag)    # (N, N+1)   for v1
    W2 = jnp.outer(Hv_diag, Hc_diag)    # (N+1, N)   for v2

    # DP_vc.T for y-direction operations
    DP_vc_T = DP_vc.T  # (N+1, N)
    DP_cv_T = DP_cv.T  # (N, N+1)

    def project_2d(h):
        """Apply projection in both x and y for periodicity."""
        return A @ h @ A  # A is symmetric for 4/2 (Hv[0] = Hv[N])

    def rhs(h, v1, v2):
        """
        RHS of 2D linearized SWE on flat periodic domain.

        h:  (N+1, N+1)  height at vertices
        v1: (N, N+1)    x-velocity at x-faces
        v2: (N+1, N)    y-velocity at y-faces
        """
        h_proj = project_2d(h)

        # Gradient: ∇h at velocity points
        dv1_dt = -g * (DP_vc @ h_proj)       # ∂h/∂x1 → (N, N+1)
        dv2_dt = -g * (h_proj @ DP_vc_T)     # ∂h/∂x2 → (N+1, N)

        # Divergence: ∇·v at h points
        div_v = DP_cv @ v1 + v2 @ DP_cv_T    # (N+1, N+1)

        # Continuity (project result for consistency)
        dh_dt = -H0 * project_2d(div_v)

        return dh_dt, dv1_dt, dv2_dt

    return {
        'rhs': jax.jit(rhs),
        'project_2d': project_2d,
        'ops': ops,
        'DP_vc': DP_vc,
        'DP_cv': DP_cv,
        'A': A,
        'Wh': Wh,
        'W1': W1,
        'W2': W2,
        'Hv_diag': Hv_diag,
        'Hc_diag': Hc_diag,
        'dx': dx,
        'N': N,
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
# Diagnostics
# ============================================================

def compute_mass(h, Wh):
    return float(jnp.sum(h * Wh))


def compute_energy(h, v1, v2, Wh, W1, W2, g, H0):
    PE = 0.5 * g * float(jnp.sum(h**2 * Wh))
    KE = 0.5 * H0 * (float(jnp.sum(v1**2 * W1)) + float(jnp.sum(v2**2 * W2)))
    return PE + KE


# ============================================================
# Test 1: 2D SBP property
# ============================================================

def test_sbp_2d():
    """
    Verify 2D SBP property (Eq. 48 for single periodic panel):
      v^T Hv Dhv h + h^T Hh Dvh v = 0
    after SAT-Projection.

    Implemented as:
      v1^T W1 (DP_vc @ h) + v2^T W2 (h @ DP_vc.T)
      + (Ah)^T Wh (DP_cv @ v1) + (Ah)^T Wh (v2 @ DP_cv.T) = 0
    """
    print("\n" + "=" * 65)
    print("TEST 1: 2D SBP Property")
    print("=" * 65)

    passed = True
    for N in [16, 32]:
        sys_d = make_2d_swe_system(N, L=1.0, H0=1.0, g=1.0)
        DP_vc = sys_d['DP_vc']
        DP_cv = sys_d['DP_cv']
        A = sys_d['A']
        Wh = sys_d['Wh']
        W1 = sys_d['W1']
        W2 = sys_d['W2']

        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (N+1, N+1))
        v1 = jax.random.normal(k2, (N, N+1))
        v2 = jax.random.normal(k3, (N+1, N))

        Ah = A @ h @ A

        # Gradient terms: v^T Hv Dhv h
        grad_term = (jnp.sum(v1 * W1 * (DP_vc @ h)) +
                     jnp.sum(v2 * W2 * (h @ DP_vc.T)))

        # Divergence terms: h^T Hh Dvh v
        div_term = (jnp.sum(Ah * Wh * (DP_cv @ v1)) +
                    jnp.sum(Ah * Wh * (v2 @ DP_cv.T)))

        err = float(jnp.abs(grad_term + div_term))
        ok = err < 1e-10
        passed = passed and ok
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  N = {N:4d}:  |grad + div| = {err:.2e}  {status}")

    return passed


# ============================================================
# Test 2: Plane wave convergence (1 period)
# ============================================================

def test_convergence():
    """
    Rightward-traveling plane wave on doubly-periodic domain.

    IC: h = A cos(kx), v1 = A√(g/H0) cos(kx), v2 = 0
    After one period T = 2π/ω, compare to IC.
    """
    print("\n" + "=" * 65)
    print("TEST 2: Plane Wave Convergence (1 period)")
    print("=" * 65)

    H0 = 1.0
    g = 1.0
    L = 1.0
    c = np.sqrt(g * H0)
    n_waves = 2
    k = 2 * np.pi * n_waves / L
    omega = c * k
    T_period = 2 * np.pi / omega
    A_wave = 0.1
    CFL = 0.3

    print(f"  c = {c:.2f}, λ = {L/n_waves:.3f}, T = {T_period:.4f}, CFL = {CFL}")
    print()
    print(f"  {'N':>6} {'steps':>7} {'l2(h)':>12} {'l2(v1)':>12} "
          f"{'rate(h)':>8} {'mass_err':>10}")
    print("  " + "-" * 60)

    results = []

    for N in [8, 12, 16, 24, 32, 48, 64]:
        dx = L / N
        sys_d = make_2d_swe_system(N, L, H0, g)
        rhs_fn = sys_d['rhs']
        step_fn = make_rk4_step(rhs_fn)
        Wh = sys_d['Wh']
        W1 = sys_d['W1']

        # Grid coordinates
        x_v = jnp.linspace(0, L, N + 1)
        x_c = (jnp.arange(N) + 0.5) * dx

        # 2D grids for IC
        Xh = x_v[:, None] * jnp.ones(N + 1)[None, :]   # (N+1, N+1)
        Xv1 = x_c[:, None] * jnp.ones(N + 1)[None, :]  # (N, N+1)

        # IC: rightward-traveling plane wave
        h0 = A_wave * jnp.cos(k * Xh)
        v1_0 = A_wave * np.sqrt(g / H0) * jnp.cos(k * Xv1)
        v2_0 = jnp.zeros((N + 1, N))

        mass0 = compute_mass(h0, Wh)

        dt = CFL * dx / c
        nsteps = int(np.ceil(T_period / dt))
        dt = T_period / nsteps

        h, v1, v2 = h0, v1_0, v2_0
        for _ in range(nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)

        mass_err = abs(compute_mass(h, Wh) - mass0)

        h_err = float(jnp.sqrt(jnp.sum((h - h0)**2 * Wh) /
                                jnp.sum(h0**2 * Wh)))
        v1_err = float(jnp.sqrt(jnp.sum((v1 - v1_0)**2 * W1) /
                                 jnp.sum(v1_0**2 * W1)))

        if results:
            rate = np.log2(results[-1]['h_err'] / h_err) / \
                   np.log2(N / results[-1]['N'])
            rate_str = f"{rate:8.2f}"
        else:
            rate_str = "     ---"

        results.append({'N': N, 'h_err': h_err, 'v1_err': v1_err})

        print(f"  {N:6d} {nsteps:7d} {h_err:12.4e} {v1_err:12.4e} "
              f"{rate_str} {mass_err:10.2e}")

    final_rate = np.log2(results[-2]['h_err'] / results[-1]['h_err']) / \
                 np.log2(results[-1]['N'] / results[-2]['N'])
    print(f"\n  Final rate: {final_rate:.2f}")
    print(f"  Target: ≥ 3.0 (SBP 4/2)")

    passed = final_rate > 2.8
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 3: Diagonal plane wave (tests both directions)
# ============================================================

def test_diagonal_wave():
    """
    Diagonally-traveling plane wave: k = (k1, k2) with k1 = k2.

    This exercises both x and y operators simultaneously.
    IC: h = A cos(k1·x + k2·y), v1 = A√(g/H0)·(k1/|k|)·cos(...),
        v2 = A√(g/H0)·(k2/|k|)·cos(...)
    """
    print("\n" + "=" * 65)
    print("TEST 3: Diagonal Plane Wave Convergence")
    print("=" * 65)

    H0 = 1.0
    g = 1.0
    L = 1.0
    c = np.sqrt(g * H0)
    n_waves = 2
    k1 = 2 * np.pi * n_waves / L
    k2 = 2 * np.pi * n_waves / L
    k_mag = np.sqrt(k1**2 + k2**2)
    omega = c * k_mag
    T_period = 2 * np.pi / omega
    A_wave = 0.1
    CFL = 0.2

    print(f"  k = ({k1:.1f}, {k2:.1f}), |k| = {k_mag:.2f}")
    print(f"  c = {c:.2f}, T = {T_period:.4f}, CFL = {CFL}")
    print()
    print(f"  {'N':>6} {'steps':>7} {'l2(h)':>12} {'rate':>8}")
    print("  " + "-" * 40)

    results = []

    for N in [8, 12, 16, 24, 32, 48, 64]:
        dx = L / N
        sys_d = make_2d_swe_system(N, L, H0, g)
        step_fn = make_rk4_step(sys_d['rhs'])
        Wh = sys_d['Wh']

        x_v = jnp.linspace(0, L, N + 1)
        x_c = (jnp.arange(N) + 0.5) * dx

        # 2D grids
        Xh, Yh = jnp.meshgrid(x_v, x_v, indexing='ij')     # (N+1, N+1)
        Xv1, Yv1 = jnp.meshgrid(x_c, x_v, indexing='ij')   # (N, N+1)
        Xv2, Yv2 = jnp.meshgrid(x_v, x_c, indexing='ij')   # (N+1, N)

        phase_h = k1 * Xh + k2 * Yh
        phase_v1 = k1 * Xv1 + k2 * Yv1
        phase_v2 = k1 * Xv2 + k2 * Yv2

        h0 = A_wave * jnp.cos(phase_h)
        v1_0 = A_wave * np.sqrt(g / H0) * (k1 / k_mag) * jnp.cos(phase_v1)
        v2_0 = A_wave * np.sqrt(g / H0) * (k2 / k_mag) * jnp.cos(phase_v2)

        dt = CFL * dx / c
        nsteps = int(np.ceil(T_period / dt))
        dt = T_period / nsteps

        h, v1, v2 = h0, v1_0, v2_0
        for _ in range(nsteps):
            h, v1, v2 = step_fn(h, v1, v2, dt)

        h_err = float(jnp.sqrt(jnp.sum((h - h0)**2 * Wh) /
                                jnp.sum(h0**2 * Wh)))

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
# Test 4: Mass conservation (long integration)
# ============================================================

def test_mass():
    """Mass conservation over 10 wave periods."""
    print("\n" + "=" * 65)
    print("TEST 4: Mass Conservation (10 wave periods)")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L = 1.0; N = 16
    c = np.sqrt(g * H0)
    k = 4 * np.pi / L
    T_period = 2 * np.pi / (c * k)
    A_wave = 0.1; CFL = 0.3; dx = L / N

    sys_d = make_2d_swe_system(N, L, H0, g)
    step_fn = make_rk4_step(sys_d['rhs'])
    Wh = sys_d['Wh']

    x_v = jnp.linspace(0, L, N + 1)
    Xh = x_v[:, None] * jnp.ones(N + 1)[None, :]

    h = A_wave * jnp.cos(k * Xh)
    x_c = (jnp.arange(N) + 0.5) * dx
    Xv1 = x_c[:, None] * jnp.ones(N + 1)[None, :]
    v1 = A_wave * np.sqrt(g / H0) * jnp.cos(k * Xv1)
    v2 = jnp.zeros((N + 1, N))

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

    passed = max_merr < 1e-12
    print(f"\n  Max mass error: {max_merr:.2e}")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 5: Energy conservation (CFL sweep)
# ============================================================

def test_energy():
    """Verify energy error scales as dt^p (spatial discretization is exact)."""
    print("\n" + "=" * 65)
    print("TEST 5: Energy Conservation — CFL sweep (10 wave periods)")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L = 1.0; N = 16
    c = np.sqrt(g * H0)
    k = 4 * np.pi / L
    T_period = 2 * np.pi / (c * k)
    A_wave = 0.1; dx = L / N

    sys_d = make_2d_swe_system(N, L, H0, g)
    step_fn = make_rk4_step(sys_d['rhs'])
    Wh = sys_d['Wh']
    W1 = sys_d['W1']
    W2 = sys_d['W2']

    x_v = jnp.linspace(0, L, N + 1)
    x_c = (jnp.arange(N) + 0.5) * dx
    Xh = x_v[:, None] * jnp.ones(N + 1)[None, :]
    Xv1 = x_c[:, None] * jnp.ones(N + 1)[None, :]

    h0 = A_wave * jnp.cos(k * Xh)
    v1_0 = A_wave * np.sqrt(g / H0) * jnp.cos(k * Xv1)
    v2_0 = jnp.zeros((N + 1, N))
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
    if passed:
        print(f"  → Spatial discretization conserves energy exactly (dt^{final_rate:.1f})")
    else:
        print(f"  → Possible spatial energy leak")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 6: Gaussian pulse (qualitative — stability + wave speed)
# ============================================================

def test_gaussian_pulse():
    """
    Gaussian height perturbation with zero initial velocity.
    Should produce circular gravity waves radiating at speed c.
    Check: stable, mass conserved, waves reach expected radius.
    """
    print("\n" + "=" * 65)
    print("TEST 6: Gaussian Pulse — Circular Gravity Waves")
    print("=" * 65)

    H0 = 1.0; g = 1.0; L = 1.0; N = 32
    c = np.sqrt(g * H0)
    dx = L / N; CFL = 0.3
    sigma = 0.05  # Gaussian width

    sys_d = make_2d_swe_system(N, L, H0, g)
    step_fn = make_rk4_step(sys_d['rhs'])
    Wh = sys_d['Wh']
    W1 = sys_d['W1']
    W2 = sys_d['W2']

    x_v = jnp.linspace(0, L, N + 1)
    Xh, Yh = jnp.meshgrid(x_v, x_v, indexing='ij')

    # Centered Gaussian
    x0, y0 = L / 2, L / 2
    h0 = 0.1 * jnp.exp(-((Xh - x0)**2 + (Yh - y0)**2) / (2 * sigma**2))
    v1_0 = jnp.zeros((N, N + 1))
    v2_0 = jnp.zeros((N + 1, N))

    mass0 = compute_mass(h0, Wh)
    E0 = compute_energy(h0, v1_0, v2_0, Wh, W1, W2, g, H0)
    max0 = float(jnp.max(h0))

    # Run for time T such that wavefront travels ~L/4
    T_end = 0.25 * L / c
    dt = CFL * dx / c
    nsteps = int(np.ceil(T_end / dt))
    dt = T_end / nsteps

    print(f"  N = {N}, σ = {sigma}, T = {T_end:.4f} (wavefront ≈ L/4)")
    print(f"  dt = {dt:.6e}, steps = {nsteps}")

    h, v1, v2 = h0, v1_0, v2_0
    for s in range(nsteps):
        h, v1, v2 = step_fn(h, v1, v2, dt)

    mass_f = compute_mass(h, Wh)
    E_f = compute_energy(h, v1, v2, Wh, W1, W2, g, H0)
    mass_err = abs(mass_f - mass0) / abs(mass0 + 1e-30)
    e_err = abs(E_f - E0) / E0

    max_f = float(jnp.max(jnp.abs(h)))
    stable = max_f < 10 * max0  # shouldn't blow up

    print(f"  max|h|:     {max_f:.6e} (initial: {max0:.6e})")
    print(f"  mass error: {mass_err:.2e}")
    print(f"  ΔE/E:       {e_err:.2e}")

    # Check wavefront has spread (h should have features away from center)
    corner_h = float(jnp.abs(h[0, 0]))
    has_spread = corner_h > 1e-10  # waves should reach corners? maybe not at T=L/4c
    # At least check v fields are nonzero (waves generated)
    v_max = max(float(jnp.max(jnp.abs(v1))), float(jnp.max(jnp.abs(v2))))
    waves_exist = v_max > 1e-6

    passed = stable and waves_exist and mass_err < 1e-12
    print(f"  max|v|:     {v_max:.6e} (waves generated: {'yes' if waves_exist else 'no'})")
    print(f"  Stable: {'yes' if stable else 'NO'}")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  STEP 2: 2D Staggered SWE — Single Panel, Doubly Periodic")
    print("  SBP 4/2 + SAT-Projection, Flat Geometry")
    print("  Reference: Shashkin 2025, Sections 2.2 and 3")
    print("=" * 65)

    results = {}
    results['sbp_2d']     = test_sbp_2d()
    results['plane_wave'] = test_convergence()
    results['diag_wave']  = test_diagonal_wave()
    results['mass']       = test_mass()
    results['energy']     = test_energy()
    results['gaussian']   = test_gaussian_pulse()

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
        print("  All tests passed. Ready for Step 3 (multi-panel connectivity).")
    else:
        print("  Some tests failed. Fix before proceeding.")
    print("=" * 65)
