"""
test_stag_step1.py — 1D Staggered SWE with SBP 4/2 + SAT-Projection
=====================================================================

GOAL: Validate SBP 4/2 operators with SAT-Projection on the simplest
possible problem before building up to the cubed sphere.

PHYSICS:
  Linearized 1D shallow water on periodic domain [0, L]:
    dh/dt = -H0 * du/dx       (continuity)
    du/dt = -g  * dh/dx       (momentum)

  Exact solution: plane wave h = A*cos(kx - ωt), u = A*√(g/H0)*cos(kx - ωt)
  with ω = k*√(gH0). After one period T = 2π/ω, solution returns to IC.

DISCRETIZATION (Shashkin's staggered grid):
  h at vertices:  N+1 points (x^v), shared at periodic boundary
  u at centers:   N points (x^c)

  Operators from sbp_staggered_1d.sbp_42:
    Dvc: gradient   (vertices → centers)  [N × (N+1)]
    Dcv: divergence (centers → vertices)  [(N+1) × N]
    Hv, Hc: quadrature norms
    l, r: boundary extrapolation

  SAT-Projection (Eqs. 26-27, 32-34):
    DP_cv = A @ DS_cv       (projected divergence)
    DP_vc = Dvc @ A         (gradient of projected h)
    A = projection matrix enforcing h[0] = h[N] (periodicity)

TARGETS:
  - Convergence rate ≥ 3 (theory: s+1 = 3 for 2s/s = 4/2 SBP)
  - Mass conservation: machine precision
  - Energy conservation: time-stepping order only (RK4)
  - SBP property (Eq. 34): machine precision

Reference: Shashkin, Goyman & Tretyak 2025, Section 2.2

Usage:
    python test_stag_step1.py
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
# Build 1D periodic SWE system
# ============================================================

def make_1d_swe_system(N, L, H0, g, order=4):
    """
    Build the 1D linearized SWE operators with SAT-Projection.

    Returns a dict with all operators and a JIT-compiled RHS function.
    """
    dx = L / N

    if order == 4:
        ops = sbp_42(N, dx)
    else:
        raise ValueError(f"Only order=4 implemented, got {order}")

    # SAT-Projection operators (Eqs. 26-27, 32)
    DP_vc, DP_cv = apply_sat_projection(ops)
    A = make_projection_matrix(ops)

    # Extract diagonal quadrature weights for efficiency
    Hv_diag = jnp.diag(ops.Hv)
    Hc_diag = jnp.diag(ops.Hc)

    def rhs(h, u):
        """
        RHS of 1D linearized SWE.

        h: (N+1,) height at vertices
        u: (N,)   velocity at centers

        Returns: (dh_dt, du_dt)
        """
        # Project h for interface continuity
        h_proj = A @ h

        # Momentum: du/dt = -g * DP_vc @ h
        du_dt = -g * (DP_vc @ h_proj)

        # Continuity: dh/dt = -H0 * A @ (DP_cv @ u)
        # (project the divergence result for consistency)
        dh_dt = -H0 * (A @ (DP_cv @ u))

        return dh_dt, du_dt

    return {
        'rhs': jax.jit(rhs),
        'ops': ops,
        'DP_vc': DP_vc,
        'DP_cv': DP_cv,
        'A': A,
        'Hv_diag': Hv_diag,
        'Hc_diag': Hc_diag,
        'dx': dx,
        'N': N,
    }


# ============================================================
# RK4 time stepper
# ============================================================

def make_rk4_step(rhs_fn):
    """Build a JIT-compiled RK4 step for (h, u) system."""
    @jax.jit
    def step(h, u, dt):
        k1h, k1u = rhs_fn(h, u)
        k2h, k2u = rhs_fn(h + 0.5*dt*k1h, u + 0.5*dt*k1u)
        k3h, k3u = rhs_fn(h + 0.5*dt*k2h, u + 0.5*dt*k2u)
        k4h, k4u = rhs_fn(h + dt*k3h, u + dt*k3u)
        h_new = h + (dt/6) * (k1h + 2*k2h + 2*k3h + k4h)
        u_new = u + (dt/6) * (k1u + 2*k2u + 2*k3u + k4u)
        return h_new, u_new
    return step


# ============================================================
# Diagnostics
# ============================================================

def compute_mass(h, Hv_diag):
    """∫ h dx using SBP quadrature."""
    return float(jnp.sum(h * Hv_diag))


def compute_energy(h, u, Hv_diag, Hc_diag, g, H0):
    """Total energy = (g/2)∫h² + (H0/2)∫u²."""
    PE = 0.5 * g * float(jnp.sum(h**2 * Hv_diag))
    KE = 0.5 * H0 * float(jnp.sum(u**2 * Hc_diag))
    return PE + KE


def verify_sbp_property(sys_dict):
    """
    Verify Eq. 34: u^T Hc DP_vc h + (Ah)^T Hv DP_cv u = 0.
    """
    ops = sys_dict['ops']
    DP_vc = sys_dict['DP_vc']
    DP_cv = sys_dict['DP_cv']
    A = sys_dict['A']
    N = sys_dict['N']

    # Random test vectors
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    h = jax.random.normal(k1, (N + 1,))
    u = jax.random.normal(k2, (N,))

    Ah = A @ h
    term1 = u @ ops.Hc @ DP_vc @ h
    term2 = Ah @ ops.Hv @ DP_cv @ u

    return float(jnp.abs(term1 + term2))


# ============================================================
# Test 1: SBP property
# ============================================================

def test_sbp_property():
    """Verify SBP property holds for 4/2 operators with SAT-Projection."""
    print("\n" + "=" * 65)
    print("TEST 1: SBP Property (Eq. 34)")
    print("=" * 65)

    for N in [16, 32, 64]:
        sys_dict = make_1d_swe_system(N, L=1.0, H0=1.0, g=1.0)
        err = verify_sbp_property(sys_dict)
        status = "✓ PASS" if err < 1e-11 else "✗ FAIL"
        print(f"  N = {N:4d}:  |u^T Hc DP_vc h + (Ah)^T Hv DP_cv u| = {err:.2e}  {status}")

    return err < 1e-11


# ============================================================
# Test 2: Convergence (plane wave, one period)
# ============================================================

def test_convergence():
    """
    Plane wave convergence test.

    IC: h = A*cos(kx), u = A*sqrt(g/H0)*cos(kx)
    After one period, compare to IC.
    Target: convergence rate ≥ 3 for SBP 4/2.
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

    print(f"  c = √(gH0) = {c:.2f}, λ = {L/n_waves:.3f}, T = {T_period:.4f}")
    print(f"  CFL = {CFL}")
    print()
    print(f"  {'N':>6} {'dx':>10} {'steps':>7} {'l2(h)':>12} "
          f"{'l2(u)':>12} {'rate(h)':>8} {'mass_err':>10}")
    print("  " + "-" * 65)

    results = []

    for N in [16, 24, 32, 48, 64, 96, 128]:
        dx = L / N
        sys_dict = make_1d_swe_system(N, L, H0, g)
        rhs_fn = sys_dict['rhs']
        step_fn = make_rk4_step(rhs_fn)
        Hv_diag = sys_dict['Hv_diag']
        Hc_diag = sys_dict['Hc_diag']
        A_proj = sys_dict['A']

        # Grid coordinates
        x_v = jnp.linspace(0, L, N + 1)  # vertices
        x_c = (jnp.arange(N) + 0.5) * dx  # centers

        # IC
        h0 = A_wave * jnp.cos(k * x_v)
        u0 = A_wave * np.sqrt(g / H0) * jnp.cos(k * x_c)

        mass0 = compute_mass(h0, Hv_diag)

        # Time stepping
        dt = CFL * dx / c
        nsteps = int(np.ceil(T_period / dt))
        dt = T_period / nsteps

        h, u = h0, u0
        for _ in range(nsteps):
            h, u = step_fn(h, u, dt)

        mass_f = compute_mass(h, Hv_diag)
        mass_err = abs(mass_f - mass0)

        # L2 errors (relative, using SBP quadrature)
        h_err = float(jnp.sqrt(jnp.sum((h - h0)**2 * Hv_diag) /
                                jnp.sum(h0**2 * Hv_diag)))
        u_err = float(jnp.sqrt(jnp.sum((u - u0)**2 * Hc_diag) /
                                jnp.sum(u0**2 * Hc_diag)))

        if results:
            rate = np.log2(results[-1]['h_err'] / h_err) / np.log2(N / results[-1]['N'])
            rate_str = f"{rate:8.2f}"
        else:
            rate_str = "     ---"

        results.append({'N': N, 'dx': dx, 'h_err': h_err, 'u_err': u_err,
                        'mass_err': mass_err})

        print(f"  {N:6d} {dx:10.4f} {nsteps:7d} {h_err:12.4e} "
              f"{u_err:12.4e} {rate_str} {mass_err:10.2e}")

    # Check final convergence rate
    final_rate = np.log2(results[-2]['h_err'] / results[-1]['h_err']) / \
                 np.log2(results[-1]['N'] / results[-2]['N'])

    print()
    print(f"  Final rate: {final_rate:.2f}")
    print(f"  Target: ≥ 3.0 (theory: s+1 = 3 for SBP 4/2)")

    passed = final_rate > 2.8
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 3: Mass conservation
# ============================================================

def test_mass_conservation():
    """Mass should be conserved to machine precision."""
    print("\n" + "=" * 65)
    print("TEST 3: Mass Conservation (10 wave periods)")
    print("=" * 65)

    H0 = 1.0
    g = 1.0
    L = 1.0
    N = 32
    c = np.sqrt(g * H0)
    k = 4 * np.pi / L
    omega = c * k
    T_period = 2 * np.pi / omega
    A_wave = 0.1
    CFL = 0.3

    sys_dict = make_1d_swe_system(N, L, H0, g)
    step_fn = make_rk4_step(sys_dict['rhs'])
    Hv_diag = sys_dict['Hv_diag']

    dx = L / N
    x_v = jnp.linspace(0, L, N + 1)
    x_c = (jnp.arange(N) + 0.5) * dx

    h = A_wave * jnp.cos(k * x_v)
    u = A_wave * np.sqrt(g / H0) * jnp.cos(k * x_c)

    mass0 = compute_mass(h, Hv_diag)

    dt = CFL * dx / c
    T_end = 10 * T_period
    nsteps = int(np.ceil(T_end / dt))
    dt = T_end / nsteps

    max_mass_err = 0.0
    for s in range(nsteps):
        h, u = step_fn(h, u, dt)
        if (s + 1) % (nsteps // 5) == 0:
            mass = compute_mass(h, Hv_diag)
            merr = abs(mass - mass0)
            max_mass_err = max(max_mass_err, merr)
            print(f"  Step {s+1:6d}/{nsteps}: mass_err = {merr:.2e}")

    passed = max_mass_err < 1e-12
    print(f"\n  Max mass error: {max_mass_err:.2e}")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 4: Energy conservation
# ============================================================

def test_energy_conservation():
    """
    Energy conservation diagnostic: sweep CFL to determine if drift is
    spatial (constant) or temporal (scales as dt^p).

    If spatial discretization conserves energy exactly (SBP guarantee),
    then ΔE/E should scale as dt^p where p ≥ 4 for RK4.
    """
    print("\n" + "=" * 65)
    print("TEST 4: Energy Conservation — CFL sweep (10 wave periods)")
    print("=" * 65)

    H0 = 1.0
    g = 1.0
    L = 1.0
    N = 32
    c = np.sqrt(g * H0)
    k = 4 * np.pi / L
    T_period = 2 * np.pi / (c * k)
    A_wave = 0.1

    sys_dict = make_1d_swe_system(N, L, H0, g)
    step_fn = make_rk4_step(sys_dict['rhs'])
    Hv_diag = sys_dict['Hv_diag']
    Hc_diag = sys_dict['Hc_diag']

    dx = L / N
    x_v = jnp.linspace(0, L, N + 1)
    x_c = (jnp.arange(N) + 0.5) * dx

    h0 = A_wave * jnp.cos(k * x_v)
    u0 = A_wave * np.sqrt(g / H0) * jnp.cos(k * x_c)
    E0 = compute_energy(h0, u0, Hv_diag, Hc_diag, g, H0)

    T_end = 10 * T_period

    CFLs = [0.4, 0.2, 0.1, 0.05, 0.025]
    results = []

    print(f"  N = {N}, T = {T_end:.4f} (10 periods)")
    print()
    print(f"  {'CFL':>6} {'dt':>12} {'steps':>7} {'ΔE/E':>12} {'rate':>8}")
    print("  " + "-" * 50)

    for CFL in CFLs:
        dt = CFL * dx / c
        nsteps = int(np.ceil(T_end / dt))
        dt = T_end / nsteps

        h, u = h0.copy(), u0.copy()
        for _ in range(nsteps):
            h, u = step_fn(h, u, dt)

        E_final = compute_energy(h, u, Hv_diag, Hc_diag, g, H0)
        rel_change = abs(E_final - E0) / E0

        if results:
            rate = np.log2(results[-1]['err'] / rel_change) / \
                   np.log2(results[-1]['dt'] / dt)
            rate_str = f"{rate:8.2f}"
        else:
            rate_str = "     ---"

        results.append({'CFL': CFL, 'dt': dt, 'err': rel_change})
        print(f"  {CFL:6.3f} {dt:12.6e} {nsteps:7d} {rel_change:12.4e} {rate_str}")

    # Check: does it scale with dt?
    final_rate = np.log2(results[-2]['err'] / results[-1]['err']) / \
                 np.log2(results[-2]['dt'] / results[-1]['dt'])

    print()
    print(f"  Final dt-scaling rate: {final_rate:.2f}")
    if final_rate > 3.5:
        print(f"  → Spatial discretization conserves energy exactly.")
        print(f"  → Drift is RK4 time-stepping error (dt^{final_rate:.1f}).")
        passed = True
    elif final_rate < 0.5:
        print(f"  → Energy error does NOT scale with dt.")
        print(f"  → Spatial energy leak detected!")
        passed = False
    else:
        print(f"  → Intermediate scaling — check for mixed spatial/temporal error.")
        passed = final_rate > 2.5

    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================
# Test 5: Projection idempotency and orthogonality
# ============================================================

def test_projection_properties():
    """Verify A² = A and (Hv·A)ᵀ = Hv·A."""
    print("\n" + "=" * 65)
    print("TEST 5: Projection Matrix Properties")
    print("=" * 65)

    for N in [16, 32, 64]:
        sys_dict = make_1d_swe_system(N, L=1.0, H0=1.0, g=1.0)
        A = sys_dict['A']
        ops = sys_dict['ops']

        # Idempotent: A² = A
        A2 = A @ A
        idemp_err = float(jnp.max(jnp.abs(A2 - A)))

        # Hv-orthogonal: (Hv A)ᵀ = Hv A
        HvA = ops.Hv @ A
        orth_err = float(jnp.max(jnp.abs(HvA - HvA.T)))

        s1 = "✓" if idemp_err < 1e-14 else "✗"
        s2 = "✓" if orth_err < 1e-14 else "✗"
        print(f"  N = {N:4d}:  A²=A err = {idemp_err:.2e} {s1}  "
              f"Hv-orth err = {orth_err:.2e} {s2}")

    return idemp_err < 1e-14 and orth_err < 1e-14


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  STEP 1: 1D Staggered SWE — SBP 4/2 + SAT-Projection")
    print("  Reference: Shashkin, Goyman & Tretyak 2025, Section 2.2")
    print("=" * 65)

    results = {}
    results['sbp']        = test_sbp_property()
    results['projection'] = test_projection_properties()
    results['convergence'] = test_convergence()
    results['mass']       = test_mass_conservation()
    results['energy']     = test_energy_conservation()

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
        print("  All tests passed. Ready for Step 2 (2D single panel).")
    else:
        print("  Some tests failed. Fix before proceeding.")
    print("=" * 65)
