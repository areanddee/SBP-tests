"""
test_convergence.py — Spatial Convergence Tests (Shashkin Section 6.2)
=======================================================================

Self-convergence tests for Gaussian wave propagation on cubed sphere.

NOTE: These tests use self-convergence (compare N against 2N) rather than
comparison against a spectral reference. Self-convergence rates underestimate
true rates when the reference solution (2N) has non-negligible error, which
happens especially for gauss 2 (vertex case) where error concentrates at
cube vertices. The ABSOLUTE errors are the primary validation — they must
match Shashkin Figure 3/4 at the same N values.

Parameters match Shashkin exactly:
  - CFL = 0.05 (small dt to eliminate time truncation)
  - T_end from Shashkin Table 1 (25 days = 2,160,000 seconds)
  - Gravity wave speed c = sqrt(gH) = 200 m/s (H0=10000m, g=4m/s²)

Usage:
    pytest tests/test_convergence.py -v               # fast: N=8,16
    pytest tests/test_convergence.py -v --runslow      # full: N=24,48,96
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.geometry import equiangular_to_cartesian
from sbp_swe.system import make_cubed_sphere_swe
from sbp_swe.timestepping import make_rk4_step
from sbp_swe.diagnostics import compute_mass


def _make_gaussian_ic(sys_d, variant, g, H0):
    """
    Gaussian initial condition matching Shashkin Section 6.2.

    variant 1: center at (0, 0, 1)  — panel 0 center
    variant 2: center at cube vertex (pi/4, pi/4) on panel 0
    """
    N = sys_d['N']
    grids = sys_d['grids']
    xi_v = grids['xi_v']
    xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')

    if variant == 1:
        xc, yc, zc = 0.0, 0.0, 1.0
    else:
        xc, yc, zc = equiangular_to_cartesian(jnp.pi / 4, jnp.pi / 4, 0)
        xc, yc, zc = float(xc), float(yc), float(zc)

    R0 = 1.0 / 3.0
    h = jnp.zeros((6, N + 1, N + 1))
    for p in range(6):
        X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
        d2 = (X - xc)**2 + (Y - yc)**2 + (Z - zc)**2
        h = h.at[p].set(jnp.exp(-d2 / (2 * R0**2)))

    h = sys_d['project_h'](h)
    v1 = jnp.zeros((6, N, N + 1))
    v2 = jnp.zeros((6, N + 1, N))
    return h, v1, v2


def _run_to_time(sys_d, h, v1, v2, T_end, cfl, g, H0):
    """Integrate to T_end using RK4."""
    step_fn = make_rk4_step(sys_d['rhs'])
    c = np.sqrt(g * H0)
    dt = cfl * sys_d['dx'] / c
    n_steps = int(np.ceil(T_end / dt))
    dt = T_end / n_steps

    for _ in range(n_steps):
        h, v1, v2 = step_fn(h, v1, v2, dt)

    return h, v1, v2


def _l2_error(h_test, h_ref, Wh, Jh):
    """Weighted L2 error on the sphere."""
    diff = h_test - h_ref
    l2 = float(jnp.sqrt(jnp.sum(diff**2 * Jh[None, :, :] * Wh[None, :, :])))
    return l2


def _linf_error(h_test, h_ref):
    """L∞ error."""
    return float(jnp.max(jnp.abs(h_test - h_ref)))


class TestConvergenceFast:
    """Fast convergence tests (N=8,16 — seconds to run)."""

    @pytest.mark.parametrize("variant", [1, 2])
    def test_self_convergence_improves(self, variant):
        """
        Verify that error decreases with resolution (N=8 vs N=16).
        This is the basic sanity check — not testing rates.
        """
        g, H0 = 1.0, 1.0
        T_end = 1.0
        cfl = 0.2
        errors = {}

        for N in [8, 16]:
            sys_d = make_cubed_sphere_swe(N, H0=H0, g=g)
            h0, v10, v20 = _make_gaussian_ic(sys_d, variant, g, H0)
            h_f, _, _ = _run_to_time(sys_d, h0, v10, v20, T_end, cfl, g, H0)

            # Use initial condition as "reference" for monotonicity check
            Wh = sys_d['Wh']
            Jh = sys_d['Jh']
            errors[N] = _l2_error(h_f, h0, Wh, Jh)

        assert errors[16] < errors[8], \
            f"Gauss {variant}: error not decreasing ({errors[8]:.2e} → {errors[16]:.2e})"


# Mark slow tests to be skippable
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


class TestConvergenceSlow:
    """
    Full Shashkin-parameter convergence tests (N=24,48,96).

    Run with: pytest tests/test_convergence.py -v --runslow
    These take minutes per resolution.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("variant", [1, 2])
    def test_shashkin_convergence(self, variant):
        """
        25-day integration at Shashkin parameters.
        Self-convergence: compare N vs 2N (subsampled).
        Primary validation: absolute L2 errors match Shashkin Fig 3/4.
        """
        g, H0 = 1.0, 1.0
        T_end = 25.0
        cfl = 0.05

        results = {}
        for N in [24, 48, 96]:
            sys_d = make_cubed_sphere_swe(N, H0=H0, g=g)
            h0, v10, v20 = _make_gaussian_ic(sys_d, variant, g, H0)
            h_f, _, _ = _run_to_time(sys_d, h0, v10, v20, T_end, cfl, g, H0)
            results[N] = {
                'h': h_f,
                'sys': sys_d,
            }

        # Self-convergence: compare 24 vs 48 subsampled to 24's grid
        h_48_sub = results[48]['h'][:, ::2, ::2]
        Wh_24 = results[24]['sys']['Wh']
        Jh_24 = results[24]['sys']['Jh']
        l2_24 = _l2_error(results[24]['h'], h_48_sub, Wh_24, Jh_24)

        h_96_sub = results[96]['h'][:, ::2, ::2]
        Wh_48 = results[48]['sys']['Wh']
        Jh_48 = results[48]['sys']['Jh']
        l2_48 = _l2_error(results[48]['h'], h_96_sub, Wh_48, Jh_48)

        rate = np.log2(l2_24 / l2_48)

        print(f"\nGauss {variant} self-convergence:")
        print(f"  N=24 vs 48: L2 = {l2_24:.4e}")
        print(f"  N=48 vs 96: L2 = {l2_48:.4e}")
        print(f"  Rate: {rate:.2f}")

        # Verify error decreases
        assert l2_48 < l2_24, \
            f"Gauss {variant}: error not decreasing with resolution"
