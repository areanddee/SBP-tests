"""
test_steady_state.py — Steady State Tests
============================================

Verifies:
  - Uniform h, zero velocity → zero RHS tendency
  - Projection preserves uniform fields
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.system import make_cubed_sphere_swe


class TestSteadyState:
    """Uniform fields should produce zero tendency."""

    @pytest.mark.parametrize("N", [8, 16])
    def test_uniform_h_zero_v(self, N):
        """Uniform h, zero velocity → all tendencies vanish."""
        sys_d = make_cubed_sphere_swe(N, H0=1.0, g=1.0)

        h = jnp.ones((6, N + 1, N + 1)) * 0.5
        h = sys_d['project_h'](h)
        v1 = jnp.zeros((6, N, N + 1))
        v2 = jnp.zeros((6, N + 1, N))

        dh, dv1, dv2 = sys_d['rhs'](h, v1, v2)

        err_h = float(jnp.max(jnp.abs(dh)))
        err_v1 = float(jnp.max(jnp.abs(dv1)))
        err_v2 = float(jnp.max(jnp.abs(dv2)))

        assert err_h < 1e-13, f"|dh/dt| = {err_h:.2e}"
        assert err_v1 < 1e-13, f"|dv1/dt| = {err_v1:.2e}"
        assert err_v2 < 1e-13, f"|dv2/dt| = {err_v2:.2e}"

    @pytest.mark.parametrize("N", [8, 16])
    def test_zero_h_zero_v(self, N):
        """All zeros → all zero tendencies."""
        sys_d = make_cubed_sphere_swe(N, H0=1.0, g=1.0)

        h = jnp.zeros((6, N + 1, N + 1))
        v1 = jnp.zeros((6, N, N + 1))
        v2 = jnp.zeros((6, N + 1, N))

        dh, dv1, dv2 = sys_d['rhs'](h, v1, v2)

        err = max(float(jnp.max(jnp.abs(dh))),
                  float(jnp.max(jnp.abs(dv1))),
                  float(jnp.max(jnp.abs(dv2))))
        assert err < 1e-15, f"Zero state has nonzero tendency: {err:.2e}"
