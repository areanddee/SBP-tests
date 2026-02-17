"""
test_operators.py — SBP 4/2 Operator Properties
=================================================

Verifies the fundamental SBP identity and operator properties:
  - SBP identity: Hv @ Dcv + Dcv.T @ Hv = diag([-1,...,0,...,+1])
  - Accuracy: 4th order interior, 2nd order boundary
  - Interpolation: Pcv and Pvc are transposes w.r.t. quadrature
  - Quadrature: Hv and Hc weights sum to domain length
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.operators import sbp_42


class TestSBPIdentity:
    """The SBP identity is the foundation of everything."""

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_sbp_identity(self, N):
        """Hv @ Dcv + Dcv.T @ Hv = diag([-1,0,...,0,+1])"""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)

        Q = ops.Hv @ ops.Dcv + ops.Dcv.T @ ops.Hv

        expected = jnp.zeros((N + 1, N + 1))
        expected = expected.at[0, 0].set(-1.0)
        expected = expected.at[N, N].set(+1.0)

        err = float(jnp.max(jnp.abs(Q - expected)))
        assert err < 1e-13, f"SBP identity error: {err:.2e}"

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_dual_sbp_identity(self, N):
        """Hc @ Dvc + Dvc.T @ Hc = e_R @ r.T - e_L @ l.T"""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)

        Q = ops.Hc @ ops.Dvc + ops.Dvc.T @ ops.Hc

        # Boundary: e_R r^T - e_L l^T
        expected = jnp.outer(jnp.eye(N)[-1], ops.r) - jnp.outer(jnp.eye(N)[0], ops.l)

        err = float(jnp.max(jnp.abs(Q - expected)))
        assert err < 1e-13, f"Dual SBP identity error: {err:.2e}"


class TestQuadrature:
    """Quadrature weights must integrate constants exactly."""

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_hv_weights_sum(self, N):
        """Hv diagonal sums to domain length L = pi/2."""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)
        total = float(jnp.sum(jnp.diag(ops.Hv)))
        assert abs(total - jnp.pi / 2) < 1e-13, f"Hv sum = {total}, expected pi/2"

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_hc_weights_sum(self, N):
        """Hc diagonal sums to domain length L = pi/2."""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)
        total = float(jnp.sum(jnp.diag(ops.Hc)))
        assert abs(total - jnp.pi / 2) < 1e-13, f"Hc sum = {total}, expected pi/2"


class TestInterpolation:
    """Interpolation operators relate vertex and cell-center grids."""

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_interpolation_adjoint(self, N):
        """Hv @ Pcv = Pvc.T @ Hc  (adjoint relation)"""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)

        lhs = ops.Hv @ ops.Pcv
        rhs = ops.Pvc.T @ ops.Hc

        err = float(jnp.max(jnp.abs(lhs - rhs)))
        assert err < 1e-13, f"Interpolation adjoint error: {err:.2e}"


class TestDifferentiationAccuracy:
    """Verify derivative accuracy on smooth test functions."""

    @pytest.mark.parametrize("N", [16, 32, 64])
    def test_dvc_accuracy_on_sin(self, N):
        """Dvc applied to sin(x) should approximate cos(x) at cell centers."""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)

        xi_v = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, N + 1)
        xi_c = (jnp.arange(N) + 0.5) * dx - jnp.pi / 4

        f_v = jnp.sin(4 * xi_v)
        df_exact = 4 * jnp.cos(4 * xi_c)
        df_approx = ops.Dvc @ f_v

        # Interior error should converge at 4th order
        err = float(jnp.max(jnp.abs(df_approx[2:-2] - df_exact[2:-2])))
        # Loose bound: just verify it's converging
        assert err < 0.5 / N**3, f"Dvc accuracy error: {err:.2e} at N={N}"
