"""
conftest.py — Shared pytest fixtures for SBP-SWE test suite
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.operators import sbp_42
from sbp_swe.geometry import equiangular_to_cartesian, compute_metric
from sbp_swe.mesh import make_staggered_grids, make_all_metrics
from sbp_swe.projection import build_projection_fn
from sbp_swe.system import make_cubed_sphere_swe
from sbp_swe.timestepping import make_rk4_step
from sbp_swe.diagnostics import compute_mass, compute_energy


@pytest.fixture(params=[8, 16])
def N(request):
    """Grid resolution."""
    return request.param


@pytest.fixture
def small_system():
    """N=8 system for fast unit tests."""
    return make_cubed_sphere_swe(8, H0=1.0, g=1.0)


@pytest.fixture
def medium_system():
    """N=16 system for conservation/convergence tests."""
    return make_cubed_sphere_swe(16, H0=1.0, g=1.0)


@pytest.fixture
def gaussian_ic():
    """Factory for Gaussian initial condition centered on panel 0."""
    def _make(sys_d, amplitude=0.1, sigma=0.2):
        N = sys_d['N']
        xi_v = sys_d['grids']['xi_v']
        xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')
        h = jnp.zeros((6, N+1, N+1))
        h = h.at[0].set(amplitude * jnp.exp(-(xi1_2d**2 + xi2_2d**2) / (2*sigma**2)))
        h = sys_d['project_h'](h)
        v1 = jnp.zeros((6, N, N+1))
        v2 = jnp.zeros((6, N+1, N))
        return h, v1, v2
    return _make
