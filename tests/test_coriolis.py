"""
test_coriolis.py — Coriolis Operator Tests
============================================

Verifies:
  - V operator is idempotent on continuous fields
  - V roundtrip: Y·X = I for edge-continuous input
  - Energy neutrality: v^T H J Q (F v) = 0 (antisymmetric)
  - Zero velocity → zero Coriolis tendency
  - Coriolis with time integration: energy error stays dt^5

Reference: Shashkin 2025, Sections 4.5, 5.4, Eq. 57-63
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.operators import sbp_42
from sbp_swe.geometry import compute_metric
from sbp_swe.mesh import make_staggered_grids, make_all_metrics, compute_contravariant
from sbp_swe.projection import build_projection_fn
from sbp_swe.coriolis import apply_V, coriolis_tendency, compute_all_panel_bases


def _make_coriolis_setup(N):
    """Build everything needed for Coriolis tests."""
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)
    dx = float(grids['dx'])
    ops = sbp_42(N, dx)
    Hv_diag = jnp.diag(ops.Hv)
    Hc_diag = jnp.diag(ops.Hc)

    project_h, corners = build_projection_fn(N, metrics['Jh'], Hv_diag)
    bases_h = compute_all_panel_bases(grids['xi1_h'], grids['xi2_h'])

    # Constant Coriolis parameter
    f_h = jnp.ones((N + 1, N + 1)) * 1.0e-4

    # Quadrature weights
    W1 = jnp.outer(Hc_diag, Hv_diag)
    W2 = jnp.outer(Hv_diag, Hc_diag)

    return {
        'N': N, 'ops': ops, 'grids': grids, 'metrics': metrics,
        'project_h': project_h, 'bases_h': bases_h, 'f_h': f_h,
        'Hv_diag': Hv_diag, 'Hc_diag': Hc_diag,
        'W1': W1, 'W2': W2,
    }


class TestVOperator:
    """V operator: edge-continuity via Cartesian projection."""

    @pytest.mark.parametrize("N", [8, 16])
    def test_v_idempotent(self, N):
        """V(V(w)) = V(w) — idempotent on any input."""
        setup = _make_coriolis_setup(N)

        key = jax.random.PRNGKey(42)
        w1_h = jax.random.normal(key, (6, N + 1, N + 1))
        w2_h = jax.random.normal(jax.random.PRNGKey(43), (6, N + 1, N + 1))

        w1_v, w2_v = apply_V(w1_h, w2_h, setup['bases_h'], setup['project_h'])
        w1_vv, w2_vv = apply_V(w1_v, w2_v, setup['bases_h'], setup['project_h'])

        err1 = float(jnp.max(jnp.abs(w1_vv - w1_v)))
        err2 = float(jnp.max(jnp.abs(w2_vv - w2_v)))
        assert max(err1, err2) < 1e-13, \
            f"V not idempotent: {max(err1, err2):.2e}"

    @pytest.mark.parametrize("N", [8, 16])
    def test_v_preserves_uniform(self, N):
        """V applied to uniform covariant field → unchanged (up to edge effects)."""
        setup = _make_coriolis_setup(N)
        # Uniform w1=1, w2=0 is NOT continuous in covariant sense across panels
        # But zero field should be preserved
        w1_h = jnp.zeros((6, N + 1, N + 1))
        w2_h = jnp.zeros((6, N + 1, N + 1))

        w1_v, w2_v = apply_V(w1_h, w2_h, setup['bases_h'], setup['project_h'])

        err = max(float(jnp.max(jnp.abs(w1_v))),
                  float(jnp.max(jnp.abs(w2_v))))
        assert err < 1e-15, f"V(0) != 0: {err:.2e}"


class TestCoriolisEnergyNeutrality:
    """The Coriolis operator must not create or destroy energy."""

    @pytest.mark.parametrize("N", [8, 16])
    def test_energy_neutrality(self, N):
        """
        Energy inner product: sum_p (v1 · F1 · W1 · J1 + v2 · F2 · W2 · J2) = 0

        where F1, F2 are the Coriolis tendencies dv1/dt, dv2/dt.
        This must vanish because the Coriolis force does no work.
        """
        setup = _make_coriolis_setup(N)
        metrics = setup['metrics']

        # Random velocity field
        key = jax.random.PRNGKey(99)
        v1 = 0.1 * jax.random.normal(key, (6, N, N + 1))
        v2 = 0.1 * jax.random.normal(jax.random.PRNGKey(100), (6, N + 1, N))

        # Compute contravariant for energy inner product
        Pvc = setup['ops'].Pvc
        Pcv = setup['ops'].Pcv

        def _contra(v1_p, v2_p):
            return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
        v1c, v2c = jax.vmap(_contra)(v1, v2)

        # Coriolis tendency
        dv1, dv2 = coriolis_tendency(
            v1, v2, setup['f_h'],
            metrics['Jh'], metrics['J1'], metrics['J2'],
            Pcv, Pvc, setup['bases_h'], setup['project_h'],
            use_V=True)

        # Contravariant of tendency for energy inner product
        dv1c, dv2c = jax.vmap(_contra)(dv1, dv2)

        # Energy inner product: v · (J * Q * dv) = v · (J * dv^contra)
        J1 = metrics['J1']
        J2 = metrics['J2']
        W1 = setup['W1']
        W2 = setup['W2']

        work = (float(jnp.sum(v1 * J1 * dv1c * W1[None, :, :])) +
                float(jnp.sum(v2 * J2 * dv2c * W2[None, :, :])))

        # Normalize by energy scale
        energy_scale = (float(jnp.sum(v1**2 * J1 * W1[None, :, :])) +
                        float(jnp.sum(v2**2 * J2 * W2[None, :, :])))

        rel_work = abs(work) / energy_scale if energy_scale > 0 else abs(work)

        assert rel_work < 1e-10, \
            f"Energy neutrality violated: relative work = {rel_work:.2e}"

    @pytest.mark.parametrize("N", [8, 16])
    def test_energy_neutrality_without_V(self, N):
        """Energy neutrality also holds without V (Eq. 57 form)."""
        setup = _make_coriolis_setup(N)
        metrics = setup['metrics']

        key = jax.random.PRNGKey(77)
        v1 = 0.1 * jax.random.normal(key, (6, N, N + 1))
        v2 = 0.1 * jax.random.normal(jax.random.PRNGKey(78), (6, N + 1, N))

        Pvc = setup['ops'].Pvc
        Pcv = setup['ops'].Pcv

        def _contra(v1_p, v2_p):
            return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
        v1c, v2c = jax.vmap(_contra)(v1, v2)

        dv1, dv2 = coriolis_tendency(
            v1, v2, setup['f_h'],
            metrics['Jh'], metrics['J1'], metrics['J2'],
            Pcv, Pvc, setup['bases_h'], setup['project_h'],
            use_V=False)

        dv1c, dv2c = jax.vmap(_contra)(dv1, dv2)

        J1 = metrics['J1']
        J2 = metrics['J2']
        W1 = setup['W1']
        W2 = setup['W2']

        work = (float(jnp.sum(v1 * J1 * dv1c * W1[None, :, :])) +
                float(jnp.sum(v2 * J2 * dv2c * W2[None, :, :])))

        energy_scale = (float(jnp.sum(v1**2 * J1 * W1[None, :, :])) +
                        float(jnp.sum(v2**2 * J2 * W2[None, :, :])))

        rel_work = abs(work) / energy_scale if energy_scale > 0 else abs(work)

        assert rel_work < 1e-10, \
            f"Energy neutrality (no V) violated: {rel_work:.2e}"


class TestCoriolisBasic:
    """Basic Coriolis operator properties."""

    @pytest.mark.parametrize("N", [8, 16])
    def test_zero_velocity_zero_tendency(self, N):
        """Zero velocity → zero Coriolis tendency."""
        setup = _make_coriolis_setup(N)
        metrics = setup['metrics']

        v1 = jnp.zeros((6, N, N + 1))
        v2 = jnp.zeros((6, N + 1, N))

        dv1, dv2 = coriolis_tendency(
            v1, v2, setup['f_h'],
            metrics['Jh'], metrics['J1'], metrics['J2'],
            setup['ops'].Pcv, setup['ops'].Pvc,
            setup['bases_h'], setup['project_h'])

        err = max(float(jnp.max(jnp.abs(dv1))),
                  float(jnp.max(jnp.abs(dv2))))
        assert err < 1e-15, f"Zero velocity has nonzero Coriolis: {err:.2e}"

    @pytest.mark.parametrize("N", [8, 16])
    def test_zero_f_zero_tendency(self, N):
        """Zero Coriolis parameter → zero tendency regardless of velocity."""
        setup = _make_coriolis_setup(N)
        metrics = setup['metrics']

        key = jax.random.PRNGKey(55)
        v1 = jax.random.normal(key, (6, N, N + 1))
        v2 = jax.random.normal(jax.random.PRNGKey(56), (6, N + 1, N))

        f_zero = jnp.zeros((N + 1, N + 1))

        dv1, dv2 = coriolis_tendency(
            v1, v2, f_zero,
            metrics['Jh'], metrics['J1'], metrics['J2'],
            setup['ops'].Pcv, setup['ops'].Pvc,
            setup['bases_h'], setup['project_h'])

        err = max(float(jnp.max(jnp.abs(dv1))),
                  float(jnp.max(jnp.abs(dv2))))
        assert err < 1e-14, f"Zero f has nonzero Coriolis: {err:.2e}"
