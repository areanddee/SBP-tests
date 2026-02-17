"""
test_conservation.py — Mass and Energy Conservation Tests
==========================================================

Verifies:
  - Mass conservation to machine precision (~1e-15) over time integration
  - Energy conservation: temporal error scales as dt^5 (RK4 property)
  - Tests both gauss 1 (panel-center) and gauss 2 (vertex) ICs
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
from sbp_swe.diagnostics import compute_mass, compute_energy


def _gaussian_ic(sys_d, variant=1):
    """
    Gaussian initial condition for h, v=0.

    variant=1: center at panel 0 center (0, 0, 1) — easy case
    variant=2: center at cube vertex (panel 0 corner) — hardest case
    """
    N = sys_d['N']
    grids = sys_d['grids']
    xi_v = grids['xi_v']
    xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')

    if variant == 1:
        xc, yc, zc = 0.0, 0.0, 1.0
    else:
        xc, yc, zc = equiangular_to_cartesian(
            jnp.pi / 4, jnp.pi / 4, 0)
        xc, yc, zc = float(xc), float(yc), float(zc)

    R0 = 1.0 / 3.0
    h = jnp.zeros((6, N + 1, N + 1))
    for p in range(6):
        X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
        d2 = (X - xc)**2 + (Y - yc)**2 + (Z - zc)**2
        chord_dist2 = d2
        h = h.at[p].set(jnp.exp(-chord_dist2 / (2 * R0**2)))

    h = sys_d['project_h'](h)
    v1 = jnp.zeros((6, N, N + 1))
    v2 = jnp.zeros((6, N + 1, N))
    return h, v1, v2


def _run_integration(sys_d, h, v1, v2, cfl, n_steps):
    """Run n_steps of RK4 and return mass/energy history."""
    step_fn = make_rk4_step(sys_d['rhs'])
    g = 1.0
    H0 = 1.0
    c = np.sqrt(g * H0)
    dt = cfl * sys_d['dx'] / c

    Wh = sys_d['Wh']
    W1 = jnp.outer(jnp.diag(sys_d['ops'].Hc), jnp.diag(sys_d['ops'].Hv))
    W2 = jnp.outer(jnp.diag(sys_d['ops'].Hv), jnp.diag(sys_d['ops'].Hc))
    Jh = sys_d['Jh']
    J1 = sys_d['J1']
    J2 = sys_d['J2']
    metrics = sys_d['metrics']
    Pvc = sys_d['Pvc']
    Pcv = sys_d['Pcv']

    mass_hist = [compute_mass(h, Wh, Jh)]
    energy_hist = [compute_energy(h, v1, v2, Wh, W1, W2, Jh, J1, J2,
                                   g, H0, metrics, Pvc, Pcv)]

    for _ in range(n_steps):
        h, v1, v2 = step_fn(h, v1, v2, dt)
        mass_hist.append(compute_mass(h, Wh, Jh))
        energy_hist.append(compute_energy(h, v1, v2, Wh, W1, W2, Jh, J1, J2,
                                           g, H0, metrics, Pvc, Pcv))

    return mass_hist, energy_hist, dt


class TestMassConservation:
    """Mass must be conserved to machine precision."""

    @pytest.mark.parametrize("variant", [1, 2])
    def test_mass_conservation(self, variant):
        """Mass change < 1e-14 after 50 steps at CFL=0.3."""
        N = 16
        sys_d = make_cubed_sphere_swe(N, H0=1.0, g=1.0)
        h, v1, v2 = _gaussian_ic(sys_d, variant=variant)

        mass_hist, _, _ = _run_integration(sys_d, h, v1, v2,
                                            cfl=0.3, n_steps=50)

        max_drift = max(abs(m - mass_hist[0]) for m in mass_hist)
        assert max_drift < 1e-14, \
            f"Gauss {variant}: mass drift = {max_drift:.2e}"


class TestEnergyConservation:
    """Energy error must scale as dt^5 (4th order RK4 property)."""

    @pytest.mark.parametrize("variant", [1, 2])
    def test_energy_dt5_scaling(self, variant):
        """
        Run at two CFL values, energy error ratio should be ~2^5 = 32.
        """
        N = 16
        sys_d = make_cubed_sphere_swe(N, H0=1.0, g=1.0)
        h0, v10, v20 = _gaussian_ic(sys_d, variant=variant)

        # Coarse dt
        _, energy_c, dt_c = _run_integration(sys_d, h0, v10, v20,
                                              cfl=0.4, n_steps=20)
        dE_c = abs(energy_c[-1] - energy_c[0])

        # Fine dt (half)
        _, energy_f, dt_f = _run_integration(sys_d, h0, v10, v20,
                                              cfl=0.2, n_steps=40)
        dE_f = abs(energy_f[-1] - energy_f[0])

        if dE_f < 1e-16:
            pytest.skip("Energy error at noise floor")

        ratio = dE_c / dE_f
        # dt^5 scaling: ratio should be ~(2)^5 = 32
        # Allow generous range [8, 128] to account for nonlinear effects
        assert 8 < ratio < 128, \
            f"Gauss {variant}: energy ratio = {ratio:.1f} (expected ~32 for dt^5)"
