"""
timestepping.py — Time Integration Methods
============================================

RK4 (classical 4th-order Runge-Kutta) for the cubed-sphere SWE.
Energy error scales as dt^5 (4th order method, one extra from symmetry).
"""

import jax


def make_rk4_step(rhs_fn):
    """
    Build a single RK4 time step function.

    Args:
        rhs_fn: function (h, v1, v2) → (dh_dt, dv1_dt, dv2_dt)

    Returns:
        step(h, v1, v2, dt) → (h_new, v1_new, v2_new)
    """
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
