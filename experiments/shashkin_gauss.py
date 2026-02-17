#!/usr/bin/env python
"""
shashkin_gauss.py — Shashkin Section 6.2 Gaussian Test Reproducer
===================================================================

Runs the full Shashkin Gaussian wave tests at publication parameters:
  - 25-day integration (T_end = 25.0 in nondimensional units)
  - CFL = 0.05 (small dt to eliminate time truncation)
  - N = 24, 48, 96, 192
  - Gauss variants 1 (panel center) and 2 (cube vertex)

Self-convergence analysis: compare N vs 2N (subsampled to N's grid).

Usage:
    python experiments/shashkin_gauss.py --gauss 1
    python experiments/shashkin_gauss.py --gauss 2
    python experiments/shashkin_gauss.py             # both
    python experiments/shashkin_gauss.py --Ns 24 48 96 192
"""

import argparse
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from sbp_swe.geometry import equiangular_to_cartesian
from sbp_swe.system import make_cubed_sphere_swe
from sbp_swe.timestepping import make_rk4_step
from sbp_swe.diagnostics import compute_mass


def make_gaussian_ic(sys_d, variant):
    """Gaussian IC per Shashkin Section 6.2."""
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


def run_experiment(variant, Ns, T_end=25.0, cfl=0.05, g=1.0, H0=1.0):
    """Run Gaussian wave test at multiple resolutions."""
    c = np.sqrt(g * H0)

    print(f"\n{'='*65}")
    print(f"  Gauss Variant {variant} -- {'Panel Center' if variant==1 else 'Cube Vertex'}")
    print(f"  T_end={T_end}, CFL={cfl}, Ns={Ns}")
    print(f"{'='*65}")

    results = {}

    for N in Ns:
        sys_d = make_cubed_sphere_swe(N, H0=H0, g=g)
        h0, v10, v20 = make_gaussian_ic(sys_d, variant)
        step_fn = make_rk4_step(sys_d['rhs'])

        dt = cfl * sys_d['dx'] / c
        n_steps = int(np.ceil(T_end / dt))
        dt = T_end / n_steps

        mass0 = compute_mass(h0, sys_d['Wh'], sys_d['Jh'])

        t0 = time.time()
        h, v1, v2 = h0, v10, v20
        for step_i in range(n_steps):
            h, v1, v2 = step_fn(h, v1, v2, dt)
            if (step_i + 1) % max(1, n_steps // 5) == 0:
                mass = compute_mass(h, sys_d['Wh'], sys_d['Jh'])
                print(f"  N={N:4d}  step {step_i+1:6d}/{n_steps}  "
                      f"mass_drift={abs(mass-mass0):.2e}")

        elapsed = time.time() - t0
        mass_f = compute_mass(h, sys_d['Wh'], sys_d['Jh'])

        results[N] = {
            'h': h, 'sys': sys_d,
            'mass_drift': abs(mass_f - mass0),
            'elapsed': elapsed, 'n_steps': n_steps, 'dt': dt,
        }
        print(f"  N={N:4d}  DONE  {elapsed:.1f}s  mass_drift={abs(mass_f-mass0):.2e}")

    # Self-convergence
    sorted_Ns = sorted(results.keys())
    print(f"\n  Self-Convergence (L2, Linf):")
    print(f"  {'N':>4s}  {'L2':>12s}  {'Linf':>12s}  {'L2 rate':>8s}  {'Linf rate':>8s}")
    print(f"  {'-'*50}")

    prev_l2, prev_linf = None, None
    for i, N in enumerate(sorted_Ns):
        if N * 2 not in results:
            print(f"  {N:4d}  (no 2N reference)")
            continue

        h_N = results[N]['h']
        h_2N_sub = results[N * 2]['h'][:, ::2, ::2]

        Wh = results[N]['sys']['Wh']
        Jh = results[N]['sys']['Jh']

        diff = h_N - h_2N_sub
        l2 = float(jnp.sqrt(jnp.sum(diff**2 * Jh[None, :, :] * Wh[None, :, :])))
        linf = float(jnp.max(jnp.abs(diff)))

        rate_l2 = f"{np.log2(prev_l2 / l2):.2f}" if prev_l2 else "   -"
        rate_linf = f"{np.log2(prev_linf / linf):.2f}" if prev_linf else "   -"

        print(f"  {N:4d}  {l2:12.4e}  {linf:12.4e}  {rate_l2:>8s}  {rate_linf:>8s}")
        prev_l2, prev_linf = l2, linf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shashkin Gaussian Wave Tests")
    parser.add_argument('--gauss', type=int, choices=[1, 2],
                        help='Run only variant 1 or 2')
    parser.add_argument('--Ns', type=int, nargs='+', default=[24, 48, 96],
                        help='Resolutions to test')
    parser.add_argument('--T', type=float, default=25.0,
                        help='Integration time (default 25.0)')
    parser.add_argument('--cfl', type=float, default=0.05,
                        help='CFL number (default 0.05)')
    args = parser.parse_args()

    variants = [args.gauss] if args.gauss else [1, 2]
    for v in variants:
        run_experiment(v, args.Ns, T_end=args.T, cfl=args.cfl)
