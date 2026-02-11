"""
Convergence study with Gaussian or cosine bell IC.

Usage:
    python step_f_convergence.py 40 80 160                  # cosine bell
    python step_f_convergence.py 40 80 160 --gaussian       # Gaussian hill
    python step_f_convergence.py 40 60 80 120 160 --gaussian
"""
import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

print(f"Platform: {jax.devices()[0].platform}")

from step_e_full_rotation import run

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    use_gaussian = '--gaussian' in flags

    if args:
        Ns = [int(x) for x in args]
    else:
        Ns = [40, 60, 80, 120, 160]

    ic_name = "Gaussian" if use_gaussian else "Cosine bell"

    results = []
    for N in Ns:
        print(f"\n{'#'*65}")
        print(f"  Running N = {N}, IC = {ic_name}")
        print(f"{'#'*65}")
        l1, l2, linf = run(N, use_gaussian=use_gaussian)
        results.append((N, l1, l2, linf))

    # Print convergence table
    print("\n")
    print("=" * 80)
    print(f"CONVERGENCE TABLE — {ic_name}")
    print("=" * 80)
    print(f"  {'N':>5}  {'l1':>12}  {'l2':>12}  {'linf':>12}  "
          f"{'l1 rate':>8}  {'l2 rate':>8}  {'linf rate':>8}")
    print("-" * 80)
    for i, (N, l1, l2, linf) in enumerate(results):
        if i == 0:
            print(f"  {N:5d}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}  "
                  f"{'---':>8}  {'---':>8}  {'---':>8}")
        else:
            N_prev, l1_prev, l2_prev, linf_prev = results[i-1]
            r = np.log2(N / N_prev)
            rate_l1 = np.log2(l1_prev / l1) / r
            rate_l2 = np.log2(l2_prev / l2) / r
            rate_linf = np.log2(linf_prev / linf) / r
            print(f"  {N:5d}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}  "
                  f"{rate_l1:8.2f}  {rate_l2:8.2f}  {rate_linf:8.2f}")
    print("=" * 80)

    if use_gaussian:
        print(f"\n  Shashkin Ch42 target: l2 rate ≈ 4.25, linf rate ≈ 3.98")
    else:
        print(f"\n  Expected for C¹ cosine bell: rate ≈ 1.5-2.0")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Ns_arr = np.array([r[0] for r in results])
        l1_arr = np.array([r[1] for r in results])
        l2_arr = np.array([r[2] for r in results])
        linf_arr = np.array([r[3] for r in results])

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.loglog(Ns_arr, l1_arr, 'o-', label='$l_1$', linewidth=2, markersize=8)
        ax.loglog(Ns_arr, l2_arr, 's-', label='$l_2$', linewidth=2, markersize=8)
        ax.loglog(Ns_arr, linf_arr, '^-', label='$l_\\infty$', linewidth=2, markersize=8)

        # Reference slopes
        N_ref = np.array([Ns_arr[0], Ns_arr[-1]])
        if use_gaussian:
            ref_orders = [(2, '--', '2nd order'), (4, ':', '4th order')]
        else:
            ref_orders = [(1.5, '--', '1.5 order'), (2, ':', '2nd order')]

        for order, ls, lbl in ref_orders:
            scale = l2_arr[0] * 1.5
            ref = scale * (N_ref / N_ref[0])**(-order)
            ax.loglog(N_ref, ref, ls, color='gray', alpha=0.7, label=lbl)

        ax.set_xlabel('N (cells per panel edge)', fontsize=13)
        ax.set_ylabel('Normalized error norm', fontsize=13)
        ax.set_title(f'SBP 4/2 {ic_name} Advection — Williamson TC1\n'
                      f'Full rotation, CFL=0.5, equatorial flow (α=0°)',
                      fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xticks(Ns_arr)
        ax.set_xticklabels([str(n) for n in Ns_arr])

        suffix = 'gaussian' if use_gaussian else 'cosinebell'
        plot_path = os.path.join(script_dir, f'convergence_{suffix}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")

    except ImportError:
        print("\nmatplotlib not available — skipping plot")
