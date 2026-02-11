"""
Convergence study: runs multiple N values and generates a plot.

Usage:
    python step_f_convergence.py                    # default: 20 30 40 60 80 120 160
    python step_f_convergence.py 20 40 80 160       # custom N values
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

# Import the run function from step_e
from step_e_full_rotation import run

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        Ns = [int(x) for x in sys.argv[1:]]
    else:
        Ns = [20, 30, 40, 60, 80, 120, 160]

    results = []
    for N in Ns:
        print(f"\n{'#'*65}")
        print(f"  Running N = {N}")
        print(f"{'#'*65}")
        l1, l2, linf = run(N)
        results.append((N, l1, l2, linf))

    # Print convergence table
    print("\n")
    print("=" * 75)
    print("CONVERGENCE TABLE")
    print("=" * 75)
    print(f"  {'N':>5}  {'l1':>12}  {'l2':>12}  {'linf':>12}  {'l1 rate':>8}  {'l2 rate':>8}  {'linf rate':>8}")
    print("-" * 75)
    for i, (N, l1, l2, linf) in enumerate(results):
        if i == 0:
            print(f"  {N:5d}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}  {'---':>8}  {'---':>8}  {'---':>8}")
        else:
            N_prev, l1_prev, l2_prev, linf_prev = results[i-1]
            r = np.log2(N / N_prev)
            rate_l1 = np.log2(l1_prev / l1) / r
            rate_l2 = np.log2(l2_prev / l2) / r
            rate_linf = np.log2(linf_prev / linf) / r
            print(f"  {N:5d}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}  {rate_l1:8.2f}  {rate_l2:8.2f}  {rate_linf:8.2f}")
    print("=" * 75)

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
        for order, ls, lbl in [(2, '--', '2nd order'), (3, ':', '3rd order')]:
            scale = l2_arr[0] * 1.5
            ref = scale * (N_ref / N_ref[0])**(-order)
            ax.loglog(N_ref, ref, ls, color='gray', alpha=0.7, label=lbl)

        ax.set_xlabel('N (cells per panel edge)', fontsize=13)
        ax.set_ylabel('Normalized error norm', fontsize=13)
        ax.set_title('SBP 4/2 Cosine Bell Advection — Williamson TC1\n'
                      'Full rotation, CFL=0.5, equatorial flow (α=0°)',
                      fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xticks(Ns_arr)
        ax.set_xticklabels([str(n) for n in Ns_arr])

        plt.tight_layout()
        plot_path = os.path.join(script_dir, 'convergence_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")

    except ImportError:
        print("\nmatplotlib not available — skipping plot generation")
        print("Install with: pip install matplotlib")
