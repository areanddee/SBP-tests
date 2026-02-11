"""
Step H comparison: Ghost-cell stencil vs SBP SAT-Projection.

Runs both methods at multiple N values with Gaussian IC,
prints side-by-side convergence table.

Usage:
    python step_h_compare.py                    # default N values, Gaussian
    python step_h_compare.py 40 60 80 120 160
    python step_h_compare.py 40 80 160 --cosbell
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

# Import both approaches
from step_e_full_rotation import run as run_ghost
from step_h_sat_projection import run as run_sbp


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    if args:
        Ns = [int(x) for x in args]
    else:
        Ns = [40, 60, 80, 120, 160]

    use_gaussian = '--cosbell' not in flags
    ic_name = "Gaussian" if use_gaussian else "Cosine bell"

    ghost_results = []
    sbp_results = []

    for N in Ns:
        print(f"\n{'#'*70}")
        print(f"  N = {N} — Ghost-cell stencil")
        print(f"{'#'*70}")
        l1g, l2g, linfg = run_ghost(N, use_gaussian=use_gaussian)
        ghost_results.append((N, l1g, l2g, linfg))

        print(f"\n{'#'*70}")
        print(f"  N = {N} — SBP SAT-Projection")
        print(f"{'#'*70}")
        l1s, l2s, linfs = run_sbp(N, use_gaussian=use_gaussian)
        sbp_results.append((N, l1s, l2s, linfs))

    # Side-by-side comparison
    print("\n")
    print("=" * 95)
    print(f"  CONVERGENCE COMPARISON — {ic_name}")
    print("=" * 95)
    print(f"  {'':5}  {'--- Ghost-cell stencil ---':>36}  {'--- SBP SAT-Projection ---':>36}")
    print(f"  {'N':>5}  {'l2':>12} {'linf':>12} {'rate_l2':>8}  {'l2':>12} {'linf':>12} {'rate_l2':>8}")
    print("-" * 95)
    for i in range(len(Ns)):
        N = Ns[i]
        _, l1g, l2g, linfg = ghost_results[i]
        _, l1s, l2s, linfs = sbp_results[i]

        if i == 0:
            rg = rs = "---"
            rig = ris = "---"
        else:
            N_prev = Ns[i-1]
            lr = np.log2(N / N_prev)
            rg  = f"{np.log2(ghost_results[i-1][2] / l2g) / lr:8.2f}"
            rs  = f"{np.log2(sbp_results[i-1][2] / l2s) / lr:8.2f}"

        print(f"  {N:5d}  {l2g:12.4e} {linfg:12.4e} {rg:>8}  "
              f"{l2s:12.4e} {linfs:12.4e} {rs:>8}")

    print("=" * 95)

    if use_gaussian:
        print(f"\n  Shashkin Ch42 target: l2 rate ≈ 4.25, linf rate ≈ 3.98")
    print(f"  FV3 (N=40):          l2 = 6.69e-2, linf = 4.94e-2")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        Ns_arr = np.array(Ns)
        g_l2 = np.array([r[2] for r in ghost_results])
        g_linf = np.array([r[3] for r in ghost_results])
        s_l2 = np.array([r[2] for r in sbp_results])
        s_linf = np.array([r[3] for r in sbp_results])

        # l2 plot
        ax = axes[0]
        ax.loglog(Ns_arr, g_l2, 'o-', label='Ghost-cell', linewidth=2, markersize=8)
        ax.loglog(Ns_arr, s_l2, 's-', label='SAT-Projection', linewidth=2, markersize=8)
        N_ref = np.array([Ns_arr[0], Ns_arr[-1]])
        for order, ls, lbl in [(2, '--', '2nd'), (4, ':', '4th')]:
            ref = g_l2[0] * 1.5 * (N_ref / N_ref[0])**(-order)
            ax.loglog(N_ref, ref, ls, color='gray', alpha=0.6, label=f'{lbl} order')
        ax.set_xlabel('N', fontsize=13)
        ax.set_ylabel('$l_2$ error', fontsize=13)
        ax.set_title(f'$l_2$ convergence — {ic_name}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

        # linf plot
        ax = axes[1]
        ax.loglog(Ns_arr, g_linf, 'o-', label='Ghost-cell', linewidth=2, markersize=8)
        ax.loglog(Ns_arr, s_linf, 's-', label='SAT-Projection', linewidth=2, markersize=8)
        for order, ls, lbl in [(2, '--', '2nd'), (4, ':', '4th')]:
            ref = g_linf[0] * 1.5 * (N_ref / N_ref[0])**(-order)
            ax.loglog(N_ref, ref, ls, color='gray', alpha=0.6, label=f'{lbl} order')
        ax.set_xlabel('N', fontsize=13)
        ax.set_ylabel('$l_\\infty$ error', fontsize=13)
        ax.set_title(f'$l_\\infty$ convergence — {ic_name}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

        plt.suptitle('Ghost-cell stencil vs SBP SAT-Projection\n'
                     'Williamson TC1, CFL=0.5, full rotation', fontsize=14, y=1.02)
        plt.tight_layout()

        suffix = 'gaussian' if use_gaussian else 'cosinebell'
        plot_path = os.path.join(script_dir, f'comparison_{suffix}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved: {plot_path}")

    except ImportError:
        print("\n  matplotlib not available — skipping plot")
