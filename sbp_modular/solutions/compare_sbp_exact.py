"""
compare_sbp_exact.py — SBP vs Exact spectral solution comparison
=================================================================

Runs the SBP-SAT solver and compares against the exact Legendre
expansion at multiple output times. Produces:
  1. Side-by-side 6-panel plots (SBP | Exact | Diff) at each output time
  2. L2, Linf error norms at each output time
  3. Error growth curve

This isolates WHERE and WHEN the SBP solution diverges from exact,
exposing edge/corner issues in the SAT treatment.

Usage (on GPU node):
    python compare_sbp_exact.py --variant 1 --N 96
    python compare_sbp_exact.py --variant 2 --N 96
    python compare_sbp_exact.py --tilt1 1.0 --tilt2 0.0 --face 4 --N 96

Requires: JAX + GPU, exact_gaussian_wave.py in same directory
"""
import os
import sys
import time as _time
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Import SBP infrastructure — adjust path as needed
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbp_staggered_1d import sbp_42
from grid import equiangular_to_cartesian
from sat_operators import build_cartesian_sat_fn
from connectivity import EDGES

# Import exact solution (pure numpy)
from exact_gaussian_wave import ExactGaussianWave


# ============================================================
# Equiangular → Cartesian (numpy, for exact solution)
# ============================================================

def equiangular_to_cartesian_np(xi1, xi2, face_id):
    t1 = np.tan(xi1)
    t2 = np.tan(xi2)
    d = np.sqrt(1.0 + t1**2 + t2**2)
    if face_id == 0:
        X, Y, Z = t1/d, t2/d, 1/d
    elif face_id == 1:
        X, Y, Z = -t1/d, 1/d, t2/d
    elif face_id == 2:
        X, Y, Z = -1/d, -t1/d, t2/d
    elif face_id == 3:
        X, Y, Z = t1/d, -1/d, t2/d
    elif face_id == 4:
        X, Y, Z = 1/d, t1/d, t2/d
    elif face_id == 5:
        X, Y, Z = -t1/d, t2/d, -1/d
    return X, Y, Z

FACE_LABELS = [
    "Face 0 (+Z)", "Face 1 (+Y)", "Face 2 (-X)",
    "Face 3 (-Y)", "Face 4 (+X)", "Face 5 (-Z)",
]


# ============================================================
# Exact solution on all panels
# ============================================================

def exact_on_panels(sol, N, t):
    """Evaluate exact h on the vertex grid (N+1 × N+1) of all 6 panels."""
    pi4 = np.pi / 4
    xi_v = np.linspace(-pi4, pi4, N + 1)
    xi1_2d, xi2_2d = np.meshgrid(xi_v, xi_v, indexing='ij')

    h_exact = np.zeros((6, N + 1, N + 1))
    for p in range(6):
        X, Y, Z = equiangular_to_cartesian_np(xi1_2d, xi2_2d, p)
        h_exact[p] = sol.evaluate_cartesian(X, Y, Z, t)

    return h_exact


# ============================================================
# Plot comparison: SBP vs Exact vs Diff
# ============================================================

def plot_comparison(h_sbp, h_exact, t, l2, linf, center_label, outfile):
    """3-row × 6-col comparison plot."""
    diff = np.array(h_sbp) - h_exact

    fig, axes = plt.subplots(3, 6, figsize=(24, 12))

    # Common color ranges
    vmax_sol = max(np.max(np.abs(h_sbp)), np.max(np.abs(h_exact)))
    if vmax_sol < 1e-12:
        vmax_sol = 1.0
    vmax_diff = np.max(np.abs(diff))
    if vmax_diff < 1e-12:
        vmax_diff = 1.0

    N = h_sbp.shape[1] - 1
    pi4 = np.pi / 4
    xi_v = np.linspace(-pi4, pi4, N + 1)
    xi1_2d, xi2_2d = np.meshgrid(xi_v, xi_v, indexing='ij')
    xi1_deg = np.degrees(xi1_2d)
    xi2_deg = np.degrees(xi2_2d)

    row_labels = ['SBP', 'Exact', 'Difference']
    row_data = [np.array(h_sbp), h_exact, diff]
    row_vmax = [vmax_sol, vmax_sol, vmax_diff]

    for row in range(3):
        for p in range(6):
            ax = axes[row, p]
            vm = row_vmax[row]
            im = ax.pcolormesh(xi1_deg, xi2_deg, row_data[row][p],
                               cmap='RdBu_r', vmin=-vm, vmax=vm,
                               shading='auto')
            if row == 0:
                ax.set_title(FACE_LABELS[p], fontsize=10)
            if p == 0:
                ax.set_ylabel(row_labels[row], fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)

        fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, pad=0.02)

    fig.suptitle(
        f"SBP vs Exact  t={t:.3f} days  N={N}\n"
        f"L2={l2:.4e}  L∞={linf:.4e}  {center_label}",
        fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Error norm summary plot
# ============================================================

def plot_error_growth(times, l2s, linfs, center_label, outfile):
    """Plot L2 and Linf error vs time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(times, l2s, 'o-b', linewidth=2, markersize=6)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('L2 error')
    ax1.set_title('L2 Error Growth')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(times, linfs, 's-r', linewidth=2, markersize=6)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('L∞ error')
    ax2.set_title('L∞ Error Growth')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Error Growth  {center_label}", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Main comparison driver
# ============================================================

def run_comparison(center_xyz, center_label, N=96,
                   output_times=None, outdir='.'):
    """
    Run SBP solver with output snapshots, compare against exact.

    NOTE: This function uses the SBP infrastructure from the test file.
    You may need to adjust imports to match your module layout.
    """
    if output_times is None:
        output_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    os.makedirs(outdir, exist_ok=True)

    print("=" * 65)
    print(f"  SBP vs Exact Comparison")
    print(f"  Center: {center_label}")
    print(f"  N = {N}")
    print(f"  Output times: {output_times}")
    print("=" * 65)

    # Shashkin parameters
    sigma_ic = 1.0 / (4.0 * np.sqrt(2.0))
    amp_ic = 1.0
    H0 = 1.0
    g = 1.0
    c = np.sqrt(g * H0)

    X0, Y0, Z0 = center_xyz

    # --- Build exact solution ---
    print("  Building exact spectral solution...")
    sol = ExactGaussianWave(amp=amp_ic, sigma=sigma_ic, c=c, a=1.0,
                             center_xyz=center_xyz, L_max=300, n_quad=600)

    # --- Build SBP system ---
    # NOTE: Import make_cubed_sphere_swe and supporting functions
    # from your test file or refactored modules.
    # This is a placeholder that you'll wire to your actual code.
    print("  Building SBP system...")

    # ---- THIS SECTION MUST BE ADAPTED TO YOUR IMPORTS ----
    # from test_stag_step5_Nconv import (make_cubed_sphere_swe,
    #     make_rk4_step, compute_mass)
    # sys_d = make_cubed_sphere_swe(N, H0, g)
    # ... etc
    # ---- END ADAPTATION SECTION ----

    # For now, print what needs to happen:
    print("  TODO: Wire to make_cubed_sphere_swe, make_rk4_step")
    print("  This script provides the comparison framework.")
    print("  Generating exact-only snapshots for now...")

    # --- Generate exact solution at all output times ---
    print(f"\n  {'Time':>8} {'max|h|':>12}")
    print(f"  {'-'*22}")

    for t in output_times:
        h_exact = exact_on_panels(sol, N, t)
        print(f"  {t:8.3f} {np.max(np.abs(h_exact)):12.4e}")

    print(f"\n  Exact snapshots verified. Wire SBP solver to complete.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SBP vs Exact comparison")
    parser.add_argument('--variant', type=int, choices=[1, 2], default=None,
                        help='1=face center, 2=cube vertex')
    parser.add_argument('--tilt1', type=float, default=None,
                        help='xi1 tilt factor (0-1, multiplied by pi/4)')
    parser.add_argument('--tilt2', type=float, default=None,
                        help='xi2 tilt factor (0-1, multiplied by pi/4)')
    parser.add_argument('--face', type=int, default=0,
                        help='Panel for tilt center (default: 0)')
    parser.add_argument('--N', type=int, default=96,
                        help='Grid resolution (default: 96)')
    parser.add_argument('--times', nargs='+', type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help='Output times in days')
    parser.add_argument('--outdir', default='comparison',
                        help='Output directory')
    args = parser.parse_args()

    if args.variant == 1:
        center_xyz = (0.0, 0.0, 1.0)
        label = "face center (0,0,1)"
    elif args.variant == 2:
        s3 = 1.0 / np.sqrt(3.0)
        center_xyz = (s3, s3, s3)
        label = "cube vertex (1,1,1)/√3"
    elif args.tilt1 is not None:
        xi1 = args.tilt1 * np.pi / 4
        xi2 = (args.tilt2 if args.tilt2 is not None else 0.0) * np.pi / 4
        X, Y, Z = equiangular_to_cartesian_np(xi1, xi2, args.face)
        center_xyz = (float(X), float(Y), float(Z))
        label = (f"tilt=({args.tilt1},{args.tilt2}) face={args.face} "
                 f"→ ({X:.4f},{Y:.4f},{Z:.4f})")
    else:
        center_xyz = (0.0, 0.0, 1.0)
        label = "face center (0,0,1)"

    run_comparison(center_xyz, label, N=args.N,
                   output_times=args.times, outdir=args.outdir)


if __name__ == "__main__":
    main()
