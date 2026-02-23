"""
visualize_exact_solution.py — Exact spectral solution on cubed sphere
=====================================================================

Plots h(x,y,z,t) on all 6 panels at specified time snapshots.
Uses ExactGaussianWave Legendre expansion (pure CPU, no JAX needed).

Usage:
    python visualize_exact_solution.py                      # face center
    python visualize_exact_solution.py --center vertex       # cube vertex
    python visualize_exact_solution.py --center 0.3 0.3 0   # custom (xi1,xi2,panel)
    python visualize_exact_solution.py --N 96 --times 0 0.25 0.5 1.0
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_gaussian_wave import ExactGaussianWave


# ============================================================
# Equiangular → Cartesian (numpy, no JAX)
# ============================================================

FACE_LABELS = [
    "Face 0 (+Z, North)",
    "Face 1 (+Y)",
    "Face 2 (-X)",
    "Face 3 (-Y)",
    "Face 4 (+X)",
    "Face 5 (-Z, South)",
]

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


# ============================================================
# Build h on all 6 panels
# ============================================================

def evaluate_all_panels(sol, N, t):
    """Evaluate exact h on all 6 panels at time t.
    
    Returns:
        h: (6, N, N) array
        xi1_2d, xi2_2d: (N, N) coordinate arrays
    """
    pi4 = np.pi / 4
    xi = np.linspace(-pi4, pi4, N)
    xi1_2d, xi2_2d = np.meshgrid(xi, xi, indexing='ij')

    h = np.zeros((6, N, N))
    for p in range(6):
        X, Y, Z = equiangular_to_cartesian_np(xi1_2d, xi2_2d, p)
        h[p] = sol.evaluate_cartesian(X, Y, Z, t)

    return h, xi1_2d, xi2_2d


# ============================================================
# Plot 6 panels for one time
# ============================================================

def plot_six_panels(h, xi1_2d, xi2_2d, t, title_prefix, vmin, vmax,
                    center_label=""):
    """
    Plot h on all 6 panels as a 2×3 grid.
    
    Returns matplotlib figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for p in range(6):
        ax = axes[p]
        xi1_deg = np.degrees(xi1_2d)
        xi2_deg = np.degrees(xi2_2d)

        im = ax.pcolormesh(xi1_deg, xi2_deg, h[p],
                           cmap='RdBu_r', vmin=vmin, vmax=vmax,
                           shading='auto')
        ax.set_title(FACE_LABELS[p], fontsize=11)
        ax.set_xlabel('ξ₁ (deg)')
        ax.set_ylabel('ξ₂ (deg)')
        ax.set_aspect('equal')

    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(im, cax=cbar_ax, label='h perturbation')

    fig.suptitle(f"{title_prefix}  t = {t:.2f} days  {center_label}",
                 fontsize=14, fontweight='bold')
    return fig


# ============================================================
# Time series visualization
# ============================================================

def visualize_time_series(center_xyz, times, N=96, prefix="exact",
                          center_label=""):
    """
    Generate h-field plots at multiple times.
    
    Args:
        center_xyz: (X0, Y0, Z0) Gaussian center on unit sphere
        times: list of times (days)
        N: grid resolution for plotting
        prefix: filename prefix
        center_label: string label for plot title
    """
    # Shashkin parameters
    sigma = 1.0 / (4.0 * np.sqrt(2.0))
    amp = 1.0

    print(f"Creating ExactGaussianWave:")
    print(f"  center = ({center_xyz[0]:.4f}, {center_xyz[1]:.4f}, {center_xyz[2]:.4f})")
    print(f"  sigma = {sigma:.6f}, amp = {amp}")
    print(f"  {center_label}")

    sol = ExactGaussianWave(amp=amp, sigma=sigma, c=1.0, a=1.0,
                             center_xyz=center_xyz, L_max=300, n_quad=600)

    # Evaluate at all times to get global colorbar range
    all_h = []
    for t in times:
        h, xi1_2d, xi2_2d = evaluate_all_panels(sol, N, t)
        all_h.append(h)

    vmax = max(np.max(np.abs(h)) for h in all_h)
    vmin = -vmax
    print(f"  Global color range: [{vmin:.4e}, {vmax:.4e}]")

    # Plot each time
    filenames = []
    for i, (t, h) in enumerate(zip(times, all_h)):
        print(f"  Plotting t = {t:.3f} days ...", end='')
        fig = plot_six_panels(h, xi1_2d, xi2_2d, t,
                              "Exact Spectral Solution",
                              vmin, vmax, center_label)

        fname = f"{prefix}_t{t:.3f}.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f" → {fname}")
        filenames.append(fname)

    # Also make a combined time strip (all times in one figure)
    n_times = len(times)
    fig_strip, axes = plt.subplots(n_times, 6, figsize=(24, 4*n_times))
    if n_times == 1:
        axes = axes[np.newaxis, :]

    for i, (t, h) in enumerate(zip(times, all_h)):
        xi1_deg = np.degrees(xi1_2d)
        xi2_deg = np.degrees(xi2_2d)
        for p in range(6):
            ax = axes[i, p]
            ax.pcolormesh(xi1_deg, xi2_deg, h[p],
                          cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          shading='auto')
            if i == 0:
                ax.set_title(FACE_LABELS[p], fontsize=9)
            if p == 0:
                ax.set_ylabel(f't={t:.2f}d', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)

    fig_strip.suptitle(f"Exact Solution Time Evolution  {center_label}",
                       fontsize=14, fontweight='bold')
    fig_strip.tight_layout()

    strip_fname = f"{prefix}_strip.png"
    fig_strip.savefig(strip_fname, dpi=150, bbox_inches='tight')
    plt.close(fig_strip)
    print(f"  Strip → {strip_fname}")
    filenames.append(strip_fname)

    return filenames


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize exact Gaussian wave on cubed sphere")
    parser.add_argument('--center', nargs='+', default=['face'],
                        help='Center: "face", "vertex", or xi1 xi2 panel')
    parser.add_argument('--N', type=int, default=96,
                        help='Grid resolution for plotting (default: 96)')
    parser.add_argument('--times', nargs='+', type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help='Times in days (default: 0 0.25 0.5 0.75 1.0)')
    parser.add_argument('--outdir', default='.',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(args.outdir)

    # Parse center
    if args.center[0] == 'face':
        # Face 0 center = north pole
        center_xyz = (0.0, 0.0, 1.0)
        prefix = "exact_face"
        label = "center=(0,0,1) [face 0 center]"
    elif args.center[0] == 'vertex':
        # Cube vertex
        s3 = 1.0 / np.sqrt(3.0)
        center_xyz = (s3, s3, s3)
        prefix = "exact_vertex"
        label = "center=(1,1,1)/√3 [cube vertex]"
    else:
        # Custom: xi1, xi2, panel
        xi1 = float(args.center[0])
        xi2 = float(args.center[1])
        panel = int(args.center[2])
        X, Y, Z = equiangular_to_cartesian_np(xi1, xi2, panel)
        center_xyz = (float(X), float(Y), float(Z))
        prefix = f"exact_xi{xi1:.2f}_{xi2:.2f}_p{panel}"
        label = f"center=({X:.4f},{Y:.4f},{Z:.4f}) [ξ=({xi1:.2f},{xi2:.2f}) panel {panel}]"

    print("=" * 65)
    print("  Exact Spectral Solution Visualization")
    print("=" * 65)
    print(f"  Center: {label}")
    print(f"  N = {args.N}")
    print(f"  Times = {args.times} days")
    print(f"  Output: {os.getcwd()}")
    print("=" * 65)

    filenames = visualize_time_series(
        center_xyz, args.times, N=args.N,
        prefix=prefix, center_label=label)

    print(f"\n  Generated {len(filenames)} files.")


if __name__ == "__main__":
    main()
