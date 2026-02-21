"""
generate_reference_solutions.py — Exact spectral solutions for Gaussian hill tests
===================================================================================

Generates exact h-field reference solutions on the cubed-sphere h-grid
using Legendre polynomial expansion of the linearized SWE (f=0).

Shashkin Eq. 84:  h(t=0) = exp(-16 θ²/a²),  a = 1 (unit sphere)
                  v(t=0) = 0
                  c = √(gH₀) = 1

Variant 1: center at panel center (0, 0, 1) — face 0 center
Variant 2: center at cube vertex (1/√3, 1/√3, 1/√3) — face 0 corner

Output:
    reference_solutions/gauss{1,2}/N{024,048,...}.zarr
    Each contains:
        h_exact : (6, N+1, N+1) float64 — exact h at t=T_end
    Attributes:
        N, T_end, sigma, amp, center_xyz, L_max, L_eff, ic_recon_err

Usage:
    python generate_reference_solutions.py

No GPU required — pure numpy.
"""
import os
import sys
import numpy as np
import zarr
import time

# Import the spectral solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_gaussian_wave import ExactGaussianWave


# ============================================================
# Equiangular → Cartesian (numpy version, no JAX dependency)
# ============================================================

def equiangular_to_cartesian_np(xi1, xi2, face_id):
    """
    Convert equiangular coordinates to Cartesian on unit sphere.
    Same as grid.py but pure numpy.
    """
    t1 = np.tan(xi1)
    t2 = np.tan(xi2)
    d = np.sqrt(1.0 + t1**2 + t2**2)

    if face_id == 0:    # +Z
        X, Y, Z = t1/d, t2/d, 1/d
    elif face_id == 1:  # +Y
        X, Y, Z = -t1/d, 1/d, t2/d
    elif face_id == 2:  # -X
        X, Y, Z = -1/d, -t1/d, t2/d
    elif face_id == 3:  # -Y
        X, Y, Z = t1/d, -1/d, t2/d
    elif face_id == 4:  # +X
        X, Y, Z = 1/d, t1/d, t2/d
    elif face_id == 5:  # -Z
        X, Y, Z = -t1/d, t2/d, -1/d
    else:
        raise ValueError(f"Invalid face_id: {face_id}")

    return X, Y, Z


# ============================================================
# Generate reference for one (variant, N)
# ============================================================

def generate_reference(N, variant, sol, T_end, output_dir):
    """
    Evaluate exact solution on the cubed-sphere h-grid and save to zarr.

    Parameters
    ----------
    N : int — grid resolution
    variant : int — 1 or 2
    sol : ExactGaussianWave — spectral solution object
    T_end : float — evaluation time
    output_dir : str — output directory
    """
    pi4 = np.pi / 4
    xi_v = np.linspace(-pi4, pi4, N + 1)
    xi1_2d, xi2_2d = np.meshgrid(xi_v, xi_v, indexing='ij')

    h_exact = np.zeros((6, N + 1, N + 1))

    for p in range(6):
        X, Y, Z = equiangular_to_cartesian_np(xi1_2d, xi2_2d, p)
        h_exact[p] = sol.evaluate_cartesian(X, Y, Z, T_end)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"N{N:03d}.zarr")
    store = zarr.open(path, mode='w', zarr_format=2)
    store['h_exact'] = h_exact
    store.attrs['N'] = N
    store.attrs['T_end'] = T_end
    store.attrs['variant'] = variant
    store.attrs['sigma'] = sol.sigma
    store.attrs['amp'] = sol.amp
    store.attrs['center_xyz'] = list(sol.center)
    store.attrs['L_max'] = sol.L_max
    store.attrs['c'] = sol.c
    store.attrs['a'] = sol.a

    return h_exact, path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("  Generate Exact Reference Solutions")
    print("  Shashkin Gaussian Hill Tests (Eq. 84)")
    print("=" * 65)

    # Shashkin parameters
    sigma = 1.0 / (4.0 * np.sqrt(2.0))   # exp(-16θ²) = exp(-θ²/(2σ²))
    amp = 1.0
    T_end = 25.0
    c = 1.0
    a = 1.0
    L_max = 300
    n_quad = 600

    Ns = [24, 48, 96, 192, 384]

    # Variant 1: center at panel center (0, 0, 1)
    center1 = (0.0, 0.0, 1.0)

    # Variant 2: center at cube vertex (1/√3, 1/√3, 1/√3)
    center2 = tuple((np.array([1., 1., 1.]) / np.sqrt(3.0)).tolist())

    for variant, center in [(1, center1), (2, center2)]:
        print(f"\n{'='*65}")
        print(f"  Variant {variant}: center = ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"{'='*65}")

        sol = ExactGaussianWave(amp=amp, sigma=sigma, c=c, a=a,
                                 center_xyz=center, L_max=L_max, n_quad=n_quad)

        # Verify IC reconstruction
        theta_test = np.linspace(0, np.pi, 1000)
        h_ic = amp * np.exp(-16.0 * theta_test**2)
        h_recon = sol.evaluate(theta_test, t=0.0)
        ic_err = np.max(np.abs(h_ic - h_recon))
        print(f"  IC reconstruction error: {ic_err:.2e}")

        outdir = os.path.join("reference_solutions", f"gauss{variant}")

        for N in Ns:
            t0 = time.time()
            h_exact, path = generate_reference(N, variant, sol, T_end, outdir)
            elapsed = time.time() - t0

            # Stats
            max_h = np.max(np.abs(h_exact))
            sum_h = np.sum(h_exact)
            print(f"  N={N:3d}: shape={h_exact.shape}, max|h|={max_h:.6e}, "
                  f"sum={sum_h:.6e}, time={elapsed:.1f}s → {path}")

    # Summary
    print(f"\n{'='*65}")
    print(f"  Reference solutions saved to reference_solutions/")
    print(f"{'='*65}")

    # Verify round-trip
    print(f"\n  Round-trip verification:")
    for variant in [1, 2]:
        for N in [48, 192]:
            path = os.path.join("reference_solutions", f"gauss{variant}", f"N{N:03d}.zarr")
            store = zarr.open(path, mode='r')
            h = np.array(store['h_exact'])
            attrs = dict(store.attrs)
            print(f"    gauss{variant}/N{N:03d}: shape={h.shape}, "
                  f"T_end={attrs['T_end']}, sigma={attrs['sigma']:.6f}, "
                  f"max|h|={np.max(np.abs(h)):.6e}")


if __name__ == "__main__":
    main()
