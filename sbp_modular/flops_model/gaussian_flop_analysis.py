"""
flop_analysis.py — FLOP accounting for SBP-SAT cubed-sphere linearized SWE
============================================================================

Counts every floating-point operation in the RHS evaluation, parameterized
by N (cells per panel edge) and s (interior stencil width).

RHS structure (Shashkin Eq. 50, f=0):
  1. project_h         — edge/corner averaging
  2. gradient (2 matmuls)   — Dvc @ h, h @ Dvc.T
  3. contravariant (4 matmuls) — Pcv.T, Pvc, Pcv, Pvc.T interleaved with metrics
  4. mass flux          — J * v_contra (elementwise)
  5. divergence (2 matmuls)   — Dcv @ u1, u2 @ Dcv.T
  6. SAT (12 edges)     — flux extrapolation + penalty
  7. project_h          — again on divergence result
  8. dh/dt              — elementwise

Usage:
    python flop_analysis.py
"""
import numpy as np


def flops_per_rhs(N, s=4, P=6):
    """
    Count FLOPs per RHS evaluation.

    Parameters
    ----------
    N : int — cells per panel edge
    s : int — interior stencil width (4 for SBP 4/2, 6 for SBP 6/3)
    P : int — number of panels (6 for cubed sphere)

    Returns dict with breakdown and totals for dense and sparse.
    """
    M = N + 1  # vertices per panel edge

    # ================================================================
    # MATMULS: 8 operator applications, all have the same flop count
    # ================================================================
    #
    #   Op(R, C) applied to X(P, C, K) via einsum:
    #     Dense:  2 * P * R * C * K
    #     Sparse: 2 * P * s * R * K  (interior, s nonzeros per row)
    #
    # All 8 matmuls have dimensions where R*C*K = N*M*M or M*N*M:
    #
    #   Gradient:
    #     Dvc(N,M) @ h(P,M,M)         → 2P·N·M·M
    #     h(P,M,M) @ Dvc(N,M).T       → 2P·M·M·N
    #   Contravariant:
    #     v2(P,M,N) @ Pcv(M,N).T      → 2P·M·N·M
    #     Pvc(N,M) @ cross(P,M,M)     → 2P·N·M·M
    #     Pcv(M,N) @ v1(P,N,M)        → 2P·M·N·M
    #     cross(P,M,M) @ Pvc(N,M).T   → 2P·M·M·N
    #   Divergence:
    #     Dcv(M,N) @ u1(P,N,M)        → 2P·M·N·M
    #     u2(P,M,N) @ Dcv(M,N).T      → 2P·M·N·M
    #
    # Every one = 2P·N·M²
    # ================================================================

    n_matmuls = 8
    matmul_dense_each = 2 * P * N * M * M
    matmul_sparse_each = 2 * P * s * N * M  # s instead of M

    matmul_dense = n_matmuls * matmul_dense_each
    matmul_sparse = n_matmuls * matmul_sparse_each

    # ================================================================
    # ELEMENTWISE operations
    # ================================================================
    #
    # Contravariant velocity (both components):
    #   JQ12 * v_at_h:         2 × P·M²          (multiply)
    #   cross / J:             2 × P·N·M          (divide)
    #   Q * v:                 2 × P·N·M          (multiply)
    #   v_contra = sum:        2 × P·N·M          (add)
    # Mass flux:
    #   J * v_contra:          2 × P·N·M          (multiply)
    # Divergence:
    #   u1_term + u2_term:     P·M²               (add)
    # Continuity:
    #   Jh_inv * div:          P·M²               (multiply)
    #   -H0 * result:          P·M²               (multiply)
    # ================================================================

    elem_contra = 2 * P * M * M + 2 * 3 * P * N * M   # 2P·M² + 6P·N·M per component, ×2
    elem_contra_total = 2 * (2 * P * M * M + 3 * P * N * M)
    elem_flux = 2 * P * N * M
    elem_div = P * M * M
    elem_cont = 2 * P * M * M
    elementwise = elem_contra_total + elem_flux + elem_div + elem_cont

    # ================================================================
    # SAT (12 edges)
    # ================================================================
    #
    # Per edge:
    #   2 flux extrapolations: einsum('c,cj->j', r(N), u(N,M)) → 2N·M each
    #   SAT arithmetic: ~6M per side × 2 sides
    # ================================================================

    n_edges = 12
    sat_extrap = n_edges * 2 * (2 * N * M)          # 48NM
    sat_arith = n_edges * 2 * 6 * M                   # 144M
    sat_total = sat_extrap + sat_arith

    # ================================================================
    # PROJECTION (2 calls per RHS)
    # ================================================================
    #
    # 12 edges: extract + average + write: ~3M per edge
    # 8 corners: ~6 flops each
    # ================================================================

    proj_total = 2 * (n_edges * 3 * M + 8 * 6)

    # ================================================================
    # TOTALS
    # ================================================================

    dense_total = matmul_dense + elementwise + sat_total + proj_total
    sparse_total = matmul_sparse + elementwise + sat_total + proj_total

    return {
        'N': N, 's': s, 'P': P, 'M': M,
        'matmul_dense': matmul_dense,
        'matmul_sparse': matmul_sparse,
        'elementwise': elementwise,
        'sat': sat_total,
        'projection': proj_total,
        'dense_total': dense_total,
        'sparse_total': sparse_total,
        'waste_fraction': 1.0 - matmul_sparse / matmul_dense,
        'matmul_fraction_dense': matmul_dense / dense_total,
    }


def flops_rk4_step(N, s=4, P=6):
    """FLOPs per RK4 step = 4 RHS + stage arithmetic."""
    rhs = flops_per_rhs(N, s, P)
    M = N + 1

    # RK4 stage updates: 3 fields × (scale + accumulate) per stage
    # h(P,M,M), v1(P,N,M), v2(P,M,N)
    field_size = P * (M*M + N*M + M*N)  # total elements across 3 fields
    # 4 intermediate updates (h + α·dt·k): 2 flops/element (multiply + add)
    # 1 final combine (h + dt/6·(k1+2k2+2k3+k4)): ~10 flops/element
    rk4_overhead = (4 * 2 + 10) * field_size

    return {
        'rhs': rhs,
        'rk4_overhead': rk4_overhead,
        'dense_total': 4 * rhs['dense_total'] + rk4_overhead,
        'sparse_total': 4 * rhs['sparse_total'] + rk4_overhead,
    }


def flops_convergence_run(N, T=25.0, s=4, P=6):
    """FLOPs for full convergence run. dt = 1/(3N) per Shashkin Table 1."""
    dt = 1.0 / (3 * N)
    nsteps = int(np.ceil(T / dt))
    step = flops_rk4_step(N, s, P)

    return {
        'N': N, 'nsteps': nsteps, 'dt': dt,
        'dense_total': nsteps * step['dense_total'],
        'sparse_total': nsteps * step['sparse_total'],
        'step': step,
    }


# ================================================================
# Summary formulas (leading terms for large N)
# ================================================================
#
#   F_matmul_dense  = 16P · N · (N+1)²        per RHS
#   F_matmul_sparse = 16P · s · N · (N+1)      per RHS
#   F_elem + SAT    ≈ 21P · N · (N+1)          per RHS
#
#   Ratio: dense_matmul / sparse_matmul = (N+1) / s
#
#   Per RK4 step:
#     F_dense  = 4 · 16P · N(N+1)² + O(N²)   = 64P · N³ + O(N²)
#     F_sparse = 4 · 16Ps · N(N+1) + O(N²)    = 64Ps · N² + O(N²)
#
#   Per convergence run (nsteps = 3TN):
#     F_dense  = 192PT · N⁴ + O(N³)            O(N⁴) scaling
#     F_sparse = 192PTs · N³ + O(N³)            O(N³) scaling
#
#   For P=6, s=4, T=25:
#     F_dense  ≈ 28800 · N⁴
#     F_sparse ≈ 28800 · 4 · N³ = 115200 · N³
# ================================================================


if __name__ == "__main__":
    print("=" * 75)
    print("  FLOP Analysis: SBP-SAT Cubed-Sphere Linearized SWE (f=0)")
    print("=" * 75)

    # --- Per-RHS breakdown ---
    print("\n  Per RHS evaluation (s=4 for SBP 4/2, P=6 panels):")
    print(f"  {'N':>5} {'matmul(D)':>12} {'matmul(S)':>12} {'elem':>10} "
          f"{'SAT':>10} {'total(D)':>12} {'total(S)':>12} {'waste':>6}")
    print(f"  {'-'*75}")

    for N in [24, 48, 96, 192, 384, 768]:
        r = flops_per_rhs(N, s=4)
        print(f"  {N:>5} {r['matmul_dense']:>12.3e} {r['matmul_sparse']:>12.3e} "
              f"{r['elementwise']:>10.3e} {r['sat']:>10.3e} "
              f"{r['dense_total']:>12.3e} {r['sparse_total']:>12.3e} "
              f"{r['waste_fraction']:>5.1%}")

    # --- Convergence run ---
    print(f"\n  Full convergence run (T=25, dt=1/(3N)):")
    print(f"  {'N':>5} {'nsteps':>8} {'dense TF':>10} {'sparse TF':>10} "
          f"{'ratio':>7} {'dense scaling':>14}")
    print(f"  {'-'*60}")

    prev_dense = None
    for N in [24, 48, 96, 192, 384, 768]:
        r = flops_convergence_run(N, T=25.0, s=4)
        d_tf = r['dense_total'] / 1e12
        s_tf = r['sparse_total'] / 1e12
        ratio = r['dense_total'] / r['sparse_total']

        scaling = ""
        if prev_dense is not None:
            scaling = f"{r['dense_total'] / prev_dense:.1f}x"
        prev_dense = r['dense_total']

        print(f"  {N:>5} {r['nsteps']:>8} {d_tf:>10.1f} {s_tf:>10.1f} "
              f"{ratio:>6.0f}x {scaling:>14}")

    # --- Verify against measured runtime ---
    print(f"\n  Verification against measured A30 runtime (N=384):")
    r384 = flops_convergence_run(384, T=25.0, s=4)
    runtime = 213.3
    achieved = r384['dense_total'] / runtime / 1e12
    peak = 5.2  # A30 FP64 TFlops
    print(f"    Dense FLOPs:  {r384['dense_total']/1e12:.0f} TFlop")
    print(f"    Runtime:      {runtime:.1f} s")
    print(f"    Throughput:   {achieved:.2f} TFlop/s ({achieved/peak:.0%} of {peak} TFlop/s A30 peak)")

    # --- Scaling comparison ---
    print(f"\n  Scaling from N=192 to N=384:")
    r192 = flops_convergence_run(192, T=25.0, s=4)
    r384 = flops_convergence_run(384, T=25.0, s=4)
    print(f"    Work ratio (dense):  {r384['dense_total']/r192['dense_total']:.1f}x  (theory: 16x for O(N⁴))")
    print(f"    Measured runtime:    192→52.7s, 384→213.3s = {213.3/52.7:.1f}x")
    print(f"    → GPU not saturated at N=192 (launch overhead dominates)")

    # --- Effect of SBP order ---
    print(f"\n  Effect of operator order (N=384):")
    for label, s in [("SBP 2/1 (s=2)", 2), ("SBP 4/2 (s=4)", 4), ("SBP 6/3 (s=6)", 6)]:
        r = flops_per_rhs(384, s=s)
        print(f"    {label}: dense={r['dense_total']:.3e}, "
              f"sparse={r['sparse_total']:.3e}, "
              f"waste={r['waste_fraction']:.1%}")

    # --- Leading-order formulas ---
    print(f"\n  Leading-order formulas (P=6, SBP 4/2, s=4):")
    print(f"    Per RHS:     F_dense ≈ 96·N·(N+1)²    F_sparse ≈ 96·4·N·(N+1)")
    print(f"    Per RK4:     F_dense ≈ 384·N³          F_sparse ≈ 1536·N² + 684·N²")
    print(f"    Per run:     F_dense ≈ 28800·N⁴        F_sparse ≈ 166500·N³")
    print(f"    Waste ratio: (N+1)/s ≈ N/4")

    print()
