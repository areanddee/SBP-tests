"""
test_sbp_vs_exact_snapshots.py — SBP vs Exact at multiple output times
======================================================================

Runs the SBP-SAT solver and compares against exact Legendre spectral
solution at short time intervals (0 to 1 day).  Produces per-time
6-panel comparison plots (SBP / Exact / Difference) at N=48 and N=96.

This is a DEBUGGING tool: short integration reveals WHERE errors grow
(edges, corners, interior) before the wave wraps the sphere.

CRITICAL: Both SBP and spectral use identical Shashkin IC parameters:
  sigma = 1/(4√2) ≈ 0.177, amp = 1.0

Usage (on GPU node):
    python test_sbp_vs_exact_snapshots.py --center face
    python test_sbp_vs_exact_snapshots.py --center vertex
    python test_sbp_vs_exact_snapshots.py --center face --Ns 48 96

Requires: JAX + GPU, exact_gaussian_wave.py in same directory
"""
import sys
import os
import time as _time
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from operators import sbp_42
from grid import equiangular_to_cartesian
from sat_operators import build_cartesian_sat_fn

# Exact solution (pure numpy, no JAX)
from exact_gaussian_wave import ExactGaussianWave


# ============================================================
# Edge connectivity (Rich's pattern)
# ============================================================

EDGES = [
    (0, 'N', 1, 'N', 'R'),
    (0, 'E', 4, 'N', 'T'),
    (0, 'W', 2, 'N', 'TR'),
    (0, 'S', 3, 'N', 'N'),
    (1, 'E', 2, 'W', 'N'),
    (1, 'S', 5, 'N', 'N'),
    (1, 'W', 4, 'E', 'N'),
    (2, 'E', 3, 'W', 'N'),
    (2, 'S', 5, 'E', 'TR'),
    (3, 'E', 4, 'W', 'N'),
    (3, 'S', 5, 'S', 'R'),
    (4, 'S', 5, 'W', 'T'),
]


def _reverses(op):
    return op in ('R', 'TR')


def _get_h_boundary(h_panel, edge, N):
    if edge == 'N':   return h_panel[:, N]
    elif edge == 'S': return h_panel[:, 0]
    elif edge == 'E': return h_panel[N, :]
    elif edge == 'W': return h_panel[0, :]


def _set_h_boundary(h, panel, edge, vals, N):
    if edge == 'N':   return h.at[panel, :, N].set(vals)
    elif edge == 'S': return h.at[panel, :, 0].set(vals)
    elif edge == 'E': return h.at[panel, N, :].set(vals)
    elif edge == 'W': return h.at[panel, 0, :].set(vals)


def _edge_k_to_ij(edge, k, N):
    if edge == 'N':   return (k, N)
    elif edge == 'S': return (k, 0)
    elif edge == 'E': return (N, k)
    elif edge == 'W': return (0, k)


def _hv_inv_index(edge, N):
    return N if edge in ('E', 'N') else 0


def _boundary_sign(edge):
    return +1.0 if edge in ('E', 'N') else -1.0


# ============================================================
# Grids & metrics (from test_stag_step5_Nconv.py)
# ============================================================

def compute_metric(xi1, xi2):
    t1 = jnp.tan(xi1)
    t2 = jnp.tan(xi2)
    c1 = jnp.cos(xi1)
    c2 = jnp.cos(xi2)
    r2 = 1.0 + t1**2 + t2**2
    r = jnp.sqrt(r2)
    J = 1.0 / (r**3 * c1**2 * c2**2)
    alpha = r**4 * c1**2 * c2**2
    Q11 = alpha * (1.0 - t1**2 / r2)
    Q12 = alpha * (-t1 * t2 / r2)
    Q22 = alpha * (1.0 - t2**2 / r2)
    return J, Q11, Q12, Q22


def make_staggered_grids(N):
    L = jnp.pi / 2
    dx = L / N
    xi_v = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, N + 1)
    xi_c = (jnp.arange(N) + 0.5) * dx - jnp.pi / 4
    xi1_h, xi2_h = jnp.meshgrid(xi_v, xi_v, indexing='ij')
    xi1_v1, xi2_v1 = jnp.meshgrid(xi_c, xi_v, indexing='ij')
    xi1_v2, xi2_v2 = jnp.meshgrid(xi_v, xi_c, indexing='ij')
    return {
        'xi_v': xi_v, 'xi_c': xi_c, 'dx': dx,
        'xi1_h': xi1_h, 'xi2_h': xi2_h,
        'xi1_v1': xi1_v1, 'xi2_v1': xi2_v1,
        'xi1_v2': xi1_v2, 'xi2_v2': xi2_v2,
    }


def make_all_metrics(grids):
    Jh, _, Q12_h, _ = compute_metric(grids['xi1_h'], grids['xi2_h'])
    J1, Q11_1, _, _ = compute_metric(grids['xi1_v1'], grids['xi2_v1'])
    J2, _, _, Q22_2 = compute_metric(grids['xi1_v2'], grids['xi2_v2'])
    return {
        'Jh': Jh, 'J1': J1, 'J2': J2,
        'Q11_1': Q11_1, 'Q12_h': Q12_h, 'Q22_2': Q22_2,
    }


# ============================================================
# Contravariant velocity
# ============================================================

def compute_contravariant(v1, v2, metrics, Pvc, Pcv):
    Q11_1 = metrics['Q11_1']
    Q12_h = metrics['Q12_h']
    Q22_2 = metrics['Q22_2']
    Jh = metrics['Jh']
    J1 = metrics['J1']
    J2 = metrics['J2']
    JQ12 = Jh * Q12_h

    v2_at_h = v2 @ Pcv.T
    cross_at_h = JQ12 * v2_at_h
    cross_at_v1 = Pvc @ cross_at_h
    v1_contra = Q11_1 * v1 + cross_at_v1 / J1

    v1_at_h = Pcv @ v1
    cross_at_h2 = JQ12 * v1_at_h
    cross_at_v2 = cross_at_h2 @ Pvc.T
    v2_contra = cross_at_v2 / J2 + Q22_2 * v2

    return v1_contra, v2_contra


# ============================================================
# h-projection (Eq. 51-52)
# ============================================================

def build_projection_fn(N, Jh, Hv_diag):
    corner_map = {}
    for pa, ea, pb, eb, op in EDGES:
        rev = _reverses(op)
        for k_a in [0, N]:
            k_b = (N - k_a) if rev else k_a
            ij_a = _edge_k_to_ij(ea, k_a, N)
            ij_b = _edge_k_to_ij(eb, k_b, N)
            key_a = (pa,) + ij_a
            key_b = (pb,) + ij_b
            if key_a not in corner_map:
                corner_map[key_a] = {key_a}
            if key_b not in corner_map:
                corner_map[key_b] = {key_b}
            corner_map[key_a].add(key_b)
            corner_map[key_b].add(key_a)

    changed = True
    while changed:
        changed = False
        for key in list(corner_map.keys()):
            group = set(corner_map[key])
            for member in list(group):
                if member in corner_map:
                    new = corner_map[member]
                    if not new.issubset(group):
                        group.update(new)
                        changed = True
            corner_map[key] = group

    seen = set()
    corner_groups = []
    for key, group in corner_map.items():
        frozen = frozenset(group)
        if frozen not in seen and len(group) == 3:
            seen.add(frozen)
            corner_groups.append(sorted(group))

    def project_h(h):
        h_orig = h
        for pa, ea, pb, eb, op in EDGES:
            rev = _reverses(op)
            bnd_a = _get_h_boundary(h_orig[pa], ea, N)
            bnd_b = _get_h_boundary(h_orig[pb], eb, N)
            if rev:
                bnd_b = bnd_b[::-1]
            avg = 0.5 * (bnd_a + bnd_b)
            h = _set_h_boundary(h, pa, ea, avg, N)
            avg_b = avg[::-1] if rev else avg
            h = _set_h_boundary(h, pb, eb, avg_b, N)

        for group in corner_groups:
            vals = jnp.array([h_orig[p, i, j] for p, i, j in group])
            avg = jnp.mean(vals)
            for p, i, j in group:
                h = h.at[p, i, j].set(avg)
        return h

    return project_h, corner_groups


# ============================================================
# Build full SWE system
# ============================================================

def make_cubed_sphere_swe(N, H0, g):
    grids = make_staggered_grids(N)
    metrics = make_all_metrics(grids)
    dx = float(grids['dx'])

    ops = sbp_42(N, dx)
    Dvc = ops.Dvc
    Dcv = ops.Dcv
    Pvc = ops.Pvc
    Pcv = ops.Pcv
    Hv_diag = jnp.diag(ops.Hv)
    Hc_diag = jnp.diag(ops.Hc)

    project_h, corners = build_projection_fn(N, metrics['Jh'], Hv_diag)
    add_sat = build_cartesian_sat_fn(N, ops, compute_metric)

    Jh = metrics['Jh']
    J1 = metrics['J1']
    J2 = metrics['J2']
    Jh_inv = 1.0 / Jh

    Wh = jnp.outer(Hv_diag, Hv_diag)
    W1 = jnp.outer(Hc_diag, Hv_diag)
    W2 = jnp.outer(Hv_diag, Hc_diag)

    def rhs(h, v1, v2):
        h_proj = project_h(h)
        dv1_dt = -g * jnp.einsum('ij,pjk->pik', Dvc, h_proj)
        dv2_dt = -g * jnp.einsum('pij,kj->pik', h_proj, Dvc)

        v1c, v2c = _contra_vmap(v1, v2)
        u1_all = J1 * v1c
        u2_all = J2 * v2c

        div = (jnp.einsum('ij,pjk->pik', Dcv, u1_all) +
               jnp.einsum('pij,kj->pik', u2_all, Dcv))

        div = add_sat(div, u1_all, u2_all, v1, v2)
        dh_dt = project_h(-H0 * Jh_inv * div)
        return dh_dt, dv1_dt, dv2_dt

    def _contra_single(v1_p, v2_p):
        return compute_contravariant(v1_p, v2_p, metrics, Pvc, Pcv)
    _contra_vmap = jax.vmap(_contra_single)

    rhs_jit = jax.jit(rhs)

    return {
        'rhs': rhs_jit,
        'project_h': project_h,
        'grids': grids,
        'metrics': metrics,
        'ops': ops,
        'corners': corners,
        'Wh': Wh, 'W1': W1, 'W2': W2,
        'Jh': Jh, 'J1': J1, 'J2': J2,
        'Pvc': Pvc, 'Pcv': Pcv,
        'N': N, 'dx': dx,
    }


# ============================================================
# RK4
# ============================================================

def make_rk4_step(rhs_fn):
    @jax.jit
    def step(h, v1, v2, dt):
        k1h, k1v1, k1v2 = rhs_fn(h, v1, v2)
        k2h, k2v1, k2v2 = rhs_fn(h + 0.5*dt*k1h, v1 + 0.5*dt*k1v1,
                                   v2 + 0.5*dt*k1v2)
        k3h, k3v1, k3v2 = rhs_fn(h + 0.5*dt*k2h, v1 + 0.5*dt*k2v1,
                                   v2 + 0.5*dt*k2v2)
        k4h, k4v1, k4v2 = rhs_fn(h + dt*k3h, v1 + dt*k3v1,
                                   v2 + dt*k3v2)
        return (h  + (dt/6)*(k1h  + 2*k2h  + 2*k3h  + k4h),
                v1 + (dt/6)*(k1v1 + 2*k2v1 + 2*k3v1 + k4v1),
                v2 + (dt/6)*(k1v2 + 2*k2v2 + 2*k3v2 + k4v2))
    return step


# ============================================================
# Diagnostics
# ============================================================

def compute_mass(h, Wh, Jh):
    return float(jnp.sum(h * Jh[None, :, :] * Wh[None, :, :]))


# ============================================================
# Exact solution on cubed-sphere grids (numpy)
# ============================================================

def equiangular_to_cartesian_np(xi1, xi2, face_id):
    t1 = np.tan(xi1)
    t2 = np.tan(xi2)
    d = np.sqrt(1.0 + t1**2 + t2**2)
    if face_id == 0:   X, Y, Z = t1/d, t2/d, 1/d
    elif face_id == 1: X, Y, Z = -t1/d, 1/d, t2/d
    elif face_id == 2: X, Y, Z = -1/d, -t1/d, t2/d
    elif face_id == 3: X, Y, Z = t1/d, -1/d, t2/d
    elif face_id == 4: X, Y, Z = 1/d, t1/d, t2/d
    elif face_id == 5: X, Y, Z = -t1/d, t2/d, -1/d
    return X, Y, Z


FACE_LABELS = [
    "Face 0 (+Z)", "Face 1 (+Y)", "Face 2 (-X)",
    "Face 3 (-Y)", "Face 4 (+X)", "Face 5 (-Z)",
]


def exact_on_panels(sol, N, t):
    """Evaluate exact h on vertex grid (N+1 × N+1) for all 6 panels."""
    pi4 = np.pi / 4
    xi_v = np.linspace(-pi4, pi4, N + 1)
    xi1_2d, xi2_2d = np.meshgrid(xi_v, xi_v, indexing='ij')
    h_exact = np.zeros((6, N + 1, N + 1))
    for p in range(6):
        X, Y, Z = equiangular_to_cartesian_np(xi1_2d, xi2_2d, p)
        h_exact[p] = sol.evaluate_cartesian(X, Y, Z, t)
    return h_exact


# ============================================================
# Plotting
# ============================================================

def plot_comparison(h_sbp, h_exact, t, N, l2, linf, center_label, outfile):
    """3-row (SBP / Exact / Diff) × 6-col comparison."""
    h_sbp_np = np.array(h_sbp)
    diff = h_sbp_np - h_exact

    fig, axes = plt.subplots(3, 6, figsize=(24, 12))

    vmax_sol = max(np.max(np.abs(h_sbp_np)), np.max(np.abs(h_exact)))
    if vmax_sol < 1e-12:
        vmax_sol = 1.0
    vmax_diff = np.max(np.abs(diff))
    if vmax_diff < 1e-12:
        vmax_diff = 1.0

    pi4 = np.pi / 4
    xi_v = np.linspace(-pi4, pi4, N + 1)
    xi1_2d, xi2_2d = np.meshgrid(xi_v, xi_v, indexing='ij')
    xi1_deg = np.degrees(xi1_2d)
    xi2_deg = np.degrees(xi2_2d)

    row_labels = ['SBP', 'Exact', 'Difference']
    row_data = [h_sbp_np, h_exact, diff]
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
        f"SBP vs Exact  t = {t:.3f} days  N = {N}\n"
        f"L2 = {l2:.4e}   L∞ = {linf:.4e}   {center_label}",
        fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_error_growth(times_list, results_by_N, center_label, outfile):
    """Plot L2 and Linf error vs time for all N values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for N, data in sorted(results_by_N.items()):
        ts = [d['t'] for d in data]
        l2s = [d['l2'] for d in data]
        linfs = [d['linf'] for d in data]
        ax1.semilogy(ts, l2s, 'o-', linewidth=2, markersize=5, label=f'N={N}')
        ax2.semilogy(ts, linfs, 's-', linewidth=2, markersize=5, label=f'N={N}')

    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('L2 error')
    ax1.set_title('L2 Error Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('L∞ error')
    ax2.set_title('L∞ Error Growth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Error Growth  {center_label}", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_combined_strip(all_snapshots, center_label, outfile):
    """
    Combined strip plot: rows = output times, 
    column groups = N values, each with 6 panels.
    """
    Ns = sorted(all_snapshots.keys())
    times = [d['t'] for d in all_snapshots[Ns[0]]]
    n_t = len(times)
    n_N = len(Ns)

    # 6 panels × n_N resolutions
    fig, axes = plt.subplots(n_t, 6 * n_N, figsize=(8 * n_N, 3.5 * n_t))
    if n_t == 1:
        axes = axes[np.newaxis, :]

    for col_group, N in enumerate(Ns):
        for row_i, snap in enumerate(all_snapshots[N]):
            diff = snap['h_sbp_np'] - snap['h_exact']
            vmax = max(np.max(np.abs(diff)), 1e-12)
            pi4 = np.pi / 4
            xi_v = np.linspace(-pi4, pi4, N + 1)
            xi1_2d, xi2_2d = np.meshgrid(xi_v, xi_v, indexing='ij')
            xi1d = np.degrees(xi1_2d)
            xi2d = np.degrees(xi2_2d)

            for p in range(6):
                col = col_group * 6 + p
                ax = axes[row_i, col]
                ax.pcolormesh(xi1d, xi2d, diff[p],
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              shading='auto')
                if row_i == 0:
                    label = FACE_LABELS[p] if col_group == 0 else f"P{p}"
                    ax.set_title(f"N={N} {label}", fontsize=7)
                if p == 0 and col_group == 0:
                    ax.set_ylabel(f"t={snap['t']:.2f}d\nL2={snap['l2']:.1e}",
                                  fontsize=8, fontweight='bold')
                ax.set_aspect('equal')
                ax.tick_params(labelsize=4)

    fig.suptitle(f"SBP − Exact Difference  {center_label}",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Core: run SBP with snapshot output
# ============================================================

def run_sbp_with_snapshots(N, center_xyz, output_times, sol,
                            sigma_ic, amp_ic, H0, g):
    """
    Run the SBP solver and capture h at specified output times.

    Returns list of dicts: {'t', 'h', 'mass_err'}
    """
    c = np.sqrt(g * H0)

    # Build system
    print(f"\n  Building SBP system N={N}...", end='', flush=True)
    sys_d = make_cubed_sphere_swe(N, H0, g)
    grids = sys_d['grids']
    dx = sys_d['dx']

    # Time step: dt = 1/(3N) per Shashkin Table 1
    dt = 1.0 / (N * 3)
    if args.nsteps is not None:
       N_ref = args.Ns[0]
       dt = 1.0 / (N_ref * 3)
       output_times = [i * dt for i in range(args.nsteps + 1)]
    elif args.times is not None:
       output_times = args.times
    else:
       output_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    CFL = c * dt / dx
    print(f" dx={dx:.4e}, dt={dt:.4e}, CFL={CFL:.4e}")

    # Gaussian IC (Shashkin parameters, matched to spectral)
    X0, Y0, Z0 = center_xyz
    xi_v = grids['xi_v']
    xi1_2d, xi2_2d = jnp.meshgrid(xi_v, xi_v, indexing='ij')

    h = jnp.zeros((6, N + 1, N + 1))
    for p in range(6):
        X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
        cos_d = jnp.clip(X * X0 + Y * Y0 + Z * Z0, -1.0, 1.0)
        d = jnp.arccos(cos_d)
        h = h.at[p].set(amp_ic * jnp.exp(-d**2 / (2 * sigma_ic**2)))

    v1 = jnp.zeros((6, N, N + 1))
    v2 = jnp.zeros((6, N + 1, N))

    mass0 = compute_mass(h, sys_d['Wh'], sys_d['Jh'])
    step_fn = make_rk4_step(sys_d['rhs'])

    # Sort output times
    output_times = sorted(output_times)
    T_end = max(output_times)
    nsteps_total = int(np.ceil(T_end / dt))
    dt = T_end / nsteps_total  # adjust to hit T_end exactly

    # JIT warmup
    print(f"  N={N}: JIT warmup...", end='', flush=True)
    t0 = _time.time()
    h, v1, v2 = step_fn(h, v1, v2, dt)
    jax.block_until_ready(h)
    print(f" {_time.time()-t0:.1f}s")

    current_step = 1
    current_time = dt

    # Collect snapshots
    snapshots = []

    # Check if t=0 is requested (use IC, step 0)
    if output_times[0] == 0.0:
        # Re-create IC (we already took one step for warmup)
        h_ic = jnp.zeros((6, N + 1, N + 1))
        for p in range(6):
            X, Y, Z = equiangular_to_cartesian(xi1_2d, xi2_2d, p)
            cos_d = jnp.clip(X * X0 + Y * Y0 + Z * Z0, -1.0, 1.0)
            d = jnp.arccos(cos_d)
            h_ic = h_ic.at[p].set(amp_ic * jnp.exp(-d**2 / (2 * sigma_ic**2)))
        snapshots.append({'t': 0.0, 'h': h_ic, 'mass_err': 0.0})
        output_times = output_times[1:]

    # Step to each output time
    for t_out in output_times:
        target_step = int(np.round(t_out / dt))

        print(f"  N={N}: stepping to t={t_out:.3f} "
              f"(steps {current_step}→{target_step})...", end='', flush=True)
        t0 = _time.time()
        while current_step < target_step:
            h, v1, v2 = step_fn(h, v1, v2, dt)
            current_step += 1
        jax.block_until_ready(h)
        current_time = current_step * dt
        run_t = _time.time() - t0
        print(f" {run_t:.1f}s")

        mass_f = compute_mass(h, sys_d['Wh'], sys_d['Jh'])
        mass_err = abs(mass_f - mass0)

        snapshots.append({
            't': current_time,
            'h': h,
            'mass_err': mass_err,
        })

    return snapshots, sys_d


# ============================================================
# Main driver
# ============================================================

def run_comparison(center_xyz, center_label, Ns=[48, 96],
                   output_times=None, outdir='.'):
    if output_times is None:
        output_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    os.makedirs(outdir, exist_ok=True)

    # Shashkin IC parameters (MUST match exact_gaussian_wave)
    sigma_ic = 1.0 / (4.0 * np.sqrt(2.0))   # 0.176777
    amp_ic = 1.0
    H0 = 1.0
    g = 1.0
    c = np.sqrt(g * H0)

    X0, Y0, Z0 = center_xyz

    print("=" * 65)
    print(f"  SBP vs Exact Snapshot Comparison")
    print(f"  Center: {center_label}")
    print(f"    Cartesian: ({X0:.6f}, {Y0:.6f}, {Z0:.6f})")
    print(f"  N values: {Ns}")
    print(f"  Output times: {output_times} days")
    print(f"  IC: sigma={sigma_ic:.6f}, amp={amp_ic}")
    print(f"  Output: {outdir}/")
    print("=" * 65)

    # Build exact spectral solution
    print("\n  Building exact spectral solution...")
    sol = ExactGaussianWave(amp=amp_ic, sigma=sigma_ic, c=c, a=1.0,
                             center_xyz=center_xyz, L_max=300, n_quad=600)

    # Run SBP at each N
    all_snapshots = {}   # N → list of snapshot dicts
    results_by_N = {}    # N → list of error dicts

    for N in Ns:
        snapshots, sys_d = run_sbp_with_snapshots(
            N, center_xyz, output_times, sol, sigma_ic, amp_ic, H0, g)

        Wh = sys_d['Wh']
        Jh = sys_d['Jh']

        # Compute errors at each snapshot
        error_list = []
        snap_data = []

        print(f"\n  N={N} error summary:")
        print(f"  {'t':>8} {'L2':>12} {'Linf':>12} {'mass_err':>12}")
        print(f"  {'-'*48}")

        for snap in snapshots:
            t = snap['t']
            h_sbp = snap['h']

            # Exact at this time
            h_exact = exact_on_panels(sol, N, t)

            # Compute error norms
            diff = np.array(h_sbp) - h_exact
            l2 = float(jnp.sqrt(jnp.sum(
                jnp.array(diff**2) * Jh[None] * Wh[None])))
            linf = float(np.max(np.abs(diff)))

            print(f"  {t:8.3f} {l2:12.4e} {linf:12.4e} "
                  f"{snap['mass_err']:12.2e}")

            error_list.append({'t': t, 'l2': l2, 'linf': linf})
            snap_data.append({
                't': t, 'l2': l2, 'linf': linf,
                'h_sbp_np': np.array(h_sbp),
                'h_exact': h_exact,
            })

            # Per-time comparison plot
            fname = os.path.join(outdir,
                f"compare_N{N:03d}_t{t:.3f}.png")
            plot_comparison(h_sbp, h_exact, t, N, l2, linf,
                            center_label, fname)
            print(f"    → {fname}")

        results_by_N[N] = error_list
        all_snapshots[N] = snap_data

    # Error growth plot
    fname = os.path.join(outdir, "error_growth.png")
    plot_error_growth(output_times, results_by_N, center_label, fname)
    print(f"\n  → {fname}")

    # Combined difference strip
    fname = os.path.join(outdir, "diff_strip.png")
    plot_combined_strip(all_snapshots, center_label, fname)
    print(f"  → {fname}")

    # Convergence rate at each time
    if len(Ns) >= 2:
        print(f"\n  Convergence rates (N={Ns[-2]}→{Ns[-1]}):")
        print(f"  {'t':>8} {'L2 rate':>10} {'Linf rate':>12}")
        print(f"  {'-'*32}")
        for i in range(len(output_times)):
            e0 = results_by_N[Ns[-2]][i]
            e1 = results_by_N[Ns[-1]][i]
            t = e0['t']
            if e0['l2'] > 0 and e1['l2'] > 0 and t > 0:
                dx0 = np.pi / 2 / Ns[-2]
                dx1 = np.pi / 2 / Ns[-1]
                l2_rate = (np.log(e0['l2'] / e1['l2']) /
                           np.log(dx0 / dx1))
                linf_rate = (np.log(e0['linf'] / e1['linf']) /
                             np.log(dx0 / dx1))
                print(f"  {t:8.3f} {l2_rate:10.2f} {linf_rate:12.2f}")
            elif t == 0:
                print(f"  {t:8.3f}      (IC)")

    print(f"\n  Done. Output in {outdir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SBP vs Exact snapshot comparison")
    parser.add_argument('--center', default='face',
                        help='"face", "vertex", or "edge"')
    parser.add_argument('--tilt1', type=float, default=None,
                        help='xi1 tilt (radians, raw)')
    parser.add_argument('--tilt2', type=float, default=None,
                        help='xi2 tilt (radians, raw)')
    parser.add_argument('--face', type=int, default=0,
                        help='Panel for custom center (default: 0)')
    parser.add_argument('--times', nargs='+', type=float, default=None,
                        help='Output times in days')
    parser.add_argument('--Ns', nargs='+', type=int, default=[48, 96],
                        help='Grid resolutions (default: 48 96)')
    parser.add_argument('--nsteps', type=int, default=None,
                        help='Output first N steps (overrides --times)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (auto-generated if not set)')
    args = parser.parse_args()

    # Parse center
    if args.tilt1 is not None:
        xi1 = args.tilt1
        xi2 = args.tilt2 if args.tilt2 is not None else 0.0
        X, Y, Z = equiangular_to_cartesian_np(xi1, xi2, args.face)
        center_xyz = (float(X), float(Y), float(Z))
        label = f"custom ξ=({xi1:.4f},{xi2:.4f}) face={args.face}"
        tag = f"tilt_{xi1:.2f}_{xi2:.2f}_f{args.face}"
    elif args.center == 'face':
        center_xyz = (0.0, 0.0, 1.0)
        label = "face center (0,0,1)"
        tag = "face"
    elif args.center == 'vertex':
        s3 = 1.0 / np.sqrt(3.0)
        center_xyz = (s3, s3, s3)
        label = "cube vertex (1,1,1)/√3"
        tag = "vertex"
    elif args.center == 'edge':
        # Edge midpoint: face 0 East edge
        X, Y, Z = equiangular_to_cartesian_np(np.pi/4, 0.0, 0)
        center_xyz = (float(X), float(Y), float(Z))
        label = "edge midpoint (π/4, 0, face 0)"
        tag = "edge"
    else:
        center_xyz = (0.0, 0.0, 1.0)
        label = "face center (0,0,1)"
        tag = "face"

    if args.outdir is None:
        Ns_str = '_'.join(str(n) for n in args.Ns)
        args.outdir = f"snapshots_{tag}_N{Ns_str}"

    run_comparison(center_xyz, label, Ns=args.Ns,
                   output_times=args.times, outdir=args.outdir)


if __name__ == "__main__":
    main()
