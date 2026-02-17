"""
Diagnostic: Why Cartesian-Averaged SAT Breaks Conservation
==========================================================

Two critical issues:
  1. sign_a * flux_cons_a + sign_b * flux_cons_b != 0  at 8 of 12 edges
     -> The consensus fluxes computed independently on each panel DON'T cancel
     -> Mass conservation is IMPOSSIBLE with this formulation

  2. flux_own != flux_cons even for the SAME panel's own velocity
     -> The Dcv operator computes flux via Eq.56 (off-diagonal Q12 handled
        through interpolation Pcv/Pvc), but _consensus_flux uses the ANALYTIC
        Q^{ij} at boundary points directly
     -> These are different discrete approximations of the same continuous flux
     -> The mismatch injects a non-conservative correction even for smooth fields

Root cause of issue 1:
  At edges where both panels have the SAME boundary type (N<->N, S<->S),
  sign_a = sign_b, so cancellation requires flux_cons_a = -flux_cons_b.
  But the off-diagonal metric Q12 has OPPOSITE signs at (xi1, pi/4) vs
  (-xi1, pi/4) on different panels, so flux = J*(Q12*v1 + Q22*v2) does
  NOT negate properly.

Usage:
  python diag_sat_conservation.py
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from sbp_staggered_1d import sbp_42
from grid import equiangular_to_cartesian
from velocity_transforms import (covariant_to_cartesian, cartesian_to_covariant)

# ============================================================
# Edge connectivity table (same as sat_operators.py)
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

def _boundary_sign(edge):
    return +1.0 if edge in ('E', 'N') else -1.0


# ============================================================
# Metric (same as test_stag_step5.py)
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


def edge_bnd_coords(edge, N):
    pi4 = jnp.pi / 4
    xi_v = jnp.linspace(-pi4, pi4, N + 1)
    if edge == 'E':
        return jnp.full(N + 1, pi4), xi_v
    elif edge == 'W':
        return jnp.full(N + 1, -pi4), xi_v
    elif edge == 'N':
        return xi_v, jnp.full(N + 1, pi4)
    elif edge == 'S':
        return xi_v, jnp.full(N + 1, -pi4)


def consensus_flux(v1_avg, v2_avg, edge, J, Q11, Q12, Q22):
    """Normal mass flux = J * Q^{n,j} * v_j"""
    if edge in ('E', 'W'):
        return J * (Q11 * v1_avg + Q12 * v2_avg)
    else:
        return J * (Q12 * v1_avg + Q22 * v2_avg)


def extrapolate_covariant_to_boundary(v1, v2, edge, ops):
    """Extrapolate covariant (v1,v2) to h-grid boundary."""
    l, r_vec, Pcv = ops.l, ops.r, ops.Pcv
    if edge == 'E':
        v1_bnd = jnp.einsum('c,cj->j', r_vec, v1)
        v2_bnd = Pcv @ v2[-1, :]
    elif edge == 'W':
        v1_bnd = jnp.einsum('c,cj->j', l, v1)
        v2_bnd = Pcv @ v2[0, :]
    elif edge == 'N':
        v1_bnd = Pcv @ v1[:, -1]
        v2_bnd = jnp.einsum('ic,c->i', v2, r_vec)
    elif edge == 'S':
        v1_bnd = Pcv @ v1[:, 0]
        v2_bnd = jnp.einsum('ic,c->i', v2, l)
    return v1_bnd, v2_bnd


# ============================================================
# Setup
# ============================================================

N = 8
dx = jnp.pi / (2 * N)
xi_v = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, N + 1)
xi_c = (jnp.arange(N) + 0.5) * dx - jnp.pi / 4
xi1_v1, xi2_v1 = jnp.meshgrid(xi_c, xi_v, indexing='ij')   # (N, N+1)
xi1_v2, xi2_v2 = jnp.meshgrid(xi_v, xi_c, indexing='ij')   # (N+1, N)

ops = sbp_42(N, float(dx))

# Initialize covariant velocity from solid body rotation V = (-Y, X, 0)
v1_all = jnp.zeros((6, N, N + 1))
v2_all = jnp.zeros((6, N + 1, N))

for p in range(6):
    X, Y, Z = equiangular_to_cartesian(xi1_v1, xi2_v1, p)
    v1_p, _ = cartesian_to_covariant(-Y, X, jnp.zeros_like(X),
                                      xi1_v1, xi2_v1, p)
    v1_all = v1_all.at[p].set(v1_p)

    X, Y, Z = equiangular_to_cartesian(xi1_v2, xi2_v2, p)
    _, v2_p = cartesian_to_covariant(-Y, X, jnp.zeros_like(X),
                                      xi1_v2, xi2_v2, p)
    v2_all = v2_all.at[p].set(v2_p)


# ============================================================
# ISSUE 1: Consensus fluxes don't cancel at 8 of 12 edges
# ============================================================

print("=" * 70)
print("ISSUE 1: Consensus flux cancellation test")
print("         sign_a * flux_cons_a + sign_b * flux_cons_b = 0 ?")
print("=" * 70)
print()
print("For mass conservation, SAT contributions from panels A and B")
print("must cancel when summed over the global domain. This requires")
print("sign_a * F*_a + sign_b * F*_b = 0 at every shared boundary.")
print()

header = (f"{'Edge':<20} {'sign_a':>6} {'sign_b':>6} {'op':>4} "
          f"{'max|imbalance|':>16} {'max|F_a|':>10} {'relative':>10}")
print(header)
print("-" * len(header))

n_pass = 0
n_fail = 0

for pa, ea, pb, eb, op in EDGES:
    rev = _reverses(op)
    sign_a = _boundary_sign(ea)
    sign_b = _boundary_sign(eb)

    # Extrapolate covariant to boundary
    v1a_bnd, v2a_bnd = extrapolate_covariant_to_boundary(
        v1_all[pa], v2_all[pa], ea, ops)
    v1b_bnd, v2b_bnd = extrapolate_covariant_to_boundary(
        v1_all[pb], v2_all[pb], eb, ops)

    # Convert to Cartesian on each panel
    xi1_a, xi2_a = edge_bnd_coords(ea, N)
    xi1_b, xi2_b = edge_bnd_coords(eb, N)

    Vx_A, Vy_A, Vz_A = covariant_to_cartesian(v1a_bnd, v2a_bnd,
                                                xi1_a, xi2_a, pa)
    Vx_B, Vy_B, Vz_B = covariant_to_cartesian(v1b_bnd, v2b_bnd,
                                                xi1_b, xi2_b, pb)

    # Align indices for reversed edges
    if rev:
        Vx_B, Vy_B, Vz_B = Vx_B[::-1], Vy_B[::-1], Vz_B[::-1]

    # Average in Cartesian
    Vx_avg = 0.5 * (Vx_A + Vx_B)
    Vy_avg = 0.5 * (Vy_A + Vy_B)
    Vz_avg = 0.5 * (Vz_A + Vz_B)

    # Convert back to covariant in each panel's frame
    v1_avg_a, v2_avg_a = cartesian_to_covariant(Vx_avg, Vy_avg, Vz_avg,
                                                  xi1_a, xi2_a, pa)

    if rev:
        Vx_b, Vy_b, Vz_b = Vx_avg[::-1], Vy_avg[::-1], Vz_avg[::-1]
    else:
        Vx_b, Vy_b, Vz_b = Vx_avg, Vy_avg, Vz_avg

    v1_avg_b, v2_avg_b = cartesian_to_covariant(Vx_b, Vy_b, Vz_b,
                                                  xi1_b, xi2_b, pb)

    # Compute consensus flux independently on each panel
    J_a, Q11_a, Q12_a, Q22_a = compute_metric(xi1_a, xi2_a)
    J_b, Q11_b, Q12_b, Q22_b = compute_metric(xi1_b, xi2_b)

    flux_a = consensus_flux(v1_avg_a, v2_avg_a, ea, J_a, Q11_a, Q12_a, Q22_a)
    flux_b = consensus_flux(v1_avg_b, v2_avg_b, eb, J_b, Q11_b, Q12_b, Q22_b)

    # Check cancellation
    imbalance = sign_a * flux_a + sign_b * flux_b
    max_imb = float(jnp.max(jnp.abs(imbalance)))
    max_fa = float(jnp.max(jnp.abs(flux_a)))
    rel = max_imb / max(max_fa, 1e-16)

    ok = max_imb < 1e-12
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    tag = "  OK" if ok else "FAIL"

    edge_label = f"({pa},{ea})<->({pb},{eb})"
    print(f"  {edge_label:<18} {sign_a:>+5.0f}  {sign_b:>+5.0f}  {op:>3} "
          f"  {max_imb:>14.2e}   {max_fa:>8.2e}   {rel:>8.2e}  {tag}")

print()
print(f"Result: {n_pass} edges cancel, {n_fail} edges DON'T cancel")
if n_fail > 0:
    print(f"*** FATAL: Mass conservation is IMPOSSIBLE with this SAT formulation.")
    print(f"    The consensus fluxes computed independently on each panel")
    print(f"    do NOT telescope when summed over the global domain.")


# ============================================================
# ISSUE 2: flux_own != flux_cons for the same panel
# ============================================================

print()
print("=" * 70)
print("ISSUE 2: flux_own vs flux_cons_self mismatch (same panel)")
print("         These should agree if the discrete scheme is consistent.")
print("=" * 70)
print()
print("flux_own  = extrapolated J*v^n from Dcv operator (Eq.56 + interp)")
print("flux_cons = J * Q^{n,j} * v_j using analytic metric at boundary")
print()
print("The Dcv operator computes contravariant velocity via Eq.56:")
print("  v^1 = Q11*v1 + J^{-1} * Pvc * (J*Q12 * (v2 @ Pcv^T))")
print("then forms u1 = J1 * v^1_contra.")
print()
print("_consensus_flux uses the analytic Q^{ij} directly at boundary:")
print("  flux = J * (Q11*v1 + Q12*v2) for E/W edges")
print()
print("These are DIFFERENT discrete approximations. The mismatch means")
print("SAT = -sign * Hv_inv * (flux_own - flux_cons) != 0 even when the")
print("solution is perfectly smooth and continuous across the interface.")
print()

# Compute mass flux via Eq.56 for panel 0
Jh, _, Q12_h, _ = compute_metric(
    *jnp.meshgrid(xi_v, xi_v, indexing='ij'))
J1, Q11_1, _, _ = compute_metric(xi1_v1, xi2_v1)
J2, _, _, Q22_2 = compute_metric(xi1_v2, xi2_v2)

Pvc = ops.Pvc
Pcv = ops.Pcv
JQ12 = Jh * Q12_h

v1_p = v1_all[0]   # (N, N+1)
v2_p = v2_all[0]   # (N+1, N)

# v^1 via Eq.56
v2_at_h = v2_p @ Pcv.T
cross_at_h = JQ12 * v2_at_h
cross_at_v1 = Pvc @ cross_at_h
v1_contra = Q11_1 * v1_p + cross_at_v1 / J1
u1 = J1 * v1_contra

# v^2 via Eq.56
v1_at_h = Pcv @ v1_p
cross_at_h2 = JQ12 * v1_at_h
cross_at_v2 = cross_at_h2 @ Pvc.T
v2_contra = cross_at_v2 / J2 + Q22_2 * v2_p
u2 = J2 * v2_contra

print(f"{'Edge':>6} {'|flux_own-flux_cons|':>22} {'|flux_own|':>12} {'relative':>10}")
print("-" * 55)

for edge in ['E', 'W', 'N', 'S']:
    # flux_own: extrapolated from mass flux arrays
    if edge == 'E':
        flux_own = jnp.einsum('c,cj->j', ops.r, u1)
    elif edge == 'W':
        flux_own = jnp.einsum('c,cj->j', ops.l, u1)
    elif edge == 'N':
        flux_own = jnp.einsum('ic,c->i', u2, ops.r)
    elif edge == 'S':
        flux_own = jnp.einsum('ic,c->i', u2, ops.l)

    # flux_cons from same panel's own extrapolated covariant velocity
    xi1_bnd, xi2_bnd = edge_bnd_coords(edge, N)
    J_bnd, Q11_bnd, Q12_bnd, Q22_bnd = compute_metric(xi1_bnd, xi2_bnd)
    v1_bnd, v2_bnd = extrapolate_covariant_to_boundary(v1_p, v2_p, edge, ops)
    flux_cons = consensus_flux(v1_bnd, v2_bnd, edge, J_bnd, Q11_bnd, Q12_bnd, Q22_bnd)

    diff = float(jnp.max(jnp.abs(flux_own - flux_cons)))
    scale = float(jnp.max(jnp.abs(flux_own)))
    rel = diff / max(scale, 1e-16)
    print(f"  {edge:>4}   {diff:>20.4e}   {scale:>10.4e}   {rel:>8.2e}")

print()
print("This ~1-2% mismatch means the SAT penalty is nonzero even for")
print("a perfectly smooth, continuous solution. Because the penalty")
print("does NOT telescope (Issue 1), it breaks global conservation.")
