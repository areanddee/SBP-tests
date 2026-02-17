"""
projection.py — Cubed-Sphere Connectivity and h-Projection
=============================================================

Defines the 12-edge connectivity table for the cubed sphere and
implements the h-field projection operator (Eq. 51-52) that enforces
continuity at shared interfaces.

Edge convention: (panel_a, edge_a, panel_b, edge_b, op)
    op: 'N' = identity, 'R' = reverse, 'T' = transpose, 'TR' = transpose+reverse
    For index mapping: 'N','T' → k↔k;  'R','TR' → k↔(N-k)

Reference: Shashkin 2025, Eq. 51-52; Ullrich cubed-sphere connectivity
"""

import jax.numpy as jnp


# ============================================================
# 12-edge connectivity table
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


# ============================================================
# Edge helper functions
# ============================================================

def reverses(op):
    """Does this edge operation reverse the index order?"""
    return op in ('R', 'TR')


def get_h_boundary(h_panel, edge, N):
    """Extract (N+1,) boundary from h of shape (N+1, N+1)."""
    if edge == 'N':   return h_panel[:, N]
    elif edge == 'S': return h_panel[:, 0]
    elif edge == 'E': return h_panel[N, :]
    elif edge == 'W': return h_panel[0, :]


def set_h_boundary(h, panel, edge, vals, N):
    """Set (N+1,) boundary in h of shape (6, N+1, N+1)."""
    if edge == 'N':   return h.at[panel, :, N].set(vals)
    elif edge == 'S': return h.at[panel, :, 0].set(vals)
    elif edge == 'E': return h.at[panel, N, :].set(vals)
    elif edge == 'W': return h.at[panel, 0, :].set(vals)


def edge_k_to_ij(edge, k, N):
    """Convert (edge, index k) to (i, j) in the (N+1, N+1) h-grid."""
    if edge == 'N':   return (k, N)
    elif edge == 'S': return (k, 0)
    elif edge == 'E': return (N, k)
    elif edge == 'W': return (0, k)


def boundary_sign(edge):
    """Sign of boundary term in SBP identity: +1 for max, -1 for min."""
    return +1.0 if edge in ('E', 'N') else -1.0


def hv_inv_index(edge, N):
    """Which Hv index corresponds to this boundary?"""
    return N if edge in ('E', 'N') else 0


# ============================================================
# h-projection (Eq. 51-52)
# ============================================================

def build_projection_fn(N, Jh, Hv_diag):
    """
    Build h-projection function for all 12 edges + 8 corners.

    For equiangular cubed sphere, J is the same on both sides of a shared
    vertex, so the weighted average (Eq. 51) simplifies to:
        (Ah·h)_m = (h_m + h_m*) / 2

    Corner points (Eq. 52): average of 3 panels.

    Args:
        N: Grid resolution
        Jh: (N+1, N+1) Jacobian at h-points (used for corner weighting)
        Hv_diag: (N+1,) diagonal of SBP quadrature weight matrix

    Returns:
        project_h: function (6, N+1, N+1) → (6, N+1, N+1)
        corner_groups: list of [(panel, i, j), ...] triples
    """
    # Build corner groups from edge connectivity
    corner_map = {}
    for pa, ea, pb, eb, op in EDGES:
        rev = reverses(op)
        for k_a in [0, N]:
            k_b = (N - k_a) if rev else k_a
            ij_a = edge_k_to_ij(ea, k_a, N)
            ij_b = edge_k_to_ij(eb, k_b, N)
            key_a = (pa,) + ij_a
            key_b = (pb,) + ij_b
            if key_a not in corner_map:
                corner_map[key_a] = {key_a}
            if key_b not in corner_map:
                corner_map[key_b] = {key_b}
            corner_map[key_a].add(key_b)
            corner_map[key_b].add(key_a)

    # Transitive closure
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

    # Deduplicate
    seen = set()
    corner_groups = []
    for key, group in corner_map.items():
        frozen = frozenset(group)
        if frozen not in seen and len(group) == 3:
            seen.add(frozen)
            corner_groups.append(sorted(group))

    def project_h(h):
        """Project h at all shared interfaces."""
        h_orig = h

        # 1. Edge averaging (12 edges)
        for pa, ea, pb, eb, op in EDGES:
            rev = reverses(op)
            bnd_a = get_h_boundary(h_orig[pa], ea, N)
            bnd_b = get_h_boundary(h_orig[pb], eb, N)
            if rev:
                bnd_b = bnd_b[::-1]
            avg = 0.5 * (bnd_a + bnd_b)
            h = set_h_boundary(h, pa, ea, avg, N)
            avg_b = avg[::-1] if rev else avg
            h = set_h_boundary(h, pb, eb, avg_b, N)

        # 2. Corner averaging (8 corners, 3 panels each)
        for group in corner_groups:
            vals = jnp.array([h_orig[p, i, j] for p, i, j in group])
            avg = jnp.mean(vals)
            for p, i, j in group:
                h = h.at[p, i, j].set(avg)

        return h

    return project_h, corner_groups
