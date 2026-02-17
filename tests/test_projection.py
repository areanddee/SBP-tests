"""
test_projection.py — h-Projection Operator Tests
==================================================

Verifies:
  - Projection enforces continuity at 12 edges and 8 corners
  - Projection is idempotent (Ah Ah = Ah)
  - Uniform field is unchanged by projection
  - Corner groups: exactly 8 groups of 3 panels each
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.operators import sbp_42
from sbp_swe.mesh import make_staggered_grids, make_all_metrics
from sbp_swe.projection import EDGES, build_projection_fn, reverses


class TestEdgeConnectivity:
    """Verify the 12-edge connectivity table."""

    def test_12_edges(self):
        """Must have exactly 12 edges."""
        assert len(EDGES) == 12

    def test_all_panels_connected(self):
        """Every panel appears in at least 4 edges (it has 4 boundaries)."""
        panel_counts = {p: 0 for p in range(6)}
        for pa, ea, pb, eb, op in EDGES:
            panel_counts[pa] += 1
            panel_counts[pb] += 1
        for p, count in panel_counts.items():
            assert count == 4, f"Panel {p} has {count} edges, expected 4"

    def test_all_edges_used(self):
        """Each (panel, edge) pair appears exactly once."""
        used = set()
        for pa, ea, pb, eb, op in EDGES:
            assert (pa, ea) not in used, f"Duplicate ({pa},{ea})"
            assert (pb, eb) not in used, f"Duplicate ({pb},{eb})"
            used.add((pa, ea))
            used.add((pb, eb))
        assert len(used) == 24  # 6 panels × 4 edges


class TestProjection:
    """h-projection operator properties."""

    def test_corner_groups(self):
        """Must find exactly 8 corner groups, each with 3 panels."""
        N = 8
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)
        grids = make_staggered_grids(N)
        metrics = make_all_metrics(grids)
        Hv_diag = jnp.diag(ops.Hv)

        _, corners = build_projection_fn(N, metrics['Jh'], Hv_diag)
        assert len(corners) == 8, f"Expected 8 corners, got {len(corners)}"
        for group in corners:
            assert len(group) == 3, f"Corner group has {len(group)} panels, expected 3"

    @pytest.mark.parametrize("N", [8, 16])
    def test_uniform_field_unchanged(self, N):
        """Projecting a uniform field should not change it."""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)
        grids = make_staggered_grids(N)
        metrics = make_all_metrics(grids)
        Hv_diag = jnp.diag(ops.Hv)

        project_h, _ = build_projection_fn(N, metrics['Jh'], Hv_diag)

        h = jnp.ones((6, N + 1, N + 1)) * 42.0
        h_proj = project_h(h)

        err = float(jnp.max(jnp.abs(h_proj - h)))
        assert err < 1e-14, f"Uniform field changed by {err:.2e}"

    @pytest.mark.parametrize("N", [8, 16])
    def test_idempotent(self, N):
        """Ah(Ah(h)) = Ah(h) — projection is idempotent."""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)
        grids = make_staggered_grids(N)
        metrics = make_all_metrics(grids)
        Hv_diag = jnp.diag(ops.Hv)

        project_h, _ = build_projection_fn(N, metrics['Jh'], Hv_diag)

        # Random discontinuous field
        key = jax.random.PRNGKey(42)
        h = jax.random.normal(key, (6, N + 1, N + 1))

        h1 = project_h(h)
        h2 = project_h(h1)

        err = float(jnp.max(jnp.abs(h2 - h1)))
        assert err < 1e-14, f"Projection not idempotent: {err:.2e}"

    @pytest.mark.parametrize("N", [8, 16])
    def test_edge_continuity(self, N):
        """After projection, values match at all 12 shared edges."""
        dx = (jnp.pi / 2) / N
        ops = sbp_42(N, dx)
        grids = make_staggered_grids(N)
        metrics = make_all_metrics(grids)
        Hv_diag = jnp.diag(ops.Hv)

        project_h, _ = build_projection_fn(N, metrics['Jh'], Hv_diag)

        key = jax.random.PRNGKey(123)
        h = jax.random.normal(key, (6, N + 1, N + 1))
        h = project_h(h)

        max_err = 0.0
        for pa, ea, pb, eb, op in EDGES:
            rev = reverses(op)
            if ea == 'N':   bnd_a = h[pa, :, N]
            elif ea == 'S': bnd_a = h[pa, :, 0]
            elif ea == 'E': bnd_a = h[pa, N, :]
            elif ea == 'W': bnd_a = h[pa, 0, :]

            if eb == 'N':   bnd_b = h[pb, :, N]
            elif eb == 'S': bnd_b = h[pb, :, 0]
            elif eb == 'E': bnd_b = h[pb, N, :]
            elif eb == 'W': bnd_b = h[pb, 0, :]

            if rev:
                bnd_b = bnd_b[::-1]
            err = float(jnp.max(jnp.abs(bnd_a - bnd_b)))
            max_err = max(max_err, err)

        assert max_err < 1e-14, f"Edge discontinuity after projection: {max_err:.2e}"
