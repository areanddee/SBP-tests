"""
test_geometry.py — Coordinate Transform and Metric Tensor Tests
================================================================

Verifies:
  - equiangular_to_cartesian maps to unit sphere
  - Panel centers map to correct Cartesian axes
  - Metric tensor: det(Q) = 1/J², positivity, symmetry
  - Metric consistent across panels (equiangular property)
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.geometry import equiangular_to_cartesian, compute_metric


class TestCartesianMapping:
    """Equiangular to Cartesian coordinate transform."""

    def test_on_unit_sphere(self):
        """All mapped points lie on unit sphere."""
        xi = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, 11)
        xi1, xi2 = jnp.meshgrid(xi, xi, indexing='ij')
        max_err = 0.0
        for face_id in range(6):
            X, Y, Z = equiangular_to_cartesian(xi1, xi2, face_id)
            r = jnp.sqrt(X**2 + Y**2 + Z**2)
            err = float(jnp.max(jnp.abs(r - 1.0)))
            max_err = max(max_err, err)
        assert max_err < 1e-14, f"Off unit sphere by {max_err:.2e}"

    def test_panel_centers(self):
        """Panel centers map to correct Cartesian axis directions."""
        expected = [
            (0, 0, 1),   # face 0: +Z
            (1, 0, 0),   # face 1: +X
            (0, 1, 0),   # face 2: +Y
            (-1, 0, 0),  # face 3: -X
            (0, -1, 0),  # face 4: -Y
            (0, 0, -1),  # face 5: -Z
        ]
        for face_id, (ex, ey, ez) in enumerate(expected):
            X, Y, Z = equiangular_to_cartesian(0.0, 0.0, face_id)
            err = max(abs(float(X) - ex), abs(float(Y) - ey), abs(float(Z) - ez))
            assert err < 1e-14, f"Face {face_id} center wrong: ({X},{Y},{Z})"

    def test_all_panels_cover_sphere(self):
        """Six panels tile the sphere: every axis-aligned point is covered."""
        # Check that panel vertices are shared
        corners = set()
        for face_id in range(6):
            for xi1 in [-jnp.pi / 4, jnp.pi / 4]:
                for xi2 in [-jnp.pi / 4, jnp.pi / 4]:
                    X, Y, Z = equiangular_to_cartesian(xi1, xi2, face_id)
                    corners.add((round(float(X), 10), round(float(Y), 10), round(float(Z), 10)))
        # Cube has 8 corners
        assert len(corners) == 8, f"Expected 8 unique corners, got {len(corners)}"


class TestMetricTensor:
    """Metric tensor properties."""

    def test_jacobian_positive(self):
        """Jacobian J > 0 everywhere on every panel."""
        xi = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, 21)
        xi1, xi2 = jnp.meshgrid(xi, xi, indexing='ij')
        J, _, _, _ = compute_metric(xi1, xi2)
        assert float(jnp.min(J)) > 0, "Jacobian not everywhere positive"

    def test_metric_determinant(self):
        """det(Q) = (Q11*Q22 - Q12^2) should equal 1/J^2."""
        xi = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, 21)
        xi1, xi2 = jnp.meshgrid(xi, xi, indexing='ij')
        J, Q11, Q12, Q22 = compute_metric(xi1, xi2)

        det_Q = Q11 * Q22 - Q12**2
        expected = 1.0 / J**2

        err = float(jnp.max(jnp.abs(det_Q - expected) / expected))
        assert err < 1e-12, f"det(Q) != 1/J² by relative error {err:.2e}"

    def test_metric_positive_definite(self):
        """Contravariant metric must be positive definite: Q11 > 0, det(Q) > 0."""
        xi = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, 21)
        xi1, xi2 = jnp.meshgrid(xi, xi, indexing='ij')
        J, Q11, Q12, Q22 = compute_metric(xi1, xi2)

        assert float(jnp.min(Q11)) > 0, "Q11 not positive"
        assert float(jnp.min(Q22)) > 0, "Q22 not positive"
        det_Q = Q11 * Q22 - Q12**2
        assert float(jnp.min(det_Q)) > 0, "det(Q) not positive"

    def test_metric_symmetric_in_xi(self):
        """Q11(xi1,xi2) = Q22(xi2,xi1) by symmetry of equiangular coordinates."""
        xi = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, 21)
        xi1, xi2 = jnp.meshgrid(xi, xi, indexing='ij')

        _, Q11_a, _, _ = compute_metric(xi1, xi2)
        _, _, _, Q22_b = compute_metric(xi2, xi1)

        err = float(jnp.max(jnp.abs(Q11_a - Q22_b)))
        assert err < 1e-14, f"Q11/Q22 symmetry error: {err:.2e}"

    def test_metric_at_center(self):
        """At panel center (0,0), metric is identity-like: Q12=0, Q11=Q22."""
        J, Q11, Q12, Q22 = compute_metric(jnp.array(0.0), jnp.array(0.0))
        assert abs(float(Q12)) < 1e-14, f"Q12 at center: {float(Q12):.2e}"
        assert abs(float(Q11) - float(Q22)) < 1e-14, "Q11 != Q22 at center"
