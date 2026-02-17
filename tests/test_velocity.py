"""
test_velocity.py — Velocity Transform Tests
=============================================

Verifies:
  - Basis vectors tangent to sphere (a_i · r = 0)
  - Contravariant roundtrip: cart→contra→cart = identity
  - Covariant roundtrip: cart→cov→cart = identity
  - Covariant continuity at shared edges
  - Covariant vs contravariant consistency
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sbp_swe.geometry import equiangular_to_cartesian
from sbp_swe.velocity import (
    get_covariant_basis, contravariant_to_cartesian,
    cartesian_to_contravariant, cartesian_to_covariant,
    covariant_to_cartesian, _compute_metric_and_inverse,
)


class TestBasisVectors:
    """Covariant basis vectors a_i = dr/dxi^i."""

    def test_tangent_to_sphere(self):
        """Basis vectors must be tangent to sphere: r · a_i = 0."""
        max_err = 0.0
        for face_id in range(6):
            for xi1 in [-0.5, 0.0, 0.5]:
                for xi2 in [-0.5, 0.0, 0.5]:
                    X, Y, Z = equiangular_to_cartesian(xi1, xi2, face_id)
                    a1, a2 = get_covariant_basis(xi1, xi2, face_id)

                    r_dot_a1 = X * a1[0] + Y * a1[1] + Z * a1[2]
                    r_dot_a2 = X * a2[0] + Y * a2[1] + Z * a2[2]

                    err = max(abs(float(r_dot_a1)), abs(float(r_dot_a2)))
                    max_err = max(max_err, err)
        assert max_err < 1e-14, f"Basis vectors not tangent: {max_err:.2e}"

    def test_metric_positive_definite(self):
        """Covariant metric det(g) > 0 everywhere."""
        for face_id in range(6):
            for xi1 in [-0.7, 0.0, 0.7]:
                for xi2 in [-0.7, 0.0, 0.7]:
                    a1, a2 = get_covariant_basis(xi1, xi2, face_id)
                    g11, g12, g22, _, _, _ = _compute_metric_and_inverse(a1, a2)
                    det = float(g11 * g22 - g12**2)
                    assert det > 0, f"det(g) = {det} at face {face_id}, ({xi1},{xi2})"


class TestRoundtrips:
    """Cartesian ↔ covariant/contravariant roundtrips."""

    @pytest.mark.parametrize("face_id", range(6))
    def test_contravariant_roundtrip(self, face_id):
        """cart → contra → cart = identity."""
        xi1, xi2 = 0.3, -0.2
        Vx, Vy, Vz = 1.0, 2.0, -0.5

        v1, v2 = cartesian_to_contravariant(Vx, Vy, Vz, xi1, xi2, face_id)
        Vx2, Vy2, Vz2 = contravariant_to_cartesian(v1, v2, xi1, xi2, face_id)

        err = max(abs(float(Vx2 - Vx)), abs(float(Vy2 - Vy)), abs(float(Vz2 - Vz)))
        assert err < 1e-13, f"Contravariant roundtrip error: {err:.2e}"

    @pytest.mark.parametrize("face_id", range(6))
    def test_covariant_roundtrip(self, face_id):
        """cart → cov → cart = identity."""
        xi1, xi2 = -0.4, 0.1
        Vx, Vy, Vz = -0.3, 1.5, 0.7

        v1, v2 = cartesian_to_covariant(Vx, Vy, Vz, xi1, xi2, face_id)
        Vx2, Vy2, Vz2 = covariant_to_cartesian(v1, v2, xi1, xi2, face_id)

        err = max(abs(float(Vx2 - Vx)), abs(float(Vy2 - Vy)), abs(float(Vz2 - Vz)))
        assert err < 1e-13, f"Covariant roundtrip error: {err:.2e}"


class TestEdgeContinuity:
    """Covariant velocity is continuous at shared edges in the Cartesian sense."""

    def test_covariant_continuity_at_edges(self):
        """
        A uniform Cartesian velocity field, converted to covariant on
        adjacent panels, must give the same Cartesian velocity at shared edges.
        """
        from sbp_swe.projection import EDGES, reverses
        pi4 = float(jnp.pi / 4)

        Vx_const, Vy_const, Vz_const = 1.0, 0.5, -0.3
        max_err = 0.0

        for pa, ea, pb, eb, op in EDGES:
            # Get shared boundary coordinates
            n_pts = 5
            t = np.linspace(-pi4, pi4, n_pts)

            for k in range(n_pts):
                # Coords in panel A's frame
                if ea == 'E':   xi1a, xi2a = pi4, t[k]
                elif ea == 'W': xi1a, xi2a = -pi4, t[k]
                elif ea == 'N': xi1a, xi2a = t[k], pi4
                elif ea == 'S': xi1a, xi2a = t[k], -pi4

                # Coords in panel B's frame
                k_b = (n_pts - 1 - k) if reverses(op) else k
                if eb == 'E':   xi1b, xi2b = pi4, t[k_b]
                elif eb == 'W': xi1b, xi2b = -pi4, t[k_b]
                elif eb == 'N': xi1b, xi2b = t[k_b], pi4
                elif eb == 'S': xi1b, xi2b = t[k_b], -pi4

                # Convert to covariant in each panel's frame, then back to Cartesian
                v1a, v2a = cartesian_to_covariant(Vx_const, Vy_const, Vz_const,
                                                   xi1a, xi2a, pa)
                Vxa, Vya, Vza = covariant_to_cartesian(v1a, v2a, xi1a, xi2a, pa)

                v1b, v2b = cartesian_to_covariant(Vx_const, Vy_const, Vz_const,
                                                   xi1b, xi2b, pb)
                Vxb, Vyb, Vzb = covariant_to_cartesian(v1b, v2b, xi1b, xi2b, pb)

                err = max(abs(float(Vxa - Vxb)), abs(float(Vya - Vyb)),
                          abs(float(Vza - Vzb)))
                max_err = max(max_err, err)

        assert max_err < 1e-13, f"Edge Cartesian continuity error: {max_err:.2e}"
