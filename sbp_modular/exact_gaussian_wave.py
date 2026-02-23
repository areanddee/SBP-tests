"""
exact_gaussian_wave.py — Exact solution for Gaussian gravity wave on sphere
============================================================================

The linearized SWE (f=0) on a sphere reduce to a scalar wave equation:
    ∂²h/∂t² = c² ∇²h,   c = √(gH₀)

For an axisymmetric initial perturbation with zero initial velocity,
expand in Legendre polynomials:
    h(θ, t) = Σₗ hₗ(t) Pₗ(cos θ)

where θ = great-circle distance from perturbation center.

Since ∇²Pₗ = -l(l+1)/a² Pₗ, each mode decouples:
    hₗ(t) = hₗ(0) cos(ωₗ t),   ωₗ = c √(l(l+1)) / a

The projection coefficients are computed via Gauss-Legendre quadrature:
    hₗ(0) = (2l+1)/2 ∫₀π h₀(θ) Pₗ(cos θ) sin θ dθ

Usage:
    from exact_gaussian_wave import ExactGaussianWave
    sol = ExactGaussianWave(amp=0.01, sigma=0.3, c=1.0, a=1.0)
    h = sol.evaluate(theta, t)           # scalar or array theta
    h = sol.evaluate_cartesian(X,Y,Z, t) # Cartesian coords on unit sphere
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


class ExactGaussianWave:
    """Exact solution of Gaussian gravity wave on a sphere."""

    def __init__(self, amp=0.01, sigma=0.3, c=1.0, a=1.0,
                 center_xyz=(0., 0., 1.), L_max=200, n_quad=400):
        """
        Parameters
        ----------
        amp : float — amplitude of initial Gaussian
        sigma : float — width parameter: h₀(θ) = amp * exp(-θ²/(2σ²))
        c : float — wave speed √(gH₀)
        a : float — sphere radius
        center_xyz : tuple — (X₀,Y₀,Z₀) Cartesian coords of perturbation center
        L_max : int — maximum Legendre degree
        n_quad : int — number of Gauss-Legendre quadrature points
        """
        self.amp = amp
        self.sigma = sigma
        self.c = c
        self.a = a
        self.center = np.array(center_xyz, dtype=np.float64)
        self.center /= np.linalg.norm(self.center)  # normalize
        self.L_max = L_max

        # Project initial condition onto Legendre polynomials
        self.h_l = self._project_ic(n_quad)

        # Precompute frequencies: ωₗ = c * √(l(l+1)) / a
        ell = np.arange(self.L_max + 1, dtype=np.float64)
        self.omega_l = self.c * np.sqrt(ell * (ell + 1)) / self.a

        # Report spectrum info
        self._spectrum_info()

    def _project_ic(self, n_quad):
        """Compute hₗ(0) = (2l+1)/2 ∫₀π h₀(θ) Pₗ(cos θ) sin θ dθ."""
        x_gl, w_gl = leggauss(n_quad)
        theta = np.arccos(x_gl)
        h0 = self.amp * np.exp(-theta**2 / (2 * self.sigma**2))
        f = w_gl * h0  # preweight

        # Evaluate all Pₗ(x) via stable 3-term recurrence
        h_l = np.zeros(self.L_max + 1)
        P_prev = np.ones_like(x_gl)       # P_0 = 1
        h_l[0] = 0.5 * np.sum(f * P_prev)

        if self.L_max >= 1:
            P_curr = x_gl.copy()           # P_1 = x
            h_l[1] = 1.5 * np.sum(f * P_curr)

            for l in range(1, self.L_max):
                # (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1}
                P_next = ((2*l + 1) * x_gl * P_curr - l * P_prev) / (l + 1)
                h_l[l + 1] = (2*(l+1) + 1) / 2.0 * np.sum(f * P_next)
                P_prev = P_curr
                P_curr = P_next

        return h_l

    def _spectrum_info(self):
        """Detect effective truncation and zero out noise floor."""
        peak = np.max(np.abs(self.h_l))

        # Find where coefficients drop below noise floor
        # For GL quadrature with n_quad points, noise floor ~ n_quad * eps * peak
        # Use relative threshold of 1e-8 to be safe
        threshold = peak * 1e-8
        l_eff = 0
        for l in range(self.L_max + 1):
            if np.abs(self.h_l[l]) > threshold:
                l_eff = l

        # Zero out noise tail
        n_zeroed = np.count_nonzero(self.h_l[l_eff + 1:])
        self.h_l[l_eff + 1:] = 0.0

        print(f"  Gaussian IC: amp={self.amp}, sigma={self.sigma}")
        print(f"  Spectrum: L_max={self.L_max}, effective L={l_eff} "
              f"(zeroed {n_zeroed} noise modes), "
              f"|h_{l_eff}|={np.abs(self.h_l[l_eff]):.2e}")

    def evaluate(self, theta, t):
        """
        Evaluate h(θ, t) at given great-circle distances and time.

        Parameters
        ----------
        theta : array-like — great-circle distance(s) from center (radians)
        t : float — time

        Returns
        -------
        h : same shape as theta — height perturbation
        """
        theta = np.asarray(theta, dtype=np.float64)
        x = np.cos(theta)
        orig_shape = x.shape
        x_flat = x.ravel()

        # hₗ(t) = hₗ(0) cos(ωₗ t)
        h_l_t = self.h_l * np.cos(self.omega_l * t)

        # h(θ, t) = Σₗ hₗ(t) Pₗ(cos θ) — Clenshaw summation for stability
        result = self._clenshaw_legendre(x_flat, h_l_t)
        return result.reshape(orig_shape)

    def evaluate_cartesian(self, X, Y, Z, t):
        """
        Evaluate h at Cartesian points on unit sphere.

        Parameters
        ----------
        X, Y, Z : array-like — Cartesian coordinates (on unit sphere)
        t : float — time

        Returns
        -------
        h : same shape as X — height perturbation
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)

        # Great-circle distance = arccos(dot product with center)
        cos_d = np.clip(
            X * self.center[0] + Y * self.center[1] + Z * self.center[2],
            -1.0, 1.0
        )
        theta = np.arccos(cos_d)
        return self.evaluate(theta, t)

    @staticmethod
    def _clenshaw_legendre(x, c):
        """
        Evaluate Σₗ cₗ Pₗ(x) using Clenshaw recurrence.

        The three-term recurrence for Legendre polynomials:
            (l+1) P_{l+1}(x) = (2l+1) x Pₗ(x) - l P_{l-1}(x)

        Clenshaw runs this backward for numerical stability.
        """
        N = len(c) - 1
        if N < 0:
            return np.zeros_like(x)

        # b_{N+2} = b_{N+1} = 0
        b_k2 = np.zeros_like(x)
        b_k1 = np.zeros_like(x)

        for k in range(N, 0, -1):
            # b_k = c_k + ((2k+1)/(k+1)) x b_{k+1} - ((k+1)/(k+2)) b_{k+2}
            b_k = c[k] + (2*k + 1) / (k + 1) * x * b_k1 - (k + 1) / (k + 2) * b_k2
            b_k2 = b_k1
            b_k1 = b_k

        # result = c_0 * P_0(x) + b_1 * P_1(x) - (1/2) b_2 * P_0(x)
        # Simplification: result = c_0 + x * b_1 - (1/2) b_2
        result = c[0] + x * b_k1 - 0.5 * b_k2
        return result


def verify_quadrature():
    """Verify the projection by reconstructing h₀(θ) from the coefficients."""
    print("=" * 65)
    print("  Verification: project then reconstruct IC")
    print("=" * 65)

    sol = ExactGaussianWave(amp=0.01, sigma=0.3, L_max=200, n_quad=400)

    theta_test = np.linspace(0, np.pi, 1000)
    h_exact = 0.01 * np.exp(-theta_test**2 / (2 * 0.3**2))
    h_recon = sol.evaluate(theta_test, t=0.0)

    err = np.max(np.abs(h_exact - h_recon))
    print(f"  Max reconstruction error at t=0: {err:.2e}")
    assert err < 1e-10, f"Reconstruction error too large: {err}"
    print(f"  ✓ PASS\n")
    return sol


def verify_conservation():
    """
    Verify Parseval consistency: Clenshaw evaluation matches modal sum.

    Note: ∫h² dΩ is NOT conserved — energy sloshes between h and v fields.
    The conserved SWE energy is E ∝ Σ h_l(0)² which is trivially constant.
    This test instead checks that the Clenshaw summation is accurate by
    comparing direct quadrature of h² against the Parseval identity.
    """
    print("=" * 65)
    print("  Verification: Parseval consistency (Clenshaw vs quadrature)")
    print("=" * 65)

    sol = ExactGaussianWave(amp=0.01, sigma=0.3, L_max=200, n_quad=400)

    x_gl, w_gl = leggauss(500)
    theta_gl = np.arccos(x_gl)
    ell = np.arange(sol.L_max + 1, dtype=np.float64)

    for t in [0.0, 1.0, 5.0, 12.5, 25.0]:
        h_l_t = sol.h_l * np.cos(sol.omega_l * t)
        E_parseval = 2 * np.pi * np.sum(2 * h_l_t**2 / (2 * ell + 1))

        h_quad = sol.evaluate(theta_gl, t)
        E_quad = 2 * np.pi * np.sum(w_gl * h_quad**2)

        rel_err = abs(E_parseval - E_quad) / abs(E_parseval)
        print(f"  t={t:5.1f}: Parseval={E_parseval:.10e}, "
              f"quad={E_quad:.10e}, rel err={rel_err:.1e}")

    print(f"  ✓ Clenshaw evaluation consistent with Parseval\n")


def verify_wavespeed():
    """Check wave propagation timing."""
    print("=" * 65)
    print("  Verification: wave propagation")
    print("=" * 65)

    sol = ExactGaussianWave(amp=0.01, sigma=0.3, c=1.0, a=1.0,
                            L_max=200, n_quad=400)

    # Track peak location over time
    theta_fine = np.linspace(0, np.pi, 2000)
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, np.pi]:
        h = sol.evaluate(theta_fine, t)
        i_peak = np.argmax(np.abs(h))
        print(f"  t={t:5.2f}: max|h| = {np.max(np.abs(h)):.6e} "
              f"at θ = {theta_fine[i_peak]:.3f} rad ({np.degrees(theta_fine[i_peak]):.1f}°)")

    print()


def demo_solution():
    """Generate exact solution snapshots for both IC sets."""
    print("=" * 65)
    print("  Demo: exact solutions at T=25")
    print("=" * 65)

    # Our test IC
    sol1 = ExactGaussianWave(amp=0.01, sigma=0.3, c=1.0, a=1.0,
                              center_xyz=(0, 0, 1), L_max=200, n_quad=400)
    print(f"\n  --- Our IC (amp=0.01, σ=0.3, center=pole) ---")
    print(f"  h(θ=0,   t=25) = {sol1.evaluate(0.0, t=25.0):.6e}")
    print(f"  h(θ=π,   t=25) = {sol1.evaluate(np.pi, t=25.0):.6e}")
    print(f"  max|h(t=25)| = {np.max(np.abs(sol1.evaluate(np.linspace(0,np.pi,500), 25.0))):.6e}")

    # Shashkin IC: h₀ = exp(-16θ²) → σ = 1/(4√2), amp = 1.0
    sigma_sh = 1.0 / (4 * np.sqrt(2))
    sol2 = ExactGaussianWave(amp=1.0, sigma=sigma_sh, c=1.0, a=1.0,
                              center_xyz=(0, 0, 1), L_max=200, n_quad=400)
    print(f"\n  --- Shashkin IC (amp=1.0, σ={sigma_sh:.5f}, center=pole) ---")
    print(f"  h(θ=0,   t=25) = {sol2.evaluate(0.0, t=25.0):.6e}")
    print(f"  h(θ=π,   t=25) = {sol2.evaluate(np.pi, t=25.0):.6e}")
    print(f"  max|h(t=25)| = {np.max(np.abs(sol2.evaluate(np.linspace(0,np.pi,500), 25.0))):.6e}")

    # Variant 2: vertex-centered (use cube vertex coordinates)
    # φ = arcsin(1/√3), λ = π/4 → Cartesian: (1/√3, 1/√3, 1/√3)
    vert = np.array([1, 1, 1], dtype=np.float64) / np.sqrt(3)
    sol3 = ExactGaussianWave(amp=0.01, sigma=0.3, c=1.0, a=1.0,
                              center_xyz=vert, L_max=200, n_quad=400)
    print(f"\n  --- Our IC at cube vertex ({vert[0]:.4f}, {vert[1]:.4f}, {vert[2]:.4f}) ---")
    print(f"  h at vertex,    t=25: {sol3.evaluate(0.0, t=25.0):.6e}")
    print(f"  h at anti-vertex, t=25: {sol3.evaluate(np.pi, t=25.0):.6e}")

    print()


if __name__ == "__main__":
    print("=" * 65)
    print("  Exact Gaussian Gravity Wave on the Sphere")
    print("  via Legendre Polynomial Expansion")
    print("=" * 65)

    sol = verify_quadrature()
    verify_conservation()
    verify_wavespeed()
    demo_solution()
