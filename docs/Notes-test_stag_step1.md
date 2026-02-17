# Notes: test_stag_step1 — 1D Staggered SWE Validation

**Date:** February 12, 2026  
**Reference:** Shashkin, Goyman & Tretyak 2025, Section 2.2  
**Code:** `test_stag_step1.py` using `sbp_staggered_1d.py` (SBP 4/2 operators)

---

## Setup

Linearized 1D shallow water equations on periodic domain [0, L]:

    dh/dt = −H₀ · du/dx     (continuity)
    du/dt = −g  · dh/dx      (momentum)

Staggered grid: h at N+1 vertices, u at N cell centers. SBP 4/2 operators (4th-order interior, 2nd-order boundary) with SAT-Projection enforcing periodicity via Eqs. 26–27 and 32–34. RK4 time integration. Plane wave IC: h = A·cos(kx), u = A·√(g/H₀)·cos(kx), exact solution returns to IC after one period T = 2π/ω.

## Key Results

**SBP property (Eq. 34):** Verified to machine precision (≤ 7e-15) for all N tested. This is the foundational guarantee — it means the spatial discretization satisfies a discrete integration-by-parts identity exactly.

**Projection properties:** A² = A (idempotent) and (Hv·A)ᵀ = Hv·A (Hv-orthogonal), both to machine precision. These are required for the energy estimate to hold.

**Convergence:** Rate 4.5 in l₂ for both h and u across N = 16 to 128. Theory predicts s+1 = 3 for SBP 4/2 (2s/s notation), so we're exceeding the guaranteed rate. This is consistent with Shashkin's observation that SAT-Projection prevents boundary error accumulation, allowing the interior 4th-order stencil to dominate.

**Mass conservation:** Machine precision (< 1e-17) over 10 wave periods. The SBP divergence operator in flux form guarantees this algebraically — it does not depend on time-stepping accuracy.

## Energy Conservation: Spatial vs. Temporal

Initial test at CFL = 0.3 showed ΔE/E = 2.0e-5 after 10 wave periods, which appeared concerning. A CFL sweep resolved this definitively:

| CFL   | dt          | ΔE/E     | dt-scaling rate |
|-------|-------------|----------|-----------------|
| 0.400 | 1.25e-02    | 8.3e-05  | —               |
| 0.200 | 6.25e-03    | 2.6e-06  | 5.0             |
| 0.100 | 3.13e-03    | 8.2e-08  | 5.0             |
| 0.050 | 1.56e-03    | 2.5e-09  | 5.0             |
| 0.025 | 7.81e-04    | 8.0e-11  | 5.0             |

The energy error scales as dt⁵, confirming that the spatial discretization conserves energy exactly and the drift is entirely from RK4 time-stepping.

**Why dt⁵ and not dt⁴?** RK4 has global truncation error O(dt⁴), but for linear Hamiltonian systems (which the linearized SWE is), the leading energy error term is one order higher than the solution error. This is well-known for symplectic analysis of Runge-Kutta methods applied to conservative systems.

**What Shashkin guarantees:** The paper proves dE_N/dt ≤ 0 for the semi-discrete system (Carpenter et al. Theorem 2.2, adapted to the staggered case). This is an energy *bound*, not conservation — the spatial operators never create energy but are permitted to dissipate it. For the periodic domain with SAT-Projection, the interface terms vanish identically (Eq. 34 gives u·Hc·D_vc·h + (Ah)·Hv·D_cv·u ≡ 0), so the semi-discrete energy is in fact exactly conserved: dE/dt = 0. The energy drift we observe is therefore purely temporal.

**Practical implication:** At operationally reasonable CFL values (0.3–0.5), RK4 energy drift is O(10⁻⁵) per 10 wave periods. This is acceptable for the linearized SWE and consistent with Shashkin's characterization of "good" energy conservation. For applications requiring tighter energy control, higher-order or symplectic time integrators could be used, but this is not a priority for proof-of-concept work.

## Validated Components

| Component | Status | Notes |
|-----------|--------|-------|
| SBP 4/2 operators (Dcv, Dvc, Hv, Hc, l, r) | ✓ | All properties verified |
| SAT correction (Eq. 26–27) | ✓ | Integrated into DP_vc, DP_cv |
| Projection matrix A (Eq. 33) | ✓ | Idempotent + Hv-orthogonal |
| SAT-Projection operators (Eq. 32) | ✓ | Closed-domain SBP property exact |
| Convergence rate | ✓ | 4.5 (exceeds s+1 = 3 theory) |
| Mass conservation | ✓ | Machine precision |
| Energy conservation | ✓ | Spatial: exact. Temporal: dt⁵ |

## Next Step

Step 2: Extend to 2D on a single panel with doubly-periodic boundary conditions using Kronecker products (Shashkin Section 3). Gaussian pulse → circular gravity waves. This validates the 2D operator construction before introducing multi-panel connectivity.
