"""
sbp_swe — SBP Finite Difference Shallow Water Equations on Cubed Sphere
========================================================================

Implements the Ch42 scheme from Shashkin, Goyman & Tretyak (2025):
"SBP FD for Linear Shallow Water Equations on Staggered Curvilinear Grids"

Modules:
    operators   — SBP 4/2 difference, interpolation, and quadrature matrices
    geometry    — Equiangular gnomonic projection, metric tensor, covariant bases
    mesh        — Staggered grid construction, metric evaluation at grid points
    projection  — h-projection (Eq. 51-52), EDGES connectivity, corner averaging
    sat         — Conservative Cartesian-averaged SAT coupling (Eq. 53-55)
    velocity    — Cartesian ↔ covariant ↔ contravariant transforms
    coriolis    — Energy-conserving Coriolis operator (Eq. 63)
    halo        — Ghost-cell halo exchange for advection
    timestepping — RK4 time integration
    diagnostics — Mass and energy computation
    system      — Full cubed-sphere SWE assembly
"""
