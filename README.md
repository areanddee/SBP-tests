# SBP-SWE: Summation-by-Parts Shallow Water Equations on Cubed Sphere

Implementation of the Ch42 scheme from Shashkin, Goyman & Tretyak (2025):
"SBP FD for Linear Shallow Water Equations on Staggered Curvilinear Grids"

## Quick Start

```bash
pip install jax jaxlib numpy pytest
pytest tests/ -v              # fast unit tests (~30 seconds)
pytest tests/ -v --runslow    # + slow convergence tests (minutes)
```

## Package Structure

```
sbp_swe/            Core library
  operators.py       SBP 4/2 difference, interpolation, quadrature matrices
  geometry.py        Equiangular gnomonic projection, metric tensor
  mesh.py            Staggered grid construction, metric at grid points
  projection.py      EDGES connectivity, h-projection (Eq. 51-52)
  sat.py             Conservative Cartesian-averaged SAT (Eq. 53-55)
  velocity.py        Cartesian <-> covariant <-> contravariant transforms
  coriolis.py        Energy-conserving Coriolis operator (Eq. 63)
  halo.py            Ghost-cell halo exchange
  timestepping.py    RK4 time integration
  diagnostics.py     Mass and energy computation
  system.py          Full SWE assembly (f=0)

tests/               pytest test suite
  test_operators.py   SBP identity, quadrature, interpolation adjoint
  test_geometry.py    Unit sphere mapping, metric tensor properties
  test_velocity.py    Roundtrips, edge continuity, basis tangency
  test_projection.py  Edge/corner continuity, idempotency
  test_sat.py         Mass conservation (machine precision)
  test_steady_state.py  Zero tendency for uniform fields
  test_conservation.py  Mass (1e-15), energy (dt^5 scaling)
  test_coriolis.py    V operator, energy neutrality
  test_convergence.py Spatial convergence (Shashkin Section 6.2)

experiments/         Shashkin reproducers
  shashkin_gauss.py   25-day Gaussian wave tests at publication parameters

archive/             Historical development code (safe to delete)
```

## Key Results

- **Mass conservation**: Machine precision (~1e-15 to 1e-18)
- **Energy conservation**: Temporal error scales as dt^5 (RK4)
- **Spatial convergence**: Absolute L2 errors match Shashkin Figure 3/4
- **Gauss variant 2** (cube vertex, hardest case): validated through 25 days
