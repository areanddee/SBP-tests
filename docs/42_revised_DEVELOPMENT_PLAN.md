# SBP 4/2 Cubed-Sphere Advection: Incremental Development Plan

## Diagnosis: Why step7 Is Broken

After reviewing all the code, **test_step7_cubed_sphere.py is essentially a throwaway** — it discards
all the verified infrastructure built up previously. Here are its specific problems:

### Problem 1: 1-Deep Halo Exchange (Fatal for 4th-order stencils)
- Uses `halo_exchange.py` → arrays are `(6, N+2, N+2)` → only **1 ghost cell** per side
- The 4th-order interpolation stencil `[-1/16, 9/16, 9/16, -1/16]` needs **2 ghost cells**
- At panel boundaries, the code falls back to 2-point averages, creating O(h²) errors that dominate
- **Fix**: Switch to `halo_exchange_2deep.py` → `(6, N+4, N+4)` → 2 ghost cells per side

### Problem 2: Fake Physics
- Uses `v1 = 0, v2 = 1` (uniform eta velocity) — NOT actual solid-body rotation
- Actual solid-body rotation requires contravariant velocities computed via `velocity_transforms.py`
- Contravariant velocities vary across each panel and differ between panels

### Problem 3: Fake Geometry
- Uses same metric tensor for all 6 panels: `sqrt_G = 1/(cos²ξ cos²η)`
- Missing the actual per-panel Cartesian-to-equiangular metric from `grid.py`
- Missing `√G·v` at cell faces (needed for flux form)

### Problem 4: Fake Initial Condition
- Uses a 2D Gaussian `exp(-(ξ² + η²)/2σ²)` on panel 0 only
- Should use Williamson/Drake cosine bell: great-circle distance formula on the sphere

### Problem 5: Wrong RHS Structure
- Hand-coded first-order upwind with MC limiter instead of SBP operators
- Not using flux-form `dφ/dt = -(1/√G) div(√G v φ)`

## Key Insight: `advection_sbp_cubesphere.py` Is the Right Starting Point

This file already has:
- ✅ Proper SBP 4/2 operator matrices (Pvc interpolation, Dvc derivative)
- ✅ Proper per-panel geometry from `grid.py` 
- ✅ Proper contravariant velocities from `velocity_transforms.py`
- ✅ Proper `√G·v` pre-computed at cell faces
- ✅ Proper cosine bell IC via great-circle distance
- ✅ Flux-form RHS: `-(1/√G)(dFx/dξ¹ + dFy/dξ²)`
- ✅ RK4 time stepping
- ❌ **Still uses 1-deep halo exchange** — this is the one thing to fix

## The Plan: 6 Incremental Steps

Each step produces a **runnable test script** that YOU run and report results back.
I produce ONE script per step. No multi-step AI development.

---

### Step A: Baseline — Run advection_sbp_cubesphere.py with Order=2

**Goal**: Confirm the existing infrastructure works with 2/1 operators (which only need 1 ghost cell).

**What I do**: Create a small test wrapper that:
- Imports `advection_sbp_cubesphere.py` infrastructure
- Runs Williamson TC1 with `order=2`, `N=40`, short run (1/12 rotation)
- Reports: mass error, max/min, amplitude

**What you do**: Run it, paste output.

**Expected**: Mass conservation ~1e-14, amplitude loss ~10-20% (2nd order is diffusive but functional)

**Pass criteria**: Mass error < 1e-10. If this fails, the base infrastructure has a bug we need to fix first.

---

### Step B: Verify 2-Deep Halo Exchange in Isolation

**Goal**: Confirm `halo_exchange_2deep.py` correctly fills 2 layers of ghost cells.

**What I do**: Create a diagnostic test that:
- Creates a known field (e.g., `field[face, i, j] = face*1000 + i*N + j`)
- Runs `make_halo_exchange_2deep`
- Checks that ghost cells match expected neighbor values for all 12 edges
- Prints detailed pass/fail for each edge + transformation type

**What you do**: Run it, paste output.

**Expected**: All 24 ghost-cell checks pass (12 edges × 2 directions).

**Pass criteria**: Every edge pair correct. If any fail, we debug connectivity before proceeding.

---

### Step C: Integrate 2-Deep Halos into Advection RHS

**Goal**: Replace 1-deep with 2-deep halo exchange in the advection code.

**What I do**: Create a modified version of `advection_sbp_cubesphere.py` that:
- Uses `extend_to_include_ghosts_2deep` → `(6, N+4, N+4)` arrays
- Uses `make_halo_exchange_2deep` for ghost cell exchange
- Adjusts all array indexing: interior is now `[2:N+2, 2:N+2]`
- Keeps `order=2` (so the ONLY change is the halo depth)
- Runs same test as Step A

**What you do**: Run it, paste output.

**Expected**: Results essentially identical to Step A (2nd-order operators only use 1 neighbor, so extra ghosts shouldn't change anything).

**Pass criteria**: Mass error < 1e-10, amplitude within 1% of Step A result.

This is the **critical integration test**: same physics, same operators, just deeper halos. If results differ, the 2-deep halo indexing is wrong.

---

### Step D: Enable 4th-Order Interpolation with 2-Deep Halos

**Goal**: Now that 2-deep halos are working, enable the 4th-order interpolation stencil at ALL faces including boundaries.

**What I do**: Modify Step C's code to:
- Use `order=4` in SBPPanelData
- The interior interpolation already uses `[-1/16, 9/16, 9/16, -1/16]`
- Boundary faces now ALSO get 4-point stencils (the 2 ghost cells provide the needed data)
- The boundary face interpolation that was `0.5*(a+b)` becomes `[-1/16, 9/16, 9/16, -1/16]`
- Run with `N=40`, short run (1/12 rotation)

**What you do**: Run it, paste output.

**Expected**: Significant improvement over Step A. Less amplitude loss, lower L∞ error.

**Pass criteria**: Mass error < 1e-10, amplitude retention > 90% for short run.

---

### Step E: Full Rotation Test at Ullrich Parameters

**Goal**: Run the complete Williamson TC1 at Ullrich's parameters.

**What I do**: Modify Step D for:
- `N = 40` (matching Ullrich Table 2)
- `CFL = 0.5`  
- Full rotation (T = 2π, or 12 days in physical units)
- Report: l₁, l₂, l∞ normalized error norms, mass error, amplitude retention

**What you do**: Run it (will take longer — maybe a few minutes), paste output.

**Expected**: 
- Mass conservation < 1e-12
- l₂ error comparable to Shashkin Ch_42 results
- Amplitude retention > 85%

**Pass criteria**: Stable completion, mass error < 1e-8, amplitude > 50%.

---

### Step F: Convergence Study

**Goal**: Verify the scheme converges at the expected rate.

**What I do**: Create a convergence test:
- N = 20, 40, 80 (each doubling)
- Same CFL = 0.5
- Full rotation
- Report l₂ error, l∞ error, convergence rates

**What you do**: Run it (this one takes longest), paste output.

**Expected**: ~2nd order for cosine bell (C¹ smoothness limits convergence), ~4th order for Gaussian bell.

**Pass criteria**: Convergence rate > 1.5 for cosine bell, > 3.0 for Gaussian bell.

---

## Summary Table

| Step | What Changes                    | Tests                        | Key Risk                        |
|------|--------------------------------|-----------------------------|---------------------------------|
| A    | Baseline: order=2, 1-deep     | Mass, amplitude             | Infrastructure bug              |
| B    | 2-deep halo in isolation       | Connectivity verification   | Edge transformations wrong      |
| C    | 2-deep halo + order=2         | Same as A (should match)    | Indexing off-by-one             |
| D    | 2-deep halo + order=4         | Improved amplitude          | Boundary stencil error          |
| E    | Full rotation, Ullrich params | l₁, l₂, l∞ norms           | Instability over long run       |
| F    | Convergence N=20,40,80        | Convergence rates           | Rate not meeting theory         |

## Ground Rules

1. **I produce ONE script per step. You run it.**
2. If a step fails, we debug THAT step before moving on.
3. No multi-file refactors. Each step is a self-contained script.
4. Each script prints clear diagnostic output — no guessing.
5. If I need to fix something, I explain exactly what changed and why.

## Ready to Begin?

Say the word and I'll produce the Step A script.
